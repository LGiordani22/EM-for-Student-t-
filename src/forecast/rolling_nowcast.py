"""
src/forecast/rolling_nowcast.py

Real-time ROLLING NOWCAST ORCHESTRATOR for the nowcasting pipeline (Fifth brick).

Walks a rolling monthly calendar (as_of = the 15th of each month, the
rolling-nowcast convention — see panel_builder._as_of_day_aware) and, for every
month, nowcasts the quarter(s) whose GDP is not yet published, with the DFM
engine (Student-t + Gaussian) and the univariate benchmarks (ARMA, random walk).

Target selection (the calendar logic) — fully automatic, never hard-coded
-------------------------------------------------------------------------
At an as_of date (day 15 of month M), a quarter T (quarter-end qe) is "in
flight" — i.e. a nowcast target — iff its GDP is NOT yet published under the
real-time rule used everywhere in the pipeline:

      gdp_available_through(as_of) < qe(T)

We enumerate the in-flight quarters from the first unpublished quarter up to the
quarter that CONTAINS as_of (the current quarter):

      last_pub = gdp_available_through(as_of)          # last published quarter
      current  = as_of rolled to its quarter-end
      targets  = { qe : last_pub < qe <= current }

This yields ONE target in most months and TWO in the FIRST month of each
quarter (Jan/Apr/Jul/Oct), where the just-ended quarter's GDP advance is not out
yet (released ~end of that month) AND the new quarter has just begun.  Example:
  * Nov/Dec: only the current quarter Q4 (horizon 2, 3).
  * Jan: Q4 of last year (last nowcast, horizon 4) + Q1 (first, horizon 1).

horizon_month = how many months of the target quarter have elapsed at as_of
(months from the quarter's first month up to and including the as_of month):
1, 2, 3 within the quarter, then 4 in the first month of the next quarter when
the advance is still pending.  The shrinking forecast horizon.

The SAME predicate (gdp_available_through < qe) governs both what enters the
panel and which quarters are targets, so the target's own GDP is, by
construction, never in the panel — cheating is structurally impossible.

Output
------
A "long" DataFrame (one row per as_of × target × method) with columns:
  as_of, target_quarter, horizon_month, method, nowcast_livello, nowcast_z,
  gdp_realizzato, errore, n_iter, converged.
Saved as CSV under output/forecast_realtime/csv/<config>/.

Run
---
  python -m src.forecast.rolling_nowcast                       # short self-test
  python -m src.forecast.rolling_nowcast --start 2008-09 --end 2009-01
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.forecast.data_import import gdp_available_through, list_available_vintages
from src.forecast.nowcast_engine import nowcast_gdp
from src.forecast.benchmarks import arma_nowcast, random_walk_nowcast
from src.config_utils import parse_config_args

# ─── Locations ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ASOF_DAY = 15
_DEFAULT_ESTIMATORS = ("student_t", "gaussian")


# ─── Calendar helpers ─────────────────────────────────────────────────────────

def _month_iter(start: str, end: str) -> list[pd.Timestamp]:
    """Day-15 as_of timestamps for every month in [start, end] ('YYYY-MM')."""
    months = pd.date_range(
        start=pd.Timestamp(start + "-01"),
        end=pd.Timestamp(end + "-01"),
        freq="MS",
    )
    return [pd.Timestamp(d.year, d.month, _ASOF_DAY) for d in months]


def quarter_label(qe: pd.Timestamp) -> str:
    """Quarter-end month-end -> 'YYYYQn'."""
    return f"{qe.year}Q{(qe.month - 1) // 3 + 1}"


def horizon_month(as_of: pd.Timestamp, target_qe: pd.Timestamp) -> int:
    """
    Months of the target quarter elapsed at as_of: from the quarter's FIRST
    month up to and including the as_of month (1, 2, 3 within the quarter; 4 in
    the next quarter's first month when the advance is still pending).
    """
    q_start_month = ((target_qe.month - 1) // 3) * 3 + 1
    return (as_of.year - target_qe.year) * 12 + (as_of.month - q_start_month) + 1


def in_flight_targets(as_of: pd.Timestamp) -> list[pd.Timestamp]:
    """
    Quarter-ends in flight at as_of: from the first unpublished quarter up to the
    quarter that contains as_of.  Automatic — driven only by
    gdp_available_through.  Returns 1 quarter-end (most months) or 2 (the first
    month of each quarter).
    """
    last_pub = gdp_available_through(as_of)                 # day-aware
    current_qe = as_of + pd.offsets.QuarterEnd(0)           # quarter containing as_of
    targets: list[pd.Timestamp] = []
    q = last_pub + pd.offsets.QuarterEnd(1)                 # first quarter-end after last_pub
    while q <= current_qe:
        targets.append(q)
        q = q + pd.offsets.QuarterEnd(1)
    return targets


def build_calendar(start: str, end: str) -> list[tuple[pd.Timestamp, list[tuple[str, pd.Timestamp, int]]]]:
    """
    Rolling calendar without running any fit: for each as_of, the list of
    (target_label, target_qe, horizon_month).  Cheap — for validating the logic.
    """
    cal = []
    for as_of in _month_iter(start, end):
        targets = [(quarter_label(qe), qe, horizon_month(as_of, qe))
                   for qe in in_flight_targets(as_of)]
        cal.append((as_of, targets))
    return cal


# ─── Realised GDP (current, revised) for scoring ──────────────────────────────

def _load_realized_gdp(config_name: str = "small") -> pd.Series:
    csv = os.path.join(_PROJECT_ROOT, "data", "processed", f"dataset_{config_name}.csv")
    df = pd.read_csv(csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df["GDPC1"].dropna()


# ─── Incremental save + resume (anti-zombie, in the spirit of monte_carlo) ────
#
# Why incremental.  The Student-t DFM fits are slow and a full period can run
# for hours; an interruption (kill, crash, OneDrive hiccup, laptop sleep) must
# never lose completed work.  So instead of writing the CSV only at the end, we
# rewrite it ATOMICALLY after every completed nowcast: at any instant the CSV on
# disk contains exactly what has been computed so far.
#
# Why resume.  Each nowcast is identified uniquely by the triple
# (as_of, target_quarter, method).  On start-up we read the period CSV (if it
# exists), collect the triples already present, and skip them — only the missing
# nowcasts are computed.  Re-launching the SAME command therefore continues from
# where it stopped instead of recomputing everything.  One CSV per period
# (rolling_nowcast_<start>_<end>.csv), so each run/period resumes off its own
# file.
#
# Failed / non-converged nowcasts.  These ARE saved (converged=False, or n_iter
# = -1 for an exception) so the resume does NOT retry them forever.  If you
# later want to re-attempt one, delete its row from the CSV by hand and re-run:
# only the removed triples will be recomputed.  (This is the convention Lorenzo
# asked for; flip the `_SKIP_FAILED_ON_RESUME` switch below to retry instead.)
#
# Atomicity & OneDrive.  We always write to "<file>.tmp" and then os.replace()
# onto the final name — os.replace is atomic on Windows, so a half-written file
# can never be observed and the CSV is never corrupted, even if killed mid-flush.
# Because the project lives on OneDrive, the sync client can briefly lock the
# target; _atomic_write_csv retries a few times on PermissionError/OSError and,
# if it still cannot write, warns but does NOT abort the run (the row stays in
# memory and the next completed nowcast rewrites the whole file anyway).

_COLUMNS = [
    "as_of", "target_quarter", "horizon_month", "method",
    "nowcast_livello", "nowcast_z", "gdp_realizzato", "errore",
    "n_iter", "converged",
]

_ARMA_ORDER = (2, 2)                                   # benchmark default
_ARMA_METHOD = f"arma{_ARMA_ORDER[0]}{_ARMA_ORDER[1]}"  # -> "arma22"
_RW_METHOD = "random_walk"

# If True, rows with converged=False are NOT trusted on resume and get retried.
# Default False: keep them, never retry (see note above).
_SKIP_FAILED_ON_RESUME = False


def _period_csv_path(out_dir: str, start: str, end: str) -> str:
    return os.path.join(out_dir, f"rolling_nowcast_{start}_{end}.csv")


def _row_key(r: dict) -> tuple[str, str, str]:
    """Unique identity of a nowcast: (as_of, target_quarter, method)."""
    return (str(r["as_of"]), str(r["target_quarter"]), str(r["method"]))


def _load_existing(path: str) -> list[dict]:
    """
    Read an existing period CSV into a list of row-dicts.  Robust to a stray
    malformed line (skipped with a warning); returns [] if the file cannot be
    parsed at all, so a corrupt file degrades to 'start fresh' rather than
    crashing the run.
    """
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"  [warn] cannot parse existing CSV ({type(exc).__name__}: {exc}); "
              f"starting fresh: {path}")
        return []
    # Tolerate an older/partial file missing some columns.
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[_COLUMNS].to_dict("records")


def _atomic_write_csv(rows: list[dict], path: str,
                      retries: int = 6, delay: float = 0.7) -> bool:
    """
    Write `rows` to `path` atomically (temp file + os.replace), sorted by
    (as_of, target_quarter, method) for a tidy, stable file.  Retries on a
    transient lock (OneDrive sync); returns True on success, False if every
    attempt failed (the caller keeps going — work stays in memory).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=_COLUMNS)
    if len(df):
        df = df.sort_values(["as_of", "target_quarter", "method"],
                            kind="stable").reset_index(drop=True)
    tmp = f"{path}.tmp"
    for attempt in range(1, retries + 1):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)                  # atomic on Windows
            return True
        except (PermissionError, OSError) as exc:
            if attempt == retries:
                print(f"  [warn] could not write CSV after {retries} attempts "
                      f"({type(exc).__name__}: {exc}); will retry on next nowcast.")
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                return False
            time.sleep(delay)
    return False


# ─── The orchestrator ─────────────────────────────────────────────────────────

def run_rolling_nowcast(
    start: str,
    end: str,
    config_name: str = "small",
    estimators: tuple[str, ...] = _DEFAULT_ESTIMATORS,
    output_dir: str | None = None,
    save: bool = True,
    verbose_dfm: bool = False,
) -> pd.DataFrame:
    """
    Run the rolling nowcast over [start, end] ('YYYY-MM', as_of = day 15).

    For every month and every in-flight target, runs the DFM nowcasts
    (`estimators`) and the two univariate benchmarks, scores them against the
    current (revised) realised GDP, and returns a long DataFrame.  Non-converged
    or failed DFM fits are recorded (converged=False) and the run continues.

    Incremental / resumable.  When `save=True` the period CSV is rewritten
    atomically after every completed nowcast, and an existing CSV is read on
    start-up so that nowcasts already present — identified by the triple
    (as_of, target_quarter, method) — are skipped and only the missing ones are
    computed.  Re-running the same command thus resumes from where it stopped.
    """
    out_dir = output_dir or os.path.join(_PROJECT_ROOT, "output", "forecast_realtime", "csv", config_name)
    realized_gdp = _load_realized_gdp(config_name=config_name)

    # --- Resume: read whatever is already on disk for THIS period ---
    path = _period_csv_path(out_dir, start, end)
    all_rows: list[dict] = []
    done_keys: set[tuple[str, str, str]] = set()
    if save and os.path.exists(path):
        all_rows = _load_existing(path)
        for r in all_rows:
            if _SKIP_FAILED_ON_RESUME and not bool(r.get("converged", False)):
                continue                            # treat failures as not-done
            done_keys.add(_row_key(r))

    # Keep only months that actually have a vintage on disk (>= first usable).
    avail = set(list_available_vintages())
    calendar = build_calendar(start, end)
    skipped = [as_of for as_of, _ in calendar if (as_of.year, as_of.month) not in avail]
    if skipped:
        print(f"  [warn] {len(skipped)} month(s) have no usable vintage and are skipped: "
              f"{[f'{d.year}-{d.month:02d}' for d in skipped]}")

    # Enumerate every expected nowcast (its triple) to size the work + resume.
    methods = list(estimators) + [_ARMA_METHOD, _RW_METHOD]
    expected: list[tuple[str, str]] = []            # (key-as-string, kind)
    n_dfm_total = 0
    for as_of, targets in calendar:
        if (as_of.year, as_of.month) not in avail:
            continue
        as_of_iso = as_of.date().isoformat()
        for label, _qe, _h in targets:
            for m in methods:
                expected.append(((as_of_iso, label, m),
                                 "dfm" if m in estimators else "bench"))
                if m in estimators:
                    n_dfm_total += 1

    n_expected = len(expected)
    n_already = sum(1 for key, _ in expected if key in done_keys)
    n_missing = n_expected - n_already
    n_dfm_todo = sum(1 for key, kind in expected
                     if kind == "dfm" and key not in done_keys)

    if save and all_rows:
        print(f"  resume: trovati {len(all_rows)} nowcast nel CSV "
              f"({path}); dei {n_expected} previsti per il periodo, "
              f"{n_already} gia' fatti, ne calcolo {n_missing} mancanti.")
    else:
        print(f"  resume: nessun CSV esistente per il periodo, parto da zero "
              f"({n_expected} nowcast da calcolare).")
    print(f"  calendar: {len(calendar)} months "
          f"-> {n_dfm_todo} DFM fits + benchmark calls da fare.\n")

    fit_done = 0                                    # DFM fits computed this run

    def _persist():
        if save:
            _atomic_write_csv(all_rows, path)

    for as_of, targets in calendar:
        if (as_of.year, as_of.month) not in avail:
            continue
        as_of_iso = as_of.date().isoformat()
        last_pub = gdp_available_through(as_of)
        tgt_str = ", ".join(f"{lab}(h{h})" for lab, _qe, h in targets)

        # Is anything left to do this month?  (purely for a tidy log)
        month_keys = [(as_of_iso, lab, m) for lab, _qe, _h in targets for m in methods]
        if all(k in done_keys for k in month_keys):
            print(f"as_of {as_of.date()}  | targets: {tgt_str}  -> tutti gia' fatti, salto")
            continue

        print(f"as_of {as_of.date()}  | last GDP published = {quarter_label(last_pub)} "
              f"| targets: {tgt_str}")

        for label, qe, h in targets:
            realized = float(realized_gdp.loc[qe]) if qe in realized_gdp.index else float("nan")

            def _row(method, liv, z, n_iter, converged):
                err = (liv - realized) if np.isfinite(liv) and np.isfinite(realized) else float("nan")
                return {
                    "as_of": as_of_iso,
                    "target_quarter": label,
                    "horizon_month": h,
                    "method": method,
                    "nowcast_livello": liv,
                    "nowcast_z": z,
                    "gdp_realizzato": realized,
                    "errore": err,
                    "n_iter": n_iter,
                    "converged": converged,
                }

            # --- DFM estimators (slow) ---
            for est in estimators:
                key = (as_of_iso, label, est)
                if key in done_keys:
                    print(f"    [cached] {label} {est:<10} (gia' nel CSV, salto)")
                    continue
                fit_done += 1
                t0 = time.perf_counter()
                try:
                    r = nowcast_gdp(as_of, label, config_name=config_name, estimator=est, verbose=verbose_dfm)
                    liv, z = r["nowcast_livello"], r["nowcast_z"]
                    n_iter, converged = int(r["n_iter"]), bool(r["converged"])
                    status = "OK" if converged else "no-converge"
                except Exception as exc:                      # robustness: never abort the run
                    liv = z = float("nan")
                    n_iter, converged = -1, False
                    status = f"FAIL ({type(exc).__name__}: {exc})"
                dt = time.perf_counter() - t0
                print(f"    [{fit_done}/{n_dfm_todo}] {label} {est:<10} "
                      f"{dt:6.1f}s  {status:<12} nowcast={liv:+.3f} (real {realized:+.3f})")
                all_rows.append(_row(est, liv, z, n_iter, converged))
                done_keys.add(key)
                _persist()                          # incremental, atomic save

            # --- benchmarks (fast) ---
            key = (as_of_iso, label, _ARMA_METHOD)
            if key not in done_keys:
                a = arma_nowcast(as_of, label, config_name=config_name, order=_ARMA_ORDER)
                all_rows.append(_row(a["metodo"], a["nowcast_livello"], a["nowcast_z"],
                                     np.nan, bool(a["converged"])))
                done_keys.add((as_of_iso, label, a["metodo"]))
                _persist()
            key = (as_of_iso, label, _RW_METHOD)
            if key not in done_keys:
                w = random_walk_nowcast(as_of, label, config_name=config_name)
                all_rows.append(_row(w["metodo"], w["nowcast_livello"], w["nowcast_z"],
                                     np.nan, True))
                done_keys.add((as_of_iso, label, w["metodo"]))
                _persist()

    df = pd.DataFrame(all_rows, columns=_COLUMNS)
    if len(df):
        df = df.sort_values(["as_of", "target_quarter", "method"],
                            kind="stable").reset_index(drop=True)

    if save:
        _atomic_write_csv(all_rows, path)           # final flush (idempotent)
        print(f"\n  saved {len(df)} rows -> {path}")

    return df


# ─── CLI / short self-test ────────────────────────────────────────────────────

def _hr(t: str) -> None:
    print("\n" + "=" * 80)
    print(t)
    print("=" * 80)


def _print_calendar_logic(start: str, end: str) -> None:
    """Cheap validation of the target-selection logic (no fits)."""
    _hr(f"CALENDAR LOGIC  (as_of = day 15)   {start} .. {end}")
    print(f"  {'as_of':<12}{'last GDP pub':<14}{'targets (horizon)':<34}{'n_targets'}")
    for as_of, targets in build_calendar(start, end):
        last_pub = gdp_available_through(as_of)
        tgt = "  ".join(f"{lab}(h{h})" for lab, _qe, h in targets)
        flag = "  <- DOUBLE (quarter change)" if len(targets) == 2 else ""
        print(f"  {as_of.date().isoformat():<12}{quarter_label(last_pub):<14}{tgt:<34}{len(targets)}{flag}")


def main() -> None:
    def _extra(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", default="2008-09",
                       help="first month 'YYYY-MM' (as_of = day 15)")
        p.add_argument("--end", default="2009-01",
                       help="last month 'YYYY-MM'")
        p.add_argument("--output-dir", default=None,
                       help="override output dir (default: output/forecast_realtime/csv/<config>/)")
        p.add_argument("--no-save", action="store_true",
                       help="do not write the CSV")
        p.add_argument("--logic-only", action="store_true",
                       help="only print the calendar/target logic, run no fits")

    args = parse_config_args("Rolling real-time GDP nowcast.", extra=_extra)

    _hr("rolling_nowcast.py")
    _print_calendar_logic(args.start, args.end)

    if args.logic_only:
        _hr("Logic-only mode: no fits run.")
        return

    _hr(f"RUNNING ROLLING NOWCAST  {args.start} .. {args.end}  [config={args.config}]")
    t0 = time.perf_counter()
    df = run_rolling_nowcast(args.start, args.end, config_name=args.config,
                             output_dir=args.output_dir, save=not args.no_save)
    elapsed = time.perf_counter() - t0

    _hr("RESULTS (long DataFrame)")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:8.3f}"))
    _hr(f"Done in {elapsed:.1f}s  ({len(df)} rows).")


if __name__ == "__main__":
    main()
