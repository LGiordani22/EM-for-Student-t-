"""
src/forecast/compute_metrics.py

NOWCAST ACCURACY METRICS for the real-time nowcasting pipeline.

Pure reader: ingests the long CSVs produced by rolling_nowcast.py and computes
accuracy metrics aggregated by (method x horizon), by method, and by period.
Nothing is recomputed or re-estimated — it only scores nowcasts against the
realised GDP already stored in each row.

Config-aware (--small / --big / --config NAME) and NOT hardcoded on periods:
it DISCOVERS the CSVs present in the config's folder and unions them.  The set
of methods and horizons is DEDUCED from the data, so adding a method, a horizon,
or a new period just works without editing this file.

Metrics, per (method x horizon) and per method (pooled over horizons/dates):
  * n_used / n_tot : observations scored vs. present (target quarters still
                     missing their realised GDP are dropped, with a count)
  * RMSE, MAE      : error magnitude   (error = nowcast_livello - gdp_realizzato)
  * Bias           : signed mean error (over/under-prediction)
  * corr           : corr(nowcast, realised)
  * RMSE_rel_rw    : RMSE / RMSE(random_walk)  at the SAME horizon  -> "how much
  * RMSE_rel_arma  : RMSE / RMSE(arma22)         better than the benchmark"
  * mean_z, |z|    : mean standardised nowcast (Volatility-Paradox metric); the
                     z column is read straight from the CSV (nowcast_z)

Output:
  * pretty tables printed to screen (columnar), plus
  * <out>/metrics_summary.txt           (all tables, human-readable)
  * <out>/metrics_by_method_horizon.csv (the detailed table, machine-readable)
  where <out> defaults to the config's CSV folder.

Run
---
  python -m src.forecast.compute_metrics                      # small, all periods
  python -m src.forecast.compute_metrics --big
  python -m src.forecast.compute_metrics --big --periods 2024-10_2025-09
  python -m src.forecast.compute_metrics --small --periods 2008-01_2009-12,2020-01_2020-12
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from src.config_utils import parse_config_args, get_project_root

# ─── Benchmark method labels (match rolling_nowcast.py constants) ─────────────
# The SET of methods is discovered from the data; only the two reference
# benchmarks need a stable label so relative-RMSE columns know the denominator.
# If a benchmark is absent from the data, its relative column is silently NaN.
_RW_METHOD = "random_walk"
_ARMA_METHOD = "arma22"

# Preferred display order; any method not listed is appended alphabetically.
_METHOD_ORDER = ["student_t", "gaussian", _ARMA_METHOD, _RW_METHOD]

# Number of extreme quarters (by |realised|) to surface for the Volatility
# Paradox discussion.
_N_EXTREME = 4


# ─── Paths & discovery ─────────────────────────────────────────────────────────

def _csv_dir(config_name: str) -> str:
    return os.path.join(
        str(get_project_root()), "output", "forecast_realtime", "csv", config_name
    )


def _period_label(path: str) -> str:
    """'rolling_nowcast_2008-01_2009-12.csv' -> '2008-01_2009-12'."""
    base = os.path.basename(path)
    stem = base[len("rolling_nowcast_"):] if base.startswith("rolling_nowcast_") else base
    return stem[:-4] if stem.endswith(".csv") else stem


def discover_csvs(config_name: str, periods: list[str] | None) -> list[tuple[str, str]]:
    """
    Find rolling-nowcast CSVs for a config.  Returns [(period_label, path), ...].

    If `periods` is given, keep only those whose label matches; otherwise return
    all discovered CSVs (sorted by label).
    """
    cands = sorted(glob.glob(os.path.join(_csv_dir(config_name), "rolling_nowcast_*.csv")))
    pairs = [(_period_label(p), p) for p in cands]
    if periods:
        wanted = set(periods)
        pairs = [(lbl, p) for (lbl, p) in pairs if lbl in wanted]
        missing = wanted - {lbl for lbl, _ in pairs}
        if missing:
            avail = ", ".join(lbl for lbl, _ in [(_period_label(p), p) for p in cands])
            raise SystemExit(
                f"Period(s) not found: {sorted(missing)}.\n  Available: {avail}"
            )
    return pairs


def load_long(config_name: str, periods: list[str] | None) -> tuple[pd.DataFrame, int, int]:
    """
    Union the selected CSVs into one long DataFrame with a `period` column.

    The error is recomputed as nowcast_livello - gdp_realizzato (self-contained).
    Rows whose realised GDP is still missing (recent quarters) are dropped, and
    counted: returns (df_scored, n_total_rows, n_dropped_no_realised).
    """
    pairs = discover_csvs(config_name, periods)
    if not pairs:
        raise SystemExit(
            f"No rolling-nowcast CSV found in {_csv_dir(config_name)}.\n"
            f"Run: python -m src.forecast.rolling_nowcast --{config_name} "
            f"--start YYYY-MM --end YYYY-MM"
        )
    frames = []
    for label, path in pairs:
        d = pd.read_csv(path)
        d["period"] = label
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["target_quarter"] = df["target_quarter"].astype(str)
    df["horizon_month"] = df["horizon_month"].astype(int)

    n_total = len(df)
    df["error"] = df["nowcast_livello"] - df["gdp_realizzato"]
    scored = df[df["gdp_realizzato"].notna() & df["nowcast_livello"].notna()].copy()
    n_dropped = n_total - len(scored)
    return scored, n_total, n_dropped


# ─── Metric primitives ─────────────────────────────────────────────────────────

def _safe_corr(nowcast: pd.Series, realised: pd.Series) -> float:
    """Pearson corr, robust to n<2 and zero-variance inputs."""
    a = nowcast.to_numpy(dtype=float)
    b = realised.to_numpy(dtype=float)
    if a.size < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(g: pd.DataFrame, has_z: bool) -> dict:
    err = g["error"].to_numpy(dtype=float)
    n = err.size
    out = {
        "n": n,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if n else float("nan"),
        "MAE": float(np.mean(np.abs(err))) if n else float("nan"),
        "Bias": float(np.mean(err)) if n else float("nan"),
        "corr": _safe_corr(g["nowcast_livello"], g["gdp_realizzato"]),
    }
    if has_z:
        z = g["nowcast_z"]
        out["mean_z"] = float(z.mean()) if z.notna().any() else float("nan")
        out["mean_abs_z"] = float(z.abs().mean()) if z.notna().any() else float("nan")
    return out


def _ordered_methods(present: list[str]) -> list[str]:
    known = [m for m in _METHOD_ORDER if m in present]
    extra = sorted(m for m in present if m not in _METHOD_ORDER)
    return known + extra


# ─── Metric tables ──────────────────────────────────────────────────────────────

def table_by_method_horizon(df: pd.DataFrame, has_z: bool) -> pd.DataFrame:
    """One row per (method, horizon) with all metrics + relative RMSE."""
    rows = []
    for (method, h), g in df.groupby(["method", "horizon_month"]):
        rec = {"method": method, "horizon": int(h)}
        rec.update(_metrics(g, has_z))
        rows.append(rec)
    tab = pd.DataFrame(rows)

    # Relative RMSE vs each benchmark, matched at the SAME horizon.
    def _bench_rmse_by_h(bench: str) -> dict[int, float]:
        b = tab[tab["method"] == bench]
        return dict(zip(b["horizon"], b["RMSE"]))

    rw_by_h = _bench_rmse_by_h(_RW_METHOD)
    arma_by_h = _bench_rmse_by_h(_ARMA_METHOD)
    tab["RMSE_rel_rw"] = tab.apply(
        lambda r: r["RMSE"] / rw_by_h[r["horizon"]]
        if rw_by_h.get(r["horizon"]) else float("nan"), axis=1)
    tab["RMSE_rel_arma"] = tab.apply(
        lambda r: r["RMSE"] / arma_by_h[r["horizon"]]
        if arma_by_h.get(r["horizon"]) else float("nan"), axis=1)

    order = _ordered_methods(list(tab["method"].unique()))
    tab["__mo"] = tab["method"].map({m: i for i, m in enumerate(order)})
    tab = tab.sort_values(["__mo", "horizon"]).drop(columns="__mo").reset_index(drop=True)
    return tab


def table_by_method(df: pd.DataFrame, has_z: bool) -> pd.DataFrame:
    """One row per method, pooled over horizons and dates, + relative RMSE."""
    rows = []
    for method, g in df.groupby("method"):
        rec = {"method": method}
        rec.update(_metrics(g, has_z))
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("method")

    rw_rmse = tab["RMSE"].get(_RW_METHOD, float("nan"))
    arma_rmse = tab["RMSE"].get(_ARMA_METHOD, float("nan"))
    tab["RMSE_rel_rw"] = tab["RMSE"] / rw_rmse if rw_rmse and not np.isnan(rw_rmse) else float("nan")
    tab["RMSE_rel_arma"] = tab["RMSE"] / arma_rmse if arma_rmse and not np.isnan(arma_rmse) else float("nan")

    order = _ordered_methods(list(tab.index))
    return tab.reindex(order).reset_index()


def table_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE per method x period (pooled over horizons) — crisis vs calm view."""
    rmse = (
        df.assign(sq_err=df["error"] ** 2)
        .groupby(["method", "period"])["sq_err"]
        .mean()
        .pow(0.5)
        .rename("RMSE")
        .reset_index()
    )
    pivot = rmse.pivot(index="method", columns="period", values="RMSE")
    order = _ordered_methods(list(pivot.index))
    return pivot.reindex(order)


def table_extremes(df: pd.DataFrame, has_z: bool) -> pd.DataFrame | None:
    """
    For the quarters with the largest |realised|, show each method's nowcast at
    the DEEPEST available horizon (closest to release) — the Volatility-Paradox
    snapshot: in extreme quarters, does the standardised nowcast stay compressed?
    """
    if not has_z:
        return None
    realised = (
        df.groupby("target_quarter")["gdp_realizzato"].first()
        .abs().sort_values(ascending=False)
    )
    top = realised.head(_N_EXTREME).index.tolist()
    rows = []
    for tq in top:
        sub = df[df["target_quarter"] == tq]
        hmax = sub["horizon_month"].max()
        deep = sub[sub["horizon_month"] == hmax]
        real = float(deep["gdp_realizzato"].iloc[0])
        for method in _ordered_methods(list(deep["method"].unique())):
            r = deep[deep["method"] == method]
            if r.empty:
                continue
            rows.append({
                "target": tq,
                "realised": real,
                "h": int(hmax),
                "method": method,
                "nowcast": float(r["nowcast_livello"].iloc[0]),
                "nowcast_z": float(r["nowcast_z"].iloc[0]),
                "error": float(r["error"].iloc[0]),
            })
    return pd.DataFrame(rows) if rows else None


# ─── Formatting ─────────────────────────────────────────────────────────────────

_FLOAT_FMT = {
    "RMSE": "{:.4f}", "MAE": "{:.4f}", "Bias": "{:+.4f}", "corr": "{:+.3f}",
    "RMSE_rel_rw": "{:.3f}", "RMSE_rel_arma": "{:.3f}",
    "mean_z": "{:+.3f}", "mean_abs_z": "{:.3f}",
    "realised": "{:+.4f}", "nowcast": "{:+.4f}", "nowcast_z": "{:+.3f}", "error": "{:+.4f}",
}


def _fmt(tab: pd.DataFrame) -> str:
    out = tab.copy()
    for col, fmt in _FLOAT_FMT.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    return out.to_string(index=False)


def _section(title: str) -> str:
    return "\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    def _extra(p):
        p.add_argument("--periods", default=None,
                       help="comma-separated period labels to include "
                            "(e.g. 2008-01_2009-12,2020-01_2020-12); default: all present")
        p.add_argument("--out-dir", default=None,
                       help="where to write metrics files (default: the config CSV folder)")

    args = parse_config_args("Compute nowcast accuracy metrics from rolling-nowcast CSVs.",
                             extra=_extra)
    periods = [s.strip() for s in args.periods.split(",")] if args.periods else None

    df, n_total, n_dropped = load_long(args.config, periods)
    has_z = "nowcast_z" in df.columns and df["nowcast_z"].notna().any()

    pairs = discover_csvs(args.config, periods)
    methods = _ordered_methods(list(df["method"].unique()))
    horizons = [int(h) for h in sorted(df["horizon_month"].unique())]

    header = (
        f"NOWCAST METRICS  (config: {args.config})\n"
        f"periods ({len(pairs)}): {', '.join(lbl for lbl, _ in pairs)}\n"
        f"methods: {methods}\n"
        f"horizons: {horizons}\n"
        f"observations scored: {len(df)} / {n_total}  "
        f"(dropped {n_dropped} rows with no realised GDP yet)\n"
        f"z-score column present: {'yes' if has_z else 'NO — nowcast_z absent/all-NaN'}"
    )

    t_mh = table_by_method_horizon(df, has_z)
    t_m = table_by_method(df, has_z)
    t_p = table_by_period(df)
    t_ext = table_extremes(df, has_z)

    blocks = [
        header,
        _section("1. BY METHOD x HORIZON  (RMSE_rel_* matched at same horizon; "
                 "<1 = beats benchmark)"),
        _fmt(t_mh),
        _section("2. BY METHOD  (pooled over horizons & dates)"),
        _fmt(t_m),
    ]
    if t_p.shape[1] > 1:
        blocks += [_section("3. RMSE BY METHOD x PERIOD  (crisis vs calm)"),
                   _fmt(t_p.reset_index())]
    if t_ext is not None:
        blocks += [_section(f"4. EXTREME QUARTERS (top {_N_EXTREME} by |realised|) — "
                            "Volatility Paradox: nowcast_z stays compressed?"),
                   _fmt(t_ext)]
    if not has_z:
        blocks += ["\nNOTE: nowcast_z is absent or all-NaN in these CSVs. "
                   "The z metric (mean_z / Volatility-Paradox snapshot) was skipped. "
                   "Re-run rolling_nowcast to repopulate nowcast_z if you need it."]

    report = "\n".join(blocks)
    print(report)

    out_dir = args.out_dir or _csv_dir(args.config)
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "metrics_summary.txt")
    csv_path = os.path.join(out_dir, "metrics_by_method_horizon.csv")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    t_mh.to_csv(csv_path, index=False)
    print(f"\nwrote: {txt_path}")
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
