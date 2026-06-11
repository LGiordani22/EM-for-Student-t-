"""
src/forecast/panel_builder.py

Real-time PANEL BUILDER for the nowcasting pipeline (Second brick).

Given an "as of" publication date, assembles the model panel in the EXACT same
format as the in-sample dataset on which theta_star is calibrated, so it can be
fed straight into fit_dfm.  This module does NOT estimate the DFM or produce
nowcasts — those are later bricks.

What it combines (all via data_import.py functions)
----------------------------------------------------
  * FRED-MD series   -> load_fredmd_vintage(as_of, config)
    Real vintage, already transformed, real ragged edge.
    small: 18 series.  big: 36 series (F2a).
  * Current series   -> get_current_vintage(sid, as_of, config) per series
    Current (revised) values truncated to publication-lag availability.
    small: GDPC1 + NFCI.  big: GDPC1 + NFCI + 9 additional (F2b).
  * Spread columns   -> NaN placeholder until F2c.
    small: none.  big: term_spread, baa_spread, aaa_spread (all NaN for now).

Target format
-------------
  * index   : monthly, MONTH-END, from `start` to the last available month.
  * columns : ORDERED_COLS order from the config (real / financial / other,
              GDPC1 last).
  * GDPC1   : placed only at quarter-end months; NaN off quarter-ends.
  * ragged edge : each series ends at its real-time publication cutoff.

HYBRID CAVEAT (data_import.py): FRED-MD series use TRUE real-time vintages;
current series use REVISED values with reconstructed temporal availability.
Spread series (F2c) will be computed from vintage component levels once
implemented; they are NaN until then.
"""

from __future__ import annotations

import re

import pandas as pd

from src.data_loader import ORDERED_COLS, SAMPLE_START, load_config, apply_transform
from src.forecast.data_import import (
    load_fredmd_vintage,
    load_fredmd_raw_levels,
    get_current_vintage,
    gdp_available_through,
)


# Rolling-nowcast convention: the information set is photographed on the 15th of
# each month.  This matches the real publication calendar — the FRED-MD vintage
# of month M is released by McCracken ~mid-M (so it is legitimately available on
# the 15th), while the BEA advance GDP of the just-closed quarter is released
# ~end of the following month (so it is NOT yet out at mid-month, and the target
# quarter's GDP stays hidden).  See src/forecast/_inspect_asof*.py.
_ASOF_DAY = 15


def _as_of_day_aware(as_of_date) -> pd.Timestamp:
    """
    Resolve an as_of input to a DAY-AWARE timestamp under the day-15 convention.

    Month-granular inputs — a (year, month) tuple, or a 'YYYY-MM' string with no
    day — are interpreted as the 15th of that month (the publication convention
    above).  A full date (with a day) keeps its day.  The DAY matters: it decides
    whether the just-closed quarter's GDP advance is already out (GDP cut) and
    which is the last complete NFCI month, but it does NOT affect the FRED-MD
    vintage FILE, which is chosen by (year, month) only.
    """
    if isinstance(as_of_date, tuple):
        y, m = int(as_of_date[0]), int(as_of_date[1])
        return pd.Timestamp(y, m, _ASOF_DAY)
    if isinstance(as_of_date, str) and re.fullmatch(r"\s*\d{4}-\d{1,2}\s*", as_of_date):
        y, m = as_of_date.strip().split("-")
        return pd.Timestamp(int(y), int(m), _ASOF_DAY)
    return pd.Timestamp(as_of_date)


def build_panel(as_of_date, config_name: str = "small", start: str = SAMPLE_START) -> pd.DataFrame:
    """
    Build the real-time model panel as known at `as_of_date`.

    Parameters
    ----------
    as_of_date : str | datetime | (year, month) tuple
        Vintage publication date.  Month-granular inputs ("2006-01", (2006, 1))
        are interpreted as the 15th of that month (rolling-nowcast convention);
        a full date keeps its day.  See _as_of_day_aware.
    config_name : str
        "small" (20 series: 18 FRED-MD + GDPC1 + NFCI) or
        "big"   (50 series: 36 FRED-MD + 11 current + 3 spread-NaN; F2b).
    start : str
        First month to keep (default "1985-01-01").

    Returns
    -------
    panel : pd.DataFrame
        Shape (T, M).  Month-end DatetimeIndex, columns = ORDERED_COLS.
        GDPC1 only at quarter-end months.  Real ragged edge at the bottom.
        For the big config, spread columns (term_spread, baa_spread,
        aaa_spread) are computed from vintage raw rate levels.
    """
    cfg = load_config(config_name)
    ordered_cols = cfg["ORDERED_COLS"]
    current_series_list = cfg["CURRENT_SERIES"]   # all source="current" names
    spread_defs = cfg["SPREAD_DEFS"]              # list of {name, components, transform}

    as_of = _as_of_day_aware(as_of_date)
    start_me = pd.Timestamp(start) + pd.offsets.MonthEnd(0)

    # 1. FRED-MD series — real vintage (already transformed, true ragged edge).
    fredmd = load_fredmd_vintage(as_of_date, config_name=config_name, start=start)

    # 2. Current series — revised values truncated to real-time availability.
    #    Late-start series (JTSJOL, T10YIE) return empty Series for early as_of
    #    dates; reindex below turns them into all-NaN columns.
    current_data: dict[str, pd.Series] = {}
    for sid in current_series_list:
        s = get_current_vintage(sid, as_of, config_name=config_name)
        current_data[sid] = s.loc[s.index >= start_me] if len(s) else s

    # 3. Master monthly month-end index spanning all available data.
    ends = [fredmd.index.max()]
    for s in current_data.values():
        if len(s):
            ends.append(s.index.max())
    last = max(ends)
    master = pd.date_range(start=start_me, end=last, freq="ME")

    # 4. Assemble panel: FRED-MD, then current series.
    panel = pd.DataFrame(index=master)
    for col in fredmd.columns:
        panel[col] = fredmd[col].reindex(master)
    for sid, s in current_data.items():
        panel[sid] = s.reindex(master)          # NaN off quarter-ends for GDPC1

    # 5. Constructed spreads from vintage raw component levels.
    #    spread = level(a) - level(b)  [raw rates in %], then apply spread's
    #    transform (t1 = level, kept in level: spread is already I(0)).
    #    Ragged edge: NaN wherever either component level is NaN.
    if spread_defs:
        comp_names = list({c for sd in spread_defs for c in sd["components"]})
        raw = load_fredmd_raw_levels(as_of_date, comp_names)
        for sd in spread_defs:
            name = sd["name"]
            a, b = sd["components"]
            level = raw[a] - raw[b]
            spread = apply_transform(level, sd["transform"])
            spread.name = name
            panel[name] = spread.reindex(master)

    panel = panel[ordered_cols]
    panel.index.name = None
    return panel


__all__ = ["build_panel"]


# ─── Smoke tests / diagnostics ────────────────────────────────────────────────
# Run from the project root with:   python -m src.forecast.panel_builder
# All local: real vintages on disk + current processed dataset.  No network.

_KEY_SERIES = ["INDPRO", "PAYEMS", "S&P 500", "NFCI"]


def _hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _quarter_label(ts: pd.Timestamp) -> str:
    return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"


def _last_obs(panel: pd.DataFrame, col: str):
    s = panel[col].dropna()
    return s.index[-1] if len(s) else None


def _diagnose(as_of, panel: pd.DataFrame) -> None:
    print(f"  shape      : {panel.shape[0]} months x {panel.shape[1]} series")
    print(f"  date range : {panel.index[0].date()} .. {panel.index[-1].date()}")

    # last observed month for a few key series
    for col in _KEY_SERIES:
        lo = _last_obs(panel, col)
        print(f"  last obs   : {col:<10s} -> {lo.date() if lo is not None else 'none'}")

    # last available GDP quarter
    g_last = _last_obs(panel, "GDPC1")
    if g_last is not None:
        print(f"  last GDP   : {_quarter_label(g_last)} ({g_last.date()}), "
              f"log-diff = {panel.loc[g_last, 'GDPC1']:.4f}")

    # ragged-edge profile: observed count over the last 4 months
    print("  ragged edge profile (observed / 20) over the last 4 months:")
    for d, row in panel.tail(4).iterrows():
        obs = int(row.notna().sum())
        present = [c for c in panel.columns if pd.notna(row[c])]
        print(f"    {d.date()}  {obs:2d}/20   present: {present}")


def _check_format(panel: pd.DataFrame, insample: pd.DataFrame) -> None:
    cols_ok = list(panel.columns) == list(insample.columns)
    me_ok = bool((panel.index == (panel.index + pd.offsets.MonthEnd(0))).all())
    print(f"  columns identical to dataset_small.csv (same set + order): "
          f"{'OK' if cols_ok else 'FAIL'}")
    if not cols_ok:
        print(f"    panel   : {list(panel.columns)}")
        print(f"    insample: {list(insample.columns)}")
    print(f"  index is month-end monthly: {'OK' if me_ok else 'FAIL'}")
    # GDP only on quarter-end months?
    gdp_off_q = panel.loc[~panel.index.month.isin([3, 6, 9, 12]), "GDPC1"]
    gdp_ok = bool(gdp_off_q.isna().all())
    print(f"  GDPC1 NaN on all non-quarter-end months: {'OK' if gdp_ok else 'FAIL'}")


def _check_consecutive(p_jan: pd.DataFrame, p_feb: pd.DataFrame) -> None:
    """The whole point of real-time: the ragged edge advances one month."""
    jan_m = pd.Timestamp("2006-01-31")
    print("  INDPRO for Jan-2006 (2006-01-31):")
    in_jan = jan_m in p_jan.index and pd.notna(p_jan.loc[jan_m, "INDPRO"])
    in_feb = jan_m in p_feb.index and pd.notna(p_feb.loc[jan_m, "INDPRO"])
    print(f"    present in panel as-of 2006-01 : {in_jan}  (expected False -> not yet released)")
    print(f"    present in panel as-of 2006-02 : {in_feb}  (expected True  -> released by Feb)")
    edge_jan = _last_obs(p_jan, "INDPRO")
    edge_feb = _last_obs(p_feb, "INDPRO")
    advanced = (edge_feb is not None and edge_jan is not None
                and edge_feb > edge_jan)
    print(f"    INDPRO ragged edge: {edge_jan.date()} (Jan panel) -> "
          f"{edge_feb.date()} (Feb panel)  "
          f"-> {'OK (advanced)' if advanced else 'FAIL (did not advance)'}")
    verdict = (not in_jan) and in_feb and advanced
    print(f"  REAL-TIME CHECK -> {'OK' if verdict else 'FAIL'}")


def _check_consistency(panel: pd.DataFrame, insample: pd.DataFrame) -> None:
    """
    For well-aged months (<= 2005-06) the FRED-MD values in the 2006-01 panel
    should be close to the in-sample (current-vintage) values.  Small gaps are
    normal (data revisions); flag only large ones.
    """
    fredmd_cols = [c for c in panel.columns if c not in ("NFCI", "GDPC1")]
    aged = panel.index[panel.index <= pd.Timestamp("2005-06-30")]
    common = aged.intersection(insample.index)
    a = panel.loc[common, fredmd_cols]
    b = insample.loc[common, fredmd_cols]
    diff = (a - b).abs()
    max_abs = diff.max().sort_values(ascending=False)
    print(f"  comparing {len(common)} aged months (<= 2005-06), {len(fredmd_cols)} "
          f"FRED-MD series, vintage 2006-01 vs current in-sample:")
    print("  top 5 series by max |difference|:")
    for col, v in max_abs.head(5).items():
        print(f"    {col:<14s} max|diff| = {v:.5f}")
    THRESH = 0.05
    big = max_abs[max_abs > THRESH]
    if len(big) == 0:
        print(f"  all series within |diff| <= {THRESH} -> OK (revisions are small)")
    else:
        print(f"  WARN: {len(big)} series exceed |diff| {THRESH} "
              f"(likely benchmark revisions, inspect): {list(big.index)}")


if __name__ == "__main__":
    insample = pd.read_csv("data/processed/dataset_small.csv", index_col=0)
    insample.index = pd.to_datetime(insample.index)

    _hr("panel_builder.py smoke tests")
    print(f"in-sample target: {insample.shape[0]} months x {insample.shape[1]} series, "
          f"{insample.index[0].date()} .. {insample.index[-1].date()}")

    example_dates = ["2006-01", "2006-02", "2008-11", "2020-05", "2024-12"]
    panels: dict[str, pd.DataFrame] = {}
    for d in example_dates:
        _hr(f"build_panel('{d}')")
        panels[d] = build_panel(d)
        _diagnose(d, panels[d])

    _hr("FORMAT CHECK  (panel 2024-12 vs dataset_usa.csv)")
    _check_format(panels["2024-12"], insample)

    _hr("REAL-TIME CHECK  (2006-01 vs 2006-02: ragged edge must advance)")
    _check_consecutive(panels["2006-01"], panels["2006-02"])

    _hr("CONSISTENCY CHECK  (aged FRED-MD values: vintage 2006-01 vs in-sample)")
    _check_consistency(panels["2006-01"], insample)

    _hr("Done.")
