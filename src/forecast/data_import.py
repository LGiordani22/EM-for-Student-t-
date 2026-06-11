"""
src/forecast/data_import.py

Real-time DATA IMPORT for the nowcasting pipeline (First brick).

This module ONLY imports raw real-time data.  It does NOT build the model
panel, estimate the DFM, or produce nowcasts — those are later bricks.

Two sources
-----------
A) Historical FRED-MD vintages (the 18 monthly FRED-MD series of the model)
   -> load_fredmd_vintage(as_of_date)

   The two folders under data/raw/ each hold one "Historical vintage" file
   per publication month.  The FILE NAME is the publication ("as of") date,
   e.g. 2008-01.csv was released in Jan 2008 and contains the COMPLETE
   history (from 1959) up to that release, WITH the real ragged edge of the
   time (last data month lags the release month by ~1 month).

B) CURRENT (revised) GDPC1 and NFCI, which are NOT in FRED-MD
   -> get_current_vintage(series_id, as_of_date)

   These two are read from the CURRENT processed dataset that data_loader.py
   builds (data/processed/dataset_usa.csv), so they are bit-for-bit identical
   to the in-sample series and need no extra FRED/ALFRED download.  They carry
   the data_loader transform already (GDPC1: log-diff*100 at quarter-end;
   NFCI: monthly-mean level).

   We deliberately do NOT use ALFRED real-time vintages here: ALFRED's NFCI
   real-time vintages only start in 2011, so a pre-2011 real-time NFCI cannot
   be reconstructed.  For coherence, GDPC1 and NFCI therefore both come from
   the current data.

   HYBRID design — caveat to declare in the thesis
   ------------------------------------------------
   The 18 FRED-MD series use REAL real-time vintages (right values + right
   availability as of each date).  GDPC1 and NFCI instead use CURRENT (revised)
   VALUES, but their temporal AVAILABILITY is reconstructed to match the real
   publication calendar (see get_current_vintage / gdp_available_through).
   So for GDPC1/NFCI there is look-ahead on the VALUES (revised, not the
   numbers known at the time) but NOT on availability (the timing rule prevents
   using a quarter's GDP before it would have been released).  This asymmetry
   is a known limitation to state in the thesis; moving GDPC1 to true ALFRED
   real-time vintages is a future upgrade.

Series-alignment notes discovered during data inspection (2026-06-03/06-08)
---------------------------------------------------------------------------
Most FRED-MD series appear in all vintages with a stable name and transform
code.  The following need special handling:

  * TWEXAFEGSMTHx (trade-weighted USD, transform 5) only exists from the
    2020-04 vintage onward.  Earlier vintages (1999-08..2020-03) carry its
    predecessor TWEXMMTH (also transform 5).  FRED-MD swapped the
    discontinued "major currencies" index for the "Advanced Foreign
    Economies, goods & services" index in April 2020.  They are conceptually
    similar broad-USD indices but NOT the identical underlying series — there
    is a definitional break at 2020-04 to be noted in the thesis.  This
    module bridges the two names transparently via VINTAGE_NAME_ALIASES.

  * VIXCLSx (VIX index, transform 1) — big config only.  In vintages up to
    2021-11, FRED-MD published the series under the name VXOCLSx (the
    CBOE VXO index, measuring implied volatility on the S&P 100).  From the
    2021-12 vintage onward, FRED-MD switched to VIXCLSx (the S&P 500-based
    VIX).  Additionally, the 8 vintages 2015-01..2015-08 carry NEITHER name
    (both absent — a documented FRED-MD gap).  CAVEAT for the thesis:
    VXO (S&P 100) ≠ VIX (S&P 500); the two series are highly correlated but
    rest on different option baskets — there is a definitional break at the
    2021-12 vintage.  This module bridges the two names via
    VINTAGE_NAME_ALIASES; the 2015-01..2015-08 gap yields NaN for VIXCLSx
    (handled gracefully — Kalman imputes).

  * PCEPI (PCE price index, transform 6) is absent from the 12 earliest
    vintages (1999-08..2000-07); FRED-MD only began including it in 2000-08.
    There is no predecessor.  Because PCEPI cannot be recovered for an as_of
    in that early window, we declare 2000-08 the FIRST USABLE vintage and make
    that an EXPLICIT, documented contract: requesting any earlier as_of raises
    a clear ValueError (see FIRST_USEFUL_VINTAGE / _resolve_vintage_path)
    rather than relying on allow_missing.  Nothing is lost as training data:
    the 2000-08 vintage already carries the COMPLETE history from 1959 (PCEPI
    included), so the model's in-sample window is unaffected.

  * IPDCONGD / IPBUSEQ (IP sub-indices, transform 5) — big config only.
    Both are absent from vintages before 2002-12 (first appear 2002-12).
    For vintages that predate their availability the column is set to NaN;
    since all rolling-nowcast periods start from 2001 at the earliest and
    these series appear in 2002-12, only the dotcom_2001 period is affected
    (Jan–Dec 2001 as_of dates → IPDCONGD/IPBUSEQ all NaN for that period).

  * WPSFD49207 (PPI proxy, transform 6) — big config only.  Absent from
    vintages before 2016-03 (first appears 2016-03).  Affects crisi_2008,
    dotcom_2001, debito_eu_2011, calma_2015, and dazi_2018 periods where
    all vintages predate 2016-03.  Kalman handles the all-NaN column.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# Reuse the canonical dataset specification + helpers from data_loader so the
# real-time panel stays bit-for-bit consistent with the in-sample dataset.
from src.data_loader import (
    ALL_FREDMD_COLS,      # the 18 FRED-MD model series (small config), in canonical order
    SAMPLE_START,         # "1985-01-01"
    apply_transform,      # FRED-MD transformation-code engine
    _to_month_end,        # snap month-start dates to month-end
    load_config,          # config-aware series specification
)

# ─── Vintage folder locations ─────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

VINTAGE_DIRS: list[str] = [
    os.path.join(_RAW_DIR, "Historical-vintages-of-FRED-MD-1999-08-to-2014-12"),
    os.path.join(_RAW_DIR, "Historical-vintages-of-FRED-MD-2015-01-to-2024-12"),
    # 15 files: 2025-01 .. 2026-03.
    os.path.join(_RAW_DIR, "Historical-vintages-of-FRED-MD-2025-01-to-2026-03"),
]

# Three file-naming conventions are present in the folders:
#   "2008-01.csv"            (most files; also 2025-01 .. 2025-03)
#   "FRED-MD_2024m12.csv"    (the 2024-03 .. 2024-12 files)
#   "2025-04-MD.csv"         (the 2025-04 .. 2026-03 files, "-MD" suffix)
# All three are parsed into a (year, month) key.
_NAME_PATTERNS = [
    re.compile(r"^(\d{4})-(\d{2})\.csv$", re.IGNORECASE),
    re.compile(r"^FRED-MD_(\d{4})m(\d{2})\.csv$", re.IGNORECASE),
    re.compile(r"^(\d{4})-(\d{2})-MD\.csv$", re.IGNORECASE),
]

# ─── Series-name aliasing across vintages ─────────────────────────────────────
# Maps a model (canonical) series name to the source column name that actually
# carries it in a given vintage.  Each entry is a list of
# (valid_until_inclusive (year, month), source_name) rules, tried in order; the
# first rule whose cutoff is >= the vintage date wins.  None as cutoff = "always".
#
# CAVEAT (TWEX, methodological — to declare in the thesis):
# The two names below are NOT the same underlying series; they are bridged as
# one continuous column only as a deliberate modelling choice.  TWEXMMTH is the
# (discontinued) trade-weighted USD index vs the "major currencies" basket;
# TWEXAFEGSMTHx is the trade-weighted USD index vs the "Advanced Foreign
# Economies (goods & services)" basket.  Both are broad-USD exchange-rate
# indices with transform code 5, conceptually similar, but they rest on
# DIFFERENT currency baskets and definitions.  FRED-MD swapped one for the
# other at the 2020-04 vintage, so there is a DEFINITIONAL BREAK at 2020-04
# inside this single column — a caveat that must be stated in the thesis.
VINTAGE_NAME_ALIASES: dict[str, list[tuple[tuple[int, int] | None, str]]] = {
    "TWEXAFEGSMTHx": [
        ((2020, 3), "TWEXMMTH"),        # 1999-08 .. 2020-03 vintages: "major currencies" index
        (None,      "TWEXAFEGSMTHx"),   # 2020-04 onward: "advanced foreign economies" index
    ],
    # CAVEAT (VIX, methodological — to declare in the thesis):
    # VXOCLSx (CBOE VXO, S&P 100 implied volatility) and VIXCLSx (CBOE VIX,
    # S&P 500 implied volatility) are related but NOT the same underlying series;
    # they are bridged here as a modelling choice.  FRED-MD carried VXOCLSx up to
    # the 2021-11 vintage, then switched to VIXCLSx from 2021-12.  Additionally,
    # the 8 vintages 2015-01..2015-08 carry NEITHER name (genuine FRED-MD gap);
    # during that window both the alias source and the canonical name are absent
    # and the column is NaN (see _KNOWN_LATE_SERIES / graceful-NaN logic below).
    "VIXCLSx": [
        ((2021, 11), "VXOCLSx"),        # 1999-08 .. 2021-11: VXO (S&P 100) definition
        (None,       "VIXCLSx"),        # 2021-12 onward: native VIX (S&P 500) definition
    ],
}

# Series known to be absent (no predecessor, or late addition) before a given
# vintage.  Any series listed here is silently NaN-filled when absent from a
# vintage — no error is raised.  The Kalman filter handles NaN columns.
_KNOWN_LATE_SERIES: dict[str, tuple[int, int]] = {
    "PCEPI":      (2000, 8),   # first available in the 2000-08 vintage
    "IPDCONGD":   (2002, 12),  # big config only; first available in the 2002-12 vintage
    "IPBUSEQ":    (2002, 12),  # big config only; first available in the 2002-12 vintage
    "WPSFD49207": (2016, 3),   # big config only; first available in the 2016-03 vintage
}

# First vintage that can produce a COMPLETE model panel, hence the first
# publication date we treat as usable.  PCEPI (a model series with no
# predecessor) is absent from the 12 earliest vintages (1999-08 .. 2000-07)
# and first appears in 2000-08, which already carries the full history from
# 1959 (PCEPI included).  Discarding the pre-2000-08 vintages therefore costs
# no training data.  We enforce this as an explicit contract: requesting an
# earlier as_of raises a clear error (see _resolve_vintage_path) instead of
# silently leaning on allow_missing to return an all-NaN PCEPI column.
FIRST_USEFUL_VINTAGE: tuple[int, int] = (2000, 8)


# ─── Vintage discovery ────────────────────────────────────────────────────────

def _parse_vintage_name(fname: str) -> tuple[int, int] | None:
    """Return (year, month) for a vintage filename, or None if it is not one."""
    for pat in _NAME_PATTERNS:
        m = pat.match(fname)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def build_vintage_index() -> dict[tuple[int, int], str]:
    """
    Scan both vintage folders and map (year, month) -> absolute file path.

    Handles both file-naming conventions and the two-folder split
    transparently, so callers never need to know which folder a vintage lives
    in or how it is named.
    """
    index: dict[tuple[int, int], str] = {}
    for folder in VINTAGE_DIRS:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            key = _parse_vintage_name(fname)
            if key is not None:
                index[key] = os.path.join(folder, fname)
    if not index:
        raise FileNotFoundError(
            "No FRED-MD vintage files found. Expected folders:\n  "
            + "\n  ".join(VINTAGE_DIRS)
        )
    return index


def list_available_vintages(include_unusable: bool = False) -> list[tuple[int, int]]:
    """
    Sorted list of (year, month) vintages available for loading.

    By default only USABLE vintages are returned (>= FIRST_USEFUL_VINTAGE):
    the pre-2000-08 vintages are on disk but cannot yield a complete panel
    (PCEPI absent), so they are excluded as useful publication dates.  Pass
    include_unusable=True to see every file present on disk.
    """
    keys = sorted(build_vintage_index().keys())
    if include_unusable:
        return keys
    return [ym for ym in keys if ym >= FIRST_USEFUL_VINTAGE]


def _as_year_month(as_of_date) -> tuple[int, int]:
    """Coerce a date-like input to a (year, month) tuple."""
    if isinstance(as_of_date, tuple):
        return int(as_of_date[0]), int(as_of_date[1])
    ts = pd.Timestamp(as_of_date)
    return ts.year, ts.month


def _resolve_vintage_path(as_of_date) -> tuple[str, tuple[int, int]]:
    """
    Find the vintage file for a given as_of month.

    Raises a clear error (listing the available range) if the month has no
    vintage on disk.
    """
    ym = _as_year_month(as_of_date)

    # Explicit, documented contract: vintages before FIRST_USEFUL_VINTAGE
    # (2000-08) lack PCEPI and cannot produce a complete panel.  Block them
    # with a clear reason rather than letting allow_missing silently paper
    # over it with an all-NaN PCEPI column.
    if ym < FIRST_USEFUL_VINTAGE:
        fy, fm = FIRST_USEFUL_VINTAGE
        raise ValueError(
            f"Vintage {ym[0]}-{ym[1]:02d} predates the first usable FRED-MD "
            f"vintage {fy}-{fm:02d}.\n"
            f"  Reason: PCEPI (a model series with no predecessor) is absent "
            f"from every vintage before {fy}-{fm:02d}, so an earlier as_of "
            f"cannot yield a complete panel.\n"
            f"  No training data is lost: the {fy}-{fm:02d} vintage already "
            f"contains the full history from 1959 (PCEPI included)."
        )

    index = build_vintage_index()
    if ym in index:
        return index[ym], ym
    avail = list_available_vintages()
    raise FileNotFoundError(
        f"No FRED-MD vintage for {ym[0]}-{ym[1]:02d}.\n"
        f"  Usable range: {avail[0][0]}-{avail[0][1]:02d} .. "
        f"{avail[-1][0]}-{avail[-1][1]:02d}  ({len(avail)} vintages)."
    )


# ─── Vintage parsing ──────────────────────────────────────────────────────────

def _read_vintage_raw(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Parse one FRED-MD vintage CSV into (data, tcodes).

    Layout (identical across all 305 vintages):
        row 0 : 'Transform:', <code>, <code>, ...
        row 1+: <date>,       <value>, <value>, ...
    The date column is 'sasdate'; date strings may be MM/DD/YYYY (older
    vintages) or M/D/YYYY (newer) — pandas infers both.
    """
    raw = pd.read_csv(path, header=0, low_memory=False)
    date_col = raw.columns[0]                       # 'sasdate'
    tcodes = raw.iloc[0].drop(date_col).astype(float)

    data = raw.iloc[1:].copy()
    data = data.rename(columns={date_col: "date"})
    # Drop fully-empty trailing rows some vintages carry, then parse dates.
    data = data[data["date"].notna()]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].notna()].set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")
    return data, tcodes


def _source_name_for(canonical: str, ym: tuple[int, int]) -> str:
    """Resolve the source column name carrying `canonical` in vintage `ym`."""
    rules = VINTAGE_NAME_ALIASES.get(canonical)
    if rules is None:
        return canonical
    for cutoff, src in rules:
        if cutoff is None or ym <= cutoff:
            return src
    return canonical


def load_fredmd_vintage(
    as_of_date,
    config_name: str = "small",
    start: str | None = SAMPLE_START,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Load the FRED-MD model series as published at `as_of_date`.

    Parameters
    ----------
    as_of_date : str | datetime | (year, month) tuple
        Publication month of the vintage to load (e.g. "2008-01", (2008, 1)).
        The returned panel contains the COMPLETE transformed history up to
        that vintage's last data month — i.e. the real ragged edge of the
        time is preserved.
    config_name : str
        Config to use ("small" or "big"; default "small").
        "small": 18 FRED-MD series.
        "big":   36 FRED-MD series (18 additional: IPCONGD, IPDCONGD, IPBUSEQ,
                 CUMFNS, AMDMNOx, ANDENOx, UNRATE, AWHMAN, CES0600000007,
                 HOUST, PERMIT, TB3MS, GS5, GS10, BAA, AAA, VIXCLSx,
                 WPSFD49207).  Aliases (VIXCLSx->VXOCLSx pre-2021-12) and
                 late-start series (IPDCONGD/IPBUSEQ from 2002-12,
                 WPSFD49207 from 2016-03) are handled transparently: columns
                 absent from a vintage are NaN-filled (Kalman handles them).
    start : str | None
        If given (default "1985-01-01"), rows before this date are dropped
        (matches the in-sample dataset start).  Pass None to keep 1959+.
    allow_missing : bool
        Controls behaviour for series UNEXPECTEDLY absent from a vintage (not
        in _KNOWN_LATE_SERIES and not in VINTAGE_NAME_ALIASES):
          False -> raise a clear KeyError;
          True  -> return that column filled with NaN.
        Series in _KNOWN_LATE_SERIES or VINTAGE_NAME_ALIASES are ALWAYS
        silently NaN-filled regardless of this flag.

    Returns
    -------
    panel : pd.DataFrame
        Index = month-end dates.  Columns = FRED-MD series in canonical order.
        Values are TRANSFORMED with the vintage's own FRED-MD transform codes
        (so the ragged-edge NaNs at the bottom are the real publication lag).
    """
    fredmd_cols = load_config(config_name)["ALL_FREDMD_COLS"]

    path, ym = _resolve_vintage_path(as_of_date)
    data, tcodes = _read_vintage_raw(path)

    transformed: dict[str, pd.Series] = {}
    truly_missing: list[tuple[str, str, bool]] = []

    for canonical in fredmd_cols:
        src = _source_name_for(canonical, ym)
        if src in data.columns:
            s = apply_transform(data[src], int(tcodes[src]))
            s.name = canonical
            transformed[canonical] = s
            continue

        # Series not found under any known name in this vintage.
        # Determine whether this is a known/expected absence:
        #   (a) _KNOWN_LATE_SERIES: series with a documented late start date.
        #   (b) VINTAGE_NAME_ALIASES: series with alias rules whose resolved
        #       source is still absent (e.g. VIXCLSx gap 2015-01..2015-08).
        # Both cases are silently NaN-filled — Kalman handles missing columns.
        # Only truly unexpected absences are governed by allow_missing.
        late_cut = _KNOWN_LATE_SERIES.get(canonical)
        has_alias = canonical in VINTAGE_NAME_ALIASES
        known_reason = (late_cut is not None) or has_alias
        if allow_missing or known_reason:
            transformed[canonical] = pd.Series(np.nan, index=data.index, name=canonical)
        truly_missing.append((canonical, src, known_reason))

    hard_missing = [c for (c, _src, _k) in truly_missing if c not in transformed]
    if hard_missing:
        lines = []
        for c, src, known in truly_missing:
            if c in transformed:
                continue
            if known:
                lc = _KNOWN_LATE_SERIES.get(c)
                if lc:
                    lines.append(
                        f"  - {c}: absent before the {lc[0]}-{lc[1]:02d} vintage "
                        f"(late-start series, no predecessor)."
                    )
                else:
                    lines.append(
                        f"  - {c}: alias source '{src}' absent in vintage "
                        f"{ym[0]}-{ym[1]:02d} (known alias gap)."
                    )
            else:
                lines.append(
                    f"  - {c}: expected source column '{src}' not found in "
                    f"vintage {ym[0]}-{ym[1]:02d}. FRED-MD may have renamed it; "
                    f"add a rule to VINTAGE_NAME_ALIASES."
                )
        raise KeyError(
            f"\n{len(hard_missing)} model series missing in vintage "
            f"{ym[0]}-{ym[1]:02d}:\n" + "\n".join(lines)
        )

    panel = pd.DataFrame(transformed, index=data.index)[fredmd_cols]
    panel.index = _to_month_end(panel.index)

    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)].copy()

    return panel


def load_fredmd_raw_levels(as_of_date, series_ids: list[str]) -> pd.DataFrame:
    """
    Load raw (untransformed) FRED-MD levels for specific series from a vintage.

    Used to compute spread series from component rate levels BEFORE any
    transformation (e.g. GS10 and TB3MS as raw yields-in-percent, not
    first-differenced).  Alias resolution is applied (VXOCLSx -> VIXCLSx, etc.)
    so callers use canonical series names.

    Returns the FULL vintage history (no start-date filter); callers crop via
    reindex() on their master date index.  Series absent from the vintage
    return an all-NaN column (same graceful-NaN logic as load_fredmd_vintage).
    """
    path, ym = _resolve_vintage_path(as_of_date)
    data, _ = _read_vintage_raw(path)

    result: dict[str, pd.Series] = {}
    for canonical in series_ids:
        src = _source_name_for(canonical, ym)
        if src in data.columns:
            result[canonical] = data[src].copy()
        else:
            result[canonical] = pd.Series(np.nan, index=data.index, name=canonical)

    df = pd.DataFrame(result, index=data.index)
    df.index = _to_month_end(df.index)
    return df


# ─── Current (non-vintage) series with reconstructed publication timing ─────────
#
# These series are NOT in FRED-MD vintage files.  Rather than pull real-time
# ALFRED vintages (NFCI's ALFRED real-time vintages only begin in 2011, making
# pre-2011 reconstruction impossible), we read CURRENT (revised) values from the
# processed dataset that data_loader.py builds, then truncate to respect the real
# publication calendar at `as_of`.
#
# HYBRID DESIGN CAVEAT (declare in thesis):
# Values are REVISED (look-ahead on the numbers), but temporal AVAILABILITY is
# reconstructed from the publication-lag rules below — so the timing is real-time
# even if the exact vintage numbers are not.  Same caveat as GDP/NFCI for small.
#
# LATE-START SERIES:
# Some current series have a historical start much later than 1985:
#   JTSJOL  : Dec 2000  (BLS JOLTS began reporting)
#   T10YIE  : Jan 2003  (FRED TIPS breakeven series starts)
# For as_of dates before a series exists, get_current_vintage returns an empty
# Series; the calling panel builder assigns NaN gracefully.

# Named scalar constants kept for backward compatibility and as documentation.
GDP_PUBLICATION_LAG_MONTHS: int = 1   # quarterly; BEA advance ~1 month after quarter-end
NFCI_PUBLICATION_LAG_MONTHS: int = 0  # weekly -> monthly mean; available same month

# Per-series publication lags for ALL current series (small + big configs).
# Lag = months subtracted from the "last complete month" at as_of to get the
# last available data point under the day-15 convention:
#   lag 0 -> last complete month  (e.g. mid-June -> May is last available)
#   lag 1 -> last complete month - 1  (e.g. mid-June -> April is last available)
# GDPC1 is quarterly and uses gdp_available_through(); its entry here is for
# documentation only and is NOT used by get_current_vintage.
CURRENT_SERIES_LAG_MONTHS: dict[str, int] = {
    "GDPC1":    1,   # quarterly special case — handled by gdp_available_through()
    "NFCI":     0,   # weekly -> monthly mean; no lag
    # ── 9 current series added in big config ──────────────────────────────────
    "JTSJOL":   1,   # BLS JOLTS; released ~1 month after reference month
    "RRSFS":    1,   # Census retail control; released ~1 month after reference month
    "PCEPILFE": 1,   # BEA core PCE; released ~1 month after reference month
    "GS2":      0,   # 2-yr Treasury daily avg; available same month
    "ANFCI":    0,   # Fed weekly financial conditions index; available same month
    "CPILFESL": 0,   # BLS core CPI; published mid-following month (~0 lag)
    "IR":       0,   # BLS import price index; available same month
    "IQ":       0,   # BLS export price index; available same month
    "T10YIE":   0,   # FRED TIPS breakeven (daily avg); available same month
}


def _as_timestamp(as_of_date) -> pd.Timestamp:
    """Coerce a date-like input (or (year, month) tuple) to a Timestamp.

    A (year, month) tuple is interpreted as the MONTH-END of that month.
    """
    if isinstance(as_of_date, tuple):
        y, m = int(as_of_date[0]), int(as_of_date[1])
        return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(as_of_date)


def _load_current_series(series_id: str, config_name: str = "small") -> pd.Series:
    """Read a current (revised, already-transformed) series from the processed
    dataset built by data_loader.py for the given config."""
    csv = os.path.join(_PROJECT_ROOT, "data", "processed", f"dataset_{config_name}.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(
            f"Processed dataset not found:\n  {csv}\n"
            f"Build it first with:  python src/data_loader.py --config {config_name}"
        )
    df = pd.read_csv(csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    if series_id not in df.columns:
        raise KeyError(f"'{series_id}' not found in {csv}.")
    return df[series_id].dropna()


def gdp_available_through(as_of_date, config_name: str = "small") -> pd.Timestamp:
    """
    Last quarter-end whose real-GDP (GDPC1) estimate is already published as of
    `as_of_date`, under the declared publication-lag approximation.

    Rule
    ----
    GDP for a quarter ending at quarter-end date Q is treated as released
    GDP_PUBLICATION_LAG_MONTHS month-ends after Q, i.e. at
        release(Q) = Q + MonthEnd(GDP_PUBLICATION_LAG_MONTHS).
    It is AVAILABLE iff  release(Q) <= as_of.

    Example (lag = 1)
    -----------------
    as_of = 2008-04-15.  Q1-2008 ends 2008-03-31, released ~2008-04-30, which
    is AFTER 2008-04-15 -> Q1-2008 GDP is NOT yet out.  The last available GDP
    is Q4-2007 (released ~2008-01-31).  Hence a Q1 nowcast made in mid-April is
    a GENUINE nowcast: the Q1 GDP figure does not exist yet.

    Returns
    -------
    pd.Timestamp : the quarter-end month-end of the last available GDP quarter.
    """
    as_of = _as_timestamp(as_of_date)
    gdp = _load_current_series("GDPC1", config_name=config_name)
    release_dates = gdp.index + pd.offsets.MonthEnd(GDP_PUBLICATION_LAG_MONTHS)
    available = gdp.index[release_dates <= as_of]
    if len(available) == 0:
        raise ValueError(
            f"No GDP quarter is published by {as_of.date()} under a "
            f"{GDP_PUBLICATION_LAG_MONTHS}-month lag "
            f"(earliest quarter-end on file is {gdp.index[0].date()})."
        )
    return available[-1]


def get_current_vintage(series_id: str, as_of_date, config_name: str = "small") -> pd.Series:
    """
    Current (revised) values for a non-vintage series, TRUNCATED to respect
    real-time publication AVAILABILITY at `as_of_date`.

    Supports GDPC1, NFCI (small and big configs) and the 9 additional current
    series of the big config: JTSJOL, RRSFS, PCEPILFE, GS2, ANFCI, CPILFESL,
    IR, IQ, T10YIE.

    Values are REVISED (hybrid caveat — look-ahead on numbers, not on timing).
    See CURRENT_SERIES_LAG_MONTHS and the module docstring.

    Lag rules (day-15 convention):
      GDPC1  : quarterly — uses gdp_available_through() (BEA advance ~1 month
               after quarter-end).  Returns quarter-end month-end index.
      lag 0  : available through the last COMPLETE calendar month before as_of
               (e.g. mid-June -> May is last available).
      lag 1  : available through (last complete month - 1 month)
               (e.g. mid-June -> April is last available).

    Late-start series (JTSJOL from Dec-2000, T10YIE from Jan-2003, etc.) return
    an EMPTY Series if all their data lies after the publication cutoff — the
    caller should treat that as all-NaN for those dates.

    Parameters
    ----------
    series_id : str
        Any key in CURRENT_SERIES_LAG_MONTHS.
    as_of_date : str | datetime | (year, month) tuple

    Returns
    -------
    pd.Series  (name = series_id), month-end DatetimeIndex, already transformed.
    May be EMPTY if the series had not started by the publication cutoff.
    """
    if series_id not in CURRENT_SERIES_LAG_MONTHS:
        raise ValueError(
            f"get_current_vintage: '{series_id}' is not a known current series.\n"
            f"  Known series: {sorted(CURRENT_SERIES_LAG_MONTHS)}.\n"
            f"  Add it to CURRENT_SERIES_LAG_MONTHS if it should be handled here."
        )
    as_of = _as_timestamp(as_of_date)
    s = _load_current_series(series_id, config_name=config_name)

    # ── GDPC1: quarterly, uses its own release-calendar logic ─────────────────
    if series_id == "GDPC1":
        last_q = gdp_available_through(as_of, config_name=config_name)
        return s.loc[s.index <= last_q].copy()

    # ── All monthly current series: apply publication lag ─────────────────────
    # Under the day-15 convention, as_of mid-month means the current month is
    # still in progress.  Last complete month = the month-end just before as_of.
    lag = CURRENT_SERIES_LAG_MONTHS[series_id]
    last_complete = as_of + pd.offsets.MonthEnd(0)   # snap forward to month-end
    if last_complete > as_of:                        # mid-month -> step back one
        last_complete = as_of - pd.offsets.MonthEnd(1)
    cutoff = last_complete - pd.offsets.MonthEnd(lag)
    return s.loc[s.index <= cutoff].copy()


__all__ = [
    "VINTAGE_DIRS",
    "VINTAGE_NAME_ALIASES",
    "FIRST_USEFUL_VINTAGE",
    "GDP_PUBLICATION_LAG_MONTHS",
    "NFCI_PUBLICATION_LAG_MONTHS",
    "CURRENT_SERIES_LAG_MONTHS",
    "build_vintage_index",
    "list_available_vintages",
    "load_fredmd_vintage",
    "load_fredmd_raw_levels",
    "gdp_available_through",
    "get_current_vintage",
]


# ─── Smoke tests ──────────────────────────────────────────────────────────────
# Run from the project root with:   python -m src.forecast.data_import
#
# All tests are local: FRED-MD reads the on-disk vintage files; GDPC1/NFCI read
# the current processed dataset (data/processed/dataset_usa.csv).  No network.

def _hr(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _ragged_edge_report(panel: pd.DataFrame, n_tail: int = 3) -> None:
    """Print the last `n_tail` rows and flag the real ragged-edge NaNs."""
    print(f"  shape      : {panel.shape[0]} months x {panel.shape[1]} series")
    print(f"  date range : {panel.index[0].date()} .. {panel.index[-1].date()}")
    last = panel.iloc[-1]
    missing = last.index[last.isna()].tolist()
    print(f"  last row   : {panel.index[-1].date()}  "
          f"(ragged edge -> {len(missing)} series still NaN at the bottom)")
    if missing:
        print(f"    NaN at edge: {missing}")
    print("  last 3 published months (a few series):")
    cols_show = [c for c in ("INDPRO", "PAYEMS", "TWEXAFEGSMTHx", "PCEPI")
                 if c in panel.columns]
    print(panel[cols_show].tail(n_tail).to_string(float_format=lambda x: f"{x:8.4f}"))


def _run_fredmd_tests() -> None:
    _hr("FRED-MD #1  load_fredmd_vintage('2008-01')  [expect TWEXMMTH alias]")
    p08 = load_fredmd_vintage("2008-01")
    _ragged_edge_report(p08)
    src08 = _source_name_for("TWEXAFEGSMTHx", (2008, 1))
    ok08 = src08 == "TWEXMMTH" and p08["TWEXAFEGSMTHx"].notna().any()
    print(f"  TWEX alias : '{src08}' carries TWEXAFEGSMTHx  "
          f"-> {'OK (pre-2020-04 predecessor active, has data)' if ok08 else 'FAIL'}")

    _hr("FRED-MD #2  load_fredmd_vintage('2024-12')  [expect TWEXAFEGSMTHx native]")
    p24 = load_fredmd_vintage("2024-12")
    _ragged_edge_report(p24)
    src24 = _source_name_for("TWEXAFEGSMTHx", (2024, 12))
    ok24 = src24 == "TWEXAFEGSMTHx" and p24["TWEXAFEGSMTHx"].notna().any()
    print(f"  TWEX alias : '{src24}' carries TWEXAFEGSMTHx  "
          f"-> {'OK (native series active, has data)' if ok24 else 'FAIL'}")

    _hr("FRED-MD #3  load_fredmd_vintage('2000-08')  [first usable vintage, PCEPI present]")
    p00 = load_fredmd_vintage("2000-08")
    _ragged_edge_report(p00)
    pcepi_ok = "PCEPI" in p00.columns and p00["PCEPI"].notna().any()
    print(f"  PCEPI      : present with data -> "
          f"{'OK' if pcepi_ok else 'FAIL'}  "
          f"(first usable vintage = {FIRST_USEFUL_VINTAGE[0]}-{FIRST_USEFUL_VINTAGE[1]:02d})")

    _hr("FRED-MD #4  load_fredmd_vintage('2000-07')  [must raise: PCEPI absent]")
    try:
        load_fredmd_vintage("2000-07")
        print("  FAIL: no error raised — pre-2000-08 vintage was accepted!")
    except ValueError as exc:
        print("  OK: explicit error raised ->")
        for line in str(exc).splitlines():
            print(f"    {line}")


def _quarter_label(ts: pd.Timestamp) -> str:
    """Human-readable quarter label for a quarter-end month-end timestamp."""
    return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"


def _run_gdp_timing_tests() -> None:
    _hr("GDP TIMING  gdp_available_through(...)  "
        f"[lag = {GDP_PUBLICATION_LAG_MONTHS} month after quarter end]")
    gdp = _load_current_series("GDPC1")

    # mid-April 2008: Q1-2008 advance (~end of April) is NOT out yet -> Q4-2007.
    q_apr = gdp_available_through("2008-04-15")
    exp_apr = pd.Timestamp("2007-12-31")
    ok_apr = q_apr == exp_apr
    print(f"  as-of 2008-04-15 -> last GDP = {_quarter_label(q_apr)} "
          f"({q_apr.date()}), log-diff = {gdp.loc[q_apr]:.4f}")
    print(f"    expected Q4 2007 (Q1-2008 advance not yet released) -> "
          f"{'OK' if ok_apr else 'FAIL'}")

    # mid-May 2008: Q1-2008 advance is out by now -> Q1-2008.
    q_may = gdp_available_through("2008-05-15")
    exp_may = pd.Timestamp("2008-03-31")
    ok_may = q_may == exp_may
    print(f"  as-of 2008-05-15 -> last GDP = {_quarter_label(q_may)} "
          f"({q_may.date()}), log-diff = {gdp.loc[q_may]:.4f}")
    print(f"    expected Q1 2008 (advance now released) -> "
          f"{'OK' if ok_may else 'FAIL'}")
    print("  sanity: both values are quarterly growth rates (log-diff x100), "
          "not levels.")


def _run_current_series_tests() -> None:
    _hr("CURRENT DATA  get_current_vintage('GDPC1' / 'NFCI', '2009-01-15')")

    g = get_current_vintage("GDPC1", "2009-01-15")
    last_q = g.index[-1]
    exp_q = pd.Timestamp("2008-09-30")     # Q3 2008 (advance ~end Oct 2008)
    ok_q = last_q == exp_q
    print("  GDPC1 (last 8 available quarters, log-diff x100):")
    print(g.tail(8).to_string(float_format=lambda x: f"{x:7.3f}"))
    print(f"  last available quarter as-of 2009-01-15: {_quarter_label(last_q)} "
          f"({last_q.date()}) -> {'OK (Q3 2008)' if ok_q else 'FAIL'}")
    sensible = g.abs().max() < 50.0      # growth rates, not levels (~10000s)
    print(f"  sanity: values look like growth rates (|max| = {g.abs().max():.2f} "
          f"< 50) -> {'OK' if sensible else 'FAIL'}")

    n = get_current_vintage("NFCI", "2009-01-15")
    last_nfci = n.index[-1]
    exp_nfci = pd.Timestamp("2008-12-31")     # mid-Jan -> last complete month = Dec 2008
    ok_nfci = last_nfci == exp_nfci
    print("\n  NFCI (last 8 available months, level):")
    print(n.tail(8).to_string(float_format=lambda x: f"{x:7.3f}"))
    print(f"  last available month as-of 2009-01-15: {last_nfci.date()} "
          f"-> {'OK (Dec 2008, last COMPLETE month; Jan not peeked)' if ok_nfci else 'FAIL'}")
    crisis_peak = n.loc["2008-09":"2009-03"].max()
    print(f"  sanity: NFCI is a level ~0 in calm times; 2008Q4-2009Q1 peak = "
          f"{crisis_peak:.3f} (should be clearly > 0 = financial stress).")


if __name__ == "__main__":
    _hr("data_import.py smoke tests")
    print(f"project root : {_PROJECT_ROOT}")
    print(f"usable vintages on disk: {len(list_available_vintages())} "
          f"({list_available_vintages()[0]} .. {list_available_vintages()[-1]})")

    _run_fredmd_tests()
    _run_gdp_timing_tests()
    _run_current_series_tests()

    _hr("Done.")
