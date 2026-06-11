"""
src/data_loader.py

Downloads, transforms, and saves the USA macroeconomic dataset for the
Student-t DFM estimation (First Stage).

CONFIG-DRIVEN (since the multi-configuration refactor, PASSO 1)
--------------------------------------------------------------
The series set is no longer hardcoded.  It is read from a JSON config file
``config/series_<name>.json`` (default name: "small"), which is the SINGLE
SOURCE OF TRUTH for:
  * which series are in the dataset and their canonical column ORDER,
  * each series' BLOCK (real/financial/other) and FREQUENCY (monthly/quarterly),
  * each FRED-MD series' transformation CODE,
  * which series are "current/non-vintage" (GDPC1, NFCI),
  * the factor sizes (r_R, r_F, r_X, r_total) and the output CSV name.

At import time the module loads the DEFAULT config ("small") and exposes the
familiar module-level constants (ORDERED_COLS, BLOCK, FREQ, ALL_FREDMD_COLS,
REAL_COLS, FINANCIAL_FREDMD_COLS, OTHER_COLS, SAMPLE_START) so that every
downstream module that imports them keeps working unchanged.  The "small"
config reproduces the original 20-series dataset bit-for-bit.

IMPORTANT — Two-step preprocessing pipeline:
  1) Stationarisation (here in data_loader.py): apply FRED-MD
     transformation codes to each series so that all series are
     approximately stationary.
  2) Standardisation (NOT here, but in em_initialization.py):
     centre to zero mean and rescale to unit variance, prior to PCA.

Pipeline
--------
1.  Download FRED-MD (monthly series + transformation codes)
2.  Download NFCI weekly from FRED direct; aggregate to monthly mean
3.  Download GDPC1 quarterly from FRED direct; compute log-diff * 100
4.  Apply FRED-MD transformation codes to the monthly series
5.  Filter to sample 1985-01 onwards
6.  Position GDP only at quarter-end months (NaN elsewhere)
7.  Save to data/processed/dataset_<config>.csv

Usage
-----
    # From the project root, with FRED_API_KEY set in the environment:
    python src/data_loader.py                       # default config "small"
    python src/data_loader.py --config small
    python src/data_loader.py --config small --api-key YOUR_KEY_HERE

FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
"""

import argparse
import io
import json
import os
import sys

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests
from fredapi import Fred

load_dotenv()

# ─── Paths and constants ──────────────────────────────────────────────────────

FREDMD_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREDMD_LOCAL = os.path.join(_PROJECT_ROOT, "data", "raw", "fredmd_current.csv")

CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")
DEFAULT_CONFIG = "small"

# Current (non-vintage) series with DEDICATED fetch logic (special frequency /
# transform handling): NFCI is weekly→monthly mean, GDPC1 is quarterly log-diff.
# Any OTHER current series declared in a config is fetched by the GENERIC
# monthly fetcher (download_current_monthly), applying the JSON transform code.
_SPECIAL_CURRENT_SERIES = {"NFCI", "GDPC1"}


# ─── Config loading (single source of truth) ──────────────────────────────────

def _config_path(name: str) -> str:
    return os.path.join(CONFIG_DIR, f"series_{name}.json")


def load_config(name: str = DEFAULT_CONFIG) -> dict:
    """
    Load ``config/series_<name>.json`` and derive every list/dict the loader
    needs.  This is the SINGLE SOURCE OF TRUTH for the series configuration.

    Returns a dict that merges the raw JSON with the following derived keys:
      ORDERED_COLS           : list[str]  — canonical column order (= series order)
      BLOCK                  : dict[str, str]  — name -> block
      FREQ                   : dict[str, str]  — name -> frequency
      ALL_FREDMD_COLS        : list[str]  — FRED-MD-sourced series, in order
      REAL_COLS / FINANCIAL_FREDMD_COLS / OTHER_COLS : list[str] (FRED-MD only)
      FREDMD_TRANSFORMS      : dict[str, int]  — name -> transform code (FRED-MD)
      CURRENT_SERIES         : list[str]  — non-vintage series (e.g. GDPC1, NFCI)
      SAMPLE_START           : str  — "YYYY-MM-DD"
      OUTPUT_CSV             : str  — output file name
    """
    path = _config_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Series config not found: {path}\n"
            f"  Expected a JSON config at config/series_{name}.json.\n"
            f"  Available configs: "
            f"{sorted(f[len('series_'):-len('.json')] for f in os.listdir(CONFIG_DIR) if f.startswith('series_') and f.endswith('.json')) if os.path.isdir(CONFIG_DIR) else '[none]'}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    series = raw["series"]

    # ── validate each entry ──────────────────────────────────────────────────
    valid_blocks = {"real", "financial", "other"}
    valid_freqs = {"monthly", "quarterly"}
    valid_sources = {"fredmd", "current", "spread"}
    for s in series:
        for key in ("name", "block", "freq", "source"):
            if key not in s:
                raise ValueError(f"Series entry {s!r} is missing required key '{key}'.")
        if s["block"] not in valid_blocks:
            raise ValueError(f"Series '{s['name']}': block '{s['block']}' not in {valid_blocks}.")
        if s["freq"] not in valid_freqs:
            raise ValueError(f"Series '{s['name']}': freq '{s['freq']}' not in {valid_freqs}.")
        if s["source"] not in valid_sources:
            raise ValueError(f"Series '{s['name']}': source '{s['source']}' not in {valid_sources}.")
        if s["source"] == "fredmd" and s.get("transform") is None:
            raise ValueError(f"Series '{s['name']}' (source 'fredmd') must declare a transform code.")
        if s["source"] == "spread":
            if s.get("transform") is None:
                raise ValueError(f"Series '{s['name']}' (source 'spread') must declare a transform code.")
            comps = s.get("components")
            if not isinstance(comps, list) or len(comps) != 2 or not all(isinstance(c, str) for c in comps):
                raise ValueError(
                    f"Series '{s['name']}' (source 'spread') must declare "
                    f"'components': [minuend, subtrahend] (two series names); "
                    f"the spread is computed as components[0] - components[1] on RAW LEVELS."
                )

    names = [s["name"] for s in series]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate series names in config '{name}': {dupes}")

    # ── derived lists/dicts ──────────────────────────────────────────────────
    ordered_cols = list(names)
    block = {s["name"]: s["block"] for s in series}
    freq = {s["name"]: s["freq"] for s in series}

    fredmd_series = [s for s in series if s["source"] == "fredmd"]
    all_fredmd_cols = [s["name"] for s in fredmd_series]
    real_cols = [s["name"] for s in fredmd_series if s["block"] == "real"]
    fin_cols = [s["name"] for s in fredmd_series if s["block"] == "financial"]
    other_cols = [s["name"] for s in fredmd_series if s["block"] == "other"]
    fredmd_transforms = {s["name"]: int(s["transform"]) for s in fredmd_series}

    # The ``source`` field is authoritative for which series are "current"
    # (non-vintage, fetched from FRED direct).  We derive the list from it
    # rather than from the optional explicit ``current_series`` key, so adding
    # a ``source: "current"`` entry to the JSON is all that is needed.
    current_series = [s["name"] for s in series if s["source"] == "current"]
    # Transform codes for current series that declare one (NFCI/GDPC1 are null:
    # they use dedicated fetchers; the generic monthly fetcher uses these codes).
    current_transforms = {
        s["name"]: int(s["transform"])
        for s in series
        if s["source"] == "current" and s.get("transform") is not None
    }

    # Constructed spreads: computed from the RAW LEVELS of two components
    # (spread = components[0] - components[1]), then the spread's own transform
    # is applied.  They are neither downloaded as columns (not 'fredmd') nor
    # fetched from FRED (not 'current'); they are derived in build_dataset.
    spread_defs = [
        {"name": s["name"], "components": list(s["components"]), "transform": int(s["transform"])}
        for s in series
        if s["source"] == "spread"
    ]

    cfg = dict(raw)
    cfg.update(
        ORDERED_COLS=ordered_cols,
        BLOCK=block,
        FREQ=freq,
        ALL_FREDMD_COLS=all_fredmd_cols,
        REAL_COLS=real_cols,
        FINANCIAL_FREDMD_COLS=fin_cols,
        OTHER_COLS=other_cols,
        FREDMD_TRANSFORMS=fredmd_transforms,
        CURRENT_SERIES=current_series,
        CURRENT_TRANSFORMS=current_transforms,
        SPREAD_DEFS=spread_defs,
        SAMPLE_START=raw.get("sample_start", "1985-01-01"),
        OUTPUT_CSV=raw.get("output_csv", f"dataset_{name}.csv"),
    )
    return cfg


# ─── Module-level constants (populated from the DEFAULT config at import) ──────
# Downstream modules import these names; keeping them at module level preserves
# backward compatibility.  They reflect the "small" config (the original 20).

_DEFAULT_CFG = load_config(DEFAULT_CONFIG)

SAMPLE_START: str = _DEFAULT_CFG["SAMPLE_START"]
REAL_COLS: list[str] = _DEFAULT_CFG["REAL_COLS"]
FINANCIAL_FREDMD_COLS: list[str] = _DEFAULT_CFG["FINANCIAL_FREDMD_COLS"]
OTHER_COLS: list[str] = _DEFAULT_CFG["OTHER_COLS"]
ALL_FREDMD_COLS: list[str] = _DEFAULT_CFG["ALL_FREDMD_COLS"]
ORDERED_COLS: list[str] = _DEFAULT_CFG["ORDERED_COLS"]
BLOCK: dict[str, str] = _DEFAULT_CFG["BLOCK"]
FREQ: dict[str, str] = _DEFAULT_CFG["FREQ"]


# ─── Helper ───────────────────────────────────────────────────────────────────

def _to_month_end(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Snap every date in dt_index to the last calendar day of its month.

    Example: 2020-01-01  →  2020-01-31
             2020-03-15  →  2020-03-31
    """
    return pd.DatetimeIndex(
        [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(1)
         for y, m in zip(dt_index.year, dt_index.month)]
    )


# ─── Transformation codes ─────────────────────────────────────────────────────

def apply_transform(series: pd.Series, code: int) -> pd.Series:
    """
    Apply a FRED-MD transformation code to a pandas Series.

    Code  Description
    ----  -----------
    1     Level (no change)
    2     First difference   Δx_t = x_t - x_{t-1}
    3     Second difference  Δ²x_t
    4     Log               ln(x_t)
    5     Log first diff    Δln(x_t) = ln(x_t) - ln(x_{t-1})
    6     Log second diff   Δ²ln(x_t)
    7     Percent change    (x_t - x_{t-1}) / x_{t-1}
    """
    code = int(code)
    if code == 1:
        return series.copy()
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    elif code == 7:
        return series.pct_change()
    else:
        raise ValueError(
            f"Unknown FRED-MD transformation code {code!r} "
            f"for series '{series.name}'"
        )


# ─── FRED-MD download and parsing ─────────────────────────────────────────────

def download_fredmd() -> tuple[pd.DataFrame, pd.Series]:
    """
    Download the latest FRED-MD file and return raw data + transformation codes.

    FRED-MD CSV structure
    ---------------------
    Row 0  (headers) : sasdate, INDPRO, IPFINAL, ...
    Row 1  (tcodes)  : transform:, 5, 5, 5, ...   ← iloc[0] of the DataFrame
    Row 2+ (data)    : 1959-01-01, 16.9, ...       ← iloc[1:] of the DataFrame

    Returns
    -------
    data   : pd.DataFrame  — raw (untransformed) monthly data
    tcodes : pd.Series     — transformation codes, indexed by series name
    """
    print("Downloading FRED-MD ...", end=" ", flush=True)
    raw_bytes: bytes | None = None

    # --- attempt network download ---
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(FREDMD_URL, headers=headers, timeout=120)
        resp.raise_for_status()
        raw_bytes = resp.content
        print("(web)  ", end="", flush=True)
    except Exception as _exc:
        print(f"\n  ! Web download failed ({_exc.__class__.__name__}: {_exc})")
        print("  ! Falling back to local copy ...")

    # --- fallback: local file ---
    if raw_bytes is None:
        if os.path.isfile(FREDMD_LOCAL):
            with open(FREDMD_LOCAL, "rb") as _f:
                raw_bytes = _f.read()
            print(f"  (local)  {FREDMD_LOCAL}")
        else:
            raise FileNotFoundError(
                "\nFRED-MD could not be downloaded and no local copy was found.\n\n"
                "  To fix this, download the file manually:\n"
                f"    URL  : {FREDMD_URL}\n"
                f"    Save to: {FREDMD_LOCAL}\n\n"
                "  In a browser, navigate to that URL and save the page as a CSV.\n"
                "  Then re-run this script."
            )

    raw = pd.read_csv(io.BytesIO(raw_bytes), header=0, low_memory=False)

    # First row of the DataFrame contains transformation codes
    date_col = raw.columns[0]          # typically 'sasdate'
    tcode_row = raw.iloc[0]
    tcodes = tcode_row.drop(date_col).astype(float)   # Series: name → code

    # Data rows start from index 1
    data = raw.iloc[1:].copy()
    data = data.rename(columns={date_col: "date"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")

    print(f"{data.shape[0]} months × {data.shape[1]} series  ✓")
    return data, tcodes


def select_and_transform_fredmd(
    data: pd.DataFrame,
    tcodes: pd.Series,
    all_fredmd_cols: list[str] | None = None,
    expected_transforms: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Select the target FRED-MD series and apply their transformation codes.

    The list of series to select is taken from ``all_fredmd_cols`` (defaults to
    the module-level ALL_FREDMD_COLS from the default config).  The actual
    transformation still uses the codes carried by the downloaded FRED-MD file
    (``tcodes``), exactly as before — this guarantees the output is bit-for-bit
    identical to the pre-refactor pipeline.  If ``expected_transforms`` is given
    (the config's declared codes), a consistency check asserts the file's codes
    match the config, so the JSON config stays authoritative.

    Raises a descriptive KeyError listing available column names if any target
    series is absent — useful when FRED-MD renames a column between vintages.
    """
    if all_fredmd_cols is None:
        all_fredmd_cols = ALL_FREDMD_COLS

    missing = [c for c in all_fredmd_cols if c not in data.columns]
    if missing:
        available = "\n    ".join(sorted(data.columns.tolist()))
        raise KeyError(
            f"\nThese series were not found in FRED-MD:\n  {missing}\n\n"
            f"Available FRED-MD columns:\n    {available}\n\n"
            "Update config/series_<config>.json (the series list)."
        )

    # Consistency check: the config's transform codes must match the file's.
    if expected_transforms is not None:
        mismatches = {
            c: (int(tcodes[c]), int(expected_transforms[c]))
            for c in all_fredmd_cols
            if c in expected_transforms and int(tcodes[c]) != int(expected_transforms[c])
        }
        if mismatches:
            lines = "\n".join(
                f"    {c}: file code {fc} != config code {cc}"
                for c, (fc, cc) in mismatches.items()
            )
            raise ValueError(
                "FRED-MD transform codes in the file disagree with the config:\n"
                f"{lines}\n"
                "  The config is the source of truth — fix the config transform "
                "code, or the FRED-MD file may have changed its code for that series."
            )

    transformed: dict[str, pd.Series] = {}
    for col in all_fredmd_cols:
        s = apply_transform(data[col], tcodes[col])
        s.name = col
        transformed[col] = s

    return pd.DataFrame(transformed, index=data.index)


# ─── FRED direct: NFCI and GDPC1 ──────────────────────────────────────────────

def _get_fred_api_key(provided: str | None = None) -> str:
    key = provided or os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED API key not found.\n"
            "  Option 1: set the environment variable  FRED_API_KEY=<your_key>\n"
            "  Option 2: pass it to build_dataset(fred_api_key='...')\n"
            "  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def download_nfci(fred: Fred) -> pd.Series:
    """
    Download weekly NFCI from FRED and aggregate to monthly mean.

    NFCI is the Chicago Fed's National Financial Conditions Index.
    It is already stationary (levels), so no further transformation is needed.
    Weekly observations within each calendar month are averaged.
    """
    print("Downloading NFCI (weekly → monthly mean) ...", end=" ", flush=True)
    nfci_w = fred.get_series("NFCI")
    nfci_w.index = pd.to_datetime(nfci_w.index)

    # 'ME' = Month End frequency (pandas >= 2.2)
    nfci_m = nfci_w.resample("ME").mean()
    nfci_m.name = "NFCI"
    print(f"{nfci_m.shape[0]} monthly observations  ✓")
    return nfci_m


def download_gdpc1(fred: Fred) -> pd.Series:
    """
    Download real GDP (GDPC1) from FRED, compute log-difference × 100,
    and re-index to the month-end of the last month of each quarter.

    FRED stores quarterly dates at the *start* of each quarter:
        2020-01-01  →  Q1 2020   (January)
        2020-04-01  →  Q2 2020   (April)
        ...

    We shift each observation to the *end* of the last month of its quarter:
        Q1 (Jan 1)  → + 2 months → Mar 1  → month-end → Mar 31
        Q2 (Apr 1)  → + 2 months → Jun 1  → month-end → Jun 30
        Q3 (Jul 1)  → + 2 months → Sep 1  → month-end → Sep 30
        Q4 (Oct 1)  → + 2 months → Dec 1  → month-end → Dec 31

    This aligns GDP with the last month of the quarter in the monthly panel.
    """
    print("Downloading GDPC1 (quarterly) ...", end=" ", flush=True)
    gdpc1 = fred.get_series("GDPC1")
    gdpc1.index = pd.to_datetime(gdpc1.index)

    # Log-difference × 100 (approximately quarterly growth rate in percent)
    gdp_growth = np.log(gdpc1).diff() * 100
    gdp_growth.name = "GDPC1"

    # Shift index from quarter-start to quarter-end month-end
    shifted_idx = gdp_growth.index + pd.DateOffset(months=2)
    gdp_growth.index = _to_month_end(shifted_idx)

    n_obs = gdp_growth.dropna().shape[0]
    print(f"{n_obs} quarterly observations (after log-diff)  ✓")
    return gdp_growth


def download_current_monthly(fred: Fred, name: str, code: int) -> pd.Series:
    """
    Download a CURRENT (non-vintage, FRED-direct) series, aggregate it to a
    monthly month-end index, and apply its FRED-MD-style transformation code.

    This is the GENERIC fetcher for any ``source: "current"`` series in the
    config other than the two with dedicated logic (NFCI, GDPC1).  The native
    FRED frequency may be monthly, weekly (e.g. ANFCI) or daily (e.g. T10YIE);
    we collapse to a monthly mean (the same convention used for NFCI) so the
    fetcher is frequency-agnostic, then apply the transform on the monthly level.

    Series that start later than the sample (e.g. JTSJOL ~2000, T10YIE ~2003)
    will have leading NaN in the assembled panel — this is expected and handled
    downstream by the Kalman filter.  The first valid date is printed.
    """
    print(f"Downloading {name} (current, FRED) ...", end=" ", flush=True)
    s = fred.get_series(name)
    s.index = pd.to_datetime(s.index)

    # Collapse any native frequency (daily/weekly/monthly) to a monthly mean;
    # 'ME' labels each month at its month-end date.
    s_m = s.resample("ME").mean()
    s_m = apply_transform(s_m, code)
    s_m.name = name

    valid = s_m.dropna()
    first = valid.index[0].date() if not valid.empty else None
    print(f"first valid {first}, {valid.shape[0]} monthly obs (tcode {int(code)})  ✓")
    return s_m


# ─── Constructed spreads ──────────────────────────────────────────────────────

def compute_spreads(raw_fm: pd.DataFrame, spread_defs: list[dict]) -> pd.DataFrame:
    """
    Build the CONSTRUCTED spread series from the RAW (untransformed) levels of
    their FRED-MD components, then apply each spread's own transformation code.

    Each ``spread_def`` is ``{"name", "components": [a, b], "transform"}`` and the
    spread is computed as ``level(a) - level(b)`` on the raw rate levels (e.g.
    term_spread = GS10 - TB3MS on the yields in percent), NOT on the already
    differenced component series.  This is deliberate: rate levels are I(1) and
    enter the model in first differences (t2), but a spread is their stationary
    cointegrating combination (I(0)) and is therefore kept in level (t1).

    ``raw_fm`` must be the raw FRED-MD frame (month-start index, untransformed),
    so the component levels are available.  The returned frame is month-end
    indexed, aligned with the transformed monthly panel.

    Where a component level is NaN the spread is NaN (subtraction propagates),
    so spreads carry no missingness beyond that of their components.
    """
    if not spread_defs:
        return pd.DataFrame(index=_to_month_end(raw_fm.index))

    out: dict[str, pd.Series] = {}
    for sd in spread_defs:
        name = sd["name"]
        a, b = sd["components"]
        code = int(sd["transform"])
        missing = [c for c in (a, b) if c not in raw_fm.columns]
        if missing:
            raise KeyError(
                f"Spread '{name}' needs raw FRED-MD levels for {missing}, "
                f"absent from the FRED-MD file. Components must be FRED-MD series."
            )
        print(f"Computing spread {name} = {a} - {b} (level) ...", end=" ", flush=True)
        level = raw_fm[a] - raw_fm[b]            # spread on raw rate levels
        s = apply_transform(level, code)
        s.name = name
        out[name] = s
        valid = s.dropna()
        rng = f"[{valid.min():+.2f}, {valid.max():+.2f}]" if not valid.empty else "[empty]"
        print(f"{valid.shape[0]} obs, range {rng} (tcode {code})  ✓")

    spreads = pd.DataFrame(out, index=raw_fm.index)
    spreads.index = _to_month_end(spreads.index)
    return spreads


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def build_dataset(
    fred_api_key: str | None = None,
    save: bool = True,
    config: str | dict = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Full pipeline: download → transform → assemble → filter → save.

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key.  Falls back to the FRED_API_KEY environment variable.
    save : bool
        Write result to data/processed/<output_csv> (from the config).
    config : str | dict
        Config name (loads config/series_<name>.json) or an already-loaded
        config dict (from :func:`load_config`).  Default "small".

    Returns
    -------
    df : pd.DataFrame
        Shape (T, M).  Index = month-end dates.  Columns = config ORDERED_COLS.
        Non-quarter-end months have NaN for quarterly series by construction.
    """
    cfg = load_config(config) if isinstance(config, str) else config

    ordered_cols = cfg["ORDERED_COLS"]
    block = cfg["BLOCK"]
    freq = cfg["FREQ"]
    all_fredmd_cols = cfg["ALL_FREDMD_COLS"]
    sample_start = cfg["SAMPLE_START"]
    current_series = cfg["CURRENT_SERIES"]
    current_transforms = cfg["CURRENT_TRANSFORMS"]
    output_csv = cfg["OUTPUT_CSV"]

    # Every current series other than the two with dedicated fetchers (NFCI,
    # GDPC1) is downloaded by the generic monthly fetcher, which needs a
    # transform code from the config.
    missing_codes = [
        n for n in current_series
        if n not in _SPECIAL_CURRENT_SERIES and n not in current_transforms
    ]
    if missing_codes:
        raise ValueError(
            f"Config '{cfg.get('name', '?')}' has current series without a "
            f"transform code (required by the generic fetcher): {missing_codes}."
        )

    key = _get_fred_api_key(fred_api_key)
    fred = Fred(api_key=key)

    # ── 1. FRED-MD ─────────────────────────────────────────────────────────────
    raw_fm, tcodes = download_fredmd()
    monthly_fm = select_and_transform_fredmd(
        raw_fm, tcodes,
        all_fredmd_cols=all_fredmd_cols,
        expected_transforms=cfg["FREDMD_TRANSFORMS"],
    )
    # FRED-MD dates are month-start (e.g. 2020-01-01) → convert to month-end
    monthly_fm.index = _to_month_end(monthly_fm.index)

    # ── 2. FRED direct (current/non-vintage series) ────────────────────────────
    # NFCI and GDPC1 have dedicated fetchers; every other current series uses
    # the generic monthly fetcher with its config transform code.
    frames = [monthly_fm]
    for name in current_series:
        if name == "NFCI":
            frames.append(download_nfci(fred))
        elif name == "GDPC1":
            frames.append(download_gdpc1(fred))
        else:
            frames.append(download_current_monthly(fred, name, current_transforms[name]))

    # ── 2b. Constructed spreads (from raw component LEVELS) ─────────────────────
    spread_defs = cfg.get("SPREAD_DEFS", [])
    if spread_defs:
        frames.append(compute_spreads(raw_fm, spread_defs))

    # ── 3. Merge (outer join) ───────────────────────────────────────────────────
    # Quarterly series have only 4 dates per year; non-quarter months become NaN
    df = pd.concat(frames, axis=1, sort=True)

    # ── 4. Filter to sample period ──────────────────────────────────────────────
    df = df.loc[df.index >= pd.Timestamp(sample_start)].copy()

    # ── 5. Enforce canonical column order ───────────────────────────────────────
    df = df[ordered_cols]

    # ── 6. Safety: ensure quarterly series are NaN on non-quarter-end months ────
    non_qend = ~df.index.month.isin([3, 6, 9, 12])
    for col in ordered_cols:
        if freq.get(col) == "quarterly":
            df.loc[non_qend, col] = np.nan

    # ── 7. Report ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Dataset assembled  (config: {cfg.get('name', '?')})")
    print(f"  Shape      : {df.shape[0]} months × {df.shape[1]} series")
    print(f"  Date range : {df.index[0].date()}  →  {df.index[-1].date()}")
    print("  Missing (%) per series:")
    miss_pct = (df.isna().mean() * 100).round(1)
    for col, pct in miss_pct.items():
        tag = f"  [{block.get(col,'?')}/{freq.get(col,'?')}]"
        print(f"    {col:<30s}{pct:5.1f}%{tag}")
    print("─" * 60)

    # ── 8. Save ─────────────────────────────────────────────────────────────────
    if save:
        out_path = os.path.join(_PROJECT_ROOT, "data", "processed", output_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path)
        print(f"\nSaved → {out_path}")

        # Build block_sizes and frequency_counts dynamically from the config
        block_sizes: dict[str, int] = {}
        for col in df.columns:
            b = block.get(col, "?")
            block_sizes[b] = block_sizes.get(b, 0) + 1

        frequency_counts: dict[str, int] = {}
        for col in df.columns:
            f = freq.get(col, "?")
            frequency_counts[f] = frequency_counts.get(f, 0) + 1

        metadata = {
            "config": cfg.get("name", "?"),
            "description": cfg.get(
                "description",
                "USA macroeconomic dataset for Student-t DFM estimation",
            ),
            "sample_start": df.index[0].strftime("%Y-%m-%d"),
            "sample_end":   df.index[-1].strftime("%Y-%m-%d"),
            "n_observations": len(df),
            "n_series": len(df.columns),
            "series": [
                {"name": col, "block": block.get(col, "?"), "freq": freq.get(col, "?")}
                for col in df.columns
            ],
            "block_sizes": block_sizes,
            "factor_sizes": cfg.get(
                "factor_sizes", {"r_R": 1, "r_F": 1, "r_X": 1, "r_total": 3}
            ),
            "frequency_counts": frequency_counts,
            "transformations_source": (
                "FRED-MD transformation codes for monthly series; "
                "log-diff x 100 for GDPC1; levels for NFCI"
            ),
            "standardization": (
                "NOT applied here; performed in em_initialization.py"
            ),
        }

        json_path = out_path.replace(".csv", "_metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved → {json_path}")

    return df


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the USA macro dataset for the Student-t DFM (config-driven)."
    )
    _grp = parser.add_mutually_exclusive_group()
    _grp.add_argument(
        "--small", dest="config", action="store_const", const="small",
        help="Use the 'small' config (20 series). Shortcut for --config small.",
    )
    _grp.add_argument(
        "--big", dest="config", action="store_const", const="big",
        help="Use the 'big' config (50 series). Shortcut for --config big.",
    )
    _grp.add_argument(
        "--config", dest="config", metavar="NAME",
        help=f"Series config name (config/series_<NAME>.json). Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="FRED API key (else taken from the FRED_API_KEY environment variable).",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIG

    build_dataset(fred_api_key=args.api_key, config=args.config)
