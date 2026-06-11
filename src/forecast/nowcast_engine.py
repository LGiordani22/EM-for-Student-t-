"""
src/forecast/nowcast_engine.py

Real-time DFM NOWCAST ENGINE for the nowcasting pipeline (Third brick).

Given an "as of" date and a target quarter, estimate the Student-t (or
Gaussian) mixed-frequency DFM on the real-time panel produced by
panel_builder.build_panel, and extract the implied nowcast of quarterly GDP
growth for the target quarter.

This module contains ONLY the DFM engine.  The univariate benchmarks (ARMA,
random walk) live in a separate module (benchmarks.py) and are not imported
here.

What it does
------------
1. build_panel(as_of) — the real-time 20-series panel (Second brick).
2. Extend the panel forward with all-NaN rows up to the target quarter-end
   month, so the Kalman smoother produces the *forecast* of the latent factors
   at the target month (a genuine nowcast: the target quarter's GDP is not yet
   published, and its last monthly indicators may not be either).
3. BLIND PCA initialisation recomputed on THIS vintage (scale + theta^(0)).
   We deliberately do NOT warm-start from theta_star (the in-sample fit):
     * scale rigour — every vintage is standardised on its own
       expanding-window mean/std (see "scale coherence" below); warm-starting
       loadings calibrated to the full-sample scale would be inconsistent;
     * out-of-sample rigour — the nowcast must not borrow any information from
       the future that theta_star implicitly contains.
4. fit_dfm(...) — the already-validated EM machine (Kalman filter/smoother +
   E-step + M-step + sign/Convention-1 post-processing).
5. Extract the GDP nowcast at the target quarter-end month via the GDP row of
   the augmented loading and the Mariano-Murasawa weights (1/3, 2/3, 1, 2/3,
   1/3), exactly the construction validated in em_main.

Scale coherence (CRITICAL)
--------------------------
Standardisation of the panel and de-standardisation of the nowcast use the
SAME per-series mean/std, computed on the sample of THIS vintage (data up to
as_of) — NOT the full sample, NOT the in-sample fit.  Each vintage has its own
expanding-window scale.  Sequence:

    mean, std    = column stats of THIS vintage's panel (non-NaN)
    panel_std    = (panel - mean) / std            ; fit_dfm(panel_std)
    nowcast_std  = Lambda[GDP] · (MM · f_smooth)   (standardised units)
    nowcast_z    = nowcast_std                      (z vs this vintage's GDP std)
    nowcast_livello = nowcast_std * std[GDP] + mean[GDP]   (SAME mean/std)

This is the *Volatility Paradox*: in a calm training window std[GDP] is small,
so the de-standardised level looks smoothed — which is CORRECT in real time
(no look-ahead to the larger full-sample volatility).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Forecast-package siblings (resolved relative to the project root, which is on
# sys.path when this module is run as `python -m src.forecast.nowcast_engine`).
from src.data_loader import BLOCK, FREQ, ORDERED_COLS, load_config
from src.forecast.panel_builder import build_panel
from src.forecast.data_import import gdp_available_through

# The First-Stage estimation machine lives in the flat src/ modules, which
# import one another with BARE module names lazily (e.g. `from em_e_step import
# run_e_step` inside run_em).  Put src/ on sys.path so those lazy imports
# resolve when fit_dfm runs.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from em_initialization import (             # noqa: E402  (after sys.path tweak)
    standardize,
    mm_fill_quarterly,
    gaussian_fill_ragged,
    pca_initialization,
    compute_theta_initial,
)
from em_main import fit_dfm                  # noqa: E402

# ─── Constants for the GDP extraction (match em_main / kalman) ────────────────
_BLOCK_ORDER: list[str] = ["real", "financial", "other"]
_MM_WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])
_REF_SERIES = {"real": "PAYEMS", "financial": "S&P 500", "other": "UMCSENTx"}
_R = 3  # one factor per block


def _quarter_end(target_quarter: str) -> pd.Timestamp:
    """Parse a 'YYYYQn' label into the month-END of that quarter's last month."""
    s = str(target_quarter).upper().replace(" ", "")
    if "Q" not in s:
        raise ValueError(f"target_quarter {target_quarter!r} is not 'YYYYQn'.")
    y_str, q_str = s.split("Q")
    y, q = int(y_str), int(q_str)
    if not 1 <= q <= 4:
        raise ValueError(f"quarter must be 1..4, got {q} in {target_quarter!r}.")
    return pd.Timestamp(y, q * 3, 1) + pd.offsets.MonthEnd(0)


def nowcast_gdp(
    as_of_date,
    target_quarter: str,
    config_name: str = "small",
    estimator: str = "student_t",
    theta_init: dict | None = None,
    max_iter: int = 250,
    verbose: bool = False,
) -> dict:
    """
    Nowcast quarterly GDP growth for `target_quarter` using data available at
    `as_of_date`, via the real-time mixed-frequency DFM.

    Parameters
    ----------
    as_of_date : str | datetime | (year, month) tuple
        Vintage publication date (information set), e.g. "2008-11-15".
    target_quarter : str
        Target quarter as 'YYYYQn', e.g. "2008Q4".
    estimator : "student_t" | "gaussian"
        Which DFM to estimate.  "gaussian" sets nu = inf inside fit_dfm.
    theta_init : dict | None
        If None (default), a blind PCA initialisation is recomputed on this
        vintage.  If given, it is used as theta^(0) (escape hatch; the
        nowcast loop should leave it None for out-of-sample rigour).
    max_iter : int
        Maximum outer-EM iterations (forwarded to fit_dfm).
    verbose : bool
        Per-iteration EM log (forwarded to fit_dfm).

    Returns
    -------
    dict with keys: nowcast_livello, nowcast_z, target_quarter, as_of,
        estimator, n_iter, converged, nu_eps_hat (None for gaussian),
        ultimo_gdp_disponibile, mean_gdp_train, std_gdp_train.
    """
    if estimator not in ("student_t", "gaussian"):
        raise ValueError(f"estimator must be 'student_t' or 'gaussian', got {estimator!r}.")

    cfg = load_config(config_name)
    _block = cfg["BLOCK"]
    _freq = cfg["FREQ"]
    _ordered_cols = cfg["ORDERED_COLS"]

    target_qe = _quarter_end(target_quarter)

    # 1. Real-time panel, extended forward to the target quarter-end so the
    #    smoother forecasts the factors there (all-NaN rows -> pure prediction).
    panel = build_panel(as_of_date, config_name=config_name)
    if target_qe < panel.index[0]:
        raise ValueError(
            f"target quarter-end {target_qe.date()} precedes the panel start "
            f"{panel.index[0].date()}."
        )
    if target_qe > panel.index[-1]:
        panel = panel.reindex(pd.date_range(panel.index[0], target_qe, freq="ME"))
    panel.index.name = None

    # 2. Blind expanding-window standardisation + PCA init on THIS vintage.
    Y_std, mean, std = standardize(panel)
    Y_mm = Y_std.copy()
    for col in Y_std.columns:
        if _freq.get(col) == "quarterly":
            Y_mm[col] = mm_fill_quarterly(Y_std[col])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    if theta_init is None:
        F0, _info = pca_initialization(Y_filled, _block)
        theta0 = compute_theta_initial(Y_filled, F0, _block)
    else:
        theta0 = theta_init

    # 3. Estimate the DFM (reuse the validated EM machine).
    freq_list = [_freq[c] for c in _ordered_cols]
    fit = fit_dfm(
        Y=Y_std.to_numpy(),
        theta_init=theta0,
        freq_list=freq_list,
        block_map=_block,
        ordered_cols=_ordered_cols,
        ref_series=_REF_SERIES,
        gaussian=(estimator == "gaussian"),
        use_full_elbo=True,
        max_iter=max_iter,
        verbose=verbose,
        save_path=None,
    )
    theta = fit["theta"]
    f_smooth = np.asarray(fit["f_smooth"])             # (T, 5r)

    # 4. GDP nowcast at the target quarter-end month, via the MM aggregation of
    #    the GDP block factor across the five lag-blocks.
    gdp_idx = _ordered_cols.index("GDPC1")
    gdp_block_j = _BLOCK_ORDER.index(_block["GDPC1"])   # 0 (real)
    idx_lags = np.array([lag * _R + gdp_block_j for lag in range(5)])
    t_idx = Y_std.index.get_loc(target_qe)
    phi = float(f_smooth[t_idx, idx_lags] @ _MM_WEIGHTS)
    nowcast_std = float(np.asarray(theta["Lambda"])[gdp_idx, gdp_block_j] * phi)

    # 5. De-standardise with the SAME vintage mean/std (scale coherence).
    mean_gdp = float(mean["GDPC1"])
    std_gdp = float(std["GDPC1"])
    nowcast_livello = nowcast_std * std_gdp + mean_gdp
    nowcast_z = nowcast_std                              # already in std[GDP] units

    nu_eps = float(theta["nu_eps"])
    return {
        "nowcast_livello": nowcast_livello,
        "nowcast_z": nowcast_z,
        "target_quarter": target_quarter,
        "as_of": str(as_of_date),
        "estimator": estimator,
        "n_iter": int(fit["n_iter"]),
        "converged": bool(fit["converged"]),
        "nu_eps_hat": (nu_eps if estimator == "student_t" else None),
        "ultimo_gdp_disponibile": gdp_available_through(as_of_date, config_name=config_name),
        "mean_gdp_train": mean_gdp,
        "std_gdp_train": std_gdp,
    }


__all__ = ["nowcast_gdp"]


# ─── Smoke tests ──────────────────────────────────────────────────────────────
# Run from the project root with:  python -m src.forecast.nowcast_engine
# Local: real vintages on disk + current processed dataset.  Estimates the full
# EM per call (minutes), so this exercises only two dates x two estimators.

def _hr(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def _realized_gdp(target_qe: pd.Timestamp) -> float:
    """Current (revised) realised GDP growth for the target quarter."""
    cur = pd.read_csv(
        os.path.join(_SRC_DIR, "..", "data", "processed", "dataset_small.csv"),
        index_col=0,
    )
    cur.index = pd.to_datetime(cur.index)
    if target_qe in cur.index:
        return float(cur.loc[target_qe, "GDPC1"])
    return float("nan")


def _run_date(as_of: str, target_quarter: str, scenario: str) -> None:
    _hr(f"{scenario}:  as_of = {as_of}   target = {target_quarter}")
    target_qe = _quarter_end(target_quarter)

    res = {}
    for est in ("student_t", "gaussian"):
        print(f"\n  estimating DFM ({est}) ...")
        res[est] = nowcast_gdp(as_of, target_quarter, estimator=est, verbose=False)
        r = res[est]
        print(f"    converged = {r['converged']}, n_iter = {r['n_iter']}, "
              f"last available GDP = "
              f"{r['ultimo_gdp_disponibile'].date()}")
        nu_str = f"{r['nu_eps_hat']:.2f}" if r["nu_eps_hat"] is not None else "inf (gaussian)"
        print(f"    nowcast: livello = {r['nowcast_livello']:+.4f}   "
              f"z = {r['nowcast_z']:+.4f}   nu_eps_hat = {nu_str}")

    # Scale check: training mean/std of GDP for this vintage (identical across
    # estimators — depends only on the panel).
    mean_gdp = res["student_t"]["mean_gdp_train"]
    std_gdp = res["student_t"]["std_gdp_train"]
    print(f"\n  training-scale GDP (this vintage): mean = {mean_gdp:+.4f}, "
          f"std = {std_gdp:.4f}")
    print(f"    (expected expanding-window std, NOT the ~1.04 full-sample value)")

    realized = _realized_gdp(target_qe)
    realized_z = (realized - mean_gdp) / std_gdp if std_gdp else float("nan")
    print(f"\n  realised {target_quarter} (current dataset): "
          f"livello = {realized:+.4f}   z = {realized_z:+.4f}")

    print("\n  comparison  (nowcast vs realised):")
    print(f"    {'method':<14}{'livello':>12}{'z':>10}{'err(liv)':>12}{'err(z)':>10}")
    for est in ("student_t", "gaussian"):
        r = res[est]
        print(f"    {est:<14}{r['nowcast_livello']:>12.4f}{r['nowcast_z']:>10.4f}"
              f"{r['nowcast_livello'] - realized:>12.4f}"
              f"{r['nowcast_z'] - realized_z:>10.4f}")
    print(f"    {'realised':<14}{realized:>12.4f}{realized_z:>10.4f}"
          f"{0.0:>12.4f}{0.0:>10.4f}")


if __name__ == "__main__":
    _hr("nowcast_engine.py smoke tests  (DFM only)")

    # Crisis quarter: GDP collapses; expect the level to be underestimated
    # (structural compression of the calm training std) but z to flag the event;
    # the Student-t may smooth more than the Gaussian (common-extreme down-weight).
    _run_date("2008-11-15", "2008Q4", "CRISIS")

    # Calm quarter: the two estimators should be close.
    _run_date("2015-05-15", "2015Q2", "CALM")

    _hr("Done.")
