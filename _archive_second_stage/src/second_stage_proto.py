"""
src/forecast/second_stage_proto.py

PROTOTYPE of the Second Stage: quantile regression of GDP on the DFM factors
(Growth-at-Risk style), for ONE as_of / target quarter.  2D density only — NO
3D surface yet.  Built to verify the mechanism end-to-end.

Pipeline
--------
1. Re-fit the (real-time) DFM at `as_of` with the existing machinery (same
   sequence as nowcast_engine.nowcast_gdp), extract the 3 SMOOTHED factors for
   the whole history up to the edge.  The fit is CACHED to disk (it is the slow
   part) so steps 2-6 iterate instantly.
2. Build the quarterly-aligned factors: for each block (real/financial/other),
   the Mariano-Murasawa weighted combination of the 5 monthly lags at each
   quarter-end — exactly the bridge the first stage uses to map factor -> GDP.
3. Training set: quarter-ends up to the last GDP realised+available at `as_of`,
   paired with realised GDP in NATURAL units (quarterly %, 100*dlog).  Reports n.
4. Quantile regression Q_tau(GDP | f) = b0 + b_real f_real + b_fin f_fin +
   b_nom f_nom for tau in {.05,.10,.25,.50,.75,.90,.95}.  Reports coefficients.
5. Predict the 7 quantiles at the target quarter; check/repair quantile crossing
   (Chernozhukov rearrangement = sort).
6. Fit an Azzalini-Capitanio skew-t to the 7 quantiles (ABG-2019 style: minimise
   squared distance between the skew-t quantiles and the estimated quantiles).
7. 2D figure: predictive density + the 7 quantile points + realised GDP line.

Run
---
  python -m src.forecast.second_stage_proto                       # 2008-11-15 / 2008Q4
  python -m src.forecast.second_stage_proto --as-of 2020-05-15 --target 2020Q2
  python -m src.forecast.second_stage_proto --estimator gaussian  # faster fit
  python -m src.forecast.second_stage_proto --refit               # ignore cache
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats, optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_config
from src.forecast.panel_builder import build_panel
from src.forecast.data_import import gdp_available_through

# Reach the flat src/ EM machinery (same sys.path tweak as nowcast_engine).
import sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from em_initialization import (              # noqa: E402
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from em_main import fit_dfm                   # noqa: E402

_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "output", "forecast_realtime", "second_stage_cache")
_FIG_DIR = os.path.join(_PROJECT_ROOT, "output", "forecast_realtime", "second_stage_proto")

_BLOCK_ORDER = ["real", "financial", "other"]
_MM_WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])
_REF_SERIES = {"real": "PAYEMS", "financial": "S&P 500", "other": "UMCSENTx"}
_R = 3
_TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
_FACTOR_NAMES = ["f_real", "f_fin", "f_nom"]


# ─── Step 1: fit DFM, extract smoothed factors (cached) ───────────────────────

def _quarter_end(target_quarter: str) -> pd.Timestamp:
    s = str(target_quarter).upper().replace(" ", "")
    y, q = s.split("Q")
    return pd.Timestamp(int(y), int(q) * 3, 1) + pd.offsets.MonthEnd(0)


def _cache_path(as_of: str, config: str, estimator: str) -> str:
    tag = f"{config}_{estimator}_{str(as_of).replace('-', '').replace(':', '')}"
    return os.path.join(_CACHE_DIR, f"factors_{tag}.npz")


def fit_factors(as_of, target_quarter, config="big", estimator="student_t",
                refit=False, verbose=False) -> dict:
    """Fit the DFM at `as_of` and return smoothed factors + panel info (cached)."""
    cache = _cache_path(as_of, config, estimator)
    target_qe = _quarter_end(target_quarter)

    if os.path.exists(cache) and not refit:
        z = np.load(cache, allow_pickle=True)
        print(f"  [cache] loaded factors from {os.path.basename(cache)}")
        return {
            "f_smooth": z["f_smooth"],
            "index": pd.DatetimeIndex(z["index"]),
            "gdp_natural": pd.Series(z["gdp_natural"], index=pd.DatetimeIndex(z["gdp_index"])),
            "n_iter": int(z["n_iter"]), "converged": bool(z["converged"]),
        }

    cfg = load_config(config)
    freq, block, ordered_cols = cfg["FREQ"], cfg["BLOCK"], cfg["ORDERED_COLS"]

    panel = build_panel(as_of, config_name=config)
    gdp_natural = panel["GDPC1"].dropna().copy()          # realised GDP, natural units
    if target_qe > panel.index[-1]:
        panel = panel.reindex(pd.date_range(panel.index[0], target_qe, freq="ME"))
    panel.index.name = None

    Y_std, mean, std = standardize(panel)
    Y_mm = Y_std.copy()
    for col in Y_std.columns:
        if freq.get(col) == "quarterly":
            Y_mm[col] = mm_fill_quarterly(Y_std[col])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    F0, _ = pca_initialization(Y_filled, block)
    theta0 = compute_theta_initial(Y_filled, F0, block)

    print(f"  fitting DFM ({estimator}, {config}) at {as_of} ... (slow, cached after)")
    fit = fit_dfm(
        Y=Y_std.to_numpy(), theta_init=theta0,
        freq_list=[freq[c] for c in ordered_cols],
        block_map=block, ordered_cols=ordered_cols, ref_series=_REF_SERIES,
        gaussian=(estimator == "gaussian"), use_full_elbo=True,
        max_iter=250, verbose=verbose, save_path=None,
    )
    f_smooth = np.asarray(fit["f_smooth"])

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez(
        cache,
        f_smooth=f_smooth,
        index=np.array([str(d.date()) for d in Y_std.index]),
        gdp_natural=gdp_natural.to_numpy(),
        gdp_index=np.array([str(d.date()) for d in gdp_natural.index]),
        n_iter=fit["n_iter"], converged=fit["converged"],
    )
    print(f"  fit done (n_iter={fit['n_iter']}, converged={fit['converged']}); "
          f"cached -> {os.path.basename(cache)}")
    return {"f_smooth": f_smooth, "index": Y_std.index,
            "gdp_natural": gdp_natural,
            "n_iter": int(fit["n_iter"]), "converged": bool(fit["converged"])}


# ─── Step 2: MM-aggregated quarterly factors ──────────────────────────────────

def quarterly_factors(f_smooth: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """MM-weighted quarterly factor per block at each quarter-end month."""
    qe_mask = index.month.isin([3, 6, 9, 12])
    qe_dates = index[qe_mask]
    out = {}
    for j, name in enumerate(_FACTOR_NAMES):
        idx_lags = np.array([lag * _R + j for lag in range(5)])
        vals = [float(f_smooth[index.get_loc(d), idx_lags] @ _MM_WEIGHTS) for d in qe_dates]
        out[name] = vals
    return pd.DataFrame(out, index=qe_dates)


# ─── Step 4: quantile regression ──────────────────────────────────────────────

def fit_quantile_reg(train: pd.DataFrame, taus: list[float]) -> pd.DataFrame:
    """QuantReg of GDP on the 3 factors for each tau.  Returns coefficient table."""
    from statsmodels.regression.quantile_regression import QuantReg
    import statsmodels.api as sm

    X = sm.add_constant(train[_FACTOR_NAMES])
    y = train["gdp"]
    rows = []
    for tau in taus:
        res = QuantReg(y, X).fit(q=tau, max_iter=2000)
        rows.append({"tau": tau, "b0": res.params["const"],
                     **{f"b_{n}": res.params[n] for n in _FACTOR_NAMES}})
    return pd.DataFrame(rows).set_index("tau")


# ─── Step 5: predict + rearrange ──────────────────────────────────────────────

def predict_quantiles(coef: pd.DataFrame, f_target: pd.Series) -> pd.Series:
    x = np.array([1.0, f_target["f_real"], f_target["f_fin"], f_target["f_nom"]])
    cols = ["b0"] + [f"b_{n}" for n in _FACTOR_NAMES]
    q = coef[cols].to_numpy() @ x
    return pd.Series(q, index=coef.index)


def rearrange(q: pd.Series) -> tuple[pd.Series, bool]:
    """Chernozhukov rearrangement: sort to enforce monotonicity. Returns (q_sorted, crossed)."""
    crossed = bool((np.diff(q.to_numpy()) < 0).any())
    return pd.Series(np.sort(q.to_numpy()), index=q.index), crossed


# ─── Step 6: skew-t fit (Azzalini-Capitanio) ──────────────────────────────────

def skewt_pdf(x, xi, omega, alpha, nu):
    z = (np.asarray(x, dtype=float) - xi) / omega
    return (2.0 / omega) * stats.t.pdf(z, nu) * stats.t.cdf(
        alpha * z * np.sqrt((nu + 1.0) / (z ** 2 + nu)), nu + 1.0)


def _skewt_grid_cdf(xi, omega, alpha, nu, lo, hi, n=4000):
    grid = np.linspace(lo, hi, n)
    pdf = skewt_pdf(grid, xi, omega, alpha, nu)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    return grid, cdf


def fit_skewt(q_hat: pd.Series) -> dict:
    """Fit skew-t to the estimated quantiles (ABG 2019): min sum (ppf(tau)-q_hat)^2."""
    taus = np.array(q_hat.index, dtype=float)
    qv = q_hat.to_numpy(dtype=float)
    span = qv[-1] - qv[0]
    lo, hi = qv[0] - 3 * span, qv[-1] + 3 * span

    # init: location=median, scale from IQR, symmetric, moderate df
    xi0 = float(q_hat.loc[0.50]) if 0.50 in q_hat.index else float(np.median(qv))
    omega0 = max((q_hat.loc[0.90] - q_hat.loc[0.10]) / 2.563, 1e-2) if 0.90 in q_hat.index else max(span / 4, 1e-2)

    # nu is bounded to (2.05, 50): with only 7 quantiles the df is weakly
    # identified and an unbounded transform lets it run to ~1e110 (degenerate,
    # skew-normal limit).  A smooth logistic cap keeps it finite; nu~50 then
    # simply reads "no fat tails needed".  ABG (2019) likewise cap the df.
    _NU_MAX = 50.0

    def unpack(p):
        nu = 2.05 + (_NU_MAX - 2.05) / (1.0 + np.exp(-p[3]))
        return p[0], np.exp(p[1]), p[2], nu      # xi, omega>0, alpha, nu in (2.05, 50)

    def obj(p):
        xi, omega, alpha, nu = unpack(p)
        grid, cdf = _skewt_grid_cdf(xi, omega, alpha, nu, lo, hi)
        ppf = np.interp(taus, cdf, grid)
        return float(np.sum((ppf - qv) ** 2))

    p0 = np.array([xi0, np.log(omega0), 0.0, -1.955])   # nu0 ~ 8 under the logistic cap
    res = optimize.minimize(obj, p0, method="Nelder-Mead",
                            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-10})
    xi, omega, alpha, nu = unpack(res.x)
    return {"xi": xi, "omega": omega, "alpha": alpha, "nu": nu,
            "converged": bool(res.success), "sse": float(res.fun),
            "grid_lo": lo, "grid_hi": hi}


# ─── Step 7: figure ───────────────────────────────────────────────────────────

def make_figure(q_hat, skewt, realised, as_of, target_quarter, config, path):
    xi, omega, alpha, nu = skewt["xi"], skewt["omega"], skewt["alpha"], skewt["nu"]
    lo = min(skewt["grid_lo"], realised - 1, q_hat.min() - 1)
    hi = max(skewt["grid_hi"], realised + 1, q_hat.max() + 1)
    xs = np.linspace(lo, hi, 1000)
    dens = skewt_pdf(xs, xi, omega, alpha, nu)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, dens, color="#1f77b4", linewidth=2.0, label="Skew-t predictive density", zorder=3)
    ax.fill_between(xs, dens, color="#1f77b4", alpha=0.10, zorder=1)

    qy = skewt_pdf(q_hat.to_numpy(), xi, omega, alpha, nu)
    ax.scatter(q_hat.to_numpy(), qy, color="#d62728", s=42, zorder=5,
               label="Estimated quantiles")
    for tau, x, y in zip(q_hat.index, q_hat.to_numpy(), qy):
        ax.annotate(f"{tau:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="#d62728")

    if np.isfinite(realised):
        ax.axvline(realised, color="black", linewidth=1.6, linestyle="--",
                   label=f"Realised GDP = {realised:+.2f}%", zorder=4)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("GDP growth (quarterly %, 100·Δlog)", fontsize=10)
    ax.set_ylabel("Predictive density", fontsize=10)
    ax.set_title(f"Second-Stage predictive density — target {target_quarter}  "
                 f"(as_of {as_of}, {config})", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Orchestration ─────────────────────────────────────────────────────────────

def run(as_of, target_quarter, config, estimator, refit):
    print("=" * 78)
    print(f"SECOND-STAGE PROTOTYPE  as_of={as_of}  target={target_quarter}  "
          f"config={config}  estimator={estimator}")
    print("=" * 78)

    # 1-2. fit + factors
    ff = fit_factors(as_of, target_quarter, config, estimator, refit=refit)
    qf = quarterly_factors(ff["f_smooth"], ff["index"])
    target_qe = _quarter_end(target_quarter)

    # 3. training set: quarter-ends up to last realised GDP available at as_of
    last_gdp = gdp_available_through(as_of, config_name=config)
    gdp = ff["gdp_natural"]
    train_qe = qf.index[(qf.index <= last_gdp) & qf.index.isin(gdp.index)]
    train = qf.loc[train_qe].copy()
    train["gdp"] = gdp.loc[train_qe].to_numpy()
    train = train.dropna()

    print(f"\n[3] TRAINING SET")
    print(f"    last realised GDP available at as_of: {last_gdp.date()}")
    print(f"    n training quarters = {len(train)}  "
          f"({train.index[0].date()} .. {train.index[-1].date()})")
    print(f"    GDP range: [{train['gdp'].min():+.2f}, {train['gdp'].max():+.2f}]  "
          f"mean {train['gdp'].mean():+.2f}")

    # 4. quantile regression
    coef = fit_quantile_reg(train, _TAUS)
    print(f"\n[4] QUANTILE REGRESSION coefficients")
    print(coef.to_string(float_format=lambda v: f"{v:+.4f}"))
    lo_tau = coef.loc[0.05]
    dom = max(_FACTOR_NAMES, key=lambda n: abs(lo_tau[f"b_{n}"]))
    print(f"    -> at tau=0.05 the dominant factor is '{dom}' "
          f"(|b|={abs(lo_tau['b_'+dom]):.3f}); "
          f"b_fin={lo_tau['b_f_fin']:+.3f}, b_real={lo_tau['b_f_real']:+.3f}, "
          f"b_nom={lo_tau['b_f_nom']:+.3f}")

    # 5. predict at target + crossing check
    if target_qe not in qf.index:
        raise SystemExit(f"target quarter-end {target_qe.date()} not in factor index "
                         f"(edge {qf.index[-1].date()}).")
    f_target = qf.loc[target_qe]
    q_raw = predict_quantiles(coef, f_target)
    q_sorted, crossed = rearrange(q_raw)
    print(f"\n[5] PREDICTED QUANTILES at {target_quarter}  (factors: "
          f"f_real={f_target['f_real']:+.3f}, f_fin={f_target['f_fin']:+.3f}, "
          f"f_nom={f_target['f_nom']:+.3f})")
    tbl = pd.DataFrame({"q_raw": q_raw, "q_rearranged": q_sorted})
    print(tbl.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"    quantile crossing detected: {'YES -> rearranged' if crossed else 'no'}")

    # 6. skew-t fit
    sk = fit_skewt(q_sorted)
    print(f"\n[6] SKEW-T FIT (Azzalini-Capitanio)")
    print(f"    converged={sk['converged']}  SSE={sk['sse']:.3e}")
    print(f"    xi(loc)={sk['xi']:+.4f}  omega(scale)={sk['omega']:.4f}  "
          f"alpha(skew)={sk['alpha']:+.4f}  nu(df)={sk['nu']:.2f}")

    # realised target (may be NaN if not yet out at "today")
    realised = float(gdp.loc[target_qe]) if target_qe in gdp.index else float("nan")
    if not np.isfinite(realised):
        # fall back to current processed dataset (revised), for context only
        cur = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "processed",
                                       f"dataset_{config}.csv"), index_col=0)
        cur.index = pd.to_datetime(cur.index)
        realised = float(cur.loc[target_qe, "GDPC1"]) if target_qe in cur.index else float("nan")
    print(f"\n    realised {target_quarter} GDP = {realised:+.4f}%  "
          f"(pdf at realised = {float(skewt_pdf(realised, sk['xi'], sk['omega'], sk['alpha'], sk['nu'])):.4f})")

    # 7. figure
    fig_path = os.path.join(
        _FIG_DIR, f"density2d_{config}_{estimator}_{target_quarter}_"
                  f"{str(as_of).replace('-', '')}.png")
    make_figure(q_sorted, sk, realised, as_of, target_quarter, config, fig_path)
    print(f"\n[7] figure -> {fig_path}")
    return {"train": train, "coef": coef, "q": q_sorted, "skewt": sk,
            "realised": realised, "fig": fig_path}


def main():
    p = argparse.ArgumentParser(description="Second-stage quantile-regression prototype (2D).")
    p.add_argument("--as-of", default="2008-11-15")
    p.add_argument("--target", default="2008Q4")
    p.add_argument("--config", default="big")
    p.add_argument("--estimator", default="student_t", choices=["student_t", "gaussian"])
    p.add_argument("--refit", action="store_true", help="ignore the factor cache and re-fit")
    args = p.parse_args()
    run(args.as_of, args.target, args.config, args.estimator, args.refit)


if __name__ == "__main__":
    main()
