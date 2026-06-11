"""
src/run_experiment_a.py
=======================
Experiment A — Heavy-tail benefit: DGP Student-t, both estimators.

Objective
---------
When the DGP is genuinely heavy-tailed (the calibrated real-data fit, nu ~ 4),
the Student-t estimator should OUTPERFORM the Gaussian one on every parametric
and latent metric.  The gap is concentrated on the innovation/idiosyncratic
covariances Q and R and on the factor amplitudes: the Gaussian estimator,
lacking the weight-attenuation mechanism, cannot down-weight the fat-tailed
observations and therefore inflates its covariance estimates.

The expected outcome:
  - loglik gap (student_t - gaussian) LARGE and POSITIVE (the tail-robust
    estimator fits the heavy-tailed data materially better).
  - Student-t recovers nu ~ 4 (the calibrated truth); the Gaussian estimator
    reports nu = inf by construction.
  - Q, R recovered with smaller relative error by the Student-t estimator;
    the Gaussian estimator over-states them.
  - factor recovery (|corr|, RMSE) at least as good for Student-t.

This is the mirror image of Experiment B: there the DGP is Gaussian and the
two estimators COINCIDE (nesting, gap ~ 0); here the DGP is heavy-tailed and
the Student-t estimator WINS (gap > 0).

Thesis reference: section "Experiment A — Monte Carlo", line ~11200-12800.

Usage
-----
    python src/run_experiment_a.py [--S 20] [--full]

    --S N     : run S replications per scenario (default 20 for validation)
    --full    : run the full design (S=1000, T_grid=[100,200,400,800,497])
                for the thesis-quality run (cluster recommended)

Construction of theta_star^A
----------------------------
theta_star^A IS the calibrated real-data fit, unchanged: the Student-t DGP
with nu_u, nu_eps ~ 4.  Experiment B modifies it (nu -> inf); Experiment A
leaves it as-is.  The difference between the two experiments isolates the
effect of the tails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_utils import parse_config_args                # noqa: E402
from data_loader import load_config                       # noqa: E402
from em_main import load_dfm_fit                          # noqa: E402
from monte_carlo import (                                  # noqa: E402
    run_grid, load_grid_results,
    run_one_replication,
    aggregate_replications, print_aggregate_table,
    _scenario_filename, _load_scenario,
)

# Block order for the per-block rows of the side-by-side kernel table.
_BLOCK_ORDER: list[str] = ["real", "financial", "other"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _tee_file(path: pathlib.Path):
    """Tee stdout to *path* for the duration of this context."""
    buf = io.StringIO()
    real_stdout = sys.stdout
    class _T:
        encoding = getattr(real_stdout, "encoding", "utf-8")
        errors   = getattr(real_stdout, "errors",   "replace")
        def write(self, s):   real_stdout.write(s); buf.write(s)
        def flush(self):      real_stdout.flush()
        def isatty(self):     return False
    sys.stdout = _T()
    try:
        yield
    finally:
        sys.stdout = real_stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Saved txt report: {path}")


def _build_theta_star_A(theta_star: dict) -> dict:
    """
    Construct the Student-t DGP for Experiment A.

    theta_star^A IS theta_star: the calibrated real-data fit with its
    heavy tails (nu_u, nu_eps ~ 4) intact.  Unlike Experiment B, which sets
    nu -> inf to obtain a Gaussian DGP, Experiment A leaves every parameter
    unchanged.  The difference between A and B isolates the effect of the
    tails.  We return a defensive copy so the caller's dict is never mutated.

    Experiment A uses theta_star (the real-data DFM fit, fit_dfm_result.npz)
    UNCHANGED as the data-generating process.  The panels are simulated from
    the FULL Student-t DFM: the factor and idiosyncratic innovations are
    scaled by weights drawn from Gamma(nu/2, nu/2) with the calibrated
    nu_u ~ 4, nu_eps ~ 4, so the simulated data carry genuine heavy tails.
    This is the DGP against which the tail-robust (Student-t) estimator
    should outperform the Gaussian one.
    """
    return {k: np.asarray(v).copy() for k, v in theta_star.items()}


def _save_metrics_npz(metrics: dict, path: pathlib.Path) -> None:
    """Persist a single-replication metrics dict to a flat ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for k, v in metrics.items():
        if isinstance(v, str):
            flat[k] = np.asarray(v)
        elif isinstance(v, bool):
            flat[k] = np.asarray(v, dtype=bool)
        elif isinstance(v, (int, float)):
            flat[k] = np.asarray(v)
        elif isinstance(v, np.ndarray):
            flat[k] = v
        else:
            flat[k] = np.asarray(v)
    np.savez(path, **flat)


def print_side_by_side(m_t: dict, m_g: dict) -> None:
    """
    Print a vertical Student-t / Gaussian comparison for a single
    (seed, T) configuration.  Both dicts come from
    :func:`monte_carlo.run_one_replication`.
    """
    bar = "=" * 82
    print("\n" + bar)
    print(f"  EXPERIMENT A KERNEL  -  seed = {m_t['seed']}, T = {m_t['T']}, pi = 0")
    print(f"  DGP = Student-t (calibrated), two estimators applied to the same panel")
    print(bar)

    def row(label: str, val_t, val_g, fmt: str = "{:>14.4f}") -> str:
        def _sv(val):
            if isinstance(val, float) and np.isinf(val):
                return "  inf(frozen)"
            if isinstance(val, float) and np.isnan(val):
                return "  NaN (check!)"   # Unexpected NaN — flag it
            if isinstance(val, (int, float)):
                return fmt.format(val)
            return f"{val!s:>14s}"        # String passthrough (e.g. "N/A (Gaussian)")
        return f"  {label:<40s}  {_sv(val_t)}  {_sv(val_g)}"

    def _nu_relerr_val(m: dict, key: str):
        """Return "N/A (Gaussian)" when nu is not a free parameter, else the float."""
        return "N/A (Gaussian)" if m.get("nu_frozen") else m[key]

    print(f"  {'metric':<40s}  {'Student-t':>14s}  {'Gaussian':>14s}")
    print("  " + "-" * 72)

    # Algorithmic reliability
    print(row("EM converged",                 m_t["converged"],  m_g["converged"],  fmt="{:>14}"))
    print(row("n_iter",                       m_t["n_iter"],     m_g["n_iter"],     fmt="{:>14d}"))
    print(row("monotonicity_violations",      m_t["n_monotonicity_violations"],
                                              m_g["n_monotonicity_violations"], fmt="{:>14d}"))
    print(row("loglik final",                 m_t["loglik_final"], m_g["loglik_final"], fmt="{:>14.2f}"))
    print("  " + "-" * 72)

    # Heavy-tail parameters
    print(row("nu_u  (truth {:.3f})".format(m_t["nu_u_star"]),
              m_t["nu_u_hat"], m_g["nu_u_hat"]))
    print(row("nu_eps (truth {:.3f})".format(m_t["nu_eps_star"]),
              m_t["nu_eps_hat"], m_g["nu_eps_hat"]))
    print(row("nu_u  rel.err",
              _nu_relerr_val(m_t, "nu_u_relerr"),
              _nu_relerr_val(m_g, "nu_u_relerr"), fmt="{:>14.4f}"))
    print(row("nu_eps rel.err",
              _nu_relerr_val(m_t, "nu_eps_relerr"),
              _nu_relerr_val(m_g, "nu_eps_relerr"), fmt="{:>14.4f}"))
    if m_g.get("nu_frozen"):
        print(f"  {'  * Gaussian nu: inf (Gaussian limit) — nu_relerr not applicable':<72s}")
    print("  " + "-" * 72)

    # Eigenvalues of A
    print(row("rho(A) (truth {:.6f})".format(m_t["rho_A_star"]),
              m_t["rho_A_hat"], m_g["rho_A_hat"], fmt="{:>14.6f}"))
    print(row("rho(A) rel.err",
              m_t["rho_A_relerr"], m_g["rho_A_relerr"], fmt="{:>14.2%}"))
    print(row("|eig(A_hat) - eig(A_star)|_2",
              m_t["eig_A_err_norm"], m_g["eig_A_err_norm"]))
    print("  " + "-" * 72)

    # Lambda
    print(row("Lambda relerr  (normalised only)",
              m_t["lambda_relerr_normalised"],
              m_g["lambda_relerr_normalised"], fmt="{:>14.2%}"))
    print(row("Lambda relerr  (Procrustes-block)",
              m_t["lambda_relerr_procrustes_blockdiag"],
              m_g["lambda_relerr_procrustes_blockdiag"], fmt="{:>14.2%}"))
    for j, b in enumerate(_BLOCK_ORDER):
        print(row(f"  h_{b}",
                  float(m_t["H_block_diag"][j]),
                  float(m_g["H_block_diag"][j])))
    print("  " + "-" * 72)

    # Q diagonal per block
    for j, b in enumerate(_BLOCK_ORDER):
        print(row(f"diag(Q)_{b} relerr",
                  float(m_t["diagQ_relerr"][j]),
                  float(m_g["diagQ_relerr"][j]), fmt="{:>14.2%}"))
    print(row("R median rel.err",
              m_t["R_median_relerr"], m_g["R_median_relerr"], fmt="{:>14.2%}"))
    print(row("R max rel.err",
              m_t["R_max_relerr"],    m_g["R_max_relerr"],    fmt="{:>14.2%}"))
    print("  " + "-" * 72)

    # Factor recovery
    for j, b in enumerate(_BLOCK_ORDER):
        print(row(f"|corr(f_hat_{b}, F_true_{b})|",
                  float(m_t["factor_abscorr"][j]),
                  float(m_g["factor_abscorr"][j])))
    pair_names = [("R", "F"), ("R", "X"), ("F", "X")]
    for k, (a, b) in enumerate(pair_names):
        print(row(f"  cross |corr(f_hat_{a}, F_true_{b})|",
                  float(m_t["factor_crosscorr"][k]),
                  float(m_g["factor_crosscorr"][k])))
    for j, b in enumerate(_BLOCK_ORDER):
        print(row(f"factor RMSE_traj  ({b})",
                  float(m_t["factor_rmse_traj"][j]),
                  float(m_g["factor_rmse_traj"][j])))
    print("  " + "-" * 72)

    # Weight recovery — Student-t-specific.  Under the Gaussian estimator the
    # weights are identically 1, so corr is 0 / NaN and overlap is the
    # expected-by-chance baseline.  Reported for symmetry; the contrast is
    # the point.
    print(row("corr(w_u_hat,   w_u_true)",
              m_t["w_u_corr"], m_g["w_u_corr"]))
    print(row("corr(w_eps_hat, w_eps_true)",
              m_t["w_eps_corr"], m_g["w_eps_corr"]))
    print(row("w_u   overlap@5%",
              m_t["w_u_overlap_5pct"],   m_g["w_u_overlap_5pct"],   fmt="{:>14.2%}"))
    print(row("w_eps overlap@5%",
              m_t["w_eps_overlap_5pct"], m_g["w_eps_overlap_5pct"], fmt="{:>14.2%}"))
    print(row("w_u   lift@5%   (overlap / chance)",
              m_t["w_u_lift_5pct"],   m_g["w_u_lift_5pct"]))
    print(row("w_eps lift@5%   (overlap / chance)",
              m_t["w_eps_lift_5pct"], m_g["w_eps_lift_5pct"]))
    print(bar)


def _advantage_table(agg_st: dict, agg_g: dict, T: int, S: int) -> None:
    """
    Print the key Student-t-vs-Gaussian metrics for one (T, S) scenario,
    emphasising the Experiment A message: the Student-t estimator BEATS the
    Gaussian one on the heavy-tailed DGP.  The 'gap' column is (student_t -
    gaussian); for the loglik it should be LARGE and POSITIVE, and for the
    Q / R relative errors the Gaussian column should be the larger (inflated).
    """
    bar = "=" * 88
    print(f"\n{bar}")
    print(f"  Experiment A — Heavy-tail benefit  |  T={T}, S={S}")
    print(f"  DGP: Student-t (calibrated nu~4)  |  Two estimators on the same panels")
    print(bar)
    print(f"  {'metric':<48s}  {'student_t':>12s}  {'gaussian':>12s}  {'gap':>10s}")
    print("  " + "-" * 86)

    def _val(agg: dict, key: str, pct: bool = False) -> tuple[float, str]:
        if key not in agg:
            return float("nan"), "  N/A"
        mean = agg[key]["mean"]
        if pct:
            return mean, f"{mean:>12.2%}"
        return mean, f"{mean:>12.4f}"

    def _row(label, key, pct=False):
        vt, st = _val(agg_st, key, pct)
        vg, sg = _val(agg_g,  key, pct)
        gap = vt - vg if (np.isfinite(vt) and np.isfinite(vg)) else float("nan")
        gap_s = f"{gap:>+10.4f}" if np.isfinite(gap) else "       N/A"
        print(f"  {label:<48s}  {st}  {sg}  {gap_s}")

    # nu estimates (student_t should recover ~4; gaussian is inf by design)
    nu_u_st   = agg_st.get("nu_u_hat",  {}).get("mean", float("nan"))
    nu_eps_st = agg_st.get("nu_eps_hat", {}).get("mean", float("nan"))
    print(f"  {'nu_u_hat  (truth ~4, student_t only)':<48s}  {nu_u_st:>12.3f}  {'inf (by design)':>12s}  {'':>10s}")
    print(f"  {'nu_eps_hat (truth ~4, student_t only)':<48s}  {nu_eps_st:>12.3f}  {'inf (by design)':>12s}  {'':>10s}")
    print("  " + "-" * 86)

    # loglik — the headline: large POSITIVE gap = benefit of tail-robustness
    _row("loglik_final (mean over replications)", "loglik_final")
    print("  " + "-" * 86)

    # parameter recovery — Gaussian should over-state Q and R
    _row("Lambda relerr Procrustes-block [PRIMARY]",
         "lambda_relerr_procrustes_blockdiag", pct=True)
    _row("rho(A) rel.err", "rho_A_relerr", pct=True)
    _row("diag(Q) rel.err [real]",   "diagQ_relerr_real",  pct=True)
    _row("diag(Q) rel.err [fin]",    "diagQ_relerr_fin",   pct=True)
    _row("diag(Q) rel.err [other]",  "diagQ_relerr_other", pct=True)
    _row("R median rel.err",         "R_median_relerr",    pct=True)
    print("  " + "-" * 86)

    # factor recovery
    _row("|corr| factor [real]",      "factor_abscorr_real")
    _row("|corr| factor [financial]", "factor_abscorr_fin")
    _row("|corr| factor [other]",     "factor_abscorr_other")
    print("  " + "-" * 86)

    # convergence
    _row("convergence rate",   "converged")
    _row("iterations (mean)",  "n_iter")
    print(bar)
    print("  Interpretation:")
    print("    loglik gap > 0 (large)  -> Student-t fits the heavy tails better.")
    print("    Q / R rel.err larger for Gaussian -> it inflates covariances")
    print("    (no down-weighting of fat-tailed observations).")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(S: int = 20, full: bool = False, config: str = "small") -> None:
    # Force UTF-8 on Windows to avoid em-dash encoding errors.
    try:
        sys.stdout.reconfigure(encoding="utf-8")                  # type: ignore
    except Exception:
        pass

    cfg_data     = load_config(config)
    ordered_cols = cfg_data["ORDERED_COLS"]
    block_map    = cfg_data["BLOCK"]
    freq_list    = [cfg_data["FREQ"][c] for c in ordered_cols]
    fit_path = _PROJECT_ROOT / "data" / "processed" / config / "fit_dfm_result.npz"
    print(f"Loading calibrated theta_star from: {fit_path}")
    real_fit   = load_dfm_fit(fit_path)
    theta_star = real_fit["theta"]
    nu_u_A     = float(theta_star["nu_u"])
    nu_eps_A   = float(theta_star["nu_eps"])
    rho_A      = float(np.max(np.abs(np.linalg.eigvals(
        np.asarray(theta_star["A"])))))
    print(f"  theta_star: rho(A)={rho_A:.4f}  nu_u={nu_u_A:.3f}  nu_eps={nu_eps_A:.3f}")

    # ── Build theta_star^A (the calibrated Student-t DGP, unchanged) ──────────
    theta_star_A = _build_theta_star_A(theta_star)
    print(f"\ntheta_star^A (Student-t DGP):")
    print(f"  nu_u   = {float(theta_star_A['nu_u']):.3f}   (heavy tails — UNCHANGED from theta_star)")
    print(f"  nu_eps = {float(theta_star_A['nu_eps']):.3f}   (heavy tails — UNCHANGED from theta_star)")
    print(f"  Lambda, A, Q, R, Sigma_0: UNCHANGED (calibrated real-data fit)")
    print(f"  -> Experiment B sets nu=inf; A keeps the tails, isolating their effect")

    # ── Grid configuration ────────────────────────────────────────────────────
    if full:
        T_GRID = [100, 200, 400, 800, 497]
        S_RUN  = 1000
        print(f"\n[FULL RUN] S={S_RUN}, T_grid={T_GRID}  (thesis-quality)")
    else:
        T_GRID = [100, 200, 400, 800, 497]
        S_RUN  = S
        print(f"\n[TEST RUN] S={S_RUN}, T_grid={T_GRID}  (validation; use --full for S=1000)")

    ESTIMATORS = ["student_t", "gaussian"]
    PI_GRID    = [0.0]
    OUT_DIR_A  = _PROJECT_ROOT / "output" / "monte_carlo" / config / "expA"

    print(f"\noutput_dir  : {OUT_DIR_A}")
    print(f"config      : {config}")
    print(f"resume      : True  (safe to interrupt and re-run)")

    # ── Run grid ──────────────────────────────────────────────────────────────
    grid_result = run_grid(
        theta_star=theta_star_A,
        S=S_RUN,
        T_grid=T_GRID,
        estimators=ESTIMATORS,
        pi_grid=PI_GRID,
        freq_list=freq_list,
        block_map=block_map,
        ordered_cols=ordered_cols,
        output_dir=OUT_DIR_A,
        resume=True,
    )
    print(f"\nrun_grid done: {grid_result['n_computed']} computed, "
          f"{grid_result['n_skipped']} skipped")

    # ── Heavy-tail-benefit diagnostics at each T ──────────────────────────────
    with _tee_file(OUT_DIR_A / "results.txt"):
        for T in T_GRID:
            path_st = OUT_DIR_A / _scenario_filename("student_t", T, 0.0, S_RUN)
            path_g  = OUT_DIR_A / _scenario_filename("gaussian",  T, 0.0, S_RUN)
            if not path_st.exists() or not path_g.exists():
                print(f"\n[WARN] missing scenario files for T={T}, skipping diagnostics")
                continue
            agg_st = _load_scenario(path_st)["aggregates"]
            agg_g  = _load_scenario(path_g)["aggregates"]

            # Per-T Student-t-advantage table.
            _advantage_table(agg_st, agg_g, T=T, S=S_RUN)

            # nu_u_hat distribution for the Student-t estimator at this T:
            # it should concentrate near the calibrated truth (~4), NOT saturate.
            reps_st = [r for r in _load_scenario(path_st)["per_replication"]
                       if isinstance(r, dict)]
            if reps_st:
                nu_u_vals = [r.get("nu_u_hat", float("nan")) for r in reps_st]
                nu_u_arr  = np.array([v for v in nu_u_vals if np.isfinite(v)])
                print(f"\n  nu_u_hat distribution (Student-t estimator, T={T}, "
                      f"S={len(nu_u_arr)} finite reps):")
                if len(nu_u_arr) > 0:
                    print(f"    min={nu_u_arr.min():.1f}  "
                          f"median={np.median(nu_u_arr):.1f}  "
                          f"mean={nu_u_arr.mean():.1f}  "
                          f"max={nu_u_arr.max():.1f}")
                    print(f"    (truth nu_u={nu_u_A:.2f}; expect concentration near it, "
                          f"NOT saturation at the Brent bound 1000)")
                else:
                    print(f"    all nu_u_hat are inf (unexpected on a heavy-tailed DGP)")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _extra(p):
        p.add_argument("--S", type=int, default=20,
                       help="Replications per scenario (default 20 for test)")
        p.add_argument("--full", action="store_true",
                       help="Full run: S=1000, T_grid=[100,200,400,800,497]")

    args = parse_config_args(
        "Run Experiment A (Heavy-tail benefit, DGP Student-t)",
        extra=_extra,
    )
    S = 1000 if args.full else args.S
    main(S=S, full=args.full, config=args.config)
