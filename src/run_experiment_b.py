"""
src/run_experiment_b.py
=======================
Experiment B — Nesting test: DGP Gaussian, both estimators.

Objective
---------
When the DGP is Gaussian (nu -> infinity), both the Student-t and the
Gaussian estimator converge to the same Maximum-Likelihood solution.  The
Student-t estimator loses nothing relative to the Gaussian one — this is the
*nesting property*: the Gaussian model is a special case of the Student-t
model (nu -> inf), so the Student-t likelihood can always match it.

The expected outcome:
  - Student-t estimator on Gaussian data -> nu_hat saturates toward the Brent
    upper bound (1000), recognising the absence of heavy tails.
  - loglik gap (student_t - gaussian) ~ 0, versus the large positive gap in
    Experiment A (DGP Student-t) where the tail-robust estimator wins.
  - Lambda, Q, R, factor recovery: both estimators achieve nearly identical
    metrics (no penalty for using the more general Student-t model).

Thesis reference: section "Experiment B — nesting check", line ~12050+.

Usage
-----
    python src/run_experiment_b.py [--S 20] [--full]

    --S N     : run S replications per scenario (default 20 for validation)
    --full    : run the full design (S=1000, T_grid=[100,200,400,800,497])
                for the thesis-quality run (cluster recommended)

Construction of theta_star^B
-----------------------------
theta_star^B is theta_star with the tails removed (nu -> infinity).
It generates Gaussian data with the SAME loadings, dynamics and variances
as the Student-t DGP of Experiment A.  This isolates the effect of the tails:
the only difference between Experiment A and B is the presence/absence of
heavy tails in the innovations and idiosyncratic noise.
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
    aggregate_replications, print_aggregate_table,
    _scenario_filename, _load_scenario,
)


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


def _build_theta_star_B(theta_star: dict) -> dict:
    """
    Construct the Gaussian DGP by setting nu_u = nu_eps = np.inf.

    theta_star^B is theta_star with the tails removed (nu -> infinity).
    It generates Gaussian data with the SAME loadings, dynamics and variances
    as the Student-t DGP of Experiment A.  This isolates the effect of the
    tails: the only difference between Experiment A and B is the presence/
    absence of heavy tails.  All other parameters (Lambda, A, Q, R, Sigma_0)
    are copied unchanged.

    Experiment B uses the SAME theta_star (the real-data DFM fit) but sends
    nu_u, nu_eps -> infinity, while leaving Lambda, A, Q, R, Sigma_0
    byte-for-byte unchanged.  The panels are then generated from a GAUSSIAN
    DGP: with nu = infinity the weight prior Gamma(nu/2, nu/2) degenerates to
    a point mass at 1, so the Student-t weight mechanism switches off (weights
    identically 1) and the innovations are purely Gaussian -- the heavy-tail
    machinery of the simulator is bypassed as redundant in this limit.  In
    other words, B reuses the calibrated loadings/dynamics/variances of the
    real fit and strips ONLY the tails.

    Why share theta_star between A and B: it makes A vs B a controlled
    ceteris-paribus experiment.  The only difference between the two DGPs is
    the presence (A) or absence (B) of heavy tails; any difference in the
    results is therefore attributable to the tails alone, not to a different
    calibration.

    Distinction DGP vs estimator: here nu=inf lives in the DGP (the data are
    generated tail-free, so the TRUE weights are 1), which is distinct from
    the gaussian=True ESTIMATOR (where nu=inf is imposed on the fitted model,
    forcing the ESTIMATED weights to 1).  Experiment B combines a Gaussian DGP
    with each of the two estimators in turn: the Student-t estimator on
    Gaussian data must recover large-but-finite nu (nesting), while the
    Gaussian estimator fits it directly.
    """
    theta_B = {k: np.asarray(v).copy() for k, v in theta_star.items()}
    theta_B["nu_u"]   = np.inf
    theta_B["nu_eps"] = np.inf
    return theta_B


def _loglik_gap_table(agg_st: dict, agg_g: dict, T: int, S: int) -> None:
    """Print the key nesting-test metrics for one (T, S) scenario."""
    bar = "=" * 88
    print(f"\n{bar}")
    print(f"  Experiment B — Nesting check  |  T={T}, S={S}")
    print(f"  DGP: Gaussian (nu=inf)  |  Two estimators on the same synthetic panels")
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

    # nu estimates
    nu_u_st  = agg_st.get("nu_u_hat",  {}).get("mean", float("nan"))
    nu_eps_st = agg_st.get("nu_eps_hat", {}).get("mean", float("nan"))
    print(f"  {'nu_u_hat  (student_t only)':<48s}  {nu_u_st:>12.1f}  {'N/A (Gaussian)':>12s}  {'':>10s}")
    print(f"  {'nu_eps_hat (student_t only)':<48s}  {nu_eps_st:>12.1f}  {'N/A (Gaussian)':>12s}  {'':>10s}")
    print("  " + "-" * 86)

    # loglik
    _row("loglik_final (mean over replications)", "loglik_final")
    print("  " + "-" * 86)

    # parameter recovery
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


def _compare_A_vs_B(
    dir_A: pathlib.Path,
    dir_B: pathlib.Path,
    T: int,
    S_A: int,
    S_B: int,
) -> None:
    """
    Print a four-column table: metric | A:student_t | A:gaussian | B:student_t | B:gaussian.
    A = DGP Student-t (Experiment A); B = DGP Gaussian (Experiment B).
    """
    def _load(directory: pathlib.Path, estimator: str, S: int) -> dict | None:
        fname = _scenario_filename(estimator, T, 0.0, S)
        path  = directory / fname
        if not path.exists():
            return None
        payload = _load_scenario(path)
        return payload.get("aggregates", {})

    agg_At = _load(dir_A, "student_t", S_A)
    agg_Ag = _load(dir_A, "gaussian",  S_A)
    agg_Bt = _load(dir_B, "student_t", S_B)
    agg_Bg = _load(dir_B, "gaussian",  S_B)

    if agg_At is None or agg_Bt is None:
        print(f"\n[INFO] Cannot print A-vs-B table at T={T}: "
              f"Experiment A file missing from {dir_A}.")
        return

    bar = "=" * 110
    print(f"\n{bar}")
    print(f"  Experiment A vs B  |  T={T}  |  A:S={S_A}  B:S={S_B}")
    print(f"  A: DGP Student-t (calibrated nu~4)   B: DGP Gaussian (nu=inf)")
    print(f"  Message: in A Student-t beats Gaussian; in B they coincide (nesting).")
    print(bar)
    print(f"  {'metric':<44s}  {'A:stud-t':>10s}  {'A:gauss':>10s}  "
          f"{'B:stud-t':>10s}  {'B:gauss':>10s}")
    print("  " + "-" * 99)

    def _get(agg: dict | None, key: str, pct: bool = False) -> str:
        if agg is None or key not in agg:
            return f"{'N/A':>10s}"
        v = agg[key]["mean"]
        if not np.isfinite(v):
            return f"{'—':>10s}"
        return f"{v:>10.2%}" if pct else f"{v:>10.4f}"

    def _row4(label: str, key: str, pct: bool = False):
        print(f"  {label:<44s}  "
              f"{_get(agg_At, key, pct)}  "
              f"{_get(agg_Ag, key, pct)}  "
              f"{_get(agg_Bt, key, pct)}  "
              f"{_get(agg_Bg, key, pct)}")

    _row4("loglik_final",   "loglik_final")
    _row4("Lambda relerr Procrustes-block",
          "lambda_relerr_procrustes_blockdiag", pct=True)
    _row4("diag(Q) rel.err [real]",  "diagQ_relerr_real",  pct=True)
    _row4("diag(Q) rel.err [fin]",   "diagQ_relerr_fin",   pct=True)
    _row4("R median rel.err",        "R_median_relerr",    pct=True)
    _row4("|corr| factor [real]",    "factor_abscorr_real")
    _row4("|corr| factor [fin]",     "factor_abscorr_fin")
    _row4("convergence rate",        "converged")
    _row4("n_iter",                  "n_iter")
    print(bar)
    print("  Interpretation:")
    print("    A: Student-t > Gaussian on loglik (+gap = benefit of tail-robustness)")
    print("    B: Student-t ≈ Gaussian on ALL metrics (nesting confirmed if gap ~ 0)")
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

    # ── Build theta_star^B ────────────────────────────────────────────────────
    theta_star_B = _build_theta_star_B(theta_star)
    print(f"\ntheta_star^B (Gaussian DGP):")
    print(f"  nu_u   = {theta_star_B['nu_u']}   (was {nu_u_A:.3f} in theta_star)")
    print(f"  nu_eps = {theta_star_B['nu_eps']}  (was {nu_eps_A:.3f} in theta_star)")
    print(f"  Lambda, A, Q, R, Sigma_0: UNCHANGED (identical to Experiment A DGP)")
    print(f"  -> only the tail behaviour differs between Experiment A and B")

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
    OUT_DIR_B  = _PROJECT_ROOT / "output" / "monte_carlo" / config / "expB"
    OUT_DIR_A  = _PROJECT_ROOT / "output" / "monte_carlo" / config / "expA"

    print(f"\noutput_dir  : {OUT_DIR_B}")
    print(f"config      : {config}")
    print(f"resume      : True  (safe to interrupt and re-run)")

    # ── Run grid ──────────────────────────────────────────────────────────────
    grid_result = run_grid(
        theta_star=theta_star_B,
        S=S_RUN,
        T_grid=T_GRID,
        estimators=ESTIMATORS,
        pi_grid=PI_GRID,
        freq_list=freq_list,
        block_map=block_map,
        ordered_cols=ordered_cols,
        output_dir=OUT_DIR_B,
        resume=True,
    )
    print(f"\nrun_grid done: {grid_result['n_computed']} computed, "
          f"{grid_result['n_skipped']} skipped")

    # ── Nesting diagnostics at each T ─────────────────────────────────────────
    with _tee_file(OUT_DIR_B / "results.txt"):
        for T in T_GRID:
            # Load the two scenarios for this T from the saved JSON files.
            path_st = OUT_DIR_B / _scenario_filename("student_t", T, 0.0, S_RUN)
            path_g  = OUT_DIR_B / _scenario_filename("gaussian",  T, 0.0, S_RUN)
            if not path_st.exists() or not path_g.exists():
                print(f"\n[WARN] missing scenario files for T={T}, skipping diagnostics")
                continue
            agg_st = _load_scenario(path_st)["aggregates"]
            agg_g  = _load_scenario(path_g)["aggregates"]

            # Print per-T nesting table.
            _loglik_gap_table(agg_st, agg_g, T=T, S=S_RUN)

            # nu_u_hat distribution for Student-t estimator at this T.
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
                    print(f"    (Brent upper bound = 1000; expect saturation toward high nu)")
                else:
                    print(f"    all nu_u_hat are inf (capped above Brent bound)")

        # ── A vs B comparison at T=497 (the headline scenario) ────────────────
        _compare_A_vs_B(
            dir_A=OUT_DIR_A,
            dir_B=OUT_DIR_B,
            T=497,
            S_A=S_RUN,
            S_B=S_RUN,
        )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _extra(p):
        p.add_argument("--S", type=int, default=20,
                       help="Replications per scenario (default 20 for test)")
        p.add_argument("--full", action="store_true",
                       help="Full run: S=1000, T_grid=[100,200,400,800,497]")

    args = parse_config_args(
        "Run Experiment B (Nesting test, DGP Gaussian)",
        extra=_extra,
    )
    S = 1000 if args.full else args.S
    main(S=S, full=args.full, config=args.config)
