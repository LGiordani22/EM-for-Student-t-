r"""
src/monte_carlo.py
==================

Monte Carlo framework for the Student-t mixed-frequency DFM —
S parallel replications, grid of scenarios, aggregate Bias/RMSE statistics.

Three layers of functionality
------------------------------

**Single-replication runner** (:func:`run_one_replication`)
    One full simulate-then-estimate cycle for a single ``(seed, T, estimator,
    π)`` configuration.  Returns a compact dict of scalar metrics (no panels,
    no factor paths, no weight series) so that a grid of S = 1000 runs stays
    memory-light.  This is the kernel executed by every worker in the parallel
    layer.

**Parallel S replications** (:func:`run_monte_carlo`)
    Runs S independent replications via :class:`multiprocessing.Pool` with
    ``imap``.  Aggregates the per-replication metric dicts into Bias / RMSE
    statistics via :func:`aggregate_replications`.  Failed replications are
    caught per-worker and counted in ``n_failed``, without crashing the pool.

**Grid + incremental saving** (:func:`run_grid`)
    Iterates :func:`run_monte_carlo` over the Cartesian product
    ``T_grid × estimators × pi_grid``.  Each completed scenario is
    immediately persisted to a JSON file in ``output_dir``; ``resume=True``
    (default) skips scenarios whose file already exists, making the runner
    safe to interrupt and restart — essential for cluster jobs with wall-clock
    limits.

This module is a PURE ENGINE
----------------------------
It owns only the generic Monte Carlo machinery: simulate-then-estimate,
parallel replication, aggregation, and grid I/O.  It contains **no**
experiment-specific logic (no calibrated DGP construction, no per-experiment
reporting).  The three thesis experiments are thin wrappers that configure
this engine via :func:`run_grid`:

    * ``run_experiment_a.py`` — DGP Student-t (calibrated, nu ~ 4).  Both
      estimators; the Student-t estimator is expected to beat the Gaussian
      one (the latter, lacking weight-attenuation, inflates Q / R under the
      heavy-tailed DGP).
    * ``run_experiment_b.py`` — DGP Gaussian (theta_star with nu → ∞).
      Nesting check: the two estimators should coincide (loglik gap ~ 0).
    * ``run_experiment_c.py`` — contamination robustness (pi > 0).
      Implemented: each observation is contaminated with probability *pi*
      (Bernoulli indicator z_t); contaminated observations replace their
      idiosyncratic shock with a draw from t_{nu_contam}(0, kappa^2 R)
      (factor signal left intact).  The simulator returns ``contam_mask``
      (binary ground-truth vector); :func:`compute_contamination_detection`
      computes detection rates from it.

Each wrapper loads the calibrated ``theta_star``, optionally transforms it
(B sets nu = inf; A leaves it unchanged), and calls :func:`run_grid` with its
own ``output_dir``.  The ``__main__`` of this module is only an engine
smoke-test (a single small replication), never an experiment.

Reference: thesis section Monte Carlo, line ~11200-12800.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import json
import multiprocessing
import os
import time

# Set BLAS thread count to 1 BEFORE importing numpy.
# Each worker process (spawn on Windows) re-imports this module from scratch,
# so these env-vars are set before numpy/BLAS are initialised in every worker.
# Without this, each of the n_jobs processes would use the default multi-thread
# BLAS, causing n_jobs × nthreads thread contention and severe slowdown.
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import numpy as np

# ─── Locate project root and make sibling modules importable ─────────────────

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader  import BLOCK, FREQ, ORDERED_COLS                  # noqa: E402
from em_main      import fit_dfm, load_dfm_fit, _theta_fingerprint  # noqa: E402
from simulate_dfm import simulate_dfm                               # noqa: E402

from monte_carlo_recovery import (                                  # noqa: E402
    init_theta_from_synthetic,
    align_sign_per_factor,
    procrustes_orthogonal,
    procrustes_block_diagonal,
    apply_factor_rotation,
    compute_outlier_rank_overlap,
    compute_contamination_detection,
)


_BLOCK_ORDER:  list[str]      = ["real", "financial", "other"]
_BLOCK_TO_COL: dict[str, int] = {b: j for j, b in enumerate(_BLOCK_ORDER)}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   METRIC HELPERS                                                         ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _signed_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN-safe via simple mean / std centred form."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xz = x - x.mean()
    yz = y - y.mean()
    sx = float(np.sqrt((xz ** 2).sum()))
    sy = float(np.sqrt((yz ** 2).sum()))
    if sx == 0 or sy == 0:
        return float("nan")
    return float((xz @ yz) / (sx * sy))


def _abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    c = _signed_corr(x, y)
    return float(abs(c)) if np.isfinite(c) else float("nan")


def _sorted_eigvals(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of A sorted by descending modulus (complex-valued)."""
    w = np.linalg.eigvals(A)
    return w[np.argsort(-np.abs(w))]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   PER-REPLICATION DRIVER                                                 ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_one_replication(
    *,
    seed: int,
    theta_star: dict,
    T: int,
    estimator: str,
    pi: float = 0.0,
    nu_contam: float = 3.0,
    kappa: float = 5.0,
    freq_list: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    ordered_cols: list[str] | None = None,
    r: int | None = None,
    max_iter: int = 200,
    tol_outer: float = 1e-5,
    verbose_em: bool = False,
) -> dict[str, Any]:
    r"""
    Execute one Monte Carlo replication and return a compact scalar
    metric dict.

    Pipeline
    --------
    1. ``sim = simulate_dfm(theta_star, T, seed=seed, pi=pi, ...)`` —
       draw a synthetic mixed-frequency panel from the calibrated DGP.
       For Experiment C (``π > 0``) the simulator additionally injects
       additive outliers in a fraction ``π`` of periods (heavy
       ``t_{nu_contam}(0, kappa^2 R)`` spikes) and returns the binary
       ground-truth ``contam_mask`` of which periods were contaminated.
       At ``π = 0`` (Experiments A / B) the panel is bit-identical to
       the contamination-free DGP and the mask is all-``False``.
    2. ``theta_0 = init_theta_from_synthetic(sim['Y'])`` — fresh PCA
       initial guess from the synthetic panel.  Never sees
       ``theta_star``.
    3. ``fit = fit_dfm(sim['Y'], theta_0, gaussian=<bool>, save_path=None)``
       — re-fit with the chosen estimator.  ``save_path=None`` is
       critical: the cached real-data fit at
       ``data/processed/fit_dfm_result.npz`` is never overwritten.
    4. Compute the recovery metrics relative to ``theta_star`` and to
       the ground-truth latents ``sim['F']``, ``sim['w_u_true']``,
       ``sim['w_eps_true']``.  Return *only scalars* (no panels, no
       factor paths) so a future :math:`S = 1000` × grid run stays
       memory-light.

    Parameters
    ----------
    seed : int
        Master seed for :func:`simulate_dfm`.  Two calls with the
        same ``seed`` and ``T`` draw the **same** synthetic panel,
        so changing only ``estimator`` between calls produces a
        head-to-head comparison on identical data.
    theta_star : dict
        Calibrated DGP — the converged real-data fit (loaded via
        :func:`em_main.load_dfm_fit`).
    T : int
        Length of the synthetic panel.
    estimator : {"student_t", "gaussian"}
        Estimator applied to the synthetic panel.
    pi : float, default 0.0
        Contamination fraction (Experiment C).  ``π = 0`` reduces the
        DGP exactly to Experiments A / B (no contamination); ``π > 0``
        injects additive outliers in a Bernoulli(``π``) fraction of
        periods.
    nu_contam : float, default 3.0
        Degrees of freedom of the heavy-tailed contaminating shock
        ``t_{nu_contam}(0, kappa^2 R)``.  Exposed as an argument so the
        Experiment-C wrapper can sweep the outlier severity; ignored
        when ``π = 0``.
    kappa : float, default 5.0
        Scale multiplier of the contaminating shock (covariance
        ``kappa^2 R``, ≈ 25x the baseline at the default).  Exposed for
        the Experiment-C wrapper; ignored when ``π = 0``.
    freq_list, block_map, ordered_cols, r
        Panel metadata.  ``None`` defaults are loaded from
        :mod:`data_loader`.
    max_iter : int
        EM outer iteration budget.  Default 200 (enough for the
        synthetic panels at the empirical sample lengths; the real
        data converges in 102--146 iterations).
    tol_outer : float
        Relative-ELBO outer convergence tolerance.  Default 1e-5.
    verbose_em : bool
        Forwarded to ``fit_dfm``; default False to keep logs short.

    Returns
    -------
    dict with scalar / small-array entries — see ``RETURNED_KEYS``
    below for the full list.  Designed so that a list of these
    dicts can be coerced to a pandas DataFrame for the
    aggregate Bias / RMSE statistics of :func:`aggregate_replications`.
    """
    if estimator not in {"student_t", "gaussian"}:
        raise ValueError(
            f"estimator must be 'student_t' or 'gaussian'; got {estimator!r}."
        )

    # ── Defaults from data_loader ────────────────────────────────────────────
    if freq_list    is None: freq_list    = [FREQ[c] for c in ORDERED_COLS]
    if block_map    is None: block_map    = BLOCK
    if ordered_cols is None: ordered_cols = ORDERED_COLS
    if r            is None: r            = int(np.asarray(theta_star["A"]).shape[0])

    # ── 1. Simulate the synthetic panel ──────────────────────────────────────
    # pi / nu_contam / kappa drive the Experiment-C additive-outlier injection.
    # At pi=0 the panel is bit-identical to the contamination-free DGP and
    # contam_mask is all-False (Experiments A / B).
    sim = simulate_dfm(
        theta=theta_star, T=T,
        freq_list=freq_list, block_map=block_map,
        ordered_cols=ordered_cols, r=r,
        seed=seed,
        pi=pi, nu_contam=nu_contam, kappa=kappa,
    )
    Y          = sim["Y"]
    F_true     = sim["F"]
    w_u_true   = sim["w_u_true"]
    w_eps_true = sim["w_eps_true"]
    contam_mask = sim["contam_mask"]                 # (T,) bool ground truth

    # ── 2. Fresh PCA init on the synthetic panel ─────────────────────────────
    theta_0, _ = init_theta_from_synthetic(
        Y,
        ordered_cols=ordered_cols, block_map=block_map, freq_map=FREQ,
    )

    # ── 3. Re-fit with the chosen estimator ──────────────────────────────────
    gaussian_flag = (estimator == "gaussian")
    fit = fit_dfm(
        Y=Y,
        theta_init=theta_0,
        freq_list=freq_list,
        block_map=block_map,
        ordered_cols=ordered_cols,
        max_iter=max_iter,
        tol_outer=tol_outer,
        verbose=verbose_em,
        save_path=None,
        gaussian=gaussian_flag,
        # Experiment B (DGP Gaussian, student_t estimator) REQUIRES the full
        # ELBO as the monitored objective — the Kalman-loglik proxy breaks
        # down as nu -> inf.  We pass use_full_elbo=True explicitly here, for
        # every experiment, so that a future change to fit_dfm's default would
        # not silently break Experiment B.
        use_full_elbo=True,
    )
    theta_hat   = fit["theta"]
    f_smooth   = np.asarray(fit["f_smooth"])
    estep      = fit["e_step_output"]
    w_u_hat    = np.asarray(estep["w_u"])
    w_eps_hat  = np.asarray(estep["w_eps"])

    # ── 4. Recovery metrics ──────────────────────────────────────────────────
    metrics = compute_replication_metrics(
        theta_star=theta_star,
        theta_hat=theta_hat,
        f_smooth_hat=f_smooth,
        F_true=F_true,
        w_u_hat=w_u_hat,
        w_u_true=w_u_true,
        w_eps_hat=w_eps_hat,
        w_eps_true=w_eps_true,
        ordered_cols=ordered_cols,
        block_map=block_map,
        # Experiments A / B (pi=0) pass None so no detection keys are produced;
        # Experiment C (pi>0) passes the binary ground-truth mask.  (At pi=0 the
        # mask is all-False anyway — None is the explicit "not applicable".)
        contam_mask=(contam_mask if pi > 0.0 else None),
    )

    # ── 4b. Gaussian nu marker ───────────────────────────────────────────────
    # em_main.fit_dfm already sets theta["nu_u"] = theta["nu_eps"] = np.inf
    # for gaussian=True, so compute_replication_metrics already returns
    # nu_u_hat = nu_eps_hat = inf.  We still override the relerr to nan
    # (inf - nu_star / |nu_star| = inf, which would look like a large error
    # rather than "not applicable") and set the nu_frozen flag used by the
    # print functions to distinguish "N/A by design" from a numeric failure.
    metrics["nu_frozen"] = gaussian_flag
    if gaussian_flag:
        metrics["nu_u_relerr"]   = float("nan")
        metrics["nu_eps_relerr"] = float("nan")

    # ── 5. Algorithmic reliability ───────────────────────────────────────────
    metrics["seed"]                   = int(seed)
    metrics["T"]                      = int(T)
    metrics["estimator"]              = estimator
    metrics["pi"]                     = float(pi)
    metrics["converged"]              = bool(fit["converged"])
    metrics["n_iter"]                 = int(fit["n_iter"])
    metrics["n_monotonicity_violations"] = int(len(fit["monotonicity_violations"]))
    metrics["loglik_initial"]         = float(fit["loglik_history"][0])
    metrics["loglik_final"]           = float(fit["loglik_history"][-1])

    return metrics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   RECOVERY METRICS (shared between estimators)                           ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_replication_metrics(
    *,
    theta_star: dict,
    theta_hat: dict,
    f_smooth_hat: np.ndarray,
    F_true: np.ndarray,
    w_u_hat: np.ndarray,
    w_u_true: np.ndarray,
    w_eps_hat: np.ndarray,
    w_eps_true: np.ndarray,
    ordered_cols: list[str],
    block_map: dict[str, str],
    contam_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    r"""
    Compute the per-replication recovery metrics.

    Returned keys (scalars and short arrays only — no full panels):

    Parameter recovery
        - ``nu_u_star``, ``nu_u_hat``, ``nu_u_relerr``
        - ``nu_eps_star``, ``nu_eps_hat``, ``nu_eps_relerr``
        - ``rho_A_star``, ``rho_A_hat``, ``rho_A_relerr``
        - ``eig_A_err_norm`` — Euclidean norm of the eigenvalue
          difference vector (after sorting by descending |·|).
        - ``lambda_relerr_normalised`` — :math:`\|\hat{\Lambda}^{\text{sgn}}
          - \Lambda^*\|_F / \|\Lambda^*\|_F`, sign-aligned.
        - ``lambda_relerr_procrustes_blockdiag`` — same after the
          block-diagonal Procrustes rescaling (the primary
          loadings metric, see monte_carlo_recovery.py (self-recovery test)).
        - ``H_block_diag`` — vector ``(h_R, h_F, h_X)``.
        - ``diagQ_star``, ``diagQ_hat``, ``diagQ_relerr`` (per block).
        - ``R_median_relerr``, ``R_max_relerr``.

    Factor recovery
        - ``factor_abscorr`` — ``(|corr_R|, |corr_F|, |corr_X|)``,
          the within-block factor correlation (the thesis prescribes
          a Pearson correlation here, and ``|·|`` neutralises the
          sign / scale ambiguity that survives Convention 1 on
          finite samples).
        - ``factor_crosscorr`` — 3 cross-block correlations
          ``(|corr(f_R, F_F)|, |corr(f_R, F_X)|, |corr(f_F, F_X)|)``,
          which should be near zero under correct block
          identification (thesis line ~12707).
        - ``factor_rmse_traj`` — RMSE of the smoothed factor
          trajectory against the truth, per block (computed *after*
          rescaling the truth by the block-Procrustes ``h_b`` and
          sign-aligning, so the comparison is on a common scale).

    Weight recovery (Student-t specific)
        - ``w_u_corr``, ``w_eps_corr`` — full-sample linear corr.
        - ``w_u_overlap_5pct``, ``w_eps_overlap_5pct`` —
          precision@5% on the lowest weights (outlier set), the
          metric introduced in monte_carlo_recovery.py.
        - ``w_u_lift_5pct``, ``w_eps_lift_5pct`` — overlap divided
          by the expected-by-chance baseline ``k / T_eff``.

    Contamination detection (Experiment C only)
        Added only when ``contam_mask is not None`` (i.e. ``π > 0``):
        - ``detection_rate_natural``, ``detection_lift_natural`` —
          recall (= precision at ``k = n_contam``) of the true
          contaminated periods among the most down-weighted, and its
          lift over chance.
        - ``detection_rate_5pct`` / ``_10pct`` (recall),
          ``detection_precision_5pct`` / ``_10pct`` (precision),
          ``detection_lift_5pct`` / ``_10pct`` — fixed-fraction variants.
        All ``nan`` for the Gaussian estimator (constant ``w_eps_hat``,
        no down-weighting to score — nan *by design*).  For Experiments
        A / B ``contam_mask`` is ``None`` and these keys are omitted.

    Parameters
    ----------
    contam_mask : np.ndarray of bool (T,), or None
        Experiment-C ground-truth contamination indicator.  ``None`` for
        the contamination-free Experiments A / B (``π = 0``), in which
        case no detection metrics are produced.
    """
    Lambda_star = np.asarray(theta_star["Lambda"])
    Lambda_hat  = np.asarray(theta_hat["Lambda"])
    A_star      = np.asarray(theta_star["A"])
    A_hat       = np.asarray(theta_hat["A"])
    Q_star      = np.asarray(theta_star["Q"])
    Q_hat       = np.asarray(theta_hat["Q"])
    R_star      = np.asarray(theta_star["R"])
    R_hat       = np.asarray(theta_hat["R"])
    r           = int(Lambda_star.shape[1])

    # ── (a) Sign-align the loadings per factor ────────────────────────────────
    d_sign = align_sign_per_factor(Lambda_hat, Lambda_star)
    D      = np.diag(d_sign.astype(float))
    theta_hat_sgn, f_smooth_sgn = apply_factor_rotation(
        theta_hat, f_smooth_hat, D
    )
    Lambda_sgn = np.asarray(theta_hat_sgn["Lambda"])
    Q_sgn      = np.asarray(theta_hat_sgn["Q"])

    # ── (b) Loadings: normalised-only and block-Procrustes-aligned ───────────
    lambda_relerr_normalised = float(
        np.linalg.norm(Lambda_sgn - Lambda_star) / np.linalg.norm(Lambda_star)
    )
    H_block = procrustes_block_diagonal(
        Lambda_hat, Lambda_star,
        ordered_cols=ordered_cols, block_map=block_map,
    )
    Lambda_proc_block = Lambda_hat @ H_block
    lambda_relerr_procrustes_blockdiag = float(
        np.linalg.norm(Lambda_proc_block - Lambda_star)
        / np.linalg.norm(Lambda_star)
    )

    # ── (c) Eigenvalues of A and spectral radius ─────────────────────────────
    eig_star = _sorted_eigvals(A_star)
    eig_hat  = _sorted_eigvals(A_hat)
    eig_A_err_norm = float(np.linalg.norm(eig_hat - eig_star))
    rho_A_star = float(np.max(np.abs(eig_star)))
    rho_A_hat  = float(np.max(np.abs(eig_hat)))

    # ── (d) Q, R recoveries (per-block diag of Q; per-series R) ──────────────
    diagQ_star   = np.diag(Q_star)
    diagQ_hat    = np.diag(Q_sgn)
    diagQ_relerr = np.abs(diagQ_hat - diagQ_star) / np.maximum(np.abs(diagQ_star), 1e-12)
    R_relerr     = np.abs(R_hat - R_star)        / np.maximum(np.abs(R_star),     1e-12)

    # ── (e) Factor recovery — within and cross-block ─────────────────────────
    # f_smooth_sgn[:, :r] is the sign-aligned contemporaneous block.  F_true
    # is the raw simulator output (Conv-1 NOT applied).  We use |corr| for the
    # within-block metric (scale + sign invariant) — exactly as in the
    # self-recovery test.
    f_hat_now = f_smooth_sgn[:, :r]
    factor_abscorr = np.array([
        _abs_corr(f_hat_now[:, j], F_true[:, j]) for j in range(r)
    ])
    # Cross-block (off-diagonal) — should be near 0 under correct block
    # identification (thesis line ~12707).
    cross_pairs = [(0, 1), (0, 2), (1, 2)]
    factor_crosscorr = np.array([
        _abs_corr(f_hat_now[:, i], F_true[:, j]) for (i, j) in cross_pairs
    ])

    # Trajectory RMSE per block.  Bring F_true to f_hat's scale via the
    # block-Procrustes h_b to make the absolute distance meaningful.
    h_diag = np.diag(H_block)  # (r,)
    factor_rmse_traj = np.empty(r)
    for j in range(r):
        # Match the same scale convention used in the loadings comparison:
        # Lambda_proc_block = Lambda_hat @ H_block  =>  the loadings are
        # rescaled by h_b, which is equivalent to dividing the factor by
        # the same h_b.  We rescale f_hat by h_b to bring it to the
        # F_true scale (and use the sign-aligned version).
        f_hat_resc = f_hat_now[:, j] * h_diag[j]
        factor_rmse_traj[j] = float(np.sqrt(np.mean((f_hat_resc - F_true[:, j]) ** 2)))

    # ── (f) Weight recovery — linear corr + precision@5% ─────────────────────
    # Under the Gaussian estimator w_u ≡ w_eps ≡ 1 (constant — the Gaussian
    # model has no scale-mixture weights).  _signed_corr returns nan when the
    # estimated series has zero variance; aggregate_replications skips all-nan
    # keys, so these metrics simply do not appear for gaussian scenarios.
    w_u_corr   = _signed_corr(w_u_hat,   w_u_true)
    w_eps_corr = _signed_corr(w_eps_hat, w_eps_true)
    ov_u   = compute_outlier_rank_overlap(w_u_hat,   w_u_true,   k_frac=0.05)
    ov_eps = compute_outlier_rank_overlap(w_eps_hat, w_eps_true, k_frac=0.05)

    # ── (g) Contamination detection — Experiment C only ──────────────────────
    # The binary-truth analogue of the overlap above: how well the lowest
    # w_eps_hat (the periods the Student-t down-weights) line up with the TRUE
    # contaminated periods (contam_mask).  For Experiments A / B contam_mask is
    # None -> empty dict -> no detection keys.  For the Gaussian estimator the
    # function itself returns nan-valued keys (constant w_eps_hat, no ranking),
    # so the keys stay present and uniform across the C-Gaussian scenario.
    detection = compute_contamination_detection(w_eps_hat, contam_mask)

    metrics_out = {
        "nu_u_star":        float(theta_star["nu_u"]),
        "nu_u_hat":         float(theta_hat["nu_u"]),
        "nu_u_relerr":      abs(float(theta_hat["nu_u"]) - float(theta_star["nu_u"]))
                              / max(abs(float(theta_star["nu_u"])), 1e-12),
        "nu_eps_star":      float(theta_star["nu_eps"]),
        "nu_eps_hat":       float(theta_hat["nu_eps"]),
        "nu_eps_relerr":    abs(float(theta_hat["nu_eps"]) - float(theta_star["nu_eps"]))
                              / max(abs(float(theta_star["nu_eps"])), 1e-12),
        "rho_A_star":       rho_A_star,
        "rho_A_hat":        rho_A_hat,
        "rho_A_relerr":     abs(rho_A_hat - rho_A_star) / max(abs(rho_A_star), 1e-12),
        "eig_A_err_norm":   eig_A_err_norm,
        "lambda_relerr_normalised":           lambda_relerr_normalised,
        "lambda_relerr_procrustes_blockdiag": lambda_relerr_procrustes_blockdiag,
        "H_block_diag":     h_diag.astype(float),
        "diagQ_star":       diagQ_star,
        "diagQ_hat":        diagQ_hat,
        "diagQ_relerr":     diagQ_relerr,
        "R_median_relerr":  float(np.median(R_relerr)),
        "R_max_relerr":     float(np.max(R_relerr)),
        "factor_abscorr":   factor_abscorr,
        "factor_crosscorr": factor_crosscorr,
        "factor_rmse_traj": factor_rmse_traj,
        "w_u_corr":         float(w_u_corr),
        "w_eps_corr":       float(w_eps_corr),
        "w_u_overlap_5pct": float(ov_u["overlap"]),
        "w_u_lift_5pct":    float(ov_u["lift"]),
        "w_eps_overlap_5pct": float(ov_eps["overlap"]),
        "w_eps_lift_5pct":    float(ov_eps["lift"]),
    }
    # Merge the (possibly empty) Experiment-C detection metrics.  Empty for
    # A / B; the scalar detection_* keys are flowed through aggregate_replications
    # automatically (they are plain floats).
    metrics_out.update(detection)
    return metrics_out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   Parallel S replications + aggregation                                  ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _mc_worker(args: tuple):
    r"""
    Top-level worker for :class:`multiprocessing.Pool`.

    Wraps :func:`run_one_replication` in a ``try/except`` so that a rare
    numerical failure (e.g. non-convergence, singular matrix) does **not**
    crash the entire pool.  Returns ``None`` on failure; the caller counts
    these in ``n_failed`` and aggregates only the successful replications.

    Must be a module-level function (not a lambda / nested def) so that
    ``pickle`` can serialise it for the spawned worker processes on Windows.

    ``nu_contam`` / ``kappa`` ride in the args tuple right after ``pi`` (the
    contamination overlay parameters for Experiment C); they are forwarded to
    :func:`run_one_replication` and ignored there when ``pi = 0`` (A / B).
    """
    (seed, theta_star, T, estimator, pi, nu_contam, kappa,
     freq_list, block_map, ordered_cols, r, max_iter, tol) = args
    try:
        return run_one_replication(
            seed=seed,
            theta_star=theta_star,
            T=T,
            estimator=estimator,
            pi=pi,
            nu_contam=nu_contam,
            kappa=kappa,
            freq_list=freq_list,
            block_map=block_map,
            ordered_cols=ordered_cols,
            r=r,
            max_iter=max_iter,
            tol_outer=tol,
            verbose_em=False,
        )
    except Exception as exc:
        print(
            f"  [WARN] replica seed={seed} failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


# ── Metric classification tables ─────────────────────────────────────────────

# Estimate-type keys: paired with their true-value key for bias / RMSE.
_ESTIMATE_TO_TRUE: dict[str, str] = {
    "nu_u_hat":        "nu_u_star",
    "nu_eps_hat":      "nu_eps_star",
    "rho_A_hat":       "rho_A_star",
    # Expanded Q-diagonal keys (after array expansion below):
    "diagQ_hat_real":  "diagQ_star_real",
    "diagQ_hat_fin":   "diagQ_star_fin",
    "diagQ_hat_other": "diagQ_star_other",
}

# Error-type keys: already deviations from truth; RMSE = sqrt(mean(x^2)).
# Matched by substring: any key whose name contains one of these patterns.
_ERROR_PATTERNS: tuple[str, ...] = ("relerr", "err_norm")

_BLOCK_NAMES: tuple[str, ...] = ("real", "fin", "other")
_CROSS_NAMES: tuple[str, ...] = ("RF", "RX", "FX")

# Array keys of shape (r,) that expand to one scalar per block:
_ARRAY_BLOCK_KEYS: frozenset[str] = frozenset({
    "factor_abscorr", "factor_rmse_traj", "H_block_diag",
    "diagQ_star", "diagQ_hat", "diagQ_relerr",
})


def _expand_one_rep(d: dict) -> dict:
    """
    Flatten array-valued entries in a single-replica dict to scalar sub-keys.

    ``factor_abscorr`` (shape (3,)) → ``factor_abscorr_real``,
    ``factor_abscorr_fin``, ``factor_abscorr_other``.
    Strings are kept as-is; other non-numeric types are dropped.
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == 3:
            if k in _ARRAY_BLOCK_KEYS:
                for j, b in enumerate(_BLOCK_NAMES):
                    out[f"{k}_{b}"] = float(v[j])
            elif k == "factor_crosscorr":
                for j, n in enumerate(_CROSS_NAMES):
                    out[f"{k}_{n}"] = float(v[j])
            else:
                for j in range(3):
                    out[f"{k}_{j}"] = float(v[j])
        elif isinstance(v, (int, float, bool, np.integer, np.floating)):
            out[k] = float(v)
        elif isinstance(v, str):
            out[k] = v
    return out


def aggregate_replications(replications: list[dict]) -> dict:
    r"""
    Aggregate scalar recovery metrics across *S* successful replications.

    For every numeric key in the metric dicts the function returns a
    sub-dict with:

    * ``mean``, ``std``, ``median``, ``q05``, ``q95`` — always present.
    * ``bias``, ``rmse`` — for *estimate-type* keys (those listed in
      :data:`_ESTIMATE_TO_TRUE`): ``bias = mean(x̂) - x*``,
      ``rmse = sqrt(mean((x̂ - x*)²))``.
    * ``rmse`` only — for *error-type* keys (containing ``'relerr'`` or
      ``'err_norm'`` in their name): ``rmse = sqrt(mean(x²))``.

    Parameters
    ----------
    replications
        List of successful (non-``None``) metric dicts from
        :func:`run_one_replication`.

    Returns
    -------
    dict
        ``{expanded_key: {mean, std, median, q05, q95, [bias], [rmse]}}``.
    """
    if not replications:
        return {}

    expanded = [_expand_one_rep(d) for d in replications]
    numeric_keys = [k for k, v in expanded[0].items() if isinstance(v, float)]

    agg: dict[str, dict] = {}
    for key in numeric_keys:
        vals   = np.array([d[key] for d in expanded if key in d], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        n = len(finite)
        stats: dict[str, float] = {
            "mean":   float(finite.mean()),
            "std":    float(finite.std(ddof=1)) if n > 1 else float("nan"),
            "median": float(np.median(finite)),
            "q05":    float(np.quantile(finite, 0.05)),
            "q95":    float(np.quantile(finite, 0.95)),
        }
        if key in _ESTIMATE_TO_TRUE:
            star_key = _ESTIMATE_TO_TRUE[key]
            star_val = float(expanded[0][star_key])
            stats["bias"] = stats["mean"] - star_val
            stats["rmse"] = float(np.sqrt(np.mean((finite - star_val) ** 2)))
        elif any(p in key for p in _ERROR_PATTERNS):
            stats["rmse"] = float(np.sqrt(np.mean(finite ** 2)))
        agg[key] = stats

    return agg


def print_aggregate_table(
    agg: dict,
    estimator: str,
    T: int,
    S: int,
    n_failed: int = 0,
) -> None:
    """Print a formatted summary of :func:`aggregate_replications` output."""
    bar = "=" * 102
    print("\n" + bar)
    print(
        f"  AGGREGATE STATISTICS  —  "
        f"S={S} ({S - n_failed} ok, {n_failed} failed), "
        f"T={T}, estimator={estimator!r}"
    )
    print(bar)

    def _fmt(s: dict) -> str:
        parts = [f"{s['mean']:>9.4f}"]
        if "bias" in s:
            parts.append(f"  bias={s['bias']:+.4f}  RMSE={s['rmse']:.4f}")
        elif "rmse" in s:
            parts.append(f"  RMSE={s['rmse']:.4f}")
        std_v = s.get("std", float("nan"))
        if np.isfinite(std_v):
            parts.append(f"  std={std_v:.4f}")
        parts.append(f"  [{s['q05']:.3f}, {s['q95']:.3f}]")
        return "".join(parts)

    groups = [
        ("Heavy-tail parameters (estimates)", [
            ("nu_u_hat",       "nu_u  (true = nu_u_star)"),
            ("nu_eps_hat",     "nu_eps (true = nu_eps_star)"),
            ("nu_u_relerr",    "nu_u  rel.err"),
            ("nu_eps_relerr",  "nu_eps rel.err"),
        ]),
        ("Spectral radius of A", [
            ("rho_A_hat",       "rho(A) (estimate)"),
            ("rho_A_relerr",    "rho(A) rel.err"),
            ("eig_A_err_norm",  "|eig(A_hat) - eig(A*)| Euclidean"),
        ]),
        ("Loading matrix Lambda", [
            ("lambda_relerr_procrustes_blockdiag",
             "Lambda relerr Procrustes-block  [PRIMARY]"),
            ("lambda_relerr_normalised",
             "Lambda relerr (sign-normalised only)"),
        ]),
        ("Block-diagonal Procrustes scale factors", [
            ("H_block_diag_real",  "h_real"),
            ("H_block_diag_fin",   "h_financial"),
            ("H_block_diag_other", "h_other"),
        ]),
        ("Q diagonal per block", [
            ("diagQ_hat_real",   "diag(Q) estimate [real]"),
            ("diagQ_hat_fin",    "diag(Q) estimate [financial]"),
            ("diagQ_hat_other",  "diag(Q) estimate [other]"),
            ("diagQ_relerr_real",  "diag(Q) rel.err  [real]"),
            ("diagQ_relerr_fin",   "diag(Q) rel.err  [financial]"),
            ("diagQ_relerr_other", "diag(Q) rel.err  [other]"),
        ]),
        ("R (idiosyncratic variances)", [
            ("R_median_relerr", "R rel.err  (median over series)"),
            ("R_max_relerr",    "R rel.err  (max over series)"),
        ]),
        ("Factor recovery", [
            ("factor_abscorr_real",    "|corr| factor  [real]"),
            ("factor_abscorr_fin",     "|corr| factor  [financial]"),
            ("factor_abscorr_other",   "|corr| factor  [other]"),
            ("factor_crosscorr_RF",    "|cross-corr|  real–financial"),
            ("factor_crosscorr_RX",    "|cross-corr|  real–other"),
            ("factor_crosscorr_FX",    "|cross-corr|  financial–other"),
            ("factor_rmse_traj_real",  "RMSE trajectory  [real]"),
            ("factor_rmse_traj_fin",   "RMSE trajectory  [financial]"),
            ("factor_rmse_traj_other", "RMSE trajectory  [other]"),
        ]),
        ("Weight recovery (Student-t specific)", [
            ("w_u_corr",          "corr(w_u_hat,   w_u_true)"),
            ("w_eps_corr",        "corr(w_eps_hat, w_eps_true)"),
            ("w_u_overlap_5pct",  "w_u   overlap@5%"),
            ("w_eps_overlap_5pct","w_eps overlap@5%"),
            ("w_u_lift_5pct",     "w_u   lift@5%   (overlap / chance)"),
            ("w_eps_lift_5pct",   "w_eps lift@5%   (overlap / chance)"),
        ]),
        ("Algorithmic reliability", [
            ("converged",                 "convergence rate"),
            ("n_iter",                    "iterations (mean)"),
            ("n_monotonicity_violations", "monotonicity violations (mean)"),
        ]),
    ]

    for title, metrics in groups:
        if not any(k in agg for k, _ in metrics):
            if title == "Heavy-tail parameters (estimates)" and estimator == "gaussian":
                print(f"\n  [{title}]")
                print(f"    {'nu_u  (estimate)':<54s}  inf (Gaussian limit)")
                print(f"    {'nu_eps (estimate)':<54s}  inf (Gaussian limit)")
                print(f"    {'nu_u  rel.err':<54s}  N/A (Gaussian)")
                print(f"    {'nu_eps rel.err':<54s}  N/A (Gaussian)")
            continue
        print(f"\n  [{title}]")
        for key, label in metrics:
            if key not in agg:
                if (title == "Heavy-tail parameters (estimates)"
                        and estimator == "gaussian"
                        and ("relerr" in key)):
                    # nu_u_relerr / nu_eps_relerr are all-nan for Gaussian
                    # (nu is not a free parameter): show N/A, not a missing row.
                    print(f"    {label:<54s}  N/A (Gaussian)")
                continue
            print(f"    {label:<54s}  {_fmt(agg[key])}")

    print("\n" + bar)


def run_monte_carlo(
    theta_star: dict,
    S: int,
    T: int,
    estimator: str,
    freq_list: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    ordered_cols: list[str] | None = None,
    r: int | None = None,
    pi: float = 0.0,
    nu_contam: float = 3.0,
    kappa: float = 5.0,
    n_jobs: int | None = None,
    base_seed: int = 1000,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> dict:
    r"""
    Run *S* independent Monte Carlo replications in parallel and aggregate
    the recovery metrics.

    Parameters
    ----------
    theta_star : dict
        Calibrated DGP — the converged real-data fit loaded via
        :func:`em_main.load_dfm_fit`.
    S : int
        Number of replications.
    T : int
        Synthetic panel length in months.
    estimator : {"student_t", "gaussian"}
        Estimator applied to each synthetic panel.
    freq_list, block_map, ordered_cols, r
        Panel metadata.  ``None`` → defaults from :mod:`data_loader`.
    pi : float, default 0.0
        Contamination fraction (Experiment C).  When pi > 0 the simulator
        injects idiosyncratic contamination (Bernoulli z_t, heavy-tailed
        replacement shock) and returns ``contam_mask`` for detection metrics.
    nu_contam : float, default 3.0
        Degrees of freedom of the contaminating Student-t shock
        ``t_{nu_contam}(0, kappa^2 R)``.  Forwarded per-replication; ignored
        when ``pi = 0``.
    kappa : float, default 5.0
        Scale-inflation factor of the contaminating shock (covariance
        ``kappa^2 R``).  Forwarded per-replication; ignored when ``pi = 0``.
    n_jobs : int or None
        Worker processes.  ``None`` → ``min(cpu_count()-1, 6)``
        (leaves one core free; capped at 6 for the Ryzen 5 5600G).
    base_seed : int, default 1000
        Replication *s* (0-indexed) uses ``seed = base_seed + s``.
        Reproducible: the same ``base_seed`` always yields the same
        set of replications regardless of execution order.
    max_iter : int, default 200
        EM outer-iteration budget per replication.
    tol : float, default 1e-5
        Relative-ELBO outer convergence tolerance.

    Returns
    -------
    dict with keys:

    ``per_replication``
        List of *S* metric dicts (``None`` for failed replicas).
    ``aggregates``
        Nested ``{metric_key: {mean, std, …}}`` dict from
        :func:`aggregate_replications`.
    ``config``
        Run configuration (S, T, estimator, pi, n_jobs, base_seed).
    ``n_failed``
        Number of replications that raised an exception.
    ``wall_time_seconds``
        Total wall-clock time of the parallel run.
    """
    if freq_list    is None: freq_list    = [FREQ[c] for c in ORDERED_COLS]
    if block_map    is None: block_map    = BLOCK
    if ordered_cols is None: ordered_cols = ORDERED_COLS
    if r            is None: r            = int(np.asarray(theta_star["A"]).shape[0])
    if n_jobs is None:
        n_jobs = max(1, min(multiprocessing.cpu_count() - 1, 6))

    print(
        f"\nrun_monte_carlo: S={S}, T={T}, estimator={estimator!r}, "
        f"pi={pi}, nu_contam={nu_contam}, kappa={kappa}, "
        f"n_jobs={n_jobs}, base_seed={base_seed}"
    )

    args_list = [
        (base_seed + s, theta_star, T, estimator, pi, nu_contam, kappa,
         freq_list, block_map, ordered_cols, r, max_iter, tol)
        for s in range(S)
    ]

    t0: float = time.perf_counter()
    replications: list = []
    n_failed: int = 0

    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap(_mc_worker, args_list)):
            replications.append(result)
            if result is None:
                n_failed += 1
            if (i + 1) % 10 == 0 or (i + 1) == S:
                elapsed = time.perf_counter() - t0
                print(
                    f"  completed {i + 1}/{S} replications "
                    f"({n_failed} failed, {elapsed:.1f}s elapsed)",
                    flush=True,
                )

    t_total = time.perf_counter() - t0
    n_success = S - n_failed
    print(
        f"  Done: {n_success}/{S} succeeded in {t_total:.2f}s "
        f"({t_total / S:.2f}s / replica wall-avg)"
    )

    good = [rep for rep in replications if rep is not None]
    aggregates = aggregate_replications(good)

    return {
        "per_replication":  replications,
        "aggregates":       aggregates,
        "config": {
            "S": S, "T": T, "estimator": estimator,
            "pi": pi, "nu_contam": nu_contam, "kappa": kappa,
            "n_jobs": n_jobs, "base_seed": base_seed,
            # Fingerprint of the DGP theta_star, so run_grid's resume check can
            # verify a cached scenario was produced from the SAME theta_star
            # before skipping it (anti zombie-result).
            "theta_fingerprint": _theta_fingerprint(theta_star),
        },
        "n_failed":          n_failed,
        "wall_time_seconds": t_total,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   Grid over (T*, estimator, pi) + incremental saving + resume            ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Design notes
# ------------
# *Why JSON for the scenario files.*  Each Monte Carlo scenario produces
# a small payload (config + aggregates + ~S per-replication metric dicts,
# all of which are pure scalars or length-3 arrays).  JSON is the most
# portable choice across the workstation/cluster boundary: human-readable,
# editor-inspectable, single file per scenario, no pickle compatibility
# issues across Python versions or numpy releases.  Size for S=1000 is
# roughly ~250 KB per scenario.
#
# *Why per-scenario, incremental saving.*  Cluster jobs frequently have
# wall-clock limits, and exploratory runs are routinely interrupted.
# Writing the result to disk as soon as a scenario completes means no
# work is ever lost, and ``resume=True`` lets a follow-up invocation
# skip everything already on disk and re-attempt only the missing
# scenarios.  This is the standard idempotent-runner pattern.
#
# *Seed convention across the grid.*  Every (estimator, π) pair shares
# the same ``base_seed``, so the synthetic panel ``Y^{(s)}`` for replica
# ``s`` is *identical* across estimators at fixed ``T`` — the two
# estimators are then compared on the same data (paired-sampling
# variance reduction).  Different ``T`` values produce different panels
# by construction, which is correct: ``T`` is a dimension of the
# experimental design, not a nuisance.


def _scenario_filename(estimator: str, T: int, pi: float, S: int) -> str:
    r"""
    Build a deterministic, sortable filename for one Monte Carlo scenario.

    Format::

        mc_<estimator>_T<TTTT>_pi<P.PP>_S<S>.json

    where ``T`` is zero-padded to 4 digits (so a directory listing sorts
    naturally: ``T0100`` before ``T0200`` before ``T0497``) and ``pi`` is
    formatted to two decimals (``0.00``, ``0.01``, ``0.05``, ``0.10``).
    """
    return f"mc_{estimator}_T{T:04d}_pi{pi:.2f}_S{S}.json"


def _to_json_safe(obj):
    r"""
    Recursively convert numpy types / arrays to JSON-serialisable
    Python natives.  Handles ``ndarray`` (→ list), ``np.integer`` (→ int),
    ``np.floating`` (→ float), ``np.bool_`` (→ bool); leaves dict / list
    / tuple structures intact while recursing through them.

    ``inf`` and ``nan`` (float or np.floating) are both converted to
    ``None`` (JSON null): ``inf`` to avoid the non-standard ``Infinity``
    token, ``nan`` to produce valid RFC 8259 JSON (``NaN`` is not a legal
    JSON value).  The inverse conversion for known ``inf`` fields is
    handled by :func:`_load_scenario`.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)  # safe fallback


def _save_scenario(path: pathlib.Path, mc_result: dict) -> None:
    """Serialise the dict returned by :func:`run_monte_carlo` to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config":            mc_result["config"],
        "n_failed":          mc_result["n_failed"],
        "wall_time_seconds": mc_result["wall_time_seconds"],
        "aggregates":        mc_result["aggregates"],
        "per_replication":   mc_result["per_replication"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_json_safe(payload), fh, indent=2, allow_nan=True)


# Fields that are stored as null (JSON) for Gaussian replications (inf → None
# in _to_json_safe) and must be restored to float("inf") on load.
#
# Theoretical motivation for null/NaN in Gaussian replications
# ------------------------------------------------------------
# Three families of metrics become null (JSON) or NaN (Python) for the
# Gaussian estimator or Gaussian DGP, for distinct model-theoretic reasons:
#
# (i)  nu_u_hat, nu_eps_hat (and their star counterparts in Exp. B):
#      The degrees-of-freedom parameter nu is a feature of the Student-t
#      scale-mixture representation.  The Gaussian model is the nu → ∞ limit
#      of the Student-t DFM; nu is not a free parameter under the Gaussian
#      estimator and is set to inf by fit_dfm(gaussian=True).  For the DGP
#      of Experiment B (also Gaussian), theta_star["nu"] = inf as well.
#      Both hats and stars are inf → _to_json_safe converts them to null;
#      _load_scenario restores nu_hat fields (listed here) to float("inf").
#
# (ii) nu_u_relerr, nu_eps_relerr:
#      Relative error is inf/|nu_star| = inf, which would masquerade as a
#      large numerical error.  These are overridden to nan in
#      run_one_replication and serialised as null.  aggregate_replications
#      drops all-nan keys automatically (finite= [] → skip), so they never
#      appear in the aggregate table for Gaussian scenarios.
#
# (iii) w_u_corr, w_eps_corr, overlap and lift metrics:
#       Under the Gaussian estimator the EM E-step sets every weight
#       w_u_t ≡ 1 and w_eps_{it} ≡ 1 (no observation-level down-weighting).
#       These are constant sequences with zero variance, so
#       corr(1_vec, w_true) = nan (_signed_corr returns nan when std = 0).
#       The overlap/lift metrics are also degenerate (all weights identical,
#       no outlier ranking possible).  These NaN values are serialised as
#       null and skipped by aggregate_replications.
_INF_FIELDS: frozenset[str] = frozenset({"nu_u_hat", "nu_eps_hat"})


def _load_scenario(path: pathlib.Path) -> dict:
    """Load and deserialise a JSON scenario file."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Restore inf for Gaussian nu fields stored as null
    for rep in payload.get("per_replication", []):
        if isinstance(rep, dict):
            for k in _INF_FIELDS:
                if rep.get(k) is None:
                    rep[k] = float("inf")
    return payload


def run_grid(
    theta_star: dict,
    S: int,
    T_grid: list[int],
    estimators: list[str] = ("student_t", "gaussian"),
    pi_grid: list[float] = (0.0,),
    nu_contam: float = 3.0,
    kappa: float = 5.0,
    freq_list: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    ordered_cols: list[str] | None = None,
    r: int | None = None,
    n_jobs: int | None = None,
    base_seed: int = 1000,
    max_iter: int = 200,
    tol: float = 1e-5,
    output_dir: str | pathlib.Path = "data/processed/mc_results",
    resume: bool = True,
) -> dict:
    r"""
    Iterate :func:`run_monte_carlo` over the Cartesian product
    ``T_grid × estimators × pi_grid`` and persist every scenario to disk
    as soon as it completes.

    Parameters
    ----------
    theta_star : dict
        Calibrated DGP (from :func:`em_main.load_dfm_fit`).
    S : int
        Number of replications per scenario.
    T_grid : list of int
        Sample sizes, e.g. ``[100, 200, 400, 800, 497]`` (the headline
        grid of Section *Choice of T\**, thesis line ~13530).
    estimators : list of str
        Subset of ``{"student_t", "gaussian"}``.
    pi_grid : list of float
        Contamination intensities.  Values > 0 activate Experiment C
        contamination: Bernoulli z_t indicator selects contaminated
        observations; their idiosyncratic shock is replaced by a draw from
        t_{nu_contam}(0, kappa^2 R) while the factor signal is left intact.
        The simulator returns ``contam_mask``; ``run_grid`` passes it through
        to :func:`compute_contamination_detection`.
    output_dir : path-like
        Directory in which per-scenario JSON files are written.
        Created if absent.
    resume : bool, default True
        If ``True`` and a scenario's JSON file already exists in
        ``output_dir``, the scenario is loaded from disk and **not**
        recomputed.  Combined with the incremental write, this makes
        the runner safe to interrupt and resume — essential for cluster
        jobs with wall-clock limits.
    nu_contam : float, default 3.0
        Degrees of freedom of the Experiment-C contaminating shock; forwarded
        verbatim to every :func:`run_monte_carlo` call (ignored at pi = 0).
    kappa : float, default 5.0
        Scale-inflation factor of the Experiment-C contaminating shock;
        forwarded verbatim to every :func:`run_monte_carlo` call (ignored at
        pi = 0).

    Other parameters (``n_jobs``, ``base_seed``, ``max_iter``, ``tol``,
    ``freq_list`` etc.) are forwarded verbatim to
    :func:`run_monte_carlo`; see its docstring.

    Returns
    -------
    dict with keys

    ``scenarios``
        List of the *loaded* per-scenario payloads (each one a dict with
        ``config``, ``aggregates``, ``per_replication``, …).  For very
        large grids prefer to use :func:`load_grid_results` directly on
        ``output_dir`` instead of holding the full list in memory.
    ``n_computed``
        Number of scenarios that were re-estimated in this call.
    ``n_skipped``
        Number of scenarios that were resumed from disk.
    ``output_dir``
        Resolved string path to the output directory.
    """
    estimators = list(estimators)
    pi_grid    = list(pi_grid)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_spec: list[tuple[int, str, float]] = [
        (T, est, pi)
        for T in T_grid
        for est in estimators
        for pi in pi_grid
    ]
    n_total = len(scenarios_spec)

    print(
        f"\nrun_grid: {n_total} scenarios "
        f"= {len(T_grid)} T x {len(estimators)} estimators x {len(pi_grid)} pi  "
        f"(S={S} replications each)"
    )
    print(f"  output_dir: {output_dir}")
    print(f"  resume    : {resume}")

    loaded_payloads: list[dict] = []
    n_skipped: int = 0
    n_computed: int = 0

    for i, (T, estimator, pi) in enumerate(scenarios_spec):
        filename  = _scenario_filename(estimator, T, pi, S)
        scen_path = output_dir / filename
        label     = (
            f"[{i + 1}/{n_total}] T={T:>4d}, "
            f"estimator={estimator:<10s}, pi={pi:.2f}"
        )

        if resume and scen_path.exists():
            payload    = _load_scenario(scen_path)
            cfg_cached = payload.get("config", {})
            current_fp = _theta_fingerprint(theta_star)

            # Validate all parameters that determine the scenario output.
            # S / T / estimator / pi are already implicit in the filename path;
            # the remaining parameters must be verified against the stored config.
            stale_reason: str | None = None
            if cfg_cached.get("theta_fingerprint") is None:
                stale_reason = "no fingerprint in cache"
            elif cfg_cached["theta_fingerprint"] != current_fp:
                stale_reason = (
                    f"theta fingerprint mismatch "
                    f"(cached {cfg_cached['theta_fingerprint']} != current {current_fp})"
                )
            elif cfg_cached.get("base_seed") != base_seed:
                stale_reason = (
                    f"base_seed mismatch "
                    f"(cached {cfg_cached.get('base_seed')} != current {base_seed})"
                )
            elif cfg_cached.get("nu_contam") != nu_contam:
                stale_reason = (
                    f"nu_contam mismatch "
                    f"(cached {cfg_cached.get('nu_contam')} != current {nu_contam})"
                )
            elif cfg_cached.get("kappa") != kappa:
                stale_reason = (
                    f"kappa mismatch "
                    f"(cached {cfg_cached.get('kappa')} != current {kappa})"
                )

            if stale_reason is not None:
                print(f"  {label}  ->  scenario stale: {stale_reason}, recomputing")
            else:
                print(f"  {label}  ->  skipped (cached): {filename}")
                loaded_payloads.append(payload)
                n_skipped += 1
                continue

        print(f"\n  {label}  ->  computing ...")
        t0 = time.perf_counter()
        mc_result = run_monte_carlo(
            theta_star=theta_star,
            S=S, T=T, estimator=estimator, pi=pi,
            nu_contam=nu_contam, kappa=kappa,
            freq_list=freq_list, block_map=block_map,
            ordered_cols=ordered_cols, r=r,
            n_jobs=n_jobs, base_seed=base_seed,
            max_iter=max_iter, tol=tol,
        )
        elapsed = time.perf_counter() - t0
        _save_scenario(scen_path, mc_result)
        print(
            f"  {label}  ->  completed in {elapsed:.1f}s, "
            f"saved {filename}"
        )
        # Round-trip-load to keep `scenarios` uniform with the cached branch.
        loaded_payloads.append(_load_scenario(scen_path))
        n_computed += 1

    print(
        f"\nrun_grid done: {n_computed} computed, "
        f"{n_skipped} skipped (cached), {n_total} total"
    )
    return {
        "scenarios":  loaded_payloads,
        "n_computed": n_computed,
        "n_skipped":  n_skipped,
        "output_dir": str(output_dir),
    }


def load_grid_results(output_dir: str | pathlib.Path):
    r"""
    Assemble all scenario JSON files in ``output_dir`` into a long-format
    pandas DataFrame.

    One row per (scenario, metric).  Columns: ``estimator``, ``T``,
    ``pi``, ``S``, ``n_failed``, ``wall_time_seconds``, ``metric``,
    ``mean``, ``std``, ``median``, ``q05``, ``q95``, ``bias``, ``rmse``.
    ``bias`` / ``rmse`` are populated only for the keys to which they
    apply (estimate-type / error-type respectively); ``NaN`` elsewhere.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    output_dir = pathlib.Path(output_dir)
    rows: list[dict] = []
    for path in sorted(output_dir.glob("mc_*.json")):
        payload = _load_scenario(path)
        cfg = payload["config"]
        for metric_key, stats in payload["aggregates"].items():
            rows.append({
                "estimator":         cfg["estimator"],
                "T":                 int(cfg["T"]),
                "pi":                float(cfg["pi"]),
                "S":                 int(cfg["S"]),
                "n_failed":          int(payload["n_failed"]),
                "wall_time_seconds": float(payload["wall_time_seconds"]),
                "metric":            metric_key,
                "mean":              stats.get("mean", float("nan")),
                "std":               stats.get("std",  float("nan")),
                "median":            stats.get("median", float("nan")),
                "q05":               stats.get("q05", float("nan")),
                "q95":               stats.get("q95", float("nan")),
                "bias":              stats.get("bias", float("nan")),
                "rmse":              stats.get("rmse", float("nan")),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ── ENGINE SMOKE-TEST ────────────────────────────────────────────────────
    # This is NOT an experiment.  It runs a single small replication just to
    # confirm the engine wires together end-to-end (simulate -> PCA init ->
    # fit -> metrics) without crashing.  The three thesis experiments live in
    # run_experiment_a.py / run_experiment_b.py / run_experiment_c.py and drive
    # this engine through run_grid.
    #
    # No caching, no parallel pool, no grid — just one cheap replication at a
    # small T with a fixed seed.
    from config_utils import parse_config_args
    from data_loader  import load_config as _load_config

    try:
        sys.stdout.reconfigure(encoding="utf-8")    # type: ignore[attr-defined]
    except Exception:
        pass

    _args = parse_config_args("monte_carlo.py engine smoke-test (not an experiment).")
    _cfg  = _args.config

    print("=" * 72)
    print(f"  monte_carlo.py  —  ENGINE SMOKE-TEST (not an experiment)  [{_cfg}]")
    print("=" * 72)

    fit_path = _PROJECT_ROOT / "data" / "processed" / _cfg / "fit_dfm_result.npz"
    print(f"Loading calibrated theta_star from: {fit_path}")
    real_fit   = load_dfm_fit(fit_path)
    theta_star = real_fit["theta"]

    cfg_data     = _load_config(_cfg)
    ordered_cols = cfg_data["ORDERED_COLS"]
    block_map    = cfg_data["BLOCK"]
    freq_list    = [cfg_data["FREQ"][c] for c in ordered_cols]

    SEED = 0
    T    = 100
    print(f"\nRunning one student_t replication  (seed={SEED}, T={T}) ...")
    m = run_one_replication(
        seed=SEED, theta_star=theta_star, T=T,
        estimator="student_t", pi=0.0, max_iter=200, verbose_em=False,
        freq_list=freq_list, block_map=block_map, ordered_cols=ordered_cols,
    )

    print("\n[smoke-test OK] engine ran end-to-end.  A few sanity scalars:")
    print(f"  converged                = {m['converged']}")
    print(f"  n_iter                   = {m['n_iter']}")
    print(f"  monotonicity_violations  = {m['n_monotonicity_violations']}")
    print(f"  nu_u_hat                 = {m['nu_u_hat']:.3f}  (truth {m['nu_u_star']:.3f})")
    print(f"  rho_A_hat                = {m['rho_A_hat']:.4f}  (truth {m['rho_A_star']:.4f})")
    print(f"  lambda_relerr (Procr.)   = {m['lambda_relerr_procrustes_blockdiag']:.2%}")
    print("=" * 72)
