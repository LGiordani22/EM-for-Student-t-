"""
src/monte_carlo_recovery.py
===========================

Monte Carlo self-recovery test for the Student-t mixed-frequency DFM.

WHAT THIS SCRIPT DOES
---------------------
1. Load ``theta_star`` from ``data/processed/fit_dfm_result.npz`` — the
   converged EM estimate on the real US macro panel.  This is the DGP we
   pretend to know.
2. Simulate a synthetic panel ``Y_sim`` of length ``T`` via
   :func:`simulate_dfm.simulate_dfm`.  The synthetic panel carries the
   same NaN structure as the real panel (mixed-frequency quarterly mask
   + ragged end-of-sample edge) and the ground-truth latent quantities
   ``F``, ``w_u_true``, ``w_eps_true`` we want the EM to recover.
3. Build a brand-new PCA-based initial parameter vector ``theta_0`` from
   ``Y_sim`` — *without* peeking at ``theta_star``.  This makes the test
   honest: the EM has to re-discover ``theta_star`` from scratch.
4. Re-fit the model on the synthetic panel via :func:`em_main.fit_dfm`
   with ``save_path=None`` (so the cached real-data fit at
   ``fit_dfm_result.npz`` is *not* overwritten).
5. Compare ``theta_hat`` against ``theta_star`` along three orthogonal
   axes:

   a. **Invariant scalars** — ``nu_u``, ``nu_eps``, and the eigenvalues
      of ``A`` (in particular the spectral radius).  Eigenvalues of ``A``
      are invariant to any change of basis on the latent factor space,
      so this is the cleanest possible comparison.
   b. **Normalised-only loadings** — :func:`em_main.fit_dfm` already
      applies sign normalisation (block-wise reference-series sign) and
      Convention 1 (unit total variance) to both fits, so ``Lambda``,
      diag(``Q``) and ``R`` should be directly comparable.  Residual
      sign mismatches per factor are realigned in
      :func:`align_sign_per_factor`.
   c. **Procrustes-aligned loadings** — solve the orthogonal Procrustes
      problem H = argmin ||Lambda_hat @ H - Lambda_star||_F to absorb
      any residual rotation in factor space.  Reported in two variants:
      a *free* 3x3 orthogonal H (which the EM should already eliminate
      via Convention 1 + sign), and a *block-restricted* H constrained
      to be diagonal (preserving the real / financial / other block
      structure).  If the free H is *not* almost diagonal, that
      signals a block-identifiability issue in the EM (factors mixing
      across blocks).  See the thesis Monte Carlo section.

6. Compare latent-state recovery: ``|corr(f_hat[:, j], F_true[:, j])|``
   per factor and ``corr(w_hat, w_true)`` for the two weight vectors.
   These metrics are scale/sign invariant and provide the strongest
   form of validation (level 2 in the Monte Carlo: do we recover the
   *latent path*, not just the parameters).

Each call to :func:`run_recovery` runs the EM once on a synthetic panel
and so takes several minutes.  Two replicas are produced in
``__main__`` at the lengths in ``T_GRID`` (currently ``T=500`` and
``T=2000``); results are persisted per cell to
``output/recovery/final/<spec>/<variant>/mc_recovery_T*.npz`` for later
inspection.

Thesis references
-----------------
``EM_for_student_t.tex``:
  - Monte Carlo section ("Self-recovery test" / "Procrustes alignment"):
    prescribes the rotation alignment used for ``Lambda``.
  - Convention 1 + sign normalisation section: explains why
    *normalised-only* comparison should match *Procrustes-aligned*
    comparison closely for a correctly identified model.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

# ─── Local imports (resolved lazily to allow this file to be run as a script)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from em.em_initialization  import (                                         # noqa: E402
    standardize,
    mm_fill_quarterly,
    gaussian_fill_ragged,
    pca_initialization,
    compute_theta_initial,
)
from em.em_main            import fit_dfm, load_dfm_fit, _theta_fingerprint  # noqa: E402
from simulate_dfm       import simulate_dfm                              # noqa: E402


_BLOCK_ORDER:  list[str]      = ["real", "financial", "other"]
_BLOCK_TO_COL: dict[str, int] = {b: j for j, b in enumerate(_BLOCK_ORDER)}

# Etichette dei fattori usate nelle tabelle. `_BLOCK_ORDER` sopra e' il default
# a tre blocchi (real/financial/other); con le spec nominate i fattori sono in
# numero e nome diversi, e le funzioni di stampa ricevono `factor_names`.
_FACTOR_NAMES: list[str] | None = None


def _r_of(metrics: dict) -> int:
    """Numero di fattori dedotto dai metrics, senza cablare r da nessuna parte."""
    return int(np.asarray(metrics["diagQ_star"]).ravel().size)


def _labels(r: int, factor_names: list[str] | None = None) -> list[str]:
    """
    Nomi degli `r` fattori per le tabelle.

    Ordine di preferenza: l'argomento esplicito, poi il default di modulo, poi
    il vecchio `_BLOCK_ORDER`. Se dopo tutto questo i nomi sono meno di `r`
    (spec con piu' fattori dei tre blocchi di base) si completa con `f{j}`
    invece di sollevare: un'etichetta generica in una tabella e' preferibile a
    un report che non viene prodotto.
    """
    names = list(factor_names or _FACTOR_NAMES or _BLOCK_ORDER)
    if len(names) < r:
        names += [f"f{j}" for j in range(len(names), r)]
    return names[:r]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   INITIALISATION ON A SYNTHETIC PANEL                                    ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _synthetic_monthly_index(T: int, start: str = "1985-01-31") -> pd.DatetimeIndex:
    """
    Build a monthly DatetimeIndex of length ``T`` starting at ``start``,
    aligned with the *real* panel's calendar so that the quarter-end
    months (March / June / September / December) coincide with the
    simulator's default ``quarter_end_offset=2`` (i.e. ``t=2`` is March).
    The PCA initialiser :func:`mm_fill_quarterly` relies on
    ``series.index.month.isin([3, 6, 9, 12])`` to detect quarter-ends, so
    the index *must* be a true ``DatetimeIndex`` with this convention.
    """
    return pd.date_range(start=start, periods=T, freq="ME")


def init_theta_from_synthetic(
    Y_sim: np.ndarray,
    *,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    freq_map: dict[str, str] | None = None,
    nu_init: float = 10.0,
    random_state: int = 42,
    structure=None,
    idio_ar1: bool = False,
) -> tuple[dict, np.ndarray]:
    """
    Replicate :func:`em_initialization.initialize_theta` on a synthetic
    panel, *without* touching the CSV pipeline or the cached real-data
    initial theta on disk.  This is the EM's "blind" starting point for
    the recovery test: a fresh PCA on the synthetic data.

    Pipeline (= identical to the real-data pipeline):
        Y_sim  --standardize-->          (column-wise z-score on observed)
              --mm_fill_quarterly-->     (locally-constant MM fill, GDPC1)
              --gaussian_fill_ragged-->  (N(0,1) fill for the ragged edge)
              --pca_initialization-->    (block-by-block first PC)
              --compute_theta_initial--> theta^(0)

    Returns ``(theta_0, F_pca)`` where ``F_pca`` is the PCA factor matrix
    (used only to compute ``theta_0``; the EM does not need it again).

    Parameters (oltre a quelli evidenti)
    ------------------------------------
    structure : FactorStructure or None
        Struttura di caricamento. Se fornita ha la **precedenza** su
        `block_map`, e va usata per ogni spec non diagonale: un dict
        serie->blocco puo' rappresentare solo un caricamento per serie, quindi
        passare `block_map` sotto `fed_overlap` degraderebbe la struttura a
        diagonale **in silenzio**, e la recovery misurerebbe un modello diverso
        da quello simulato.
    idio_ar1 : bool
        Se True `theta^(0)` include `rho` (e `sigma2`), cioe' l'inizializzazione
        e' quella dell'Asse B. Deve corrispondere al DGP: simulare con `rho` e
        inizializzare senza (o viceversa) non e' una recovery.
    """
    T = Y_sim.shape[0]
    df = pd.DataFrame(
        Y_sim,
        columns=ordered_cols,
        index=_synthetic_monthly_index(T),
    )

    # 1. Standardise: synthetic data is *already* in standardised scale
    #    (because theta_star is Convention-1 normalised), so this is
    #    essentially a no-op modulo sample-mean / sample-std fluctuations,
    #    but we keep it to faithfully mirror the real-data pipeline.
    Y_std, _mean, _std = standardize(df)

    # 2. MM-fill the quarterly columns
    Y_mm = Y_std.copy()
    for col in Y_std.columns:
        if freq_map.get(col) == "quarterly":
            Y_mm[col] = mm_fill_quarterly(Y_std[col])

    # 3. Gaussian-fill any remaining NaN (ragged edge + boundary NaN)
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=random_state)

    # 4. PCA guidata dalla struttura (la mask, non una mappa a blocchi)
    _struct = structure if structure is not None else block_map

    F_pca, _info = pca_initialization(Y_filled, _struct)

    # 5. Closed-form theta^(0)
    _kw = {"nu_init": nu_init}
    if idio_ar1:
        # `freq_list` serve perche' sotto AR(1) le trimestrali portano 5 slot
        # idiosincratici nello stato invece di 1.
        _kw["idio_ar1"] = True
        _kw["freq_list"] = [freq_map[c] for c in ordered_cols]
    theta_0 = compute_theta_initial(Y_filled, F_pca, _struct, **_kw)

    return theta_0, F_pca


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   SIGN AND PROCRUSTES ALIGNMENT                                          ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def align_sign_per_factor(
    Lambda_hat: np.ndarray,
    Lambda_star: np.ndarray,
) -> np.ndarray:
    r"""
    Return a vector ``d \in \{-1, +1\}^r`` such that flipping factor
    column ``j`` by ``d[j]`` (i.e. Lambda_hat[:, j] *= d[j]) maximises
    the *positive* alignment of ``Lambda_hat`` and ``Lambda_star``
    column by column.

    Both fits are already sign-normalised against their own reference
    series by :func:`em_main.normalize_signs`, so ``d`` is *expected*
    to be ``(+1, +1, +1)`` whenever the reference series have the same
    sign in the smoothed factors of the two fits.  A non-trivial ``d``
    flags a sign discrepancy and is corrected before computing
    relative errors.

    Rule: pick the sign of ``Lambda_hat[:, j].T @ Lambda_star[:, j]``
    (positive inner product = same orientation).
    """
    r = Lambda_hat.shape[1]
    d = np.ones(r, dtype=int)
    for j in range(r):
        s = float(Lambda_hat[:, j] @ Lambda_star[:, j])
        if s < 0:
            d[j] = -1
    return d


def procrustes_orthogonal(
    Lambda_hat: np.ndarray,
    Lambda_star: np.ndarray,
) -> np.ndarray:
    r"""
    Solve the *orthogonal* Procrustes problem

    .. math::

        H^\star \;=\; \arg\min_{H \in \mathcal{O}(r)}
                       \| \mathbf{\Lambda}_{\text{hat}} H - \mathbf{\Lambda}_\star \|_F.

    Classical closed-form solution: SVD of
    :math:`M = \mathbf{\Lambda}_{\text{hat}}^\top \mathbf{\Lambda}_\star = U \Sigma V^\top`,
    then :math:`H^\star = U V^\top`.

    Applied to the factor space, ``H`` corresponds to the change of
    variables :math:`g_t = H^\top f_t`, under which the observation
    equation becomes :math:`y_t = (\mathbf{\Lambda} H) g_t + \varepsilon_t`
    and the dynamics become :math:`g_t = (H^\top A H) g_{t-1} +
    (H^\top u_t)` — see :func:`apply_factor_rotation`.
    """
    M = Lambda_hat.T @ Lambda_star
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def procrustes_block_diagonal(
    Lambda_hat: np.ndarray,
    Lambda_star: np.ndarray,
    *,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    structure=None,
) -> np.ndarray:
    r"""
    Block-restricted Procrustes: find the *diagonal* matrix
    :math:`H = \mathrm{diag}(h_R, h_F, h_X)` that minimises
    :math:`\| \mathbf{\Lambda}_{\text{hat}} H - \mathbf{\Lambda}_\star \|_F`.

    Because the model has *one factor per block* (``r_b = 1``), a
    block-diagonal :math:`H` reduces to a diagonal matrix with one
    scalar :math:`h_b` per block.  The closed-form solution per block:

    .. math::

        h_b \;=\;
        \frac{
            \mathbf{\Lambda}_{\text{hat},\,b,b}^\top \,
            \mathbf{\Lambda}_{\star,\,b,b}
        }{
            \| \mathbf{\Lambda}_{\text{hat},\,b,b} \|^2
        },

    where :math:`\mathbf{\Lambda}_{*, b, b}` is the column of
    :math:`\mathbf{\Lambda}_*` restricted to the rows of the block
    :math:`b` (the only rows with non-zero entries in column :math:`b`).

    Note: if Convention 1 is correctly applied to both fits, the
    *magnitude* of each :math:`h_b` should be close to 1, and its sign
    should match :func:`align_sign_per_factor`.  Block-restricted
    Procrustes therefore *adds nothing* over sign normalisation when
    the block structure is well identified — which is precisely what
    the recovery test should verify.
    """
    r       = Lambda_hat.shape[1]
    H_diag  = np.zeros(r)
    _st = structure if structure is not None else None
    for j in range(r):
        if _st is not None:
            # Mask-driven: i membri del fattore j sono quelli che ci caricano
            # davvero, anche quando una serie carica su piu' fattori.
            rows = [int(i) for i in _st.members(j)]
        else:
            b = _labels(r)[j]
            rows = [i for i, c in enumerate(ordered_cols) if block_map.get(c) == b]
        lh   = Lambda_hat[rows, j]
        ls   = Lambda_star[rows, j]
        denom = float(lh @ lh)
        H_diag[j] = float(lh @ ls) / denom if denom > 0 else 1.0
    return np.diag(H_diag)


def apply_factor_rotation(
    theta: dict,
    f_smooth: np.ndarray,
    H: np.ndarray,
) -> tuple[dict, np.ndarray]:
    r"""
    Apply a change of basis :math:`g_t = H^{-1} f_t` (= :math:`H^\top f_t`
    for orthogonal :math:`H`) to a fitted theta and its smoothed
    factors.  Returns a *new* theta dict (does not mutate the input)
    and the rotated smoothed factor matrix.

    Algebra (for general invertible :math:`H`):

    .. code-block::

        y = Lambda f + eps          ==>  y = (Lambda H) (H^{-1} f) + eps
        f_t = A f_{t-1} + u_t       ==>  g_t = (H^{-1} A H) g_{t-1} + (H^{-1} u_t)
        Q   = Cov(u_t)              ==>  Q'  = H^{-1} Q (H^{-1})^T

    For *orthogonal* H this simplifies to the usual congruence /
    similarity formulas with :math:`H^{-1} = H^\top`.

    Only ``Lambda``, ``A``, ``Q`` and the contemporaneous-block of
    ``f_smooth[:, :r]`` are returned — the higher-lag blocks of the
    augmented smoother (the Mariano-Murasawa state at lags 1..4) are
    *not* rotated here, because (a) they are not used in the parameter
    comparison and (b) rotating them properly would require rebuilding
    the full augmented state.  The contemporaneous block is what the
    factor-correlation diagnostic compares against ``F_true``.
    """
    H_inv = np.linalg.inv(H)
    Lambda_new = np.asarray(theta["Lambda"]) @ H
    A_new      = H_inv @ np.asarray(theta["A"]) @ H
    Q_new      = H_inv @ np.asarray(theta["Q"]) @ H_inv.T

    new_theta: dict = {key: np.asarray(val).copy() for key, val in theta.items()}
    new_theta["Lambda"] = Lambda_new
    new_theta["A"]      = A_new
    new_theta["Q"]      = Q_new

    r = Lambda_new.shape[1]
    f_new = f_smooth.copy()
    # f_new[t, :r] = H^{-1} @ f_smooth[t, :r]   <=>   f_new[:, :r] = f_smooth[:, :r] @ H^{-T}
    f_new[:, :r] = f_smooth[:, :r] @ H_inv.T

    return new_theta, f_new


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   RECOVERY METRICS                                                       ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _sorted_eigvals(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of A sorted by descending modulus (complex-valued)."""
    w = np.linalg.eigvals(A)
    return w[np.argsort(-np.abs(w))]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Correlazione di Pearson (con segno), calcolata sugli array APPIATTITI.

    L'appiattimento non e' una comodita': coi pesi PER SERIE `w^eps` e' una
    matrice (T, M), non un vettore (T,), e la versione precedente faceva
    `xz @ yz` — un prodotto matriciale che su due (T, M) fallisce con un errore
    di dimensione. Appiattire e' la generalizzazione naturale: la correlazione
    e' fra le coppie (periodo, serie) di peso stimato e peso vero, che e'
    esattamente la domanda a cui la metrica deve rispondere. Sui pesi condivisi
    (T,) l'appiattimento e' l'identita', quindi i risultati sui pesi condivisi
    non cambiano.

    Richiede che le due forme coincidano: correlare un peso per-serie con uno
    condiviso sarebbe un confronto fra oggetti diversi, e va segnalato.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"forme incompatibili: {x.shape} vs {y.shape}. Stimatore e DGP "
            f"devono usare lo stesso schema di pesi (per-serie o condiviso).")
    x = x.ravel()
    y = y.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    xz = x[ok] - x[ok].mean()
    yz = y[ok] - y[ok].mean()
    sx = float(np.sqrt((xz ** 2).sum()))
    sy = float(np.sqrt((yz ** 2).sum()))
    if sx == 0 or sy == 0:
        return float("nan")
    return float((xz @ yz) / (sx * sy))


def _abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    """|Pearson correlation|. Vedi `_corr` per il trattamento delle forme."""
    c = _corr(x, y)
    return float("nan") if np.isnan(c) else float(abs(c))


def _signed_corr(x: np.ndarray, y: np.ndarray) -> float:
    return _corr(x, y)


def compute_recovery_metrics(
    *,
    theta_star: dict,
    theta_hat: dict,
    f_smooth_hat: np.ndarray,
    F_true: np.ndarray,
    w_u_hat: np.ndarray,
    w_u_true: np.ndarray,
    w_eps_hat: np.ndarray,
    w_eps_true: np.ndarray,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    structure=None,
) -> dict[str, Any]:
    r"""
    Compute every recovery diagnostic described in the docstring of
    this module.  Returns a plain dict (no pandas) so that the result
    can be serialised via ``np.savez``.

    Inputs
    ------
    theta_star : converged EM estimate on the real data (the DGP).
    theta_hat  : converged EM estimate on the synthetic panel.
                 Both are sign-normalised + Convention-1 by
                 :func:`em_main.fit_dfm`, so they are *already* in
                 the canonical frame.
    f_smooth_hat : (T, 5r) augmented smoothed state from the EM
                   re-fit.  Only ``[:, :r]`` is used.
    F_true      : (T, r) ground-truth monthly factors from
                  :func:`simulate_dfm.simulate_factors`.  *Not*
                  Convention-1 rescaled — the comparison uses
                  scale-invariant ``|corr|``.
    w_u_hat / w_u_true     : (T,) posterior mean vs ground truth
                             for the factor-innovation weights.
    w_eps_hat / w_eps_true : same for the idiosyncratic weights.

    Outputs
    -------
    dict with the following keys (Python-native types where possible
    for clean printing):

    - ``"nu_u_star"``, ``"nu_u_hat"``, ``"nu_u_relerr"``
    - ``"nu_eps_star"``, ``"nu_eps_hat"``, ``"nu_eps_relerr"``
    - ``"eig_A_star"``, ``"eig_A_hat"``  (sorted by |w|, complex)
    - ``"rho_A_star"``, ``"rho_A_hat"``, ``"rho_A_relerr"``
    - ``"sign_flips"``  (r,)  -- d ∈ {-1, +1}^r used in normalised-only
    - ``"Lambda_relerr_per_series"``  (M,)  -- after sign alignment
    - ``"Lambda_relerr_norm"``  scalar
    - ``"diagQ_star"``, ``"diagQ_hat"``, ``"diagQ_relerr"``  (r,)
    - ``"R_star"``, ``"R_hat"``, ``"R_relerr"``  (M,)
    - ``"H_free"``  (r, r)  -- free orthogonal Procrustes rotation
    - ``"H_block"`` (r, r)  -- block-diagonal Procrustes rotation
    - ``"Lambda_relerr_norm_proc_free"``  scalar
    - ``"Lambda_relerr_norm_proc_block"`` scalar
    - ``"factor_abscorr"`` (r,)
    - ``"w_u_corr"``, ``"w_eps_corr"`` scalars
    """
    Lambda_star = np.asarray(theta_star["Lambda"])
    Lambda_hat  = np.asarray(theta_hat["Lambda"])
    A_star      = np.asarray(theta_star["A"])
    A_hat       = np.asarray(theta_hat["A"])
    Q_star      = np.asarray(theta_star["Q"])
    Q_hat       = np.asarray(theta_hat["Q"])
    R_star      = np.asarray(theta_star["R"])
    R_hat       = np.asarray(theta_hat["R"])
    r           = Lambda_star.shape[1]

    # ── Invariants: degrees of freedom and eigenvalues of A ────────────────
    nu_u_star   = float(theta_star["nu_u"])
    nu_u_hat    = float(theta_hat["nu_u"])
    nu_eps_star = float(theta_star["nu_eps"])
    nu_eps_hat  = float(theta_hat["nu_eps"])
    eig_A_star  = _sorted_eigvals(A_star)
    eig_A_hat   = _sorted_eigvals(A_hat)
    rho_A_star  = float(np.max(np.abs(eig_A_star)))
    rho_A_hat   = float(np.max(np.abs(eig_A_hat)))

    # ── Sign alignment per factor and normalised-only loadings ─────────────
    d_sign         = align_sign_per_factor(Lambda_hat, Lambda_star)
    D              = np.diag(d_sign.astype(float))
    # Apply the sign flip to the FULL theta_hat (factors and dynamics
    # transform consistently under a diagonal ±1 H).
    theta_hat_sgn, f_smooth_hat_sgn = apply_factor_rotation(
        theta_hat, f_smooth_hat, D
    )
    Lambda_hat_sgn = np.asarray(theta_hat_sgn["Lambda"])
    Q_hat_sgn      = np.asarray(theta_hat_sgn["Q"])

    Lambda_diff   = Lambda_hat_sgn - Lambda_star
    # Per-series relative error: only series with non-zero loading have
    # a meaningful relative error (block-diagonality => most entries are 0).
    M = Lambda_star.shape[0]
    Lambda_relerr_per_series = np.zeros(M)
    for i in range(M):
        norm_star_i = np.linalg.norm(Lambda_star[i])
        if norm_star_i > 0:
            Lambda_relerr_per_series[i] = (
                np.linalg.norm(Lambda_diff[i]) / norm_star_i
            )
    Lambda_relerr_norm = float(
        np.linalg.norm(Lambda_diff) / np.linalg.norm(Lambda_star)
    )

    # ── Procrustes alignments ──────────────────────────────────────────────
    # Free orthogonal H (full rotation in factor space — should be close to
    # diag(±1) if the model is well identified).
    H_free  = procrustes_orthogonal(Lambda_hat, Lambda_star)
    # Block-restricted H (= diag(h_R, h_F, h_X) here since r_b = 1).  Both
    # versions are computed against the *raw* (pre-sign-alignment) Lambda_hat
    # because Procrustes naturally absorbs any sign ambiguity.
    H_block = procrustes_block_diagonal(
        Lambda_hat, Lambda_star,
        ordered_cols=ordered_cols, block_map=block_map, structure=structure,
    )
    Lambda_proc_free  = Lambda_hat @ H_free
    Lambda_proc_block = Lambda_hat @ H_block

    Lambda_relerr_norm_proc_free  = float(
        np.linalg.norm(Lambda_proc_free  - Lambda_star) / np.linalg.norm(Lambda_star)
    )
    Lambda_relerr_norm_proc_block = float(
        np.linalg.norm(Lambda_proc_block - Lambda_star) / np.linalg.norm(Lambda_star)
    )

    # ── Q and R per-component recoveries ───────────────────────────────────
    diagQ_star    = np.diag(Q_star)
    diagQ_hat     = np.diag(Q_hat_sgn)            # diagonal is sign-invariant
    diagQ_relerr  = np.abs(diagQ_hat - diagQ_star) / np.maximum(np.abs(diagQ_star), 1e-12)
    R_relerr      = np.abs(R_hat - R_star)        / np.maximum(np.abs(R_star),     1e-12)

    # ── Factor-path recovery: |corr(f_hat, F_true)| per factor ─────────────
    # f_smooth_hat is in the EM's canonical frame (Conv. 1 + sign).  F_true
    # is the raw simulator output — NOT Convention-1 rescaled.  |corr| is
    # invariant to any scale and sign, so this is the right metric.
    factor_abscorr = np.array([
        _abs_corr(f_smooth_hat[:, j], F_true[:, j]) for j in range(r)
    ])

    # ── Weight-path recovery ───────────────────────────────────────────────
    w_u_corr   = _signed_corr(w_u_hat,   w_u_true)
    w_eps_corr = _signed_corr(w_eps_hat, w_eps_true)

    return {
        "nu_u_star":      nu_u_star,
        "nu_u_hat":       nu_u_hat,
        "nu_u_relerr":    abs(nu_u_hat - nu_u_star) / max(abs(nu_u_star), 1e-12),
        "nu_eps_star":    nu_eps_star,
        "nu_eps_hat":     nu_eps_hat,
        "nu_eps_relerr":  abs(nu_eps_hat - nu_eps_star) / max(abs(nu_eps_star), 1e-12),
        "eig_A_star":     eig_A_star,
        "eig_A_hat":      eig_A_hat,
        "rho_A_star":     rho_A_star,
        "rho_A_hat":      rho_A_hat,
        "rho_A_relerr":   abs(rho_A_hat - rho_A_star) / max(abs(rho_A_star), 1e-12),
        "sign_flips":     d_sign,
        "Lambda_relerr_per_series":      Lambda_relerr_per_series,
        "Lambda_relerr_norm":            Lambda_relerr_norm,
        "Lambda_relerr_norm_proc_free":  Lambda_relerr_norm_proc_free,
        "Lambda_relerr_norm_proc_block": Lambda_relerr_norm_proc_block,
        "H_free":         H_free,
        "H_block":        H_block,
        "diagQ_star":     diagQ_star,
        "diagQ_hat":      diagQ_hat,
        "diagQ_relerr":   diagQ_relerr,
        "R_star":         R_star,
        "R_hat":          R_hat,
        "R_relerr":       R_relerr,
        "factor_abscorr": factor_abscorr,
        "w_u_corr":       w_u_corr,
        "w_eps_corr":     w_eps_corr,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   PRETTY PRINTING                                                        ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _fmt_complex(z: complex, precision: int = 4) -> str:
    if abs(z.imag) < 1e-10:
        return f"{z.real:+.{precision}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:+.{precision}f}{sign}{abs(z.imag):.{precision}f}j"


def print_recovery_table(
    metrics: dict[str, Any],
    *,
    label: str,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    series_highlight: tuple[str, ...] = ("INDPRO", "PAYEMS", "S&P 500",
                                         "BAAFFM", "CPIAUCSL", "GDPC1",
                                         "UMCSENTx", "NFCI"),
    factor_names: list[str] | None = None,
) -> None:
    """Print a human-readable recovery table for a single replica."""
    bar = "=" * 76
    print("\n" + bar)
    print(f"  RECOVERY TABLE — {label}")
    print(bar)

    # ── Scalars ──────────────────────────────────────────────────────────────
    print(f"\n  {'parameter':<26s}  {'theta_star':>14s}  {'theta_hat':>14s}  {'rel.err':>10s}")
    print("  " + "-" * 70)
    print(f"  {'nu_u':<26s}  {metrics['nu_u_star']:>14.4f}  "
          f"{metrics['nu_u_hat']:>14.4f}  {metrics['nu_u_relerr']:>10.2%}")
    print(f"  {'nu_eps':<26s}  {metrics['nu_eps_star']:>14.4f}  "
          f"{metrics['nu_eps_hat']:>14.4f}  {metrics['nu_eps_relerr']:>10.2%}")
    print(f"  {'spectral_radius(A)':<26s}  {metrics['rho_A_star']:>14.6f}  "
          f"{metrics['rho_A_hat']:>14.6f}  {metrics['rho_A_relerr']:>10.2%}")

    # ── Eigenvalues of A ─────────────────────────────────────────────────────
    print(f"\n  Eigenvalues of A  (sorted by descending |w|):")
    print(f"    {'#':>2s}  {'theta_star':>22s}    {'theta_hat':>22s}")
    print("    " + "-" * 52)
    for k, (ws, wh) in enumerate(zip(metrics["eig_A_star"], metrics["eig_A_hat"])):
        print(f"    {k:>2d}  {_fmt_complex(ws, 5):>22s}    {_fmt_complex(wh, 5):>22s}")

    # ── Diagonal Q recovery ──────────────────────────────────────────────────
    print(f"\n  diag(Q):")
    print(f"    {'block':<12s}  {'theta_star':>12s}  {'theta_hat':>12s}  {'rel.err':>10s}")
    print("    " + "-" * 50)
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        print(f"    {b:<12s}  {metrics['diagQ_star'][j]:>12.4f}  "
              f"{metrics['diagQ_hat'][j]:>12.4f}  {metrics['diagQ_relerr'][j]:>10.2%}")

    # ── Sign alignment ───────────────────────────────────────────────────────
    print(f"\n  Sign alignment per factor (theta_hat vs theta_star, expected +1):")
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        print(f"    factor {j} ({b:<10s}):  d = {int(metrics['sign_flips'][j]):+d}")

    # ── Lambda recovery — selected series after sign alignment ───────────────
    print(f"\n  Lambda recovery — selected reference series  "
          f"(sign-aligned per factor):")
    print(f"    {'series':<14s}  {'block':<10s}  {'j':>2s}  "
          f"{'Lambda_star':>12s}  {'Lambda_hat':>12s}  {'rel.err':>10s}")
    print("    " + "-" * 64)
    Lambda_star_arr = np.asarray(metrics["_extras"]["Lambda_star"])
    Lambda_hat_sgn  = np.asarray(metrics["_extras"]["Lambda_hat_sgn"])
    for name in series_highlight:
        if name not in ordered_cols:
            continue
        i = ordered_cols.index(name)
        b = block_map.get(name, "?")
        j = _BLOCK_TO_COL.get(b, -1)
        lam_s = float(Lambda_star_arr[i, j]) if j >= 0 else float("nan")
        lam_h = float(Lambda_hat_sgn[i, j])  if j >= 0 else float("nan")
        relerr = abs(lam_h - lam_s) / max(abs(lam_s), 1e-12)
        print(f"    {name:<14s}  {b:<10s}  {j:>2d}  "
              f"{lam_s:>+12.4f}  {lam_h:>+12.4f}  {relerr:>10.2%}")

    # ── Procrustes summary ───────────────────────────────────────────────────
    print(f"\n  Procrustes alignment (||Lambda_hat H - Lambda_star||_F / ||Lambda_star||_F):")
    print(f"    {'normalised-only (sign+Conv1)':<32s}  "
          f"{metrics['Lambda_relerr_norm']:>10.3%}")
    print(f"    {'Procrustes (free 3x3 ortho)':<32s}  "
          f"{metrics['Lambda_relerr_norm_proc_free']:>10.3%}")
    print(f"    {'Procrustes (block-diagonal)':<32s}  "
          f"{metrics['Lambda_relerr_norm_proc_block']:>10.3%}")

    print(f"\n  Free-rotation H (should be close to diag(±1) if block-identified):")
    H_free = np.asarray(metrics["H_free"])
    for row in H_free:
        print("    " + "  ".join(f"{v:>+8.4f}" for v in row))
    off_diag_max = float(np.max(np.abs(H_free - np.diag(np.diag(H_free)))))
    diag_dev     = float(np.max(np.abs(np.abs(np.diag(H_free)) - 1.0)))
    print(f"    max |off-diagonal| = {off_diag_max:.4f}   "
          f"max ||diag| - 1|   = {diag_dev:.4f}")

    print(f"\n  Block-diagonal H = diag(h_R, h_F, h_X):")
    H_block = np.asarray(metrics["H_block"])
    h_diag = np.diag(H_block)
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        print(f"    h_{b:<10s} = {h_diag[j]:>+8.4f}")

    # ── Factor and weight correlations ───────────────────────────────────────
    print(f"\n  Latent-path recovery — correlations:")
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        print(f"    |corr(f_hat_{j}, F_true_{j})|  "
              f"[{b:<10s}]  = {metrics['factor_abscorr'][j]:.4f}")
    print(f"    corr(w_u_hat,   w_u_true)               "
          f"= {metrics['w_u_corr']:+.4f}")
    print(f"    corr(w_eps_hat, w_eps_true)             "
          f"= {metrics['w_eps_corr']:+.4f}")

    # ── R recovery (aggregate) ───────────────────────────────────────────────
    R_relerr = np.asarray(metrics["R_relerr"])
    print(f"\n  R (per-series) — median rel.err = {np.median(R_relerr):.2%}, "
          f"max rel.err = {np.max(R_relerr):.2%}")
    print(bar)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   END-TO-END RECOVERY DRIVER                                             ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_recovery(
    theta_star: dict,
    *,
    T: int,
    seed: int,
    verbose_em: bool = False,
    max_iter: int = 500,
    # `tol_outer` e' ESPOSTA perche' il criterio di arresto e' RELATIVO
    # (|dL|/|L| < tol_outer) mentre |L| cresce con T: a parita' di `tol_outer`
    # la soglia ASSOLUTA sul guadagno per iterazione e' proporzionale a T, cioe'
    # l'EM si ferma prima su pannelli lunghi. Misurato su queste celle: a T=500
    # l'ultimo passo vale ~0.14, a T=2000 ~0.55, con lo STESSO numero di
    # iterazioni. Chi confronta due lunghezze campionarie sta quindi
    # confrontando anche due gradi di convergenza diversi, e i parametri lenti
    # (`nu_u`, `nu_eps`) restano piu' vicini al loro valore iniziale sul
    # pannello lungo — che con `nu_init = 10` significa distorti verso l'ALTO.
    # Per un confronto a convergenza equivalente si passa una `tol_outer`
    # riscalata di |L(T=500)|/|L(T)| ~ 500/T.
    tol_outer: float = 1e-5,
    save_path: str | pathlib.Path | None = None,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    freq_map: dict[str, str] | None = None,
    structure=None,
    gaussian: bool = False,
    per_series_weights: bool = False,
) -> dict[str, Any]:
    """
    One full recovery replica: simulate, init from scratch, re-fit, score.

    **La variante non si passa: si deduce da `theta_star`.**  Se `theta_star`
    porta `rho`, il DGP e' idio-AR(1) e lo stimatore viene inizializzato di
    conseguenza; altrimenti no.  E' l'unico modo di garantire che simulatore e
    stimatore parlino dello stesso modello — un disallineamento qui non
    darebbe un errore, darebbe una recovery che "fallisce" per un motivo che
    non c'entra con la stima.  `gaussian` e `per_series_weights`, che non sono
    leggibili da `theta_star`, restano espliciti.

    `structure` (FactorStructure) ha la precedenza su `block_map` e va sempre
    passata per le spec non diagonali: vedi `init_theta_from_synthetic`.

    Returns the metrics dict together with the simulated panel and the
    EM result (for further inspection).  ``save_path`` writes a flat
    ``.npz`` archive with every numerical quantity (separate from the
    cached real-data ``fit_dfm_result.npz`` — never overwritten).
    """
    print("\n" + "#" * 76)
    print(f"#  RECOVERY REPLICA — T = {T}, seed = {seed}")
    print("#" * 76)

    # ── 1. Simulate the synthetic panel from theta_star ──────────────────────
    # La variante del DGP e' scritta dentro theta_star: `rho` presente =>
    # idiosincratico AR(1) (con l'aggregazione MM anche dell'idio per le
    # trimestrali). Non c'e' un flag da tenere in sincronia a mano.
    _idio_ar1 = "rho" in set(theta_star.keys())
    _struct = structure if structure is not None else block_map
    print("\n[1/4] Simulating synthetic panel from theta_star ...")
    print(f"  DGP: idio_ar1={_idio_ar1}, gaussian={gaussian}, "
          f"per_series_weights={per_series_weights}")
    sim = simulate_dfm(
        theta=theta_star, T=T,
        freq_list=[freq_map[c] for c in ordered_cols],
        block_map=block_map,
        ordered_cols=ordered_cols,
        r=int(np.asarray(theta_star["A"]).shape[0]),
        seed=seed,
        per_series_weights=per_series_weights,
    )
    Y_sim, F_true       = sim["Y"], sim["F"]
    w_u_true, w_eps_true = sim["w_u_true"], sim["w_eps_true"]
    print(f"  Y_sim.shape = {Y_sim.shape};  finite cells = "
          f"{int(np.isfinite(Y_sim).sum())} / {Y_sim.size} "
          f"({100*np.isfinite(Y_sim).mean():.1f}%)")

    # ── 2. PCA-based init on the synthetic panel (honest start) ──────────────
    print("\n[2/4] PCA initialisation on the synthetic panel ...")
    theta_0, _F_pca = init_theta_from_synthetic(
        Y_sim,
        ordered_cols=ordered_cols,
        block_map=block_map,
        freq_map=freq_map,
        structure=structure,
        idio_ar1=_idio_ar1,
    )
    if per_series_weights:
        theta_0["per_series_weights"] = True
    rho_init = float(max(abs(np.linalg.eigvals(theta_0["A"]))))
    print(f"  theta_0: rho(A_init) = {rho_init:.4f}, "
          f"nu_u_init = {theta_0['nu_u']:.2f}, "
          f"nu_eps_init = {theta_0['nu_eps']:.2f}")

    # ── 3. Re-fit the EM from scratch (save_path=None — do NOT clobber cache)
    print("\n[3/4] Re-fitting the EM on the synthetic panel "
          "(save_path=None) ...")
    result = fit_dfm(
        Y=Y_sim,
        theta_init=theta_0,
        freq_list=[freq_map[c] for c in ordered_cols],
        block_map=_struct,
        ordered_cols=ordered_cols,
        verbose=verbose_em,
        save_path=None,
        max_iter=max_iter,
        tol_outer=tol_outer,
        use_full_elbo=True,
        gaussian=gaussian,
    )
    theta_hat    = result["theta"]
    f_smooth_hat = np.asarray(result["f_smooth"])
    estep        = result["e_step_output"]
    w_u_hat      = np.asarray(estep["w_u"])
    w_eps_hat    = np.asarray(estep["w_eps"])
    _n_viol = len(result["monotonicity_violations"])
    if _n_viol == 0:
        _viol_msg = "0 (clean)"
    elif _n_viol <= 2:
        _viol_msg = (f"{_n_viol} (transient, expected at small-nu transit"
                     " — not a bug)")
    else:
        _viol_msg = f"{_n_viol}  *** WARNING: unexpectedly many violations ***"
    print(f"  EM converged: {result['converged']}, "
          f"n_iter = {result['n_iter']}, "
          f"monotonicity violations = {_viol_msg}")
    print(f"  loglik trajectory: {result['loglik_history'][0]:.2f}  ->  "
          f"{result['loglik_history'][-1]:.2f}")

    # NOTE — Monotonicity violations: synthetic vs real data (empirical observation)
    # On the synthetic panels tested here (T=500 and T=2000, seed 42) the EM
    # converged with ZERO monotonicity violations, whereas fitting on the real
    # dataset produced 1 transient violation.  The reason is NOT that synthetic
    # data are immune by construction: any dataset *could* trigger a violation.
    # What differs is the convergence trajectory.  In these runs the PCA
    # initialisation places the starting point such that the path through
    # parameter space crosses the narrow unstable band (nu ~ 5) more smoothly
    # — or avoids the critical point entirely — compared with the real-data fit.
    # The synthetic recovery also converges faster than the real fit, giving
    # the EM less opportunity to linger near the instability.  The phenomenon
    # is confined to a narrow nu range and depends
    # sensitively on both the starting point and the realised data; a different
    # seed or sample size could easily produce a violation.

    # ── 4. Compute recovery metrics ──────────────────────────────────────────
    print("\n[4/4] Computing recovery metrics ...")
    metrics = compute_recovery_metrics(
        theta_star=theta_star,
        theta_hat=theta_hat,
        f_smooth_hat=f_smooth_hat,
        F_true=F_true,
        w_u_hat=w_u_hat,
        w_u_true=w_u_true,
        w_eps_hat=w_eps_hat,
        w_eps_true=w_eps_true,
        ordered_cols=ordered_cols,
        block_map=block_map,
        structure=structure,
    )

    # Attach pre-aligned and aligned Lambda for the pretty printer.
    d_sign = metrics["sign_flips"]
    D      = np.diag(d_sign.astype(float))
    Lambda_hat_sgn = np.asarray(theta_hat["Lambda"]) @ D
    metrics["_extras"] = {
        "Lambda_star":    np.asarray(theta_star["Lambda"]),
        "Lambda_hat":     np.asarray(theta_hat["Lambda"]),
        "Lambda_hat_sgn": Lambda_hat_sgn,
        "T":              T,
        "seed":           seed,
        "n_iter":         int(result["n_iter"]),
        "converged":      bool(result["converged"]),
    }

    # ── Optional persistence ─────────────────────────────────────────────────
    if save_path is not None:
        save_p = pathlib.Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, np.ndarray] = {}
        for k, v in metrics.items():
            if k == "_extras":
                for ek, ev in v.items():
                    flat[f"extras__{ek}"] = np.asarray(ev)
            elif isinstance(v, np.ndarray):
                flat[k] = v
            else:
                flat[k] = np.asarray(v)
        # Ground truth and final theta for downstream replays.
        for k, v in theta_hat.items():
            flat[f"theta_hat__{k}"] = np.asarray(v)
        flat["F_true"]     = F_true
        flat["w_u_true"]   = w_u_true
        flat["w_eps_true"] = w_eps_true
        flat["w_u_hat"]    = w_u_hat
        flat["w_eps_hat"]  = w_eps_hat
        flat["loglik_history"] = np.asarray(result["loglik_history"], dtype=float)
        # Fingerprint of the DGP theta_star this recovery was generated from,
        # so the reuse check can verify the cache is not stale (anti zombie-result).
        flat["theta_fingerprint"] = np.asarray(_theta_fingerprint(theta_star))
        np.savez(save_p, **flat)
        print(f"\n  Saved recovery archive to: {save_p}")

    return {"metrics": metrics, "sim": sim, "result": result}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   OUTLIER RANK-OVERLAP DIAGNOSTIC (precision@k)                          ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_outlier_rank_overlap(
    w_hat: np.ndarray,
    w_true: np.ndarray,
    k_frac: float,
    *,
    exclude_t0: bool = True,
) -> dict[str, Any]:
    r"""
    Outlier-focused recovery diagnostic for the Student-t mixing weights.

    Under the scale-mixture representation,
    :math:`u_t \mid w_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q} / w_t)`
    with :math:`\mathbb{E}[w_t] = 1`.  A *low* :math:`w_t` corresponds to
    a *heavy-tailed outlier period*: the conditional covariance
    :math:`\mathbf{Q} / w_t` is inflated, the realised innovation is
    large, and the EM down-weights that period in the M-step.  Robust
    inference depends on the model *identifying these outlier periods*
    far more than on getting the exact weight value at every normal
    period right.  Linear correlation pools normal and outlier periods
    with equal weight and so understates this property.

    The diagnostic computed here is the *precision@k* of the
    bottom-:math:`k` set: take the :math:`k = \lceil k_{\text{frac}} \cdot T
    \rceil` periods with the lowest *true* weight (the true outliers of
    the DGP) and the :math:`k` periods with the lowest *estimated*
    weight (the EM-identified outliers), and report the fraction of the
    true outliers that fall in the estimated outlier set:

    .. math::

        \mathrm{overlap}@k \;=\;
            \frac{
                \left|
                  \{t : w^{\text{true}}_t \in \text{bottom-}k\}
                  \;\cap\;
                  \{t : w^{\text{hat}}_t  \in \text{bottom-}k\}
                \right|
            }{k}.

    A purely random guess would produce
    :math:`\mathrm{overlap} \approx k / T`, so the *lift* over chance
    is :math:`\mathrm{overlap} / (k/T)`.  As a bonus we also report the
    Spearman rank correlation on the subset of *true outlier* periods,
    which measures whether the EM not only identifies the outliers but
    also ranks them by severity correctly.

    Parameters
    ----------
    w_hat, w_true : np.ndarray of shape (T,)
        Estimated and ground-truth mixing weights.  Must be aligned
        and strictly positive.
    k_frac : float
        Fraction of the sample to treat as the outlier set
        (e.g. ``0.05`` for the bottom 5%).  Converted to an integer
        :math:`k` via rounding to the nearest period.
    exclude_t0 : bool, default True
        Drop ``t = 0`` before ranking.  For ``w_u``, the EM's posterior
        at ``t = 0`` is dominated by the Sigma_0 prior and is not a
        proper data-driven weight; for ``w_eps`` the effect is negligible
        but we exclude it uniformly for consistency.

    Returns
    -------
    dict with keys:
      - ``k`` : int — number of periods in the top-:math:`k` set.
      - ``T_eff`` : int — effective sample length after ``exclude_t0``.
      - ``overlap`` : float — precision@k, in :math:`[0, 1]`.
      - ``expected_random`` : float — :math:`k / T_{\text{eff}}`,
        the overlap a random ranking would produce.
      - ``lift`` : float — ``overlap / expected_random`` (>> 1 means
        the model identifies outliers far better than chance).
      - ``spearman_on_outliers`` : float — Spearman :math:`\rho` of
        :math:`w^{\text{hat}}` against :math:`w^{\text{true}}`
        restricted to the true outlier set.  Measures *severity
        ordering* among the outliers.
    """
    from scipy.stats import spearmanr                                # noqa: PLC0415

    w_hat_a  = np.asarray(w_hat,  dtype=float)
    w_true_a = np.asarray(w_true, dtype=float)
    if w_hat_a.shape != w_true_a.shape:
        raise ValueError(
            f"w_hat and w_true must have the same shape; got "
            f"{w_hat_a.shape} vs {w_true_a.shape}."
        )

    if exclude_t0:
        w_hat_a  = w_hat_a[1:]
        w_true_a = w_true_a[1:]

    T_eff = int(w_true_a.shape[0])
    k     = max(1, int(round(k_frac * T_eff)))

    # Indices of the bottom-k (lowest) weights in each series.  argpartition
    # is O(T) and only guarantees that the first k elements are <= the rest,
    # which is all we need for set membership.
    idx_true = np.argpartition(w_true_a, k - 1)[:k]
    idx_hat  = np.argpartition(w_hat_a,  k - 1)[:k]
    overlap  = len(set(idx_true.tolist()) & set(idx_hat.tolist())) / k

    expected_random = k / T_eff
    lift            = overlap / expected_random if expected_random > 0 else float("nan")

    # Spearman correlation restricted to the true-outlier subset.  Severity
    # ordering: does the EM not just *flag* outliers but also rank them by
    # how extreme they are?
    #
    # spearmanr emits a ConstantInputWarning and returns an undefined
    # coefficient whenever *either* input is constant on the subset.  Two
    # by-design cases trigger this and must be skipped, returning nan:
    #   - GAUSSIAN ESTIMATOR: w_hat is identically 1, so w_hat_sub is constant
    #     (Experiment A, Gaussian column).
    #   - GAUSSIAN DGP: the true weights are identically 1, so w_true_sub is
    #     constant (Experiment B, where the DGP carries no outliers to rank).
    # The previous guard checked only w_hat_sub and so missed the second case.
    w_hat_sub  = w_hat_a[idx_true]
    w_true_sub = w_true_a[idx_true]
    if np.ptp(w_hat_sub) == 0.0 or np.ptp(w_true_sub) == 0.0:   # no ranking to recover
        spearman_on_outliers = float("nan")
    else:
        sp = spearmanr(w_hat_sub, w_true_sub)
        spearman_on_outliers = float(
            getattr(sp, "statistic", None) if hasattr(sp, "statistic") else sp[0]
        )

    return {
        "k":                    int(k),
        "k_frac":               float(k_frac),
        "T_eff":                int(T_eff),
        "overlap":              float(overlap),
        "expected_random":      float(expected_random),
        "lift":                 float(lift),
        "spearman_on_outliers": float(spearman_on_outliers),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   CONTAMINATION DETECTION DIAGNOSTIC (Experiment C)                      ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_contamination_detection(
    w_eps_hat: np.ndarray,
    contam_mask: np.ndarray | None,
    k_fracs: tuple[float, ...] = (0.05, 0.10),
    *,
    exclude_t0: bool = True,
) -> dict[str, Any]:
    r"""
    Contamination-detection diagnostic for Experiment C (``pi > 0``).

    This is the BINARY-ground-truth analogue of
    :func:`compute_outlier_rank_overlap`.  There the "truth" is the
    *continuous* mixing weight ``w_eps_true`` and we score the overlap of
    the bottom-:math:`k` sets of estimated vs. true weights.  Here the
    truth is the *contamination mask* ``z_t \in \{0, 1\}`` returned by the
    additive-outlier injection of the Experiment-C simulator: a hard,
    externally-imposed label of which periods carry a contaminating spike
    (drawn from a heavy ``t_{\nu_\mathrm{contam}}(0, \kappa^2 R)`` shock,
    see :func:`simulate_dfm.apply_contamination`).  The bottom-:math:`k`
    side of the comparison is *identical* — the periods the Student-t
    estimator down-weights most, i.e. the lowest ``w_eps_hat`` — only the
    "truth" side changes (the set ``{t : z_t = 1}`` instead of a bottom-k
    of true weights).

    Interpretation (thesis)
    -----------------------
    The *detection rate* is the fraction of the genuinely contaminated
    periods that the Student-t estimator successfully flags by assigning
    them the lowest mixing weights.  In the thesis' language it is the
    "detection rate of contaminated periods": *a high detection rate,
    combined with a low false-positive rate, is the diagnostic signature
    of a successful robustness mechanism* — the model must concentrate its
    down-weighting on the true outliers, not scatter it over clean
    periods.  The Gaussian estimator has no such mechanism: its
    ``w_eps_hat`` is identically 1 (constant), so there is no ranking to
    detect with and the metric is ``nan`` *by design* (not a failure —
    there simply is no down-weighting to score).

    Metrics
    -------
    Let :math:`n_c = \#\{t : z_t = 1\}` be the number of contaminated
    periods and :math:`T_\mathrm{eff}` the effective sample length.

    * **Natural-:math:`k` detection rate** (``detection_rate_natural``).
      Take exactly :math:`k = n_c` lowest-weight periods.  At this
      :math:`k`, precision and recall *coincide* (both equal
      ``hits / n_c``), which is the cleanest single number — it asks
      "if the estimator were allowed to flag exactly as many periods as
      were truly contaminated, what fraction would it get right?".
    * **Lift** (``detection_lift_natural``).  Detection rate divided by
      the rate a *random* bottom-:math:`k` set would achieve,
      :math:`k / T_\mathrm{eff} = n_c / T_\mathrm{eff}` (the
      contamination fraction).  ``lift = 1`` is chance; ``lift >> 1``
      means the down-weighting locks onto the true outliers.
    * **Fixed-fraction precision / recall / lift** at each ``k_frac``
      (default 5% and 10%), mirroring the ``overlap@5%`` / ``lift@5%``
      already reported for the continuous-weight diagnostic.  Here
      ``detection_rate_<p>`` is the *recall* (fraction of true outliers
      captured in the bottom :math:`p\%`) and ``detection_precision_<p>``
      the precision (fraction of the flagged set that is truly
      contaminated).

    Parameters
    ----------
    w_eps_hat : np.ndarray of shape (T,)
        Estimated observation-error mixing weights.  Low values are the
        periods the Student-t EM down-weights (its outlier flags).
    contam_mask : np.ndarray of bool, shape (T,), or None
        Ground-truth contamination indicator ``z_t`` from
        :func:`simulate_dfm.simulate_dfm` (key ``"contam_mask"``).
        ``None`` for the contamination-free Experiments A / B — in that
        case the function returns an **empty dict** (nothing to detect),
        so the caller can merge it unconditionally.
    k_fracs : tuple of float, default (0.05, 0.10)
        Fixed sample fractions for the auxiliary precision@k columns.
    exclude_t0 : bool, default True
        Drop ``t = 0`` before ranking, for exact consistency with
        :func:`compute_outlier_rank_overlap` (the ``w_eps`` effect is
        negligible, but a contaminated period at ``t = 0`` is dropped
        from *both* the truth and the candidate set, so the comparison
        stays aligned).

    Returns
    -------
    dict
        ``{}`` if ``contam_mask is None``.  Otherwise a flat dict with the
        keys ``detection_n_contam``, ``detection_T_eff``,
        ``detection_rate_natural``, ``detection_lift_natural`` and, per
        ``k_frac`` tag ``<p>pct``, ``detection_rate_<p>pct``,
        ``detection_precision_<p>pct``, ``detection_lift_<p>pct``.  The
        rate / precision / lift entries are ``nan`` when detection is
        undefined (no contaminated periods, or a constant — Gaussian —
        ``w_eps_hat``); the keys are always present so that aggregation
        across replications stays uniform.
    """
    # ── Edge case (A / B): no mask -> nothing to detect ──────────────────────
    # The contamination-free experiments pass contam_mask=None.  Returning an
    # empty dict lets compute_replication_metrics merge the result blindly.
    if contam_mask is None:
        return {}

    w_hat_a = np.asarray(w_eps_hat,  dtype=float)
    mask_a  = np.asarray(contam_mask, dtype=bool)
    if w_hat_a.shape != mask_a.shape:
        raise ValueError(
            f"w_eps_hat and contam_mask must have the same shape; got "
            f"{w_hat_a.shape} vs {mask_a.shape}."
        )

    if exclude_t0:
        w_hat_a = w_hat_a[1:]
        mask_a  = mask_a[1:]

    T_eff    = int(mask_a.shape[0])
    true_idx = np.flatnonzero(mask_a)
    n_contam = int(true_idx.size)

    # Build the full key set up-front (all nan), so that every replication —
    # contaminated or not, Student-t or Gaussian — yields the SAME keys and
    # aggregate_replications can pool them without special-casing.
    tags = [f"{int(round(f * 100))}pct" for f in k_fracs]
    out: dict[str, float] = {
        "detection_n_contam":     float(n_contam),
        "detection_T_eff":        float(T_eff),
        "detection_rate_natural": float("nan"),
        "detection_lift_natural": float("nan"),
    }
    for tag in tags:
        out[f"detection_rate_{tag}"]      = float("nan")
        out[f"detection_precision_{tag}"] = float("nan")
        out[f"detection_lift_{tag}"]      = float("nan")

    # ── Edge case (pi = 0): mask all-False -> 0 contaminated periods ──────────
    # Detection rate would be 0/0 (undefined), NOT 0.  Leave the rate / lift
    # entries at nan; aggregate_replications drops nan before averaging.
    if n_contam == 0:
        return out
    # ── Edge case (Gaussian estimator): w_eps_hat is identically 1 ───────────
    # A constant weight series carries no ranking, so there is no
    # down-weighting to score: detection is undefined by design -> nan.
    # (Same guard philosophy as the spearman block of the overlap diagnostic.)
    if np.ptp(w_hat_a) == 0.0:
        return out

    true_set = set(true_idx.tolist())

    def _detect_at_k(k: int) -> tuple[float, float, float]:
        """(recall, precision, lift) for the bottom-``k`` lowest-weight set."""
        k = max(1, min(k, T_eff))
        # argpartition is O(T) and only guarantees the first k entries are the
        # k smallest — exactly what set-membership needs (identical to the
        # bottom-k logic of compute_outlier_rank_overlap).
        idx_hat   = np.argpartition(w_hat_a, k - 1)[:k]
        hits      = len(true_set & set(idx_hat.tolist()))
        recall    = hits / n_contam            # frac. of true outliers caught
        precision = hits / k                   # frac. of flagged that are true
        expected  = k / T_eff                  # random-baseline recall
        lift      = recall / expected if expected > 0 else float("nan")
        return float(recall), float(precision), float(lift)

    # Natural k = n_contam : precision == recall (the cleanest single number).
    rate_nat, _, lift_nat = _detect_at_k(n_contam)
    out["detection_rate_natural"] = rate_nat
    out["detection_lift_natural"] = lift_nat

    # Fixed-fraction k (5%, 10%): the same precision@k columns reported for the
    # continuous-weight overlap, now scored against the binary mask.
    for k_frac, tag in zip(k_fracs, tags):
        rate_k, prec_k, lift_k = _detect_at_k(int(round(k_frac * T_eff)))
        out[f"detection_rate_{tag}"]      = rate_k
        out[f"detection_precision_{tag}"] = prec_k
        out[f"detection_lift_{tag}"]      = lift_k

    return out


def print_outlier_overlap_tables(
    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    *,
    k_fracs: tuple[float, ...] = (0.05, 0.10),
) -> None:
    """
    Pretty-print the outlier rank-overlap table for one or more replicas.

    Parameters
    ----------
    cases : list of (label, w_u_hat, w_u_true, w_eps_hat, w_eps_true,
                    linear_corr_u, linear_corr_eps).
        Actually a tuple with 6 entries — see ``run_outlier_diagnostic``
        for the canonical builder.
    k_fracs : tuple of float
        Fractions of T to treat as outlier sets.  Default (5%, 10%).
    """
    bar = "=" * 88
    print("\n" + bar)
    print("  OUTLIER RANK-OVERLAP DIAGNOSTIC  —  precision@k on the lowest weights")
    print(bar)

    # Header
    hdr = (f"  {'weight':<8s}  {'replica':<12s}  {'k%':>4s}  {'k':>5s}  "
           f"{'T_eff':>6s}  {'overlap':>9s}  {'rand':>8s}  "
           f"{'lift':>7s}  {'spear@outl':>11s}  {'lin.corr':>9s}")
    print(hdr)
    print("  " + "-" * 86)

    for case in cases:
        label, w_u_hat, w_u_true, w_eps_hat, w_eps_true, corr_u, corr_eps = case
        for w_label, w_hat, w_true, linear_corr in [
            ("w_u",   w_u_hat,   w_u_true,   corr_u),
            ("w_eps", w_eps_hat, w_eps_true, corr_eps),
        ]:
            for k_frac in k_fracs:
                res = compute_outlier_rank_overlap(
                    w_hat, w_true, k_frac=k_frac, exclude_t0=True,
                )
                print(f"  {w_label:<8s}  {label:<12s}  "
                      f"{100*k_frac:>4.0f}  {res['k']:>5d}  "
                      f"{res['T_eff']:>6d}  {res['overlap']:>9.2%}  "
                      f"{res['expected_random']:>8.2%}  "
                      f"{res['lift']:>7.2f}  "
                      f"{res['spearman_on_outliers']:>11.3f}  "
                      f"{linear_corr:>9.3f}")
        print("  " + "-" * 86)

    # Interpretation footer
    print("\n  Interpretation:")
    print("  - 'overlap' = precision@k = fraction of the TRUE bottom-k outlier")
    print("    periods that the EM also ranks in its bottom-k.")
    print("  - 'rand'    = k / T_eff = overlap a random ranking would achieve.")
    print("  - 'lift'    = overlap / rand.  Large lift (>> 1) means the EM")
    print("    identifies real outliers far better than chance, even when the")
    print("    full-sample linear correlation is moderate (the linear-corr")
    print("    'noise' lives at NORMAL periods, which are irrelevant for")
    print("    robustness; outlier identification is what protects the M-step).")
    print("  - 'spear@outl' = Spearman correlation of w_hat vs w_true restricted")
    print("    to the TRUE outlier subset — does the EM also rank outliers by")
    print("    severity?  Positive => yes; near zero => only the set is right,")
    print("    not the ranking within it.")
    print(bar)


def load_recovery_metrics(save_path: str | pathlib.Path) -> dict[str, Any]:
    """
    Reconstruct a metrics dict (suitable for :func:`print_recovery_table`)
    from a ``.npz`` previously written by :func:`run_recovery`.  Lets
    callers print or compare results without re-running the EM.
    """
    arc = np.load(save_path)
    keys = set(arc.files)
    metrics: dict[str, Any] = {}

    # Plain numeric and array fields (anything that is not prefixed).
    plain_keys = {
        "nu_u_star", "nu_u_hat", "nu_u_relerr",
        "nu_eps_star", "nu_eps_hat", "nu_eps_relerr",
        "rho_A_star", "rho_A_hat", "rho_A_relerr",
        "eig_A_star", "eig_A_hat",
        "sign_flips",
        "Lambda_relerr_per_series",
        "Lambda_relerr_norm",
        "Lambda_relerr_norm_proc_free",
        "Lambda_relerr_norm_proc_block",
        "H_free", "H_block",
        "diagQ_star", "diagQ_hat", "diagQ_relerr",
        "R_star", "R_hat", "R_relerr",
        "factor_abscorr",
        "w_u_corr", "w_eps_corr",
    }
    for k in plain_keys:
        if k not in keys:
            continue
        v = arc[k]
        # Unwrap scalar 0-d arrays into native floats / ints.
        if isinstance(v, np.ndarray) and v.shape == ():
            metrics[k] = v.item()
        else:
            metrics[k] = np.asarray(v)

    extras: dict[str, Any] = {}
    for k in keys:
        if k.startswith("extras__"):
            ev = arc[k]
            if isinstance(ev, np.ndarray) and ev.shape == ():
                extras[k.removeprefix("extras__")] = ev.item()
            else:
                extras[k.removeprefix("extras__")] = np.asarray(ev)
    metrics["_extras"] = extras
    return metrics


def _recovery_cache_valid(
    path: pathlib.Path,
    theta_star: dict,
    *,
    T: int,
    seed: int,
    force_recompute: bool,
) -> bool:
    r"""
    Decide whether the recovery archive at ``path`` may be reused as-is.

    Returns ``True`` only when reuse is safe: the file exists AND all three
    of the following conditions hold:
      1. ``theta_fingerprint`` matches the current ``theta_star`` (same DGP).
      2. ``extras__T`` matches the requested ``T`` (same sample length).
      3. ``extras__seed`` matches the requested ``seed`` (same simulation draw).

    Any other situation (force_recompute, missing file, missing keys, or any
    mismatch) returns ``False`` with an explanatory message so the caller
    recomputes instead of serving a zombie result.
    """
    if force_recompute:
        print(f"[cache] {path.name}: force_recompute=True — recomputing.")
        return False
    if not path.exists():
        return False
    with np.load(path) as arc:
        cached_fp   = (
            str(arc["theta_fingerprint"])
            if "theta_fingerprint" in arc.files else None
        )
        cached_T    = (
            int(arc["extras__T"])
            if "extras__T"    in arc.files else None
        )
        cached_seed = (
            int(arc["extras__seed"])
            if "extras__seed" in arc.files else None
        )
    current_fp = _theta_fingerprint(theta_star)
    if cached_fp is None:
        print(f"[cache] {path.name}: file has no fingerprint, cannot verify — "
              f"recomputing.")
        return False
    if cached_fp != current_fp:
        print(f"[cache] {path.name}: theta fingerprint mismatch "
              f"(cached {cached_fp} != current {current_fp}) — recomputing.")
        return False
    if cached_T != T:
        print(f"[cache] {path.name}: T mismatch "
              f"(cached {cached_T} != requested {T}) — recomputing.")
        return False
    if cached_seed != seed:
        print(f"[cache] {path.name}: seed mismatch "
              f"(cached {cached_seed} != requested {seed}) — recomputing.")
        return False
    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   RECOVERY TEXT OUTPUT                                                   ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def write_recovery_txt(
    metrics: dict[str, Any],
    *,
    T: int,
    out_path: pathlib.Path,
    ordered_cols: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    series_highlight: tuple[str, ...] = (
        "INDPRO", "PAYEMS", "S&P 500", "BAAFFM", "CPIAUCSL",
        "GDPC1", "UMCSENTx", "NFCI",
    ),
    factor_names: list[str] | None = None,
) -> None:
    """Write a plain-text recovery table to ``out_path``."""
    import datetime
    import io

    buf = io.StringIO()

    def w(line: str = "") -> None:
        buf.write(line + "\n")

    bar = "=" * 78
    w(bar)
    w(f"  RECOVERY TABLE  —  T = {T}")
    w(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    w(bar)

    # ── 1. Scalar parameters ─────────────────────────────────────────────────
    w()
    w("  SCALAR PARAMETERS")
    w(f"  {'parameter':<26s}  {'theta_star':>12s}  {'theta_hat':>12s}  "
      f"{'bias':>12s}  {'rel.err':>8s}")
    w("  " + "-" * 74)
    rows_s = [
        ("nu_u",               float(metrics["nu_u_star"]),
                               float(metrics["nu_u_hat"]),
                               float(metrics["nu_u_relerr"])),
        ("nu_eps",             float(metrics["nu_eps_star"]),
                               float(metrics["nu_eps_hat"]),
                               float(metrics["nu_eps_relerr"])),
        ("spectral_radius(A)", float(metrics["rho_A_star"]),
                               float(metrics["rho_A_hat"]),
                               float(metrics["rho_A_relerr"])),
    ]
    for name, star, hat, relerr in rows_s:
        bias = hat - star
        w(f"  {name:<26s}  {star:>12.4f}  {hat:>12.4f}  "
          f"  {bias:>+11.4f}  {relerr:>8.2%}")

    # ── 2. Eigenvalues of A ──────────────────────────────────────────────────
    w()
    w("  EIGENVALUES OF A  (sorted by descending |w|)")
    w(f"    {'#':>2s}  {'theta_star':>22s}    {'theta_hat':>22s}")
    w("    " + "-" * 52)
    for k, (ws, wh) in enumerate(zip(
            metrics["eig_A_star"], metrics["eig_A_hat"])):
        w(f"    {k:>2d}  {_fmt_complex(ws, 5):>22s}    {_fmt_complex(wh, 5):>22s}")

    # ── 3. Diagonal of Q ─────────────────────────────────────────────────────
    w()
    w("  DIAGONAL OF Q")
    w(f"  {'block':<12s}  {'theta_star':>12s}  {'theta_hat':>12s}  "
      f"{'bias':>12s}  {'rel.err':>8s}")
    w("  " + "-" * 60)
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        ds = float(metrics["diagQ_star"][j])
        dh = float(metrics["diagQ_hat"][j])
        re = float(metrics["diagQ_relerr"][j])
        w(f"  {b:<12s}  {ds:>12.4f}  {dh:>12.4f}  {dh - ds:>+12.4f}  {re:>8.2%}")

    # ── 4. Lambda — reference series ─────────────────────────────────────────
    w()
    w("  LAMBDA — REFERENCE SERIES  (after sign alignment)")
    w(f"  {'series':<14s}  {'block':<10s}  {'j':>2s}  "
      f"{'Lambda_star':>12s}  {'Lambda_hat':>12s}  {'bias':>12s}  {'rel.err':>8s}")
    w("  " + "-" * 78)
    Lambda_star_arr = np.asarray(metrics["_extras"]["Lambda_star"])
    Lambda_hat_sgn  = np.asarray(metrics["_extras"]["Lambda_hat_sgn"])
    for name in series_highlight:
        if name not in ordered_cols:
            continue
        i      = ordered_cols.index(name)
        b      = block_map.get(name, "?")
        j      = _BLOCK_TO_COL.get(b, -1)
        ls     = float(Lambda_star_arr[i, j]) if j >= 0 else float("nan")
        lh     = float(Lambda_hat_sgn[i, j])  if j >= 0 else float("nan")
        re     = abs(lh - ls) / max(abs(ls), 1e-12)
        bias_l = lh - ls
        w(f"  {name:<14s}  {b:<10s}  {j:>2d}  "
          f"{ls:>+12.4f}  {lh:>+12.4f}  {bias_l:>+12.4f}  {re:>8.2%}")

    # ── 5. Procrustes summary ────────────────────────────────────────────────
    w()
    w("  PROCRUSTES ALIGNMENT  "
      "(||Lambda_hat H - Lambda_star||_F / ||Lambda_star||_F)")
    w(f"  {'normalised-only (sign+Conv1)':<34s}  "
      f"{float(metrics['Lambda_relerr_norm']):>10.3%}")
    w(f"  {'Procrustes (free 3x3 ortho)':<34s}  "
      f"{float(metrics['Lambda_relerr_norm_proc_free']):>10.3%}")
    w(f"  {'Procrustes (block-diagonal)':<34s}  "
      f"{float(metrics['Lambda_relerr_norm_proc_block']):>10.3%}")

    # ── 6. R summary ─────────────────────────────────────────────────────────
    R_relerr = np.asarray(metrics["R_relerr"])
    w()
    w("  R (IDIOSYNCRATIC VARIANCES)")
    w(f"  median rel.err = {float(np.median(R_relerr)):.2%}   "
      f"max rel.err = {float(np.max(R_relerr)):.2%}")

    # ── 7. Factor and weight correlations ────────────────────────────────────
    w()
    w("  LATENT-PATH RECOVERY")
    for j, b in enumerate(_labels(_r_of(metrics), factor_names)):
        w(f"  |corr(f_hat_{j}, F_true_{j})|  [{b:<10s}]  = "
          f"{float(metrics['factor_abscorr'][j]):.4f}")
    w(f"  corr(w_u_hat,   w_u_true)                       "
      f"= {float(metrics['w_u_corr']):+.4f}")
    w(f"  corr(w_eps_hat, w_eps_true)                     "
      f"= {float(metrics['w_eps_corr']):+.4f}")

    w()
    w(bar)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  Saved recovery table to: {out_path}")


def write_recovery_summary(
    cases: list[tuple[int, dict[str, Any]]],
    out_path: pathlib.Path,
    factor_names: list[str] | None = None,
) -> None:
    """Write side-by-side recovery summary comparing multiple T values."""
    import datetime

    lines: list[str] = []
    bar = "=" * 78

    lines.append(bar)
    lines.append("  RECOVERY SUMMARY  —  comparison across sample sizes")
    lines.append(f"  Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("  NOTE: rel.err is relative to theta_star.")
    lines.append("  Improvement indicator: rel.err should DECREASE as T grows.")
    lines.append(bar)

    hdr = f"  {'metric':<42s}" + "".join(
        f"  {'T='+str(T):>12s}" for T, _ in cases
    )
    sep = "  " + "-" * (42 + 14 * len(cases))
    lines.append("")
    lines.append(hdr)
    lines.append(sep)

    metric_rows: list[tuple[str, Any]] = [
        ("nu_u          star",
         lambda m: float(m["nu_u_star"])),
        ("nu_u          hat",
         lambda m: float(m["nu_u_hat"])),
        ("nu_u          rel.err",
         lambda m: float(m["nu_u_relerr"])),
        ("nu_eps        star",
         lambda m: float(m["nu_eps_star"])),
        ("nu_eps        hat",
         lambda m: float(m["nu_eps_hat"])),
        ("nu_eps        rel.err",
         lambda m: float(m["nu_eps_relerr"])),
        ("rho(A)        star",
         lambda m: float(m["rho_A_star"])),
        ("rho(A)        hat",
         lambda m: float(m["rho_A_hat"])),
        ("rho(A)        rel.err",
         lambda m: float(m["rho_A_relerr"])),
        ("Lambda relerr (normalised)",
         lambda m: float(m["Lambda_relerr_norm"])),
        ("Lambda relerr (Proc.free)",
         lambda m: float(m["Lambda_relerr_norm_proc_free"])),
        ("Lambda relerr (Proc.block)",
         lambda m: float(m["Lambda_relerr_norm_proc_block"])),
    ]

    # Righe PER FATTORE. Generate su `r` letto dai metrics, NON cablate a 3.
    #
    # Erano tre righe fisse con etichette `f_R`/`f_F`/`f_X` e `[0=real]`/
    # `[1=fin.]`/`[2=other]`, cioe' i nomi dei tre blocchi di base. Con le spec
    # nominate questo sbagliava due volte: i nomi non corrispondevano ai fattori
    # veri, e su ogni spec con r > 3 il quarto fattore veniva SILENZIOSAMENTE
    # omesso dal summary. Non era cosmetico: su `fed_overlap` (r=4) il quarto
    # fattore e' quello con la correlazione peggiore (0.74) e il quarto elemento
    # di Q e' il piu' fuori scala della cella (5.4x) — leggendo il summary quella
    # cella sembrava la piu' sana delle tre, mentre e' quella con il problema.
    # Il resto del file usava gia' `_labels()`; solo questa tabella era rimasta
    # indietro.
    _r = _r_of(cases[0][1])
    _lab = _labels(_r, factor_names)
    _w = max(len(b) for b in _lab)

    for j, b in enumerate(_lab):
        metric_rows.append((
            f"|corr(f_{b}, F_true_{b})|",
            # `j=j` lega l'indice alla DEFINIZIONE della lambda. Senza, tutte
            # le chiusure vedrebbero l'ultimo `j` del ciclo e ogni riga
            # stamperebbe il medesimo fattore.
            lambda m, j=j: float(np.asarray(m["factor_abscorr"])[j]),
        ))

    metric_rows += [
        ("corr(w_u_hat,   w_u_true)",
         lambda m: float(m["w_u_corr"])),
        ("corr(w_eps_hat, w_eps_true)",
         lambda m: float(m["w_eps_corr"])),
        ("R median rel.err",
         lambda m: float(np.median(np.asarray(m["R_relerr"])))),
        ("R max rel.err",
         lambda m: float(np.max(np.asarray(m["R_relerr"])))),
    ]

    for j, b in enumerate(_lab):
        metric_rows.append((
            f"diag(Q)[{j}={b:<{_w}s}] rel.err",
            lambda m, j=j: float(np.asarray(m["diagQ_relerr"])[j]),
        ))

    for label, fn in metric_rows:
        row = f"  {label:<42s}"
        for _T, m in cases:
            try:
                v = fn(m)
                row += f"  {v:>12.4f}"
            except Exception:
                row += f"  {'N/A':>12s}"
        lines.append(row)

    lines.append("")
    lines.append(bar)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved recovery summary to: {out_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   MAIN — RUN THE TWO REPLICAS AND PRINT THE TABLES                       ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # ── Recovery su TUTTE le celle del dataset `final` ───────────────────────
    # Per ogni cella (spec x variante) si simula dal theta stimato su quella
    # cella e si ri-stima da zero, a due lunghezze campionarie. Gli artefatti
    # vanno in output/recovery/final/<spec>/<variante>/.
    #
    # NIENTE E' CABLATO SU r: i fattori si contano dalla struttura, e le
    # tabelle usano i nomi della spec. La versione precedente indicizzava
    # `factor_abscorr[0..2]` e avrebbe perso il quarto fattore delle spec a
    # r = 4 senza dare errore.
    import itertools

    from config_utils import parse_spec_variant_args, SPECS, VARIANTS as _VNAMES
    from em.selftest_fixture import load_fixture
    from run_final_artifacts import VARIANTS as _V, load_final_fit

    T_GRID = (500, 2000)
    SEED = 42

    def _extra(ap):
        ap.add_argument("--all", action="store_true",
                        help=f"tutte le {len(SPECS) * len(_VNAMES)} celle "
                             f"(ignora --spec/--variant)")
        ap.add_argument("--force", action="store_true",
                        help="ignora la cache e ricalcola")

    args = parse_spec_variant_args(
        "Recovery Monte Carlo sulle celle del dataset final.", extra=_extra)
    force_recompute = bool(args.force) or os.environ.get("MC_RECOVERY_FORCE") == "1"

    celle = (list(itertools.product(SPECS, _VNAMES)) if args.all
             else [(args.spec, args.variant)])

    print("=" * 78)
    print(f"  RECOVERY — {len(celle)} cella/e x {len(T_GRID)} lunghezze "
          f"(T = {', '.join(str(t) for t in T_GRID)}), seed = {SEED}")
    print("=" * 78)

    esiti: list[tuple] = []
    for spec, variant in celle:
        print("\n" + "#" * 78)
        print(f"#  {spec} / {variant}")
        print("#" * 78)

        try:
            theta_star = load_final_fit(spec, variant)["theta"]
        except FileNotFoundError as exc:
            print(f"  [salto] {exc}".splitlines()[0])
            esiti.append((spec, variant, None, "cella non stimata"))
            continue

        fx = load_fixture(spec)
        ordered_cols = fx.ordered_cols
        structure    = fx.structure
        block_map    = structure.display_block_map()     # solo per le tabelle
        freq_map     = dict(zip(ordered_cols, fx.freq_list))
        factor_names = list(structure.factor_names)
        _vcfg = _V[variant]

        out_dir = _PROJECT_ROOT / "output" / "recovery" / "final" / spec / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  r = {structure.r}, fattori = {factor_names}, "
              f"{'diagonale' if structure.diagonal else 'NON diagonale'}")
        print(f"  DGP: idio_ar1={'rho' in theta_star}, "
              f"gaussian={_vcfg['gaussian']}, "
              f"per_series_weights={_vcfg.get('per_series_weights', False)}")
        print(f"  output -> {out_dir}")

        per_T: list[tuple[int, dict]] = []
        for T in T_GRID:
            npz = out_dir / f"mc_recovery_T{T}.npz"
            if _recovery_cache_valid(npz, theta_star, T=T, seed=SEED,
                                     force_recompute=force_recompute):
                print(f"\n  [riuso] {npz.name} (fingerprint + T + seed verificati)")
                m = load_recovery_metrics(npz)
            else:
                m = run_recovery(
                    theta_star, T=T, seed=SEED, verbose_em=False,
                    save_path=npz,
                    ordered_cols=ordered_cols, block_map=block_map,
                    freq_map=freq_map, structure=structure,
                    gaussian=_vcfg["gaussian"],
                    per_series_weights=_vcfg.get("per_series_weights", False),
                )["metrics"]
            per_T.append((T, m))

            print_recovery_table(m, label=f"T = {T}  [{spec}/{variant}]",
                                 ordered_cols=ordered_cols, block_map=block_map,
                                 factor_names=factor_names)
            write_recovery_txt(
                m, T=T,
                out_path=out_dir / f"recovery_T{T}_{spec}_{variant}.txt",
                ordered_cols=ordered_cols, block_map=block_map,
                factor_names=factor_names,
            )

        # ── Confronto fianco a fianco fra le lunghezze ───────────────────────
        # Le righe per-fattore sono generate da `r`, non elencate a mano.
        righe = [
            ("nu_u rel.err",                     "nu_u_relerr"),
            ("nu_eps rel.err",                   "nu_eps_relerr"),
            ("rho(A) rel.err",                   "rho_A_relerr"),
            ("Lambda rel.err (normalised)",      "Lambda_relerr_norm"),
            ("Lambda rel.err (Procr. free)",     "Lambda_relerr_norm_proc_free"),
            ("Lambda rel.err (Procr. block)",    "Lambda_relerr_norm_proc_block"),
        ]
        print("\n" + "=" * 78)
        print(f"  CONFRONTO  {spec}/{variant}")
        print("=" * 78)
        intest = "  " + f"{'metric':<38s}" + "".join(f"{'T='+str(t):>13s}" for t, _ in per_T)
        print(intest)
        print("  " + "-" * (38 + 13 * len(per_T)))
        for nome, chiave in righe:
            vals = "".join(f"{float(m[chiave]):>13.4f}" for _, m in per_T)
            print(f"  {nome:<38s}{vals}")
        for j in range(structure.r):
            nome = f"|corr(f_{factor_names[j]}, F_true)|"
            vals = "".join(f"{float(np.asarray(m['factor_abscorr'])[j]):>13.4f}"
                           for _, m in per_T)
            print(f"  {nome:<38s}{vals}")
        for nome, chiave in (("corr(w_u_hat, w_u_true)", "w_u_corr"),
                             ("corr(w_eps_hat, w_eps_true)", "w_eps_corr")):
            vals = "".join(f"{float(m[chiave]):>13.4f}" for _, m in per_T)
            print(f"  {nome:<38s}{vals}")

        write_recovery_summary(
            per_T,
            out_path=out_dir / f"recovery_summary_{spec}_{variant}.txt",
            factor_names=factor_names,
        )
        esiti.append((spec, variant, per_T, "ok"))

    # ── Riepilogo finale ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RIEPILOGO RECOVERY")
    print("=" * 78)
    print(f"  {'cella':<32s}" + "".join(f"{'Lam relerr T='+str(t):>20s}"
                                        for t in T_GRID))
    print("  " + "-" * (32 + 20 * len(T_GRID)))
    for spec, variant, per_T, stato in esiti:
        if per_T is None:
            print(f"  {spec+'/'+variant:<32s}  {stato}")
            continue
        vals = "".join(
            f"{float(m['Lambda_relerr_norm_proc_block']):>20.4f}"
            for _, m in per_T)
        print(f"  {spec+'/'+variant:<32s}{vals}")
    print("\n  (Lambda rel.err Procruste-block: piu' basso = recupero migliore;")
    print("   deve CALARE al crescere di T se il modello e' identificato)")
    print("=" * 78)
