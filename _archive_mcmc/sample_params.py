"""
src/mcmc/sample_params.py
=========================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: i parametri del modello dai loro condizionali completi, dato un percorso
f già estratto e i pesi w (la volatilità entra solo via la precisione combinata
g_t = w_t / h_t; con h≡1 tornano i draw pesati semplici):

    (A, Q)      dal VAR di stato   f_t = A f_{t-1} + u_t           [eq:state]
                → statistiche di secondo momento pesate + draw MNIW
                  (Matrix-Normal–Inverse-Wishart)
    (Λ, R)      dall'osservazione  y_t = Λ f_t + ε_t              [eq:obs, eq:idio]
                → per serie, regressione pesata mixed-frequency + draw NIG
                  (Normal–Inverse-Gamma); R = diag(r_1..r_M)
    (ν_u, ν_ε)  dai pesi  w_t ~ Gamma(ν/2, ν/2)  → griddy-Gibbs sul condizionale
                1-D log-concavo, statistiche sufficienti  Σ_t log w_t  e  Σ_t w_t

Ogni condizionale è l'EM col PUNTO/MEDIA sostituito da un DRAW dallo stesso posterior
coniugato.

Gibbs step (d): draw the model parameters ``(A, Q)``, ``(Lambda, R)`` and
``(nu_u, nu_eps)`` from their full conditionals, given a sampled factor path and
sampled weights.  These draws are SV/leverage-agnostic: the volatility enters
only through the combined precision ``g_t = w_t / h_t`` fed by the orchestrator
(with ``h == 1`` they reduce to the plain weighted draws).

Reuse vs new
------------
Every conditional is built on EM machinery, replacing the EM *point/mean* with a
*draw* from the same posterior:

* ``(A, Q)`` — sufficient statistics from :func:`em_m_step.compute_weighted_moments`
  (fed the *sampled* augmented path with ``P_smooth = 0``, so the moments are the
  realized weighted second moments) → MNIW draw :func:`mcmc.shared.draw_A_Q`.
* ``(Lambda, R)`` — per-series weighted sufficient statistics, identical to the
  block-restricted mixed-frequency M-step (monthly regressor = block factor;
  quarterly regressor = MM composite :func:`mcmc.shared.composite_regressor`),
  with the posterior covariance term dropped (sampled state) → NIG draw
  :func:`mcmc.shared.draw_lambda_r_series`.
* ``(nu_u, nu_eps)`` — griddy-Gibbs on :func:`mcmc.shared.nu_log_target`, which
  is the 1-D log-concave full conditional (log-concavity proven in the thesis);
  the sufficient statistics are ``sum_t log w_t`` and ``sum_t w_t`` of the
  *sampled* weights.

Config-aware
------------
The block->column map and the MM weights are imported from ``em_m_step`` (the
single source of truth used by the EM), so MCMC and EM agree by construction.
Dimensions (``r``, ``M``), the block structure and the per-series frequencies all
come from the arguments (``theta``, ``block_map``, ``freq_list``, ``ordered_cols``),
never hard-coded.
"""

from __future__ import annotations

import numpy as np

from em.em_m_step import (
    compute_weighted_moments,
    _MM_WEIGHTS,
)
from em.factor_structure import as_structure

from mcmc.shared import (
    composite_regressor,
    draw_A_Q,
    draw_lambda_r_series,
    nu_log_target,
)


# ─────────────────────────────────────────────────────────────────────────────
# (d.1) Transition block (A, Q)
# ─────────────────────────────────────────────────────────────────────────────

def draw_A_Q_block(
    f_aug: np.ndarray,
    w_u: np.ndarray,
    r: int,
    rng: np.random.Generator,
    *,
    Psi0: np.ndarray | None = None,
    nu0: float = 0.0,
    A0: np.ndarray | None = None,
    kappa: float = 0.0,
    fix_q_identity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Draw ``(A, Q)`` from the MNIW full conditional given the sampled factor path
    and factor-side combined precisions ``g^u_t``.

    The realized weighted moments are obtained from
    :func:`compute_weighted_moments` with ``P_smooth = P_lag = 0`` (the state is
    *sampled*, so there is no posterior covariance), then passed to
    :func:`draw_A_Q`.  ``T_eff = T - 1`` matches the EM convention (one
    transition per consecutive pair, summed over ``t = 1..T-1``).

    This is the **scalar-volatility** draw of eq:param-AQ: it is exact whenever
    ``G^u_t`` factors as ``g^u_t Q^{-1}`` — the no-SV cell (``g = w``) and the
    degenerate ``H^u_t = h^u_t I``.  Per-factor Spec II uses
    :func:`mcmc.shared.draw_A_Q_perfactor`.

    ``Psi0, nu0`` (IW on ``Q``) and ``A0, kappa`` (natural-conjugate matrix-Normal
    on ``A``, prior precision ``kappa I`` on the regressors) are the Family~A
    priors of ``tab:param-prior-tuning``; the defaults are the flat limit and
    reproduce the EM seam bit-for-bit.
    """
    T = f_aug.shape[0]
    zeros = np.zeros((T, 5 * r, 5 * r))
    mom = compute_weighted_moments(f_aug, zeros, zeros, w_u, r)
    return draw_A_Q(mom["P00"], mom["P10"], mom["P11"], T - 1, rng,
                    Psi0=Psi0, nu0=nu0, A0=A0, kappa=kappa,
                    fix_q_identity=fix_q_identity)


# ─────────────────────────────────────────────────────────────────────────────
# (d.2) Loading / idiosyncratic-variance block (Lambda, R)
# ─────────────────────────────────────────────────────────────────────────────

def draw_Lambda_R_block(
    Y: np.ndarray,
    f_aug: np.ndarray,
    w_eps: np.ndarray,
    block_map: dict[str, str],
    freq_list: list[str],
    ordered_cols: list[str],
    r: int,
    rng: np.random.Generator,
    *,
    a0: float = 0.0,
    b0: float | np.ndarray = 0.0,
    m0: np.ndarray | None = None,
    M0_inv: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Draw the block-diagonal loading matrix ``Lambda`` (M, r) and the
    idiosyncratic variances ``R`` (M,) from their per-series NIG full
    conditionals, mirroring the block-restricted mixed-frequency M-step.

    For each series ``i`` the regressor is the contemporaneous block factor
    (monthly) or its MM composite (quarterly); the weighted sufficient
    statistics are formed over the observed set and the posterior-covariance
    term is dropped (sampled state).  Each ``(lambda_i, r_i)`` is then a draw
    from :func:`draw_lambda_r_series`.

    ``w_eps`` may be ``(T,)`` (Passo 1: scalar tail weight per period) or
    ``(T, M)`` (Passo 2: the combined precision ``g^eps_{i,t}=w^eps_t/h^eps_{i,t}``
    per series).  In both cases series ``i`` uses its own column.

    Family~A priors (tab:param-prior-tuning, ``L,R`` rows).  The per-series NIG
    prior ``r_i ~ IG(a0, b0_i)``, ``L_i | r_i ~ N(m0_i, r_i / M0_inv)`` is threaded
    to :func:`draw_lambda_r_series`:

    * ``a0`` : scalar IG shape (default ``2`` in the tuning table).
    * ``b0`` : scalar or ``(M,)`` IG scale; ``b0_i=(a0-1)\widehat r_{i,\text{EM}}``
      centres the idiosyncratic-variance prior mean at the EM value.
    * ``m0`` : ``None`` or ``(M, r)`` prior mean matrix; series ``i`` reads its
      own block column ``m0[i, j]`` (``=\widehat L_{i,\text{EM}}``).  The whole
      loading matrix is passed so the block column is picked with the same ``j``
      used for the likelihood term.
    * ``M0_inv`` : scalar prior precision ``1/M0`` on the loading (``\kappa``).

    The flat limit ``a0=b0=M0_inv=0``, ``m0=None`` reproduces the weighted-LS
    draw **bit-for-bit** (the EM seam of ``test_shared``).
    """
    T, M = Y.shape
    Lambda = np.zeros((M, r))
    R = np.zeros(M)
    w_eps = np.asarray(w_eps, float)
    per_series = (w_eps.ndim == 2)
    b0_arr = np.broadcast_to(np.asarray(b0, float), (M,))

    phi = composite_regressor(f_aug, _MM_WEIGHTS, r)   # (T, r), quarterly regressor

    # Colonna-fattore di ciascuna serie, dalla struttura invece che dalla
    # vecchia mappa cablata blocco->colonna (real/financial/other).
    # NOTA: questo campionatore assume UNA colonna per serie (Lambda diagonale
    # a blocchi). Vale per le strutture diagonali; una struttura con overlap
    # (piu' fattori per riga) richiederebbe un draw congiunto multivariato ed
    # e' fuori dallo scope di src/mcmc. Meglio fallire qui che prendere
    # silenziosamente il primo fattore e stimare meta' modello.
    _fs = as_structure(block_map, list(ordered_cols))
    if not _fs.diagonal:
        raise NotImplementedError(
            f"draw_Lambda_R_block: la struttura '{_fs.spec_name}' non e' "
            f"diagonale (righe con piu' di un fattore: "
            f"{int((_fs.mask.sum(axis=1) > 1).sum())}). Il campionatore MCMC "
            f"assume una colonna per serie."
        )
    _col_of = {i: int(_fs.factors_of_series(i)[0]) for i in range(M)}

    for i in range(M):
        name = ordered_cols[i]
        j = _col_of[i]
        freq = freq_list[i]

        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        if obs_t.size == 0:
            # No observations: leave loading at 0, draw R from a weak default.
            # (Does not arise for the configured panels; guard only.)
            Lambda[i, j] = 0.0
            R[i] = 1.0
            continue

        y_i = Y[obs_t, i]
        w_i = w_eps[obs_t, i] if per_series else w_eps[obs_t]
        E_x = (f_aug[obs_t, j] if freq == "monthly" else phi[obs_t, j])

        num = float(np.sum(w_i * y_i * E_x))
        den = float(np.sum(w_i * E_x * E_x))
        s_yy = float(np.sum(w_i * y_i * y_i))
        n_obs = int(obs_t.size)

        m0_i = float(m0[i, j]) if m0 is not None else 0.0
        lam, rr = draw_lambda_r_series(
            num, den, s_yy, n_obs, rng,
            a0=a0, b0=float(b0_arr[i]), m0=m0_i, M0_inv=M0_inv,
        )
        Lambda[i, j] = lam
        R[i] = rr

    return Lambda, R


# ─────────────────────────────────────────────────────────────────────────────
# (d.3) Degrees of freedom (nu_u, nu_eps) — griddy-Gibbs
# ─────────────────────────────────────────────────────────────────────────────

def nu_log_prior_exponential(mean: float = 20.0):
    r"""
    Proper **exponential** prior on the Student-t degrees of freedom (Family~D,
    ``subsec:param-familyD`` / ``tab:param-prior-tuning``): ``p(nu) ∝ exp(-nu/mean)``
    on ``(2, ∞)``.  Decreasing in ``nu`` — it expresses that the tails are
    genuinely heavy but not pathologically so ("exponential with mean around 20").

    Returns a callable ``nu -> log p(nu)`` (up to an additive constant, which the
    griddy normalisation discards) for :func:`draw_nu_griddy`'s ``log_prior`` hook.
    It is log-concave (linear in ``nu``), so it preserves the log-concavity of the
    ``nu`` full conditional (thesis ``eq:param-nu-logtarget``).  The lower bound
    ``nu > 2`` is enforced by the griddy bracket, not here.
    """
    inv = 1.0 / float(mean)
    return lambda nu: -inv * float(nu)


def nu_log_prior_uniform(lo: float = 2.0, hi: float = 50.0):
    r"""
    Proper **bounded-uniform** prior on ``nu`` (Family~D): flat on ``(lo, hi)``,
    ``-inf`` outside — "a bounded uniform, Unif(2, 50)".  The griddy grid bracket
    is itself a bounded-uniform prior (thesis note), so this simply *tightens* the
    effective support to ``(lo, hi)``; grid points outside receive zero weight.

    Returns a callable ``nu -> log p(nu)`` for :func:`draw_nu_griddy`.
    """
    def _lp(nu):
        return 0.0 if (lo < nu < hi) else -np.inf
    return _lp

def draw_nu_griddy(
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    nu_bounds: tuple[float, float] = (2.001, 1000.0),
    grid_size: int = 600,
    log_prior=None,
) -> float:
    r"""
    Griddy-Gibbs draw of a Student-t degrees-of-freedom parameter from its
    full conditional ``p(nu | w) propto p(nu) * prod_t Gamma(w_t; nu/2, nu/2)``.

    The (un-normalised) log full conditional is :func:`nu_log_target`; it is
    1-D and log-concave (thesis result), so a fine grid is more than adequate.
    A **geometric** grid spans the bracket (nu ranges over ~2..1000), and each
    grid point is weighted by the local cell width (trapezoidal) so the discrete
    sample approximates the continuous posterior rather than the density on a
    non-uniform grid.

    Parameters
    ----------
    weights : (n,)        the *sampled* mixing weights (all Gamma(nu/2, nu/2));
                          for nu_u pass ``w_u``, for nu_eps pass ``w_eps``.
    nu_bounds : (lo, hi)  bracket, identical default to the EM ``update_nu``.
    grid_size : int       number of grid points.
    log_prior : callable or None  optional ``nu -> log p(nu)``; ``None`` = flat.
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    sum_w = float(np.sum(w))
    sum_log_w = float(np.sum(np.log(w)))

    lo, hi = nu_bounds
    grid = np.geomspace(lo, hi, grid_size)
    logp = np.array([nu_log_target(nu, sum_log_w, sum_w, n, log_prior) for nu in grid])
    logp -= logp.max()
    dens = np.exp(logp)
    cell = np.gradient(grid)                 # local width (trapezoidal weights)
    probs = dens * cell
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate target (should not happen): fall back to grid midpoint.
        return float(grid[grid_size // 2])
    probs /= total
    return float(rng.choice(grid, p=probs))
