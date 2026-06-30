"""
src/mcmc/sample_params.py
=========================

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
    _BLOCK_TO_COL,
    _MM_WEIGHTS,
)

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
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Draw ``(A, Q)`` from the flat-prior MNIW full conditional given the sampled
    factor path and factor-side weights.

    The realized weighted moments are obtained from
    :func:`compute_weighted_moments` with ``P_smooth = P_lag = 0`` (the state is
    *sampled*, so there is no posterior covariance), then passed to
    :func:`draw_A_Q`.  ``T_eff = T - 1`` matches the EM convention (one
    transition per consecutive pair, summed over ``t = 1..T-1``).
    """
    T = f_aug.shape[0]
    zeros = np.zeros((T, 5 * r, 5 * r))
    mom = compute_weighted_moments(f_aug, zeros, zeros, w_u, r)
    return draw_A_Q(mom["P00"], mom["P10"], mom["P11"], T - 1, rng)


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
    """
    T, M = Y.shape
    Lambda = np.zeros((M, r))
    R = np.zeros(M)
    w_eps = np.asarray(w_eps, float)
    per_series = (w_eps.ndim == 2)

    phi = composite_regressor(f_aug, _MM_WEIGHTS, r)   # (T, r), quarterly regressor

    for i in range(M):
        name = ordered_cols[i]
        j = _BLOCK_TO_COL[block_map[name]]
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

        lam, rr = draw_lambda_r_series(num, den, s_yy, n_obs, rng)
        Lambda[i, j] = lam
        R[i] = rr

    return Lambda, R


# ─────────────────────────────────────────────────────────────────────────────
# (d.3) Degrees of freedom (nu_u, nu_eps) — griddy-Gibbs
# ─────────────────────────────────────────────────────────────────────────────

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
