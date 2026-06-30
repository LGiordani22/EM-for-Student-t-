"""
src/mcmc/shared.py
==================

Reusable mathematical kernels for the Gibbs sampler of the Student-t DFM with
stochastic volatility and leverage (master cell D1-b x D2-b).

These are the helpers that, in the EM code, live "incastrate" (embedded) inside
the E-/M-step functions.  They are extracted here, unchanged in formula, so the
Gibbs blocks can call them directly.

PASSO 0 contract
----------------
* **Additive only.**  This module depends on ``numpy`` and ``scipy`` ONLY.  It
  does NOT import the EM code, and the EM code is left untouched.
* **Verified against EM.**  Every helper reproduces the corresponding EM
  quantity (``test_shared.py`` asserts this to machine precision for the
  deterministic seams, and within Monte-Carlo error for the stochastic draws).
* **Draw, not mean.**  Where EM took a posterior *mean* (the plug-in E-step) or
  a *point* (the closed-form M-step), the helper here either keeps the exact
  formula (the deterministic kernels) or replaces the mean/point with a *draw*
  from the same conjugate posterior (the ``draw_*`` helpers).

Mapping to the EM source
------------------------
+----------------------------+-------------------------------------------------+
| helper                     | EM counterpart (same formula)                   |
+============================+=================================================+
| realized_deflated_d_eps    | em_e_step.compute_d_eps with P_smooth = 0,       |
|                            |   plus per-series deflation by h^eps_t           |
| realized_deflated_d_u      | em_e_step.compute_d_u with P_smooth = P_lag = 0, |
|                            |   plus deflation by h^u_t                        |
| draw_weights               | em_e_step.compute_weights (draw, not mean)       |
| nu_foc / nu_log_target     | the g(nu) closure inside em_m_step.update_nu     |
| draw_A_Q                   | em_m_step.update_A_Q (MNIW draw, same moments)   |
| composite_regressor        | the phi^k_t regressor inside em_m_step.update_*  |
| draw_lambda_r_series       | em_m_step.update_Lambda + update_R (NIG draw)    |
+----------------------------+-------------------------------------------------+

Stochastic-volatility hook
--------------------------
The single new ingredient relative to EM is the deflation by the stochastic
volatility path ``h_t``.  In the conditional model the idiosyncratic variance
is ``h^eps_{i,t} * r_i`` and the factor-innovation covariance is ``h^u_t * Q``;
the Mahalanobis residuals must therefore be deflated by ``h`` before they enter
the weight / parameter conditionals.  When ``h`` is omitted (``None``), every
helper reduces to its exact EM counterpart at ``h == 1``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import invwishart

# Mariano-Murasawa aggregation weights c_l, l = 0..4 — identical to
# em_m_step._MM_WEIGHTS and kalman.MM_WEIGHTS_DEFAULT.  Fixed (not estimated).
MM_WEIGHTS: np.ndarray = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Realized, volatility-deflated Mahalanobis residuals
# ─────────────────────────────────────────────────────────────────────────────

def realized_deflated_d_eps(
    Y: np.ndarray,
    f_states: np.ndarray,
    Lambda_tilde: np.ndarray,
    R: np.ndarray,
    W_list: list[np.ndarray],
    h_eps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Idiosyncratic squared Mahalanobis residual from a *sampled* state path,
    optionally deflated by the idiosyncratic stochastic-volatility path.

    This is :func:`em_e_step.compute_d_eps` stripped of the posterior-uncertainty
    trace term (because, conditional on a *drawn* state path, there is no
    posterior covariance — the state is fixed), and with each squared residual
    additionally divided by its own volatility ``h^eps_{i,t}``.

    Equivalence to EM (verified in ``test_shared``)
    -----------------------------------------------
    With ``h_eps = None`` this returns exactly
    ``compute_d_eps(Y, f_states, P_smooth=0, Lambda_tilde, R, W_list)`` — the
    realized-residual limit of the EM quantity (``term2`` vanishes when
    ``P_smooth = 0``).

    Parameters
    ----------
    Y : (T, M)            observation panel (NaN = missing).
    f_states : (T, 5r)    a *sampled* augmented state path (FFBS draw), or any
                          state at which to evaluate the realized residual.
    Lambda_tilde : (M, 5r) effective augmented loading (kalman.build_Lambda_tilde).
    R : (M,)              idiosyncratic variances r_i (diagonal of R).
    W_list : length T     selection matrices (kalman.build_all_selection_matrices).
    h_eps : (T, M) or None idiosyncratic volatilities h^eps_{i,t}.  ``None`` ⇒ 1.

    Returns
    -------
    d_eps : (T,)          realized deflated residual, ``check d^eps_t``.
    m_obs : (T,)  int     number of observed series at each t.
    """
    T, M = Y.shape
    d_eps = np.zeros(T)
    m_obs = np.zeros(T, dtype=int)
    Y_filled = np.where(np.isnan(Y), 0.0, Y)

    for t in range(T):
        W_t = W_list[t]
        m_t = W_t.shape[0]
        m_obs[t] = m_t
        if m_t == 0:
            continue

        obs_idx = np.argmax(W_t, axis=1)          # observed series indices
        var_obs = R[obs_idx].astype(float)        # base idiosyncratic variances
        if h_eps is not None:
            var_obs = var_obs * h_eps[t, obs_idx]  # deflate: Var = h^eps * r
        inv_var = 1.0 / var_obs

        WL = W_t @ Lambda_tilde                    # (m_t, 5r)
        y_obs = W_t @ Y_filled[t]                  # (m_t,)
        resid = y_obs - WL @ f_states[t]           # (m_t,)
        d_eps[t] = float(np.sum((resid * resid) * inv_var))

    return d_eps, m_obs


def realized_deflated_d_u(
    f_states: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    r: int,
    h_u: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Factor-side squared Mahalanobis residual from a *sampled* state path,
    optionally deflated by the common stochastic-volatility path.

    This is :func:`em_e_step.compute_d_u` stripped of the posterior-uncertainty
    trace term (``term2``), with the innovation quadratic divided by ``h^u_t``.

    Equivalence to EM (verified in ``test_shared``)
    -----------------------------------------------
    With ``h_u = None`` this returns exactly
    ``compute_d_u(f_states, P_smooth=0, P_lag=0, A, Q, r)``.

    Convention: ``d_u[0] = NaN`` (no f_{-1} in sample), as in EM.
    """
    T = f_states.shape[0]
    d_u = np.full(T, np.nan)
    for t in range(1, T):
        f_t = f_states[t][0:r]
        f_tm1 = f_states[t - 1][0:r]
        innovation = f_t - A @ f_tm1
        term1 = float(innovation @ np.linalg.solve(Q, innovation))
        if h_u is not None:
            term1 = term1 / float(h_u[t])
        d_u[t] = term1
    return d_u


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tail-weight draw (Gibbs step (c)) — Gamma draw instead of the E-step mean
# ─────────────────────────────────────────────────────────────────────────────

def draw_weights(
    d_eps: np.ndarray,
    d_u: np.ndarray,
    m_obs: np.ndarray,
    nu_eps: float,
    nu_u: float,
    r: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    r"""
    Draw the Student-t tail weights from their conjugate Gamma full conditionals.

    Same posteriors as :func:`em_e_step.compute_weights`, but we *draw* instead
    of returning the mean ``alpha/beta``::

        w^eps_t | .  ~  Gamma( (nu_eps + m_t)/2 ,  (nu_eps + check d^eps_t)/2 )
        w^u_t   | .  ~  Gamma( (nu_u   + r  )/2 ,  (nu_u   + check d^u_t  )/2 )

    where the residuals are the volatility-deflated ones of
    :func:`realized_deflated_d_eps` / :func:`realized_deflated_d_u`.

    Boundary ``t = 0`` (factor side): ``d_u[0] = NaN`` (no data), so ``w^u_0`` is
    drawn from its *prior* ``Gamma(nu_u/2, nu_u/2)`` (mean 1), matching the EM
    convention ``w_u[0] = 1`` in expectation.

    Returns ``{"w_eps": (T,), "w_u": (T,)}``.  (Log-weights are not returned: the
    nu-update of the sampler uses the realized ``sum_t log w_t`` directly.)
    """
    T = d_eps.shape[0]

    alpha_eps = (nu_eps + m_obs) / 2.0
    beta_eps = (nu_eps + d_eps) / 2.0
    w_eps = rng.gamma(shape=alpha_eps, scale=1.0 / beta_eps)

    w_u = np.empty(T)
    # t = 0: draw from the prior Gamma(nu_u/2, nu_u/2)  (rate = nu_u/2 -> scale = 2/nu_u)
    w_u[0] = rng.gamma(shape=nu_u / 2.0, scale=2.0 / nu_u)
    alpha_u = (nu_u + r) / 2.0
    beta_u = (nu_u + d_u[1:]) / 2.0
    w_u[1:] = rng.gamma(shape=alpha_u, scale=1.0 / beta_u)

    return {"w_eps": w_eps, "w_u": w_u}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Degrees-of-freedom: log-target and first-order condition (Gibbs step (d))
# ─────────────────────────────────────────────────────────────────────────────

def nu_foc(nu: float, mean_log_w: float, mean_w: float) -> float:
    r"""
    First-order condition g(nu) for the Student-t degrees of freedom.

    Identical to the ``g(nu)`` closure inside :func:`em_m_step.update_nu`:

    .. math::
        g(\nu) = \log\tfrac{\nu}{2} - \psi\!\big(\tfrac{\nu}{2}\big)
                 + 1 + \overline{\log w} - \bar{w}.

    ``g`` is the gradient of :func:`nu_log_target` up to the positive factor
    ``T/2`` (so ``g(nu) = 0`` locates the mode used by the M-step, and bounds the
    Metropolis / griddy-Gibbs draw of the sampler).
    """
    half = 0.5 * nu
    return float(np.log(half) - digamma(half) + 1.0 + mean_log_w - mean_w)


def nu_log_target(
    nu: float,
    sum_log_w: float,
    sum_w: float,
    T: int,
    log_prior=None,
) -> float:
    r"""
    Log full-conditional of nu (up to an additive constant), Gibbs step (d):

    .. math::
        \log p(\nu \mid w) = \mathrm{const}
          + T\big[\tfrac{\nu}{2}\log\tfrac{\nu}{2} - \log\Gamma(\tfrac{\nu}{2})\big]
          + \tfrac{\nu}{2}\,(\textstyle\sum_t \log w_t - \sum_t w_t)
          + \log p(\nu).

    Its derivative equals ``(T/2) * nu_foc(nu, sum_log_w/T, sum_w/T)`` — the link
    verified in ``test_shared`` and used by the sampler's nu draw (the thesis
    log-concavity result makes this a 1-D log-concave target).

    ``log_prior`` is an optional callable ``nu -> log p(nu)``; ``None`` ⇒ flat.
    """
    half = 0.5 * nu
    val = T * (half * np.log(half) - gammaln(half)) + half * (sum_log_w - sum_w)
    if log_prior is not None:
        val += float(log_prior(nu))
    return float(val)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Linear-Gaussian parameter draws (Gibbs step (d), Family A)
# ─────────────────────────────────────────────────────────────────────────────

def draw_A_Q(
    P00: np.ndarray,
    P10: np.ndarray,
    P11: np.ndarray,
    T_eff: int | float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Draw (A, Q) from the flat-prior matrix-normal--inverse-Wishart posterior,
    built on the *same* weighted moments used by :func:`em_m_step.update_A_Q`.

    The moments ``P00, P10, P11`` come from
    :func:`em_m_step.compute_weighted_moments` — in the sampler with the combined
    precision ``g_t = w_t / h_t`` in place of the EM weight ``w_t``.

    Point-estimate seam (verified in ``test_shared``)
    -------------------------------------------------
    The posterior mean of A and the inverse-Wishart scale reproduce the EM
    point estimate exactly::

        A_hat = P10 @ inv(P00)                       == update_A_Q(...)[0]
        S     = P11 - A_hat @ P10.T  (= T_eff * Q)   ;  S / T_eff == update_A_Q(...)[1]

    Draw (thesis eq:param-AQ, flat-prior limit)
    -------------------------------------------
        Q ~ InvWishart(scale = S, df = T_eff)
        vec(A) | Q ~ N( vec(A_hat),  Q (x) inv(P00) )

    i.e. A is matrix-normal with among-row covariance Q and among-column
    (regressor) covariance ``inv(P00)``.
    """
    r = P00.shape[0]

    A_hat = np.linalg.solve(P00, P10.T).T          # A_hat = P10 inv(P00)
    S = P11 - A_hat @ P10.T                         # residual scatter = T_eff * Q_EM
    S = 0.5 * (S + S.T)

    Q_draw = np.atleast_2d(invwishart.rvs(df=T_eff, scale=S, random_state=rng))

    P00_inv = np.linalg.inv(P00)
    P00_inv = 0.5 * (P00_inv + P00_inv.T)
    L_row = np.linalg.cholesky(Q_draw)              # among-row cov = Q
    L_col = np.linalg.cholesky(P00_inv)             # among-col cov = inv(P00)
    Z = rng.standard_normal((r, r))
    A_draw = A_hat + L_row @ Z @ L_col.T

    return A_draw, Q_draw


def composite_regressor(
    f_aug: np.ndarray,
    mm_weights: np.ndarray | None = None,
    r: int | None = None,
) -> np.ndarray:
    r"""
    Mariano-Murasawa composite regressor phi_t for the quarterly loadings.

    For each block column j and each t,

    .. math::
        \phi_t[j] = \sum_{l=0}^{4} c_l \, \tilde f_t[\,l\,r + j\,],

    with ``c = (1/3, 2/3, 1, 2/3, 1/3)``.  This is exactly the ``E_phi`` formed
    inside :func:`em_m_step.update_Lambda` / :func:`em_m_step.update_R` for the
    quarterly rows (here read off a sampled augmented path instead of the
    smoothed mean).

    Parameters
    ----------
    f_aug : (T, 5r)        augmented state path.
    mm_weights : (5,) or None   default :data:`MM_WEIGHTS`.
    r : int or None             factor dimension; inferred as ``dim // 5`` if None.

    Returns
    -------
    phi : (T, r)           composite regressor, column j = block-j composite.
    """
    if mm_weights is None:
        mm_weights = MM_WEIGHTS
    c = np.asarray(mm_weights, dtype=float)
    T, dim = f_aug.shape
    if r is None:
        r = dim // 5
    phi = np.zeros((T, r))
    for l in range(5):
        phi += c[l] * f_aug[:, l * r:(l + 1) * r]
    return phi


def draw_lambda_r_series(
    num: float,
    den: float,
    s_yy: float,
    n_obs: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    r"""
    Draw the (scalar, block-restricted) loading and idiosyncratic variance of one
    series from the flat-prior normal--inverse-gamma posterior, built on the same
    weighted sufficient statistics used by :func:`em_m_step.update_Lambda` /
    :func:`em_m_step.update_R`.

    Sufficient statistics (weighted by w^eps_t, summed over the observed set):
        num   = sum_t w_t y_t  E[phi_t]          (cross moment)   -> num in update_Lambda
        den   = sum_t w_t  E[phi_t^2]            (regressor 2nd moment, = F_i)
        s_yy  = sum_t w_t y_t^2
        n_obs = |T_i|
    where ``phi_t`` is the contemporaneous block-factor (monthly) or the MM
    composite (quarterly, via :func:`composite_regressor`).

    Point-estimate seam (verified in ``test_shared``)
    -------------------------------------------------
        lambda_hat = num / den                       == update_Lambda(...) loading
        ssr        = s_yy - num^2/den  (= sum_t w_t E[resid^2])
        ssr / n_obs                                  == update_R(...) variance

    Draw (thesis eq:param-LR / eq:param-mf-LQ, flat-prior limit)
    -----------------------------------------------------------
        r      ~ InvGamma( n_obs/2 ,  ssr/2 )
        lambda ~ N( lambda_hat ,  r / den )
    """
    lam_hat = num / den
    ssr = s_yy - num * num / den           # completion-of-square residual
    # r ~ InvGamma(shape = n_obs/2, scale = ssr/2)  <=>  1/r ~ Gamma(n_obs/2, scale=2/ssr)
    r_draw = 1.0 / rng.gamma(shape=n_obs / 2.0, scale=2.0 / ssr)
    lam_draw = lam_hat + np.sqrt(r_draw / den) * rng.standard_normal()
    return float(lam_draw), float(r_draw)
