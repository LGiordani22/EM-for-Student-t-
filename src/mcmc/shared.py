"""
src/mcmc/shared.py
==================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: i kernel matematici riusabili dei condizionali completi (stessa formula
dell'EM, con "media/punto → draw").  I principali:

  Pesi delle code (mistura di scala gaussiana; residui di Mahalanobis deflazionati per h):
    d^u_t = (f_t - A f_{t-1})' Q^{-1} (f_t - A f_{t-1}) / h^u_t    [eq:d-u]
    w^u_t | · ~ Gamma((ν_u + r)/2,  (ν_u + d^u_t)/2)               [eq:post-w-u]
    d^ε_t = (y_t - Λ f_t)' R^{-1} (y_t - Λ f_t) / h^ε_t            [eq:d-eps]
    w^ε_t | · ~ Gamma((ν_ε + m_t)/2, (ν_ε + d^ε_t)/2)             [eq:post-w-eps]

  (A, Q)   draw MNIW dai momenti di secondo ordine pesati di  f_t = A f_{t-1} + u_t
  (Λ, R)   draw NIG per serie da  y_t = Λ f_t + ε_t  (regressore MM per le trimestrali)
  ν        target log-concavo g(ν) del condizionale di ν, da  w_t ~ Gamma(ν/2, ν/2)   [eq:g-nu-u]

Tutto additivo: dipende solo da numpy/scipy, non importa l'EM, ed è verificato contro
l'EM in test_shared.py (uguaglianza a macchina per i pezzi deterministici, entro errore
Monte-Carlo per i draw).

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
    trace term (``term2``).  The tail weight ``w^u_t`` is conjugate-Gamma with
    rate driven by ``d^u_t = u_t' Var(u_t|.)^{-1} u_t`` (up to ``w``), and under
    the stochastic-volatility sandwich ``Var(u_t|.) = (1/w) sqrt(H^u_t) Q
    sqrt(H^u_t)`` (eq:param-hu-matrix) this is
    ``d^u_t = u_t' H_t^{-1/2} Q^{-1} H_t^{-1/2} u_t = (H_t^{-1/2} u_t)' Q^{-1}
    (H_t^{-1/2} u_t)``.

    ``h_u`` shape selects the specification:

    * ``None`` — no SV; returns exactly ``compute_d_u(P=0, P_lag=0)`` (EM seam).
    * ``(T,)`` — a **single** common volatility ``h^u_t``: the scalar-common
      restriction ``H^u_t = h^u_t I``, at which the inside placement
      ``Q^{1/2} H Q^{1/2}`` (eq:vol-inside) and the outside one
      ``sqrt(H) Q sqrt(H)`` (eq:vol-outside) coincide, both ``= h^u_t Q``, so the
      specification fork is vacuous; the whole quadratic is divided by that scalar.
    * ``(T, r)`` — **per-factor** volatilities under the Specification II sandwich
      (the model actually estimated, ``eq:vol-outside``): the
      innovation is deflated component-wise by ``sqrt(h^u_{i,t})`` *before* the
      ``Q^{-1}`` quadratic.  With equal columns this reduces to the ``(T,)`` case.

    Convention: ``d_u[0] = NaN`` (no f_{-1} in sample), as in EM.
    """
    T = f_states.shape[0]
    d_u = np.full(T, np.nan)
    per_factor = h_u is not None and np.ndim(h_u) == 2
    for t in range(1, T):
        f_t = f_states[t][0:r]
        f_tm1 = f_states[t - 1][0:r]
        innovation = f_t - A @ f_tm1
        if per_factor:
            innovation = innovation / np.sqrt(h_u[t])       # H_t^{-1/2} u_t
        term1 = float(innovation @ np.linalg.solve(Q, innovation))
        if h_u is not None and not per_factor:
            term1 = term1 / float(h_u[t])                   # scalar-common H = h I
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
    *,
    Psi0: np.ndarray | None = None,
    nu0: float = 0.0,
    A0: np.ndarray | None = None,
    kappa: float = 0.0,
    fix_q_identity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Draw (A, Q) from the matrix-normal--inverse-Wishart posterior, built on the
    *same* weighted moments used by :func:`em_m_step.update_A_Q`.

    The moments ``P00, P10, P11`` come from
    :func:`em_m_step.compute_weighted_moments` — in the sampler with the combined
    precision ``g_t = w_t / h_t`` in place of the EM weight ``w_t``.  This is the
    **scalar-volatility** draw: it applies when ``G^u_t = g^u_t Q^{-1}`` factors
    out of eq:param-A-precision — i.e. when the ``r`` common volatilities coincide
    (``h^u_t I``) or are absent (``h ≡ 1``, the no-SV "current model" cell).  The
    per-factor Spec II case does *not* collapse and is handled by
    :func:`draw_A_Q_perfactor`.

    Point-estimate seam (verified in ``test_shared``)
    -------------------------------------------------
    The posterior mean of A and the inverse-Wishart scale reproduce the EM
    point estimate exactly (flat prior)::

        A_hat = P10 @ inv(P00)                       == update_A_Q(...)[0]
        S     = P11 - A_hat @ P10.T  (= T_eff * Q)   ;  S / T_eff == update_A_Q(...)[1]

    Priors (tab:param-prior-tuning; ``None``/``0`` ⇒ the flat limit above,
    reproduced **bit-for-bit**)
    ---------------------------------------------------------------------
    Natural-conjugate MNIW, ``Q ~ IW(Psi0, nu0)`` (eq:param-Q-post) and, given
    ``Q``, the matrix-Normal ``A | Q ~ MN(A0, Q, Omega0)`` with among-column
    (regressor) covariance ``Omega0 = (kappa I)^{-1}`` — the "diffuse
    matrix-Normal centred at the EM fit with a tiny prior precision ``kappa I``"
    of the tuning table.  (It is the Kronecker special case ``V0 = Q ⊗ Omega0``
    of the general Gaussian prior ``vec(A) ~ N(vec(A0), V0)`` of eq:param-AQ,
    which is what keeps the *joint* conjugacy alive here.)  Posterior::

        P00n  = P00 + kappa I,      P10n = P10 + kappa A0
        A_hat = P10n @ inv(P00n)
        S     = Psi0 + P11 + kappa A0 A0' - A_hat @ P10n.T
        Q         ~ InvWishart(scale = S, df = nu0 + T_eff)
        vec(A)|Q  ~ N( vec(A_hat),  Q (x) inv(P00n) )

    Huang--Wand (eq:param-Q-hw-post) is the same draw with the IW hyperparameters
    swapped for ``(2 nu* diag(1/a), nu* + r - 1)`` — see :func:`hw_iw_prior`.
    """
    r = P00.shape[0]

    if kappa:
        P00 = P00 + kappa * np.eye(r)
        if A0 is not None:
            A0 = np.asarray(A0, float)
            P10 = P10 + kappa * A0
            P11 = P11 + kappa * (A0 @ A0.T)

    A_hat = np.linalg.solve(P00, P10.T).T          # A_hat = P10 inv(P00)

    # Identificazione unit-Q: Q fissata a I, A | Q=I gaussiano (among-row cov = I).
    if fix_q_identity:
        Q_draw = np.eye(r)
    else:
        S = P11 - A_hat @ P10.T                     # residual scatter = T_eff * Q_EM
        if Psi0 is not None:
            S = S + np.asarray(Psi0, float)
        S = 0.5 * (S + S.T)
        Q_draw = np.atleast_2d(invwishart.rvs(df=T_eff + nu0, scale=S, random_state=rng))

    P00_inv = np.linalg.inv(P00)
    P00_inv = 0.5 * (P00_inv + P00_inv.T)
    L_row = np.linalg.cholesky(Q_draw)              # among-row cov = Q (=I if fixed)
    L_col = np.linalg.cholesky(P00_inv)             # among-col cov = inv(P00)
    Z = rng.standard_normal((r, r))
    A_draw = A_hat + L_row @ Z @ L_col.T

    return A_draw, Q_draw


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Huang--Wand hierarchical inverse-Wishart on Q (optional, sec ~20639-20694)
# ─────────────────────────────────────────────────────────────────────────────

def hw_iw_prior(a: np.ndarray, nu_star: float = 2.0) -> tuple[np.ndarray, float]:
    r"""
    The inverse-Wishart hyperparameters ``(Psi0, nu0)`` implied by the
    Huang--Wand hierarchical prior (eq:param-Q-hw-prior) at the current auxiliary
    scales ``a = (a_1, ..., a_r)``::

        Q | a ~ IW( 2 nu*  diag(1/a_1, ..., 1/a_r),   nu* + r - 1 )

    Conditional on ``a`` this is an *ordinary* inverse-Wishart, so the Family~A
    draw (:func:`draw_A_Q` / :func:`draw_A_Q_perfactor`) runs unchanged with
    ``Psi0 -> 2 nu* diag(1/a)``, ``nu0 -> nu* + r - 1`` — the Gibbs step of
    eq:param-Q-hw-post.  Marginalising ``a`` out gives a **half-t_{nu*}** prior on
    each innovation standard deviation ``sqrt(Q_jj)`` (scale ``A_j``) and a
    marginal correlation law ``propto (1-rho^2)^{(nu*-2)/2}`` — **uniform** at
    ``nu* = 2``: the variance tails and the correlation shape are controlled
    separately, which the plain IW's single ``nu0`` cannot do.
    """
    a = np.asarray(a, float).ravel()
    r = a.size
    Psi0 = 2.0 * float(nu_star) * np.diag(1.0 / a)
    nu0 = float(nu_star) + r - 1.0
    return Psi0, nu0


def draw_hw_aux(
    Q: np.ndarray,
    rng: np.random.Generator,
    *,
    nu_star: float = 2.0,
    A_scales: float | np.ndarray = 1e5,
) -> np.ndarray:
    r"""
    Draw the ``r`` Huang--Wand auxiliary scales from their closed inverse-gamma
    full conditionals (eq:param-Q-hw-aux) — the ``r`` scalar draws appended to
    Family~A when the hierarchical prior is switched on::

        a_j | Q, .  ~  IG( (nu* + r)/2,   nu* (Q^{-1})_jj + 1/A_j^2 ),   j = 1..r

    Parameters
    ----------
    Q : (r, r)                 the current factor-innovation covariance draw.
    nu_star : float            ``nu*`` — the half-t degrees of freedom on each
                               ``sqrt(Q_jj)``; ``2`` gives near-half-Cauchy
                               margins *and* uniform marginal correlations.
    A_scales : float or (r,)   the half-t scale(s) ``A_j``, "set large relative to
                               the plausible innovation scale of the standardised
                               panel" (``A ~ 10`` mildly informative, ``~1e5``
                               near-flat).

    Returns ``a`` (r,).  Prior draw (no data): ``a_j ~ IG(1/2, 1/A_j^2)``.
    """
    Q = np.asarray(Q, float)
    r = Q.shape[0]
    Qinv_diag = np.diag(np.linalg.inv(Q))
    A_j = np.broadcast_to(np.asarray(A_scales, float), (r,))
    shape = 0.5 * (float(nu_star) + r)
    rate = float(nu_star) * Qinv_diag + 1.0 / (A_j ** 2)      # IG scale parameter
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / rate, size=r)


def draw_A_Q_perfactor(
    f_head: np.ndarray,
    h_u: np.ndarray,
    w_u: np.ndarray,
    A_cur: np.ndarray,
    rng: np.random.Generator,
    *,
    Psi0: np.ndarray | None = None,
    nu0: float = 0.0,
    A0: np.ndarray | None = None,
    V0_inv: np.ndarray | None = None,
    fix_q_identity: bool = False,
    _return_moments: bool = False,
):
    r"""
    Draw (A, Q) under Specification II — *per-factor* common stochastic
    volatility — from the two-step conditional Gibbs move of the thesis
    (\S ``Sampling the Parameters (Gibbs Step (d))``, Family~A).

    Under Spec II the VAR(1) innovation covariance is the sandwich
    ``Var(u_t|.) = (1/w^u_t) sqrt(H^u_t) Q sqrt(H^u_t)`` (eq:param-hu-matrix,
    ``H^u_t = diag(h_u[t, :])``).  The per-period whitening
    ``C_t = sqrt(w^u_t) H_t^{-1/2}`` does **not** commute with ``A``, so the
    scalar-weighted moments of the first stage are no longer sufficient and the
    closed matrix-normal--inverse-Wishart draw does not survive.  We therefore
    draw the block as a two-step Gibbs move:

    * **Q | A_cur** (eq:param-Q-post): with the whitened residuals
      ``check_u_t = C_t (f_t - A_cur f_{t-1}) ~ N(0, Q)``,
      ``Q ~ IW(Psi0 + sum_t check_u_t check_u_t', nu0 + T_eff)``.
    * **vec(A) | Q** (eq:param-A-vec-model, eq:param-AQ, eq:param-A-precision):
      a genuine ``r^2``-dimensional Gaussian with precision
      ``V_n^{-1} = V0_inv + sum_t (f_{t-1} f_{t-1}') (x) G^u_t`` and
      ``G^u_t = C_t Q^{-1} C_t = w^u_t H_t^{-1/2} Q^{-1} H_t^{-1/2}`` — a *sum of
      Kronecker products*, one per period, that does not collapse to a single
      Kronecker because ``G^u_t`` varies with ``t``.

    ``T_eff = T - 1``: the innovation ``u_t = f_t - A f_{t-1}`` exists only for
    ``t = 1..T-1`` (the effective-count convention of the thesis; the factor side
    conditions on ``f_1``).

    Collapse seam (verified in ``test_shared``): when ``h_u ≡ 1`` and
    ``A_cur = A_hat`` the Q-scale reduces to the scalar residual scatter
    ``P11 - A_hat P10'`` and the A-posterior mean to ``vec(A_hat)``,
    ``A_hat = P10 P00^{-1}`` with the scalar-weighted moments ``P.. = sum_t w_t ..``
    — i.e. the per-factor draw reduces to the common-scalar
    matrix-normal--inverse-Wishart draw of :func:`draw_A_Q`.

    Parameters
    ----------
    f_head : (T, r)     the head factor path (first ``r`` of the augmented state).
    h_u : (T, r)        per-factor common volatilities ``h^u_{i,t}`` (``≡1`` off).
    w_u : (T,)          factor tail weights ``w^u_t``.
    A_cur : (r, r)      current transition (the Q|A step conditions on it).
    Psi0, nu0 : IW prior scale / dof (``None``/``0`` ⇒ flat).
    A0, V0_inv : Gaussian prior on ``vec(A)`` (``None`` ⇒ flat).
    _return_moments : if True, also return the posterior moments (for testing).
    """
    f_head = np.asarray(f_head, float)
    h_u = np.asarray(h_u, float)
    w_u = np.asarray(w_u, float)
    A_cur = np.asarray(A_cur, float)
    T, r = f_head.shape

    ft = f_head[1:]                       # (T-1, r)  f_t
    ftm1 = f_head[:-1]                     # (T-1, r)  f_{t-1}
    c = np.sqrt(w_u[1:])[:, None] * h_u[1:] ** -0.5   # (T-1, r): diag of C_t
    T_eff = T - 1

    # ── Q | A_cur  (eq:param-Q-post) ────────────────────────────────────────
    # Identificazione unit-Q: Q e' FISSATA a I per tutta la catena (non campionata),
    # la scala del fattore e' portata da Lambda (e ancorata da mu_h=0).  Il draw di
    # vec(A) qui sotto e' esatto: e' gia' condizionato su Q via Qinv, che qui e' I.
    if fix_q_identity:
        Q_draw = np.eye(r)
        Qinv = np.eye(r)
        S, df = None, None
    else:
        U = ft - ftm1 @ A_cur.T               # (T-1, r) innovations u_t
        Uc = c * U                            # (T-1, r) whitened residuals check_u_t
        S = Uc.T @ Uc                         # sum_t check_u_t check_u_t'
        if Psi0 is not None:
            S = S + np.asarray(Psi0, float)
        S = 0.5 * (S + S.T)
        df = float(T_eff) + float(nu0)
        Q_draw = np.atleast_2d(invwishart.rvs(df=df, scale=S, random_state=rng))
        Qinv = np.linalg.inv(Q_draw)
        Qinv = 0.5 * (Qinv + Qinv.T)

    # ── vec(A) | Q  (eq:param-AQ, eq:param-A-precision) ──────────────────────
    P = np.zeros((r * r, r * r))          # V_n^{-1}
    rhs = np.zeros(r * r)                 # V0_inv vec(A0) + sum_t X_t' Qinv check_f_t
    for i in range(T_eff):
        ci = c[i]                                     # diag of C_t (r,)
        G = (ci[:, None] * Qinv) * ci[None, :]        # G^u_t = C_t Qinv C_t
        P += np.kron(np.outer(ftm1[i], ftm1[i]), G)
        fcheck = ci * ft[i]                           # C_t f_t = check_f_t
        rhs += np.kron(ftm1[i], ci * (Qinv @ fcheck)) # X_t' Qinv check_f_t
    if V0_inv is not None:
        P = P + np.asarray(V0_inv, float)
        if A0 is not None:
            rhs = rhs + np.asarray(V0_inv, float) @ np.asarray(A0, float).ravel(order="F")
    P = 0.5 * (P + P.T)
    V_n = np.linalg.inv(P)
    V_n = 0.5 * (V_n + V_n.T)
    m_n = V_n @ rhs                                    # vec(A) posterior mean

    L = np.linalg.cholesky(V_n)
    vecA = m_n + L @ rng.standard_normal(r * r)
    A_draw = vecA.reshape((r, r), order="F")

    if _return_moments:
        return A_draw, Q_draw, {
            "Q_scale": S, "Q_df": df,
            "A_mean": m_n.reshape((r, r), order="F"),
            "A_prec": P,
        }
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
    *,
    a0: float = 0.0,
    b0: float = 0.0,
    m0: float = 0.0,
    M0_inv: float = 0.0,
) -> tuple[float, float]:
    r"""
    Draw the (scalar, block-restricted) loading and idiosyncratic variance of one
    series from the conjugate **normal--inverse-gamma** posterior (thesis
    eq:param-LR-post / eq:param-LR-hyper), built on the same weighted sufficient
    statistics used by :func:`em_m_step.update_Lambda` / :func:`em_m_step.update_R`.

    Because the block restriction leaves each series loading on a *single* factor,
    ``L_i`` is scalar and the whole NIG is scalar.

    Sufficient statistics (weighted by ``g^eps_t = w^eps_t/h^eps_{i,t}``, summed
    over the observed set):
        num   = sum_t g_t y_t phi_t              (cross moment,  = c_i)
        den   = sum_t g_t phi_t^2                (regressor 2nd moment,  = F_i)
        s_yy  = sum_t g_t y_t^2
        n_obs = |T_i|
    where ``phi_t`` is the contemporaneous block-factor (monthly) or the MM
    composite (quarterly, via :func:`composite_regressor`).

    Prior (eq:param-LR-post): ``r_i ~ IG(a0, b0)``, ``L_i | r_i ~ N(m0, r_i M0)``
    (pass the prior precision ``M0_inv = 1/M0``).  Posterior:
        Fbar = den + M0_inv,   cbar = num + M0_inv m0
        b_n  = b0 + 1/2 ( s_yy + M0_inv m0^2 - cbar^2/Fbar )
        r_i     ~ IG( a0 + n_obs/2, b_n )
        L_i|r_i ~ N( cbar/Fbar, r_i/Fbar )

    Flat-prior limit ``a0=b0=m0=M0_inv=0`` reproduces the weighted-LS draw
    ``r ~ IG(n_obs/2, ssr/2)``, ``L ~ N(num/den, r/den)`` **bit-for-bit** (the EM
    seam of ``test_shared``).
    """
    Fbar = den + M0_inv                              # F_i + M0^{-1}
    cbar = num + M0_inv * m0                         # c_i + M0^{-1} m0
    lam_hat = cbar / Fbar
    b_n = b0 + 0.5 * (s_yy + M0_inv * m0 * m0 - cbar * cbar / Fbar)
    shape = a0 + n_obs / 2.0
    # r ~ InvGamma(shape, b_n)  <=>  1/r ~ Gamma(shape, scale = 1/b_n)
    r_draw = 1.0 / rng.gamma(shape=shape, scale=1.0 / b_n)
    lam_draw = lam_hat + np.sqrt(r_draw / Fbar) * rng.standard_normal()
    return float(lam_draw), float(r_draw)
