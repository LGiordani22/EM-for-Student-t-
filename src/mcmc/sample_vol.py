"""
src/mcmc/sample_vol.py
======================

Gibbs step (b), **base case** (KSC, no leverage): sample the M+1 stochastic
log-volatility paths and their AR(1) parameters, given the states, the tail
weights and the parameters.

Theory: ``docs/EM_for_student_t.tex`` §"Sampling the Volatility Paths: Base
Case (Gibbs Step (b))".  The construction is Kim-Shephard-Chib (1998):

  1. Reconstruct the residual that each volatility process scales
     (common factor: ``u_t = f_t - A f_{t-1}``; series i: ``eps_{i,t}=y_{i,t}-(Lf_t)_i``).
  2. Whiten by base scale and tail weight so each measurement is N(0, h_t):
       idiosyncratic:  e_{i,t} = sqrt(w^eps_t / r_i) * eps_{i,t}
       common factor:  tilde_u_t = sqrt(w^u_t) * Q^{-1/2} u_t  -> r scalar
                       measurements of the SAME state log h^u_t.
  3. Log-square with offset: ``y* = log(e^2 + c)`` -> linear in log h_t with a
     log chi^2_1 error.
  4. Approximate that error by the 7-component KSC mixture (constants.KSC7) via
     a component indicator s; conditional on s the system is linear-Gaussian.
  5. Sub-sweep: (i) draw indicators s (multinomial per measurement), (ii) draw
     the whole log-vol path by the scalar FFBS recursion (the dimension-1
     instance of the state sampler's backward pass).
  6. Family B: draw the AR(1) parameters (mu, phi, sigma_eta^2), |phi|<1, by a
     conjugate regression on the sampled log-vol path.

Config-aware: ``r``, ``M`` and the per-series structure come from the inputs;
nothing is hard-coded.  ``rho == 0`` throughout (leverage is Passi 3-4).
"""

from __future__ import annotations

import numpy as np

from mcmc.constants import KSC7


# ─────────────────────────────────────────────────────────────────────────────
# Scalar AR(1) FFBS (the dimension-1 instance of eq:ffbs-backward)
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_ar1_ffbs(
    y_eff: np.ndarray,
    V_eff: np.ndarray,
    mask: np.ndarray,
    mu: float,
    phi: float,
    sigma2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Forward-filter / backward-sample a scalar AR(1) state ``x_t = log h_t`` with

        x_t = mu + phi (x_{t-1} - mu) + N(0, sigma2),     x_0 ~ N(mu, sigma2/(1-phi^2))
        y_eff_t = x_t + N(0, V_eff_t)   (only where ``mask_t``; else no update)

    Returns one draw of ``x_{0:T-1}``.  This is exactly the FFBS backward pass of
    the state block (eq:ffbs-backward) at state dimension 1.
    """
    T = mask.shape[0]
    a = np.zeros(T)          # filtered mean
    P = np.zeros(T)          # filtered variance
    stat_var = sigma2 / (1.0 - phi * phi)
    for t in range(T):
        if t == 0:
            a_pred, P_pred = mu, stat_var
        else:
            a_pred = mu + phi * (a[t - 1] - mu)
            P_pred = phi * phi * P[t - 1] + sigma2
        if mask[t]:
            S = P_pred + V_eff[t]
            K = P_pred / S
            a[t] = a_pred + K * (y_eff[t] - a_pred)
            P[t] = (1.0 - K) * P_pred
        else:
            a[t] = a_pred
            P[t] = P_pred

    x = np.zeros(T)
    x[T - 1] = a[T - 1] + np.sqrt(max(P[T - 1], 0.0)) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        P_pred_next = phi * phi * P[t] + sigma2
        J = phi * P[t] / P_pred_next
        m = a[t] + J * (x[t + 1] - (mu + phi * (a[t] - mu)))
        V = P[t] * (1.0 - J * phi)
        x[t] = m + np.sqrt(max(V, 0.0)) * rng.standard_normal()
    return x


# ─────────────────────────────────────────────────────────────────────────────
# One log-vol process: indicator draw + path draw
# ─────────────────────────────────────────────────────────────────────────────

def sample_log_vol_process(
    ys: np.ndarray,
    tidx: np.ndarray,
    T: int,
    logh_cur: np.ndarray,
    mu: float,
    phi: float,
    sigma2: float,
    rng: np.random.Generator,
    ksc: dict = KSC7,
) -> np.ndarray:
    r"""
    One KSC sub-sweep for a single log-volatility process.

    Parameters
    ----------
    ys : (N,)             all log-square measurements ``y*`` for this process.
    tidx : (N,) int       the time index t in ``0..T-1`` of each measurement
                          (the common factor contributes r measurements per t;
                          an idiosyncratic series contributes one per observed t).
    logh_cur : (T,)       current log-vol path (for the indicator conditional).
    mu, phi, sigma2 : current AR(1) parameters.

    Returns the newly drawn ``log h_{0:T-1}``.
    """
    q = ksc["q"]; m = ksc["m"]; v2 = ksc["v2"]
    log_q = np.log(q)
    half_log_v2 = 0.5 * np.log(v2)

    # ── (i) indicators: s_n ~ q_j N(y*_n; logh_t(n) + m_j, v2_j) ──────────────
    if ys.size == 0:
        # No measurements (e.g. fully unobserved): path is a prior AR(1) draw.
        return _scalar_ar1_ffbs(np.zeros(T), np.zeros(T), np.zeros(T, bool),
                                mu, phi, sigma2, rng)
    d = ys[:, None] - logh_cur[tidx][:, None] - m[None, :]      # (N, 7)
    logp = log_q[None, :] - half_log_v2[None, :] - 0.5 * d * d / v2[None, :]
    # Gumbel-max categorical sampling.
    g = rng.gumbel(size=logp.shape)
    s = np.argmax(logp + g, axis=1)                            # (N,)

    m_sel = m[s]; v2_sel = v2[s]
    inv_v2 = 1.0 / v2_sel

    # ── combine measurements per time into one effective Gaussian obs ─────────
    prec = np.zeros(T)
    rhs = np.zeros(T)
    np.add.at(prec, tidx, inv_v2)
    np.add.at(rhs, tidx, (ys - m_sel) * inv_v2)
    mask = prec > 0
    V_eff = np.zeros(T)
    y_eff = np.zeros(T)
    V_eff[mask] = 1.0 / prec[mask]
    y_eff[mask] = rhs[mask] / prec[mask]

    # ── (ii) draw the path ────────────────────────────────────────────────────
    return _scalar_ar1_ffbs(y_eff, V_eff, mask, mu, phi, sigma2, rng)


# ─────────────────────────────────────────────────────────────────────────────
# Family B: AR(1) parameters of one log-vol process
# ─────────────────────────────────────────────────────────────────────────────

def draw_ar1_params(
    logh: np.ndarray,
    rng: np.random.Generator,
    *,
    fix_mu0: bool = True,
    a_sig: float = 2.0,
    b_sig: float = 0.05,
    max_tries: int = 50,
) -> tuple[float, float, float]:
    r"""
    Draw the AR(1) parameters of a log-volatility process from a conjugate
    regression on the sampled path, with a weak inverse-gamma prior on
    ``sigma2`` and stationarity ``|phi|<1`` enforced by rejection.

    Identification convention (``fix_mu0=True``, the default and adopted
    convention)
    -----------------------------------------------------------------------
    The unconditional level ``mu`` of each log-volatility process is **not
    identified separately** from the static scale (Q for the common factor, r_i
    for series i): only the products ``h^u_t Q`` and ``h^eps_{i,t} r_i`` enter
    the likelihood, so a constant shift in ``log h`` trades off against a rescale
    of Q / r_i.  We therefore **fix ``mu = 0`` for every process**, pinning each
    ``h_t`` as a multiplier around 1 and letting all of the base scale live in Q
    and R.  The model becomes ``x_t = phi x_{t-1} + N(0,sigma2)`` (no intercept,
    stationary ``x_0 ~ N(0, sigma2/(1-phi^2))``), and only ``(phi, sigma2)`` are
    drawn; ``mu`` is returned as exactly 0.

    With ``fix_mu0=False`` the full ``x_t = c + phi x_{t-1} + eta`` is drawn and
    ``mu = c/(1-phi)`` returned (the unidentified parametrisation, kept only for
    diagnostics / ablation).

    Prior on ``sigma2``
    -------------------
    ``sigma2 ~ InvGamma(a_sig, b_sig)``.  The default ``IG(2.0, 0.05)`` is
    *weakly informative*: prior mean ``b/(a-1) = 0.05`` (a plausible log-vol
    innovation variance, not pulling toward zero) with a heavy tail (``a=2`` =>
    infinite prior variance, ~2 prior pseudo-observations against T-1 data).
    This replaces the tighter KSC-style ``IG(2.5, 0.025)`` (prior mean 0.0167):
    the likelihood for ``sigma2`` is intrinsically weak — ``log h_t`` is observed
    only through the very noisy KSC log-square measurement (error variance
    ``pi^2/2 ≈ 4.93`` per measurement) — so an over-tight prior would visibly
    bias the estimate downward.  Both ``a_sig, b_sig`` are exposed so the prior
    is parametrisable; the same prior is used for all M+1 processes.
    """
    x = np.asarray(logh, float)
    y = x[1:]
    n = y.size

    if fix_mu0:
        xp = x[:-1]
        xtx = float(xp @ xp) + 1e-12
        phi_hat = float((xp @ y) / xtx)
        resid = y - phi_hat * xp
        ssr = float(resid @ resid)
        shape = a_sig + n / 2.0
        for _ in range(max_tries):
            scale = b_sig + 0.5 * ssr
            sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)
            phi = phi_hat + np.sqrt(sigma2 / xtx) * rng.standard_normal()
            if abs(phi) < 0.999:
                return 0.0, float(phi), float(sigma2)
        return 0.0, float(np.clip(phi_hat, -0.98, 0.98)), max(float(sigma2), 1e-6)

    # ── unidentified parametrisation with free intercept (diagnostics only) ──
    X = np.column_stack([np.ones(n), x[:-1]])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(2))
    beta_hat = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_hat
    ssr = float(resid @ resid)
    shape = a_sig + n / 2.0
    for _ in range(max_tries):
        scale = b_sig + 0.5 * ssr
        sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)
        L = np.linalg.cholesky(sigma2 * XtX_inv)
        beta = beta_hat + L @ rng.standard_normal(2)
        c, phi = float(beta[0]), float(beta[1])
        if abs(phi) < 0.999:
            return c / (1.0 - phi), phi, sigma2
    phi = float(np.clip(beta_hat[1], -0.98, 0.98))
    return float(beta_hat[0] / (1.0 - phi)), phi, max(sigma2, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# The full step (b): all M+1 processes
# ─────────────────────────────────────────────────────────────────────────────

def _inv_sqrt_spd(Q: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root ``Q^{-1/2}`` of an SPD matrix."""
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    vals = np.clip(vals, 1e-12, None)
    return (vecs / np.sqrt(vals)) @ vecs.T


def sample_volatility_block(
    Y: np.ndarray,
    f_aug: np.ndarray,
    theta: dict,
    w_u: np.ndarray,
    w_eps: np.ndarray,
    logh_u: np.ndarray,
    logh_eps: np.ndarray,
    sv_u: tuple[float, float, float],
    sv_eps: np.ndarray,
    rng: np.random.Generator,
    *,
    offset: float = 1e-6,
    fix_mu0: bool = True,
    prior_a: float = 2.0,
    prior_b: float = 0.05,
    ksc: dict = KSC7,
) -> dict:
    r"""
    Sample all M+1 log-volatility paths and their AR(1) parameters (no leverage).

    Parameters
    ----------
    f_aug : (T, 5r)       sampled augmented state path (step (a)).
    theta : dict          current parameters (A, Q, Lambda, R).
    w_u, w_eps : (T,)     current tail weights.
    logh_u : (T,)         current common log-vol path.
    logh_eps : (T, M)     current idiosyncratic log-vol paths.
    sv_u : (mu, phi, sigma2)            current common AR(1) params.
    sv_eps : (M, 3)       current idiosyncratic AR(1) params.
    offset : float        log-square offset c.

    Returns dict: ``h_u`` (T,), ``h_eps`` (T, M), ``logh_u``, ``logh_eps``,
    ``sv_u`` (3,), ``sv_eps`` (M, 3).
    """
    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    T, M = Y.shape
    r = A.shape[0]
    F = f_aug[:, :r]

    # ── Common factor: r whitened measurements per t = 1..T-1 ────────────────
    Qinv_half = _inv_sqrt_spd(Q)
    u = F[1:] - F[:-1] @ A.T                       # (T-1, r), u_t = f_t - A f_{t-1}
    tilde_u = (np.sqrt(w_u[1:])[:, None]) * (u @ Qinv_half.T)   # (T-1, r), N(0, h^u_t I)
    ys_u = np.log(tilde_u.reshape(-1) ** 2 + offset)           # (r(T-1),)
    tidx_u = np.repeat(np.arange(1, T), r)                      # times 1..T-1, r each
    mu_u, phi_u, s2_u = sv_u
    logh_u_new = sample_log_vol_process(ys_u, tidx_u, T, logh_u, mu_u, phi_u, s2_u, rng, ksc)
    sv_u_new = np.array(draw_ar1_params(logh_u_new, rng, fix_mu0=fix_mu0,
                                        a_sig=prior_a, b_sig=prior_b))

    # ── Idiosyncratic: one whitened measurement per observed (i, t) ───────────
    signal = F @ Lambda.T                          # (T, M), (Lambda f_t)_i
    logh_eps_new = np.zeros((T, M))
    sv_eps_new = np.zeros((M, 3))
    for i in range(M):
        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        eps = Y[obs_t, i] - signal[obs_t, i]
        e = np.sqrt(w_eps[obs_t] / R[i]) * eps     # N(0, h^eps_{i,t})
        ys_i = np.log(e ** 2 + offset)
        mu_i, phi_i, s2_i = sv_eps[i]
        logh_eps_new[:, i] = sample_log_vol_process(
            ys_i, obs_t, T, logh_eps[:, i], mu_i, phi_i, s2_i, rng, ksc)
        sv_eps_new[i] = draw_ar1_params(logh_eps_new[:, i], rng, fix_mu0=fix_mu0,
                                        a_sig=prior_a, b_sig=prior_b)

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
    }
