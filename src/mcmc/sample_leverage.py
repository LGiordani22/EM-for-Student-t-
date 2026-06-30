"""
src/mcmc/sample_leverage.py
===========================

Gibbs step (b) + Family C, **Branch A** (contemporaneous timing, Metropolis):
the first sampler with *leverage* — the asymmetric coupling between the level
shock and the log-volatility innovation that gives the predictive density its
skewness.  This is also the first non-pure-Gibbs block: the leverage drift makes
the log-volatility transition non-linear-Gaussian, so the KSC mixture + FFBS of
the base case (Passo 2) is replaced by a single-move Metropolis-Hastings update.

Theory: ``docs/EM_for_student_t.tex``, "Two Routes for the Leverage", Branch~A
(``subsec:lev-branch-A``); the leverage conditional law ``eta_t | z_t``
(``eq:lev-cond-scalar`` / ``eq:lev-cond-common``); the single-move target
``eq:lev-mh-target``; Family~C ``eq:param-rho-cond`` with regressor
``k_t = sigma_eta z_t`` (contemporaneous).

Branch A is implementable *from first principles* — no Omori constants (those are
Branch~B / Passo~4).  The Metropolis target is self-contained: model parameters
+ data.

Identification convention ``mu = 0`` (adopted in Passo 2) is in force here too.

Nesting: at ``rho = 0`` every leverage term vanishes (drift ``rho*sigma*z = 0``,
variance ``sigma^2(1-rho^2) = sigma^2``), so the contemporaneous sampler reduces
to the base KSC block of Passo 2 — but we still route ``rho = 0`` runs through the
base block (``sample_vol.sample_volatility_block``) to keep that path bit-identical.

Config-aware: ``r``, ``M`` and the per-series structure come from the inputs.
The common factor carries an ``r``-vector leverage ``rho`` (constraint
``rho'rho < 1``); each idiosyncratic series carries a scalar ``rho_i``
(``|rho_i| < 1``).
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# (b) single-move Metropolis on a scalar log-vol path with contemporaneous leverage
# ─────────────────────────────────────────────────────────────────────────────

def _lev_path_mh(
    logh: np.ndarray,
    S: np.ndarray,
    kdim: np.ndarray,
    gcoef: np.ndarray,
    has_obs: np.ndarray,
    phi: float,
    sigma2: float,
    rho2: float,
    prop_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    r"""
    One single-move random-walk Metropolis sweep over a scalar log-vol path
    ``x_t = log h_t`` under contemporaneous leverage (``mu = 0``).

    The full conditional of ``x_t`` collects (eq:lev-mh-target): the level
    likelihood ``h_t^{-k/2} exp(-S_t/(2 h_t))``; the transition *into* ``t``
    carrying the state-dependent leverage drift
    ``sigma * gcoef_t * exp(-x_t/2)`` with conditional variance
    ``sigma^2(1-rho^2)``; and the transition *out of* ``t``.

    Parameters
    ----------
    logh : (T,)           current path (updated in place on a copy).
    S : (T,)              sum of squares of the (whitened) level residual at t
                          (``||tilde_u_t||^2`` common, ``e_t^2`` idiosyncratic);
                          used only where ``has_obs[t]``.
    kdim : (T,) int       level dimension at t (``r`` common, ``1`` idiosyncratic).
    gcoef : (T,)          ``rho . residual_t`` (scalar): the drift coefficient, so
                          the drift is ``sigma * gcoef_t * exp(-x_t/2)``.
    has_obs : (T,) bool   whether a level residual (hence leverage) is present at t.
    phi, sigma2 : AR(1) persistence and innovation variance.
    rho2 : float          ``rho'rho`` (common) or ``rho^2`` (idiosyncratic).
    prop_sd : float       random-walk proposal standard deviation.

    Returns ``(logh_new, n_accept, n_propose)``.
    """
    T = logh.shape[0]
    x = logh.copy()
    sig = np.sqrt(sigma2)
    var_lev = sigma2 * (1.0 - rho2)
    stat_var = sigma2 / (1.0 - phi * phi)

    def _terms(t: int, xt: float) -> float:
        val = 0.0
        if has_obs[t]:
            val += -0.5 * kdim[t] * xt - 0.5 * S[t] * np.exp(-xt)
        # transition into t (or stationary prior at t = 0)
        if t >= 1:
            if has_obs[t]:
                m = phi * x[t - 1] + sig * gcoef[t] * np.exp(-0.5 * xt)
                vt = var_lev
            else:
                m = phi * x[t - 1]
                vt = sigma2
            val += -0.5 * (xt - m) ** 2 / vt
        else:
            val += -0.5 * xt * xt / stat_var
        # transition out of t
        if t <= T - 2:
            if has_obs[t + 1]:
                m1 = phi * xt + sig * gcoef[t + 1] * np.exp(-0.5 * x[t + 1])
                vt1 = var_lev
            else:
                m1 = phi * xt
                vt1 = sigma2
            val += -0.5 * (x[t + 1] - m1) ** 2 / vt1
        return val

    n_acc = 0
    for t in range(T):
        xt = x[t]
        xs = xt + prop_sd * rng.standard_normal()
        if np.log(rng.random()) < _terms(t, xs) - _terms(t, xt):
            x[t] = xs
            n_acc += 1
    return x, n_acc, T


# ─────────────────────────────────────────────────────────────────────────────
# Family B under leverage: (phi, sigma^2) — leverage-aware draws
# ─────────────────────────────────────────────────────────────────────────────

def _draw_phi_lev(x, zeta, has_obs, sigma2, rho2, rng):
    r"""Draw ``phi`` from its Gaussian full conditional under leverage.

    ``x_t = phi x_{t-1} + sigma*zeta_t + N(0, sigma^2(1-rho^2))`` on leverage
    transitions (``zeta_t = rho . z_t`` known given the path), and
    ``x_t = phi x_{t-1} + N(0, sigma^2)`` on non-leverage transitions — a
    heteroskedastic linear regression of ``x_t`` on ``x_{t-1}``.
    """
    T = x.shape[0]
    sig = np.sqrt(sigma2)
    var_lev = sigma2 * (1.0 - rho2)
    num = 0.0
    den = 0.0
    for t in range(1, T):
        vt = var_lev if has_obs[t] else sigma2
        yt = x[t] - (sig * zeta[t] if has_obs[t] else 0.0)
        num += x[t - 1] * yt / vt
        den += x[t - 1] * x[t - 1] / vt
    if den <= 0:
        return 0.0
    phi_hat = num / den
    post_var = 1.0 / den
    for _ in range(50):
        phi = phi_hat + np.sqrt(post_var) * rng.standard_normal()
        if abs(phi) < 0.999:
            return float(phi)
    return float(np.clip(phi_hat, -0.98, 0.98))


def _draw_sigma2_lev(x, zeta, has_obs, phi, rho2, sigma2_cur, a_sig, b_sig, prop_sd, rng):
    r"""RW-Metropolis draw of ``sigma^2`` (on ``log sigma^2``) under leverage.

    Target: ``IG(a_sig,b_sig)`` prior times the leverage likelihood, where on
    leverage transitions the residual is ``x_t - phi x_{t-1} - sqrt(v) zeta_t``
    with variance ``v(1-rho^2)``, and on non-leverage transitions
    ``x_t - phi x_{t-1}`` with variance ``v``.
    """
    T = x.shape[0]
    eta = x[1:] - phi * x[:-1]                # (T-1,)
    obs = has_obs[1:]
    zt = zeta[1:]

    def _logp(v):
        if v <= 0:
            return -np.inf
        lp = -(a_sig + 1.0) * np.log(v) - b_sig / v   # IG kernel
        sv = np.sqrt(v)
        # leverage transitions
        rl = eta[obs] - sv * zt[obs]
        vlev = v * (1.0 - rho2)
        lp += -0.5 * np.sum(rl * rl) / vlev - 0.5 * np.sum(obs) * np.log(vlev)
        # non-leverage transitions
        rn = eta[~obs]
        lp += -0.5 * np.sum(rn * rn) / v - 0.5 * np.sum(~obs) * np.log(v)
        return lp

    log_v = np.log(sigma2_cur)
    log_vs = log_v + prop_sd * rng.standard_normal()
    # proposal on log v: add the log-Jacobian (target in v -> +log v on each side)
    cur = _logp(sigma2_cur) + log_v
    new = _logp(np.exp(log_vs)) + log_vs
    if np.log(rng.random()) < new - cur:
        return float(np.exp(log_vs)), 1
    return float(sigma2_cur), 0


# ─────────────────────────────────────────────────────────────────────────────
# Family C: leverage correlation rho (scalar / vector) — Metropolis
# ─────────────────────────────────────────────────────────────────────────────

def _rho_logpost_scalar(rho, eta, k, n_lev, sigma2):
    if abs(rho) >= 1.0:
        return -np.inf
    om = 1.0 - rho * rho
    res = eta - rho * k
    return -0.5 * n_lev * np.log(om) - 0.5 * np.sum(res * res) / (sigma2 * om)


def draw_rho_scalar(rho_cur, eta, k, sigma2, prop_sd, rng):
    """RW-Metropolis for a scalar leverage ``rho`` (idiosyncratic), ``|rho|<1``.

    ``eta_t`` realised AR(1) innovations, ``k_t = sigma * z_t`` the contemporaneous
    leverage regressor, both over the leverage-bearing transitions.
    """
    n_lev = eta.shape[0]
    rs = rho_cur + prop_sd * rng.standard_normal()
    cur = _rho_logpost_scalar(rho_cur, eta, k, n_lev, sigma2)
    new = _rho_logpost_scalar(rs, eta, k, n_lev, sigma2)
    if np.log(rng.random()) < new - cur:
        return float(rs), 1
    return float(rho_cur), 0


def _rho_logpost_vec(rho, eta, K, n_lev, sigma2):
    rr = float(rho @ rho)
    if rr >= 1.0:
        return -np.inf
    om = 1.0 - rr
    res = eta - K @ rho
    return -0.5 * n_lev * np.log(om) - 0.5 * np.sum(res * res) / (sigma2 * om)


def draw_rho_vec(rho_cur, eta, K, sigma2, prop_sd, rng):
    """RW-Metropolis for the common-factor leverage vector ``rho`` (``rho'rho<1``).

    ``K`` is ``(n_lev, r)`` with row ``k_t = sigma * z^u_t``.
    """
    n_lev = eta.shape[0]
    rs = rho_cur + prop_sd * rng.standard_normal(rho_cur.shape[0])
    cur = _rho_logpost_vec(rho_cur, eta, K, n_lev, sigma2)
    new = _rho_logpost_vec(rs, eta, K, n_lev, sigma2)
    if np.log(rng.random()) < new - cur:
        return rs, 1
    return rho_cur, 0


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: step (b) + Family B + Family C for all M+1 processes (Branch A)
# ─────────────────────────────────────────────────────────────────────────────

def sample_volatility_block_leverage(
    Y: np.ndarray,
    f_aug: np.ndarray,
    theta: dict,
    w_u: np.ndarray,
    w_eps: np.ndarray,
    logh_u: np.ndarray,
    logh_eps: np.ndarray,
    sv_u: np.ndarray,
    sv_eps: np.ndarray,
    rho_u: np.ndarray,
    rho_eps: np.ndarray,
    rng: np.random.Generator,
    *,
    prior_a: float = 2.0,
    prior_b: float = 0.05,
    prop_path: float = 0.25,
    prop_sigma2: float = 0.20,
    prop_rho: float = 0.06,
    inv_sqrt_spd=None,
) -> dict:
    r"""
    Contemporaneous-leverage volatility + leverage-parameter sampler (Branch A).

    Sweeps, for the common factor and each idiosyncratic series: (b) the
    single-move Metropolis log-vol path; Family~B ``(phi, sigma^2)`` (leverage
    aware); Family~C ``rho`` (Metropolis).  ``mu = 0`` throughout.

    Returns ``h_u, h_eps, logh_u, logh_eps, sv_u, sv_eps, rho_u, rho_eps`` and the
    Metropolis acceptance rates ``acc`` (dict).
    """
    from mcmc.sample_vol import _inv_sqrt_spd as _isqrt
    if inv_sqrt_spd is None:
        inv_sqrt_spd = _isqrt

    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    T, M = Y.shape
    r = A.shape[0]
    F = f_aug[:, :r]

    acc = {"path_u": 0.0, "path_eps": 0.0, "sigma2": 0.0, "rho_u": 0.0, "rho_eps": 0.0}

    # ── Common factor ────────────────────────────────────────────────────────
    Qinv_half = inv_sqrt_spd(Q)
    u = F[1:] - F[:-1] @ A.T                                # (T-1, r)
    tilde_u = (np.sqrt(w_u[1:])[:, None]) * (u @ Qinv_half.T)  # (T-1, r); z = tilde_u/sqrt(h)
    tilde_full = np.zeros((T, r)); tilde_full[1:] = tilde_u
    has_u = np.zeros(T, bool); has_u[1:] = True
    S_u = np.sum(tilde_full ** 2, axis=1)                  # ||tilde_u_t||^2
    kdim_u = np.where(has_u, r, 0)

    mu_u, phi_u, s2_u = float(sv_u[0]), float(sv_u[1]), float(sv_u[2])
    rho_u = np.asarray(rho_u, float).copy()
    rho2_u = float(rho_u @ rho_u)
    gcoef_u = tilde_full @ rho_u                            # rho . tilde_u_t  (T,)

    logh_u_new, na, npp = _lev_path_mh(logh_u, S_u, kdim_u, gcoef_u, has_u,
                                       phi_u, s2_u, rho2_u, prop_path, rng)
    acc["path_u"] = na / npp

    # leverage regressors at the new path: z^u_t = tilde_u_t * exp(-x_t/2)
    z_u = tilde_full * np.exp(-0.5 * logh_u_new)[:, None]   # (T, r)
    zeta_u = z_u @ rho_u                                    # rho . z^u_t  (T,)
    # Family B: phi then sigma^2 (leverage aware)
    phi_u = _draw_phi_lev(logh_u_new, zeta_u, has_u, s2_u, rho2_u, rng)
    s2_u, a1 = _draw_sigma2_lev(logh_u_new, zeta_u, has_u, phi_u, rho2_u, s2_u,
                                prior_a, prior_b, prop_sigma2, rng)
    # Family C: rho (vector) on the leverage-bearing transitions t = 1..T-1
    eta_u = logh_u_new[1:] - phi_u * logh_u_new[:-1]        # (T-1,)
    K_u = np.sqrt(s2_u) * z_u[1:]                           # k_t = sigma * z^u_t (T-1, r)
    rho_u, ar = draw_rho_vec(rho_u, eta_u, K_u, s2_u, prop_rho, rng)
    acc["sigma2"] += a1; acc["rho_u"] = ar
    sv_u_new = np.array([0.0, phi_u, s2_u])

    # ── Idiosyncratic series ─────────────────────────────────────────────────
    signal = F @ Lambda.T
    logh_eps_new = np.zeros((T, M))
    sv_eps_new = np.zeros((M, 3))
    rho_eps_new = np.zeros(M)
    acc_pe = 0.0; acc_s = 0.0; acc_re = 0.0
    for i in range(M):
        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        e_full = np.zeros(T)
        has_i = np.zeros(T, bool)
        eps = Y[obs_t, i] - signal[obs_t, i]
        e_full[obs_t] = np.sqrt(w_eps[obs_t] / R[i]) * eps   # e_{i,t} ~ N(0, h)
        has_i[obs_t] = True
        S_i = e_full ** 2
        kdim_i = np.where(has_i, 1, 0)

        phi_i, s2_i = float(sv_eps[i, 1]), float(sv_eps[i, 2])
        rho_i = float(rho_eps[i]); rho2_i = rho_i * rho_i
        gcoef_i = rho_i * e_full

        lh_i, na_i, np_i = _lev_path_mh(logh_eps[:, i], S_i, kdim_i, gcoef_i, has_i,
                                        phi_i, s2_i, rho2_i, prop_path, rng)
        acc_pe += na_i / np_i

        z_i = e_full * np.exp(-0.5 * lh_i)                    # z_t (T,)
        zeta_i = rho_i * z_i
        phi_i = _draw_phi_lev(lh_i, zeta_i, has_i, s2_i, rho2_i, rng)
        s2_i, a_si = _draw_sigma2_lev(lh_i, zeta_i, has_i, phi_i, rho2_i, s2_i,
                                      prior_a, prior_b, prop_sigma2, rng)
        # rho on leverage transitions (t>=1 and observed)
        lev_mask = has_i[1:]
        eta_i = (lh_i[1:] - phi_i * lh_i[:-1])[lev_mask]
        k_i = (np.sqrt(s2_i) * z_i[1:])[lev_mask]
        rho_i, a_ri = draw_rho_scalar(rho_i, eta_i, k_i, s2_i, prop_rho, rng)
        acc_s += a_si; acc_re += a_ri

        logh_eps_new[:, i] = lh_i
        sv_eps_new[i] = (0.0, phi_i, s2_i)
        rho_eps_new[i] = rho_i

    if M > 0:
        acc["path_eps"] = acc_pe / M
        acc["sigma2"] = (acc["sigma2"] + acc_s) / (1 + M)
        acc["rho_eps"] = acc_re / M

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
        "rho_u": rho_u, "rho_eps": rho_eps_new,
        "acc": acc,
    }
