"""
src/mcmc/sample_leverage_lagged.py
==================================

Gibbs step (b) + Family C, **Branch B** (lagged timing, Omori sign-augmented
mixture + FFBS): the *second* leverage sampler.  Unlike Branch A (contemporaneous
+ single-move Metropolis on the path), Branch B keeps the volatility path a
**direct FFBS draw** — the route's whole point, and the reason it should mix
better than Branch A's Metropolis.

Theory: ``docs/EM_for_student_t.tex``, "Branch B: lagged timing with the
sign-augmented mixture" (``subsec:lev-branch-B``):

  * Lagged timing couples the level shock ``z_t`` with ``eta_{t+1}`` — the
    innovation driving ``log h_{t+1}`` (the *next* transition), Yu (2005).
  * Sign identity (eq:lev-sign-identity): ``z_t = d_t exp(xi_t/2)``,
    ``d_t = sign(e_t)``, ``xi_t = y*_t - log h_t``, ``y*_t = log(e_t^2 + c)``.
  * Omori et al. (2007) ten-component mixture (constants.OMORI10) approximates
    the *joint* law of ``(xi_t, eta_{t+1})``; conditional on the indicator
    ``s_t in {1..10}`` AND the sign ``d_t`` (eq:lev-omori-cond):
        xi_t | s=j           ~ N(m_j, v2_j)
        eta_{t+1} | (j,d,xi)  ~ N( d_t rho sigma e^{m_j/2}(a_j + b_j(xi_t-m_j)),
                                   sigma^2 (1-rho^2) ).
    The conditional mean is **linear in log h_t** (since xi_t = y*_t - log h_t),
    so conditional on ``s_{1:T}`` and ``d_{1:T}`` the system is genuinely
    linear-Gaussian -> the base block's forward-filter/backward-sample (FFBS)
    pass draws the whole path at once.  Both obstacles (state-dependent magnitude
    + sign) dissolve.

Linear-Gaussian reduction used by the FFBS (absorbing xi_t = y*_t - log h_t into
the transition so the errors are *independent*).  With ``mu = 0`` and writing
``P_{t,k} = sigma * d_{t,k} * e^{m_{s}/2}`` for measurement ``k`` at period ``t``:

    log h_{t+1} = (phi - sum_k rho_k P_{t,k} b_{s})  log h_t
                + sum_k rho_k P_{t,k}(a_{s} + b_{s}(y*_{t,k} - m_{s}))
                + N(0, sigma^2 (1 - rho'rho))          (leverage transition)
    log h_{t+1} = phi log h_t + N(0, sigma^2)           (no-leverage transition)

i.e. a scalar AR(1) with a **time-varying** transition coefficient ``G_t`` and
intercept ``c_t`` — a mild generalisation of ``sample_vol._scalar_ar1_ffbs``,
implemented here as :func:`_ffbs_tv`.

Common factor (``r`` whitened measurements per period): the leverage acts
through the scalar combination ``rho' z^u_t`` (``.tex`` subsec:lev-branches-allproc),
so the ``r`` per-period indicators couple *through their sum* in the transition.
The indicator full conditional therefore carries a transition term, drawn by a
leave-one-out Gibbs scan over the ``r`` components — the technically delicate
piece, isolated in :func:`_branch_b_one_process`.

Family B ``(phi, sigma^2)`` and Family C ``rho`` reuse the Branch-A routines of
:mod:`mcmc.sample_leverage` *verbatim* — the conditionals are the **same**
algebraic skeleton (eq:param-rho-cond), only the regressor changes to the Omori
one ``k_t = sigma * d_{t-1} e^{m_j/2}(a_j + b_j(xi_{t-1}-m_j))``
(eq:param-rho-regressor, Branch B).  ``mu = 0`` throughout.

Nesting: at ``rho = 0`` the drift vanishes, ``G_t = phi``, ``W_t = sigma^2``, and
the FFBS reduces to the base KSC path draw of Passo 2 (with a 10- rather than
7-component indicator for the measurement, but the path law is identical at
``rho = 0``).
"""

from __future__ import annotations

import numpy as np

from mcmc.constants import OMORI10
from mcmc.sample_vol import _inv_sqrt_spd
from mcmc.sample_leverage import (
    _draw_phi_lev,
    _draw_sigma2_lev,
    dominant_dir_z,
    draw_rho_common,
    draw_rho_scalar,
    draw_rho_vec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Generalised scalar FFBS: time-varying transition coefficient and intercept
# ─────────────────────────────────────────────────────────────────────────────

def _ffbs_tv(
    y_eff: np.ndarray,
    V_eff: np.ndarray,
    mask: np.ndarray,
    G: np.ndarray,
    c: np.ndarray,
    W: np.ndarray,
    stat_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Forward-filter / backward-sample a scalar state ``x_t = log h_t`` with a
    **time-varying** linear-Gaussian transition,

        x_0      ~ N(0, stat_var),
        x_{t+1}  = G_t x_t + c_t + N(0, W_t),
        y_eff_t  = x_t + N(0, V_eff_t)   (only where ``mask_t``),

    where ``G_t, c_t, W_t`` (length-``T`` arrays; index ``t`` is the transition
    ``t -> t+1``) carry the Omori leverage drift, already linearised in ``x_t``.
    Returns one draw of ``x_{0:T-1}``.  Reduces to
    :func:`mcmc.sample_vol._scalar_ar1_ffbs` when ``G==phi``, ``c==0``,
    ``W==sigma^2`` (the ``rho = 0`` / no-leverage case).
    """
    T = mask.shape[0]
    a = np.zeros(T)
    P = np.zeros(T)
    for t in range(T):
        if t == 0:
            a_pred, P_pred = 0.0, stat_var
        else:
            a_pred = G[t - 1] * a[t - 1] + c[t - 1]
            P_pred = G[t - 1] * G[t - 1] * P[t - 1] + W[t - 1]
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
        P_pred_next = G[t] * G[t] * P[t] + W[t]
        J = G[t] * P[t] / P_pred_next
        m = a[t] + J * (x[t + 1] - (G[t] * a[t] + c[t]))
        V = P[t] * (1.0 - J * G[t])
        x[t] = m + np.sqrt(max(V, 0.0)) * rng.standard_normal()
    return x


# ─────────────────────────────────────────────────────────────────────────────
# One log-vol process under lagged leverage: indicators (+sign+transition) + path
# ─────────────────────────────────────────────────────────────────────────────

def _branch_b_one_process(
    ystar: np.ndarray,
    signs: np.ndarray,
    has: np.ndarray,
    logh_cur: np.ndarray,
    phi: float,
    sigma2: float,
    rho_vec: np.ndarray,
    rng: np.random.Generator,
    omori: dict = OMORI10,
) -> dict:
    r"""
    One Branch-B sub-sweep for a single log-volatility process.

    Parameters
    ----------
    ystar : (T, K)        log-square measurements ``y*_{t,k}`` (``K = r`` for the
                          common factor, ``K = 1`` for an idiosyncratic series);
                          rows where ``~has[t]`` are ignored.
    signs : (T, K)        ``d_{t,k} = sign(e_{t,k})`` of the underlying residual.
    has : (T,) bool       whether period ``t`` carries a measurement (hence the
                          shock ``z_t`` exists, hence the transition ``t->t+1``
                          carries leverage).
    logh_cur : (T,)       current path (for the indicator full conditional).
    phi, sigma2 : current AR(1) parameters (``mu = 0``).
    rho_vec : (K,)        leverage correlation (vector for the common factor).

    Returns ``{"logh": (T,), "g": (T, K), "s": (T, K)}`` where ``g`` is the Omori
    leverage regressor ``g_{t,k} = d_{t,k} e^{m_s/2}(a_s + b_s(xi_{t,k}-m_s))``
    evaluated at the **new** path (so that the Family-B/C regressor is
    ``k_t = sigma * (rho . g_{t-1})``), and ``s`` are the drawn indicators.
    """
    T, K = ystar.shape
    q = np.asarray(omori["q"]); m = np.asarray(omori["m"]); v2 = np.asarray(omori["v2"])
    a = np.asarray(omori["a"]); b = np.asarray(omori["b"])
    logq = np.log(q); half_log_v2 = 0.5 * np.log(v2)
    em2_all = np.exp(0.5 * m)                       # e^{m_j/2}, (10,)

    rho2 = float(rho_vec @ rho_vec)
    sig = np.sqrt(sigma2)
    Wlev = sigma2 * (1.0 - rho2)
    stat_var = sigma2 / (1.0 - phi * phi)

    x = np.asarray(logh_cur, float)
    xi = ystar - x[:, None]                          # (T, K) current xi

    # ── (i) measurement-only indicator draw (initialisation) ──────────────────
    dm = xi[:, :, None] - m[None, None, :]            # (T, K, 10)
    logp = logq[None, None, :] - half_log_v2[None, None, :] - 0.5 * dm * dm / v2[None, None, :]
    s = np.argmax(logp + rng.gumbel(size=logp.shape), axis=2)     # (T, K)

    # leverage-out mask: period t has a shock and is not the last period
    lev_out = has.copy()
    lev_out[T - 1] = False

    # current next-period innovation eta_{t+1} = x_{t+1} - phi x_t (index by t)
    eta_next = np.full(T, np.nan)
    eta_next[: T - 1] = x[1:] - phi * x[:-1]

    def _g_of(s_idx: np.ndarray, xi_: np.ndarray) -> np.ndarray:
        return signs * em2_all[s_idx] * (a[s_idx] + b[s_idx] * (xi_ - m[s_idx]))

    # ── (ii) refine indicators with the TRANSITION term (sign + leverage) ─────
    # Full conditional of s_{t,k} includes N(eta_{t+1}; drift_j, sigma^2(1-rho^2)),
    # the drift summing over k -> leave-one-out Gibbs scan over the K components.
    if rho2 > 0.0:
        g = _g_of(s, xi)                              # (T, K)
        Dsum = g @ rho_vec                            # (T,) sum_k rho_k g_{t,k}
        for k in range(K):
            rk = float(rho_vec[k])
            if rk == 0.0:
                continue
            # candidate component-k regressor for every indicator j
            g_kj = signs[:, k:k + 1] * em2_all[None, :] * (
                a[None, :] + b[None, :] * (xi[:, k:k + 1] - m[None, :]))   # (T, 10)
            Dother = Dsum - rk * g[:, k]              # (T,) contribution of k'!=k
            resid = eta_next - sig * Dother           # target mean = sig*rk*g_kj
            # measurement log-prob for component k
            dmk = xi[:, k:k + 1] - m[None, :]
            lpk = logq[None, :] - half_log_v2[None, :] - 0.5 * dmk * dmk / v2[None, :]
            # transition log-prob, only on leverage transitions
            tr = np.zeros((T, 10))
            mlo = lev_out & ~np.isnan(resid)
            diff = resid[:, None] - sig * rk * g_kj   # (T, 10)
            tr[mlo] = -0.5 * diff[mlo] ** 2 / Wlev
            tot = lpk + tr
            s_k = np.argmax(tot + rng.gumbel(size=tot.shape), axis=1)      # (T,)
            s[:, k] = np.where(has, s_k, s[:, k])
            g_new_k = signs[:, k] * em2_all[s[:, k]] * (
                a[s[:, k]] + b[s[:, k]] * (xi[:, k] - m[s[:, k]]))
            Dsum = Dsum - rk * g[:, k] + rk * g_new_k
            g[:, k] = g_new_k

    # ── (iii) build the time-varying FFBS transition coefficients ─────────────
    sj = s                                            # (T, K)
    P = sig * signs * em2_all[sj]                     # (T, K), P_{t,k}
    bj = b[sj]; aj = a[sj]; mj = m[sj]
    red = (rho_vec[None, :] * P * bj).sum(axis=1)               # sum_k rho_k P b
    cterm = (rho_vec[None, :] * P * (aj + bj * (ystar - mj))).sum(axis=1)
    G = np.full(T, float(phi))
    c = np.zeros(T)
    W = np.full(T, float(sigma2))
    G[lev_out] = phi - red[lev_out]
    c[lev_out] = cterm[lev_out]
    W[lev_out] = Wlev

    # ── (iv) effective per-period measurement (combine the K whitened obs) ────
    inv_v2 = 1.0 / v2[sj]                             # (T, K)
    prec = np.where(has[:, None], inv_v2, 0.0).sum(axis=1)
    rhs = np.where(has[:, None], (ystar - mj) * inv_v2, 0.0).sum(axis=1)
    mask = prec > 0
    V_eff = np.zeros(T); y_eff = np.zeros(T)
    V_eff[mask] = 1.0 / prec[mask]
    y_eff[mask] = rhs[mask] / prec[mask]

    # ── (v) draw the whole path (direct FFBS, no Metropolis) ──────────────────
    x_new = _ffbs_tv(y_eff, V_eff, mask, G, c, W, stat_var, rng)

    # ── (vi) Omori regressor at the new path, for Family B/C ──────────────────
    xi_new = ystar - x_new[:, None]
    g_new = signs * em2_all[sj] * (aj + bj * (xi_new - mj))      # (T, K)
    return {"logh": x_new, "g": g_new, "s": sj}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: step (b) + Family B + Family C for all M+1 processes (Branch B)
# ─────────────────────────────────────────────────────────────────────────────

def sample_volatility_block_leverage_lagged(
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
    offset: float = 1e-6,
    prop_sigma2: float = 0.20,
    prop_rho: float = 0.06,
    common_lev_scalar: bool = True,
    omori: dict = OMORI10,
    inv_sqrt_spd=None,
    **_ignored,
) -> dict:
    r"""
    Lagged-leverage volatility + leverage-parameter sampler (Branch B, Omori).

    Mirrors :func:`mcmc.sample_leverage.sample_volatility_block_leverage` (Branch A)
    block-for-block, but the volatility path is a **direct FFBS draw** (sign-
    augmented 10-component mixture), not a single-move Metropolis sweep.  ``mu = 0``
    throughout.  ``prop_path`` is accepted and ignored (no path Metropolis); the
    returned ``acc["path_*"] = 1.0`` records that the FFBS path is always accepted
    — the structural mixing advantage over Branch A.

    Returns ``h_u, h_eps, logh_u, logh_eps, sv_u, sv_eps, rho_u, rho_eps`` and the
    acceptance rates ``acc`` (path = 1.0; sigma2 and rho remain Metropolis).
    """
    if inv_sqrt_spd is None:
        inv_sqrt_spd = _inv_sqrt_spd

    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    T, M = Y.shape
    r = A.shape[0]
    F = f_aug[:, :r]

    acc = {"path_u": 1.0, "path_eps": 1.0, "sigma2": 0.0, "rho_u": 0.0, "rho_eps": 0.0}

    # ── Common factor (K = r whitened measurements per t = 1..T-1) ────────────
    Qinv_half = inv_sqrt_spd(Q)
    u = F[1:] - F[:-1] @ A.T                              # (T-1, r)
    tilde_u = (np.sqrt(w_u[1:])[:, None]) * (u @ Qinv_half.T)   # (T-1, r) signed
    ystar_u = np.zeros((T, r)); signs_u = np.zeros((T, r))
    ystar_u[1:] = np.log(tilde_u ** 2 + offset)
    signs_u[1:] = np.sign(tilde_u)
    signs_u[signs_u == 0.0] = 1.0
    has_u = np.zeros(T, bool); has_u[1:] = True

    mu_u, phi_u, s2_u = float(sv_u[0]), float(sv_u[1]), float(sv_u[2])
    rho_u = np.asarray(rho_u, float).copy()
    rho2_u = float(rho_u @ rho_u)

    out_u = _branch_b_one_process(ystar_u, signs_u, has_u, logh_u,
                                  phi_u, s2_u, rho_u, rng, omori)
    logh_u_new = out_u["logh"]; g_u = out_u["g"]          # (T, r)

    # transition INTO t carries leverage iff z_{t-1} exists -> has_u[t-1]
    has_tr_u = np.zeros(T, bool); has_tr_u[1:] = has_u[:-1]
    # Family B regressor zeta_t = rho . g_{t-1}  (drift into t = sigma * zeta_t)
    zeta_u = np.zeros(T)
    zeta_u[1:] = g_u[:-1] @ rho_u
    phi_u = _draw_phi_lev(logh_u_new, zeta_u, has_tr_u, s2_u, rho2_u, rng)
    s2_u, a1 = _draw_sigma2_lev(logh_u_new, zeta_u, has_tr_u, phi_u, rho2_u, s2_u,
                                prior_a, prior_b, prop_sigma2, rng)
    # Family C: rho on leverage-bearing transitions (k_t = sigma * g_{t-1}).
    # General case = free r-vector; adopted specialization = scalar along the
    # identified dominant direction g (subsec:common-lev-identif).
    eta_u = logh_u_new[1:] - phi_u * logh_u_new[:-1]      # (T-1,)
    lev_u = has_tr_u[1:]
    K_u = (np.sqrt(s2_u) * g_u[:-1])[lev_u]               # (n_lev, r)
    g_dom = dominant_dir_z(F, Qinv_half) if (common_lev_scalar and r > 1) else None
    rho_u, ar = draw_rho_common(rho_u, eta_u[lev_u], K_u, s2_u, prop_rho, rng, g_dom)
    acc["sigma2"] += a1; acc["rho_u"] = ar
    sv_u_new = np.array([0.0, phi_u, s2_u])

    # ── Idiosyncratic series (K = 1) ──────────────────────────────────────────
    signal = F @ Lambda.T
    logh_eps_new = np.zeros((T, M))
    sv_eps_new = np.zeros((M, 3))
    rho_eps_new = np.zeros(M)
    acc_s = 0.0; acc_re = 0.0
    for i in range(M):
        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        ystar_i = np.zeros((T, 1)); signs_i = np.zeros((T, 1))
        has_i = np.zeros(T, bool)
        eps = Y[obs_t, i] - signal[obs_t, i]
        e = np.sqrt(w_eps[obs_t] / R[i]) * eps           # signed N(0, h)
        ystar_i[obs_t, 0] = np.log(e ** 2 + offset)
        si = np.sign(e); si[si == 0.0] = 1.0
        signs_i[obs_t, 0] = si
        has_i[obs_t] = True

        phi_i, s2_i = float(sv_eps[i, 1]), float(sv_eps[i, 2])
        rho_i = float(rho_eps[i]); rho2_i = rho_i * rho_i

        out_i = _branch_b_one_process(ystar_i, signs_i, has_i, logh_eps[:, i],
                                      phi_i, s2_i, np.array([rho_i]), rng, omori)
        lh_i = out_i["logh"]; g_i = out_i["g"][:, 0]     # (T,)

        has_tr_i = np.zeros(T, bool); has_tr_i[1:] = has_i[:-1]
        zeta_i = np.zeros(T); zeta_i[1:] = rho_i * g_i[:-1]
        phi_i = _draw_phi_lev(lh_i, zeta_i, has_tr_i, s2_i, rho2_i, rng)
        s2_i, a_si = _draw_sigma2_lev(lh_i, zeta_i, has_tr_i, phi_i, rho2_i, s2_i,
                                      prior_a, prior_b, prop_sigma2, rng)
        eta_i = lh_i[1:] - phi_i * lh_i[:-1]
        lev_i = has_tr_i[1:]
        k_i = (np.sqrt(s2_i) * g_i[:-1])[lev_i]
        rho_i, a_ri = draw_rho_scalar(rho_i, eta_i[lev_i], k_i, s2_i, prop_rho, rng)
        acc_s += a_si; acc_re += a_ri

        logh_eps_new[:, i] = lh_i
        sv_eps_new[i] = (0.0, phi_i, s2_i)
        rho_eps_new[i] = rho_i

    if M > 0:
        acc["sigma2"] = (acc["sigma2"] + acc_s) / (1 + M)
        acc["rho_eps"] = acc_re / M

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
        "rho_u": rho_u, "rho_eps": rho_eps_new,
        "acc": acc,
    }
