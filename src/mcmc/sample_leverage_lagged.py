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

Common factor — **Option A, per-factor (Spec II, Phase 7)**: the ``r`` common
volatilities are ``r`` **independent scalar** Omori/FFBS channels (``K = 1`` each,
via :func:`_branch_b_one_process`), structurally identical to the idiosyncratic
series (``subsec:lev-branches-allproc``) — no vector ``rho``, no ``rho'rho<1``.
Two boundary details specific to the common block: **(i)** the augmenting sign is
``d^u_k = sign(z^u_k)`` of the FULL-whitening raw shock
``z^u = sqrt(w) Q^{-1/2}(sqrt H)^{-1} u`` (not ``sign(u_k)``; the magnitude uses
the per-component ``e_k = sqrt(w/q_kk) u_k`` so ``xi_k = y*_k - log h_k`` stays
linear in ``log h_k`` — exact at diagonal ``Q``);
**(ii)** ``z^u_0`` is undefined (``u_t`` exists for ``t>=1``), so the first
leverage-bearing transition is into ``log h_2`` (``has_u[0]=False``).

*How wrong is the magnitude whitening away from diagonal Q?* (``docs/audit_P1-P5.md``
§P5, quantified — the old "differs mildly for full Q" was vague.)  The Family~C
regressor is ``g_k = sign(z_k)|zbar_k|`` against a true drift ``rho z_k``, so with
``Var(zbar_k)=1`` the estimate is attenuated by exactly

    rho_hat/rho = E[|z_k||zbar_k|] = lambda(c_k),
    lambda(c) = (2/pi)(c*arcsin c + sqrt(1-c^2)),   c_k = (Q^{1/2})_kk / sqrt(q_kk),

verified by Monte Carlo.  Always an attenuation, **never a sign flip** (the sign uses
the exact whitening): ``lambda(1)=1`` at diagonal ``Q``, ``0.98`` at ``corr(Q)=0.3``,
``0.88`` at ``0.8``.  An attenuated ``|rho|`` understates the predictive left skew, so
it errs *against* the Growth-at-Risk objective — but on the real panel ``c_k>=0.9986``
and the bias is ``-0.1%``, so no correction is applied.  The Omori linearisation
itself (``a_j + b_j(xi-m_j)`` for ``exp(xi/2)``) does **not** attenuate:
``E[zg]/E[g^2] = 1.0000``, ``corr(g,z) = 0.997``.  Inspect at any ``Q`` with
:func:`mcmc.diagnostics.leverage_whitening_attenuation`.

Family B ``(phi, sigma^2)`` and Family C scalar ``rho_i`` reuse the Branch-A
routines of :mod:`mcmc.sample_leverage` *verbatim* — the **same** algebraic
skeleton (eq:param-rho-cond), only the regressor changes to the Omori one
``k_t = sigma * d_{t-1} e^{m_j/2}(a_j + b_j(xi_{t-1}-m_j))``
(eq:param-rho-regressor, Branch B).  ``mu = 0`` throughout.  ASIS (``use_asis``,
Phase 6) wraps each channel's Family~B draw, with the lagged drift mask.

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
    draw_rho_scalar,
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
# Orchestrator: step (b) + Family B + Family C for all M+r processes (Branch B)
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
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    use_asis: bool = False,
    fix_mu0: bool = True,
    sv_idio: bool = True,
    offset: float = 1e-6,
    prop_sigma2: float = 0.20,
    prop_rho: float = 0.06,
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

    ``sv_idio=False`` is the **D2-a** restriction (``subsec:variants-restrictions``):
    ``h^eps ≡ 1`` frozen, no idiosyncratic Family~B / Family~C draw.

    Returns ``h_u, h_eps, logh_u, logh_eps, sv_u, sv_eps, rho_u, rho_eps`` and the
    acceptance rates ``acc`` (path = 1.0; sigma2 and rho remain Metropolis).
    """
    if not fix_mu0:
        raise ValueError("sample_volatility_block_leverage_lagged supports mu=0 only "
                         "(fix_mu0=True); the leverage AR(1) has no intercept.")
    if use_asis:
        if sigma_prior != "half_normal":
            raise ValueError("use_asis=True requires sigma_prior='half_normal' "
                             "(CP and NCP must share the Gaussian prior on sigma_eta).")
        from mcmc.sample_asis import asis_scale_interweave      # lazy: avoid import cycle
    if inv_sqrt_spd is None:
        inv_sqrt_spd = _inv_sqrt_spd

    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    T, M = Y.shape
    r = A.shape[0]
    F = f_aug[:, :r]

    acc = {"path_u": 1.0, "path_eps": 1.0, "sigma2": 0.0, "rho_u": 0.0, "rho_eps": 0.0}

    # ── Common factor: r per-factor channels under Option A (Spec II) ─────────
    # Each factor is its own scalar Omori/FFBS channel (K = 1), structurally like
    # an idiosyncratic series (subsec:lev-branches-allproc).  Two boundary details
    # specific to the common block (subsec:lev-branches-allproc (i)-(ii)):
    #   (i) the augmenting sign is d^u_k = sign(z^u_k) of the FULL-whitening raw
    #       shock z^u = sqrt(w) Q^{-1/2}(sqrt H)^{-1} u — not sign(u_k) — computed
    #       once from the current path (frozen for the sweep).  The magnitude uses
    #       the per-component measurement e_k = sqrt(w/q_kk) u_k (decoupled, Phase 1)
    #       so xi_k = y*_k - log h_k stays linear in log h_k (exact at diagonal Q).
    #   (ii) u_t exists only for t >= 1, so z^u_0 is undefined: the first leverage-
    #        bearing transition is into log h_2 (driven by z_1) — honoured by
    #        has_u[0] = False (=> has_tr_u[t] = has_u[t-1], leverage into t >= 2).
    Qinv_half = inv_sqrt_spd(Q)
    qdiag = np.diag(Q)
    u = F[1:] - F[:-1] @ A.T                              # (T-1, r)
    has_u = np.zeros(T, bool); has_u[1:] = True
    b_full = np.zeros((T, r)); b_full[1:] = np.sqrt(w_u[1:])[:, None] * u   # sqrt(w) u
    E = b_full / np.sqrt(qdiag)[None, :]                  # per-component e_k (T, r); E[0]=0
    ystar_all = np.log(E ** 2 + offset)                  # (T, r)
    # full-whitening raw shock at the current path -> the augmenting signs (i)
    a_u = np.exp(-0.5 * np.asarray(logh_u, float)) * b_full          # diag(exp(-x/2)) sqrt(w) u
    z_u = a_u @ Qinv_half                                            # (T, r) z^u_k, full whitening
    signs_all = np.sign(z_u); signs_all[signs_all == 0.0] = 1.0      # (T, r)
    has_tr_u = np.zeros(T, bool); has_tr_u[1:] = has_u[:-1]          # leverage into t >= 2

    sv_u = np.asarray(sv_u, float)
    rho_u = np.asarray(rho_u, float).copy()
    logh_u = np.asarray(logh_u, float)
    logh_u_new = np.zeros((T, r))
    sv_u_new = np.zeros((r, 3))
    rho_u_new = np.zeros(r)
    acc_s_u = 0.0; acc_r_u = 0.0
    for k in range(r):
        phi_k, s2_k = float(sv_u[k, 1]), float(sv_u[k, 2])
        rho_k = float(rho_u[k]); rho2_k = rho_k * rho_k
        out_k = _branch_b_one_process(ystar_all[:, k:k + 1], signs_all[:, k:k + 1],
                                      has_u, logh_u[:, k], phi_k, s2_k,
                                      np.array([rho_k]), rng, omori)
        lh_k = out_k["logh"]; g_k = out_k["g"][:, 0]     # (T,)
        # Family B: phi, sigma2 on the lagged drift zeta_t = rho_k g_{t-1}
        zeta_k = np.zeros(T); zeta_k[1:] = rho_k * g_k[:-1]
        phi_k = _draw_phi_lev(lh_k, zeta_k, has_tr_u, s2_k, rho2_k, rng)
        s2_k, a1 = _draw_sigma2_lev(lh_k, zeta_k, has_tr_u, phi_k, rho2_k, s2_k,
                                    prior_a, prior_b, prop_sigma2, rng,
                                    sigma_prior=sigma_prior, half_normal_B=half_normal_B)
        if use_asis:                                     # (2)-(4) NCP interweave (lagged)
            z_lag = np.zeros(T); z_lag[1:] = g_k[:-1]    # drift into t = rho g_{t-1}
            lh_k, phi_k, s2_k = asis_scale_interweave(
                lh_k, ystar_all[:, k], has_u, s2_k, rho_k, z_lag, rng,
                has_lev=has_tr_u, half_normal_B=half_normal_B)
            s_k = out_k["s"][:, 0]                        # recompute Omori g at rescaled path
            g_k = signs_all[:, k] * np.exp(0.5 * omori["m"][s_k]) * (
                omori["a"][s_k] + omori["b"][s_k] * (ystar_all[:, k] - lh_k - omori["m"][s_k]))
        # Family C: scalar rho_k on the leverage transitions (k_t = sigma_k g_{t-1})
        eta_k = lh_k[1:] - phi_k * lh_k[:-1]
        lev_k = has_tr_u[1:]
        k_reg = (np.sqrt(s2_k) * g_k[:-1])[lev_k]
        rho_k, a_rk = draw_rho_scalar(rho_k, eta_k[lev_k], k_reg, s2_k, prop_rho, rng)
        logh_u_new[:, k] = lh_k
        sv_u_new[k] = (0.0, phi_k, s2_k)
        rho_u_new[k] = rho_k
        acc_s_u += a1; acc_r_u += a_rk
    acc["sigma2"] = acc_s_u
    acc["rho_u"] = acc_r_u / max(1, r)
    rho_u = rho_u_new

    # ── Idiosyncratic series (K = 1; omitted under D2-a) ──────────────────────
    signal = F @ Lambda.T
    logh_eps_new = np.zeros((T, M))
    sv_eps_new = np.asarray(sv_eps, float).copy() if not sv_idio else np.zeros((M, 3))
    rho_eps_new = np.asarray(rho_eps, float).copy() if not sv_idio else np.zeros(M)
    acc_s = 0.0; acc_re = 0.0
    M_lev = M if sv_idio else 0
    for i in range(M_lev):
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
                                      prior_a, prior_b, prop_sigma2, rng,
                                      sigma_prior=sigma_prior, half_normal_B=half_normal_B)
        if use_asis:                                     # (2)-(4) NCP interweave (lagged)
            z_lag = np.zeros(T); z_lag[1:] = g_i[:-1]
            lh_i, phi_i, s2_i = asis_scale_interweave(
                lh_i, ystar_i[:, 0], has_i, s2_i, rho_i, z_lag, rng,
                has_lev=has_tr_i, half_normal_B=half_normal_B)
            s_ii = out_i["s"][:, 0]
            g_i = signs_i[:, 0] * np.exp(0.5 * omori["m"][s_ii]) * (
                omori["a"][s_ii] + omori["b"][s_ii] * (ystar_i[:, 0] - lh_i - omori["m"][s_ii]))
        eta_i = lh_i[1:] - phi_i * lh_i[:-1]
        lev_i = has_tr_i[1:]
        k_i = (np.sqrt(s2_i) * g_i[:-1])[lev_i]
        rho_i, a_ri = draw_rho_scalar(rho_i, eta_i[lev_i], k_i, s2_i, prop_rho, rng)
        acc_s += a_si; acc_re += a_ri

        logh_eps_new[:, i] = lh_i
        sv_eps_new[i] = (0.0, phi_i, s2_i)
        rho_eps_new[i] = rho_i

    acc["sigma2"] = (acc["sigma2"] + acc_s) / (r + M_lev)   # r common + M idio draws
    if M_lev > 0:
        acc["rho_eps"] = acc_re / M_lev

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
        "rho_u": rho_u, "rho_eps": rho_eps_new,
        "acc": acc,
    }
