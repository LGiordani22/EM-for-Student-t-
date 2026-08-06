"""
src/mcmc/sample_leverage_lagged.py
==================================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: blocco (b) + Famiglia C con LEVERAGE laggato (Branch B).  Come Branch A
estrae log h, i parametri AR(1) e le ρ_i, ma con timing LAGGATO: z_t è correlato con
η_{t+1}, l'innovazione della transizione SUCCESSIVA.  Identità di segno:

    z_t = d_t exp(ξ_t/2),  d_t = sign(e_t),  ξ_t = y*_t - log h_t,  y*_t = log(e_t² + c)
                                                                   [eq:lev-sign-identity]

Mistura a 10 componenti di Omori (constants.OMORI10) per la legge congiunta (ξ_t, η_{t+1});
condizionata all'indicatore s_t ∈ {1..10} E al segno d_t:

    ξ_t | s=j            ~ N(m_j, v²_j)
    η_{t+1} | (j, d, ξ)  ~ N( d_t ρ σ e^{m_j/2}(a_j + b_j(ξ_t - m_j)),  σ²(1-ρ²) )   [eq:lev-omori-cond]

La media è LINEARE in log h_t (perché ξ_t = y*_t - log h_t) ⇒ condizionatamente a
(s, d) il sistema è genuinamente lineare-gaussiano → il percorso è un vero draw FFBS
(il punto di Branch B: dovrebbe mescolare meglio del Metropolis di Branch A).

Gibbs step (b) + Family C, **Branch B** (lagged timing, Omori sign-augmented
mixture + FFBS): the *second* leverage sampler.  Unlike Branch A (contemporaneous
+ single-move Metropolis on the path), Branch B keeps the volatility path a
**direct FFBS draw** — the route's whole point, and the reason it should mix
better than Branch A's Metropolis.

Theory: ``docs/tesi/EM_for_student_t.tex``, "Branch B: lagged timing with the
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

from mcmc.constants import LOG_CHI2_MEAN, LOG_CHI2_VAR, OMORI10, QML_A, QML_B
from mcmc.sample_vol import _inv_sqrt_spd, _psd_chol, logsq_corr_matrix
from mcmc.sample_leverage import (
    _draw_phi_lev,
    _draw_sigma2_lev,
    draw_rho,
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
# Multivariate generalisation of _ffbs_tv: r-dim state, DIAGONAL time-varying
# transition, FULL measurement covariance — the kernel of the coupled QML pass
# ─────────────────────────────────────────────────────────────────────────────

def _mv_ffbs_tv(
    y_eff: np.ndarray,
    R_eff: np.ndarray,
    mask: np.ndarray,
    G: np.ndarray,
    c: np.ndarray,
    W: np.ndarray,
    stat_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Forward-filter / backward-sample the ``r``-vector state ``x_t = log h^u_t`` with

        x_0      ~ N(0, diag(stat_var)),
        x_{t+1}  = G_t x_t + c_t + N(0, diag(W_t)),
        y_eff_t  = x_t + N(0, R_eff_t)      (only where ``mask_t``),

    i.e. :func:`_ffbs_tv` at state dimension ``r``.  ``G`` is ``(T, r, r)`` — a
    **full**, time-varying transition — ``c`` and ``W`` are ``(T, r)``; index ``t``
    is the transition ``t -> t+1``.

    Why ``G_t`` must be allowed to be full: under a non-diagonal ``Q`` the leverage
    drift of factor ``k`` keys on the fully whitened shock ``z_k``, which is a linear
    combination of **every** factor's standardised shock (``z = M eps``,
    ``M = Q^{-1/2} diag(sqrt(q_kk))``).  Once ``eps_j = d_j exp(xi_j/2)`` is
    linearised in ``xi_j = y*_j - x_j``, that drift is linear in the whole vector
    ``x_t``, not just in ``x_{k,t}``.  A diagonal ``G`` would silently drop the
    off-diagonal mixing — which is exactly the error that made the first coupled
    attempt unstable.  At diagonal ``Q``, ``M = I`` and ``G_t`` is diagonal again.
    """
    T = mask.shape[0]
    r = stat_var.shape[0]

    a = np.zeros((T, r))
    P = np.zeros((T, r, r))
    for t in range(T):
        if t == 0:
            a_pred, P_pred = np.zeros(r), np.diag(stat_var)
        else:
            Gm = G[t - 1]
            a_pred = Gm @ a[t - 1] + c[t - 1]
            P_pred = Gm @ P[t - 1] @ Gm.T + np.diag(W[t - 1])
        if mask[t]:
            S = P_pred + R_eff[t]
            K = P_pred @ np.linalg.inv(0.5 * (S + S.T))
            a[t] = a_pred + K @ (y_eff[t] - a_pred)
            P[t] = P_pred - K @ P_pred
        else:
            a[t], P[t] = a_pred, P_pred
        P[t] = 0.5 * (P[t] + P[t].T)

    x = np.zeros((T, r))
    x[T - 1] = a[T - 1] + _psd_chol(P[T - 1]) @ rng.standard_normal(r)
    for t in range(T - 2, -1, -1):
        Gm = G[t]
        P_pred_next = Gm @ P[t] @ Gm.T + np.diag(W[t])
        J = P[t] @ Gm.T @ np.linalg.inv(P_pred_next)
        m = a[t] + J @ (x[t + 1] - (Gm @ a[t] + c[t]))
        V = P[t] - J @ (Gm @ P[t])
        x[t] = m + _psd_chol(0.5 * (V + V.T)) @ rng.standard_normal(r)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# The COUPLED common block under leverage: QML measurement + linearised drift
# ─────────────────────────────────────────────────────────────────────────────

def _branch_b_common_qml(
    ystar: np.ndarray,
    signs: np.ndarray,
    has: np.ndarray,
    phi: np.ndarray,
    sigma2: np.ndarray,
    rho: np.ndarray,
    Sigma_xi: np.ndarray,
    M: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    r"""
    Draw the ``r`` per-factor common log-vol paths **jointly** under lagged leverage,
    with a **non-diagonal ``Q`` handled exactly on both sides** — measurement *and*
    leverage drift.  This is the coupled pass the ``.tex`` leaves open ("would demand
    a joint r-dimensional sign-augmented mixture that we do not derive"): QML supplies
    it, because dropping the mixture is what lets a full covariance through.

    **The two objects, and why they differ.**  The measurement reads the
    per-component standardised shock ``eps_k = e_k/sqrt(h_k)``, whose log-square is
    linear in ``log h_k``.  The leverage keys on the **fully whitened** shock
    ``z = Q^{-1/2} H^{-1/2} sqrt(w) u`` (Option A, ``eq:lev-cond-common``).  The two
    are linearly related, *exactly*:

        z = M eps,      M = Q^{-1/2} diag(sqrt(q_kk)),

    and ``eps_j = d_j exp(xi_j/2)`` with ``xi_j = y*_j - x_j`` and
    ``d_j = sign(e_j) = sign(u_j)`` — a sign that is **observable and independent of
    the path** (unlike ``sign(z_k)``, which mixes the ``h``'s).  Hence

        z_{k,t} = sum_j M_kj d_{j,t} exp(xi_{j,t}/2).

    **The linearisation.**  QML replaces ``exp(xi/2) = |eps|`` by its best linear
    predictor under the exact log-chi^2 law, ``QML_A + QML_B (xi - LOG_CHI2_MEAN)``
    (closed form, ``constants``) — the single-Gaussian counterpart of Omori's
    per-component ``(a_j, b_j)``.  The drift is then linear in the **whole vector**
    ``x_t``:

        x_{t+1,k} = phi_k x_{t,k}
                  - rho_k sigma_k QML_B sum_j M_kj d_{j,t} x_{j,t}
                  + rho_k sigma_k sum_j M_kj d_{j,t}(QML_A + QML_B(y*_{j,t} - m))
                  + N(0, sigma_k^2 (1 - rho_k^2)),

    i.e. a **full** time-varying transition ``G_t``, while the measurement
    ``y*_t = x_t + N(m, Sigma_xi)`` carries the full ``Sigma_xi = (pi^2/2) R_xi``.
    One :func:`_mv_ffbs_tv` pass draws all ``r`` paths at once.

    **What this fixes.**  The first attempt at this block kept ``G_t`` *diagonal* —
    that is, it used ``z_k ~= d_k exp(xi_k/2)``, the ``M = I`` approximation — and
    took the sign from the full whitening to compensate.  Sign and magnitude then sat
    on *different* objects, and the coupled measurement let the FFBS indulge the
    mismatch: at ``corr(Q) = 0.8`` the least identified factor's ``phi`` collapsed and
    its ``rho`` pinned to the boundary.  The error was the **truncated drift**, not
    the coupling.  With the mixing restored, both sides key on ``eps`` and are
    consistent by construction.

    Two consequences worth naming.  **(i)** At diagonal ``Q``, ``M = I`` and this
    reduces exactly to the per-factor formula of :func:`_branch_b_one_process` (up to
    the QML-vs-mixture measurement, which remains the price of the coupled route).
    **(ii)** The whitening attenuation of audit **P5** disappears: the decoupled block
    uses ``|e_k|`` as a proxy for ``|z_k|`` and pays ``lambda(c_k)``; here ``z_k`` is
    assembled exactly from the linear map, so there is nothing to attenuate.

    Returns ``{"logh": (T, r), "g": (T, r)}`` with ``g_{t,k} = z_{k,t}`` the leverage
    regressor evaluated at the NEW path, so Family B/C read ``k_t = sigma_k g_{t-1,k}``
    exactly as in the Omori block.
    """
    T, r = ystar.shape
    sig = np.sqrt(sigma2)                                   # (r,)

    lev_out = has.copy()
    lev_out[T - 1] = False                                  # no transition out of T-1

    # ── the whitening map applied to the SIGNED, path-independent component signs ──
    # md[t] = M * diag(d_t): row k, column j = M_kj d_{j,t}
    md = M[None, :, :] * signs[:, None, :]                  # (T, r, r)
    rs = (rho * sig)[None, :, None]                         # (1, r, 1)

    # ── transition: FULL, time-varying, leverage linearised by (QML_A, QML_B) ─────
    G = np.tile(np.diag(phi), (T, 1, 1))                    # (T, r, r)
    c = np.zeros((T, r))
    W = np.tile(sigma2, (T, 1))                             # (T, r)

    lin = QML_A + QML_B * (ystar - LOG_CHI2_MEAN)           # (T, r), the level part
    G[lev_out] = (np.diag(phi)[None, :, :] - QML_B * rs * md)[lev_out]
    c[lev_out] = np.einsum("tkj,tj->tk", rs * md, lin)[lev_out]
    W[lev_out] = np.tile(sigma2 * (1.0 - rho * rho), (T, 1))[lev_out]

    # ── measurement: ONE Gaussian, constant FULL covariance (the coupling) ──────
    y_eff = np.zeros((T, r))
    y_eff[has] = (ystar - LOG_CHI2_MEAN)[has]
    R_eff = np.zeros((T, r, r))
    R_eff[has] = Sigma_xi

    stat_var = sigma2 / (1.0 - phi * phi)
    x_new = _mv_ffbs_tv(y_eff, R_eff, has, G, c, W, stat_var, rng)

    # ── Family B/C regressor: z_k at the NEW path, assembled through M (no proxy) ──
    xi_new = ystar - x_new                                  # (T, r)
    eps_lin = signs * (QML_A + QML_B * (xi_new - LOG_CHI2_MEAN))     # (T, r) ~ eps
    z_new = eps_lin @ M.T                                   # (T, r): z_k = sum_j M_kj eps_j
    return {"logh": x_new, "g": z_new}


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
    rho_sampler: str = "griddy",
    rho_grid_size: int = 401,
    rho_log_prior=None,
    common_vol_coupling: str = "decoupled",
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

    ``rho_sampler`` selects the Family~C kernel (fix P6, ``docs/fix_P6_map.md``):
    ``"griddy"`` (default) draws ``rho`` from its full conditional on the compact
    ``(-1,1)`` support, **independently of the current value**; ``"rw"`` is the
    RW-Metropolis baseline (``ESS/draw ~ 0.5%``), kept for comparison.  With the
    griddy the returned ``acc["rho_*"]`` is ``1.0`` by construction — as for the FFBS
    path draw — so **acceptance is not a mixing diagnostic**: read ``ESS``.

    Returns ``h_u, h_eps, logh_u, logh_eps, sv_u, sv_eps, rho_u, rho_eps`` and the
    acceptance rates ``acc`` (path = 1.0; sigma2 and rho remain Metropolis).
    """
    if not fix_mu0:
        raise ValueError("sample_volatility_block_leverage_lagged supports mu=0 only "
                         "(fix_mu0=True); the leverage AR(1) has no intercept.")
    if common_vol_coupling not in ("decoupled", "qml"):
        raise ValueError(
            f"common_vol_coupling={common_vol_coupling!r} under leverage: 'decoupled' "
            f"(Omori mixture, r independent channels) or 'qml' (coupled, one r-dim "
            f"FFBS).  'literal' is not available under leverage: a full R_xi does not "
            f"factorise over the Omori indicators — that is what the QML form fixes."
        )
    if use_asis:
        if sigma_prior != "half_normal":
            raise ValueError("use_asis=True requires sigma_prior='half_normal' "
                             "(CP and NCP must share the Gaussian prior on sigma_eta).")
        if common_vol_coupling == "qml":
            # The interweave re-derives the measurement from the KSC mixture; under
            # QML the measurement is a single Gaussian, so the two would target
            # different models.  Same guard as the no-leverage common block.
            raise ValueError("use_asis=True is not supported with "
                             "common_vol_coupling='qml' (the ASIS interweave assumes "
                             "the mixture measurement).")
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

    # Coupled QML pass: the r paths are drawn TOGETHER (one r-dim FFBS with the full
    # log-square covariance), before the per-factor Family B/C loop.  Decoupled: each
    # factor is its own Omori channel, drawn inside the loop.
    qml = (common_vol_coupling == "qml")
    if qml:
        Sigma_xi = LOG_CHI2_VAR * logsq_corr_matrix(Q)          # (pi^2/2) R_xi
        # The coupled pass keys EVERYTHING on the per-component shock eps: the
        # measurement reads its log-square (linear in log h), and the leverage drift
        # reconstructs the fully whitened z EXACTLY through the linear map
        #     z = M eps,   M = Q^{-1/2} diag(sqrt(q_kk)).
        # The augmenting sign is therefore sign(e_k) = sign(u_k) — OBSERVABLE and
        # path-independent — not sign(z_k), which mixes the h's.  (The first version
        # of this block used sign(z) with a *diagonal* drift, i.e. z_k ~= d_k
        # exp(xi_k/2): sign and magnitude on different objects.  That truncated drift,
        # not the coupling, is what made it unstable at strong corr(Q).)
        M_mix = Qinv_half * np.sqrt(qdiag)[None, :]             # Q^{-1/2} diag(sqrt q)
        sg = np.sign(E)
        sg = np.where(sg == 0.0, 1.0, sg)
        out_q = _branch_b_common_qml(ystar_all, sg, has_u,
                                     sv_u[:, 1].copy(), sv_u[:, 2].copy(),
                                     rho_u.copy(), Sigma_xi, M_mix, rng)
        logh_qml, g_qml = out_q["logh"], out_q["g"]             # (T, r), (T, r)

    for k in range(r):
        phi_k, s2_k = float(sv_u[k, 1]), float(sv_u[k, 2])
        rho_k = float(rho_u[k]); rho2_k = rho_k * rho_k
        if qml:
            lh_k = logh_qml[:, k]; g_k = g_qml[:, k]     # (T,)
        else:
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
        rho_k, a_rk = draw_rho(rho_k, eta_k[lev_k], k_reg, s2_k, rng,
                               sampler=rho_sampler, prop_sd=prop_rho,
                               grid_size=rho_grid_size, log_prior=rho_log_prior)
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
        rho_i, a_ri = draw_rho(rho_i, eta_i[lev_i], k_i, s2_i, rng,
                               sampler=rho_sampler, prop_sd=prop_rho,
                               grid_size=rho_grid_size, log_prior=rho_log_prior)
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
