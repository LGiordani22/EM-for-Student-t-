"""
src/mcmc/sample_leverage.py
===========================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: blocco (b) + Famiglia C con LEVERAGE contemporaneo (Branch A).  Oltre
alle traiettorie log h e ai parametri AR(1), estrae le correlazioni di leverage ρ_i
(una per canale).  La coppia (shock di livello, innovazione di log-vol):

    (z_t, η_t) ~ N(0, [[1, ρσ_η], [ρσ_η, σ_η²]])
  ⇒ η_t | z_t ~ N(ρ σ_η z_t,  σ_η² (1-ρ²))                        [eq:lev-cond-scalar]

Il drift di leverage entra nella transizione dello STESSO periodo:

    log h_t | log h_{t-1}, z_t ~ N(φ log h_{t-1} + ρ σ_η z_t,  σ_η² (1-ρ²))

Residuo grezzo sbiancato (Spec. II, Option A) — componente i = shock proprio del
fattore i, accoppiato a η^u_{i,t}:

    z^u_t = sqrt(w^u_t) Q^{-1/2} (sqrt H^u_t)^{-1} u_t
    η^u_{i,t} | z^u_{i,t} ~ N(ρ_i σ_{u,i} z^u_{i,t}, σ_{u,i}²(1-ρ_i²))   [eq:lev-cond-common]

Il drift rende la transizione NON lineare-gaussiana ⇒ niente KSC/FFBS: il percorso
si aggiorna con un Metropolis-Hastings single-move; ogni ρ_i è una mossa scalare.
A ρ=0 il blocco si riduce esattamente al caso base (drift 0, varianza σ_η²).

Nota (coupling / QML): il target di Branch A usa la verosimiglianza ESATTA col
whitening pieno Q^{-1/2}, quindi tratta una Q piena senza approssimazioni — non gli
serve alcun passo "accoppiato" (QML).  Il coupling è un dispositivo del solo Branch B
(che linearizza con Omori) e del blocco comune SENZA leverage; sotto Branch A è vietato
(guard in gibbs).

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
to the base KSC block — but we still route ``rho = 0`` runs through the base block
(``sample_vol.sample_volatility_block_specII``) to keep that path bit-identical
and, as ``subsec:variants-restrictions`` notes, because the symmetric case needs
only the seven-component KSC mixture.

Config-aware: ``r``, ``M`` and the per-series structure come from the inputs.

Common leverage — **Option A** (``subsec:lev-attach-choice``, ``eq:lev-cond-common``;
Phase 4).  Under Specification~II the common block carries ``r`` **per-factor**
volatilities, and the leverage is ``r`` **independent scalar** correlations
``rho_i``, each coupling factor ``i``'s log-vol innovation ``eta^u_i`` to its own
**raw** base shock ``z^u_i`` — the ``i``-th component of
``z^u_t = sqrt(w) Q^{-1/2} (sqrt H^u_t)^{-1} u_t`` (the full symmetric ``Q^{-1/2}``).
There is **no vector draw and no ``rho'rho<1`` region** (the thesis retires the
dominant-direction specialisation): every ``rho_i`` is one scalar Metropolis move
(:func:`draw_rho_scalar`), exactly as on the idiosyncratic side.  The path draw is
the *coupled* multivariate single-move Metropolis :func:`_lev_path_mh_mv_common`,
because ``z^u_{k,t}`` mixes every factor's volatility through ``Q^{-1/2}``.  The
old vector machinery (``draw_rho_vec``, ``draw_rho_common``, ``dominant_dir_z``,
``common_lev_scalar``) is gone: both branches are per-factor Option A.
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
# (b) multivariate single-move Metropolis on the r per-factor common log-vol
#     paths under Option A leverage (Spec II) — the coupled path draw
# ─────────────────────────────────────────────────────────────────────────────

def _lev_path_mh_mv_common(
    logh_u: np.ndarray,          # (T, r) current per-factor log-vol paths
    b: np.ndarray,               # (T, r) weighted innovations sqrt(w_t) u_t (0 at t=0)
    S: np.ndarray,               # (T, r) per-component level SS e_{k,t}^2 (0 at t=0)
    has_obs: np.ndarray,         # (T,) bool: leverage-bearing transition present at t
    Qinv_half: np.ndarray,       # (r, r) symmetric Q^{-1/2}
    phi: np.ndarray,             # (r,)
    sigma2: np.ndarray,          # (r,)
    rho: np.ndarray,             # (r,)
    prop_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    r"""
    One single-move random-walk Metropolis sweep over the ``r`` per-factor common
    log-vol paths ``x_{k,t} = log h^u_{k,t}`` under **Option A** leverage,
    Specification~II (``mu = 0``).  This is the *coupled* path draw: the leverage
    drift keys on the **raw shock**
    ``z^u_t = sqrt(w) Q^{-1/2} (sqrt(H^u_t))^{-1} u_t`` (eq:lev-cond-common,
    subsec:lev-attach-choice), whose ``k``-th component
    ``z^u_{k,t} = [Q^{-1/2} (exp(-x_{.,t}/2) * b_t)]_k`` mixes **every** factor's
    volatility through the symmetric root ``Q^{-1/2}``.  Moving a single
    ``x_{k,t}`` therefore shifts the transition-*into*-``t`` drift of **all** ``r``
    factors (eq:lev-transition-contemp), so the move's Metropolis target sums the
    ``r`` transition densities into ``t`` — the genuine multivariate coupling the
    scalar :func:`_lev_path_mh` cannot express (it held only under the scalar-common
    restriction ``H^u_t = h^u_t I``, where ``Q^{-1/2}`` and ``(sqrt H)^{-1}``
    commute; that restriction is *not* "Specification I" — the two specifications
    of ``subsec:vol-placement`` are both per-factor and differ in the placement of
    ``H^u_t`` around ``Q^{1/2}``).

    Terms of the full conditional of ``x_{k,t}`` that move with it:
      * **level** (``t>=1``, factor ``k`` only): ``-x/2 - S_{k,t} e^{-x}/2``;
      * **transition into ``t``** (``t>=1``, **all** factors ``j``): each
        ``N(x_{j,t}; phi_j x_{j,t-1} + rho_j sqrt(sigma2_j) z^u_{j,t},
        sigma2_j(1-rho_j^2))`` — ``z^u_{.,t}`` recomputed at the proposal;
      * **stationary prior** (``t=0``, factor ``k`` only):
        ``-x^2 / (2 sigma2_k/(1-phi_k^2))``;
      * **transition out of ``t``** (into ``t+1``, factor ``k`` only): the AR(1)
        term ``N(x_{k,t+1}; phi_k x_{k,t} + drift_{k,t+1}, .)`` — its drift
        ``z^u_{k,t+1}`` does **not** depend on ``x_{k,t}`` and is held fixed.

    Returns ``(logh_new (T, r), n_accept, n_propose)``.
    """
    T, r = logh_u.shape
    x = logh_u.copy()
    sig = np.sqrt(sigma2)                      # (r,)
    drift_coef = rho * sig                     # (r,) = rho_i sigma_{u,i}
    var_lev = sigma2 * (1.0 - rho * rho)       # (r,)
    stat_var = sigma2 / (1.0 - phi * phi)      # (r,)
    n_acc = 0
    n_prop = 0

    def _zrow(t: int, xrow: np.ndarray) -> np.ndarray:
        """z^u_{.,t} = Q^{-1/2} (exp(-xrow/2) * b_t)."""
        return Qinv_half @ (np.exp(-0.5 * xrow) * b[t])

    for t in range(T):
        # transition-into-t precomputables (all factors j) — only when t>=1
        if t >= 1:
            v_in = var_lev if has_obs[t] else sigma2      # (r,)
            base_mean = phi * x[t - 1]                    # (r,) phi_j x_{j,t-1}
        # out-of-t drift into t+1 (factor-wise), z^u_{k,t+1} independent of x[t]
        if t <= T - 2:
            v_out = var_lev if has_obs[t + 1] else sigma2
            z_tp1 = _zrow(t + 1, x[t + 1]) if has_obs[t + 1] else np.zeros(r)
            drift_tp1 = drift_coef * z_tp1                # (r,)

        for k in range(r):
            xt_old = x[t, k]
            xt_new = xt_old + prop_sd * rng.standard_normal()

            # level term (measurement of factor k at t)
            dlevel = 0.0
            if has_obs[t]:
                dlevel = ((-0.5 * xt_new - 0.5 * S[t, k] * np.exp(-xt_new))
                          - (-0.5 * xt_old - 0.5 * S[t, k] * np.exp(-xt_old)))

            # transition into t
            if t == 0:
                dinto = -0.5 * (xt_new * xt_new - xt_old * xt_old) / stat_var[k]
            else:
                row_old = x[t]
                row_new = x[t].copy(); row_new[k] = xt_new
                mean_old = base_mean + drift_coef * _zrow(t, row_old)
                mean_new = base_mean + drift_coef * _zrow(t, row_new)
                res_old = row_old - mean_old
                res_new = row_new - mean_new
                dinto = np.sum(-0.5 * res_new * res_new / v_in
                               + 0.5 * res_old * res_old / v_in)

            # transition out of t (factor k only)
            dout = 0.0
            if t <= T - 2:
                mo_old = phi[k] * xt_old + drift_tp1[k]
                mo_new = phi[k] * xt_new + drift_tp1[k]
                dd = ((x[t + 1, k] - mo_new) ** 2 - (x[t + 1, k] - mo_old) ** 2)
                dout = -0.5 * dd / v_out[k]

            n_prop += 1
            if np.log(rng.random()) < dlevel + dinto + dout:
                x[t, k] = xt_new
                n_acc += 1

    return x, n_acc, n_prop


# ─────────────────────────────────────────────────────────────────────────────
# (b') BLOCK path draw for Branch A: Laplace proposal + exact Metropolis
#
# The single-move sweep above is Branch A's one real defect (audit P3 + P2):
#   * from the flat warm start (log h == 0) it needs a KSC seed not to sit in the
#     h~1 trap — a patch on the *initialisation*, not on the kernel;
#   * it moves one coordinate per sweep with a fixed step, so doubling T doubles
#     the degrees of freedom at constant effort: Branch A *degrades* with T
#     (rho collapses at T=1200; audit P2).
#
# Both die if the path is drawn as ONE block.  The obstacle is that A's target is
# genuinely non-linear-Gaussian: the drift rho*sigma*z_t keys on the CURRENT shock
# z_t = Q^{-1/2}(exp(-x_t/2) b_t), so exp(-x_t/2) sits inside the transition mean.
# (This is why Omori's mixture — built for the *lagged* coupling, Branch B — does
# not apply here.)
#
# The fix keeps A exact and buys the block move from a proposal:
#   1. build a LINEAR-GAUSSIAN approximation of A's conditional, per factor, by
#      Taylor-expanding the two non-linear pieces (the log-chi^2 level term and
#      the exp(-x/2) drift) — a standard Laplace/Durbin-Koopman construction;
#   2. iterate it to its MODE from a *fixed* start, so the approximation depends on
#      the data and the parameters but NOT on the current draw — an independence
#      proposal;
#   3. draw the whole path from it by FFBS and accept it against the EXACT,
#      fully coupled Branch-A target.
#
# The approximation therefore touches only the *efficiency*: whatever it gets wrong
# the Metropolis ratio corrects.  Branch A stays what it is — the branch with no
# linearisation — which is the whole reason to keep it (it is the gold standard
# against which Branch B's Omori linearisation can be measured).
# ─────────────────────────────────────────────────────────────────────────────

_A_MIN = 0.1          # guard on the linearised transition coefficient (proposal only)


def _logpost_A_common(
    x: np.ndarray,               # (T, r) path
    b: np.ndarray,               # (T, r) sqrt(w_t) u_t   (0 at t=0)
    S: np.ndarray,               # (T, r) per-component level SS e_{k,t}^2
    has_obs: np.ndarray,         # (T,) bool
    Qinv_half: np.ndarray,       # (r, r)
    phi: np.ndarray,
    sigma2: np.ndarray,
    rho: np.ndarray,
) -> float:
    r"""
    Branch A's **exact** log full conditional of the common log-vol paths, up to an
    additive constant in ``x`` — the same three groups of terms the single-move
    Metropolis differences one coordinate at a time (:func:`_lev_path_mh_mv_common`),
    here evaluated on the whole ``(T, r)`` path:

      * level:        ``-x/2 - S e^{-x}/2``               (where ``has_obs``)
      * transition:   ``N(x_t; phi x_{t-1} + rho sigma z_t, sigma^2(1-rho^2))``
                      with the **exact** raw shock ``z_t = Q^{-1/2}(e^{-x_t/2} b_t)``
                      — the full symmetric root, so every factor's volatility enters
                      every factor's drift;
      * stationary prior at ``t = 0``.

    Variances are constants of the path move, so their log-determinants are dropped.
    """
    T, r = x.shape
    sig = np.sqrt(sigma2)
    cdrift = rho * sig                                   # rho_k sigma_k
    var_lev = sigma2 * (1.0 - rho * rho)
    stat_var = sigma2 / (1.0 - phi * phi)

    lp = float(np.sum(-0.5 * x[has_obs] - 0.5 * S[has_obs] * np.exp(-x[has_obs])))
    lp += float(np.sum(-0.5 * x[0] ** 2 / stat_var))

    Z = (np.exp(-0.5 * x) * b) @ Qinv_half               # (T, r): row t = Q^{-1/2} a_t
    drift = cdrift[None, :] * Z[1:] * has_obs[1:, None]  # no drift where no shock
    resid = x[1:] - (phi[None, :] * x[:-1] + drift)
    v_in = np.where(has_obs[1:, None], var_lev[None, :], sigma2[None, :])
    lp += float(np.sum(-0.5 * resid * resid / v_in))
    return lp


def _lin_gauss_approx(x_hat, S_k, e_k, has_obs, phi, sigma2, rho):
    r"""
    Linear-Gaussian approximation of Branch A's scalar conditional for one factor,
    expanded at ``x_hat``.  Returns ``(y_eff, V_eff, mask, G, c, W, stat_var)`` in
    the convention of :func:`mcmc.sample_leverage_lagged._ffbs_tv` (index ``t`` is
    the transition ``t -> t+1``).

    **Level.**  ``g(x) = -x/2 - S e^{-x}/2`` is concave; its second-order expansion
    is a Gaussian pseudo-observation with precision ``-g'' = S e^{-x_hat}/2`` and
    pseudo-datum ``x_hat + g'/(-g'')`` — the textbook Gaussian approximation to the
    log-chi^2 measurement.

    **Drift.**  ``rho sigma e_t e^{-x_t/2}`` is linearised in ``x_t``:
    with ``D_t = rho sigma e_t e^{-x_hat_t/2}``,

        drift(x_t) ~= D_t (1 + x_hat_t/2) - (D_t/2) x_t,

    so the transition ``x_t = phi x_{t-1} + drift(x_t) + N(0, v)`` becomes

        A_t x_t = phi x_{t-1} + C_t + N(0, v),   A_t = 1 + D_t/2,
                                                 C_t = D_t (1 + x_hat_t/2)

    i.e. an AR(1) with time-varying coefficient ``phi/A_t``, intercept ``C_t/A_t``
    and variance ``v/A_t^2``.  The proposal only needs ``A_t`` to stay away from 0;
    where it does not (a violent shock against a near-1 ``|rho|``) we fall back to
    the no-leverage transition — a *worse proposal*, never a wrong target.

    The cross-factor coupling of ``Q^{-1/2}`` is deliberately **left out** here: it
    would make the proposal a ``(T x r)`` Gaussian for a gain the Metropolis ratio
    already guarantees.  Exact at diagonal ``Q``, a good proposal near it — which is
    where the real panel lives (``corr(Q)`` max off-diagonal 0.099).
    """
    T = x_hat.shape[0]
    sig = float(np.sqrt(sigma2))
    v_lev = float(sigma2 * (1.0 - rho * rho))
    stat_var = float(sigma2 / (1.0 - phi * phi))

    # ── level: Gaussian pseudo-observation ───────────────────────────────────
    mask = has_obs & (S_k > 0.0)
    prec = np.zeros(T)
    prec[mask] = 0.5 * S_k[mask] * np.exp(-x_hat[mask])
    V_eff = np.full(T, np.inf)
    y_eff = np.zeros(T)
    ok = mask & (prec > 1e-12)
    V_eff[ok] = 1.0 / prec[ok]
    gprime = -0.5 + 0.5 * S_k[ok] * np.exp(-x_hat[ok])
    y_eff[ok] = x_hat[ok] + gprime / prec[ok]
    mask = ok

    # ── transition: linearised leverage drift ────────────────────────────────
    D = rho * sig * e_k * np.exp(-0.5 * x_hat)            # (T,)
    A = 1.0 + 0.5 * D
    C = D * (1.0 + 0.5 * x_hat)
    v_t = np.where(has_obs, v_lev, sigma2)                # variance of the transition INTO t

    G = np.full(T, float(phi))
    c = np.zeros(T)
    W = np.full(T, float(sigma2))
    for t in range(T - 1):                                # transition t -> t+1
        tn = t + 1
        if has_obs[tn] and abs(A[tn]) > _A_MIN:
            G[t] = phi / A[tn]
            c[t] = C[tn] / A[tn]
            W[t] = v_t[tn] / (A[tn] * A[tn])
        else:
            G[t] = phi
            c[t] = 0.0
            W[t] = v_t[tn]
    return y_eff, V_eff, mask, G, c, W, stat_var


def _tv_filter(y_eff, V_eff, mask, G, c, W, stat_var):
    """Forward Kalman filter of the scalar linear-Gaussian approximation."""
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
            Sv = P_pred + V_eff[t]
            K = P_pred / Sv
            a[t] = a_pred + K * (y_eff[t] - a_pred)
            P[t] = (1.0 - K) * P_pred
        else:
            a[t], P[t] = a_pred, P_pred
    return a, P


def _tv_backward(a, P, G, c, W, rng=None):
    """Backward pass: RTS smoothed **mean** (``rng=None``) or one FFBS **draw**."""
    T = a.shape[0]
    x = np.zeros(T)
    x[T - 1] = a[T - 1]
    if rng is not None:
        x[T - 1] += np.sqrt(max(P[T - 1], 0.0)) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        P_pred_next = G[t] * G[t] * P[t] + W[t]
        J = G[t] * P[t] / P_pred_next
        m = a[t] + J * (x[t + 1] - (G[t] * a[t] + c[t]))
        x[t] = m
        if rng is not None:
            V = P[t] * (1.0 - J * G[t])
            x[t] += np.sqrt(max(V, 0.0)) * rng.standard_normal()
    return x


def _lin_gauss_logdens(x, y_eff, V_eff, mask, G, c, W, stat_var):
    r"""``log q(x)`` of the linear-Gaussian approximation, up to the constant
    ``-log p(y_eff)`` — which is the SAME for both paths in the Metropolis ratio
    (the proposal is an independence proposal: one approximation, two evaluations),
    so it cancels and is not computed."""
    lp = -0.5 * (np.log(stat_var) + x[0] * x[0] / stat_var)
    m = G[:-1] * x[:-1] + c[:-1]
    d = x[1:] - m
    lp += float(np.sum(-0.5 * (np.log(W[:-1]) + d * d / W[:-1])))
    if np.any(mask):
        dm = y_eff[mask] - x[mask]
        lp += float(np.sum(-0.5 * (np.log(V_eff[mask]) + dm * dm / V_eff[mask])))
    return float(lp)


def _lev_path_laplace_mh_common(
    logh_u: np.ndarray,          # (T, r) current paths
    b: np.ndarray,               # (T, r) sqrt(w) u
    S: np.ndarray,               # (T, r) e^2
    E: np.ndarray,               # (T, r) signed e_{k,t} = sqrt(w/q_kk) u_k
    has_obs: np.ndarray,         # (T,)
    Qinv_half: np.ndarray,       # (r, r)
    phi: np.ndarray,
    sigma2: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
    *,
    n_newton: int = 10,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int, int]:
    r"""
    Branch A's block path draw: per factor, an **independence** Metropolis move whose
    proposal is the mode-centred Laplace approximation of :func:`_lin_gauss_approx`,
    accepted against the exact coupled target :func:`_logpost_A_common`.

    Per factor ``k``:
      1. iterate ``x_hat <- smoothed mean of the approximation built at x_hat`` from
         the FIXED start ``x_hat = 0`` (Durbin-Koopman mode finding).  Starting from
         a constant — not from the current draw — is what makes the proposal an
         *independence* proposal: ``q`` depends on the data and the parameters only,
         so the reverse density needs no second linearisation and the ratio is just
         ``q(x_cur)/q(x_prop)``.  It is also what retires the flat-start trap (P3):
         the very first proposal is already a sensible path, so no KSC warm seed is
         needed;
      2. draw the whole path from ``q`` by FFBS;
      3. accept with the EXACT Branch-A target, other factors held fixed.

    Returns ``(logh_new (T, r), n_accept, n_propose)`` with one accept/reject per
    factor per sweep (so ``n_propose = r``, not ``T*r`` — read the acceptance rate
    accordingly: it is a *block* acceptance).
    """
    T, r = logh_u.shape
    x = np.asarray(logh_u, float).copy()
    n_acc = 0

    for k in range(r):
        # ── 1. mode of the approximation, from a fixed start (independence) ───
        x_hat = np.zeros(T)
        approx = None
        for _ in range(n_newton):
            approx = _lin_gauss_approx(x_hat, S[:, k], E[:, k], has_obs,
                                       float(phi[k]), float(sigma2[k]), float(rho[k]))
            a, P = _tv_filter(*approx)
            x_new_hat = _tv_backward(a, P, approx[3], approx[4], approx[5])
            if np.max(np.abs(x_new_hat - x_hat)) < tol:
                x_hat = x_new_hat
                break
            x_hat = x_new_hat
        approx = _lin_gauss_approx(x_hat, S[:, k], E[:, k], has_obs,
                                   float(phi[k]), float(sigma2[k]), float(rho[k]))
        y_eff, V_eff, mask, G, c, W, stat_var = approx
        a, P = _tv_filter(*approx)

        # ── 2. propose the whole path ─────────────────────────────────────────
        x_prop = _tv_backward(a, P, G, c, W, rng=rng)

        # ── 3. accept against the EXACT target (other factors fixed) ──────────
        x_cur_k = x[:, k].copy()
        lp_cur = _logpost_A_common(x, b, S, has_obs, Qinv_half, phi, sigma2, rho)
        x[:, k] = x_prop
        lp_prop = _logpost_A_common(x, b, S, has_obs, Qinv_half, phi, sigma2, rho)

        lq_cur = _lin_gauss_logdens(x_cur_k, y_eff, V_eff, mask, G, c, W, stat_var)
        lq_prop = _lin_gauss_logdens(x_prop, y_eff, V_eff, mask, G, c, W, stat_var)

        if np.log(rng.random()) < (lp_prop - lp_cur) + (lq_cur - lq_prop):
            n_acc += 1                          # keep x[:, k] = x_prop
        else:
            x[:, k] = x_cur_k
    return x, n_acc, r


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


def _draw_sigma2_lev(x, zeta, has_obs, phi, rho2, sigma2_cur, a_sig, b_sig, prop_sd, rng,
                     *, sigma_prior="inverse_gamma", half_normal_B=1.0):
    r"""RW-Metropolis draw of ``sigma^2`` (on ``log sigma^2``) under leverage.

    Target: the ``sigma_eta`` prior times the leverage likelihood, where on
    leverage transitions the residual is ``x_t - phi x_{t-1} - sqrt(v) zeta_t``
    with variance ``v(1-rho^2)``, and on non-leverage transitions
    ``x_t - phi x_{t-1}`` with variance ``v``.

    ``sigma_prior`` selects the prior kernel in ``v = sigma^2``:
    ``"inverse_gamma"`` -> ``IG(a_sig,b_sig)`` (``-(a+1)log v - b/v``, the conjugate
    baseline); ``"half_normal"`` -> half-Normal on ``sigma_eta ~ N(0,B)``,
    ``B=half_normal_B`` (``-0.5 log v - v/(2B)``, incl.\ the ``sigma_eta->v``
    Jacobian).  Both are combined with the same leverage likelihood, so the
    ``(1-rho^2)`` variance factor is honoured either way.
    """
    T = x.shape[0]
    eta = x[1:] - phi * x[:-1]                # (T-1,)
    obs = has_obs[1:]
    zt = zeta[1:]
    inv_2B = 0.5 / half_normal_B

    def _logp(v):
        if v <= 0:
            return -np.inf
        if sigma_prior == "half_normal":
            lp = -0.5 * np.log(v) - inv_2B * v            # half-Normal kernel (in v)
        else:
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

    .. warning::
       **Mixes very badly** (``docs/audit_P1-P5.md`` §P6): one move per sweep from a
       fixed proposal, anchored at the current value, against a posterior ridge with
       the log-vol path and ``sigma_eta^2``.  Measured efficiency ``ESS/draw``
       ``0.35–1.25%`` on the per-factor DGP; widening ``prop_sd`` does **not** help
       (acceptance collapses, ESS does not rise).  Retained for Branch~A and as the
       comparison baseline; the default on Branch~B is
       :func:`draw_rho_griddy`.
    """
    n_lev = eta.shape[0]
    rs = rho_cur + prop_sd * rng.standard_normal()
    cur = _rho_logpost_scalar(rho_cur, eta, k, n_lev, sigma2)
    new = _rho_logpost_scalar(rs, eta, k, n_lev, sigma2)
    if np.log(rng.random()) < new - cur:
        return float(rs), 1
    return float(rho_cur), 0


def _rho_logpost_grid(grid, eta, k, n_lev, sigma2):
    r"""The Family~C log-posterior of :func:`_rho_logpost_scalar`, evaluated on a whole
    grid at once (``(G, n_lev)`` work, one pass).  Kept separate so that the scalar
    form stays the single readable statement of the target."""
    grid = np.asarray(grid, float)
    om = 1.0 - grid * grid                                   # (G,)
    res = eta[None, :] - grid[:, None] * k[None, :]          # (G, n_lev)
    ssr = np.einsum("gt,gt->g", res, res)                    # (G,)
    return -0.5 * n_lev * np.log(om) - 0.5 * ssr / (sigma2 * om)


def draw_rho_griddy(rho_cur, eta, k, sigma2, rng, *, grid_size=401, log_prior=None,
                    eps=1e-6):
    r"""
    **Griddy-Gibbs** draw of a scalar leverage ``rho`` from its full conditional
    (``eq:param-rho-cond``), on the compact support ``(-1, 1)``.

    The fix for P6 (``docs/audit_P1-P5.md``, ``docs/fix_P6_map.md``).  The
    RW-Metropolis :func:`draw_rho_scalar` anchors every proposal at the current value
    and takes **one** step per sweep, so it random-walks along the ``rho``--path--
    ``sigma_eta`` ridge at ``ESS/draw ~ 0.5%``.  A griddy draw is **independent of the
    current value**: the walk disappears.

    Same pattern as :func:`mcmc.sample_params.draw_nu_griddy` — evaluate the
    un-normalised log-target on a grid, stabilise by its max, weight each point by its
    cell width (trapezoidal), normalise, sample — with **one** deliberate difference:

    * ``nu`` lives on the unbounded ``(2, infty)``, so its grid is **geometric** and
      the *log-concavity* of its target (a thesis result) is what justifies a coarse
      grid;
    * ``rho`` lives on the **compact** ``(-1, 1)``, so its grid is **uniform**, and no
      log-concavity is needed — nor does it hold.  The first term of the target,
      ``-(n/2) log(1 - rho^2)``, has second derivative ``n(1+rho^2)/(1-rho^2)^2 > 0``:
      it is **convex**.  Empirically the full target is unimodal in the informative
      regime, but on 400 random draws of ``(n_lev, sigma^2, scales)`` about 0.8% are
      non-concave and one was bimodal.  A fine grid on a bounded support resolves any
      of those; a log-concavity argument copied from ``nu`` would be wrong.

    Parameters
    ----------
    rho_cur : float       accepted and **ignored** (the draw is independent of it);
                          kept so the call sites are interchangeable with the RW.
    eta, k : (n_lev,)     as in :func:`draw_rho_scalar`.
    sigma2 : float        current ``sigma_eta^2``.
    grid_size : int       uniform grid points on ``(-1+eps, 1-eps)``.
    log_prior : callable or None   ``rho -> log p(rho)``; ``None`` is the flat
                          Uniform(-1,1) the thesis adopts as default.  This is the hook
                          for the Fisher-``z`` shrinkage it names as the alternative,
                          without touching the kernel.

    Returns ``(rho_new, 1)`` — the trailing ``1`` mirrors the RW's acceptance flag; a
    griddy draw is always "accepted", exactly as Branch~B's FFBS path draw reports
    ``acc = 1.0``.  Acceptance is therefore *not* a mixing diagnostic here: read ESS.
    """
    n_lev = eta.shape[0]
    if n_lev == 0:
        return float(rho_cur), 1
    grid = np.linspace(-1.0 + eps, 1.0 - eps, grid_size)
    logp = _rho_logpost_grid(grid, eta, k, n_lev, sigma2)
    if log_prior is not None:
        logp = logp + np.array([float(log_prior(g)) for g in grid])
    logp -= logp.max()
    dens = np.exp(logp)
    probs = dens * np.gradient(grid)              # trapezoidal cell weights
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0.0, 1                             # degenerate target -> no leverage
    probs /= total
    return float(rng.choice(grid, p=probs)), 1


def draw_rho(rho_cur, eta, k, sigma2, rng, *, sampler="griddy", prop_sd=0.06,
             grid_size=401, log_prior=None):
    """Dispatch the Family~C draw: ``"griddy"`` (default, P6 fix) or ``"rw"`` (the
    RW-Metropolis baseline, kept for the GATE-3 comparison and for Branch~A)."""
    if sampler == "griddy":
        return draw_rho_griddy(rho_cur, eta, k, sigma2, rng,
                               grid_size=grid_size, log_prior=log_prior)
    if sampler == "rw":
        return draw_rho_scalar(rho_cur, eta, k, sigma2, prop_sd, rng)
    raise ValueError(f"rho_sampler={sampler!r}: 'griddy' or 'rw'.")


# NB: the vector-rho machinery (draw_rho_vec / draw_rho_common / dominant_dir_z,
# the rho'rho<1 region and the dominant-direction scalar specialisation) is
# **removed** (Phase 7).  Under Option A the common leverage is r *independent
# scalar* correlations (draw_rho_scalar per factor, eq:lev-cond-common) in both
# Branch A and Branch B — there is no vector draw and no rho'rho<1 region.


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: step (b) + Family B + Family C for all M+r processes (Branch A)
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
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    use_asis: bool = False,
    fix_mu0: bool = True,
    sv_idio: bool = True,
    prop_path: float = 0.25,
    prop_sigma2: float = 0.20,
    prop_rho: float = 0.06,
    lev_path_sampler: str = "single",
    rho_sampler: str = "rw",
    rho_grid_size: int = 401,
    rho_log_prior=None,
    inv_sqrt_spd=None,
    **_ignored,
) -> dict:
    r"""
    Contemporaneous-leverage volatility + leverage-parameter sampler (Branch A).

    Sweeps, for the common factor and each idiosyncratic series: (b) the
    single-move Metropolis log-vol path; Family~B ``(phi, sigma^2)`` (leverage
    aware); Family~C ``rho`` (Metropolis).  ``mu = 0`` throughout (the
    identification convention ``eq:sv-mu-identification``): the leverage AR(1)
    regressions carry **no intercept**, so ``mu`` is structurally pinned at 0 and
    every returned ``sv`` row is ``(0, phi, sigma2)``.  ``fix_mu0=False`` is not
    supported here (raises) — the leverage derivation assumes ``mu=0``.

    ``sigma_prior`` (``"inverse_gamma"`` baseline / ``"half_normal"``,
    ``B=half_normal_B``) selects the Family~B ``sigma_eta`` prior, threaded to the
    leverage-aware :func:`_draw_sigma2_lev` (the ``(1-rho^2)`` variance factor is
    honoured either way).

    ``sv_idio=False`` is the **D2-a** restriction (``subsec:variants-restrictions``):
    the idiosyncratic volatilities are frozen at ``h^eps ≡ 1`` and neither their
    Family~B nor their Family~C (``rho_eps``) is drawn — with no ``h^eps`` there is
    no idiosyncratic log-vol innovation for a leverage correlation to attach to.

    Returns ``h_u, h_eps, logh_u, logh_eps, sv_u, sv_eps, rho_u, rho_eps`` and the
    Metropolis acceptance rates ``acc`` (dict).
    """
    if not fix_mu0:
        raise ValueError("sample_volatility_block_leverage supports mu=0 only "
                         "(fix_mu0=True); the leverage AR(1) has no intercept.")
    if use_asis:
        if sigma_prior != "half_normal":
            raise ValueError("use_asis=True requires sigma_prior='half_normal' "
                             "(CP and NCP must share the Gaussian prior on sigma_eta).")
        from mcmc.sample_asis import asis_scale_interweave      # lazy: avoid import cycle
    from mcmc.sample_vol import _inv_sqrt_spd as _isqrt
    if inv_sqrt_spd is None:
        inv_sqrt_spd = _isqrt

    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    T, M = Y.shape
    r = A.shape[0]
    F = f_aug[:, :r]

    acc = {"path_u": 0.0, "path_eps": 0.0, "sigma2": 0.0, "rho_u": 0.0, "rho_eps": 0.0}

    # ── Common factor: r per-factor volatilities under Option A (Spec II) ──────
    # Two distinct whitenings (subsec:lev-attach-choice, caution at eq ~16368):
    #   * measurement (decoupled, per-component): e_{k,t}=sqrt(w/q_kk) u_{k,t};
    #   * raw shock (the leverage target): z^u_t=sqrt(w) Q^{-1/2}(sqrt H^u_t)^{-1}u_t
    #     with the FULL symmetric Q^{-1/2} — mixes every factor's volatility.
    Qinv_half = inv_sqrt_spd(Q)                            # Q^{-1/2} (symmetric)
    qdiag = np.diag(Q)
    u = F[1:] - F[:-1] @ A.T                               # (T-1, r), u_t
    has_u = np.zeros(T, bool); has_u[1:] = True
    b_u = np.zeros((T, r)); b_u[1:] = np.sqrt(w_u[1:])[:, None] * u   # sqrt(w) u_t
    S_u = np.zeros((T, r))                                 # per-component level SS
    S_u[1:] = (w_u[1:][:, None] / qdiag[None, :]) * u ** 2

    logh_u = np.asarray(logh_u, float)
    sv_u = np.asarray(sv_u, float)                         # (r, 3)
    rho_u = np.asarray(rho_u, float).copy()                # (r,)
    phi_u = sv_u[:, 1].copy(); s2_u = sv_u[:, 2].copy()

    if lev_path_sampler not in ("single", "laplace"):
        raise ValueError(f"lev_path_sampler={lev_path_sampler!r}: 'single' or 'laplace'.")

    if lev_path_sampler == "single":
        # Warm-start the path from a blocked KSC-FFBS draw when it enters flat (the
        # warm start's log h == 0).  The per-factor single-move Metropolis mixes far
        # more slowly than the blocked KSC-FFBS and, coupled to the state feedback,
        # can otherwise sit in the flat-init trap (h ~ 1 -> homoskedastic states ->
        # u ~ homoskedastic -> h ~ 1).  Seeding the sweep with the exact rho=0 path
        # (sample_common_vol_mv, the Spec II common block) leaves the target
        # unchanged and lets the subsequent Metropolis explore around a sensible path.
        # NB: the seed is a patch on the *initialisation*; 'laplace' does not need it
        # (its proposal is mode-centred, so the first sweep already lands on a
        # sensible path) — audit P3.
        if not np.any(np.abs(logh_u) > 1e-9):
            from mcmc.sample_vol import sample_common_vol_mv
            seed_sv = np.column_stack([np.zeros(r), phi_u, s2_u])
            logh_u = sample_common_vol_mv(u, Q, w_u, logh_u, seed_sv, rng,
                                          offset=1e-6)["logh_u"]

        # (b) coupled multivariate single-move path draw (drift keys on the raw shock)
        logh_u_new, na, npp = _lev_path_mh_mv_common(
            logh_u, b_u, S_u, has_u, Qinv_half, phi_u, s2_u, rho_u, prop_path, rng)
    else:
        # (b') block draw: Laplace proposal, exact Branch-A acceptance (audit P2/P3)
        E_u = b_u / np.sqrt(qdiag)[None, :]                 # signed e_{k,t}; S_u = E_u^2
        logh_u_new, na, npp = _lev_path_laplace_mh_common(
            logh_u, b_u, S_u, E_u, has_u, Qinv_half, phi_u, s2_u, rho_u, rng)
    acc["path_u"] = na / npp

    # raw shocks at the new path: z^u_t = Q^{-1/2} (exp(-x_t/2) * b_t)  (T, r)
    a_u = np.exp(-0.5 * logh_u_new) * b_u                  # (T, r); 0 at t=0
    z_u = a_u @ Qinv_half                                  # (T, r): row t = Q^{-1/2} a_t
    y_star_u = np.log(S_u + 1e-6)                          # KSC log-square for ASIS (T, r)

    # Family B + Family C per factor — r independent scalar channels (Option A,
    # eq:lev-cond-common): no vector draw, no rho'rho<1 region.
    sv_u_new = np.zeros((r, 3))
    rho_u_new = np.zeros(r)
    acc_s_u = 0.0; acc_r_u = 0.0
    if not use_asis:
        # single interleaved pass (phi, sigma2, rho per factor) — the raw shock z_u
        # is fixed (no path rescale), so this is the plain Option A block.
        for k in range(r):
            rho_k = float(rho_u[k]); rho2_k = rho_k * rho_k
            zeta_k = rho_k * z_u[:, k]                          # rho_k z^u_{k,t} (T,)
            ph_k = _draw_phi_lev(logh_u_new[:, k], zeta_k, has_u, s2_u[k], rho2_k, rng)
            s2_k, a1 = _draw_sigma2_lev(logh_u_new[:, k], zeta_k, has_u, ph_k, rho2_k,
                                        s2_u[k], prior_a, prior_b, prop_sigma2, rng,
                                        sigma_prior=sigma_prior, half_normal_B=half_normal_B)
            eta_k = logh_u_new[1:, k] - ph_k * logh_u_new[:-1, k]
            k_reg = np.sqrt(s2_k) * z_u[1:, k]                 # k_t = sigma_k z^u_{k,t}
            rho_k, a_rk = draw_rho(rho_k, eta_k, k_reg, s2_k, rng,
                                   sampler=rho_sampler, prop_sd=prop_rho,
                                   grid_size=rho_grid_size, log_prior=rho_log_prior)
            sv_u_new[k] = (0.0, ph_k, s2_k); rho_u_new[k] = rho_k
            acc_s_u += a1; acc_r_u += a_rk
    else:
        # ASIS interweave rescales the path per factor, and z_u couples all factors
        # through Q^{-1/2}, so the sweep is two-pass: (1) CP Family B + ASIS with z_u
        # FROZEN (the separable leverage stance, subsec:asis-leverage); recompute z_u
        # at the rescaled path; (2) Family C rho on the current path.
        for k in range(r):
            rho_k = float(rho_u[k]); rho2_k = rho_k * rho_k
            zeta_k = rho_k * z_u[:, k]
            ph_k = _draw_phi_lev(logh_u_new[:, k], zeta_k, has_u, s2_u[k], rho2_k, rng)
            s2_k, a1 = _draw_sigma2_lev(logh_u_new[:, k], zeta_k, has_u, ph_k, rho2_k,
                                        s2_u[k], prior_a, prior_b, prop_sigma2, rng,
                                        sigma_prior=sigma_prior, half_normal_B=half_normal_B)
            x_a, ph_k, s2_k = asis_scale_interweave(          # (2)-(4) NCP interweave
                logh_u_new[:, k], y_star_u[:, k], has_u, s2_k, rho_k, z_u[:, k], rng,
                half_normal_B=half_normal_B)
            logh_u_new[:, k] = x_a
            sv_u_new[k] = (0.0, ph_k, s2_k); acc_s_u += a1
        a_u = np.exp(-0.5 * logh_u_new) * b_u                  # recompute z_u at rescaled path
        z_u = a_u @ Qinv_half
        for k in range(r):
            ph_k = float(sv_u_new[k, 1]); s2_k = float(sv_u_new[k, 2])
            eta_k = logh_u_new[1:, k] - ph_k * logh_u_new[:-1, k]
            k_reg = np.sqrt(s2_k) * z_u[1:, k]
            rho_k, a_rk = draw_rho(float(rho_u[k]), eta_k, k_reg, s2_k, rng,
                                   sampler=rho_sampler, prop_sd=prop_rho,
                                   grid_size=rho_grid_size, log_prior=rho_log_prior)
            rho_u_new[k] = rho_k; acc_r_u += a_rk
    acc["sigma2"] = acc_s_u                                # accepts over the r factors
    acc["rho_u"] = acc_r_u / max(1, r)
    rho_u = rho_u_new

    # ── Idiosyncratic series (omitted, h^eps frozen at 1, under D2-a) ─────────
    signal = F @ Lambda.T
    logh_eps_new = np.zeros((T, M))
    sv_eps_new = np.asarray(sv_eps, float).copy() if not sv_idio else np.zeros((M, 3))
    rho_eps_new = np.asarray(rho_eps, float).copy() if not sv_idio else np.zeros(M)
    acc_pe = 0.0; acc_s = 0.0; acc_re = 0.0
    M_lev = M if sv_idio else 0
    for i in range(M_lev):
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

        z_i = e_full * np.exp(-0.5 * lh_i)                    # z_t (T,), frozen for CP+ASIS
        zeta_i = rho_i * z_i
        phi_i = _draw_phi_lev(lh_i, zeta_i, has_i, s2_i, rho2_i, rng)
        s2_i, a_si = _draw_sigma2_lev(lh_i, zeta_i, has_i, phi_i, rho2_i, s2_i,
                                      prior_a, prior_b, prop_sigma2, rng,
                                      sigma_prior=sigma_prior, half_normal_B=half_normal_B)
        if use_asis:                                          # (2)-(4) NCP interweave
            y_star_i = np.zeros(T); y_star_i[obs_t] = np.log(S_i[obs_t] + 1e-6)
            x_a, phi_i, s2_i = asis_scale_interweave(
                lh_i, y_star_i, has_i, s2_i, rho_i, z_i, rng, half_normal_B=half_normal_B)
            lh_i = x_a
            z_i = e_full * np.exp(-0.5 * lh_i)                # recompute for Family C
        # rho on leverage transitions (t>=1 and observed)
        lev_mask = has_i[1:]
        eta_i = (lh_i[1:] - phi_i * lh_i[:-1])[lev_mask]
        k_i = (np.sqrt(s2_i) * z_i[1:])[lev_mask]
        rho_i, a_ri = draw_rho(rho_i, eta_i, k_i, s2_i, rng,
                               sampler=rho_sampler, prop_sd=prop_rho,
                               grid_size=rho_grid_size, log_prior=rho_log_prior)
        acc_s += a_si; acc_re += a_ri

        logh_eps_new[:, i] = lh_i
        sv_eps_new[i] = (0.0, phi_i, s2_i)
        rho_eps_new[i] = rho_i

    acc["sigma2"] = (acc["sigma2"] + acc_s) / (r + M_lev)   # r common + M idio draws
    if M_lev > 0:
        acc["path_eps"] = acc_pe / M_lev
        acc["rho_eps"] = acc_re / M_lev

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
        "rho_u": rho_u, "rho_eps": rho_eps_new,
        "acc": acc,
    }
