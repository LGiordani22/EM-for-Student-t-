"""
src/mcmc/sample_vol.py
======================

SISTEMA (equazioni dal .tex — notazione originale)
--------------------------------------------------
Cosa calcola: le M+r traiettorie di log-volatilità log h_{1:T} e i loro parametri
AR(1) (φ, σ_η²), caso base SENZA leverage (ρ=0).  Ogni processo di volatilità scala
un residuo:

    fattore i:  u_t = sqrt(H^u_t) Q^{1/2} z^u_t / sqrt(w^u_t)     [eq:vol-outside, Spec. II]
    serie i:    ε_{i,t} = sqrt(h^ε_{i,t} r_i / w^ε_t) · z^ε_{i,t}

Sbiancato e log-quadrato (offset c) → LINEARE nello stato log h:

    y*_t = log(e_t² + c) = log h_t + log(z_t²)

L'errore log-χ²_1 è approssimato dalla mistura a 7 componenti KSC (constants.KSC7);
condizionatamente all'indicatore di componente s il sistema è lineare-gaussiano →
FFBS scalare.  Transizione (μ=0):

    log h_t = φ log h_{t-1} + η_t,   η_t ~ N(0, σ_η²)             [eq:sv-logvol-u/eps]

Specification II: H^u_t = diag(h^u_{1,t}..h^u_{r,t}) fuori dal sandwich Q^{1/2}, così
Var(u_t|·) = sqrt(H^u_t) Q sqrt(H^u_t)/w^u_t e h^u_{i,t} è la vol del fattore i.

Gibbs step (b), **base case** (KSC, no leverage): sample the M+r stochastic
log-volatility paths and their AR(1) parameters, given the states, the tail
weights and the parameters.  Specification II (``eq:vol-outside``): r per-factor
common volatilities + M idiosyncratic.  The legacy :func:`sample_volatility_block`
is the **scalar-common restriction** ``H^u_t = h^u_t I`` (where the inside/outside
placements coincide, so it selects no specification) — a test seam, not a model.

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

from mcmc.constants import KSC7, LOG_CHI2_MEAN


# ─────────────────────────────────────────────────────────────────────────────
# Cross-covariance of the multivariate log-squares (Spec II common block)
# ─────────────────────────────────────────────────────────────────────────────
#
# Under Spec II the per-factor log-square residuals are ``xi_bar_k = log zeta_bar_k^2``
# with ``zeta_bar ~ N(0, corr(Q))`` (subsec:vol-all-processes).  Their correlation
# matrix ``R_xi`` is the *genuine* correlation matrix of that derived vector, so it
# is PSD by construction; its (j,k) entry is a fixed function of the underlying
# Gaussian correlation ``rho = corr(Q)_jk``:
#     R_xi[j,k] = g(rho) / (pi^2/2),   g(rho) = Cov(log X^2, log Y^2),
# for a standard bivariate normal (X,Y) of correlation rho (g(0)=0, g(+-1)=pi^2/2).
# We tabulate ``R_xi(rho)`` once by common-random-number Monte Carlo (deterministic,
# smooth in rho) and interpolate.

_LOGSQ_TABLE: tuple[np.ndarray, np.ndarray] | None = None


def _build_logsq_corr_table(n_grid: int = 161, n_mc: int = 3_000_000,
                            seed: int = 20260708) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_mc)
    W = rng.standard_normal(n_mc)
    lX = np.log(X * X); lX -= lX.mean()
    var = float(np.mean(lX * lX))                     # ~ pi^2/2
    rhos = np.linspace(-0.999, 0.999, n_grid)
    corr = np.empty(n_grid)
    for i, rho in enumerate(rhos):
        Y = rho * X + np.sqrt(1.0 - rho * rho) * W
        lY = np.log(Y * Y); lY -= lY.mean()
        corr[i] = float(np.mean(lX * lY)) / var
    return rhos, corr


def logsq_corr(rho: np.ndarray) -> np.ndarray:
    """``Corr(log X^2, log Y^2)`` for a std bivariate normal of correlation ``rho``
    (vectorised, via the cached table)."""
    global _LOGSQ_TABLE
    if _LOGSQ_TABLE is None:
        _LOGSQ_TABLE = _build_logsq_corr_table()
    grid, val = _LOGSQ_TABLE
    return np.interp(rho, grid, val)


def logsq_corr_matrix(Q: np.ndarray) -> np.ndarray:
    """The log-square correlation matrix ``R_xi`` implied by ``corr(Q)``
    (subsec:vol-all-processes).  PSD by construction; 1 on the diagonal."""
    Q = np.asarray(Q, float)
    d = np.sqrt(np.clip(np.diag(Q), 1e-300, None))
    C = Q / np.outer(d, d)                             # corr(Q)
    R = np.asarray(logsq_corr(np.clip(C, -0.999, 0.999)), float)
    np.fill_diagonal(R, 1.0)
    return 0.5 * (R + R.T)


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
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    sigma2_cur: float | None = None,
    prop_log_sigma: float = 0.2,
    max_tries: int = 50,
) -> tuple[float, float, float]:
    r"""
    Draw the AR(1) parameters of a log-volatility process from a conjugate
    regression on the sampled path, with a weak inverse-gamma prior on
    ``sigma2`` and stationarity ``|phi|<1`` enforced by rejection.

    Prior on ``sigma_eta`` (``sigma_prior``, Family~B, thesis
    ``subsec:param-familyB`` + delicate case (a) of ``subsec:param-prior-tuning``)
    ---------------------------------------------------------------------------
    * ``"inverse_gamma"`` (default, the exact **conjugate baseline** used when
      interweaving is off): ``sigma2 ~ IG(a_sig, b_sig)``, drawn directly
      (``eq:param-logvol``).
    * ``"half_normal"`` (Gelman 2006; the ASIS-conjugate choice — the master
      sampler's recommended prior, turned on with the interweaving of Phase 6):
      a half-Normal on the **standard deviation** ``sigma_eta ~ N(0, B)``,
      ``B = half_normal_B``.  This is non-conjugate on the scale, so ``sigma2``
      is drawn by a **light random-walk Metropolis** on ``log sigma2`` (the
      "light Metropolis correction the Family~B update already tolerates"),
      requiring the current ``sigma2_cur`` as the RW anchor; ``phi`` is then the
      Gaussian conditional truncated to ``|phi|<1``.  In the density of
      ``v = sigma2`` the half-Normal prior (with the ``sigma_eta -> v`` Jacobian)
      is ``p(v) ∝ v^{-1/2} exp(-v / (2B))``.  Only supported under ``fix_mu0``
      (the ``mu = 0`` identification the half-Normal derivation assumes).

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
    is parametrisable; the same prior is used for all M+r processes.
    """
    x = np.asarray(logh, float)
    y = x[1:]
    n = y.size

    if sigma_prior == "half_normal":
        # Half-Normal on sigma_eta (Gelman 2006 / ASIS): light RW-Metropolis on
        # log sigma2, mu=0 (fix_mu0) only.  Valid Gibbs order: phi | sigma2_cur,
        # then sigma2 | phi_new.
        if not fix_mu0:
            raise ValueError("sigma_prior='half_normal' requires fix_mu0=True "
                             "(the mu=0 identification the half-Normal assumes).")
        if sigma2_cur is None or sigma2_cur <= 0:
            raise ValueError("sigma_prior='half_normal' needs a positive sigma2_cur "
                             "(the RW-Metropolis anchor).")
        xp = x[:-1]
        xtx = float(xp @ xp) + 1e-12
        phi_hat = float((xp @ y) / xtx)
        # phi | sigma2_cur  (Gaussian, truncated |phi|<1)
        phi = float(np.clip(phi_hat, -0.98, 0.98))
        for _ in range(max_tries):
            cand = phi_hat + np.sqrt(sigma2_cur / xtx) * rng.standard_normal()
            if abs(cand) < 0.999:
                phi = float(cand)
                break
        # sigma2 | phi  via RW-MH on log v, target = Gaussian likelihood x
        # half-Normal-in-v prior  p(v) ∝ v^{-1/2} exp(-v/(2B)).
        resid = y - phi * xp
        ssr = float(resid @ resid)
        inv_2B = 0.5 / half_normal_B

        def _logp_v(v: float) -> float:
            if v <= 0:
                return -np.inf
            # likelihood v^{-n/2} exp(-ssr/2v) + half-Normal prior v^{-1/2} exp(-v/2B)
            return -(0.5 * n + 0.5) * np.log(v) - 0.5 * ssr / v - inv_2B * v

        log_v = np.log(sigma2_cur)
        log_vs = log_v + prop_log_sigma * rng.standard_normal()
        # RW on log v: +log v Jacobian on each side (target expressed in v)
        cur = _logp_v(sigma2_cur) + log_v
        new = _logp_v(float(np.exp(log_vs))) + log_vs
        sigma2 = float(np.exp(log_vs)) if np.log(rng.random()) < new - cur else float(sigma2_cur)
        return 0.0, phi, max(sigma2, 1e-8)

    if sigma_prior != "inverse_gamma":
        raise ValueError(f"sigma_prior={sigma_prior!r} not in {{'inverse_gamma','half_normal'}}")

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
# The full step (b): all M+r processes (Spec II: r per-factor common + M idio)
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
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    ksc: dict = KSC7,
) -> dict:
    r"""
    Sample the common + M idiosyncratic log-volatility paths and their AR(1)
    parameters (no leverage) under the **scalar-common restriction**
    ``H^u_t = h^u_t I``: one volatility shared by the r factors, read through the
    ``r`` whitened components ``Q^{-1/2} sqrt(w) u_t`` at once.

    This is *not* "Specification I".  The two specifications of
    ``subsec:vol-placement`` both give **each factor its own** ``h^u_{i,t}`` in a
    diagonal ``H^u_t``; they differ only in **where** it enters relative to the
    mixing ``Q^{1/2}`` — inside, ``Var(u_t) = Q^{1/2} H^u_t Q^{1/2}``
    (eq:vol-inside), or outside, ``Var(u_t) = sqrt(H^u_t) Q sqrt(H^u_t)``
    (eq:vol-outside, adopted).  Under the restriction ``H^u_t = h^u_t I`` used
    here the two placements **coincide** (both equal ``h^u_t Q``) and the fork
    disappears altogether.

    .. warning::
       **Not on any sampler path.**  Every SV path of ``fit_dfm_mcmc`` is now
       per-factor (Specification II); this block is retained solely as the
       **bit-exact ``r = 1`` seam** against which the per-factor
       :func:`sample_common_vol_mv` is validated (``test_shared`` [4c]), i.e. as
       a reference implementation, not as a variant of the model.  The scalar
       common volatility is *not* a cell of the D1 x D2 grid.

    ``sigma_prior`` (``"inverse_gamma"`` conjugate baseline / ``"half_normal"``
    with scale ``half_normal_B``) selects the Family~B ``sigma_eta`` prior
    threaded to :func:`draw_ar1_params`; under the half-Normal the current
    ``sigma2`` (from ``sv_u`` / ``sv_eps``) anchors the light RW-Metropolis.

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
                                        a_sig=prior_a, b_sig=prior_b,
                                        sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                                        sigma2_cur=float(s2_u)))

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
                                        a_sig=prior_a, b_sig=prior_b,
                                        sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                                        sigma2_cur=float(s2_i))

    return {
        "h_u": np.exp(logh_u_new), "h_eps": np.exp(logh_eps_new),
        "logh_u": logh_u_new, "logh_eps": logh_eps_new,
        "sv_u": sv_u_new, "sv_eps": sv_eps_new,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Specification II: the per-factor common-volatility block (multivariate)
# ─────────────────────────────────────────────────────────────────────────────

def _psd_chol(M: np.ndarray) -> np.ndarray:
    """Cholesky of ``M`` with a small eigenvalue floor (numerical PSD safeguard)."""
    M = 0.5 * (M + M.T)
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 1e-12, None)
        return np.linalg.cholesky((vecs * vals) @ vecs.T)


def _mv_ar1_ffbs(
    y_eff: np.ndarray,
    R_eff: np.ndarray,
    mask: np.ndarray,
    phi: np.ndarray,
    sigma2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Multivariate **diagonal**-AR(1) forward-filter / backward-sample (eq:ffbs-backward
    at state dimension ``r``).  The state ``x_t = log h^u_t`` follows ``r`` independent
    AR(1)s (``Phi = diag(phi)``, ``Q = diag(sigma2)``, ``mu = 0``,
    ``x_0 ~ N(0, diag(sigma2/(1-phi^2)))``); the measurement (where ``mask_t``) is
    ``y_eff_t = x_t + N(0, R_eff_t)`` with ``H = I`` and the **full** cross-covariance
    ``R_eff_t`` that couples the factors (subsec:vol-all-processes).  It is the
    coupling in the measurement — not the transition — that makes this one r-dim pass.
    """
    T = mask.shape[0]
    r = phi.shape[0]
    Phi = np.diag(phi)
    Qd = np.diag(sigma2)
    stat = np.diag(sigma2 / (1.0 - phi * phi))

    a = np.zeros((T, r))
    P = np.zeros((T, r, r))
    for t in range(T):
        if t == 0:
            a_pred, P_pred = np.zeros(r), stat
        else:
            a_pred = phi * a[t - 1]
            P_pred = (phi[:, None] * P[t - 1]) * phi[None, :] + Qd
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
        P_pred_next = (phi[:, None] * P[t]) * phi[None, :] + Qd
        J = P[t] @ Phi.T @ np.linalg.inv(P_pred_next)
        m = a[t] + J @ (x[t + 1] - phi * a[t])
        V = P[t] - J @ (Phi @ P[t])
        x[t] = m + _psd_chol(V) @ rng.standard_normal(r)
    return x


def sample_common_vol_mv(
    u_head: np.ndarray,
    Q: np.ndarray,
    w_u: np.ndarray,
    logh_u: np.ndarray,
    sv_u: np.ndarray,
    rng: np.random.Generator,
    *,
    offset: float = 1e-6,
    fix_mu0: bool = True,
    prior_a: float = 2.0,
    prior_b: float = 0.05,
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    use_asis: bool = False,
    coupling: str = "decoupled",
    R_xi: np.ndarray | None = None,
    allow_experimental: bool = False,
    ksc: dict = KSC7,
) -> dict:
    r"""
    Specification-II common-volatility block: the ``r`` **per-factor** log-volatility
    paths ``log h^u_{k,·}`` and their AR(1) parameters.

    ``use_asis`` (Phase 6, ``sec:asis``): wrap each per-factor Family~B draw with
    the ancillarity--sufficiency interweaving move (redraw ``(phi, sigma_eta^2)`` in
    the non-centred coordinates and rescale the path) to break the path/scale ridge.
    Requires ``sigma_prior='half_normal'`` (CP and NCP must share the σ_η prior).

    Theory: ``subsec:vol-all-processes`` (multivariate common block),
    ``subsec:vol-placement`` / ``eq:vol-outside`` (the sandwich
    ``Var(u_t|.) = sqrt(H^u_t) Q sqrt(H^u_t)/w^u_t``), ``eq:vol-logsquare``,
    ``eq:ffbs-backward``.

    Under Spec II the k-th component of the innovation obeys
    ``sqrt(w^u_t) u_{k,t} = sqrt(h^u_{k,t}) zeta_{k,t}`` with
    ``Var(zeta_{k,t}) = q_kk``, so the **per-component** standardized residual
    ``e_{k,t} = sqrt(w^u_t / q_kk) u_{k,t} ~ N(0, h^u_{k,t})`` and the measurement is
    ``y*_{k,t} = log(e_{k,t}^2 + c) = log h^u_{k,t} + log chi^2_1`` — exactly the
    idiosyncratic KSC template with ``q_kk = (Q)_kk`` playing the role of ``r_i``
    (the known offset ``log q_kk`` of the thesis is absorbed by standardising with
    ``q_kk`` before the log-square).

    Three couplings, selected by ``coupling`` (``subsec:vol-all-processes``, the
    "caveat on the cross-covariance"):

    * ``"decoupled"`` (**default**) — the ``r`` measurements are treated as marginally
      independent → ``r`` independent scalar KSC sub-sweeps
      (:func:`sample_log_vol_process`). **Exact when ``Q`` is diagonal**; the
      theory-sanctioned near-diagonal fast path otherwise. At ``r = 1`` the
      standardisation ``e_1 = sqrt(w/q_11) u_1`` equals the scalar block's
      ``tilde_u = sqrt(w) Q^{-1/2} u``, so with the same RNG this reproduces the
      common path of :func:`sample_volatility_block` bit-for-bit.
    * ``"qml"`` — the **quasi-maximum-likelihood** coupled pass (Harvey, Ruiz and
      Shephard 1994): keep the *exact constant* covariance of the log-square vector,
      ``Sigma_xi = (pi^2/2) R_xi`` with ``R_xi = corr`` of the log-squares
      (:func:`logsq_corr_matrix`), and drop the componentwise mixture — the
      measurement is a **single** Gaussian, centred at the log-χ²₁ mean
      (``LOG_CHI2_MEAN``), with that constant covariance, drawn by one r-dim FFBS
      **without** an indicator step.  This is the *consistent* coupled form: unlike
      the literal one below it never mixes conditional variances with an
      unconditional correlation, so it is **stable at strong ``corr(Q)``** — the
      reason it is the option the thesis holds in reserve.  Its price is that it
      **drops the mixture's marginal refinement**, so at diagonal ``Q`` it does *not*
      reduce to ``"decoupled"`` (a single ``N(·, pi^2/2)`` in place of the 7-component
      KSC mixture), and off the near-diagonal regime its benefit is *calibration*, not
      point accuracy — positively-correlated measurement noise is redundant
      information.  **Not a default; a data-driven robustness option** to be switched
      on only if the estimated ``corr(Q)`` proves materially non-diagonal *and* the
      predictive tail is seen to be mis-calibrated (``docs/audit_P1-P5.md`` §P4;
      inspect with :func:`mcmc.diagnostics.recommend_coupling`).  **Reachable only on
      the no-leverage Spec II path**: under Branch~A/B the common block is not this
      function (``subsec:lev-branches-allproc``, detail (iii)).
    * ``"literal"`` — the correlation-scaled form ``Sigma_xi,t = diag(v_{s_k,t})
      R_xi diag(v_{s_k,t})`` with per-factor KSC indicators (pass the matrix in
      ``R_xi``, or leave ``None`` to build it from ``Q``).  ⚠️ **EXPERIMENTAL, requires
      ``allow_experimental=True``** — it mixes the *unconditional* correlation with the
      mixture's *conditional* variances, and the inconsistency **distorts** the
      less-persistent factor at strong ``corr(Q)`` (phi 0.90 → 0.42 at 0.92;
      ``test_spec2_recovery`` diagnostic).  Kept only as the reference implementation
      of the finding that this literal realisation is unstable — the ``"qml"`` form is
      the one to use if a coupled pass is ever wanted.

    Parameters
    ----------
    u_head : (T-1, r)     innovations ``u_t = f_t - A f_{t-1}`` for ``t = 1..T-1``.
    Q : (r, r)            factor-innovation scale (its diagonal standardises each factor).
    w_u : (T,)            factor tail weights (``w_u[0]`` unused: no ``u_0``).
    logh_u : (T, r)       current per-factor log-vol paths (indicator conditional).
    sv_u : (r, 3)         current per-factor ``(mu, phi, sigma2)``.
    coupling : str        ``"decoupled"`` (default) / ``"qml"`` / ``"literal"``.
    R_xi : None or (r, r) the ``"literal"`` correlation matrix; ``None`` builds it from
                          ``Q``.  Passing a matrix selects ``"literal"`` for backward
                          compatibility.  Ignored by ``"decoupled"`` and ``"qml"``.
    allow_experimental : bool  opt-in required for ``"literal"`` only.

    Returns ``{"logh_u": (T, r), "h_u": (T, r), "sv_u": (r, 3)}``.
    """
    if coupling not in ("decoupled", "qml", "literal"):
        raise ValueError(f"coupling={coupling!r}: 'decoupled', 'qml' or 'literal'.")
    # Back-compat: a matrix R_xi is the old way of selecting the literal branch.
    if R_xi is not None and coupling == "decoupled":
        coupling = "literal"
    if coupling == "literal" and not allow_experimental:
        raise ValueError(
            "coupling='literal' is EXPERIMENTAL: the correlation-scaled form "
            "Sigma = diag(v_s) R_xi diag(v_s) is unstable (it distorts the "
            "less-persistent factor, phi 0.90 -> 0.42 at corr(Q)=0.92). For a "
            "coupled pass use coupling='qml' (stable); pass allow_experimental=True "
            "only for a deliberate study of the literal form (docs/audit_P1-P5.md)."
        )
    if coupling != "decoupled" and use_asis:
        raise ValueError(
            f"use_asis is not wired on the coupled joint-FFBS path (coupling="
            f"{coupling!r}); ASIS wraps the per-factor scalar Family B draw of the "
            f"decoupled block. Use coupling='decoupled' with use_asis, or coupling="
            f"{coupling!r} without."
        )
    u_head = np.asarray(u_head, float)
    Tm1, r = u_head.shape
    T = Tm1 + 1
    qdiag = np.diag(np.asarray(Q, float))
    inv_sqrt_q = 1.0 / np.sqrt(qdiag)                  # per-factor 1/sqrt(q_kk)
    sqrt_w1 = np.sqrt(np.asarray(w_u, float)[1:])      # sqrt-weights at t = 1..T-1
    tidx = np.arange(1, T)
    logh_cur = np.asarray(logh_u, float)

    # per-factor standardized residuals and log-squares (measurements at t=1..T-1)
    E = sqrt_w1[:, None] * (u_head * inv_sqrt_q[None, :])       # (T-1, r), e_{k,t} ~ N(0,h_k)
    YS = np.log(E ** 2 + offset)                                # (T-1, r), y*_{k,t}

    logh_new = np.zeros((T, r))
    sv_new = np.zeros((r, 3))

    if coupling == "decoupled":
        # ── R_xi = I → r independent scalar KSC sub-sweeps ────────────────────
        for k in range(r):
            mu_k, phi_k, s2_k = sv_u[k]
            # sqrt(w)*(u_k/sqrt(q_kk)) with the operand order of the scalar block:
            e_k = sqrt_w1 * (u_head[:, k] * inv_sqrt_q[k])
            ys_k = np.log(e_k ** 2 + offset)
            logh_new[:, k] = sample_log_vol_process(
                ys_k, tidx, T, logh_cur[:, k], mu_k, phi_k, s2_k, rng, ksc)
            sv_new[k] = draw_ar1_params(                             # (1) CP draw
                logh_new[:, k], rng, fix_mu0=fix_mu0, a_sig=prior_a, b_sig=prior_b,
                sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                sigma2_cur=float(s2_k))
            if use_asis:                                             # (2)-(4) NCP interweave
                from mcmc.sample_asis import asis_scale_interweave
                y_star = np.empty(T); y_star[1:] = ys_k             # measured t = 1..T-1
                has = np.zeros(T, bool); has[1:] = True
                x_a, phi_a, s2_a = asis_scale_interweave(
                    logh_new[:, k], y_star, has, float(sv_new[k, 2]),
                    0.0, np.zeros(T), rng, half_normal_B=half_normal_B, ksc=ksc)
                logh_new[:, k] = x_a; sv_new[k] = (0.0, phi_a, s2_a)
        return {"logh_u": logh_new, "h_u": np.exp(logh_new), "sv_u": sv_new}

    if coupling == "qml":
        # ── QML: constant exact covariance, NO mixture, one r-dim FFBS ────────
        # Harvey-Ruiz-Shephard (1994): approximate the log-square vector by a single
        # Gaussian keeping its exact first two moments — mean LOG_CHI2_MEAN (per
        # component) and covariance Sigma_xi = (pi^2/2) R_xi, R_xi the log-square
        # correlation matrix from corr(Q).  Constant in t (no indicator), stable at
        # strong corr(Q).  At diagonal Q, Sigma_xi = (pi^2/2) I: a single Gaussian
        # of variance pi^2/2 — NOT the KSC mixture, so this does NOT nest 'decoupled'.
        R_xi_corr = logsq_corr_matrix(np.asarray(Q, float))    # (r, r), 1 on diagonal
        Sigma_xi = (np.pi * np.pi / 2.0) * R_xi_corr           # exact constant cov
        mask = np.zeros(T, bool); mask[1:] = True
        y_eff = np.zeros((T, r))
        y_eff[1:] = YS - LOG_CHI2_MEAN                         # mean-centred measurement
        R_eff = np.zeros((T, r, r))
        R_eff[1:] = Sigma_xi                                   # constant across t
        phi_vec = np.asarray(sv_u, float)[:, 1].copy()
        s2_vec = np.asarray(sv_u, float)[:, 2].copy()
        logh_new = _mv_ar1_ffbs(y_eff, R_eff, mask, phi_vec, s2_vec, rng)
        for k in range(r):
            sv_new[k] = draw_ar1_params(
                logh_new[:, k], rng, fix_mu0=fix_mu0, a_sig=prior_a, b_sig=prior_b,
                sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                sigma2_cur=float(s2_vec[k]))
        return {"logh_u": logh_new, "h_u": np.exp(logh_new), "sv_u": sv_new}

    # ── literal: coupled r-dim FFBS with the log-square cross-correlation ─────
    if R_xi is None:                                           # coupling='literal', no matrix
        R_xi = logsq_corr_matrix(np.asarray(Q, float))
    R_xi = np.asarray(R_xi, float)
    m_ksc = ksc["m"]; v2_ksc = ksc["v2"]
    log_q = np.log(ksc["q"]); half_log_v2 = 0.5 * np.log(v2_ksc)

    y_eff = np.zeros((T, r))
    vstd = np.ones((T, r))                             # measurement std per factor per t
    for k in range(r):
        # (i) per-factor indicators s_{k,t} ~ q_j N(y*; logh_k + m_j, v2_j)
        d = YS[:, k][:, None] - logh_cur[1:, k][:, None] - m_ksc[None, :]      # (T-1, 7)
        logp = log_q[None, :] - half_log_v2[None, :] - 0.5 * d * d / v2_ksc[None, :]
        s = np.argmax(logp + rng.gumbel(size=logp.shape), axis=1)             # (T-1,)
        y_eff[1:, k] = YS[:, k] - m_ksc[s]
        vstd[1:, k] = np.sqrt(v2_ksc[s])

    # (ii) one r-dim FFBS with Sigma_xi,t = diag(v) R_xi diag(v) (mask t=1..T-1)
    mask = np.zeros(T, bool); mask[1:] = True
    R_eff = np.zeros((T, r, r))
    for t in range(1, T):
        Dt = vstd[t]
        R_eff[t] = (Dt[:, None] * R_xi) * Dt[None, :]
    phi_vec = np.asarray(sv_u, float)[:, 1].copy()
    s2_vec = np.asarray(sv_u, float)[:, 2].copy()
    logh_new = _mv_ar1_ffbs(y_eff, R_eff, mask, phi_vec, s2_vec, rng)         # (T, r)
    for k in range(r):
        sv_new[k] = draw_ar1_params(
            logh_new[:, k], rng, fix_mu0=fix_mu0, a_sig=prior_a, b_sig=prior_b,
            sigma_prior=sigma_prior, half_normal_B=half_normal_B,
            sigma2_cur=float(s2_vec[k]))
    return {"logh_u": logh_new, "h_u": np.exp(logh_new), "sv_u": sv_new}


def _sample_idio_vol(Y, F, Lambda, R, w_eps, logh_eps, sv_eps, rng,
                     offset, fix_mu0, prior_a, prior_b, ksc,
                     *, sigma_prior="inverse_gamma", half_normal_B=1.0, use_asis=False):
    """The idiosyncratic KSC sub-sweeps (one scalar process per series) — shared
    by the scalar and the Spec-II common blocks (R diagonal, so unchanged by the
    per-factor common treatment).  ``use_asis`` wraps each series' Family~B draw
    with the ASIS interweave (Phase 6, ``sec:asis``)."""
    T, M = Y.shape
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
                                        a_sig=prior_a, b_sig=prior_b,
                                        sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                                        sigma2_cur=float(s2_i))
        if use_asis:                               # (2)-(4) NCP interweave (per series)
            from mcmc.sample_asis import asis_scale_interweave
            y_star = np.zeros(T); y_star[obs_t] = ys_i
            has = np.zeros(T, bool); has[obs_t] = True
            x_a, phi_a, s2_a = asis_scale_interweave(
                logh_eps_new[:, i], y_star, has, float(sv_eps_new[i, 2]),
                0.0, np.zeros(T), rng, half_normal_B=half_normal_B, ksc=ksc)
            logh_eps_new[:, i] = x_a; sv_eps_new[i] = (0.0, phi_a, s2_a)
    return logh_eps_new, sv_eps_new


def sample_volatility_block_specII(
    Y: np.ndarray,
    f_aug: np.ndarray,
    theta: dict,
    w_u: np.ndarray,
    w_eps: np.ndarray,
    logh_u: np.ndarray,
    logh_eps: np.ndarray,
    sv_u: np.ndarray,
    sv_eps: np.ndarray,
    rng: np.random.Generator,
    *,
    offset: float = 1e-6,
    fix_mu0: bool = True,
    prior_a: float = 2.0,
    prior_b: float = 0.05,
    sigma_prior: str = "inverse_gamma",
    half_normal_B: float = 1.0,
    use_asis: bool = False,
    sv_idio: bool = True,
    common_vol_coupling: str = "decoupled",
    R_xi: np.ndarray | None = None,
    ksc: dict = KSC7,
) -> dict:
    r"""
    Step (b), **Specification II, no leverage**: the ``r`` per-factor common
    volatilities (via :func:`sample_common_vol_mv`) and the ``M`` idiosyncratic
    ones (unchanged), and their AR(1) parameters.  ``common_vol_coupling`` selects
    the common-block measurement coupling — ``"decoupled"`` (default), ``"qml"``
    (stable coupled) or ``"literal"`` (experimental); see
    :func:`sample_common_vol_mv`.

    ``sv_idio=False`` is the **D2-a** restriction (``subsec:variants-restrictions``,
    "common volatility only"): the idiosyncratic part of the sampler is omitted,
    the ``h^eps_{i,t}`` are frozen at 1, and Family~B is drawn only for the ``r``
    common processes rather than for all ``M+r``.  ``sv_eps`` is then returned
    unchanged (frozen, not drawn).

    Returns ``h_u`` (T, r), ``h_eps`` (T, M), ``logh_u`` (T, r), ``logh_eps`` (T, M),
    ``sv_u`` (r, 3), ``sv_eps`` (M, 3) — the per-factor counterpart of
    :func:`sample_volatility_block` (whose ``h_u`` is (T,) scalar).
    """
    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    r = A.shape[0]
    F = f_aug[:, :r]

    u_head = F[1:] - F[:-1] @ A.T                   # (T-1, r), u_t = f_t - A f_{t-1}
    if use_asis and sigma_prior != "half_normal":
        raise ValueError("use_asis=True requires sigma_prior='half_normal' "
                         "(CP and NCP must share the Gaussian prior on sigma_eta).")
    cm = sample_common_vol_mv(u_head, Q, w_u, logh_u, sv_u, rng, offset=offset,
                              fix_mu0=fix_mu0, prior_a=prior_a, prior_b=prior_b,
                              sigma_prior=sigma_prior, half_normal_B=half_normal_B,
                              use_asis=use_asis, coupling=common_vol_coupling,
                              R_xi=R_xi, ksc=ksc)

    if sv_idio:
        logh_eps_new, sv_eps_new = _sample_idio_vol(
            Y, F, Lambda, R, w_eps, logh_eps, sv_eps, rng,
            offset, fix_mu0, prior_a, prior_b, ksc,
            sigma_prior=sigma_prior, half_normal_B=half_normal_B, use_asis=use_asis)
    else:                                           # D2-a: h^eps frozen at 1
        logh_eps_new = np.zeros_like(np.asarray(logh_eps, float))
        sv_eps_new = np.asarray(sv_eps, float).copy()

    return {
        "h_u": cm["h_u"], "h_eps": np.exp(logh_eps_new),
        "logh_u": cm["logh_u"], "logh_eps": logh_eps_new,
        "sv_u": cm["sv_u"], "sv_eps": sv_eps_new,
    }
