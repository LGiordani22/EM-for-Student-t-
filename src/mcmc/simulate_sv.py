"""
src/mcmc/simulate_sv.py
=======================

Stochastic-volatility extension of the DFM simulator, for the Passo-2 recovery
test.  Self-contained in ``mcmc/`` (the EM-side ``simulate_dfm`` is left
untouched), but it reuses the masking (:func:`simulate_dfm.apply_missing_pattern`)
and the MM aggregation (:func:`kalman.build_Lambda_tilde`) so the synthetic panel
has exactly the mixed-frequency + ragged structure the sampler expects.

Data-generating process (base case, no leverage, rho = 0)
---------------------------------------------------------
For the common factor and each idiosyncratic series an AR(1) log-volatility
path is drawn,

    log h_t = mu + phi (log h_{t-1} - mu) + N(0, sigma^2),
    log h_0 ~ N(mu, sigma^2/(1-phi^2)),

and the Student-t tail weights are drawn ``w_t ~ Gamma(nu/2, nu/2)`` (mean 1).
The factor innovation and idiosyncratic noise then carry *both* multipliers:

    u_t ~ N(0, h^u_t Q / w^u_t),      eps_{i,t} ~ N(0, h^eps_{i,t} r_i / w^eps_t).

Setting all ``sigma == 0`` (constant unit h) reproduces the Passo-1 Student-t
DGP; setting in addition ``nu -> inf`` reproduces the Gaussian one.

Two common-volatility DGPs
--------------------------
* **scalar-common** (``sv_u``, the historical default): the restriction
  ``H^u_t = h^u_t I`` — a *single* volatility shared by all r factors, with a
  *vector* ``rho_u`` under leverage.  At ``H = h I`` the two placements of
  ``subsec:vol-placement`` coincide (``Q^{1/2} H Q^{1/2} = sqrt(H) Q sqrt(H)
  = h^u_t Q``), so this DGP does **not** select a specification; it is simply
  the degenerate case.  Kept bit-identical (the Passo-2/3/4 gates use it).
* **per-factor, Specification II** (``sv_u_perfactor``): ``r`` *independent*
  volatilities ``h^u_{k,t}`` on the diagonal of ``H^u_t``, entering through the
  **outside** sandwich ``u_t = sqrt(H^u_t) Q^{1/2} z_t / sqrt(w^u_t)``
  (eq:vol-outside), so ``Var(u_{k,t}) = h^u_{k,t} q_kk`` reads factor k's *own*
  volatility off the diagonal — the reason the thesis adopts II over the inside
  placement I (eq:vol-inside), whose diagonal blends every ``h^u_j`` through
  ``Q^{1/2}``.  Under Option A each channel carries its own scalar ``rho_k``,
  coupling ``eta_k`` to the raw shock ``z_k``.  This is what the sampler
  estimates on every SV path, and the only DGP in which distinct per-factor
  ``(phi_k, sigma_k, rho_k)`` are identified.
"""

from __future__ import annotations

import numpy as np

from kalman import build_Lambda_tilde
from simulate_dfm import apply_missing_pattern


def _ar1_path(mu: float, phi: float, sigma: float, L: int,
              rng: np.random.Generator) -> np.ndarray:
    """Draw a stationary AR(1) path of length L."""
    x = np.empty(L)
    if sigma <= 0.0:
        x[:] = mu
        return x
    stat_sd = sigma / np.sqrt(1.0 - phi * phi)
    x[0] = mu + stat_sd * rng.standard_normal()
    for t in range(1, L):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * rng.standard_normal()
    return x


def _sqrt_spd(Q: np.ndarray) -> np.ndarray:
    """Symmetric square root ``Q^{1/2}`` of an SPD matrix (so the sampler's
    symmetric ``Q^{-1/2}`` whitening recovers the generative shock z directly,
    aligning the leverage parametrisation between simulator and sampler)."""
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    vals = np.clip(vals, 1e-12, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def simulate_dfm_sv(
    theta: dict,
    T: int,
    freq_list: list[str],
    block_map: dict[str, str],
    ordered_cols: list[str],
    r: int,
    seed: int,
    *,
    sv_u: tuple[float, float, float],
    sv_eps: tuple[float, float, float] | np.ndarray,
    sv_u_perfactor: np.ndarray | None = None,
    rho_u: np.ndarray | float | None = None,
    rho_eps: float | np.ndarray | None = None,
    timing: str = "contemporaneous",
    burn_in: int = 200,
    ragged_months: int = 2,
    ragged_series: list[str] | None = None,
    quarter_end_offset: int = 2,
) -> dict:
    r"""
    Simulate a Student-t mixed-frequency DFM panel with stochastic volatility.

    Parameters
    ----------
    theta : dict          DGP parameters ``A, Q, Lambda, R, nu_u, nu_eps``.
    sv_u : (mu, phi, sigma)            common-factor log-vol AR(1) — the
                          **scalar-common** DGP ``H^u_t = h^u_t I``: a *single*
                          ``h^u_t`` shared by all r factors (with, under leverage,
                          a vector ``rho_u`` loading the r shocks).  At ``H = h I``
                          the inside and outside placements coincide, so this DGP
                          picks no specification.  Ignored when ``sv_u_perfactor``
                          is given.
    sv_u_perfactor : (r, 3) or None    the **Specification II / Option A** DGP:
                          ``r`` *independent* per-factor log-vol AR(1)s
                          ``(mu_k, phi_k, sigma_k)``, and under leverage ``r``
                          *scalar* channels ``rho_k`` coupling ``eta_k`` to the
                          raw shock ``z_k`` (``eq:lev-cond-common``).  This is the
                          specification the sampler now estimates on every SV
                          path, and the only one in which distinct per-factor
                          ``(phi_k, sigma_k, rho_k)`` are identified.  The
                          innovation is the outside sandwich
                          ``u_t = sqrt(H^u_t) Q^{1/2} z_t / sqrt(w^u_t)``, so the
                          sampler's whitening ``z^u = sqrt(w) Q^{-1/2}(sqrt H)^{-1} u``
                          recovers ``z_t`` exactly.
    sv_eps : (mu, phi, sigma) or (M, 3)  idiosyncratic log-vol AR(1) (shared
                          spec broadcast to all series, or per-series).
    burn_in : int         factor/vol burn-in (>= 4 so the MM lags are genuine).

    .. warning::
       **sigma in, sigma-SQUARED out** — and the two are *not* the same field.

       Every ``sv_*`` **input** above is ``(mu, phi, sigma)``: the log-vol
       innovation **standard deviation** (that is what ``_ar1_path`` multiplies the
       normal draw by).  The **sampler**, however, parameterises the same process by
       the **variance**: ``fit_dfm_mcmc`` reads ``sv_init=(mu, phi, sigma^2)`` and
       its ``draws["sv_u"][:, :, 2]`` is ``sigma^2``.

       The two conventions once collided on the same key: this function used to
       *return* its ``sigma`` input under the name ``sv_u``, so
       ``sim["sv_u"]`` and ``res["draws"]["sv_u"]`` looked comparable and were not.
       A recovery check written against it silently compared ``sigma`` to
       ``sigma^2`` — the kind of error that returns plausible numbers.

       The inputs are left in ``sigma`` (they define the DGPs of the whole suite,
       and re-reading them as variances would silently change every one of them),
       and the **returned** ``sv_u`` / ``sv_eps`` are converted to the sampler's
       ``(mu, phi, sigma^2)``.  So ``sim["sv_u"]`` may be compared with
       ``res["draws"]["sv_u"]`` directly — that is the whole point.  The inputs are
       echoed back, unconverted, as ``sv_u_sigma`` / ``sv_eps_sigma``.

    Returns dict with ``Y`` (T, M), ``F`` (T, r), ``h_u_true`` — (T,) scalar-common,
    (T, r) under ``sv_u_perfactor`` — ``h_eps_true`` (T, M), ``logh_u_true``,
    ``logh_eps_true``, ``w_u_true`` (T,), ``w_eps_true`` (T,), ``sv_u``,
    ``sv_eps`` (M, 3) — **both in the sampler's ``(mu, phi, sigma^2)``** — plus
    ``sv_u_sigma`` / ``sv_eps_sigma`` (the inputs, in ``sigma``), ``rho_u_true``,
    ``rho_eps_true``, ``theta_used``.
    """
    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    nu_u = float(theta["nu_u"]); nu_eps = float(theta["nu_eps"])
    M = len(ordered_cols)
    if burn_in < 4:
        burn_in = 4
    L = burn_in + T

    sv_eps = np.asarray(sv_eps, dtype=float)
    if sv_eps.ndim == 1:
        sv_eps = np.tile(sv_eps, (M, 1))

    perfactor = sv_u_perfactor is not None
    if perfactor:
        sv_u_pf = np.asarray(sv_u_perfactor, dtype=float)
        if sv_u_pf.shape != (r, 3):
            raise ValueError(f"sv_u_perfactor must be (r, 3) = ({r}, 3), "
                             f"got {sv_u_pf.shape}.")

    rng = np.random.default_rng(seed)
    rng_v = np.random.default_rng(seed + 101)
    rng_w = np.random.default_rng(seed + 202)
    rng_e = np.random.default_rng(seed + 303)

    # ── volatility paths (full length, then keep the tail window) ────────────
    logh_u_full = _ar1_path(sv_u[0], sv_u[1], sv_u[2], L, rng_v)
    h_u_full = np.exp(logh_u_full)
    logh_eps_full = np.empty((L, M))
    for i in range(M):
        logh_eps_full[:, i] = _ar1_path(sv_eps[i, 0], sv_eps[i, 1], sv_eps[i, 2], L, rng_v)
    h_eps_full = np.exp(logh_eps_full)

    # ── tail weights (scalar per period) ─────────────────────────────────────
    w_u_full = rng_w.gamma(shape=nu_u / 2.0, scale=2.0 / nu_u, size=L)
    w_eps_full = rng_w.gamma(shape=nu_eps / 2.0, scale=2.0 / nu_eps, size=L)

    # ── leverage configuration (contemporaneous, Branch A) ───────────────────
    lev = (rho_u is not None) or (rho_eps is not None)
    rho_u_vec = np.zeros(r) if rho_u is None else np.broadcast_to(
        np.asarray(rho_u, float), (r,)).copy()
    rho_eps_vec = np.zeros(M) if rho_eps is None else np.broadcast_to(
        np.asarray(rho_eps, float), (M,)).copy()

    Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)

    def _idio_leverage_paths(zlag: int):
        """Per-series coupling of the level shock and its log-vol (Family C, idio).
        An idiosyncratic residual exists every period (incl. t = 0), so under
        lagged timing eta_1 pairs with z_0 (the sampler sees it: period 0 is
        observed).  Shared verbatim by the scalar-common and per-factor branches."""
        z_e = rng_e.standard_normal((L, M))          # standardized idio shocks
        nu_indep_e = rng_v.standard_normal((L, M))   # independent parts
        x_e = np.zeros((L, M))
        for i in range(M):
            s_i, phi_i, rho_i = float(sv_eps[i, 2]), float(sv_eps[i, 1]), float(rho_eps_vec[i])
            x_e[0, i] = (s_i / np.sqrt(1.0 - phi_i * phi_i)) * nu_indep_e[0, i] if s_i > 0 else 0.0
            for t in range(1, L):
                if s_i > 0:
                    zsrc = t - zlag
                    eta = s_i * rho_i * z_e[zsrc, i] \
                        + s_i * np.sqrt(max(1.0 - rho_i ** 2, 0.0)) * nu_indep_e[t, i]
                else:
                    eta = 0.0
                x_e[t, i] = phi_i * x_e[t - 1, i] + eta
        return z_e, x_e

    if perfactor:
        # ── Specification II + Option A: r per-factor volatilities, r scalar rho.
        # Outside sandwich u_t = sqrt(H^u_t) Q^{1/2} z_t / sqrt(w^u_t) with the
        # SYMMETRIC Q^{1/2}, so the sampler's z^u = sqrt(w) Q^{-1/2}(sqrt H)^{-1} u
        # returns exactly the generative z_t — the shock rho_k keys on
        # (eq:lev-cond-common).  Each channel is independent:
        #     eta_{k,t} = s_k [ rho_k z_{k,t-zlag} + sqrt(1-rho_k^2) nu_{k,t} ],
        #     x_{k,t}   = phi_k x_{k,t-1} + eta_{k,t}.
        if timing not in ("contemporaneous", "lagged"):
            raise ValueError(f"timing={timing!r} must be 'contemporaneous' or 'lagged'.")
        zlag = 1 if timing == "lagged" else 0
        Qhalf = _sqrt_spd(Q)
        mu_k, phi_k, s_k = sv_u_pf[:, 0], sv_u_pf[:, 1], sv_u_pf[:, 2]
        if np.any(mu_k != 0.0):
            raise ValueError("sv_u_perfactor: mu must be 0 (eq:sv-mu-identification).")
        nu_indep_u = rng_v.standard_normal((L, r))   # independent parts of eta^u
        z_u_full = rng.standard_normal((L, r))       # raw standardized factor shocks
        x_u = np.zeros((L, r))
        f = np.zeros((L, r))
        stat_sd = np.where(s_k > 0, s_k / np.sqrt(1.0 - phi_k ** 2), 0.0)
        x_u[0] = stat_sd * nu_indep_u[0]
        for t in range(1, L):
            # the common factor has a level shock only from t = 1 on (no u_0), so a
            # lagged eta_1 (which would pair with z_0) is left plain — matching the
            # sampler, whose first leverage transition is into t = 2.
            zsrc = t - zlag
            if zsrc >= 1:
                eta = s_k * (rho_u_vec * z_u_full[zsrc]
                             + np.sqrt(np.maximum(1.0 - rho_u_vec ** 2, 0.0)) * nu_indep_u[t])
            else:
                eta = s_k * nu_indep_u[t]
            x_u[t] = phi_k * x_u[t - 1] + eta
            u = np.sqrt(np.exp(x_u[t]) / w_u_full[t]) * (Qhalf @ z_u_full[t])
            f[t] = A @ f[t - 1] + u
        logh_u_full = x_u; h_u_full = np.exp(x_u)
        if lev:
            z_eps_full, x_eps = _idio_leverage_paths(zlag)
            logh_eps_full = x_eps; h_eps_full = np.exp(x_eps)
        else:
            z_eps_full = None
    elif not lev:
        # ── no-leverage path (Passo 2; bit-identical RNG stream) ─────────────
        cQ = np.linalg.cholesky(0.5 * (Q + Q.T))
        f = np.zeros((L, r))
        for t in range(1, L):
            scale = np.sqrt(h_u_full[t] / w_u_full[t])
            f[t] = A @ f[t - 1] + scale * (cQ @ rng.standard_normal(r))
        z_eps_full = None
    else:
        # ── leverage: couple the standardized level shock with the log-vol
        # innovation (corr rho).  Timing selects WHICH vol innovation:
        #   contemporaneous (Branch A): z_t couples eta_t  (zlag = 0);
        #   lagged          (Branch B): z_t couples eta_{t+1}, i.e. eta_t couples
        #                               z_{t-1}             (zlag = 1).
        # Symmetric Q^{1/2} is used so the sampler's whitening (symmetric Q^{-1/2})
        # recovers z directly, aligning the rho parametrisation.
        if timing not in ("contemporaneous", "lagged"):
            raise ValueError(f"timing={timing!r} must be 'contemporaneous' or 'lagged'.")
        zlag = 1 if timing == "lagged" else 0
        Qhalf = _sqrt_spd(Q)
        s_u, phi_u = float(sv_u[2]), float(sv_u[1])
        rr_u = float(rho_u_vec @ rho_u_vec)
        nu_indep_u = rng_v.standard_normal(L)        # independent part of eta^u
        z_u_full = rng.standard_normal((L, r))       # standardized factor shocks
        x_u = np.zeros(L)
        x_u[0] = (s_u / np.sqrt(1.0 - phi_u ** 2)) * nu_indep_u[0] if s_u > 0 else 0.0
        f = np.zeros((L, r))
        for t in range(1, L):
            if s_u > 0:
                # the common factor has a level shock only from t = 1 on (no u_0),
                # so a lagged eta_1 (which would pair with z_0) is left plain —
                # matching the sampler, whose first leverage transition is t = 2.
                zsrc = t - zlag
                if zsrc >= 1:
                    eta = s_u * (rho_u_vec @ z_u_full[zsrc]) \
                        + s_u * np.sqrt(max(1.0 - rr_u, 0.0)) * nu_indep_u[t]
                else:
                    eta = s_u * nu_indep_u[t]
            else:
                eta = 0.0
            x_u[t] = phi_u * x_u[t - 1] + eta
            h = np.exp(x_u[t])
            u = np.sqrt(h / w_u_full[t]) * (Qhalf @ z_u_full[t])
            f[t] = A @ f[t - 1] + u
        logh_u_full = x_u; h_u_full = np.exp(x_u)

        z_eps_full, x_eps = _idio_leverage_paths(zlag)
        logh_eps_full = x_eps; h_eps_full = np.exp(x_eps)

    keep = slice(burn_in, L)
    F = f[keep]                                            # (T, r)
    h_u = h_u_full[keep]; h_eps = h_eps_full[keep]
    logh_u = logh_u_full[keep]; logh_eps = logh_eps_full[keep]
    w_u = w_u_full[keep].copy(); w_eps = w_eps_full[keep]
    w_u[0] = 1.0                                           # convention (no u_0)

    # ── augmented state over the window, with genuine pre-window lags ─────────
    f_aug = np.zeros((T, 5 * r))
    for l in range(5):
        f_aug[:, l * r:(l + 1) * r] = f[burn_in - l: L - l]

    # ── observation signal (MM aggregation) + idiosyncratic noise ────────────
    signal = f_aug @ Lambda_tilde.T                        # (T, M)
    eps_std = np.sqrt(h_eps * R[None, :] / w_eps[:, None])             # (T, M)
    if not lev:
        # OLD stream: one (T, M) draw — keeps Passo 2 bit-identical.
        obs_noise = rng_e.standard_normal((T, M))
    else:
        obs_noise = z_eps_full[keep]                       # leverage-coupled shock
    Y_complete = signal + eps_std * obs_noise

    Y = apply_missing_pattern(
        Y_complete, freq_list, ordered_cols,
        ragged_months=ragged_months, ragged_series=ragged_series,
        quarter_end_offset=quarter_end_offset,
    )

    # sigma in, sigma^2 out (see the warning in the docstring): the inputs define the
    # DGP in standard-deviation units, the sampler speaks variance.  Convert on the
    # way out so that sim["sv_u"] and res["draws"]["sv_u"] mean the SAME thing, and
    # echo the raw inputs under a name that cannot be mistaken for either.
    sv_u_sigma = np.asarray(sv_u_pf if perfactor else sv_u, float)
    sv_eps_sigma = np.asarray(sv_eps, float)

    def _to_var(sv: np.ndarray) -> np.ndarray:
        out = np.array(sv, float, copy=True)
        out[..., 2] = out[..., 2] ** 2
        return out

    return {
        "Y": Y, "F": F,
        "h_u_true": h_u, "h_eps_true": h_eps,
        "logh_u_true": logh_u, "logh_eps_true": logh_eps,
        "w_u_true": w_u, "w_eps_true": w_eps,
        "sv_u": _to_var(sv_u_sigma), "sv_eps": _to_var(sv_eps_sigma),
        "sv_u_sigma": sv_u_sigma, "sv_eps_sigma": sv_eps_sigma,
        "rho_u_true": rho_u_vec, "rho_eps_true": rho_eps_vec,
        "theta_used": dict(theta),
    }
