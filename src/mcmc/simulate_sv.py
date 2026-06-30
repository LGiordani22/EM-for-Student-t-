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
    sv_u : (mu, phi, sigma)            common-factor log-vol AR(1).
    sv_eps : (mu, phi, sigma) or (M, 3)  idiosyncratic log-vol AR(1) (shared
                          spec broadcast to all series, or per-series).
    burn_in : int         factor/vol burn-in (>= 4 so the MM lags are genuine).

    Returns dict with ``Y`` (T, M), ``F`` (T, r), ``h_u_true`` (T,),
    ``h_eps_true`` (T, M), ``logh_u_true``, ``logh_eps_true``,
    ``w_u_true`` (T,), ``w_eps_true`` (T,), ``sv_u``, ``sv_eps`` (M, 3),
    ``theta_used``.
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

    if not lev:
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

        # idiosyncratic: per-series coupling of its level shock and log-vol.  An
        # idiosyncratic residual exists every period (incl. t = 0), so under lagged
        # timing eta_1 pairs with z_0 (the sampler sees it: period 0 is observed).
        z_eps_full = rng_e.standard_normal((L, M))   # standardized idio shocks
        nu_indep_e = rng_v.standard_normal((L, M))   # independent parts
        x_eps = np.zeros((L, M))
        for i in range(M):
            s_i, phi_i, rho_i = float(sv_eps[i, 2]), float(sv_eps[i, 1]), float(rho_eps_vec[i])
            x_eps[0, i] = (s_i / np.sqrt(1.0 - phi_i ** 2)) * nu_indep_e[0, i] if s_i > 0 else 0.0
            for t in range(1, L):
                if s_i > 0:
                    zsrc = t - zlag
                    eta = s_i * rho_i * z_eps_full[zsrc, i] \
                        + s_i * np.sqrt(max(1.0 - rho_i ** 2, 0.0)) * nu_indep_e[t, i]
                else:
                    eta = 0.0
                x_eps[t, i] = phi_i * x_eps[t - 1, i] + eta
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

    return {
        "Y": Y, "F": F,
        "h_u_true": h_u, "h_eps_true": h_eps,
        "logh_u_true": logh_u, "logh_eps_true": logh_eps,
        "w_u_true": w_u, "w_eps_true": w_eps,
        "sv_u": np.asarray(sv_u, float), "sv_eps": sv_eps,
        "rho_u_true": rho_u_vec, "rho_eps_true": rho_eps_vec,
        "theta_used": dict(theta),
    }
