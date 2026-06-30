"""
src/mcmc/gibbs.py
=================

Gibbs orchestrator for the Student-t mixed-frequency DFM — the **complete model**
with stochastic volatility and leverage, selectable by flag (``sv``, ``leverage``,
``timing``).  This is the MCMC counterpart of the EM: the same model and the same
Kalman / mixed-frequency machinery, with the estimation engine replaced by a
Gibbs sampler whose blocks draw from the exact full conditionals.

Sweep
-----
    (a) states      f_{0:T-1}      FFBS  (mcmc.sample_states)
    (b) volatility  h_u, h_eps     KSC base / leverage (mcmc.sample_vol,
                    + sv_u, sv_eps    mcmc.sample_leverage[_lagged]); SV only
                    (+ rho_u, rho_eps when leverage)
    (c) weights     w_u, w_eps     Gamma (mcmc.shared.draw_weights)
    (d) parameters  A, Q           MNIW  (mcmc.sample_params.draw_A_Q_block)
                    Lambda, R      NIG   (mcmc.sample_params.draw_Lambda_R_block)
                    nu_u, nu_eps   griddy-Gibbs (mcmc.sample_params.draw_nu_griddy)

Flags / nesting.  With ``sv=False`` block (b) is skipped (``h == 1``) and the
weights enter the filter and the moments exactly as in the EM E-step.  With
``sv=True, leverage=False`` block (b) is the base KSC sampler.  With
``leverage=True`` block (b) adds the leverage drift, Branch A
(``timing='contemporaneous'``, Metropolis path) or Branch B
(``timing='lagged'``, Omori mixture + FFBS path); ``rho == 0`` recovers the
no-leverage case.

Config-aware (vincolo trasversale)
----------------------------------
Nothing is hard-coded: ``r``, ``M``, ``T``, the block structure and the
per-series frequencies all come from the arguments.  The warm start is loaded
from the current config's ``fit_dfm_result.npz`` via :func:`load_warm_init`
(``load_dfm_fit`` + ``resolve_output_path``), and the draw storage is the
caller's responsibility (no cache is written here, so the real EM caches are
never touched).
"""

from __future__ import annotations

import time

import numpy as np

from kalman import build_Lambda_tilde, build_all_selection_matrices, kalman_filter

from mcmc.shared import (
    draw_weights,
    realized_deflated_d_eps,
    realized_deflated_d_u,
)
from mcmc.sample_states import ffbs_sample_states
from mcmc.sample_params import draw_A_Q_block, draw_Lambda_R_block, draw_nu_griddy
from mcmc.sample_vol import sample_volatility_block
from mcmc.sample_leverage import sample_volatility_block_leverage
from mcmc.sample_leverage_lagged import sample_volatility_block_leverage_lagged


def load_warm_init(config_name: str = "small") -> dict:
    """
    Load the warm start (EM fit) for ``config_name`` — the config's own
    ``data/processed/<config>/fit_dfm_result.npz`` — via ``load_dfm_fit``.

    Returns a dict with the canonical ``theta`` plus the panel metadata
    (``freq_list``, ``block_map``, ``ordered_cols``, ``r``) derived from the
    matching ``load_config(config_name)``.  This is the config-aware entry point
    the plan mandates: the sampler is initialised from the EM fit of the *same*
    config, never a generic one.
    """
    from config_utils import resolve_output_path
    from data_loader import load_config
    from em_main import load_dfm_fit

    cfg = load_config(config_name)
    ordered_cols = cfg["ORDERED_COLS"]
    block_map = cfg["BLOCK"]
    freq_list = [cfg["FREQ"][c] for c in ordered_cols]

    fit_path = resolve_output_path("processed", "fit_dfm_result.npz", config_name)
    fit = load_dfm_fit(fit_path)
    return {
        "theta": fit["theta"],
        "freq_list": freq_list,
        "block_map": block_map,
        "ordered_cols": ordered_cols,
        "r": int(fit["r"]),
        "fit": fit,
    }


def fit_dfm_mcmc(
    Y: np.ndarray,
    theta_init: dict,
    freq_list: list[str],
    block_map: dict[str, str],
    ordered_cols: list[str],
    *,
    n_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 1,
    seed: int = 0,
    update_nu: bool = True,
    nu_bounds: tuple[float, float] = (2.001, 1000.0),
    nu_grid_size: int = 600,
    sv: bool = False,
    sv_offset: float = 1e-6,
    sv_init: tuple[float, float, float] = (0.0, 0.95, 0.05),
    sv_fix_mu0: bool = True,
    sv_prior_a: float = 2.0,
    sv_prior_b: float = 0.05,
    leverage: bool = False,
    timing: str = "contemporaneous",
    lev_prop_path: float = 0.25,
    lev_prop_sigma2: float = 0.20,
    lev_prop_rho: float = 0.06,
    store_states: bool = False,
    store_vol: bool = False,
    verbose: bool = True,
) -> dict:
    r"""
    Run the Gibbs sampler for the Student-t mixed-frequency DFM.

    Stochastic volatility (``sv``) and leverage (``leverage`` + ``timing``) are
    selectable by flag; with both off this reduces to the no-SV MCMC counterpart
    of the EM.

    Parameters
    ----------
    Y : (T, M)                observation panel (NaN = missing).
    theta_init : dict         warm start; keys ``A`` (r, r), ``Q`` (r, r),
                              ``Lambda`` (M, r), ``R`` (M,), ``Sigma_0`` (5r, 5r),
                              ``nu_u``, ``nu_eps``.  Sigma_0 is held fixed.
    freq_list, block_map, ordered_cols : panel metadata (config-driven).
    n_iter, burn_in, thin : sampler schedule.  ``n_keep = ceil((n_iter-burn_in)/thin)``.
    seed : int                master RNG seed.
    update_nu : bool          draw nu_u, nu_eps (True) or hold them fixed (False).
    nu_bounds, nu_grid_size : griddy-Gibbs settings for the nu draws.
    store_states : bool       also store the per-draw head factor path F (T, r)
                              (memory ~ n_keep * T * r floats).
    verbose : bool            progress log.

    Returns
    -------
    dict with
        ``draws`` : dict of stacked draws —
            ``A`` (n_keep, r, r), ``Q`` (n_keep, r, r),
            ``Lambda`` (n_keep, M, r), ``R`` (n_keep, M),
            ``nu_u`` (n_keep,), ``nu_eps`` (n_keep,), ``loglik`` (n_keep,),
            and ``F`` (n_keep, T, r) iff ``store_states``.
        ``theta_mean`` : dict — posterior means (A, Q, Lambda, R, nu_u, nu_eps,
            Sigma_0), in the same layout as a ``theta`` dict.
        ``meta`` : dict — schedule, dimensions, seed, wall-clock.
    """
    Y = np.asarray(Y, dtype=float)
    T, M = Y.shape
    A = np.array(theta_init["A"], dtype=float)
    Q = np.array(theta_init["Q"], dtype=float)
    Lambda = np.array(theta_init["Lambda"], dtype=float)
    R = np.array(theta_init["R"], dtype=float).ravel()
    Sigma_0 = np.array(theta_init["Sigma_0"], dtype=float)
    nu_u = float(theta_init["nu_u"])
    nu_eps = float(theta_init["nu_eps"])
    r = A.shape[0]

    rng = np.random.default_rng(seed)
    W_list = build_all_selection_matrices(Y)

    # Warm-start weights at 1 (no down-weighting yet) — the FFBS at the very
    # first sweep needs weights before step (c) produces them.
    w_u = np.ones(T)
    w_eps = np.ones(T)

    # Stochastic-volatility state (Passo 2).  h == 1 (log h == 0) at start; the
    # combined precision is g = w / h, so with h == 1 the SV path reduces to
    # Passo 1.  M+1 processes: common h^u + M idiosyncratic h^eps.
    logh_u = np.zeros(T)
    logh_eps = np.zeros((T, M))
    h_u = np.ones(T)
    h_eps = np.ones((T, M))
    sv_u = np.array(sv_init, dtype=float)
    sv_eps = np.tile(np.array(sv_init, dtype=float), (M, 1))
    # Leverage state (Passi 3-4).  rho == 0 == no leverage (nesting).
    rho_u = np.zeros(r)
    rho_eps = np.zeros(M)
    if leverage and timing not in ("contemporaneous", "lagged"):
        raise ValueError(
            f"timing={timing!r} not recognised: 'contemporaneous' (Branch A, "
            f"Passo 3) or 'lagged' (Branch B / Omori, Passo 4)."
        )

    n_keep = max(0, (n_iter - burn_in + thin - 1) // thin)
    draws = {
        "A": np.zeros((n_keep, r, r)),
        "Q": np.zeros((n_keep, r, r)),
        "Lambda": np.zeros((n_keep, M, r)),
        "R": np.zeros((n_keep, M)),
        "nu_u": np.zeros(n_keep),
        "nu_eps": np.zeros(n_keep),
        "loglik": np.zeros(n_keep),
    }
    if store_states:
        draws["F"] = np.zeros((n_keep, T, r))
    if sv:
        draws["sv_u"] = np.zeros((n_keep, 3))        # (mu, phi, sigma2)
        draws["sv_eps"] = np.zeros((n_keep, M, 3))
        if store_vol:
            draws["h_u"] = np.zeros((n_keep, T))
            draws["h_eps"] = np.zeros((n_keep, T, M))
    if leverage:
        draws["rho_u"] = np.zeros((n_keep, r))
        draws["rho_eps"] = np.zeros((n_keep, M))
    acc_accum = {"path_u": 0.0, "path_eps": 0.0, "sigma2": 0.0,
                 "rho_u": 0.0, "rho_eps": 0.0}
    acc_count = 0

    t0 = time.time()
    keep = 0
    for it in range(n_iter):
        theta_cur = {"A": A, "Q": Q, "Lambda": Lambda, "R": R, "Sigma_0": Sigma_0}

        # ── (a) States: FFBS (fed combined precision g = w / h when sv) ──────
        if sv:
            st = ffbs_sample_states(Y, theta_cur, w_u, w_eps, freq_list, rng,
                                    h_u=h_u, h_eps=h_eps)
        else:
            st = ffbs_sample_states(Y, theta_cur, w_u, w_eps, freq_list, rng)
        f_aug = st["f_aug"]
        loglik = st["loglik"]

        # ── (b) Volatility paths: KSC base, or leverage (A=MH / B=Omori+FFBS) ─
        if sv and leverage:
            lev_sampler = (sample_volatility_block_leverage
                           if timing == "contemporaneous"
                           else sample_volatility_block_leverage_lagged)
            vb = lev_sampler(
                Y, f_aug, theta_cur, w_u, w_eps, logh_u, logh_eps,
                sv_u, sv_eps, rho_u, rho_eps, rng,
                prior_a=sv_prior_a, prior_b=sv_prior_b,
                prop_path=lev_prop_path, prop_sigma2=lev_prop_sigma2,
                prop_rho=lev_prop_rho,
            )
            h_u, h_eps = vb["h_u"], vb["h_eps"]
            logh_u, logh_eps = vb["logh_u"], vb["logh_eps"]
            sv_u, sv_eps = vb["sv_u"], vb["sv_eps"]
            rho_u, rho_eps = vb["rho_u"], vb["rho_eps"]
            for kk in acc_accum:
                acc_accum[kk] += vb["acc"][kk]
            acc_count += 1
        elif sv:
            vb = sample_volatility_block(
                Y, f_aug, theta_cur, w_u, w_eps, logh_u, logh_eps,
                tuple(sv_u), sv_eps, rng, offset=sv_offset, fix_mu0=sv_fix_mu0,
                prior_a=sv_prior_a, prior_b=sv_prior_b,
            )
            h_u, h_eps = vb["h_u"], vb["h_eps"]
            logh_u, logh_eps = vb["logh_u"], vb["logh_eps"]
            sv_u, sv_eps = vb["sv_u"], vb["sv_eps"]

        # ── (c) Weights: Gamma draws (residuals deflated by h) ──────────────
        Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)
        h_eps_arg = h_eps if sv else None
        h_u_arg = h_u if sv else None
        d_eps, m_obs = realized_deflated_d_eps(Y, f_aug, Lambda_tilde, R, W_list, h_eps=h_eps_arg)
        d_u = realized_deflated_d_u(f_aug, A, Q, r, h_u=h_u_arg)
        wd = draw_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r, rng)
        w_eps, w_u = wd["w_eps"], wd["w_u"]

        # ── (d) Parameters (combined precision g = w / h enters A,Q,Lambda,R) ─
        if sv:
            g_u = w_u / h_u
            g_eps = w_eps[:, None] / h_eps
        else:
            g_u = w_u
            g_eps = w_eps
        A, Q = draw_A_Q_block(f_aug, g_u, r, rng)
        Lambda, R = draw_Lambda_R_block(
            Y, f_aug, g_eps, block_map, freq_list, ordered_cols, r, rng
        )
        if update_nu:
            # nu conditionals use the raw tail weights w (h does not enter).
            nu_u = draw_nu_griddy(w_u, rng, nu_bounds=nu_bounds, grid_size=nu_grid_size)
            nu_eps = draw_nu_griddy(w_eps, rng, nu_bounds=nu_bounds, grid_size=nu_grid_size)

        # ── storage ─────────────────────────────────────────────────────────
        if it >= burn_in and ((it - burn_in) % thin == 0) and keep < n_keep:
            draws["A"][keep] = A
            draws["Q"][keep] = Q
            draws["Lambda"][keep] = Lambda
            draws["R"][keep] = R
            draws["nu_u"][keep] = nu_u
            draws["nu_eps"][keep] = nu_eps
            draws["loglik"][keep] = loglik
            if store_states:
                draws["F"][keep] = st["F"]
            if sv:
                draws["sv_u"][keep] = sv_u
                draws["sv_eps"][keep] = sv_eps
                if store_vol:
                    draws["h_u"][keep] = h_u
                    draws["h_eps"][keep] = h_eps
            if leverage:
                draws["rho_u"][keep] = rho_u
                draws["rho_eps"][keep] = rho_eps
            keep += 1

        if verbose and (it % max(1, n_iter // 20) == 0 or it == n_iter - 1):
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
            print(
                f"  [gibbs] it {it + 1:5d}/{n_iter}  "
                f"loglik={loglik:11.2f}  rho(A)={rho:.4f}  "
                f"nu_u={nu_u:7.2f}  nu_eps={nu_eps:7.2f}"
            )

    # Trim storage in case n_keep overshot (it never undershoots).
    if keep < n_keep:
        for k in list(draws):
            draws[k] = draws[k][:keep]

    theta_mean = {
        "A": draws["A"].mean(axis=0),
        "Q": draws["Q"].mean(axis=0),
        "Lambda": draws["Lambda"].mean(axis=0),
        "R": draws["R"].mean(axis=0),
        "nu_u": float(draws["nu_u"].mean()),
        "nu_eps": float(draws["nu_eps"].mean()),
        "Sigma_0": Sigma_0,
    }
    if sv:
        theta_mean["sv_u"] = draws["sv_u"].mean(axis=0)        # (mu, phi, sigma2)
        theta_mean["sv_eps"] = draws["sv_eps"].mean(axis=0)    # (M, 3)
    if leverage:
        theta_mean["rho_u"] = draws["rho_u"].mean(axis=0)      # (r,)
        theta_mean["rho_eps"] = draws["rho_eps"].mean(axis=0)  # (M,)

    meta = {
        "n_iter": n_iter, "burn_in": burn_in, "thin": thin, "n_keep": keep,
        "seed": seed, "T": T, "M": M, "r": r,
        "update_nu": update_nu, "sv": sv, "leverage": leverage, "timing": timing,
        "wall_clock_s": time.time() - t0,
    }
    if leverage and acc_count > 0:
        meta["acceptance"] = {k: v / acc_count for k, v in acc_accum.items()}
    if verbose:
        print(f"  [gibbs] done: {keep} kept draws in {meta['wall_clock_s']:.1f}s")
    return {"draws": draws, "theta_mean": theta_mean, "meta": meta}
