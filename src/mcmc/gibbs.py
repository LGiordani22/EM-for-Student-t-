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
    (d) parameters  A, Q           per-factor two-step (mcmc.shared.draw_A_Q_perfactor)
                                   or MNIW (mcmc.sample_params.draw_A_Q_block, no SV)
                    Lambda, R      NIG   (mcmc.sample_params.draw_Lambda_R_block)
                    nu_u, nu_eps   griddy-Gibbs (mcmc.sample_params.draw_nu_griddy)

Flags / nesting (``sec:gibbs-variants``; see :data:`VARIANT_FLAGS`).  Every cell
of the D1 x D2 grid is a *restriction* of this one sweep:

    ``sv=False``      block (b) skipped (``h == 1``, Families B/C gone); the
                      weights enter the filter and the moments exactly as in the
                      EM E-step — the "current model" row.
    ``sv_idio=False`` D2-a: block (b) drops its idiosyncratic half (``h^eps==1``),
                      Family B runs for the r common processes only.
    ``tails='gaussian'`` D1-a: ``w == 1``, block (c) and Family D omitted.
    ``leverage=True`` block (b) adds the leverage drift, Branch A
                      (``timing='contemporaneous'``, Metropolis path) or Branch B
                      (``timing='lagged'``, Omori mixture + FFBS path); Family C
                      draws rho.  ``leverage=False`` (rho == 0) is the symmetric
                      base case — the 7-component KSC mixture, no Omori.

``q_prior='huang_wand'`` swaps the inverse-Wishart prior on Q for the
hierarchical half-t construction (r auxiliary scales appended to Family A) — a
robustness switch, not a default.  ``use_asis`` toggles no block: it wraps the
Family B draw wherever a volatility is sampled.

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
    draw_A_Q_perfactor,
    draw_hw_aux,
    hw_iw_prior,
    realized_deflated_d_eps,
    realized_deflated_d_u,
)
from mcmc.sample_states import ffbs_sample_states
from mcmc.sample_params import draw_A_Q_block, draw_Lambda_R_block, draw_nu_griddy
from mcmc.sample_vol import sample_volatility_block_specII
from mcmc.sample_leverage import sample_volatility_block_leverage
from mcmc.sample_leverage_lagged import sample_volatility_block_leverage_lagged


# ─────────────────────────────────────────────────────────────────────────────
# The D1 x D2 grid as restrictions of the master sweep (sec:gibbs-variants)
# ─────────────────────────────────────────────────────────────────────────────

#: Flags that specialise the master sampler to each cell of Table
#: ``tab:gibbs-variants``.  The skeleton never changes — states (a), volatilities
#: (b), tails (c), parameters (d) — only *which* blocks execute.
#:
#: **Every cell that samples a common volatility does so under Specification II**
#: (``eq:vol-outside``, ``subsec:vol-placement``): ``H^u_t`` is diagonal with the
#: r *per-factor* volatilities and enters *outside* the mixing,
#: ``Var(u_t|.) = sqrt(H^u_t) Q sqrt(H^u_t) / w^u_t``.  Restricting a cell (D1-a,
#: D2-a) switches *blocks* off; it never changes that sandwich, and it never turns
#: the r per-factor h^u into one scalar.  (The only place h is literally 1 is where
#: the .tex freezes it: h^eps == 1 under D2-a, and h == 1 in the no-SV row, where
#: sigma_eta^2 = 0 makes phi inoperative — subsec:variants-currentmodel.)
#:
#:  * ``D2-a`` (common volatility only) ⇒ ``sv_idio=False``: step (b) omits the
#:    idiosyncratic part, ``h^eps ≡ 1``, Family~B drawn for the ``r`` common
#:    processes only (not all ``M+r``).
#:  * ``D1-a`` (Gaussian, pure SV) ⇒ ``tails="gaussian"``: ``w ≡ 1``, step (c)
#:    and Family~D are omitted; the combined precision collapses to ``g=1/h``.
#:  * ``D1-c`` (explicit outliers) *replaces* step (c) and Family~D by the
#:    outlier indicators/parameters — **not implemented** (no outlier block in
#:    this engine); ``variant_kwargs`` raises rather than silently mis-mapping.
#:  * ``rho = 0`` ⇒ ``leverage=False``: the step-(b) leverage correction and
#:    Family~C drop out; the 7-component KSC mixture suffices (no Omori).
#:  * ``current model`` ⇒ ``sv=False``: the *entire* step (b) is omitted
#:    (``h ≡ 1``, ``phi`` inoperative), together with Family~B and Family~C —
#:    the MCMC counterpart of the first-stage EM.
#:
#: ASIS toggles no block: it is a wrapper on the Family~B draw and rides along
#: wherever a volatility is sampled (``subsec:variants-principle``).
VARIANT_FLAGS: dict[str, dict] = {
    "D1a x D2a": dict(sv=True, sv_idio=False, tails="gaussian", update_nu=False),
    "D1a x D2b": dict(sv=True, sv_idio=True, tails="gaussian", update_nu=False),
    "D1b x D2a": dict(sv=True, sv_idio=False, tails="student_t", update_nu=True),
    "D1b x D2b": dict(sv=True, sv_idio=True, tails="student_t", update_nu=True),   # master
    "current model": dict(sv=False, sv_idio=True, tails="student_t", update_nu=True),
}


def _chain_diagnostics(draws: dict, *, sv: bool, sv_idio: bool,
                       leverage: bool, student_t: bool) -> dict:
    r"""
    ESS and split-R-hat for the slowly-mixing parameters of the sweep — computed
    **from the stored draws**, after the loop.

    Pure instrumentation: it reads ``draws``, never writes them, and consumes **no**
    RNG.  The draws of a given seed are therefore bit-for-bit identical whether or
    not this runs (asserted by ``test_diagnostics`` [5]).

    Why it is not optional (``docs/audit_P1-P5.md`` §P6).  ``rho`` is updated by a
    single random-walk Metropolis move per sweep, sits on a posterior ridge with the
    log-vol path and ``sigma_eta^2``, and shows ``ESS ~ 3-23`` per 2000 draws — while
    being the parameter that gives the GDP predictive its **skew**.  A ``rho_hat``
    without its ESS is not a measurement.  The same ridge afflicts ``(phi, sigma^2)``
    (which is what ASIS exists for), so all three are reported.

    Note on ``split_r_hat`` from a **single** chain: it splits the chain in half and
    compares the halves, so it detects a chain that is still travelling (exactly the
    ``rho`` failure mode) — but it cannot detect a chain stuck in one of several
    modes.  For that, run several seeds.

    Returns a dict; per-factor quantities are reported channel by channel, per-series
    ones (``M`` of them) as a summary so the dict stays readable::

        {"rho_u":       {"ess": (r,), "r_hat": (r,)},
         "sv_u_phi":    {"ess": (r,), "r_hat": (r,)},
         "sv_u_sigma2": {"ess": (r,), "r_hat": (r,)},
         "rho_eps":     {"ess_min":…, "ess_median":…, "r_hat_max":…},
         "sv_eps_phi":  {…}, "sv_eps_sigma2": {…},
         "nu_u": {"ess":…, "r_hat":…}, "nu_eps": {…}}
    """
    from mcmc.diagnostics import ess, split_r_hat        # lazy: diagnostics imports gibbs

    def _scalar(x: np.ndarray) -> dict:
        ch = np.asarray(x, float)[None, :]               # (1, n_draws)
        return {"ess": float(ess(ch)), "r_hat": float(split_r_hat(ch))}

    def _per_column(X: np.ndarray) -> dict:
        """X is (n_draws, k) -> per-column ESS / R-hat."""
        X = np.asarray(X, float)
        return {"ess": np.array([ess(X[:, j][None, :]) for j in range(X.shape[1])]),
                "r_hat": np.array([split_r_hat(X[:, j][None, :]) for j in range(X.shape[1])])}

    def _summary(X: np.ndarray) -> dict:
        """X is (n_draws, k) with k large (per-series) -> a readable summary."""
        d = _per_column(X)
        e, rh = d["ess"], d["r_hat"]
        return {"ess_min": float(np.nanmin(e)), "ess_median": float(np.nanmedian(e)),
                "r_hat_max": float(np.nanmax(rh))}

    out: dict = {}
    n_keep = draws["A"].shape[0]
    if n_keep < 4:                                       # ess/split_r_hat need a chain
        return out

    if leverage:
        out["rho_u"] = _per_column(draws["rho_u"])
        if sv_idio and "rho_eps" in draws:
            out["rho_eps"] = _summary(draws["rho_eps"])
    if sv:
        out["sv_u_phi"] = _per_column(draws["sv_u"][:, :, 1])
        out["sv_u_sigma2"] = _per_column(draws["sv_u"][:, :, 2])
        if sv_idio and "sv_eps" in draws:
            out["sv_eps_phi"] = _summary(draws["sv_eps"][:, :, 1])
            out["sv_eps_sigma2"] = _summary(draws["sv_eps"][:, :, 2])
    if student_t:
        out["nu_u"] = _scalar(draws["nu_u"])
        out["nu_eps"] = _scalar(draws["nu_eps"])
    return out


def variant_kwargs(cell: str, *, leverage: bool = False,
                   timing: str = "contemporaneous") -> dict:
    r"""
    ``fit_dfm_mcmc`` keyword arguments that restrict the master sampler to the
    grid cell ``cell`` of ``tab:gibbs-variants``.

    ``leverage`` is the *first-level* choice (the dagger of the table: Family~C
    and the step-(b) leverage correction are active only if ``rho != 0``), and
    ``timing`` the *second-level* one (Branch~A contemporaneous / Branch~B
    lagged) — an internal bifurcation of the leveraged cell, not a restriction
    (``subsec:variants-leverage-levels``).  The ``current model`` cell has no
    volatility, hence no leverage.
    """
    key = cell.strip().lower().replace("×", "x").replace("*", "")
    table = {k.lower(): v for k, v in VARIANT_FLAGS.items()}
    if key.startswith("d1c"):
        raise NotImplementedError(
            "D1-c (explicit outliers) *replaces* the weights block and Family D "
            "by the outlier indicators and their parameters "
            "(subsec:variants-restrictions); no outlier block is implemented in "
            "this sampler."
        )
    if key not in table:
        raise KeyError(f"unknown grid cell {cell!r}; known: {sorted(VARIANT_FLAGS)}")
    kw = dict(table[key])
    if leverage and not kw["sv"]:
        raise ValueError("the 'current model' cell freezes all volatility, so "
                         "there is no log-vol innovation for leverage to "
                         "correlate with (subsec:variants-currentmodel).")
    kw["leverage"] = leverage
    if leverage:
        kw["timing"] = timing
    return kw


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
    from em.em_main import load_dfm_fit

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
    nu_log_prior=None,
    tails: str = "student_t",
    use_family_a_priors: bool = False,
    family_a_kappa: float = 1e-2,
    q_prior: str = "inverse_wishart",
    hw_nu_star: float = 2.0,
    hw_A: float = 1e5,
    sv: bool = False,
    sv_idio: bool = True,
    common_vol_coupling: str = "decoupled",
    sv_offset: float = 1e-6,
    sv_init: tuple[float, float, float] = (0.0, 0.95, 0.05),
    sv_fix_mu0: bool = True,
    sv_prior_a: float = 2.0,
    sv_prior_b: float = 0.05,
    sv_sigma_prior: str = "inverse_gamma",
    sv_half_normal_B: float = 1.0,
    use_asis: bool = False,
    leverage: bool = False,
    timing: str = "contemporaneous",
    lev_prop_path: float = 0.25,
    lev_prop_sigma2: float = 0.20,
    lev_prop_rho: float = 0.06,
    rho_sampler: str = "griddy",
    rho_grid_size: int = 401,
    rho_log_prior=None,
    store_states: bool = False,
    store_vol: bool = False,
    compute_diagnostics: bool = True,
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
    nu_log_prior : callable or None  Family D (Phase 5) proper prior on the d.o.f.
                              ``nu -> log p(nu)`` applied to both ``nu_u`` and
                              ``nu_eps``; ``None`` (default) is flat over the
                              bounded griddy bracket (already proper).  Build with
                              ``sample_params.nu_log_prior_exponential(mean=20)`` or
                              ``nu_log_prior_uniform(2, 50)``.
    tails : str               ``"student_t"`` (D1-b, default) or ``"gaussian"``
                              (D1-a): the latter sets ``w == 1`` and omits step (c)
                              and Family D altogether (sec:gibbs-variants).
    sv_idio : bool            ``False`` is D2-a (common volatility only): step (b)
                              omits the idiosyncratic block, ``h^eps == 1``, and
                              Family B is drawn for the r common processes only.
    common_vol_coupling : str  measurement coupling of the r per-factor common
                              volatilities, **no-leverage Spec II path only**:
                              ``"decoupled"`` (default, exact at diagonal Q),
                              ``"qml"`` (stable coupled — the Harvey-Ruiz-Shephard
                              constant-covariance form, a data-driven robustness
                              option, ``docs/audit_P1-P5.md`` §P4), or ``"literal"``
                              (experimental, unstable).  Raises under ``leverage``
                              (the common block is then r independent Omori channels,
                              so it would be a silent no-op).  Inspect the
                              recommendation with
                              :func:`mcmc.diagnostics.recommend_coupling`.
    use_family_a_priors : bool  turn on the weakly-informative Family A priors
                              (tab:param-prior-tuning), anchored on the EM warm
                              start; ``False`` (default) keeps the flat-prior
                              limit, bit-identical to the EM seam.  Wired on every
                              path: the Spec II per-factor draw takes the general
                              Gaussian prior on ``vec(A)`` (``V0^{-1}=kappa I``),
                              the scalar/no-SV MNIW draw its natural-conjugate
                              matrix-Normal counterpart (same ``kappa``).
    family_a_kappa : float    prior precision scale: ``V0^{-1}=kappa I`` on
                              vec(A) and ``M0^{-1}=kappa`` on each loading
                              (``~1e-2``, weak); only used when the priors are on.
    q_prior : str             ``"inverse_wishart"`` (default: the plain IW of
                              eq:param-Q-post, hyperparameters from
                              ``use_family_a_priors``) or ``"huang_wand"`` — the
                              optional hierarchical IW with half-t margins
                              (eq:param-Q-hw-prior), which severs the
                              variance–correlation link while keeping the Gibbs
                              draw, via r auxiliary scales ``a_j`` appended to
                              Family A (eq:param-Q-hw-aux).  Not a default: it is
                              the *robustness check* the thesis prescribes —
                              re-estimate and confirm the posterior correlations
                              among factor innovations do not move.
    hw_nu_star, hw_A : float  Huang--Wand dials: ``nu*=2`` gives half-t_2
                              (near-half-Cauchy) margins on each ``sqrt(Q_jj)``
                              *and* uniform marginal correlations; ``A`` is the
                              common half-t scale, set large relative to the
                              innovation scale of the standardised panel
                              (``~10`` mildly informative, ``1e5`` near-flat).
    store_states : bool       also store the per-draw head factor path F (T, r)
                              (memory ~ n_keep * T * r floats).
    rho_sampler : str         Family C kernel, **Branch B only** (fix P6):
                              ``"griddy"`` (default) draws rho from its full
                              conditional on the compact (-1,1) support, independently
                              of the current value; ``"rw"`` is the RW-Metropolis
                              baseline.  Branch A is untouched and always uses the RW.
                              Under the griddy ``meta["acceptance"]["rho_*"] == 1.0``
                              by construction — read ESS, not acceptance.
    rho_grid_size : int       griddy grid points on (-1, 1).
    rho_log_prior : callable or None  ``rho -> log p(rho)``; ``None`` is the flat
                              Uniform(-1,1) default of the thesis.  The hook for the
                              Fisher-z shrinkage it names as the alternative.
    compute_diagnostics : bool  compute ESS and split-R-hat for rho, phi, sigma^2
                              (and nu) from the stored draws and return them under
                              ``res["diagnostics"]``.  Default **on**: rho mixes so
                              badly (audit P6) that a rho_hat without its ESS is not
                              a measurement.  Pure instrumentation — it reads the
                              draws and consumes no RNG, so switching it off cannot
                              change a single draw (asserted in ``test_diagnostics``).
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
        ``diagnostics`` : dict — ESS / split-R-hat for ``rho_u`` (per factor),
            ``sv_u_phi``, ``sv_u_sigma2`` (per factor), ``nu_*``, and a summary
            (min / median ESS, max R-hat) for the ``M`` per-series quantities.
            Present iff ``compute_diagnostics``.
    """
    if tails not in ("student_t", "gaussian"):
        raise ValueError(
            f"tails={tails!r}: 'student_t' (D1-b) or 'gaussian' (D1-a). D1-c "
            f"(explicit outliers) replaces the weights block and Family D by the "
            f"outlier indicators — not implemented (see variant_kwargs)."
        )
    if q_prior not in ("inverse_wishart", "huang_wand"):
        raise ValueError(f"q_prior={q_prior!r}: 'inverse_wishart' or 'huang_wand'.")
    if common_vol_coupling not in ("decoupled", "qml", "literal"):
        raise ValueError(f"common_vol_coupling={common_vol_coupling!r}: "
                         f"'decoupled', 'qml' or 'literal'.")
    if common_vol_coupling != "decoupled" and leverage:
        # Under leverage the common block is r independent Omori/FFBS channels, not
        # sample_common_vol_mv, so the coupling would be a silent no-op (audit P1,
        # .tex subsec:lev-branches-allproc (iii)).  Fail loudly instead.
        raise ValueError(
            f"common_vol_coupling={common_vol_coupling!r} is only reachable on the "
            f"no-leverage Spec II path: under leverage the common block is r "
            f"independent Omori/FFBS channels, with no joint measurement covariance "
            f"to couple. Use leverage=False, or common_vol_coupling='decoupled'."
        )
    if leverage and not sv:
        raise ValueError(
            "leverage=True requires sv=True: with the volatility frozen "
            "(sigma_eta^2 = 0, h == 1) there is no log-vol innovation for rho to "
            "correlate with (subsec:variants-currentmodel)."
        )
    student_t = (tails == "student_t")
    if not student_t:
        # D1-a: w == 1, the weights block (c) and Family D (nu) are omitted.
        update_nu = False

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

    # ── Family A weakly-informative priors (tab:param-prior-tuning) ──────────
    # Anchored ("bridged") on the EM warm start, frozen from the *initial* theta
    # before the sweep mutates A/Q/Lambda/R.  Off by default: the flat limit
    # (Psi0/nu0/A0/V0_inv = None/0, a0/b0/m0/M0_inv = 0) keeps every EM seam
    # bit-identical.  When on, the recommended defaults are wired here and reach
    # the conjugate draw_* kernels of Family A "by a change of arguments"
    # (thesis §From table to code).  Two shapes of the same (A, Q) prior:
    #   * per-factor (Spec II, draw_A_Q_perfactor): the general Gaussian prior on
    #     vec(A), V0^{-1} = kappa I (eq:param-AQ — no Kronecker structure needed,
    #     the posterior is not matrix-Normal anyway);
    #   * scalar / no-SV (draw_A_Q_block): its natural-conjugate matrix-Normal
    #     counterpart A | Q ~ MN(A0, Q, (kappa I)^{-1}), which keeps the joint MNIW.
    if use_family_a_priors:
        aq_prior_pf = {
            "A0": A.copy(),                        # A0 = A_hat_EM (or 0)
            "V0_inv": family_a_kappa * np.eye(r * r),   # V0^{-1} = kappa I, kappa~1e-2
            "Psi0": (2.0 * r + 2.0) * Q.copy(),    # Psi0 = (2r+2) Q_hat_EM (mode at EM)
            "nu0": float(r + 1),                   # nu0 = r+1 -> uniform marginal corr.
        }
        aq_prior_mniw = {
            "A0": A.copy(),
            "kappa": family_a_kappa,
            "Psi0": (2.0 * r + 2.0) * Q.copy(),
            "nu0": float(r + 1),
        }
        lr_prior = {
            "a0": 2.0,                             # IG shape (2a0=4 pseudo-obs)
            "b0": (2.0 - 1.0) * R.copy(),          # b0_i=(a0-1) r_hat_i -> mean r_hat_i
            "m0": Lambda.copy(),                   # (M, r): series i reads m0[i, j]=L_hat_i
            "M0_inv": family_a_kappa,              # loading prior precision kappa
        }
    else:
        aq_prior_pf = {}
        aq_prior_mniw = {}
        lr_prior = {}

    # ── Huang--Wand hierarchical IW on Q (optional, eq:param-Q-hw-prior) ──────
    # Conditional on the r auxiliary scales a_j the prior is an ordinary IW, so
    # the Family A draw runs unchanged with (Psi0, nu0) swapped for
    # (2 nu* diag(1/a), nu* + r - 1); the a_j are r scalar IG draws appended to
    # the block (eq:param-Q-hw-aux).  It *overrides* the plain-IW (Psi0, nu0) —
    # the A and (Lambda, R) priors, if on, are untouched.
    hw = (q_prior == "huang_wand")

    # ASIS (Phase 6) interweaves the Family B sigma_eta draw; CP and NCP must share
    # the Gaussian prior on sigma_eta, so ASIS forces the half-Normal (sec:asis).
    if use_asis and sv_sigma_prior != "half_normal":
        sv_sigma_prior = "half_normal"

    rng = np.random.default_rng(seed)
    # a_j initialised from its own full conditional at the warm-start Q.
    hw_a = draw_hw_aux(Q, rng, nu_star=hw_nu_star, A_scales=hw_A) if hw else None
    W_list = build_all_selection_matrices(Y)

    # Warm-start weights at 1 (no down-weighting yet) — the FFBS at the very
    # first sweep needs weights before step (c) produces them.
    w_u = np.ones(T)
    w_eps = np.ones(T)

    # Stochastic-volatility state (Passo 2).  h == 1 (log h == 0) at start; the
    # combined precision is g = w / h, so with h == 1 the SV path reduces to
    # Passo 1.  Under Specification II the common block carries r *per-factor*
    # volatilities h^u_{k,·} (state (T, r), params (r, 3)) — M+r processes.
    # Per-factor holds for ALL SV paths: no leverage, contemporaneous leverage
    # (Branch A) and lagged leverage (Branch B, Omori), all Option A (r scalar
    # rho_i on the raw shock z^u = sqrt(w) Q^{-1/2}(sqrt H)^{-1} u).  With
    # ``sv=False`` the whole of step (b) is omitted and h stays at 1.
    logh_eps = np.zeros((T, M))
    h_eps = np.ones((T, M))
    sv_eps = np.tile(np.array(sv_init, dtype=float), (M, 1))
    logh_u = np.zeros((T, r)); h_u = np.ones((T, r))
    sv_u = np.tile(np.array(sv_init, dtype=float), (r, 1))          # (r, 3)
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
        "loglik": np.zeros(n_keep),
    }
    if student_t:                       # Family D exists only under D1-b
        draws["nu_u"] = np.zeros(n_keep)
        draws["nu_eps"] = np.zeros(n_keep)
    if store_states:
        draws["F"] = np.zeros((n_keep, T, r))
    if hw:
        draws["hw_a"] = np.zeros((n_keep, r))    # the r auxiliary half-t scales
    if sv:
        draws["sv_u"] = np.zeros((n_keep, r, 3))
        if store_vol:
            draws["h_u"] = np.zeros((n_keep, T, r))
        if sv_idio:                     # omitted under D2-a (h^eps frozen at 1)
            draws["sv_eps"] = np.zeros((n_keep, M, 3))
            if store_vol:
                draws["h_eps"] = np.zeros((n_keep, T, M))
    if leverage:
        draws["rho_u"] = np.zeros((n_keep, r))
        if sv_idio:
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
            # The Family C griddy (fix P6) is wired on Branch B only — Branch A keeps
            # its RW-Metropolis untouched, by explicit scope decision.
            lev_kw = {} if timing == "contemporaneous" else {
                "rho_sampler": rho_sampler, "rho_grid_size": rho_grid_size,
                "rho_log_prior": rho_log_prior,
            }
            vb = lev_sampler(
                Y, f_aug, theta_cur, w_u, w_eps, logh_u, logh_eps,
                sv_u, sv_eps, rho_u, rho_eps, rng,
                prior_a=sv_prior_a, prior_b=sv_prior_b,
                sigma_prior=sv_sigma_prior, half_normal_B=sv_half_normal_B,
                use_asis=use_asis, fix_mu0=sv_fix_mu0, sv_idio=sv_idio,
                prop_path=lev_prop_path, prop_sigma2=lev_prop_sigma2,
                prop_rho=lev_prop_rho, **lev_kw,
            )
            h_u, h_eps = vb["h_u"], vb["h_eps"]
            logh_u, logh_eps = vb["logh_u"], vb["logh_eps"]
            sv_u, sv_eps = vb["sv_u"], vb["sv_eps"]
            rho_u, rho_eps = vb["rho_u"], vb["rho_eps"]
            for kk in acc_accum:
                acc_accum[kk] += vb["acc"][kk]
            acc_count += 1
        elif sv:
            # per-factor common volatilities (Spec II) + idiosyncratic, no leverage.
            vb = sample_volatility_block_specII(
                Y, f_aug, theta_cur, w_u, w_eps, logh_u, logh_eps,
                sv_u, sv_eps, rng, offset=sv_offset, fix_mu0=sv_fix_mu0,
                prior_a=sv_prior_a, prior_b=sv_prior_b,
                sigma_prior=sv_sigma_prior, half_normal_B=sv_half_normal_B,
                use_asis=use_asis, sv_idio=sv_idio,
                common_vol_coupling=common_vol_coupling,
            )
            h_u, h_eps = vb["h_u"], vb["h_eps"]
            logh_u, logh_eps = vb["logh_u"], vb["logh_eps"]
            sv_u, sv_eps = vb["sv_u"], vb["sv_eps"]

        # ── (c) Weights: Gamma draws (residuals deflated by h) ──────────────
        # Omitted under D1-a (Gaussian): w == 1 throughout, and with it Family D.
        if student_t:
            Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)
            h_eps_arg = h_eps if sv else None
            h_u_arg = h_u if sv else None
            d_eps, m_obs = realized_deflated_d_eps(Y, f_aug, Lambda_tilde, R, W_list,
                                                   h_eps=h_eps_arg)
            d_u = realized_deflated_d_u(f_aug, A, Q, r, h_u=h_u_arg)
            wd = draw_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r, rng)
            w_eps, w_u = wd["w_eps"], wd["w_u"]

        # ── (d) Parameters (combined precision g = w / h enters A,Q,Lambda,R) ─
        # Huang--Wand: swap the IW hyperparameters for the ones implied by the
        # current auxiliary scales (eq:param-Q-hw-post); everything else unchanged.
        if hw:
            Psi0_t, nu0_t = hw_iw_prior(hw_a, hw_nu_star)
        if sv:
            # Spec II factor side: matrix whitening sqrt(H^u) — the per-factor
            # (A, Q) draw (eq:param-AQ), conditioning Q|A on the current A.
            aq_kw = dict(aq_prior_pf)
            if hw:
                aq_kw["Psi0"], aq_kw["nu0"] = Psi0_t, nu0_t
            A, Q = draw_A_Q_perfactor(f_aug[:, :r], h_u, w_u, A, rng, **aq_kw)
            g_eps = w_eps[:, None] / h_eps
        else:
            # No SV (h == 1): G^u_t = w^u_t Q^{-1} factors out and the closed
            # matrix-normal--inverse-Wishart draw of the scalar case is recovered.
            aq_kw = dict(aq_prior_mniw)
            if hw:
                aq_kw["Psi0"], aq_kw["nu0"] = Psi0_t, nu0_t
            g_eps = w_eps
            A, Q = draw_A_Q_block(f_aug, w_u, r, rng, **aq_kw)
        if hw:
            hw_a = draw_hw_aux(Q, rng, nu_star=hw_nu_star, A_scales=hw_A)
        Lambda, R = draw_Lambda_R_block(
            Y, f_aug, g_eps, block_map, freq_list, ordered_cols, r, rng, **lr_prior
        )
        if update_nu:
            # nu conditionals use the raw tail weights w (h does not enter).
            # nu_log_prior (Family D, Phase 5): None => flat over the griddy bracket
            # (already a proper bounded-uniform); a callable adds a proper prior on
            # (2, inf) — e.g. sample_params.nu_log_prior_exponential(mean=20).
            nu_u = draw_nu_griddy(w_u, rng, nu_bounds=nu_bounds, grid_size=nu_grid_size,
                                  log_prior=nu_log_prior)
            nu_eps = draw_nu_griddy(w_eps, rng, nu_bounds=nu_bounds, grid_size=nu_grid_size,
                                    log_prior=nu_log_prior)

        # ── storage ─────────────────────────────────────────────────────────
        if it >= burn_in and ((it - burn_in) % thin == 0) and keep < n_keep:
            draws["A"][keep] = A
            draws["Q"][keep] = Q
            draws["Lambda"][keep] = Lambda
            draws["R"][keep] = R
            draws["loglik"][keep] = loglik
            if student_t:
                draws["nu_u"][keep] = nu_u
                draws["nu_eps"][keep] = nu_eps
            if store_states:
                draws["F"][keep] = st["F"]
            if hw:
                draws["hw_a"][keep] = hw_a
            if sv:
                draws["sv_u"][keep] = sv_u
                if store_vol:
                    draws["h_u"][keep] = h_u
                if sv_idio:
                    draws["sv_eps"][keep] = sv_eps
                    if store_vol:
                        draws["h_eps"][keep] = h_eps
            if leverage:
                draws["rho_u"][keep] = rho_u
                if sv_idio:
                    draws["rho_eps"][keep] = rho_eps
            keep += 1

        if verbose and (it % max(1, n_iter // 20) == 0 or it == n_iter - 1):
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
            tail_txt = (f"  nu_u={nu_u:7.2f}  nu_eps={nu_eps:7.2f}" if student_t
                        else "  (gaussian)")
            print(
                f"  [gibbs] it {it + 1:5d}/{n_iter}  "
                f"loglik={loglik:11.2f}  rho(A)={rho:.4f}{tail_txt}"
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
        "Sigma_0": Sigma_0,
    }
    if student_t:
        theta_mean["nu_u"] = float(draws["nu_u"].mean())
        theta_mean["nu_eps"] = float(draws["nu_eps"].mean())
    if sv:
        theta_mean["sv_u"] = draws["sv_u"].mean(axis=0)        # (r, 3)
        if sv_idio:
            theta_mean["sv_eps"] = draws["sv_eps"].mean(axis=0)    # (M, 3)
    if leverage:
        theta_mean["rho_u"] = draws["rho_u"].mean(axis=0)      # (r,)
        if sv_idio:
            theta_mean["rho_eps"] = draws["rho_eps"].mean(axis=0)  # (M,)

    meta = {
        "n_iter": n_iter, "burn_in": burn_in, "thin": thin, "n_keep": keep,
        "seed": seed, "T": T, "M": M, "r": r,
        "update_nu": update_nu, "sv": sv, "sv_idio": sv_idio, "tails": tails,
        "leverage": leverage, "timing": timing, "q_prior": q_prior,
        "use_family_a_priors": use_family_a_priors,
        "wall_clock_s": time.time() - t0,
    }
    if leverage and acc_count > 0:
        meta["acceptance"] = {k: v / acc_count for k, v in acc_accum.items()}

    # ── convergence diagnostics (pure instrumentation: reads draws, no RNG) ──
    # Mandatory by default: no rho_hat without its ESS (audit P6).
    res = {"draws": draws, "theta_mean": theta_mean, "meta": meta}
    if compute_diagnostics:
        res["diagnostics"] = _chain_diagnostics(
            draws, sv=sv, sv_idio=sv_idio, leverage=leverage, student_t=student_t)

    if verbose:
        print(f"  [gibbs] done: {keep} kept draws in {meta['wall_clock_s']:.1f}s")
        d = res.get("diagnostics", {})
        if "rho_u" in d:
            print(f"  [gibbs] ESS(rho_u)={np.round(d['rho_u']['ess'], 0)}  "
                  f"split-Rhat={np.round(d['rho_u']['r_hat'], 3)}")
        if "sv_u_phi" in d:
            print(f"  [gibbs] ESS(phi_u)={np.round(d['sv_u_phi']['ess'], 0)}  "
                  f"ESS(sigma2_u)={np.round(d['sv_u_sigma2']['ess'], 0)}")
    return res
