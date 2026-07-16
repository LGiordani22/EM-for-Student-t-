"""
src/mcmc/diagnostics.py
=======================

Cosa calcola (in due parole): le diagnostiche di convergenza (R-hat, ESS) e la
harness di recovery a 3 livelli — lo strumento di correttezza del sampler.  Non è un
motore del modello: non tocca le equazioni del sistema, misura se le catene hanno
recuperato la verità (parametri coperti dagli intervalli, percorsi correlati col vero).

Convergence diagnostics (R-hat, ESS) and the **3-level recovery harness** for
the Gibbs sampler.  The recovery test is the central correctness instrument:
MCMC bugs do not crash, they produce subtly wrong posteriors, so the only
reliable check is self-recovery — simulate from a known theta, verify the
sampler recovers it (parameters covered by credible intervals; latent paths
correlated with the truth; healthy R-hat / ESS), and cross-check the posterior
means against the EM fit on the same panel (the no-SV MCMC must reproduce the EM).

Reuses the existing simulator (:func:`simulate_dfm.simulate_dfm`), the EM fit
(:func:`em_main.fit_dfm`, used both as warm init and as the cross-check
baseline) and the rotational-alignment routines
(:mod:`monte_carlo_recovery`) — the model is one-factor-per-block, so the only
indeterminacy is a per-factor sign.
"""

from __future__ import annotations

import numpy as np

from mcmc.gibbs import fit_dfm_mcmc


# ─────────────────────────────────────────────────────────────────────────────
# Audit metrics (docs/audit_P1-P5.md) — pure functions of Q / of the draws.
#
# corr(Q) is the single number that closes P1, P4 and P5 at once: the decoupled
# common-volatility block is *exact* at diagonal Q; the coupling's benefit and the
# Branch-B leverage attenuation are both second-order in the off-diagonals.  These
# helpers make it inspectable at every run instead of asserted once.
# ─────────────────────────────────────────────────────────────────────────────

def _corr_from_cov(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(np.asarray(Q, float)))
    return np.asarray(Q, float) / np.outer(d, d)


def posterior_corr_Q(draws: dict, *, cred: float = 0.90) -> dict:
    r"""
    Posterior correlation matrix of the factor innovations, from ``draws["Q"]``.

    **Closes P1, P4 and P5 empirically.**  The decoupled block is exact at diagonal
    ``Q``; the coupled-``R_xi`` correction (P1/P4) and the Branch-B whitening
    attenuation (P5) are both second-order in ``corr(Q)``.  On the EM fit the
    off-diagonals are ``<= 0.099``; this recomputes them from the *posterior* draws,
    where the volatility may have moved them.

    It is also the check the thesis prescribes for the Huang--Wand switch
    (``eq:param-Q-hw-prior``, ".tex" ~20684): *re-estimate under the hierarchical
    prior and confirm the posterior correlations among the factor innovations do not
    move*.  Run it on both and compare ``mean``.

    Returns ``{"mean": (r,r), "lo": (r,r), "hi": (r,r), "max_offdiag": float}``
    (element-wise posterior mean and central ``cred`` credible band of the
    correlations).
    """
    Qs = np.asarray(draws["Q"], float)                  # (n_keep, r, r)
    C = np.array([_corr_from_cov(Q) for Q in Qs])       # (n_keep, r, r)
    a = 0.5 * (1.0 - float(cred))
    mean = C.mean(axis=0)
    off = mean[np.triu_indices_from(mean, k=1)]
    return {
        "mean": mean,
        "lo": np.quantile(C, a, axis=0),
        "hi": np.quantile(C, 1.0 - a, axis=0),
        "max_offdiag": float(np.max(np.abs(off))) if off.size else 0.0,
    }


def coupling_overconfidence(Q: np.ndarray) -> dict:
    r"""
    **P4.**  How overconfident the *decoupled* common-volatility block is, at a given
    ``Q``.

    The block treats the ``r`` per-factor log-squares ``y*_{k,t}`` as marginally
    independent.  They are in fact correlated through ``corr(Q)``, with log-square
    correlation ``R_xi[j,k] = g(corr(Q)_{jk}) / (pi^2/2)``
    (:func:`mcmc.sample_vol.logsq_corr_matrix`).  Ignoring that correlation credits
    the sampler with more independent information than there is; for ``r`` roughly
    equicorrelated channels the joint precision is inflated by

        sqrt(1 + (r-1) * mean_offdiag(R_xi))  -  1 .

    Second-order in ``corr(Q)``: ``+0.4%`` at ``corr(Q)=0.10`` (the real panel),
    ``+3.6%`` at ``0.30``, ``+42%`` at ``0.90``.  Under Spec II each channel carries
    its *own* state, so the effect on a single ``h_k`` is weaker still.

    Returns ``{"R_xi": (r,r), "max_R_xi": float, "overconfidence": float}``.
    """
    from mcmc.sample_vol import logsq_corr_matrix
    Q = np.asarray(Q, float)
    r = Q.shape[0]
    R_xi = logsq_corr_matrix(_corr_from_cov(Q))
    off = R_xi[np.triu_indices_from(R_xi, k=1)]
    mean_off = float(np.mean(off)) if off.size else 0.0
    return {
        "R_xi": R_xi,
        "max_R_xi": float(np.max(np.abs(off))) if off.size else 0.0,
        "overconfidence": float(np.sqrt(1.0 + (r - 1) * mean_off) - 1.0),
    }


def recommend_coupling(Q: np.ndarray, *, tol: float = 0.05) -> dict:
    r"""
    **The data-driven recommendation** for ``fit_dfm_mcmc(common_vol_coupling=...)``
    — the decision the thesis says to make *once*, from the estimated ``Q``, and hold
    fixed for the whole chain (``docs/audit_P1-P5.md`` §P4; ``.tex``
    ``subsec:vol-all-processes``, the cross-covariance caveat).

    It is **advice, not a switch**: the choice is yours.  The decoupled common-vol
    block is *exact* at diagonal ``Q`` and over-confident away from it; the QML
    coupled pass widens the posterior to its honest width (calibration, not point
    accuracy).  The relevant quantity is not ``corr(Q)`` itself but the induced
    over-confidence :func:`coupling_overconfidence` — second-order in the
    off-diagonals — so the threshold is on that:

    * ``overconfidence <= tol``  →  ``"decoupled"`` (default; on the real panel
      ``corr(Q) ~ 0.1`` gives ``~0.4%`` and this is what it returns);
    * ``overconfidence >  tol``  →  ``"qml"`` **worth considering** — but the thesis
      makes it a *joint* condition: switch only if the tail is *also* seen to be
      mis-calibrated (a PIT / coverage check on the second stage), never on this
      number alone.

    ``tol = 0.05`` (5% over-confidence, ``corr(Q) ~ 0.35``) is a deliberately loose
    default: below it the coupling buys nothing worth the doubled cost and the lost
    mixture refinement.  Run on the **posterior** ``Q`` draws (mean), not only the EM
    fit, since SV can move it.

    Returns ``{"overconfidence": float, "corr_max_offdiag": float,
    "recommend": "decoupled"|"qml", "tol": float, "note": str}``.
    """
    oc = coupling_overconfidence(Q)
    corr = _corr_from_cov(Q)
    off = corr[np.triu_indices_from(corr, k=1)]
    rec = "qml" if oc["overconfidence"] > tol else "decoupled"
    note = ("decoupled is near-exact here" if rec == "decoupled" else
            "consider qml IFF a PIT/coverage check also shows tail mis-calibration")
    return {
        "overconfidence": oc["overconfidence"],
        "corr_max_offdiag": float(np.max(np.abs(off))) if off.size else 0.0,
        "recommend": rec, "tol": float(tol), "note": note,
    }


def leverage_whitening_attenuation(Q: np.ndarray) -> dict:
    r"""
    **P5.**  The attenuation of the Branch-B leverage ``rho_k`` induced by the
    per-component magnitude whitening.

    Branch~B takes the *sign* from the fully-whitened raw shock
    ``d_k = sign(z^u_k)`` but the *magnitude* from the per-component
    ``|zbar_k| = exp(xi_k/2)`` — the linearisation that makes the FFBS possible
    (``eq:lev-omori-cond``).  Its Family~C regressor is therefore
    ``g_k = sign(z_k)|zbar_k|`` while the true drift is ``rho z_k``, so with
    ``Var(zbar_k) = 1``

        rho_hat / rho  =  E[|z_k| |zbar_k|]  =  lambda(c_k),
        lambda(c) = (2/pi) ( c*arcsin(c) + sqrt(1-c^2) ),
        c_k = Corr(z_k, zbar_k) = (Q^{1/2})_kk / sqrt(q_kk).

    ``lambda(1) = 1`` (exact at diagonal ``Q``), ``lambda(0) = 0.6366``.  It is always
    an **attenuation, never a sign flip** — the sign uses the full whitening.  An
    attenuated ``|rho|`` *understates* the predictive left skew, i.e. errs against
    the Growth-at-Risk objective; but on the real panel ``c_k >= 0.9986``, so the
    bias is ``-0.1%`` and no correction is warranted (``docs/audit_P1-P5.md`` §P5).

    Returns ``{"c": (r,), "lambda": (r,), "bias_pct": (r,)}``.
    """
    Q = np.asarray(Q, float)
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    Q_half = (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
    c = np.diag(Q_half) / np.sqrt(np.diag(Q))
    c = np.clip(c, -1.0, 1.0)
    lam = (2.0 / np.pi) * (c * np.arcsin(c) + np.sqrt(np.clip(1.0 - c * c, 0.0, None)))
    return {"c": c, "lambda": lam, "bias_pct": 100.0 * (lam - 1.0)}


# ─────────────────────────────────────────────────────────────────────────────
# Convergence diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def split_r_hat(chains: np.ndarray) -> float:
    r"""
    Split-:math:`\hat R` (Gelman-Rubin) for a scalar quantity.

    ``chains`` is ``(n_chains, n_draws)``.  Each chain is split in half (so a
    single long chain still yields a usable diagnostic), giving ``2*n_chains``
    sequences of length ``n_draws//2``.  Returns ``sqrt(var_hat / W)``.
    """
    chains = np.asarray(chains, dtype=float)
    m0, n0 = chains.shape
    half = n0 // 2
    if half < 2:
        return float("nan")
    seqs = np.vstack([chains[:, :half], chains[:, half:2 * half]])  # (2*m0, half)
    m, n = seqs.shape
    means = seqs.mean(axis=1)
    variances = seqs.var(axis=1, ddof=1)
    W = variances.mean()
    B = n * means.var(ddof=1)
    if W <= 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def ess(chains: np.ndarray) -> float:
    r"""
    Effective sample size for a scalar, pooled across chains via the
    autocorrelation-based estimator (Stan/Geyer initial-positive-sequence).

    ``chains`` is ``(n_chains, n_draws)``.  Returns the total ESS.
    """
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if n < 4:
        return float("nan")

    # Per-chain autocovariance via FFT, averaged across chains (variogram form).
    means = chains.mean(axis=1, keepdims=True)
    centered = chains - means
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    acov = np.zeros(n)
    for c in range(m):
        f = np.fft.rfft(centered[c], n=nfft)
        ac = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
        acov += ac / n
    acov /= m
    if acov[0] <= 0:
        return float("nan")
    rho = acov / acov[0]

    # Geyer initial positive sequence: sum paired autocorrelations while positive.
    tau = 1.0
    t = 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
        t += 2
    return float(m * n / tau)


def loadings_unit_factor(Lambda: np.ndarray, F: np.ndarray) -> np.ndarray:
    r"""
    Loadings in the unit-variance-factor scale: ``Lambda_.k * sd(f_k)``.

    ``Y = Lambda f + eps`` is invariant under ``f_k -> c f_k, Lambda_.k -> Lambda_.k / c``,
    so the raw ``Lambda`` draws are not identified in scale; ``Lambda_.k * sd(f_k)`` is
    the identified, scale-invariant object.  Reads a coordinate — changes no draw.

    ``Lambda`` : (..., M, r),  ``F`` : (..., T, r)  ->  (..., M, r).
    """
    Lambda = np.asarray(Lambda, float)
    F = np.asarray(F, float)
    c = F.std(axis=-2)
    return Lambda * c[..., None, :]


def innovation_correlation(Q: np.ndarray) -> np.ndarray:
    r"""
    Factor-innovation correlation ``R_Q = D^{-1/2} Q D^{-1/2}`` (unit diagonal).

    The innovation covariance factors as ``Q = D^{1/2} R_Q D^{1/2}``: the scale ``D`` (its
    diagonal) is redundant with the factor scale — absorbed by ``Lambda`` and anchored by
    ``mu_h`` — while the correlation ``R_Q`` is the identified, scale-invariant part.  The
    raw ``Q`` diagonal mixes badly for the same reason ``Lambda`` did (scale ridge); ``R_Q``
    is well identified.  Reads a coordinate — changes no draw.

    ``Q`` : (..., r, r)  ->  (..., r, r).
    """
    Q = np.asarray(Q, float)
    d = np.sqrt(np.einsum("...kk->...k", Q))
    return Q / (d[..., :, None] * d[..., None, :])


def diagnostics_table(per_chain: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    R-hat + ESS for each named scalar quantity.

    ``per_chain[name]`` is ``(n_chains, n_draws)``.  Returns
    ``{name: {"r_hat": ..., "ess": ...}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for name, ch in per_chain.items():
        out[name] = {"r_hat": split_r_hat(ch), "ess": ess(ch)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Recovery harness (Passo 1)
# ─────────────────────────────────────────────────────────────────────────────

def _diag_align(H: np.ndarray, A: np.ndarray, Q: np.ndarray,
                Lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a diagonal change of basis ``g = H^{-1} f`` (per-block sign+scale)
    to a draw — the diagonal case of :func:`monte_carlo_recovery.apply_factor_rotation`.

    With ``h = diag(H)``:
        Lambda -> Lambda H        = Lambda * h[None, :]
        A      -> H^{-1} A H       = A * (h[None, :] / h[:, None])
        Q      -> H^{-1} Q H^{-T}  = Q / (h[:, None] * h[None, :])
    """
    h = np.diag(H).astype(float)
    Lambda_new = Lambda * h[None, :]
    A_new = A * (h[None, :] / h[:, None])
    Q_new = Q / (h[:, None] * h[None, :])
    return A_new, Q_new, Lambda_new


def run_recovery_mcmc(
    theta_true: dict,
    *,
    T: int,
    seed: int = 12345,
    config_name: str = "small",
    n_iter: int = 3000,
    burn_in: int = 1000,
    thin: int = 2,
    n_chains: int = 3,
    ci: float = 0.90,
    em_max_iter: int = 300,
    verbose: bool = True,
) -> dict:
    r"""
    One full Passo-1 recovery replica: simulate -> EM (warm init + baseline) ->
    multi-chain Gibbs -> score (coverage, latent paths, MCMC diagnostics,
    EM cross-check).

    Parameters
    ----------
    theta_true : dict       DGP parameters (typically the real-data EM fit of the
                            config; ``A, Q, Lambda, R, nu_u, nu_eps`` required).
    T : int                 synthetic sample length.
    seed : int              master seed (simulation + chains derive from it).
    config_name : str       config providing the panel metadata (small/big).
    n_iter, burn_in, thin, n_chains : sampler schedule.
    ci : float              central credible-interval mass for coverage.
    em_max_iter : int       EM iterations for warm init / baseline.

    Returns
    -------
    dict with ``metrics`` (coverage, correlations, EM cross-check, diagnostics),
    ``sim``, ``em``, and the per-chain ``gibbs`` results.
    """
    import sys, pathlib
    _src = str(pathlib.Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data_loader import load_config
    from em.em_main import fit_dfm
    from simulate_dfm import simulate_dfm
    from monte_carlo_recovery import procrustes_block_diagonal

    cfg = load_config(config_name)
    ordered_cols = cfg["ORDERED_COLS"]
    block_map = cfg["BLOCK"]
    freq_list = [cfg["FREQ"][c] for c in ordered_cols]
    r = int(np.asarray(theta_true["A"]).shape[0])

    # ── 1. Simulate ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[1/5] Simulating T={T} panel from theta_true ({config_name}) ...")
    sim = simulate_dfm(
        theta=theta_true, T=T, freq_list=freq_list, block_map=block_map,
        ordered_cols=ordered_cols, r=r, seed=seed,
    )
    Y = sim["Y"]
    F_true = sim["F"]
    w_u_true, w_eps_true = sim["w_u_true"], sim["w_eps_true"]

    # ── 2. EM fit on the synthetic panel (warm init AND cross-check baseline) ─
    if verbose:
        print("[2/5] EM fit on the synthetic panel (warm init + baseline) ...")
    em = fit_dfm(
        Y=Y, theta_init=_pca_init(Y, ordered_cols, block_map, freq_list),
        freq_list=freq_list, block_map=block_map, ordered_cols=ordered_cols,
        verbose=False, save_path=None, max_iter=em_max_iter, use_full_elbo=True,
    )
    theta_em = em["theta"]
    theta_init_mcmc = dict(theta_em)
    theta_init_mcmc["Sigma_0"] = np.asarray(theta_em["Sigma_0"])

    # ── 3. Multi-chain Gibbs ────────────────────────────────────────────────
    if verbose:
        print(f"[3/5] Running {n_chains} Gibbs chains "
              f"(n_iter={n_iter}, burn_in={burn_in}, thin={thin}) ...")
    chains = []
    for c in range(n_chains):
        res = fit_dfm_mcmc(
            Y, theta_init_mcmc, freq_list, block_map, ordered_cols,
            n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed + 1000 * (c + 1),
            store_states=(c == 0), verbose=False,
        )
        chains.append(res)
        if verbose:
            print(f"      chain {c}: {res['meta']['n_keep']} draws, "
                  f"{res['meta']['wall_clock_s']:.1f}s")

    # ── 4. Per-draw scale+sign alignment to the true frame, then summaries ──
    # The block-diagonal DFM is invariant under a per-block factor rescaling
    # f_b -> c_b f_b (Lambda_b -> Lambda_b/c_b, A_ij -> A_ij c_i/c_j,
    # Q_ij -> Q_ij c_i c_j); with Sigma_0 held fixed this scale is only weakly
    # pinned, so each draw lives in its own frame.  We quotient it out exactly as
    # the EM recovery does: block-diagonal Procrustes maps each draw's Lambda
    # onto the truth (one signed scalar h_b per block), and the same H is applied
    # to (A, Q, Lambda, F).  This is the identified comparison.
    Lambda_true = np.asarray(theta_true["Lambda"])

    per_chain_A, per_chain_Q, per_chain_Lam = [], [], []
    A_draws, Q_draws, Lam_draws, R_draws, nu_u_draws, nu_eps_draws = [], [], [], [], [], []
    H_list = []  # per (chain,draw) diagonal H, in chain-major order
    for ch in chains:
        d = ch["draws"]
        cA, cQ, cLam, cH = [], [], [], []
        for k in range(d["A"].shape[0]):
            H = procrustes_block_diagonal(d["Lambda"][k], Lambda_true,
                                          ordered_cols=ordered_cols,
                                          block_map=block_map)  # (r, r) diagonal
            A_a, Q_a, Lam_a = _diag_align(H, d["A"][k], d["Q"][k], d["Lambda"][k])
            cA.append(A_a); cQ.append(Q_a); cLam.append(Lam_a); cH.append(H)
            A_draws.append(A_a); Q_draws.append(Q_a); Lam_draws.append(Lam_a)
        per_chain_A.append(np.array(cA)); per_chain_Q.append(np.array(cQ))
        per_chain_Lam.append(np.array(cLam))
        H_list.append(np.array(cH))
        R_draws.append(d["R"]); nu_u_draws.append(d["nu_u"]); nu_eps_draws.append(d["nu_eps"])
    A_draws = np.array(A_draws); Q_draws = np.array(Q_draws); Lam_draws = np.array(Lam_draws)
    R_draws = np.concatenate(R_draws)
    nu_u_draws = np.concatenate(nu_u_draws); nu_eps_draws = np.concatenate(nu_eps_draws)

    alpha = (1.0 - ci) / 2.0

    def _coverage(draws_stacked: np.ndarray, truth: np.ndarray) -> dict:
        lo = np.quantile(draws_stacked, alpha, axis=0)
        hi = np.quantile(draws_stacked, 1.0 - alpha, axis=0)
        inside = (truth >= lo) & (truth <= hi)
        return {"lo": lo, "hi": hi, "mean": draws_stacked.mean(axis=0),
                "inside": inside, "frac": float(np.mean(inside))}

    cov_A = _coverage(A_draws, np.asarray(theta_true["A"]))
    cov_Q = _coverage(Q_draws, np.asarray(theta_true["Q"]))
    cov_Lam = _coverage(Lam_draws, np.asarray(theta_true["Lambda"]))
    cov_R = _coverage(R_draws, np.asarray(theta_true["R"]).ravel())
    cov_nu_u = _coverage(nu_u_draws[:, None], np.array([float(theta_true["nu_u"])]))
    cov_nu_eps = _coverage(nu_eps_draws[:, None], np.array([float(theta_true["nu_eps"])]))

    # Loading coverage restricted to the structurally non-zero entries.
    nz = np.abs(np.asarray(theta_true["Lambda"])) > 1e-12
    cov_Lam_nz_frac = float(np.mean(cov_Lam["inside"][nz]))

    # ── 5. Latent-path + weight recovery (chain 0, stored states) ───────────
    # Align each stored F draw by its own H (F -> F @ H^{-T} = F / h), then mean.
    F_draws0 = chains[0]["draws"]["F"]                  # (n_keep, T, r)
    h0 = np.array([np.diag(Hk) for Hk in H_list[0]])    # (n_keep, r)
    F_aligned = (F_draws0 / h0[:, None, :]).mean(axis=0)  # (T, r)
    factor_corr = np.array([
        _abscorr(F_aligned[:, j], F_true[:, j]) for j in range(r)
    ])
    # weight recovery: re-derive the posterior-mean weights is non-trivial here;
    # instead correlate the EM weights (same panel) with the truth as a sanity
    # anchor, plus report nu recovery as the primary tail diagnostic.
    w_u_em = np.asarray(em["e_step_output"]["w_u"])
    w_eps_em = np.asarray(em["e_step_output"]["w_eps"])
    w_u_corr = _signedcorr(w_u_em, w_u_true)
    w_eps_corr = _signedcorr(w_eps_em, w_eps_true)

    # ── EM cross-check: MCMC posterior mean vs EM point (same alignment) ─────
    H_em = procrustes_block_diagonal(
        np.asarray(theta_em["Lambda"]), Lambda_true,
        ordered_cols=ordered_cols, block_map=block_map)  # (r, r) diagonal
    A_em_a, Q_em_a, Lam_em_a = _diag_align(
        H_em, np.asarray(theta_em["A"]), np.asarray(theta_em["Q"]),
        np.asarray(theta_em["Lambda"]),
    )
    xcheck = {
        "A_relerr": _relerr(cov_A["mean"], A_em_a),
        "Q_relerr": _relerr(cov_Q["mean"], Q_em_a),
        "Lambda_relerr": _relerr(cov_Lam["mean"][nz], Lam_em_a[nz]),
        "R_relerr": _relerr(cov_R["mean"], np.asarray(theta_em["R"]).ravel()),
        "nu_u_em": float(theta_em["nu_u"]), "nu_u_mcmc": float(cov_nu_u["mean"][0]),
        "nu_eps_em": float(theta_em["nu_eps"]), "nu_eps_mcmc": float(cov_nu_eps["mean"][0]),
    }

    # ── MCMC diagnostics: R-hat / ESS on key scalars across chains ──────────
    # Computed in the *aligned* (identified) frame for scale-dependent
    # quantities (Q, R); rho(A) and nu are alignment-invariant.
    rhoA_pc = np.array([
        [np.max(np.abs(np.linalg.eigvals(pcA[k]))) for k in range(pcA.shape[0])]
        for pcA in per_chain_A
    ])
    Q00_pc = np.array([pcQ[:, 0, 0] for pcQ in per_chain_Q])
    nu_u_pc = np.array([ch["draws"]["nu_u"] for ch in chains])
    nu_eps_pc = np.array([ch["draws"]["nu_eps"] for ch in chains])
    R0_pc = np.array([ch["draws"]["R"][:, 0] for ch in chains])
    per_chain = {
        "rho(A)": rhoA_pc, "Q[0,0]": Q00_pc, "nu_u": nu_u_pc,
        "nu_eps": nu_eps_pc, "R[0]": R0_pc,
    }
    diag = diagnostics_table(per_chain)

    metrics = {
        "coverage": {
            "A": cov_A["frac"], "Q": cov_Q["frac"], "Lambda_nz": cov_Lam_nz_frac,
            "R": cov_R["frac"], "nu_u": cov_nu_u["frac"], "nu_eps": cov_nu_eps["frac"],
        },
        "ci_level": ci,
        "factor_abscorr": factor_corr,
        "w_u_corr_em": w_u_corr, "w_eps_corr_em": w_eps_corr,
        "xcheck_mcmc_vs_em": xcheck,
        "diagnostics": diag,
        "align_h_mean": np.concatenate([np.array([np.diag(Hk) for Hk in Hc])
                                        for Hc in H_list], axis=0).mean(axis=0),
        "detail": {"A": cov_A, "Q": cov_Q, "Lambda": cov_Lam, "R": cov_R,
                   "nu_u": cov_nu_u, "nu_eps": cov_nu_eps},
    }
    if verbose:
        _print_recovery(metrics, theta_true)
    return {"metrics": metrics, "sim": sim, "em": em, "gibbs": chains}


# ─────────────────────────────────────────────────────────────────────────────
# Recovery harness (Passo 2 — base SV, no leverage)
# ─────────────────────────────────────────────────────────────────────────────

def run_recovery_mcmc_sv(
    theta_true: dict,
    *,
    sv_u: tuple[float, float, float],
    sv_eps: tuple[float, float, float],
    T: int,
    seed: int = 4321,
    config_name: str = "small",
    n_iter: int = 1500,
    burn_in: int = 500,
    thin: int = 2,
    n_chains: int = 2,
    ci: float = 0.90,
    em_max_iter: int = 250,
    verbose: bool = True,
) -> dict:
    r"""
    Passo-2 recovery: simulate with dynamic volatility (log-AR(1), rho=0) ->
    EM warm init -> multi-chain SV Gibbs -> score the volatility paths, the
    log-vol AR(1) parameters and theta, with explicit attention to the
    ``h^u`` vs ``nu_u`` separation flagged for scope D1-b.

    Identification notes used in the scoring:
      * the per-block factor scale is unidentified (as in Passo 1) -> theta is
        compared after block-diagonal Procrustes alignment;
      * the *level* of each log-vol process (``mu``) is confounded with the
        static scale (Q for h^u, r_i for h^eps_i), so log-vol paths are scored
        by **correlation** (level-invariant) and ``mu`` is reported but not
        required to be covered;  ``phi`` and ``sigma_eta^2`` ARE identified.
    """
    import sys, pathlib
    _src = str(pathlib.Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data_loader import load_config
    from em.em_main import fit_dfm
    from monte_carlo_recovery import procrustes_block_diagonal
    from mcmc.simulate_sv import simulate_dfm_sv

    cfg = load_config(config_name)
    ordered_cols = cfg["ORDERED_COLS"]; block_map = cfg["BLOCK"]
    freq_list = [cfg["FREQ"][c] for c in ordered_cols]
    r = int(np.asarray(theta_true["A"]).shape[0])

    if verbose:
        print(f"\n[1/4] Simulating SV panel T={T} ({config_name}) ...")
    sim = simulate_dfm_sv(theta_true, T=T, freq_list=freq_list, block_map=block_map,
                          ordered_cols=ordered_cols, r=r, seed=seed,
                          sv_u=sv_u, sv_eps=sv_eps)
    Y = sim["Y"]

    if verbose:
        print("[2/4] EM warm init on the SV panel ...")
    em = fit_dfm(Y=Y, theta_init=_pca_init(Y, ordered_cols, block_map, freq_list),
                 freq_list=freq_list, block_map=block_map, ordered_cols=ordered_cols,
                 verbose=False, save_path=None, max_iter=em_max_iter, use_full_elbo=True)
    theta_init = dict(em["theta"]); theta_init["Sigma_0"] = np.asarray(em["theta"]["Sigma_0"])

    if verbose:
        print(f"[3/4] {n_chains} SV Gibbs chains (n_iter={n_iter}) ...")
    chains = []
    for c in range(n_chains):
        res = fit_dfm_mcmc(Y, theta_init, freq_list, block_map, ordered_cols,
                           n_iter=n_iter, burn_in=burn_in, thin=thin,
                           seed=seed + 1000 * (c + 1), sv=True,
                           store_states=(c == 0), store_vol=(c == 0), verbose=False)
        chains.append(res)
        if verbose:
            print(f"      chain {c}: {res['meta']['n_keep']} draws, "
                  f"{res['meta']['wall_clock_s']:.1f}s")

    Lambda_true = np.asarray(theta_true["Lambda"])

    # ── theta coverage (block-Procrustes aligned, pooled) ───────────────────
    # Two frames are reported for Q, R:
    #   * raw     : Q, R as drawn (factor scale aligned only);
    #   * id.     : Q, R folded to the unit-vol-level frame by multiplying by
    #               exp(mu) of the matching log-vol process — because the SV
    #               model identifies only the products h^u_t Q and h^eps_{i,t} r_i,
    #               the split between the vol *level* (mu) and the static scale is
    #               free.  At unit vol level (mu=0, the DGP convention) Q, R are
    #               identified, so this is the fair comparison.
    A_draws, Q_draws, Qid_draws, Lam_draws = [], [], [], []
    R_draws, Rid_draws, nuu, nue = [], [], [], []
    for ch in chains:
        d = ch["draws"]
        mu_u = d["sv_u"][:, 0]                 # (n_keep,)
        mu_eps = d["sv_eps"][:, :, 0]          # (n_keep, M)
        for k in range(d["A"].shape[0]):
            H = procrustes_block_diagonal(d["Lambda"][k], Lambda_true,
                                          ordered_cols=ordered_cols, block_map=block_map)
            A_a, Q_a, Lam_a = _diag_align(H, d["A"][k], d["Q"][k], d["Lambda"][k])
            A_draws.append(A_a); Q_draws.append(Q_a); Lam_draws.append(Lam_a)
            Qid_draws.append(Q_a * np.exp(mu_u[k]))            # fold vol level into Q
            Rid_draws.append(d["R"][k] * np.exp(mu_eps[k]))    # fold vol level into R
        R_draws.append(d["R"]); nuu.append(d["nu_u"]); nue.append(d["nu_eps"])
    A_draws = np.array(A_draws); Q_draws = np.array(Q_draws); Lam_draws = np.array(Lam_draws)
    Qid_draws = np.array(Qid_draws); Rid_draws = np.array(Rid_draws)
    R_draws = np.concatenate(R_draws); nuu = np.concatenate(nuu); nue = np.concatenate(nue)
    alpha = (1 - ci) / 2

    def _covfrac(draws, truth):
        lo = np.quantile(draws, alpha, axis=0); hi = np.quantile(draws, 1 - alpha, axis=0)
        return float(np.mean((truth >= lo) & (truth <= hi)))
    nz = np.abs(Lambda_true) > 1e-12
    cov = {
        "A": _covfrac(A_draws, np.asarray(theta_true["A"])),
        "Q_raw": _covfrac(Q_draws, np.asarray(theta_true["Q"])),
        "Q_id": _covfrac(Qid_draws, np.asarray(theta_true["Q"])),
        "Lambda_nz": float(np.mean(((np.quantile(Lam_draws, alpha, 0) <= Lambda_true) &
                                    (np.quantile(Lam_draws, 1 - alpha, 0) >= Lambda_true))[nz])),
        "R_raw": _covfrac(R_draws, np.asarray(theta_true["R"]).ravel()),
        "R_id": _covfrac(Rid_draws, np.asarray(theta_true["R"]).ravel()),
    }

    # ── volatility recovery (chain 0, stored paths) ─────────────────────────
    d0 = chains[0]["draws"]
    logh_u_mean = np.log(d0["h_u"].mean(axis=0))
    h_u_corr = _signedcorr(logh_u_mean, sim["logh_u_true"])
    logh_eps_mean = np.log(d0["h_eps"].mean(axis=0))                 # (T, M)
    h_eps_corrs = np.array([_signedcorr(logh_eps_mean[:, i], sim["logh_eps_true"][:, i])
                            for i in range(Y.shape[1])])

    # log-vol AR(1) params (pooled): phi, sigma2 identified; mu confounded.
    sv_u_draws = np.concatenate([ch["draws"]["sv_u"] for ch in chains], axis=0)   # (N,3)
    sv_u_mean = sv_u_draws.mean(axis=0)
    sv_eps_draws = np.concatenate([ch["draws"]["sv_eps"] for ch in chains], axis=0)  # (N,M,3)
    sv_eps_mean = sv_eps_draws.mean(axis=0)

    # factor path corr (aligned)
    h0 = np.array([np.diag(procrustes_block_diagonal(Lk, Lambda_true,
                  ordered_cols=ordered_cols, block_map=block_map))
                  for Lk in d0["Lambda"]])
    F_aligned = (d0["F"] / h0[:, None, :]).mean(axis=0)
    factor_corr = np.array([_abscorr(F_aligned[:, j], sim["F"][:, j]) for j in range(r)])

    # ── h^u vs nu_u separation: nu_u CI and h^u variability ────────────────
    nu_u_lo, nu_u_hi = np.quantile(nuu, alpha), np.quantile(nuu, 1 - alpha)
    # per-draw path std (unbiased for the true path amplitude) vs std of the
    # posterior-MEAN path (always attenuated by averaging — reported for context).
    logh_u_std_perdraw = float(np.std(np.log(d0["h_u"]), axis=1).mean())
    sep = {
        "nu_u_true": float(theta_true["nu_u"]), "nu_u_mean": float(nuu.mean()),
        "nu_u_ci": (float(nu_u_lo), float(nu_u_hi)),
        "logh_u_std_postmean": float(logh_u_mean.std()),       # attenuated (mean path)
        "logh_u_std_perdraw": logh_u_std_perdraw,              # unbiased (per-draw)
        "logh_u_std_true": float(sim["logh_u_true"].std()),
        "sigma2_u_mean": float(sv_u_mean[2]),
        "h_u_corr": h_u_corr,
    }

    # ── diagnostics: R-hat / ESS (vol scalars included) ────────────────────
    per_chain = {
        "rho(A)": np.array([[np.max(np.abs(np.linalg.eigvals(ch["draws"]["A"][k])))
                             for k in range(ch["draws"]["A"].shape[0])] for ch in chains]),
        "nu_u": np.array([ch["draws"]["nu_u"] for ch in chains]),
        "nu_eps": np.array([ch["draws"]["nu_eps"] for ch in chains]),
        "sv_u.phi": np.array([ch["draws"]["sv_u"][:, 1] for ch in chains]),
        "sv_u.sig2": np.array([ch["draws"]["sv_u"][:, 2] for ch in chains]),
    }
    diag = diagnostics_table(per_chain)

    metrics = {
        "coverage": cov, "ci_level": ci, "factor_abscorr": factor_corr,
        "h_u_corr": h_u_corr, "h_eps_corr_mean": float(np.nanmean(h_eps_corrs)),
        "h_eps_corr_median": float(np.nanmedian(h_eps_corrs)),
        "sv_u_mean": sv_u_mean, "sv_u_true": (sv_u[0], sv_u[1], sv_u[2] ** 2),
        "sv_eps_phi_mean": float(sv_eps_mean[:, 1].mean()),
        "sv_eps_sig2_mean": float(sv_eps_mean[:, 2].mean()),
        "sv_eps_true": (sv_eps[0], sv_eps[1], sv_eps[2] ** 2),
        "nu_u_mean": float(nuu.mean()), "nu_eps_mean": float(nue.mean()),
        "separation": sep, "diagnostics": diag,
    }
    if verbose:
        _print_recovery_sv(metrics, theta_true)
    return {"metrics": metrics, "sim": sim, "em": em, "gibbs": chains}


def _print_recovery_sv(m, theta_true):
    print("\n" + "=" * 72)
    print(f"  RECOVERY (Passo 2, base SV) — {int(m['ci_level']*100)}% credible intervals")
    print("=" * 72)
    cov = m["coverage"]
    print(f"  theta coverage (aligned):  A={cov['A']*100:.0f}%  Lambda_nz={cov['Lambda_nz']*100:.0f}%")
    print(f"    Q: raw={cov['Q_raw']*100:.0f}% / identified(fold vol level)={cov['Q_id']*100:.0f}%   "
          f"R: raw={cov['R_raw']*100:.0f}% / identified={cov['R_id']*100:.0f}%")
    print(f"  factor |corr|: {', '.join(f'{c:.3f}' for c in m['factor_abscorr'])}")
    print(f"  VOL PATHS:  h^u corr(log)={m['h_u_corr']:.3f}   "
          f"h^eps corr(log) mean={m['h_eps_corr_mean']:.3f} median={m['h_eps_corr_median']:.3f}")
    svm, svt = m["sv_u_mean"], m["sv_u_true"]
    print(f"  h^u AR(1):  mu={svm[0]:+.2f}(true {svt[0]:+.2f}, confounded)  "
          f"phi={svm[1]:.3f}(true {svt[1]:.3f})  sig2={svm[2]:.4f}(true {svt[2]:.4f})")
    print(f"  h^eps AR(1) mean: phi={m['sv_eps_phi_mean']:.3f}(true {m['sv_eps_true'][1]:.3f})  "
          f"sig2={m['sv_eps_sig2_mean']:.4f}(true {m['sv_eps_true'][2]:.4f})")
    sep = m["separation"]
    print("  h^u vs nu_u SEPARATION:")
    print(f"    nu_u true={sep['nu_u_true']:.2f}  mean={sep['nu_u_mean']:.2f}  "
          f"CI=[{sep['nu_u_ci'][0]:.2f},{sep['nu_u_ci'][1]:.2f}]")
    print(f"    std(log h^u): per-draw={sep['logh_u_std_perdraw']:.3f} "
          f"postmean={sep['logh_u_std_postmean']:.3f} true={sep['logh_u_std_true']:.3f}  "
          f"(h^u corr={sep['h_u_corr']:.3f})")
    print(f"  nu_eps mean={m['nu_eps_mean']:.2f} (true {float(theta_true['nu_eps']):.2f})")
    print("  Convergence (R-hat / ESS):")
    for name, dd in m["diagnostics"].items():
        print(f"    {name:10s}: R-hat={dd['r_hat']:.3f}  ESS={dd['ess']:8.1f}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Recovery harness (Passo 3 — leverage, Branch A: contemporaneous + Metropolis)
# ─────────────────────────────────────────────────────────────────────────────

def _skewness(x):
    x = np.asarray(x, float); x = x - x.mean()
    s = np.sqrt(np.mean(x * x))
    return float(np.mean(x ** 3) / s ** 3) if s > 0 else float("nan")


def leverage_skewness_check(theta_true, sv_u, rho_u, *, T, seed, config_name="small",
                            timing="contemporaneous"):
    r"""DGP-level test that leverage ``rho<0`` produces a *left-skewed*
    factor-innovation density (the mechanism behind Growth-at-Risk skewness).
    ``timing`` selects the contemporaneous (Branch A) or lagged (Branch B) DGP.

    Measures the skewness of the factor innovation projected onto the
    *recessionary leverage direction* ``d = -Q^{1/2} rho`` (normalised).  By
    construction the level shock in this direction is negatively correlated with
    the vol driver (``(Q^{1/2} d) . rho = -rho'Q rho < 0`` for any ``rho != 0``),
    so high-volatility events fall in its negative tail -> a left-skewed
    predictive density.  This orientation is robust (independent of ``Q``'s
    off-diagonal signs, unlike a single block), and the same fixed direction is
    used for the ``rho=0`` control.  The tail degrees of freedom are pushed to
    (near-)Gaussian so the *only* source of skewness is the leverage, not the
    Student-t tails (whose sample skewness is extremely noisy); a long ``T`` and
    a moderate ``sigma`` keep the stochastic-volatility tail noise small.
    """
    import sys, pathlib
    _src = str(pathlib.Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data_loader import load_config
    from mcmc.simulate_sv import simulate_dfm_sv, _sqrt_spd

    cfg = load_config(config_name)
    oc = cfg["ORDERED_COLS"]; bm = cfg["BLOCK"]
    fl = [cfg["FREQ"][c] for c in oc]
    r = int(np.asarray(theta_true["A"]).shape[0])
    A = np.asarray(theta_true["A"]); Q = np.asarray(theta_true["Q"])
    rho_u = np.broadcast_to(np.asarray(rho_u, float), (r,)).copy()
    d = -_sqrt_spd(Q) @ rho_u                  # recessionary leverage direction
    d = d / np.linalg.norm(d)
    theta_g = dict(theta_true); theta_g["nu_u"] = 1000.0; theta_g["nu_eps"] = 1000.0

    # Under CONTEMPORANEOUS leverage the asymmetry is in the one-step innovation
    # u_t (z_t couples its own vol).  Under LAGGED leverage z_t couples eta_{t+1},
    # so u_t is symmetric and the asymmetry is cross-temporal — a large negative
    # shock today raises tomorrow's volatility — surfacing only in the *cumulative*
    # (multi-step) innovation.  We therefore project the cumulative innovation over
    # a short window for the lagged DGP (window 1 reproduces the contemporaneous
    # measure).
    win = 1 if timing == "contemporaneous" else 4

    def _proj_skew(rho_arg):
        sim = simulate_dfm_sv(theta_g, T=T, freq_list=fl, block_map=bm,
                              ordered_cols=oc, r=r, seed=seed, sv_u=sv_u,
                              sv_eps=(0.0, 0.0, 0.0), rho_u=rho_arg, rho_eps=None,
                              timing=timing)
        F = sim["F"]
        u = (F[1:] - F[:-1] @ A.T) @ d            # (T-1,) one-step innovation on d
        if win > 1:
            c = np.cumsum(u)
            u = c[win:] - c[:-win]                 # rolling win-step cumulative
        return _skewness(u)

    return {"skew_leverage": _proj_skew(rho_u), "skew_symmetric": _proj_skew(np.zeros(r)),
            "rho_u": rho_u, "timing": timing, "window": win}


def run_recovery_mcmc_leverage(
    theta_true: dict,
    *,
    rho_u: np.ndarray | float,
    rho_eps: float = 0.0,
    sv_u: tuple[float, float, float] = (0.0, 0.97, 0.22),
    sv_eps: tuple[float, float, float] = (0.0, 0.95, 0.15),
    T: int = 300,
    seed: int = 7777,
    config_name: str = "small",
    n_iter: int = 3000,
    burn_in: int = 1200,
    thin: int = 2,
    n_chains: int = 2,
    ci: float = 0.90,
    em_max_iter: int = 250,
    rho0_control: bool = True,
    timing: str = "contemporaneous",
    verbose: bool = True,
) -> dict:
    r"""
    Leverage recovery: Branch A (``timing='contemporaneous'``, Passo 3, Metropolis
    path) or Branch B (``timing='lagged'``, Passo 4, Omori sign-augmented mixture +
    FFBS path).  The DGP timing is matched to the sampler timing throughout
    (simulation, the rho=0 control and the skewness mechanism).

    Checks: sign/magnitude/precision of the recovered leverage ``rho`` (expected
    weakly identified for the common vector, like ``nu_u``); the skewness the
    leverage produces (DGP-level); the Metropolis acceptance rates; the
    diagnostics (R-hat/ESS, expected slow for the single-move MH); and that the
    rest (h, theta, nu, h^u-vs-nu_u) stays sane.  With ``rho0_control`` it also
    fits a ``rho=0`` DGP and checks the sampler does not invent leverage.
    """
    import sys, pathlib
    _src = str(pathlib.Path(__file__).resolve().parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from data_loader import load_config
    from em.em_main import fit_dfm
    from monte_carlo_recovery import procrustes_block_diagonal
    from mcmc.simulate_sv import simulate_dfm_sv

    cfg = load_config(config_name)
    oc = cfg["ORDERED_COLS"]; bm = cfg["BLOCK"]
    fl = [cfg["FREQ"][c] for c in oc]
    r = int(np.asarray(theta_true["A"]).shape[0])
    rho_u = np.broadcast_to(np.asarray(rho_u, float), (r,)).copy()

    if verbose:
        print(f"\n[1/5] Simulating {timing} leverage panel T={T} (rho_u={rho_u}) ...")
    sim = simulate_dfm_sv(theta_true, T=T, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=seed, sv_u=sv_u, sv_eps=sv_eps,
                          rho_u=rho_u, rho_eps=rho_eps, timing=timing)
    Y = sim["Y"]

    if verbose:
        print("[2/5] EM warm init ...")
    em = fit_dfm(Y=Y, theta_init=_pca_init(Y, oc, bm, fl), freq_list=fl, block_map=bm,
                 ordered_cols=oc, verbose=False, save_path=None, max_iter=em_max_iter,
                 use_full_elbo=True)
    theta_init = dict(em["theta"]); theta_init["Sigma_0"] = np.asarray(em["theta"]["Sigma_0"])

    if verbose:
        print(f"[3/5] {n_chains} leverage Gibbs chains (n_iter={n_iter}) ...")
    chains = []
    for c in range(n_chains):
        res = fit_dfm_mcmc(Y, theta_init, fl, bm, oc, n_iter=n_iter, burn_in=burn_in,
                           thin=thin, seed=seed + 1000 * (c + 1), sv=True, leverage=True,
                           timing=timing, store_states=(c == 0),
                           store_vol=(c == 0), verbose=False)
        chains.append(res)
        if verbose:
            print(f"      chain {c}: {res['meta']['n_keep']} draws, "
                  f"{res['meta']['wall_clock_s']:.0f}s, acc={ {k: round(v,2) for k,v in res['meta']['acceptance'].items()} }")

    Lambda_true = np.asarray(theta_true["Lambda"])
    # ── rho recovery (pooled) ────────────────────────────────────────────────
    rho_u_draws = np.concatenate([ch["draws"]["rho_u"] for ch in chains], axis=0)  # (N, r)
    rho_eps_draws = np.concatenate([ch["draws"]["rho_eps"] for ch in chains], axis=0)
    alpha = (1 - ci) / 2
    rho_u_mean = rho_u_draws.mean(axis=0)
    rho_u_lo = np.quantile(rho_u_draws, alpha, axis=0)
    rho_u_hi = np.quantile(rho_u_draws, 1 - alpha, axis=0)
    # dominant true component
    jdom = int(np.argmax(np.abs(rho_u)))
    rho_eps_mean = rho_eps_draws.mean(axis=0)

    # ── theta / vol sanity (block-Procrustes aligned Q raw cov) ──────────────
    Q_draws = []
    for ch in chains:
        d = ch["draws"]
        for k in range(d["A"].shape[0]):
            H = procrustes_block_diagonal(d["Lambda"][k], Lambda_true,
                                          ordered_cols=oc, block_map=bm)
            _, Q_a, _ = _diag_align(H, d["A"][k], d["Q"][k], d["Lambda"][k])
            Q_draws.append(Q_a)
    Q_draws = np.array(Q_draws)
    Q_cov = float(np.mean(((np.quantile(Q_draws, alpha, 0) <= np.asarray(theta_true["Q"])) &
                           (np.quantile(Q_draws, 1 - alpha, 0) >= np.asarray(theta_true["Q"])))))
    d0 = chains[0]["draws"]
    h_u_corr = _signedcorr(np.log(d0["h_u"].mean(axis=0)), sim["logh_u_true"])
    nuu = np.concatenate([ch["draws"]["nu_u"] for ch in chains])
    nue = np.concatenate([ch["draws"]["nu_eps"] for ch in chains])
    sv_u_mean = np.concatenate([ch["draws"]["sv_u"] for ch in chains], axis=0).mean(axis=0)

    # ── diagnostics ──────────────────────────────────────────────────────────
    per_chain = {
        f"rho_u[{jdom}]": np.array([ch["draws"]["rho_u"][:, jdom] for ch in chains]),
        "sv_u.phi": np.array([ch["draws"]["sv_u"][:, 1] for ch in chains]),
        "sv_u.sig2": np.array([ch["draws"]["sv_u"][:, 2] for ch in chains]),
        "nu_u": np.array([ch["draws"]["nu_u"] for ch in chains]),
    }
    diag = diagnostics_table(per_chain)
    acc = {k: float(np.mean([ch["meta"]["acceptance"][k] for ch in chains]))
           for k in chains[0]["meta"]["acceptance"]}

    # ── skewness (DGP mechanism; moderate sigma + long T to tame SV-tail noise) ─
    skew = leverage_skewness_check(theta_true, (0.0, 0.95, 0.15), rho_u, T=10000,
                                   seed=seed + 55, config_name=config_name, timing=timing)

    # ── rho=0 control: no false leverage ─────────────────────────────────────
    rho0 = None
    if rho0_control:
        if verbose:
            print("[4/5] rho=0 control fit (no false leverage) ...")
        sim0 = simulate_dfm_sv(theta_true, T=T, freq_list=fl, block_map=bm, ordered_cols=oc,
                               r=r, seed=seed + 9, sv_u=sv_u, sv_eps=sv_eps,
                               rho_u=np.zeros(r), rho_eps=0.0, timing=timing)
        em0 = fit_dfm(Y=sim0["Y"], theta_init=_pca_init(sim0["Y"], oc, bm, fl), freq_list=fl,
                      block_map=bm, ordered_cols=oc, verbose=False, save_path=None,
                      max_iter=em_max_iter, use_full_elbo=True)
        ti0 = dict(em0["theta"]); ti0["Sigma_0"] = np.asarray(em0["theta"]["Sigma_0"])
        res0 = fit_dfm_mcmc(sim0["Y"], ti0, fl, bm, oc, n_iter=n_iter, burn_in=burn_in,
                            thin=thin, seed=seed + 321, sv=True, leverage=True,
                            timing=timing, verbose=False)
        rd0 = res0["draws"]["rho_u"]
        rho0 = {"mean": rd0.mean(axis=0),
                "lo": np.quantile(rd0, alpha, axis=0), "hi": np.quantile(rd0, 1 - alpha, axis=0)}

    metrics = {
        "rho_u_true": rho_u, "rho_u_mean": rho_u_mean,
        "rho_u_ci": (rho_u_lo, rho_u_hi), "rho_u_norm_true": float(np.linalg.norm(rho_u)),
        "rho_u_norm_mean": float(np.linalg.norm(rho_u_mean)), "jdom": jdom,
        "rho_eps_true": float(sim["rho_eps_true"][0]) if rho_eps else 0.0,
        "rho_eps_mean": float(rho_eps_mean.mean()),
        "Q_cov": Q_cov, "h_u_corr": h_u_corr,
        "phi_u": float(sv_u_mean[1]), "sig2_u": float(sv_u_mean[2]),
        "nu_u_mean": float(nuu.mean()), "nu_eps_mean": float(nue.mean()),
        "acceptance": acc, "diagnostics": diag, "skewness": skew, "rho0_control": rho0,
        "ci": ci, "timing": timing,
    }
    if verbose:
        _print_recovery_lev(metrics, theta_true)
    return {"metrics": metrics, "sim": sim, "em": em, "gibbs": chains}


def _print_recovery_lev(m, theta_true):
    branch = ("Passo 3, Branch A (contemporaneous, Metropolis path)"
              if m.get("timing", "contemporaneous") == "contemporaneous"
              else "Passo 4, Branch B (lagged, Omori mixture + FFBS path)")
    print("\n" + "=" * 72)
    print(f"  RECOVERY ({branch}) — {int(m['ci']*100)}% CI")
    print("=" * 72)
    jd = m["jdom"]; lo, hi = m["rho_u_ci"]
    print(f"  rho_u true: {np.round(m['rho_u_true'],3)}")
    print(f"  rho_u mean: {np.round(m['rho_u_mean'],3)}")
    print(f"    dominant comp [{jd}]: {m['rho_u_mean'][jd]:+.3f}  "
          f"CI=[{lo[jd]:+.3f},{hi[jd]:+.3f}]  (true {m['rho_u_true'][jd]:+.3f})"
          f"  {'sign OK' if np.sign(m['rho_u_mean'][jd])==np.sign(m['rho_u_true'][jd]) else 'SIGN WRONG'}"
          f"  {'CI excludes 0' if (lo[jd]>0 or hi[jd]<0) else 'CI includes 0'}")
    print(f"    ||rho_u||: mean={m['rho_u_norm_mean']:.3f} true={m['rho_u_norm_true']:.3f}")
    print(f"  rho_eps mean={m['rho_eps_mean']:+.3f} (true {m['rho_eps_true']:+.3f})")
    sk = m["skewness"]
    print(f"  SKEWNESS (DGP, leverage direction): rho<0 -> {sk['skew_leverage']:+.3f}  "
          f"vs rho=0 -> {sk['skew_symmetric']:+.3f}   "
          f"{'LEFT-SKEW OK' if sk['skew_leverage'] < -0.1 else 'weak'}")
    print(f"  Metropolis acceptance: "
          + "  ".join(f"{k}={v:.2f}" for k, v in m["acceptance"].items()))
    print(f"  Rest sane: Q cov={m['Q_cov']*100:.0f}%  h^u corr={m['h_u_corr']:.3f}  "
          f"phi_u={m['phi_u']:.3f}  sig2_u={m['sig2_u']:.4f}")
    print(f"    nu_u={m['nu_u_mean']:.2f} (true {float(theta_true['nu_u']):.2f})  "
          f"nu_eps={m['nu_eps_mean']:.2f} (true {float(theta_true['nu_eps']):.2f})")
    if m["rho0_control"] is not None:
        c0 = m["rho0_control"]; jd0 = int(np.argmax(np.abs(c0["mean"])))
        incl0 = all(c0["lo"][j] <= 0 <= c0["hi"][j] for j in range(len(c0["mean"])))
        print(f"  rho=0 CONTROL: rho_u mean={np.round(c0['mean'],3)}  "
              f"{'all CIs include 0 (no false leverage)' if incl0 else 'WARNING: some CI excludes 0'}")
    print("  Convergence (R-hat / ESS; single-move MH mixes slowly):")
    for name, dd in m["diagnostics"].items():
        print(f"    {name:10s}: R-hat={dd['r_hat']:.3f}  ESS={dd['ess']:8.1f}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Branch A vs Branch B: the mixing / attenuation comparison (Passo 4 hypothesis)
# ─────────────────────────────────────────────────────────────────────────────

def compare_branches_AB(
    theta_true: dict,
    *,
    rho_u: np.ndarray | float,
    rho_eps: float = -0.4,
    sv_u: tuple[float, float, float] = (0.0, 0.97, 0.22),
    sv_eps: tuple[float, float, float] = (0.0, 0.95, 0.15),
    T: int = 300,
    seed: int = 20240,
    config_name: str = "small",
    n_iter: int = 3000,
    burn_in: int = 1200,
    thin: int = 1,
    n_chains: int = 2,
    verbose: bool = True,
) -> dict:
    r"""
    Head-to-head test of the Passo-4 hypothesis: *does Branch B (lagged, FFBS) mix
    better than Branch A (contemporaneous, single-move Metropolis), and is its
    ``rho`` less attenuated?*

    Each branch is run on its **own matched** DGP (Branch A on a contemporaneous
    panel, Branch B on a lagged panel) with the *same* leverage values ``rho_u``,
    ``rho_eps`` and the *same* sampler schedule, so the only difference is the
    timing/sampler pair.  Compares, for the dominant common component and the
    idiosyncratic ``rho``:

      * **mixing** — the effective sample size (ESS) of ``rho`` and of the log-vol
        AR(1) parameters, and the path acceptance (Branch A < 1 single-move MH;
        Branch B = 1 by construction, the FFBS draw is always taken);
      * **attenuation** — the recovered ``|rho|`` relative to the truth.

    Interpretation (the user's question on rho / ASIS):
      * if B mixes better *and* is less attenuated -> part of Branch A's
        attenuation was a *mixing* artefact of the single-move Metropolis;
      * if B is *still* attenuated despite better mixing -> the attenuation is
        *structural* (missing information about the common leverage vector),
        confirming the Passo-3 diagnosis.
    """
    common = dict(theta_true=theta_true, rho_u=rho_u, rho_eps=rho_eps, sv_u=sv_u,
                  sv_eps=sv_eps, T=T, seed=seed, config_name=config_name,
                  n_iter=n_iter, burn_in=burn_in, thin=thin, n_chains=n_chains,
                  rho0_control=False, verbose=False)
    if verbose:
        print("\n[A] Branch A (contemporaneous, Metropolis path) ...")
    resA = run_recovery_mcmc_leverage(timing="contemporaneous", **common)
    if verbose:
        print("[B] Branch B (lagged, Omori mixture + FFBS path) ...")
    resB = run_recovery_mcmc_leverage(timing="lagged", **common)

    def _summ(res):
        m = res["metrics"]
        jd = m["jdom"]
        dg = m["diagnostics"]
        rt = m["rho_u_true"]
        return {
            "ess_rho_dom": dg[f"rho_u[{jd}]"]["ess"],
            "rhat_rho_dom": dg[f"rho_u[{jd}]"]["r_hat"],
            "ess_phi": dg["sv_u.phi"]["ess"],
            "ess_sig2": dg["sv_u.sig2"]["ess"],
            "acc_path": m["acceptance"]["path_u"],
            "acc_rho_u": m["acceptance"]["rho_u"],
            "rho_dom_mean": float(m["rho_u_mean"][jd]),
            "rho_dom_true": float(rt[jd]),
            "atten_dom": float(abs(m["rho_u_mean"][jd]) / abs(rt[jd])) if rt[jd] != 0 else float("nan"),
            "atten_norm": float(m["rho_u_norm_mean"] / m["rho_u_norm_true"]) if m["rho_u_norm_true"] > 0 else float("nan"),
            "rho_eps_mean": m["rho_eps_mean"], "rho_eps_true": m["rho_eps_true"],
            "atten_eps": float(abs(m["rho_eps_mean"]) / abs(m["rho_eps_true"])) if m["rho_eps_true"] != 0 else float("nan"),
            "h_u_corr": m["h_u_corr"], "skew": m["skewness"]["skew_leverage"],
        }

    A, B = _summ(resA), _summ(resB)
    out = {"A": A, "B": B, "resA": resA, "resB": resB,
           "ess_gain_rho": B["ess_rho_dom"] / A["ess_rho_dom"] if A["ess_rho_dom"] > 0 else float("nan"),
           "ess_gain_phi": B["ess_phi"] / A["ess_phi"] if A["ess_phi"] > 0 else float("nan")}
    if verbose:
        _print_compare_AB(A, B, out)
    return out


def _print_compare_AB(A, B, out):
    print("\n" + "=" * 72)
    print("  BRANCH A  vs  BRANCH B   —   mixing & attenuation (Passo 4 hypothesis)")
    print("=" * 72)
    print(f"  {'metric':28s} {'Branch A (MH)':>16s} {'Branch B (FFBS)':>16s}")
    print("  " + "-" * 64)
    rows = [
        ("ESS rho_dom",            "ess_rho_dom",  "{:.1f}"),
        ("R-hat rho_dom",          "rhat_rho_dom", "{:.3f}"),
        ("ESS sv_u.phi",           "ess_phi",      "{:.1f}"),
        ("ESS sv_u.sig2",          "ess_sig2",     "{:.1f}"),
        ("path acceptance",        "acc_path",     "{:.3f}"),
        ("rho_u acceptance",       "acc_rho_u",    "{:.3f}"),
        ("rho_dom mean",           "rho_dom_mean", "{:+.3f}"),
        ("rho_dom true",           "rho_dom_true", "{:+.3f}"),
        ("atten. dom (|mean|/|true|)", "atten_dom","{:.2f}"),
        ("atten. ||rho||",         "atten_norm",   "{:.2f}"),
        ("rho_eps mean",           "rho_eps_mean", "{:+.3f}"),
        ("atten. rho_eps",         "atten_eps",    "{:.2f}"),
        ("h^u corr",               "h_u_corr",     "{:.3f}"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:28s} {fmt.format(A[key]):>16s} {fmt.format(B[key]):>16s}")
    print("  " + "-" * 64)
    print(f"  ESS gain B/A:  rho_dom x{out['ess_gain_rho']:.1f}   sv_u.phi x{out['ess_gain_phi']:.1f}")
    print("  Reading:")
    better_mix = out["ess_gain_rho"] > 1.2
    less_atten = B["atten_dom"] > A["atten_dom"] + 0.05
    if better_mix and less_atten:
        print("    Branch B mixes better AND is less attenuated")
        print("    -> part of Branch A's attenuation was a MIXING artefact.")
    elif better_mix and not less_atten:
        print("    Branch B mixes better but rho is STILL attenuated")
        print("    -> the attenuation is STRUCTURAL (missing info on the common")
        print("       leverage vector), confirming the Passo-3 diagnosis.")
    else:
        print("    inconclusive at this schedule (lengthen the chains / raise T).")
    print("=" * 72)


# ─── small helpers ───────────────────────────────────────────────────────────

def _pca_init(Y, ordered_cols, block_map, freq_list):
    """PCA warm-start theta on the synthetic panel (honest start for the EM)."""
    from monte_carlo_recovery import init_theta_from_synthetic
    freq_map = {c: f for c, f in zip(ordered_cols, freq_list)}
    theta0, _ = init_theta_from_synthetic(
        Y, ordered_cols=ordered_cols, block_map=block_map, freq_map=freq_map,
    )
    return theta0


def _abscorr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xz, yz = x - x.mean(), y - y.mean()
    sx, sy = np.sqrt((xz ** 2).sum()), np.sqrt((yz ** 2).sum())
    return float(abs((xz @ yz) / (sx * sy))) if sx > 0 and sy > 0 else float("nan")


def _signedcorr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xz, yz = x - x.mean(), y - y.mean()
    sx, sy = np.sqrt((xz ** 2).sum()), np.sqrt((yz ** 2).sum())
    return float((xz @ yz) / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def _relerr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def _print_recovery(m, theta_true):
    print("\n" + "=" * 72)
    print(f"  RECOVERY (Passo 1, no SV) — {int(m['ci_level']*100)}% credible intervals")
    print("=" * 72)
    cov = m["coverage"]
    print("  Coverage of truth in credible intervals:")
    for k in ("A", "Q", "Lambda_nz", "R", "nu_u", "nu_eps"):
        print(f"    {k:10s}: {cov[k]*100:5.1f}%")
    print(f"  Factor |corr| (per block): "
          f"{', '.join(f'{c:.3f}' for c in m['factor_abscorr'])}")
    print(f"  EM-weight |corr| anchor: w_u={m['w_u_corr_em']:.3f}  "
          f"w_eps={m['w_eps_corr_em']:.3f}")
    xc = m["xcheck_mcmc_vs_em"]
    print("  MCMC posterior mean vs EM point (relerr; must be small):")
    print(f"    A={xc['A_relerr']:.3f}  Q={xc['Q_relerr']:.3f}  "
          f"Lambda={xc['Lambda_relerr']:.3f}  R={xc['R_relerr']:.3f}")
    print(f"    nu_u: EM={xc['nu_u_em']:.2f} MCMC={xc['nu_u_mcmc']:.2f}   "
          f"nu_eps: EM={xc['nu_eps_em']:.2f} MCMC={xc['nu_eps_mcmc']:.2f}")
    print(f"    nu_true: nu_u={float(theta_true['nu_u']):.2f} "
          f"nu_eps={float(theta_true['nu_eps']):.2f}")
    print("  Convergence (R-hat / ESS):")
    for name, d in m["diagnostics"].items():
        print(f"    {name:8s}: R-hat={d['r_hat']:.3f}  ESS={d['ess']:8.1f}")
    print("=" * 72)
