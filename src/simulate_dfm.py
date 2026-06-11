"""
src/simulate_dfm.py

Monte Carlo simulator for the Student-t Dynamic Factor Model — the
exact mirror of the generative model that ``em_e_step`` / ``em_m_step`` /
``kalman`` *assume*.  Used to produce synthetic datasets from a known
parameter vector :math:`\\theta^\\star` (typically the EM estimate on
real macro data) so that the EM machinery can be validated by a
self-recovery test: simulate from ``theta_star`` -> re-fit -> compare
``theta_hat`` against ``theta_star``.

Scope of this file
------------------
TASK 1 — *factor process*:
    :func:`simulate_factors` simulates the VAR(1) Student-t latent
    factor process

    .. math::

        f_t \\;=\\; \\mathbf{A}\\, f_{t-1} + u_t,
        \\qquad
        u_t \\mid w^u_t \\sim \\mathcal{N}\\!\\big(\\mathbf{0},\\, \\mathbf{Q} / w^u_t\\big),
        \\qquad
        w^u_t \\stackrel{\\mathrm{iid}}{\\sim} \\mathrm{Gamma}(\\nu_u / 2,\\, \\nu_u / 2),

    using the scale-mixture-of-normals representation of the
    multivariate Student-t.  The function returns both the factors
    ``F`` and the *ground-truth* mixing weights ``w_u_true`` so that
    the EM E-step's recovery of the latent weights can later be
    evaluated against the true ones (``corr(w_hat, w_true)``).

TASK 2 — *observation equation*:
    :func:`simulate_observations` generates the panel
    :math:`y_t = \\mathbf{\\Lambda}^{\\mathrm{eff}}(t)\\, \\tilde{f}_t +
    \\varepsilon_t` with idiosyncratic Student-t noise
    :math:`\\varepsilon_t \\mid w^\\varepsilon_t \\sim
    \\mathcal{N}(\\mathbf{0}, \\mathbf{R} / w^\\varepsilon_t)`,
    :math:`w^\\varepsilon_t \\stackrel{\\mathrm{iid}}{\\sim}
    \\mathrm{Gamma}(\\nu_\\varepsilon / 2, \\nu_\\varepsilon / 2)`
    *shared across series at a given t*, and the
    Mariano-Murasawa five-month weighted aggregation
    :math:`(1/3, 2/3, 1, 2/3, 1/3)` for quarterly rows — the exact
    mirror of :func:`kalman.build_Lambda_tilde`.  The function
    returns both the (complete) ``Y_complete`` and the ground-truth
    weights ``w_eps_true``.

TASK 3 (this revision) — *missing-data pattern*:
    :func:`apply_missing_pattern` overlays the mixed-frequency
    quarterly mask (GDP visible only at quarter-end months) and the
    ragged end-of-sample mask (only the timely series — NFCI in our
    real panel — observed in the last few months) on top of
    ``Y_complete``.  This produces the ``Y`` panel actually seen by
    the EM-side Kalman filter, with the same ``NaN`` pattern the
    real data carry.

TASK 4 — *contamination mechanism* (Experiment C):
    :func:`apply_contamination` overlays the additive-outlier
    contamination of the thesis (section "The contamination
    mechanism", ~line 12419).  For each period an indicator
    :math:`z_t \\sim \\mathrm{Bernoulli}(\\pi)` is drawn; when
    :math:`z_t = 1` the *whole* idiosyncratic shock vector of that
    period (all :math:`M` series) is REPLACED — not augmented — by a
    draw from an inflated Student-t
    :math:`t_{\\nu_{\\mathrm{contam}}}(\\mathbf{0},
    \\kappa^2 \\mathbf{R})` (defaults :math:`\\nu_{\\mathrm{contam}} = 3`,
    :math:`\\kappa = 5`, i.e. ~25x the baseline scale).  The
    contamination is realised as a scale-mixture coherent with the
    rest of the simulator (a single shared :math:`w^{\\mathrm{contam}}_t
    \\sim \\mathrm{Gamma}(\\nu_{\\mathrm{contam}}/2,
    \\nu_{\\mathrm{contam}}/2)` per contaminated period), and it is
    driven by a *dedicated* RNG stream so that, at :math:`\\pi = 0`,
    the simulator output is bit-identical to the contamination-free
    version.  Only the *idiosyncratic* component is contaminated; the
    factor process (:func:`simulate_factors`) is left untouched — see
    the theory comment in :func:`apply_contamination` for why.

WRAPPER — *full simulator*:
    :func:`simulate_dfm` chains :func:`simulate_factors`,
    :func:`simulate_observations` and :func:`apply_missing_pattern`
    into a single call that returns ``Y`` together with the *ground
    truth* (``F``, ``w_u_true``, ``w_eps_true``, ``theta_used``)
    needed by the Monte Carlo self-recovery test.

Thesis reference
----------------
``EM_for_student_t.tex``:
  - "Scale mixture representation" (search for "scale-mixture" /
    "scale mixture") — the Gaussian-Gamma identity underlying the
    Student-t.  We use the standard parametrisation in which the
    *prior* mixing weight has mean one,
    :math:`\\mathbb{E}[w^u_t] = 1` and
    :math:`\\mathrm{Var}(w^u_t) = 2 / \\nu_u`, so that the marginal
    factor innovation is
    :math:`u_t \\sim t_{\\nu_u}(\\mathbf{0}, \\mathbf{Q})` and
    :math:`\\mathrm{Cov}(u_t) = \\mathbf{Q}\\, \\nu_u / (\\nu_u - 2)`
    for :math:`\\nu_u > 2`.
  - The same parametrisation is used in :func:`em_e_step.compute_weights`
    (the conjugate posterior is :math:`\\mathrm{Gamma}((\\nu_u + r)/2,
    (\\nu_u + d^u_t)/2)`, which collapses to the prior when the
    Mahalanobis residual :math:`d^u_t = 0`), and in
    :func:`kalman.build_Q_tilde` (the augmented innovation covariance
    is :math:`\\mathbf{Q}/w^u_t` in the top-left :math:`r \\times r`
    block).  The simulator must therefore use exactly the same
    prior to remain the "mirror" of the EM-side model.
"""

from __future__ import annotations

import numpy as np

# Threshold above which nu is treated as the Gaussian limit (weights ≡ 1).
# Any nu > _NU_GAUSSIAN_THRESHOLD is numerically indistinguishable from
# infinity: the Gamma(nu/2, 2/nu) mixing weight has Var = 2/nu < 2e-6 and
# concentrates to a point mass at 1.  np.inf is always the preferred input
# for the exact Gaussian limit (theta_star_B of Experiment B), but a large
# finite nu (e.g. from the Brent bracket upper bound 1000) falls well below
# this threshold and is still treated as Student-t.
_NU_GAUSSIAN_THRESHOLD: float = 1e6


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   FACTOR-PROCESS SIMULATOR — VAR(1) Student-t scale-mixture              ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_factors(
    A: np.ndarray,
    Q: np.ndarray,
    nu_u: float,
    T: int,
    r: int,
    seed: int,
    burn_in: int = 1000,
) -> dict:
    r"""
    Simulate ``T`` periods of the VAR(1) Student-t latent factor
    process

    .. math::

        f_t \;=\; \mathbf{A}\, f_{t-1} + u_t,
        \qquad
        u_t \mid w^u_t \sim \mathcal{N}\!\big(\mathbf{0},\, \mathbf{Q} / w^u_t\big),
        \qquad
        w^u_t \stackrel{\mathrm{iid}}{\sim} \mathrm{Gamma}(\nu_u / 2,\, \nu_u / 2),

    using the scale-mixture-of-normals representation of the
    multivariate Student-t.  Returns the simulated factor matrix and
    the *ground-truth* mixing weights for later use in the EM
    self-recovery test.

    Parameters
    ----------
    A : np.ndarray, shape (r, r)
        VAR(1) transition matrix.  Must have spectral radius
        :math:`\rho(\mathbf{A}) < 1` for the simulated process to be
        stationary.  Typically loaded from ``theta_star`` (the EM
        estimate on the real macro panel).
    Q : np.ndarray, shape (r, r)
        Innovation covariance matrix — *conditional* on
        :math:`w^u_t = 1`, i.e. the "Gaussian-equivalent" scale.  Must
        be symmetric positive definite.  The marginal (unconditional)
        innovation covariance is
        :math:`\mathrm{Cov}(u_t) = \mathbf{Q}\, \nu_u / (\nu_u - 2)`
        for :math:`\nu_u > 2`.
    nu_u : float
        Degrees of freedom of the factor-innovation Student-t.  Must
        be ``> 2`` for a finite marginal innovation covariance; ``> 4``
        for finite marginal kurtosis.  Lower values produce heavier
        tails.
    T : int
        Number of *post-burn-in* periods to return.  The simulator
        actually generates ``burn_in + T`` periods and discards the
        first ``burn_in`` of them.
    r : int
        Latent factor dimension (= number of blocks = 3 in this
        project).  Checked against the shapes of ``A`` and ``Q``.
    seed : int
        Seed for ``numpy.random.default_rng``.  Two calls with the
        same ``(A, Q, nu_u, T, r, seed, burn_in)`` produce
        bit-identical output.
    burn_in : int, default 1000
        Number of initial periods to discard so that the simulated
        path is (numerically) drawn from the stationary distribution
        of the VAR.  See Notes for why this default is *long*.

    Returns
    -------
    dict
        - ``F`` : np.ndarray, shape ``(T, r)`` — simulated factors
          *after* the burn-in.  Row ``t`` is :math:`f_t`.
        - ``w_u_true`` : np.ndarray, shape ``(T,)`` — the mixing
          weights actually drawn for the *post-burn-in* periods,
          aligned one-to-one with the rows of ``F``.  These are the
          ground-truth weights against which the E-step's posterior
          weights ``w_u_hat`` can be benchmarked
          (``corr(w_hat, w_true)``).

    Notes
    -----
    **Scale-mixture representation of the Student-t.**

    A multivariate Student-t random vector :math:`u \sim t_\nu(\mathbf{0},
    \mathbf{Q})` admits the equivalent *Gaussian scale mixture*
    representation

    .. math::

        u \mid w \sim \mathcal{N}(\mathbf{0}, \mathbf{Q} / w),
        \qquad
        w \sim \mathrm{Gamma}(\nu / 2,\, \nu / 2).

    Marginalising over the Gamma-distributed weight reproduces the
    Student-t density.  This is the representation the EM exploits:
    the E-step computes the *posterior* mean of :math:`w` given the
    data (a Gaussian-Gamma conjugate update) and uses it as a
    *weight* in the M-step's weighted least squares, which turns
    Student-t maximum likelihood into IRLS (iteratively reweighted
    least squares).  The simulator must therefore use the *same*
    prior on :math:`w` to remain a faithful mirror of the EM model.

    In the parametrisation :math:`w \sim \mathrm{Gamma}(\nu/2, \nu/2)`
    (shape :math:`\nu/2`, rate :math:`\nu/2`):

      - :math:`\mathbb{E}[w] = 1` (the prior is calibrated so that
        the conditional covariance :math:`\mathbf{Q}/w` averages
        back to roughly :math:`\mathbf{Q}` — the EM is in the
        Gaussian limit when :math:`\nu_u \to \infty` and :math:`w \to 1`),
      - :math:`\mathrm{Var}(w) = 2 / \nu` — heavier tails (smaller
        :math:`\nu`) come with more variable weights.

    **Heavy tails come from low w_u_t.**

    Periods in which the Gamma sample :math:`w^u_t` happens to be
    *small* (the left tail of the Gamma — rare but heavy) get a
    *large* conditional innovation covariance :math:`\mathbf{Q} / w^u_t`
    and so produce *large* :math:`u_t` — i.e. outliers in the factor
    path.  These are the periods the EM should later identify as
    "down-weight" via small posterior :math:`\hat{w}^u_t`.  The
    returned ``w_u_true`` lets us *verify* that recovery later by
    comparing it to the E-step's posterior mean
    :math:`\hat{w}^u_t = (\nu_u + r) / (\nu_u + d^u_t)`.

    **NumPy Gamma parametrisation gotcha.**

    NumPy's ``np.random.Generator.gamma(shape, scale)`` uses the
    *shape-scale* parametrisation, with mean ``shape * scale``.  Our
    target is mean ``1`` and shape :math:`\nu_u / 2`, so we pass
    ``shape = nu_u / 2`` and ``scale = 2 / nu_u``: then
    ``shape * scale = (nu_u / 2) * (2 / nu_u) = 1``.  Variance is
    ``shape * scale**2 = (nu_u / 2) * (2 / nu_u)**2 = 2 / nu_u``,
    matching the closed form above.

    **Why a long burn-in of 1000 is needed.**

    The VAR(1) is a contraction map with rate equal to the spectral
    radius :math:`\rho(\mathbf{A})`.  Starting from
    :math:`f_{-\mathrm{burn\_in}} = \mathbf{0}` (the stationary mean,
    since :math:`\mathbb{E}[f_t] = \mathbf{0}`), the deterministic
    component of the path decays as
    :math:`\mathbf{A}^{\mathrm{burn\_in}}`, which is of order
    :math:`\rho(\mathbf{A})^{\mathrm{burn\_in}}` in operator norm.
    On the real macro panel the EM converged to
    :math:`\rho(\mathbf{A}) \approx 0.978` (very persistent dynamics —
    the factors track slow-moving business-cycle quantities), so a
    burn-in of :math:`n` collapses the transient by a factor of
    :math:`0.978^n`.  At :math:`n = 1000`, this is
    :math:`0.978^{1000} \approx 7 \times 10^{-11}` — essentially
    machine precision.  A shorter burn-in (say :math:`n = 100`,
    contracting by :math:`0.978^{100} \approx 0.107`) would leave a
    visible transient at the start of ``F`` that pollutes downstream
    statistics (especially the OLS recovery check and the kurtosis
    diagnostic in ``__main__``).  We default to ``1000`` so the
    behaviour is robust to persistent ``A``.

    Equivalent alternatives — drawing the initial state from the
    stationary distribution :math:`\mathcal{N}(\mathbf{0},
    \mathbf{\Sigma}_\infty)` where
    :math:`\mathrm{vec}(\mathbf{\Sigma}_\infty) =
    (\mathbf{I} - \mathbf{A} \otimes \mathbf{A})^{-1}\,
    \mathrm{vec}(\mathbf{Q})` — would also work but introduces a
    second Gaussian-only object that does not exist in the
    Student-t generative model.  The burn-in approach keeps the
    simulator *purely* in the Student-t world end-to-end.

    **Why we return ``w_u_true``.**

    The Monte Carlo validation has two distinct levels of recovery:

      1. *Parameter* recovery: does ``run_em`` re-estimate ``theta``
         close to ``theta_star``?  This needs only ``F`` (well, plus
         the eventual ``Y`` from Task 2).
      2. *Latent-state* recovery: does the E-step's posterior mean
         :math:`\hat{w}^u_t` track the *true* :math:`w^u_t` that
         drove the heavy-tailed innovation at each ``t``?  This
         needs the ground-truth weights, which only the simulator
         knows.

    Returning ``w_u_true`` enables level (2) — a stronger and more
    informative test than level (1) alone.

    Raises
    ------
    ValueError
        If ``A`` is not ``(r, r)``, if ``Q`` is not ``(r, r)``,
        if ``nu_u <= 0`` or ``T <= 0`` or ``burn_in < 0``, or if
        ``Q`` is not (numerically) symmetric positive definite (the
        Cholesky factorisation fails).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0.9, 0.0, 0.0],
    ...               [0.0, 0.95, 0.0],
    ...               [0.0, 0.0, 0.85]])
    >>> Q = np.eye(3)
    >>> out = simulate_factors(A, Q, nu_u=4.0, T=500, r=3, seed=0)
    >>> out["F"].shape
    (500, 3)
    >>> out["w_u_true"].shape
    (500,)
    """
    # ── 1. Argument validation ───────────────────────────────────────────────
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if A.shape != (r, r):
        raise ValueError(f"A shape {A.shape} does not match (r, r) = ({r}, {r}).")
    if Q.shape != (r, r):
        raise ValueError(f"Q shape {Q.shape} does not match (r, r) = ({r}, {r}).")
    # nu_u = np.inf is the Gaussian limit (weights ≡ 1); NaN or non-positive is invalid.
    if np.isnan(nu_u) or nu_u <= 0:
        raise ValueError(
            f"nu_u must be a positive number (np.inf for the Gaussian limit); "
            f"got {nu_u}."
        )
    if T <= 0:
        raise ValueError(f"T must be a positive integer; got {T}.")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative; got {burn_in}.")

    # ── 1b. Gaussian-limit flag ──────────────────────────────────────────────
    # np.inf or nu_u > _NU_GAUSSIAN_THRESHOLD -> pure Gaussian innovations
    # (weights w_u_t ≡ 1 exactly; no Gamma draws).
    _gaussian_factors = np.isinf(nu_u) or nu_u > _NU_GAUSSIAN_THRESHOLD

    # Symmetrise Q defensively before Cholesky to absorb any tiny asymmetry
    # introduced upstream by EM arithmetic (theta_star comes from a sequence
    # of M-step updates, so its Q is symmetric up to ~1e-15 only).
    Q = 0.5 * (Q + Q.T)
    try:
        L = np.linalg.cholesky(Q)                  # L L' = Q, L lower-triangular
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"Q is not numerically positive definite — Cholesky failed: {err}.  "
            "Inspect the spectrum of Q upstream."
        ) from err

    # ── 2. RNG and shapes ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_total = burn_in + T

    # ── 3. Sample the mixing weights w_u_t for every period ──────────────────
    # Gamma(shape = nu_u/2, scale = 2/nu_u)  =>  mean = 1, var = 2/nu_u.
    # NumPy: rng.gamma(shape, scale) returns Gamma(shape, scale) with mean shape*scale.
    # Gaussian limit (nu_u = inf): weights are identically 1; no Gamma draws.
    if _gaussian_factors:
        w_full = np.ones(n_total)                  # (n_total,), exactly 1.0
    else:
        w_full = rng.gamma(
            shape=nu_u / 2.0,
            scale=2.0 / nu_u,
            size=n_total,
        )                                          # (n_total,), strictly > 0

    # ── 4. Sample the Gaussian "seeds" z_t ~ N(0, I_r) ───────────────────────
    Z = rng.standard_normal(size=(n_total, r))     # (n_total, r)

    # ── 5. Build the Student-t innovations u_t = (1/sqrt(w_u_t)) * L @ z_t ───
    # Conditional on w, Cov(L z / sqrt(w)) = L Cov(z) L' / w = Q / w.
    # Vectorised over t:  (n_total, r) = (n_total, r) @ (r, r) / (n_total, 1).
    U = (Z @ L.T) / np.sqrt(w_full)[:, np.newaxis]  # (n_total, r)

    # ── 6. Propagate the VAR(1):  f_t = A f_{t-1} + u_t,  f_{-1} = 0 ─────────
    # We initialise the path at the stationary mean E[f_t] = 0 and rely on the
    # long burn-in to wash out the deterministic start.  Equivalently, the
    # first row of F_full is just u_0.
    F_full = np.empty((n_total, r), dtype=float)
    F_full[0] = U[0]
    for t in range(1, n_total):
        F_full[t] = A @ F_full[t - 1] + U[t]

    # ── 7. Discard the burn-in and return aligned arrays ─────────────────────
    F        = F_full[burn_in:]                    # (T, r)
    w_u_true = w_full[burn_in:]                    # (T,)

    return {"F": F, "w_u_true": w_u_true}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   OBSERVATION-EQUATION SIMULATOR — block Lambda + MM + Student-t eps     ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Default Mariano-Murasawa aggregation weights — IDENTICAL to those used in
# kalman.build_Lambda_tilde (kalman.MM_WEIGHTS_DEFAULT).  Hard-coded here to
# avoid a circular import; the equality is checked in the self-test.
_MM_WEIGHTS_DEFAULT: list[float] = [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]


def simulate_observations(
    F: np.ndarray,
    Lambda: np.ndarray,
    R: np.ndarray,
    nu_eps: float,
    freq_list: list[str],
    block_map: dict[str, str],
    ordered_cols: list[str],
    r: int,
    seed: int,
    mm_weights: list[float] | None = None,
    pi: float = 0.0,
    nu_contam: float = 3.0,
    kappa: float = 5.0,
    contam_seed: int | None = None,
) -> dict:
    r"""
    Simulate the *complete* (no missing) observation panel
    :math:`\mathbf{Y} \in \mathbb{R}^{T \times M}` from a given set of
    latent factors and parameters, using the EXACT generative model the
    EM machinery assumes: a block-diagonal monthly loading,
    Mariano-Murasawa (MM) five-month aggregation for quarterly rows,
    and idiosyncratic Student-t noise via the time-shared Gamma
    scale-mixture.

    Returns the complete panel ``Y_complete`` (with the early-sample
    quarterly rows set to ``NaN`` because the MM kernel needs five
    lags) and the *ground-truth* mixing weights ``w_eps_true``.  No
    additional missing-data pattern is applied here — that is Task 3.

    Parameters
    ----------
    F : np.ndarray, shape (T, r)
        Simulated monthly factor matrix from :func:`simulate_factors`
        (or any other matrix with the same shape).  Row ``t`` is
        :math:`f_t \in \mathbb{R}^r`.
    Lambda : np.ndarray, shape (M, r)
        Block-diagonal monthly loading matrix from
        :math:`\theta^\star`.  Each row has non-zero entries only in
        the factor-column corresponding to the series' own economic
        block (real / financial / other).  ``Lambda`` is used unchanged
        for monthly rows; for quarterly rows it is the *latent monthly*
        loading whose row is spread across five lag-blocks with MM
        weights.
    R : np.ndarray, shape (M,)
        Diagonal idiosyncratic variances (one per series).  Strictly
        positive.  The *conditional* idiosyncratic variance at time
        :math:`t` is :math:`R_i / w^\varepsilon_t`.
    nu_eps : float
        Idiosyncratic degrees of freedom.  ``> 2`` for finite marginal
        idiosyncratic variance; ``> 4`` for finite marginal kurtosis.
    freq_list : list[str], length M
        ``"monthly"`` / ``"quarterly"`` per series, in the same row
        order as ``Lambda`` and ``R``.
    block_map : dict[str, str]
        Maps series name -> block name (``"real"`` / ``"financial"`` /
        ``"other"``).  Accepted for API symmetry with the EM-side
        functions; not algebraically required here because
        block-diagonality of ``Lambda`` already encodes which factor
        each series loads on.  We use it only for diagnostic / sanity
        purposes in the self-test.
    ordered_cols : list[str], length M
        Series names in the order of the rows of ``Lambda``.  Accepted
        for the same reason as ``block_map``.
    r : int
        Latent factor dimension (= number of blocks = 3 in this
        project).  Checked against the shapes of ``F`` and ``Lambda``.
    seed : int
        Seed for ``numpy.random.default_rng``.  Independent of the
        seed used by :func:`simulate_factors`: the factor and
        observation processes are conditionally independent given
        their state, so they may be driven by independent RNGs.
    mm_weights : list[float], optional
        Five MM aggregation weights for quarterly rows.  Default
        ``[1/3, 2/3, 1, 2/3, 1/3]``, the standard Mariano-Murasawa
        weights for the log-difference of a chain-weighted quarterly
        index.  This must be the *same* list as
        :data:`kalman.MM_WEIGHTS_DEFAULT` — otherwise simulator and
        estimator would not be mirror images of each other.  The
        equality is verified in the self-test.
    pi : float, default 0.0
        Contamination intensity (Experiment C).  Probability that any
        given period ``t`` is an additive outlier.  ``pi = 0`` (the
        default) disables contamination entirely and returns output
        bit-identical to the contamination-free simulator.  Must lie
        in ``[0, 1]``.  See :func:`apply_contamination`.
    nu_contam : float, default 3.0
        Degrees of freedom of the inflated Student-t used at
        contaminated periods.  Heavier than the baseline ``nu_eps`` by
        design (default 3 → very heavy tails).  ``np.inf`` collapses
        the contaminated draw to a Gaussian of scale ``kappa^2 R``.
    kappa : float, default 5.0
        Scale-inflation factor of the contaminated idiosyncratic
        covariance: contaminated shocks have covariance
        ``kappa^2 * R`` (≈ 25x the baseline at the default
        ``kappa = 5``).  Must be strictly positive.
    contam_seed : int or None, default None
        Seed for the *dedicated* contamination RNG.  Drawn from a
        stream **separate** from the ``seed`` that drives ``Z`` and
        ``w_eps_true``, so that toggling contamination on/off never
        perturbs the baseline idiosyncratic draws (this is what
        guarantees the bit-identity at ``pi = 0``).  ``None`` defaults
        to ``seed + 1`` — disjoint from the baseline stream seeded at
        ``seed``.  :func:`simulate_dfm` passes an explicit value.

    Returns
    -------
    dict
        - ``Y_complete`` : np.ndarray, shape ``(T, M)``.  The
          simulated observations.  For *monthly* rows this is
          complete in every time period; for *quarterly* rows the
          first four time periods (``t = 0, 1, 2, 3``) are ``NaN``
          because the MM kernel requires five consecutive lags
          :math:`\{f_t, f_{t-1}, \ldots, f_{t-4}\}` and not enough
          history is yet available (see Notes).  No further missing
          mask is applied — Task 3 will overlay the mixed-frequency
          and ragged-edge patterns of the real panel on top of this.
        - ``w_eps_true`` : np.ndarray, shape ``(T,)``, all strictly
          positive.  The ground-truth idiosyncratic mixing weights
          actually drawn at each ``t``.  Used in the recovery test to
          benchmark the E-step's posterior mean :math:`\hat{w}^\varepsilon_t =
          (\nu_\varepsilon + m_t)/(\nu_\varepsilon + d^\varepsilon_t)`.
        - ``contam_mask`` : np.ndarray of bool, shape ``(T,)``.  The
          ground-truth contamination indicator: ``True`` at the
          periods drawn as additive outliers
          (:math:`z_t = 1`).  All ``False`` when ``pi = 0``.  This is
          the ground truth against which Experiment C's detection-rate
          metric will be scored (a later step).

    Notes
    -----
    **The observation model — what we are mirroring.**

    The EM-side model (``EM_for_student_t.tex``, eq. at line ~1877 /
    line ~3071 / line ~3137 and ``em_e_step.compute_weights``) writes

    .. math::

        y_t \;=\; \mathbf{\Lambda} f_t + \varepsilon_t,
        \qquad
        \varepsilon_t \mid w^\varepsilon_t \;\sim\;
        \mathcal{N}\!\big(\mathbf{0},\; \mathbf{R} / w^\varepsilon_t\big),
        \qquad
        w^\varepsilon_t \stackrel{\mathrm{iid}}{\sim}
        \mathrm{Gamma}(\nu_\varepsilon/2,\, \nu_\varepsilon/2).

    The scalar :math:`w^\varepsilon_t` is *shared across the M series*
    at a given ``t``: a single Gamma draw rescales the whole
    idiosyncratic covariance :math:`\mathbf{R}` by :math:`1 / w^\varepsilon_t`.
    Marginally, :math:`\varepsilon_t \sim t_{\nu_\varepsilon}(\mathbf{0}, \mathbf{R})`
    — a *multivariate* Student-t with full :math:`M`-dimensional
    covariance :math:`\mathbf{R}` and the same degrees of freedom for
    every series.  This is the structure the EM exploits: the
    posterior of :math:`w^\varepsilon_t` is a single Gamma
    :math:`\mathrm{Gamma}((\nu_\varepsilon + m_t)/2, (\nu_\varepsilon + d^\varepsilon_t)/2)`
    — one weight per ``t``, not one per ``(series, t)`` — and we
    must therefore *draw exactly one* :math:`w^\varepsilon_t` per
    ``t`` in the simulator, otherwise the EM's recovered
    :math:`\hat{w}^\varepsilon_t` would not be the right comparison
    target.

    Within a given ``t``, however, the actual noise scalars
    :math:`\varepsilon_{i, t}` are *conditionally* independent across
    series because ``R`` is diagonal in this project (see the
    idiosyncratic-variances structure used everywhere downstream):
    given :math:`w^\varepsilon_t`,

    .. math::

        \varepsilon_{i, t} \;=\; \sqrt{R_i / w^\varepsilon_t} \cdot z_{i, t},
        \qquad
        z_{i, t} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1).

    So the recipe is: at each ``t``, draw one Gamma weight and one
    vector of ``M`` independent standard normals, then scale
    component-by-component by :math:`\sqrt{R_i / w^\varepsilon_t}`.

    **Block diagonality of Lambda.**

    By assumption (and by all EM-side code in this project) each row
    of :math:`\mathbf{\Lambda}` has non-zero entries in exactly one
    column — the column of the series' own economic block.  The
    simulator uses the *full* matrix product ``Lambda @ F[t]`` rather
    than branching on the block, because the zero entries in the
    off-block columns of ``Lambda`` make the result automatically
    equal to ``Lambda[i, j_block(i)] * F[t, j_block(i)]``.  This keeps
    the implementation symmetric with the EM side (which uses the
    full augmented matrix product) and trivially generalises to a
    future setting in which an off-block loading is allowed.

    **MM aggregation for quarterly series — the mirror of build_Lambda_tilde.**

    For a quarterly series :math:`i` indexed by the (latent) monthly
    loading row :math:`\mathbf{\Lambda}_{i.}`, the underlying
    quarterly log-difference at month :math:`t` is the
    Mariano-Murasawa weighted sum of five consecutive monthly
    factor contributions:

    .. math::

        \mu_{i, t} \;=\; \sum_{k=0}^{4} c_k \,(\mathbf{\Lambda}_{i.} \cdot f_{t-k})
                  \;=\; \mathbf{\Lambda}_{i.} \cdot \Phi_t,
        \qquad
        \Phi_t \;\equiv\; \sum_{k=0}^{4} c_k\, f_{t-k},
        \qquad
        c = (1/3,\, 2/3,\, 1,\, 2/3,\, 1/3).

    The same identity is used to build the *quarterly rows of*
    ``Lambda_tilde`` in :func:`kalman.build_Lambda_tilde` (rows are
    :math:`(c_0 \mathbf{\Lambda}_{i.}, c_1 \mathbf{\Lambda}_{i.},
    \ldots, c_4 \mathbf{\Lambda}_{i.})` over the five lag-blocks of
    the augmented state).  Both objects compute the *same* number,
    so the simulator and the EM observation equation are bit-exact
    mirrors:

    .. math::

        \mu_{i, t} \;=\; \tilde{\mathbf{\Lambda}}_{i.} \cdot \tilde{f}_t
                  \;=\; \mathbf{\Lambda}_{i.} \cdot \Phi_t.

    The self-test checks this equivalence numerically as a
    correctness guard.

    **The five-lag boundary: t in {0, 1, 2, 3}.**

    The MM kernel needs five consecutive monthly lags
    :math:`\{f_t, f_{t-1}, \ldots, f_{t-4}\}` to compute
    :math:`\Phi_t`, so it is defined only for ``t >= 4``.  At
    ``t in {0, 1, 2, 3}`` we set the simulated quarterly rows to
    ``NaN`` for the following reasons:

      1. The EM-side state-space model handles the same boundary by
         carrying ``Sigma_0`` (a prior over the augmented state at
         ``t = 0``) and letting the Kalman recursion propagate.
         There is no analogue of ``Sigma_0`` in the simulator
         because we are *generating* the data, not filtering — the
         monthly factor itself is well-defined at ``t < 4`` (it's
         just :math:`f_t` from :func:`simulate_factors`), but the
         quarterly aggregate is not.  Pretending otherwise (e.g.
         using a partial MM sum with the available lags only) would
         no longer match the assumed observation equation, and the
         EM would systematically misfit those rows.
      2. In any case, the *real* mixed-frequency panel has the GDP
         series missing for every non-quarter-end month, including
         the very first months of the sample.  The Task-3 missing
         pattern will mask out these entries again.  Setting them to
         ``NaN`` here is therefore semantically consistent with how
         the EM will see them: as unobserved.
      3. Monthly series are unaffected by this boundary: they need
         only the contemporaneous factor :math:`f_t` and are
         observable from ``t = 0``.

    **Why returning w_eps_true.**

    Same reason as ``w_u_true`` in :func:`simulate_factors`: the
    Monte Carlo validation has two levels of recovery,
    *parameter* recovery (does ``run_em`` re-estimate
    ``theta_star``?) and *latent-state* recovery (does the E-step's
    posterior mean :math:`\hat{w}^\varepsilon_t` track the true
    :math:`w^\varepsilon_t`?).  Level (2) needs the ground-truth
    weights, which only the simulator knows.

    Raises
    ------
    ValueError
        If shapes are inconsistent (``F`` not ``(T, r)``, ``Lambda``
        not ``(M, r)``, ``R`` not ``(M,)``, ``len(freq_list) != M``,
        ``len(ordered_cols) != M``, ``len(mm_weights) != 5``), if
        ``nu_eps <= 0``, if any ``R[i] <= 0``, or if a frequency
        label is not ``"monthly"`` / ``"quarterly"``.

    Examples
    --------
    >>> out = simulate_observations(
    ...     F=F, Lambda=Lambda, R=R, nu_eps=4.5,
    ...     freq_list=freq_list, block_map=BLOCK,
    ...     ordered_cols=ORDERED_COLS, r=3, seed=123,
    ... )
    >>> out["Y_complete"].shape, out["w_eps_true"].shape
    ((2000, 20), (2000,))
    """
    # ── 1. Argument validation ───────────────────────────────────────────────
    F      = np.asarray(F, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    R      = np.asarray(R, dtype=float)

    if F.ndim != 2 or F.shape[1] != r:
        raise ValueError(
            f"F shape {F.shape} inconsistent with r = {r}; expected (T, r)."
        )
    T = F.shape[0]
    M = Lambda.shape[0]
    if Lambda.shape != (M, r):
        raise ValueError(f"Lambda shape {Lambda.shape} expected (M, r) = (?, {r}).")
    if R.shape != (M,):
        raise ValueError(f"R shape {R.shape} does not match Lambda's M = {M}.")
    if np.any(R <= 0):
        raise ValueError("R must be strictly positive component-wise.")
    if len(freq_list) != M:
        raise ValueError(
            f"len(freq_list) = {len(freq_list)} does not match M = {M}."
        )
    if len(ordered_cols) != M:
        raise ValueError(
            f"len(ordered_cols) = {len(ordered_cols)} does not match M = {M}."
        )
    # nu_eps = np.inf is the Gaussian limit; NaN or non-positive is invalid.
    if np.isnan(nu_eps) or nu_eps <= 0:
        raise ValueError(
            f"nu_eps must be a positive number (np.inf for the Gaussian limit); "
            f"got {nu_eps}."
        )
    _gaussian_obs = np.isinf(nu_eps) or nu_eps > _NU_GAUSSIAN_THRESHOLD

    # Contamination arguments validated up-front (fail fast).  The heavy
    # lifting / the remaining checks live in apply_contamination.
    if np.isnan(pi) or not (0.0 <= pi <= 1.0):
        raise ValueError(f"pi must lie in [0, 1]; got {pi}.")

    if mm_weights is None:
        mm_weights = _MM_WEIGHTS_DEFAULT
    if len(mm_weights) != 5:
        raise ValueError(
            f"mm_weights must have length 5; got {len(mm_weights)}."
        )
    mm = np.asarray(mm_weights, dtype=float)

    valid_freq = {"monthly", "quarterly"}
    bad = [(i, freq_list[i]) for i in range(M) if freq_list[i] not in valid_freq]
    if bad:
        raise ValueError(
            f"freq_list contains invalid labels: {bad}. "
            f"Expected one of {valid_freq}."
        )

    # ── 2. RNG ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # ── 3. Draw the shared idiosyncratic weights w_eps_t  (T,) ───────────────
    # Gamma(shape = nu_eps/2, scale = 2/nu_eps)  =>  mean = 1, var = 2/nu_eps.
    # ONE scalar per t, SHARED across the M series at that t.
    # Gaussian limit (nu_eps = inf): weights are identically 1; no Gamma draws.
    if _gaussian_obs:
        w_eps_true = np.ones(T)                    # (T,), exactly 1.0
    else:
        w_eps_true = rng.gamma(
            shape=nu_eps / 2.0,
            scale=2.0 / nu_eps,
            size=T,
        )                                          # (T,), strictly > 0

    # ── 4. Build the noiseless signal mu_t,i = Lambda^eff(t) @ tilde_f_t ─────
    # Monthly: contemporaneous-factor signal,  mu = F @ Lambda_M.T, where
    # Lambda_M is the sub-matrix of Lambda restricted to monthly rows.
    # Quarterly: MM-aggregated factor times Lambda_Q rows of Lambda.
    freq_arr  = np.asarray(freq_list)
    is_quart  = (freq_arr == "quarterly")
    is_month  = (freq_arr == "monthly")

    # Pre-compute the MM-aggregated factor Phi_t  (shape (T, r)):
    #    Phi_t = c_0 f_t + c_1 f_{t-1} + ... + c_4 f_{t-4}    for t >= 4.
    # We initialise to NaN so that any read at t < 4 surfaces as NaN and
    # propagates cleanly to the quarterly rows of Y_complete.
    Phi = np.full((T, r), np.nan, dtype=float)
    for t in range(4, T):
        Phi[t] = (
            mm[0] * F[t]
            + mm[1] * F[t - 1]
            + mm[2] * F[t - 2]
            + mm[3] * F[t - 3]
            + mm[4] * F[t - 4]
        )

    # Allocate the full signal matrix and fill block by frequency.
    mu = np.empty((T, M), dtype=float)
    # Monthly rows: standard contemporaneous product.  (T, r) @ (r, m_count).
    if is_month.any():
        mu[:, is_month] = F @ Lambda[is_month, :].T
    # Quarterly rows: same product but against Phi rather than F.  Rows 0..3
    # of Phi are NaN by construction, so quarterly rows of mu are NaN there.
    if is_quart.any():
        mu[:, is_quart] = Phi @ Lambda[is_quart, :].T

    # ── 5. Add the Student-t idiosyncratic noise eps_it ──────────────────────
    # eps_{i,t} = sqrt(R_i / w_eps_t) * z_{i,t}, with z iid N(0,1).
    # Scale factor sqrt(R_i) is series-specific; sqrt(1/w_eps_t) is t-specific
    # and shared across series.
    Z = rng.standard_normal(size=(T, M))           # (T, M) iid N(0,1)
    eps = Z * np.sqrt(R)[np.newaxis, :] / np.sqrt(w_eps_true)[:, np.newaxis]

    # ── 6. Contamination overlay (Experiment C) ──────────────────────────────
    # Substitute the idiosyncratic shock at outlier periods by an inflated
    # Student-t draw.  Driven by a DEDICATED RNG (contam_seed) so the baseline
    # draws above (w_eps_true, Z) are untouched: at pi=0 this is a strict no-op
    # and the output is bit-identical to the contamination-free simulator.
    _contam_seed = contam_seed if contam_seed is not None else seed + 1
    contam_rng = np.random.default_rng(_contam_seed)
    eps, contam_mask = apply_contamination(
        eps, R, pi=pi, rng=contam_rng,
        nu_contam=nu_contam, kappa=kappa,
    )

    Y_complete = mu + eps
    # Quarterly rows at t < 4 inherit NaN from mu (eps is finite, but NaN+x=NaN
    # propagates).  This is intentional — see "five-lag boundary" in Notes.

    return {
        "Y_complete": Y_complete,
        "w_eps_true": w_eps_true,
        "contam_mask": contam_mask,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   CONTAMINATION OVERLAY — substitutive inflated-Student-t outliers       ║
# ║   (Experiment C)                                                         ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def apply_contamination(
    eps: np.ndarray,
    R: np.ndarray,
    pi: float,
    rng: np.random.Generator,
    nu_contam: float = 3.0,
    kappa: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Overlay the additive-outlier contamination of Experiment C on a
    *clean* idiosyncratic-shock matrix ``eps``.

    For every period :math:`t` an indicator
    :math:`z_t \sim \mathrm{Bernoulli}(\pi)` is drawn.  Wherever
    :math:`z_t = 1`, the **entire** idiosyncratic shock vector of that
    period (all :math:`M` series) is *replaced* — not augmented — by a
    draw from an inflated multivariate Student-t,

    .. math::

        \varepsilon_t^{\mathrm{contam}} \;\sim\;
        t_{\nu_{\mathrm{contam}}}\!\big(\mathbf{0},\, \kappa^2 \mathbf{R}\big),

    realised, coherently with the rest of the simulator, as a Gaussian
    scale mixture with a *single* shared weight per contaminated period:

    .. math::

        \varepsilon_{i,t}^{\mathrm{contam}}
        \;=\; \frac{\kappa \sqrt{R_i}}{\sqrt{w^{\mathrm{contam}}_t}}\, z_{i,t},
        \qquad z_{i,t} \sim \mathcal{N}(0, 1),
        \qquad w^{\mathrm{contam}}_t \sim
               \mathrm{Gamma}(\nu_{\mathrm{contam}}/2,\, \nu_{\mathrm{contam}}/2).

    Parameters
    ----------
    eps : np.ndarray, shape (T, M)
        Clean idiosyncratic shocks (baseline Student-t draws).  Not
        mutated: a copy is returned with the contaminated rows
        overwritten.
    R : np.ndarray, shape (M,)
        Diagonal idiosyncratic variances (the same ``R`` used to build
        ``eps``).  The contaminated covariance is ``kappa^2 * R``.
    pi : float
        Contamination intensity in ``[0, 1]``.  ``pi = 0`` is a strict
        no-op (no draws taken — see Notes on bit-identity).
    rng : numpy.random.Generator
        Dedicated RNG for the contamination draws (the Bernoulli mask,
        the Gamma weights, and the Gaussian seeds).  Kept separate from
        the baseline idiosyncratic RNG by the caller so toggling
        contamination never perturbs the baseline ``eps`` / ``w_eps``.
    nu_contam : float, default 3.0
        Degrees of freedom of the contaminated Student-t.  ``np.inf``
        (or a value above :data:`_NU_GAUSSIAN_THRESHOLD`) collapses the
        contaminated draw to a Gaussian of scale ``kappa^2 R``.
    kappa : float, default 5.0
        Scale-inflation factor; contaminated shocks have covariance
        ``kappa^2 * R`` (≈ 25x baseline at ``kappa = 5``).

    Returns
    -------
    (eps_out, contam_mask) : tuple
        - ``eps_out`` : np.ndarray, shape ``(T, M)`` — ``eps`` with the
          contaminated rows replaced.  At ``pi = 0`` this is the input
          array unchanged (same values).
        - ``contam_mask`` : np.ndarray of bool, shape ``(T,)`` — the
          ground-truth indicator (``True`` where :math:`z_t = 1`).

    Notes
    -----
    **Why substitutive, inflated-Student-t, and idiosyncratic only.**

    This mirrors the thesis contamination model (section "The
    contamination mechanism", ~line 12419) and its economic
    interpretation:

      - *Substitutive, not additive on top of a normal shock.*  A
        contaminated period is one whose idiosyncratic disturbance is
        generated by a different, fat-tailed regime — the normal shock
        does not also occur underneath.  The replacement keeps the
        scale-mixture algebra clean (one weight per period, exactly as
        the baseline) so the contaminated panel is still a valid input
        to the same Kalman/EM machinery.

      - *Only the idiosyncratic component is contaminated; the factor
        process is left untouched.*  Outliers in macro/financial panels
        manifest as deviations *from* the common factor structure, not
        as moves *of* the latent factor: a flash crash or a data glitch
        is a spike in a series' residual, while a genuine business-cycle
        shock that moves the whole economy is, by definition, a factor
        move and is already modelled by the heavy-tailed factor
        innovation.  Contaminating :math:`\varepsilon_t` (and never
        :math:`u_t`) is therefore the economically correct way to test
        robustness — it is precisely the case the Student-t idiosyncratic
        weights :math:`w^\varepsilon_t` are meant to down-weight.

      - *Limits in pi.*  At :math:`\pi = 0` the DGP reduces exactly to
        the uncontaminated model (Experiment B if additionally
        :math:`\nu \to \infty`; the calibrated Student-t DGP of
        Experiment A otherwise).  As :math:`\pi` grows the panel is
        increasingly polluted by outliers, and the gap between the
        tail-robust (Student-t) and the Gaussian estimator should widen
        — the headline of Experiment C.

    **Bit-identity at pi = 0.**

    When ``pi == 0`` the function returns immediately *without drawing
    anything* from ``rng``.  Combined with the caller using a dedicated
    contamination RNG (so the baseline ``w_eps`` / ``Z`` draws are made
    on a different stream), this guarantees that, at fixed seed, the
    ``pi = 0`` panel is bit-for-bit identical to the panel produced by
    the contamination-free simulator.

    Raises
    ------
    ValueError
        If ``eps`` is not 2-D, ``R`` is not ``(M,)``, ``pi`` is outside
        ``[0, 1]``, ``nu_contam <= 0`` (or NaN), or ``kappa <= 0``.
    """
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 2:
        raise ValueError(f"eps must be 2-D (T, M); got ndim={eps.ndim}.")
    T, M = eps.shape
    R = np.asarray(R, dtype=float)
    if R.shape != (M,):
        raise ValueError(f"R shape {R.shape} does not match eps' M = {M}.")
    if np.isnan(pi) or not (0.0 <= pi <= 1.0):
        raise ValueError(f"pi must lie in [0, 1]; got {pi}.")
    if np.isnan(nu_contam) or nu_contam <= 0:
        raise ValueError(
            f"nu_contam must be a positive number (np.inf for the Gaussian "
            f"limit); got {nu_contam}."
        )
    if kappa <= 0:
        raise ValueError(f"kappa must be strictly positive; got {kappa}.")

    # ── pi == 0: strict no-op ─────────────────────────────────────────────────
    # Return eps UNCHANGED and an all-False mask, WITHOUT drawing from rng.
    # This is what makes the pi=0 panel bit-identical to the old simulator.
    if pi == 0.0:
        return eps, np.zeros(T, dtype=bool)

    # ── Bernoulli(pi) indicator per period — shared across the M series ──────
    contam_mask = rng.random(T) < pi                # (T,) bool
    n_c = int(contam_mask.sum())
    if n_c == 0:
        # No period happened to be contaminated this draw; eps unchanged.
        return eps, contam_mask

    # ── Inflated-Student-t draw at the contaminated periods ──────────────────
    # One shared weight per contaminated period (mirror of w_eps_t), then a
    # fresh N(0,1) per (series, contaminated-period).  Marginally this is
    # t_{nu_contam}(0, kappa^2 R).
    _gaussian_contam = np.isinf(nu_contam) or nu_contam > _NU_GAUSSIAN_THRESHOLD
    if _gaussian_contam:
        w_contam = np.ones(n_c)                     # Gaussian limit: weights ≡ 1
    else:
        w_contam = rng.gamma(
            shape=nu_contam / 2.0,
            scale=2.0 / nu_contam,
            size=n_c,
        )                                           # (n_c,), strictly > 0
    Zc = rng.standard_normal(size=(n_c, M))         # (n_c, M) iid N(0,1)
    eps_contam = (
        Zc * (kappa * np.sqrt(R))[np.newaxis, :]
        / np.sqrt(w_contam)[:, np.newaxis]
    )                                               # (n_c, M)

    eps_out = eps.copy()
    eps_out[contam_mask] = eps_contam
    return eps_out, contam_mask


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   MISSING-DATA-PATTERN OVERLAY                                           ║
# ║   (mixed-frequency quarterly mask + ragged end-of-sample edge)           ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Default name of the "timely" series — the one that remains observed at the
# end-of-sample ragged edge.  In the real panel of this project it is NFCI
# (the Chicago Fed's National Financial Conditions Index, weekly aggregated
# to monthly), the financial-block series with the shortest publication lag.
# If a different panel is simulated and this series name is absent, the
# fallback in apply_missing_pattern documents what happens.
_DEFAULT_TIMELY_SERIES: str = "NFCI"


def apply_missing_pattern(
    Y_complete: np.ndarray,
    freq_list: list[str],
    ordered_cols: list[str],
    ragged_months: int = 2,
    ragged_series: list[str] | None = None,
    quarter_end_offset: int = 2,
) -> np.ndarray:
    r"""
    Overlay the *mixed-frequency* quarterly mask (GDP visible only at
    quarter-end months) and the *ragged* end-of-sample mask (only a
    handful of timely series observed in the last few months) on top
    of the complete simulated panel ``Y_complete``.  Returns the
    resulting ``(T, M)`` panel ``Y`` with ``NaN`` everywhere a real
    observation would not be available.

    The returned ``Y`` has *exactly* the same kind of missing
    structure that the EM-side Kalman filter is built to handle via
    the time-varying selection matrices ``W_t`` (see
    :func:`kalman.build_all_selection_matrices`).  No explicit
    ``W_t`` is constructed here: ``W_t`` is rebuilt at runtime from
    the ``NaN`` pattern of ``Y`` inside the EM, so all we need to do
    is *place the NaNs in the right cells*.

    Parameters
    ----------
    Y_complete : np.ndarray, shape (T, M)
        Complete simulated panel from :func:`simulate_observations`.
        Quarterly rows already carry ``NaN`` at ``t in {0, 1, 2, 3}``
        (the five-lag MM boundary).  Those entries remain ``NaN``
        — both kinds of missingness compose by union, never overwrite.
    freq_list : list[str], length M
        ``"monthly"`` / ``"quarterly"`` per series.  Determines which
        rows receive the quarterly mask.
    ordered_cols : list[str], length M
        Series names in the order of the columns of ``Y_complete``.
        Used (a) to identify ``ragged_series`` by name, and (b) to
        locate the default *timely* series (``NFCI``) when
        ``ragged_series=None``.
    ragged_months : int, default 2
        Number of trailing months in which the end-of-sample ragged
        mask is applied.  The real panel of this project has
        ``ragged_months = 2`` (the last two months: only ``NFCI`` is
        observed).  Set to ``0`` to disable the ragged overlay
        entirely.
    ragged_series : list[str] or None, default None
        Series to mask at the trailing ``ragged_months`` months.  If
        ``None`` the default is "*every series except the timely
        one*" — i.e. every series except :data:`_DEFAULT_TIMELY_SERIES`
        (``"NFCI"``) — which faithfully reproduces the ragged
        structure observed in the real panel (see Notes).  Series
        names not present in ``ordered_cols`` are silently ignored;
        names present are looked up to obtain their column indices.
    quarter_end_offset : int, default 2
        Offset that defines the quarter-end month indices.  With
        ``offset = 2`` and the convention that ``t = 0`` is January,
        the quarter-end months are
        ``t = 2, 5, 8, 11, ...`` (March, June, September, December),
        matching the real dataset (which starts 1985-01 with GDP
        first observed at 1985-03 = ``t = 2``).  See Notes for
        alternative conventions.

    Returns
    -------
    np.ndarray, shape ``(T, M)``
        The masked panel ``Y``.  Cells set to ``NaN``:

        - *quarterly rows* (``freq == "quarterly"``) at every ``t``
          that is NOT a quarter-end (~ two-thirds of the months),
        - any *named ragged series* at the last ``ragged_months``
          months of the sample,
        - any ``NaN`` already present in ``Y_complete`` (the
          quarterly MM-boundary rows at ``t < 4``).

        ``Y_complete`` is not mutated.

    Raises
    ------
    ValueError
        If ``Y_complete`` is not ``(T, M)`` with ``len(freq_list) == M``
        and ``len(ordered_cols) == M``, if ``ragged_months < 0``, or
        if ``ragged_months > T``.

    Notes
    -----
    **Why a quarter-end mask: mixed-frequency observation.**

    The macro panel mixes monthly indicators (industrial production,
    payrolls, prices, financial spreads, ...) with one quarterly
    indicator (real GDP).  In a *monthly* state-space model the
    quarterly series simply cannot be observed in two out of three
    months — by definition.  We encode this by setting
    ``Y[~quarter_end_mask, q_idx] = NaN``, which is exactly what the
    EM observes on the real panel and what the time-varying
    selection matrices ``W_t`` of the Kalman filter are designed for.
    See :func:`kalman.build_all_selection_matrices`.

    **The quarter-end convention.**

    With ``T`` monthly time indices ``t = 0, 1, ..., T-1`` and the
    convention that ``t = 0`` is the first month of the simulated
    sample (= January if calibrated against this project's real
    panel which starts 1985-01), the quarter-end months are
    those with ``(t - quarter_end_offset) % 3 == 0`` for some
    integer offset.  Setting ``quarter_end_offset = 2`` gives
    ``t = 2, 5, 8, 11, ...`` — i.e. March, June, September, December
    — which matches what is observed in
    ``data/processed/dataset_usa.csv``: the first ``GDPC1`` non-NaN
    row is at ``t = 2`` (1985-03), and ``GDPC1`` is observed at
    exactly ``165 = floor(497/3) + 1`` of the 497 months in the real
    panel.  The same convention is used here for symmetry.

    Any other ``quarter_end_offset`` in ``{0, 1, 2}`` would simply
    shift which monthly index counts as the quarter-end; the EM
    machinery does not care about the absolute calendar — it
    only needs the *fraction* of months in which GDP is observed to
    be ~1/3, which holds for any offset.

    **Why a ragged end-of-sample edge.**

    Real macro data are published with a publication lag: at the
    "vintage" date when one runs the model, the most recent two or
    three months have only a *subset* of indicators available.  In
    this project's real panel the last two months
    (April-2026, May-2026) have ``m_t = 1`` — only ``NFCI`` is
    observed.  This is what makes the *nowcasting* problem
    non-trivial: the model has to forecast GDP using only the
    indicators that are out at the vintage date.

    To replicate this in the simulator we mask every series except
    ``NFCI`` for the last ``ragged_months`` months (default 2).
    The choice of NFCI is data-driven: it is the financial-block
    series with the shortest publication lag in the project's panel
    (it is weekly, then aggregated to monthly).  The user can
    override the default via ``ragged_series``.

    **The two masks compose by UNION.**

    The quarterly mask and the ragged mask act on different cells in
    general — the quarterly mask only ever touches the GDP column,
    while the ragged mask touches whichever series are listed in
    ``ragged_series``.  Their effects compose by union: a cell is
    NaN in ``Y`` if it is NaN for *any* of the reasons (already
    NaN in ``Y_complete`` because of the MM boundary at ``t < 4``,
    or masked by the quarterly rule, or masked by the ragged rule).
    The implementation just calls ``Y[...] = np.nan`` in sequence —
    NaN is idempotent under overwriting, so we never need to
    remember which rule set a cell to NaN first.

    **Distribution of m_t after masking.**

    For ``T`` large and ``ragged_months`` small relative to ``T``:

        - ``m_t = M``     (quarter-end months not in the ragged tail):
                          ~ ``T / 3`` of them.
        - ``m_t = M - 1`` (non-quarter-end months not in the ragged
                          tail; GDP missing, everything else present):
                          ~ ``2 T / 3`` of them.
        - ``m_t = 1``     (ragged-tail months; only ``NFCI`` observed,
                          including GDP missing because the ragged
                          tail is generically not a quarter-end):
                          ``ragged_months`` of them, with the
                          possible exception of a tail month that
                          *happens* to coincide with a quarter-end
                          (in which case ``m_t = 1`` still holds
                          because the ragged mask wipes the GDP
                          column too — see the default
                          ``ragged_series``).

    This distribution is what the EM expects, and what
    :func:`kalman.build_all_selection_matrices` is built to handle
    via the per-``t`` selection matrices ``W_t``.

    Examples
    --------
    >>> Y_complete = obs["Y_complete"]
    >>> Y = apply_missing_pattern(
    ...     Y_complete,
    ...     freq_list=[FREQ[c] for c in ORDERED_COLS],
    ...     ordered_cols=ORDERED_COLS,
    ...     ragged_months=2,
    ... )
    >>> # Quarter-end months (t = 2, 5, 8, ...) have GDP observed:
    >>> int(np.isfinite(Y[2, ORDERED_COLS.index("GDPC1")]))
    1
    >>> # Last 2 months have only NFCI observed:
    >>> np.sum(np.isfinite(Y[-1]))
    1
    """
    # ── 1. Argument validation ───────────────────────────────────────────────
    Y_complete = np.asarray(Y_complete, dtype=float)
    if Y_complete.ndim != 2:
        raise ValueError(f"Y_complete must be 2D; got ndim={Y_complete.ndim}.")
    T, M = Y_complete.shape
    if len(freq_list) != M:
        raise ValueError(
            f"len(freq_list) = {len(freq_list)} does not match M = {M}."
        )
    if len(ordered_cols) != M:
        raise ValueError(
            f"len(ordered_cols) = {len(ordered_cols)} does not match M = {M}."
        )
    if ragged_months < 0:
        raise ValueError(
            f"ragged_months must be non-negative; got {ragged_months}."
        )
    if ragged_months > T:
        raise ValueError(
            f"ragged_months ({ragged_months}) exceeds T ({T})."
        )

    Y = Y_complete.copy()

    # ── 2. Quarterly mask (GDP-like series) ──────────────────────────────────
    # t is a quarter-end iff (t - quarter_end_offset) % 3 == 0.  With offset=2,
    # quarter-end months are t = 2, 5, 8, ..., i.e. March, June, ... when the
    # sample starts in January (1985-01 in the real panel).
    t_arr = np.arange(T)
    is_quarter_end = ((t_arr - quarter_end_offset) % 3 == 0)
    q_idx = [i for i, f in enumerate(freq_list) if f == "quarterly"]
    for q in q_idx:
        Y[~is_quarter_end, q] = np.nan

    # ── 3. Ragged end-of-sample edge ─────────────────────────────────────────
    if ragged_months > 0:
        if ragged_series is None:
            # Default mirror of the real panel: mask every series except the
            # timely one (NFCI).  If NFCI is absent from ordered_cols (e.g.
            # a Monte Carlo run on a custom subset that does not include it),
            # we mask every monthly series and leave the user to inspect.
            if _DEFAULT_TIMELY_SERIES in ordered_cols:
                ragged_series = [
                    c for c in ordered_cols if c != _DEFAULT_TIMELY_SERIES
                ]
            else:
                # Conservative fallback: mask every series, including any
                # quarterly one (the quarterly mask has already taken care of
                # those, so this is a no-op for them).
                ragged_series = list(ordered_cols)
        ragged_idx = [
            ordered_cols.index(name) for name in ragged_series
            if name in ordered_cols
        ]
        for r_i in ragged_idx:
            Y[-ragged_months:, r_i] = np.nan

    return Y


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   simulate_dfm — full end-to-end simulator wrapper                       ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_dfm(
    theta: dict | "np.lib.npyio.NpzFile",
    T: int,
    freq_list: list[str],
    block_map: dict[str, str],
    ordered_cols: list[str],
    r: int,
    seed: int,
    burn_in: int = 1000,
    ragged_months: int = 2,
    ragged_series: list[str] | None = None,
    mm_weights: list[float] | None = None,
    quarter_end_offset: int = 2,
    pi: float = 0.0,
    nu_contam: float = 3.0,
    kappa: float = 5.0,
) -> dict:
    r"""
    End-to-end Student-t DFM simulator: chain the three building
    blocks (factor process, observation equation, missing-data
    overlay) and return the synthetic panel ``Y`` *together with the
    ground truth* needed by the Monte Carlo self-recovery test.

    A single call to :func:`simulate_dfm` produces everything the
    self-recovery test needs:

      1. ``Y`` — the synthetic panel actually fed to ``run_em`` /
         ``fit_dfm``, with the same NaN structure as the real
         dataset.
      2. ``F``, ``w_u_true``, ``w_eps_true`` — the ground-truth
         latent factors and mixing weights actually drawn.  These
         are not available on the real dataset but are essential for
         level-2 (latent-state) recovery diagnostics.
      3. ``Y_complete`` — the pre-mask panel, useful for diagnostics
         (e.g. comparing fitted values on the *complete* signal to
         see how much of the residual error is genuine noise vs
         loss from missingness).
      4. ``theta_used`` — the parameter dict from which the data were
         generated.  Saved alongside the simulated panel so that
         downstream code can compare ``theta_hat`` to
         ``theta_used`` without ambiguity.

    Parameters
    ----------
    theta : dict-like
        Parameter set from which to generate.  Typically the
        converged EM estimate on the real panel (``theta_star``),
        but any structurally valid set works.  Required keys:
        ``A`` (r, r), ``Q`` (r, r), ``Lambda`` (M, r), ``R`` (M,),
        ``nu_u`` (scalar), ``nu_eps`` (scalar).  Other keys
        (``Sigma_0``, ``F``, ``w_u``, ``w_eps``, ...) are ignored —
        the simulator does not depend on them, but they are *not*
        cleared from the returned ``theta_used`` (it is a shallow
        copy of the input).
    T : int
        Number of monthly periods to simulate.  Typical Monte Carlo
        choices: ``T = 497`` (= length of the real panel, for a
        like-for-like comparison) or ``T = 2000`` (for tighter
        recovery statistics).
    freq_list, block_map, ordered_cols, r
        Panel metadata, passed through unchanged to
        :func:`simulate_observations` and
        :func:`apply_missing_pattern`.  Must be mutually consistent
        (``len(freq_list) == len(ordered_cols) == M``, etc.).
    seed : int
        Master seed.  The factor and observation RNGs are derived
        from this seed using disjoint integer offsets so that
        independent randomness is used for the two stages (see
        ``simulate_factors`` and ``simulate_observations``).
    burn_in : int, default 1000
        VAR burn-in for the factor process.  Long enough to make the
        transient negligible at ``rho(A) ~ 0.978``; see
        :func:`simulate_factors`.
    ragged_months : int, default 2
        Length of the end-of-sample ragged edge; see
        :func:`apply_missing_pattern`.
    ragged_series : list[str] or None, default None
        Series to mask at the ragged edge.  ``None`` defaults to "all
        except NFCI" — the real-panel mirror; see
        :func:`apply_missing_pattern`.
    mm_weights : list[float] or None, default None
        Five MM aggregation weights for the quarterly rows;
        ``None`` defaults to ``[1/3, 2/3, 1, 2/3, 1/3]`` (= the
        kalman-side default).
    quarter_end_offset : int, default 2
        Quarter-end convention for the quarterly mask; see
        :func:`apply_missing_pattern`.
    pi : float, default 0.0
        Contamination intensity (Experiment C).  ``0`` ⇒ no
        contamination (output bit-identical to the contamination-free
        simulator at fixed seed).  See :func:`apply_contamination`.
    nu_contam : float, default 3.0
        Degrees of freedom of the inflated contaminating Student-t.
    kappa : float, default 5.0
        Scale-inflation factor of the contaminated idiosyncratic
        covariance (``kappa^2 * R``).

    Returns
    -------
    dict
        - ``Y`` : np.ndarray, shape ``(T, M)`` — the synthetic
          panel with mixed-frequency + ragged ``NaN`` pattern.  The
          object passed to ``run_em`` / ``fit_dfm`` in the Monte
          Carlo.
        - ``F`` : np.ndarray, shape ``(T, r)`` — the true monthly
          factor matrix.
        - ``w_u_true`` : np.ndarray, shape ``(T,)`` — true factor-
          innovation mixing weights.
        - ``w_eps_true`` : np.ndarray, shape ``(T,)`` — true
          idiosyncratic mixing weights.
        - ``Y_complete`` : np.ndarray, shape ``(T, M)`` — the
          pre-mask panel (only the MM-boundary NaNs for ``t < 4``).
        - ``contam_mask`` : np.ndarray of bool, shape ``(T,)`` — the
          ground-truth contamination indicator (``True`` at outlier
          periods; all ``False`` when ``pi = 0``).  Ground truth for
          Experiment C's detection-rate metric.
        - ``theta_used`` : dict — shallow copy of the input
          ``theta``, for downstream parameter-recovery comparison.

    Notes
    -----
    **Independent randomness for the factor and observation processes.**

    The factor and observation processes are conditionally
    independent given the latent state:  factor innovations and
    idiosyncratic noise enter the model in different equations, with
    no cross-coupling.  We therefore draw them with *different*
    RNG seeds — concretely, ``seed`` for the factor process,
    ``seed + 1`` for the observations, and ``seed + 2`` for the
    contamination overlay (the Bernoulli mask + the inflated
    Student-t outlier draws).  This keeps the three stages decoupled:
    re-running the simulator with a different seed gives independent
    variation in all of them, refactoring one stage does not perturb
    the others' draws, and — crucially — toggling contamination
    on/off (``pi``) never moves the baseline factor/observation
    streams, so the ``pi = 0`` panel is bit-identical to the
    contamination-free simulator at fixed seed.

    Examples
    --------
    >>> sim = simulate_dfm(
    ...     theta=theta_star, T=2000, freq_list=freq_list,
    ...     block_map=BLOCK, ordered_cols=ORDERED_COLS, r=3,
    ...     seed=42,
    ... )
    >>> sim["Y"].shape, sim["F"].shape
    ((2000, 20), (2000, 3))
    """
    # ── 1. Factor process ────────────────────────────────────────────────────
    factors = simulate_factors(
        A=np.asarray(theta["A"]),
        Q=np.asarray(theta["Q"]),
        nu_u=float(np.asarray(theta["nu_u"])),
        T=T, r=r,
        seed=seed,
        burn_in=burn_in,
    )
    F        = factors["F"]
    w_u_true = factors["w_u_true"]

    # ── 2. Observation equation ──────────────────────────────────────────────
    obs = simulate_observations(
        F=F,
        Lambda=np.asarray(theta["Lambda"]),
        R=np.asarray(theta["R"]),
        nu_eps=float(np.asarray(theta["nu_eps"])),
        freq_list=freq_list,
        block_map=block_map,
        ordered_cols=ordered_cols,
        r=r,
        seed=seed + 1,                # independent stream from the factor RNG
        mm_weights=mm_weights,
        pi=pi,
        nu_contam=nu_contam,
        kappa=kappa,
        contam_seed=seed + 2,         # third disjoint stream: contamination only
    )
    Y_complete  = obs["Y_complete"]
    w_eps_true  = obs["w_eps_true"]
    contam_mask = obs["contam_mask"]

    # ── 3. Missing-data overlay ──────────────────────────────────────────────
    Y = apply_missing_pattern(
        Y_complete=Y_complete,
        freq_list=freq_list,
        ordered_cols=ordered_cols,
        ragged_months=ragged_months,
        ragged_series=ragged_series,
        quarter_end_offset=quarter_end_offset,
    )

    # ── 4. Package ground truth + reference theta into the return dict ───────
    # Shallow copy of theta: keep the original around for parameter recovery
    # comparison without mutating the caller's dict.  For NpzFile inputs we
    # materialise into a plain dict.
    theta_used: dict = {key: np.asarray(theta[key]) for key in theta.keys()}

    return {
        "Y":           Y,
        "F":           F,
        "w_u_true":    w_u_true,
        "w_eps_true":  w_eps_true,
        "Y_complete":  Y_complete,
        "contam_mask": contam_mask,
        "theta_used":  theta_used,
    }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    import sys

    # Locate project root and make sibling modules importable.
    project_root = pathlib.Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from config_utils import parse_config_args, resolve_output_path
    args = parse_config_args("simulate_dfm self-test")
    cfg  = args.config
    print(f"Config: {cfg!r}")

    # ── 1. Load theta_star from the cached EM result ─────────────────────────
    fit_path = resolve_output_path("processed", "fit_dfm_result.npz", cfg)
    print(f"Loading theta_star from: {fit_path}")
    archive = np.load(fit_path)
    A_star    = np.asarray(archive["theta__A"])
    Q_star    = np.asarray(archive["theta__Q"])
    nu_u_star = float(archive["theta__nu_u"])
    r         = int(archive["r"])
    print(f"  A     shape {A_star.shape},  spectral radius "
          f"rho(A) = {max(abs(np.linalg.eigvals(A_star))):.6f}")
    print(f"  Q     shape {Q_star.shape},  diag(Q) = "
          f"{np.array2string(np.diag(Q_star), precision=4)}")
    print(f"  nu_u  = {nu_u_star:.4f}")
    print(f"  r     = {r}")

    # ── 2. Simulate ───────────────────────────────────────────────────────────
    T_sim       = 2000
    seed        = 42
    burn_in     = 1000
    print(f"\nSimulating: T={T_sim}, seed={seed}, burn_in={burn_in} ...")
    out = simulate_factors(
        A=A_star, Q=Q_star, nu_u=nu_u_star,
        T=T_sim, r=r, seed=seed, burn_in=burn_in,
    )
    F        = out["F"]
    w_u_true = out["w_u_true"]

    # ── 3. Shape, finiteness, positivity sanity checks ───────────────────────
    print("\n" + "=" * 64)
    print("Sanity checks")
    print("=" * 64)
    assert F.shape == (T_sim, r), f"F shape {F.shape} != ({T_sim}, {r})"
    assert w_u_true.shape == (T_sim,), \
        f"w_u_true shape {w_u_true.shape} != ({T_sim},)"
    assert np.all(np.isfinite(F)), "F contains NaN/inf"
    assert np.all(np.isfinite(w_u_true)), "w_u_true contains NaN/inf"
    assert np.all(w_u_true > 0), "w_u_true contains non-positive entries"
    print(f"[OK] F.shape        = {F.shape}")
    print(f"[OK] w_u_true.shape = {w_u_true.shape}")
    print(f"[OK] no NaN/inf in F or w_u_true")
    print(f"[OK] w_u_true > 0 everywhere "
          f"(min = {w_u_true.min():.4f}, max = {w_u_true.max():.4f})")

    # Theoretical: w ~ Gamma(nu/2, scale=2/nu) has mean 1, var = 2/nu.
    # Sample mean has approx std sqrt(2/nu/T) for the sample mean of w.
    mean_w  = float(w_u_true.mean())
    var_w   = float(w_u_true.var(ddof=0))
    theo_se = np.sqrt(2.0 / nu_u_star / T_sim)
    print(f"\nGround-truth weights w_u_true:")
    print(f"  mean(w_u_true) = {mean_w:.6f}   (theoretical 1.0, "
          f"sample SE ~ {theo_se:.4f})")
    print(f"  var (w_u_true) = {var_w:.6f}    (theoretical {2.0 / nu_u_star:.4f})")
    assert abs(mean_w - 1.0) < 6.0 * theo_se, (
        f"sample mean of w_u_true ({mean_w:.4f}) is more than 6 SE from 1; "
        f"check the Gamma parametrisation."
    )
    print(f"[OK] mean(w_u_true) within 6 SE of 1")

    # ── 4. VAR(1) recovery via OLS — verifies the simulator's dynamics ───────
    # F_t = A F_{t-1} + u_t  =>  let X = F[:-1], Y = F[1:].
    #   Y = X A' + u    =>    A' = (X'X)^{-1} X'Y    =>    A = Y' X (X'X)^{-1}.
    # Equivalently:  np.linalg.lstsq(X, Y) returns the coefficient C of shape
    # (r, r) with Y ≈ X @ C, and we have A = C.T.
    X = F[:-1]
    Y = F[1:]
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_ols = C.T

    print("\n" + "=" * 64)
    print("VAR(1) recovery — OLS on simulated factors")
    print("=" * 64)
    print(f"\nA_true (theta_star):")
    print(np.array2string(A_star, precision=4, suppress_small=True))
    print(f"\nA_ols  (OLS on simulated F):")
    print(np.array2string(A_ols, precision=4, suppress_small=True))
    diff_A     = np.linalg.norm(A_ols - A_star)
    relerr_A   = diff_A / np.linalg.norm(A_star)
    rho_true   = float(max(abs(np.linalg.eigvals(A_star))))
    rho_ols    = float(max(abs(np.linalg.eigvals(A_ols))))
    print(f"\n||A_ols - A_true||_F     = {diff_A:.4f}")
    print(f"||A_ols - A_true|| / ||A_true|| = {relerr_A:.4f}")
    print(f"spectral radius A_true = {rho_true:.6f}")
    print(f"spectral radius A_ols  = {rho_ols:.6f}")

    # ── 5. Heavy-tail diagnostic — kurtosis of implied innovations ───────────
    # u_t implied = F[1:] - F[:-1] @ A.T   (using the true A, not A_ols).
    U_implied = F[1:] - F[:-1] @ A_star.T
    # Pearson kurtosis: gaussian -> 3, Student-t with nu dof -> 3(nu-2)/(nu-4)
    # for nu > 4 (undefined for nu <= 4).  At nu_u ~ 4 the sample kurtosis can
    # be very large and sample-dependent — we just check it is well above 3.
    kurt = np.array([
        np.mean((U_implied[:, j] - U_implied[:, j].mean()) ** 4)
        / U_implied[:, j].var(ddof=0) ** 2
        for j in range(r)
    ])
    print("\n" + "=" * 64)
    print("Heavy-tail check — Pearson kurtosis of implied innovations u_t")
    print("=" * 64)
    print(f"  per-factor kurtosis = "
          f"{np.array2string(kurt, precision=2)}   (gaussian = 3)")
    print(f"  Student-t reference at nu_u = {nu_u_star:.2f}: "
          + (f"3*(nu-2)/(nu-4) = {3*(nu_u_star-2)/(nu_u_star-4):.2f}"
             if nu_u_star > 4 else
             "undefined (nu_u <= 4; expect *very* large sample kurtosis)"))
    assert np.all(kurt > 3.0), (
        f"At least one per-factor kurtosis is <= 3 "
        f"({kurt.tolist()}); heavy tails not observed."
    )
    print(f"[OK] every per-factor kurtosis > 3")

    # ── 6. Outlier alignment — low w_u_true should coincide with large |u| ───
    # Use the threshold 5th percentile of w_u_true as the "outlier" set.
    # Periods t in 1..T-1 indexed in U_implied as t-1.
    w_post = w_u_true[1:]                          # aligned with U_implied
    thr    = np.percentile(w_post, 5.0)
    mask_outlier = w_post < thr
    norm_U = np.linalg.norm(U_implied, axis=1)
    print(f"\nOutlier alignment (low w_u_true <-> large ||u_t||):")
    print(f"  5th-pctl threshold of w_u_true = {thr:.4f}")
    print(f"  mean ||u_t||  on outlier set   = "
          f"{norm_U[mask_outlier].mean():.4f}")
    print(f"  mean ||u_t||  on rest          = "
          f"{norm_U[~mask_outlier].mean():.4f}")
    print(f"  ratio (outlier / rest)         = "
          f"{norm_U[mask_outlier].mean() / norm_U[~mask_outlier].mean():.2f}x")
    assert (
        norm_U[mask_outlier].mean() > 1.5 * norm_U[~mask_outlier].mean()
    ), "Low-w_u_true periods do NOT show larger |u_t| — simulator bug?"
    print(f"[OK] low-w periods have substantially larger ||u_t||")

    # ── 7. Plot the simulated factors and highlight outlier periods ──────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_path = resolve_output_path("figures", "sim_factors.png", cfg)

        # Outlier periods on the *full* T_sim grid (not the t-1 shifted one).
        thr_full     = np.percentile(w_u_true, 5.0)
        mask_full    = w_u_true < thr_full
        block_labels = ["real (f^R)", "financial (f^F)", "other (f^X)"]

        fig, axes = plt.subplots(r, 1, figsize=(11, 7), sharex=True)
        for j in range(r):
            ax = axes[j]
            ax.plot(F[:, j], lw=0.8, color="C0", label=f"factor {j}")
            ax.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
            # Vertical bands for outlier periods (5th-pctl of w_u_true).
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                np.where(mask_full)[0],
                ymin=ymin, ymax=ymax,
                colors="C3", alpha=0.18, lw=0.8,
                label=("low w_u_true (5th pctl)" if j == 0 else None),
            )
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(block_labels[j] if j < len(block_labels) else f"f{j}")
            if j == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("simulated time index (post burn-in)")
        fig.suptitle(
            f"Simulated factors  —  T={T_sim}, seed={seed}, nu_u={nu_u_star:.2f}, "
            f"rho(A)={rho_true:.3f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        print(f"\n[OK] saved plot: {fig_path}")
    except Exception as exc:
        print(f"\n[WARN] plot skipped: {exc!r}")

    print("\n" + "=" * 64)
    print("simulate_factors  —  all self-test checks passed")
    print("=" * 64)


    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║   TASK 2 — observation-equation self-test                            ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    from data_loader import load_config                               # noqa: E402
    from kalman      import MM_WEIGHTS_DEFAULT, build_Lambda_tilde   # noqa: E402
    _cfg_dict   = load_config(cfg)
    ORDERED_COLS = _cfg_dict["ORDERED_COLS"]
    BLOCK        = _cfg_dict["BLOCK"]
    FREQ         = _cfg_dict["FREQ"]

    # ── 8. Load Lambda, R, nu_eps from theta_star ────────────────────────────
    Lambda_star  = np.asarray(archive["theta__Lambda"])
    R_star       = np.asarray(archive["theta__R"])
    nu_eps_star  = float(archive["theta__nu_eps"])
    freq_list    = [FREQ[c] for c in ORDERED_COLS]
    M_panel      = len(ORDERED_COLS)
    print("\n" + "=" * 64)
    print("Task 2 — simulate_observations")
    print("=" * 64)
    print(f"\nLoaded from theta_star:")
    print(f"  Lambda  shape {Lambda_star.shape}")
    print(f"  R       shape {R_star.shape},  range [{R_star.min():.4f}, "
          f"{R_star.max():.4f}]")
    print(f"  nu_eps  = {nu_eps_star:.4f}")
    print(f"  M       = {M_panel}   (monthly: "
          f"{sum(f == 'monthly' for f in freq_list)},  quarterly: "
          f"{sum(f == 'quarterly' for f in freq_list)})")

    # Sanity: the MM weights used here MUST coincide with the EM-side ones.
    assert np.allclose(_MM_WEIGHTS_DEFAULT, MM_WEIGHTS_DEFAULT), (
        f"MM weights mismatch between simulator ({_MM_WEIGHTS_DEFAULT}) and "
        f"kalman.MM_WEIGHTS_DEFAULT ({MM_WEIGHTS_DEFAULT})."
    )
    print(f"[OK] simulator MM weights match kalman.MM_WEIGHTS_DEFAULT exactly")

    # ── 9. Simulate Y_complete ───────────────────────────────────────────────
    seed_obs = 7
    print(f"\nSimulating observations: T={T_sim}, M={M_panel}, seed={seed_obs}")
    obs = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=seed_obs,
    )
    Y_complete  = obs["Y_complete"]
    w_eps_true  = obs["w_eps_true"]

    # Identify the quarterly column(s) for the boundary check.
    q_idx = [i for i, f in enumerate(freq_list) if f == "quarterly"]
    m_idx = [i for i, f in enumerate(freq_list) if f == "monthly"]
    assert len(q_idx) == 1, f"expected 1 quarterly series, found {len(q_idx)}"
    q   = q_idx[0]
    print(f"  quarterly series:  {ORDERED_COLS[q]}  (column {q})")

    # ── 10. Shape, finiteness, boundary checks ───────────────────────────────
    print("\n" + "-" * 64)
    print("Shape & finiteness")
    print("-" * 64)
    assert Y_complete.shape == (T_sim, M_panel), \
        f"Y shape {Y_complete.shape} != ({T_sim}, {M_panel})"
    assert w_eps_true.shape == (T_sim,)
    # Monthly rows must be fully finite everywhere.
    assert np.all(np.isfinite(Y_complete[:, m_idx])), \
        "monthly rows of Y_complete contain NaN/inf — should be impossible"
    # Quarterly row must be NaN at t < 4 and finite at t >= 4.
    assert np.all(np.isnan(Y_complete[:4, q])), \
        f"quarterly rows for t<4 should be NaN, got {Y_complete[:4, q]}"
    assert np.all(np.isfinite(Y_complete[4:, q])), \
        "quarterly rows for t>=4 should be finite"
    assert np.all(np.isfinite(w_eps_true)) and np.all(w_eps_true > 0)
    print(f"[OK] Y_complete shape {Y_complete.shape}")
    print(f"[OK] monthly rows fully finite; quarterly NaN only at t<4")
    print(f"[OK] w_eps_true > 0, finite (min={w_eps_true.min():.4f}, "
          f"max={w_eps_true.max():.4f})")

    # ── 11. Statistical moments of w_eps_true ────────────────────────────────
    mean_we   = float(w_eps_true.mean())
    var_we    = float(w_eps_true.var(ddof=0))
    theo_se_e = np.sqrt(2.0 / nu_eps_star / T_sim)
    print(f"\nGround-truth weights w_eps_true:")
    print(f"  mean = {mean_we:.6f}   (theoretical 1.0, sample SE ~ {theo_se_e:.4f})")
    print(f"  var  = {var_we:.6f}    (theoretical {2.0 / nu_eps_star:.4f})")
    assert abs(mean_we - 1.0) < 6.0 * theo_se_e
    print(f"[OK] mean(w_eps_true) within 6 SE of 1")

    # ── 12. Loading recovery via OLS on a few monthly series ─────────────────
    # For a monthly series i in block j(i): y_{i,t} = Lambda[i, j] * F[t, j] + eps.
    # OLS of y on F[:, j] should recover Lambda[i, j] within O(sqrt(R_i/T)).
    print("\n" + "-" * 64)
    print("Loading recovery via OLS  (monthly series)")
    print("-" * 64)
    print(f"  {'series':<14s}  {'block':<10s}  {'j':>2s}  "
          f"{'Lambda_true':>12s}  {'Lambda_OLS':>12s}  {'rel.err':>8s}")
    print("  " + "-" * 64)
    pick = ["PAYEMS", "S&P 500", "UMCSENTx", "INDPRO"]
    block_to_col = {"real": 0, "financial": 1, "other": 2}
    for name in pick:
        i_row = ORDERED_COLS.index(name)
        b     = BLOCK[name]
        j_col = block_to_col[b]
        # Univariate OLS of Y[:, i_row] on F[:, j_col] (no intercept — both
        # have zero population mean by construction).
        x      = F[:, j_col]
        y      = Y_complete[:, i_row]
        lam_ols = float((x @ y) / (x @ x))
        lam_t   = float(Lambda_star[i_row, j_col])
        relerr  = abs(lam_ols - lam_t) / max(abs(lam_t), 1e-12)
        print(f"  {name:<14s}  {b:<10s}  {j_col:>2d}  "
              f"{lam_t:>+12.4f}  {lam_ols:>+12.4f}  {relerr:>8.3%}")

    # ── 13. MM aggregation verification on the quarterly series ──────────────
    # Rebuild Phi_t = sum_l c_l F[t-l, j_real] and check
    #   Y[GDP, t] ~ Lambda[GDP, j_real] * Phi[t] + noise.
    j_real = block_to_col[BLOCK[ORDERED_COLS[q]]]   # = 0 for GDPC1
    Phi    = np.full(T_sim, np.nan)
    for t in range(4, T_sim):
        Phi[t] = sum(
            _MM_WEIGHTS_DEFAULT[k] * F[t - k, j_real] for k in range(5)
        )
    y_q    = Y_complete[4:, q]                     # (T-4,)
    phi_q  = Phi[4:]                               # (T-4,)
    lam_q_true = float(Lambda_star[q, j_real])
    lam_q_ols  = float((phi_q @ y_q) / (phi_q @ phi_q))
    relerr_q   = abs(lam_q_ols - lam_q_true) / max(abs(lam_q_true), 1e-12)
    print("\n" + "-" * 64)
    print("MM aggregation verification (quarterly series)")
    print("-" * 64)
    print(f"  {'series':<10s}  {'j':>2s}  {'Lambda_true':>12s}  "
          f"{'Lambda_OLS(phi)':>16s}  {'rel.err':>8s}")
    print("  " + "-" * 52)
    print(f"  {ORDERED_COLS[q]:<10s}  {j_real:>2d}  "
          f"{lam_q_true:>+12.4f}  {lam_q_ols:>+16.4f}  {relerr_q:>8.3%}")
    # Also a direct numerical cross-check that Lambda @ Phi_t = Lambda_tilde @
    # tilde_f_t for ONE quarterly row at one t.  This pins down the simulator
    # as a bit-exact mirror of build_Lambda_tilde.
    Lambda_tilde = build_Lambda_tilde(Lambda_star, freq_list)
    t_check      = 100
    # Build tilde_f_t = (f_t, f_{t-1}, ..., f_{t-4})  -- shape (5r,)
    tilde_ft = np.concatenate([F[t_check - k] for k in range(5)])
    mu_via_tilde = float(Lambda_tilde[q] @ tilde_ft)
    mu_via_phi   = lam_q_true * float(Phi[t_check])
    diff         = abs(mu_via_tilde - mu_via_phi)
    print(f"\n  cross-check at t={t_check}:")
    print(f"    Lambda_tilde[q] @ tilde_f_t = {mu_via_tilde:+.10f}")
    print(f"    Lambda[q, j_real] * Phi[t]  = {mu_via_phi:+.10f}")
    print(f"    |diff|                      = {diff:.2e}")
    assert diff < 1e-12, (
        f"simulator's MM aggregation disagrees with build_Lambda_tilde "
        f"at t={t_check}: diff = {diff:.3e}"
    )
    print(f"  [OK] simulator's MM aggregation matches build_Lambda_tilde "
          f"to machine precision")

    # ── 14. Heavy-tail check on the idiosyncratic residuals ──────────────────
    # E[w_eps_t]=1 so for a monthly series  resid_it = y_it - Lambda[i,:] @ F[t]
    # is the actual eps_it = sqrt(R_i / w_eps_t) z_it.  Its marginal is
    # Student-t with nu_eps dof (Gaussian scale-mixture), so kurtosis > 3.
    print("\n" + "-" * 64)
    print("Heavy-tail check — kurtosis of monthly idiosyncratic residuals")
    print("-" * 64)
    resid_m = Y_complete[:, m_idx] - F @ Lambda_star[m_idx, :].T  # (T, n_month)
    kurt_eps = np.array([
        np.mean((resid_m[:, k] - resid_m[:, k].mean()) ** 4)
        / resid_m[:, k].var(ddof=0) ** 2
        for k in range(resid_m.shape[1])
    ])
    print(f"  per-series kurtosis: min={kurt_eps.min():.2f}  "
          f"median={np.median(kurt_eps):.2f}  max={kurt_eps.max():.2f}  "
          f"(gaussian = 3)")
    if nu_eps_star > 4:
        theo_kurt = 3.0 * (nu_eps_star - 2.0) / (nu_eps_star - 4.0)
        print(f"  Student-t reference kurtosis at nu_eps = {nu_eps_star:.2f}: "
              f"{theo_kurt:.2f}")
    else:
        print(f"  Student-t reference kurtosis undefined (nu_eps <= 4); "
              f"expect very large sample kurtosis")
    assert np.median(kurt_eps) > 3.0, (
        f"median kurtosis {np.median(kurt_eps):.2f} <= 3 — heavy tails "
        f"not observed"
    )
    print(f"  [OK] median kurtosis > 3 across monthly series")

    print("\n" + "=" * 64)
    print("simulate_observations  —  all self-test checks passed")
    print("=" * 64)


    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║   TASK 3 — apply_missing_pattern self-test                           ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    from collections import Counter
    print("\n" + "=" * 64)
    print("Task 3 — apply_missing_pattern")
    print("=" * 64)

    # ── 15. Apply missing pattern on Y_complete from Task 2 ──────────────────
    print(f"\nApplying default missing pattern: ragged_months=2, "
          f"timely series = NFCI")
    Y_masked = apply_missing_pattern(
        Y_complete=Y_complete,
        freq_list=freq_list,
        ordered_cols=ORDERED_COLS,
        ragged_months=2,
    )
    assert Y_masked.shape == (T_sim, M_panel)
    print(f"[OK] Y.shape = {Y_masked.shape}")

    # ── 16. GDP observation count ────────────────────────────────────────────
    # Quarter-end months with offset=2: t = 2, 5, 8, ...
    # The first one (t = 2) falls inside the MM-boundary [t < 4] zone where
    # Task 2 has already set the quarterly row to NaN, so the GDP count is
    # (# quarter-ends) - 1.  For T_sim = 2000 with offset=2 this gives
    #   #qe = (1997 - 2) / 3 + 1 = 666;  observed = 666 - 1 = 665.
    gdp_col   = ORDERED_COLS.index("GDPC1")
    n_obs_gdp = int(np.isfinite(Y_masked[:, gdp_col]).sum())
    expected  = sum(
        1 for t in range(T_sim)
        if (t - 2) % 3 == 0 and t >= 4         # exclude MM-boundary qe at t=2
    )
    print(f"\nGDP observation count:")
    print(f"  non-NaN entries in GDPC1                  = {n_obs_gdp}")
    print(f"  expected (quarter-ends with t>=4)         = {expected}")
    print(f"  fraction                                  = {n_obs_gdp / T_sim:.3%}"
          f"  (~ 1/3)")
    assert n_obs_gdp == expected, (
        f"GDP non-NaN count {n_obs_gdp} != expected {expected}"
    )
    assert 0.32 < n_obs_gdp / T_sim < 0.34
    print(f"[OK] GDP observed at every quarter-end with t>=4 (MM boundary respected)")

    # ── 17. Ragged-edge check: last 2 months have only NFCI ──────────────────
    nfci_col = ORDERED_COLS.index("NFCI")
    last_rows = np.isfinite(Y_masked[-2:])             # (2, M)
    m_t_last  = last_rows.sum(axis=1)
    obs_idx_last = np.where(last_rows[0])[0]
    print(f"\nRagged-edge check (last 2 months):")
    print(f"  m_t for last 2 months = {m_t_last.tolist()}")
    print(f"  observed series at t=T-2: {[ORDERED_COLS[i] for i in obs_idx_last]}")
    assert np.all(m_t_last == 1), \
        f"last 2 months expected m_t=1, got {m_t_last.tolist()}"
    assert obs_idx_last.tolist() == [nfci_col], \
        f"the only observed series should be NFCI, got {obs_idx_last.tolist()}"
    print(f"[OK] last 2 months have m_t = 1, only NFCI observed")

    # ── 18. Distribution of m_t over the whole simulated sample ──────────────
    m_t_arr = np.isfinite(Y_masked).sum(axis=1)        # (T,)
    print(f"\nDistribution of m_t over T={T_sim} months:")
    print(f"  {'m_t':>4s}  {'count':>6s}  {'share':>7s}")
    print(f"  ----  ------  -------")
    for k, v in sorted(Counter(m_t_arr.tolist()).items()):
        print(f"  {k:>4d}  {v:>6d}  {v / T_sim:>7.3%}")
    # Sanity ranges: m_t = M (quarter-end inside the panel), m_t = M-1
    # (non-quarter-end, no ragged), m_t = 1 (ragged tail).  At T = 2000,
    # ragged = 2, offset = 2:  m_t=20 count ~ floor(1998/3)+1 minus any
    # quarter-end falling inside the ragged tail; m_t=19 count fills the rest.
    n_mM  = int((m_t_arr == M_panel).sum())
    n_mM1 = int((m_t_arr == M_panel - 1).sum())
    n_m1  = int((m_t_arr == 1).sum())
    assert n_m1 == 2,  f"expected 2 ragged months, got {n_m1}"
    assert n_mM + n_mM1 + n_m1 == T_sim, \
        f"m_t distribution missing some months: {n_mM}+{n_mM1}+{n_m1} != {T_sim}"
    print(f"[OK] m_t in {{1, {M_panel-1}, {M_panel}}} as expected for the default pattern")

    # ── 19. Sanity vs the real panel's m_t distribution ──────────────────────
    # The real panel has m_t = 19 dominant (~66%), m_t = 20 at ~33%, and 2
    # months of m_t = 1 (ragged) — match the qualitative pattern here.
    print(f"\nQualitative comparison to the real panel:")
    print(f"  real-panel dominant m_t   = 19    (~66%)")
    print(f"  simulator dominant m_t    = {Counter(m_t_arr.tolist()).most_common(1)[0][0]}    "
          f"({Counter(m_t_arr.tolist()).most_common(1)[0][1] / T_sim:.1%})")
    print(f"  real-panel quarter-end %  = 33.2% (165/497)")
    print(f"  simulator quarter-end %   = {n_mM / T_sim:.1%}")

    print("\n" + "=" * 64)
    print("apply_missing_pattern  —  all self-test checks passed")
    print("=" * 64)


    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║   WRAPPER — simulate_dfm self-test                                   ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 64)
    print("simulate_dfm — end-to-end wrapper")
    print("=" * 64)

    # Build the theta dict expected by simulate_dfm from the archive.
    theta_star: dict = {
        "A":      A_star,
        "Q":      Q_star,
        "nu_u":   nu_u_star,
        "Lambda": Lambda_star,
        "R":      R_star,
        "nu_eps": nu_eps_star,
    }

    # ── 20. Wrapper at T = 497 (real-panel length) ────────────────────────────
    print(f"\nWrapper run at T=497 (same length as the real panel)")
    sim_497 = simulate_dfm(
        theta=theta_star, T=497,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2026,
    )
    assert sim_497["Y"].shape          == (497, M_panel)
    assert sim_497["F"].shape          == (497, 3)
    assert sim_497["w_u_true"].shape   == (497,)
    assert sim_497["w_eps_true"].shape == (497,)
    assert sim_497["Y_complete"].shape == (497, M_panel)
    assert set(sim_497["theta_used"].keys()) >= {
        "A", "Q", "nu_u", "Lambda", "R", "nu_eps"
    }
    n_gdp_497 = int(np.isfinite(sim_497["Y"][:, gdp_col]).sum())
    print(f"  Y shape         = {sim_497['Y'].shape}")
    print(f"  F shape         = {sim_497['F'].shape}")
    print(f"  w_u_true shape  = {sim_497['w_u_true'].shape}")
    print(f"  w_eps_true shape= {sim_497['w_eps_true'].shape}")
    print(f"  GDP non-NaN     = {n_gdp_497}  (real panel: 165, expected here 164 "
          f"— first quarter-end t=2 falls in the MM boundary t<4)")
    print(f"  theta_used keys = {sorted(sim_497['theta_used'].keys())}")
    # Expected: 165 quarter-ends in [0, 496] minus 1 because t=2 falls in the
    # MM boundary [t < 4] where the simulator set the GDP row to NaN.  The
    # real panel does not have this 1-obs deficit because its preprocessing
    # uses data from pre-1985 to populate the MM-aggregated GDP at t=2.
    assert n_gdp_497 == 164, (
        f"expected exactly 164 GDP obs at T=497 (165 quarter-ends - 1 MM "
        f"boundary at t=2), got {n_gdp_497}"
    )

    # ── 21. Wrapper at T = 2000 ──────────────────────────────────────────────
    print(f"\nWrapper run at T=2000")
    sim_2000 = simulate_dfm(
        theta=theta_star, T=2000,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2027,
    )
    assert sim_2000["Y"].shape          == (2000, M_panel)
    assert sim_2000["F"].shape          == (2000, 3)
    assert sim_2000["w_u_true"].shape   == (2000,)
    assert sim_2000["w_eps_true"].shape == (2000,)
    print(f"  Y shape         = {sim_2000['Y'].shape}")
    print(f"  F shape         = {sim_2000['F'].shape}")

    # ── 22. Independence of factor and observation RNGs ──────────────────────
    # Changing only the master seed should change *both* F and w_eps_true; the
    # factor and observation streams are decoupled by the seed+1 offset, so a
    # different master seed produces different ground truth across the board.
    sim_alt = simulate_dfm(
        theta=theta_star, T=200,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2028,
    )
    sim_ref = simulate_dfm(
        theta=theta_star, T=200,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2029,
    )
    assert not np.allclose(sim_alt["F"], sim_ref["F"])
    assert not np.allclose(sim_alt["w_eps_true"], sim_ref["w_eps_true"])
    print(f"\n[OK] master seed changes propagate to both F and w_eps_true")

    # ── 23. Reproducibility check ────────────────────────────────────────────
    sim_a = simulate_dfm(
        theta=theta_star, T=200,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2030,
    )
    sim_b = simulate_dfm(
        theta=theta_star, T=200,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=2030,
    )
    assert np.array_equal(sim_a["F"], sim_b["F"])
    assert np.array_equal(sim_a["w_u_true"], sim_b["w_u_true"])
    assert np.array_equal(sim_a["w_eps_true"], sim_b["w_eps_true"])
    # Y comparison must handle NaN: use np.array_equal with equal_nan=True.
    assert np.array_equal(sim_a["Y"], sim_b["Y"], equal_nan=True)
    print(f"[OK] same master seed -> bit-exact identical output")

    print("\n" + "=" * 64)
    print("simulate_dfm wrapper  —  all self-test checks passed")
    print("=" * 64)


    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║   TASK 4 — apply_contamination / contamination overlay self-test     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 64)
    print("Task 4 — contamination mechanism (Experiment C)")
    print("=" * 64)

    SEED_C = 4242
    # Mirror the simulate_observations gaussian-limit branch when rebuilding the
    # baseline draws (nu_eps_star is finite ~4.39 in practice, so the gamma
    # branch is taken; the guard keeps the reconstruction correct regardless).
    _gauss_eps = np.isinf(nu_eps_star) or nu_eps_star > _NU_GAUSSIAN_THRESHOLD
    LamM = Lambda_star[m_idx, :]                          # monthly-row loadings

    def _eps_monthly(obs_dict) -> np.ndarray:
        """Recover the EXACT idiosyncratic shocks on monthly rows: for monthly
        series mu = Lambda @ F (no MM), so eps = Y_complete - F @ Lambda^T."""
        return obs_dict["Y_complete"][:, m_idx] - F @ LamM.T

    # ── 1. pi = 0  =>  BIT-IDENTICAL to the contamination-free simulator ─────
    print("\n[1] pi = 0 : bit-identity + all-False mask")
    obs_pi0 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=0.0,
    )
    assert "contam_mask" in obs_pi0, "contam_mask missing from return dict"
    assert obs_pi0["contam_mask"].shape == (T_sim,), "contam_mask wrong shape"
    assert obs_pi0["contam_mask"].dtype == bool, "contam_mask must be bool"
    assert not obs_pi0["contam_mask"].any(), "pi=0 must give an all-False mask"
    print(f"    contam_mask: shape {obs_pi0['contam_mask'].shape}, "
          f"dtype {obs_pi0['contam_mask'].dtype}, sum = "
          f"{int(obs_pi0['contam_mask'].sum())}  (expected 0)")

    # (1a) At pi=0 the output must NOT depend on the contamination seed — the
    #      contamination RNG is never consumed, so it cannot leak into Y.
    obs_pi0_b = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=0.0, contam_seed=12345,
    )
    obs_pi0_c = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=0.0, contam_seed=99999,
    )
    assert np.array_equal(obs_pi0_b["Y_complete"], obs_pi0_c["Y_complete"], equal_nan=True)
    assert np.array_equal(obs_pi0_b["w_eps_true"], obs_pi0_c["w_eps_true"])
    print(f"    [OK] pi=0 output independent of contam_seed (no RNG leakage)")

    # (1b) Reconstruct the baseline draws in the documented order (w_eps FIRST,
    #      then Z) on a fresh RNG seeded at SEED_C, and verify the pi=0 panel
    #      used EXACTLY those draws — bit-for-bit.  This pins the new codepath
    #      as a strict no-op at pi=0 (the main stream is neither shifted nor
    #      perturbed by the contamination machinery).
    rng_ref = np.random.default_rng(SEED_C)
    if _gauss_eps:
        w_ref = np.ones(T_sim)
    else:
        w_ref = rng_ref.gamma(nu_eps_star / 2.0, 2.0 / nu_eps_star, size=T_sim)
    Z_ref   = rng_ref.standard_normal(size=(T_sim, M_panel))
    eps_ref = Z_ref * np.sqrt(R_star)[None, :] / np.sqrt(w_ref)[:, None]
    assert np.array_equal(w_ref, obs_pi0["w_eps_true"]), \
        "baseline w_eps draw order/stream changed by the new code"
    # Rebuild the monthly panel by ADDING the reference signal back, rather than
    # recovering eps by SUBTRACTING it off Y (as _eps_monthly does).  The
    # subtractive form computes (mu + eps) - mu, which is NOT bit-identical to
    # eps in floating point (catastrophic cancellation at the ~1e-16 level, so
    # np.array_equal would spuriously fail even though the DGP is a strict
    # no-op).  The simulator builds Y_complete[:, monthly] as the elementwise
    # sum mu_monthly + eps_monthly, and addition commutes with column slicing,
    # so reconstructing that same sum here reproduces Y_complete bit-for-bit —
    # a genuine "main RNG stream not shifted/perturbed" check.
    Y_ref_monthly = F @ LamM.T + eps_ref[:, m_idx]
    assert np.array_equal(obs_pi0["Y_complete"][:, m_idx], Y_ref_monthly), \
        "pi=0 panel is NOT bit-identical to the baseline draws (main RNG stream shifted)"
    print(f"    [OK] pi=0 panel is bit-identical to the contamination-free DGP")

    # ── 2. pi = 0.05, T large : contaminated fraction ~ pi ───────────────────
    print("\n[2] pi = 0.05 : contaminated fraction within binomial tolerance")
    PI = 0.05
    obs_c = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI,                       # nu_contam=3, kappa=5 (defaults)
    )
    mask  = obs_c["contam_mask"]
    n_c   = int(mask.sum())
    frac  = n_c / T_sim
    se    = np.sqrt(PI * (1.0 - PI) / T_sim)           # binomial SE of the fraction
    print(f"    contaminated periods = {n_c} / {T_sim}  (fraction {frac:.4f}, "
          f"target {PI}, ±5·SE = ±{5 * se:.4f})")
    assert abs(frac - PI) < 5.0 * se, (
        f"contaminated fraction {frac:.4f} more than 5 binomial SE from {PI}"
    )
    print(f"    [OK] contaminated fraction ~ pi")

    # ── 3. Outliers are VISIBLE : |eps| much larger at contaminated periods ──
    print("\n[3] outliers visible : |eps[contam]| >> |eps[clean]|")
    eps_c_m  = _eps_monthly(obs_c)                      # (T, n_month) exact eps
    mag      = np.abs(eps_c_m).mean(axis=1)            # per-period mean |eps|
    mean_con = float(mag[mask].mean())
    mean_cln = float(mag[~mask].mean())
    ratio    = mean_con / mean_cln
    print(f"    mean |eps| contaminated = {mean_con:.4f}")
    print(f"    mean |eps| clean        = {mean_cln:.4f}")
    print(f"    ratio                   = {ratio:.2f}x   "
          f"(expected ~5-6 at kappa=5, nu_contam=3)")
    assert ratio > 3.0, (
        f"contaminated periods only {ratio:.2f}x larger — outliers not visible"
    )
    print(f"    [OK] contaminated periods carry markedly larger idiosyncratic shocks")

    # ── 4. kappa scales the effect : kappa=5 outliers larger than kappa=3 ────
    print("\n[4] kappa scales the outlier magnitude")
    obs_k3 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI, kappa=3.0,
    )
    obs_k5 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI, kappa=5.0,
    )
    # Same seed (hence same contam_seed, same Bernoulli mask, same w_contam and
    # same Gaussian seeds): kappa enters ONLY as a multiplicative scale, so the
    # contaminated rows must satisfy eps(k=5) = (5/3) * eps(k=3) EXACTLY, and
    # the clean rows must be untouched and identical.
    mask_k = obs_k5["contam_mask"]
    assert np.array_equal(obs_k3["contam_mask"], mask_k), \
        "kappa must not change which periods are contaminated"
    eps_k3 = _eps_monthly(obs_k3)
    eps_k5 = _eps_monthly(obs_k5)
    assert np.allclose(eps_k5[mask_k], (5.0 / 3.0) * eps_k3[mask_k]), \
        "contaminated shocks do not scale linearly with kappa"
    assert np.array_equal(eps_k5[~mask_k], eps_k3[~mask_k]), \
        "clean periods must be identical across kappa (baseline untouched)"
    mag_k3 = float(np.abs(eps_k3[mask_k]).mean())
    mag_k5 = float(np.abs(eps_k5[mask_k]).mean())
    print(f"    mean |eps[contam]| at kappa=3 = {mag_k3:.4f}")
    print(f"    mean |eps[contam]| at kappa=5 = {mag_k5:.4f}")
    print(f"    measured ratio = {mag_k5 / mag_k3:.4f}   (exact 5/3 = "
          f"{5.0 / 3.0:.4f})")
    assert mag_k5 > mag_k3, "kappa=5 outliers should be larger than kappa=3"
    print(f"    [OK] outlier magnitude scales exactly with kappa")

    # ── 5. Reproducibility : same seed -> same mask & data; diff seed -> diff ─
    print("\n[5] reproducibility")
    obs_r1 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI,
    )
    obs_r2 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI,
    )
    assert np.array_equal(obs_r1["contam_mask"], obs_r2["contam_mask"])
    assert np.array_equal(obs_r1["Y_complete"], obs_r2["Y_complete"], equal_nan=True)
    obs_diff = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C + 777, pi=PI,
    )
    assert not np.array_equal(obs_r1["contam_mask"], obs_diff["contam_mask"]), \
        "a different master seed should produce a different contamination mask"
    print(f"    [OK] same seed -> identical mask & data; different seed -> different mask")

    # (5a) Baseline / contamination RNG isolation: holding the baseline seed
    #      fixed but changing contam_seed leaves w_eps_true untouched while the
    #      mask changes — proof the two streams are genuinely disjoint.
    obs_cs1 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI, contam_seed=1,
    )
    obs_cs2 = simulate_observations(
        F=F, Lambda=Lambda_star, R=R_star, nu_eps=nu_eps_star,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_C, pi=PI, contam_seed=2,
    )
    assert np.array_equal(obs_cs1["w_eps_true"], obs_cs2["w_eps_true"]), \
        "baseline w_eps must not depend on the contamination seed"
    assert not np.array_equal(obs_cs1["contam_mask"], obs_cs2["contam_mask"]), \
        "contam_seed must drive the contamination mask"
    print(f"    [OK] baseline and contamination RNG streams are isolated")

    # ── 6. The FACTOR is never contaminated (wrapper level) ──────────────────
    print("\n[6] factor untouched by contamination (wrapper level)")
    SEED_W = 3030
    sim_pi0 = simulate_dfm(
        theta=theta_star, T=500,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_W, pi=0.0,
    )
    sim_pic = simulate_dfm(
        theta=theta_star, T=500,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_W, pi=0.10,
    )
    assert np.array_equal(sim_pi0["F"], sim_pic["F"]), \
        "the factor path must be identical with and without contamination"
    assert np.array_equal(sim_pi0["w_u_true"], sim_pic["w_u_true"]), \
        "factor-innovation weights must be untouched by contamination"
    assert np.array_equal(sim_pi0["w_eps_true"], sim_pic["w_eps_true"]), \
        "baseline idiosyncratic weights must be untouched by contamination"
    assert int(sim_pi0["contam_mask"].sum()) == 0, "pi=0 wrapper mask must be empty"
    assert int(sim_pic["contam_mask"].sum()) > 0,  "pi=0.10 should contaminate some periods"
    assert not np.array_equal(sim_pi0["Y"], sim_pic["Y"], equal_nan=True), \
        "the contaminated panel should differ from the clean one"
    print(f"    F, w_u_true, w_eps_true identical across pi=0 and pi=0.10")
    print(f"    contaminated periods at pi=0.10: {int(sim_pic['contam_mask'].sum())} / 500")

    # (6a) The pi=0 wrapper panel must be invariant to the contamination params
    #      themselves (kappa, nu_contam) — they only matter once pi>0.
    sim_pi0_b = simulate_dfm(
        theta=theta_star, T=500,
        freq_list=freq_list, block_map=BLOCK, ordered_cols=ORDERED_COLS,
        r=r, seed=SEED_W, pi=0.0, kappa=99.0, nu_contam=2.0,
    )
    assert np.array_equal(sim_pi0["Y"], sim_pi0_b["Y"], equal_nan=True), \
        "pi=0 panel must not depend on kappa / nu_contam"
    print(f"    [OK] factor untouched; pi=0 panel invariant to kappa / nu_contam")

    print("\n" + "=" * 64)
    print("apply_contamination  —  all self-test checks passed")
    print("=" * 64)
