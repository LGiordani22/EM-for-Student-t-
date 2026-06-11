"""
src/em_m_step.py

Student-t M-step machinery for the mixed-frequency Dynamic Factor Model.

This module implements the parameter updates of the EM loop.  At outer
iteration j, the E-step (:mod:`em_e_step`) produces the converged
smoothed factor moments and Student-t mixing weights; the M-step uses
them to compute the weighted sufficient statistics from which the next
iterate ``theta^(j+1) = (A, Q, Lambda, R, nu_eps, nu_u)`` is derived in
closed form (or, for the nu's, via a one-dimensional root-finding step).

Thesis reference
----------------
EM_for_student_t.tex:
  - Section "The M-Step in the Unified Model" (subsec:full-m-step,
    line ~9265) — block-by-block M-step in the mixed-frequency unified
    model with Student-t down-weighting.
  - Section "Update of the VAR Transition Matrix A"
    (subsec:update-A, line ~5669) — full derivation of the A-update
    that yields A^(j+1) = P10^u (P00^u)^{-1}.  This derivation is the
    prototype for the analogous updates of Lambda, Q and R.
  - eq:Pu-definitions (line ~5714) — boxed definitions of the three
    weighted second-moment matrices P11^u, P10^u, P00^u that
    constitute the sufficient statistics for the (A, Q) block.

TASK 1 — weighted factor second moments  (this file):
  - compute_weighted_moments(f_smooth, P_smooth, P_lag, w_u, r)
      -> dict with keys ``P00`` (r,r), ``P10`` (r,r), ``P11`` (r,r).
    These are the sufficient statistics for the closed-form update of
    the VAR transition matrix A and the innovation covariance Q.

TASK 2 — block-restricted, mixed-frequency Lambda update  (this file):
  - update_Lambda(Y, f_smooth, P_smooth, w_eps, W_list,
                  block_map, freq_list, ordered_cols, r)
      -> Lambda_new of shape (M, r), block-diagonal by construction.
    For each row i this is a *scalar* weighted OLS of y_{i,t} on the
    regressor of its economic block (f^k_t for monthly series, the MM
    composite phi^k_t for quarterly series).

TASK 5 — degrees-of-freedom update via Brent root-finding  (this file):
  - update_nu(w_bar, log_w_bar, nu_bounds=(2.1, 200.0))
      -> nu_new (float).  Identical functional form for nu_u and
    nu_eps: only the posterior summaries differ.  See thesis
    eq:foc-nu-u (line ~6464) and eq:foc-nu-eps-summary (line ~6671).

TASK 6 — high-level M-step wrapper  (this file):
  - run_m_step(Y, e_step_output, theta_old, ...)
      -> theta_new (dict).  Orchestrates Tasks 1–5 in the correct ECM
    order (sequential conditional maximisation):
      1) observation pair  :  Lambda  ->  R   (R uses Lambda^(j+1))
      2) transition pair   :  A       ->  Q   (Q uses A^(j+1))
      3) degrees of freedom: nu_u and nu_eps  (independent ECME steps;
                                                optional freeze for the
                                                first ``freeze_nu_iters``
                                                outer iterations).
    Sigma_0 is *not* updated by the unified M-step (thesis
    subsec:full-m-step lists only updates (a)–(e)); it is carried
    forward from ``theta_old``.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma


# ─── 1. Weighted second moments of the monthly factors ───────────────────────

def compute_weighted_moments(
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    P_lag: np.ndarray,
    w_u: np.ndarray,
    r: int,
) -> dict[str, np.ndarray]:
    r"""
    Compute the three weighted posterior second-moment matrices of the
    monthly latent factors that constitute the sufficient statistics for
    the closed-form M-step update of A and Q.

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - eq:Pu-definitions (line ~5714), boxed definitions of P11^u,
        P10^u and P00^u as the *weighted* posterior moments that enter
        the trace-form objective of the state-transition block.
      - Section "The M-Step in the Unified Model" (line ~9279), where
        the same sums are written in compact form
        :math:`\mathcal{P}_{ab}^u = \sum_t \hat{w}^u_t\,
        \mathbb{E}[f_{t-a} f_{t-b}' \mid \mathbf{Y}]`.
      - Posterior second-moment identity (line ~5703-5706):
        :math:`\mathbb{E}[f_a f_b' \mid \mathbf{Y}] =
        \mathrm{Cov}(f_a, f_b \mid \mathbf{Y}) +
        \hat{f}_{a \mid T}\, \hat{f}_{b \mid T}'`.

    Parameters
    ----------
    f_smooth : np.ndarray, shape (T, 5r)
        Smoothed augmented state means f_{t|T} from
        :func:`kalman.kalman_smoother`.  Block 0 (columns ``0:r``) is the
        contemporaneous monthly factor; block 1 (columns ``r:2r``) is
        the one-lag factor encoded inside the augmented state at time t.
    P_smooth : np.ndarray, shape (T, 5r, 5r)
        Smoothed augmented state covariances P_{t|T} from the same
        routine.  Diagonal blocks contain marginal covariances of each
        lagged factor; off-diagonal blocks encode lag cross-covariances
        within the same time t.
    P_lag : np.ndarray, shape (T, 5r, 5r)
        Smoothed lag-one cross-covariance P_{t, t-1 | T}.  Returned by
        the smoother for completeness; in the production code path of
        this routine the cross moment is obtained from the *internal*
        lag-block of ``P_smooth[t]`` (see below) and ``P_lag`` is unused.
        Kept in the signature so callers (e.g. the validation test) can
        cross-check the alternative form.
    w_u : np.ndarray, shape (T,)
        Posterior mean of the factor-side Student-t weights, from the
        E-step (:func:`em_e_step.compute_weights`).  ``w_u[0] = 1`` by
        convention (prior mean — no factor innovation at t=0); this
        entry does not enter any sum below because the t=0 term is
        excluded.
    r : int
        Number of monthly latent factors (= 3 in this project).

    Returns
    -------
    dict with keys
        ``P00`` : np.ndarray (r, r)
            :math:`\mathcal{P}_{00}^u = \sum_{t=1}^{T-1} \hat{w}^u_t\,
            \mathbb{E}[f_{t-1} f_{t-1}' \mid \mathbf{Y}]`.
            Symmetric, positive-definite.
        ``P10`` : np.ndarray (r, r)
            :math:`\mathcal{P}_{10}^u = \sum_{t=1}^{T-1} \hat{w}^u_t\,
            \mathbb{E}[f_t f_{t-1}' \mid \mathbf{Y}]`.
            NOT symmetric in general (cross-time moment).
        ``P11`` : np.ndarray (r, r)
            :math:`\mathcal{P}_{11}^u = \sum_{t=1}^{T-1} \hat{w}^u_t\,
            \mathbb{E}[f_t f_t' \mid \mathbf{Y}]`.
            Symmetric, positive-definite.

    Notes
    -----
    **Role in the M-step.**
    These three matrices are the *only* sufficient statistics for the
    closed-form M-step updates of A and Q (thesis eq:A-update,
    line ~5814):

    .. math::

        \mathbf{A}^{(j+1)} \;=\; \mathcal{P}_{10}^u\, (\mathcal{P}_{00}^u)^{-1},
        \qquad
        \mathbf{Q}^{(j+1)} \;=\; \tfrac{1}{T}\Big(\mathcal{P}_{11}^u - \mathbf{A}^{(j+1)}\, (\mathcal{P}_{10}^u)'\Big).

    Everything else in the (A, Q) block — the trace identity in the
    Q-function, the matrix-calculus derivation, the first-order
    condition — collapses onto these three sums.  Computing them once
    in this routine is therefore the natural seam between E-step output
    (per-t smoothed moments + weights) and M-step output (parameter
    updates).

    **Posterior second-moment identities (Banbura-Modugno 2014 §3.2,
    thesis line ~5703-5706).**
    For any two random vectors x, y and any conditioning sigma-algebra
    that makes them measurable, the second moment decomposes into
    posterior covariance plus the outer product of posterior means:

    .. math::

        \mathbb{E}[x y' \mid \mathbf{Y}] \;=\;
        \mathrm{Cov}(x, y \mid \mathbf{Y}) \;+\;
        \mathbb{E}[x \mid \mathbf{Y}]\, \mathbb{E}[y \mid \mathbf{Y}]'.

    Applied with x = f_t, y = f_t (or x = f_{t-1}, y = f_{t-1}; or
    x = f_t, y = f_{t-1}) this yields the per-t contributions assembled
    below.  The covariance term captures *posterior uncertainty* about
    the factor (P_smooth blocks); the outer-product term captures the
    *point estimate* of the moment.  Both are needed: dropping the
    covariance term would underestimate the true posterior expectation
    of the squared residual and bias Q upward — this is the EM analogue
    of the "regression to the mean" correction.

    **Extraction of monthly blocks from the augmented state
    (consistency with the E-step).**
    The smoother returns the *augmented* state
    :math:`\tilde{f}_t = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4})`.
    The two monthly sub-states needed here are extracted as follows:

    - ``f_t      = f_smooth[t][0:r]``         — contemporaneous block
    - ``f_{t-1}  = f_smooth[t][r:2r]``        — internal one-lag block
                                              of the augmented state at
                                              the SAME time t
    - ``Var(f_t)         = P_smooth[t][0:r, 0:r]``
    - ``Var(f_{t-1})     = P_smooth[t][r:2r, r:2r]``
    - ``Cov(f_t, f_{t-1}) = P_smooth[t][0:r, r:2r]``

    This is the *internal-block* approach: every quantity at time t is
    read off the augmented smoothed moments at the same index t, with
    no recourse to P_lag and no need to access the smoothed state at
    t-1.  It is the same approach used in
    :func:`em_e_step._compute_d_u_internal_blocks`, and was already
    verified to be numerically identical to the P_lag form at machine
    precision in the E-step test suite.  The companion sanity check in
    the ``__main__`` block of this file repeats that verification for
    the M-step second moments, confirming that the augmented
    state-space encodes the lag structure of the monthly factors
    consistently.

    **Why the internal-block form is preferred here.**

    1. It accesses the augmented state at a single time t per iteration,
       which is conceptually cleaner and avoids any indexing edge cases
       near the sample boundary.
    2. It does not depend on the lag-one smoothed covariance P_lag,
       which is the most algebraically intricate output of the RTS
       smoother (built from the Kalman gain at t-1 and the smoother
       gain at t).  Decoupling the M-step from P_lag isolates any
       smoother bug in the E-step where it can be caught by the
       internal-block-vs-P_lag consistency test, rather than
       contaminating the parameter updates.
    3. Performance: skipping P_lag avoids one (5r)x(5r) matrix read
       per t — negligible here but architecturally cleaner.

    **Time-index convention (sum over t = 1, ..., T-1).**
    The thesis sums run from math-index t = 1 to T, with the
    convention that f_0 is the initial state and is treated diffusely
    via Sigma_0.  In Python (0-indexed) the corresponding range is
    ``range(1, T)``: at each t in this range we read both f_t (block 0
    of the augmented state at time t) and f_{t-1} (block 1 of the same
    augmented state at time t).  The boundary t = 0 carries no
    f_{-1} and is excluded; ``w_u[0] = 1`` by convention but does not
    enter any sum.  This is the same convention used by
    :func:`em_e_step.compute_d_u`, which sets ``d_u[0] = NaN``.

    **Role of the weights w_u.**
    Each per-t contribution is multiplied by the posterior mean of the
    factor-side mixing weight ``w_u[t]``.  Months with a large
    Mahalanobis residual d_u_t (i.e. months in which the factors moved
    in a way poorly predicted by their own VAR dynamics — *factor
    outliers*) receive a small ``w_u[t]`` and contribute proportionally
    less to the sums.  This is exactly the robust-weighting mechanism
    that distinguishes the Student-t DFM from the Gaussian
    Banbura-Modugno model: in the latter, the analogous unweighted
    sums treat every month equally and a few extreme innovation events
    can distort the entire (A, Q) estimate.

    **Symmetry, positive-definiteness, finiteness.**
    Both P11 and P00 are explicit weighted sums of *symmetric positive
    semi-definite* matrices (covariance block plus outer product), with
    *strictly positive* weights (w_u > 0 by construction in the
    E-step).  As long as at least one term is positive definite — which
    is the case as soon as one smoothed covariance is PD, generically
    the case for T >> r — the sums are themselves symmetric
    positive-definite.  P10 is the cross-time sum and is NOT symmetric
    in general (this is expected: it encodes the directional one-step
    cross-covariance).  These properties are asserted in the self-test.

    **Vectorisation.**
    The three sums are accumulated in a single Python loop over
    ``range(1, T)``.  Vectorising over t would require pre-extracting
    three (T, r) means and three (T, r, r) covariance blocks via fancy
    indexing, then summing with ``np.einsum``.  This is left as a
    future optimisation: T = 497 here, so the loop runs in a few
    milliseconds and the readable form is preferred.
    """
    T = f_smooth.shape[0]

    P00 = np.zeros((r, r))
    P10 = np.zeros((r, r))
    P11 = np.zeros((r, r))

    for t in range(1, T):
        # ── Monthly factor means at time t (contemporaneous + internal lag) ──
        f_t   = f_smooth[t][0:r]          # (r,)        E[f_t     | Y]
        f_tm1 = f_smooth[t][r:2 * r]      # (r,)        E[f_{t-1} | Y]

        # ── Posterior covariance blocks read off P_smooth[t] ──────────────────
        cov_tt     = P_smooth[t][0:r,       0:r]        # Var(f_t)
        cov_tm1tm1 = P_smooth[t][r:2 * r,   r:2 * r]    # Var(f_{t-1})
        cov_t_tm1  = P_smooth[t][0:r,       r:2 * r]    # Cov(f_t, f_{t-1})

        w = w_u[t]

        # ── Second-moment contributions  E[f_a f_b' | Y] = Cov + outer ────────
        P11 += w * (cov_tt     + np.outer(f_t,   f_t))
        P00 += w * (cov_tm1tm1 + np.outer(f_tm1, f_tm1))
        P10 += w * (cov_t_tm1  + np.outer(f_t,   f_tm1))

    return {"P00": P00, "P10": P10, "P11": P11}


# ─── 2. Block-restricted, mixed-frequency Lambda update ──────────────────────

# Canonical block ordering used throughout the project.  Must match the
# convention set by ``em_initialization.pca_initialization`` and
# ``em_initialization.compute_theta_initial``: column j of every factor
# matrix corresponds to the block at position j of this list.
_BLOCK_ORDER: list[str] = ["real", "financial", "other"]
_BLOCK_TO_COL: dict[str, int] = {b: j for j, b in enumerate(_BLOCK_ORDER)}

# Mariano-Murasawa aggregation weights c_l for l = 0, ..., 4.
# Applied to (f^k_t, f^k_{t-1}, f^k_{t-2}, f^k_{t-3}, f^k_{t-4}) to form the
# composite regressor phi^k_t used in the quarterly observation equation.
_MM_WEIGHTS: np.ndarray = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])


def update_Lambda(
    Y: np.ndarray,
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    w_eps: np.ndarray,
    W_list: list[np.ndarray],
    block_map: dict[str, str],
    freq_list: list[str],
    ordered_cols: list[str],
    r: int,
) -> np.ndarray:
    r"""
    Block-restricted, mixed-frequency M-step update of the loading matrix
    :math:`\mathbf{\Lambda}` (monthly + quarterly rows in a single pass).

    Each row of :math:`\mathbf{\Lambda}` is updated independently as a
    *scalar* weighted OLS regression of the corresponding series on the
    one factor it loads on (block restriction).  Monthly rows regress on
    the contemporaneous block-factor :math:`f^k_t`; quarterly rows
    regress on the Mariano-Murasawa composite block-factor
    :math:`\phi^k_t` (a fixed linear combination of the block factor at
    the current month and at four lags).

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - Section "The M-Step in the Unified Model", paragraphs (b) and
        (c) (line ~9281-9296): the row-by-row Lambda updates for the
        monthly and quarterly rows in the unified block-restricted,
        mixed-frequency setting.  Equation (b) gives the monthly update
        with regressor :math:`f^k_t`; equation (c) gives the quarterly
        update with composite regressor
        :math:`\phi^k_t = \tfrac{1}{3} f^k_t + \tfrac{2}{3} f^k_{t-1}
        + f^k_{t-2} + \tfrac{2}{3} f^k_{t-3} + \tfrac{1}{3} f^k_{t-4}`.
      - Section "M-Step with Block Restrictions" (subsec:block-mstep,
        line ~8310): the block-diagonal exclusion restrictions collapse
        the multivariate OLS of each row of :math:`\mathbf{\Lambda}`
        into a one-dimensional scalar weighted OLS on the single factor
        of the row's economic block.  This is the structural reason why
        the update below loops over rows and writes into a single
        entry per row.
      - Section "Update of the Factor Loading Matrix" (subsec:update-Lambda,
        line ~5854): full derivation of the unrestricted closed-form
        Lambda update — the prototype that, after applying block and
        missing-data restrictions, reduces to the row-by-row scalar OLS
        used here.  In particular, the appearance of :math:`\hat{w}^\varepsilon_t`
        as the multiplicative weight in every sum (in place of the
        Gaussian unit weights) is the Student-t down-weighting
        mechanism inherited unchanged from that derivation.
      - Section "(iii) Update of L^Q: the composite-regressor form"
        (line ~7935): the quarterly observation equation
        :math:`y^Q_{j,t} = \mathbf{\Lambda}^Q_{j\cdot} \phi_t +
        \varepsilon^Q_{j,t}` is *formally identical* to the monthly one
        except that :math:`f_t` is replaced by :math:`\phi_t`; the
        M-step therefore inherits the same weighted-OLS structure with
        :math:`\phi_t` in place of :math:`f_t`.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Mixed-frequency observation panel with ``NaN`` for missing
        entries.  For monthly series, NaN appears at the ragged edge
        (publication lag).  For quarterly series, NaN appears at all
        non-quarter-end months by construction.  This NaN pattern
        defines the observation set :math:`\mathcal{T}_i` of each row.
    f_smooth : np.ndarray, shape (T, 5r)
        Smoothed augmented state means
        :math:`\hat{\tilde{f}}_{t \mid T}` from the E-step.  The
        augmented state stacks five consecutive lags of the monthly
        factors: block ``l`` (columns ``l*r:(l+1)*r``) carries the
        contemporaneous-at-t image of :math:`f_{t-l}`.  In particular,
        the contemporaneous block-factor :math:`f^k_t` is
        ``f_smooth[t, j]`` where ``j = block-to-column index`` and the
        lagged block-factor :math:`f^k_{t-l}` is ``f_smooth[t, l*r + j]``.
    P_smooth : np.ndarray, shape (T, 5r, 5r)
        Smoothed augmented state covariances
        :math:`\tilde{P}_{t \mid T}` from the E-step.  The 5x5
        sub-matrix on indices ``[j, r+j, 2r+j, 3r+j, 4r+j]`` of
        ``P_smooth[t]`` is the joint posterior covariance of the five
        lagged block-factors used to assemble :math:`\phi^k_t`.
    w_eps : np.ndarray, shape (T,)
        Posterior mean of the *idiosyncratic-side* Student-t weights
        :math:`\hat{w}^\varepsilon_t` from the converged inner ECM loop
        (E-step).  Months in which the panel is far from its factor-
        implied value in Mahalanobis sense (idiosyncratic outliers)
        receive a small weight and contribute proportionally less to
        each row's sums.  All weights are strictly positive.
    W_list : list of np.ndarray, length T
        Selection matrices ``W_list[t]`` of shape ``(m_t, M)`` from
        :func:`kalman.build_all_selection_matrices`.  Kept in the
        signature for API consistency with the E-step.  The
        observation set :math:`\mathcal{T}_i` for each row is read off
        the NaN pattern of ``Y`` directly (equivalent to projecting
        ``W_t`` onto column i and dropping zero rows), which is the
        more natural primitive for the row-by-row update.
    block_map : dict[str, str]
        Maps each series name to its economic block:
        ``"real"``, ``"financial"``, or ``"other"``.  Typically
        :data:`data_loader.BLOCK`.
    freq_list : list[str], length M
        Frequency tag (``"monthly"`` or ``"quarterly"``) of each
        series, *in the same order as the columns of* ``Y``
        (i.e. aligned with ``ordered_cols``).
    ordered_cols : list[str], length M
        Series names in the order of the columns of ``Y``.  Used
        together with ``block_map`` to map each row of Y to its
        block, and hence to the single column of Lambda that is
        allowed to be non-zero.
    r : int
        Number of monthly latent factors (= number of blocks = 3 in
        this project).

    Returns
    -------
    Lambda_new : np.ndarray, shape (M, r)
        Updated loading matrix.  By construction, row i has at most one
        non-zero entry, at column ``j = _BLOCK_TO_COL[block_map[ordered_cols[i]]]``;
        all other columns of row i are exactly zero.  No NaN, no inf.

    Notes
    -----
    **Block restriction collapses the row update to a scalar weighted OLS.**
    The unrestricted closed-form update (thesis eq:Lambda-update, line
    ~5947) is
    :math:`\mathbf{\Lambda}^{(j+1)} = \mathcal{S}^\varepsilon_{yf}\,
    (\mathcal{P}^\varepsilon_{11})^{-1}`, an :math:`M \times r`
    matrix obtained as the row-wise solution of an *r*-dimensional
    weighted OLS for each series.  Under the block-diagonal exclusion
    restrictions (Section 8, line ~8310) each row is allowed to load
    on a single factor :math:`f^k`, so :math:`r-1` columns are
    *constrained to zero* and never enter the optimisation.  The
    remaining one-dimensional unknown — the scalar
    :math:`\mathbf{\Lambda}^{k}_i` — is recovered by the closed-form
    scalar weighted OLS:

    .. math::

        \mathbf{\Lambda}^{M,k,(j+1)}_i \;=\;
        \frac{\sum_{t \in \mathcal{T}^M_i} \hat{w}^\varepsilon_t\,
              y^M_{i,t}\, \mathbb{E}[f^k_t \mid \mathbf{Y}]}{
              \sum_{t \in \mathcal{T}^M_i} \hat{w}^\varepsilon_t\,
              \mathbb{E}[(f^k_t)^2 \mid \mathbf{Y}]}

    for monthly series (thesis line ~9285), and the analogous
    formula with :math:`f^k_t` replaced by :math:`\phi^k_t` for
    quarterly series (thesis line ~9293).  No matrix inversion is
    required: the denominator is a positive scalar.

    **Composite regressor :math:`\phi^k_t` for quarterly series.**
    The MM identity (Section 7) writes the quarterly log-difference
    as a fixed linear combination of five consecutive monthly
    log-differences with weights :math:`(c_0, \ldots, c_4) =
    (\tfrac{1}{3}, \tfrac{2}{3}, 1, \tfrac{2}{3}, \tfrac{1}{3})`.
    Under the latent-monthly-factor representation, this becomes a
    linear combination of the contemporaneous and four lagged block-
    factors:

    .. math::

        \phi^k_t \;\equiv\; \tfrac{1}{3} f^k_t + \tfrac{2}{3} f^k_{t-1}
        + f^k_{t-2} + \tfrac{2}{3} f^k_{t-3} + \tfrac{1}{3} f^k_{t-4}.

    Inside the augmented state these five quantities are exactly the
    j-th entry of each of the five lag-blocks of
    ``f_smooth[t]``: ``f^k_{t-l} = f_smooth[t, l*r + j]`` for
    :math:`l = 0, \ldots, 4`.  The first and second posterior moments
    of :math:`\phi^k_t` are therefore read off the smoothed augmented
    moments at the same time index t (thesis eq:mm-Ephi /
    eq:mm-Ephiphi, line ~7980-7984):

    .. math::

        \mathbb{E}[\phi^k_t \mid \mathbf{Y}] \;&=\; \mathbf{c}'\,
        \hat{\tilde{f}}_{t \mid T}[\,\mathrm{idx}\,], \\[2pt]
        \mathrm{Var}(\phi^k_t \mid \mathbf{Y}) \;&=\; \mathbf{c}'\,
        \tilde{P}_{t \mid T}[\,\mathrm{idx},\mathrm{idx}\,]\, \mathbf{c}, \\[2pt]
        \mathbb{E}[(\phi^k_t)^2 \mid \mathbf{Y}] \;&=\;
        \mathrm{Var}(\phi^k_t \mid \mathbf{Y})
        + (\mathbb{E}[\phi^k_t \mid \mathbf{Y}])^2,

    where ``idx = [j, r+j, 2r+j, 3r+j, 4r+j]`` and
    :math:`\mathbf{c}` is the vector of MM weights.  The variance
    term is *essential*: dropping it would underestimate the true
    posterior second moment of the regressor and bias the loadings
    upward.  This is the same "posterior covariance correction" that
    appears in :func:`compute_weighted_moments` (the trace term in
    the Q-function).

    **Why monthly rows still pick up posterior uncertainty.**
    For monthly rows the formula
    :math:`\mathbb{E}[(f^k_t)^2 \mid \mathbf{Y}] =
    P_{t|T}[j,j] + (f_{t|T}[j])^2` is the same identity
    specialised to the contemporaneous block of the augmented state
    (lag :math:`l = 0` only), so no separate treatment is needed:
    ignoring the :math:`P_{t|T}[j,j]` term would similarly bias
    monthly loadings.

    **Time set :math:`\mathcal{T}_i` and missing data.**
    The observation set of each row is the set of times at which the
    corresponding entry of ``Y`` is non-NaN.  For monthly series this
    excludes the ragged edge; for the quarterly series (GDPC1) this
    automatically restricts the sums to quarter-end months *that have
    been released* (Q3 2025 ragged edge in the May 2026 vintage is
    naturally excluded).  The selection matrix :math:`\mathbf{W}_t`
    used in the E-step encodes the same information at the time-by-
    time level; here we use the row-by-row primitive (column NaN
    mask) because the row-by-row update treats each series in
    isolation.

    **Role of :math:`\hat{w}^\varepsilon_t`.**
    Every per-t term in both numerator and denominator is multiplied
    by :math:`\hat{w}^\varepsilon_t`.  Months in which the observed
    panel is far from its factor-implied value in Mahalanobis sense
    (idiosyncratic outliers — e.g. COVID April 2020 for the activity
    block) receive a small weight and contribute proportionally less.
    This is the Student-t robustification of the loading update: in
    the Gaussian limit :math:`\nu_\varepsilon \to \infty` we have
    :math:`\hat{w}^\varepsilon_t \to 1` and the formula collapses to
    the Bańbura-Modugno (2014) row-by-row OLS.

    **Numerical guards.**
    A denominator below a tolerance threshold indicates a degenerate
    situation (the block-factor is essentially zero throughout the
    observation set of the row, or all weights vanish).  This is not
    expected to occur with the present dataset (T = 497, observation
    sets of size ~165 for the quarterly row and ~330+ for the
    monthly rows), but we raise an explicit ``RuntimeError`` if it
    does, rather than silently producing inf/nan.
    """
    T, M = Y.shape
    if len(ordered_cols) != M:
        raise ValueError(
            f"ordered_cols has length {len(ordered_cols)} but Y has {M} columns."
        )
    if len(freq_list) != M:
        raise ValueError(
            f"freq_list has length {len(freq_list)} but Y has {M} columns."
        )

    Lambda_new = np.zeros((M, r))

    # Pre-build the augmented-state indices [j, r+j, 2r+j, 3r+j, 4r+j] for each
    # block column j.  This is the index pattern that extracts the j-th block-
    # factor across the five lags of the augmented state.
    quarterly_indices: dict[int, np.ndarray] = {
        j: np.array([l * r + j for l in range(5)]) for j in range(r)
    }

    eps_denom = 1e-12   # safety tolerance against degenerate sums

    for i in range(M):
        col   = ordered_cols[i]
        block = block_map[col]
        if block not in _BLOCK_TO_COL:
            raise KeyError(
                f"Series '{col}' has unknown block '{block}'. "
                f"Expected one of {_BLOCK_ORDER}."
            )
        j = _BLOCK_TO_COL[block]
        freq = freq_list[i]

        # T_i = set of times at which series i is observed (Y[t, i] not NaN).
        # For monthly series this excludes the ragged edge; for the quarterly
        # series this is naturally only the released quarter-end months.
        obs_mask = ~np.isnan(Y[:, i])
        obs_t    = np.where(obs_mask)[0]

        if obs_t.size == 0:
            # No observed entries — leave Lambda[i, j] = 0 (degenerate; would
            # mean the series carries no information about the loading).
            continue

        y_i  = Y[obs_t, i]            # (n_obs,)
        w_i  = w_eps[obs_t]           # (n_obs,)  posterior weights

        if freq == "monthly":
            # Regressor: contemporaneous block-factor f^k_t.
            # E[f^k_t | Y]       = f_smooth[t, j]
            # E[(f^k_t)^2 | Y]   = P_smooth[t, j, j] + f_smooth[t, j]^2
            E_f  = f_smooth[obs_t, j]                # (n_obs,)
            V_f  = P_smooth[obs_t, j, j]             # (n_obs,)
            E_f2 = V_f + E_f ** 2                    # (n_obs,)

            num = float(np.sum(w_i * y_i * E_f))
            den = float(np.sum(w_i * E_f2))

        elif freq == "quarterly":
            # Regressor: MM composite block-factor phi^k_t.
            # Extract the 5 lagged block-factors from the augmented state.
            idx = quarterly_indices[j]               # (5,)

            # Means: (n_obs, 5) -> (n_obs,) via mm-weighted sum
            f_block = f_smooth[obs_t][:, idx]        # (n_obs, 5)
            E_phi   = f_block @ _MM_WEIGHTS          # (n_obs,)

            # Covariance sub-matrices: (n_obs, 5, 5) -> (n_obs,) via c' P c
            # P_block[t] = P_smooth[obs_t[t]][np.ix_(idx, idx)]  (read out as
            # a 3-D advanced-indexing slice over the observed times).
            P_block = P_smooth[
                obs_t[:, None, None],
                idx[None, :, None],
                idx[None, None, :],
            ]                                         # (n_obs, 5, 5)
            V_phi   = np.einsum(
                "s,nsl,l->n", _MM_WEIGHTS, P_block, _MM_WEIGHTS
            )                                         # (n_obs,)
            E_phi2  = V_phi + E_phi ** 2              # (n_obs,)

            num = float(np.sum(w_i * y_i * E_phi))
            den = float(np.sum(w_i * E_phi2))

        else:
            raise ValueError(
                f"Series '{col}' has unknown freq '{freq}'. "
                "Expected 'monthly' or 'quarterly'."
            )

        if abs(den) < eps_denom:
            raise RuntimeError(
                f"Degenerate denominator for series '{col}' (block '{block}', "
                f"freq '{freq}'): sum = {den:.3e}.  The block-factor appears "
                f"to be (nearly) zero across the observation set, or all "
                f"posterior weights vanish."
            )

        Lambda_new[i, j] = num / den

    return Lambda_new


# ─── 3. Sequential update of the VAR pair (A, Q) ─────────────────────────────

def update_A_Q(
    P00: np.ndarray,
    P10: np.ndarray,
    P11: np.ndarray,
    T_eff: int | float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Closed-form M-step update of the VAR(1) transition matrix
    :math:`\mathbf{A}` and the factor-innovation scale matrix
    :math:`\mathbf{Q}`, performed *sequentially* (conditional
    maximisation: first :math:`\mathbf{A}`, then :math:`\mathbf{Q}`
    evaluated at the just-updated :math:`\mathbf{A}^{(j+1)}`).

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - Section "The M-Step in the Unified Model", paragraph (a)
        (line ~9271-9279): the compact statement of the joint
        :math:`(\mathbf{A}, \mathbf{Q})` update in the unified Student-t
        mixed-frequency model.  Reads, in our notation:

        .. math::

            \mathbf{A}^{(j+1)} \;=\; \mathcal{P}_{10}^{u}\,
                (\mathcal{P}_{00}^{u})^{-1}, \qquad
            \mathbf{Q}^{(j+1)} \;=\; \tfrac{1}{T}\,\Big(\mathcal{P}_{11}^{u}
                - \mathbf{A}^{(j+1)}\, (\mathcal{P}_{10}^{u})'\Big).

      - Section "Update of the VAR Transition Matrix A"
        (subsec:update-A, line ~5669-5852): full derivation of the
        :math:`\mathbf{A}` update from the first-order condition
        :math:`\mathbf{Q}^{-1}\, \mathcal{P}_{10}^{u} -
        \mathbf{Q}^{-1}\, \mathbf{A}\, \mathcal{P}_{00}^{u} = 0`, which
        — after left-multiplying by :math:`\mathbf{Q}` and right-
        multiplying by :math:`(\mathcal{P}_{00}^{u})^{-1}` — produces
        :math:`\mathbf{A}^{(j+1)} = \mathcal{P}_{10}^{u}
        (\mathcal{P}_{00}^{u})^{-1}` (eq:A-update, line ~5812).
      - Section "Sequential conditional maximisation" (line ~5652-5667):
        the explicit statement that :math:`\mathbf{A}` and :math:`\mathbf{Q}`
        are updated sequentially — first :math:`\mathbf{A}` with
        :math:`\mathbf{Q}^{(j)}` held fixed, then :math:`\mathbf{Q}`
        re-evaluated at :math:`\mathbf{A}^{(j+1)}`.  This is the ECM
        scheme of Meng-Rubin (1993); the two updates yield the same
        fixed point as a fully simultaneous maximisation but each step
        is a standard weighted-OLS problem.
      - Section "Update of the Factor Innovation Scale Matrix Q"
        (subsec:update-Q, line ~5984-6098): full derivation of the
        :math:`\mathbf{Q}` update.  Two key passages:
          * "Simplify the sum using the new A" (line ~6013-6037): the
            sum :math:`\mathcal{S}_{ff}^{u}(\mathbf{A}^{(j+1)})`
            collapses, *at the optimum in A*, to the compact form
            :math:`\mathcal{P}_{11}^{u} - \mathbf{A}^{(j+1)}
            (\mathcal{P}_{10}^{u})'` (eq:Sff-simplified, line ~6034).
            The two "cross-terms" cancel because of the first-order
            condition of the A-update.
          * "Set the gradient to zero and solve" (line ~6070-6098):
            the Magnus-Neudecker identities applied to the determinant
            and trace-with-inverse terms yield
            :math:`\mathbf{Q}^{(j+1)} = \tfrac{1}{T} C_{\mathbf{Q}}`
            (eq:Q-update, line ~6094).

    Parameters
    ----------
    P00 : np.ndarray, shape (r, r)
        Weighted posterior second moment :math:`\mathcal{P}_{00}^{u} =
        \sum_t \hat{w}^u_t\, \mathbb{E}[f_{t-1} f_{t-1}' \mid \mathbf{Y}]`
        from :func:`compute_weighted_moments`.  Symmetric positive
        definite.
    P10 : np.ndarray, shape (r, r)
        Weighted cross-time moment :math:`\mathcal{P}_{10}^{u} =
        \sum_t \hat{w}^u_t\, \mathbb{E}[f_t f_{t-1}' \mid \mathbf{Y}]`.
        NOT symmetric in general (encodes the directional one-step
        cross-covariance).
    P11 : np.ndarray, shape (r, r)
        Weighted posterior second moment :math:`\mathcal{P}_{11}^{u} =
        \sum_t \hat{w}^u_t\, \mathbb{E}[f_t f_t' \mid \mathbf{Y}]`.
        Symmetric positive definite.
    T_eff : int or float
        Effective number of transitions in the sums above.  In the
        thesis this is the "T" of :math:`-\tfrac{T}{2}\log|\mathbf{Q}|`
        in the log-likelihood: it counts the number of state-equation
        terms :math:`f_t = \mathbf{A} f_{t-1} + u_t` actually summed.
        In this project's Python convention, with ``T`` total time
        points indexed ``0..T-1`` and :func:`compute_weighted_moments`
        summing over ``range(1, T)``, this equals ``T - 1`` (one
        transition per consecutive pair).  Passed by the caller to
        keep this routine purely linear-algebraic and decoupled from
        the indexing convention of the smoother.

    Returns
    -------
    A_new : np.ndarray, shape (r, r)
        Updated VAR transition matrix
        :math:`\mathbf{A}^{(j+1)} = \mathcal{P}_{10}^{u}\,
        (\mathcal{P}_{00}^{u})^{-1}`.  NOT block-diagonal in general:
        unlike :math:`\mathbf{\Lambda}`, the transition matrix is left
        *unrestricted* so that the VAR can capture dynamic cross-block
        spillovers (e.g. financial conditions in :math:`t-1` driving
        real activity in :math:`t`).
    Q_new : np.ndarray, shape (r, r)
        Updated factor-innovation covariance
        :math:`\mathbf{Q}^{(j+1)} = \tfrac{1}{T_{\text{eff}}}\,
        (\mathcal{P}_{11}^{u} - \mathbf{A}^{(j+1)}\, (\mathcal{P}_{10}^{u})')`.
        Symmetrised after computation; should be positive definite at
        the optimum.  A diagnostic warning is printed if the smallest
        eigenvalue is non-positive (numerical edge case).

    Notes
    -----
    **Why sequential and not simultaneous.**
    A fully simultaneous maximisation of :math:`L(\mathbf{A}, \mathbf{Q})`
    is a coupled non-linear system (the optimal :math:`\mathbf{Q}`
    depends on :math:`\mathbf{A}` through the residuals, and the optimal
    :math:`\mathbf{A}` depends on :math:`\mathbf{Q}` through the
    weighting :math:`\mathbf{Q}^{-1}` inside the trace).  Updating
    sequentially — first :math:`\mathbf{A}` with :math:`\mathbf{Q}`
    held at :math:`\mathbf{Q}^{(j)}`, then :math:`\mathbf{Q}` at the
    new :math:`\mathbf{A}^{(j+1)}` — decouples the two problems: each
    becomes a closed-form weighted OLS.  This is the ECM scheme of
    Meng-Rubin (1993).  Two facts:

    1. The :math:`\mathbf{A}` first-order condition,
       :math:`\mathbf{Q}^{-1}\mathcal{P}_{10}^{u}
       = \mathbf{Q}^{-1}\mathbf{A}\mathcal{P}_{00}^{u}`, factors out
       :math:`\mathbf{Q}^{-1}` on the left of both sides.  Left-
       multiplying by :math:`\mathbf{Q}` kills the dependence on
       :math:`\mathbf{Q}` entirely, leaving
       :math:`\mathbf{A}\mathcal{P}_{00}^{u} = \mathcal{P}_{10}^{u}`.
       *This is why* the :math:`\mathbf{A}`-update does not depend on
       :math:`\mathbf{Q}` at all — the only thing the "current"
       :math:`\mathbf{Q}^{(j)}` controls is the trace objective's
       overall scale, not the location of its maximum in :math:`\mathbf{A}`.
       So holding :math:`\mathbf{Q}` fixed is harmless: any positive-
       definite :math:`\mathbf{Q}` gives the same :math:`\mathbf{A}^{(j+1)}`.
    2. The cross-terms in :math:`\mathcal{S}_{ff}^{u}(\mathbf{A}^{(j+1)})`
       cancel because of (1): substituting
       :math:`\mathbf{A}^{(j+1)} \mathcal{P}_{00}^{u} =
       \mathcal{P}_{10}^{u}` into
       :math:`\mathcal{P}_{11}^{u} - \mathcal{P}_{10}^{u}
       \mathbf{A}^{(j+1)\prime} - \mathbf{A}^{(j+1)}
       (\mathcal{P}_{10}^{u})' + \mathbf{A}^{(j+1)}
       \mathcal{P}_{00}^{u} \mathbf{A}^{(j+1)\prime}` collapses to
       :math:`\mathcal{P}_{11}^{u} - \mathbf{A}^{(j+1)}
       (\mathcal{P}_{10}^{u})'` (thesis eq:Sff-simplified, line ~6034).
       This is the form used below for :math:`\mathbf{Q}^{(j+1)}`.

    **Why** :math:`\mathbf{A}` **is not block-diagonal (unlike** :math:`\mathbf{\Lambda}` **).**
    The block-diagonal restriction on :math:`\mathbf{\Lambda}` encodes
    the *static* economic exclusion that each observed series loads on
    one factor only.  The transition matrix :math:`\mathbf{A}`,
    however, governs the *dynamic* propagation of factor innovations
    across blocks: it must remain dense so that financial-block
    innovations in :math:`t-1` are allowed to feed into the
    real-block factor at :math:`t` (and similarly for other cross-block
    spillover channels).  Restricting :math:`\mathbf{A}` to block-
    diagonal would force three statistically independent univariate
    AR(1) factor processes — a strong, economically implausible
    restriction.  Hence ``A_new`` typically has substantial off-diagonal
    entries.

    **Numerical stability of the** :math:`\mathbf{A}` **inverse.**
    Forming :math:`\mathbf{A}^{(j+1)} = \mathcal{P}_{10}^{u}
    (\mathcal{P}_{00}^{u})^{-1}` via an explicit ``np.linalg.inv``
    inverts a (possibly mildly ill-conditioned) :math:`r \times r`
    matrix.  We instead solve the equivalent linear system, which is
    typically more accurate and faster: writing
    :math:`\mathbf{A} \mathcal{P}_{00}^{u} = \mathcal{P}_{10}^{u}` and
    transposing gives :math:`\mathcal{P}_{00}^{u} \mathbf{A}' =
    (\mathcal{P}_{10}^{u})'`, so :math:`\mathbf{A}' =
    \mathrm{solve}(\mathcal{P}_{00}^{u}, (\mathcal{P}_{10}^{u})')`
    and :math:`\mathbf{A}` is its transpose.  For :math:`r = 3` this
    is microseconds either way, but the ``solve`` form is the safer
    primitive.

    **Symmetrisation of** :math:`\mathbf{Q}`.
    The product :math:`\mathbf{A}^{(j+1)} (\mathcal{P}_{10}^{u})'` is
    not symmetric on the nose (it equals
    :math:`\mathbf{A}^{(j+1)} \mathcal{P}_{00}^{u}
    \mathbf{A}^{(j+1)\prime}` at the exact optimum, but only up to
    floating-point error).  We symmetrise explicitly:
    :math:`\mathbf{Q}_{\text{sym}} = \tfrac{1}{2}(\mathbf{Q} + \mathbf{Q}')`.
    Without this, asymmetry of order ``1e-15`` propagates into the
    next E-step's :math:`\tilde{\mathbf{Q}}` and may trigger Cholesky
    failures in the Kalman covariance update.

    **Positive-definiteness check.**
    At a valid local optimum :math:`\mathbf{Q}^{(j+1)}` is positive
    definite (it is a weighted average of posterior-residual outer
    products).  We compute its eigenvalues post-symmetrisation and
    print a one-line diagnostic if the smallest is non-positive — this
    is rare in practice but worth flagging because the next E-step
    relies on :math:`\mathbf{Q}^{-1}` for the Mahalanobis residual.

    **Normalisation: why** :math:`1/T_{\text{eff}}` **and not**
    :math:`1/\sum_t \hat{w}^u_t`.
    The thesis derivation (eq. 6094 and eq. 9277) gives the prefactor
    :math:`1/T` where :math:`T` is the same :math:`T` appearing in
    :math:`-\tfrac{T}{2}\log|\mathbf{Q}|` of the expected complete-data
    log-likelihood — i.e. the *number of state-equation terms*
    actually summed, not the sum of the posterior weights.  The
    weights :math:`\hat{w}^u_t` enter only *inside* the sums
    :math:`\mathcal{P}_{ab}^{u}`; the :math:`1/T` outside is a pure
    counting factor.  In the Gaussian limit
    :math:`\hat{w}^u_t \to 1` the formula reduces to the standard
    Bańbura-Modugno (2014) update with the same :math:`1/T`
    prefactor.  In our Python indexing (``range(1, T)``) we have
    :math:`T_{\text{eff}} = T - 1` transitions, hence the parameter
    name.

    Examples
    --------
    >>> mom = compute_weighted_moments(f_smooth, P_smooth, P_lag, w_u, r)
    >>> A_new, Q_new = update_A_Q(mom["P00"], mom["P10"], mom["P11"],
    ...                           T_eff=f_smooth.shape[0] - 1)
    """
    r = P00.shape[0]
    if P00.shape != (r, r) or P10.shape != (r, r) or P11.shape != (r, r):
        raise ValueError(
            f"Shape mismatch: P00={P00.shape}, P10={P10.shape}, P11={P11.shape}; "
            f"expected all ({r}, {r})."
        )
    if T_eff <= 0:
        raise ValueError(f"T_eff must be strictly positive, got {T_eff}.")

    # ── A-update via solve (no explicit inverse) ─────────────────────────────
    # Closed-form: A = P10 @ inv(P00).
    # Equivalent linear system:  A @ P00 = P10  =>  P00 @ A' = P10'  =>
    # solve P00 @ X = P10'  for X = A', then transpose.
    A_new_T = np.linalg.solve(P00, P10.T)        # (r, r)  contains A'
    A_new   = A_new_T.T                          # (r, r)

    # ── Q-update at the just-updated A ───────────────────────────────────────
    # Q = (1/T_eff) * (P11 - A_new @ P10').
    Q_raw = (P11 - A_new @ P10.T) / float(T_eff)

    # Symmetrise Q before returning.
    #
    # Theoretically Q is a covariance matrix and must be symmetric.
    # The closed-form update Q_raw = (1/T)(P11 - A_new @ P10.T) is
    # symmetric at the exact optimum: P11 is symmetric by construction,
    # and at the updated A^(j+1) the cross-term A_new @ P10.T is also
    # symmetric (see thesis eq. for S_ff at the new A, riga ~5982).
    #
    # In floating-point arithmetic, however, the matrix products and
    # subtraction introduce rounding errors of order 1e-12 to 1e-15,
    # so Q_raw[i,j] and Q_raw[j,i] may differ in their last digits.
    # This tiny asymmetry is pure numerical noise, NOT signal.
    #
    # Taking Q_new = 0.5*(Q_raw + Q_raw.T) keeps the symmetric part
    # (the true value) and discards the antisymmetric part (the noise).
    # It does not alter the result: it removes a numerical artefact
    # that should not be there. After this step Q_new is exactly
    # symmetric (max|Q_new - Q_new.T| = 0).
    #
    # Why this matters: Q feeds into the Kalman recursions (Q_tilde,
    # then P_pred = A_tilde P A_tilde' + Q_tilde). An asymmetric Q
    # would let asymmetries accumulate in the state covariances across
    # the ~50-100 EM iterations, and could produce complex eigenvalues
    # that break the positive-definiteness check and the matrix
    # inversions. Symmetrising covariance matrices at every step is
    # standard practice for numerical stability in Kalman/EM routines
    # (the same is done for P_filt and P_smooth in kalman.py).
    Q_new = 0.5 * (Q_raw + Q_raw.T)

    # Positive-definiteness check.
    #
    # A valid covariance matrix must be positive definite (all
    # eigenvalues > 0). At theta^(0) and in early EM iterations Q is
    # comfortably PD. We check the minimum eigenvalue and emit a
    # (non-fatal) warning if it is <= 0, which would signal a
    # degenerate factor innovation (a near-deterministic factor) or
    # numerical trouble. In the Student-t model Q tends to CONTRACT
    # relative to the Gaussian case (outlier innovations are
    # down-weighted via w_u), so a collapse towards a singular Q is
    # something to watch for if a factor's innovations are almost
    # entirely explained away as outliers — though we do not expect
    # this in practice.
    eigvals_Q = np.linalg.eigvalsh(Q_new)
    if eigvals_Q.min() <= 0.0:
        print(
            f"[update_A_Q WARNING] Q_new has a non-positive eigenvalue: "
            f"min eig = {eigvals_Q.min():.3e}   "
            f"(eigenvalues = {eigvals_Q}).  The next E-step's Mahalanobis "
            f"residual will fail.  Inspect the weighted moments."
        )

    return A_new, Q_new


# ─── 4. Diagonal update of the idiosyncratic variances R ─────────────────────

def update_R(
    Y: np.ndarray,
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    Lambda_new: np.ndarray,
    w_eps: np.ndarray,
    W_list: list[np.ndarray],
    block_map: dict[str, str],
    freq_list: list[str],
    ordered_cols: list[str],
    r: int,
) -> np.ndarray:
    r"""
    Closed-form, entry-by-entry M-step update of the idiosyncratic
    variance vector :math:`\mathbf{R} = \mathrm{diag}(r_1, \ldots, r_M)`,
    evaluated at the *just-updated* loading matrix
    :math:`\mathbf{\Lambda}^{(j+1)}` (sequential conditional
    maximisation: R after Lambda).

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - Section "The M-Step in the Unified Model", paragraph (d)
        (line ~9298-9309): compact statement of the R-update in the
        unified Student-t mixed-frequency model.  The formula reads:

        .. math::

            r^{(j+1)}_i \;=\; \frac{1}{|\mathcal{T}_i|}\,
                \sum_{t \in \mathcal{T}_i} \hat{w}^{\varepsilon}_t\,
                \mathbb{E}\!\big[\,e_{i,t}^2 \,\big|\, \mathbf{Y}\,\big],

        with :math:`e_{i,t} = y^M_{i,t} - \mathbf{\Lambda}^{M,(j+1)}_{i\cdot}\, f^k_t`
        for monthly series and
        :math:`e_{i,t} = y^Q_{i,t} - \mathbf{\Lambda}^{Q,(j+1)}_{i\cdot}\, \phi^k_t`
        for quarterly series.
      - Section "Update of the Idiosyncratic Scale Matrix R"
        (subsec:update-R, line ~6217-6339): full derivation of the
        unrestricted closed-form update from the first-order condition
        of :math:`L_{\mathbf{R}}(\mathbf{R}) =
        -\tfrac{T}{2}\log|\mathbf{R}| - \tfrac{1}{2}\,\mathrm{tr}[
        \mathbf{R}^{-1}\, \mathcal{S}_{yy}^{\,\varepsilon}(\mathbf{\Lambda}^{(j+1)})]`,
        followed by the projection onto the diagonal class (the DGP
        assumption that :math:`\mathbf{R}` is diagonal).  The
        residual-based form (eq:R-update-residuals, line ~6308-6311)
        is the per-series scalar identity used here.
      - Mixed-frequency specialisation (line ~7994-8008):
        eq:mm-RM-update and eq:mm-RQ-update separate monthly and
        quarterly rows; the monthly part uses :math:`f^k_t` as
        regressor, the quarterly part uses the composite MM regressor
        :math:`\phi^k_t = \tfrac{1}{3}f^k_t + \tfrac{2}{3}f^k_{t-1}
        + f^k_{t-2} + \tfrac{2}{3}f^k_{t-3} + \tfrac{1}{3}f^k_{t-4}`.
      - Normalisation note (line ~8116-8136): explicit justification
        of the :math:`1/|\mathcal{T}_i|` prefactor (the count of
        observed periods for series i) over the BM-style :math:`1/T`
        prefactor.  The latter systematically *underestimates*
        :math:`r_i` by a factor :math:`|\mathcal{T}_i|/T` for series
        with substantial missing data — most strikingly for quarterly
        GDP, where two-thirds of the months are missing by
        construction.  Using :math:`1/|\mathcal{T}_i|` recovers the
        standard ML estimator on the observed sample (Bańbura,
        Giannone, Modugno, Reichlin 2013; Jungbacker-Koopman 2014).

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Mixed-frequency observation panel, *already standardised*,
        with ``NaN`` for missing entries.  Monthly NaNs sit at the
        ragged edge (publication lag); quarterly NaNs cover all
        non-quarter-end months by construction.  The NaN pattern of
        column ``i`` defines :math:`\mathcal{T}_i`.
    f_smooth : np.ndarray, shape (T, 5r)
        Smoothed augmented state means
        :math:`\hat{\tilde{f}}_{t \mid T}` from the E-step.  Block
        ``l`` (columns ``l*r:(l+1)*r``) carries the
        contemporaneous-at-t image of :math:`f_{t-l}`.  The
        contemporaneous block-factor :math:`f^k_t` is
        ``f_smooth[t, j]`` and the five lagged block-factors used to
        build :math:`\phi^k_t` are ``f_smooth[t, l*r + j]`` for
        :math:`l = 0, \ldots, 4`.
    P_smooth : np.ndarray, shape (T, 5r, 5r)
        Smoothed augmented state covariances
        :math:`\tilde{P}_{t \mid T}` from the E-step.  The diagonal
        entry ``P_smooth[t, j, j]`` is :math:`\mathrm{Var}(f^k_t \mid \mathbf{Y})`;
        the :math:`5 \times 5` sub-block on indices
        ``[j, r+j, 2r+j, 3r+j, 4r+j]`` is the joint posterior
        covariance of the five lagged block-factors used in
        :math:`\phi^k_t`.
    Lambda_new : np.ndarray, shape (M, r)
        Updated loading matrix :math:`\mathbf{\Lambda}^{(j+1)}` from
        :func:`update_Lambda`.  Block-diagonal by construction: row
        ``i`` has at most one non-zero entry, at column
        ``j = _BLOCK_TO_COL[block_map[ordered_cols[i]]]``.  *Must*
        be the updated value, not :math:`\mathbf{\Lambda}^{(j)}` —
        this is the defining feature of sequential conditional
        maximisation (thesis line ~5608).
    w_eps : np.ndarray, shape (T,)
        Posterior mean of the idiosyncratic Student-t weights
        :math:`\hat{w}^\varepsilon_t` from the converged inner ECM
        loop.  All weights strictly positive.  Outlier months
        (large Mahalanobis residual on the observation side, e.g.
        COVID April 2020) receive small weights and contribute less
        to each row's sum — the robust shielding mechanism of the
        Student-t model.
    W_list : list of np.ndarray, length T
        Selection matrices kept for API consistency with the E-step;
        the row-by-row primitive used here reads the observation
        sets directly from ``np.isnan(Y[:, i])``.
    block_map : dict[str, str]
        Maps each series name to its economic block
        (``"real"``, ``"financial"``, ``"other"``).
    freq_list : list[str], length M
        Frequency tag (``"monthly"`` or ``"quarterly"``) of each
        series, aligned with ``ordered_cols`` and the columns of Y.
    ordered_cols : list[str], length M
        Series names in the order of the columns of Y.
    r : int
        Number of monthly latent factors (= number of blocks = 3 in
        this project).

    Returns
    -------
    R_new : np.ndarray, shape (M,)
        Updated diagonal of :math:`\mathbf{R}^{(j+1)}`: ``R_new[i]``
        is the variance of the idiosyncratic residual of series ``i``
        at the updated loadings.  All entries strictly positive
        (asserted; an entry near zero would indicate a degenerate
        residual or vanishing weights — neither expected in practice).
        Stored as a 1-D vector; the full matrix is
        :math:`\mathrm{diag}(\mathtt{R\_new})`.

    Notes
    -----
    **Why R is diagonal.**
    The DGP assumes that the idiosyncratic components
    :math:`\varepsilon_{1,t}, \ldots, \varepsilon_{M,t}` are
    *contemporaneously uncorrelated* across series: all comovement is
    captured by the common factors, and what remains is
    series-specific noise.  Imposing this at the M-step means that
    the unrestricted closed-form maximiser (an :math:`M \times M`
    weighted residual outer-product matrix
    :math:`\mathcal{S}_{yy}^{\,\varepsilon}(\mathbf{\Lambda}^{(j+1)})`,
    thesis eq:Syy-simplified, line ~6244) is projected onto the
    diagonal class — equivalent to differentiating with respect to
    each :math:`r_i` separately (thesis line ~6275-6283).  This
    collapses the multivariate problem into M *scalar* updates, one
    per series, that we implement here.

    **Sequential conditional maximisation: R uses :math:`\mathbf{\Lambda}^{(j+1)}`.**
    A fully simultaneous maximisation of
    :math:`L(\mathbf{\Lambda}, \mathbf{R})` couples Lambda and R
    through the residual variance.  Conditional maximisation breaks
    the coupling: first update Lambda holding R fixed (gives the
    weighted-OLS scalar formula in :func:`update_Lambda`), then
    update R *at the new* Lambda.  The cross-terms in
    :math:`\mathcal{S}_{yy}^{\,\varepsilon}(\mathbf{\Lambda}^{(j+1)})`
    cancel because of the Lambda first-order condition (thesis
    eq:Syy-simplified), exactly mirroring the algebra of the
    (A, Q) sequential update.  This is the ECM scheme of Meng-Rubin
    (1993) — the fixed point is identical to the simultaneous one,
    but each step is closed-form.

    **Posterior second moment of the residual.**
    The conditional expectation inside the sum expands as:

    .. math::

        \mathbb{E}\!\big[(y_{i,t} - \boldsymbol{\lambda}_i'\, f_t)^2 \,\big|\, \mathbf{Y}\big]
        \;=\; (y_{i,t} - \boldsymbol{\lambda}_i'\, \mathbb{E}[f_t \mid \mathbf{Y}])^2
        \;+\; \boldsymbol{\lambda}_i'\, \mathrm{Var}(f_t \mid \mathbf{Y})\, \boldsymbol{\lambda}_i,

    using
    :math:`\mathbb{E}[X^2] = (\mathbb{E}[X])^2 + \mathrm{Var}(X)`
    applied to the scalar
    :math:`X = y_{i,t} - \boldsymbol{\lambda}_i' f_t`.  The first
    term is the *point residual squared* at the posterior mean of
    :math:`f_t`; the second is the *contribution of posterior
    uncertainty about the factor to the residual variance*.  Under
    block restrictions :math:`\boldsymbol{\lambda}_i` has a single
    non-zero entry :math:`\Lambda_{ij}`, so the variance term
    simplifies to :math:`\Lambda_{ij}^2 \cdot P_{t \mid T}[j, j]`
    (monthly) or :math:`\Lambda_{ij}^2 \cdot \mathrm{Var}(\phi^k_t \mid \mathbf{Y})`
    (quarterly).  Dropping this term would systematically
    *underestimate* :math:`r_i`: in periods of high posterior
    uncertainty about the factor — typically the ragged edge — the
    estimator would treat all the unexplained variation as if the
    factor were known exactly, biasing R downward and Lambda upward
    in subsequent iterations.  This is the analogue of the
    posterior-covariance correction in :func:`compute_weighted_moments`
    (Banbura-Modugno 2014 §3.2).

    **Composite regressor for quarterly series.**
    For the (unique, in this project) quarterly row GDPC1, the
    regressor :math:`f^k_t` is replaced by the MM composite
    :math:`\phi^k_t`.  Its posterior moments are read off the
    augmented smoothed state at the *same* time index t (no recourse
    to P_lag), exactly as in :func:`update_Lambda`:

    .. math::

        \mathbb{E}[\phi^k_t \mid \mathbf{Y}] \;&=\; \mathbf{c}'\,
        \hat{\tilde{f}}_{t \mid T}[\,\mathrm{idx}\,], \\[2pt]
        \mathrm{Var}(\phi^k_t \mid \mathbf{Y}) \;&=\; \mathbf{c}'\,
        \tilde{P}_{t \mid T}[\,\mathrm{idx}, \mathrm{idx}\,]\, \mathbf{c},

    with ``idx = [j, r+j, 2r+j, 3r+j, 4r+j]`` and
    :math:`\mathbf{c} = (\tfrac{1}{3}, \tfrac{2}{3}, 1, \tfrac{2}{3}, \tfrac{1}{3})`.
    The first quarter-end at :math:`t = 2` (Python-indexed: zero-based)
    requires pre-sample factors :math:`f_{-1}, f_{-2}`: the augmented
    initial-state prior supplies these as latent variables, and the
    smoother delivers their posterior moments together with the
    in-sample ones (thesis "Boundary at the start of the sample",
    line ~8038-8078).  No special boundary handling needed here.

    **Why** :math:`1/|\mathcal{T}_i|` **and not** :math:`1/T` **or** :math:`1/\sum_t \hat{w}^\varepsilon_t`.
    Three candidate normalisations exist; we use the first:

    1. :math:`1/|\mathcal{T}_i|` (count of observed periods for
       series i).  This is the standard maximum-likelihood estimator
       of the idiosyncratic variance on the observed sample, and is
       what the thesis derivation in the mixed-frequency setting
       yields (eq:mm-RM-update / eq:mm-RQ-update, line ~7999-8008,
       and the explicit normalisation discussion at line ~8116-8136).

    2. :math:`1/T` (Bańbura-Modugno 2014 original).  Underestimates
       :math:`r_i` by the factor :math:`|\mathcal{T}_i|/T` for series
       with missing data.  For quarterly GDP, where roughly
       two-thirds of the months are missing, this is a *factor-three*
       underestimate — material.  Refined in BM-Giannone-Reichlin
       2013 and Jungbacker-Koopman 2014 to the form we use.

    3. :math:`1/\sum_{t \in \mathcal{T}_i} \hat{w}^\varepsilon_t`
       (weighted-average form).  This is the form your spec sheet
       wrote; it differs from (1) when the weights deviate from 1.
       The thesis derivation does *not* use this form: the prefactor
       :math:`1/T` (or :math:`1/|\mathcal{T}_i|` in the missing-data
       case) comes from :math:`-\tfrac{T}{2}\log|\mathbf{R}|` in the
       expected complete-data log-likelihood, while the weights
       :math:`\hat{w}^\varepsilon_t` enter only inside the sum.  In
       the Gaussian limit :math:`\nu_\varepsilon \to \infty` all
       weights collapse to 1 and forms (1) and (3) coincide; for
       finite :math:`\nu_\varepsilon` they differ by a small
       finite-sample correction (thesis line ~6198-6211 makes the
       same point for Q).  We use form (1) for consistency with the
       thesis derivation, with the Gaussian-limit reduction, and
       with the BM-style mixed-frequency literature.

    **Role of** :math:`\hat{w}^\varepsilon_t`.
    Every per-t residual square inside the sum is multiplied by
    :math:`\hat{w}^\varepsilon_t`.  Outlier months
    (e.g. COVID April 2020 for the real-activity block) receive a
    small weight and contribute proportionally less.  Without this
    weighting, a single extreme outlier residual would inflate
    :math:`r_i` substantially; the Student-t weighting shields
    :math:`\mathbf{R}` from outlier-driven variance inflation.

    **Positivity.**
    Each :math:`r_i` is a weighted sum of strictly non-negative
    quantities (squared residuals plus a non-negative
    variance-correction term) divided by a positive integer.  It is
    strictly positive whenever at least one observed period
    contributes a non-zero residual, which is the case for every
    series in this dataset.  A near-zero entry would indicate a
    perfectly explained series — implausible in macro data — and is
    raised as a ``RuntimeError`` for safety.

    Examples
    --------
    >>> Lambda_new = update_Lambda(Y, f_smooth, P_smooth, w_eps, W_list,
    ...                            block_map, freq_list_M, ORDERED_COLS, r)
    >>> R_new = update_R(Y, f_smooth, P_smooth, Lambda_new, w_eps,
    ...                  W_list, block_map, freq_list_M, ORDERED_COLS, r)
    >>> R_new.shape
    (20,)
    """
    T, M = Y.shape
    if Lambda_new.shape != (M, r):
        raise ValueError(
            f"Lambda_new.shape = {Lambda_new.shape}, expected ({M}, {r})."
        )
    if len(ordered_cols) != M:
        raise ValueError(
            f"ordered_cols has length {len(ordered_cols)} but Y has {M} columns."
        )
    if len(freq_list) != M:
        raise ValueError(
            f"freq_list has length {len(freq_list)} but Y has {M} columns."
        )

    R_new = np.zeros(M)

    # Same augmented-state lag indices used in update_Lambda for the
    # quarterly composite regressor phi^k_t.
    quarterly_indices: dict[int, np.ndarray] = {
        j: np.array([l * r + j for l in range(5)]) for j in range(r)
    }

    eps_r = 1e-12   # safety tolerance for the positivity check

    for i in range(M):
        col   = ordered_cols[i]
        block = block_map[col]
        if block not in _BLOCK_TO_COL:
            raise KeyError(
                f"Series '{col}' has unknown block '{block}'. "
                f"Expected one of {_BLOCK_ORDER}."
            )
        j     = _BLOCK_TO_COL[block]
        freq  = freq_list[i]
        lam   = float(Lambda_new[i, j])     # scalar loading (block restriction)
        lam2  = lam * lam

        # T_i  = set of times at which series i is observed.
        obs_mask = ~np.isnan(Y[:, i])
        obs_t    = np.where(obs_mask)[0]
        n_obs    = obs_t.size

        if n_obs == 0:
            # All-NaN series (e.g. WPSFD49207 before 2016-03 in the big config).
            # update_Lambda already set lambda[i,:] = 0 via its own n_obs==0 branch,
            # so this series carries no information in the current vintage.
            # Keep r_i = 1.0 (unit variance of the standardised data) as a
            # neutral placeholder; the Kalman filter ignores it when Y[:,i] is all NaN.
            R_new[i] = 1.0
            continue

        y_i = Y[obs_t, i]                   # (n_obs,)
        w_i = w_eps[obs_t]                  # (n_obs,)

        if freq == "monthly":
            # Regressor: contemporaneous block-factor f^k_t.
            E_f  = f_smooth[obs_t, j]                   # (n_obs,)
            V_f  = P_smooth[obs_t, j, j]                # (n_obs,)

            # Point residual at the posterior mean of f^k_t.
            resid_point = y_i - lam * E_f               # (n_obs,)

            # E[(y - lam f)^2 | Y] = resid_point^2 + lam^2 * Var(f^k_t | Y).
            E_resid2 = resid_point ** 2 + lam2 * V_f    # (n_obs,)

        elif freq == "quarterly":
            # Regressor: MM composite block-factor phi^k_t.
            idx = quarterly_indices[j]                  # (5,)

            f_block = f_smooth[obs_t][:, idx]           # (n_obs, 5)
            E_phi   = f_block @ _MM_WEIGHTS             # (n_obs,)

            P_block = P_smooth[
                obs_t[:, None, None],
                idx[None, :, None],
                idx[None, None, :],
            ]                                            # (n_obs, 5, 5)
            V_phi   = np.einsum(
                "s,nsl,l->n", _MM_WEIGHTS, P_block, _MM_WEIGHTS
            )                                            # (n_obs,)

            resid_point = y_i - lam * E_phi              # (n_obs,)
            E_resid2    = resid_point ** 2 + lam2 * V_phi

        else:
            raise ValueError(
                f"Series '{col}' has unknown freq '{freq}'. "
                "Expected 'monthly' or 'quarterly'."
            )

        # r_i = (1/|T_i|) * sum_{t in T_i} w_eps[t] * E[resid^2 | Y].
        r_i = float(np.sum(w_i * E_resid2)) / float(n_obs)

        if r_i <= eps_r:
            raise RuntimeError(
                f"Degenerate idiosyncratic variance for series '{col}' "
                f"(block '{block}', freq '{freq}'): r_i = {r_i:.3e}. "
                f"The residual is essentially zero across the observation "
                f"set — implausible for macro data, inspect Lambda_new[i, j]."
            )

        R_new[i] = r_i

    return R_new


# ─── 5. Degrees-of-freedom update via Brent root-finding ─────────────────────

def update_nu(
    w_bar: float,
    log_w_bar: float,
    nu_bounds: tuple[float, float] = (2.001, 1000.0),
) -> float:
    r"""
    M-step update of a Student-t degrees-of-freedom parameter
    (:math:`\nu_u` or :math:`\nu_\varepsilon`) by one-dimensional
    root-finding on the first-order condition.

    The two degrees-of-freedom parameters are the *only* parameters of
    the model that cannot be updated in closed form: the relevant terms
    of the expected complete-data log-likelihood involve
    :math:`\log \Gamma(\nu/2)`, whose derivative is the (transcendental)
    digamma function :math:`\psi`.  The first-order condition is
    therefore an *implicit equation in* :math:`\nu`, solvable only
    numerically.  This makes the overall algorithm an ECME
    (Expectation-Conditional-Maximisation Either) algorithm in the
    sense of Liu and Rubin (1994): all other M-step blocks are
    closed-form CM steps; the two :math:`\nu`-updates are "Either"
    steps that maximise the FOC numerically.

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - subsec:update-nu (line ~6362): full derivation of both
        :math:`\nu_u` and :math:`\nu_\varepsilon` updates.
      - eq:L-nu-u-compact (line ~6411): the per-T objective.
      - eq:foc-nu-u (boxed, line ~6464) — *the* first-order condition
        solved here:

        .. math::

            \log\!\tfrac{\nu}{2} \;-\; \psi\!\big(\tfrac{\nu}{2}\big)
            \;+\; 1 \;+\; \overline{\log w} \;-\; \bar{w} \;=\; 0.

      - eq:foc-nu-eps-summary (line ~6671): the analogous condition for
        :math:`\nu_\varepsilon`.  Identical functional form — same
        digamma identity, same posterior summaries.
      - "Properties of the First-Order Condition" (line ~6490):
        monotonicity and boundary behaviour of g.
      - "Brent's method (our choice)" (line ~6590): rationale for
        Brent over Newton.

    Parameters
    ----------
    w_bar : float
        Posterior mean of the Student-t weights, averaged over time:
        :math:`\bar{w} \equiv \frac{1}{T}\sum_{t=1}^{T} \hat{w}_t`.
        For :math:`\nu_u` this is the mean of ``w_u`` (factor-side
        weights) from the converged E-step; for :math:`\nu_\varepsilon`
        the mean of ``w_eps`` (idiosyncratic-side weights).
    log_w_bar : float
        Posterior mean of the *log* weights, averaged over time:
        :math:`\overline{\log w} \equiv \frac{1}{T}\sum_{t=1}^{T}
        \widehat{\log w}_t`.  For :math:`\nu_u` from ``log_w_u``; for
        :math:`\nu_\varepsilon` from ``log_w_eps``.
    nu_bounds : tuple of float, optional
        Bracketing interval :math:`[\nu^{\min}, \nu^{\max}]` for Brent.
        Default ``(2.001, 1000.0)`` — the conservative bracket
        recommended by the thesis (riga ~6614).  The lower bound is
        just above 2 so the Student-t second moment exists (a natural
        constraint for a covariance-based DGP); the upper bound is
        large enough that any root above it can be safely treated as
        "essentially Gaussian" — at :math:`\nu = 1000` the
        difference from the Gaussian DGP in any practical statistic
        of interest is far below sampling noise.

    Returns
    -------
    nu_new : float
        The updated degrees of freedom.  Always inside
        ``[nu_min, nu_max]``.  See "Handling of the boundary cases"
        below for the clamping convention.

    Notes
    -----
    **Why the same function for both** :math:`\nu_u` **and**
    :math:`\nu_\varepsilon`.
    The FOC eq:foc-nu-u and its idiosyncratic counterpart
    eq:foc-nu-eps-summary are *identical in functional form*: they
    differ only in which posterior summaries
    :math:`(\bar{w}, \overline{\log w})` enter on the right.  The
    dimension of the underlying Student-t distribution (:math:`r` for
    the factor side, :math:`M` or, with missing data, :math:`m_t` for
    the idiosyncratic side) does *not* appear explicitly in the FOC.
    Where the dimension enters is *upstream*, in the E-step
    computation of :math:`\hat{w}_t` and :math:`\widehat{\log w}_t`:
    the posterior conjugacy of the Gamma mixing distribution yields
    shape :math:`\tfrac{\nu + r}{2}` (resp. :math:`\tfrac{\nu + m_t}{2}`)
    and rate :math:`\tfrac{\nu + d_t}{2}`, so the dimension is already
    baked into the per-t weights.  By the time we average them into
    :math:`\bar{w}` and :math:`\overline{\log w}` and plug them into
    :math:`g(\nu)`, no explicit dimension remains.  Hence one
    parameterless ``update_nu`` serves both updates — see the thesis
    discussion at line ~6712-6734 confirming this point and noting the
    "two quantitative differences" between the two cases (which manifest
    in the *numerical values* of the summaries, not in the FOC's form).

    **Properties of the root function g.**

    .. math::

        g(\nu) \;\equiv\; \log\!\tfrac{\nu}{2} \;-\;
        \psi\!\big(\tfrac{\nu}{2}\big) \;+\; 1 \;+\;
        \overline{\log w} \;-\; \bar{w}.

    1. *Monotonicity.*  :math:`g'(\nu) = 1/\nu - \tfrac{1}{2}\psi'(\nu/2)`.
       The trigamma inequality :math:`\psi'(x) > 1/x` (Abramowitz-Stegun)
       gives :math:`g'(\nu) < 0` for all :math:`\nu > 0`: g is
       *strictly decreasing*.  The root, if it exists, is therefore
       unique.
    2. *Limits.*  As :math:`\nu \to 0^+`, :math:`g(\nu) \to +\infty`
       (digamma diverges faster than log near 0).  As
       :math:`\nu \to \infty`, the standard expansion
       :math:`\psi(x) = \log x - 1/(2x) + O(1/x^2)` gives
       :math:`g(\infty) = 1 + \overline{\log w} - \bar{w}`.
    3. *Sign of g at infinity.*  Jensen + the elementary bound
       :math:`\log x \leq x - 1` give
       :math:`\overline{\log w} - \bar{w} \leq -1`, with equality only
       in the degenerate case :math:`w \equiv 1`.  So generically
       :math:`g(\infty) < 0`, and the IVT yields exactly one root in
       :math:`(0, \infty)`.  The boundary case :math:`g(\infty) \to 0`
       corresponds to *posterior weights essentially equal to 1
       everywhere* — i.e. the data look Gaussian, no down-weighting is
       needed, and the FOC pushes :math:`\nu \to \infty`.

    **Handling of the boundary cases (thesis sign-test, riga ~6618-6626).**
    Before invoking Brent we evaluate :math:`g(\nu^{\min})` and
    :math:`g(\nu^{\max})`.  Because g is strictly decreasing
    (monotonicity proved above), exactly one of three configurations
    occurs:

    1. ``g_min > 0  and  g_max < 0``  —  *bracket valid*:
       a unique root sits inside the interval.  ``brentq`` is
       guaranteed to converge to it.  This is the *typical* outcome
       for any realistic posterior, since the Jensen-bound argument
       gives :math:`g(\infty) \leq 0` and :math:`g(0^+) = +\infty`.

    2. ``g_min > 0  and  g_max > 0``  —  *both positive, root above*
       :math:`\nu^{\max}`: the FOC wants a :math:`\nu` larger than
       :math:`\nu^{\max}`.  The data are essentially Gaussian
       (posterior weights tightly concentrated near 1, so
       :math:`\Delta_{\text{post}} \approx 1`).  We **return**
       :math:`\nu^{\max}`, effectively treating the process as
       Gaussian — exactly the thesis prescription at riga ~6623:
       *"if* :math:`g(\nu^{\max}) > 0` *[...] we set*
       :math:`\nu^{(j+1)} = \nu^{\max}` *, effectively treating the
       process as Gaussian"*.

    3. ``g_min < 0  and  g_max < 0``  —  *both negative, root below*
       :math:`\nu^{\min}`: the FOC wants a :math:`\nu` smaller than
       :math:`\nu^{\min}`.  The data are extremely heavy-tailed
       (so much that even :math:`\nu^{\min} \approx 2` is too large
       to fit the posterior dispersion).  We **return**
       :math:`\nu^{\min}`.  This case is rare and signals either
       pathological outliers or a model misspecification; the clamp
       preserves the moment-existence constraint :math:`\nu > 2` and
       prevents the EM iterates from drifting into a region where
       :math:`\nu \leq 2` and the Student-t covariance is undefined.

    The fourth logical configuration (``g_min < 0  and  g_max > 0``)
    would contradict the strict monotonicity ``g' < 0`` and cannot
    occur — its appearance would indicate a bug in the upstream
    computation of :math:`(\bar{w}, \overline{\log w})`.

    **Why Brent and not Newton.**
    Newton converges in fewer iterations (quadratic near the root) but
    can overshoot into negative :math:`\nu` without safeguards, and
    requires the trigamma function (which explodes as
    :math:`\nu \to 0`).  Brent (Brent 1973) combines bisection,
    secant, and inverse quadratic interpolation: it is *guaranteed* to
    converge once a sign-bracketing interval is provided, and uses
    only :math:`\log` and :math:`\psi` (no derivative).  The thesis
    rationale is at line ~6628-6637.

    **Statistical interpretation.**
    The FOC can be rewritten (thesis eq:nu-interpretation, line ~6755)
    as

    .. math::

        \underbrace{\log\!\tfrac{\nu}{2} - \psi\!\big(\tfrac{\nu}{2}\big)}_{\Delta_{\text{prior}}(\nu)}
        \;+\; 1 \;=\; \underbrace{\bar{w} - \overline{\log w}}_{\Delta_{\text{post}}},

    *matching the concavity gap of* :math:`\log` *under the
    Gamma*\ :math:`(\nu/2, \nu/2)` *prior to the empirical concavity
    gap implied by the posterior weights*.  A large posterior gap
    (dispersed weights — fat tails) forces a small :math:`\nu`; a
    posterior gap near 1 (concentrated weights — Gaussian-like data)
    pushes :math:`\nu` to infinity.  This is the precise mechanism by
    which EM "reads" tail heaviness from the posterior weights.

    **Note on the boundary weight at** ``t = 0`` **(``w_u`` only).**
    The factor-side weight at :math:`t = 0` is set to the prior mean
    (``w_u[0] = 1.0``) and the corresponding log-weight to the prior
    log-mean (``log_w_u[0] = psi(nu_u/2) - log(nu_u/2)``) by the
    E-step convention, since :math:`d_u[0]` is undefined (no
    :math:`f_{-1}`).  These boundary values are included in the
    averages by the caller in this self-test (one period out of
    :math:`T = 497`; the effect on :math:`\nu^{(j+1)}` is negligible).
    The thesis sums (line ~6405) run from :math:`t = 1` to :math:`T`,
    consistent with including this prior term.

    Examples
    --------
    >>> # After running the E-step:
    >>> w_u_bar     = float(np.mean(w_u))
    >>> log_w_u_bar = float(np.mean(log_w_u))
    >>> nu_u_new    = update_nu(w_u_bar, log_w_u_bar)
    >>> # And analogously for nu_eps with w_eps, log_w_eps.
    """
    nu_min, nu_max = nu_bounds
    if not (0 < nu_min < nu_max):
        raise ValueError(
            f"Invalid nu_bounds = {nu_bounds}: require 0 < nu_min < nu_max."
        )

    # Root function g(nu) — closure over the posterior summaries.
    def g(nu: float) -> float:
        half = 0.5 * nu
        return float(np.log(half) - digamma(half) + 1.0 + log_w_bar - w_bar)

    # Sign test at the bracket endpoints (thesis riga ~6618-6626).
    # Because g is strictly decreasing on (0, inf), exactly one of three
    # outcomes is possible:
    #   (1) g_min > 0 and g_max < 0  ->  valid bracket: brentq finds the root.
    #   (2) g_min > 0 and g_max > 0  ->  root above nu_max: clamp to nu_max
    #                                    (data essentially Gaussian).
    #   (3) g_min < 0 and g_max < 0  ->  root below nu_min: clamp to nu_min
    #                                    (extremely heavy-tailed; rare).
    # The configuration (g_min < 0 and g_max > 0) is ruled out by g' < 0
    # and would indicate a bug upstream.
    g_min = g(nu_min)
    g_max = g(nu_max)

    if g_min > 0.0 and g_max < 0.0:
        # Case (1): valid sign-changing bracket.
        nu_new = brentq(g, nu_min, nu_max, xtol=1e-8, rtol=1e-10, maxiter=200)
        return float(nu_new)

    if g_min > 0.0 and g_max > 0.0:
        # Case (2): essentially Gaussian.  Clamp at nu_max.
        return float(nu_max)

    if g_min < 0.0 and g_max < 0.0:
        # Case (3): extremely heavy-tailed.  Clamp at nu_min to preserve
        # the moment-existence constraint nu > 2.
        return float(nu_min)

    # Defensive guard: the remaining (g_min < 0, g_max > 0) configuration
    # contradicts the monotonicity of g and should never arise.  If it
    # does, there is a bug upstream in (w_bar, log_w_bar).
    raise RuntimeError(
        f"update_nu: sign-test failure that contradicts the strict "
        f"monotonicity of g.  g({nu_min}) = {g_min:.3e}, "
        f"g({nu_max}) = {g_max:.3e}.  Check the upstream computation of "
        f"(w_bar, log_w_bar) — Jensen requires bar(w) - log_bar(w) >= 1."
    )


# ─── 6. High-level M-step wrapper ────────────────────────────────────────────

def run_m_step(
    Y: np.ndarray,
    e_step_output: dict,
    theta_old: dict | "np.lib.npyio.NpzFile",
    freq_list: list[str] | None = None,
    block_map: dict[str, str] | None = None,
    ordered_cols: list[str] | None = None,
    freeze_nu_iters: int = 0,
    current_iter: int = 0,
    nu_bounds: tuple[float, float] = (2.001, 1000.0),
) -> dict:
    r"""
    Run a full sequential-ECM M-step at outer iteration ``j``, taking the
    converged E-step output and the previous parameter iterate
    :math:`\theta^{(j)}` and returning the next iterate
    :math:`\theta^{(j+1)}`.

    This is the high-level orchestrator that wires together Tasks 1–5
    (:func:`compute_weighted_moments`, :func:`update_Lambda`,
    :func:`update_A_Q`, :func:`update_R`, :func:`update_nu`) in the
    block-by-block order prescribed by the thesis.  It contains *no*
    new statistical logic — every algebraic update is delegated to the
    per-block function — but it does encode three choices that matter:

    1. *Decoupling and sequencing of the M-step blocks* (see below).
    2. *Optional freezing of the* :math:`\nu` *updates* in the first
       ``freeze_nu_iters`` outer iterations.
    3. *Sigma_0 is held fixed*: the unified M-step in the thesis
       (subsec:full-m-step, riga ~9265) does NOT update the
       initial-state covariance.  See the "Notes" section for a full
       explanation.

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - Section "The M-Step in the Unified Model" (subsec:full-m-step,
        riga ~9265-9319): the five M-step paragraphs (a) A and Q,
        (b) :math:`\Lambda^M`, (c) :math:`\Lambda^Q`, (d) R,
        (e) :math:`\nu_u` and :math:`\nu_\varepsilon`.
      - Section "Sequential conditional maximisation" (riga ~5652-5667):
        the explicit statement that the M-step is performed as a sequence
        of conditional maximisation (CM) steps in the sense of
        Meng-Rubin (1993), each maximising the expected complete-data
        log-likelihood holding the previously updated blocks fixed.
      - Section "(iii) Student-t extensions affect only the *weights*
        in the sums" (riga ~9341): the structural updates of A, Lambda,
        R are the Gaussian Bańbura-Modugno updates with the per-t
        Student-t weights :math:`\hat{w}^u_t`, :math:`\hat{w}^\varepsilon_t`
        inserted; the only block that is genuinely non-closed-form is
        :math:`\nu`, hence the ECME label.
      - Section "Properties of the First-Order Condition" (riga ~6490)
        and "Brent's method (our choice)" (riga ~6590): rationale for
        the numerical :math:`\nu`-update.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Mixed-frequency observation panel (standardised), with ``NaN``
        for missing entries.  Same array used in the E-step.
    e_step_output : dict
        Output of :func:`em_e_step.run_e_step`.  Required keys:
        ``f_smooth`` (T, 5r), ``P_smooth`` (T, 5r, 5r),
        ``P_lag`` (T, 5r, 5r), ``w_u`` (T,), ``w_eps`` (T,),
        ``log_w_u`` (T,), ``log_w_eps`` (T,).  An optional key
        ``W_list`` (list of T selection matrices) is honoured if
        present; otherwise the selection matrices are rebuilt at runtime
        from the NaN pattern of ``Y`` via
        :func:`kalman.build_all_selection_matrices`.
    theta_old : dict-like
        Current parameter iterate :math:`\theta^{(j)}`.  Read access via
        ``theta_old[key]``; both Python dicts and ``np.lib.npyio.NpzFile``
        objects (returned by :func:`numpy.load`) work.  Required keys:
        ``Lambda`` (M, r), ``A`` (r, r), ``Q`` (r, r), ``R`` (M,),
        ``nu_u`` (scalar), ``nu_eps`` (scalar), ``Sigma_0`` (5r, 5r).
        Auxiliary keys (``w_u``, ``w_eps``, ``F``) carried by the
        initialisation NPZ are propagated to ``theta_new`` if present.
    freq_list : list[str], length M, optional
        Frequency tag (``"monthly"`` or ``"quarterly"``) for each
        column of ``Y``.  Defaults to
        ``[FREQ[c] for c in ordered_cols]`` using the canonical
        ``ORDERED_COLS`` and ``FREQ`` mapping from
        :mod:`data_loader`.
    block_map : dict[str, str], optional
        Maps each series name to its economic block
        (``"real"``, ``"financial"``, ``"other"``).  Defaults to
        :data:`data_loader.BLOCK`.
    ordered_cols : list[str], length M, optional
        Series names in the order of the columns of ``Y``.  Defaults to
        :data:`data_loader.ORDERED_COLS`.
    freeze_nu_iters : int, default 0
        Number of outer EM iterations during which the two
        degrees-of-freedom parameters are *not* updated and instead
        kept fixed at their ``theta_old`` values.  Useful for
        stabilising the EM trajectory in the very first iterations,
        when the smoothed factors and the implied Mahalanobis residuals
        are still adjusting to the initial PCA configuration and an
        early :math:`\nu`-update can be dominated by initialisation
        artefacts.  Default 0 = update :math:`\nu_u, \nu_\varepsilon`
        from the very first iteration.  A typical safe value in
        practice is 1–3.
    current_iter : int, default 0
        Zero-based index of the current outer EM iteration.  Compared
        against ``freeze_nu_iters`` to decide whether to actually run
        the :math:`\nu`-updates this iteration.
    nu_bounds : tuple of float, default (2.001, 1000.0)
        Bracket :math:`[\nu^{\min}, \nu^{\max}]` passed to
        :func:`update_nu` for Brent root-finding (thesis riga ~6614).

    Returns
    -------
    theta_new : dict
        Next parameter iterate :math:`\theta^{(j+1)}`.  Contains
        *every* key of ``theta_old``, with the following keys overwritten
        by their updated values:

          - ``Lambda``  (M, r)  — block-diagonal updated loadings.
          - ``A``       (r, r)  — updated VAR transition matrix.
          - ``Q``       (r, r)  — updated factor-innovation covariance,
                                 symmetric and (generically) PD.
          - ``R``       (M,)    — updated diagonal of the idiosyncratic
                                 variances, strictly positive.
          - ``nu_u``    scalar  — updated factor-side d.o.f., in
                                 ``nu_bounds`` (or frozen).
          - ``nu_eps``  scalar  — updated idiosyncratic-side d.o.f.,
                                 in ``nu_bounds`` (or frozen).
          - ``Sigma_0`` (5r, 5r) — carried forward unchanged from
                                 ``theta_old`` (NOT updated).

        Auxiliary keys present in ``theta_old`` are propagated:
          - ``w_u``   : replaced with the converged E-step weights.
          - ``w_eps`` : replaced with the converged E-step weights.
          - ``F``     : replaced with the smoothed monthly factors
                       ``f_smooth[:, :r]`` (the contemporaneous block of
                       the augmented state).
        Any other key in ``theta_old`` is copied through verbatim.

    Notes
    -----
    **The ECM ordering: two independent pairs + an independent ECME step.**

    The full Q-function of the unified Student-t mixed-frequency model
    factorises into three additive pieces (thesis "Sequential conditional
    maximisation", riga ~5652-5667):

    .. math::

        \mathcal{Q}(\theta) \;=\;
        \underbrace{\mathcal{Q}_{\mathrm{trans}}(\mathbf{A}, \mathbf{Q})}_{\text{depends on } f_t, f_{t-1}, w^u}
        \;+\;
        \underbrace{\mathcal{Q}_{\mathrm{obs}}(\mathbf{\Lambda}, \mathbf{R})}_{\text{depends on } y_t, f_t, w^\varepsilon}
        \;+\;
        \underbrace{\mathcal{Q}_{\nu}(\nu_u, \nu_\varepsilon)}_{\text{depends on } w, \log w}
        \;+\; \text{const.}

    Each summand depends on a *disjoint* subset of the parameters, so
    the three blocks can be maximised independently in any order — the
    M-step is, at the top level, three parallel sub-problems.  We
    update them in the order (observation pair) -> (transition pair) ->
    (nu) for readability; any permutation would yield the same
    :math:`\theta^{(j+1)}`.

    *Inside* each of the two block pairs there is a genuine
    sequentiality dictated by Meng-Rubin (1993):

    - **Observation pair** :math:`(\mathbf{\Lambda}, \mathbf{R})`:
      :math:`\mathbf{R}` enters the trace in
      :math:`\mathcal{Q}_{\mathrm{obs}}` through :math:`\mathbf{R}^{-1}`,
      which weights the residual outer-product matrix.  Holding
      :math:`\mathbf{R}` fixed at :math:`\mathbf{R}^{(j)}` decouples the
      :math:`\mathbf{\Lambda}`-update (a weighted OLS per row) from
      :math:`\mathbf{R}`.  Then :math:`\mathbf{R}^{(j+1)}` is computed
      from the residuals *at* :math:`\mathbf{\Lambda}^{(j+1)}` — this
      yields a strictly larger increase in
      :math:`\mathcal{Q}_{\mathrm{obs}}` than holding :math:`\mathbf{R}`
      at the old value would, and is what the thesis derivation (and
      Banbura-Modugno 2014) prescribes.

    - **Transition pair** :math:`(\mathbf{A}, \mathbf{Q})`:
      analogous structure.  The trick is that the
      :math:`\mathbf{A}`-FOC is :math:`\mathbf{Q}^{-1} \mathcal{P}_{10}^u =
      \mathbf{Q}^{-1} \mathbf{A} \mathcal{P}_{00}^u`, and left-
      multiplying by :math:`\mathbf{Q}` *kills* the dependence on
      :math:`\mathbf{Q}` entirely.  So the :math:`\mathbf{A}` update
      does not actually depend on :math:`\mathbf{Q}^{(j)}` — any
      positive-definite :math:`\mathbf{Q}` produces the same
      :math:`\mathbf{A}^{(j+1)}`.  :math:`\mathbf{Q}^{(j+1)}` is then
      computed at the new :math:`\mathbf{A}^{(j+1)}`, using the
      "Sff-simplified" form (thesis eq. 6034) in which the cross-terms
      cancel because of the :math:`\mathbf{A}`-FOC.  This collapses
      the joint maximisation into two closed-form scalar problems.

    - **ECME block** :math:`(\nu_u, \nu_\varepsilon)`:
      independent of every other parameter once the posterior summaries
      :math:`(\bar{w}, \overline{\log w})` are in hand.  The FOC is
      *not* closed-form (involves the digamma function), hence the
      "Either" label of ECME (Liu and Rubin 1994): we maximise the
      FOC numerically via Brent.  See :func:`update_nu`.

    **Why** ``freeze_nu_iters`` **exists.**
    In the very first outer iteration, :math:`(\bar{w}, \overline{\log w})`
    are computed from posterior weights at :math:`\nu^{(0)} = 10`
    applied to the PCA-initialised smoothed factors and residuals,
    which themselves carry a fair amount of initialisation noise.  In
    practice this is usually fine (the :math:`\nu`-update is well-
    behaved at :math:`\theta^{(0)}` for the macro data we use), but the
    option to freeze :math:`\nu` for a few iterations is a standard
    safeguard for less well-conditioned datasets: it lets
    :math:`(\mathbf{A}, \mathbf{Q}, \mathbf{\Lambda}, \mathbf{R})`
    settle into a good neighbourhood before the tail-heaviness is
    re-estimated.  Defaults to 0 (no freezing).

    **Why** :math:`\mathbf{\Sigma}_0` **is held fixed.**
    The unified M-step in subsec:full-m-step lists exactly five
    update paragraphs (a–e); the initial-state covariance
    :math:`\mathbf{\Sigma}_0` is *not* one of them.  Conceptually,
    :math:`\mathbf{\Sigma}_0` is the prior covariance of
    :math:`\tilde{f}_0` and enters the expected complete-data
    log-likelihood through a single term
    :math:`-\tfrac{1}{2}\,(\mathbb{E}[\tilde{f}_0] - 0)' \mathbf{\Sigma}_0^{-1}
    (\mathbb{E}[\tilde{f}_0] - 0) - \tfrac{1}{2}\log|\mathbf{\Sigma}_0|`
    that contributes O(1) to a sum of O(T) terms — re-optimising it
    each iteration provides essentially no information beyond the
    initialisation (the Lyapunov solution
    :math:`\mathbf{\Sigma}_0 = \mathrm{Lyap}(\mathbf{A}, \mathbf{Q})`
    from Algorithm 1 step 2, EM_for_student_t.tex riga ~4305).  We
    therefore freeze it.  Should a future variant of the thesis
    require an explicit :math:`\mathbf{\Sigma}_0` update — e.g. the
    standard Shumway-Stoffer (1982) closed form
    :math:`\mathbf{\Sigma}_0^{(j+1)} = \tilde{P}_{0 \mid T} +
    \hat{\tilde{f}}_{0 \mid T} \hat{\tilde{f}}_{0 \mid T}'` — it can be
    slotted in here without affecting the other updates.

    **W_list propagation.**
    Both :func:`update_Lambda` and :func:`update_R` accept ``W_list``
    in their signature *for API consistency with the E-step*; both
    actually derive the per-series observation sets from
    ``np.isnan(Y[:, i])`` directly (the row-by-row primitive that
    matches the row-by-row update form).  We therefore rebuild
    ``W_list`` at runtime from the NaN pattern of ``Y`` if it is not
    carried inside ``e_step_output``; this is a few-millisecond
    operation at :math:`T = 497` and avoids forcing the E-step to
    serialise a list of 497 small matrices.

    Examples
    --------
    >>> estep   = run_e_step(Y, theta, freq_list=freq_list)
    >>> theta1  = run_m_step(Y, estep, theta_old=theta)
    >>> # one full outer EM iteration is now (E + M).
    """
    # ── Lazy defaults from data_loader.  Keeping these optional ensures
    #    that run_m_step can be exercised in isolation in a unit test that
    #    fabricates synthetic block / freq metadata, while in production the
    #    caller almost always wants the project-canonical mapping. ─────────
    if (block_map is None) or (freq_list is None) or (ordered_cols is None):
        from data_loader import BLOCK, FREQ, ORDERED_COLS
        if ordered_cols is None:
            ordered_cols = ORDERED_COLS
        if block_map is None:
            block_map = BLOCK
        if freq_list is None:
            freq_list = [FREQ[c] for c in ordered_cols]

    # ── Extract per-period posterior moments from the E-step output ──────
    f_smooth  = e_step_output["f_smooth"]
    P_smooth  = e_step_output["P_smooth"]
    P_lag     = e_step_output["P_lag"]
    w_u       = e_step_output["w_u"]
    w_eps     = e_step_output["w_eps"]
    log_w_u   = e_step_output["log_w_u"]
    log_w_eps = e_step_output["log_w_eps"]

    # W_list: prefer the one carried by the E-step output if present, else
    # rebuild from the NaN pattern of Y.  See "W_list propagation" in the
    # docstring for why this is harmless.
    W_list = e_step_output.get("W_list", None)
    if W_list is None:
        from kalman import build_all_selection_matrices
        W_list = build_all_selection_matrices(Y)

    T = Y.shape[0]
    r = f_smooth.shape[1] // 5
    if 5 * r != f_smooth.shape[1]:
        raise ValueError(
            f"f_smooth has {f_smooth.shape[1]} columns; expected a multiple "
            f"of 5 (augmented state dimension 5r).  r inferred = {r}."
        )

    # ── 1. Observation pair (sequential: Lambda first, then R at new Lambda)
    #    -- See "Observation pair" in the docstring above. --
    Lambda_new = update_Lambda(
        Y=Y,
        f_smooth=f_smooth,
        P_smooth=P_smooth,
        w_eps=w_eps,
        W_list=W_list,
        block_map=block_map,
        freq_list=freq_list,
        ordered_cols=ordered_cols,
        r=r,
    )
    R_new = update_R(
        Y=Y,
        f_smooth=f_smooth,
        P_smooth=P_smooth,
        Lambda_new=Lambda_new,           # uses Lambda^(j+1), not Lambda^(j)
        w_eps=w_eps,
        W_list=W_list,
        block_map=block_map,
        freq_list=freq_list,
        ordered_cols=ordered_cols,
        r=r,
    )

    # ── 2. Transition pair (sequential: A first, then Q at new A) ─────────
    #    The number of transitions actually summed in compute_weighted_moments
    #    is T - 1 (the loop runs over range(1, T)); this is the same T_eff
    #    that appears in the -T/2 log|Q| prefactor of the state-equation
    #    log-likelihood (thesis "Normalisation", riga ~6094).
    moments = compute_weighted_moments(f_smooth, P_smooth, P_lag, w_u, r)
    A_new, Q_new = update_A_Q(
        P00=moments["P00"],
        P10=moments["P10"],
        P11=moments["P11"],
        T_eff=T - 1,
    )

    # ── 3. Degrees of freedom (independent ECME steps; optional freeze) ───
    # The freeze branch also serves the Gaussian estimator, which calls
    # run_m_step with freeze_nu_iters > max_iter (so this branch is always
    # taken) and nu_u = nu_eps = inf.  For the Gaussian estimator nu is not a
    # free parameter to be estimated but is fixed at infinity by definition of
    # the model.  The nu-update (Brent root-finding on the ECME first-order
    # condition) is therefore skipped: there is no finite stationary point to
    # solve for when the weights are identically 1 — the FOC g(nu) -> 0 only as
    # nu -> infinity, which is the boundary the Gaussian limit already sits on.
    if current_iter < freeze_nu_iters:
        # Freeze both nu's at their previous-iteration values.  Per the
        # docstring of update_nu, the FOC has the same functional form for
        # u and eps, so the same conditional logic applies to both.
        # (For the Gaussian estimator these previous values are inf, and
        # float(inf) = inf propagates forward cleanly.)
        nu_u_new   = float(theta_old["nu_u"])
        nu_eps_new = float(theta_old["nu_eps"])
    else:
        # Posterior summaries: simple time-averages over all T periods.
        # The boundary entries at t = 0 (w_u[0] = 1.0 = prior mean,
        # log_w_u[0] = psi(nu_u/2) - log(nu_u/2)) are included in the
        # mean — this is consistent with the thesis sum at riga ~6405 and
        # has negligible numerical effect (1 period out of T = 497).
        w_u_bar       = float(np.mean(w_u))
        log_w_u_bar   = float(np.mean(log_w_u))
        w_eps_bar     = float(np.mean(w_eps))
        log_w_eps_bar = float(np.mean(log_w_eps))
        nu_u_new   = update_nu(w_u_bar,   log_w_u_bar,   nu_bounds=nu_bounds)
        nu_eps_new = update_nu(w_eps_bar, log_w_eps_bar, nu_bounds=nu_bounds)

    # ── 4. Sigma_0: NOT updated by the unified M-step (subsec:full-m-step).
    #    Carry forward the previous value.  See "Why Sigma_0 is held fixed"
    #    in the docstring above. ─────────────────────────────────────────────
    Sigma_0_new = np.asarray(theta_old["Sigma_0"]).copy()

    # ── 5. Assemble theta_new with the same keys as theta_old. ─────────────
    #    We start from a shallow copy of theta_old to preserve every key
    #    (Sigma_0, F, w_u, w_eps, ...) and overwrite the updated entries.
    #    Using a dict comprehension makes the routine robust to both Python
    #    dicts and NpzFile objects (the latter does not implement .copy()
    #    in the dict sense).
    theta_new: dict = {key: np.asarray(theta_old[key]) for key in theta_old.keys()}
    theta_new["Lambda"]  = Lambda_new
    theta_new["A"]       = A_new
    theta_new["Q"]       = Q_new
    theta_new["R"]       = R_new
    theta_new["nu_u"]    = nu_u_new
    theta_new["nu_eps"]  = nu_eps_new
    theta_new["Sigma_0"] = Sigma_0_new

    # Propagate the latest E-step weights / smoothed factors into the
    # auxiliary keys carried by theta_initial.npz, when present.  These
    # entries are not strictly parameters of the model (they are E-step
    # outputs), but keeping them in sync with the latest iteration is a
    # convenience: it lets a caller introspect the most-recent posterior
    # state of the model directly from theta_new without re-running the
    # E-step.
    if "w_u" in theta_new:
        theta_new["w_u"] = w_u.copy()
    if "w_eps" in theta_new:
        theta_new["w_eps"] = w_eps.copy()
    if "F" in theta_new:
        theta_new["F"] = f_smooth[:, :r].copy()

    return theta_new


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    import sys

    import pandas as pd

    # ── parse config flag ─────────────────────────────────────────────────────
    _src_dir = str(pathlib.Path(__file__).resolve().parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from config_utils import parse_config_args, resolve_output_path, get_project_root

    _args = parse_config_args("em_m_step self-test — M-step parameter updates.")
    _cfg  = _args.config

    # ── Locate project root & make sibling modules importable ────────────────
    project_root = get_project_root()
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from data_loader       import load_config as _dl_load_config  # noqa: E402
    from em_e_step         import run_e_step                      # noqa: E402
    from em_initialization import load_standardized_data          # noqa: E402
    from kalman            import build_all_selection_matrices     # noqa: E402

    _cfg_dict    = _dl_load_config(_cfg)
    BLOCK        = _cfg_dict["BLOCK"]
    FREQ         = _cfg_dict["FREQ"]
    ORDERED_COLS = _cfg_dict["ORDERED_COLS"]

    npz_path  = resolve_output_path("processed", "theta_initial.npz", _cfg)
    csv_path  = resolve_output_path("dataset", "", _cfg)
    meta_path = resolve_output_path("processed", "theta_initial_metadata.json", _cfg)

    print(f"Loading theta^(0) from: {npz_path}")
    theta = np.load(npz_path)

    # Y is loaded *standardised*, NaN preserved.
    Y, mean_, std_, series_names = load_standardized_data(
        dataset_path=str(csv_path),
        metadata_path=str(meta_path),
    )
    freq_list = [FREQ[name] for name in series_names]
    T, M = Y.shape
    r = int(theta["A"].shape[0])
    print(f"Y shape: T={T}, M={M}   r={r}  [standardised, NaN preserved]")
    print(f"  mean/std consistency vs theta_initial_metadata.json: PASSED")

    # ── 1. Run the E-step at theta^(0) to obtain the inputs ──────────────────
    print("\nRunning E-step (verbose=False) to obtain smoothed moments + weights ...")
    estep = run_e_step(Y, theta, freq_list=freq_list, verbose=False)
    f_smooth = estep["f_smooth"]                # (T, 5r)
    P_smooth = estep["P_smooth"]                # (T, 5r, 5r)
    P_lag    = estep["P_lag"]                   # (T, 5r, 5r)
    w_u      = estep["w_u"]                     # (T,)
    w_eps    = estep["w_eps"]                   # (T,)  needed for Lambda
    print(f"  E-step converged in {estep['n_inner_iter']} inner iterations  "
          f"(loglik = {estep['loglik']:.2f})")
    print(f"  w_u   range: [{w_u.min():.4f}, {w_u.max():.4f}]   "
          f"w_u[0] = {w_u[0]:.4f} (prior mean)")
    print(f"  w_eps range: [{w_eps.min():.4f}, {w_eps.max():.4f}]")

    # ── 2. Compute weighted second moments ───────────────────────────────────
    print("\n" + "=" * 64)
    print("compute_weighted_moments")
    print("=" * 64)
    moments = compute_weighted_moments(f_smooth, P_smooth, P_lag, w_u, r)
    P00 = moments["P00"]
    P10 = moments["P10"]
    P11 = moments["P11"]

    # ── 3. Shape / finiteness assertions ─────────────────────────────────────
    assert P00.shape == (r, r), f"P00.shape = {P00.shape}, expected ({r},{r})"
    assert P10.shape == (r, r), f"P10.shape = {P10.shape}, expected ({r},{r})"
    assert P11.shape == (r, r), f"P11.shape = {P11.shape}, expected ({r},{r})"
    assert np.all(np.isfinite(P00)), "P00 contains NaN/inf"
    assert np.all(np.isfinite(P10)), "P10 contains NaN/inf"
    assert np.all(np.isfinite(P11)), "P11 contains NaN/inf"
    print(f"[OK] shapes all ({r},{r})   no NaN/inf in any matrix")

    # ── 4. Symmetry of P00 and P11 (P10 is NOT symmetric in general) ─────────
    sym_err_00 = float(np.max(np.abs(P00 - P00.T)))
    sym_err_11 = float(np.max(np.abs(P11 - P11.T)))
    asym_P10   = float(np.max(np.abs(P10 - P10.T)))
    assert sym_err_00 < 1e-10, f"P00 not symmetric: max|P00 - P00.T| = {sym_err_00:.3e}"
    assert sym_err_11 < 1e-10, f"P11 not symmetric: max|P11 - P11.T| = {sym_err_11:.3e}"
    print(f"[OK] P00 symmetric   max|P00 - P00.T| = {sym_err_00:.2e}")
    print(f"[OK] P11 symmetric   max|P11 - P11.T| = {sym_err_11:.2e}")
    print(f"[OK] P10 NOT symmetric (expected)   max|P10 - P10.T| = {asym_P10:.4f}")

    # ── 5. Positive-definiteness of P00 and P11 ──────────────────────────────
    eig_00 = np.linalg.eigvalsh(0.5 * (P00 + P00.T))
    eig_11 = np.linalg.eigvalsh(0.5 * (P11 + P11.T))
    assert eig_00.min() > 0, f"P00 not positive-definite: min eig = {eig_00.min():.3e}"
    assert eig_11.min() > 0, f"P11 not positive-definite: min eig = {eig_11.min():.3e}"
    print(f"[OK] P00 positive-definite   eigvals = {eig_00}")
    print(f"[OK] P11 positive-definite   eigvals = {eig_11}")

    # ── 6. Consistency check: internal-block P10 vs P_lag-form P10 ───────────
    # Verifies that the augmented state-space encodes the lag structure of
    # the monthly factors consistently — the analogue, for the M-step
    # sufficient statistics, of the d_u consistency test in the E-step.
    P10_alt = np.zeros((r, r))
    for t in range(1, T):
        f_t      = f_smooth[t][0:r]
        f_tm1_at = f_smooth[t - 1][0:r]                # contemporaneous block at t-1
        cov_lag  = P_lag[t][0:r, 0:r]                  # Cov(f_t, f_{t-1} | Y) via P_lag
        P10_alt += w_u[t] * (cov_lag + np.outer(f_t, f_tm1_at))

    diff_P10 = float(np.max(np.abs(P10 - P10_alt)))
    print(f"\n[OK] P10 (internal-block) vs P10 (P_lag form): "
          f"max|diff| = {diff_P10:.3e}")
    assert diff_P10 < 1e-8, (
        f"Internal-block vs P_lag mismatch in P10: max|diff| = {diff_P10:.3e}. "
        f"The augmented smoother may be inconsistent (re-run d_u test)."
    )

    # ── 7. Print the three matrices and the implied A^(j+1) ──────────────────
    np.set_printoptions(precision=4, suppress=True)

    print("\n" + "=" * 64)
    print("Weighted second-moment matrices")
    print("=" * 64)
    print(f"\nP00  (sum_t w_u[t] * E[f_{{t-1}} f_{{t-1}}' | Y]):")
    print(P00)
    print(f"\nP10  (sum_t w_u[t] * E[f_t f_{{t-1}}' | Y]):")
    print(P10)
    print(f"\nP11  (sum_t w_u[t] * E[f_t f_t' | Y]):")
    print(P11)

    # ── 8. Anticipo del prossimo step: A_implied = P10 @ inv(P00) ────────────
    A_implied = P10 @ np.linalg.inv(P00)
    eig_A = np.linalg.eigvals(A_implied)
    abs_eig_A = np.abs(eig_A)
    print("\n" + "=" * 64)
    print("Anticipo M-step:  A^(j+1) = P10 @ inv(P00)")
    print("=" * 64)
    print(f"\nA_implied:")
    print(A_implied)
    print(f"\nEigenvalues of A_implied: {eig_A}")
    print(f"|eigenvalues|             : {abs_eig_A}")
    print(f"Spectral radius           : {abs_eig_A.max():.4f}   "
          f"(< 1 means a stable VAR(1))")

    # ── 9. Comparison with theta^(0) A for context ───────────────────────────
    A0 = theta["A"]
    eig_A0 = np.abs(np.linalg.eigvals(A0))
    print(f"\nFor comparison, theta^(0) A:")
    print(A0)
    print(f"|eigenvalues| of A^(0)   : {eig_A0}")
    print(f"Spectral radius of A^(0) : {eig_A0.max():.4f}")

    print("\n" + "=" * 64)
    print("compute_weighted_moments test passed.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #                       TASK 2 — update_Lambda
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("update_Lambda  (block-restricted, mixed-frequency)")
    print("=" * 64)

    # ── Build the inputs not already available in this scope ─────────────────
    # block_map  : dict series-name -> block        (BLOCK)
    # freq_list  : list[str] aligned with columns of Y, in ORDERED_COLS order
    # W_list     : selection matrices (kept for API consistency)
    block_map    = BLOCK
    freq_list_M  = [FREQ[c] for c in ORDERED_COLS]    # (M,)
    W_list       = build_all_selection_matrices(Y)
    assert series_names == ORDERED_COLS, (
        "Column order of Y must match data_loader.ORDERED_COLS"
    )

    Lambda_new = update_Lambda(
        Y=Y,
        f_smooth=f_smooth,
        P_smooth=P_smooth,
        w_eps=w_eps,
        W_list=W_list,
        block_map=block_map,
        freq_list=freq_list_M,
        ordered_cols=ORDERED_COLS,
        r=r,
    )

    # ── 1. Shape + finiteness ────────────────────────────────────────────────
    assert Lambda_new.shape == (M, r), (
        f"Lambda_new.shape = {Lambda_new.shape}, expected ({M}, {r})"
    )
    assert np.all(np.isfinite(Lambda_new)), "Lambda_new contains NaN/inf"
    print(f"[OK] shape = {Lambda_new.shape}   no NaN/inf")

    # ── 2. Block-diagonality: every off-block entry must be exactly zero ─────
    off_block_max = 0.0
    off_block_violations: list[tuple[str, int, int, float]] = []
    for i, col in enumerate(ORDERED_COLS):
        j_allowed = _BLOCK_TO_COL[block_map[col]]
        for jj in range(r):
            if jj != j_allowed and Lambda_new[i, jj] != 0.0:
                off_block_max = max(off_block_max, abs(Lambda_new[i, jj]))
                off_block_violations.append((col, i, jj, float(Lambda_new[i, jj])))
    assert off_block_max == 0.0, (
        f"Lambda_new is not block-diagonal: "
        f"max off-block |entry| = {off_block_max:.3e}\n"
        f"  first violations: {off_block_violations[:5]}"
    )
    print(f"[OK] Lambda_new is exactly block-diagonal "
          f"(max off-block |entry| = {off_block_max:.2e})")

    # ── 3. No NaN/inf among the on-block entries either (already covered by
    #       the assertion above, but be explicit) ──────────────────────────────
    on_block_vals = np.array([
        Lambda_new[i, _BLOCK_TO_COL[block_map[col]]]
        for i, col in enumerate(ORDERED_COLS)
    ])
    assert np.all(np.isfinite(on_block_vals)), "Some on-block loadings are NaN/inf"
    print(f"[OK] on-block loadings all finite   "
          f"range = [{on_block_vals.min():+.4f}, {on_block_vals.max():+.4f}]")

    # ── 4. Side-by-side comparison Lambda^(0) vs Lambda_new ───────────────────
    Lambda_0 = theta["Lambda"]
    factor_labels = ["f_R", "f_F", "f_X"]
    print("\n--- Lambda^(0) vs Lambda_new (on-block loadings only) ---")
    print(f"  {'Series':<22s}  {'block':<10s}  {'freq':<10s}  "
          f"{'Lambda^(0)':>11s}  {'Lambda_new':>11s}  {'delta':>10s}")
    print("  " + "-" * 80)
    for i, col in enumerate(ORDERED_COLS):
        b = block_map[col]
        j = _BLOCK_TO_COL[b]
        lam0 = Lambda_0[i, j]
        lam1 = Lambda_new[i, j]
        delta = lam1 - lam0
        print(
            f"  {col:<22s}  {b:<10s}  {freq_list_M[i]:<10s}  "
            f"{lam0:>+11.4f}  {lam1:>+11.4f}  {delta:>+10.4f}"
        )

    # ── 5. Full Lambda_new printed ───────────────────────────────────────────
    print("\n--- Lambda_new (M=20 rows × r=3 cols) ---")
    print(f"  {'Series':<22s}  {'f_R':>10s}  {'f_F':>10s}  {'f_X':>10s}  Block")
    print("  " + "-" * 72)
    for i, col in enumerate(ORDERED_COLS):
        b = block_map[col]
        print(
            f"  {col:<22s}  {Lambda_new[i, 0]:>+10.4f}  "
            f"{Lambda_new[i, 1]:>+10.4f}  {Lambda_new[i, 2]:>+10.4f}  {b}"
        )

    # ── 6. Spot-check: GDPC1 row (the only quarterly row) ────────────────────
    gdp_idx     = ORDERED_COLS.index("GDPC1")
    gdp_block_j = _BLOCK_TO_COL[block_map["GDPC1"]]   # 0 (real)
    print(
        f"\nGDPC1 spot-check:\n"
        f"  initial loading  Lambda^(0)[GDPC1, f_R] = {Lambda_0[gdp_idx, gdp_block_j]:+.6f}\n"
        f"  updated loading  Lambda_new [GDPC1, f_R] = {Lambda_new[gdp_idx, gdp_block_j]:+.6f}\n"
        f"  (the update uses the MM composite regressor phi^R_t and weights "
        f"w_eps, restricted to the {(~np.isnan(Y[:, gdp_idx])).sum()} observed "
        f"quarter-end months)"
    )

    print("\n" + "=" * 64)
    print("update_Lambda test passed.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #         DIAGNOSTICA SCALA — perche' Lambda monthly collassa?
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Sintomo osservato dopo un M-step:
    #   - INDPRO  (real,      monthly): 0.40 -> ~0.0006
    #   - PAYEMS  (real,      monthly): 0.38 -> ~0.0003
    #   - CPIAUCSL(other,     monthly): 0.68 -> ~0.0001
    #   - BAAFFM  (financial, monthly): 0.50 -> ~1.155  (esplosa)
    #
    # Ipotesi:  E[(f^k_t)^2 | Y] = P_smooth[t,j,j] + f_smooth[t,j]^2
    # potrebbe essere molto piu' grande della corrispondente quantita'
    # calcolata sui fattori PCA originali F (usati in compute_theta_initial),
    # facendo collassare il rapporto num/den.
    #
    # Stampiamo, per 3 serie campione (una per ciascun block), la scala dei
    # regressori, il numeratore e denominatore della scalar OLS, e
    # confrontiamo con la scala dei fattori PCA originali F.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("DIAGNOSTICA Lambda — scala regressori (f_smooth vs F_pca)")
    print("=" * 64)

    # ── 0. Recupero dei fattori PCA originali F dal theta_initial.npz ────────
    F_pca = theta["F"]   # (T, r) — fattori PCA usati in compute_theta_initial
    assert F_pca.shape == (T, r), f"F_pca.shape={F_pca.shape}, atteso ({T},{r})"

    # ── 1. Confronto scala globale F_pca vs f_smooth vs diag(P_smooth) ───────
    print("\n--- Scala globale dei regressori (per ciascun blocco j) ---")
    print(
        f"  {'block':<10s}  {'j':>2s}  "
        f"{'std(F_pca[:,j])':>17s}  "
        f"{'std(f_smooth[:,j])':>20s}  "
        f"{'mean(P_smooth[t,j,j])':>22s}  "
        f"{'corr(F_pca,f_smooth)':>22s}"
    )
    print("  " + "-" * 100)
    for j in range(r):
        block_name = _BLOCK_ORDER[j]
        std_F_pca   = float(np.std(F_pca[:, j], ddof=1))
        std_f_sm    = float(np.std(f_smooth[:, j], ddof=1))
        mean_P_diag = float(np.mean(P_smooth[:, j, j]))
        corr        = float(np.corrcoef(F_pca[:, j], f_smooth[:, j])[0, 1])
        print(
            f"  {block_name:<10s}  {j:>2d}  "
            f"{std_F_pca:>17.6f}  "
            f"{std_f_sm:>20.6f}  "
            f"{mean_P_diag:>22.6f}  "
            f"{corr:>+22.6f}"
        )

    # ── 2. Per 3 serie campione: decomposizione num / den ────────────────────
    target_series = ["INDPRO", "BAAFFM", "CPIAUCSL"]
    print("\n--- Decomposizione num/den per 3 serie campione (monthly) ---")
    for name in target_series:
        if name not in ORDERED_COLS:
            print(f"\n  [SKIP] '{name}' non e' in ORDERED_COLS")
            continue
        i    = ORDERED_COLS.index(name)
        b    = block_map[name]
        j    = _BLOCK_TO_COL[b]
        freq = freq_list_M[i]

        obs_mask = ~np.isnan(Y[:, i])
        obs_t    = np.where(obs_mask)[0]
        y_i      = Y[obs_t, i]
        w_i      = w_eps[obs_t]
        E_f      = f_smooth[obs_t, j]                  # E[f^k_t | Y]
        V_f      = P_smooth[obs_t, j, j]               # Var(f^k_t | Y)
        E_f2     = V_f + E_f ** 2                      # E[(f^k_t)^2 | Y]

        num   = float(np.sum(w_i * y_i * E_f))
        den   = float(np.sum(w_i * E_f2))
        ratio = num / den

        # Per confronto, calcolo l'analoga OLS sui fattori PCA originali
        # (usando le STESSE osservazioni obs_t, NO posterior covariance,
        # NO weights w_eps — replica esattamente compute_theta_initial
        # ristretta agli obs_t per isolare il solo effetto scala).
        f_pca_obs = F_pca[obs_t, j]
        num_pca   = float(np.sum(y_i * f_pca_obs))
        den_pca   = float(np.sum(f_pca_obs ** 2))
        ratio_pca = num_pca / den_pca

        lam0 = float(theta["Lambda"][i, j])
        lam1 = float(Lambda_new[i, j])

        corr_Y_f      = float(np.corrcoef(y_i, E_f)[0, 1])
        corr_Y_fpca   = float(np.corrcoef(y_i, f_pca_obs)[0, 1])
        mean_Vf       = float(V_f.mean())
        mean_Ef2_pt   = float((E_f ** 2).mean())
        ratio_Vf_Ef2  = mean_Vf / mean_Ef2_pt if mean_Ef2_pt > 0 else np.inf

        print(f"\n  >>  Serie: {name:<10s}  block={b:<10s} (j={j})   "
              f"freq={freq:<10s}  n_obs={obs_t.size}")
        print(f"     Scala Y[obs_t, i]   : mean={y_i.mean():+10.6f}   "
              f"std={y_i.std(ddof=1):>10.6f}")
        print(f"     Scala E[f^k_t]      : mean={E_f.mean():+10.6f}   "
              f"std={E_f.std(ddof=1):>10.6f}")
        print(f"     Scala F_pca[:,j]    : mean={f_pca_obs.mean():+10.6f}   "
              f"std={f_pca_obs.std(ddof=1):>10.6f}")
        print(f"     mean( V_f )         = {mean_Vf:.6f}      "
              f"mean( E_f^2 ) = {mean_Ef2_pt:.6f}      "
              f"ratio mean(V_f)/mean(E_f^2) = {ratio_Vf_Ef2:.4f}")
        print(f"     corr(Y , E[f^k])    = {corr_Y_f:+.4f}      "
              f"corr(Y , F_pca[:,j]) = {corr_Y_fpca:+.4f}")
        print(f"     -- Formula M-step (con f_smooth, V_f, w_eps) --")
        print(f"        num   = sum_t w*Y*E[f^k]            = {num:+.6e}")
        print(f"        den   = sum_t w*(V_f + E[f^k]^2)    = {den:+.6e}")
        print(f"        num/den (= Lambda^(1)_new)          = {ratio:+.6e}")
        print(f"     -- Formula PCA-init (con F_pca, no V, no w) --")
        print(f"        num_pca = sum_t Y * F_pca           = {num_pca:+.6e}")
        print(f"        den_pca = sum_t F_pca^2             = {den_pca:+.6e}")
        print(f"        num_pca/den_pca                     = {ratio_pca:+.6e}")
        print(f"     Lambda^(0) salvato in theta_initial    = {lam0:+.6f}")
        print(f"     Lambda_new restituito da update_Lambda = {lam1:+.6f}   "
              f"[ratio==lam1? {np.isclose(ratio, lam1)}]")

    # ── 3. Riepilogo: tabella sintetica per tutte e 3 le serie ───────────────
    print("\n--- Riepilogo (tutte le 3 serie, una riga ciascuna) ---")
    print(
        f"  {'Serie':<10s}  {'std(Y)':>8s}  {'std(F_pca)':>10s}  "
        f"{'std(f_sm)':>10s}  {'mean(V_f)':>10s}  "
        f"{'lam_PCA':>10s}  {'lam_smooth':>12s}  {'lam_smooth/lam_PCA':>20s}"
    )
    print("  " + "-" * 105)
    for name in target_series:
        if name not in ORDERED_COLS:
            continue
        i = ORDERED_COLS.index(name)
        j = _BLOCK_TO_COL[block_map[name]]
        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        y_i   = Y[obs_t, i]
        E_f   = f_smooth[obs_t, j]
        V_f   = P_smooth[obs_t, j, j]
        f_pca = F_pca[obs_t, j]
        w_i   = w_eps[obs_t]

        std_Y      = float(y_i.std(ddof=1))
        std_F_pca  = float(f_pca.std(ddof=1))
        std_f_sm   = float(E_f.std(ddof=1))
        mean_Vf    = float(V_f.mean())
        lam_pca    = float(np.sum(y_i * f_pca) / np.sum(f_pca ** 2))
        lam_smooth = float(np.sum(w_i * y_i * E_f) /
                           np.sum(w_i * (V_f + E_f ** 2)))
        rapp       = lam_smooth / lam_pca if lam_pca != 0 else np.nan
        print(
            f"  {name:<10s}  {std_Y:>8.4f}  {std_F_pca:>10.4f}  "
            f"{std_f_sm:>10.4f}  {mean_Vf:>10.4f}  "
            f"{lam_pca:>+10.4f}  {lam_smooth:>+12.6e}  {rapp:>+20.6e}"
        )

    print("\n" + "=" * 64)
    print("FINE diagnostica.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #                       TASK 3 — update_A_Q
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("update_A_Q  (sequential ECM: A first, then Q at the new A)")
    print("=" * 64)

    # Number of transitions actually summed in compute_weighted_moments:
    # the loop is for t in range(1, T) -> T-1 transitions.
    T_eff = T - 1
    print(f"\nT_eff = {T_eff}  (= T - 1; the number of f_{{t-1}} -> f_t "
          f"transitions summed)")

    A_new, Q_new = update_A_Q(P00=P00, P10=P10, P11=P11, T_eff=T_eff)

    # ── 1. Shape / finiteness assertions ─────────────────────────────────────
    assert A_new.shape == (r, r), f"A_new.shape = {A_new.shape}, expected ({r},{r})"
    assert Q_new.shape == (r, r), f"Q_new.shape = {Q_new.shape}, expected ({r},{r})"
    assert np.all(np.isfinite(A_new)), "A_new contains NaN/inf"
    assert np.all(np.isfinite(Q_new)), "Q_new contains NaN/inf"
    print(f"[OK] shapes A_new={A_new.shape}, Q_new={Q_new.shape}   no NaN/inf")

    # ── 2. Symmetry + positive-definiteness of Q_new ─────────────────────────
    sym_err_Q = float(np.max(np.abs(Q_new - Q_new.T)))
    assert sym_err_Q < 1e-12, (
        f"Q_new not symmetric after explicit symmetrisation: "
        f"max|Q - Q'| = {sym_err_Q:.3e}"
    )
    print(f"[OK] Q_new symmetric    max|Q_new - Q_new.T| = {sym_err_Q:.2e}")

    eig_Q = np.linalg.eigvalsh(Q_new)
    assert eig_Q.min() > 0, (
        f"Q_new not positive-definite: min eigenvalue = {eig_Q.min():.3e}"
    )
    print(f"[OK] Q_new positive-definite   eigvals = {eig_Q}")

    # ── 3. Stability of A_new: spectral radius < 1 ───────────────────────────
    eig_A_new     = np.linalg.eigvals(A_new)
    abs_eig_A_new = np.abs(eig_A_new)
    rho_A_new     = abs_eig_A_new.max()
    assert rho_A_new < 1.0, (
        f"A_new is NOT a stable VAR(1): spectral radius = {rho_A_new:.4f}"
    )
    print(f"[OK] A_new stable VAR(1)   spectral radius = {rho_A_new:.4f}")

    # ── 4. Cross-check vs the Task 1 "A_implied" computed with explicit inv ──
    # A_implied = P10 @ inv(P00) was printed earlier; verify that the
    # solve-based form returns the same A (up to round-off).
    diff_A = float(np.max(np.abs(A_new - A_implied)))
    print(f"[OK] A_new (solve form) vs A_implied (inv form)   "
          f"max|diff| = {diff_A:.3e}")
    assert diff_A < 1e-10, (
        f"solve-based A disagrees with inv-based A: max|diff| = {diff_A:.3e}"
    )

    # ── 5. Side-by-side comparison with theta^(0) ─────────────────────────────
    A0          = theta["A"]
    Q0          = theta["Q"]
    eig_A0      = np.abs(np.linalg.eigvals(A0))
    rho_A0      = eig_A0.max()
    diag_Q0     = np.diag(Q0)
    diag_Q_new  = np.diag(Q_new)

    np.set_printoptions(precision=4, suppress=True)
    print("\n--- A: theta^(0)  vs  A_new ---")
    print(f"A^(0):\n{A0}")
    print(f"\nA_new:\n{A_new}")
    print(f"\nspectral radius   A^(0) = {rho_A0:.4f}     "
          f"A_new = {rho_A_new:.4f}")

    print("\n--- Q: diag(Q^(0))  vs  diag(Q_new) ---")
    print(f"  {'factor':<6s}  {'diag(Q^(0))':>12s}  {'diag(Q_new)':>12s}  "
          f"{'ratio Q_new/Q^(0)':>20s}")
    print("  " + "-" * 60)
    for j, lab in enumerate(["f_R", "f_F", "f_X"]):
        rat = diag_Q_new[j] / diag_Q0[j] if diag_Q0[j] != 0 else np.nan
        print(f"  {lab:<6s}  {diag_Q0[j]:>+12.4f}  {diag_Q_new[j]:>+12.4f}  "
              f"{rat:>+20.4f}")

    print(f"\nFull Q^(0):\n{Q0}")
    print(f"\nFull Q_new:\n{Q_new}")

    # ── 6. Interpretation snippet ────────────────────────────────────────────
    # Q_new tipicamente piu' piccola di Q^(0): i pesi w_u down-weightano
    # le innovazioni outlier, quindi le varianze residue dei fattori sono
    # ridotte.  rho(A_new) ~ 0.96 conferma che il VAR resta vicino al
    # bordo della regione di stabilita' ma stabile.
    mean_diag_ratio = float(np.mean(diag_Q_new / diag_Q0))
    print(
        f"\nMean diagonal ratio Q_new/Q^(0) = {mean_diag_ratio:.4f}    "
        f"(<1: Student-t down-weighting riduce la varianza residua)"
    )

    print("\n" + "=" * 64)
    print("update_A_Q test passed.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #                       TASK 4 — update_R
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("update_R  (diagonal idiosyncratic variances, at the new Lambda)")
    print("=" * 64)

    R_new = update_R(
        Y=Y,
        f_smooth=f_smooth,
        P_smooth=P_smooth,
        Lambda_new=Lambda_new,           # j+1 loadings from Task 2
        w_eps=w_eps,
        W_list=W_list,
        block_map=block_map,
        freq_list=freq_list_M,
        ordered_cols=ORDERED_COLS,
        r=r,
    )

    # ── 1. Shape / finiteness / positivity assertions ────────────────────────
    assert R_new.shape == (M,), f"R_new.shape = {R_new.shape}, expected ({M},)"
    assert np.all(np.isfinite(R_new)), "R_new contains NaN/inf"
    assert np.all(R_new > 0), (
        f"R_new has non-positive entries (idiosyncratic variances must be > 0): "
        f"min = {R_new.min():.3e}, "
        f"argmin = '{ORDERED_COLS[int(np.argmin(R_new))]}'"
    )
    print(f"[OK] shape = {R_new.shape}   no NaN/inf   all entries > 0")
    print(f"[OK] R_new range: [{R_new.min():.4e}, {R_new.max():.4e}]   "
          f"mean = {R_new.mean():.4e}")

    # ── 2. Side-by-side comparison vs R^(0) ─────────────────────────────────
    # theta_initial stores R as either a (M,) vector or an (M, M) diagonal
    # matrix — accept both.
    R0_raw = theta["R"]
    if R0_raw.ndim == 2:
        R0 = np.diag(R0_raw)
    else:
        R0 = R0_raw
    assert R0.shape == (M,), f"R^(0).shape = {R0.shape}, expected ({M},)"

    print("\n--- R^(0)  vs  R_new  (per series) ---")
    print(f"  {'Series':<22s}  {'block':<10s}  {'freq':<10s}  "
          f"{'R^(0)':>11s}  {'R_new':>11s}  {'R_new/R^(0)':>13s}")
    print("  " + "-" * 84)
    for i, col in enumerate(ORDERED_COLS):
        b    = block_map[col]
        rat  = R_new[i] / R0[i] if R0[i] > 0 else np.nan
        print(
            f"  {col:<22s}  {b:<10s}  {freq_list_M[i]:<10s}  "
            f"{R0[i]:>11.4f}  {R_new[i]:>11.4f}  {rat:>+13.4f}"
        )

    # ── 3. Summary statistics ─────────────────────────────────────────────
    mean_R0     = float(np.mean(R0))
    mean_Rnew   = float(np.mean(R_new))
    median_rat  = float(np.median(R_new / R0))
    print(
        f"\nmean(R^(0))   = {mean_R0:.4f}     "
        f"mean(R_new)   = {mean_Rnew:.4f}     "
        f"median ratio  = {median_rat:.4f}"
    )

    # ── 4. Best- vs worst-explained series ─────────────────────────────────
    # Lower R = the common factor explains more of the series' variance
    # (since Y is standardised, var(Y_i) ~ 1, so signal share ~ 1 - R_i).
    order_asc  = np.argsort(R_new)
    print("\n--- Best 3 explained by the factor (lowest R_new) ---")
    for k in range(3):
        i = order_asc[k]
        col = ORDERED_COLS[i]
        print(f"  {col:<22s}  block={block_map[col]:<10s}  "
              f"freq={freq_list_M[i]:<10s}  R_new = {R_new[i]:.4f}")
    print("\n--- Worst 3 explained by the factor (highest R_new) ---")
    for k in range(3):
        i = order_asc[-(k + 1)]
        col = ORDERED_COLS[i]
        print(f"  {col:<22s}  block={block_map[col]:<10s}  "
              f"freq={freq_list_M[i]:<10s}  R_new = {R_new[i]:.4f}")

    # ── 5. Spot-check: GDPC1 (quarterly) ──────────────────────────────────
    gdp_idx = ORDERED_COLS.index("GDPC1")
    n_obs_gdp = int((~np.isnan(Y[:, gdp_idx])).sum())
    print(
        f"\nGDPC1 spot-check:\n"
        f"  R^(0)[GDPC1] = {R0[gdp_idx]:.4f}     "
        f"R_new[GDPC1] = {R_new[gdp_idx]:.4f}     "
        f"ratio = {R_new[gdp_idx]/R0[gdp_idx]:+.4f}\n"
        f"  (computed with composite phi^R_t and w_eps over "
        f"{n_obs_gdp} observed quarter-end months,\n"
        f"   normalised by 1/|T^Q_GDPC1| = 1/{n_obs_gdp} per the thesis "
        f"derivation eq:mm-RQ-update)"
    )

    # ── 6. Posterior-uncertainty correction: how much does it matter? ─────
    # Compare R_new (with the lam^2 * Var[f] correction) against the
    # "naive" point-residual estimator that drops the variance term.
    R_naive = np.zeros(M)
    for i in range(M):
        col = ORDERED_COLS[i]
        j   = _BLOCK_TO_COL[block_map[col]]
        lam = float(Lambda_new[i, j])
        obs_t = np.where(~np.isnan(Y[:, i]))[0]
        y_i   = Y[obs_t, i]
        w_i   = w_eps[obs_t]
        if freq_list_M[i] == "monthly":
            E_f = f_smooth[obs_t, j]
            resid = y_i - lam * E_f
        else:
            idx = np.array([l * r + j for l in range(5)])
            E_phi = f_smooth[obs_t][:, idx] @ _MM_WEIGHTS
            resid = y_i - lam * E_phi
        R_naive[i] = float(np.sum(w_i * resid ** 2)) / obs_t.size

    extra_share = (R_new - R_naive) / R_new   # share of R_new due to Var[f]
    print(
        f"\nPosterior-uncertainty correction (lam^2 * Var[f] term):\n"
        f"  median share in R_new: {float(np.median(extra_share)):.4f}     "
        f"max share: {float(np.max(extra_share)):.4f} "
        f"({ORDERED_COLS[int(np.argmax(extra_share))]})\n"
        f"  (dropping the Var[f] term would bias R downward by this share "
        f"on average)"
    )

    print("\n" + "=" * 64)
    print("update_R test passed.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #                       TASK 5 — update_nu
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("update_nu  (ECME step: Brent root-finding on the FOC)")
    print("=" * 64)

    # Posterior summaries.  We average over ALL t (= include t=0):
    #   - w_u[0] = 1.0       (prior mean, set by the E-step boundary
    #                         convention since d_u[0] is undefined)
    #   - log_w_u[0] = psi(nu_u^(j)/2) - log(nu_u^(j)/2)
    #                        (prior log-mean at the CURRENT nu_u)
    # The thesis sum (riga ~6405) is (1/T) sum_{t=1}^T = mean over all
    # T periods in 0-indexed Python, consistent with this convention.
    # The single boundary period out of T=497 contributes negligibly.
    log_w_u   = estep["log_w_u"]
    log_w_eps = estep["log_w_eps"]

    w_u_bar       = float(np.mean(w_u))
    log_w_u_bar   = float(np.mean(log_w_u))
    w_eps_bar     = float(np.mean(w_eps))
    log_w_eps_bar = float(np.mean(log_w_eps))

    print(f"\nPosterior summaries (averaged over all T={T} periods):")
    print(f"  factor-side:        w_u_bar     = {w_u_bar:.6f}   "
          f"log_w_u_bar     = {log_w_u_bar:+.6f}")
    print(f"  idiosyncratic-side: w_eps_bar   = {w_eps_bar:.6f}   "
          f"log_w_eps_bar   = {log_w_eps_bar:+.6f}")

    # ── Sanity: Jensen — bar{w} - overline{log w} >= 1 (>= 1 strictly
    #    unless weights are all identically 1; Eq. ~6787 of thesis).
    jensen_u   = w_u_bar   - log_w_u_bar
    jensen_eps = w_eps_bar - log_w_eps_bar
    assert jensen_u   >= 1.0 - 1e-12, f"Jensen violated for u: {jensen_u:.6e}"
    assert jensen_eps >= 1.0 - 1e-12, f"Jensen violated for eps: {jensen_eps:.6e}"
    print(f"\n[OK] Jensen gap u:   bar{{w}} - log_bar{{w}} = "
          f"{jensen_u:.6f}  (>= 1)")
    print(f"[OK] Jensen gap eps: bar{{w}} - log_bar{{w}} = "
          f"{jensen_eps:.6f}  (>= 1)")

    # ── Run the updates ──────────────────────────────────────────────────────
    # Bracket della tesi (riga ~6614): [2.001, 1000].  2.001 garantisce
    # nu > 2 (esistenza del secondo momento del Student-t); 1000 e' un
    # upper bound "essenzialmente gaussiano" per qualsiasi statistica
    # pratica.
    nu_bounds = (2.001, 1000.0)
    nu_u_new   = update_nu(w_u_bar,   log_w_u_bar,   nu_bounds=nu_bounds)
    nu_eps_new = update_nu(w_eps_bar, log_w_eps_bar, nu_bounds=nu_bounds)

    # ── 1. Shape/finiteness/bounds assertions ────────────────────────────────
    assert np.isfinite(nu_u_new)   and nu_u_new   > 2.0, (
        f"nu_u_new   = {nu_u_new} out of (2, inf)")
    assert np.isfinite(nu_eps_new) and nu_eps_new > 2.0, (
        f"nu_eps_new = {nu_eps_new} out of (2, inf)")
    assert nu_bounds[0] <= nu_u_new   <= nu_bounds[1]
    assert nu_bounds[0] <= nu_eps_new <= nu_bounds[1]
    print(f"\n[OK] nu_u_new   = {nu_u_new:.4f}   in [{nu_bounds[0]}, "
          f"{nu_bounds[1]}], > 2, finite")
    print(f"[OK] nu_eps_new = {nu_eps_new:.4f}   in [{nu_bounds[0]}, "
          f"{nu_bounds[1]}], > 2, finite")

    # ── 2. Verify the FOC at the returned roots (residual ~ 0) ───────────────
    def _g(nu: float, wb: float, lwb: float) -> float:
        return float(np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 + lwb - wb)

    res_u   = _g(nu_u_new,   w_u_bar,   log_w_u_bar)
    res_eps = _g(nu_eps_new, w_eps_bar, log_w_eps_bar)
    # If clamped to the boundary, g may not be exactly zero.
    if nu_u_new not in nu_bounds:
        assert abs(res_u) < 1e-6, f"FOC residual nu_u   = {res_u:.3e}"
        print(f"[OK] FOC residual at nu_u_new   : {res_u:+.3e}  (< 1e-6)")
    else:
        print(f"[INFO] nu_u_new clamped at bound  : g(nu_u_new) = {res_u:+.3e}")
    if nu_eps_new not in nu_bounds:
        assert abs(res_eps) < 1e-6, f"FOC residual nu_eps = {res_eps:.3e}"
        print(f"[OK] FOC residual at nu_eps_new : {res_eps:+.3e}  (< 1e-6)")
    else:
        print(f"[INFO] nu_eps_new clamped at bound: g(nu_eps_new) = {res_eps:+.3e}")

    # ── 3. Comparison vs nu^(0) = 10 (theta_initial default) ─────────────────
    nu_u_0   = float(theta["nu_u"])
    nu_eps_0 = float(theta["nu_eps"])
    print("\n--- Comparison: nu^(0)  vs  nu^(1) ---")
    print(f"  nu_u   : {nu_u_0:>7.2f}  ->  {nu_u_new:>7.4f}   "
          f"(delta = {nu_u_new - nu_u_0:+.4f})")
    print(f"  nu_eps : {nu_eps_0:>7.2f}  ->  {nu_eps_new:>7.4f}   "
          f"(delta = {nu_eps_new - nu_eps_0:+.4f})")
    print(
        "\nInterpretation:\n"
        "  * nu_new < nu^(0)  =>  code piu' spesse del previsto: i dati\n"
        "    contengono piu' outlier di quanto nu^(0) = 10 assumesse.\n"
        "    Tipico nel campione 1985-2026 per via di Covid (Apr 2020) e\n"
        "    delle crisi finanziarie (2008, 2020).\n"
        "  * nu_new > nu^(0)  =>  vicino al gaussiano: i pesi posteriori\n"
        "    sono concentrati intorno a 1, poca dispersione.\n"
        "  * Confronto delle posterior gaps (Delta_post = bar{w} - log_bar{w},\n"
        "    interpretazione concavity-gap-matching, tesi eq:nu-interpretation,\n"
        "    riga ~6755):\n"
        f"      factor-side       Delta_post(u)   = {jensen_u:.4f}\n"
        f"      idiosyncratic     Delta_post(eps) = {jensen_eps:.4f}\n"
        "    Il blocco con gap piu' grande -> nu piu' piccolo. La tesi\n"
        "    (riga ~6819) suggerisce in generale nu_eps > nu_u perche'\n"
        "    m_t (~20) > r (3): w_eps stimati con piu' informazione,\n"
        "    distribuzione posteriore piu' concentrata -> gap piu' piccolo.\n"
        "    NEL NOSTRO DATASET il pattern e' INVERTITO: gli outlier\n"
        "    idiosyncratici (Covid + crisi finanziarie) sono cosi' severi\n"
        "    che il loro gap supera quello factor-side, producendo\n"
        "    nu_eps < nu_u. Risultato empirico interessante: la non-\n"
        "    normalita' macro USA e' piu' visibile nei residui specifici\n"
        "    di serie che nelle innovazioni del fattore comune."
    )

    # ── 4. Sanity — Gaussian limit: w_bar = 1, log_w_bar = 0  =>  clamp to nu_max
    nu_gauss = update_nu(1.0, 0.0, nu_bounds=nu_bounds)
    assert nu_gauss == nu_bounds[1], (
        f"Gaussian-limit test failed: update_nu(1.0, 0.0) = {nu_gauss}, "
        f"expected nu_max = {nu_bounds[1]} (clamp)"
    )
    print(f"\n[OK] Gaussian-limit sanity: update_nu(w_bar=1.0, log_w_bar=0.0) "
          f"= {nu_gauss}   (clamp at nu_max = {nu_bounds[1]}; FOC satisfied "
          f"only at nu -> inf)")

    # ── 5. Sanity — extreme heavy-tail: large posterior gap  =>  clamp to nu_min
    # Construct artificial pesi con un gap molto grande: bar{w}=2, log_bar{w}=-5
    # =>  bar{w} - log_bar{w} = 7, ben sopra 1, FOC vuole nu molto piccolo.
    nu_heavy = update_nu(2.0, -5.0, nu_bounds=nu_bounds)
    print(f"[INFO] heavy-tail sanity (w_bar=2.0, log_w_bar=-5.0): "
          f"nu_new = {nu_heavy:.4f}   (should be at or near nu_min = "
          f"{nu_bounds[0]})")

    # ── 6. Plot di g(nu) sul bracket per nu_u (qualitativo) ───────────────────
    nu_grid = np.linspace(nu_bounds[0], nu_bounds[1], 100)
    g_grid  = np.array([_g(nu, w_u_bar, log_w_u_bar) for nu in nu_grid])
    # Verifica monotonia decrescente sulla griglia
    diffs = np.diff(g_grid)
    assert np.all(diffs < 0), (
        f"g(nu) non monotonamente decrescente sulla griglia: "
        f"max diff = {diffs.max():.3e}"
    )
    print(f"\n[OK] g_u(nu) strettamente decrescente sulla griglia "
          f"[{nu_bounds[0]}, {nu_bounds[1]}]   "
          f"(max diff = {diffs.max():+.3e}, deve essere < 0)")
    print(f"     g_u(nu_min={nu_bounds[0]})    = {g_grid[0]:+.6f}")
    print(f"     g_u(nu_max={nu_bounds[1]})    = {g_grid[-1]:+.6f}")
    print(f"     => root deve stare fra nu_min e nu_max (segni opposti): "
          f"{'OK' if g_grid[0] * g_grid[-1] < 0 else 'BOUND'}")

    print("\n" + "=" * 64)
    print("update_nu test passed.")
    print("=" * 64)

    # ─────────────────────────────────────────────────────────────────────────
    #                       TASK 6 — run_m_step (full wrapper)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("run_m_step  (Task 6 — full ECM M-step wrapper)")
    print("=" * 64)
    print(
        "  Orchestrates Tasks 1-5 in the correct ECM order:\n"
        "    1) observation pair :  Lambda  ->  R   (R at Lambda^(j+1))\n"
        "    2) transition  pair :  A       ->  Q   (Q at A^(j+1))\n"
        "    3) ECME step        :  nu_u, nu_eps    (independent; freezable)\n"
        "    Sigma_0 carried forward unchanged (thesis subsec:full-m-step).\n"
    )

    # The E-step output is already in scope as ``estep`` from the earlier
    # section.  We pass it through run_m_step together with theta^(0).
    theta_new = run_m_step(
        Y=Y,
        e_step_output=estep,
        theta_old=theta,
        freq_list=freq_list_M,
        block_map=block_map,
        ordered_cols=ORDERED_COLS,
        freeze_nu_iters=0,        # default: do update nu from iter 0
        current_iter=0,
        nu_bounds=nu_bounds,
    )

    # ── 1. Required keys are present ─────────────────────────────────────────
    required_keys = {"Lambda", "A", "Q", "R", "nu_u", "nu_eps", "Sigma_0"}
    missing = required_keys - set(theta_new.keys())
    assert not missing, f"theta_new is missing required keys: {missing}"
    print(f"[OK] theta_new contains all required keys: {sorted(required_keys)}")
    print(f"     full key set ({len(theta_new)} keys): {sorted(theta_new.keys())}")

    # ── 2. Shapes ─────────────────────────────────────────────────────────────
    assert theta_new["Lambda"].shape  == (M, r),     f"Lambda shape  = {theta_new['Lambda'].shape}"
    assert theta_new["A"].shape       == (r, r),     f"A shape       = {theta_new['A'].shape}"
    assert theta_new["Q"].shape       == (r, r),     f"Q shape       = {theta_new['Q'].shape}"
    assert theta_new["R"].shape       == (M,),       f"R shape       = {theta_new['R'].shape}"
    assert theta_new["Sigma_0"].shape == (5 * r, 5 * r), (
        f"Sigma_0 shape = {theta_new['Sigma_0'].shape}"
    )
    assert np.ndim(theta_new["nu_u"])   == 0, "nu_u must be a scalar"
    assert np.ndim(theta_new["nu_eps"]) == 0, "nu_eps must be a scalar"
    print(f"[OK] all shapes correct  "
          f"(Lambda {theta_new['Lambda'].shape}, A {theta_new['A'].shape}, "
          f"Q {theta_new['Q'].shape}, R {theta_new['R'].shape}, "
          f"Sigma_0 {theta_new['Sigma_0'].shape})")

    # ── 3. Finiteness (no NaN/inf anywhere) ──────────────────────────────────
    for k in ("Lambda", "A", "Q", "R", "Sigma_0"):
        assert np.all(np.isfinite(theta_new[k])), f"{k} contains NaN/inf"
    assert np.isfinite(theta_new["nu_u"])   and np.isfinite(theta_new["nu_eps"])
    print(f"[OK] no NaN/inf in any field of theta_new")

    # ── 4. Structural properties of each block ────────────────────────────────
    # 4a. Lambda block-diagonal
    off_block_max = 0.0
    for i, col in enumerate(ORDERED_COLS):
        j_allowed = _BLOCK_TO_COL[block_map[col]]
        for jj in range(r):
            if jj != j_allowed:
                off_block_max = max(off_block_max, abs(theta_new["Lambda"][i, jj]))
    assert off_block_max == 0.0, (
        f"Lambda_new not block-diagonal: max off-block |entry| = {off_block_max:.3e}"
    )
    print(f"[OK] Lambda block-diagonal  (max off-block |entry| = {off_block_max:.2e})")

    # 4b. A stable VAR(1) (spectral radius < 1)
    rho_A_new = float(np.max(np.abs(np.linalg.eigvals(theta_new["A"]))))
    assert rho_A_new < 1.0, f"A not stable: spectral radius = {rho_A_new:.4f}"
    print(f"[OK] A stable VAR(1)        spectral radius = {rho_A_new:.4f}  (< 1)")

    # 4c. Q symmetric PD
    Qn = theta_new["Q"]
    sym_err_Q = float(np.max(np.abs(Qn - Qn.T)))
    eig_Q_new = np.linalg.eigvalsh(Qn)
    assert sym_err_Q < 1e-12,           f"Q not symmetric: {sym_err_Q:.3e}"
    assert eig_Q_new.min() > 0,         f"Q not PD: min eigenvalue = {eig_Q_new.min():.3e}"
    print(f"[OK] Q symmetric PD          min eigenvalue = {eig_Q_new.min():.4e}")

    # 4d. R strictly positive
    assert np.all(theta_new["R"] > 0), (
        f"R has non-positive entries: min = {theta_new['R'].min():.3e}"
    )
    print(f"[OK] R > 0 entry-wise       min = {theta_new['R'].min():.4f}, "
          f"max = {theta_new['R'].max():.4f}")

    # 4e. nu_u, nu_eps inside the bracket
    assert nu_bounds[0] <= theta_new["nu_u"]   <= nu_bounds[1]
    assert nu_bounds[0] <= theta_new["nu_eps"] <= nu_bounds[1]
    assert theta_new["nu_u"]   > 2.0 and theta_new["nu_eps"] > 2.0
    print(f"[OK] nu_u={float(theta_new['nu_u']):.4f}, "
          f"nu_eps={float(theta_new['nu_eps']):.4f}   "
          f"in [{nu_bounds[0]}, {nu_bounds[1]}], > 2")

    # 4f. Sigma_0 unchanged (identity check vs theta_old)
    Sigma_0_old = np.asarray(theta["Sigma_0"])
    diff_Sigma0 = float(np.max(np.abs(theta_new["Sigma_0"] - Sigma_0_old)))
    assert diff_Sigma0 == 0.0, (
        f"Sigma_0 changed but should be fixed: max|diff| = {diff_Sigma0:.3e}"
    )
    print(f"[OK] Sigma_0 carried forward unchanged  "
          f"(max|Sigma_0_new - Sigma_0_old| = {diff_Sigma0:.2e})")

    # ── 5. Cross-check: run_m_step output coincides with the per-block calls ─
    # The wrapper must produce *exactly* the same numbers as the per-block
    # tests above (Lambda_new, A_new, Q_new, R_new, nu_u_new, nu_eps_new from
    # the earlier sections of this self-test).  This guards against silent
    # ordering or argument bugs in the wrapper.
    diff_Lambda = float(np.max(np.abs(theta_new["Lambda"] - Lambda_new)))
    diff_A      = float(np.max(np.abs(theta_new["A"]      - A_new)))
    diff_Q      = float(np.max(np.abs(theta_new["Q"]      - Q_new)))
    diff_R      = float(np.max(np.abs(theta_new["R"]      - R_new)))
    diff_nu_u   = abs(float(theta_new["nu_u"])   - nu_u_new)
    diff_nu_eps = abs(float(theta_new["nu_eps"]) - nu_eps_new)
    assert diff_Lambda < 1e-12, f"Lambda mismatch wrapper vs direct: {diff_Lambda:.3e}"
    assert diff_A      < 1e-12, f"A mismatch wrapper vs direct:      {diff_A:.3e}"
    assert diff_Q      < 1e-12, f"Q mismatch wrapper vs direct:      {diff_Q:.3e}"
    assert diff_R      < 1e-12, f"R mismatch wrapper vs direct:      {diff_R:.3e}"
    assert diff_nu_u   < 1e-12, f"nu_u mismatch wrapper vs direct:   {diff_nu_u:.3e}"
    assert diff_nu_eps < 1e-12, f"nu_eps mismatch wrapper vs direct: {diff_nu_eps:.3e}"
    print(f"[OK] wrapper output identical to direct per-block calls  "
          f"(max diff = {max(diff_Lambda, diff_A, diff_Q, diff_R, diff_nu_u, diff_nu_eps):.2e})")

    # ── 6. CONSOLIDATED COMPARISON: theta^(0)  vs  theta^(1) ─────────────────
    A0          = theta["A"]
    Q0          = theta["Q"]
    R0_raw      = theta["R"]
    R0          = np.diag(R0_raw) if R0_raw.ndim == 2 else R0_raw
    Lambda_0    = theta["Lambda"]
    nu_u_0      = float(theta["nu_u"])
    nu_eps_0    = float(theta["nu_eps"])

    rho_A_0     = float(np.max(np.abs(np.linalg.eigvals(A0))))
    mean_diagQ0 = float(np.mean(np.diag(Q0)))
    mean_diagQn = float(np.mean(np.diag(theta_new["Q"])))
    mean_R0     = float(np.mean(R0))
    mean_R_new  = float(np.mean(theta_new["R"]))
    # On-block loadings only (off-block entries are exactly zero in both):
    on_block_0  = np.array([
        Lambda_0[i, _BLOCK_TO_COL[block_map[c]]] for i, c in enumerate(ORDERED_COLS)
    ])
    on_block_n  = np.array([
        theta_new["Lambda"][i, _BLOCK_TO_COL[block_map[c]]]
        for i, c in enumerate(ORDERED_COLS)
    ])
    mean_abs_lam_0 = float(np.mean(np.abs(on_block_0)))
    mean_abs_lam_n = float(np.mean(np.abs(on_block_n)))

    print("\n" + "-" * 72)
    print(f"  CONSOLIDATED UPDATE TABLE   theta^(0)  ->  theta^(1)")
    print("-" * 72)
    print(f"  {'quantity':<35s}  {'theta^(0)':>14s}  {'theta^(1)':>14s}")
    print("-" * 72)
    print(f"  {'spectral radius A':<35s}  {rho_A_0:>14.4f}  {rho_A_new:>14.4f}")
    print(f"  {'mean diag(Q)':<35s}  {mean_diagQ0:>14.4f}  {mean_diagQn:>14.4f}")
    print(f"  {'mean R':<35s}  {mean_R0:>14.4f}  {mean_R_new:>14.4f}")
    print(f"  {'mean |on-block Lambda|':<35s}  {mean_abs_lam_0:>14.4f}  {mean_abs_lam_n:>14.4f}")
    print(f"  {'nu_u':<35s}  {nu_u_0:>14.4f}  {float(theta_new['nu_u']):>14.4f}")
    print(f"  {'nu_eps':<35s}  {nu_eps_0:>14.4f}  {float(theta_new['nu_eps']):>14.4f}")
    print("-" * 72)

    # ── 7. TEST FREEZE_NU: con freeze_nu_iters=5, current_iter=0 ──────────────
    # nu_u_new e nu_eps_new devono essere ESATTAMENTE quelli di theta_old (10).
    theta_frozen = run_m_step(
        Y=Y,
        e_step_output=estep,
        theta_old=theta,
        freq_list=freq_list_M,
        block_map=block_map,
        ordered_cols=ORDERED_COLS,
        freeze_nu_iters=5,
        current_iter=0,
        nu_bounds=nu_bounds,
    )
    assert float(theta_frozen["nu_u"])   == nu_u_0, (
        f"freeze_nu: nu_u   = {float(theta_frozen['nu_u'])} != theta_old nu_u = {nu_u_0}"
    )
    assert float(theta_frozen["nu_eps"]) == nu_eps_0, (
        f"freeze_nu: nu_eps = {float(theta_frozen['nu_eps'])} != theta_old nu_eps = {nu_eps_0}"
    )
    # Also: the other blocks SHOULD still be updated even with freeze_nu_iters>0.
    diff_Lambda_frozen = float(np.max(np.abs(theta_frozen["Lambda"] - theta_new["Lambda"])))
    diff_A_frozen      = float(np.max(np.abs(theta_frozen["A"]      - theta_new["A"])))
    assert diff_Lambda_frozen < 1e-12, "freeze_nu should not change Lambda"
    assert diff_A_frozen      < 1e-12, "freeze_nu should not change A"
    print(f"\n[OK] freeze_nu_iters=5, current_iter=0:  "
          f"nu_u = {float(theta_frozen['nu_u']):.2f} (frozen),  "
          f"nu_eps = {float(theta_frozen['nu_eps']):.2f} (frozen)")
    print(f"     (Lambda, A, Q, R still updated normally — only nu's are frozen)")

    # ── 8. Sanity:  current_iter >= freeze_nu_iters releases the freeze ──────
    theta_unfrozen = run_m_step(
        Y=Y,
        e_step_output=estep,
        theta_old=theta,
        freq_list=freq_list_M,
        block_map=block_map,
        ordered_cols=ORDERED_COLS,
        freeze_nu_iters=5,
        current_iter=5,            # = freeze_nu_iters -> update from here
        nu_bounds=nu_bounds,
    )
    assert abs(float(theta_unfrozen["nu_u"])   - float(theta_new["nu_u"]))   < 1e-12
    assert abs(float(theta_unfrozen["nu_eps"]) - float(theta_new["nu_eps"])) < 1e-12
    print(f"[OK] freeze_nu_iters=5, current_iter=5:  nu's updated normally "
          f"(matches the unfrozen baseline)")

    # ── 9. Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("M-step completo - theta^(1) pronto per la prossima iterazione EM")
    print("=" * 64)

    # ── 10. Final summary: complete structure of em_m_step.py ────────────────
    print("\nStruttura completa di em_m_step.py (in ordine):")
    print("-" * 72)
    print("  1. compute_weighted_moments(f_smooth, P_smooth, P_lag, w_u, r)")
    print("     -> dict{P00, P10, P11} : weighted posterior second moments of")
    print("        the monthly factors (sufficient statistics for A, Q).")
    print()
    print("  2. update_Lambda(Y, f_smooth, P_smooth, w_eps, W_list, block_map,")
    print("                   freq_list, ordered_cols, r)")
    print("     -> Lambda_new (M, r)  : block-restricted, mixed-frequency")
    print("        scalar weighted-OLS row-by-row update of the loadings.")
    print()
    print("  3. update_A_Q(P00, P10, P11, T_eff)")
    print("     -> (A_new, Q_new)     : closed-form sequential ECM update of")
    print("        the VAR transition matrix and innovation covariance.")
    print()
    print("  4. update_R(Y, f_smooth, P_smooth, Lambda_new, w_eps, W_list,")
    print("              block_map, freq_list, ordered_cols, r)")
    print("     -> R_new (M,)         : closed-form per-series update of the")
    print("        diagonal idiosyncratic variances at the new Lambda.")
    print()
    print("  5. update_nu(w_bar, log_w_bar, nu_bounds=(2.001, 1000.0))")
    print("     -> nu_new (float)     : Brent root-finding ECME update of a")
    print("        Student-t degrees-of-freedom parameter (nu_u or nu_eps).")
    print()
    print("  6. run_m_step(Y, e_step_output, theta_old, freq_list, block_map,")
    print("               ordered_cols, freeze_nu_iters, current_iter, nu_bounds)")
    print("     -> theta_new (dict)   : high-level wrapper orchestrating 1-5 in")
    print("        ECM order (Lambda->R, A->Q, nu's); Sigma_0 carried forward.")
    print("-" * 72)

    print("\n" + "=" * 64)
    print("run_m_step test passed.")
    print("=" * 64)
