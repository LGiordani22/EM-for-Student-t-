"""
src/em_initialization.py

Step 2 of Algorithm 1 in EM_for_student_t.tex: initialisation of the
EM algorithm for the Student-t Dynamic Factor Model.

This module currently implements:
  - standardize(df)                 : centre and scale each observed series
  - mm_fill_quarterly(series)       : locally-constant MM fill for GDPC1
                                      (Section 7.2, eq. 7.2)
  - gaussian_fill_ragged(df)        : N(0,1) fill for ragged-edge NaN
  - pca_initialization(Y, bm)       : block-by-block PCA for initial factors F^(0)
                                      (Section 4.2, Algorithm 1 step 2)
  - compute_theta_initial(Y, F, bm) : compute all initial parameters theta^(0)
                                      (Lambda, A, Q, R, nu, w, Sigma_0)
                                      (Section 4.3-4.5, Algorithm 1 step 2)

Reference: EM_for_student_t.tex, Section 4 (Initialisation) and
           Section 2.2 (Preprocessing: Stationarity, Centring and
           Standardisation).
"""

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd


# ─── Standardisation ──────────────────────────────────────────────────────────

def standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Centre and scale each column of a mixed-frequency panel to zero mean
    and unit variance, preserving NaN entries.

    Parameters
    ----------
    df : pd.DataFrame
        Shape (T, M).  Rows are monthly time periods; columns are the M
        observed series.  NaN entries represent either the ragged edge
        (publication lag of FRED-MD series), the quarterly mask of GDPC1
        (non-quarter-end months), or structural missing values.

    Returns
    -------
    Y_std : pd.DataFrame
        Standardised panel, same shape and index as ``df``.
        Column j satisfies  mean(Y_std[:, j]) ≈ 0  and
        std(Y_std[:, j]) ≈ 1  over the non-NaN observations.
        NaN entries in ``df`` remain NaN in ``Y_std``.
    mean : pd.Series
        Column-wise sample means, computed on observed (non-NaN) values.
        Index = df.columns.
    std : pd.Series
        Column-wise sample standard deviations (ddof=1), computed on
        observed (non-NaN) values.  Index = df.columns.

    Raises
    ------
    ValueError
        If any column has fewer than 2 non-NaN observations (std undefined).

    Notes
    -----
    **Thesis reference:**
    EM_for_student_t.tex, Section 2.2
    "Preprocessing: Stationarity, Centring and Standardisation".

    **Why centring:**
    The DGP of the Student-t DFM assumes zero-mean observables,
    i.e. E[y_{it}] = 0 for all i, t (Section 2, eq. (DGP-obs)).
    The FRED-MD transformation codes render the series approximately
    stationary but do not impose a zero mean.  Subtracting the
    sample mean enforces this assumption before estimation.

    **Why standardisation (unit variance):**
    The 20 series operate on very different scales — e.g. INDPRO
    log-differences are O(10^{-2}) while BAAFFM spread levels are
    O(1).  Without rescaling, PCA-based initialisation (step 2 of
    Algorithm 1) would be dominated by high-variance series,
    producing a distorted initial loading matrix Lambda_0.
    Dividing by the sample standard deviation places all series on
    a common scale.  Note: standardisation does NOT Gaussianise the
    data — it is a linear rescaling that preserves the shape of each
    marginal distribution, including its kurtosis and skewness.
    The empirical excess kurtosis (motivation for the Student-t model,
    Section 1) is unaffected by this transformation.

    **Why statistics are computed on observed values only:**
    Mean and std are estimated from the actually observed data points
    (NaN ignored) rather than from any imputed values.  This ensures
    that the location and scale statistics reflect the true sample
    distribution, not an artefact of the fill procedure.  In
    particular, GDPC1 statistics are computed from the ~165 observed
    quarterly values, not from the ~497 monthly slots.

    **Inverse transform:**
    The returned (mean, std) tuple allows the inverse transformation

        y_original = Y_std * std + mean

    to be applied later — e.g. to map estimated factor scores and
    loadings back to the original economic scale for interpretation.
    """
    # ── input validation ───────────────────────────────────────────────────────
    n_valid = df.notna().sum()
    too_few = n_valid[n_valid < 2]
    if not too_few.empty:
        # Columns with <2 non-NaN values cannot be standardised (std undefined).
        # This is expected for late-start series (e.g. WPSFD49207 before 2016-03,
        # VIXCLSx gap 2015-01..2015-08) when running the big config on early
        # vintages.  Treat them as permanently missing: mean=0, std=1 so the
        # column stays all-NaN after standardisation; gaussian_fill_ragged will
        # fill it with N(0,1) noise and the EM/Kalman treats it as uninformative.
        import warnings
        warnings.warn(
            f"standardize: {too_few.index.tolist()} have <2 non-NaN obs — "
            f"treating as missing (mean=0, std=1). Expected for late-start "
            f"series in early vintages of the big config.",
            RuntimeWarning, stacklevel=2,
        )

    # ── compute statistics on observed values (NaN ignored by default) ────────
    mean = df.mean()          # NaN for all-NaN columns
    std  = df.std()           # NaN for all-NaN columns (ddof=1)
    # Fill undefined stats for all-NaN / single-obs columns.
    mean = mean.fillna(0.0)
    std  = std.fillna(1.0).replace(0.0, 1.0)

    # ── standardise — NaN propagate through arithmetic automatically ──────────
    Y_std = (df - mean) / std

    return Y_std, mean, std


# ─── MM fill – quarterly series ───────────────────────────────────────────────

def mm_fill_quarterly(series: pd.Series) -> pd.Series:
    """
    Fill intra-quarter NaN entries in a monthly-indexed quarterly series using
    the locally-constant Mariano–Murasawa (MM) identity.

    Thesis reference: EM_for_student_t.tex, Section 7.2
    ('Using the MM Identity for Initialisation'), eq. 7.2
    (mm-fill-recursion).

    Parameters
    ----------
    series : pd.Series
        Monthly time series indexed by month-end ``pd.Timestamp`` values.
        The series is assumed to be standardised (output of ``standardize``).
        Non-NaN values appear **only** at quarter-end months (March, June,
        September, December); all other months are NaN.

    Returns
    -------
    filled : pd.Series
        Same length and index as ``series``.  Intra-quarter NaN entries are
        replaced by the monthly log-difference ξ_m derived from the MM
        recursion.  The quarter-end month is likewise overwritten with ξ_m
        (see note below).  Months that precede the first observed quarter-end
        remain NaN.

    Notes
    -----
    **Locally-constant assumption.**
    Within each quarter m the three monthly log-differences are assumed equal:

        x_{3m} = x_{3m-1} = x_{3m-2}  ≡  ξ_m

    This is the simplest consistent interpolation: it neither introduces
    artificial intra-quarter dynamics nor violates the aggregation identity.

    **Derivation of the recursion.**
    The Mariano–Murasawa identity expresses the quarterly log-difference of
    a GDP-chain index as a weighted sum of five consecutive monthly
    log-differences, with weights {1/3, 2/3, 1, 2/3, 1/3}:

        x^Q_{3m} = (1/3) x_{3m-4}
                 + (2/3) x_{3m-3}
                 + (1)   x_{3m-2}
                 + (2/3) x_{3m-1}
                 + (1/3) x_{3m}

    Substituting the locally-constant assumption
    (x_{3m-2} = x_{3m-1} = x_{3m} = ξ_m  and  x_{3m-4} = x_{3m-3} = ξ_{m-1}):

        x^Q_{3m} = (1/3 + 2/3) ξ_{m-1}  +  (1 + 2/3 + 1/3) ξ_m
                 = ξ_{m-1}  +  2 ξ_m

    Solving for ξ_m gives the recursion (eq. 7.2):

        ξ_m = (1/2) (x^Q_{3m} − ξ_{m-1})

    **Boundary condition.**
    For the first observed quarter (m = 1) there is no preceding quarter in
    the sample.  The boundary value is set to:

        ξ_0 = x^Q_1 / 3

    which is equivalent to assuming the "virtual" months before the sample
    grew at the same constant rate ξ_0 = x^Q_1/3.  This implies
    ξ_1 = (1/2)(x^Q_1 − x^Q_1/3) = x^Q_1/3 as well, so every month of the
    first quarter receives the value x^Q_1/3, and the identity is trivially
    satisfied: 2(x^Q_1/3) + x^Q_1/3 = x^Q_1.

    **Why the quarter-end month is overwritten.**
    The observed value at the quarter-end month is x^Q_m — the aggregated
    quarterly figure.  Under the locally-constant assumption the *monthly*
    value for that same calendar month is ξ_m ≠ x^Q_m (in general).
    Replacing x^Q_m with ξ_m ensures that the filled series is internally
    consistent: every month within a quarter carries the same monthly
    log-difference ξ_m, and the MM aggregation of those three values
    reconstructs x^Q_m exactly.

    **Scope of use.**
    This fill is used ONLY to construct θ^(0) via PCA in step 2 of
    Algorithm 1 (Section 4).  Once the EM iterations start, the original
    quarterly observations re-enter the model through the selection matrix
    W_t and the MM-augmented state vector (Section 7.3), not through this
    fill.

    **Post-fill variance.**
    After the fill the variance of the series will differ from 1 because the
    fill expands ~N quarterly observations into ~3N monthly observations with
    repeated values within each quarter.  This is acceptable: the fill is
    used solely for PCA initialisation, which is invariant to small scale
    variations.
    """
    # locate observed quarter-end months (months 3, 6, 9, 12 with non-NaN values)
    is_qend = series.index.month.isin([3, 6, 9, 12])
    observed = series[is_qend & series.notna()]

    if observed.empty:
        return series.copy()

    filled = series.copy()

    # boundary condition: xi_0 = x_Q[first_quarter] / 3
    xi_prev = observed.iloc[0] / 3.0

    for q_date, x_q in observed.items():
        # recursion eq. 7.2: xi_m = (1/2)(x_Q_m - xi_{m-1})
        xi_m = 0.5 * (x_q - xi_prev)

        # fill all three months of this quarter (start_month .. quarter-end)
        start_month = q_date.month - 2   # always in [1, 10] for months 3,6,9,12
        for offset in range(3):
            target = pd.Timestamp(q_date.year, start_month + offset, 1) + pd.offsets.MonthEnd(0)
            if target in filled.index:
                filled[target] = xi_m

        xi_prev = xi_m

    return filled


# ─── Gaussian fill – ragged edge ──────────────────────────────────────────────

def gaussian_fill_ragged(
    df: pd.DataFrame,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Replace all remaining NaN entries in a standardised panel with independent
    draws from N(0, 1).

    Thesis reference: EM_for_student_t.tex, Algorithm 1 step 2.6
    (Initialisation: ragged-edge fill).

    Parameters
    ----------
    df : pd.DataFrame
        Standardised panel (output of ``standardize`` and, for GDPC1,
        ``mm_fill_quarterly``).  Shape (T, M).  Remaining NaN entries are
        typically:

        * 1–2 rows at the start of each series (lost to log-differencing).
        * 1–2 rows at the end of series affected by publication lag
          (ragged edge).

    random_state : int or None, optional
        Seed for ``numpy.random.default_rng``.  Default 42 ensures that
        successive runs with the same dataset produce identical theta_0.
        Pass ``None`` for a random seed.

    Returns
    -------
    filled : pd.DataFrame
        Same shape, index, and columns as ``df``.  Every NaN entry is
        replaced by an independent N(0, 1) draw; all observed (non-NaN)
        values are left unchanged.

    Notes
    -----
    **Why N(0, 1):**
    The panel has been standardised to zero mean and unit variance.
    Consequently N(0, 1) is the standardised marginal distribution of each
    series, and drawing from it is the minimal-information assumption — we
    draw from the standardised marginal because we have no cross-series
    information to do better at this stage.

    **Why this crude fill is acceptable:**
    The filled values are used *exclusively* to construct the initial
    parameter vector theta_0 via PCA (Algorithm 1, step 2).  Once the EM
    iterations begin, the original observations re-enter the model through
    the selection matrix W_t: observed entries are conditioned on exactly,
    and missing entries are integrated out via the Kalman smoother.  The
    Gaussian fill is therefore discarded after PCA and has no effect on
    the EM fixed point.

    **Reproducibility:**
    ``random_state`` is fixed to 42 by default so that theta_0 is
    deterministic across runs.  Changing the seed will produce a different
    fill and, consequently, a different PCA initialisation, but the EM
    algorithm should converge to the same maximum-likelihood estimate
    provided the log-likelihood surface is well-behaved.
    """
    rng = np.random.default_rng(random_state)
    nan_mask = df.isna()

    # Draw a full (T, M) noise matrix — only NaN positions will be used.
    noise = pd.DataFrame(
        rng.standard_normal(df.shape),
        index=df.index,
        columns=df.columns,
    )

    # Keep observed values; replace NaN positions with N(0,1) draws.
    filled = df.where(~nan_mask, noise)
    return filled


# ─── PCA initialisation ───────────────────────────────────────────────────────

def pca_initialization(
    Y_filled: pd.DataFrame,
    block_map: dict[str, str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Extract block-specific initial factors F^(0) = [f_R, f_F, f_X] via PCA.

    Thesis reference: EM_for_student_t.tex, Section 4.2
    (Initialising the Factors by PCA), eq. (4.1).

    Note: while the thesis describes the global PCA on the full standardised
    matrix, we implement an equivalent but more interpretable variant:
    block-by-block PCA.  This approach directly produces factors aligned with
    the economic blocks (real, financial, other), eliminating the need for a
    post-hoc reordering.  The block-diagonal structure of Lambda (Section 8
    of the thesis) is thus respected from the very start of the EM iteration.

    Parameters
    ----------
    Y_filled : pd.DataFrame
        Shape (T, M).  Fully-filled standardised panel (output of
        ``gaussian_fill_ragged``).  Must contain no NaN entries.
    block_map : dict[str, str]
        Maps each series name to its economic block: ``"real"``,
        ``"financial"``, or ``"other"``.

    Returns
    -------
    F : np.ndarray, shape (T, 3)
        Initial factor matrix.  Columns are ordered [f_R, f_F, f_X],
        corresponding to the real, financial, and other blocks.
    block_factors_info : dict[str, np.ndarray]
        Keys ``"real"``, ``"financial"``, ``"other"``.  Each value is the
        first eigenvector v_b (shape (M_b,)) used to project its block onto
        the factor.  Useful for inspecting block loadings.

    Raises
    ------
    ValueError
        If any expected block (``"real"``, ``"financial"``, ``"other"``) has
        no series assigned to it in ``block_map``.

    Notes
    -----
    **Why PCA for initialisation:**
    The EM algorithm (Algorithm 1, Section 4) requires a starting parameter
    vector theta^(0).  The most natural choice for the initial factors F^(0)
    is the first principal component of each block: PCA extracts the direction
    of maximum variance in the sub-panel, which under the factor model is
    precisely the latent factor driving co-movement within the block.

    **Why block-by-block:**
    The DFM assumes a block-diagonal loading matrix Lambda (Section 8): series
    in the real block load only on f_R, financial series on f_F, and so on.
    Running PCA on the full (T x M) panel would mix signals across blocks.
    Block-by-block PCA respects this structure from the outset: each sub-panel
    Y_b is driven primarily by its own latent factor f_b, and the first PC of
    Y_b is a consistent estimator of f_b under the block-factor structure.

    **Sign convention (Convention 1, Section 8.4):**
    The sign of an eigenvector is algebraically indeterminate: if v is an
    eigenvector, so is -v.  We resolve this by requiring that the sum of the
    loading vector v_b be non-negative (i.e. most loadings are positive).
    This means that f_b = Y_b @ v_b tends to be positive during economic
    expansions (when most series are above their mean) and negative during
    contractions, giving the factors an intuitive economic sign.

    **Implementation note:**
    The sample covariance is computed as Sigma_b = Y_b.T @ Y_b / T (without
    subtracting the mean, because Y_filled is already standardised to zero
    mean).  ``numpy.linalg.eigh`` is used for the symmetric eigendecomposition;
    it guarantees real eigenvalues and returns them in ascending order, so the
    first principal component is the *last* column of the eigenvector matrix.
    """
    BLOCK_ORDER = ["real", "financial", "other"]

    # group columns by block
    block_cols: dict[str, list[str]] = {b: [] for b in BLOCK_ORDER}
    for col in Y_filled.columns:
        b = block_map.get(col)
        if b in block_cols:
            block_cols[b].append(col)

    for b in BLOCK_ORDER:
        if not block_cols[b]:
            raise ValueError(f"No series assigned to block '{b}' in block_map.")

    T = len(Y_filled)
    factors: list[np.ndarray] = []
    block_factors_info: dict[str, np.ndarray] = {}

    for b in BLOCK_ORDER:
        Y_b = Y_filled[block_cols[b]].to_numpy()   # (T, M_b)
        Sigma_b = (Y_b.T @ Y_b) / T                # (M_b, M_b) — symmetric

        # eigh returns eigenvalues ascending → last column = first PC
        _, V = np.linalg.eigh(Sigma_b)
        v_b = V[:, -1].copy()                       # (M_b,)

        # sign normalisation: majority of loadings positive
        if v_b.sum() < 0:
            v_b = -v_b

        factors.append(Y_b @ v_b)                  # (T,)
        block_factors_info[b] = v_b

    F = np.column_stack(factors)   # (T, 3)
    return F, block_factors_info


# ─── Initial parameter vector ─────────────────────────────────────────────────

def compute_theta_initial(
    Y_filled: pd.DataFrame,
    F: np.ndarray,
    block_map: dict[str, str],
    nu_init: float = 10.0,
    sigma_0_method: str = "identity",
) -> dict:
    """
    Compute the initial parameter vector theta^(0) from PCA factors F^(0).

    This is the final step of Algorithm 1 initialisation (Section 4 of the
    thesis): given the fully balanced panel Y_filled and the block-by-block
    PCA factors F, compute all model parameters needed to start the EM
    iterations.

    Parameters
    ----------
    Y_filled : pd.DataFrame
        Shape (T, M). Fully balanced panel with **no NaN entries** (output of
        the pipeline: ``standardize`` → ``mm_fill_quarterly`` →
        ``gaussian_fill_ragged``). All T time points are used for every series.
    F : np.ndarray, shape (T, r)
        Initial factor matrix from ``pca_initialization``. Columns are
        ordered [f_R, f_F, f_X] corresponding to BLOCK_ORDER =
        [``"real"``, ``"financial"``, ``"other"``]. Must be fully filled.
    block_map : dict[str, str]
        Maps each series name to its economic block: ``"real"``,
        ``"financial"``, or ``"other"``. Typically ``data_loader.BLOCK``.
    nu_init : float, optional
        Initial degrees-of-freedom for both factor innovations (nu_u) and
        idiosyncratic errors (nu_eps). Default 10.
        Reference: Thesis Section 4.4.
    sigma_0_method : str, optional
        How to initialise the state covariance Sigma_0.
        ``"identity"`` (default): Sigma_0 = I_{5r}.
        ``"lyapunov"``: stationary covariance via the discrete Lyapunov
        equation (not yet implemented — raises NotImplementedError).
        Reference: Thesis Section 4.5.

    Returns
    -------
    theta_0 : dict with the following keys

        ``"Lambda"``  : np.ndarray, shape (M, r)
            Block-diagonal loading matrix.  Entry (i, j) is non-zero only
            when series i belongs to the block corresponding to column j of F.
        ``"A"``       : np.ndarray, shape (r, r)
            VAR(1) transition matrix for the factors (full, not block-diagonal).
        ``"Q"``       : np.ndarray, shape (r, r)
            VAR(1) innovation covariance (symmetric positive semi-definite).
        ``"R"``       : np.ndarray, shape (M,)
            Diagonal of the idiosyncratic error covariance.  R[i] is the
            sample variance of the residuals for series i over all T time
            points of the balanced panel.
        ``"nu_u"``    : float
            Degrees-of-freedom for factor innovations (= ``nu_init``).
        ``"nu_eps"``  : float
            Degrees-of-freedom for idiosyncratic errors (= ``nu_init``).
        ``"w_u"``     : np.ndarray, shape (T,)
            Initial mixing weights for factor innovations (all ones).
        ``"w_eps"``   : np.ndarray, shape (T,)
            Initial mixing weights for idiosyncratic errors (all ones).
        ``"Sigma_0"`` : np.ndarray, shape (5r, 5r)
            Initial state covariance for the augmented state vector
            tilde_f_t in R^{5r} (companion form of the MM-augmented VAR).

    Notes
    -----
    **Overview — Thesis Section 4.3 (Initialising from PCA):**
    Given F^(0) from block-by-block PCA, each parameter is obtained by a
    simple closed-form estimator that ignores the Student-t mixing structure.
    These estimates are intentionally coarse: the EM iterations quickly
    refine them toward the maximum-likelihood optimum.

    **Lambda^(0) — Thesis Section 4.3, eq. (4.2).  Block-diagonal restriction
    imposed by construction, in line with Section 8 (Block-Structure
    Identification).**
    All loadings (monthly and quarterly) are initialised on the fully balanced
    panel (after MM fill for the quarterly series and N(0,1) fill for the
    ragged edge).  This ensures a uniform treatment of all series, consistent
    with the fact that the initial factors F^(0) are themselves extracted from
    the balanced panel.  The MM aggregation structure (composite regressor
    phi^b) enters rigorously only in the M-step of the EM iteration (Section 8
    of the thesis); at initialisation, a direct OLS of each (filled) series on
    its block factor suffices.

    Series i in block b loads only on factor f_b (column j in F).  The scalar
    loading is estimated by scalar OLS on the full T-point sample:

        Lambda_{i,b} = (sum_{t=1}^{T} y_filled_{i,t} * f_bt)
                       / (sum_{t=1}^{T} f_bt^2)

    **A^(0) and Q^(0) — Thesis Section 4.3, eqs. (4.3)-(4.4) (VAR OLS).**
    OLS on the fully-filled factor series F (no mask needed since F has no NaN):

        A = (Z.T @ X) @ inv(X.T @ X)    X = F[:-1], Z = F[1:]
        Q = residuals.T @ residuals / (T-1)

    A is a full (r x r) matrix — it captures dynamic correlations across
    blocks and is NOT block-diagonal.

    **R^(0) — Thesis Section 4.3, eq. (4.5) (diagonal residual variance).**
    For each series i, R_i is the sample variance (ddof=1) of the residuals
    computed on the full balanced panel:

        residual_{i,t} = y_filled_{i,t} - Lambda_{i,:} @ F_t   for t = 1..T
        R_i = Var( residual_i )

    Lambda and R are thus both computed on the balanced panel, ensuring a
    uniform treatment consistent with the PCA step.

    **nu^(0) — Thesis Section 4.4, default nu = 10.**
    Both nu parameters are initialised to nu_init (default 10), placing the
    Student-t close to a moderate-tailed distribution.  The EM M-step updates
    nu toward the data-implied degrees of freedom.

    **w_u^(0), w_eps^(0) — Thesis Section 4.4.**
    All mixing weights are initialised to 1, consistent with the Student-t
    scale-mixture representation.

    **Sigma_0 — Thesis Section 4.5, either identity or Lyapunov.**
    The augmented state tilde_f_t in R^{5r} stacks five lags of f_t:
    tilde_f_t = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4}).
    ``"identity"``: Sigma_0 = I_{5r}  (default, fast, sufficient for init).
    ``"lyapunov"`` (TODO): stationary covariance from
        P = tilde_A @ P @ tilde_A.T + tilde_Q,
    where tilde_A is the (5r x 5r) companion-form transition matrix built from
    A and tilde_Q is the (5r x 5r) companion noise covariance built from Q.
    Implementation: scipy.linalg.solve_discrete_lyapunov(tilde_A, tilde_Q).
    """
    BLOCK_ORDER = ["real", "financial", "other"]
    r = len(BLOCK_ORDER)   # 3

    series_list = list(Y_filled.columns)
    M = len(series_list)
    T, r_F = F.shape
    if r_F != r:
        raise ValueError(f"F has {r_F} columns but BLOCK_ORDER has r={r} blocks.")

    # ── 1. Lambda^(0): block-diagonal scalar OLS on full balanced panel ───────
    # Thesis Section 4.3, eq. (4.2).  Block-diagonal by construction: Lambda
    # is initialised to zero and only the (i, j) entry for the matching block
    # is filled, so all off-block entries remain exactly 0.
    # All series (monthly and quarterly) use all T observations; no masking is
    # needed because Y_filled contains no NaN entries.
    Lambda = np.zeros((M, r))
    for i, col in enumerate(series_list):
        b = block_map[col]
        j = BLOCK_ORDER.index(b)
        y_i = Y_filled[col].to_numpy()   # (T,) — no NaN
        f_b = F[:, j]                     # (T,) — factor for block b

        Lambda[i, j] = np.dot(y_i, f_b) / np.dot(f_b, f_b)

    # ── 2. A^(0) and Q^(0): VAR(1) OLS on F ──────────────────────────────────
    # Thesis Section 4.3, eqs. (4.3)-(4.4).
    # Model: f_t = A @ f_{t-1} + u_t   =>  Z = X @ A.T + U
    X = F[:-1, :]    # (T-1, r) lagged factors
    Z = F[1:, :]     # (T-1, r) current factors
    A = (Z.T @ X) @ np.linalg.inv(X.T @ X)   # (r, r)
    residuals_var = Z - X @ A.T               # (T-1, r)
    Q = (residuals_var.T @ residuals_var) / (T - 1)   # (r, r)

    # ── 3. R^(0): diagonal residual variance on full balanced panel ──────────
    # Thesis Section 4.3, eq. (4.5).
    # Residuals and variance are computed on all T points, consistent with the
    # full-sample OLS used for Lambda above.
    R = np.zeros(M)
    for i, col in enumerate(series_list):
        y_i = Y_filled[col].to_numpy()   # (T,) — no NaN
        resid = y_i - F @ Lambda[i, :]
        R[i] = np.var(resid, ddof=1)

    # ── 4. nu_u^(0) and nu_eps^(0) ───────────────────────────────────────────
    # Thesis Section 4.4, default nu = 10.
    nu_u   = float(nu_init)
    nu_eps = float(nu_init)

    # ── 5. w_u^(0) and w_eps^(0) ─────────────────────────────────────────────
    # Thesis Section 4.4.
    w_u   = np.ones(T)
    w_eps = np.ones(T)

    # ── 6. Sigma_0: initial state covariance ─────────────────────────────────
    # Thesis Section 4.5, either identity or Lyapunov.
    # Augmented state tilde_f_t in R^{5r}: stacks five consecutive factor lags.
    dim = 5 * r    # 15 when r=3
    if sigma_0_method == "identity":
        Sigma_0 = np.eye(dim)
    elif sigma_0_method == "lyapunov":
        # TODO: build companion tilde_A (5r x 5r) and tilde_Q (5r x 5r),
        # then call scipy.linalg.solve_discrete_lyapunov(tilde_A, tilde_Q).
        raise NotImplementedError(
            "sigma_0_method='lyapunov' is not yet implemented. "
            "Use the default 'identity' for now."
        )
    else:
        raise ValueError(
            f"Unknown sigma_0_method='{sigma_0_method}'. "
            "Valid choices: 'identity', 'lyapunov'."
        )

    return {
        "Lambda":  Lambda,
        "A":       A,
        "Q":       Q,
        "R":       R,
        "nu_u":    nu_u,
        "nu_eps":  nu_eps,
        "w_u":     w_u,
        "w_eps":   w_eps,
        "Sigma_0": Sigma_0,
    }


# ─── Full pipeline wrapper ────────────────────────────────────────────────────

def initialize_theta(
    dataset_path: str | None = None,
    nu_init: float = 10.0,
    sigma_0_method: str = "identity",
    save: bool = True,
    random_state: int = 42,
    config_name: str | None = None,
) -> tuple[dict, np.ndarray, dict]:
    """
    Entry point for computing the initial parameter vector theta^(0).

    Combines standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, and compute_theta_initial into a single call —
    Algorithm 1 step 2 in full.

    Reference: EM_for_student_t.tex, Section 4 (complete initialisation),
    Algorithm 1 step 2.

    Saving theta^(0) to disk allows downstream modules (kalman.py,
    em_e_step.py) to load it without recomputing the full pipeline; the
    .npz file contains all numerical arrays and the .json file contains
    sample metadata and quick-reference eigenvalues.

    Parameters
    ----------
    dataset_path : str or None, optional
        Path to the preprocessed CSV panel.  If ``None``, ``config_name``
        must be provided and the path is derived as
        ``data/processed/dataset_<config_name>.csv``.
    nu_init : float, optional
        Initial degrees-of-freedom for both nu_u and nu_eps.  Default 10.
        Reference: Thesis Section 4.4.
    sigma_0_method : str, optional
        How to initialise Sigma_0.  ``"identity"`` (default) or
        ``"lyapunov"`` (not yet implemented).
        Reference: Thesis Section 4.5.
    save : bool, optional
        If True (default), writes two files to ``data/processed/``:

        * ``theta_initial.npz`` — compressed NumPy archive with arrays
          Lambda, A, Q, R, w_u, w_eps, Sigma_0, F, nu_u, nu_eps.
        * ``theta_initial_metadata.json`` — JSON dict with sample info,
          per-series mean/std (needed for inverse standardisation), and
          quick-reference eigenvalues of A and Q.
    random_state : int, optional
        RNG seed for the Gaussian ragged-edge fill.  Default 42
        (deterministic theta^(0) across runs).

    Returns
    -------
    theta_0 : dict
        Initial parameter vector with keys Lambda, A, Q, R, nu_u, nu_eps,
        w_u, w_eps, Sigma_0.  See ``compute_theta_initial`` for shapes.
    F : np.ndarray, shape (T, 3)
        Initial factor matrix [f_R, f_F, f_X] from block-by-block PCA.
    metadata : dict
        JSON-serializable dict with keys:

        * ``sample_start``, ``sample_end`` — ISO-8601 date strings.
        * ``T``, ``M``, ``r`` — panel dimensions.
        * ``block_sizes`` — number of series per block.
        * ``series_mean``, ``series_std`` — per-series location/scale
          (needed for the inverse transform y_original = Y_std*std + mean).
        * ``nu_init``, ``sigma_0_method``, ``random_state`` — call args.
        * ``Lambda_sv`` — singular values of Lambda (list of floats).
        * ``A_eigenvalues`` — list of dicts {real, imag, mod} per eigenvalue.
        * ``Q_eigenvalues`` — eigenvalues of Q (list of floats, ascending).
    """
    # ── locate project root and resolve dataset path ──────────────────────────
    _project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if dataset_path is None:
        if config_name is None:
            raise ValueError(
                "initialize_theta requires either dataset_path or config_name. "
                "Pass config_name='small' or config_name='big'."
            )
        dataset_path = os.path.join(
            _project_root, "data", "processed", f"dataset_{config_name}.csv"
        )

    # ── import BLOCK and FREQ from data_loader ────────────────────────────────
    _src = os.path.join(_project_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    if config_name is not None:
        from data_loader import load_config as _dl_load_config  # noqa: PLC0415
        _dl_cfg = _dl_load_config(config_name)
        BLOCK = _dl_cfg["BLOCK"]
        FREQ  = _dl_cfg["FREQ"]
    else:
        from data_loader import BLOCK, FREQ  # noqa: PLC0415

    # ── 1. load dataset ───────────────────────────────────────────────────────
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

    # ── 2a. standardize ───────────────────────────────────────────────────────
    Y_std, mean, std = standardize(df)

    # ── 2b. mm_fill_quarterly on all quarterly series ─────────────────────────
    Y_mm = Y_std.copy()
    for col in Y_std.columns:
        if FREQ.get(col) == "quarterly":
            Y_mm[col] = mm_fill_quarterly(Y_std[col])

    # ── 2c. gaussian_fill_ragged ──────────────────────────────────────────────
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=random_state)

    # ── 2d. pca_initialization ────────────────────────────────────────────────
    F, block_factors_info = pca_initialization(Y_filled, BLOCK)

    # ── 2e. compute_theta_initial ─────────────────────────────────────────────
    theta_0 = compute_theta_initial(Y_filled, F, BLOCK, nu_init, sigma_0_method)

    # ── 3. build metadata ─────────────────────────────────────────────────────
    T, M = Y_filled.shape
    r = F.shape[1]

    block_order = ["real", "financial", "other"]
    block_sizes = {
        b: sum(1 for c in Y_filled.columns if BLOCK.get(c) == b)
        for b in block_order
    }

    # singular values of Lambda (M x r, not square → no eigenvalues)
    sv_Lambda = np.linalg.svd(theta_0["Lambda"], compute_uv=False).tolist()

    # eigenvalues of A (may be complex)
    eigvals_A = np.linalg.eigvals(theta_0["A"])
    eigvals_A_list = [
        {"real": float(ev.real), "imag": float(ev.imag), "mod": float(abs(ev))}
        for ev in eigvals_A
    ]

    # eigenvalues of Q (symmetric → real, ascending)
    eigvals_Q = np.linalg.eigvalsh(theta_0["Q"]).tolist()

    metadata: dict = {
        "sample_start":   df.index[0].strftime("%Y-%m-%d"),
        "sample_end":     df.index[-1].strftime("%Y-%m-%d"),
        "T":              T,
        "M":              M,
        "r":              r,
        "block_sizes":    block_sizes,
        "series_mean":    {col: float(mean[col]) for col in df.columns},
        "series_std":     {col: float(std[col])  for col in df.columns},
        "nu_init":        nu_init,
        "sigma_0_method": sigma_0_method,
        "random_state":   random_state,
        "Lambda_sv":      sv_Lambda,
        "A_eigenvalues":  eigvals_A_list,
        "Q_eigenvalues":  eigvals_Q,
    }

    # ── 4. optionally save to disk ────────────────────────────────────────────
    if save:
        if config_name is not None:
            out_dir = os.path.join(_project_root, "data", "processed", config_name)
        else:
            out_dir = os.path.join(_project_root, "data", "processed")
        os.makedirs(out_dir, exist_ok=True)

        npz_path  = os.path.join(out_dir, "theta_initial.npz")
        json_path = os.path.join(out_dir, "theta_initial_metadata.json")

        np.savez_compressed(
            npz_path,
            Lambda  = theta_0["Lambda"],
            A       = theta_0["A"],
            Q       = theta_0["Q"],
            R       = theta_0["R"],
            w_u     = theta_0["w_u"],
            w_eps   = theta_0["w_eps"],
            Sigma_0 = theta_0["Sigma_0"],
            F       = F,
            nu_u    = np.array(theta_0["nu_u"]),
            nu_eps  = np.array(theta_0["nu_eps"]),
        )

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return theta_0, F, metadata


# ─── Standardized-data loader (shared across the EM pipeline) ────────────────

def load_standardized_data(
    dataset_path: str | None = None,
    metadata_path: str | None = None,
    check_metadata: bool = True,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load a config-aware dataset CSV and return its column-wise standardised
    version with missing entries preserved as ``NaN``.

    This is the canonical input representation for **every** stage of the EM
    pipeline after initialisation — Kalman filter / smoother, E-step
    (Mahalanobis residuals and Student-t weights) and M-step (parameter
    updates).  The standardisation is identical to the one used inside
    :func:`initialize_theta` to compute :math:`\\theta^{(0)}`, so all
    downstream modules operate on the same numerical scale as the initial
    parameters; a scale mismatch between ``Y`` and ``\\theta^{(0)}`` would
    otherwise force the Kalman to compress/expand the latent state to
    compensate, biasing the loadings (see ``em_m_step`` diagnostics).

    Parameters
    ----------
    dataset_path : str
        Path to the preprocessed CSV panel (required).  Pass the
        config-aware path, e.g. ``data/processed/small/dataset_small.csv``.
    metadata_path : str or None, optional
        Path to ``theta_initial_metadata.json`` (output of
        :func:`initialize_theta`).  Default: same directory as the dataset.
        Used only for the consistency check controlled by ``check_metadata``.
    check_metadata : bool, optional
        When ``True`` (default), assert that the column means and standard
        deviations computed here coincide, within ``atol``, with the
        values stored in ``theta_initial_metadata.json``.  This guarantees
        that the data representation seen by the EM modules is exactly
        the one against which :math:`\\theta^{(0)}` was calibrated.
        Set to ``False`` only if the metadata file is not (yet) available.
    atol : float, optional
        Absolute tolerance for the metadata consistency check
        (default ``1e-10``).

    Returns
    -------
    Y_std : np.ndarray, shape (T, M)
        Standardised observation panel with ``NaN`` preserved at all
        positions where the raw dataset had missing values (ragged edge,
        non-quarter-end months for the quarterly series, etc.).  No fill
        is applied — fills are an initialisation-only artefact handled
        inside :func:`initialize_theta`.
    mean : np.ndarray, shape (M,)
        Per-series sample means computed on observed (non-NaN) values.
        Identical to those reported by :func:`standardize`.
    std : np.ndarray, shape (M,)
        Per-series sample standard deviations (ddof=1) computed on
        observed values.
    series_names : list[str]
        Column order of ``Y_std``, ``mean`` and ``std``.

    Raises
    ------
    AssertionError
        If ``check_metadata`` is ``True`` and the (mean, std) computed
        here differ from those stored in ``theta_initial_metadata.json``
        by more than ``atol`` for any series.

    Notes
    -----
    **Why no fill.**
    ``initialize_theta`` applies ``mm_fill_quarterly`` and
    ``gaussian_fill_ragged`` to build a fully balanced panel for PCA.
    Those fills exist **only** to bootstrap :math:`F^{(0)}` and the
    initial loadings; once EM starts, missing entries re-enter the
    model rigorously through the selection matrix :math:`W_t` (Kalman
    filter, eq. (6) of the thesis) and the per-time observed Mahalanobis
    residual (E-step, eq:d-eps-missing).  Re-applying the fill here
    would discard that probabilistic handling.

    **Idempotency of the standardisation.**
    The (mean, std) values returned here are the same that were saved
    in ``theta_initial_metadata.json`` (both come from :func:`standardize`
    on the same CSV).  The ``check_metadata`` assertion makes this
    explicit and protects against silent dataset drift between two runs.
    """
    # ── locate dataset and metadata files ─────────────────────────────────────
    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if dataset_path is None:
        raise ValueError(
            "load_standardized_data requires dataset_path. "
            "Pass the path to the config-aware CSV, e.g. data/processed/small/dataset_small.csv."
        )
    if metadata_path is None:
        metadata_path = str(_project_root / "data" / "processed" / "theta_initial_metadata.json")

    # ── load CSV (raw, with NaN) and standardise NaN-aware ───────────────────
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    Y_std_df, mean_sr, std_sr = standardize(df)

    series_names = list(df.columns)
    Y_std = Y_std_df.to_numpy()
    mean  = mean_sr.to_numpy()
    std   = std_sr.to_numpy()

    # ── optional metadata consistency check ──────────────────────────────────
    if check_metadata:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"theta_initial_metadata.json not found at: {metadata_path}\n"
                f"Run initialize_theta(save=True) first, or pass check_metadata=False."
            )
        with open(metadata_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        meta_mean = np.array([meta["series_mean"][c] for c in series_names])
        meta_std  = np.array([meta["series_std"][c]  for c in series_names])

        max_diff_mean = float(np.max(np.abs(meta_mean - mean)))
        max_diff_std  = float(np.max(np.abs(meta_std  - std)))

        assert max_diff_mean < atol, (
            f"Series-wise means computed here differ from theta_initial_metadata.json "
            f"by max |diff| = {max_diff_mean:.3e} > atol = {atol:.0e}.  "
            f"The dataset may have changed since theta^(0) was computed; "
            f"rerun initialize_theta to refresh."
        )
        assert max_diff_std < atol, (
            f"Series-wise stds computed here differ from theta_initial_metadata.json "
            f"by max |diff| = {max_diff_std:.3e} > atol = {atol:.0e}.  "
            f"The dataset may have changed since theta^(0) was computed; "
            f"rerun initialize_theta to refresh."
        )

    return Y_std, mean, std, series_names


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── parse config flag ─────────────────────────────────────────────────────
    import pathlib
    import sys as _sys_main
    _src_dir = str(pathlib.Path(__file__).resolve().parent)
    if _src_dir not in _sys_main.path:
        _sys_main.path.insert(0, _src_dir)
    from config_utils import parse_config_args, resolve_output_path, get_project_root

    _args = parse_config_args("em_initialization self-test — standardise, PCA, theta^(0).")
    _cfg = _args.config

    # ── locate project root and resolve config-specific paths ─────────────────
    project_root = str(get_project_root())
    csv_path     = str(resolve_output_path("dataset", "", _cfg))

    # ── load BLOCK and FREQ for this config ───────────────────────────────────
    from data_loader import load_config as _dl_load_config
    _cfg_dict    = _dl_load_config(_cfg)
    BLOCK        = _cfg_dict["BLOCK"]
    FREQ         = _cfg_dict["FREQ"]
    ORDERED_COLS = _cfg_dict["ORDERED_COLS"]

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Shape: {df.shape[0]} months x {df.shape[1]} series\n")

    # ── apply standardise ─────────────────────────────────────────────────────
    Y_std, mu, sigma = standardize(df)

    # ── assertions ────────────────────────────────────────────────────────────
    tol = 1e-10

    # 1. Mean of standardised columns ≈ 0
    mean_after = Y_std.mean()
    assert (mean_after.abs() < tol).all(), (
        f"Standardised means not close to zero:\n{mean_after[mean_after.abs() >= tol]}"
    )

    # 2. Std of standardised columns ≈ 1
    std_after = Y_std.std()
    assert ((std_after - 1.0).abs() < tol).all(), (
        f"Standardised stds not close to 1:\n{std_after[(std_after - 1.0).abs() >= tol]}"
    )

    # 3. NaN count unchanged
    nan_before = df.isna().sum()
    nan_after  = Y_std.isna().sum()
    assert (nan_before == nan_after).all(), (
        "NaN count changed after standardisation — fill must NOT happen here.\n"
        f"Differences:\n{(nan_after - nan_before)[nan_after != nan_before]}"
    )

    print("All assertions passed.\n")

    # ── report table ──────────────────────────────────────────────────────────
    report = pd.DataFrame({
        "block"      : [None] * len(df.columns),   # placeholder
        "mean_before": df.mean().round(6),
        "mean_after" : mean_after.round(6),
        "std_before" : df.std().round(6),
        "std_after"  : std_after.round(6),
        "n_obs"      : df.notna().sum(),
    })

    # add block metadata from the active config (already loaded above)
    report["block"] = [BLOCK.get(c, "?") for c in df.columns]

    print("Standardisation report (mean and std, before vs after):")
    print(report.to_string())

    # ── mm_fill_quarterly self-test ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("mm_fill_quarterly self-test")
    print("=" * 60)

    gdp_raw    = Y_std["GDPC1"]
    gdp_filled = mm_fill_quarterly(gdp_raw)

    is_qend = gdp_filled.index.month.isin([3, 6, 9, 12])

    # 1. Every month that belongs to an *observed* quarter must be filled.
    #    Months of ragged-edge quarters (quarter-end is NaN or out of range) are
    #    permitted to remain NaN — those quarters were never processed.
    observed_qends = gdp_raw[is_qend & gdp_raw.notna()].index
    nan_in_observed = 0
    for qe in observed_qends:
        start_m = qe.month - 2
        for offset in range(3):
            target = pd.Timestamp(qe.year, start_m + offset, 1) + pd.offsets.MonthEnd(0)
            if target in gdp_filled.index and pd.isna(gdp_filled[target]):
                nan_in_observed += 1

    assert nan_in_observed == 0, (
        f"{nan_in_observed} NaN found in months belonging to observed quarters"
    )
    nan_total = gdp_filled.isna().sum()
    print(f"[OK] All months of observed quarters filled. "
          f"Remaining NaN: {nan_total} (ragged-edge / pre-sample, expected).")

    # 2. MM reconstruction: x^Q_m == 2*xi_m + xi_{m-1}  for m >= 2
    xi_qend = gdp_filled[is_qend & gdp_filled.notna()]   # filled xi_m values
    x_Q     = gdp_raw[is_qend & gdp_raw.notna()]         # original quarterly values

    xi_m  = xi_qend.values[1:]    # quarters 2 .. end
    xi_m1 = xi_qend.values[:-1]   # quarters 1 .. end-1
    x_Q_m = x_Q.values[1:]        # quarters 2 .. end

    recon_error = np.abs(2.0 * xi_m + xi_m1 - x_Q_m).max()
    assert recon_error < 1e-10, (
        f"MM reconstruction failed: max error = {recon_error:.2e}"
    )
    print(f"[OK] MM reconstruction: max |2*xi_m + xi_{{m-1}} - x^Q_m| = {recon_error:.2e}")

    # 3. Table: first 12 months of the sample
    print("\nFirst 12 months — GDP original (quarterly) vs MM-filled (monthly):")
    tbl = pd.DataFrame({
        "GDP_original" : gdp_raw.iloc[:12].round(4),
        "GDP_mm_filled": gdp_filled.iloc[:12].round(4),
    })
    tbl.index = tbl.index.strftime("%Y-%m")
    print(tbl.to_string())

    # 4. Optional plot (2007-2010 and 2019-2021 recessions)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        periods = [("2007-01", "2010-12", "GFC 2007–2010"),
                   ("2019-01", "2021-12", "COVID 2019–2021")]

        for ax, (start, end, title) in zip(axes, periods):
            seg_filled = gdp_filled.loc[start:end]
            seg_raw    = gdp_raw.loc[start:end]
            ax.plot(seg_filled.index, seg_filled.values, lw=1.2,
                    label="MM-filled (ξ_m)", color="steelblue")
            ax.scatter(seg_raw.dropna().index, seg_raw.dropna().values,
                       s=30, color="crimson", zorder=5, label="x^Q (observed)")
            ax.set_title(title)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.legend(fontsize=8)
            ax.set_ylabel("Standardised log-diff")

        fig.suptitle("GDPC1: MM-filled monthly series vs quarterly observations",
                     fontsize=11)
        fig.tight_layout()

        plot_path = resolve_output_path("figures", "mm_fill_verification.png", _cfg)
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n[OK] Recession plot saved to: {plot_path}")

        print("""
============================================================
EXPLANATION OF THE PLOT
============================================================

The plot shows MM-filled monthly values (blue line) plotted
against the original quarterly observations (red dots). The
filled line appears "smoothed" relative to the quarterly
observations because of the algebraic structure of the MM
aggregation:

The MM aggregation identity (eq. 7.1 of the thesis) relates
the observed quarterly value to 5 latent monthly values:

    x_Q_m = (1/3) x_{m-1} + (2/3) x_{m-1}
          + x_m + (2/3) x_m + (1/3) x_m

Under the locally-constant assumption (xi_{m-1}, xi_m), this
simplifies to:

    x_Q_m = xi_{m-1} + 2 * xi_m

so xi_m absorbs only ~half of the quarterly value. When the
quarterly value spikes (e.g. -8.5 during Covid Q2 2020), the
filled monthly value is approximately half of it (-3.7),
not the full spike. This is mathematically exact, not a flaw:

    3 * xi_m + 2 * xi_{m-1} reconstructs the full quarterly
    cumulative log-difference.

The information content of the quarterly observation is
preserved — it is just redistributed across 3 monthly values
under the locally-constant assumption.

This fill is used ONLY to construct theta_0 via PCA. Once the
EM iteration begins, the original quarterly observation
re-enters the model via the selection matrix W_t and the
MM-augmented state representation (Section 7.3), not via the
filled values.

------------------------------------------------------------
In sintesi (ITA): il MM-fill produce visivamente una serie
"smoothed" rispetto ai valori quarterly osservati, ma e'
matematicamente corretto. Il valore mensile xi_m e' circa
meta' del quarterly per costruzione della MM aggregation
(pesi 1/3, 2/3, 1, 2/3, 1/3): lo spike originale viene
redistribuito su 3 mesi, mantenendo l'informazione totale
del quarter. Questo fill serve solo per costruire theta_0
via PCA; dal primo passo EM in poi, l'osservazione quarterly
rientra nel modello attraverso la matrice di selezione W_t
e la rappresentazione MM-augmented dello stato (Sez. 7.3).
============================================================
""")
    except ImportError:
        print("\n[SKIP] matplotlib not available — plot skipped.")

    # ── gaussian_fill_ragged self-test ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("gaussian_fill_ragged self-test")
    print("=" * 60)

    # Build the panel as it would enter PCA: Y_std with GDPC1 MM-filled.
    Y_std_mmfill = Y_std.copy()
    Y_std_mmfill["GDPC1"] = gdp_filled

    nan_before_gauss = Y_std_mmfill.isna().sum()
    total_nan_before = int(nan_before_gauss.sum())
    print(f"NaN entries before Gaussian fill: {total_nan_before}")

    Y_filled = gaussian_fill_ragged(Y_std_mmfill, random_state=42)

    # 1. No NaN in output.
    assert Y_filled.isna().sum().sum() == 0, (
        "gaussian_fill_ragged left NaN entries in the output."
    )
    print("[OK] Y_filled contains no NaN.")

    # 2. Observed values are unchanged.
    observed_mask = ~Y_std_mmfill.isna()
    max_diff = (Y_filled[observed_mask] - Y_std_mmfill[observed_mask]).abs().max().max()
    assert max_diff < 1e-14, (
        f"Observed values changed after Gaussian fill: max diff = {max_diff:.2e}"
    )
    print("[OK] All originally observed values are preserved (max diff = 0).")

    # 3. Per-column NaN count (filled values).
    nan_counts = nan_before_gauss[nan_before_gauss > 0].sort_values(ascending=False)
    print(f"\nNaN filled per column ({len(nan_counts)} series affected):")
    for col, cnt in nan_counts.items():
        print(f"  {col:<20s}: {cnt} NaN filled")

    # ── final pipeline verification ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL PIPELINE VERIFICATION")
    print("=" * 60)

    print("Final dataset shape:", Y_filled.shape)
    print("Total NaN remaining:", Y_filled.isna().sum().sum())
    print("Per-column NaN counts:", Y_filled.isna().sum())

    assert Y_filled.isna().sum().sum() == 0, "Pipeline incomplete: NaN remaining"
    _T_fill, _M_fill = Y_filled.shape
    assert _M_fill == len(ORDERED_COLS), (
        f"Wrong number of columns: {_M_fill} vs {len(ORDERED_COLS)} expected"
    )

    print(f"Final dataset is complete: {_T_fill} × {_M_fill} with zero NaN.")
    print("Ready for PCA initialisation.")

    # ── pca_initialization self-test ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("pca_initialization self-test")
    print("=" * 60)

    # BLOCK already loaded from active config at the top of __main__
    F, block_info = pca_initialization(Y_filled, BLOCK)

    # 1. shape
    T_obs  = len(Y_filled)
    r_pca  = F.shape[1]
    assert F.shape[0] == T_obs, f"Wrong F shape[0]: {F.shape[0]} vs {T_obs}"
    print(f"[OK] F.shape = {F.shape}  (T={T_obs}, r={r_pca} factors: f_R, f_F, f_X)")

    # 2. means ~0
    f_means = F.mean(axis=0)
    print(f"[OK] Factor means: "
          f"f_R={f_means[0]:+.4f}  f_F={f_means[1]:+.4f}  f_X={f_means[2]:+.4f}")

    # 3. correlation table: each series vs its block factor
    BLOCK_ORDER = ["real", "financial", "other"]
    factor_labels = ["f_R", "f_F", "f_X"]

    print()
    for b_idx, (b, fname) in enumerate(zip(BLOCK_ORDER, factor_labels)):
        b_cols = [c for c in Y_filled.columns if BLOCK.get(c) == b]
        print(f"Correlation — {b.upper()} series vs {fname}:")
        for col in b_cols:
            r = np.corrcoef(Y_filled[col].to_numpy(), F[:, b_idx])[0, 1]
            print(f"  {col:<22s}  {r:+.4f}")
        print()

    # 4. eigenvector (loading) table
    print("Block loadings (first eigenvectors v_b):")
    for b, fname in zip(BLOCK_ORDER, factor_labels):
        b_cols = [c for c in Y_filled.columns if BLOCK.get(c) == b]
        v = block_info[b]
        print(f"  {b.upper()} ({fname}):")
        for col, loading in zip(b_cols, v):
            print(f"    {col:<22s}  {loading:+.4f}")
    print()

    # 5. optional plot: 3 factors in time with NBER recession shading
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        colors = ["steelblue", "darkorange", "seagreen"]
        plot_labels = [r"$f_R$ (Real)", r"$f_F$ (Financial)", r"$f_X$ (Other)"]
        # NBER recession dates
        recessions = [("2007-12", "2009-06"), ("2020-02", "2020-04")]
        dates = Y_filled.index

        for idx, (ax, label, color) in enumerate(zip(axes, plot_labels, colors)):
            ax.plot(dates, F[:, idx], lw=1.0, color=color, label=label)
            for r_start, r_end in recessions:
                ax.axvspan(pd.Timestamp(r_start), pd.Timestamp(r_end),
                           alpha=0.15, color="grey", zorder=0)
            ax.axhline(0, lw=0.5, color="black", linestyle="--")
            ax.set_ylabel(label, fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.legend(loc="upper left", fontsize=8)

        axes[0].set_title(r"Initial factors $F^{(0)}$ via block-by-block PCA"
                          " (grey bands = NBER recessions)", fontsize=11)
        fig.tight_layout()

        plot_path = resolve_output_path("figures", "pca_factors.png", _cfg)
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[OK] Factor plot saved to: {plot_path}")

    except ImportError:
        print("[SKIP] matplotlib not available — plot skipped.")

    # ── compute_theta_initial self-test ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("compute_theta_initial self-test")
    print("=" * 60)

    theta_0 = compute_theta_initial(Y_filled, F, BLOCK)

    Lambda  = theta_0["Lambda"]
    A_var   = theta_0["A"]
    Q_var   = theta_0["Q"]
    R_var   = theta_0["R"]
    Sigma_0 = theta_0["Sigma_0"]

    # 1. Shape verification  (M and r derived from data, config-agnostic)
    _M_init = Y_filled.shape[1]
    _r_init = F.shape[1]
    assert Lambda.shape  == (_M_init, _r_init), f"Lambda.shape  = {Lambda.shape}"
    assert A_var.shape   == (_r_init, _r_init), f"A.shape       = {A_var.shape}"
    assert Q_var.shape   == (_r_init, _r_init), f"Q.shape       = {Q_var.shape}"
    assert R_var.shape   == (_M_init,),          f"R.shape       = {R_var.shape}"
    assert Sigma_0.shape == (5 * _r_init, 5 * _r_init), \
        f"Sigma_0.shape = {Sigma_0.shape}"
    print(f"[OK] Lambda.shape  = {Lambda.shape}")
    print(f"[OK] A.shape       = {A_var.shape}")
    print(f"[OK] Q.shape       = {Q_var.shape}")
    print(f"[OK] R.shape       = {R_var.shape}")
    print(f"[OK] Sigma_0.shape = {Sigma_0.shape}")

    # 2. Block-diagonality of Lambda (off-block entries must be exactly zero)
    THETA_BLOCK_ORDER = ["real", "financial", "other"]
    series_names = list(Y_std.columns)
    off_block_max = 0.0
    for i, col in enumerate(series_names):
        b = BLOCK.get(col)
        j = THETA_BLOCK_ORDER.index(b)
        for jj in range(3):
            if jj != j:
                off_block_max = max(off_block_max, abs(Lambda[i, jj]))
    assert off_block_max == 0.0, (
        f"Lambda off-block entries are not zero: max = {off_block_max}"
    )
    print(f"[OK] Lambda is exactly block-diagonal (off-block max = {off_block_max:.2e})")

    # 3. R positivity
    assert (R_var > 0).all(), (
        f"R has non-positive entries at indices: "
        f"{np.where(R_var <= 0)[0].tolist()}"
    )
    print(f"[OK] All R values positive (min = {R_var.min():.6f})")

    # 4. Q symmetry and positive definiteness
    sym_err = np.max(np.abs(Q_var - Q_var.T))
    assert sym_err < 1e-14, f"Q not symmetric: max |Q - Q.T| = {sym_err:.2e}"
    eigvals_Q = np.linalg.eigvalsh(Q_var)
    assert (eigvals_Q > 0).all(), (
        f"Q not positive definite: min eigenvalue = {eigvals_Q.min():.4e}"
    )
    print(f"[OK] Q symmetric (max |Q - Q.T| = {sym_err:.2e})")
    print(f"[OK] Q positive definite (min eigenvalue = {eigvals_Q.min():.6f})")

    # ── Summary table: Lambda non-zero loadings per block ─────────────────────
    print("\n--- Lambda: non-zero loadings per block ---")
    factor_labels = ["f_R", "f_F", "f_X"]
    for b_idx, (b, fname) in enumerate(zip(THETA_BLOCK_ORDER, factor_labels)):
        b_cols = [c for c in series_names if BLOCK.get(c) == b]
        print(f"\n  {b.upper()} block  (factor {fname}, column j={b_idx}):")
        for col in b_cols:
            i = series_names.index(col)
            lam = Lambda[i, b_idx]
            print(f"    {col:<22s}  Lambda = {lam:+.4f}")

    # ── Full Lambda matrix ────────────────────────────────────────────────────
    print("\n--- Lambda full matrix (M=20 rows × r=3 cols) ---")
    print(f"  {'Series':<22s}  {'f_R':>8s}  {'f_F':>8s}  {'f_X':>8s}  Block")
    print("  " + "-" * 60)
    for i, col in enumerate(series_names):
        b = BLOCK.get(col)
        print(
            f"  {col:<22s}  {Lambda[i, 0]:>8.4f}  "
            f"{Lambda[i, 1]:>8.4f}  {Lambda[i, 2]:>8.4f}  {b}"
        )

    # ── Eigenvalues of A (VAR stability) ─────────────────────────────────────
    eigvals_A = np.linalg.eigvals(A_var)
    moduli = np.abs(eigvals_A)
    print("\n--- A eigenvalues (VAR(1) stability: |lam| < 1 ?) ---")
    for k, (ev, mod) in enumerate(zip(eigvals_A, moduli)):
        sign = "+" if ev.imag >= 0 else ""
        stable_tag = "stable" if mod < 1.0 else "UNSTABLE"
        print(
            f"  lam_{k+1} = {ev.real:+.6f}{sign}{ev.imag:.6f}i  "
            f"|lam| = {mod:.6f}  [{stable_tag}]"
        )

    # ── Diagonal of Q ─────────────────────────────────────────────────────────
    print("\n--- Q diagonal (factor innovation variances) ---")
    for k, fname in enumerate(factor_labels):
        print(f"  Q[{k},{k}] = {Q_var[k, k]:.6f}  ({fname})")

    # ── R per series ──────────────────────────────────────────────────────────
    print("\n--- R: idiosyncratic variances per series ---")
    for i, col in enumerate(series_names):
        b = BLOCK.get(col)
        print(f"  {col:<22s}  R = {R_var[i]:.6f}  [{b}]")

    # ── nu and w ──────────────────────────────────────────────────────────────
    print(f"\nnu_u    = {theta_0['nu_u']}")
    print(f"nu_eps  = {theta_0['nu_eps']}")
    w_u_ok   = bool(np.all(theta_0["w_u"]   == 1.0))
    w_eps_ok = bool(np.all(theta_0["w_eps"] == 1.0))
    print(f"w_u     : shape={theta_0['w_u'].shape},   all-ones={w_u_ok}")
    print(f"w_eps   : shape={theta_0['w_eps'].shape},   all-ones={w_eps_ok}")
    print(
        f"Sigma_0 : shape={Sigma_0.shape}, "
        f"= I_15 = {bool(np.allclose(Sigma_0, np.eye(15)))}"
    )

    print("\n[OK] compute_theta_initial: all checks passed.")

    # ── initialize_theta self-test (full pipeline wrapper) ────────────────────
    print("\n" + "=" * 60)
    print("initialize_theta self-test (full pipeline wrapper)")
    print("=" * 60)

    theta_w, F_w, meta_w = initialize_theta(save=True, config_name=_cfg)

    # 1. verify output files exist (config-specific paths)
    npz_path_w  = resolve_output_path("processed", "theta_initial.npz", _cfg)
    json_path_w = resolve_output_path("processed", "theta_initial_metadata.json", _cfg)
    assert npz_path_w.exists(),  f"Missing: {npz_path_w}"
    assert json_path_w.exists(), f"Missing: {json_path_w}"
    print(f"[OK] theta_initial.npz             saved: {npz_path_w}")
    print(f"[OK] theta_initial_metadata.json   saved: {json_path_w}")

    # 2. round-trip test: reload and compare arrays
    loaded_npz = np.load(str(npz_path_w))
    for key in ["Lambda", "A", "Q", "R"]:
        diff = np.max(np.abs(theta_w[key] - loaded_npz[key]))
        assert diff == 0.0, f"Round-trip mismatch for '{key}': max diff = {diff:.2e}"
        print(f"[OK] {key:<8s} round-trip identical  (max |orig - loaded| = {diff:.2e})")

    # 3. summary
    print(f"\n  T={meta_w['T']}, M={meta_w['M']}, r={meta_w['r']}")
    print(f"  sample : {meta_w['sample_start']} -> {meta_w['sample_end']}")
    print(f"  blocks : {meta_w['block_sizes']}")
    print(f"  nu_init={meta_w['nu_init']},  sigma_0_method='{meta_w['sigma_0_method']}'")
    print(f"  Lambda singular values : {[round(v, 4) for v in meta_w['Lambda_sv']]}")
    print(f"  A eigenvalue moduli    : {[round(ev['mod'], 4) for ev in meta_w['A_eigenvalues']]}")
    print(f"  Q eigenvalues          : {[round(v, 6) for v in meta_w['Q_eigenvalues']]}")

    print("\ntheta^(0) salvato. Pronto per il Kalman.")
