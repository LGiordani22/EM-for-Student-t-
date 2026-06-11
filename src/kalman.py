"""
src/kalman.py

Kalman filter / smoother machinery for the Student-t Dynamic Factor Model
with mixed-frequency (Mariano-Murasawa) aggregation.

This module implements the augmented state-space representation derived in
EM_for_student_t.tex:
  - Section "Adapting the State-Space Model: the Structural Role of MM"
    (subsec:mm-augmentation) — the companion-form augmentation that lets
    quarterly series load on five consecutive monthly factors.
  - Section "The Full State-Space Model" (subsec:full-ssm) — the complete
    system (eq:final-state, eq:final-obs, eq:final-Lambda-tilde) used at
    runtime by the E-step.

TASK 1 — augmented-matrix constructors (pure functions):
  - build_A_tilde(A)                    : companion-form transition (5r x 5r)
  - build_Q_tilde(Q, w_u_t)             : augmented innovation cov (5r x 5r, rank r)
  - build_Lambda_tilde(Lambda, freq_list, mm_weights) : effective loading (M x 5r)
  - build_R_tilde(R, w_eps_t)           : idiosyncratic cov (M x M, diagonal)

TASK 2 — selection matrix machinery:
  - build_selection_matrix(y_t)         : W_t (m_t x M) for a single time step
  - build_all_selection_matrices(Y)     : W_list for the full panel (T entries)

TASK 3 — single filter step (no time loop):
  - kalman_predict(...)                 : prediction -> f_{t|t-1}, P_{t|t-1}
  - kalman_update(...)                  : update    -> f_{t|t}, P_{t|t}, eta, S, loglik_t, K, WL

TASK 4 — full forward filter:
  - kalman_filter(...)                  : forward recursion t = 0..T-1; returns
                                          f_pred, P_pred, f_filt, P_filt, K_list, WL_list, loglik

TASK 5 — backward RTS smoother + lag-one smoothed covariance:
  - kalman_smoother(...)                : f_{t|T}, P_{t|T}, P_{t,t-1|T}, J

TASK 6 — high-level wrapper (entry point of the Kalman E-step):
  - run_kalman(Y, theta, w_u, w_eps, freq_list, save_path)
                                        : forward filter + RTS smoother in
                                          one call; returns smoothed and
                                          filtered moments, log-likelihood,
                                          augmented matrices and metadata.

The M-step and the full EM iteration (which also updates the Student-t
mixing weights) are deferred to later modules.

Notation
--------
r   : number of monthly latent factors (r = r_R + r_F + r_X = 3 here).
5r  : dimension of the augmented state tilde_f_t = (f_t, f_{t-1}, ..., f_{t-4}).
M   : number of observed series (= 20 here, 19 monthly + 1 quarterly).
"""

import numpy as np


# ─── Default Mariano-Murasawa aggregation weights ─────────────────────────────
# Weights c = (1/3, 2/3, 1, 2/3, 1/3) applied to (f_t, f_{t-1}, f_{t-2},
# f_{t-3}, f_{t-4}) when aggregating a monthly log-difference factor into a
# quarterly log-difference.  Reference: EM_for_student_t.tex, eq:mm-aggregation
# and eq:mm-observation.  These coefficients are KNOWN and FIXED (not estimated):
# they are a structural restriction implied by the log-difference aggregation of
# a chain-weighted quarterly index, not free parameters.
MM_WEIGHTS_DEFAULT: list[float] = [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]


# ─── 1. Augmented transition matrix A_tilde ───────────────────────────────────

def build_A_tilde(A: np.ndarray) -> np.ndarray:
    r"""
    Build the companion-form augmented transition matrix tilde_A (5r x 5r).

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Adapting the State-Space Model: the
    Structural Role of MM" (subsec:mm-augmentation), eq:mm-augmented-transition;
    restated in Section "The Full State-Space Model" (subsec:full-ssm),
    eq:final-state.

    Parameters
    ----------
    A : np.ndarray, shape (r, r)
        The unrestricted monthly VAR(1) transition matrix.  Entry A[i, j]
        is the effect of factor j at time t-1 on factor i at time t.  A is
        full (not block-diagonal): it captures dynamic spillovers across the
        real / financial / other blocks.

    Returns
    -------
    A_tilde : np.ndarray, shape (5r, 5r)
        The companion-form transition matrix for the augmented state
        tilde_f_t = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4}) in R^{5r}:

            A_tilde = [[A,   0,   0,   0,   0],
                       [I_r, 0,   0,   0,   0],
                       [0,   I_r, 0,   0,   0],
                       [0,   0,   I_r, 0,   0],
                       [0,   0,   0,   I_r, 0]]

    Notes
    -----
    **Why the companion form.**
    The quarterly aggregation (Mariano-Murasawa) requires the *current and
    four lagged* monthly factors to be simultaneously available in the state
    at time t.  A standard first-order state-space model carries only f_t,
    so we augment the state to stack five consecutive factor vectors.  The
    Kalman filter then operates on this 5r-dimensional state with first-order
    dynamics.

    **Reading the block rows.**
    - First block row [A, 0, 0, 0, 0] encodes the genuine dynamic equation:
      reading the first block of tilde_f_t reproduces f_t = A f_{t-1} + u_t,
      the original VAR(1).
    - Block rows 2-5 are *identity shifts*.  Block row k (for k = 1..4, in
      0-based factor-block terms) places I_r so that the new block
      f_{t-k} simply copies the previous-period block f_{t-(k-1)}.  These
      encode the definitional tautologies f_{t-k} = f_{t-k}; they propagate
      lagged factors forward in time and carry no dynamics of their own.

    **Construction.**
    Block row k (1-based, k = 1, 2, 3, 4) has I_r in block column (k-1):
    rows [k*r : (k+1)*r], columns [(k-1)*r : k*r].
    """
    r = A.shape[0]
    dim = 5 * r

    A_tilde = np.zeros((dim, dim))

    # First block row: the genuine VAR(1) transition.
    A_tilde[:r, :r] = A

    # Block rows 2..5: identity shifts that carry each lag forward by one period.
    # Block row k (k = 1, 2, 3, 4) -> I_r at rows [k*r:(k+1)*r], cols [(k-1)*r:k*r].
    for k in range(1, 5):
        A_tilde[k * r:(k + 1) * r, (k - 1) * r:k * r] = np.eye(r)

    return A_tilde


# ─── 2. Augmented innovation covariance Q_tilde ───────────────────────────────

def build_Q_tilde(Q: np.ndarray, w_u_t: float = 1.0) -> np.ndarray:
    r"""
    Build the augmented (singular) innovation covariance tilde_Q_t (5r x 5r).

    Thesis reference
    ----------------
    EM_for_student_t.tex, subsec:mm-augmentation, eq:mm-augmented-Q;
    restated in subsec:full-ssm, eq:final-state.

    Parameters
    ----------
    Q : np.ndarray, shape (r, r)
        The monthly innovation covariance, Cov(u_t) for the VAR(1) factor
        innovations.  Symmetric positive (semi-)definite.
    w_u_t : float, optional
        The factor-side mixing weight at time t in the Student-t scale-mixture
        representation, w_u_t ~ Gamma(nu_u/2, nu_u/2).  The conditional
        innovation covariance is Q / w_u_t.  Default 1.0, which recovers the
        Gaussian (un-tilted) covariance Q.

    Returns
    -------
    Q_tilde : np.ndarray, shape (5r, 5r)
        The augmented innovation covariance:

            Q_tilde = [[Q/w_u_t, 0, ..., 0],
                       [0,       0, ..., 0],
                       ...
                       [0,       0, ..., 0]]

        Only the top-left (r x r) block equals Q / w_u_t; every other entry
        is exactly zero.

    Notes
    -----
    **This matrix is SINGULAR (rank r, not 5r).**
    The augmented innovation tilde_u_t = (u_t, 0, 0, 0, 0) injects fresh
    noise only into the contemporaneous factor block f_t.  The four
    lagged-factor blocks are *deterministic* functions of past states
    (identity shifts in tilde_A), not new innovations, so they receive zero
    variance.  Consequently rank(Q_tilde) = rank(Q) = r.

    **Why singularity is not a problem.**
    The Kalman filter never inverts tilde_Q_t directly.  The augmented
    covariance enters only the prediction step

        P_{t|t-1} = tilde_A P_{t-1|t-1} tilde_A' + tilde_Q_t,

    which uses tilde_Q_t additively.  The only matrix actually inverted in the
    filter is the *innovation covariance* S_t = W_t (Lambda_tilde P Lambda_tilde'
    + R_tilde) W_t', which is non-singular as long as R_tilde is positive
    definite.  Hence a rank-deficient tilde_Q_t is handled without any special
    treatment (no pseudo-inverse required).

    **Role of w_u_t.**
    Dividing Q by the mixing weight w_u_t tilts the innovation variance:
    a small w_u_t (rare, drawn in the tail of the Gamma) inflates the
    variance, producing the heavy tails of the Student-t.  At initialisation
    all weights are 1, so Q_tilde reduces to diag(Q, 0, 0, 0, 0).
    """
    r = Q.shape[0]
    dim = 5 * r

    Q_tilde = np.zeros((dim, dim))
    Q_tilde[:r, :r] = Q / w_u_t

    return Q_tilde


# ─── 3. Augmented (effective) loading matrix Lambda_tilde ─────────────────────

def build_Lambda_tilde(
    Lambda: np.ndarray,
    freq_list: list[str],
    mm_weights: list[float] | None = None,
) -> np.ndarray:
    r"""
    Build the effective augmented loading matrix tilde_Lambda (M x 5r).

    Thesis reference
    ----------------
    EM_for_student_t.tex, subsec:mm-observation (eq:mm-observation,
    eq:mm-loading-full) and subsec:full-ssm (eq:final-obs,
    eq:final-Lambda-tilde).

    Parameters
    ----------
    Lambda : np.ndarray, shape (M, r)
        The block-diagonal monthly loading matrix from theta^(0).  Row i is
        the loading L_{i.} of series i on the r factors; by the block-diagonal
        restriction only the column of series i's own economic block is
        non-zero.
    freq_list : list[str], length M
        Frequency label of each series, "monthly" or "quarterly", in the SAME
        ROW ORDER as Lambda (i.e. the dataset column order).
    mm_weights : list[float], optional
        The five Mariano-Murasawa aggregation weights
        (w0, w1, w2, w3, w4) applied to (f_t, f_{t-1}, f_{t-2}, f_{t-3},
        f_{t-4}).  Default MM_WEIGHTS_DEFAULT = [1/3, 2/3, 1, 2/3, 1/3].

    Returns
    -------
    Lambda_tilde : np.ndarray, shape (M, 5r)
        The effective loading matrix in the augmented representation:

            Lambda_tilde = [[L^M,      0,        0,    0,        0     ],
                            [w0 L^Q,  w1 L^Q,  w2 L^Q, w3 L^Q,  w4 L^Q]]

        (monthly rows in the first block, quarterly rows in the second; the
        actual interleaving follows freq_list).

    Raises
    ------
    ValueError
        If len(freq_list) != M, if mm_weights does not have length 5, or if
        any frequency label is not "monthly"/"quarterly".

    Notes
    -----
    **What each series "sees" in the augmented state.**
    The augmented state is tilde_f_t = (f_t, f_{t-1}, f_{t-2}, f_{t-3},
    f_{t-4}) in R^{5r}, partitioned into five lag-blocks of width r.

    - *Monthly series* are observed at the same frequency as the factor, so
      they load only on the contemporaneous block f_t (the first r columns):

          y^M_{i,t} = (L_{i.}, 0, 0, 0, 0) tilde_f_t + eps_{i,t}.

      The four trailing zero-blocks mean monthly series do not load on lagged
      factors.

    - *Quarterly series* are observed only at quarter-end months and represent
      an aggregate of the unobserved monthly path.  Via the Mariano-Murasawa
      identity (eq:mm-aggregation) the quarterly log-difference equals a
      weighted sum of five consecutive monthly factor contributions:

          y^Q_{j,3m} = 1/3 L_{j.} f_{3m}   + 2/3 L_{j.} f_{3m-1}
                     +     L_{j.} f_{3m-2} + 2/3 L_{j.} f_{3m-3}
                     + 1/3 L_{j.} f_{3m-4} + eps_{j,3m}.

      Hence the effective loading row spreads L_{j.} across all five
      lag-blocks with the MM weights:

          (w0 L_{j.}, w1 L_{j.}, w2 L_{j.}, w3 L_{j.}, w4 L_{j.}).

    **Two restrictions encoded simultaneously.**
    1. *Block-diagonal (vertical):* because Lambda is block-diagonal, each
       output row has non-zero entries only in the factor-columns of its own
       block — within every lag-block.  Replicating Lambda across lag-blocks
       preserves this.
    2. *Mariano-Murasawa (horizontal):* the fixed weights {1/3, 2/3, 1, 2/3,
       1/3} are a known restriction (not estimated), dictated by the
       log-difference aggregation of a chain-weighted quarterly index.

    **Note on the underlying loading.**
    The thesis distinguishes L^M (monthly loadings) from L^Q (the *latent
    monthly* loadings of quarterly series).  In our single-quarterly-series
    setting both are simply rows of the same theta^(0) Lambda; the MM pattern
    is applied to whichever rows are flagged "quarterly" in freq_list.
    """
    if mm_weights is None:
        mm_weights = MM_WEIGHTS_DEFAULT

    M, r = Lambda.shape

    if len(freq_list) != M:
        raise ValueError(
            f"freq_list has length {len(freq_list)} but Lambda has M={M} rows."
        )
    if len(mm_weights) != 5:
        raise ValueError(
            f"mm_weights must have length 5 (one per lag-block); "
            f"got {len(mm_weights)}."
        )

    dim = 5 * r
    Lambda_tilde = np.zeros((M, dim))

    for i in range(M):
        freq = freq_list[i]
        if freq == "monthly":
            # Load only on the contemporaneous block f_t (first r columns).
            Lambda_tilde[i, :r] = Lambda[i, :]
        elif freq == "quarterly":
            # Spread the loading across all five lag-blocks with MM weights.
            for k, w in enumerate(mm_weights):
                Lambda_tilde[i, k * r:(k + 1) * r] = w * Lambda[i, :]
        else:
            raise ValueError(
                f"freq_list[{i}] = {freq!r}; expected 'monthly' or 'quarterly'."
            )

    return Lambda_tilde


# ─── 4. Augmented idiosyncratic covariance R_tilde ────────────────────────────

def build_R_tilde(R: np.ndarray, w_eps_t: float = 1.0) -> np.ndarray:
    r"""
    Build the idiosyncratic noise covariance tilde_R_t (M x M, diagonal).

    Thesis reference
    ----------------
    EM_for_student_t.tex, subsec:full-ssm, eq:final-obs
    (eps_t | w^eps_t ~ N(0, R / w^eps_t)) and the "Idiosyncratic covariance"
    paragraph (R = diag(r_1, ..., r_M)).

    Parameters
    ----------
    R : np.ndarray, shape (M,)
        The idiosyncratic variances stored as a 1-D vector (the diagonal of
        the idiosyncratic covariance), as produced by theta^(0).
    w_eps_t : float, optional
        The idiosyncratic mixing weight at time t in the Student-t scale
        mixture, w^eps_t ~ Gamma(nu_eps/2, nu_eps/2).  The conditional
        idiosyncratic covariance is R / w^eps_t.  Default 1.0 (Gaussian limit).

    Returns
    -------
    R_tilde : np.ndarray, shape (M, M)
        The diagonal matrix diag(R) / w_eps_t.  Off-diagonal entries are zero
        because the idiosyncratic errors are cross-sectionally independent.

    Notes
    -----
    **Why diagonal.**
    The DFM assumes that, conditional on the common factors, the M series are
    cross-sectionally independent: all co-movement is captured by the factors,
    leaving series-specific idiosyncratic noise that is mutually uncorrelated.
    Hence R (and tilde_R_t) is diagonal.

    **Role of w_eps_t.**
    Exactly as for the factor side, dividing by the mixing weight tilts the
    idiosyncratic variance to generate heavy tails: a small w_eps_t inflates
    the variance for that observation, accommodating outliers.  At
    initialisation all weights are 1, so tilde_R_t = diag(R).

    **Selection at runtime (later tasks).**
    For missing data, the observation equation applies a selection matrix W_t,
    so the *effective* idiosyncratic covariance for the observed sub-vector at
    time t is W_t tilde_R_t W_t'.  This module returns the full M x M matrix;
    the row/column selection is handled by the filter in a later task.
    """
    return np.diag(R) / w_eps_t


# ─── 5. Selection matrix W_t ──────────────────────────────────────────────────

def build_selection_matrix(y_t: np.ndarray) -> np.ndarray:
    r"""
    Build the selection matrix W_t for a single observation vector y_t.

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Setup: The Selection Matrix W_t"
    (subsec:selection-matrix), eq. at line ~6579.

    Parameters
    ----------
    y_t : np.ndarray, shape (M,)
        Observation vector at time t.  NaN entries indicate series that are
        not observed at time t (either ragged-edge publication lag or
        quarterly series in a non-quarter-end month).

    Returns
    -------
    W_t : np.ndarray, shape (m_t, M)
        Binary selection matrix with exactly one 1 per row.  Multiplying
        W_t @ y_t extracts the m_t observed entries from y_t (NaNs removed).

        Special case: if all M entries are NaN, W_t has shape (0, M) — zero
        rows.  Reference: line ~6816 ("limiting case m_t = 0").

    Notes
    -----
    **Construction.**
    W_t is the sub-matrix of the M x M identity obtained by keeping only the
    rows whose index appears in obs_idx = {i : y_t[i] is not NaN}:

        W_t = I_M[obs_idx, :]

    Each row of W_t has a single 1 in the column of the corresponding
    observed series, and zeros elsewhere.  Hence W_t @ y_t is the m_t-vector
    of observed values, and W_t @ Lambda_tilde is the effective loading matrix
    restricted to observed series.

    **W_t is known and fixed.**
    W_t is determined entirely by the release schedule (which series are
    published at time t), not estimated.  It is NOT saved to disk; it is
    reconstructed at runtime from the NaN pattern of the dataset.
    Reference: line ~6588.

    **Why float.**
    W_t is stored as float64 (not int) so that matrix products such as
    W_t @ Lambda_tilde and W_t @ R_tilde @ W_t.T remain in a uniform numeric
    type throughout the Kalman filter.
    """
    M = y_t.shape[0]
    obs_idx = np.where(~np.isnan(y_t))[0]
    W_t = np.eye(M, dtype=float)[obs_idx, :]
    return W_t


def build_all_selection_matrices(Y: np.ndarray) -> list[np.ndarray]:
    r"""
    Build selection matrices W_t for every time step in the panel Y.

    Thesis reference
    ----------------
    EM_for_student_t.tex, subsec:selection-matrix, line ~6588:
    "W_t is known and fixed: determined entirely by the release schedule,
    not estimated."

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Full observation panel with NaN for missing values.  Rows are time
        periods (months), columns are the M observed series.

    Returns
    -------
    W_list : list of np.ndarray, length T
        W_list[t] = W_t, a float64 array of shape (m_t, M), where
        m_t = number of non-NaN entries in Y[t, :].

    Notes
    -----
    **Not saved to disk.**
    The W_t matrices are NOT persisted anywhere.  They are lightweight
    (each at most M x M, typically sparse) and reconstructed at runtime from
    the NaN pattern of the dataset on every call.  The NaN pattern itself is
    fully determined by the data release schedule and never needs to be
    estimated or stored separately.

    **Usage in the Kalman filter.**
    At each time step t the filter uses W_list[t] to project the full
    observation equation onto the observed sub-space:

        y_t_obs = W_t @ y_t       (m_t-vector, NaN-free)
        H_t     = W_t @ Lambda_tilde           (m_t x 5r effective loading)
        S_t     = H_t @ P_{t|t-1} @ H_t.T + W_t @ R_tilde @ W_t.T   (m_t x m_t)

    **Characteristic m_t values in our dataset (M=20: 19 monthly + 1 quarterly).**
    m_t=20: quarter-end month, all series observed (richest information set).
    m_t=19: non-quarter-end month, GDP structurally absent (quarterly mask).
    m_t<19: ragged edge near sample end, one or more monthly series not yet released.
    m_t=1:  extreme ragged edge (Apr–May 2026), only NFCI available.
    The quarterly mask and the ragged edge are handled by the SAME W_t mechanism —
    no special-case code is needed for quarterly vs. ragged missingness.
    """
    T = Y.shape[0]
    return [build_selection_matrix(Y[t, :]) for t in range(T)]


# ─── 6. Kalman prediction step ───────────────────────────────────────────────

def kalman_predict(
    f_filt_prev: np.ndarray,
    P_filt_prev: np.ndarray,
    A_tilde: np.ndarray,
    Q_tilde: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    One-step-ahead prediction of the augmented state (5r-dimensional).

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Forward Recursion: the Filter"
    (subsec:kf-forward, riga ~4645), eq:kf-predict (riga ~4688).

    Parameters
    ----------
    f_filt_prev : np.ndarray, shape (5r,)
        Filtered mean f_{t-1|t-1} from the previous time step.
    P_filt_prev : np.ndarray, shape (5r, 5r)
        Filtered covariance P_{t-1|t-1} from the previous time step.
    A_tilde : np.ndarray, shape (5r, 5r)
        Companion-form transition matrix (time-invariant, see build_A_tilde).
    Q_tilde : np.ndarray, shape (5r, 5r)
        Augmented innovation covariance at time t, i.e. with top-left block
        Q / w_u_t (see build_Q_tilde).  In the Student-t model this is
        time-varying; caller is responsible for passing the correct version
        for the current t.

    Returns
    -------
    f_pred : np.ndarray, shape (5r,)
        Predicted mean f_{t|t-1} = A_tilde @ f_{t-1|t-1}.
    P_pred : np.ndarray, shape (5r, 5r)
        Predicted covariance P_{t|t-1} =
        A_tilde @ P_{t-1|t-1} @ A_tilde.T + Q_tilde.

    Notes
    -----
    **No dependence on y_t.**
    The prediction step uses only the previous filtered state and the
    transition model; it is identical whether or not y_t contains NaNs.
    The missing-data logic enters exclusively in kalman_update via W_t.
    Reference: subsec:kf-missing, riga ~6788.

    **Time-varying Q_tilde in the Student-t model.**
    The mixing weight w_u_t ~ Gamma(nu_u/2, nu_u/2) scales the factor
    innovation variance at each t.  During EM initialisation all w_u_t = 1,
    so Q_tilde is constant across time; in the E-step it varies with the
    drawn weights.
    """
    f_pred = A_tilde @ f_filt_prev
    P_pred = A_tilde @ P_filt_prev @ A_tilde.T + Q_tilde
    return f_pred, P_pred


# ─── 7. Kalman update step ───────────────────────────────────────────────────

def kalman_update(
    f_pred: np.ndarray,
    P_pred: np.ndarray,
    y_t: np.ndarray,
    W_t: np.ndarray,
    Lambda_tilde: np.ndarray,
    R_tilde: np.ndarray,
) -> dict:
    r"""
    Kalman update step for a single time point with missing data.

    Thesis reference
    ----------------
    EM_for_student_t.tex, subsec:kf-missing (riga ~6780),
    eq:kf-innov-missing through eq:kf-update (righe ~6800-6806).
    Log-likelihood contribution: eq:kf-loglik (riga ~6810).

    Parameters
    ----------
    f_pred : np.ndarray, shape (5r,)
        Predicted mean f_{t|t-1} from kalman_predict.
    P_pred : np.ndarray, shape (5r, 5r)
        Predicted covariance P_{t|t-1} from kalman_predict.
    y_t : np.ndarray, shape (M,)
        Raw observation vector at time t.  NaN entries mark unobserved
        series (ragged edge or quarterly mask); they are handled by W_t
        and do not enter the algebra directly.
    W_t : np.ndarray, shape (m_t, M)
        Selection matrix for time t (see build_selection_matrix).  Rows
        correspond to the m_t observed series.  If m_t = 0 (all-NaN row),
        W_t has shape (0, M) and the update is skipped entirely.
    Lambda_tilde : np.ndarray, shape (M, 5r)
        Augmented loading matrix (see build_Lambda_tilde).
    R_tilde : np.ndarray, shape (M, M)
        Full idiosyncratic covariance at time t, i.e. diag(R) / w_eps_t
        (see build_R_tilde).  In the Student-t model this is time-varying;
        caller passes the version scaled by the current mixing weight w_eps_t.

    Returns
    -------
    dict with keys
        f_filt   : np.ndarray (5r,)     -- filtered mean f_{t|t}
        P_filt   : np.ndarray (5r, 5r)  -- filtered covariance P_{t|t}
        eta      : np.ndarray (m_t,)    -- observed innovation y_obs - WL @ f_pred
        S        : np.ndarray (m_t,m_t) -- innovation covariance
        loglik_t : float                -- log-likelihood contribution at time t
        K        : np.ndarray (5r, m_t) -- Kalman gain at time t
        WL       : np.ndarray (m_t, 5r) -- effective loading W_t @ Lambda_tilde

    ``K`` and ``WL`` are returned so that the caller (e.g. the full filter and
    the RTS smoother) can build the lag-one smoothed covariance without
    recomputing them.  For ``m_t = 0`` they are stored as zero-row/-column
    empty arrays of the correct dtype.

    Notes
    -----
    **Missing-data projection.**
    The full observation equation is y_t = Lambda_tilde @ f_t + eps_t.
    When some entries are missing, only the m_t observed rows are used:

        y_obs = W_t @ y_t              (m_t-vector of observed values)
        WL    = W_t @ Lambda_tilde     (m_t x 5r, effective loading)
        WR    = W_t @ R_tilde @ W_t.T  (m_t x m_t, effective idiosyncratic cov)

    Every filter equation then reduces to its fully-observed counterpart
    evaluated on the m_t-dimensional observed sub-space.
    Reference: riga ~6800.

    **NaN substitution before W_t multiplication.**
    W_t selects only columns corresponding to observed series, so NaN values
    in y_t are never algebraically accessed.  Nevertheless, to prevent
    0 * NaN = NaN propagation in numpy, NaNs in y_t are replaced with 0
    before computing y_obs = W_t @ y_filled.

    **Kalman gain via np.linalg.solve.**
    K = P_pred @ WL.T @ inv(S) is computed without forming inv(S) explicitly.
    Writing K.T = inv(S) @ WL @ P_pred and exploiting symmetry of S and P_pred:

        S @ K.T = WL @ P_pred   =>   K.T = np.linalg.solve(S, WL @ P_pred)

    **Log-likelihood.**
    loglik_t = -0.5 * [m_t * log(2*pi) + log|S| + eta.T @ inv(S) @ eta].
    log|S| is computed via np.linalg.slogdet to avoid overflow/underflow for
    large m_t or poorly conditioned S.

    **m_t = 0 (all-NaN time step).**
    If W_t has 0 rows, the update is skipped: f_filt = f_pred, P_filt = P_pred,
    eta and S are empty, loglik_t = 0.  Reference: riga ~6816.
    """
    m_t = W_t.shape[0]

    # ── Special case: no observations at time t ───────────────────────────────
    if m_t == 0:
        dim_aug = f_pred.shape[0]
        return {
            "f_filt": f_pred.copy(),
            "P_filt": P_pred.copy(),
            "eta": np.empty(0),
            "S": np.empty((0, 0)),
            "loglik_t": 0.0,
            "K": np.zeros((dim_aug, 0)),
            "WL": np.zeros((0, Lambda_tilde.shape[1])),
        }

    # ── Step 1: extract observed values (replace NaN before multiplying W_t) ──
    y_filled = np.where(np.isnan(y_t), 0.0, y_t)
    y_obs = W_t @ y_filled           # (m_t,)

    # ── Step 2: effective loading and effective idiosyncratic covariance ──────
    WL = W_t @ Lambda_tilde          # (m_t, 5r)
    WR = W_t @ R_tilde @ W_t.T      # (m_t, m_t)

    # ── Step 3: innovation ────────────────────────────────────────────────────
    eta = y_obs - WL @ f_pred        # (m_t,)

    # ── Step 4: innovation covariance ─────────────────────────────────────────
    S = WL @ P_pred @ WL.T + WR     # (m_t, m_t)

    # ── Step 5: Kalman gain via solve (avoids explicit matrix inversion) ──────
    # K = P_pred @ WL.T @ inv(S)  <==>  S @ K.T = WL @ P_pred
    K_T = np.linalg.solve(S, WL @ P_pred)   # (m_t, 5r)
    K = K_T.T                                # (5r, m_t)

    # ── Step 6: state update ──────────────────────────────────────────────────
    f_filt = f_pred + K @ eta
    P_filt = P_pred - K @ WL @ P_pred
    P_filt = 0.5 * (P_filt + P_filt.T)      # symmetrise against floating-point drift

    # ── Step 7: log-likelihood contribution ───────────────────────────────────
    _, logdet_S = np.linalg.slogdet(S)
    S_inv_eta = np.linalg.solve(S, eta)
    loglik_t = -0.5 * (m_t * np.log(2.0 * np.pi) + logdet_S + float(eta @ S_inv_eta))

    return {
        "f_filt": f_filt,
        "P_filt": P_filt,
        "eta": eta,
        "S": S,
        "loglik_t": loglik_t,
        "K": K,
        "WL": WL,
    }


# ─── 8. Full forward Kalman filter ──────────────────────────────────────────

def kalman_filter(
    Y: np.ndarray,
    theta: dict,
    w_u: np.ndarray | None = None,
    w_eps: np.ndarray | None = None,
    freq_list: list[str] | None = None,
) -> dict:
    r"""
    Full forward Kalman filter over the entire sample T.

    Runs the predict + update loop from t=0 to T-1, storing all filtered
    and predicted moments needed by the smoother (Task 5) and the E-step.

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Forward Recursion: the Filter"
    (subsec:kf-forward, riga ~4645).

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Full observation panel.  NaN entries mark unobserved series at a
        given time step (quarterly mask, ragged edge, etc.).
    theta : dict-like
        Parameter container.  Accessed keys:
          ``A``       (r, r)     VAR(1) transition matrix.
          ``Q``       (r, r)     Factor innovation covariance.
          ``Lambda``  (M, r)     Block-diagonal loading matrix.
          ``R``       (M,)       Idiosyncratic variances (diagonal of R_tilde).
          ``Sigma_0`` (5r, 5r)   Initial state covariance P_{-1|-1}.
        Typically a ``numpy.lib.npyio.NpzFile`` loaded via ``np.load``.
    w_u : np.ndarray (T,) or None
        Factor-side Student-t mixing weights w^u_t > 0.  In the scale-mixture
        representation the conditional factor innovation covariance is Q/w_u_t.
        If None, all weights default to 1.0 (Gaussian / EM initialisation).
    w_eps : np.ndarray (T,) or None
        Idiosyncratic-side Student-t mixing weights w^eps_t > 0.  The
        conditional idiosyncratic covariance is diag(R)/w_eps_t.
        If None, all weights default to 1.0.
    freq_list : list[str] or None
        Frequency label (``'monthly'`` or ``'quarterly'``) for each of the M
        columns of Y, in the same order as the dataset columns.  Passed to
        :func:`build_Lambda_tilde`.  Required; raises ``ValueError`` if None.

    Returns
    -------
    dict with keys
        ``f_pred``  : np.ndarray (T, 5r)      -- predicted means f_{t|t-1}
        ``P_pred``  : np.ndarray (T, 5r, 5r)  -- predicted covariances P_{t|t-1}
        ``f_filt``  : np.ndarray (T, 5r)      -- filtered means f_{t|t}
        ``P_filt``  : np.ndarray (T, 5r, 5r)  -- filtered covariances P_{t|t}
        ``K_list``  : list[np.ndarray]        -- length T; entry t has shape
                                                 (5r, m_t), Kalman gain at time t
                                                 (empty (5r, 0) when m_t = 0)
        ``WL_list`` : list[np.ndarray]        -- length T; entry t has shape
                                                 (m_t, 5r), effective loading
                                                 W_t @ Lambda_tilde (empty (0, 5r)
                                                 when m_t = 0)
        ``loglik``  : float                    -- total log-likelihood (sum over t)

    ``K_list`` and ``WL_list`` are stored as Python lists (not stacked arrays)
    because m_t varies across t (ragged edge + quarterly mask).  The RTS
    smoother (Task 5) consumes them to initialise the lag-one cross-covariance
    P_{T-1, T-2 | T} = (I - K_{T-1} WL_{T-1}) A_tilde P_{T-2 | T-2}.

    Indexing convention (important — avoids off-by-one errors)
    ----------------------------------------------------------
    The thesis (riga ~4654) initialises with ``(f_{0|0}, P_{0|0}) = (0, Sigma_0)``
    *before* seeing any data and then loops over t = 1, 2, ..., T (1-based).

    In Python (0-indexed), we adopt the following convention:

    ::

        f_init = 0_{5r},  P_init = Sigma_0      [conceptually f_{-1|-1},
                                                  the state before any data]

        t = 0:  predict from (f_init, P_init)  --> f_pred[0], P_pred[0]
                update  with Y[0]              --> f_filt[0], P_filt[0]

        t = 1..T-1:
                predict from (f_filt[t-1], P_filt[t-1]) --> f_pred[t], P_pred[t]
                update  with Y[t]                        --> f_filt[t], P_filt[t]

    Python index t corresponds to thesis index t+1 (1-based), but both
    represent the same information state: the filter conditioned on
    Y[0], ..., Y[t].  There is no off-by-one in the stored moments.

    Notes
    -----
    **Time-varying noise matrices.**
    ``Q_tilde`` and ``R_tilde`` are rebuilt at every t because the Student-t
    weights can vary across time steps.  In the Gaussian baseline (all weights 1)
    these matrices are constant, but rebuilding them costs little and keeps the
    code general.

    **Pre-allocation.**
    Output arrays are pre-allocated to ``np.zeros`` before the loop to avoid
    repeated dynamic allocation and to simplify indexing.

    **m_t = 0 handling.**
    If a time step has no observed series (all-NaN row), ``kalman_update``
    skips the update: ``f_filt[t] = f_pred[t]``, ``P_filt[t] = P_pred[t]``,
    ``loglik_t = 0``.  No special-case code is needed here.

    **Interpretation of the filtered factors (Gaussian baseline, theta^(0)).**
    f[0] (real): near-zero with rare large spikes (GFC ~-0.06, COVID ~-0.36),
    reflecting high Q[0,0] and low persistence — large but short-lived shocks.
    f[1] (financial): smooth, persistent, cycling 0..6 over the sample (dominated
    by credit/term spreads); low Q[1,1] and high VAR eigenvalue accumulate small
    shocks into long swings. f[2] (other): noisy, mean-reverting around zero
    (CPI/PCE inflation differences, Q[2,2] moderate, low persistence).
    At the ragged edge (Apr-May 2026, m_t=1), only NFCI is observed: f[1]
    keeps updating while f[0] and f[2] are extrapolated via A_tilde.
    These are Gaussian (all weights=1) paths; Student-t weights will down-weight
    outlier periods such as COVID, producing more robust factor estimates.
    """
    if freq_list is None:
        raise ValueError(
            "freq_list is required: pass a list of 'monthly'/'quarterly' "
            "with one entry per column of Y."
        )

    T, M = Y.shape
    A       = theta["A"]        # (r, r)
    Q       = theta["Q"]        # (r, r)
    Lambda  = theta["Lambda"]   # (M, r)
    R       = theta["R"]        # (M,)
    Sigma_0 = theta["Sigma_0"]  # (5r, 5r)

    r   = A.shape[0]
    dim = 5 * r

    if w_u is None:
        w_u = np.ones(T)
    if w_eps is None:
        w_eps = np.ones(T)

    # ── Build time-invariant matrices once ────────────────────────────────────
    A_tilde      = build_A_tilde(A)
    Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)

    # ── Selection matrices for all T time steps ───────────────────────────────
    W_list = build_all_selection_matrices(Y)

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    f_pred_arr = np.zeros((T, dim))
    P_pred_arr = np.zeros((T, dim, dim))
    f_filt_arr = np.zeros((T, dim))
    P_filt_arr = np.zeros((T, dim, dim))
    # K and WL have time-varying shape (m_t differs across t); store as lists.
    K_list: list[np.ndarray]  = [np.zeros((dim, 0))] * T
    WL_list: list[np.ndarray] = [np.zeros((0, dim))] * T
    loglik_total = 0.0

    # ── Initial state: f_{-1|-1} = 0, P_{-1|-1} = Sigma_0 ───────────────────
    f_prev = np.zeros(dim)
    P_prev = Sigma_0.copy()

    # ── Main forward loop ─────────────────────────────────────────────────────
    for t in range(T):
        # Rebuild noise matrices at every t (handles Student-t weight variation)
        Q_tilde_t = build_Q_tilde(Q, float(w_u[t]))
        R_tilde_t = build_R_tilde(R, float(w_eps[t]))

        # Prediction step: f_{t|t-1}, P_{t|t-1}
        f_p, P_p = kalman_predict(f_prev, P_prev, A_tilde, Q_tilde_t)
        f_pred_arr[t] = f_p
        P_pred_arr[t] = P_p

        # Update step: f_{t|t}, P_{t|t}, loglik_t
        upd = kalman_update(f_p, P_p, Y[t], W_list[t], Lambda_tilde, R_tilde_t)
        f_filt_arr[t] = upd["f_filt"]
        P_filt_arr[t] = upd["P_filt"]
        K_list[t]     = upd["K"]
        WL_list[t]    = upd["WL"]
        loglik_total  += upd["loglik_t"]

        # Pass filtered state to next iteration
        f_prev = f_filt_arr[t]
        P_prev = P_filt_arr[t]

    return {
        "f_pred": f_pred_arr,
        "P_pred": P_pred_arr,
        "f_filt": f_filt_arr,
        "P_filt": P_filt_arr,
        "K_list": K_list,
        "WL_list": WL_list,
        "loglik": loglik_total,
    }


# ─── 9. Rauch-Tung-Striebel backward smoother + lag-one covariance ──────────

def kalman_smoother(
    filter_out: dict,
    A_tilde: np.ndarray,
    jitter: float = 0.0,
) -> dict:
    r"""
    Backward Rauch-Tung-Striebel (RTS) smoother + lag-one smoothed covariance.

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Backward Recursion: the Rauch-Tung-Striebel
    Smoother" (subsec:rts-backward, riga ~4858) — equations rts-gain, rts-mean,
    rts-cov for the smoothed mean and covariance.  Section "The Lag-One Smoothed
    Covariance" (subsec:rts-lag-cov, righe ~4911-4925) — initialisation and
    recursion for P_{t, t-1 | T}.

    Parameters
    ----------
    filter_out : dict
        Output of :func:`kalman_filter`.  Required keys: ``f_pred``, ``P_pred``,
        ``f_filt``, ``P_filt``, ``K_list``, ``WL_list``.
    A_tilde : np.ndarray, shape (5r, 5r)
        Companion-form state transition matrix (time-invariant; see
        :func:`build_A_tilde`).  Must be the same matrix used inside the filter
        that produced ``filter_out``.
    jitter : float, optional
        Diagonal regularisation added to ``P_pred[t+1]`` before inversion.
        If ``jitter == 0`` (default) the inverse is computed with
        ``np.linalg.pinv`` (SVD-based pseudo-inverse) to handle the structural
        singularity of ``P_pred`` (see Notes).  A small positive value (e.g.
        ``1e-10``) switches to the regular ``np.linalg.inv`` of
        ``P_pred[t+1] + jitter * I``.

    Returns
    -------
    dict with keys
        ``f_smooth`` : np.ndarray (T, 5r)      -- smoothed means f_{t|T}
        ``P_smooth`` : np.ndarray (T, 5r, 5r)  -- smoothed covariances P_{t|T}
        ``P_lag``    : np.ndarray (T, 5r, 5r)  -- lag-one smoothed cross-covariance
                                                  P_{t, t-1 | T}.  Defined for
                                                  t = 1, ..., T-1; ``P_lag[0]`` is
                                                  filled with zeros (no prior
                                                  period at t = 0).
        ``J``        : np.ndarray (T, 5r, 5r)  -- smoother gains J_t for
                                                  t = 0, ..., T-2; ``J[T-1]`` is
                                                  filled with zeros (unused).

    Algorithm
    ---------
    **Boundary condition (no future beyond t = T-1).**

    ::

        f_smooth[T-1] = f_filt[T-1]
        P_smooth[T-1] = P_filt[T-1]

    **Backward recursion for t = T-2, T-3, ..., 0** (eq:rts-gain, eq:rts-mean,
    eq:rts-cov):

    ::

        J_t           = P_filt[t] @ A_tilde.T @ inv(P_pred[t+1])
        f_smooth[t]   = f_filt[t]   + J_t @ (f_smooth[t+1] - f_pred[t+1])
        P_smooth[t]   = P_filt[t]   + J_t @ (P_smooth[t+1] - P_pred[t+1]) @ J_t.T

    **Lag-one cross-covariance** (eq:rts-lag-cov):

    ::

        P_lag[T-1]    = (I - K_{T-1} WL_{T-1}) @ A_tilde @ P_filt[T-2]

        for t = T-2, T-3, ..., 1:
            P_lag[t]  = P_filt[t] @ J_{t-1}.T
                      + J_t @ (P_lag[t+1] - A_tilde @ P_filt[t]) @ J_{t-1}.T

    The initialisation comes from the standard identity
    Cov(f_{T-1}, f_{T-2} | y_{1:T-1}) = (I - K_{T-1} WL_{T-1}) A_tilde P_filt[T-2],
    which uses the Kalman gain and effective loading at the LAST update step.
    Both ``K_{T-1}`` and ``WL_{T-1}`` are read from ``filter_out``; if
    ``m_{T-1} = 0`` (no observation at the last step), they are stored as
    empty arrays whose product is the zero matrix, and the formula reduces to
    ``P_lag[T-1] = A_tilde @ P_filt[T-2]`` as expected.

    Notes
    -----
    **Singularity of P_pred[t+1].**
    In the augmented state-space, only the contemporaneous factor block carries
    fresh innovation (Q_tilde has rank r, not 5r), while the four lag-blocks are
    deterministic shifts.  Hence ``P_pred = A_tilde P_filt A_tilde.T + Q_tilde``
    is in general rank-deficient (rank up to 4r + r = 5r minus the lag-blocks
    that perfectly replicate earlier filtered components).  Direct
    ``np.linalg.inv`` may either fail or amplify floating-point noise.

    We offer two safe options:

    * **Default (jitter = 0)**: invert with ``np.linalg.pinv`` (Moore-Penrose
      pseudo-inverse via SVD).  Numerically robust to exact singularity; SVD
      of a 15 x 15 matrix done T times costs a few milliseconds.
    * **jitter > 0**: invert ``P_pred[t+1] + jitter * I`` with ``np.linalg.inv``.
      Faster and avoids SVD, at the cost of an O(jitter) bias.  Set jitter =
      1e-10 if the smoother is to be called many times inside an EM loop.

    Both are theoretically valid because ``J_t`` only enters multiplicatively
    against the column space of ``A_tilde.T``, on which ``P_pred[t+1]`` is in
    fact invertible.

    **Why the lag-one covariance.**
    The M-step needs the second moment
    ``E[f_t f_{t-1}.T | y_{1:T}] = P_lag[t] + f_smooth[t] f_smooth[t-1].T``
    to update the VAR(1) transition matrix A and the innovation covariance Q
    (see subsec:rts-lag-cov, righe ~4914-4917).  Without ``P_lag``, the
    M-step estimators of A and Q would be biased toward zero.

    **Symmetrisation.**
    ``P_smooth[t]`` is symmetrised after each backward step to absorb
    floating-point asymmetry; ``P_lag[t]`` is NOT symmetric in general
    (lag-one cross-covariance is asymmetric).

    **Smoother reduces uncertainty.**
    Because the smoother conditions on the full sample y_{1:T} rather than the
    causal cone y_{1:t}, the smoothed covariance is no larger than the filtered
    one in the trace sense: ``trace(P_smooth[t]) <= trace(P_filt[t])`` for every
    t (modulo floating-point tolerance).  This is checked in the self-test.

    **Filtered vs smoothed factors: three key properties (self-test).**
    (1) *Variance reduction*: ``trace(P_smooth[t]) <= trace(P_filt[t])`` at
    every t by construction. (2) *Sharper recession troughs*: the smoothed real
    factor reaches more pronounced extrema at turning points — at COVID (2020-04)
    roughly 60 % deeper than the filtered estimate, because the smoother
    retroactively incorporates the sharp recovery that follows. The filtered path
    is necessarily more tentative at turning points (causal: past and present
    only). (3) *Convergence at the sample end*: ``f_smooth[T-1] == f_filt[T-1]``
    by initialisation; the smoothing correction grows as we move backward from T.
    Under the Student-t extension, the latent weight ``w_t`` for April 2020 will
    be small (outlier), dampening the COVID trough relative to the Gaussian
    smoother — comparing the two paths is a key diagnostic of the robustness
    gain (Section 1 of the thesis).
    """
    f_pred  = filter_out["f_pred"]
    P_pred  = filter_out["P_pred"]
    f_filt  = filter_out["f_filt"]
    P_filt  = filter_out["P_filt"]
    K_list  = filter_out["K_list"]
    WL_list = filter_out["WL_list"]

    T, dim = f_filt.shape   # dim = 5r
    I_dim  = np.eye(dim)

    # ── Pre-allocate outputs ─────────────────────────────────────────────────
    f_smooth = np.zeros_like(f_filt)
    P_smooth = np.zeros_like(P_filt)
    P_lag    = np.zeros_like(P_filt)      # P_lag[0] left at zero (undefined)
    J_arr    = np.zeros_like(P_filt)      # J_arr[T-1] left at zero (unused)

    # ── Boundary at t = T-1 ──────────────────────────────────────────────────
    f_smooth[T - 1] = f_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    # ── Backward recursion: t = T-2, T-3, ..., 0 ─────────────────────────────
    for t in range(T - 2, -1, -1):
        # Invert the (possibly singular) prediction covariance.
        if jitter > 0.0:
            P_pred_inv = np.linalg.inv(P_pred[t + 1] + jitter * I_dim)
        else:
            P_pred_inv = np.linalg.pinv(P_pred[t + 1])

        J_t = P_filt[t] @ A_tilde.T @ P_pred_inv      # (5r, 5r)
        J_arr[t] = J_t

        # Smoothed mean and covariance.
        f_smooth[t] = f_filt[t] + J_t @ (f_smooth[t + 1] - f_pred[t + 1])
        P_smooth[t] = (
            P_filt[t]
            + J_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ J_t.T
        )
        # Symmetrise against floating-point drift.
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    # ── Lag-one smoothed covariance P_{t, t-1 | T} ───────────────────────────
    if T >= 2:
        K_last  = K_list[T - 1]    # (5r, m_{T-1})
        WL_last = WL_list[T - 1]   # (m_{T-1}, 5r)
        # If m_{T-1} = 0, K_last and WL_last are empty: their product is the
        # zero (5r x 5r) matrix and the formula degenerates to A_tilde P_filt[T-2].
        P_lag[T - 1] = (I_dim - K_last @ WL_last) @ A_tilde @ P_filt[T - 2]

        # Backward recursion for t = T-2, T-3, ..., 1.
        for t in range(T - 2, 0, -1):
            J_prev = J_arr[t - 1]
            J_t    = J_arr[t]
            P_lag[t] = (
                P_filt[t] @ J_prev.T
                + J_t @ (P_lag[t + 1] - A_tilde @ P_filt[t]) @ J_prev.T
            )

    return {
        "f_smooth": f_smooth,
        "P_smooth": P_smooth,
        "P_lag":    P_lag,
        "J":        J_arr,
    }


# ─── 10. High-level wrapper: filter + smoother (Task 6) ──────────────────────

def run_kalman(
    Y: np.ndarray,
    theta: dict,
    w_u: np.ndarray | None = None,
    w_eps: np.ndarray | None = None,
    freq_list: list[str] | None = None,
    save_path=None,
) -> dict:
    r"""
    High-level entry point: forward Kalman filter + RTS smoother in one call.

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Summary of the E-Step So Far"
    (subsec:e-step-summary, riga ~4936).  This wrapper produces the smoothed
    moments that the M-step needs via the identities (righe ~4945-4947):

        E[f_t f_t.T     | y_{1:T}] = P_smooth[t] + f_smooth[t] f_smooth[t].T
        E[f_t f_{t-1}.T | y_{1:T}] = P_lag[t]    + f_smooth[t] f_smooth[t-1].T

    These second moments enter the closed-form M-step updates for the loading
    matrix Lambda, the VAR(1) transition A, the factor innovation covariance Q
    and the idiosyncratic variances R.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Full observation panel with NaN for missing values (ragged edge,
        quarterly mask).
    theta : dict-like
        Parameter container.  Accessed keys: ``A`` (r, r), ``Q`` (r, r),
        ``Lambda`` (M, r), ``R`` (M,), ``Sigma_0`` (5r, 5r).  Typically a
        ``numpy.lib.npyio.NpzFile`` loaded via ``np.load``.
    w_u : np.ndarray (T,) or None, optional
        Factor-side Student-t mixing weights.  Default None -> all ones
        (Gaussian baseline / EM initialisation).
    w_eps : np.ndarray (T,) or None, optional
        Idiosyncratic-side Student-t mixing weights.  Default None -> all ones.
    freq_list : list[str] or None, optional
        Frequency label ('monthly' / 'quarterly') for each of the M columns of
        Y.  Default None -> imported from :mod:`data_loader` as
        ``[FREQ[col] for col in ORDERED_COLS]`` (the canonical column order
        used throughout the project).
    save_path : str or pathlib.Path or None, optional
        If provided, persist ``f_smooth``, ``P_smooth``, ``P_lag`` and
        ``loglik`` to an ``.npz`` file at this path.  Useful for inspecting
        results or for passing them to downstream modules without re-running
        filter + smoother.  Default None (nothing saved).

    Returns
    -------
    dict with keys
        # Smoothed moments — principal E-step outputs.
        ``f_smooth``     : np.ndarray (T, 5r)      E[f_t | y_{1:T}]
        ``P_smooth``     : np.ndarray (T, 5r, 5r)  Var[f_t | y_{1:T}]
        ``P_lag``        : np.ndarray (T, 5r, 5r)  Cov[f_t, f_{t-1} | y_{1:T}]

        # Filtered moments — useful for diagnostics.
        ``f_filt``       : np.ndarray (T, 5r)
        ``P_filt``       : np.ndarray (T, 5r, 5r)

        # Log-likelihood — monitor EM convergence (must increase monotonically).
        ``loglik``       : float

        # Augmented matrices built once — exposed for the M-step / debug.
        ``A_tilde``      : np.ndarray (5r, 5r)
        ``Lambda_tilde`` : np.ndarray (M, 5r)

        # Metadata.
        ``T``, ``M``, ``r`` : int

    Notes
    -----
    **What this wrapper does (and does NOT do).**
    ``run_kalman`` is the ENTRY POINT for the Kalman portion of the E-step:
    given the current parameters theta and the latent weights (w_u, w_eps), it
    computes the conditional means and covariances of the factor states given
    the full data sample.  It does NOT compute the Student-t mixing weights —
    those are produced by the full E-step (a separate module, ``em_e_step``),
    which calls ``run_kalman`` inside its own update loop.

    **Gaussian baseline vs Student-t.**
    With all weights = 1 (default), this reduces to the standard mixed-frequency
    Kalman filter/smoother of Bañbura & Modugno (2014).  In the full Student-t
    EM, the weights are time-varying and rebuilt at every iteration.

    **No code duplication.**
    All algebra lives in :func:`kalman_filter` and :func:`kalman_smoother`;
    this wrapper only orchestrates them and packages the combined output.
    """
    # ── 1. Resolve freq_list (default: from data_loader) ─────────────────────
    if freq_list is None:
        import pathlib as _pathlib  # noqa: PLC0415
        import sys as _sys          # noqa: PLC0415
        _src_dir = str(_pathlib.Path(__file__).resolve().parent)
        if _src_dir not in _sys.path:
            _sys.path.insert(0, _src_dir)
        from data_loader import FREQ, ORDERED_COLS  # noqa: PLC0415
        freq_list = [FREQ[col] for col in ORDERED_COLS]

    # ── 2. Extract dimensions ─────────────────────────────────────────────────
    T, M = Y.shape
    r = theta["A"].shape[0]

    # ── 3. Default weights = 1 (Gaussian baseline) ────────────────────────────
    if w_u is None:
        w_u = np.ones(T)
    if w_eps is None:
        w_eps = np.ones(T)

    # ── 4. Build augmented matrices (computed once; reused in output dict) ────
    A_tilde      = build_A_tilde(theta["A"])
    Lambda_tilde = build_Lambda_tilde(theta["Lambda"], freq_list)

    # ── 5. Forward Kalman filter ──────────────────────────────────────────────
    filter_out = kalman_filter(
        Y, theta, w_u=w_u, w_eps=w_eps, freq_list=freq_list,
    )

    # ── 6. Backward RTS smoother + lag-one covariance ─────────────────────────
    smoother_out = kalman_smoother(filter_out, A_tilde, jitter=0.0)

    # ── 7. Assemble combined output ───────────────────────────────────────────
    out = {
        # smoothed moments (principal E-step outputs)
        "f_smooth": smoother_out["f_smooth"],
        "P_smooth": smoother_out["P_smooth"],
        "P_lag":    smoother_out["P_lag"],
        # filtered moments (diagnostics)
        "f_filt":   filter_out["f_filt"],
        "P_filt":   filter_out["P_filt"],
        # log-likelihood (EM monitoring)
        "loglik":   filter_out["loglik"],
        # augmented matrices (reusable by M-step / debug)
        "A_tilde":      A_tilde,
        "Lambda_tilde": Lambda_tilde,
        # metadata
        "T": T, "M": M, "r": r,
    }

    # ── 8. Optional persistence to .npz ───────────────────────────────────────
    if save_path is not None:
        np.savez(
            save_path,
            f_smooth=out["f_smooth"],
            P_smooth=out["P_smooth"],
            P_lag=out["P_lag"],
            loglik=out["loglik"],
        )

    return out


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    # ── parse config flag ─────────────────────────────────────────────────────
    _src_dir = str(pathlib.Path(__file__).resolve().parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from config_utils import parse_config_args, resolve_output_path, get_project_root

    _args = parse_config_args("kalman.py self-test — Kalman filter + smoother.")
    _cfg  = _args.config

    # ── locate project root and resolve config-specific paths ─────────────────
    project_root = get_project_root()
    npz_path     = resolve_output_path("processed", "theta_initial.npz", _cfg)
    csv_path     = resolve_output_path("dataset", "", _cfg)

    # ── load FREQ for this config ─────────────────────────────────────────────
    from data_loader import load_config as _dl_load_config
    FREQ = _dl_load_config(_cfg)["FREQ"]

    print(f"Loading theta^(0) from: {npz_path}")
    theta = np.load(npz_path)
    A = theta["A"]
    Q = theta["Q"]
    Lambda = theta["Lambda"]
    R = theta["R"]

    r = A.shape[0]
    M = Lambda.shape[0]
    print(f"r = {r}  (monthly factors),  5r = {5 * r}  (augmented state),  M = {M}")

    # ── build freq_list in dataset column order ───────────────────────────────
    import pandas as pd

    series_names = list(pd.read_csv(str(csv_path), index_col=0, nrows=0).columns)
    freq_list = [FREQ[name] for name in series_names]
    n_monthly = sum(f == "monthly" for f in freq_list)
    n_quarterly = sum(f == "quarterly" for f in freq_list)
    print(f"freq_list: {n_monthly} monthly + {n_quarterly} quarterly")
    quarterly_idx = [i for i, f in enumerate(freq_list) if f == "quarterly"]
    print(f"Quarterly series: {[series_names[i] for i in quarterly_idx]} "
          f"(row index {quarterly_idx})\n")

    tol = 1e-12

    # ══════════════════════════════════════════════════════════════════════════
    # 1. build_A_tilde
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 64)
    print("1. build_A_tilde")
    print("=" * 64)

    A_tilde = build_A_tilde(A)

    assert A_tilde.shape == (5 * r, 5 * r), f"A_tilde.shape = {A_tilde.shape}"

    # top-left block == A
    assert np.allclose(A_tilde[:r, :r], A, atol=tol), "top-left block != A"

    # identity shifts: block row k (k=1..4) has I_r at cols [(k-1)*r:k*r]
    for k in range(1, 5):
        block = A_tilde[k * r:(k + 1) * r, (k - 1) * r:k * r]
        assert np.allclose(block, np.eye(r), atol=tol), (
            f"identity shift missing/incorrect at block row {k}"
        )

    # everything else must be zero: reconstruct expected and compare
    A_expected = np.zeros((5 * r, 5 * r))
    A_expected[:r, :r] = A
    for k in range(1, 5):
        A_expected[k * r:(k + 1) * r, (k - 1) * r:k * r] = np.eye(r)
    assert np.allclose(A_tilde, A_expected, atol=tol), "unexpected non-zero entries"

    print(f"[OK] shape = {A_tilde.shape}")
    print(f"[OK] top-left (r x r) block == A   (max diff {np.abs(A_tilde[:r, :r] - A).max():.2e})")
    print(f"[OK] four identity-shift blocks I_{r} at correct positions")
    print(f"[OK] all other entries == 0")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. build_Q_tilde
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("2. build_Q_tilde")
    print("=" * 64)

    Q_tilde = build_Q_tilde(Q, w_u_t=1.0)

    assert Q_tilde.shape == (5 * r, 5 * r), f"Q_tilde.shape = {Q_tilde.shape}"

    # top-left block == Q (w_u_t = 1)
    assert np.allclose(Q_tilde[:r, :r], Q, atol=tol), "top-left block != Q"

    # rank == r (singular!)
    rank = np.linalg.matrix_rank(Q_tilde)
    assert rank == r, f"rank(Q_tilde) = {rank}, expected {r}"

    # everything except top-left block is zero
    Q_expected = np.zeros((5 * r, 5 * r))
    Q_expected[:r, :r] = Q
    assert np.allclose(Q_tilde, Q_expected, atol=tol), "unexpected non-zero entries"

    # w_u_t = 2.0 : top-left == Q / 2
    Q_tilde_2 = build_Q_tilde(Q, w_u_t=2.0)
    assert np.allclose(Q_tilde_2[:r, :r], Q / 2.0, atol=tol), "Q/w_u_t scaling failed"

    print(f"[OK] shape = {Q_tilde.shape}")
    print(f"[OK] top-left (r x r) block == Q   (max diff {np.abs(Q_tilde[:r, :r] - Q).max():.2e})")
    print(f"[OK] rank == {rank}  (SINGULAR: rank r, not 5r)")
    print(f"[OK] all other entries == 0")
    print(f"[OK] w_u_t=2.0 -> top-left == Q/2   (max diff {np.abs(Q_tilde_2[:r, :r] - Q / 2).max():.2e})")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. build_Lambda_tilde
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("3. build_Lambda_tilde")
    print("=" * 64)

    Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)

    assert Lambda_tilde.shape == (M, 5 * r), f"Lambda_tilde.shape = {Lambda_tilde.shape}"

    # monthly rows: first r columns == Lambda[i], remaining 4r columns == 0
    for i, f in enumerate(freq_list):
        if f == "monthly":
            assert np.allclose(Lambda_tilde[i, :r], Lambda[i, :], atol=tol), (
                f"monthly row {i}: first r columns != Lambda[i]"
            )
            assert np.allclose(Lambda_tilde[i, r:], 0.0, atol=tol), (
                f"monthly row {i}: trailing 4r columns not zero"
            )

    # quarterly row(s): MM pattern with weights [1/3, 2/3, 1, 2/3, 1/3]
    mm = MM_WEIGHTS_DEFAULT
    for i in quarterly_idx:
        for k, w in enumerate(mm):
            block = Lambda_tilde[i, k * r:(k + 1) * r]
            assert np.allclose(block, w * Lambda[i, :], atol=tol), (
                f"quarterly row {i}, lag-block {k}: expected {w}*Lambda[i]"
            )

    # explicit MM weight sum check: 1/3 + 2/3 + 1 + 2/3 + 1/3 == 3
    weight_sum = sum(mm)
    assert abs(weight_sum - 3.0) < tol, f"MM weights sum = {weight_sum}, expected 3"

    # block-diagonal preserved in the first r columns for monthly rows:
    # the contemporaneous block of Lambda_tilde must equal Lambda exactly.
    monthly_mask = np.array([f == "monthly" for f in freq_list])
    assert np.allclose(
        Lambda_tilde[monthly_mask, :r], Lambda[monthly_mask, :], atol=tol
    ), "block-diagonal structure not preserved in contemporaneous block"

    # show the GDPC1 quarterly row explicitly
    gdp_i = quarterly_idx[0]
    gdp_load = Lambda[gdp_i, :]
    nonzero_col = int(np.argmax(np.abs(gdp_load)))  # its block factor column
    print(f"[OK] shape = {Lambda_tilde.shape}")
    print(f"[OK] {n_monthly} monthly rows: first {r} cols == Lambda_i, trailing {4 * r} cols == 0")
    print(f"[OK] MM weights = {mm},  sum = {weight_sum:.4f} (== 3)")
    print(f"[OK] block-diagonal preserved in contemporaneous block (monthly rows)")
    print(f"\n     Quarterly series '{series_names[gdp_i]}' (row {gdp_i}):")
    print(f"       underlying loading Lambda[{gdp_i}] = {np.round(gdp_load, 4)}  "
          f"(loads on factor column {nonzero_col})")
    print(f"       MM-spread value on its block factor across the 5 lag-blocks:")
    spread = [Lambda_tilde[gdp_i, k * r + nonzero_col] for k in range(5)]
    print(f"         lags 0..4 : {np.round(spread, 4)}")
    print(f"         expected  : {np.round([w * gdp_load[nonzero_col] for w in mm], 4)}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. build_R_tilde
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("4. build_R_tilde")
    print("=" * 64)

    R_tilde = build_R_tilde(R, w_eps_t=1.0)

    assert R_tilde.shape == (M, M), f"R_tilde.shape = {R_tilde.shape}"

    # diagonal == R (w_eps_t = 1)
    assert np.allclose(np.diag(R_tilde), R, atol=tol), "diagonal != R"

    # off-diagonal all zero
    off_diag = R_tilde - np.diag(np.diag(R_tilde))
    assert np.allclose(off_diag, 0.0, atol=tol), "off-diagonal not zero"

    # w_eps_t = 2.0 : diagonal == R / 2
    R_tilde_2 = build_R_tilde(R, w_eps_t=2.0)
    assert np.allclose(np.diag(R_tilde_2), R / 2.0, atol=tol), "R/w_eps_t scaling failed"

    print(f"[OK] shape = {R_tilde.shape}")
    print(f"[OK] diagonal == R   (max diff {np.abs(np.diag(R_tilde) - R).max():.2e})")
    print(f"[OK] off-diagonal entries == 0   (max abs {np.abs(off_diag).max():.2e})")
    print(f"[OK] w_eps_t=2.0 -> diagonal == R/2   (max diff {np.abs(np.diag(R_tilde_2) - R / 2).max():.2e})")

    print("\n" + "=" * 64)
    print("All augmented-matrix constructors passed. Ready for Task 2 (filter).")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. build_selection_matrix  &  build_all_selection_matrices
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("5. build_selection_matrix / build_all_selection_matrices")
    print("=" * 64)

    # ── 5a. Known example (thesis line ~6602): M=4, one NaN at index 1 ───────
    print("\n--- 5a. Known example (M=4, y_t=[1.0, NaN, 3.0, 4.0]) ---")
    y_known = np.array([1.0, np.nan, 3.0, 4.0])
    W_known = build_selection_matrix(y_known)

    assert W_known.shape == (3, 4), f"W_known.shape = {W_known.shape}, expected (3, 4)"
    # Replace NaN with 0 before multiplying: W_t zeros out those columns anyway,
    # so the mathematical result is identical, but NumPy avoids 0*NaN = NaN propagation.
    y_known_nz = np.nan_to_num(y_known, nan=0.0)
    assert np.allclose(W_known @ y_known_nz, [1.0, 3.0, 4.0], atol=tol), (
        f"W_known @ y_known_nz = {W_known @ y_known_nz}, expected [1.0, 3.0, 4.0]"
    )
    W_expected = np.array([[1., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
    assert np.allclose(W_known, W_expected, atol=tol), (
        f"W_known != expected:\n{W_known}"
    )
    print(f"[OK] m_t = {W_known.shape[0]}  (expected 3)")
    print(f"[OK] W_t.shape = {W_known.shape}  (expected (3, 4))")
    print(f"[OK] W_t @ y_t (NaN->0) = {W_known @ y_known_nz}  (expected [1. 3. 4.])")
    print(f"[OK] W_t == [[1,0,0,0],[0,0,1,0],[0,0,0,1]]")

    # ── 5b. Edge case: all-NaN (m_t = 0) ─────────────────────────────────────
    print("\n--- 5b. All-NaN edge case: m_t = 0 ---")
    y_all_nan = np.array([np.nan, np.nan, np.nan, np.nan])
    W_zero = build_selection_matrix(y_all_nan)

    assert W_zero.shape == (0, 4), f"W_zero.shape = {W_zero.shape}, expected (0, 4)"
    prod_zero = W_zero @ y_all_nan
    assert prod_zero.shape == (0,), f"(W_zero @ y_all_nan).shape = {prod_zero.shape}, expected (0,)"
    print(f"[OK] W_t.shape = {W_zero.shape}  (expected (0, 4))")
    print(f"[OK] (W_t @ y_t).shape = {prod_zero.shape}  (expected (0,))")

    # ── 5c. Real data: build W_list for entire panel ──────────────────────────
    print("\n--- 5c. Real data: dataset_usa.csv (standardised) ---")
    # Load Y *standardised*, NaN preserved — same representation used by
    # E-step and M-step.  The selection matrices depend ONLY on the NaN
    # pattern (preserved by standardisation), so this section is functionally
    # equivalent to using the raw panel, just consistent with the rest.
    from em_initialization import load_standardized_data  # noqa: PLC0415
    _meta_path_sel = resolve_output_path("processed", "theta_initial_metadata.json", _cfg)
    Y_raw, _mean_raw, _std_raw, _series_names_sel = load_standardized_data(
        dataset_path=str(csv_path),
        metadata_path=str(_meta_path_sel),
    )
    _dates = pd.read_csv(str(csv_path), index_col=0, parse_dates=True).index
    T_raw, M_raw = Y_raw.shape
    print(f"Dataset shape: T={T_raw}, M={M_raw}")

    W_list = build_all_selection_matrices(Y_raw)

    assert len(W_list) == T_raw, f"len(W_list) = {len(W_list)}, expected {T_raw}"
    print(f"[OK] len(W_list) = {len(W_list)}  (T = {T_raw})")

    # distribution of m_t
    mt_array = np.array([W.shape[0] for W in W_list])
    unique_mt, counts_mt = np.unique(mt_array, return_counts=True)
    print("\nDistribution of m_t across T=497 time steps:")
    for mt_val, cnt in zip(unique_mt, counts_mt):
        print(f"  m_t = {mt_val:2d}  :  {cnt:4d} months")

    print("""
============================================================
INTERPRETATION OF m_t (number of observed series at time t)
============================================================

In our dataset (M = 20 series: 19 monthly + 1 quarterly GDP),
the number of observed series m_t at each month takes a small
set of characteristic values, each with a clear structural
meaning:

- m_t = 20: a quarter-end month (March, June, September,
  December) in the interior of the sample, where all 19 monthly
  series AND the quarterly GDP are observed. This is the richest
  information set. (~163 months)

- m_t = 19: a non-quarter-end month in the interior of the
  sample. All 19 monthly series are observed, but GDP is absent
  because it is only released at quarter-end. This is the
  "quarterly mask": GDP is structurally missing 2 out of every
  3 months by design, not by ragged edge. (~330 months)

- m_t = 18 (or other intermediate values near the sample end):
  ragged edge. One or two monthly series with longer publication
  lags (e.g. CMRMTSPLx, real manufacturing and trade sales) are
  not yet released for the most recent months, on top of the
  GDP quarterly mask.

- m_t = 1: the very end of the sample (April-May 2026). Only
  NFCI is observed, because NFCI is downloaded fresh from FRED
  (weekly, low publication lag) while all FRED-MD series stop at
  the vintage cutoff (March 2026) and GDP is absent. This is the
  extreme ragged-edge typical of real-time nowcasting: near the
  present, only the most timely, high-frequency indicators are
  available.

============================================================
WHY THIS MATTERS FOR THE KALMAN FILTER
============================================================

The selection matrix W_t handles all these cases uniformly
(Section 6 of the thesis): at each t, only the m_t observed
rows of the observation equation are used. Two consequences:

1. The quarterly mask (m_t = 19) and the ragged edge (m_t < 19)
   are treated by the SAME mechanism — there is no special-case
   code for quarterly vs. ragged missingness.

2. In information-poor months (small m_t, e.g. m_t = 1 at the
   sample end), the filter update relies on very few series.
   The factors of the unobserved blocks are then effectively
   EXTRAPOLATED via the state transition A_tilde, with only
   minimal correction from the few observed series. For example,
   in April-May 2026 only NFCI (a financial-block series) is
   observed, so the financial factor receives a small update
   while the real and other factors are propagated almost purely
   by the VAR dynamics. This is the expected behaviour of
   nowcasting at the edge of the sample, not a defect.
============================================================
""")

    # verify GDP (quarterly) logic: find GDPC1 column index
    gdp_col = _series_names_sel.index("GDPC1")
    is_qend = _dates.month.isin([3, 6, 9, 12])

    for t in range(T_raw):
        gdp_observed = not np.isnan(Y_raw[t, gdp_col])
        if _dates[t].month in [3, 6, 9, 12]:
            # GDP is observed at quarter-end only if not ragged-edge missing
            if gdp_observed:
                # make sure the selection matrix includes column gdp_col
                assert gdp_col in np.where(~np.isnan(Y_raw[t, :]))[0], (
                    f"t={t}: GDP observed but not in W_t"
                )
        else:
            # non-quarter-end month: GDP MUST be NaN
            assert np.isnan(Y_raw[t, gdp_col]), (
                f"t={t} ({_dates[t].strftime('%Y-%m')}): GDP should be NaN on non-quarter-end month"
            )
    print("[OK] GDP quarterly mask respected: GDP present in W_t iff quarter-end month")

    # quarter-end vs non-quarter-end comparison for interior months (no ragged edge)
    interior_qend = [t for t in range(T_raw) if is_qend[t] and mt_array[t] == M_raw]
    interior_non_qend = [t for t in range(T_raw) if not is_qend[t] and mt_array[t] == M_raw - 1]
    print(f"[OK] Fully-observed quarter-end months (m_t={M_raw}):     {len(interior_qend)}")
    print(f"[OK] Fully-observed non-quarter-end months (m_t={M_raw-1}): {len(interior_non_qend)}")

    # ragged edge: last 5 months
    print("\nRagged edge — m_t for the last 5 months:")
    for t in range(T_raw - 5, T_raw):
        dt_str = _dates[t].strftime("%Y-%m")
        qend_tag = "quarter-end" if _dates[t].month in [3, 6, 9, 12] else "non-qend"
        print(f"  t={t}  {dt_str}  ({qend_tag})  m_t = {mt_array[t]}")

    print("\n[OK] Selection matrix tests passed.")

    print("\n" + "=" * 64)
    print("All Task-1 + Task-2 tests passed.")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. kalman_predict  &  kalman_update   (Task 3)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("6. kalman_predict / kalman_update  (Task 3)")
    print("=" * 64)

    # ── 6a. Toy example: r=1, M=1 monthly series, fully observed ─────────────
    print("\n--- 6a. Toy example: r=1, M=1, fully observed ---")

    A_toy = np.array([[0.5]])
    A_tilde_toy = build_A_tilde(A_toy)            # (5, 5)
    Q_toy = np.array([[1.0]])
    Q_tilde_toy = build_Q_tilde(Q_toy)            # (5, 5)
    Lambda_toy = np.array([[2.0]])
    Lambda_tilde_toy = build_Lambda_tilde(Lambda_toy, ["monthly"])  # (1, 5)
    R_toy = np.array([0.5])
    R_tilde_toy = build_R_tilde(R_toy)            # (1, 1)

    f0_toy = np.zeros(5)
    P0_toy = np.eye(5)

    # --- predict step ---
    f_pred_toy, P_pred_toy = kalman_predict(f0_toy, P0_toy, A_tilde_toy, Q_tilde_toy)

    # manual check:
    #   f_pred = A_tilde @ 0 = 0
    #   P_pred[0,0] = A_tilde[0,:] @ A_tilde[0,:] + Q[0,0] = 0.5^2 + 1.0 = 1.25
    assert np.allclose(f_pred_toy, np.zeros(5), atol=tol), "f_pred != 0"
    assert abs(P_pred_toy[0, 0] - 1.25) < tol, \
        f"P_pred[0,0] = {P_pred_toy[0, 0]}, expected 1.25"

    print(f"  f_pred      = {f_pred_toy}")
    print(f"  P_pred[0,0] = {P_pred_toy[0, 0]:.6f}  (expected 1.25)")

    # --- update step ---
    y_t_toy = np.array([1.0])
    W_t_toy = np.array([[1.0]])    # (1, 1), fully observed
    res_toy = kalman_update(f_pred_toy, P_pred_toy, y_t_toy, W_t_toy,
                            Lambda_tilde_toy, R_tilde_toy)

    # manual derivation:
    #   WL = [[2,0,0,0,0]], WR = [[0.5]]
    #   eta = 1.0 - 2*0 = 1.0
    #   S   = 4 * P_pred[0,0] + 0.5 = 4*1.25 + 0.5 = 5.5
    #   K[0]= P_pred[0,0] * 2 / S = 2.5 / 5.5
    #   f_filt[0] = 0 + K[0]*1.0 = 2.5/5.5 ~ 0.454545
    expected_eta_toy = 1.0
    expected_S_toy   = 4.0 * P_pred_toy[0, 0] + 0.5          # = 5.5
    expected_f0_toy  = P_pred_toy[0, 0] * 2.0 / expected_S_toy  # = 2.5/5.5

    assert abs(res_toy["eta"][0] - expected_eta_toy) < tol, \
        f"eta = {res_toy['eta']}, expected {expected_eta_toy}"
    assert abs(res_toy["S"][0, 0] - expected_S_toy) < tol, \
        f"S = {res_toy['S'][0, 0]:.6f}, expected {expected_S_toy}"
    assert abs(res_toy["f_filt"][0] - expected_f0_toy) < tol, \
        f"f_filt[0] = {res_toy['f_filt'][0]:.6f}, expected {expected_f0_toy:.6f}"
    assert res_toy["f_filt"][0] > 0, "filter did not move toward the observation"
    assert res_toy["P_filt"][0, 0] < P_pred_toy[0, 0], \
        "posterior variance must decrease after observing data"
    assert np.isfinite(res_toy["loglik_t"]), "loglik_t is not finite"

    print(f"  eta         = {res_toy['eta'][0]:.6f}  (expected {expected_eta_toy})")
    print(f"  S           = {res_toy['S'][0, 0]:.6f}  (expected {expected_S_toy:.4f})")
    print(f"  f_filt      = {np.round(res_toy['f_filt'], 6)}")
    print(f"  f_filt[0]   = {res_toy['f_filt'][0]:.6f}  "
          f"(expected {expected_f0_toy:.6f}  = 2.5/5.5)")
    print(f"  P_filt[0,0] = {res_toy['P_filt'][0, 0]:.6f}  "
          f"(< P_pred[0,0] = {P_pred_toy[0, 0]:.4f})")
    print(f"  loglik_t    = {res_toy['loglik_t']:.6f}")
    print("[OK] Toy example: predict + update match manual calculation")

    # ── 6b. Missing data (m_t = 0, all NaN) ──────────────────────────────────
    print("\n--- 6b. Missing data: m_t = 0 (all NaN, update skipped) ---")

    y_t_nan_toy = np.array([np.nan])
    W_t_empty_toy = build_selection_matrix(y_t_nan_toy)    # shape (0, 1)
    assert W_t_empty_toy.shape == (0, 1)

    res_miss = kalman_update(f_pred_toy, P_pred_toy, y_t_nan_toy, W_t_empty_toy,
                             Lambda_tilde_toy, R_tilde_toy)

    assert np.allclose(res_miss["f_filt"], f_pred_toy, atol=tol), \
        "f_filt should equal f_pred when m_t = 0"
    assert np.allclose(res_miss["P_filt"], P_pred_toy, atol=tol), \
        "P_filt should equal P_pred when m_t = 0"
    assert res_miss["eta"].shape == (0,)
    assert res_miss["S"].shape == (0, 0)
    assert res_miss["loglik_t"] == 0.0

    print(f"  W_t.shape   = {W_t_empty_toy.shape}  (m_t = 0)")
    print(f"  f_filt == f_pred: {np.allclose(res_miss['f_filt'], f_pred_toy)}")
    print(f"  P_filt == P_pred: {np.allclose(res_miss['P_filt'], P_pred_toy)}")
    print(f"  eta.shape   = {res_miss['eta'].shape},  S.shape = {res_miss['S'].shape}")
    print(f"  loglik_t    = {res_miss['loglik_t']:.1f}  (no observation, no contribution)")
    print("[OK] m_t = 0: update skipped, state and covariance unchanged")

    # ── 6c. Real data sanity check (first time step, t=0) ────────────────────
    print("\n--- 6c. Real data sanity check (t=0) ---")

    A_tilde_real = build_A_tilde(A)
    Q_tilde_real = build_Q_tilde(Q)
    Lambda_tilde_real = build_Lambda_tilde(Lambda, freq_list)
    R_tilde_real = build_R_tilde(R)

    f0_real = np.zeros(5 * r)
    if "Sigma_0" in theta.files and theta["Sigma_0"].shape == (5 * r, 5 * r):
        P0_real = theta["Sigma_0"]
        p0_label = "theta['Sigma_0']"
    else:
        P0_real = 10.0 * np.eye(5 * r)
        p0_label = "10 * I_{5r}  (Sigma_0 absent or wrong shape)"
    print(f"  P_0 source  : {p0_label}")

    y_t0 = Y_raw[0, :]
    W_t0 = W_list[0]
    print(f"  t=0  ({_dates[0].strftime('%Y-%m')})  m_t = {W_t0.shape[0]}")

    f_pred_real, P_pred_real = kalman_predict(f0_real, P0_real, A_tilde_real, Q_tilde_real)
    res_real = kalman_update(f_pred_real, P_pred_real, y_t0, W_t0,
                             Lambda_tilde_real, R_tilde_real)

    f_filt_real = res_real["f_filt"]
    P_filt_real = res_real["P_filt"]

    assert f_filt_real.shape == (5 * r,)
    assert P_filt_real.shape == (5 * r, 5 * r)
    assert np.allclose(P_filt_real, P_filt_real.T, atol=1e-10), "P_filt not symmetric"
    _eigs = np.linalg.eigvalsh(P_filt_real)
    assert _eigs.min() >= -1e-8, f"P_filt not PSD: min eigenvalue = {_eigs.min():.2e}"
    assert np.isfinite(res_real["loglik_t"]), f"loglik_t not finite: {res_real['loglik_t']}"

    print(f"  f_filt  shape = {f_filt_real.shape}")
    print(f"  P_filt  shape = {P_filt_real.shape}")
    print(f"  P_filt symmetric: max|P - P.T| = "
          f"{np.abs(P_filt_real - P_filt_real.T).max():.2e}")
    print(f"  P_filt PSD:       min eigenvalue = {_eigs.min():.4e}")
    print(f"  loglik_t        = {res_real['loglik_t']:.6f}")
    print("[OK] Real-data t=0: shapes, symmetry, PSD, finite loglik all pass")

    print("\n" + "=" * 64)
    print("All Task-1 + Task-2 + Task-3 tests passed.")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. kalman_filter   (Task 4 — full forward filter)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("7. kalman_filter  (Task 4 — full forward filter)")
    print("=" * 64)

    # ── Load full panel (standardised, NaN preserved) ─────────────────────────
    # The Kalman filter/smoother operate on the *standardised* data — the same
    # representation against which theta^(0) was calibrated.  See
    # ``em_initialization.load_standardized_data`` for the rationale.
    _meta_path = resolve_output_path("processed", "theta_initial_metadata.json", _cfg)
    Y_full, _, _, _ = load_standardized_data(
        dataset_path=str(csv_path),
        metadata_path=str(_meta_path),
    )
    dates_idx = pd.read_csv(str(csv_path), index_col=0, parse_dates=True).index

    # ── Run filter (Gaussian: w_u=1, w_eps=1 via None defaults) ──────────────
    print("\nRunning kalman_filter (Gaussian baseline, w_u=1, w_eps=1)...")
    import time as _time
    _t0 = _time.perf_counter()
    kf = kalman_filter(Y_full, theta, w_u=None, w_eps=None, freq_list=freq_list)
    _elapsed = _time.perf_counter() - _t0
    print(f"  Elapsed: {_elapsed:.3f} s")

    f_filt_kf = kf["f_filt"]   # (T, 15)
    P_filt_kf = kf["P_filt"]   # (T, 15, 15)
    loglik_kf = kf["loglik"]
    T_kf      = Y_full.shape[0]

    # ── Shape checks ─────────────────────────────────────────────────────────
    assert f_filt_kf.shape == (T_kf, 5 * r), \
        f"f_filt.shape = {f_filt_kf.shape}, expected ({T_kf}, {5*r})"
    assert P_filt_kf.shape == (T_kf, 5 * r, 5 * r), \
        f"P_filt.shape = {P_filt_kf.shape}, expected ({T_kf}, {5*r}, {5*r})"
    print(f"[OK] f_filt.shape = {f_filt_kf.shape}  (expected ({T_kf}, {5*r}))")
    print(f"[OK] P_filt.shape = {P_filt_kf.shape}  (expected ({T_kf}, {5*r}, {5*r}))")

    # ── No NaN / inf in f_filt ────────────────────────────────────────────────
    n_bad = int(np.sum(~np.isfinite(f_filt_kf)))
    assert n_bad == 0, f"f_filt has {n_bad} non-finite entries"
    print(f"[OK] f_filt: all {f_filt_kf.size} entries finite  (no NaN, no inf)")

    # ── Symmetry of P_filt (all T steps) ─────────────────────────────────────
    asym_max = float(np.max([
        np.abs(P_filt_kf[t] - P_filt_kf[t].T).max() for t in range(T_kf)
    ]))
    assert asym_max < 1e-8, f"max asymmetry of P_filt = {asym_max:.2e}"
    print(f"[OK] P_filt symmetric at all t:  max|P - P.T| = {asym_max:.2e}")

    # ── PSD of P_filt (min eigenvalue >= -1e-8) ───────────────────────────────
    # eigvalsh is cheaper than eig and correct for symmetric matrices
    min_eig = float(min(np.linalg.eigvalsh(P_filt_kf[t]).min() for t in range(T_kf)))
    assert min_eig >= -1e-8, f"P_filt not PSD: min eigenvalue across all t = {min_eig:.2e}"
    print(f"[OK] P_filt PSD at all t:        min eigenvalue = {min_eig:.4e}")

    # ── Log-likelihood ────────────────────────────────────────────────────────
    assert np.isfinite(loglik_kf), f"loglik not finite: {loglik_kf}"
    print(f"[OK] Total log-likelihood = {loglik_kf:.4f}")

    # ── First 3 monthly filtered factors at the ragged edge (last 5 months) ──
    print("\nFirst 3 monthly filtered factors — last 5 months (ragged edge):")
    print(f"  {'Date':<12}  {'f[0]':>12}  {'f[1]':>12}  {'f[2]':>12}")
    for t in range(T_kf - 5, T_kf):
        dt_str = dates_idx[t].strftime("%Y-%m")
        print(
            f"  {dt_str:<12}  {f_filt_kf[t, 0]:>+12.6f}"
            f"  {f_filt_kf[t, 1]:>+12.6f}  {f_filt_kf[t, 2]:>+12.6f}"
        )

    # ── Recession sanity check ────────────────────────────────────────────────
    # f_filt[:, 0] is the first monthly factor (loads on real-activity series).
    # It should show large negative values during the GFC (2008-2009) and
    # the COVID crash (2020-04).
    recession_checks = [
        ("2008-12", "GFC trough"),
        ("2009-06", "GFC recovery"),
        ("2020-04", "COVID crash"),
    ]
    print("\nRecession sanity check — first factor f_filt[:, 0]")
    print("(expected: strongly negative at GFC trough and COVID crash)")
    date_strs = [d.strftime("%Y-%m") for d in dates_idx]
    for ym, label in recession_checks:
        if ym in date_strs:
            idx = date_strs.index(ym)
            val = f_filt_kf[idx, 0]
            print(f"  t={idx:3d}  {ym}  ({label:<18})  f[0] = {val:+.6f}")
        else:
            print(f"  {ym}  ({label:<18})  date not found in index")

    # ── Interpretation of the filtered factors ────────────────────────────────
    print("""
============================================================
INTERPRETATION OF THE FILTERED FACTORS (Gaussian baseline)
============================================================

The three filtered monthly factors display distinct dynamic
characteristics that reflect both their economic content and
the initial parameters theta^(0):

REAL FACTOR (f[0]):
- Stays close to zero in normal times (std ~ 2.3), with small
  fluctuations.
- Shows a milder dip during the 2008-2009 Great Recession
  (f[0] ~ -5.5 at the trough).
- Shows a MASSIVE spike during the 2020 COVID crash
  (f[0] ~ -38 in April 2020), the largest movement in the
  sample by far, followed by a sharp positive rebound.
- A value of ~ -38 on a series with std ~ 2.3 is roughly a
  16-sigma event — a textbook outlier, and exactly why the
  Student-t down-weighting matters: a Gaussian likelihood
  treats this point as a literal 16-sigma realisation and lets
  it dominate parameter estimates.
- This "near-zero with rare large spikes" behaviour reflects
  the high innovation variance (Q[0,0] ~ 5.2) combined with
  low persistence (the real-block companion eigenvalue is
  small): shocks are large but quickly die out, so the factor
  reverts rapidly to zero.

FINANCIAL FACTOR (f[1]):
- Highly persistent and cyclical, ranging roughly between -4
  and +3 over the sample.
- Rises during periods of financial stress / tight conditions
  and falls in calm periods.
- This smooth, trending behaviour reflects low innovation
  variance (Q[1,1] ~ 0.26) combined with high persistence
  (the dominant VAR eigenvalue ~0.95 loads on this factor):
  small shocks accumulate into long swings.
- Recall that f[1] is driven mainly by credit and term spreads
  (T10YFFM, BAAFFM, etc.), so it should be read as a
  "spread / financial-stress" factor rather than a stock-market
  level.

OTHER FACTOR (f[2]):
- Noisy and mean-reverting around zero (range ~ ±4), with no
  strong trend.
- This reflects its content (volatile monthly inflation
  differences, CPI/PCE), which are intrinsically noisy at the
  monthly frequency (Q[2,2] ~ 1.86, low persistence).

RAGGED EDGE (last months, April-May 2026):
- Only NFCI (a financial-block series) is observed at the very
  end (m_t = 1). Consequently:
  * f[1] (financial) keeps updating, anchored by NFCI;
  * f[0] (real) and f[2] (other) are essentially extrapolated
    via the state transition A_tilde, since no series from
    their blocks is observed.
- This is the expected nowcasting behaviour at the sample edge,
  not a numerical artefact.

NOTE: these are the FILTERED factors under the Gaussian baseline
(all weights = 1). The Student-t weights (estimated in the E-step)
will down-weight outlier periods such as COVID, which is expected
to make the factor paths more robust to extreme observations.

============================================================
""")

    # ── Optional plot: 3 monthly factors with recession shading ──────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig_path = resolve_output_path("figures", "kalman_filtered_factors.png", _cfg)

        dates_ts = [pd.Timestamp(d) for d in dates_idx]
        factor_labels = ["Factor 0 (real)", "Factor 1 (financial)", "Factor 2 (other)"]
        recessions = [
            (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
            (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(dates_ts, f_filt_kf[:, k], lw=0.8, color="steelblue")
            for rs, re in recessions:
                ax.axvspan(rs, re, alpha=0.20, color="gray",
                           label=("NBER recession" if k == 0 else None))
            ax.axhline(0, color="black", lw=0.5, ls="--")
            ax.set_ylabel(factor_labels[k], fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(
            "Kalman filtered monthly factors — Gaussian baseline", fontsize=11
        )
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()

        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] Plot saved to {fig_path}")
    except ImportError:
        print("\n[SKIP] matplotlib not available; plot skipped")
    except Exception as _plot_err:
        print(f"\n[SKIP] Plot failed: {_plot_err}")

    print("\n" + "=" * 64)
    print("All Task-1 + Task-2 + Task-3 + Task-4 tests passed.")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. kalman_smoother  (Task 5 — RTS backward + lag-one covariance)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("8. kalman_smoother  (Task 5 — RTS backward + lag-one covariance)")
    print("=" * 64)

    # Reuse A_tilde from the full-data filter (theta['A'] -> companion form).
    A_tilde_full = build_A_tilde(theta["A"])

    print("\nRunning kalman_smoother (jitter = 0, uses np.linalg.pinv)...")
    _t0 = _time.perf_counter()
    ks = kalman_smoother(kf, A_tilde_full, jitter=0.0)
    _elapsed = _time.perf_counter() - _t0
    print(f"  Elapsed: {_elapsed:.3f} s")

    f_smooth = ks["f_smooth"]   # (T, 15)
    P_smooth = ks["P_smooth"]   # (T, 15, 15)
    P_lag    = ks["P_lag"]      # (T, 15, 15); P_lag[0] is zeros (undefined)
    J_arr    = ks["J"]          # (T, 15, 15); J[T-1] is zeros (unused)

    # ── Shape checks ─────────────────────────────────────────────────────────
    assert f_smooth.shape == (T_kf, 5 * r), \
        f"f_smooth.shape = {f_smooth.shape}, expected ({T_kf}, {5*r})"
    assert P_smooth.shape == (T_kf, 5 * r, 5 * r), \
        f"P_smooth.shape = {P_smooth.shape}"
    assert P_lag.shape == (T_kf, 5 * r, 5 * r), \
        f"P_lag.shape = {P_lag.shape}"
    assert J_arr.shape == (T_kf, 5 * r, 5 * r), \
        f"J.shape = {J_arr.shape}"
    print(f"[OK] f_smooth.shape = {f_smooth.shape}")
    print(f"[OK] P_smooth.shape = {P_smooth.shape}")
    print(f"[OK] P_lag.shape    = {P_lag.shape}")
    print(f"[OK] J.shape        = {J_arr.shape}")

    # ── Finiteness ────────────────────────────────────────────────────────────
    n_bad_f = int(np.sum(~np.isfinite(f_smooth)))
    n_bad_P = int(np.sum(~np.isfinite(P_smooth)))
    n_bad_L = int(np.sum(~np.isfinite(P_lag)))
    assert n_bad_f == 0, f"f_smooth has {n_bad_f} non-finite entries"
    assert n_bad_P == 0, f"P_smooth has {n_bad_P} non-finite entries"
    assert n_bad_L == 0, f"P_lag has {n_bad_L} non-finite entries"
    print(f"[OK] All entries finite (no NaN/inf) in f_smooth, P_smooth, P_lag")

    # ── Boundary: f_smooth[T-1] == f_filt[T-1] ───────────────────────────────
    diff_mean = float(np.abs(f_smooth[-1] - f_filt_kf[-1]).max())
    diff_cov  = float(np.abs(P_smooth[-1] - P_filt_kf[-1]).max())
    assert diff_mean < 1e-10, f"f_smooth[T-1] != f_filt[T-1]: max diff {diff_mean}"
    assert diff_cov  < 1e-10, f"P_smooth[T-1] != P_filt[T-1]: max diff {diff_cov}"
    print(f"[OK] Boundary: f_smooth[T-1] == f_filt[T-1]  (max diff {diff_mean:.2e})")
    print(f"[OK] Boundary: P_smooth[T-1] == P_filt[T-1]  (max diff {diff_cov:.2e})")

    # ── Symmetry of P_smooth ─────────────────────────────────────────────────
    asym_max = float(np.max([
        np.abs(P_smooth[t] - P_smooth[t].T).max() for t in range(T_kf)
    ]))
    assert asym_max < 1e-8, f"max asymmetry of P_smooth = {asym_max:.2e}"
    print(f"[OK] P_smooth symmetric at all t:  max|P - P.T| = {asym_max:.2e}")

    # ── PSD of P_smooth ──────────────────────────────────────────────────────
    min_eig = float(min(np.linalg.eigvalsh(P_smooth[t]).min() for t in range(T_kf)))
    # Slightly looser tolerance than P_filt (pinv-based recursion can leak ~1e-7).
    assert min_eig >= -1e-6, \
        f"P_smooth not PSD: min eigenvalue across all t = {min_eig:.2e}"
    print(f"[OK] P_smooth PSD at all t:        min eigenvalue = {min_eig:.4e}")

    # ── Smoother reduces uncertainty: trace(P_smooth) <= trace(P_filt) ──────
    trace_filt   = np.array([np.trace(P_filt_kf[t]) for t in range(T_kf)])
    trace_smooth = np.array([np.trace(P_smooth[t]) for t in range(T_kf)])
    tol_trace = 1e-6
    n_violations = int(np.sum(trace_smooth > trace_filt + tol_trace))
    mean_reduction = float(np.mean(trace_filt - trace_smooth))
    max_reduction  = float(np.max(trace_filt - trace_smooth))
    print(f"[OK] trace(P_smooth[t]) <= trace(P_filt[t]) at all t  "
          f"(tol = {tol_trace:g}): {n_violations} violations / {T_kf}")
    print(f"     Mean trace reduction across t : {mean_reduction:+.4f}")
    print(f"     Max  trace reduction across t : {max_reduction:+.4f}")
    assert n_violations == 0, \
        f"{n_violations} time steps violate trace(P_smooth) <= trace(P_filt)"

    # ── Filtered vs smoothed: last 5 months (ragged edge) ────────────────────
    print("\nFiltered vs smoothed factors — last 5 months (ragged edge):")
    print(f"  {'Date':<10}"
          f"  {'filt[0]':>10} {'smth[0]':>10}"
          f"  {'filt[1]':>10} {'smth[1]':>10}"
          f"  {'filt[2]':>10} {'smth[2]':>10}")
    for t in range(T_kf - 5, T_kf):
        dt_str = dates_idx[t].strftime("%Y-%m")
        print(
            f"  {dt_str:<10}"
            f"  {f_filt_kf[t, 0]:>+10.4f} {f_smooth[t, 0]:>+10.4f}"
            f"  {f_filt_kf[t, 1]:>+10.4f} {f_smooth[t, 1]:>+10.4f}"
            f"  {f_filt_kf[t, 2]:>+10.4f} {f_smooth[t, 2]:>+10.4f}"
        )

    # ── Recession sanity check on smoothed real factor ───────────────────────
    print("\nRecession check — smoothed real factor f_smooth[:, 0] "
          "(vs filtered f_filt[:, 0])")
    for ym, label in recession_checks:
        if ym in date_strs:
            idx = date_strs.index(ym)
            print(
                f"  t={idx:3d}  {ym}  ({label:<18})  "
                f"f_filt = {f_filt_kf[idx, 0]:+.6f}   "
                f"f_smooth = {f_smooth[idx, 0]:+.6f}"
            )
        else:
            print(f"  {ym}  ({label:<18})  date not found in index")

    print("""
============================================================
INTERPRETATION: FILTERED vs SMOOTHED FACTORS
============================================================

The RTS smoother revises the filtered factor estimates using
the ENTIRE sample, including observations after time t. Four
properties are visible in the results:

1. VARIANCE REDUCTION. By construction, the smoothed covariance
   satisfies trace(P_smooth[t]) <= trace(P_filt[t]) at every t
   (verified above: 0 violations). The smoother is strictly more
   informed than the filter, because it conditions on future as
   well as past observations.

2. SHARPER RECESSION TROUGHS. The smoothed real factor reaches
   slightly more pronounced extrema than the filtered one during
   downturns:
   - GFC trough (2008-12): filtered ~ -5.46, smoothed ~ -5.52
   - COVID crash (2020-04): filtered ~ -37.9, smoothed ~ -38.0
   The direction is the expected one — when the filter processes
   April 2020 it has not yet "seen" the rebound, while the
   smoother retroactively recognises how deep the contraction
   was — but the magnitude of the correction is small (see
   point 4 below).

3. CONVERGENCE AT THE SAMPLE END. At t = T-1 the smoothed and
   filtered estimates coincide exactly (the smoother is
   initialised at the filter's final output). Moving backward
   from T, the smoothing correction grows. This is why the last
   month (May 2026) shows filt == smooth, while earlier months
   differ.

4. FILTER AND SMOOTHER NEARLY COINCIDE. With correctly
   standardised data, the posterior factor uncertainty is small
   relative to the signal (V_f / E_f^2 ~ 0.03), so the filtered
   and smoothed paths are nearly identical (mean trace reduction
   ~ 0.05). This is a sign of precise factor estimation: the
   filter is already confident using past data alone, and future
   information adds little. (Under the earlier mis-scaled data
   the two paths diverged strongly — that divergence was an
   artefact of the scale mismatch, now resolved.)

IMPORTANT — link to the Student-t extension:
Under this Gaussian baseline (all weights = 1), the smoother
still places the COVID trough at ~ -38 — a 16-sigma realisation
that a Gaussian likelihood treats literally. This is precisely
the behaviour the Student-t model is designed to temper: in the
full model, the latent weight w_t for the April 2020 observation
will be small (it is a heavy-tailed outlier), down-weighting its
influence on the factor path and on the parameter updates.
Comparing the Gaussian and Student-t smoothed factors around
COVID will be one of the key diagnostics of the robustness gain
(Section 1 of the thesis).
============================================================
""")

    # ── Optional plot: filtered vs smoothed real factor in two crisis windows─
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates_pd = pd.DatetimeIndex(dates_idx)
        windows = [
            (pd.Timestamp("2007-01-01"), pd.Timestamp("2010-12-31"),
             "Great Recession (2007–2010)"),
            (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-06-30"),
             "COVID-19 shock (2019–2021)"),
        ]
        recessions = [
            (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
            (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
        ]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        for ax, (start, end, title) in zip(axes, windows):
            mask = (dates_pd >= start) & (dates_pd <= end)
            d_win = dates_pd[mask]
            ax.plot(d_win, f_filt_kf[mask, 0],
                    label="filtered f[0]", lw=1.1, color="tab:blue", ls="--")
            ax.plot(d_win, f_smooth[mask, 0],
                    label="smoothed f[0]", lw=1.4, color="tab:red")
            for rs, re_ in recessions:
                if (rs <= end) and (re_ >= start):
                    ax.axvspan(max(rs, start), min(re_, end),
                               alpha=0.20, color="gray")
            ax.axhline(0, color="black", lw=0.5, ls="--")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower left")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle("Real factor f[0]: filtered vs RTS-smoothed",
                     fontsize=11)
        fig.tight_layout()
        fig_path = resolve_output_path("figures", "kalman_filtered_vs_smoothed_real.png", _cfg)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] Plot saved to {fig_path}")
    except ImportError:
        print("\n[SKIP] matplotlib not available; plot skipped")
    except Exception as _plot_err:
        print(f"\n[SKIP] Plot failed: {_plot_err}")

    # ══════════════════════════════════════════════════════════════════════════
    # 9. run_kalman  (Task 6 — high-level wrapper: filter + smoother)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("9. run_kalman  (Task 6 — high-level wrapper)")
    print("=" * 64)

    # ── 9a. Single call: filter + smoother, weights default to 1 ─────────────
    save_path_test = resolve_output_path("processed", "_kalman_test_output.npz", _cfg)
    print("\nRunning run_kalman (Gaussian baseline, save_path provided)...")
    _t0 = _time.perf_counter()
    rk = run_kalman(Y_full, theta, w_u=None, w_eps=None,
                    freq_list=freq_list, save_path=save_path_test)
    _elapsed = _time.perf_counter() - _t0
    print(f"  Elapsed: {_elapsed:.3f} s")

    # ── 9b. Verify dict keys ─────────────────────────────────────────────────
    expected_keys = {
        "f_smooth", "P_smooth", "P_lag",
        "f_filt", "P_filt",
        "loglik",
        "A_tilde", "Lambda_tilde",
        "T", "M", "r",
    }
    actual_keys = set(rk.keys())
    missing = expected_keys - actual_keys
    extra   = actual_keys - expected_keys
    assert not missing, f"run_kalman missing keys: {missing}"
    print(f"[OK] All expected keys present ({len(expected_keys)} keys)")
    if extra:
        print(f"     (also exposes extra keys: {sorted(extra)})")

    # ── 9c. Shape & metadata coherence ───────────────────────────────────────
    assert rk["T"] == T_kf, f"T mismatch: {rk['T']} vs {T_kf}"
    assert rk["M"] == M,    f"M mismatch: {rk['M']} vs {M}"
    assert rk["r"] == r,    f"r mismatch: {rk['r']} vs {r}"
    assert rk["A_tilde"].shape      == (5 * r, 5 * r)
    assert rk["Lambda_tilde"].shape == (M, 5 * r)
    assert rk["f_smooth"].shape     == (T_kf, 5 * r)
    assert rk["P_smooth"].shape     == (T_kf, 5 * r, 5 * r)
    assert rk["P_lag"].shape        == (T_kf, 5 * r, 5 * r)
    print(f"[OK] Metadata: T={rk['T']}, M={rk['M']}, r={rk['r']}")
    print(f"[OK] Shapes:   A_tilde {rk['A_tilde'].shape}, "
          f"Lambda_tilde {rk['Lambda_tilde'].shape}")

    # ── 9d. Round-trip consistency vs separate calls ─────────────────────────
    # run_kalman must produce IDENTICAL output to kalman_filter + kalman_smoother
    # called separately (it merely orchestrates them; any mismatch would indicate
    # accidental code drift in the wrapper).
    diff_loglik  = abs(rk["loglik"] - loglik_kf)
    diff_ffilt   = float(np.abs(rk["f_filt"]   - f_filt_kf).max())
    diff_fsmooth = float(np.abs(rk["f_smooth"] - f_smooth ).max())
    diff_Psmooth = float(np.abs(rk["P_smooth"] - P_smooth ).max())
    diff_Plag    = float(np.abs(rk["P_lag"]    - P_lag    ).max())
    assert diff_loglik  < 1e-10, f"loglik drift: {diff_loglik:.2e}"
    assert diff_ffilt   < 1e-12, f"f_filt drift: {diff_ffilt:.2e}"
    assert diff_fsmooth < 1e-12, f"f_smooth drift: {diff_fsmooth:.2e}"
    assert diff_Psmooth < 1e-12, f"P_smooth drift: {diff_Psmooth:.2e}"
    assert diff_Plag    < 1e-12, f"P_lag drift: {diff_Plag:.2e}"
    print("[OK] Round-trip vs separate calls (kalman_filter + kalman_smoother):")
    print(f"     max|d loglik|   = {diff_loglik:.2e}")
    print(f"     max|d f_filt|   = {diff_ffilt:.2e}")
    print(f"     max|d f_smooth| = {diff_fsmooth:.2e}")
    print(f"     max|d P_smooth| = {diff_Psmooth:.2e}")
    print(f"     max|d P_lag|    = {diff_Plag:.2e}")

    # ── 9e. Second moments at t = 1 (the E-step identities) ──────────────────
    # E[f_t f_t.T   | Y] = P_smooth[t] + outer(f_smooth[t], f_smooth[t])
    # E[f_t f_{t-1}.T | Y] = P_lag[t]  + outer(f_smooth[t], f_smooth[t-1])
    f1 = rk["f_smooth"][1]
    f0 = rk["f_smooth"][0]
    E_f1_f1 = rk["P_smooth"][1] + np.outer(f1, f1)
    E_f1_f0 = rk["P_lag"][1]    + np.outer(f1, f0)

    assert E_f1_f1.shape == (5 * r, 5 * r), f"E[f1 f1'] shape: {E_f1_f1.shape}"
    assert E_f1_f0.shape == (5 * r, 5 * r), f"E[f1 f0'] shape: {E_f1_f0.shape}"
    assert np.all(np.isfinite(E_f1_f1)),     "E[f1 f1'] has non-finite entries"
    assert np.all(np.isfinite(E_f1_f0)),     "E[f1 f0'] has non-finite entries"
    # Symmetry of the contemporaneous second moment.
    asym = float(np.abs(E_f1_f1 - E_f1_f1.T).max())
    assert asym < 1e-8, f"E[f1 f1'] not symmetric: max|M - M.T| = {asym:.2e}"
    print(f"\n[OK] Second moments at t=1 (E-step identities):")
    print(f"     E[f_1 f_1.T]   shape = {E_f1_f1.shape}, "
          f"trace = {np.trace(E_f1_f1):.4f}, asym = {asym:.2e}")
    print(f"     E[f_1 f_0.T]   shape = {E_f1_f0.shape}, "
          f"max|.| = {np.abs(E_f1_f0).max():.4f}")

    # ── 9f. Check the .npz save produced a readable file ─────────────────────
    assert save_path_test.exists(), f"save_path .npz not written: {save_path_test}"
    _saved = np.load(save_path_test)
    saved_keys = set(_saved.files)
    expected_saved = {"f_smooth", "P_smooth", "P_lag", "loglik"}
    assert expected_saved <= saved_keys, (
        f".npz missing keys: {expected_saved - saved_keys}"
    )
    assert np.allclose(_saved["f_smooth"], rk["f_smooth"]), \
        ".npz f_smooth differs from in-memory copy"
    _saved.close()
    save_path_test.unlink()  # cleanup
    print(f"[OK] save_path .npz round-trip: keys {sorted(expected_saved)} "
          f"written and reloaded; file removed")

    # ── 9g. Final summary ─────────────────────────────────────────────────────
    print(f"""
============================================================
SUMMARY — Kalman E-step (factor side) complete
============================================================
  total log-likelihood : {rk['loglik']:.4f}
  smoothed mean shape  : {rk['f_smooth'].shape}
  smoothed cov shape   : {rk['P_smooth'].shape}
  lag-one cov shape    : {rk['P_lag'].shape}
  augmented state dim  : 5r = {5 * rk['r']}
  observation dim      : M  = {rk['M']}
  sample length        : T  = {rk['T']}
============================================================
Kalman E-step (factor side) completo — pronto per il prossimo
modulo (em_e_step: aggiornamento dei pesi Student-t w_u, w_eps).
============================================================
""")

    print("\n" + "=" * 64)
    print("All Task-1 + Task-2 + Task-3 + Task-4 + Task-5 + Task-6 tests passed.")
    print("=" * 64)
