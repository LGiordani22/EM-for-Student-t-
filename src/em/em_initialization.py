"""
src/em/em_initialization.py

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

import numpy as np
import pandas as pd

try:  # package (from em.em_initialization) o script (python src/em/em_initialization.py)
    from em.factor_structure import FactorStructure, as_structure
except ModuleNotFoundError:  # pragma: no cover
    from factor_structure import FactorStructure, as_structure


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
    The series operate on very different scales — e.g. INDPRO
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
    particular, GDPC1 statistics are computed from the observed
    quarterly values only, not from the more numerous monthly slots.

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
        # This is expected for late-start series (e.g. PPIFIS, which begins in
        # 2009-12) in early real-time vintages.  Treat them as permanently
        # missing: mean=0, std=1 so the column stays all-NaN after
        # standardisation; gaussian_fill_ragged will fill it with N(0,1) noise
        # and the EM/Kalman treats it as uninformative.
        import warnings
        warnings.warn(
            f"standardize: {too_few.index.tolist()} have <2 non-NaN obs — "
            f"treating as missing (mean=0, std=1). Expected for series that "
            f"start late, in vintages earlier than their first observation.",
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
    structure: "FactorStructure | dict[str, str]",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Estrai i fattori iniziali F^(0) via PCA sequenziale, guidata dalla loading
    mask.

    Thesis reference: EM_for_student_t.tex, Section 4.2
    (Initialising the Factors by PCA), eq. (4.1).

    Algoritmo unico per TUTTE le strutture (diagonali e non-diagonali con
    globale). I fattori si estraggono dal piu' LARGO al piu' STRETTO
    (``structure.extraction_order()``), residualizzando i membri dopo ogni
    estrazione, **rispettando la mask** (si toglie il fattore j dalla serie i
    solo se ``mask[i, j] == 1``):

      1. fattore j: PCA sulla sotto-matrice dei RESIDUI correnti dei suoi membri
         -> primo autovettore v_j (unit norm), sign-normalizzato;
      2. f_j = Y_membri @ v_j;
      3. deflazione: per ogni membro i, si sottrae la proiezione OLS di
         resid_i su f_j.

    - **Diagonali** (mask a 1 entrata/riga, insiemi disgiunti): la deflazione non
      tocca gli altri fattori (nessuna serie condivisa), quindi ogni PCA e' sulla
      sotto-matrice GREZZA del blocco -> coincide con la PCA per-blocco classica.
    - **Overlap** (globale + locali): il globale (piu' largo) e' estratto per
      primo su tutte le serie; i locali si estraggono sui RESIDUI post-globale
      -> catturano il co-movimento specifico del blocco, ortogonale al globale.

    Parameters
    ----------
    Y_filled : pd.DataFrame
        Shape (T, M). Pannello standardizzato e riempito (nessun NaN). L'ordine
        delle colonne DEVE coincidere con ``structure.ordered_cols``.
    structure : FactorStructure | dict
        La struttura di fattore (loading mask). Un dict serie->blocco viene
        adattato a una struttura diagonale generica.

    Returns
    -------
    F : np.ndarray, shape (T, r)
        Matrice dei fattori iniziali; colonna j = fattore ``factor_names[j]``.
    factor_info : dict[str, np.ndarray]
        Nome del fattore -> autovettore v_j (loadings sui suoi membri).

    Notes
    -----
    **Sign convention (Convention 1).** L'autovettore e' indeterminato nel segno;
    si impone somma dei loadings >= 0, cosi' f_j e' positivo nelle espansioni.

    **Covarianza.** Sigma_b = Y_b.T @ Y_b / T (media gia' nulla per la
    standardizzazione); ``np.linalg.eigh`` -> primo PC = ultima colonna.
    """
    cols = list(Y_filled.columns)
    fs = as_structure(structure, cols)
    if fs.ordered_cols != cols:
        raise ValueError(
            "pca_initialization: l'ordine delle colonne di Y_filled non coincide "
            "con structure.ordered_cols. Ricostruisci la mask con lo stesso ordine."
        )

    Yv = Y_filled.to_numpy()          # (T, M)
    T = Yv.shape[0]
    r = fs.r

    resid = Yv.copy()                 # residui correnti (deflazione progressiva)
    F = np.zeros((T, r))
    factor_info: dict[str, np.ndarray] = {}

    for j in fs.extraction_order():
        members = fs.members(j)
        Y_b = resid[:, members]                    # (T, M_b) — residui dei membri
        Sigma_b = (Y_b.T @ Y_b) / T                # (M_b, M_b), simmetrica
        _, V = np.linalg.eigh(Sigma_b)             # autovalori crescenti
        v_j = V[:, -1].copy()                       # primo PC

        if v_j.sum() < 0:                           # sign convention
            v_j = -v_j

        f_j = Y_b @ v_j                             # (T,)
        F[:, j] = f_j
        factor_info[fs.factor_names[j]] = v_j

        # deflazione OLS dei membri per f_j (rispetta la mask: solo i membri)
        denom = float(f_j @ f_j)
        if denom > 0:
            for idx in members:
                lam = (resid[:, idx] @ f_j) / denom
                resid[:, idx] = resid[:, idx] - lam * f_j

    return F, factor_info


# ─── Initial parameter vector ─────────────────────────────────────────────────

def compute_theta_initial(
    Y_filled: pd.DataFrame,
    F: np.ndarray,
    structure: "FactorStructure | dict[str, str]",
    nu_init: float = 10.0,
    sigma_0_method: str = "identity",
    idio_ar1: bool = False,
    freq_list: list[str] | None = None,
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
        Initial factor matrix from ``pca_initialization``. Le colonne
        seguono l'ordine dei fattori della struttura
        (``FactorStructure.factor_names``). Must be fully filled.
    structure : FactorStructure or dict[str, str]
        Struttura di fattore: una ``FactorStructure`` (mask M x r) oppure
        una mappa serie -> blocco, adattata a struttura diagonale.
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
    cols = list(Y_filled.columns)
    fs = as_structure(structure, cols)
    if fs.ordered_cols != cols:
        raise ValueError(
            "compute_theta_initial: l'ordine delle colonne di Y_filled non "
            "coincide con structure.ordered_cols."
        )
    r = fs.r
    M = len(cols)
    T, r_F = F.shape
    if r_F != r:
        raise ValueError(f"F ha {r_F} colonne ma la mask ha r={r} fattori.")

    Yv = Y_filled.to_numpy()   # (T, M) — nessun NaN

    # ── 1. Lambda^(0): OLS mask-driven sul pannello bilanciato ────────────────
    # Thesis Section 4.3, eq. (4.2). Per ogni serie i, OLS sui SOLI fattori
    # attivi della sua riga di mask. Con una colonna attiva -> scalar OLS
    # (identico al caso diagonale classico); con due (globale+locale) -> OLS
    # multivariato. Le colonne non attive restano esattamente 0.
    Lambda = np.zeros((M, r))
    for i in range(M):
        cols_i = fs.factors_of_series(i)          # colonne attive (mask[i]==1)
        y_i = Yv[:, i]
        if cols_i.size == 1:
            j = int(cols_i[0])
            f_j = F[:, j]
            Lambda[i, j] = np.dot(y_i, f_j) / np.dot(f_j, f_j)
        else:
            Xf = F[:, cols_i]                     # (T, k_i)
            beta = np.linalg.solve(Xf.T @ Xf, Xf.T @ y_i)
            Lambda[i, cols_i] = beta

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
    resid_all = np.zeros((T, M))
    for i in range(M):
        resid = Yv[:, i] - F @ Lambda[i, :]
        resid_all[:, i] = resid
        R[i] = np.var(resid, ddof=1)

    # ── 3-bis. ASSE B: seed di rho^(0) e sigma^2^(0) ─────────────────────────
    # Un AR(1)-OLS sui residui idiosincratici del passo precedente. E' un seed,
    # non una stima: serve solo a far partire l'EM da un punto sensato invece
    # che da rho = 0 (che sarebbe il baseline e potrebbe lasciare l'algoritmo
    # in una zona piatta). Si tronca a |rho| <= 0.95 per stazionarieta'.
    rho_0 = None
    sigma2_0 = None
    if idio_ar1:
        rho_0 = np.zeros(M)
        sigma2_0 = np.zeros(M)
        for i in range(M):
            x, y = resid_all[:-1, i], resid_all[1:, i]
            den = float(x @ x)
            rho_0[i] = float(np.clip(y @ x / den, -0.95, 0.95)) if den > 1e-12 else 0.0
            innov = y - rho_0[i] * x
            sigma2_0[i] = max(float(np.var(innov, ddof=1)), 1e-8)

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
    dim = 5 * r    # stato aumentato MM (solo fattori); r generico
    if sigma_0_method == "identity":
        Sigma_0 = np.eye(dim)
        if idio_ar1:
            # Estende Sigma_0 al blocco idiosincratico con la STAZIONARIA
            # dell'AR(1): sigma^2/(1-rho^2) e le covarianze fra i lag delle
            # trimestrali. L'identita' sarebbe incoerente (la varianza
            # incondizionata di un AR(1) non e' 1).
            if freq_list is None:
                raise ValueError(
                    "compute_theta_initial: con idio_ar1=True serve freq_list "
                    "per costruire il layout del blocco idiosincratico."
                )
            from em.idio_ar1 import (  # noqa: PLC0415
                build_idio_layout, build_augmented_Sigma0,
            )
            Sigma_0 = build_augmented_Sigma0(
                Sigma_0, rho_0, sigma2_0, build_idio_layout(freq_list))
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
        **({"rho": rho_0, "sigma2": sigma2_0} if idio_ar1 else {}),
    }


# ─── Full pipeline wrapper ────────────────────────────────────────────────────

def initialize_theta(
    dataset_path: str | None = None,
    nu_init: float = 10.0,
    sigma_0_method: str = "identity",
    save: bool = True,
    random_state: int = 42,
    spec_name: str | None = None,
    out_dir: str | None = None,
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
        Path to the preprocessed CSV panel.  If ``None`` (with ``spec_name``
        given), defaults to ``data/processed/final/dataset_final.csv``.
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
    # Percorso unico: spec_name -> dataset 'final' (37 serie) + series_final.json.
    # Il vecchio percorso config_name via data_loader (small/big) e' stato rimosso.
    _project_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)

    if spec_name is None:
        raise ValueError(
            "initialize_theta richiede spec_name (dataset 'final'); il percorso "
            "config_name via data_loader (small/big) e' stato rimosso."
        )
    if dataset_path is None:
        dataset_path = os.path.join(
            _project_root, "data", "processed", "final", "dataset_final.csv")
    _series_cfg = os.path.join(_project_root, "config", "series_final.json")
    with open(_series_cfg, "r", encoding="utf-8") as _fh:
        _sc = json.load(_fh)
    FREQ = {s["series_id"]: ("monthly" if s["freq"] == "M" else "quarterly")
            for s in _sc["series"]}

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

    # ── struttura di fattore (loading mask) ───────────────────────────────────
    # spec_name -> build_loading_mask (dataset final).
    # Se arriva un dict serie->blocco (BLOCK), as_structure lo adatta a una
    # struttura diagonale generica (nessun nome di blocco cablato qui).
    if spec_name is not None:
        from em.factor_structure import build_loading_mask  # noqa: PLC0415
        structure = build_loading_mask(spec_name, list(df.columns))
    else:
        structure = as_structure(BLOCK, list(df.columns))

    # ── 2d. pca_initialization ────────────────────────────────────────────────
    F, _factor_info = pca_initialization(Y_filled, structure)

    # ── 2e. compute_theta_initial ─────────────────────────────────────────────
    theta_0 = compute_theta_initial(Y_filled, F, structure, nu_init, sigma_0_method)

    # ── 3. build metadata ─────────────────────────────────────────────────────
    T, M = Y_filled.shape
    r = F.shape[1]

    # dimensioni per fattore, derivate dalla mask (nomi generici)
    block_sizes = {
        structure.factor_names[j]: int(structure.mask[:, j].sum())
        for j in range(structure.r)
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
        # `out_dir` esplicito ha la precedenza: serve ai self-test per scrivere
        # in una cartella temporanea invece che fra gli artefatti veri.
        if out_dir is not None:
            pass
        elif spec_name is not None:
            # una cartella per spec: le tre strutture non si sovrascrivono.
            out_dir = os.path.join(_project_root, "data", "processed", "final", spec_name)
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
        config-aware path, e.g. ``data/processed/final/dataset_final.csv``.
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
    _project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    if dataset_path is None:
        raise ValueError(
            "load_standardized_data requires dataset_path. "
            "Pass the path to the config-aware CSV, e.g. data/processed/final/dataset_final.csv."
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
    # ─────────────────────────────────────────────────────────────────────────
    # Self-test dell'inizializzazione, parametrico sulla struttura di fattore.
    #
    #     python -m em.em_initialization --spec fed_overlap
    #     python -m em.em_initialization --spec diag4
    #     python -m em.em_initialization --spec diag3
    #
    # Nessun nome di serie, nome di fattore o valore di r cablato: tutto viene
    # dal dataset 'final' e da config/factor_specs.json via selftest_fixture.
    # ─────────────────────────────────────────────────────────────────────────
    import pathlib as _pl
    import sys as _sys_main
    _src_dir = str(_pl.Path(__file__).resolve().parent.parent)
    if _src_dir not in _sys_main.path:
        _sys_main.path.insert(0, _src_dir)

    from em.selftest_fixture import (
        parse_spec_args, load_fixture, selftest_scratch,
    )

    _args = parse_spec_args(
        "em_initialization self-test — standardise, MM fill, PCA, theta^(0).")
    fx = load_fixture(_args.spec)
    fs = fx.structure

    print("=" * 70)
    print(fx.describe())
    print("=" * 70)

    df, Y_std = fx.df, fx.Y_std_df
    tol = 1e-10

    # ── standardize ──────────────────────────────────────────────────────────
    print("\n--- standardize ---")
    mean_after, std_after = Y_std.mean(), Y_std.std()
    assert (mean_after.abs() < tol).all(), \
        f"medie non nulle:\n{mean_after[mean_after.abs() >= tol]}"
    assert ((std_after - 1.0).abs() < tol).all(), \
        f"std non unitarie:\n{std_after[(std_after - 1.0).abs() >= tol]}"
    assert (df.isna().sum() == Y_std.isna().sum()).all(), \
        "il conteggio dei NaN e' cambiato: la standardizzazione non deve riempire nulla."
    print(f"[OK] mean~0, std~1, NaN invariati su tutte le {fx.M} serie.")

    # ── mm_fill_quarterly: su OGNI serie trimestrale, non solo il target ──────
    print("\n--- mm_fill_quarterly ---")
    for qcol in fx.quarterly_cols:
        q_raw = Y_std[qcol]
        q_fill = mm_fill_quarterly(q_raw)
        is_qend = q_fill.index.month.isin([3, 6, 9, 12])

        # 1. ogni mese di un trimestre OSSERVATO deve essere riempito
        nan_in_observed = 0
        for qe in q_raw[is_qend & q_raw.notna()].index:
            for off in range(3):
                tgt = pd.Timestamp(qe.year, qe.month - 2 + off, 1) + pd.offsets.MonthEnd(0)
                if tgt in q_fill.index and pd.isna(q_fill[tgt]):
                    nan_in_observed += 1
        assert nan_in_observed == 0, \
            f"{qcol}: {nan_in_observed} NaN in mesi di trimestri osservati"

        # 2. identita' MM sotto ipotesi locally-constant: x^Q_m = 2*xi_m + xi_{m-1}
        xi = q_fill[is_qend & q_fill.notna()].values
        xq = q_raw[is_qend & q_raw.notna()].values
        recon = np.abs(2.0 * xi[1:] + xi[:-1] - xq[1:]).max()
        assert recon < 1e-10, f"{qcol}: ricostruzione MM fallita, max err = {recon:.2e}"
        print(f"  [OK] {qcol:<20s} max |2*xi_m + xi_(m-1) - x^Q_m| = {recon:.2e}  "
              f"(NaN residui: {int(q_fill.isna().sum())}, bordo/pre-campione)")

    # ── gaussian_fill_ragged ─────────────────────────────────────────────────
    print("\n--- gaussian_fill_ragged ---")
    Y_filled = fx.Y_filled
    assert Y_filled.isna().sum().sum() == 0, "gaussian_fill_ragged ha lasciato NaN."
    Y_mm_ref = Y_std.copy()
    for qcol in fx.quarterly_cols:
        Y_mm_ref[qcol] = mm_fill_quarterly(Y_std[qcol])
    obs = ~Y_mm_ref.isna()
    max_diff = (Y_filled[obs] - Y_mm_ref[obs]).abs().max().max()
    assert max_diff < 1e-14, \
        f"valori osservati alterati dal fill: max diff = {max_diff:.2e}"
    n_filled = int(Y_mm_ref.isna().sum().sum())
    print(f"[OK] zero NaN in output; {n_filled} celle riempite; "
          f"osservati preservati (max diff = {max_diff:.2e}).")

    # ── pca_initialization ───────────────────────────────────────────────────
    print("\n--- pca_initialization ---")
    F = fx.F0
    assert F.shape == (fx.T, fs.r), f"F.shape = {F.shape}, atteso ({fx.T}, {fs.r})"
    print(f"[OK] F.shape = {F.shape}  (fattori: {fs.factor_names})")
    print("     medie dei fattori: "
          + "  ".join(f"{n}={F[:, j].mean():+.4f}" for j, n in enumerate(fs.factor_names)))

    # correlazione di ciascun fattore con le SUE serie (membri dalla mask).
    print("\n     corr serie-membro vs fattore (membri = mask[:, j] == 1):")
    for j, fname in enumerate(fs.factor_names):
        members = fs.members(j)
        cors = [np.corrcoef(Y_filled.iloc[:, i].to_numpy(), F[:, j])[0, 1]
                for i in members]
        print(f"       {fname:<4s} ({len(members):>2d} serie)  media |corr| = "
              f"{np.mean(np.abs(cors)):.4f}   range [{min(cors):+.3f}, {max(cors):+.3f}]")

    # ── compute_theta_initial ────────────────────────────────────────────────
    print("\n--- compute_theta_initial ---")
    theta_0 = fx.theta0
    Lambda, A_var, Q_var = theta_0["Lambda"], theta_0["A"], theta_0["Q"]
    R_var, Sigma_0 = theta_0["R"], theta_0["Sigma_0"]
    M, r = fx.M, fs.r

    assert Lambda.shape == (M, r), f"Lambda.shape = {Lambda.shape}"
    assert A_var.shape == (r, r), f"A.shape = {A_var.shape}"
    assert Q_var.shape == (r, r), f"Q.shape = {Q_var.shape}"
    assert R_var.shape == (M,), f"R.shape = {R_var.shape}"
    assert Sigma_0.shape == (5 * r, 5 * r), f"Sigma_0.shape = {Sigma_0.shape}"
    print(f"[OK] shape: Lambda{Lambda.shape}  A{A_var.shape}  Q{Q_var.shape}  "
          f"R{R_var.shape}  Sigma_0{Sigma_0.shape}")

    # Lambda deve rispettare la MASK: zero esatto fuori dalle entrate ammesse.
    # Per diag3/diag4 la mask ha una entrata per riga, per fed_overlap ne ha
    # due (globale + locale), e il test e' lo stesso.
    off_mask = np.abs(Lambda[fs.mask == 0])
    off_mask_max = float(off_mask.max()) if off_mask.size else 0.0
    assert off_mask_max == 0.0, \
        f"Lambda ha entrate non nulle fuori dalla mask: max = {off_mask_max:.2e}"
    n_active = int(fs.mask.sum())
    print(f"[OK] Lambda rispetta la mask: {n_active} entrate attive su {M * r}, "
          f"fuori-mask esattamente 0 (max = {off_mask_max:.2e})")

    assert (R_var > 0).all(), \
        f"R non positiva agli indici {np.where(R_var <= 0)[0].tolist()}"
    sym_err = float(np.max(np.abs(Q_var - Q_var.T)))
    assert sym_err < 1e-14, f"Q non simmetrica: max |Q - Q.T| = {sym_err:.2e}"
    eig_Q = np.linalg.eigvalsh(Q_var)
    assert (eig_Q > 0).all(), \
        f"Q non definita positiva: min autovalore = {eig_Q.min():.4e}"
    print(f"[OK] R > 0 (min = {R_var.min():.6f});  Q simmetrica ({sym_err:.2e}) e "
          f"definita positiva (min autoval = {eig_Q.min():.6f})")

    eig_A = np.linalg.eigvals(A_var)
    mod_A = np.abs(eig_A)
    assert (mod_A < 1.0).all(), \
        f"VAR(1) iniziale non stabile: moduli = {np.round(mod_A, 4).tolist()}"
    print("[OK] VAR(1) stabile: moduli autovalori di A = "
          f"{np.round(np.sort(mod_A)[::-1], 4).tolist()}")

    assert bool(np.all(theta_0["w_u"] == 1.0)), "w_u^(0) non tutti 1"
    assert bool(np.all(theta_0["w_eps"] == 1.0)), "w_eps^(0) non tutti 1"
    assert np.allclose(Sigma_0, np.eye(5 * r)), "Sigma_0^(0) != I_{5r}"
    print(f"[OK] pesi iniziali == 1, Sigma_0 == I_{{{5 * r}}}, "
          f"nu_u = {theta_0['nu_u']}, nu_eps = {theta_0['nu_eps']}")

    # ── Lambda per fattore (leggibile) ───────────────────────────────────────
    print("\n--- Lambda: caricamenti attivi per fattore ---")
    for j, fname in enumerate(fs.factor_names):
        members = fs.members(j)
        vals = Lambda[members, j]
        print(f"  {fname:<4s} ({len(members):>2d} serie): "
              f"media {vals.mean():+.4f}  |min| {np.abs(vals).min():.4f}  "
              f"max |.| {np.abs(vals).max():.4f}")
    tgt_i = fx.col(fx.target)
    tgt_f = [fs.factor_names[j] for j in fs.factors_of_series(tgt_i)]
    print(f"\n  target '{fx.target}' carica su {tgt_f}: "
          + "  ".join(f"{fs.factor_names[j]}={Lambda[tgt_i, j]:+.4f}"
                      for j in fs.factors_of_series(tgt_i)))

    # ── initialize_theta: wrapper completo + round-trip su disco ─────────────
    print("\n--- initialize_theta (pipeline completa + round-trip) ---")
    # scrive in una cartella TEMPORANEA: il self-test verifica, non produce
    # artefatti (vedi selftest_scratch).
    _scratch = selftest_scratch(f"em_init_{_args.spec}")
    theta_w, F_w, meta_w = initialize_theta(save=True, spec_name=_args.spec,
                                            out_dir=_scratch)

    out_dir = _pl.Path(_scratch)
    npz_path = out_dir / "theta_initial.npz"
    json_path = out_dir / "theta_initial_metadata.json"
    assert npz_path.exists(), f"manca: {npz_path}"
    assert json_path.exists(), f"manca: {json_path}"

    loaded = np.load(str(npz_path))
    for key in ("Lambda", "A", "Q", "R"):
        d = float(np.max(np.abs(theta_w[key] - loaded[key])))
        assert d == 0.0, f"round-trip non identico per '{key}': max diff = {d:.2e}"
    print(f"[OK] salvato in {out_dir} e ricaricato bit-identico (Lambda, A, Q, R).")

    # il wrapper deve riprodurre la pipeline passo-passo: stessa spec, stesso seed
    for key in ("Lambda", "A", "Q", "R"):
        d = float(np.max(np.abs(np.asarray(theta_w[key]) - np.asarray(theta_0[key]))))
        assert d < 1e-12, f"initialize_theta != fixture per '{key}': max diff = {d:.2e}"
    print("[OK] initialize_theta coincide con la pipeline passo-passo (max diff < 1e-12).")

    assert meta_w["r"] == r and meta_w["M"] == M and meta_w["T"] == fx.T
    assert set(meta_w["block_sizes"]) == set(fs.factor_names), \
        f"factor_sizes {list(meta_w['block_sizes'])} != mask {fs.factor_names}"
    print(f"     T={meta_w['T']}, M={meta_w['M']}, r={meta_w['r']}  "
          f"sample {meta_w['sample_start']} -> {meta_w['sample_end']}")
    print(f"     serie per fattore : {meta_w['block_sizes']}")

    print(f"\n[OK] em_initialization self-test superato per spec '{_args.spec}'.")
