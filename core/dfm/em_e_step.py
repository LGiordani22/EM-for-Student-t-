"""
core/dfm/em_e_step.py

Student-t E-step machinery for the mixed-frequency Dynamic Factor Model.

This module implements the *weight side* of the E-step: given the smoothed
factor moments produced by :mod:`kalman`, it computes the squared Mahalanobis
residuals d^u_t and d^eps_t that drive the posterior updates of the
Student-t mixing weights w^u_t and w^eps_t.

Thesis reference
----------------
EM_for_student_t.tex:
  - Section "The E-Step" (line ~4129) — overall structure.
  - Section "Resolving the Coupling: ECM with an Inner Iteration"
    (line ~4963) — ECM block-coordinate procedure that ties together the
    Kalman moments and the weights.
  - Section "The Weight Update with Missing Data"
    (subsec:weight-missing, line ~7046), eq:d-eps-missing (line ~7054)
    and eq:w-eps-hat-missing (line ~7073).

TASK 1 — idiosyncratic Mahalanobis residual with missing data:
  - compute_d_eps(Y, f_smooth, P_smooth, Lambda_tilde, R, W_list)
    -> d_eps (T,), m_obs (T,)

TASK 2 — factor-side Mahalanobis residual:
  - compute_d_u(f_smooth, P_smooth, P_lag, A, Q, r)
    -> d_u (T,), with d_u[0] = NaN (no f_{-1} in sample).

TASK 3 — posterior mean and log-mean of the mixing weights from the
Gamma conjugate posterior:
  - compute_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r)
    -> dict with w_eps, w_u, log_w_eps, log_w_u (each shape (T,)).

TASK 4 — ECM inner loop: coordinate ascent that resolves the coupling
between Student-t mixing weights and smoothed factor moments:
  - ecm_inner_loop(Y, theta, freq_list, tol_inner, max_inner, verbose)
    -> dict with converged f_smooth/P_smooth/P_lag, w_eps/w_u, log_w_eps/log_w_u,
       loglik, n_inner_iter, converged.

TASK 5 — high-level wrapper (entry point of the full E-step):
  - run_e_step(Y, theta, freq_list, tol_inner, max_inner, verbose, save_path)
    -> dict with all converged E-step outputs needed by the M-step
       (smoothed factor moments + weights + log-weights + Lambda_tilde),
       plus loglik and inner-loop diagnostics.  Called once per outer
       EM iteration by em_main.py.

ASSE B — idiosyncratic AR(1) in the state:
  When ``theta`` carries ``rho`` the idiosyncratic errors live INSIDE the
  augmented state (see :mod:`em.idio_ar1`), so the observation equation is
  (nearly) deterministic and the baseline ``compute_d_eps`` degenerates.  The
  residual is then the fresh AR(1) innovation,
    d^eps_t = sum_i E[(eps_{i,t} - rho_i eps_{i,t-1})^2 | Y] / sigma_i^2,
  computed by :func:`compute_d_eps_ar1`, symmetric to ``d_u``: its degrees of
  freedom count M (every series has an innovation at every t, observed or not),
  not m_t.  With ``per_series_weights`` the E-step returns one weight per cell
  w^eps_{i,t} of shape (T, M) instead of a shared w^eps_t; this is coherent
  only under the AR(1) state and is used by the ``student_t_ar1`` variant.
"""

import numpy as np
from scipy.special import digamma


# ─── 1. Idiosyncratic Mahalanobis residual with missing data ──────────────────

def compute_d_eps(
    Y: np.ndarray,
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    Lambda_tilde: np.ndarray,
    R: np.ndarray,
    W_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the squared Mahalanobis residual d^eps_t for every time step t,
    using only the m_t observed series at that t.

    Thesis reference
    ----------------
    EM_for_student_t.tex, eq:d-eps-missing (line ~7054), Section
    "The Weight Update with Missing Data" (subsec:weight-missing,
    line ~7046).  The shape parameter (nu_eps + m_t)/2 of the posterior
    Gamma distribution for w^eps_t (eq:w-eps-hat-missing, line ~7073)
    requires both d_eps[t] and m_obs[t] = m_t; this routine returns both.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Full observation panel with NaN for missing entries (ragged edge
        and/or quarterly mask).  NaN entries are not algebraically accessed:
        they are silently replaced with 0 prior to the W_t projection.
    f_smooth : np.ndarray, shape (T, 5r)
        Smoothed augmented state means f_{t|T} from
        :func:`kalman.kalman_smoother`.
    P_smooth : np.ndarray, shape (T, 5r, 5r)
        Smoothed augmented state covariances P_{t|T} from the same routine.
    Lambda_tilde : np.ndarray, shape (M, 5r)
        Augmented loading matrix; see :func:`kalman.build_Lambda_tilde`.
        Row i is the effective loading of series i on the 5r-dimensional
        augmented state.  For monthly series only the contemporaneous
        block is non-zero; for quarterly series the row is spread across
        the five lag-blocks with Mariano-Murasawa weights.
    R : np.ndarray, shape (M,)
        Idiosyncratic variances (diagonal of R).  Strictly positive.
    W_list : list of np.ndarray, length T
        Selection matrices, ``W_list[t]`` of shape ``(m_t, M)``; see
        :func:`kalman.build_all_selection_matrices`.  ``m_t = 0`` is
        allowed (empty array of shape ``(0, M)``).

    Returns
    -------
    d_eps : np.ndarray, shape (T,)
        Squared Mahalanobis residual at each t, computed on the observed
        sub-vector y_t^obs.  Non-negative by construction.  ``d_eps[t] = 0``
        when ``m_t = 0``.
    m_obs : np.ndarray, shape (T,), dtype int
        Number of observed series at each t (rows of W_list[t]).  Used in
        the downstream weight update as the shape parameter
        ``(nu_eps + m_t)/2``.

    Notes
    -----
    **Formula (eq:d-eps-missing).**
    For every t with m_t > 0:

    .. math::

        d^{\varepsilon,\mathrm{obs}}_t \;=\;
        \underbrace{(y_t^{\mathrm{obs}} - W_t \tilde{\Lambda}\, \hat{f}_{t|T})'
                    \,(W_t R W_t')^{-1}\,
                    (y_t^{\mathrm{obs}} - W_t \tilde{\Lambda}\, \hat{f}_{t|T})}_{\text{point-estimate Mahalanobis}}
        \;+\;
        \underbrace{\mathrm{tr}\!\big[(W_t R W_t')^{-1}\, (W_t \tilde{\Lambda})\,
                                       P_{t|T}\, (W_t \tilde{\Lambda})'\big]}_{\text{posterior-uncertainty trace}}.

    **Two-term decomposition.**
    The first term measures how far the observation is from its
    factor-implied point estimate, scaled by the idiosyncratic precision.
    The second term penalises posterior uncertainty about the factors:
    even when the point-estimate residual is exactly zero, smoothed
    covariance P_{t|T} contributes positively to the *expected* squared
    residual through the trace of inv(WR) @ WL @ P_smooth @ WL.T.  Both
    terms come out of taking the expectation of the squared residual under
    the conditional posterior of the factors given Y.

    **Why ``Lambda_tilde`` (augmented), not ``Lambda``.**
    The thesis writes the formula with the monthly loading Lambda and the
    monthly factor f_t; here the smoothed state is the 5r-dimensional
    augmented state tilde_f_t = (f_t, f_{t-1}, ..., f_{t-4}).
    ``Lambda_tilde`` is built so that ``Lambda_tilde @ f_smooth`` is exactly
    the right factor-implied quantity for *both* frequency types:

    - Monthly series row i has zeros outside the first r columns, so the
      product reduces to L^M_{i.} @ f_t — the contemporaneous expectation.
    - Quarterly series row i carries L^Q_{i.} across all five lag-blocks
      with MM weights, so the product equals
      (1/3) L^Q f_t + (2/3) L^Q f_{t-1} + L^Q f_{t-2} + (2/3) L^Q f_{t-3}
      + (1/3) L^Q f_{t-4}, the MM-aggregated quarterly value.

    The same multiplication therefore handles both frequencies without
    any branching, and ``d_eps[t]`` is correctly computed in both cases.

    **Why m_t (not M).**
    With missing data, only the m_t observed series contribute to the
    expected complete-data log-likelihood term that yields d_eps.  Series
    not observed at t carry no information about the residual and must be
    excluded.  This is encoded by the W_t projection in the formula above.
    The companion quantity ``m_obs[t]`` is returned because the posterior
    Gamma distribution for w^eps_t has shape ``(nu_eps + m_t)/2`` — the
    weight update needs it.

    **m_t = 0 (all-NaN time step).**
    When no observation is available at t, the weight w^eps_t is not
    identified by data: its posterior collapses to the prior
    ``Gamma(nu_eps/2, nu_eps/2)`` with posterior mean 1.  Setting
    ``d_eps[t] = 0`` and ``m_obs[t] = 0`` ensures the downstream weight
    update reproduces exactly this prior mean (since
    ``(nu_eps + 0) / (nu_eps + 0) = 1``).

    **Exploiting the diagonal structure of R.**
    Because R is diagonal, ``WR = W_t diag(R) W_t' = diag(R[obs_idx])`` is
    also diagonal (with obs_idx the indices of observed series at t), and
    its inverse is the trivial ``diag(1/R[obs_idx])``.  No m_t-dimensional
    matrix inverse or linear solve is needed:

    - term1 collapses to ``sum_i resid_i^2 / R[obs_idx[i]]``.
    - term2 collapses to ``sum_i (WL @ P_smooth @ WL.T)[i, i] / R[obs_idx[i]]``.

    Only the diagonal of ``WL @ P_smooth @ WL.T`` is needed.  This is O(m_t^2 * 5r)
    arithmetic per t, with no inversion at all.

    **Recovering observed indices from W_t.**
    Each row of W_t (built by :func:`kalman.build_selection_matrix`) has
    exactly one 1, in the column corresponding to its observed series.
    Hence ``obs_idx = np.argmax(W_t, axis=1)`` recovers the indices of
    observed series, in the same row order as the projected sub-vector
    ``W_t @ y_t``.

    **Non-negativity.**
    Both terms are quadratic forms / traces involving the positive
    definite ``inv(WR) = diag(1/R[obs_idx])`` and the positive
    semi-definite ``P_smooth[t]``, so d_eps[t] >= 0 for every t.  This is
    asserted in the self-test.
    """
    T, M = Y.shape

    d_eps = np.zeros(T)
    m_obs = np.zeros(T, dtype=int)

    # NaN -> 0 once, so that subsequent W_t @ y_t is safe everywhere.
    Y_filled = np.where(np.isnan(Y), 0.0, Y)

    for t in range(T):
        W_t = W_list[t]
        m_t = W_t.shape[0]
        m_obs[t] = m_t

        # ── Boundary: no observations at t -> weight stays at prior mean ──────
        if m_t == 0:
            d_eps[t] = 0.0
            continue

        # ── Recover indices of observed series and the corresponding R ────────
        # W_t row i has a single 1 in column obs_idx[i]; argmax extracts it.
        obs_idx = np.argmax(W_t, axis=1)              # (m_t,)
        R_obs = R[obs_idx]                            # (m_t,) variances
        inv_R_obs = 1.0 / R_obs                       # diag(inv(WR))

        # ── Effective loading restricted to observed series ───────────────────
        # WL shape (m_t, 5r); rows are the augmented loadings of observed series.
        WL = W_t @ Lambda_tilde                       # (m_t, 5r)

        # ── Observed sub-vector y_t^obs ───────────────────────────────────────
        y_obs = W_t @ Y_filled[t]                     # (m_t,)

        # ── Point-estimate residual and term1 = resid' inv(WR) resid ──────────
        resid = y_obs - WL @ f_smooth[t]              # (m_t,)
        term1 = float(np.sum((resid * resid) * inv_R_obs))

        # ── Term2 = tr(inv(WR) @ WL @ P_smooth[t] @ WL.T):                    ─
        #     compute only the diagonal of WL @ P @ WL.T, weighted by inv_R_obs.
        WLP = WL @ P_smooth[t]                        # (m_t, 5r)
        diag_quad = np.einsum("ij,ij->i", WLP, WL)    # (m_t,)  -- diag(WLP @ WL.T)
        term2 = float(np.sum(diag_quad * inv_R_obs))

        d_eps[t] = term1 + term2

    return d_eps, m_obs


def compute_d_eps_ar1(f_smooth, P_smooth, P_lag, rho, sigma2, layout, d_f,
                      Y=None, per_series: bool = False):
    r"""
    Residuo di Mahalanobis idiosincratico quando eps E' NELLO STATO (Asse B).

    PERCHE' SERVE UNA VERSIONE DIVERSA
    ----------------------------------
    :func:`compute_d_eps` misura il residuo dell'equazione di OSSERVAZIONE,
    pesato per ``1/R``. Sotto l'Asse B quell'equazione diventa deterministica
    (``R_tilde -> 0``): il residuo tende a zero e ``1/R`` a infinito, quindi la
    formula del baseline e' degenere e non puo' generare i pesi.

    Il residuo giusto e' quello dell'INNOVAZIONE FRESCA dell'AR(1) — la
    quantita' che sotto l'Asse B porta effettivamente lo shock idiosincratico,
    esattamente come ``d_u`` fa per l'innovazione dei fattori:

    .. math::
        d^\varepsilon_t \;=\; \sum_i
            \frac{\mathbb{E}\big[(\varepsilon_{i,t}
                  - \rho_i \varepsilon_{i,t-1})^2 \mid Y\big]}{\sigma_i^2}.

    La simmetria con ``d_u`` non e' un'analogia: entrambi sono ora residui
    dell'equazione di STATO, e i pesi restano l'inverso di una media di
    quadrati standardizzati. Il conteggio dei gradi di liberta' cambia di
    conseguenza: non piu' ``m_t`` (serie osservate a t) ma ``M``, perche' ogni
    serie ha una propria innovazione a ogni t **anche quando non e'
    osservata** — l'informazione su di essa viene dal solo prior, e lo
    smoother la propaga.

    Parameters
    ----------
    f_smooth, P_smooth, P_lag : np.ndarray
        Momenti posteriori dello stato AUMENTATO (T, d_f + n_e).
    rho, sigma2 : np.ndarray, shape (M,)
    layout : IdioLayout
    d_f : int
        Dimensione del blocco fattori (5r): dove inizia il blocco idio.
    Y : np.ndarray, optional
        Non usato nel calcolo; accettato per simmetria di firma.

    Returns
    -------
    d_eps : np.ndarray, shape (T,)
    m_eff : np.ndarray, shape (T,)
        Gradi di liberta' per periodo: costante = M (vedi sopra).
    """
    from dfm.idio_ar1 import (  # noqa: PLC0415
        extract_idio_moments, idio_innovation_second_moment,
    )
    mom = extract_idio_moments(f_smooth, P_smooth, P_lag, layout, d_f)
    S = idio_innovation_second_moment(mom, rho)          # (T, M)
    sigma2 = np.asarray(sigma2, float).ravel()
    D = S / sigma2[None, :]                              # (T, M), per serie
    # t = 0: eps_{i,-1} non e' definito, come per d_u.  Si usa il secondo
    # momento non condizionato, cioe' E[eps_0^2]/sigma^2.
    D[0, :] = mom["E_e2"][0, :] / sigma2

    if per_series:
        # Un peso per SERIE e per periodo (formulazione del .tex): ciascuna
        # innovazione e' scalare, quindi 1 grado di liberta' ciascuna.
        return D, np.ones_like(D)
    # Peso CONDIVISO (semplificazione del baseline): un solo residuo per
    # periodo, con M gradi di liberta'.
    return D.sum(axis=1), np.full(f_smooth.shape[0], float(layout.M))


# ─── 2. Factor-side Mahalanobis residual ──────────────────────────────────────

def compute_d_u(
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    P_lag: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    r: int,
) -> np.ndarray:
    r"""
    Compute the squared Mahalanobis residual d^u_t of the factor innovation
    for every time step t >= 1, taking expectation under the smoothed
    posterior of the factors.

    Thesis reference
    ----------------
    EM_for_student_t.tex, eq:d-u (line ~4649) — definition of
    ``d^u_t = (f_t - A f_{t-1})' Q^{-1} (f_t - A f_{t-1})`` (point form,
    i.e. for known f_t, f_{t-1}).  In the E-step we replace the latent
    factors by their smoothed posterior moments, which adds a trace
    correction for posterior uncertainty (analogous to the trace term
    of d^eps_t in eq:d-eps-missing, line ~7054).

    Parameters
    ----------
    f_smooth : np.ndarray, shape (T, 5r)
        Smoothed augmented state means f_{t|T} from
        :func:`kalman.kalman_smoother`.  Block 0 (columns 0:r) is the
        contemporaneous monthly factor; block 1 (columns r:2r) is the
        one-lag factor encoded inside the augmented state at time t.
    P_smooth : np.ndarray, shape (T, 5r, 5r)
        Smoothed augmented state covariances P_{t|T} from the same routine.
    P_lag : np.ndarray, shape (T, 5r, 5r)
        Smoothed lag-one cross-covariance P_{t, t-1 | T} from the same
        routine; ``P_lag[t][0:r, 0:r]`` is the r-by-r monthly block of
        Cov(f_t, f_{t-1} | Y_{1:T}).  ``P_lag[0]`` is undefined (returned
        as zeros by the smoother).
    A : np.ndarray, shape (r, r)
        Monthly VAR(1) transition matrix (NOT the augmented A_tilde).
    Q : np.ndarray, shape (r, r)
        Monthly innovation covariance (NOT the augmented Q_tilde).
        Must be symmetric positive-definite.
    r : int
        Number of monthly latent factors (= 3 in this project).

    Returns
    -------
    d_u : np.ndarray, shape (T,)
        Squared expected Mahalanobis residual of the factor innovation at
        each t.  ``d_u[0] = NaN`` because f_{-1} does not exist within the
        sample; ``d_u[t] >= 0`` for t >= 1 by construction (sum of a
        quadratic form and a trace of two PSD matrices' product).

    Notes
    -----
    **Formula (eq:d-u, expected form).**
    For every t >= 1:

    .. math::

        d^u_t \;=\;
        \underbrace{(\hat{f}_{t \mid T} - \mA \hat{f}_{t-1 \mid T})'
                    \,\mQ^{-1}\,
                    (\hat{f}_{t \mid T} - \mA \hat{f}_{t-1 \mid T})}_{\text{innovation point estimate}}
        \;+\;
        \underbrace{\mathrm{tr}\!\big[\mQ^{-1}\, V_t\big]}_{\text{posterior uncertainty}},

    where the conditional innovation covariance under the smoother is

    .. math::

        V_t \;=\; P_{t \mid T} \;+\; \mA\, P_{t-1 \mid T}\, \mA'
                  \;-\; \mA\, P_{t, t-1 \mid T}^{\,\prime}
                  \;-\; P_{t, t-1 \mid T}\, \mA'.

    Both blocks refer to the **monthly** (r-dimensional) sub-state, not
    the augmented (5r) state, because the VAR(1) dynamics live in the
    monthly space.

    **Derivation in one line.** Taking the conditional expectation of
    ``(f_t - A f_{t-1})' Q^{-1} (f_t - A f_{t-1})`` under
    ``p(f_t, f_{t-1} | Y_{1:T})`` and using
    ``E[xx' | Y] = Var[x | Y] + E[x | Y] E[x | Y]'`` produces
    quadratic-form + trace exactly as above, with the cross term
    ``Cov(f_t, f_{t-1} | Y) = P_{t, t-1 | T}`` entering through the
    ``-A P_{t,t-1}' - P_{t,t-1} A'`` cancellations.

    **Extracting monthly blocks from the augmented state.**
    The smoother returns the augmented state
    ``tilde_f_t = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4})``.  The
    monthly factor ``f_t`` is therefore ``f_smooth[t][0:r]``; the lag
    block ``f_{t-1}`` is encoded **twice** in the augmented system —
    inside ``f_smooth[t][r:2r]`` (because tilde_f_t contains its own past)
    and inside ``f_smooth[t-1][0:r]`` (because tilde_f_{t-1} has it as
    its contemporaneous block).  Both must coincide if the smoother is
    consistent.  This implementation uses the **second** form (state at
    time t-1 + ``P_lag[t]`` for the cross-covariance), which is closer to
    the thesis notation and makes the role of the lag-one smoothed
    covariance explicit.  The companion test in ``__main__`` numerically
    verifies that the alternative form (internal blocks of tilde_f_t)
    gives the same d_u up to machine precision — a non-trivial
    confirmation that the augmented state-space encodes the lag structure
    correctly.

    **Why Q (monthly), not Q_tilde (augmented).**
    The factor innovation is r-dimensional: ``u_t = f_t - A f_{t-1}`` with
    ``u_t ~ t_{nu_u}(0, Q)``.  The augmented ``Q_tilde`` is just Q
    embedded in a 5r x 5r block-zero matrix that reflects the fact that
    the four lag-blocks of tilde_f_t carry no fresh innovation — it is
    rank-r and not invertible.  All quadratic forms and traces for
    ``d^u_t`` therefore operate in the monthly r-dimensional space with
    the original Q.

    **Boundary at t = 0.**
    The model has no ``f_{-1}`` within the sample — the initial state is
    treated diffusely via ``Sigma_0`` in the filter.  We therefore set
    ``d_u[0] = NaN`` to signal that the factor-side weight ``w^u_0`` is
    not informed by data and the downstream weight update must default
    to the prior mean ``(nu_u + r) / nu_u`` — or equivalently use shape
    ``nu_u / 2`` (and 0 in the numerator dimension) to recover the
    prior mean of 1.  Returning NaN (instead of 0) prevents accidental
    contamination of sample means computed over d_u in the M-step.

    **Implementation.**
    We solve ``Q v = innovation`` and ``Q M = V_t`` with
    ``np.linalg.solve`` (Q is 3x3 SPD, well-conditioned ~20 at theta^(0))
    instead of explicitly inverting Q.  This is both more accurate and
    slightly cheaper.  ``term2 = trace(solve(Q, V_t))``.

    **Non-negativity.**
    ``term1 = innovation' Q^{-1} innovation >= 0`` because Q^{-1} is
    PSD; ``term2 = tr(Q^{-1} V_t) >= 0`` because V_t is the smoothed
    posterior covariance of the innovation ``f_t - A f_{t-1}``
    (PSD by construction) and the trace of the product of two PSD
    matrices is non-negative.  Both bounds hold up to numerical noise.
    """
    T = f_smooth.shape[0]
    d_u = np.full(T, np.nan)

    for t in range(1, T):
        f_t   = f_smooth[t][0:r]                  # (r,)
        f_tm1 = f_smooth[t - 1][0:r]              # (r,)  -- contemporaneous block of state at t-1
        P_t   = P_smooth[t][0:r, 0:r]             # (r, r)
        P_tm1 = P_smooth[t - 1][0:r, 0:r]         # (r, r)
        P_t_tm1 = P_lag[t][0:r, 0:r]              # (r, r)  -- Cov(f_t, f_{t-1} | Y_{1:T})

        # ── Term 1: innovation' Q^{-1} innovation ────────────────────────────
        innovation = f_t - A @ f_tm1              # (r,)
        # Solve Q v = innovation, then term1 = innovation @ v = innovation' Q^{-1} innovation
        v = np.linalg.solve(Q, innovation)
        term1 = float(innovation @ v)

        # ── Term 2: trace(Q^{-1} V_t) ────────────────────────────────────────
        V_t = (
            P_t
            + A @ P_tm1 @ A.T
            - A @ P_t_tm1.T
            - P_t_tm1 @ A.T
        )                                          # (r, r), symmetric in theory
        # trace(Q^{-1} V_t) = trace(solve(Q, V_t))
        term2 = float(np.trace(np.linalg.solve(Q, V_t)))

        d_u[t] = term1 + term2

    return d_u


def _compute_d_u_internal_blocks(
    f_smooth: np.ndarray,
    P_smooth: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    r: int,
) -> np.ndarray:
    """
    Alternative computation of d_u using ONLY the augmented state at time t.

    For each t >= 1, f_{t-1} and its covariance are extracted from the
    internal lag-block of the smoothed augmented state at time t, instead
    of from the smoothed state at time t-1 + P_lag[t]:

        f_{t-1}   = f_smooth[t][r:2r]
        P_{t-1}   = P_smooth[t][r:2r, r:2r]
        P_{t,t-1} = P_smooth[t][0:r, r:2r]

    This helper is intended for **testing only**.  Algebraically it must
    coincide with :func:`compute_d_u` because the augmented smoother
    propagates a single coherent posterior over the lag-stack; the test
    in ``__main__`` verifies the equivalence to machine precision and
    therefore confirms that the augmented state-space implementation is
    internally consistent.
    """
    T = f_smooth.shape[0]
    d_u = np.full(T, np.nan)

    for t in range(1, T):
        f_t     = f_smooth[t][0:r]
        f_tm1   = f_smooth[t][r:2 * r]
        P_t     = P_smooth[t][0:r, 0:r]
        P_tm1   = P_smooth[t][r:2 * r, r:2 * r]
        P_t_tm1 = P_smooth[t][0:r, r:2 * r]

        innovation = f_t - A @ f_tm1
        v = np.linalg.solve(Q, innovation)
        term1 = float(innovation @ v)

        V_t = (
            P_t
            + A @ P_tm1 @ A.T
            - A @ P_t_tm1.T
            - P_t_tm1 @ A.T
        )
        term2 = float(np.trace(np.linalg.solve(Q, V_t)))

        d_u[t] = term1 + term2

    return d_u


# ─── 3. Posterior mean and log-mean of the mixing weights ─────────────────────

def compute_weights(
    d_eps: np.ndarray,
    d_u: np.ndarray,
    m_obs: np.ndarray,
    nu_eps: float,
    nu_u: float,
    r: int,
) -> dict[str, np.ndarray]:
    r"""
    Compute the posterior mean and log-mean of the Student-t mixing weights
    w^eps_t and w^u_t under their conjugate Gamma posteriors.

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - eq:w-hat (line ~4684), full-data posterior means
            \hat{w}^eps_t = (nu_eps + M) / (nu_eps + d^eps_t),
            \hat{w}^u_t   = (nu_u + r) / (nu_u + d^u_t).
      - eq:log-w-hat-eps (line ~4693), posterior log-mean for w^eps_t,
            E[log w^eps_t] = psi((nu_eps + M)/2) - log((nu_eps + d^eps_t)/2).
      - eq:log-w-hat-u (line ~4701), posterior log-mean for w^u_t.
      - eq:w-eps-hat-missing (line ~7123), missing-data variant: M is
            replaced by m_t (number of observed series at time t).  The
            same substitution carries through to the log-mean.

    Parameters
    ----------
    d_eps : np.ndarray, shape (T,)
        Idiosyncratic Mahalanobis residuals from :func:`compute_d_eps`.
        Non-negative; equals 0 when m_obs[t] = 0.
    d_u : np.ndarray, shape (T,)
        Factor-side Mahalanobis residuals from :func:`compute_d_u`.
        Non-negative for t >= 1; ``d_u[0] = NaN`` (no f_{-1} in sample).
    m_obs : np.ndarray, shape (T,)
        Number of observed series at each t.  Used as the time-varying
        dimension that replaces M in the missing-data posterior; see
        eq:w-eps-hat-missing.
    nu_eps : float
        Degrees of freedom of the idiosyncratic Student-t distribution;
        strictly > 2 in practice.
    nu_u : float
        Degrees of freedom of the factor-innovation Student-t
        distribution; strictly > 2 in practice.
    r : int
        Number of monthly latent factors (= dimension of the factor
        innovation u_t).

    Returns
    -------
    dict with keys
        ``w_eps``     : np.ndarray (T,) -- posterior mean  E[w^eps_t | .]
        ``w_u``       : np.ndarray (T,) -- posterior mean  E[w^u_t   | .]
        ``log_w_eps`` : np.ndarray (T,) -- posterior mean  E[log w^eps_t | .]
        ``log_w_u``   : np.ndarray (T,) -- posterior mean  E[log w^u_t   | .]

    Notes
    -----
    **Closed-form Gamma moments.**
    The Gaussian-Gamma conjugacy yields posteriors
    ``w^eps_t | . ~ Gamma((nu_eps + m_t)/2, (nu_eps + d^eps_t)/2)`` and
    ``w^u_t   | . ~ Gamma((nu_u + r)/2,   (nu_u + d^u_t)/2)``.
    For a generic ``Gamma(alpha, beta)`` distribution,

    .. math::

        \mathbb{E}[W] = \frac{\alpha}{\beta},
        \qquad
        \mathbb{E}[\log W] = \psi(\alpha) - \log \beta,

    where ``psi`` is the digamma function (the derivative of
    ``log Gamma``).  Applying this to the two posteriors gives the four
    formulas computed below.

    **Anatomy of the weight formula (eq:w-hat, thesis ~line 4709).**
    Take w^eps_t:

    .. math::

        \hat{w}^{\varepsilon}_t \;=\;
        \frac{\nu_\varepsilon + m_t}{\nu_\varepsilon + d^{\varepsilon}_t}.

    The numerator depends only on nu_eps and the observation count
    m_t (so it varies with t only through how many series are observed,
    not through the data values).
    The denominator is ``nu_eps + Mahalanobis residual``: a month with
    large d_eps (outlier) gets a weight much smaller than 1 -- the
    M-step will automatically down-weight that observation.  A month
    with small d_eps (well-fit by the factors) gets a weight close to
    ``(nu_eps + m_t)/nu_eps`` -- approximately 1 for moderate d_eps.
    The same logic applies to w^u_t with r and d_u in place of m_t and
    d_eps.  No threshold, no tuning constant: down-weighting is smooth
    and entirely governed by the estimated nu's.

    **Gaussian limit.**
    Let nu -> infinity (both nu_eps and nu_u):

    .. math::

        \frac{\nu + k}{\nu + d} \;=\; \frac{1 + k/\nu}{1 + d/\nu}
        \;\longrightarrow\; 1.

    All weights collapse to 1 and the model reduces to a standard
    Gaussian DFM -- no down-weighting.  The self-test verifies this
    numerically at nu = 10000.

    **Role of the log-mean.**
    ``log_w_eps`` and ``log_w_u`` are needed ONLY by the M-step update
    of the degrees-of-freedom parameters (the score equation for nu has
    a ``psi`` term that is solved against sample means of log-weights).
    The Kalman filter and the L/A/Q/R updates use the plain weights
    only.  This is why we return both: the same E-step pass produces
    everything the M-step will need.

    **Why digamma.**
    ``psi`` arises because the Gamma is in the exponential family with
    log-partition function ``log Gamma(alpha)``: taking the expectation
    of ``log w`` with respect to a Gamma posterior pulls in the
    derivative of ``log Gamma``, which is exactly ``psi``.  This is the
    same reason ``psi`` shows up in the M-step update of nu (the score
    equation for the Student-t shape parameter).

    **Handling t = 0 for w^u (d_u[0] = NaN).**
    The first time step has no factor-side innovation (f_{-1} is not in
    the sample; the initial state is treated diffusely via Sigma_0 in
    the filter).  We therefore set the posterior of w^u_0 equal to its
    PRIOR ``Gamma(nu_u/2, nu_u/2)`` (no data evidence), which yields:

    .. math::

        \hat{w}^u_0       \;=\; 1,
        \qquad
        \widehat{\log w}^u_0 \;=\; \psi(\nu_u / 2) - \log(\nu_u / 2).

    The plain mean is exactly 1 (Gamma(a, a) has mean 1).  The
    log-mean is the PRIOR log-mean -- generally NOT zero -- but it is
    the value coherent with the prior so that the downstream M-step
    update of nu_u sees a sample mean of log-weights that integrates
    correctly with the rest of the sample.

    **Handling m_obs[t] = 0 for w^eps.**
    No special case is needed: when m_t = 0 we have d_eps[t] = 0 by
    construction in :func:`compute_d_eps`, so the formula gives
    ``(nu_eps + 0) / (nu_eps + 0) = 1`` and
    ``psi(nu_eps/2) - log(nu_eps/2)`` -- exactly the prior mean and
    log-mean, as desired.  In the current dataset m_t >= 1 everywhere
    so this branch is never exercised, but the formula is correct.

    **Non-negativity.**
    Both w_eps and w_u are strictly positive by construction (numerator
    and denominator are both > 0 since nu > 0, d >= 0, m >= 0, r > 0).
    log-means can take any real value.
    """
    T = d_eps.shape[0]

    # ── Posterior mean of w_eps  (eq:w-eps-hat-missing) ──────────────────────
    # alpha_eps[t] = (nu_eps + m_obs[t]) / 2
    # beta_eps[t]  = (nu_eps + d_eps[t]) / 2
    # E[w_eps_t]   = alpha / beta = (nu_eps + m_obs[t]) / (nu_eps + d_eps[t])
    w_eps = (nu_eps + m_obs) / (nu_eps + d_eps)

    # ── Posterior mean of w_u  (eq:w-hat for u; unaffected by missing data) ──
    # alpha_u = (nu_u + r) / 2,  beta_u[t] = (nu_u + d_u[t]) / 2
    w_u = (nu_u + r) / (nu_u + d_u)
    # t = 0: d_u[0] = NaN -> w_u[0] = NaN. Replace with prior mean (= 1).
    w_u[0] = 1.0

    # ── Posterior log-mean of w_eps  (eq:log-w-hat-eps with m_t) ─────────────
    alpha_eps = (nu_eps + m_obs) / 2.0
    beta_eps  = (nu_eps + d_eps) / 2.0
    log_w_eps = digamma(alpha_eps) - np.log(beta_eps)

    # ── Posterior log-mean of w_u  (eq:log-w-hat-u) ──────────────────────────
    alpha_u = (nu_u + r) / 2.0
    beta_u  = (nu_u + d_u) / 2.0
    log_w_u = digamma(alpha_u) - np.log(beta_u)
    # t = 0: d_u[0] = NaN -> log_w_u[0] = NaN. Replace with PRIOR log-mean,
    # i.e. evaluate at the prior Gamma(nu_u/2, nu_u/2): psi(nu_u/2) - log(nu_u/2).
    log_w_u[0] = digamma(nu_u / 2.0) - np.log(nu_u / 2.0)

    return {
        "w_eps":     w_eps,
        "w_u":       w_u,
        "log_w_eps": log_w_eps,
        "log_w_u":   log_w_u,
    }


# ─── 4. ECM inner loop: coupling weights and factor moments ──────────────────

def ecm_inner_loop(
    Y: np.ndarray,
    theta: dict,
    freq_list: list[str] | None = None,
    tol_inner: float = 1e-4,
    max_inner: int = 100,
    verbose: bool = False,
    gaussian: bool = False,
    w_init: dict | None = None,
    strict_inner: bool = False,
) -> dict:
    r"""
    Run the ECM inner loop that resolves the coupling between the
    Student-t mixing weights and the smoothed factor moments.

    Thesis reference
    ----------------
    EM_for_student_t.tex:
      - Section "Resolving the Coupling: ECM with an Inner Iteration"
        (subsec:coupling, line ~5173) — motivation and mean-field
        variational interpretation.
      - Section "The Inner Loop" (line ~5240) — algorithm, inner
        initialisation at k=0, F-update / W-update sub-steps,
        convergence criterion eq. line ~5346-5349, and convergence
        rate (3-10 iterations in practice).
      - Section "Summary of the E-Step" (subsec:e-step-summary,
        line ~5465) — full algorithmic summary of the outer E-step
        in which this inner loop is embedded.

    Background: the coupling problem
    ---------------------------------
    The E-step faces a fundamental coupling: the Kalman smoother
    (F-update) requires the weights w^eps_t and w^u_t to build the
    time-varying covariances Q/w^u_t and R/w^eps_t, but the weight
    update (W-update) requires the smoothed factor residuals
    d^eps_t and d^u_t, which are themselves Kalman output.  Neither
    update can run without the other.

    Resolution: coordinate ascent on the ELBO
    ------------------------------------------
    The standard ECM solution (Meng and Rubin 1993) embeds an inner
    loop inside the E-step that alternates the two updates with
    theta held fixed:

      k = 0:  initialise  w^eps = w^u = 1  (prior means)
      k >= 1:
        (F-update)  Kalman(theta, w^u_{k-1}, w^eps_{k-1})
                    -> smoothed moments {f_{t|T}^[k], P^[k], P_lag^[k]}
        (W-update)  d_eps, d_u from new moments
                    -> weights w^eps^[k], w^u^[k]
        (converge?) max_t|w^eps^[k]-w^eps^[k-1]|
                  + max_t|w^u^[k]  -w^u^[k-1]  |  <  tol_inner => stop

    Each sub-step minimises the KL divergence between the variational
    factor q and the true posterior with respect to one conditional
    factor while holding the other fixed (coordinate descent on the
    ELBO).  This guarantees monotone improvement in the mean-field
    ELBO and convergence to the factorised approximation
    q = q_F * q_{W^u} * q_{W^eps}.

    Key invariants
    --------------
    - theta (Lambda, A, Q, R, nu_eps, nu_u, Sigma_0) is NEVER
      modified.  The inner loop moves only weights and factor moments.
      nu_eps and nu_u are held fixed throughout the entire E-step
      (thesis line ~5497).
    - W_list (selection matrices) is built once from Y and reused
      across all inner iterations.
    - Weights are reinitialised to 1 at the START of every call
      (i.e. every outer EM iteration); the previous outer iteration's
      converged weights are NOT carried forward, because theta has
      changed.
    - run_kalman reconstructs Q_tilde and R_tilde at every call,
      because their time-varying entries depend on the current weights.

    Convergence
    -----------
    In practice 3 to 10 inner iterations suffice (thesis line ~5352).
    The weights move quickly toward their posterior means in the first
    few steps and then oscillate within tol_inner of the fixed point.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Observation panel with NaN for missing entries.
    theta : dict
        Current EM parameter estimates.  Required keys:
        ``Lambda`` (M, r), ``A`` (r, r), ``Q`` (r, r), ``R`` (M,),
        ``Sigma_0`` (5r, 5r), ``nu_eps`` (float), ``nu_u`` (float).
    freq_list : list[str] or None
        Frequency label for each series ('monthly' / 'quarterly').
        If None, imported from :mod:`data_loader`.
    tol_inner : float
        Convergence tolerance on the sum of max absolute weight
        changes across time steps (default 1e-4).
    max_inner : int
        Safety cap on inner iterations; a RuntimeWarning is issued
        if this limit is reached without convergence (default 50).
    verbose : bool
        If True, print a table of (k, delta, loglik) at each inner
        iteration so the convergence trajectory is visible.

    Returns
    -------
    dict with keys
        # Factor moments (converged)
        ``f_smooth``     : np.ndarray (T, 5r)
        ``P_smooth``     : np.ndarray (T, 5r, 5r)
        ``P_lag``        : np.ndarray (T, 5r, 5r)
        # Weight moments (converged)
        ``w_eps``        : np.ndarray (T,)
        ``w_u``          : np.ndarray (T,)
        ``log_w_eps``    : np.ndarray (T,)
        ``log_w_u``      : np.ndarray (T,)
        # Diagnostics
        ``n_inner_iter`` : int   -- number of inner iterations executed
        ``loglik``       : float -- log-likelihood at the last iteration
        ``converged``    : bool  -- True iff tol_inner was reached
    """
    import warnings

    from kalman import build_all_selection_matrices, run_kalman

    T, _ = Y.shape
    A      = theta["A"]
    Q      = theta["Q"]
    R      = theta["R"]

    # ── ASSE B: idiosincratico AR(1) nello stato ─────────────────────────────
    # Con eps nello stato l'osservazione e' deterministica, quindi il residuo
    # di osservazione non puo' generare i pesi: si usa quello dell'innovazione
    # AR(1) (vedi compute_d_eps_ar1).  Il resto del ciclo ECM — punto fisso
    # F-update / W-update, criterio di convergenza, best-iterate — resta
    # identico: cambia solo COME si calcola d_eps.
    _idio_ar1 = ("rho" in theta) and (theta["rho"] is not None)
    # Pesi idiosincratici per-serie (w^eps_{i,t}) invece del peso condiviso
    # (w^eps_t).  E' la formulazione che il .tex sviluppa come "conceptually
    # correct": con la persistenza modellata serie per serie, un abbattimento
    # delle code globale sarebbe internamente incoerente.  Vale solo sotto
    # l'Asse B: nel baseline eps non e' nello stato e i residui per-serie non
    # sono separabili dall'equazione di osservazione.
    _per_series_w = bool(theta.get("per_series_weights", False)) and _idio_ar1

    # Criterio di arresto del ciclo interno: "max" (DEFAULT) o "rms".
    # Viaggia dentro theta come gli altri flag di variante, cosi' non serve
    # infilarlo nella firma di ogni chiamante intermedio.
    #
    # IL DEFAULT RESTA "max", DELIBERATAMENTE. "rms" e' attivato dalla sola
    # variante `student_t_ar1` (vedi run_final_artifacts.VARIANTS), E IL MOTIVO
    # E' `per_series_weights`, NON IL TEMPO DI CALCOLO.
    #
    # Il discrimine: `max` e' DIPENDENTE DALLA DIMENSIONE. Coi pesi per serie il
    # massimo e' preso su T*M variazioni invece che su T, e il massimo di un
    # insieme piu' grande e' sistematicamente piu' grande: la stessa tolleranza
    # diventa una richiesta piu' severa per pura dimensione, e `max` su questa
    # cella non e' confrontabile con `max` sulle celle a pesi condivisi. "rms"
    # de-distorce, ed e' comunque la scala giusta nel merito, perche' il M-step
    # consuma MEDIE pesate: cio' che deve convergere e' l'aggregato, non il
    # singolo peso peggiore.
    #
    # Le altre tre mantengono il criterio "max" — la distorsione dimensionale
    # li' non si presenta:
    #
    #   gaussian, gaussian_ar1 : saltano del tutto il ciclo interno (nu->inf
    #                            => pesi ≡ 1), quindi il criterio e' irrilevante;
    #   student_t              : il ciclo lo usa, ma i pesi sono CONDIVISI (max
    #                            su T, nessuna distorsione dimensionale) e gli
    #                            bastano ~10 iterazioni; col solo warm start
    #                            scende da 69 s a 35 s (theta invariato a
    #                            1.1e-06). Non c'e' niente da correggere.
    #
    # Il guadagno su `student_t_ar1` (~47 iterazioni interne) e' una conseguenza
    # gradita, non la motivazione: 581 s -> 157 s a tolleranza esterna invariata,
    # con loglik NON peggiore (-9722.26 contro -9721.54 del solo warm start;
    # entrambe migliori del -9731.60 della baseline a freddo).
    #
    # CAVEAT da dichiarare se il criterio viene discusso: il rapporto max/RMS
    # misurato si stabilizza a ~36.7, cioe' nella coda i due criteri sono
    # PROPORZIONALI. Il passaggio equivale quindi a riscalare la tolleranza per
    # la dimensione, non a imporre un criterio sostanzialmente diverso — e ne
    # segue che le quattro celle non condividono la stessa tolleranza interna
    # effettiva. E' l'intento (era `max` a essere non comparabile), ma va
    # descritto cosi'.
    _criterion = str(theta.get("inner_criterion", "max")).lower()
    if _criterion not in ("max", "rms"):
        raise ValueError(
            f"inner_criterion deve essere 'max' o 'rms', non {_criterion!r}.")

    if _per_series_w:
        # Piu' spazio al ciclo interno quando i pesi sono per-serie, con
        # QUALUNQUE criterio.
        #
        # Con `max` serve perche' il criterio e' un massimo su T*M valori
        # invece che su T: con molte piu' entrate il massimo di un insieme di
        # variazioni e' sistematicamente piu' grande, e la stessa tolleranza
        # diventa una richiesta piu' severa per pura dimensione.
        #
        # Con `rms` serve per un motivo diverso e altrettanto concreto: il
        # tetto non deve MAI essere il vincolo che determina il risultato. Un
        # pannello diverso — un altro vintage, una finestra piu' corta, un
        # theta iniziale piu' lontano — puo' richiedere piu' passate di quelle
        # osservate finora, e un tetto stretto le troncherebbe in silenzio
        # producendo un E-step peggiore senza che nulla lo segnali se non un
        # warning. Il tetto e' una rete contro il ciclo infinito, non un
        # parametro di taratura: va tenuto largo, e se viene raggiunto e' un
        # sintomo da guardare, non una configurazione da accettare.
        max_inner = max(int(max_inner), 300)
    if _idio_ar1:
        from dfm.idio_ar1 import build_idio_layout  # noqa: PLC0415
        _layout = build_idio_layout(freq_list)
        _rho = np.asarray(theta["rho"], float).ravel()
        _sigma2 = np.asarray(theta.get("sigma2", R), float).ravel()
        _d_f = 5 * int(np.asarray(theta["A"]).shape[0])
    nu_eps = float(theta["nu_eps"])
    nu_u   = float(theta["nu_u"])
    r      = A.shape[0]

    # Selection matrices are fixed for all inner iterations (depend only on Y).
    W_list = build_all_selection_matrices(Y)

    # ── k = 0: punto di partenza del punto fisso ─────────────────────────────
    # Di default le medie a priori, w = 1 (thesis line ~5248-5257).
    #
    # WARM START. Il punto fisso converge allo stesso w qualunque sia il punto
    # di partenza — quello che cambia e' solo quante iterate servono. Dentro
    # l'EM il theta si muove pochissimo da un'iterazione esterna alla
    # successiva (nella coda, |dL|/|L| ~ 1e-5), quindi i pesi corretti sono
    # quasi esattamente quelli gia' calcolati al giro precedente: ripartire da
    # w = 1 significa ri-percorrere ogni volta l'intera traiettoria.
    #
    # Misurato su diag3/student_t_ar1 al theta convergiuto (tol 1e-4):
    #   partenza a freddo (w=1)          : 47 iterazioni, 18.6 s
    #   partenza dai pesi precedenti     :  1 iterazione,   0.4 s
    #   idem con theta perturbato di 1e-3:  6 iterazioni,   2.3 s
    #   idem con theta perturbato di 1e-2:  8 iterazioni,   3.0 s
    #
    # Il criterio di arresto NON viene indebolito: resta |w_new - w_old| < tol,
    # cioe' "la mappa di punto fisso non muove piu' i pesi". Partire vicino
    # alla soluzione rende quel test soddisfatto prima perche' la soluzione e'
    # davvero li', non perche' il test sia diventato piu' permissivo.
    if w_init is not None:
        w_eps = np.array(w_init["w_eps"], dtype=float, copy=True)
        w_u   = np.array(w_init["w_u"], dtype=float, copy=True)
        _want = (T, Y.shape[1]) if _per_series_w else (T,)
        if w_eps.shape != _want or w_u.shape != (T,):
            raise ValueError(
                f"w_init incoerente: w_eps {w_eps.shape} (atteso {_want}), "
                f"w_u {w_u.shape} (atteso {(T,)}).")
        if not (np.all(np.isfinite(w_eps)) and np.all(w_eps > 0)
                and np.all(np.isfinite(w_u)) and np.all(w_u > 0)):
            raise ValueError("w_init deve essere finito e strettamente positivo.")
    else:
        w_eps = np.ones((T, Y.shape[1])) if _per_series_w else np.ones(T)
        w_u   = np.ones(T)

    # ── Gaussian bypass (Bańbura-Modugno 2014 limit) ─────────────────────────
    # When nu -> infinity the scale-mixture weight prior Gamma(nu/2, nu/2)
    # collapses to a point mass at w = 1: there is no latent scale dispersion
    # to infer.  The E-step therefore bypasses the inner ECM loop entirely and
    # performs a single Kalman pass with unit weights, recovering the Gaussian
    # (Bańbura-Modugno) DFM exactly.  This is the nu -> infinity limit in which
    # the Student-t model nests the Gaussian one.
    #
    # Concretely, with ``gaussian=True``: ν_u, ν_ε → ∞, all weights are
    # identically one, the inner ECM fixed point degenerates, and the E-step
    # reduces to a single Kalman pass with w_u = w_eps = 1.  This is exactly
    # the E-step of the standard Gaussian mixed-frequency DFM (Bańbura-Modugno
    # 2014) — the benchmark to which the thesis compares the proposed estimator
    # in the Monte Carlo experiments (see EM_for_student_t.tex, line
    # ~11061-11070: "In the limit ν_u, ν_ε → ∞ ... reduce exactly to the
    # Bańbura-Modugno (2014) formulation").  The single pass returns log-weights
    # equal to zero (E[log w | w=1] = 0), so the M-step's weighted moments
    # collapse to their unweighted Gaussian counterparts automatically.
    if gaussian:
        ks_gauss = run_kalman(Y, theta, w_u=w_u, w_eps=w_eps, freq_list=freq_list)
        return {
            "f_smooth":     ks_gauss["f_smooth"],
            "P_smooth":     ks_gauss["P_smooth"],
            "P_lag":        ks_gauss["P_lag"],
            "w_eps":        w_eps,
            "w_u":          w_u,
            "log_w_eps":    np.zeros(T),
            "log_w_u":      np.zeros(T),
            "Lambda_tilde": ks_gauss["Lambda_tilde"],
            "n_inner_iter": 1,
            "loglik":       float(ks_gauss["loglik"]),
            "converged":    True,
            # Limite gaussiano: i pesi sono ≡1 e non c'e' nessun posteriore da
            # cui contare gradi di liberta'. La correzione ELBO in questo ramo
            # non viene nemmeno calcolata (nu = inf), ma la chiave c'e' lo
            # stesso perche' la forma del dizionario non dipenda dalla variante.
            "m_obs_eps":    None,
        }

    converged    = False
    n_inner_iter = 0
    loglik       = float("nan")
    # Keep a reference so the return dict has weights_new in scope after break.
    weights_new  = {"log_w_eps": np.zeros(T), "log_w_u": np.zeros(T)}

    # Best-iterate safeguard: the fixed-point can be unstable at low nu (the
    # delta decreases to a local minimum then diverges before the cap is hit).
    # We track the best state seen so far and return it on non-convergence
    # instead of the final (potentially diverged) iterate.
    best_delta = float("inf")
    best_state: dict | None = None

    if verbose:
        print(f"  {'iter':>4}  {'delta':>12}  {'loglik':>14}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*14}")

    for k in range(1, max_inner + 1):
        n_inner_iter = k

        # ── F-update: run Kalman with current weights ─────────────────────────
        ks = run_kalman(Y, theta, w_u=w_u, w_eps=w_eps, freq_list=freq_list)
        f_smooth     = ks["f_smooth"]
        P_smooth     = ks["P_smooth"]
        P_lag        = ks["P_lag"]
        Lambda_tilde = ks["Lambda_tilde"]
        loglik       = ks["loglik"]

        # ── W-update: new weights from updated smoothed moments ───────────────
        if _idio_ar1:
            d_eps, m_obs = compute_d_eps_ar1(
                f_smooth, P_smooth, P_lag, _rho, _sigma2, _layout, _d_f,
                per_series=_per_series_w,
            )
            # d_u guarda solo il blocco fattori dello stato aumentato.
            d_u = compute_d_u(f_smooth[:, :_d_f], P_smooth[:, :_d_f, :_d_f],
                              P_lag[:, :_d_f, :_d_f], A, Q, r)
        else:
            d_eps, m_obs = compute_d_eps(
                Y, f_smooth, P_smooth, Lambda_tilde, R, W_list
            )
            d_u         = compute_d_u(f_smooth, P_smooth, P_lag, A, Q, r)
        weights_new = compute_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r)
        w_eps_new   = weights_new["w_eps"]
        w_u_new     = weights_new["w_u"]

        # ── Convergence check (thesis eq. line ~5346-5349) ───────────────────
        # w_u[0] = 1.0 always (d_u[0]=NaN -> prior mean), so its contribution
        # to delta is identically 0; np.nanmax / np.nanmean sono NaN-safe.
        #
        # DUE CRITERI.
        #  "max" (default, storico): max|Delta w|. E' un massimo su T*M valori
        #    coi pesi per-serie invece che su T: con 37x piu' entrate la stessa
        #    tolleranza e' una richiesta piu' severa per pura DIMENSIONE.
        #  "rms": radice della media dei quadrati. E' la scala che conta per il
        #    M-step, che consuma MEDIE pesate: cio' che deve convergere e'
        #    l'aggregato, non il singolo peso peggiore.
        #
        # Misurato su diag3/student_t_ar1 al theta convergiuto: il rapporto
        # max/RMS si stabilizza a ~36.7, cioe' nella coda i due criteri sono
        # PROPORZIONALI. Passare a "rms" non e' quindi un criterio diverso in
        # senso sostanziale: e' lo stesso criterio con la tolleranza riscalata
        # per la dimensione del problema. Va detto, per non vendere come
        # cambiamento di sostanza quello che e' un cambiamento di scala.
        d_eps_w = np.abs(w_eps_new - w_eps)
        d_u_w   = np.abs(w_u_new - w_u)
        if _criterion == "rms":
            delta = (float(np.sqrt(np.mean(d_eps_w ** 2)))
                     + float(np.sqrt(np.nanmean(d_u_w ** 2))))
        else:
            delta = float(np.max(d_eps_w)) + float(np.nanmax(d_u_w))

        if verbose:
            print(f"  {k:>4}  {delta:>12.6f}  {loglik:>14.2f}")

        # ── Track best iterate (minimum delta seen so far) ────────────────────
        if delta < best_delta:
            best_delta = delta
            best_state = {
                "f_smooth":     f_smooth,
                "P_smooth":     P_smooth,
                "P_lag":        P_lag,
                "Lambda_tilde": Lambda_tilde,
                "w_eps":        w_eps_new,
                "w_u":          w_u_new,
                "log_w_eps":    weights_new["log_w_eps"],
                "log_w_u":      weights_new["log_w_u"],
                "loglik":       loglik,
                "m_obs_eps":    m_obs,
            }

        w_eps = w_eps_new
        w_u   = w_u_new

        if delta < tol_inner:
            converged = True
            break

    if not converged and strict_inner:
        # IL TETTO RAGGIUNTO E' UN SINTOMO, NON UNA CONFIGURAZIONE.
        # E' scritto sopra, dove `max_inner` viene alzato a 300 per i pesi
        # per-serie: "se viene raggiunto e' un sintomo da guardare, non una
        # configurazione da accettare".  Restituire comunque il best iterate
        # accetta un E-step troncato e lo passa all'M-step, che ci costruisce
        # sopra un theta; nella passata del 16-8-2026 e' cosi' che
        # `diag3/student_t_ar1` e' arrivato a 60,6 minuti PER STIMA — una
        # iterazione EM da ~6 passate di Kalman ne faceva 300, ripetuta ~90
        # volte fa i 2709 s che si leggono nel log.
        #
        # Sul percorso di STIMA quindi si rifiuta: la ri-stima abortisce e il
        # chiamante ricade sul theta precedente, esattamente come per un theta
        # degenere.  Si spende una iterazione patologica invece di novanta.
        #
        # `strict_inner` resta False sul percorso di FILTRO (`filter_only`,
        # theta congelato): li' non si sta stimando niente, il best iterate e'
        # il meglio disponibile, e sollevare toglierebbe il ripiego proprio a
        # chi serve da ripiego.
        raise RuntimeError(
            f"ecm_inner_loop non converge in {max_inner} iterazioni "
            f"(delta finale {delta:.3e}, migliore {best_delta:.3e}, "
            f"tol {tol_inner:.3e}): E-step troncato, la stima non si accetta."
        )
    if not converged:
        warnings.warn(
            f"ecm_inner_loop did not converge in {max_inner} iterations "
            f"(final delta = {delta:.3e}, best delta = {best_delta:.3e}, "
            f"tol = {tol_inner:.3e}). "
            "Returning best iterate (minimum delta) rather than last iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Return the best iterate, not the diverged last one.
        bs = best_state  # guaranteed non-None: loop ran at least once
        return {
            "f_smooth":          bs["f_smooth"],
            "P_smooth":          bs["P_smooth"],
            "P_lag":             bs["P_lag"],
            "w_eps":             bs["w_eps"],
            "w_u":               bs["w_u"],
            "log_w_eps":         bs["log_w_eps"],
            "log_w_u":           bs["log_w_u"],
            "Lambda_tilde":      bs["Lambda_tilde"],
            "n_inner_iter":      n_inner_iter,
            "loglik":            bs["loglik"],
            "converged":         False,
            "used_best_iterate": True,
            "m_obs_eps":         bs["m_obs_eps"],
        }

    return {
        "f_smooth":          f_smooth,
        "P_smooth":          P_smooth,
        "P_lag":             P_lag,
        "w_eps":             w_eps,
        "w_u":               w_u,
        "log_w_eps":         weights_new["log_w_eps"],
        "log_w_u":           weights_new["log_w_u"],
        "Lambda_tilde":      Lambda_tilde,
        "n_inner_iter":      n_inner_iter,
        "loglik":            loglik,
        "converged":         True,
        "used_best_iterate": False,
        # Gradi di liberta' EFFETTIVI di ciascun peso idiosincratico, cioe' il
        # numero di osservazioni che informano quel peso. E' l'`m_obs` usato
        # QUI per costruire i pesi, ed e' esposto perche' la correzione ELBO
        # deve usare lo STESSO oggetto invece di ricalcolarselo: le due strade
        # erano divergenti e il risultato era un ELBO che non corrispondeva al
        # modello stimato. Forma (T,) col peso condiviso, (T, M) per-serie.
        "m_obs_eps":         m_obs,
    }


# ─── 5. High-level E-step wrapper (entry point) ───────────────────────────────

def run_e_step(
    Y: np.ndarray,
    theta: dict,
    freq_list: list[str] | None = None,
    tol_inner: float = 1e-4,
    max_inner: int = 100,
    verbose: bool = False,
    save_path=None,
    gaussian: bool = False,
    w_init: dict | None = None,
    strict_inner: bool = False,
) -> dict:
    r"""
    High-level entry point: full E-step at a single outer EM iteration.

    Thesis reference
    ----------------
    EM_for_student_t.tex, Section "Summary of the E-Step"
    (subsec:e-step-summary, line ~5465).  This wrapper executes the full
    E-step described there: it runs the ECM inner loop (Task 4) to
    convergence and packages the converged factor moments, weights and
    log-weights into a single dictionary that constitutes the input to
    the M-step.

    Parameters
    ----------
    Y : np.ndarray, shape (T, M)
        Observation panel with NaN for missing entries.
    theta : dict-like
        Current EM parameter estimates theta^(j).  Required keys:
        ``Lambda`` (M, r), ``A`` (r, r), ``Q`` (r, r), ``R`` (M,),
        ``Sigma_0`` (5r, 5r), ``nu_eps`` (float), ``nu_u`` (float).
        Typically an ``npz`` archive loaded via ``np.load`` or a plain
        ``dict`` produced by the M-step.
    freq_list : list[str] or None, optional
        Frequency label ('monthly' / 'quarterly') for each of the M
        columns of Y.  If None, imported from :mod:`data_loader` in the
        canonical project column order.
    tol_inner : float, optional
        Convergence tolerance for the ECM inner loop (default 1e-4).
    max_inner : int, optional
        Safety cap on inner iterations (default 50); a ``RuntimeWarning``
        is issued by :func:`ecm_inner_loop` if reached without convergence.
    verbose : bool, optional
        If True, print the per-iteration (k, delta, loglik) table of the
        inner loop.  Default False.
    save_path : str or pathlib.Path or None, optional
        If provided, persist the principal E-step outputs to an ``.npz``
        archive at this path: ``f_smooth``, ``P_smooth``, ``P_lag``,
        ``w_eps``, ``w_u``, ``log_w_eps``, ``log_w_u``, ``Lambda_tilde``,
        ``loglik``.  Default None (nothing saved).

    Returns
    -------
    dict with keys
        # Smoothed factor moments — principal M-step inputs.
        ``f_smooth``     : np.ndarray (T, 5r)      E[f_t | y_{1:T}]
        ``P_smooth``     : np.ndarray (T, 5r, 5r)  Var[f_t | y_{1:T}]
        ``P_lag``        : np.ndarray (T, 5r, 5r)  Cov[f_t, f_{t-1} | y_{1:T}]

        # Weight moments — second principal M-step input.
        ``w_eps``        : np.ndarray (T,)         E[w^eps_t | y_{1:T}]
        ``w_u``          : np.ndarray (T,)         E[w^u_t   | y_{1:T}]
        ``log_w_eps``    : np.ndarray (T,)         E[log w^eps_t | y_{1:T}]
        ``log_w_u``      : np.ndarray (T,)         E[log w^u_t   | y_{1:T}]

        # Augmented loading matrix — reused by the M-step / debug.
        ``Lambda_tilde`` : np.ndarray (M, 5r)

        # Log-likelihood at convergence — monitor outer EM convergence.
        ``loglik``       : float

        # Inner-loop diagnostics.
        ``n_inner_iter`` : int   -- inner iterations executed
        ``converged``    : bool  -- True iff tol_inner was reached

        # Metadata.
        ``T``, ``M``, ``r`` : int

    Notes
    -----
    **What this wrapper does (and does NOT do).**
    ``run_e_step`` is the ENTRY POINT for the full E-step at outer
    iteration j.  It calls :func:`ecm_inner_loop`, which itself
    alternates between :func:`kalman.run_kalman` (F-update) and
    :func:`compute_d_eps` + :func:`compute_d_u` + :func:`compute_weights`
    (W-update) until the inner-loop convergence criterion is met.
    No M-step update of theta is performed here — that is the role of
    the next module (``em_m_step``).

    **Where this fits in the EM cycle.**
    At outer iteration j:
        theta^(j)  --run_e_step-->  {f_smooth, P_smooth, P_lag,
                                     w_eps, w_u, log_w_eps, log_w_u}
                   --em_m_step  -->  theta^(j+1).
    The two procedures are wrapped by ``em_main.run_em`` (Task 1 of
    em_main), which iterates until the OUTER log-likelihood stops
    increasing.

    **Weights are NOT carried across outer iterations.**
    :func:`ecm_inner_loop` reinitialises ``w^eps = w^u = 1`` at every
    call, by design (thesis line ~5472-5473): theta has just changed,
    so the previous outer iteration's converged weights are no longer
    the right warm-start.  This is invisible to the caller; we mention
    it here only because it explains why ``run_e_step`` has no
    "warm-start" argument.

    **nu_eps and nu_u are held fixed within the E-step.**
    Both degrees-of-freedom parameters are fixed at their current
    estimates ``theta["nu_eps"]``, ``theta["nu_u"]`` for the entirety of
    the E-step (thesis line ~5497).  They are updated separately by the
    ECME numerical root-finding step in the M-step.

    **Persistence.**
    ``save_path`` writes a numpy ``.npz`` archive that can be reloaded
    with ``np.load`` for inspection or to seed the M-step without
    re-running the inner loop.  Only the principal outputs are saved —
    inner-loop metadata is for in-memory diagnostics only.

    **No code duplication.**
    All algebra lives in Tasks 1-4 of this module and in
    :mod:`kalman`; this wrapper only orchestrates them and packages the
    output dictionary in the form expected by the M-step.
    """
    # ── 1. Run the ECM inner loop to convergence (F + W updates) ─────────────
    ecm = ecm_inner_loop(
        Y, theta,
        freq_list=freq_list,
        tol_inner=tol_inner,
        w_init=w_init,
        max_inner=max_inner,
        verbose=verbose,
        gaussian=gaussian,
        strict_inner=strict_inner,
    )

    # ── 2. Metadata ───────────────────────────────────────────────────────────
    T, M = Y.shape
    r = int(theta["A"].shape[0])

    # ── 3. Assemble the M-step input dictionary ──────────────────────────────
    out = {
        # smoothed factor moments
        "f_smooth":     ecm["f_smooth"],
        "P_smooth":     ecm["P_smooth"],
        "P_lag":        ecm["P_lag"],
        # weight moments
        "w_eps":        ecm["w_eps"],
        "w_u":          ecm["w_u"],
        "log_w_eps":    ecm["log_w_eps"],
        "log_w_u":      ecm["log_w_u"],
        # Gradi di liberta' effettivi dei pesi idiosincratici: li consuma la
        # correzione ELBO, che DEVE usare gli stessi dell'E-step.
        "m_obs_eps":    ecm.get("m_obs_eps"),
        # augmented loading matrix (computed once inside the inner loop)
        "Lambda_tilde": ecm["Lambda_tilde"],
        # outer-EM monitoring
        "loglik":       ecm["loglik"],
        # inner-loop diagnostics
        "n_inner_iter":      ecm["n_inner_iter"],
        "converged":         ecm["converged"],
        "used_best_iterate": ecm.get("used_best_iterate", False),
        # metadata
        "T": T, "M": M, "r": r,
    }

    # ── 4. Optional persistence ──────────────────────────────────────────────
    if save_path is not None:
        np.savez(
            save_path,
            f_smooth=out["f_smooth"],
            P_smooth=out["P_smooth"],
            P_lag=out["P_lag"],
            w_eps=out["w_eps"],
            w_u=out["w_u"],
            log_w_eps=out["log_w_eps"],
            log_w_u=out["log_w_u"],
            Lambda_tilde=out["Lambda_tilde"],
            loglik=out["loglik"],
        )

    return out


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # Self-test dell'E-step, parametrico sulla struttura di fattore.
    #
    #     python -m em.em_e_step --spec fed_overlap
    #     python -m em.em_e_step --spec diag4
    #     python -m em.em_e_step --spec diag3
    #
    # I test numerici qui sotto (d_eps, d_u, pesi, limite gaussiano, ciclo ECM)
    # sono per loro natura indipendenti dalla struttura: leggono theta^(0), Y e
    # freq_list e verificano identita' algebriche. L'unica cosa che cambiava fra
    # le config era il PREAMBOLO, ora unificato nella fixture.
    # ─────────────────────────────────────────────────────────────────────────
    import pathlib
    import sys

    import pandas as pd

    _src_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from kalman import build_all_selection_matrices, run_kalman        # noqa: E402
    from dfm.selftest_fixture import (                                 # noqa: E402
        parse_spec_args, load_fixture, selftest_scratch,
    )

    _args = parse_spec_args(
        "em_e_step self-test — pesi dell'E-step e residui di Mahalanobis.")
    fx = load_fixture(_args.spec)

    print("=" * 64)
    print(fx.describe())
    print("=" * 64)

    # theta^(0) dalla fixture (stessa pipeline di em_initialization).
    theta = fx.theta0
    R = np.asarray(theta["R"])           # (M,)

    # Y standardizzato, NaN preservati — la rappresentazione su cui theta^(0)
    # e' calibrato.  `dates` resta separato perche' le diagnostiche sotto
    # citano mesi di calendario (recessioni NBER, aprile 2020).
    Y = fx.Y
    dates = fx.Y_std_df.index
    series_names = fx.ordered_cols
    freq_list = fx.freq_list
    T, M = Y.shape
    print(f"Y shape: T={T}, M={M}   "
          f"({sum(f == 'monthly' for f in freq_list)} monthly + "
          f"{sum(f == 'quarterly' for f in freq_list)} quarterly)   "
          f"[standardised, NaN preserved]")

    # ── 1. Forward+smoothing Kalman with all weights = 1 (Gaussian baseline) ─
    print("\nRunning forward + smoothing Kalman (Gaussian baseline) ...")
    ks = run_kalman(Y, theta, freq_list=freq_list)
    f_smooth     = ks["f_smooth"]      # (T, 5r)
    P_smooth     = ks["P_smooth"]      # (T, 5r, 5r)
    Lambda_tilde = ks["Lambda_tilde"]  # (M, 5r)
    print(f"  f_smooth.shape = {f_smooth.shape}")
    print(f"  P_smooth.shape = {P_smooth.shape}")
    print(f"  Lambda_tilde.shape = {Lambda_tilde.shape}")
    print(f"  loglik = {ks['loglik']:.2f}")

    # ── 2. Selection matrices ────────────────────────────────────────────────
    W_list = build_all_selection_matrices(Y)

    # ── 3. Compute d_eps ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("compute_d_eps")
    print("=" * 64)
    d_eps, m_obs = compute_d_eps(Y, f_smooth, P_smooth, Lambda_tilde, R, W_list)

    # ── 4. Shape / sanity assertions ─────────────────────────────────────────
    assert d_eps.shape == (T,), f"d_eps.shape = {d_eps.shape}, expected ({T},)"
    assert m_obs.shape == (T,), f"m_obs.shape = {m_obs.shape}, expected ({T},)"

    assert np.all(np.isfinite(d_eps)), "d_eps contains NaN/inf"
    assert np.all(d_eps >= -1e-10), (
        f"d_eps has negative values; min = {d_eps.min():.3e}"
    )
    # clip mild floating-point negative noise (none expected; defensive only)
    assert d_eps.min() >= 0.0, "d_eps min < 0 (should be exactly non-negative)"

    print(f"[OK] d_eps.shape = {d_eps.shape}   m_obs.shape = {m_obs.shape}")
    print(f"[OK] all d_eps finite, all d_eps >= 0   "
          f"(min = {d_eps.min():.3e}, max = {d_eps.max():.3e})")

    # m_obs must equal W_list[t].shape[0]
    for t in range(T):
        assert m_obs[t] == W_list[t].shape[0], (
            f"m_obs[{t}] = {m_obs[t]}, W_list[{t}].shape[0] = {W_list[t].shape[0]}"
        )
    print(f"[OK] m_obs[t] == W_list[t].shape[0] for all t")

    # Coerenza con la distribuzione di m_t.  Nei mesi di FINE TRIMESTRE
    # completamente osservati m_t == M; nei mesi NON di fine trimestre cadono
    # TUTTE le serie trimestrali, quindi m_t == M - n_quarterly.  Il numero di
    # trimestrali viene da freq_list (sul dataset 'final' sono tre).
    is_qend = dates.month.isin([3, 6, 9, 12])
    n_quarterly = sum(f == "quarterly" for f in freq_list)
    M_full = int(np.max(m_obs))          # massimo osservato = M a fine trimestre
    M_nq   = M_full - n_quarterly        # mesi non di fine trimestre
    n_qend_full  = int(np.sum((m_obs == M_full) & is_qend))
    n_nonqend_nq = int(np.sum((m_obs == M_nq) & ~is_qend))
    print(f"[OK] m_obs == {M_full} in {n_qend_full} mesi (fine trimestre, tutte le serie)")
    print(f"[OK] m_obs == {M_nq} in {n_nonqend_nq} mesi (non fine trimestre, "
          f"cadono le {n_quarterly} trimestrali)")
    assert n_qend_full > 0 and n_nonqend_nq > 0, (
        f"attesi mesi con m_t={M_full} (fine trimestre) e m_t={M_nq} "
        f"(non fine trimestre, M - {n_quarterly} trimestrali)"
    )

    # ── 5. Top-10 d_eps months: should fall on shock periods (2008-09, 2020) ─
    print("\n" + "=" * 64)
    print("Top-10 months by d_eps (highest Mahalanobis residual)")
    print("=" * 64)
    top10_idx = np.argsort(d_eps)[-10:][::-1]
    print(f"{'rank':>4}  {'date':>10}  {'m_t':>4}  {'d_eps':>12}")
    print(f"{'-'*4}  {'-'*10}  {'-'*4}  {'-'*12}")
    for rank, t in enumerate(top10_idx, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{m_obs[t]:>4}  {d_eps[t]:>12.4f}")

    print("""
INTERPRETATION — d_eps top months:
The highest idiosyncratic Mahalanobis residuals correspond to
well-known crisis episodes: the COVID crash (April 2020, the
single largest residual in the sample), the Global Financial
Crisis (October 2008, April 2009), and the 1990-91 recession.
A high d_eps means the observed data at that month are far from
what the factors imply — i.e. the observation is an outlier
relative to the factor structure. These are precisely the months
that the Student-t mechanism will down-weight.""")

    # ── 6. Summary statistics and the April 2020 value ───────────────────────
    print("\n" + "=" * 64)
    print("Summary statistics")
    print("=" * 64)
    print(f"  mean(d_eps)   = {d_eps.mean():>12.4f}")
    print(f"  median(d_eps) = {np.median(d_eps):>12.4f}")
    print(f"  std(d_eps)    = {d_eps.std():>12.4f}")
    print(f"  max(d_eps)    = {d_eps.max():>12.4f}   "
          f"at {dates[int(np.argmax(d_eps))].strftime('%Y-%m')}")
    print(f"  min(d_eps)    = {d_eps.min():>12.4f}   "
          f"at {dates[int(np.argmin(d_eps))].strftime('%Y-%m')}")

    # April 2020 — emblematic Covid outlier month
    apr2020_mask = (dates.year == 2020) & (dates.month == 4)
    if apr2020_mask.any():
        t_apr20 = int(np.where(apr2020_mask)[0][0])
        print(f"\n  d_eps[Apr-2020]   = {d_eps[t_apr20]:.4f}   "
              f"(m_t = {m_obs[t_apr20]})")
        rank_apr20 = int(np.sum(d_eps > d_eps[t_apr20])) + 1
        print(f"  Rank of Apr-2020 within T={T}: #{rank_apr20}")

    print("\n" + "=" * 64)
    print("compute_d_eps test passed.")
    print("=" * 64)

    # ── 7. Compute d_u  (factor-side Mahalanobis residual) ────────────────────
    print("\n" + "=" * 64)
    print("compute_d_u")
    print("=" * 64)
    P_lag = ks["P_lag"]                    # (T, 5r, 5r); P_lag[0] is zeros
    A     = theta["A"]                     # (r, r)  monthly transition
    Q     = theta["Q"]                     # (r, r)  monthly innovation cov
    r     = A.shape[0]
    print(f"  A.shape = {A.shape}   Q.shape = {Q.shape}   r = {r}")
    print(f"  cond(Q) = {np.linalg.cond(Q):.3f}")

    # Production form (Version B: explicit smoothed states at t and t-1 + P_lag).
    d_u = compute_d_u(f_smooth, P_smooth, P_lag, A, Q, r)
    # Alternative form (Version A: only internal lag-blocks of state at time t).
    d_u_A = _compute_d_u_internal_blocks(f_smooth, P_smooth, A, Q, r)

    # ── 8. Shape / sanity assertions ──────────────────────────────────────────
    assert d_u.shape == (T,), f"d_u.shape = {d_u.shape}, expected ({T},)"
    assert np.isnan(d_u[0]), "d_u[0] should be NaN (no f_{-1} in sample)"
    assert np.all(np.isfinite(d_u[1:])), "d_u has NaN/inf at t >= 1"
    assert np.all(d_u[1:] >= -1e-10), (
        f"d_u has negative values at t >= 1; min = {np.nanmin(d_u):.3e}"
    )
    print(f"[OK] d_u.shape = {d_u.shape}")
    print(f"[OK] d_u[0] = NaN  (no f_{{-1}} in sample; weight w^u_0 -> prior)")
    print(f"[OK] d_u[t>=1] all finite, all >= 0   "
          f"(min = {np.nanmin(d_u):.3e}, max = {np.nanmax(d_u):.3e})")

    # ── 9. CONSISTENCY CHECK: Version A (internal blocks) vs Version B ────────
    #     (state at t-1 + P_lag).  Confirms the augmented state-space
    #     correctly encodes the lag structure of the monthly factors.
    diff_AB = np.nanmax(np.abs(d_u - d_u_A))
    print(f"[OK] max|d_u_B - d_u_A| = {diff_AB:.3e}   "
          f"(internal-blocks vs P_lag-form)")
    assert diff_AB < 1e-8, (
        f"Version A vs B mismatch: max|diff| = {diff_AB:.3e} "
        f"(expected < 1e-8) -- augmented smoother may be inconsistent."
    )

    # ── 10. Top-10 d_u months ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Top-10 months by d_u (largest expected factor-innovation residual)")
    print("=" * 64)
    # argsort handles NaN by placing it last; restrict to t >= 1 for safety.
    top10_idx_u = np.argsort(d_u[1:])[-10:][::-1] + 1
    print(f"{'rank':>4}  {'date':>10}  {'d_u':>12}")
    print(f"{'-'*4}  {'-'*10}  {'-'*12}")
    for rank, t in enumerate(top10_idx_u, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  {d_u[t]:>12.4f}")

    print("""
INTERPRETATION — d_u top months:
With correctly standardised data, COVID April 2020 is the #1
outlier on the factor-side residual as well, and by a very large
margin over the second-highest d_u. This contrasts with d_eps,
where COVID also leads but by a smaller relative margin over the
runner-up. So COVID dominates BOTH channels, but it is RELATIVELY
even more extreme on the factor-innovation channel: the pandemic
shock was so abrupt that the latent factor moved in a way the VAR
dynamics could not anticipate at all.

The two residuals still capture distinct information. d_eps
flags months where the OBSERVATIONS disagree with the factor
structure (its top ranks mix COVID, the 2021 reopening, 2001,
and the GFC). d_u flags months where the FACTOR DYNAMICS are
surprising relative to the VAR prediction (its top ranks are
dominated by the acute 2020 phase and the 2008-2009 crisis).
The model carries two weight processes (w_eps, w_u) with two
degrees of freedom (nu_eps, nu_u) precisely so that these two
kinds of fat-tailed behaviour can be down-weighted
independently.""")

    # ── 11. Summary statistics + April 2020 value ────────────────────────────
    print("\n" + "=" * 64)
    print("Summary statistics  (over t >= 1)")
    print("=" * 64)
    print(f"  mean(d_u)   = {np.nanmean(d_u):>12.4f}")
    print(f"  median(d_u) = {np.nanmedian(d_u):>12.4f}")
    print(f"  std(d_u)    = {np.nanstd(d_u):>12.4f}")
    print(f"  max(d_u)    = {np.nanmax(d_u):>12.4f}   "
          f"at {dates[int(np.nanargmax(d_u))].strftime('%Y-%m')}")
    print(f"  min(d_u)    = {np.nanmin(d_u):>12.4f}   "
          f"at {dates[int(np.nanargmin(d_u))].strftime('%Y-%m')}")

    if apr2020_mask.any():
        t_apr20 = int(np.where(apr2020_mask)[0][0])
        print(f"\n  d_u[Apr-2020]   = {d_u[t_apr20]:.4f}")
        rank_apr20_u = int(np.sum(d_u[1:] > d_u[t_apr20])) + 1
        print(f"  Rank of Apr-2020 within T-1 = {T - 1}: #{rank_apr20_u}")

    # Theoretical anchor: under Gaussian innovations with weights = 1, the
    # expected value of d^u_t is E[d^u_t] = r (mean of a chi-squared(r)).
    # On correctly standardised data the sample mean sits close to r under a
    # well-specified model — a clean diagnostic of correct scaling.
    print(f"\n  Theoretical Gaussian baseline E[d^u_t] = r = {r}   "
          f"(sample mean is {np.nanmean(d_u):.3f})")

    print("\n" + "=" * 64)
    print("compute_d_u test passed.")
    print("=" * 64)

    # ── 12. Compute weights  (posterior mean and log-mean) ───────────────────
    print("\n" + "=" * 64)
    print("compute_weights")
    print("=" * 64)
    nu_eps = 10.0
    nu_u   = 10.0
    print(f"  nu_eps = {nu_eps}   nu_u = {nu_u}   r = {r}")

    weights = compute_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r)
    w_eps     = weights["w_eps"]
    w_u       = weights["w_u"]
    log_w_eps = weights["log_w_eps"]
    log_w_u   = weights["log_w_u"]

    # ── 13. Shape / sanity assertions ────────────────────────────────────────
    for name, arr in weights.items():
        assert arr.shape == (T,), f"{name}.shape = {arr.shape}, expected ({T},)"
    print(f"[OK] all four arrays have shape ({T},)")

    assert np.all(w_eps > 0), "w_eps has non-positive entries"
    assert np.all(w_u > 0),   "w_u has non-positive entries"
    assert np.all(np.isfinite(w_eps)), "w_eps has NaN/inf"
    assert np.all(np.isfinite(w_u)),   "w_u has NaN/inf"
    assert np.all(np.isfinite(log_w_eps)), "log_w_eps has NaN/inf"
    assert np.all(np.isfinite(log_w_u)),   "log_w_u has NaN/inf"
    print(f"[OK] w_eps, w_u strictly positive and finite everywhere")
    print(f"[OK] log_w_eps, log_w_u finite everywhere")

    # Boundary t = 0: w_u[0] = 1.0 (prior mean), log_w_u[0] = psi(nu_u/2) - log(nu_u/2)
    assert w_u[0] == 1.0, f"w_u[0] = {w_u[0]}, expected 1.0 (prior mean)"
    expected_log_w_u_0 = float(digamma(nu_u / 2.0) - np.log(nu_u / 2.0))
    assert abs(log_w_u[0] - expected_log_w_u_0) < 1e-12, (
        f"log_w_u[0] = {log_w_u[0]}, expected {expected_log_w_u_0} (prior log-mean)"
    )
    print(f"[OK] w_u[0]     = 1.0                (prior mean, no f_{{-1}})")
    print(f"[OK] log_w_u[0] = {log_w_u[0]:+.4f}   "
          f"(prior log-mean = psi(nu_u/2) - log(nu_u/2))")

    # ── 14. KEY OUTPUT: down-weighting of outlier months for w_eps ───────────
    print("\n" + "=" * 64)
    print("Down-weighting in action  (w_eps for high-d_eps months)")
    print("=" * 64)
    top5_idx_eps = np.argsort(d_eps)[-5:][::-1]
    print(f"{'rank':>4}  {'date':>10}  {'m_t':>4}  {'d_eps':>10}  {'w_eps':>10}")
    print(f"{'-'*4}  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*10}")
    for rank, t in enumerate(top5_idx_eps, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{m_obs[t]:>4}  {d_eps[t]:>10.4f}  {w_eps[t]:>10.4f}")

    # "Normal" months: take 5 closest to the median d_eps
    median_d_eps = float(np.median(d_eps))
    normal_idx_eps = np.argsort(np.abs(d_eps - median_d_eps))[:5]
    print(f"\n  Reference: median(d_eps) = {median_d_eps:.4f}")
    print(f"\n{'rank':>4}  {'date':>10}  {'m_t':>4}  {'d_eps':>10}  {'w_eps':>10}")
    print(f"{'-'*4}  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*10}")
    for rank, t in enumerate(normal_idx_eps, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{m_obs[t]:>4}  {d_eps[t]:>10.4f}  {w_eps[t]:>10.4f}")

    # April 2020: anchor against thesis-style expected value (nu_eps+m_t)/(nu_eps+d_eps)
    if apr2020_mask.any():
        t_apr20 = int(np.where(apr2020_mask)[0][0])
        expected_w_apr20 = (nu_eps + m_obs[t_apr20]) / (nu_eps + d_eps[t_apr20])
        print(f"\n  April-2020 sanity check:")
        print(f"    d_eps        = {d_eps[t_apr20]:.4f}")
        print(f"    m_t          = {m_obs[t_apr20]}")
        print(f"    expected w   = (nu_eps + m_t) / (nu_eps + d_eps) "
              f"= ({nu_eps:.0f} + {m_obs[t_apr20]}) / ({nu_eps:.0f} + {d_eps[t_apr20]:.4f}) "
              f"= {expected_w_apr20:.4f}")
        print(f"    computed w   = {w_eps[t_apr20]:.4f}")
        assert abs(w_eps[t_apr20] - expected_w_apr20) < 1e-12, "Apr-2020 mismatch"

    # ── 15. KEY OUTPUT: down-weighting of outlier months for w_u ─────────────
    print("\n" + "=" * 64)
    print("Down-weighting in action  (w_u for high-d_u months)")
    print("=" * 64)
    top5_idx_u = np.argsort(d_u[1:])[-5:][::-1] + 1
    print(f"{'rank':>4}  {'date':>10}  {'d_u':>10}  {'w_u':>10}")
    print(f"{'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    for rank, t in enumerate(top5_idx_u, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{d_u[t]:>10.4f}  {w_u[t]:>10.4f}")

    # "Normal" months for d_u
    median_d_u = float(np.nanmedian(d_u))
    # exclude t=0 (NaN) when selecting
    d_u_for_sort = np.where(np.isnan(d_u), np.inf, np.abs(d_u - median_d_u))
    normal_idx_u = np.argsort(d_u_for_sort)[:5]
    print(f"\n  Reference: median(d_u) = {median_d_u:.4f}")
    print(f"\n{'rank':>4}  {'date':>10}  {'d_u':>10}  {'w_u':>10}")
    print(f"{'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    for rank, t in enumerate(normal_idx_u, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{d_u[t]:>10.4f}  {w_u[t]:>10.4f}")

    # ── 16. Gaussian limit: nu -> large ==> w -> 1 ───────────────────────────
    print("\n" + "=" * 64)
    print("Gaussian limit  (nu = 10000 ==> weights -> 1, no down-weighting)")
    print("=" * 64)
    nu_huge = 10000.0
    weights_gauss = compute_weights(d_eps, d_u, m_obs, nu_huge, nu_huge, r)
    w_eps_g = weights_gauss["w_eps"]
    w_u_g   = weights_gauss["w_u"]
    max_dev_eps = float(np.max(np.abs(w_eps_g - 1.0)))
    max_dev_u   = float(np.max(np.abs(w_u_g[1:] - 1.0)))  # skip t=0 (set to 1)
    print(f"  max |w_eps - 1| = {max_dev_eps:.3e}")
    print(f"  max |w_u   - 1| = {max_dev_u:.3e}   (over t >= 1)")
    assert max_dev_eps < 0.10, (
        f"Gaussian limit failed: max|w_eps - 1| = {max_dev_eps:.3e}"
    )
    assert max_dev_u < 0.10, (
        f"Gaussian limit failed: max|w_u - 1| = {max_dev_u:.3e}"
    )
    print(f"[OK] Gaussian limit: all weights collapse to 1 as nu -> infinity")

    print("""
INTERPRETATION — down-weighting and the Gaussian limit:

(a) Idiosyncratic down-weighting (w_eps): outlier months receive
much smaller weights than normal months. COVID April 2020 gets a
far smaller w_eps than a typical month, so in the M-step it
contributes proportionally less to the parameter updates and a
single extreme data point does not dominate estimation. This is
the core robustness property of the Student-t DFM.

(b) Factor-side down-weighting (w_u): on correctly standardised
data the factor-side down-weighting of COVID is now very strong,
w_u well below 1, because COVID is the #1 factor-innovation
outlier (by a wide margin). Normal months can get w_u > 1
(up-weighting): when d_u < r, the month is more informative than
average and is weighted up. The mechanism is symmetric around
the typical residual.

(c) Gaussian limit: as nu -> infinity the weights collapse to 1
and no down-weighting occurs. With nu = 10000 the weights are
already essentially 1. This confirms that the Student-t DFM NESTS the
Gaussian DFM (Banbura-Modugno 2014) as the limiting case nu ->
infinity. The heavy-tailed model is therefore a strict
generalisation: it reduces to the Gaussian one when the data show
no excess kurtosis, and departs from it (down-weighting outliers)
when they do. This is the empirical motivation discussed in
Section 1 of the thesis (excess kurtosis in most of the series).""")

    # ── 17. Summary statistics for all four weight arrays ────────────────────
    print("\n" + "=" * 64)
    print("Summary statistics  (weights at nu_eps = nu_u = 10)")
    print("=" * 64)
    print(f"  w_eps      : mean = {w_eps.mean():>8.4f}   "
          f"min = {w_eps.min():>8.4f}   max = {w_eps.max():>8.4f}")
    print(f"  w_u        : mean = {w_u.mean():>8.4f}   "
          f"min = {w_u.min():>8.4f}   max = {w_u.max():>8.4f}")
    print(f"  log_w_eps  : mean = {log_w_eps.mean():>8.4f}   "
          f"min = {log_w_eps.min():>8.4f}   max = {log_w_eps.max():>8.4f}")
    print(f"  log_w_u    : mean = {log_w_u.mean():>8.4f}   "
          f"min = {log_w_u.min():>8.4f}   max = {log_w_u.max():>8.4f}")

    print("\n" + "=" * 64)
    print("compute_weights test passed.")
    print("=" * 64)

    # ── 18. ECM inner loop ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ecm_inner_loop  (Task 4 — ECM E-step inner loop)")
    print("=" * 64)
    print(f"  nu_eps = {float(theta['nu_eps']):.1f}   "
          f"nu_u = {float(theta['nu_u']):.1f}   "
          f"tol_inner = 1e-4   max_inner = 50")
    print()

    ecm = ecm_inner_loop(Y, theta, freq_list=freq_list, verbose=True)

    f_sm_ecm   = ecm["f_smooth"]
    P_sm_ecm   = ecm["P_smooth"]
    P_l_ecm    = ecm["P_lag"]
    w_e_ecm    = ecm["w_eps"]
    w_u_ecm    = ecm["w_u"]
    lwe_ecm    = ecm["log_w_eps"]
    lwu_ecm    = ecm["log_w_u"]
    n_iter     = ecm["n_inner_iter"]
    ll_ecm     = ecm["loglik"]
    conv_ecm   = ecm["converged"]

    # ── 19. Shape / sanity assertions ────────────────────────────────────────
    r_ecm = theta["A"].shape[0]
    aug   = 5 * r_ecm

    assert f_sm_ecm.shape  == (T, aug),        f"f_smooth.shape = {f_sm_ecm.shape}"
    assert P_sm_ecm.shape  == (T, aug, aug),   f"P_smooth.shape = {P_sm_ecm.shape}"
    assert P_l_ecm.shape   == (T, aug, aug),   f"P_lag.shape = {P_l_ecm.shape}"
    assert w_e_ecm.shape   == (T,),            f"w_eps.shape = {w_e_ecm.shape}"
    assert w_u_ecm.shape   == (T,),            f"w_u.shape = {w_u_ecm.shape}"
    assert lwe_ecm.shape   == (T,),            f"log_w_eps.shape = {lwe_ecm.shape}"
    assert lwu_ecm.shape   == (T,),            f"log_w_u.shape = {lwu_ecm.shape}"
    print(f"\n[OK] all output shapes correct  (T={T}, 5r={aug})")

    assert np.all(np.isfinite(f_sm_ecm)),  "f_smooth has NaN/inf"
    assert np.all(np.isfinite(P_sm_ecm)),  "P_smooth has NaN/inf"
    assert np.all(np.isfinite(P_l_ecm)),   "P_lag has NaN/inf"
    assert np.all(np.isfinite(w_e_ecm)),   "w_eps has NaN/inf"
    assert np.all(np.isfinite(w_u_ecm)),   "w_u has NaN/inf"
    assert np.all(np.isfinite(lwe_ecm)),   "log_w_eps has NaN/inf"
    assert np.all(np.isfinite(lwu_ecm)),   "log_w_u has NaN/inf"
    assert np.all(w_e_ecm > 0),  "w_eps has non-positive entries"
    assert np.all(w_u_ecm > 0),  "w_u has non-positive entries"
    print(f"[OK] no NaN/inf in factor moments or weights")
    print(f"[OK] w_eps > 0, w_u > 0 everywhere")

    assert conv_ecm, (
        f"ecm_inner_loop did not converge in {n_iter} iterations"
    )
    assert n_iter < 20, f"n_inner_iter = {n_iter} (expected < 20)"
    print(f"[OK] converged = {conv_ecm}   n_inner_iter = {n_iter}  "
          f"(< 20, thesis expects 3-10)")
    print(f"[OK] loglik at convergence = {ll_ecm:.2f}")

    # ── 20. Loglik evolution: k=1 (Gaussian) vs converged ────────────────────
    print("\n" + "=" * 64)
    print("Log-likelihood: Gaussian baseline (k=1) vs converged")
    print("=" * 64)
    # k=1 loglik: run Kalman with all weights = 1 (the Gaussian baseline)
    from kalman import run_kalman as _rk
    ks_gauss = _rk(Y, theta, freq_list=freq_list)
    ll_gauss = ks_gauss["loglik"]
    print(f"  loglik  k=1 (all weights=1, Gaussian) : {ll_gauss:.2f}")
    print(f"  loglik  converged (Student-t weights)  : {ll_ecm:.2f}")
    print(f"  delta loglik                           : {ll_ecm - ll_gauss:+.2f}")

    # ── 21. Comparison: w_eps[Apr-2020] at k=1 vs converged ─────────────────
    print("\n" + "=" * 64)
    print("w_eps[Apr-2020]: k=1 (Gaussian factors) vs converged")
    print("=" * 64)
    if apr2020_mask.any():
        t_apr20 = int(np.where(apr2020_mask)[0][0])
        # At k=1 the weights are computed from purely-Gaussian smoothed moments
        # (all w=1 in the Kalman); these are the same as those computed in the
        # earlier compute_weights self-test with nu_eps = theta["nu_eps"].
        nu_e_theta = float(theta["nu_eps"])
        nu_u_theta = float(theta["nu_u"])
        wts_k1 = compute_weights(d_eps, d_u, m_obs, nu_e_theta, nu_u_theta, r)
        w_eps_k1 = wts_k1["w_eps"]
        print(f"  w_eps[Apr-2020] at k=1      = {w_eps_k1[t_apr20]:.4f}  "
              f"(Gaussian smoothed factors, nu_eps={nu_e_theta:.1f})")
        print(f"  w_eps[Apr-2020] converged   = {w_e_ecm[t_apr20]:.4f}")
        print(f"  change                      = {w_e_ecm[t_apr20] - w_eps_k1[t_apr20]:+.4f}")
    else:
        print("  April 2020 not in sample.")

    # ── 22. Five most down-weighted months at convergence (w_eps) ────────────
    print("\n" + "=" * 64)
    print("Five most down-weighted months at convergence (lowest w_eps)")
    print("=" * 64)
    bot5_eps = np.argsort(w_e_ecm)[:5]
    print(f"{'rank':>4}  {'date':>10}  {'w_eps':>10}  {'d_eps':>10}")
    print(f"{'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    # Recompute d_eps from the converged Kalman run (already in ks at this point)
    ks_conv = _rk(Y, theta, w_u=w_u_ecm, w_eps=w_e_ecm, freq_list=freq_list)
    d_eps_conv, _ = compute_d_eps(
        Y, ks_conv["f_smooth"], ks_conv["P_smooth"],
        ks_conv["Lambda_tilde"], R, W_list
    )
    for rank, t in enumerate(bot5_eps, start=1):
        print(f"{rank:>4}  {dates[t].strftime('%Y-%m'):>10}  "
              f"{w_e_ecm[t]:>10.4f}  {d_eps_conv[t]:>10.4f}")

    print("\n" + "=" * 64)
    print("ecm_inner_loop test passed.")
    print("=" * 64)

    # ── 23. run_e_step  (Task 5 — high-level E-step wrapper) ─────────────────
    print("\n" + "=" * 64)
    print("run_e_step  (Task 5 — high-level E-step wrapper)")
    print("=" * 64)
    print(f"  Wrapping ecm_inner_loop and packaging M-step inputs.")
    print(f"  Calling with verbose=False (compact diagnostics only).\n")

    estep = run_e_step(Y, theta, freq_list=freq_list, verbose=False)

    # ── 24. Output dictionary: keys, shapes, types ───────────────────────────
    expected_keys = {
        "f_smooth", "P_smooth", "P_lag",
        "w_eps", "w_u", "log_w_eps", "log_w_u",
        "Lambda_tilde", "m_obs_eps",
        "loglik",
        "n_inner_iter", "converged", "used_best_iterate",
        "T", "M", "r",
    }
    got_keys = set(estep.keys())
    assert got_keys == expected_keys, (
        f"key mismatch: missing {expected_keys - got_keys}, "
        f"extra {got_keys - expected_keys}"
    )
    print(f"[OK] returned dict has the expected {len(expected_keys)} keys")

    r_e = estep["r"]
    aug_e = 5 * r_e
    assert estep["f_smooth"].shape     == (T, aug_e)
    assert estep["P_smooth"].shape     == (T, aug_e, aug_e)
    assert estep["P_lag"].shape        == (T, aug_e, aug_e)
    assert estep["w_eps"].shape        == (T,)
    assert estep["w_u"].shape          == (T,)
    assert estep["log_w_eps"].shape    == (T,)
    assert estep["log_w_u"].shape      == (T,)
    assert estep["Lambda_tilde"].shape == (M, aug_e)
    assert isinstance(estep["loglik"], float)
    assert isinstance(estep["n_inner_iter"], int)
    assert isinstance(estep["converged"], bool)
    assert estep["T"] == T and estep["M"] == M and estep["r"] == r_e
    print(f"[OK] all output shapes correct  (T={T}, M={M}, 5r={aug_e})")
    print(f"[OK] scalar types correct  "
          f"(loglik={type(estep['loglik']).__name__}, "
          f"n_inner_iter={type(estep['n_inner_iter']).__name__}, "
          f"converged={type(estep['converged']).__name__})")

    # ── 25. Finiteness and positivity ────────────────────────────────────────
    for name in ("f_smooth", "P_smooth", "P_lag",
                 "w_eps", "w_u", "log_w_eps", "log_w_u", "Lambda_tilde"):
        assert np.all(np.isfinite(estep[name])), f"{name} has NaN/inf"
    assert np.all(estep["w_eps"] > 0), "w_eps has non-positive entries"
    assert np.all(estep["w_u"]   > 0), "w_u has non-positive entries"
    print(f"[OK] no NaN/inf in any array output")
    print(f"[OK] w_eps > 0 and w_u > 0 everywhere")

    # ── 26. Equivalence with direct call to ecm_inner_loop ───────────────────
    # run_e_step should be a pure repackaging of ecm_inner_loop — same numbers.
    ecm_direct = ecm_inner_loop(Y, theta, freq_list=freq_list, verbose=False)
    for name in ("f_smooth", "P_smooth", "P_lag",
                 "w_eps", "w_u", "log_w_eps", "log_w_u", "Lambda_tilde"):
        diff = float(np.max(np.abs(estep[name] - ecm_direct[name])))
        assert diff < 1e-12, (
            f"{name}: run_e_step != ecm_inner_loop  (max|diff| = {diff:.3e})"
        )
    assert abs(estep["loglik"] - ecm_direct["loglik"]) < 1e-9, "loglik mismatch"
    assert estep["converged"] == ecm_direct["converged"], "converged flag mismatch"
    print(f"[OK] run_e_step output identical to ecm_inner_loop  "
          f"(max array diff < 1e-12, loglik diff < 1e-9)")

    # ── 27. Lambda_tilde structure (consistency with kalman.build_Lambda_tilde) ─
    # Re-import build_Lambda_tilde explicitly and check that the Lambda_tilde
    # carried back by run_e_step matches the one built directly from theta.
    from kalman import build_Lambda_tilde as _build_Lambda_tilde
    Lt_direct = _build_Lambda_tilde(theta["Lambda"], freq_list)
    diff_L = float(np.max(np.abs(estep["Lambda_tilde"] - Lt_direct)))
    assert diff_L < 1e-12, (
        f"Lambda_tilde mismatch with build_Lambda_tilde: max|diff| = {diff_L:.3e}"
    )
    print(f"[OK] Lambda_tilde matches build_Lambda_tilde(theta['Lambda'], freq_list)  "
          f"(max diff {diff_L:.2e})")

    # ── 28. Persistence: save -> load -> compare ─────────────────────────────
    import tempfile
    # cartella temporanea: il round-trip su disco non deve lasciare file
    # negli artefatti veri (vedi selftest_scratch).
    save_dir = pathlib.Path(selftest_scratch(f"em_e_step_{_args.spec}"))
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=str(save_dir), delete=False
    ) as tmp:
        save_path = tmp.name
    try:
        _ = run_e_step(Y, theta, freq_list=freq_list, save_path=save_path)
        loaded = np.load(save_path)
        for name in ("f_smooth", "P_smooth", "P_lag",
                     "w_eps", "w_u", "log_w_eps", "log_w_u", "Lambda_tilde"):
            diff = float(np.max(np.abs(loaded[name] - estep[name])))
            assert diff < 1e-12, f"{name} round-trip mismatch: {diff:.3e}"
        ll_loaded = float(loaded["loglik"])
        assert abs(ll_loaded - estep["loglik"]) < 1e-9, "loglik round-trip mismatch"
        loaded.close()
        print(f"[OK] save -> load round-trip identical for all 8 arrays + loglik")
    finally:
        import os as _os
        if _os.path.exists(save_path):
            _os.remove(save_path)

    # ── 29. M-step input summary  (concise diagnostic for the caller) ────────
    print("\n" + "=" * 64)
    print("E-step output  (inputs to the M-step)")
    print("=" * 64)
    print(f"  Factor moments  : f_smooth {estep['f_smooth'].shape}, "
          f"P_smooth {estep['P_smooth'].shape}, P_lag {estep['P_lag'].shape}")
    print(f"  Weight moments  : w_eps {estep['w_eps'].shape}, w_u {estep['w_u'].shape}, "
          f"log_w_eps {estep['log_w_eps'].shape}, log_w_u {estep['log_w_u'].shape}")
    print(f"  Loading matrix  : Lambda_tilde {estep['Lambda_tilde'].shape}")
    print(f"  Outer-EM signal : loglik = {estep['loglik']:.2f}")
    print(f"  Inner loop      : converged = {estep['converged']}, "
          f"n_inner_iter = {estep['n_inner_iter']}")
    print(f"  Weight ranges   : w_eps in [{estep['w_eps'].min():.3f}, "
          f"{estep['w_eps'].max():.3f}], "
          f"w_u in [{estep['w_u'].min():.3f}, {estep['w_u'].max():.3f}]")

    print("\n" + "=" * 64)
    print("run_e_step test passed.")
    print("=" * 64)
