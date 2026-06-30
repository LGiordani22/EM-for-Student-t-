"""
src/mcmc/sample_states.py
=========================

Gibbs step (a): draw the latent factor path from its full conditional
``p(f_{0:T-1} | Y, theta, w)`` by Forward-Filtering Backward-Sampling (FFBS).

Reuse vs new
------------
* **Forward pass — reused as-is.**  :func:`kalman.kalman_filter` produces the
  filtered moments ``(f_filt[t], P_filt[t]) = p(f_aug[t] | Y_{1:t})`` for the
  augmented state, fed with the time-varying weights ``w_u, w_eps`` exactly as
  the EM E-step does.  Nothing is re-derived here.
* **Backward sampling — the only new ingredient.**  A Carter-Kohn backward
  sweep that correctly handles the **singular companion** state-space.

The singular-companion subtlety (the crux)
------------------------------------------
The augmented state is ``f_aug[t] = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4})``
and the transition ``f_aug[t+1] = A_tilde f_aug[t] + u`` injects fresh noise
**only** into the top r-block (``Q_tilde`` has rank r).  The four lag-blocks of
``f_aug[t+1]`` are *deterministic copies* of ``f_aug[t]``:

    f_aug[t+1][r:2r] = f_t,  f_aug[t+1][2r:3r] = f_{t-1},
    f_aug[t+1][3r:4r] = f_{t-2},  f_aug[t+1][4r:5r] = f_{t-3}.

So conditioning ``f_aug[t]`` on the *full* ``f_aug[t+1]`` (the textbook
Carter-Kohn kernel — ``f_aug[t+1]`` d-separates ``f_aug[t]`` from the future,
which is essential here because the **quarterly** rows make ``y_{t+1}`` load on
``f_t``) pins the first four blocks of ``f_aug[t]`` exactly and leaves only the
**deepest lag** ``f_{t-4}`` (the 5th block) free.  Conditioning on the next head
``f_{t+1}`` alone would be WRONG: ``y_{t+1}`` carries information about ``f_t``
that ``f_{t+1}`` does not summarise.

Hence the correct, non-degenerate FFBS draws, at backward step ``t``, the single
new head ``f_{t-4}`` from its Gaussian conditional under
``N(f_filt[t], P_filt[t])`` given the already-sampled blocks
``(f_t, f_{t-1}, f_{t-2}, f_{t-3})``; the boundary step ``t = T-1`` draws the
whole augmented vector at once (it introduces the five most recent heads).  We
then **reconstruct** the augmented path by stacking the sampled head sequence,
which makes the internal lag identity ``f_aug[t][r:2r] == f_aug[t-1][0:r]`` hold
*exactly* — so every downstream reader (``compute_weighted_moments`` reads the
internal lag block; ``realized_deflated_d_u`` reads consecutive heads) sees a
mutually consistent path.

Config-aware
------------
``r`` is read from ``theta["A"]`` and the augmented dimension is ``5r``; nothing
about the number of series, factors or the mixed-frequency structure is
hard-coded.  ``freq_list`` is passed straight through to the filter.
"""

from __future__ import annotations

import numpy as np

from kalman import (
    kalman_filter,
    kalman_predict,
    kalman_update,
    build_A_tilde,
    build_Q_tilde,
    build_R_tilde,
    build_Lambda_tilde,
    build_all_selection_matrices,
)


def _chol_sample(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator,
                 jitter: float = 1e-10) -> np.ndarray:
    """Draw one sample from N(mean, cov), robust to mild non-PD ``cov``.

    Symmetrises ``cov`` and adds escalating jitter until the Cholesky factor
    exists.  Falls back to an eigen-decomposition (clipping tiny negative
    eigenvalues) only if jittered Cholesky keeps failing — this never happens
    for genuine filtered covariances but guards the ragged edge.
    """
    cov = 0.5 * (cov + cov.T)
    d = cov.shape[0]
    eye = np.eye(d)
    s = jitter
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + s * eye)
            return mean + L @ rng.standard_normal(d)
        except np.linalg.LinAlgError:
            s = max(s * 10.0, 1e-12)
    # Eigen fallback: clip negatives, reconstruct a PSD square root.
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    L = vecs @ np.diag(np.sqrt(vals))
    return mean + L @ rng.standard_normal(d)


def _build_augmented(F: np.ndarray, r: int) -> np.ndarray:
    """Stack the head sequence ``F`` (T, r) into the augmented path (T, 5r).

    ``f_aug[t] = (f_t, f_{t-1}, f_{t-2}, f_{t-3}, f_{t-4})`` with zeros for the
    pre-sample lags (``t - l < 0``) — the same convention used by the simulator
    and the smoother, so the moments / residual helpers consume it unchanged.
    """
    T = F.shape[0]
    f_aug = np.zeros((T, 5 * r))
    for l in range(5):
        if l == 0:
            f_aug[:, 0:r] = F
        else:
            f_aug[l:, l * r:(l + 1) * r] = F[: T - l]
    return f_aug


def forward_filter_combined(
    Y: np.ndarray,
    theta: dict,
    g_u: np.ndarray,
    g_eps: np.ndarray,
    freq_list: list[str],
) -> dict:
    r"""
    Forward Kalman filter fed *combined precisions*, for the SV sampler.

    Identical to :func:`kalman.kalman_filter` except that the noise covariances
    are built from the combined precision ``g_t = w_t / h_t`` rather than a
    scalar tail weight, and the idiosyncratic precision is **per series**:

        Q_tilde_t = build_Q_tilde(Q, g_u[t])            ->  Q / g^u_t = h^u_t Q / w^u_t
        R_tilde_t = build_R_tilde(R, g_eps[t])          ->  diag(R_i / g^eps_{i,t})
                                                          = diag(h^eps_{i,t} r_i / w^eps_t)

    ``build_R_tilde(R, g_eps[t])`` with ``g_eps[t]`` a length-M vector returns
    ``diag(R) / g_eps[t]`` whose diagonal is ``R_i / g^eps_{i,t}`` (off-diagonals
    stay zero), so the per-series deflation comes for free.  Only the Kalman
    *loop* is re-implemented here (the stock ``kalman_filter`` casts the weight
    to a scalar); ``kalman_predict`` / ``kalman_update`` / the matrix builders are
    reused verbatim.

    Parameters
    ----------
    g_u : (T,)            combined factor-side precision ``w^u_t / h^u_t``.
    g_eps : (T, M)        combined idiosyncratic precision ``w^eps_t / h^eps_{i,t}``.

    Returns the same dict layout as :func:`kalman.kalman_filter`.
    """
    A = np.asarray(theta["A"]); Q = np.asarray(theta["Q"])
    Lambda = np.asarray(theta["Lambda"]); R = np.asarray(theta["R"]).ravel()
    Sigma_0 = np.asarray(theta["Sigma_0"])
    r = A.shape[0]
    dim = 5 * r
    T, M = Y.shape

    A_tilde = build_A_tilde(A)
    Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)
    W_list = build_all_selection_matrices(Y)

    f_pred_arr = np.zeros((T, dim)); P_pred_arr = np.zeros((T, dim, dim))
    f_filt_arr = np.zeros((T, dim)); P_filt_arr = np.zeros((T, dim, dim))
    K_list = [np.zeros((dim, 0))] * T
    WL_list = [np.zeros((0, dim))] * T
    loglik_total = 0.0

    f_prev = np.zeros(dim); P_prev = Sigma_0.copy()
    for t in range(T):
        Q_tilde_t = build_Q_tilde(Q, float(g_u[t]))
        R_tilde_t = build_R_tilde(R, g_eps[t])           # per-series vector
        f_p, P_p = kalman_predict(f_prev, P_prev, A_tilde, Q_tilde_t)
        f_pred_arr[t] = f_p; P_pred_arr[t] = P_p
        upd = kalman_update(f_p, P_p, Y[t], W_list[t], Lambda_tilde, R_tilde_t)
        f_filt_arr[t] = upd["f_filt"]; P_filt_arr[t] = upd["P_filt"]
        K_list[t] = upd["K"]; WL_list[t] = upd["WL"]; loglik_total += upd["loglik_t"]
        f_prev = f_filt_arr[t]; P_prev = P_filt_arr[t]

    return {"f_pred": f_pred_arr, "P_pred": P_pred_arr, "f_filt": f_filt_arr,
            "P_filt": P_filt_arr, "K_list": K_list, "WL_list": WL_list,
            "loglik": loglik_total}


def ffbs_sample_states(
    Y: np.ndarray,
    theta: dict,
    w_u: np.ndarray,
    w_eps: np.ndarray,
    freq_list: list[str],
    rng: np.random.Generator,
    *,
    jitter: float = 1e-10,
    filter_out: dict | None = None,
    h_u: np.ndarray | None = None,
    h_eps: np.ndarray | None = None,
) -> dict:
    r"""
    Draw a factor path from ``p(f | Y, theta, w_u, w_eps)`` via FFBS.

    Parameters
    ----------
    Y : (T, M)            observation panel (NaN = missing).
    theta : dict          model parameters; ``A`` (r, r), ``Q``, ``Lambda``,
                          ``R``, ``Sigma_0`` are read by :func:`kalman_filter`.
    w_u, w_eps : (T,)     current Student-t weights (the combined precision
                          ``g_t = w_t / h_t`` when SV is active; with
                          ``h == 1`` these are the plain weights).
    freq_list : list[str] per-column frequency, passed to the filter.
    rng : np.random.Generator
    jitter : float        diagonal regularisation for the Cholesky draws.
    filter_out : dict or None
        Optional pre-computed :func:`kalman_filter` output (avoids re-running the
        forward pass when the caller already has it).

    Returns
    -------
    dict with
        ``f_aug`` : (T, 5r)   sampled augmented state path (internally consistent).
        ``F``     : (T, r)    contemporaneous head sequence f_{0:T-1}.
        ``loglik``: float     forward-pass marginal log-likelihood (diagnostic).
    """
    A = np.asarray(theta["A"])
    r = A.shape[0]

    if filter_out is None:
        if h_u is None and h_eps is None:
            # Passo-1 path: scalar tail weights, stock filter (bit-identical).
            filter_out = kalman_filter(Y, theta, w_u=w_u, w_eps=w_eps, freq_list=freq_list)
        else:
            # SV path: combined precision g = w / h (per-series for idiosyncratic).
            T, M = Y.shape
            g_u = np.asarray(w_u, float) if h_u is None else np.asarray(w_u, float) / np.asarray(h_u, float)
            if h_eps is None:
                g_eps = np.broadcast_to(np.asarray(w_eps, float)[:, None], (T, M)).copy()
            else:
                g_eps = np.asarray(w_eps, float)[:, None] / np.asarray(h_eps, float)
            filter_out = forward_filter_combined(Y, theta, g_u, g_eps, freq_list)
    f_filt = filter_out["f_filt"]      # (T, 5r)
    P_filt = filter_out["P_filt"]      # (T, 5r, 5r)
    T = f_filt.shape[0]

    F = np.zeros((T, r))

    # ── Boundary t = T-1: sample the whole augmented vector (5 newest heads) ──
    aug_last = _chol_sample(f_filt[T - 1], P_filt[T - 1], rng, jitter)
    for l in range(5):
        if T - 1 - l >= 0:
            F[T - 1 - l] = aug_last[l * r:(l + 1) * r]

    # ── Backward sweep: at step t draw the deepest lag f_{t-4} = head s=t-4 ──
    # conditioned on the already-sampled (f_t, f_{t-1}, f_{t-2}, f_{t-3}).
    a_idx = np.arange(0, 4 * r)        # blocks 1..4 (pinned)
    b_idx = np.arange(4 * r, 5 * r)    # block 5 (free, = f_{t-4})
    for t in range(T - 2, -1, -1):
        s = t - 4
        if s < 0:
            break  # remaining initial lags are pre-sample (zeros by convention)

        mu = f_filt[t]
        Sig = P_filt[t]
        Saa = Sig[np.ix_(a_idx, a_idx)]      # (4r, 4r)
        Sbb = Sig[np.ix_(b_idx, b_idx)]      # (r, r)
        Sba = Sig[np.ix_(b_idx, a_idx)]      # (r, 4r)

        # pinned value = already-sampled (f_t, f_{t-1}, f_{t-2}, f_{t-3})
        a_val = np.concatenate([F[t], F[t - 1], F[t - 2], F[t - 3]])

        # Gaussian conditional of block 5 given blocks 1..4.
        Saa_reg = Saa + jitter * np.eye(4 * r)
        gain = np.linalg.solve(Saa_reg, Sba.T).T          # (r, 4r) = Sba Saa^{-1}
        cond_mean = mu[b_idx] + gain @ (a_val - mu[a_idx])
        cond_cov = Sbb - gain @ Sba.T
        F[s] = _chol_sample(cond_mean, cond_cov, rng, jitter)

    f_aug = _build_augmented(F, r)
    return {"f_aug": f_aug, "F": F, "loglik": float(filter_out["loglik"])}
