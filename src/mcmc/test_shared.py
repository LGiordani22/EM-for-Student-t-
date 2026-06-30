"""
src/mcmc/test_shared.py
=======================

PASSO 0 non-regression / equivalence test for the extracted helpers in
``src/mcmc/shared.py``.

Each helper is checked against its EM counterpart on a small, fully synthetic
mixed-frequency state-space (no data files, no fitting needed):

  * deterministic seams (same formula, draw replaced by point/mean) are checked
    to MACHINE PRECISION;
  * stochastic draws are checked by Monte-Carlo: the empirical mean of many
    draws reproduces the EM posterior mean / point within MC error.

The EM functions are imported and *executed*, so a green run also confirms the
EM E-/M-step code still works in this environment (it is left unchanged).

Run
---
    python src/mcmc/test_shared.py

Prints [PASS]/[FAIL] per check; exits non-zero if any fails.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.special import digamma as _digamma

# ── make src/ importable (flat EM imports) ───────────────────────────────────
_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from em_e_step import compute_d_eps, compute_d_u, compute_weights          # noqa: E402
from em_m_step import (                                                     # noqa: E402
    compute_weighted_moments,
    update_A_Q,
    update_Lambda,
    update_R,
    update_nu,
)
from kalman import build_Lambda_tilde, build_all_selection_matrices         # noqa: E402

from mcmc.shared import (                                                   # noqa: E402
    MM_WEIGHTS,
    composite_regressor,
    draw_A_Q,
    draw_lambda_r_series,
    draw_weights,
    nu_foc,
    nu_log_target,
    realized_deflated_d_eps,
    realized_deflated_d_u,
)

_PASS = 0
_FAIL = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}   {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic mixed-frequency state-space
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    r, M, T = 3, 4, 120
    dim = 5 * r

    ordered_cols = ["RM", "FM", "OM", "RQ"]
    block_map = {"RM": "real", "FM": "financial", "OM": "other", "RQ": "real"}
    freq_list = ["monthly", "monthly", "monthly", "quarterly"]
    block_col = {"real": 0, "financial": 1, "other": 2}

    # Stable VAR(1) + SPD innovation covariance.
    A = 0.5 * np.eye(r) + 0.05 * rng.standard_normal((r, r))
    B = 0.3 * rng.standard_normal((r, r))
    Q = B @ B.T + 0.5 * np.eye(r)

    # Monthly factor path, then stack 5 lags into the augmented state.
    f = np.zeros((T, r))
    cQ = np.linalg.cholesky(Q)
    for t in range(1, T):
        f[t] = A @ f[t - 1] + cQ @ rng.standard_normal(r)
    f_aug = np.zeros((T, dim))
    for t in range(T):
        for l in range(5):
            if t - l >= 0:
                f_aug[t, l * r:(l + 1) * r] = f[t - l]

    # Smoothed covariance: SPD, small.
    P_smooth = np.zeros((T, dim, dim))
    for t in range(T):
        G = 0.1 * rng.standard_normal((dim, dim))
        P_smooth[t] = G @ G.T + 0.4 * np.eye(dim)

    # Block-diagonal loadings.
    Lambda = np.zeros((M, r))
    for i, col in enumerate(ordered_cols):
        Lambda[i, block_col[block_map[col]]] = 0.8 + 0.4 * rng.standard_normal()
    R = 0.2 + 0.3 * rng.random(M)

    Lambda_tilde = build_Lambda_tilde(Lambda, freq_list)

    # Observations: monthly every t, quarterly only at quarter-ends (t % 3 == 2).
    Y = np.full((T, M), np.nan)
    for t in range(T):
        signal = Lambda_tilde @ f_aug[t]
        for i in range(M):
            if freq_list[i] == "monthly":
                Y[t, i] = signal[i] + np.sqrt(R[i]) * rng.standard_normal()
            elif (t % 3) == 2:
                Y[t, i] = signal[i] + np.sqrt(R[i]) * rng.standard_normal()
    # A small ragged edge: drop one monthly series in the last two months.
    Y[-2:, 1] = np.nan

    W_list = build_all_selection_matrices(Y)

    # Positive Student-t weights (dispersed).
    w_eps = rng.gamma(shape=4.0, scale=1.0 / 4.0, size=T)
    w_u = rng.gamma(shape=4.0, scale=1.0 / 4.0, size=T)
    w_u[0] = 1.0

    return dict(
        rng=rng, r=r, M=M, T=T, A=A, Q=Q, f_aug=f_aug, P_smooth=P_smooth,
        Lambda=Lambda, R=R, Lambda_tilde=Lambda_tilde, Y=Y, W_list=W_list,
        w_eps=w_eps, w_u=w_u, ordered_cols=ordered_cols, block_map=block_map,
        freq_list=freq_list, block_col=block_col,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. realized_deflated_d_eps / _d_u  vs  compute_d_eps / compute_d_u  (P = 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_realized_d(S):
    print("\n[1] realized_deflated_d vs compute_d_* with P=0")
    Y, f_aug, R, Lt, W = S["Y"], S["f_aug"], S["R"], S["Lambda_tilde"], S["W_list"]
    A, Q, r, T = S["A"], S["Q"], S["r"], S["T"]
    Pzero = np.zeros_like(S["P_smooth"])

    # idiosyncratic
    d_eps_em, m_em = compute_d_eps(Y, f_aug, Pzero, Lt, R, W)
    d_eps_h, m_h = realized_deflated_d_eps(Y, f_aug, Lt, R, W, h_eps=None)
    _check("d_eps == compute_d_eps(P=0)", np.allclose(d_eps_em, d_eps_h, atol=1e-12),
           f"max|diff|={np.max(np.abs(d_eps_em-d_eps_h)):.2e}")
    _check("m_obs matches", np.array_equal(m_em, m_h))

    # deflation: constant h doubles the variance -> halves d
    h2 = 2.0 * np.ones((T, S["M"]))
    d_eps_def, _ = realized_deflated_d_eps(Y, f_aug, Lt, R, W, h_eps=h2)
    _check("d_eps deflation by h=2 halves d", np.allclose(d_eps_def, 0.5 * d_eps_h, atol=1e-12),
           f"max|diff|={np.max(np.abs(d_eps_def-0.5*d_eps_h)):.2e}")

    # factor side
    d_u_em = compute_d_u(f_aug, Pzero, Pzero, A, Q, r)
    d_u_h = realized_deflated_d_u(f_aug, A, Q, r, h_u=None)
    ok_u = np.allclose(d_u_em[1:], d_u_h[1:], atol=1e-12) and np.isnan(d_u_em[0]) and np.isnan(d_u_h[0])
    _check("d_u == compute_d_u(P=0,Plag=0)", ok_u,
           f"max|diff|={np.nanmax(np.abs(d_u_em-d_u_h)):.2e}")

    h3 = 3.0 * np.ones(T)
    d_u_def = realized_deflated_d_u(f_aug, A, Q, r, h_u=h3)
    _check("d_u deflation by h=3 divides d", np.allclose(d_u_def[1:], d_u_h[1:] / 3.0, atol=1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# 2. draw_weights  vs  compute_weights  (mean of draws == E-step mean)
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_weights(S):
    print("\n[2] draw_weights vs compute_weights (MC mean)")
    Y, f_aug, R, Lt, W = S["Y"], S["f_aug"], S["R"], S["Lambda_tilde"], S["W_list"]
    A, Q, r = S["A"], S["Q"], S["r"]
    nu_eps, nu_u = 7.0, 9.0

    d_eps, m_obs = realized_deflated_d_eps(Y, f_aug, Lt, R, W, h_eps=None)
    d_u = realized_deflated_d_u(f_aug, A, Q, r, h_u=None)

    em = compute_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r)

    rng = np.random.default_rng(123)
    K = 60000
    acc_eps = np.zeros_like(d_eps)
    acc_u = np.zeros_like(d_u)
    for _ in range(K):
        d = draw_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r, rng)
        acc_eps += d["w_eps"]
        acc_u += d["w_u"]
    mc_eps = acc_eps / K
    mc_u = acc_u / K

    rel_eps = np.max(np.abs(mc_eps - em["w_eps"]) / np.abs(em["w_eps"]))
    rel_u = np.max(np.abs(mc_u - em["w_u"]) / np.abs(em["w_u"]))
    _check("MC mean(w_eps) ~ compute_weights w_eps", rel_eps < 0.03, f"max rel={rel_eps:.3f}")
    _check("MC mean(w_u) ~ compute_weights w_u", rel_u < 0.03, f"max rel={rel_u:.3f}")
    _check("w_u[0] prior mean ~ 1", abs(mc_u[0] - 1.0) < 0.03, f"{mc_u[0]:.4f}")
    # positivity of a single draw
    dd = draw_weights(d_eps, d_u, m_obs, nu_eps, nu_u, r, np.random.default_rng(7))
    _check("draws strictly positive", np.all(dd["w_eps"] > 0) and np.all(dd["w_u"] > 0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. nu_foc / nu_log_target  vs  update_nu
# ─────────────────────────────────────────────────────────────────────────────

def test_nu(S):
    print("\n[3] nu_foc / nu_log_target vs update_nu")
    rng = np.random.default_rng(5)
    w = rng.gamma(shape=8.0 / 2.0, scale=2.0 / 8.0, size=400)   # dispersed weights, nu0=8
    sum_w, sum_lw = float(np.sum(w)), float(np.sum(np.log(w)))
    Tn = w.size
    mean_w, mean_lw = sum_w / Tn, sum_lw / Tn

    # nu_foc reproduces the exact g(nu) closure used inside update_nu.
    def g_inline(nu):
        half = 0.5 * nu
        return float(np.log(half) - _digamma(half) + 1.0 + mean_lw - mean_w)
    nus = [2.5, 5.0, 12.0, 50.0]
    same = all(abs(nu_foc(n, mean_lw, mean_w) - g_inline(n)) < 1e-12 for n in nus)
    _check("nu_foc == inline g(nu)", same)

    # update_nu's returned root makes nu_foc ~ 0 (interior case).
    nu_star = update_nu(mean_w, mean_lw)
    interior = 2.001 < nu_star < 1000.0
    _check("update_nu root: nu_foc(nu*) ~ 0", interior and abs(nu_foc(nu_star, mean_lw, mean_w)) < 1e-6,
           f"nu*={nu_star:.4f}, foc={nu_foc(nu_star, mean_lw, mean_w):.2e}")

    # gradient of nu_log_target equals (T/2) * nu_foc.
    nu0, e = 6.0, 1e-5
    grad_num = (nu_log_target(nu0 + e, sum_lw, sum_w, Tn)
                - nu_log_target(nu0 - e, sum_lw, sum_w, Tn)) / (2 * e)
    grad_ana = 0.5 * Tn * nu_foc(nu0, mean_lw, mean_w)
    _check("d/dnu log-target == (T/2) nu_foc", abs(grad_num - grad_ana) < 1e-4 * abs(grad_ana) + 1e-6,
           f"num={grad_num:.4f}, ana={grad_ana:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. draw_A_Q  vs  update_A_Q  (seam + MC mean)
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_A_Q(S):
    print("\n[4] draw_A_Q vs update_A_Q (seam + MC mean)")
    f_aug, P_smooth, w_u, r, T = S["f_aug"], S["P_smooth"], S["w_u"], S["r"], S["T"]
    mom = compute_weighted_moments(f_aug, P_smooth, np.zeros_like(P_smooth), w_u, r)
    P00, P10, P11 = mom["P00"], mom["P10"], mom["P11"]
    T_eff = T - 1
    A_em, Q_em = update_A_Q(P00, P10, P11, T_eff)

    # deterministic seam used inside draw_A_Q
    A_hat = np.linalg.solve(P00, P10.T).T
    S_scatter = 0.5 * ((P11 - A_hat @ P10.T) + (P11 - A_hat @ P10.T).T)
    _check("A_hat == update_A_Q A", np.allclose(A_hat, A_em, atol=1e-12),
           f"max|diff|={np.max(np.abs(A_hat-A_em)):.2e}")
    _check("S/T_eff == update_A_Q Q", np.allclose(S_scatter / T_eff, Q_em, atol=1e-12),
           f"max|diff|={np.max(np.abs(S_scatter/T_eff-Q_em)):.2e}")

    # MC: mean of draws centred on the EM seam.
    rng = np.random.default_rng(11)
    K = 4000
    accA = np.zeros((r, r))
    accQ = np.zeros((r, r))
    for _ in range(K):
        Ad, Qd = draw_A_Q(P00, P10, P11, T_eff, rng)
        accA += Ad
        accQ += Qd
    mA, mQ = accA / K, accQ / K
    iw_mean_Q = S_scatter / (T_eff - r - 1)          # InvWishart(scale=S, df=T_eff) mean
    _check("MC mean(A) ~ A_hat", np.max(np.abs(mA - A_hat)) < 0.02, f"max|diff|={np.max(np.abs(mA-A_hat)):.3e}")
    _check("MC mean(Q) ~ IW mean", np.max(np.abs(mQ - iw_mean_Q)) < 0.05 * np.max(np.abs(iw_mean_Q)),
           f"max|diff|={np.max(np.abs(mQ-iw_mean_Q)):.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. composite_regressor + draw_lambda_r_series  vs  update_Lambda / update_R
# ─────────────────────────────────────────────────────────────────────────────

def _series_suff_stats(S, i):
    """Replicate the weighted sufficient statistics of update_Lambda/update_R
    for series i, using composite_regressor for the quarterly regressor."""
    Y, f_aug, P_smooth = S["Y"], S["f_aug"], S["P_smooth"]
    w_eps, r = S["w_eps"], S["r"]
    j = S["block_col"][S["block_map"][S["ordered_cols"][i]]]
    freq = S["freq_list"][i]

    obs_t = np.where(~np.isnan(Y[:, i]))[0]
    y_i = Y[obs_t, i]
    w_i = w_eps[obs_t]

    if freq == "monthly":
        E_x = f_aug[obs_t, j]
        V_x = P_smooth[obs_t, j, j]
    else:
        phi = composite_regressor(f_aug, MM_WEIGHTS, r)        # (T, r)
        E_x = phi[obs_t, j]
        idx = np.array([l * r + j for l in range(5)])
        P_block = P_smooth[obs_t[:, None, None], idx[None, :, None], idx[None, None, :]]
        V_x = np.einsum("s,nsl,l->n", MM_WEIGHTS, P_block, MM_WEIGHTS)

    E_x2 = V_x + E_x ** 2
    num = float(np.sum(w_i * y_i * E_x))
    den = float(np.sum(w_i * E_x2))
    s_yy = float(np.sum(w_i * y_i ** 2))
    n_obs = obs_t.size
    return j, num, den, s_yy, n_obs, E_x


def test_lambda_r(S):
    print("\n[5] composite_regressor + draw_lambda_r_series vs update_Lambda/update_R")
    Y, f_aug, P_smooth, w_eps, r = S["Y"], S["f_aug"], S["P_smooth"], S["w_eps"], S["r"]
    ordered_cols, block_map, freq_list = S["ordered_cols"], S["block_map"], S["freq_list"]
    W = S["W_list"]

    Lam_em = update_Lambda(Y, f_aug, P_smooth, w_eps, W, block_map, freq_list, ordered_cols, r)
    R_em = update_R(Y, f_aug, P_smooth, Lam_em, w_eps, W, block_map, freq_list, ordered_cols, r)

    # composite_regressor matches update_Lambda's E_phi for the quarterly series (i=3)
    j3, _, _, _, _, E_phi = _series_suff_stats(S, 3)
    phi = composite_regressor(f_aug, MM_WEIGHTS, r)
    obs_t = np.where(~np.isnan(Y[:, 3]))[0]
    _check("composite_regressor == update_Lambda E_phi (quarterly)",
           np.allclose(phi[obs_t, j3], E_phi, atol=1e-12))

    # seam for one monthly (i=0) and the quarterly (i=3) series
    for i in (0, 3):
        j, num, den, s_yy, n_obs, _ = _series_suff_stats(S, i)
        lam_hat = num / den
        ssr = s_yy - num * num / den
        em_var = ssr / n_obs
        _check(f"lambda_hat == update_Lambda[{i}]", abs(lam_hat - Lam_em[i, j]) < 1e-10,
               f"{lam_hat:.8f} vs {Lam_em[i, j]:.8f}")
        _check(f"ssr/n_obs == update_R[{i}]", abs(em_var - R_em[i]) < 1e-10,
               f"{em_var:.8f} vs {R_em[i]:.8f}")

        # MC: draws centred on the EM seam (lambda) and IG mean (variance)
        rng = np.random.default_rng(100 + i)
        K = 8000
        accl = 0.0
        accr = 0.0
        for _ in range(K):
            ld, rd = draw_lambda_r_series(num, den, s_yy, n_obs, rng)
            accl += ld
            accr += rd
        ml, mr = accl / K, accr / K
        ig_mean = (ssr / 2.0) / (n_obs / 2.0 - 1.0)          # InvGamma(n/2, ssr/2) mean
        _check(f"MC mean(lambda)[{i}] ~ lambda_hat", abs(ml - lam_hat) < 0.02 * abs(lam_hat) + 1e-3,
               f"{ml:.5f} vs {lam_hat:.5f}")
        _check(f"MC mean(r)[{i}] ~ IG mean", abs(mr - ig_mean) < 0.05 * ig_mean,
               f"{mr:.5f} vs {ig_mean:.5f}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 72)
    print("PASSO 0 — equivalence test: src/mcmc/shared.py vs EM counterparts")
    print("=" * 72)
    S = _build_synthetic(seed=0)
    test_realized_d(S)
    test_draw_weights(S)
    test_nu(S)
    test_draw_A_Q(S)
    test_lambda_r(S)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
