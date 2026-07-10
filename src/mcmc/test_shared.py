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

from em.em_e_step import compute_d_eps, compute_d_u, compute_weights          # noqa: E402
from em.em_m_step import (                                                     # noqa: E402
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
    draw_A_Q_perfactor,
    draw_hw_aux,
    draw_lambda_r_series,
    draw_weights,
    hw_iw_prior,
    nu_foc,
    nu_log_target,
    realized_deflated_d_eps,
    realized_deflated_d_u,
)
from mcmc.sample_vol import sample_volatility_block, sample_common_vol_mv, draw_ar1_params  # noqa: E402
from mcmc.sample_states import ffbs_sample_states                           # noqa: E402

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

    # Spec II per-factor deflation (T, r): equal columns == the scalar case,
    # and component-wise deflation matches (H^{-1/2}u)' Q^{-1} (H^{-1/2}u).
    d_u_pf_flat = realized_deflated_d_u(f_aug, A, Q, r, h_u=3.0 * np.ones((T, r)))
    _check("d_u per-factor (H=3 I) == scalar h=3", np.allclose(d_u_pf_flat[1:], d_u_def[1:], atol=1e-12),
           f"max|diff|={np.nanmax(np.abs(d_u_pf_flat-d_u_def)):.2e}")
    rngh = np.random.default_rng(3)
    Hpf = 0.5 + rngh.random((T, r))                     # distinct per-factor vols
    d_u_pf = realized_deflated_d_u(f_aug, A, Q, r, h_u=Hpf)
    d_ref = np.full(T, np.nan)
    for t in range(1, T):
        u = f_aug[t, :r] - A @ f_aug[t - 1, :r]
        uw = u / np.sqrt(Hpf[t])
        d_ref[t] = float(uw @ np.linalg.solve(Q, uw))
    _check("d_u per-factor == (H^-1/2 u)' Qinv (H^-1/2 u)", np.allclose(d_u_pf[1:], d_ref[1:], atol=1e-12),
           f"max|diff|={np.nanmax(np.abs(d_u_pf-d_ref)):.2e}")


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

    # ── Family D proper prior (Phase 5): exponential / uniform hooks ──────────
    from mcmc.sample_params import (draw_nu_griddy, nu_log_prior_exponential,   # noqa: E402
                                    nu_log_prior_uniform)
    lp_exp = nu_log_prior_exponential(mean=20.0)
    # prior enters the log-target additively (eq:param-nu-logtarget)
    base = nu_log_target(6.0, sum_lw, sum_w, Tn)
    withp = nu_log_target(6.0, sum_lw, sum_w, Tn, log_prior=lp_exp)
    _check("nu_log_target adds log_prior additively", abs((withp - base) - (-6.0 / 20.0)) < 1e-12,
           f"delta={withp-base:.6f} vs {-6.0/20.0:.6f}")

    # flat (None) == an all-zero prior, bit-for-bit (the seam)
    d_flat = draw_nu_griddy(w, np.random.default_rng(9), nu_bounds=(2.001, 1000.0), grid_size=400)
    d_zero = draw_nu_griddy(w, np.random.default_rng(9), nu_bounds=(2.001, 1000.0), grid_size=400,
                            log_prior=lambda nu: 0.0)
    _check("griddy: zero log_prior == flat (bitwise)", d_flat == d_zero, f"{d_flat} vs {d_zero}")

    # exponential prior (decreasing in nu) shifts the posterior mean DOWN vs flat.
    # Use a SMALL weak-data sample so the prior is visible against the likelihood.
    ws = np.random.default_rng(2).gamma(8.0 / 2.0, 2.0 / 8.0, size=40)
    def _mc(lp, seed, K=3000):
        g = np.random.default_rng(seed); s = 0.0
        for _ in range(K):
            s += draw_nu_griddy(ws, g, nu_bounds=(2.001, 200.0), grid_size=400, log_prior=lp)
        return s / K
    m_flat = _mc(None, 11); m_exp = _mc(nu_log_prior_exponential(mean=5.0), 11)
    _check("exponential prior lowers posterior mean(nu)", m_exp < m_flat,
           f"exp={m_exp:.2f} < flat={m_flat:.2f}")

    # uniform(2, 50): all draws truncated below 50 even on a wide bracket
    lp_u = nu_log_prior_uniform(2.0, 50.0)
    g = np.random.default_rng(3); mx = 0.0
    for _ in range(2000):
        mx = max(mx, draw_nu_griddy(w, g, nu_bounds=(2.001, 1000.0), grid_size=500, log_prior=lp_u))
    _check("uniform(2,50) prior truncates nu < 50", mx < 50.0, f"max nu={mx:.3f}")


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
# 4b. draw_A_Q_perfactor (Spec II) — collapse seam: H ≡ 1 == scalar MNIW moments
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_A_Q_perfactor(S):
    print("\n[4b] draw_A_Q_perfactor (Spec II) collapse to scalar MNIW at H=1")
    f_aug, w_u, r, T = S["f_aug"], S["w_u"], S["r"], S["T"]
    P_smooth = S["P_smooth"]

    # scalar-weighted realized moments (P_smooth=0), the first-stage seam
    mom = compute_weighted_moments(f_aug, np.zeros_like(P_smooth), np.zeros_like(P_smooth), w_u, r)
    P00, P10, P11 = mom["P00"], mom["P10"], mom["P11"]
    A_hat = np.linalg.solve(P00, P10.T).T
    S_scatter = 0.5 * ((P11 - A_hat @ P10.T) + (P11 - A_hat @ P10.T).T)

    # per-factor draw with H ≡ 1 and A_cur = A_hat: the posterior moments must
    # reduce to the scalar ones (eq:param-Q-post / eq:param-A-precision collapse).
    f_head = f_aug[:, :r]
    h_u = np.ones((T, r))
    rng = np.random.default_rng(7)
    _, Q_draw, pm = draw_A_Q_perfactor(f_head, h_u, w_u, A_hat, rng, _return_moments=True)

    _check("Q_scale == scalar residual scatter", np.allclose(pm["Q_scale"], S_scatter, atol=1e-10),
           f"max|diff|={np.max(np.abs(pm['Q_scale'] - S_scatter)):.2e}")
    _check("A posterior mean == A_hat", np.allclose(pm["A_mean"], A_hat, atol=1e-9),
           f"max|diff|={np.max(np.abs(pm['A_mean'] - A_hat)):.2e}")
    prec_scalar = np.kron(P00, np.linalg.inv(Q_draw))
    _check("A precision == P00 (x) Qinv", np.allclose(pm["A_prec"], prec_scalar, atol=1e-9),
           f"max|diff|={np.max(np.abs(pm['A_prec'] - prec_scalar)):.2e}")

    # deflation sanity: h_u = 4 everywhere must halve the whitening (H^{-1/2}=1/2),
    # so the Q-scale scatter shrinks by 1/4 relative to H=1 (at the same A_cur).
    _, _, pm4 = draw_A_Q_perfactor(f_head, 4.0 * np.ones((T, r)), w_u, A_hat,
                                   np.random.default_rng(7), _return_moments=True)
    _check("H=4 quarters the Q-scale", np.allclose(pm4["Q_scale"], 0.25 * S_scatter, atol=1e-10),
           f"max|diff|={np.max(np.abs(pm4['Q_scale'] - 0.25*S_scatter)):.2e}")

    # ── proper priors (Phase 2): Q ~ IW(Psi0, nu0), vec(A) ~ N(A0, V0) ────────
    Psi0 = 2.0 * np.eye(r); nu0 = float(r + 1)              # nu0 = r+1 (uniform corr)
    V0_inv = 0.05 * np.eye(r * r); A0 = np.zeros((r, r))
    _, Qd, pmp = draw_A_Q_perfactor(f_head, np.ones((T, r)), w_u, A_hat,
                                    np.random.default_rng(7),
                                    Psi0=Psi0, nu0=nu0, A0=A0, V0_inv=V0_inv,
                                    _return_moments=True)
    _check("prior Q: scale = Psi0 + scatter (eq:param-Q-post)",
           np.allclose(pmp["Q_scale"], Psi0 + S_scatter, atol=1e-10),
           f"max|diff|={np.max(np.abs(pmp['Q_scale'] - (Psi0 + S_scatter))):.2e}")
    _check("prior Q: df = nu0 + T_eff", abs(pmp["Q_df"] - (nu0 + (T - 1))) < 1e-9,
           f"df={pmp['Q_df']}")
    _check("prior A: precision = V0_inv + P00 (x) Qinv (eq:param-AQ)",
           np.allclose(pmp["A_prec"], V0_inv + np.kron(P00, np.linalg.inv(Qd)), atol=1e-9),
           f"max|diff|={np.max(np.abs(pmp['A_prec'] - (V0_inv + np.kron(P00, np.linalg.inv(Qd))))):.2e}")
    # shrinkage: with A0=0 and a finite prior precision, |A_mean| < |A_hat|
    _check("prior A: mean shrinks toward A0=0",
           np.linalg.norm(pmp["A_mean"]) < np.linalg.norm(A_hat),
           f"|A_mean|={np.linalg.norm(pmp['A_mean']):.3f} < |A_hat|={np.linalg.norm(A_hat):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4c. sample_common_vol_mv (Spec II, R_xi=I) — r=1 seam vs scalar common block
# ─────────────────────────────────────────────────────────────────────────────

def test_common_vol_mv_seam(_S):
    print("\n[4c] sample_common_vol_mv (Spec II, R_xi=I) r=1 seam vs scalar common")
    gen = np.random.default_rng(123)
    T = 80
    A = np.array([[0.5]]); Q = np.array([[1.3]])
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.5 * f[t - 1] + np.sqrt(1.3) * gen.standard_normal()
    f_aug = np.zeros((T, 5))
    for t in range(T):
        for l in range(5):
            if t - l >= 0:
                f_aug[t, l] = f[t - l]
    Lambda = np.array([[0.9]]); R = np.array([0.3])
    Y = (0.9 * f + np.sqrt(0.3) * gen.standard_normal(T)).reshape(T, 1)
    w_u = gen.gamma(4.0, 1.0 / 4.0, T); w_u[0] = 1.0
    w_eps = gen.gamma(4.0, 1.0 / 4.0, T)
    logh_u0 = np.zeros(T); logh_eps0 = np.zeros((T, 1))
    sv_u = (0.0, 0.95, 0.05); sv_eps = np.array([[0.0, 0.95, 0.05]])
    theta = {"A": A, "Q": Q, "Lambda": Lambda, "R": R}

    # scalar common block (first RNG consumer inside the block is the common path)
    vb = sample_volatility_block(Y, f_aug, theta, w_u, w_eps, logh_u0, logh_eps0,
                                 sv_u, sv_eps, np.random.default_rng(7))
    # per-factor block at r=1, same RNG seed
    u_head = f_aug[1:, :1] - f_aug[:-1, :1] @ A.T                # (T-1, 1)
    mv = sample_common_vol_mv(u_head, Q, w_u, logh_u0.reshape(T, 1),
                              np.array([sv_u]), np.random.default_rng(7))

    _check("mv logh_u == scalar common path (r=1, bitwise)",
           np.array_equal(mv["logh_u"][:, 0], vb["logh_u"]),
           f"max|diff|={np.max(np.abs(mv['logh_u'][:, 0] - vb['logh_u'])):.2e}")
    _check("mv sv_u == scalar common sv (r=1, bitwise)",
           np.array_equal(mv["sv_u"][0], vb["sv_u"]),
           f"{mv['sv_u'][0]} vs {vb['sv_u']}")
    _check("mv shapes (T,r)/(r,3)", mv["logh_u"].shape == (T, 1) and mv["sv_u"].shape == (1, 3))
    # Commit 2 (coupled r-dim FFBS) runs and is finite at r=1 (R_xi=[[1]]).
    mvc = sample_common_vol_mv(u_head, Q, w_u, logh_u0.reshape(T, 1), np.array([sv_u]),
                               np.random.default_rng(1), R_xi=np.eye(1))
    _check("coupled r-dim FFBS runs finite (r=1)",
           mvc["logh_u"].shape == (T, 1) and np.all(np.isfinite(mvc["logh_u"])))


# ─────────────────────────────────────────────────────────────────────────────
# 4d. states step (a): per-factor companion (Spec II) seam — H = h I == scalar h
# ─────────────────────────────────────────────────────────────────────────────

def test_states_perfactor_seam(S):
    print("\n[4d] ffbs states: per-factor companion (H=hI) == scalar common (seam)")
    Y, r, T, M = S["Y"], S["r"], S["T"], S["M"]
    theta = {"A": S["A"], "Q": S["Q"], "Lambda": S["Lambda"], "R": S["R"],
             "Sigma_0": np.eye(5 * r)}
    freq = S["freq_list"]; w_u, w_eps = S["w_u"], S["w_eps"]
    hvec = 0.5 + 0.8 * np.random.default_rng(9).random(T)     # a positive common vol path
    h_eps = np.ones((T, M))                                   # flat idio, isolate the common

    st_sc = ffbs_sample_states(Y, theta, w_u, w_eps, freq, np.random.default_rng(3),
                               h_u=hvec, h_eps=h_eps)                    # scalar H^u = h I
    H = np.tile(hvec[:, None], (1, r))                                   # per-factor, equal cols
    st_pf = ffbs_sample_states(Y, theta, w_u, w_eps, freq, np.random.default_rng(3),
                               h_u=H, h_eps=h_eps)                       # Spec II sandwich
    _check("per-factor H=hI companion == scalar h companion (sqrt(H)Q sqrt(H)=hQ)",
           np.allclose(st_pf["F"], st_sc["F"], atol=1e-9),
           f"max|diff|={np.max(np.abs(st_pf['F'] - st_sc['F'])):.2e}")
    _check("per-factor states finite, shape (T,r)",
           st_pf["F"].shape == (T, r) and np.all(np.isfinite(st_pf["F"])))


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

    # ── proper NIG prior (Phase 2): MC mean matches the analytic posterior ────
    a0, b0, m0, M0_inv = 3.0, 0.5, 0.2, 2.0
    Fbar = den + M0_inv; cbar = num + M0_inv * m0
    lam_post = cbar / Fbar
    b_n = b0 + 0.5 * (s_yy + M0_inv * m0 * m0 - cbar * cbar / Fbar)
    shape = a0 + n_obs / 2.0
    ig_mean_p = b_n / (shape - 1.0)
    rng = np.random.default_rng(555)
    K = 8000; al = 0.0; ar = 0.0
    for _ in range(K):
        ld, rd = draw_lambda_r_series(num, den, s_yy, n_obs, rng,
                                      a0=a0, b0=b0, m0=m0, M0_inv=M0_inv)
        al += ld; ar += rd
    _check("prior NIG: MC mean(lambda) ~ cbar/Fbar (eq:param-LR-post)",
           abs(al / K - lam_post) < 0.02 * abs(lam_post) + 1e-3, f"{al/K:.5f} vs {lam_post:.5f}")
    _check("prior NIG: MC mean(r) ~ IG(a0+n/2, b_n) mean",
           abs(ar / K - ig_mean_p) < 0.05 * ig_mean_p, f"{ar/K:.5f} vs {ig_mean_p:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. draw_Lambda_R_block prior threading (Phase 2c wiring)
# ─────────────────────────────────────────────────────────────────────────────

def test_lambda_r_block_priors(S):
    print("\n[6] draw_Lambda_R_block: flat-default seam + per-series prior routing")
    from mcmc.sample_params import draw_Lambda_R_block

    Y, f_aug, w_eps, r, M = S["Y"], S["f_aug"], S["w_eps"], S["r"], S["M"]
    ordered_cols, block_map, freq_list = S["ordered_cols"], S["block_map"], S["freq_list"]
    block_col = S["block_col"]

    # (a) flat-limit seam: default args == explicit zero/None prior, bit-identical.
    Lam_a, R_a = draw_Lambda_R_block(
        Y, f_aug, w_eps, block_map, freq_list, ordered_cols, r,
        np.random.default_rng(7))
    Lam_b, R_b = draw_Lambda_R_block(
        Y, f_aug, w_eps, block_map, freq_list, ordered_cols, r,
        np.random.default_rng(7), a0=0.0, b0=0.0, m0=None, M0_inv=0.0)
    _check("block flat default == explicit flat (bitwise)",
           np.array_equal(Lam_a, Lam_b) and np.array_equal(R_a, R_b))

    # (b) per-series routing: an overwhelming prior forces each drawn (L_i, r_i)
    # to its own prior mean, at the correct block column j_i (m0 as (M, r) matrix,
    # b0 as (M,) per-series).  Distinct targets per series catch a mis-routed j.
    j = np.array([block_col[block_map[c]] for c in ordered_cols])
    m0 = np.zeros((M, r))
    r_target = np.zeros(M)
    for i in range(M):
        m0[i, j[i]] = -2.0 - i             # distinct, off the data-driven value
        r_target[i] = 0.5 + 0.3 * i        # distinct prior-mean variance
    a0 = 1e6
    b0 = (a0 - 1.0) * r_target             # (M,): IG mean = r_target
    Lam_p, R_p = draw_Lambda_R_block(
        Y, f_aug, w_eps, block_map, freq_list, ordered_cols, r,
        np.random.default_rng(11), a0=a0, b0=b0, m0=m0, M0_inv=1e8)
    lam_at_j = np.array([Lam_p[i, j[i]] for i in range(M)])
    m0_at_j = np.array([m0[i, j[i]] for i in range(M)])
    _check("strong prior: loading -> m0[i, j_i] (column routing)",
           np.allclose(lam_at_j, m0_at_j, atol=1e-3),
           f"{lam_at_j} vs {m0_at_j}")
    _check("strong prior: r_i -> b0_i/(a0-1) (per-series b0 routing)",
           np.allclose(R_p, r_target, rtol=5e-3),
           f"{R_p} vs {r_target}")
    # off-block columns stay exactly zero (block restriction preserved).
    off = Lam_p.copy()
    for i in range(M):
        off[i, j[i]] = 0.0
    _check("block restriction preserved (off-column loadings == 0)",
           np.all(off == 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# 7. draw_ar1_params half-Normal sigma_eta prior (Phase 3, Family B)
# ─────────────────────────────────────────────────────────────────────────────

def test_ar1_half_normal():
    print("\n[7] draw_ar1_params: half-Normal sigma_eta (mu=0, recovery, guards)")
    rng = np.random.default_rng(0)

    # stationary AR(1) log-vol path with known (phi, sigma2), mu=0
    phi_true, s2_true, T = 0.95, 0.10, 2000
    x = np.zeros(T)
    x[0] = np.sqrt(s2_true / (1 - phi_true ** 2)) * rng.standard_normal()
    for t in range(1, T):
        x[t] = phi_true * x[t - 1] + np.sqrt(s2_true) * rng.standard_normal()

    # mini RW-MH chain: feed sigma2_cur back in (B large => weakly informative)
    n_it, burn = 4000, 1000
    s2 = 0.5
    acc_s2, acc_phi, n_kept, mu_max = 0.0, 0.0, 0, 0.0
    for it in range(n_it):
        mu, phi, s2 = draw_ar1_params(
            x, rng, fix_mu0=True, sigma_prior="half_normal",
            half_normal_B=10.0, sigma2_cur=s2, prop_log_sigma=0.25)
        mu_max = max(mu_max, abs(mu))
        if it >= burn:
            acc_s2 += s2; acc_phi += phi; n_kept += 1
    s2_bar, phi_bar = acc_s2 / n_kept, acc_phi / n_kept
    _check("half-Normal: mu returned exactly 0", mu_max == 0.0, f"max|mu|={mu_max:.2e}")
    _check("half-Normal: sigma2 recovered (0.05..0.20)", 0.05 < s2_bar < 0.20,
           f"s2_bar={s2_bar:.4f} vs {s2_true}")
    _check("half-Normal: phi recovered (>0.90)", phi_bar > 0.90, f"phi_bar={phi_bar:.4f}")

    # the half-Normal shrinks sigma_eta toward 0 harder than the flat/IG for small B:
    # a very tight B must pull the posterior mean of sigma2 below the truth.
    s2 = 0.5; acc = 0.0; nk = 0
    for it in range(3000):
        _, _, s2 = draw_ar1_params(x, rng, fix_mu0=True, sigma_prior="half_normal",
                                   half_normal_B=0.01, sigma2_cur=s2, prop_log_sigma=0.25)
        if it >= 1000:
            acc += s2; nk += 1
    _check("half-Normal: tight B shrinks sigma2 below truth", acc / nk < s2_true,
           f"s2_bar(B=0.01)={acc/nk:.4f} < {s2_true}")

    # guards
    try:
        draw_ar1_params(x, rng, fix_mu0=True, sigma_prior="half_normal", sigma2_cur=None)
        ok_g1 = False
    except ValueError:
        ok_g1 = True
    _check("half-Normal: raises without sigma2_cur", ok_g1)
    try:
        draw_ar1_params(x, rng, fix_mu0=False, sigma_prior="half_normal", sigma2_cur=0.1)
        ok_g2 = False
    except ValueError:
        ok_g2 = True
    _check("half-Normal: raises with fix_mu0=False", ok_g2)
    try:
        draw_ar1_params(x, rng, sigma_prior="bogus")
        ok_g3 = False
    except ValueError:
        ok_g3 = True
    _check("unknown sigma_prior raises", ok_g3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Family A priors on the MNIW draw + Huang-Wand hierarchical IW on Q
#    (eq:param-Q-hw-prior / -post / -aux, tab:param-prior-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def test_A_Q_priors_and_huang_wand(S):
    print("\n[8] draw_A_Q priors + Huang-Wand hierarchical IW on Q")
    from scipy.stats import invwishart, t as student_t_dist

    f_aug, P_smooth, w_u, r, T = S["f_aug"], S["P_smooth"], S["w_u"], S["r"], S["T"]
    mom = compute_weighted_moments(f_aug, P_smooth, np.zeros_like(P_smooth), w_u, r)
    P00, P10, P11 = mom["P00"], mom["P10"], mom["P11"]
    T_eff = T - 1

    # (a) flat default == explicit flat args, bit-for-bit (the EM seam is untouched)
    A1, Q1 = draw_A_Q(P00, P10, P11, T_eff, np.random.default_rng(7))
    A2, Q2 = draw_A_Q(P00, P10, P11, T_eff, np.random.default_rng(7),
                      Psi0=None, nu0=0.0, A0=None, kappa=0.0)
    _check("draw_A_Q: flat default == explicit flat (bitwise)",
           np.array_equal(A1, A2) and np.array_equal(Q1, Q2))

    # (b) a strong matrix-Normal prior pulls A_hat to A0 and the IW df up by nu0
    A_hat = np.linalg.solve(P00, P10.T).T
    A0 = np.zeros((r, r))
    rngp = np.random.default_rng(8)
    acc = np.zeros((r, r))
    K = 400
    for _ in range(K):
        Ad, _ = draw_A_Q(P00, P10, P11, T_eff, rngp, A0=A0, kappa=1e6)
        acc += Ad
    _check("draw_A_Q: kappa -> inf shrinks A to A0",
           np.max(np.abs(acc / K - A0)) < 0.05 * max(1e-9, np.max(np.abs(A_hat))),
           f"max|mean(A)-A0|={np.max(np.abs(acc/K-A0)):.3e} vs |A_hat|={np.max(np.abs(A_hat)):.3e}")

    # Psi0 enters the IW scale additively and nu0 the df (eq:param-Q-post):
    # E[Q] = (Psi0 + S) / (nu0 + T_eff - r - 1).
    Psi0 = 5.0 * np.eye(r); nu0 = 6.0
    Sres = P11 - A_hat @ P10.T
    rngq = np.random.default_rng(9)
    accQ = np.zeros((r, r))
    K = 3000
    for _ in range(K):
        _, Qd = draw_A_Q(P00, P10, P11, T_eff, rngq, Psi0=Psi0, nu0=nu0)
        accQ += Qd
    target = 0.5 * (Psi0 + Sres + (Psi0 + Sres).T) / (nu0 + T_eff - r - 1)
    _check("draw_A_Q: (Psi0, nu0) enter scale/df -> IW mean",
           np.max(np.abs(accQ / K - target)) < 0.05 * np.max(np.abs(target)),
           f"max|diff|={np.max(np.abs(accQ/K - target)):.3e}")

    # (c) hw_iw_prior: Psi0 = 2 nu* diag(1/a), nu0 = nu* + r - 1  (eq:param-Q-hw-prior)
    a = np.array([0.5, 2.0, 4.0])
    Psi_hw, nu_hw = hw_iw_prior(a, nu_star=2.0)
    _check("hw_iw_prior: Psi0 = 2 nu* diag(1/a)",
           np.allclose(Psi_hw, 4.0 * np.diag(1.0 / a), atol=1e-15))
    _check("hw_iw_prior: nu0 = nu* + r - 1", nu_hw == 2.0 + 3 - 1)

    # (d) draw_hw_aux: IG((nu*+r)/2, nu*(Q^{-1})_jj + 1/A_j^2)  (eq:param-Q-hw-aux)
    Qc = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 0.5]])
    nu_star, A_sc = 2.0, 3.0
    rate = nu_star * np.diag(np.linalg.inv(Qc)) + 1.0 / A_sc ** 2
    shape = 0.5 * (nu_star + 3)                       # = 2.5 > 1, IG mean exists
    ig_mean = rate / (shape - 1.0)
    rng = np.random.default_rng(21)
    K = 60000
    acc_a = np.zeros(3)
    for _ in range(K):
        acc_a += draw_hw_aux(Qc, rng, nu_star=nu_star, A_scales=A_sc)
    _check("draw_hw_aux: MC mean == IG mean",
           np.max(np.abs(acc_a / K - ig_mean)) < 0.03 * np.max(ig_mean),
           f"mc={acc_a/K}, ig={ig_mean}")

    # (e) THE construction claim (eq:param-Q-hw-prior): marginalising the a_j out,
    #     sqrt(Q_jj) ~ half-t_{nu*}(scale A)  and  corr(Q)_jk ~ Uniform(-1, 1) at nu*=2.
    #     Sample the *prior* by iterating its two conditionals (no data).
    rr, nu_star, A_sc = 2, 2.0, 1.0
    rng = np.random.default_rng(31)
    a_cur = np.ones(rr)
    K, burn = 12000, 200
    sd0 = np.empty(K); corr = np.empty(K)
    for i in range(K + burn):
        Psi_i, nu_i = hw_iw_prior(a_cur, nu_star)
        Qd = np.atleast_2d(invwishart.rvs(df=nu_i, scale=Psi_i, random_state=rng))
        a_cur = draw_hw_aux(Qd, rng, nu_star=nu_star, A_scales=A_sc)
        if i >= burn:
            sd0[i - burn] = np.sqrt(Qd[0, 0])
            corr[i - burn] = Qd[0, 1] / np.sqrt(Qd[0, 0] * Qd[1, 1])
    med_emp = float(np.median(sd0))
    med_th = A_sc * float(student_t_dist.ppf(0.75, nu_star))   # half-t median
    _check("HW prior: median sqrt(Q_jj) == half-t_{nu*}(A) median",
           abs(med_emp - med_th) < 0.08 * med_th,
           f"emp={med_emp:.4f} vs half-t={med_th:.4f}")
    _check("HW prior: marginal correlation ~ Uniform(-1,1) at nu*=2",
           abs(float(np.mean(corr))) < 0.03 and abs(float(np.std(corr)) - 1 / np.sqrt(3)) < 0.03,
           f"mean={np.mean(corr):.4f} (0), sd={np.std(corr):.4f} (0.5774)")

    # (f) the plain IW at nu0 = r+1 also gives uniform marginal correlations
    #     (eq:param-Q-uniform-nu) — the default the HW switch is benchmarked against.
    rng = np.random.default_rng(41)
    cc = np.array([np.atleast_2d(invwishart.rvs(df=rr + 1, scale=np.eye(rr),
                                                random_state=rng))
                   for _ in range(8000)])
    rho_iw = cc[:, 0, 1] / np.sqrt(cc[:, 0, 0] * cc[:, 1, 1])
    _check("plain IW at nu0=r+1: marginal correlation ~ Uniform(-1,1)",
           abs(float(np.std(rho_iw)) - 1 / np.sqrt(3)) < 0.03,
           f"sd={np.std(rho_iw):.4f} (0.5774)")


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
    test_draw_A_Q_perfactor(S)
    test_common_vol_mv_seam(S)
    test_states_perfactor_seam(S)
    test_lambda_r(S)
    test_lambda_r_block_priors(S)
    test_ar1_half_normal()
    test_A_Q_priors_and_huang_wand(S)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
