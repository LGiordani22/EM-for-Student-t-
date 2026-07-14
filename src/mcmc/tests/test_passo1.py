"""
src/mcmc/test_passo1.py
=======================

Fast correctness gate for **Passo 1** (the no-SV Gibbs sampler = the MCMC
version of the EM).  Complements the full distributional recovery harness in
:func:`mcmc.diagnostics.run_recovery_mcmc` (which is slower: multi-chain, T~400)
with three quick, decisive checks that a CI / a human can run in ~1-2 minutes:

  [1] FFBS exactness + latent recovery
      - the reconstructed augmented path is internally consistent to MACHINE
        PRECISION (``f_aug[t][r:2r] == f_aug[t-1][0:r]``), which is what makes it
        safe for every downstream reader (moments use the internal lag block,
        the residual helpers use consecutive heads);
      - a single FFBS draw correlates strongly with the true factors.

  [2] MNIW Kronecker orientation (the user's flagged risk n.1)
      - the empirical covariance of the ``draw_A_Q`` draws matches the analytic
        ``P00^{-1} (x) Q`` (column-major vec), NOT the transpose.  A wrong
        Kronecker orientation would pass the posterior-*mean* test (already in
        test_shared) yet fail here on the *dispersion*.

  [3] EM cross-check on a short panel
      - a single short Gibbs chain's posterior means reproduce the EM point
        estimate on the same synthetic panel (the defining property of the
        no-SV sampler).

Run
---
    python src/mcmc/test_passo1.py

Prints [PASS]/[FAIL] per check; exits non-zero if any fails.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from monte_carlo_recovery import procrustes_block_diagonal               # noqa: E402
from simulate_dfm import simulate_dfm                                     # noqa: E402

from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                       # noqa: E402
from mcmc.sample_states import ffbs_sample_states                        # noqa: E402
from mcmc.shared import draw_A_Q                                          # noqa: E402

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


def _abscorr(x, y):
    xz, yz = x - x.mean(), y - y.mean()
    return abs((xz @ yz) / (np.linalg.norm(xz) * np.linalg.norm(yz)))


# ─────────────────────────────────────────────────────────────────────────────

def test_ffbs(theta, freq_list, block_map, ordered_cols, r):
    print("\n[1] FFBS exactness + latent recovery")
    sim = simulate_dfm(theta=theta, T=200, freq_list=freq_list, block_map=block_map,
                       ordered_cols=ordered_cols, r=r, seed=7)
    Y = sim["Y"]
    rng = np.random.default_rng(0)
    st = ffbs_sample_states(Y, {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                            np.ones(200), np.ones(200), freq_list, rng)
    f_aug = st["f_aug"]
    lag_err = float(np.max(np.abs(f_aug[1:, r:2 * r] - f_aug[:-1, 0:r])))
    _check("augmented path internal-lag consistency (exact)", lag_err < 1e-12,
           f"max|err|={lag_err:.2e}")
    corrs = [_abscorr(st["F"][:, j], sim["F"][:, j]) for j in range(r)]
    _check("FFBS draw correlates with true factors (>0.85)",
           all(c > 0.85 for c in corrs), f"|corr|={[f'{c:.3f}' for c in corrs]}")


def test_kronecker():
    print("\n[2] draw_A_Q MNIW Kronecker orientation (dispersion)")
    rng = np.random.default_rng(1)
    r = 3
    T_eff = 300
    # Random SPD moments consistent with a stable VAR.
    B = rng.standard_normal((r, r))
    P00 = B @ B.T + r * np.eye(r)
    A_true = 0.4 * np.eye(r) + 0.05 * rng.standard_normal((r, r))
    P10 = A_true @ P00
    Qc = rng.standard_normal((r, r)); Qc = Qc @ Qc.T + np.eye(r)
    P11 = A_true @ P00 @ A_true.T + T_eff * Qc / T_eff + P10 @ np.linalg.solve(P00, P10.T) * 0
    # Make P11 a valid residual-scatter parent: P11 = A P10' + S with S = T_eff*Q-ish.
    A_hat = np.linalg.solve(P00, P10.T).T
    S = T_eff * (Qc)                       # target IW scale
    P11 = S + A_hat @ P10.T
    S_sym = 0.5 * (S + S.T)

    K = 8000
    vecA = np.zeros((K, r * r))
    Qsum = np.zeros((r, r))
    for k in range(K):
        Ad, Qd = draw_A_Q(P00, P10, P11, T_eff, rng)
        vecA[k] = Ad.flatten(order="F")    # column-major vec
        Qsum += Qd
    EQ = Qsum / K
    emp_cov = np.cov(vecA, rowvar=False)
    P00inv = np.linalg.inv(P00)
    analytic = np.kron(P00inv, EQ)         # Cov(vec_col A) = P00^{-1} (x) E[Q]
    transpose = np.kron(EQ, P00inv)        # the WRONG orientation
    rel = np.linalg.norm(emp_cov - analytic) / np.linalg.norm(analytic)
    rel_t = np.linalg.norm(emp_cov - transpose) / np.linalg.norm(transpose)
    _check("Cov(vec A) ~ P00^{-1} (x) Q  (correct orientation)", rel < 0.08,
           f"relerr={rel:.3f}")
    _check("correct orientation beats the transpose", rel < rel_t,
           f"correct={rel:.3f} vs transpose={rel_t:.3f}")


def test_em_crosscheck(theta, freq_list, block_map, ordered_cols, r):
    print("\n[3] EM cross-check on a short panel (MCMC mean ~ EM point)")
    from em.em_main import fit_dfm
    from monte_carlo_recovery import init_theta_from_synthetic

    sim = simulate_dfm(theta=theta, T=250, freq_list=freq_list, block_map=block_map,
                       ordered_cols=ordered_cols, r=r, seed=11)
    Y = sim["Y"]
    freq_map = {c: f for c, f in zip(ordered_cols, freq_list)}
    theta0, _ = init_theta_from_synthetic(Y, ordered_cols=ordered_cols,
                                          block_map=block_map, freq_map=freq_map)
    em = fit_dfm(Y=Y, theta_init=theta0, freq_list=freq_list, block_map=block_map,
                 ordered_cols=ordered_cols, verbose=False, save_path=None,
                 max_iter=250, use_full_elbo=True)
    theta_em = em["theta"]
    res = fit_dfm_mcmc(Y, {**theta_em, "Sigma_0": np.asarray(theta_em["Sigma_0"])},
                       freq_list, block_map, ordered_cols, n_iter=800, burn_in=300,
                       thin=1, seed=3, verbose=False)
    tm = res["theta_mean"]
    Lt = np.asarray(theta_em["Lambda"])
    # Put the MCMC mean in the EM frame (per-block sign+scale).
    H = procrustes_block_diagonal(tm["Lambda"], Lt, ordered_cols=ordered_cols,
                                  block_map=block_map)
    h = np.diag(H)
    A_a = tm["A"] * (h[None, :] / h[:, None])
    Q_a = tm["Q"] / (h[:, None] * h[None, :])
    Lam_a = tm["Lambda"] * h[None, :]
    nz = np.abs(Lt) > 1e-12

    def _re(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    reA = _re(A_a, np.asarray(theta_em["A"]))
    reQ = _re(Q_a, np.asarray(theta_em["Q"]))
    reL = _re(Lam_a[nz], Lt[nz])
    reR = _re(tm["R"], np.asarray(theta_em["R"]).ravel())
    _check("A  posterior mean ~ EM (relerr<0.15)", reA < 0.15, f"relerr={reA:.3f}")
    _check("Q  posterior mean ~ EM (relerr<0.20)", reQ < 0.20, f"relerr={reQ:.3f}")
    _check("Lambda posterior mean ~ EM (relerr<0.10)", reL < 0.10, f"relerr={reL:.3f}")
    _check("R  posterior mean ~ EM (relerr<0.15)", reR < 0.15, f"relerr={reR:.3f}")
    _check("nu_eps MCMC ~ EM (rel<0.30)",
           abs(tm["nu_eps"] - float(theta_em["nu_eps"])) / float(theta_em["nu_eps"]) < 0.30,
           f"MCMC={tm['nu_eps']:.2f} EM={float(theta_em['nu_eps']):.2f}")


def main() -> int:
    print("=" * 72)
    print("PASSO 1 — no-SV Gibbs sampler correctness gate")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    freq_list, block_map, ordered_cols, r = (
        w["freq_list"], w["block_map"], w["ordered_cols"], w["r"])
    test_ffbs(theta, freq_list, block_map, ordered_cols, r)
    test_kronecker()
    test_em_crosscheck(theta, freq_list, block_map, ordered_cols, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
