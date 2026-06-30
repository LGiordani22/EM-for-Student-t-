"""
src/mcmc/test_passo2.py
=======================

Fast correctness gate for **Passo 2** (base stochastic volatility, no leverage).
Complements the full SV recovery harness
(:func:`mcmc.diagnostics.run_recovery_mcmc_sv`, slower) with quick decisive checks:

  [1] KSC-7 table validates *with tolerance* (sum q = 1, sum q m ~ -1.2704) —
      never exact equality (the mixture only approximates log chi^2_1).

  [2] Combined-precision filter reduces to the stock filter at h == 1.
      ``forward_filter_combined`` with g = w must equal ``kalman_filter`` with the
      same scalar weights to machine precision — i.e. turning SV "on" with unit
      volatility does not perturb the Passo-1 state block.

  [3] Scalar log-vol FFBS recovers a known AR(1) volatility path from log-square
      observations: the posterior-mean log-vol correlates strongly with the truth
      (this isolates the genuinely new volatility machinery from the rest).

  [4] End-to-end: a short SV Gibbs run on a simulated SV panel recovers the
      common volatility path (corr > 0.6) and keeps theta/nu sane.

Run
---
    python src/mcmc/test_passo2.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kalman import kalman_filter                                          # noqa: E402
from simulate_dfm import simulate_dfm                                     # noqa: E402

from mcmc.constants import KSC7, validate_mixture                         # noqa: E402
from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                       # noqa: E402
from mcmc.sample_states import forward_filter_combined                    # noqa: E402
from mcmc.sample_vol import sample_log_vol_process                        # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                              # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _corr(x, y):
    xz, yz = x - x.mean(), y - y.mean()
    return float((xz @ yz) / (np.linalg.norm(xz) * np.linalg.norm(yz)))


def test_ksc():
    print("\n[1] KSC-7 tolerant validation")
    v = validate_mixture(KSC7)
    _check("sum q_j == 1 (within tol)", v["sum_q"][1], f"{v['sum_q'][0]:.8f}")
    _check("sum q_j m_j ~ -1.2704 (within tol)", v["sum_qm"][1], f"{v['sum_qm'][0]:.5f}")


def test_filter_equiv(theta, fl, bm, oc, r):
    print("\n[2] combined-precision filter == stock filter at h==1")
    sim = simulate_dfm(theta=theta, T=180, freq_list=fl, block_map=bm,
                       ordered_cols=oc, r=r, seed=3)
    Y = sim["Y"]
    rng = np.random.default_rng(0)
    T = Y.shape[0]
    w_u = rng.gamma(4.0, 1.0 / 4.0, T)
    w_eps = rng.gamma(4.0, 1.0 / 4.0, T)
    th = {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])}
    fk = kalman_filter(Y, th, w_u=w_u, w_eps=w_eps, freq_list=fl)
    g_eps = np.broadcast_to(w_eps[:, None], (T, Y.shape[1])).copy()
    fc = forward_filter_combined(Y, th, w_u, g_eps, fl)
    e_f = np.max(np.abs(fk["f_filt"] - fc["f_filt"]))
    e_P = np.max(np.abs(fk["P_filt"] - fc["P_filt"]))
    e_ll = abs(fk["loglik"] - fc["loglik"])
    _check("f_filt identical", e_f < 1e-10, f"max|d|={e_f:.2e}")
    _check("P_filt identical", e_P < 1e-10, f"max|d|={e_P:.2e}")
    _check("loglik identical", e_ll < 1e-8, f"|d|={e_ll:.2e}")


def test_scalar_vol_ffbs():
    print("\n[3] scalar log-vol FFBS recovers a known AR(1) path")
    rng = np.random.default_rng(11)
    T = 800
    mu, phi, sigma = -0.2, 0.97, 0.30
    logh = np.empty(T)
    logh[0] = mu + sigma / np.sqrt(1 - phi ** 2) * rng.standard_normal()
    for t in range(1, T):
        logh[t] = mu + phi * (logh[t - 1] - mu) + sigma * rng.standard_normal()
    # r=3 log-square measurements per t (common-factor style).
    h = np.exp(logh)
    z = rng.standard_normal((T, 3)) * np.sqrt(h)[:, None]
    ys = np.log(z.reshape(-1) ** 2 + 1e-6)
    tidx = np.repeat(np.arange(T), 3)
    # run several sub-sweeps from a flat start, average the path
    cur = np.zeros(T)
    acc = np.zeros(T); n = 0
    for it in range(60):
        cur = sample_log_vol_process(ys, tidx, T, cur, mu, phi, sigma ** 2, rng)
        if it >= 20:
            acc += cur; n += 1
    post = acc / n
    c = _corr(post, logh)
    _check("posterior-mean log-vol correlates with truth (>0.8)", c > 0.8, f"corr={c:.3f}")


def test_end_to_end(theta, fl, bm, oc, r):
    print("\n[4] short SV Gibbs recovers the common volatility path")
    sv_u = (0.0, 0.97, 0.25)
    sim = simulate_dfm_sv(theta, T=220, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=sv_u, sv_eps=(0.0, 0.95, 0.15))
    Y = sim["Y"]
    res = fit_dfm_mcmc(Y, {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=300, burn_in=120, thin=1, seed=4,
                       sv=True, store_vol=True, verbose=False)
    hu = res["draws"]["h_u"].mean(axis=0)
    c = _corr(np.log(hu), sim["logh_u_true"])
    _check("common log h^u corr with truth (>0.6)", c > 0.6, f"corr={c:.3f}")
    phi_hat = res["theta_mean"]["sv_u"][1]
    _check("h^u AR(1) phi recovered (>0.85)", phi_hat > 0.85, f"phi={phi_hat:.3f}")
    nu_eps = res["theta_mean"]["nu_eps"]
    _check("nu_eps stays sane (2.5..15)", 2.5 < nu_eps < 15.0, f"nu_eps={nu_eps:.2f}")


def main():
    print("=" * 72)
    print("PASSO 2 — base stochastic-volatility sampler correctness gate")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    test_ksc()
    test_filter_equiv(theta, fl, bm, oc, r)
    test_scalar_vol_ffbs()
    test_end_to_end(theta, fl, bm, oc, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
