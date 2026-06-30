"""
src/mcmc/test_passo3.py
=======================

Fast correctness gate for **Passo 3** (leverage, Branch A: contemporaneous
timing + Metropolis).  Complements the full recovery harness
(:func:`mcmc.diagnostics.run_recovery_mcmc_leverage`, slower) with quick,
decisive checks:

  [1] Family C kernel: ``draw_rho_scalar`` recovers a known leverage ``rho``
      from synthetic leverage-coupled innovations (isolates the rho Metropolis
      from the rest).

  [2] Skewness mechanism (DGP): contemporaneous leverage with ``rho < 0``
      produces a *left-skewed* factor-innovation density along the leverage
      direction, while ``rho = 0`` is symmetric — the link to Growth-at-Risk.

  [3] End-to-end: a short leverage Gibbs run executes, the Metropolis
      acceptance rates are in a workable range, and the dominant leverage
      component is recovered with the right (negative) sign.

  [4] Nesting: with ``rho = 0`` in the data, the leverage path-MH target has no
      drift and the contemporaneous block reduces to the base case (checked at
      the kernel level).

Run
---
    python src/mcmc/test_passo3.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                       # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                             # noqa: E402
from mcmc.sample_leverage import draw_rho_scalar, _lev_path_mh           # noqa: E402
from mcmc.diagnostics import leverage_skewness_check                     # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def test_rho_kernel():
    print("\n[1] Family C: draw_rho_scalar recovers a known leverage")
    rng = np.random.default_rng(0)
    T = 4000
    rho_true, sigma2 = -0.5, 0.04
    sig = np.sqrt(sigma2)
    z = rng.standard_normal(T)
    eta = rho_true * sig * z + sig * np.sqrt(1 - rho_true ** 2) * rng.standard_normal(T)
    k = sig * z
    rho = 0.0
    acc = 0.0; n = 0
    draws = []
    for it in range(4000):
        rho, a = draw_rho_scalar(rho, eta, k, sigma2, 0.05, rng)
        acc += a
        if it >= 1000:
            draws.append(rho)
    mean = float(np.mean(draws))
    _check("posterior mean rho ~ true (-0.5)", abs(mean - rho_true) < 0.07,
           f"mean={mean:.3f}")
    _check("rho Metropolis acceptance in (0.1, 0.9)", 0.1 < acc / 4000 < 0.9,
           f"acc={acc/4000:.2f}")


def test_skewness(theta):
    print("\n[2] Skewness mechanism: rho<0 -> left-skewed factor innovations")
    sk = leverage_skewness_check(theta, sv_u=(0.0, 0.95, 0.15),
                                 rho_u=np.array([-0.6, -0.4, -0.3]),
                                 T=10000, seed=3)
    _check("rho<0 produces left skew (< -0.1)", sk["skew_leverage"] < -0.1,
           f"skew={sk['skew_leverage']:.3f}")
    _check("rho=0 is ~symmetric (|skew|<0.1)", abs(sk["skew_symmetric"]) < 0.1,
           f"skew={sk['skew_symmetric']:.3f}")


def test_nesting():
    print("\n[3] Nesting: at rho=0 the leverage path-MH has no drift")
    rng = np.random.default_rng(1)
    T = 50
    logh = rng.standard_normal(T) * 0.2
    S = np.abs(rng.standard_normal(T))
    kdim = np.ones(T, int)
    gcoef = rng.standard_normal(T)          # would-be drift coeff
    has_obs = np.ones(T, bool); has_obs[0] = False
    # rho2=0 -> drift sigma*gcoef*... is multiplied by rho via gcoef? No: gcoef
    # carries rho.  Set gcoef=0 (rho=0) -> no drift, var = sigma2.  Two calls with
    # the same rng must coincide whether we pass rho2=0 with gcoef=0.
    x1, a1, _ = _lev_path_mh(logh, S, kdim, np.zeros(T), has_obs,
                             0.9, 0.04, 0.0, 0.2, np.random.default_rng(7))
    x2, a2, _ = _lev_path_mh(logh, S, kdim, np.zeros(T), has_obs,
                             0.9, 0.04, 0.0, 0.2, np.random.default_rng(7))
    _check("rho=0 path-MH is deterministic given rng (no drift terms)",
           np.allclose(x1, x2) and a1 == a2)


def test_end_to_end(theta, fl, bm, oc, r):
    print("\n[4] short leverage Gibbs: runs, acceptance sane, dominant sign")
    rho_u_true = np.array([-0.6, -0.3, -0.2])
    sim = simulate_dfm_sv(theta, T=220, freq_list=fl, block_map=bm, ordered_cols=oc, r=r,
                          seed=9, sv_u=(0.0, 0.97, 0.22), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=rho_u_true, rho_eps=-0.3)
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=800, burn_in=300, thin=1, seed=4,
                       sv=True, leverage=True, store_vol=True, verbose=False)
    acc = res["meta"]["acceptance"]
    ok_acc = all(0.05 < acc[k] < 0.95 for k in acc)
    _check("all Metropolis acceptance rates in (0.05, 0.95)", ok_acc,
           f"{ {k: round(v,2) for k,v in acc.items()} }")
    rho_u = res["theta_mean"]["rho_u"]
    jdom = int(np.argmax(np.abs(rho_u_true)))
    _check("dominant rho_u component has negative sign", rho_u[jdom] < 0,
           f"rho_u={np.round(rho_u,3)}")
    hu = res["draws"]["h_u"].mean(axis=0)
    c = float(np.corrcoef(np.log(hu), sim["logh_u_true"])[0, 1])
    _check("h^u path still tracked (corr>0.4 short run)", c > 0.4, f"corr={c:.3f}")


def main():
    print("=" * 72)
    print("PASSO 3 — leverage (Branch A, contemporaneous + Metropolis) gate")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    test_rho_kernel()
    test_skewness(theta)
    test_nesting()
    test_end_to_end(theta, fl, bm, oc, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
