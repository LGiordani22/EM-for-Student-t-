"""
src/mcmc/test_diagnostics.py
============================

Gate for the **audit metrics** of ``docs/audit_P1-P5.md`` — the three pure
functions of ``Q`` (and of the draws) that make P1, P4 and P5 *measurable* at every
run instead of asserted once:

  [1] ``posterior_corr_Q``  — the single number that closes all three: the decoupled
      common-volatility block is **exact** at diagonal ``Q``, so both the coupling's
      benefit (P4) and the Branch-B leverage attenuation (P5) are second-order in the
      off-diagonals of ``corr(Q)``.  It is also the Huang--Wand robustness check the
      thesis prescribes (``.tex`` ~20684).

  [2] ``coupling_overconfidence`` — P4.  Ignoring the log-square cross-correlation
      credits the sampler with more independent information than there is.  Verified
      against the known scale: second-order in ``corr(Q)``, ``+0.4%`` at ``0.10``.

  [3] ``leverage_whitening_attenuation`` — P5.  The closed form
      ``lambda(c) = (2/pi)(c*arcsin c + sqrt(1-c^2))`` is checked **against Monte
      Carlo on the actual definition** ``E[|z_k| |zbar_k|]`` — not against itself.
      Also: ``lambda`` is an attenuation, never a sign flip.

Run
---
    python src/mcmc/test_diagnostics.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.diagnostics import (                                            # noqa: E402
    coupling_overconfidence,
    leverage_whitening_attenuation,
    posterior_corr_Q,
)

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _sqrtm(Q):
    v, V = np.linalg.eigh(0.5 * (Q + Q.T))
    return (V * np.sqrt(np.clip(v, 0, None))) @ V.T


def _equicorr(r, rho, scales=None):
    Q = np.full((r, r), float(rho))
    np.fill_diagonal(Q, 1.0)
    if scales is not None:
        d = np.asarray(scales, float)
        Q = Q * np.outer(d, d)
    return Q


# ─────────────────────────────────────────────────────────────────────────────

def test_posterior_corr_Q():
    print("\n[1] posterior_corr_Q — closes P1/P4/P5 from the draws")
    rng = np.random.default_rng(0)
    r, n = 3, 500
    C_true = _equicorr(r, 0.4)
    # draws of Q with heterogeneous scales: corr must be scale-invariant
    scales = np.array([0.2, 1.0, 3.0])
    Qs = np.array([_equicorr(r, 0.4, scales) * (1.0 + 0.02 * rng.standard_normal())
                   for _ in range(n)])
    out = posterior_corr_Q({"Q": Qs})
    _check("corr is scale-invariant (heterogeneous diag(Q))",
           np.allclose(out["mean"], C_true, atol=1e-8),
           f"mean=\n{np.round(out['mean'],4)}")
    _check("max_offdiag reads the true correlation",
           abs(out["max_offdiag"] - 0.4) < 1e-8, f"{out['max_offdiag']:.6f}")
    _check("credible band brackets the mean",
           np.all(out["lo"] <= out["mean"] + 1e-12) and np.all(out["hi"] >= out["mean"] - 1e-12))
    _check("diagonal is exactly 1", np.allclose(np.diag(out["mean"]), 1.0))

    # a diagonal Q reports zero coupling: the decoupled block is exact there
    out0 = posterior_corr_Q({"Q": np.array([np.diag([0.2, 1.0, 3.0])] * 10)})
    _check("diagonal Q -> max_offdiag == 0 (decoupled block exact)",
           out0["max_offdiag"] == 0.0)


def test_coupling_overconfidence():
    print("\n[2] coupling_overconfidence — P4, second-order in corr(Q)")
    # Theory: g(0) = 0 exactly, so R_xi = I at diagonal Q.  In code g(rho) is
    # tabulated by common-random-number Monte Carlo with n_mc = 3e6, whose standard
    # error on a correlation is ~1/sqrt(n_mc) = 5.8e-4; the table returns
    # g(0)/(pi^2/2) = -4.2e-4, i.e. one MC sd from zero.  That noise floor — not
    # exact zero — is what "decoupled == coupled at diagonal Q" means numerically.
    d = coupling_overconfidence(np.diag([0.2, 1.0, 3.0]))
    _MC_FLOOR = 1e-3
    _check("diagonal Q -> R_xi = I and zero overconfidence (to the MC-table floor)",
           d["max_R_xi"] < _MC_FLOOR and abs(d["overconfidence"]) < _MC_FLOOR,
           f"max_R_xi={d['max_R_xi']:.2e}, oc={d['overconfidence']:.2e}")

    vals = {rho: coupling_overconfidence(_equicorr(3, rho)) for rho in
            (0.05, 0.10, 0.30, 0.50, 0.90)}
    oc = {k: v["overconfidence"] for k, v in vals.items()}
    _check("monotone in corr(Q)",
           all(oc[a] < oc[b] for a, b in zip([0.05, 0.10, 0.30, 0.50], [0.10, 0.30, 0.50, 0.90])),
           f"{ {k: round(v,4) for k,v in oc.items()} }")
    # the numbers quoted in the audit
    _check("corr(Q)=0.10 -> overconfidence ~ 0.4% (the real panel)",
           0.002 < oc[0.10] < 0.007, f"{100*oc[0.10]:.2f}%")
    _check("corr(Q)=0.90 -> overconfidence ~ 40% (why it would matter, if it were true)",
           0.35 < oc[0.90] < 0.50, f"{100*oc[0.90]:.1f}%")
    # second order: doubling corr(Q) from 0.05 more than doubles R_xi
    _check("R_xi is second-order in corr(Q)",
           vals[0.10]["max_R_xi"] > 3.0 * vals[0.05]["max_R_xi"],
           f"{vals[0.05]['max_R_xi']:.5f} -> {vals[0.10]['max_R_xi']:.5f}")

    # scale invariance: only corr(Q) matters, not diag(Q)
    a = coupling_overconfidence(_equicorr(3, 0.5))
    b = coupling_overconfidence(_equicorr(3, 0.5, scales=[0.1, 2.0, 5.0]))
    _check("depends on corr(Q) only, not on diag(Q)",
           abs(a["overconfidence"] - b["overconfidence"]) < 1e-9)


def test_leverage_whitening_attenuation():
    print("\n[3] leverage_whitening_attenuation — P5, closed form vs Monte Carlo")
    lam0 = leverage_whitening_attenuation(np.diag([0.2, 1.0, 3.0]))
    _check("diagonal Q -> c = 1, lambda = 1 (no attenuation, Branch B exact)",
           np.allclose(lam0["c"], 1.0, atol=1e-12) and np.allclose(lam0["lambda"], 1.0, atol=1e-12),
           f"c={lam0['c']}, lam={lam0['lambda']}")

    # MC on the DEFINITION: lambda_k = E[|z_k| * |zbar_k|],  zbar = Q^{1/2}z / sqrt(q_kk)
    rng = np.random.default_rng(7)
    N = 300_000
    for rho in (0.3, 0.6, 0.9):
        Q = _equicorr(3, rho, scales=[0.5, 1.0, 2.0])   # scales must not matter
        Qh = _sqrtm(Q)
        z = rng.standard_normal((N, 3))
        zbar = (z @ Qh.T) / np.sqrt(np.diag(Q))
        mc = np.array([np.mean(np.abs(z[:, k]) * np.abs(zbar[:, k])) for k in range(3)])
        cf = leverage_whitening_attenuation(Q)["lambda"]
        _check(f"corr(Q)={rho}: closed form == E[|z||zbar|] (MC)",
               np.max(np.abs(mc - cf)) < 0.01,
               f"closed={np.round(cf,4)}  mc={np.round(mc,4)}")

    # properties the audit relies on
    lam_hi = leverage_whitening_attenuation(_equicorr(3, 0.8))["lambda"]
    lam_lo = leverage_whitening_attenuation(_equicorr(3, 0.2))["lambda"]
    _check("attenuation grows with corr(Q)", np.all(lam_hi < lam_lo))
    _check("always an attenuation, never > 1 (no sign flip, never amplification)",
           np.all(lam_hi <= 1.0) and np.all(lam_lo <= 1.0) and np.all(lam_hi > 0.63),
           f"lam(0.8)={np.round(lam_hi,4)}")
    _check("corr(Q)=0.10 -> bias on rho is under 0.5% (the real panel)",
           np.all(np.abs(leverage_whitening_attenuation(_equicorr(3, 0.10))["bias_pct"]) < 0.5))
    _check("bias_pct is (lambda-1)*100",
           np.allclose(lam0["bias_pct"], 100.0 * (lam0["lambda"] - 1.0)))


def test_real_panel():
    print("\n[4] the real panel: corr(Q) is near-diagonal => P1/P4/P5 all negligible")
    try:
        from mcmc.gibbs import load_warm_init
        w = load_warm_init("small")
    except Exception as e:                              # config not on this machine
        print(f"  [SKIP] warm init unavailable ({type(e).__name__})")
        return
    Q = np.asarray(w["theta"]["Q"], float)
    oc = coupling_overconfidence(Q)
    lam = leverage_whitening_attenuation(Q)
    print(f"         max R_xi = {oc['max_R_xi']:.5f}   overconfidence = {100*oc['overconfidence']:.2f}%"
          f"   max|bias rho| = {np.max(np.abs(lam['bias_pct'])):.2f}%")
    _check("P4: coupling would buy < 2% of calibration on the EM Q",
           oc["overconfidence"] < 0.02, f"{100*oc['overconfidence']:.2f}%")
    _check("P5: Branch-B rho attenuation < 1% on the EM Q",
           np.max(np.abs(lam["bias_pct"])) < 1.0,
           f"{np.max(np.abs(lam['bias_pct'])):.3f}%")


def main():
    print("=" * 72)
    print("AUDIT METRICS — P1 / P4 / P5 made measurable (docs/audit_P1-P5.md)")
    print("=" * 72)
    test_posterior_corr_Q()
    test_coupling_overconfidence()
    test_leverage_whitening_attenuation()
    test_real_panel()
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
