"""
src/mcmc/test_perfactor_leverage.py
===================================

**Phase 8 / P0** — the test-infrastructure gap closed: a genuine **Spec II +
Option A** DGP (``simulate_dfm_sv(..., sv_u_perfactor=...)``) with ``r``
*independent* per-factor volatilities ``h^u_k`` and ``r`` *scalar* leverage
channels ``rho_k`` (``eq:lev-cond-common``).

Until now the simulator produced only the **scalar-common** restriction
``H^u_t = h^u_t I`` with a *vector* ``rho`` (a DGP at which the inside/outside
placements of ``subsec:vol-placement`` coincide, so it cannot even distinguish the
two specifications), and the two claims the sampler now makes could not be
*expressed* as a test:

  [1] **Channel separation, and where it stops.**  Each per-factor path ``h^u_k``
      is identified — the posterior ``h^u_k`` tracks *its own* true path better
      than any other factor's — **on the channels that carry enough volatility**.
      Under the old scalar-common DGP every factor read the same truth, so this
      was vacuous.  The DGP deliberately includes one **weak** channel
      (unconditional log-vol sd 0.46 against 1.03 and 0.70), and the test asserts
      that at ``T = 600`` it is *not* recovered: that is **P2** — with Spec II each
      ``h^u_k`` is read through one log-square per period, the ``sqrt(r)``
      averaging is gone, so identification needs volatility *and* length.  Pinning
      the boundary is the point: the real-time nowcast runs at short effective
      ``T``, so this is the operational regime, and a future mitigation must show
      up here as a deliberate re-read rather than as a silent pass.

  [2] **Distinct per-factor rho.**  The r scalar leverage channels are separately
      identified: the recovered ``rho_k`` are ordered as the true ones and the
      dominant negative channel is signed correctly.  (The old vector-rho draw
      could not even represent distinct scalar channels.)

  [3] **A/B parity.**  Branch A (contemporaneous, Metropolis) and Branch B
      (lagged, Omori-FFBS) are *not* a restriction of one another — they are the
      two timings of the same master cell (``subsec:variants-leverage-levels``) —
      but each, run on the DGP of its own timing, must recover the same objects.
      They agree on the shared parameters only up to the whitening convention
      (**P5**: A uses the exact full-Q ``|z^u_k|``, B the per-component
      ``|e_k|``), which is why ``Q`` is taken **diagonal** here: at diagonal
      ``Q`` the two whitenings coincide and the comparison is clean.

``Q`` diagonal also makes the decoupled common-vol block exact (Appendix 1d), so
what is being measured is the leverage machinery, not the coupling caveat (P1).

Cost: the per-factor volatilities are read through **one** log-square per period
each (no sqrt(r) averaging — **P2**), so recovery needs a long panel; T is set
accordingly and this file is the slow gate of the suite.

Run
---
    python src/mcmc/test_perfactor_leverage.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.gibbs import fit_dfm_mcmc, load_warm_init                          # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                                 # noqa: E402

_PASS = 0
_FAIL = 0

T_SIM = 600
N_ITER, BURN_IN = 700, 250

# distinct per-factor leverage channels: a dominant negative one, a weak negative
# one, and a positive one (so ordering, not just sign, has to be recovered).
RHO_TRUE = np.array([-0.70, -0.15, 0.45])
SV_U_PF = np.array([[0.0, 0.97, 0.25],
                    [0.0, 0.92, 0.18],
                    [0.0, 0.95, 0.22]])

# Unconditional sd of each log-vol channel, sigma_k / sqrt(1 - phi_k^2) =
# [1.03, 0.46, 0.70]: how much volatility there is *to* estimate.  Channel 1 is
# deliberately the weak one.  Under Spec II each h^u_k is read through **one**
# log-square per period (the sqrt(r) averaging of the scalar-common restriction is
# gone — P2, `.tex` subsec:param-familyB delicate case (a)), so at T = 600 the weak
# channel is **not identified** while the two strong ones are.  The test measures
# that boundary rather than tiptoeing around it: it is the regime the real-time
# nowcast will live in (short effective T), so the threshold is the finding.
STRONG = [0, 2]
WEAK = 1


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _corr(x, y):
    xz, yz = x - x.mean(), y - y.mean()
    return float((xz @ yz) / (np.linalg.norm(xz) * np.linalg.norm(yz)))


def _diagonal_theta(theta):
    """Diagonal Q: the per-component and full-Q whitenings coincide, so Branch A
    and Branch B key their leverage on the *same* shock (P5) and the decoupled
    common-vol block is exact (Appendix 1d)."""
    th = dict(theta)
    th["Q"] = np.diag(np.diag(np.asarray(theta["Q"], float)))
    th["Sigma_0"] = np.asarray(theta["Sigma_0"], float)
    return th


def _simulate(theta, fl, bm, oc, r, timing, seed):
    return simulate_dfm_sv(
        theta, T=T_SIM, freq_list=fl, block_map=bm, ordered_cols=oc, r=r, seed=seed,
        sv_u=(0.0, 0.95, 0.0),                 # ignored: sv_u_perfactor takes over
        sv_u_perfactor=SV_U_PF, sv_eps=(0.0, 0.94, 0.12),
        rho_u=RHO_TRUE, rho_eps=-0.3, timing=timing,
    )


def _fit(sim, theta, fl, bm, oc, timing, seed):
    return fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc,
                        n_iter=N_ITER, burn_in=BURN_IN, seed=seed,
                        sv=True, leverage=True, timing=timing,
                        store_vol=True, verbose=False)


def test_dgp_shape(sim, r):
    print("\n[0] the Spec II DGP: r independent per-factor volatilities")
    lh = sim["logh_u_true"]
    _check("logh_u_true is (T, r)", lh.shape == (T_SIM, r), f"shape={lh.shape}")
    off = [abs(_corr(lh[:, j], lh[:, k])) for j in range(r) for k in range(j + 1, r)]
    _check("the r true paths are independent (|corr| < 0.25)", max(off) < 0.25,
           f"max|corr|={max(off):.3f}")
    sd = lh.std(axis=0)
    _check("per-factor unconditional sd ordered as sigma_k/sqrt(1-phi_k^2)",
           np.argmax(sd) == int(np.argmax(SV_U_PF[:, 2] / np.sqrt(1 - SV_U_PF[:, 1] ** 2))),
           f"sd={np.round(sd,3)}")


def test_recovery(res, sim, r, branch):
    print(f"\n[{branch}] per-factor recovery")
    lhu = np.log(res["draws"]["h_u"].mean(axis=0))        # (T, r) posterior mean
    truth = sim["logh_u_true"]

    # [1] channel separation, on the channels the data can actually speak about
    C = np.array([[_corr(lhu[:, k], truth[:, j]) for j in range(r)] for k in range(r)])
    own = np.diag(C)
    cross = C - np.diag(np.diag(C))
    _check(f"{branch}: the STRONG channels track their OWN true path (corr > 0.6)",
           all(own[k] > 0.6 for k in STRONG), f"own={np.round(own,3)}")
    _check(f"{branch}: STRONG channels: own corr beats every cross corr",
           all(own[k] > np.max(np.abs(cross[k])) for k in STRONG),
           f"own={np.round(own,3)}  cross_max={np.round(np.abs(cross).max(1),3)}")
    # P2, measured: one log-square per period per factor => the weak channel
    # (unconditional sd 0.46) is NOT identified at this T.  Asserted as a *known
    # boundary*, so that a future mitigation (longer T, tighter sigma_eta prior,
    # pooling across factors) shows up here as a failure to be re-read, not as a
    # silent improvement.
    _check(f"{branch}: P2 boundary - the WEAK channel is under-identified at "
           f"T={T_SIM} (own corr < 0.6)", own[WEAK] < 0.6,
           f"own[{WEAK}]={own[WEAK]:.3f} — if this now PASSES the boundary moved, "
           f"re-read P2 before tightening")

    # [2] distinct per-factor rho: ordering and the dominant sign
    rho = res["theta_mean"]["rho_u"]
    _check(f"{branch}: rho_k ordering recovered (argsort matches truth)",
           np.array_equal(np.argsort(rho), np.argsort(RHO_TRUE)),
           f"rho={np.round(rho,3)} true={RHO_TRUE}")
    jdom = int(np.argmax(np.abs(RHO_TRUE)))
    _check(f"{branch}: dominant channel rho_{jdom} recovered negative",
           rho[jdom] < 0, f"rho={np.round(rho,3)}")
    _check(f"{branch}: the positive channel is recovered positive",
           rho[int(np.argmax(RHO_TRUE))] > 0, f"rho={np.round(rho,3)}")

    # persistence, and the mu = 0 identification
    phi = res["theta_mean"]["sv_u"][:, 1]
    _check(f"{branch}: phi_k persistent on the STRONG channels (> 0.85)",
           all(phi[k] > 0.85 for k in STRONG), f"phi={np.round(phi,3)}")
    _check(f"{branch}: mu == 0 in every draw (identification)",
           np.all(res["draws"]["sv_u"][:, :, 0] == 0.0))
    return rho, phi


def test_parity(rho_a, rho_b):
    print("\n[3] A/B parity on the shared leverage parameters (diagonal Q)")
    _check("Branch A and Branch B agree on the rho ordering",
           np.array_equal(np.argsort(rho_a), np.argsort(rho_b)),
           f"A={np.round(rho_a,3)}  B={np.round(rho_b,3)}")
    d = float(np.max(np.abs(rho_a - rho_b)))
    _check("Branch A and Branch B agree on rho within MC error (max|d| < 0.35)",
           d < 0.35, f"A={np.round(rho_a,3)}  B={np.round(rho_b,3)}  max|d|={d:.3f}")


def main():
    print("=" * 72)
    print("PHASE 8 / P0 — per-factor leverage DGP: channels, distinct rho, A/B")
    print("=" * 72)
    w = load_warm_init("small")
    theta = _diagonal_theta(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]

    sim_a = _simulate(theta, fl, bm, oc, r, "contemporaneous", seed=21)
    test_dgp_shape(sim_a, r)

    print(f"\n  (T={T_SIM}, n_iter={N_ITER}: the per-factor sqrt(r) data cost, P2)")
    sim_b = _simulate(theta, fl, bm, oc, r, "lagged", seed=22)
    res_b = _fit(sim_b, theta, fl, bm, oc, "lagged", seed=7)
    rho_b, _ = test_recovery(res_b, sim_b, r, "Branch B (lagged, Omori-FFBS)")

    res_a = _fit(sim_a, theta, fl, bm, oc, "contemporaneous", seed=7)
    rho_a, _ = test_recovery(res_a, sim_a, r, "Branch A (contemp., Metropolis)")

    test_parity(rho_a, rho_b)

    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
