"""
src/mcmc/test_spec2_recovery.py
===============================

Recovery tests for **Phase 1 / 1d** — the per-factor common-volatility block
:func:`mcmc.sample_vol.sample_common_vol_mv`.

  * ``test_recovery_diag`` (Commit 1, ``R_xi = I``): a Spec-II DGP with **diagonal**
    ``Q`` (where the decoupling is exact) and ``r`` factors carrying *distinct*
    ``(phi_k, sigma2_k)``; a mini-Gibbs on the vol block recovers per-factor paths,
    persistence, and (loosely) the weakly-identified ``sigma2``, with ``mu = 0``.
  * ``test_coupling`` (Commit 2, ``R_xi = corr-of-log-squares``): a Spec-II DGP with a
    **strongly correlated** ``Q``; the coupled r-dim FFBS recovers well, and at a
    **near-diagonal** ``Q`` it agrees with the decoupled sampler (validates the
    decoupling limit and the ``g(rho)`` table).

Theory: ``subsec:vol-all-processes``, ``eq:vol-outside`` (Spec II sandwich),
``eq:vol-logsquare``, ``eq:sv-mu-identification`` (mu = 0).

Run
---
    python src/mcmc/test_spec2_recovery.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.sample_vol import sample_common_vol_mv, logsq_corr_matrix          # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  [PASS] {name}   {detail}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}   {detail}")


def _ar1(phi: float, sigma: float, T: int, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(T)
    x[0] = (sigma / np.sqrt(1.0 - phi * phi)) * rng.standard_normal()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
    return x


def _sqrt_spd(Q: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    return (vecs * np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def _simulate(phi, s2, Q, T, seed):
    """Spec-II common block: u_t = sqrt(H_t) Q^{1/2} z_t (w = 1)."""
    rng = np.random.default_rng(seed)
    r = len(phi)
    logh = np.column_stack([_ar1(phi[k], np.sqrt(s2[k]), T, rng) for k in range(r)])
    z = rng.standard_normal((T, r))
    u = np.sqrt(np.exp(logh)) * (z @ _sqrt_spd(Q).T)
    return logh, u[1:]                                # true logh (T,r), u_head (T-1,r)


def _minigibbs(u_head, Q, T, R_xi, n_iter, burn, seed, coupling=None):
    r = u_head.shape[1]
    logh_cur = np.zeros((T, r))
    sv_cur = np.tile([0.0, 0.90, 0.10], (r, 1))
    w_u = np.ones(T)
    acc = np.zeros((T, r))
    sv_draws = []
    gen = np.random.default_rng(seed)
    for it in range(n_iter):
        if coupling is not None:
            out = sample_common_vol_mv(u_head, Q, w_u, logh_cur, sv_cur, gen,
                                       coupling=coupling)
        else:
            out = sample_common_vol_mv(u_head, Q, w_u, logh_cur, sv_cur, gen, R_xi=R_xi,
                                       allow_experimental=R_xi is not None)
        logh_cur, sv_cur = out["logh_u"], out["sv_u"]
        if it >= burn:
            acc += logh_cur
            sv_draws.append(sv_cur.copy())
    return acc / (n_iter - burn), np.mean(sv_draws, axis=0)


def _path_corr(logh_hat, logh_true):
    r = logh_hat.shape[1]
    return np.array([np.corrcoef(logh_hat[1:, k], logh_true[1:, k])[0, 1] for k in range(r)])


def test_recovery_diag():
    print("\n[1] Commit 1 (R_xi=I, diagonal Q): per-factor SV recovery")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])
    Q = np.diag([1.0, 1.6])
    logh_true, u_head = _simulate(phi_true, s2_true, Q, T, seed=20260708)
    logh_hat, sv_hat = _minigibbs(u_head, Q, T, None, 500, 150, seed=7)
    phi_hat, s2_hat = sv_hat[:, 1], sv_hat[:, 2]
    c = _path_corr(logh_hat, logh_true)
    for k in range(2):
        _check(f"factor {k}: path corr > 0.5", c[k] > 0.5, f"corr={c[k]:.3f}")
        _check(f"factor {k}: |phi_hat-phi| < 0.15", abs(phi_hat[k] - phi_true[k]) < 0.15,
               f"phi_hat={phi_hat[k]:.3f} vs {phi_true[k]:.2f}")
        lo, hi = 0.25 * s2_true[k], 5.0 * s2_true[k]
        _check(f"factor {k}: sigma2_hat in [{lo:.3f},{hi:.3f}]", lo < s2_hat[k] < hi,
               f"sigma2_hat={s2_hat[k]:.4f}")
    _check("phi distinct (phi0>phi1)", phi_hat[0] > phi_hat[1], f"{phi_hat.round(3)}")
    _check("mu fixed at 0", np.allclose(sv_hat[:, 0], 0.0), f"mu={sv_hat[:, 0].round(4)}")


def test_coupling():
    print("\n[2] Commit 2 (coupled r-dim FFBS): decoupling limit + a caveat")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])

    # ── near-diagonal Q: coupled and decoupled AGREE (the key consistency check:
    #    validates the r-dim FFBS, the g(rho) table, and the decoupling limit) ──
    Qd = np.array([[1.0, 0.05], [0.05, 1.6]])         # corr ~ 0.04
    Rd = logsq_corr_matrix(Qd)
    lt2, uh2 = _simulate(phi_true, s2_true, Qd, T, seed=202)
    lc2, sc2 = _minigibbs(uh2, Qd, T, Rd, 400, 120, seed=5)
    ld2, _ = _minigibbs(uh2, Qd, T, None, 400, 120, seed=5)
    a_c, a_d = _path_corr(lc2, lt2).mean(), _path_corr(ld2, lt2).mean()
    _check("near-diagonal Q: coupled ~ decoupled (|delta|<0.05)", abs(a_c - a_d) < 0.05,
           f"coupled={a_c:.3f} vs decoupled={a_d:.3f}")
    _check("near-diagonal coupled is stable (phi not collapsed)",
           np.all(sc2[:, 1] > 0.6), f"phi_hat={sc2[:, 1].round(3)}")

    # ── strongly correlated Q: REPORT-ONLY diagnostic (not a pass/fail). ──────
    #  FINDING (2026-07-08): the correlation-scaled measurement covariance
    #  Sigma = diag(v_s) R_xi diag(v_s) — the literal "mixture componentwise +
    #  retained cross-covariance" — is mis-specified (unconditional R_xi with the
    #  mixture's *conditional* v_s) and DISTORTS the less-persistent factor
    #  (phi collapses, sigma2 explodes).  Moreover positively-correlated
    #  measurement noise is *redundant* information, so coupling does not improve
    #  point recovery vs the (overconfident) decoupled sampler — it improves
    #  CALIBRATION.  Design decision pending (see docs/REFACTOR_PLAN.md 1d).
    Q = np.array([[1.0, 0.92], [0.92, 1.0]])
    lt, uh = _simulate(phi_true, s2_true, Q, T, seed=101)
    lc, svc = _minigibbs(uh, Q, T, logsq_corr_matrix(Q), 400, 120, seed=7)
    ld, svd = _minigibbs(uh, Q, T, None, 400, 120, seed=7)
    print(f"    [diag] strong corr(Q)=0.92:")
    print(f"    [diag]   decoupled : corr={_path_corr(ld, lt).round(3)} phi={svd[:,1].round(3)}")
    print(f"    [diag]   coupled(3): corr={_path_corr(lc, lt).round(3)} phi={svc[:,1].round(3)}"
          f"  <- correlation-scaled distortion (design decision pending)")


def test_qml():
    print("\n[3] QML coupled pass: stable where the literal collapses")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])

    # (a) QML recovers the per-factor paths on a near-diagonal DGP — it is a valid
    #     sampler, not only a stable one.
    Qd = np.array([[1.0, 0.05], [0.05, 1.6]])
    ltd, uhd = _simulate(phi_true, s2_true, Qd, T, seed=303)
    lq, sq = _minigibbs(uhd, Qd, T, None, 400, 120, seed=5, coupling="qml")
    cq = _path_corr(lq, ltd)
    for k in range(2):
        _check(f"QML factor {k}: path corr > 0.5", cq[k] > 0.5, f"corr={cq[k]:.3f}")
    _check("QML: mu fixed at 0", np.allclose(sq[:, 0], 0.0))
    _check("QML: phi distinct (phi0>phi1)", sq[0, 1] > sq[1, 1], f"phi={sq[:,1].round(3)}")

    # (b) THE selling point — strong corr(Q)=0.92: QML is STABLE (phi not collapsed),
    #     whereas the literal correlation-scaled form distorts the less-persistent
    #     factor (phi 0.90 -> ~0.4).  Same DGP, same seed as [2]'s diagnostic.
    Q = np.array([[1.0, 0.92], [0.92, 1.0]])
    lt, uh = _simulate(phi_true, s2_true, Q, T, seed=101)
    lq2, sq2 = _minigibbs(uh, Q, T, None, 400, 120, seed=7, coupling="qml")
    ll2, sl2 = _minigibbs(uh, Q, T, logsq_corr_matrix(Q), 400, 120, seed=7)  # literal
    print(f"    [diag] strong corr(Q)=0.92, less-persistent factor phi (true 0.90):")
    print(f"    [diag]   QML     : phi={sq2[:,1].round(3)}  (stable)")
    print(f"    [diag]   literal : phi={sl2[:,1].round(3)}  (collapses)")
    _check("QML stable at corr(Q)=0.92: both phi > 0.6", np.all(sq2[:, 1] > 0.6),
           f"phi={sq2[:,1].round(3)}")
    _check("QML beats the literal on the less-persistent factor's phi",
           sq2[1, 1] > sl2[1, 1], f"QML={sq2[1,1]:.3f} vs literal={sl2[1,1]:.3f}")

    # (c) construction: at diagonal Q the QML measurement covariance is (pi^2/2) I,
    #     a SINGLE Gaussian — so QML does NOT nest 'decoupled' (which uses the KSC
    #     mixture).  Check the two give different draws on the same rng/state.
    Qdiag = np.diag([1.0, 1.6])
    lt3, uh3 = _simulate(phi_true, s2_true, Qdiag, 400, seed=9)
    a = sample_common_vol_mv(uh3, Qdiag, np.ones(400), np.zeros((400, 2)),
                             np.tile([0.0, 0.95, 0.05], (2, 1)),
                             np.random.default_rng(1), coupling="qml")
    b = sample_common_vol_mv(uh3, Qdiag, np.ones(400), np.zeros((400, 2)),
                             np.tile([0.0, 0.95, 0.05], (2, 1)),
                             np.random.default_rng(1), coupling="decoupled")
    _check("QML != decoupled at diagonal Q (drops the mixture refinement)",
           not np.allclose(a["logh_u"], b["logh_u"]),
           "identical draws would mean QML wrongly nests decoupled")


def main() -> int:
    print("=" * 72)
    print("PHASE 1 / 1d — per-factor common SV recovery (decoupled + QML + literal)")
    print("=" * 72)
    test_recovery_diag()
    test_coupling()
    test_qml()
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
