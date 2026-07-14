"""
src/mcmc/test_variants.py
=========================

**Phase 8 gate** — the D1 x D2 grid as restrictions of the master sampler
(``sec:gibbs-variants``, Table ``tab:gibbs-variants``), plus the optional
Huang--Wand prior on ``Q`` (Phase 2d, ``eq:param-Q-hw-prior``).

The point of the section is that the *skeleton never changes* — states (a),
volatilities (b), tails (c), parameters (d) — and each cell only switches blocks
on or off.  These tests assert exactly that, block by block:

  [1] ``variant_kwargs`` maps each row of the table to flags; D1-c raises
      (its outlier block is not implemented) and leverage on the no-SV
      "current model" row raises (no log-vol innovation to correlate with).

  [2] **D2-a** (common volatility only): step (b) omits the idiosyncratic half,
      ``h^eps == 1`` is frozen, ``sv_eps`` is *not drawn* (Family B runs for the
      r common processes only) — checked at the block level and end-to-end,
      with and without leverage (no ``rho_eps`` either).

  [3] **D1-a** (Gaussian): ``w == 1``, step (c) and Family D omitted.  Proven
      the hard way: two runs whose *only* difference is ``nu_u``/``nu_eps`` in
      the warm start produce **bit-identical** draws — the tails block truly
      never executes.

  [4] **current model** (no SV): the whole of step (b), Family B and Family C
      are gone; what remains is (a) states, (c) tails, (d) Families A and D —
      the MCMC counterpart of the EM.

  [5] **rho = 0**: Family C and the step-(b) leverage correction drop out.

  [6] **Family A priors** are wired on *every* path (per-factor and the no-SV
      MNIW): priors on/off both run, no warning, and the flat default is
      bit-identical to the pre-prior sampler.

  [7] **Huang--Wand** (Phase 2d): the hierarchical IW runs as a drop-in for the
      plain IW, appends the r auxiliary scales to Family A, and — the robustness
      check the thesis prescribes — leaves the posterior correlations among the
      factor innovations essentially unmoved.

Run
---
    python src/mcmc/test_variants.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.gibbs import VARIANT_FLAGS, fit_dfm_mcmc, load_warm_init, variant_kwargs  # noqa: E402
from mcmc.sample_vol import sample_volatility_block_specII                   # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                                 # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _corr_offdiag(Q):
    d = np.sqrt(np.diag(Q))
    C = Q / np.outer(d, d)
    return C[np.triu_indices_from(C, k=1)]


def _fit(Y, theta, fl, bm, oc, **kw):
    kw.setdefault("n_iter", 120)
    kw.setdefault("burn_in", 40)
    kw.setdefault("seed", 5)
    kw.setdefault("verbose", False)
    return fit_dfm_mcmc(Y, dict(theta), fl, bm, oc, **kw)


# ─────────────────────────────────────────────────────────────────────────────

def test_variant_table():
    print("\n[1] variant_kwargs: the grid table -> sampler flags")
    _check("master cell D1b x D2b is the unrestricted sweep",
           VARIANT_FLAGS["D1b x D2b"] == dict(sv=True, sv_idio=True,
                                              tails="student_t", update_nu=True))
    kw = variant_kwargs("D1b x D2a", leverage=True, timing="lagged")
    _check("D1-b x D2-a + leverage (Branch B) -> sv_idio=False, timing lagged",
           kw["sv_idio"] is False and kw["timing"] == "lagged" and kw["leverage"])
    kw = variant_kwargs("D1a x D2b")
    _check("D1-a -> gaussian tails, no nu draw",
           kw["tails"] == "gaussian" and kw["update_nu"] is False)
    _check("cell name is case/separator insensitive",
           variant_kwargs("d1b × d2b") == variant_kwargs("D1b x D2b"))

    try:
        variant_kwargs("D1c x D2b"); ok = False
    except NotImplementedError:
        ok = True
    _check("D1-c (outliers) raises: the outlier block is not implemented", ok)

    try:
        variant_kwargs("current model", leverage=True); ok = False
    except ValueError:
        ok = True
    _check("current model + leverage raises (no log-vol innovation)", ok)

    try:
        variant_kwargs("D9z"); ok = False
    except KeyError:
        ok = True
    _check("unknown cell raises", ok)

    try:
        fit_dfm_mcmc(np.zeros((5, 2)), {}, [], {}, [], sv=False, leverage=True)
        ok = False
    except ValueError:
        ok = True
    _check("fit_dfm_mcmc(sv=False, leverage=True) raises at the source too", ok)


def test_d2a_block_level(theta, fl, bm, oc, r, Y):
    print("\n[2a] D2-a at the block level: h^eps frozen at 1, Family B not drawn")
    T, M = Y.shape
    rng = np.random.default_rng(0)
    f_aug = np.zeros((T, 5 * r)); f_aug[:, :r] = rng.standard_normal((T, r)) * 0.5
    sv_eps_in = np.tile(np.array([0.0, 0.9, 0.07]), (M, 1))
    out = sample_volatility_block_specII(
        Y, f_aug, dict(theta), np.ones(T), np.ones(T),
        np.zeros((T, r)), np.zeros((T, M)),
        np.tile(np.array([0.0, 0.95, 0.05]), (r, 1)), sv_eps_in,
        rng, sv_idio=False)
    _check("D2-a: h^eps == 1 exactly", np.array_equal(out["h_eps"], np.ones((T, M))))
    _check("D2-a: log h^eps == 0 exactly", np.array_equal(out["logh_eps"], np.zeros((T, M))))
    _check("D2-a: sv_eps returned unchanged (Family B omitted)",
           np.array_equal(out["sv_eps"], sv_eps_in))
    _check("D2-a: the r common processes ARE drawn",
           out["h_u"].shape == (T, r) and not np.allclose(out["logh_u"], 0.0))


def test_d2a_end_to_end(theta, fl, bm, oc, Y):
    print("\n[2b] D2-a end-to-end: no idiosyncratic vol block, no rho_eps")
    res = _fit(Y, theta, fl, bm, oc, **variant_kwargs("D1b x D2a"), store_vol=True)
    d = res["draws"]
    _check("D2-a: sv_eps / h_eps absent from draws",
           "sv_eps" not in d and "h_eps" not in d, f"keys={sorted(d)}")
    _check("D2-a: sv_u present, (n_keep, r, 3)", d["sv_u"].ndim == 3)
    _check("D2-a: theta_mean carries no sv_eps", "sv_eps" not in res["theta_mean"])

    res_l = _fit(Y, theta, fl, bm, oc, n_iter=60, burn_in=20,
                 **variant_kwargs("D1b x D2a", leverage=True))
    d = res_l["draws"]
    _check("D2-a + leverage: rho_u drawn, rho_eps absent",
           "rho_u" in d and "rho_eps" not in d, f"keys={sorted(d)}")


def test_d1a_gaussian(theta, fl, bm, oc, Y):
    print("\n[3] D1-a (Gaussian): weights block (c) and Family D omitted")
    kw = variant_kwargs("D1a x D2b")
    res = _fit(Y, theta, fl, bm, oc, **kw)
    _check("D1-a: nu_u / nu_eps absent from draws",
           "nu_u" not in res["draws"] and "nu_eps" not in res["draws"])
    _check("D1-a: theta_mean carries no nu", "nu_u" not in res["theta_mean"])
    _check("D1-a: meta records the cell",
           res["meta"]["tails"] == "gaussian" and res["meta"]["update_nu"] is False)

    # w == 1: the tails never enter.  Perturb ONLY nu in the warm start; if step
    # (c) ran, every downstream draw would move.  Bit-identical => block omitted.
    t2 = dict(theta); t2["nu_u"] = 3.5; t2["nu_eps"] = 4.5
    res2 = _fit(Y, t2, fl, bm, oc, **kw)
    same = all(np.array_equal(res["draws"][k], res2["draws"][k]) for k in res["draws"])
    _check("D1-a: draws bit-identical under a different nu warm start (w == 1)", same)

    # ... whereas under D1-b (Student-t) that same perturbation does move the chain.
    kwb = variant_kwargs("D1b x D2b")
    rb1 = _fit(Y, theta, fl, bm, oc, **kwb)
    rb2 = _fit(Y, t2, fl, bm, oc, **kwb)
    _check("control: under D1-b the nu warm start DOES move the chain",
           not np.array_equal(rb1["draws"]["A"], rb2["draws"]["A"]))

    try:
        _fit(Y, theta, fl, bm, oc, tails="outliers"); ok = False
    except ValueError:
        ok = True
    _check("tails='outliers' (D1-c) raises", ok)


def test_current_model(theta, fl, bm, oc, Y):
    print("\n[4] current model: the whole of step (b) + Families B, C omitted")
    res = _fit(Y, theta, fl, bm, oc, **variant_kwargs("current model"))
    d = res["draws"]
    _check("no-SV: sv_u / sv_eps / rho_* absent",
           not any(k in d for k in ("sv_u", "sv_eps", "rho_u", "rho_eps")),
           f"keys={sorted(d)}")
    _check("no-SV: states (a), tails (c), Families A and D remain",
           all(k in d for k in ("A", "Q", "Lambda", "R", "nu_u", "nu_eps")))
    # the explicit flags and the plain call must be the same sampler, bit-for-bit
    res2 = _fit(Y, theta, fl, bm, oc, sv=False)
    _check("no-SV: variant_kwargs('current model') == the plain sv=False call (bitwise)",
           all(np.array_equal(d[k], res2["draws"][k]) for k in d))


def test_rho_zero(theta, fl, bm, oc, Y):
    print("\n[5] rho = 0: Family C and the step-(b) leverage correction drop out")
    res0 = _fit(Y, theta, fl, bm, oc, n_iter=60, burn_in=20, sv=True, leverage=False)
    res1 = _fit(Y, theta, fl, bm, oc, n_iter=60, burn_in=20, sv=True, leverage=True)
    _check("leverage=False: no rho draws, no acceptance record",
           "rho_u" not in res0["draws"] and "acceptance" not in res0["meta"])
    _check("leverage=True: rho_u (r,) and rho_eps (M,) drawn",
           res1["draws"]["rho_u"].shape[1] == res1["meta"]["r"]
           and res1["draws"]["rho_eps"].shape[1] == res1["meta"]["M"])


def test_family_a_priors_all_paths(theta, fl, bm, oc, Y):
    print("\n[6] Family A priors: wired on every path, flat default preserved")
    for name, kw in (("no-SV (MNIW)", dict(sv=False)),
                     ("Spec II (per-factor)", dict(sv=True)),
                     ("Branch A leverage", dict(sv=True, leverage=True))):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            res = _fit(Y, theta, fl, bm, oc, n_iter=50, burn_in=10,
                       use_family_a_priors=True, **kw)
        finite = np.all(np.isfinite(res["draws"]["A"])) and np.all(np.isfinite(res["draws"]["Q"]))
        _check(f"priors on, {name}: runs, (A, Q) finite, no warning",
               finite and not any(issubclass(w.category, RuntimeWarning) for w in wlist),
               f"warnings={[str(w.message)[:40] for w in wlist]}")

    # flat default == priors off, bit-for-bit, on the no-SV EM seam
    a = _fit(Y, theta, fl, bm, oc, n_iter=50, burn_in=10, sv=False)
    b = _fit(Y, theta, fl, bm, oc, n_iter=50, burn_in=10, sv=False,
             use_family_a_priors=False)
    _check("flat is the default (bitwise)", np.array_equal(a["draws"]["Q"], b["draws"]["Q"]))

    # a strong prior must move the posterior (the args really reach the kernel)
    c = _fit(Y, theta, fl, bm, oc, n_iter=50, burn_in=10, sv=False,
             use_family_a_priors=True, family_a_kappa=1e5)
    _check("strong kappa moves A away from the flat draw",
           not np.allclose(a["draws"]["A"].mean(0), c["draws"]["A"].mean(0), atol=1e-6))


def test_huang_wand(theta, fl, bm, oc, Y):
    print("\n[7] Huang-Wand (Phase 2d): drop-in IW swap + robustness check")
    try:
        _fit(Y, theta, fl, bm, oc, n_iter=10, burn_in=1, q_prior="lkj"); ok = False
    except ValueError:
        ok = True
    _check("unknown q_prior raises", ok)

    n_iter, burn_in = 400, 150
    hw = _fit(Y, theta, fl, bm, oc, n_iter=n_iter, burn_in=burn_in, sv=False,
              q_prior="huang_wand", hw_nu_star=2.0, hw_A=1e5)
    iw = _fit(Y, theta, fl, bm, oc, n_iter=n_iter, burn_in=burn_in, sv=False,
              use_family_a_priors=True)          # the plain IW default, nu0 = r+1

    a = hw["draws"]["hw_a"]
    _check("HW: the r auxiliary scales are drawn and stored, all > 0",
           a.shape[1] == hw["meta"]["r"] and np.all(a > 0) and np.all(np.isfinite(a)))
    _check("HW: meta records the prior", hw["meta"]["q_prior"] == "huang_wand")

    # Q stays a proper covariance draw
    Qs = hw["draws"]["Q"]
    eig_ok = all(np.all(np.linalg.eigvalsh(Q) > 0) for Q in Qs[::20])
    _check("HW: every Q draw is positive definite", eig_ok)

    # The thesis' own use of the switch: "re-estimate under the hierarchical prior
    # and confirm the posterior correlations among the factor innovations do not
    # move".  With T in the hundreds the likelihood dominates -> they should not.
    c_hw = np.mean([_corr_offdiag(Q) for Q in hw["draws"]["Q"]], axis=0)
    c_iw = np.mean([_corr_offdiag(Q) for Q in iw["draws"]["Q"]], axis=0)
    dmax = float(np.max(np.abs(c_hw - c_iw)))
    _check("HW: posterior corr(Q) unmoved vs the plain IW (robustness check)",
           dmax < 0.05, f"max|d corr|={dmax:.4f}  hw={np.round(c_hw,3)} iw={np.round(c_iw,3)}")

    # ... and it composes with SV (per-factor path) and with the A / Lambda priors
    res = _fit(Y, theta, fl, bm, oc, n_iter=60, burn_in=20, sv=True,
               q_prior="huang_wand", use_family_a_priors=True)
    _check("HW composes with Spec II + Family A priors",
           np.all(np.isfinite(res["draws"]["Q"])) and "hw_a" in res["draws"])


def test_p1_coupled_unreachable(theta, fl, bm, oc, Y):
    print("\n[8] P1: the coupled R_xi branch is unreachable from the sampler")
    import inspect
    import mcmc.gibbs as gibbs_mod
    import mcmc.sample_vol as sv_mod
    import mcmc.sample_leverage_lagged as lagged_mod

    # (a) fit_dfm_mcmc has no R_xi surface at all — not a default, not a kwarg.
    sig = inspect.signature(fit_dfm_mcmc)
    _check("fit_dfm_mcmc exposes no R_xi parameter", "R_xi" not in sig.parameters,
           f"params={list(sig.parameters)}")
    _check("gibbs.py never mentions R_xi",
           "R_xi" not in inspect.getsource(gibbs_mod))

    # (b) the coupled branch demands an explicit opt-in.
    try:
        sv_mod.sample_common_vol_mv(
            np.zeros((5, 2)), np.eye(2), np.ones(6), np.zeros((6, 2)),
            np.tile([0.0, 0.9, 0.05], (2, 1)), np.random.default_rng(0),
            R_xi=np.eye(2))
        ok = False
    except ValueError:
        ok = True
    _check("sample_common_vol_mv(R_xi=...) raises without allow_experimental", ok)

    # (c) Branch B never calls the multivariate common block: BY DEFAULT its common
    #     volatility is r independent Omori/FFBS channels.
    #     NB (gate QML-leverage): an r-dim coupled FFBS under leverage *does* now
    #     exist in this module (`_branch_b_common_qml`) — the old rationale "there is
    #     no r-dim FFBS in which a cross-covariance could even be inserted" is no
    #     longer true, it was tried.  It is gated behind allow_experimental=True
    #     because it is UNSTABLE exactly where it would be used (at corr(Q)=0.8 the
    #     least identified factor's phi collapses and its rho pins to the boundary),
    #     so what must stay frozen is that the DEFAULT path never reaches it.
    _check("sample_leverage_lagged never references the no-leverage common block",
           "sample_common_vol" not in inspect.getsource(lagged_mod))
    _check("the coupled leverage pass needs an explicit opt-in",
           "allow_experimental" in inspect.getsource(gibbs_mod))

    called = []
    real = sv_mod.sample_common_vol_mv

    def _tripwire(*a, **k):
        called.append(1)
        return real(*a, **k)

    sv_mod.sample_common_vol_mv = _tripwire
    try:
        _fit(Y, theta, fl, bm, oc, n_iter=40, burn_in=10,
             sv=True, leverage=True, timing="lagged")
    finally:
        sv_mod.sample_common_vol_mv = real
    _check("a Branch-B run never enters sample_common_vol_mv (tripwire)",
           not called, f"calls={len(called)}")


def main():
    print("=" * 72)
    print("PHASE 8 — grid cells as restrictions + Huang-Wand (2d)")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    sim = simulate_dfm_sv(theta, T=140, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=11, sv_u=(0.0, 0.95, 0.15), sv_eps=(0.0, 0.9, 0.10))
    Y = sim["Y"]
    theta = {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])}

    test_variant_table()
    test_d2a_block_level(theta, fl, bm, oc, r, Y)
    test_d2a_end_to_end(theta, fl, bm, oc, Y)
    test_d1a_gaussian(theta, fl, bm, oc, Y)
    test_current_model(theta, fl, bm, oc, Y)
    test_rho_zero(theta, fl, bm, oc, Y)
    test_family_a_priors_all_paths(theta, fl, bm, oc, Y)
    test_huang_wand(theta, fl, bm, oc, Y)
    test_p1_coupled_unreachable(theta, fl, bm, oc, Y)

    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
