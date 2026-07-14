"""
src/mcmc/test_passo4.py
=======================

Fast correctness gate for **Passo 4** (leverage, Branch B: lagged timing +
Omori sign-augmented 10-component mixture + FFBS path).  Complements the full
recovery / comparison harness (:func:`mcmc.diagnostics.run_recovery_mcmc_leverage`
with ``timing='lagged'`` and :func:`mcmc.diagnostics.compare_branches_AB`) with
quick, decisive checks:

  [1] Omori-10 constants present and pass the three TOLERANT consistency checks
      (sum q = 1, sum q*m = log-chi^2 mean, b_j = a_j/2).

  [2] Time-varying FFBS nesting: at rho = 0 (G == phi, c == 0, W == sigma2) the
      generalised :func:`_ffbs_tv` reproduces the base scalar AR(1) FFBS draw
      bit-for-bit given the same RNG — Branch B nests Passo 2.

  [3] Family C kernel (Omori regressor): ``draw_rho_scalar`` recovers a known
      lagged ``rho`` from synthetic sign-augmented innovations.

  [4] Skewness mechanism (lagged DGP): lagged leverage with ``rho < 0`` produces
      a *left-skewed* CUMULATIVE factor-innovation density, while ``rho = 0`` is
      symmetric (the asymmetry is cross-temporal under lagged timing).

  [5] End-to-end: a short Branch-B Gibbs run executes, the FFBS path acceptance
      is exactly 1.0, the dominant common leverage is recovered with the right
      (negative) sign, and the h^u path is tracked.

  [6] **P3 immunity** (``docs/audit_P1-P5.md``): Branch B escapes the flat warm
      start ``log h = 0`` **without any warm-seed**.  Branch A cannot — its
      single-move Metropolis sits in the state-vol feedback trap and needs the
      blocked KSC-FFBS seed of ``sample_leverage.py:463`` — whereas B draws the
      whole path from its full conditional (``_ffbs_tv``), so at ``rho = 0`` the
      first sweep is already a full Omori/FFBS jump.  This test asserts that no
      such rescue exists in the lagged sampler and that none is needed.

Run
---
    python src/mcmc/test_passo4.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.constants import OMORI10, validate_mixture                    # noqa: E402
from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                      # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv
from mcmc.tests.test_leverage_common import leverage_end_to_end                            # noqa: E402
from mcmc.sample_vol import _scalar_ar1_ffbs                            # noqa: E402
from mcmc.sample_leverage_lagged import _ffbs_tv                        # noqa: E402
from mcmc.sample_leverage import draw_rho_scalar                        # noqa: E402
from mcmc.diagnostics import leverage_skewness_check                    # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def test_constants():
    print("\n[1] Omori-10 constants present + tolerant consistency checks")
    ok_present = all(k in OMORI10 for k in ("q", "m", "v2", "a", "b")) and len(OMORI10["q"]) == 10
    _check("OMORI10 present with 10 components, keys q,m,v2,a,b", ok_present)
    try:
        out = validate_mixture(OMORI10, has_linearization=True)
        _check("sum q = 1 (tol)", out["sum_q"][1], f"{out['sum_q'][0]:.8f}")
        _check("sum q*m ~ -1.2704 (tol)", out["sum_qm"][1], f"{out['sum_qm'][0]:.5f}")
        _check("b_j = a_j/2 (tol)", out["lin"][1], f"max|b-a/2|={out['lin'][0]:.2e}")
    except AssertionError as e:
        _check("validate_mixture", False, str(e))


def test_ffbs_nesting():
    print("\n[2] Time-varying FFBS nests the base AR(1) FFBS at rho=0")
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    T = 60
    phi, sigma2 = 0.95, 0.05
    V_eff = np.full(T, 4.9)
    mask = np.ones(T, bool); mask[3] = False
    y_eff = np.linspace(-1, 1, T)
    x_base = _scalar_ar1_ffbs(y_eff, V_eff, mask, 0.0, phi, sigma2, rng_a)
    G = np.full(T, phi); c = np.zeros(T); W = np.full(T, sigma2)
    stat_var = sigma2 / (1 - phi * phi)
    x_tv = _ffbs_tv(y_eff, V_eff, mask, G, c, W, stat_var, rng_b)
    _check("rho=0 FFBS_tv == base FFBS (same RNG)", np.allclose(x_base, x_tv),
           f"max diff={np.max(np.abs(x_base-x_tv)):.2e}")


def test_rho_kernel_omori():
    print("\n[3] Family C: draw_rho_scalar recovers a known lagged rho (Omori regressor)")
    rng = np.random.default_rng(0)
    T = 4000
    rho_true, sigma2 = -0.5, 0.04
    sig = np.sqrt(sigma2)
    # synthetic Omori regressor g_t (sign * magnitude), innovations eta = rho*sig*g + ...
    g = rng.standard_normal(T)                  # stand-in for d*e^{m/2}(a+b(xi-m))
    eta = rho_true * sig * g + sig * np.sqrt(1 - rho_true ** 2) * rng.standard_normal(T)
    k = sig * g                                 # k_t = sigma * g  (Branch B regressor)
    rho = 0.0; acc = 0.0; draws = []
    for it in range(4000):
        rho, a = draw_rho_scalar(rho, eta, k, sigma2, 0.05, rng)
        acc += a
        if it >= 1000:
            draws.append(rho)
    mean = float(np.mean(draws))
    _check("posterior mean rho ~ true (-0.5)", abs(mean - rho_true) < 0.07, f"mean={mean:.3f}")
    _check("rho acceptance in (0.1,0.9)", 0.1 < acc / 4000 < 0.9, f"acc={acc/4000:.2f}")


def test_skewness_lagged(theta):
    print("\n[4] Skewness mechanism (lagged): rho<0 -> left-skewed cumulative innovation")
    sk = leverage_skewness_check(theta, sv_u=(0.0, 0.97, 0.20),
                                 rho_u=np.array([-0.7, -0.5, -0.4]),
                                 T=12000, seed=3, timing="lagged")
    _check("rho<0 produces left skew (< -0.05)", sk["skew_leverage"] < -0.05,
           f"skew={sk['skew_leverage']:.3f} (window={sk['window']})")
    _check("rho=0 is ~symmetric (|skew|<0.07)", abs(sk["skew_symmetric"]) < 0.07,
           f"skew={sk['skew_symmetric']:.3f}")


def test_end_to_end(theta, fl, bm, oc, r):
    """Branch B del gate end-to-end — stesso corpo di Branch A, in `test_leverage_common`.
    Cio' che cambia e' cio' che DEVE cambiare: sotto B il path e' un draw FFBS diretto
    (acceptance 1.0 per costruzione, non un tasso), e T e' piu' lungo perche' la SV
    per-fattore legge ogni h^u_k da UNA sola log-square per periodo (P2)."""
    print("\n[5] Branch B end-to-end: FFBS accept=1, segno dominante, tracking")
    leverage_end_to_end("B", theta, fl, bm, oc, r, _check)

def test_rho_griddy_same_target():
    print("\n[7] fix P6: the griddy draws from the SAME target as the RW-Metropolis")
    from mcmc.sample_leverage import draw_rho, draw_rho_griddy, draw_rho_scalar
    rng = np.random.default_rng(0)

    # Correctness seam: on a fixed (eta, k, sigma2) the RW chain and the iid griddy
    # draws must agree on the posterior mean AND sd — not just the mean, which a
    # biased-but-centred kernel could also match.
    for rho_true, n in ((-0.70, 600), (-0.15, 60), (0.45, 600), (0.0, 600)):
        s2 = 0.0625; sig = np.sqrt(s2)
        z = rng.standard_normal(n)
        eta = rho_true * sig * z + sig * np.sqrt(1 - rho_true ** 2) * rng.standard_normal(n)
        k = sig * z
        r_rw = 0.0; chain = []
        for it in range(20000):
            r_rw, _ = draw_rho_scalar(r_rw, eta, k, s2, 0.06, rng)
            if it >= 8000:
                chain.append(r_rw)
        gr = np.array([draw_rho_griddy(0.0, eta, k, s2, rng)[0] for _ in range(3000)])
        dm = abs(float(np.mean(chain)) - float(gr.mean()))
        ds = abs(float(np.std(chain)) - float(gr.std()))
        _check(f"rho_true={rho_true:+.2f}, n={n}: same posterior mean and sd",
               dm < 0.02 and ds < 0.02,
               f"|d mean|={dm:.4f}, |d sd|={ds:.4f}")

    # the griddy ignores the current value (that is the whole point)
    s2 = 0.05
    z = rng.standard_normal(300); eta = -0.5 * np.sqrt(s2) * z + np.sqrt(s2 * 0.75) * rng.standard_normal(300)
    k = np.sqrt(s2) * z
    a = draw_rho_griddy(-0.99, eta, k, s2, np.random.default_rng(5))[0]
    b = draw_rho_griddy(+0.99, eta, k, s2, np.random.default_rng(5))[0]
    _check("griddy draw is independent of the current value (no random walk)", a == b,
           f"from -0.99 -> {a:.4f};  from +0.99 -> {b:.4f}")
    _check("griddy always 'accepts' (flag = 1)",
           draw_rho_griddy(0.0, eta, k, s2, rng)[1] == 1)

    # the dispatcher, and its guard
    r1, _ = draw_rho(0.0, eta, k, s2, np.random.default_rng(9), sampler="griddy")
    r2, _ = draw_rho(0.0, eta, k, s2, np.random.default_rng(9), sampler="rw")
    _check("dispatcher: 'griddy' and 'rw' are different kernels", r1 != r2)
    try:
        draw_rho(0.0, eta, k, s2, rng, sampler="hmc"); ok = False
    except ValueError:
        ok = True
    _check("dispatcher: unknown rho_sampler raises", ok)

    # a flat log_prior must not move the draw (the Fisher-z hook is additive)
    c = draw_rho_griddy(0.0, eta, k, s2, np.random.default_rng(11))[0]
    d = draw_rho_griddy(0.0, eta, k, s2, np.random.default_rng(11), log_prior=lambda r: 0.0)[0]
    _check("flat log_prior leaves the draw bit-identical", c == d)


def test_p3_flat_start_immunity(theta, fl, bm, oc, r):
    print("\n[6] P3 immunity: Branch B escapes the flat warm start with NO warm-seed")
    import inspect
    import mcmc.sample_leverage_lagged as lagged
    import mcmc.sample_leverage as contemp

    src_b = inspect.getsource(lagged)
    src_a = inspect.getsource(contemp)
    _check("Branch A carries the warm-seed rescue (the P3 fix)",
           "sample_common_vol_mv" in src_a)
    _check("Branch B carries NO warm-seed: it needs none (direct FFBS)",
           "sample_common_vol_mv" not in src_b)

    # ... and it is not merely absent, it is unnecessary: from log h == 0 the first
    # sweep is a full Omori/FFBS jump (rho = 0 => G = phi, c = 0, W = sigma2), so the
    # sampler leaves the flat state at once and recovers a persistent phi.  Branch A,
    # without its seed, sits in the trap h~1 -> homoskedastic states -> u~homosk. -> h~1.
    sim = simulate_dfm_sv(theta, T=500, freq_list=fl, block_map=bm, ordered_cols=oc, r=r,
                          seed=9, sv_u=(0.0, 0.97, 0.22), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=np.array([-0.6, -0.3, -0.2])[:r], rho_eps=-0.3,
                          timing="lagged")
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=400, burn_in=150, seed=4,
                       sv=True, leverage=True, timing="lagged",
                       store_vol=True, verbose=False)
    logh = np.log(res["draws"]["h_u"])                 # (n_keep, T, r)
    phi = res["theta_mean"]["sv_u"][:, 1]
    _check("Branch B leaves the flat path (h is not stuck at 1)",
           float(np.mean(np.std(logh, axis=1))) > 0.1,
           f"mean sd(log h) = {float(np.mean(np.std(logh, axis=1))):.3f}")
    _check("Branch B recovers a persistent phi from the flat start (all > 0.7)",
           np.all(phi > 0.7), f"phi={np.round(phi,3)}")
    _check("Branch B never degrades phi to negative (the Branch-A failure mode)",
           np.all(phi > 0.0), f"phi={np.round(phi,3)}")


def main():
    print("=" * 72)
    print("PASSO 4 — leverage (Branch B, lagged + Omori mixture + FFBS) gate")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    test_constants()
    test_ffbs_nesting()
    test_rho_kernel_omori()
    test_skewness_lagged(theta)
    test_end_to_end(theta, fl, bm, oc, r)
    test_rho_griddy_same_target()
    test_p3_flat_start_immunity(theta, fl, bm, oc, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
