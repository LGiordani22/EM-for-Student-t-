"""
src/mcmc/test_asis.py
=====================

PASSO 6 gate for the ASIS interweaving (``mcmc.sample_asis``, thesis ``sec:asis``).

ASIS is judged by its *effect*, not a point seam (it changes the chain, not the
target): the two things that must hold are

  * **posterior unchanged** — CP-only and CP+ASIS target the same log-vol posterior,
    so their draw means agree within Monte-Carlo error; and
  * **mixing improves** — the effective sample size of the worst-mixing parameter
    (the persistence ``phi`` of a strong-signal volatility, where CP crawls along
    the path/scale ridge) rises markedly under interweaving.

Both are checked on a self-contained single-process log-vol mini-Gibbs (no data
files), and once end-to-end through ``fit_dfm_mcmc`` (no leverage) to confirm the
wired sampler runs and its posterior matches the non-ASIS run.

Run
---
    python src/mcmc/test_asis.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.constants import KSC7                                       # noqa: E402
from mcmc.sample_vol import sample_log_vol_process, draw_ar1_params  # noqa: E402
from mcmc.sample_asis import asis_scale_interweave                   # noqa: E402

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


def _ess(chain: np.ndarray) -> float:
    """Effective sample size via the initial-positive-sequence autocorrelation sum."""
    x = np.asarray(chain, float) - np.mean(chain)
    n = x.size
    v = float(np.dot(x, x) / n)
    if v == 0.0:
        return float(n)
    ac = np.correlate(x, x, "full")[n - 1:] / (v * n)
    s = 1.0
    for k in range(1, n):
        if ac[k] <= 0.0:
            break
        s += 2.0 * ac[k]
    return n / s


def _simulate_logvol(phi: float, s2: float, T: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    x[0] = np.sqrt(s2 / (1.0 - phi * phi)) * rng.standard_normal()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + np.sqrt(s2) * rng.standard_normal()
    e = np.exp(x / 2.0) * rng.standard_normal(T)
    return x, np.log(e ** 2 + 1e-6)          # true path, KSC log-square


def _mini_gibbs(y_star, T, use_asis, seed, n_iter=4000, burn=1000, B=10.0):
    """Single-process log-vol Gibbs (mu=0, no leverage): step (b) KSC-FFBS path,
    step (1) CP Family B (half-Normal), optional ASIS interweave."""
    g = np.random.default_rng(seed)
    tidx = np.arange(T)
    has = np.ones(T, bool)
    z0 = np.zeros(T)
    lc = np.zeros(T)
    phi, s2 = 0.9, 0.1
    phis, s2s = [], []
    for it in range(n_iter):
        lc = sample_log_vol_process(y_star, tidx, T, lc, 0.0, phi, s2, g, KSC7)
        _, phi, s2 = draw_ar1_params(lc, g, fix_mu0=True, sigma_prior="half_normal",
                                     half_normal_B=B, sigma2_cur=s2)
        if use_asis:
            lc, phi, s2 = asis_scale_interweave(lc, y_star, has, s2, 0.0, z0, g,
                                                half_normal_B=B)
        if it >= burn:
            phis.append(phi); s2s.append(s2)
    return np.array(phis), np.array(s2s)


def test_invariance_and_ess():
    print("\n[1] ASIS: posterior invariance + phi mixing gain (strong-signal process)")
    # strong signal (large sigma2): CP is worst here, NCP best -> ASIS should help.
    _, y_star = _simulate_logvol(phi=0.98, s2=0.30, T=500, seed=0)
    T = y_star.size
    pc, sc = _mini_gibbs(y_star, T, use_asis=False, seed=1)
    pa, sa = _mini_gibbs(y_star, T, use_asis=True, seed=1)

    # (a) same target: posterior means agree within MC error
    _check("posterior mean(phi) unchanged by ASIS", abs(pc.mean() - pa.mean()) < 0.01,
           f"CP={pc.mean():.4f} ASIS={pa.mean():.4f}")
    _check("posterior mean(sigma2) unchanged by ASIS",
           abs(sc.mean() - sa.mean()) < 0.05 * sc.mean() + 0.01,
           f"CP={sc.mean():.4f} ASIS={sa.mean():.4f}")

    # (b) mixing improves: ESS(phi) rises markedly
    ec, ea = _ess(pc), _ess(pa)
    _check("ASIS raises ESS(phi) (>1.3x)", ea > 1.3 * ec, f"CP={ec:.0f} ASIS={ea:.0f} ({ea/ec:.1f}x)")
    _check("ESS(sigma2) not degraded", _ess(sa) > 0.8 * _ess(sc),
           f"CP={_ess(sc):.0f} ASIS={_ess(sa):.0f}")


def test_signed_sigma():
    print("\n[2] ASIS: signed sigma_eta draw + rescale identity")
    # With sigma2_cp fixed and a path, asis returns a rescaled path x = sigma*x_tilde;
    # x_tilde = x/sigma_cp is exact, so the returned path equals (sigma_new/sigma_cp)*x.
    rng = np.random.default_rng(3)
    _, y_star = _simulate_logvol(0.95, 0.2, 300, seed=2)
    T = y_star.size
    x = 0.3 * rng.standard_normal(T)
    has = np.ones(T, bool)
    s2_cp = 0.15
    x_new, phi_new, s2_new = asis_scale_interweave(x, y_star, has, s2_cp, 0.0,
                                                   np.zeros(T), rng, half_normal_B=10.0)
    ratio = np.sqrt(s2_new) / np.sqrt(s2_cp)          # |sigma_new|/sigma_cp
    # x_new = sigma_new * (x / sigma_cp) => |x_new| == ratio*|x| (up to the sign)
    ok_scale = np.allclose(np.abs(x_new), ratio * np.abs(x), atol=1e-9)
    _check("rescale identity |x_new| = (|sigma_new|/sigma_cp)|x|", ok_scale)
    _check("phi_new finite in (-1,1)", np.isfinite(phi_new) and abs(phi_new) < 1.0,
           f"phi_new={phi_new:.4f}")
    _check("sigma2_new finite > 0", np.isfinite(s2_new) and s2_new > 0, f"s2_new={s2_new:.4f}")


def test_end_to_end_matches():
    print("\n[3] ASIS end-to-end (no leverage): runs, mu=0, posterior ~ non-ASIS")
    from mcmc.gibbs import load_warm_init, fit_dfm_mcmc
    from mcmc.simulate_sv import simulate_dfm_sv
    w = load_warm_init("small")
    theta = dict(w["theta"]); theta["Sigma_0"] = np.asarray(theta["Sigma_0"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    sim = simulate_dfm_sv(theta, T=500, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=(0.0, 0.97, 0.25), sv_eps=(0.0, 0.95, 0.15))

    def run(asis):
        return fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc, n_iter=400, burn_in=150,
                            seed=4, sv=True, use_asis=asis,
                            sv_sigma_prior="half_normal", verbose=False)
    base = run(False); asis = run(True)
    svb = np.asarray(base["draws"]["sv_u"]); sva = np.asarray(asis["draws"]["sv_u"])
    _check("ASIS run: mu_u == 0", np.all(sva[..., 0] == 0.0))
    _check("ASIS run: sv_u finite", np.isfinite(sva).all())
    phib = svb[..., 1].mean(); phia = sva[..., 1].mean()
    _check("phi_u posterior ~ non-ASIS (|d|<0.05)", abs(phib - phia) < 0.05,
           f"base={phib:.3f} asis={phia:.3f}")
    s2b = svb[..., 2].mean(); s2a = sva[..., 2].mean()
    _check("sigma2_u posterior ~ non-ASIS (rel<0.30)", abs(s2b - s2a) < 0.30 * s2b + 0.02,
           f"base={s2b:.3f} asis={s2a:.3f}")


def test_end_to_end_leverage():
    print("\n[4] ASIS under leverage (Branch A): runs, mu=0, posterior ~ non-ASIS")
    from mcmc.gibbs import load_warm_init, fit_dfm_mcmc
    from mcmc.simulate_sv import simulate_dfm_sv
    w = load_warm_init("small")
    theta = dict(w["theta"]); theta["Sigma_0"] = np.asarray(theta["Sigma_0"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    sim = simulate_dfm_sv(theta, T=500, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=(0.0, 0.97, 0.25), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=np.array([-0.5, -0.3, -0.2]), rho_eps=-0.3)

    def run(asis):
        return fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc, n_iter=300, burn_in=120,
                            seed=4, sv=True, leverage=True, timing="contemporaneous",
                            use_asis=asis, sv_sigma_prior="half_normal", verbose=False)
    base = run(False); asis = run(True)
    sva = np.asarray(asis["draws"]["sv_u"])
    _check("ASIS+leverage: mu_u == 0", np.all(sva[..., 0] == 0.0))
    _check("ASIS+leverage: sv_u, rho_u finite",
           np.isfinite(sva).all() and np.isfinite(asis["draws"]["rho_u"]).all())
    phib = np.asarray(base["draws"]["sv_u"])[..., 1].mean()
    phia = sva[..., 1].mean()
    _check("ASIS+leverage: phi_u posterior ~ non-ASIS (|d|<0.08)", abs(phib - phia) < 0.08,
           f"base={phib:.3f} asis={phia:.3f}")


def main() -> int:
    print("=" * 72)
    print("PASSO 6 — ASIS interweaving gate (sample_asis.py)")
    print("=" * 72)
    test_invariance_and_ess()
    test_signed_sigma()
    test_end_to_end_matches()
    test_end_to_end_leverage()
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
