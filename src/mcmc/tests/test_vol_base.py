"""
src/mcmc/tests/test_vol_base.py
===============================

Il gate della **volatilita' stocastica di base** (blocco (b), SENZA leverage) e del
suo recupero.  Il leverage vive in ``test_leverage``; qui si campiona ``h`` col solo
KSC, e si verifica che sia giusto e che si recuperi.

COPERTURA (parametri/percorsi -- notazione del README/.tex)
-----------------------------------------------------------
Livello RECOVERY, SENZA leverage.  Recupera:
  * h^u_{1:T}, (phi_k, sigma^2_k)  [Fam B comune]   r vol per-fattore (Spec II)
  * h^eps_{1:T}, phi^eps           [Fam B idio]     il lato idiosincratico
  * + KSC, filtro combinato, convenzione sigma / sigma^2
Non tocca rho (Fam C): quello sta in test_leverage.

Cosa verifica
-------------
  [1] costanti **KSC-7** validate *con tolleranza* (sum q=1, sum q*m ~ -1.2704) --
      mai uguaglianza esatta (la mistura approssima log chi^2_1);
  [2] il filtro a precisione combinata == filtro stock a ``h==1`` (accendere la SV a
      volatilita' unitaria non perturba il blocco di stato);
  [3] FFBS scalare del log-vol: recupera un path AR(1) noto dalle log-square;
  [4] recovery per-fattore (Spec II) sul blocco di vol, ``Q`` diagonale: path,
      persistenza ``phi_k`` distinti, ``sigma2`` (mini-Gibbs sul solo blocco (b));
  [5] end-to-end: una corsa SV corta recupera il path comune (corr>0.6), theta/nu sani;
  [6] wiring del prior half-Normal su ``sigma_eta`` (mu=0 su ogni processo);
  [7] la **convenzione sigma / sigma^2** (la mina): la sd empirica del path vero
      identifica l'ingresso come ``sigma``, non ``sigma^2``; + round-trip;
  [8] recovery del lato **IDIOSINCRATICO**: il path ``h^eps`` e ``phi^eps`` (non solo
      "spento vs acceso": si verifica che si recuperino davvero).

Run
---
    python src/mcmc/tests/test_vol_base.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kalman import kalman_filter                                          # noqa: E402
from simulate_dfm import simulate_dfm                                     # noqa: E402

from mcmc.constants import KSC7, validate_mixture                         # noqa: E402
from mcmc.gibbs import load_warm_init, fit_dfm_mcmc                       # noqa: E402
from mcmc.sample_states import forward_filter_combined                    # noqa: E402
from mcmc.sample_vol import (                                             # noqa: E402
    sample_log_vol_process, sample_common_vol_mv, logsq_corr_matrix,
)
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
    print("\n[4] SV Gibbs (Spec II, per-factor) recovers the common volatility path")
    # Spec II splits the common vol into r per-factor states, each read by ONE
    # log-square per period (vs the old scalar-common's r simultaneous readings) —
    # the "sqrt(r)-averaging" the thesis notes is gone, so each state needs ~r x more
    # data.  We therefore run a longer panel here (r=3 -> T=750) than the old
    # scalar-common gate did (T=220); recovery is then strong and stable.
    sv_u = (0.0, 0.97, 0.25)
    sim = simulate_dfm_sv(theta, T=750, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=sv_u, sv_eps=(0.0, 0.95, 0.15))
    Y = sim["Y"]
    res = fit_dfm_mcmc(Y, {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=300, burn_in=120, thin=1, seed=4,
                       sv=True, store_vol=True, verbose=False)
    # Spec II (no leverage): the common block is now r *per-factor* volatilities.
    # The scalar-common DGP has each factor read the same common h (e_k ~ N(0,h)),
    # so every per-factor h^u_k should recover that single true common path.
    lhu = np.log(res["draws"]["h_u"].mean(axis=0))          # (T, r) under Spec II
    if lhu.ndim == 2:
        c = float(np.mean([_corr(lhu[:, k], sim["logh_u_true"]) for k in range(lhu.shape[1])]))
    else:
        c = _corr(lhu, sim["logh_u_true"])
    _check("per-factor log h^u corr with truth (>0.6)", c > 0.6, f"avg corr={c:.3f}")
    sv_u_mean = np.asarray(res["theta_mean"]["sv_u"])
    phi_hat = float(sv_u_mean[:, 1].mean()) if sv_u_mean.ndim == 2 else float(sv_u_mean[1])
    _check("h^u AR(1) phi recovered (>0.85)", phi_hat > 0.85, f"phi={phi_hat:.3f}")
    nu_eps = res["theta_mean"]["nu_eps"]
    _check("nu_eps stays sane (2.5..15)", 2.5 < nu_eps < 15.0, f"nu_eps={nu_eps:.2f}")


def test_half_normal_e2e(theta, fl, bm, oc, r):
    print("\n[5] Family B half-Normal sigma_eta prior: end-to-end wiring + mu=0")
    # Short Spec II run under the half-Normal prior: the point is that
    # the gibbs wiring runs and mu is pinned at 0 for every process; recovery
    # quality is gated by the kernel test in test_shared [7].
    sim = simulate_dfm_sv(theta, T=300, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=11, sv_u=(0.0, 0.95, 0.2), sv_eps=(0.0, 0.95, 0.15))
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=120, burn_in=50, thin=1, seed=5,
                       sv=True, sv_sigma_prior="half_normal", sv_half_normal_B=1.0,
                       store_vol=True, verbose=False)
    sv_u = np.asarray(res["draws"]["sv_u"])          # (n_keep, r, 3)
    sv_eps = np.asarray(res["draws"]["sv_eps"])      # (n_keep, M, 3)
    _check("half-Normal e2e: mu_u == 0 for all draws/factors",
           np.all(sv_u[..., 0] == 0.0), f"max|mu_u|={np.abs(sv_u[...,0]).max():.2e}")
    _check("half-Normal e2e: mu_eps == 0 for all draws/series",
           np.all(sv_eps[..., 0] == 0.0), f"max|mu_eps|={np.abs(sv_eps[...,0]).max():.2e}")
    s2u_mean = float(sv_u[..., 2].mean())
    _check("half-Normal e2e: sigma2_u finite & positive", np.isfinite(s2u_mean) and s2u_mean > 0,
           f"mean sigma2_u={s2u_mean:.4f}")
    nu_eps = res["theta_mean"]["nu_eps"]
    _check("half-Normal e2e: nu_eps sane (2.5..15)", 2.5 < nu_eps < 15.0, f"nu_eps={nu_eps:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# [4] recovery per-fattore (Spec II) sul solo blocco (b), Q diagonale
# ─────────────────────────────────────────────────────────────────────────────

def _ar1(phi, sigma, T, rng):
    x = np.empty(T)
    x[0] = (sigma / np.sqrt(1.0 - phi * phi)) * rng.standard_normal()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
    return x


def _sqrt_spd(Q):
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    return (vecs * np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def _simulate_specII(phi, s2, Q, T, seed):
    """Blocco comune Spec-II: u_t = sqrt(H_t) Q^{1/2} z_t (w = 1)."""
    rng = np.random.default_rng(seed)
    r = len(phi)
    logh = np.column_stack([_ar1(phi[k], np.sqrt(s2[k]), T, rng) for k in range(r)])
    z = rng.standard_normal((T, r))
    u = np.sqrt(np.exp(logh)) * (z @ _sqrt_spd(Q).T)
    return logh, u[1:]                                # true logh (T,r), u_head (T-1,r)


def _minigibbs(u_head, Q, T, n_iter, burn, seed):
    r = u_head.shape[1]
    logh_cur = np.zeros((T, r))
    sv_cur = np.tile([0.0, 0.90, 0.10], (r, 1))
    w_u = np.ones(T)
    sv_draws = []
    gen = np.random.default_rng(seed)
    acc = np.zeros((T, r))
    for it in range(n_iter):
        out = sample_common_vol_mv(u_head, Q, w_u, logh_cur, sv_cur, gen)
        logh_cur, sv_cur = out["logh_u"], out["sv_u"]
        if it >= burn:
            acc += logh_cur
            sv_draws.append(sv_cur.copy())
    return acc / (n_iter - burn), np.mean(sv_draws, axis=0)


def _path_corr(logh_hat, logh_true):
    r = logh_hat.shape[1]
    return np.array([np.corrcoef(logh_hat[1:, k], logh_true[1:, k])[0, 1] for k in range(r)])


def test_recovery_diag():
    print("\n[4] recovery per-fattore (R_xi=I, Q diagonale): path, phi distinti, sigma2")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])
    Q = np.diag([1.0, 1.6])
    logh_true, u_head = _simulate_specII(phi_true, s2_true, Q, T, seed=20260708)
    logh_hat, sv_hat = _minigibbs(u_head, Q, T, 500, 150, seed=7)
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


# ─────────────────────────────────────────────────────────────────────────────
# [7] la convenzione sigma / sigma^2 (la mina) — il test che l'avrebbe trovata da solo
# ─────────────────────────────────────────────────────────────────────────────

def _diagonal_theta(theta):
    th = dict(theta)
    th["Q"] = np.diag(np.diag(np.asarray(theta["Q"], float)))
    th["Sigma_0"] = np.asarray(theta["Sigma_0"], float)
    return th


def _perfactor_sim(seed=22):
    """Un pannello dal DGP per-fattore (per la guard di convenzione, che guarda solo
    il DGP, non il sampler)."""
    w = load_warm_init("small")
    theta = _diagonal_theta(w["theta"])
    sv_u_pf = np.array([[0.0, 0.97, 0.25], [0.0, 0.92, 0.18], [0.0, 0.95, 0.22]])
    rho = np.array([-0.70, -0.15, 0.45])
    r = w["r"]
    return simulate_dfm_sv(theta, T=600, freq_list=w["freq_list"], block_map=w["block_map"],
                           ordered_cols=w["ordered_cols"], r=r, seed=seed,
                           sv_u=(0.0, 0.95, 0.0), sv_u_perfactor=sv_u_pf[:r],
                           sv_eps=(0.0, 0.94, 0.12), rho_u=rho[:r], rho_eps=-0.3,
                           timing="lagged")


def test_convention():
    print("\n[7] sigma in, sigma^2 out -- la convenzione, congelata")
    sim = _perfactor_sim()
    sig = np.asarray(sim["sv_u_sigma"], float)          # (r, 3): (mu, phi, SIGMA)
    var = np.asarray(sim["sv_u"], float)                # (r, 3): (mu, phi, SIGMA^2)
    x = np.asarray(sim["logh_u_true"], float)           # (T, r) path VERO

    _check("round-trip: sv_u[:,2] == sv_u_sigma[:,2]**2",
           np.allclose(var[:, 2], sig[:, 2] ** 2),
           f"{np.round(var[:, 2], 4)} vs {np.round(sig[:, 2] ** 2, 4)}")

    # La sd EMPIRICA del path vero: le due letture danno numeri lontani (a phi=0.97:
    # 1.03 contro 2.06), quindi il dato decide QUALE colonna sia sigma.  E' un test di
    # DISCRIMINAZIONE, non di precisione (la sd incondizionata di un AR(1) su T=600 ha
    # una variabilita' campionaria enorme).
    emp = x.std(axis=0)
    as_sd = sig[:, 2] / np.sqrt(1.0 - sig[:, 1] ** 2)            # se la colonna e' sigma
    as_var = np.sqrt(sig[:, 2]) / np.sqrt(1.0 - sig[:, 1] ** 2)  # se fosse sigma^2
    err_sd = float(np.max(np.abs(emp - as_sd) / as_sd))
    err_var = float(np.max(np.abs(emp - as_var) / as_var))
    _check("la sd empirica del path vero identifica l'INPUT come sigma, non sigma^2",
           err_sd < 0.5 * err_var,
           f"empirica={np.round(emp, 3)}, come-sigma={np.round(as_sd, 3)} "
           f"(err {err_sd:.0%}), come-sigma^2={np.round(as_var, 3)} (err {err_var:.0%})")


# ─────────────────────────────────────────────────────────────────────────────
# [8] recovery del lato IDIOSINCRATICO (il buco che mancava)
# ─────────────────────────────────────────────────────────────────────────────

def test_idio_vol_recovery(theta, fl, bm, oc, r):
    print("\n[8] recovery IDIOSINCRATICO: path h^eps e phi^eps (senza leverage)")
    sim = simulate_dfm_sv(theta, T=600, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=13, sv_u=(0.0, 0.95, 0.15), sv_eps=(0.0, 0.95, 0.20))
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=300, burn_in=120, thin=1, seed=6,
                       sv=True, store_vol=True, verbose=False)
    lhe = np.log(res["draws"]["h_eps"].mean(axis=0))         # (T, M) posterior mean
    truth = np.asarray(sim["logh_eps_true"])                 # (T, M)
    cs = [_corr(lhe[:, i], truth[:, i]) for i in range(lhe.shape[1])]
    med = float(np.median(cs))
    _check("idio: la mediana della corr(h^eps, vero) e' positiva e sostanziale (>0.25)",
           med > 0.25, f"mediana corr={med:.3f} su {len(cs)} serie")
    phi_eps = np.asarray(res["theta_mean"]["sv_eps"])[:, 1]
    _check("idio: la mediana dei phi^eps e' persistente (>0.6)",
           float(np.median(phi_eps)) > 0.6, f"mediana phi^eps={float(np.median(phi_eps)):.3f}")
    _check("idio: mu^eps == 0 in ogni draw (identificazione)",
           np.all(res["draws"]["sv_eps"][:, :, 0] == 0.0))


def main():
    print("=" * 72)
    print("VOL BASE -- volatilita' stocastica senza leverage: correttezza + recovery")
    print("=" * 72)
    w = load_warm_init("small")
    theta = dict(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    # dal piu' veloce al piu' lento
    test_ksc()
    test_filter_equiv(theta, fl, bm, oc, r)
    test_scalar_vol_ffbs()
    test_recovery_diag()
    test_convention()
    test_end_to_end(theta, fl, bm, oc, r)
    test_half_normal_e2e(theta, fl, bm, oc, r)
    test_idio_vol_recovery(theta, fl, bm, oc, r)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
