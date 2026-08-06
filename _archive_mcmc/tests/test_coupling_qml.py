"""
src/mcmc/tests/test_coupling_qml.py
===================================

Il passo accoppiato del blocco di volatilita' comune (Spec II), raccolto in un posto.

Il coupling nasce dal fatto che, con ``Q`` piena, le ``r`` log-square comuni sono
correlate: accoppiarle e' teoricamente corretto ma nella pratica delicato.  Esiste
**solo dove serve**: il blocco comune SENZA leverage e, sotto leverage, il **Branch B**
(laggato, che linearizza con la mistura di Omori).  **Branch A NON ne ha bisogno**: il
suo target di Metropolis e' esatto, con il whitening pieno ``Q^{-1/2}`` -- non c'e'
nessuna mistura da accoppiare, quindi ``coupling='qml'`` sotto Branch A e' un errore
(il guard lo impone, vedi [4]).

  * Senza leverage il coupling e' una scelta di CALIBRAZIONE (le log-square positivamente
    correlate sono ridondanti, quindi accoppiarle *allarga* onestamente il posterior):
    la forma **QML** (gaussiana singola con le due costanti ``QML_A/QML_B``) e' stabile
    dove la forma **literal** (covarianza scalata dalla correlazione) distorce il fattore
    meno persistente.
  * **Con leverage** il passo accoppiato sta dietro ``allow_experimental=True``: il suo
    comportamento a ``corr(Q)`` forte e' misurato qui (dietro ``--slow``) e sara'
    caratterizzato a fondo nel ``validate``.  Il default resta ``decoupled``.

COPERTURA (parametri/percorsi -- notazione del README/.tex)
-----------------------------------------------------------
Copre il blocco (b) comune col passo accoppiato:
  * h^u_{1:T}, (phi_k, sigma^2_k)  [Fam B comune]   recupero sotto coupling QML
  * costanti QML (QML_A, QML_B), guard del coupling
Il default e' decoupled; QML e' l'opzione accoppiata (sotto leverage: allow_experimental,
da caratterizzare nel validate).

Cosa verifica
-------------
  [1] recovery per-fattore del blocco accoppiato: near-diagonal ``Q`` -> coupled ~=
      decoupled (valida la FFBS r-dim e la tabella g(rho)); + diagnostica a corr forte;
  [2] la QML recupera i path, e' STABILE a corr(Q)=0.92 dove la literal collassa, e a
      ``Q`` diagonale NON coincide col decoupled (droppa il raffinamento della mistura);
  [3] le due costanti ``(QML_A, QML_B)`` ri-derivate in Monte Carlo;
  [4] i guard: il coupling esiste sotto Branch B, mai sotto Branch A, mai in forma literal
      sotto leverage;
  [5] (``--slow``) il comportamento a corr(Q)=0.8 sotto leverage (decoupled vs qml) --
      una prima misura; la caratterizzazione piena e' materia del ``validate``.

Run
---
    python src/mcmc/tests/test_coupling_qml.py
    python src/mcmc/tests/test_coupling_qml.py --slow    # include il gate a corr(Q)=0.8
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.constants import LOG_CHI2_MEAN, LOG_CHI2_VAR, QML_A, QML_B      # noqa: E402
from mcmc.gibbs import fit_dfm_mcmc, load_warm_init                       # noqa: E402
from mcmc.sample_vol import sample_common_vol_mv, logsq_corr_matrix       # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                              # noqa: E402

_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}   {detail}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


# -----------------------------------------------------------------------------
# Blocco comune Spec-II in miniatura (mini-Gibbs sul solo blocco (b))
# -----------------------------------------------------------------------------

def _ar1(phi, sigma, T, rng):
    x = np.empty(T)
    x[0] = (sigma / np.sqrt(1.0 - phi * phi)) * rng.standard_normal()
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
    return x


def _sqrt_spd(Q):
    vals, vecs = np.linalg.eigh(0.5 * (Q + Q.T))
    return (vecs * np.sqrt(np.clip(vals, 1e-12, None))) @ vecs.T


def _simulate(phi, s2, Q, T, seed):
    """Blocco comune Spec-II: u_t = sqrt(H_t) Q^{1/2} z_t (w = 1)."""
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
    sv_draws = []
    gen = np.random.default_rng(seed)
    acc = np.zeros((T, r))
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


# -----------------------------------------------------------------------------
# [1] la FFBS r-dim accoppiata e il limite di disaccoppiamento
# -----------------------------------------------------------------------------

def test_coupling():
    print("\n[1] blocco accoppiato r-dim: limite di disaccoppiamento + un caveat")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])

    # near-diagonal Q: coupled e decoupled CONCORDANO (valida la FFBS r-dim, la tabella
    # g(rho), e il limite di disaccoppiamento)
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

    # corr forte: REPORT-ONLY (non pass/fail): la covarianza scalata dalla correlazione
    # (mistura componentwise + cross-covarianza) e' mis-specificata e distorce il fattore
    # meno persistente.  Il coupling non migliora il punto (info ridondante): migliora la
    # CALIBRAZIONE.
    Q = np.array([[1.0, 0.92], [0.92, 1.0]])
    lt, uh = _simulate(phi_true, s2_true, Q, T, seed=101)
    lc, svc = _minigibbs(uh, Q, T, logsq_corr_matrix(Q), 400, 120, seed=7)
    ld, svd = _minigibbs(uh, Q, T, None, 400, 120, seed=7)
    print(f"    [diag] strong corr(Q)=0.92:")
    print(f"    [diag]   decoupled : corr={_path_corr(ld, lt).round(3)} phi={svd[:,1].round(3)}")
    print(f"    [diag]   coupled(3): corr={_path_corr(lc, lt).round(3)} phi={svc[:,1].round(3)}"
          f"  <- correlation-scaled distortion (design decision pending)")


def test_qml():
    print("\n[2] passo QML: stabile dove la literal collassa")
    T = 1500
    phi_true = np.array([0.98, 0.90]); s2_true = np.array([0.05, 0.12])

    # (a) la QML recupera i path su un DGP near-diagonal -- e' un sampler valido, non
    #     solo stabile.
    Qd = np.array([[1.0, 0.05], [0.05, 1.6]])
    ltd, uhd = _simulate(phi_true, s2_true, Qd, T, seed=303)
    lq, sq = _minigibbs(uhd, Qd, T, None, 400, 120, seed=5, coupling="qml")
    cq = _path_corr(lq, ltd)
    for k in range(2):
        _check(f"QML factor {k}: path corr > 0.5", cq[k] > 0.5, f"corr={cq[k]:.3f}")
    _check("QML: mu fixed at 0", np.allclose(sq[:, 0], 0.0))
    _check("QML: phi distinct (phi0>phi1)", sq[0, 1] > sq[1, 1], f"phi={sq[:,1].round(3)}")

    # (b) IL punto -- corr(Q)=0.92: la QML e' STABILE (phi non collassa), la literal
    #     distorce il fattore meno persistente (phi 0.90 -> ~0.4).  Stesso DGP/seed di [1].
    Q = np.array([[1.0, 0.92], [0.92, 1.0]])
    lt, uh = _simulate(phi_true, s2_true, Q, T, seed=101)
    lq2, sq2 = _minigibbs(uh, Q, T, None, 400, 120, seed=7, coupling="qml")
    ll2, sl2 = _minigibbs(uh, Q, T, logsq_corr_matrix(Q), 400, 120, seed=7)  # literal
    print(f"    [diag] strong corr(Q)=0.92, fattore meno persistente phi (vero 0.90):")
    print(f"    [diag]   QML     : phi={sq2[:,1].round(3)}  (stable)")
    print(f"    [diag]   literal : phi={sl2[:,1].round(3)}  (collapses)")
    _check("QML stable at corr(Q)=0.92: both phi > 0.6", np.all(sq2[:, 1] > 0.6),
           f"phi={sq2[:,1].round(3)}")
    _check("QML beats the literal on the less-persistent factor's phi",
           sq2[1, 1] > sl2[1, 1], f"QML={sq2[1,1]:.3f} vs literal={sl2[1,1]:.3f}")

    # (c) a Q diagonale la QML usa una gaussiana SINGOLA (pi^2/2)I, quindi NON coincide
    #     col decoupled (che usa la mistura KSC).  Draw diversi su stesso rng/stato.
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


# -----------------------------------------------------------------------------
# [3] le costanti QML, ri-derivate in Monte Carlo
# -----------------------------------------------------------------------------

def test_qml_constants():
    """Ri-deriva (QML_A, QML_B) in Monte Carlo invece di fidarsi dell'algebra: sono il
    miglior predittore lineare di |z| dato xi = log z^2, z~N(0,1) -- la controparte a
    gaussiana singola dei (a_j, b_j) di Omori."""
    print("\n[3] costanti QML ri-derivate in Monte Carlo")
    rng = np.random.default_rng(0)
    z = rng.standard_normal(4_000_000)
    xi = np.log(z ** 2)
    az = np.abs(z)

    _check("momenti di log chi^2_1 (media, varianza)",
           abs(xi.mean() - LOG_CHI2_MEAN) < 5e-3 and abs(xi.var() - LOG_CHI2_VAR) < 1e-2,
           f"mean {xi.mean():.4f} vs {LOG_CHI2_MEAN}, var {xi.var():.4f} vs {LOG_CHI2_VAR:.4f}")

    a_mc = float(az.mean())
    b_mc = float(np.cov(az, xi)[0, 1] / xi.var())
    _check("QML_A = E|z| (forma chiusa vs Monte Carlo)",
           abs(a_mc - QML_A) < 5e-3, f"{QML_A:.6f} vs MC {a_mc:.6f}")
    _check("QML_B = Cov(|z|,xi)/Var(xi) (forma chiusa vs Monte Carlo)",
           abs(b_mc - QML_B) < 5e-3, f"{QML_B:.6f} vs MC {b_mc:.6f}")


# -----------------------------------------------------------------------------
# [4] i guard: dove il coupling esiste e dove no
# -----------------------------------------------------------------------------

def test_guards(theta, fl, bm, oc, r):
    """Il coupling esiste sotto Branch B (QML), mai sotto Branch A (target esatto, nulla
    da accoppiare), mai in forma 'literal' sotto leverage (non fattorizza)."""
    print("\n[4] guard del coupling")
    sim = simulate_dfm_sv(theta, T=40, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=1, sv_u=(0.0, 0.95, 0.05),
                          sv_eps=(0.0, 0.94, 0.12), timing="lagged")
    common = dict(n_iter=2, burn_in=0, sv=True, leverage=True, verbose=False)

    def _raises(match, **kw):
        try:
            fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc, **common, **kw)
        except ValueError as e:
            return match in str(e)
        return False

    _check("Branch A + coupling='qml' -> ValueError (nulla da accoppiare)",
           _raises("Branch A", timing="contemporaneous", common_vol_coupling="qml"))
    _check("Branch B + coupling='literal' -> ValueError (non fattorizza)",
           _raises("does not factorise", timing="lagged", common_vol_coupling="literal"))
    _check("Branch B + coupling='qml' senza opt-in -> ValueError (instabile)",
           _raises("EXPERIMENTAL", timing="lagged", common_vol_coupling="qml"))

    res = fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc, timing="lagged",
                       common_vol_coupling="qml", allow_experimental=True, **common)
    _check("Branch B + coupling='qml' con opt-in gira e restituisce r canali di rho",
           res["draws"]["rho_u"].shape[1] == r)


# -----------------------------------------------------------------------------
# [5] (--slow) il comportamento del coupling sotto leverage a corr(Q)=0.8 (baseline)
# -----------------------------------------------------------------------------

def test_qml_leverage_instability(theta0, fl, bm, oc, r):
    r"""Una prima misura del passo accoppiato sotto leverage a corr(Q)=0.8: oggi il
    decoupled regge mentre la QML muove un phi verso il basso e un rho verso il bordo.
    Il test registra questo comportamento (dietro --slow) come baseline; la
    caratterizzazione piena -- e l'eventuale mistura sign-augmented r-dim che il .tex
    lascia non derivata -- e' materia del ``validate``."""
    print("\n[5] comportamento del coupling sotto leverage a corr(Q)=0.8 (baseline)")
    theta = dict(theta0)
    corr = 0.8
    sd = np.sqrt(np.diag(np.asarray(theta["Q"], float)))
    C = np.full((r, r), corr) + (1.0 - corr) * np.eye(r)
    theta["Q"] = (sd[:, None] * C) * sd[None, :]

    rho_true = np.array([-0.70, -0.15, 0.45])[:r]
    sv_pf = np.tile(np.array([0.0, 0.95, 0.25]), (r, 1))
    sim = simulate_dfm_sv(theta, T=400, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=(0.0, 0.95, 0.0), sv_u_perfactor=sv_pf,
                          sv_eps=(0.0, 0.94, 0.12), rho_u=rho_true, rho_eps=-0.3,
                          timing="lagged")

    def _fit(coupling, **kw):
        res = fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc, n_iter=800, burn_in=300, seed=4,
                           sv=True, leverage=True, timing="lagged",
                           common_vol_coupling=coupling, verbose=False, **kw)
        return (res["draws"]["sv_u"][:, :, 1].mean(axis=0),
                res["draws"]["rho_u"].mean(axis=0))

    try:
        _fit("qml")
        gated = False
    except ValueError:
        gated = True
    _check("il passo accoppiato sotto leverage richiede allow_experimental", gated)

    phi_d, rho_d = _fit("decoupled")
    phi_q, rho_q = _fit("qml", allow_experimental=True)

    _check("decoupled REGGE a corr(Q)=0.8 (phi tutti persistenti)",
           bool(np.all(phi_d > 0.5)), f"phi={np.round(phi_d, 3)}")
    _check("qml a corr(Q)=0.8: un phi si abbassa (baseline, cfr. validate)",
           bool(np.any(phi_q < 0.5)), f"phi={np.round(phi_q, 3)}")
    _check("qml a corr(Q)=0.8: un rho va verso il bordo (baseline, cfr. validate)",
           bool(np.max(np.abs(rho_q)) > 0.85), f"rho={np.round(rho_q, 3)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slow", action="store_true", help="include il gate a corr(Q)=0.8 (lento)")
    a = p.parse_args()

    print("=" * 72)
    print("COUPLING / QML -- il passo accoppiato del blocco di volatilita' comune")
    print("=" * 72)
    test_coupling()
    test_qml()
    test_qml_constants()
    w = load_warm_init("small")
    theta, fl, bm, oc, r = (w["theta"], w["freq_list"], w["block_map"],
                            w["ordered_cols"], w["r"])
    test_guards(theta, fl, bm, oc, r)
    if a.slow:
        test_qml_leverage_instability(theta, fl, bm, oc, r)
    else:
        print("\n  [skip] test_qml_leverage_instability (usa --slow)")

    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
