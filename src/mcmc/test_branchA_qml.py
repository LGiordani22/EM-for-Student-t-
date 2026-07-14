"""
src/mcmc/test_branchA_qml.py
============================

Due aggiunte, un file di test.

**(1) Branch A riparato** — il path draw a blocco (proposta di Laplace + Metropolis
esatto, ``lev_path_sampler="laplace"``).  La tesi da difendere **non** è "mescola
meglio": è **"campiona la stessa legge"**.  Una proposta può essere rozza quanto
vuole; se il rapporto di Metropolis è giusto, la distribuzione invariante non si
muove.  Quindi il test che conta è di **invarianza** (:func:`test_invariance`):
si porta la catena al target col kernel *vecchio*, poi si applica il kernel
*nuovo*, e la legge non deve spostarsi.  Se la proposta o il rapporto fossero
sbagliati (una densità inversa dimenticata, la linearizzazione infilata nel target
invece che nella proposta), il kernel a blocco convergerebbe a una legge *diversa*
e i due riassunti divergerebbero.  L'efficienza, invece, si **misura**
(:func:`test_block_beats_single`) — non si asserisce.

**(2) QML sotto leverage** (``common_vol_coupling="qml"``, Branch B) — il blocco
comune *accoppiato*: implementato, e **non funzionante**.  Le sue due costanti sono
in forma chiusa, quindi si **ri-derivano** in Monte Carlo invece di fidarsi
dell'algebra (:func:`test_qml_constants`); i guard dicono ciò che dice la teoria
(:func:`test_guards`); e il gate lento **congela l'instabilità**
(:func:`test_qml_leverage_instability`): a ``corr(Q)=0.8`` — il solo regime che
giustificherebbe un passo accoppiato — il fattore meno identificato vede ``phi``
crollare e ``rho`` incollarsi al bordo, mentre il blocco decoupled regge.  Il test
asserisce il *divario*, così il fallimento resta visibile invece di essere sepolto.

Run
---
    python src/mcmc/test_branchA_qml.py
    python src/mcmc/test_branchA_qml.py --slow      # include il gate a corr(Q)=0.8
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.constants import LOG_CHI2_MEAN, LOG_CHI2_VAR, QML_A, QML_B      # noqa: E402
from mcmc.gibbs import fit_dfm_mcmc, load_warm_init                       # noqa: E402
from mcmc.sample_leverage import (                                        # noqa: E402
    _lev_path_laplace_mh_common,
    _lev_path_mh_mv_common,
    _logpost_A_common,
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


# ─────────────────────────────────────────────────────────────────────────────
# Un target di Branch A piccolo e autosufficiente
# ─────────────────────────────────────────────────────────────────────────────

def _toy_target(T=60, r=2, seed=3):
    """Blocco comune di Branch A in miniatura: parametri fissi, shock fissi, un
    solo target esatto — così i due kernel si confrontano sulla *stessa* legge."""
    rng = np.random.default_rng(seed)
    phi = np.array([0.95, 0.90])[:r]
    sigma2 = np.array([0.05, 0.08])[:r]
    rho = np.array([-0.60, 0.40])[:r]
    Q = np.array([[1.0, 0.3], [0.3, 1.2]])[:r, :r]          # Q NON diagonale: il
    qdiag = np.diag(Q)                                      # target è quello coupled
    vals, vecs = np.linalg.eigh(Q)
    Qinv_half = vecs @ np.diag(vals ** -0.5) @ vecs.T

    has = np.zeros(T, bool); has[1:] = True
    b = np.zeros((T, r))
    b[1:] = rng.standard_normal((T - 1, r)) @ np.linalg.cholesky(Q).T
    E = b / np.sqrt(qdiag)[None, :]
    S = E ** 2
    return dict(b=b, S=S, E=E, has=has, Qinv_half=Qinv_half,
                phi=phi, sigma2=sigma2, rho=rho, T=T, r=r)


def _sweep(kernel, x, tg, rng):
    if kernel == "single":
        return _lev_path_mh_mv_common(x, tg["b"], tg["S"], tg["has"], tg["Qinv_half"],
                                      tg["phi"], tg["sigma2"], tg["rho"], 0.25, rng)
    return _lev_path_laplace_mh_common(x, tg["b"], tg["S"], tg["E"], tg["has"],
                                       tg["Qinv_half"], tg["phi"], tg["sigma2"],
                                       tg["rho"], rng)


def _run_chain(kernel, tg, n, seed, keep_from=None):
    rng = np.random.default_rng(seed)
    x = np.zeros((tg["T"], tg["r"]))
    keep_from = n // 2 if keep_from is None else keep_from
    keep = []
    for i in range(n):
        x, _, _ = _sweep(kernel, x, tg, rng)
        if i >= keep_from:
            keep.append(x.copy())
    return np.stack(keep)                                   # (n_keep, T, r)


# ─────────────────────────────────────────────────────────────────────────────
# [1] Invarianza — l'unica proprietà che un cambio di proposta deve preservare
# ─────────────────────────────────────────────────────────────────────────────

def test_invariance(n_chains=4, n_single=3000, n_block=500):
    """I due kernel sono Metropolis sullo **stesso** target esatto, quindi la loro
    legge stazionaria coincide.  Il kernel a blocco è nuovo: se la sua proposta o il
    suo rapporto di accettazione fossero sbagliati (una densità inversa dimenticata,
    la linearizzazione infilata nel target invece che nella proposta), convergerebbe
    a una legge *diversa* — e si vedrebbe qui.

    **La soglia non è cablata.**  La catena single-move è proprio quella che mescola
    male: la sua media ha un errore Monte Carlo grosso, e una tolleranza fissa
    sarebbe una scommessa, non un test (troppo stretta -> rosso a caso; troppo larga
    -> non falsifica nulla).  Si stima quindi l'errore standard di ciascun gruppo
    **dalla dispersione fra catene indipendenti** e si confronta lo scarto con
    ``3 * SE`` — un test a due campioni che si autocalibra."""
    tg = _toy_target(T=40, r=2)

    def _group(kernel, n, seed0):
        # media e sd del path per catena -> la dispersione FRA catene stima l'errore
        # Monte Carlo di ciascun kernel, incluso l'effetto della sua autocorrelazione.
        means = np.zeros((n_chains, tg["r"]))
        sds = np.zeros((n_chains, tg["r"]))
        for c in range(n_chains):
            X = _run_chain(kernel, tg, n, seed=seed0 + c)
            means[c] = X.mean(axis=(0, 1))
            sds[c] = X.std(axis=(0, 1))
        return means, sds

    ms, ss = _group("single", n_single, 100)
    mb, sb = _group("laplace", n_block, 200)

    def _two_sample(As, Bs, label):
        ok = True
        detail = []
        for k in range(tg["r"]):
            a, b = As[:, k], Bs[:, k]
            se = float(np.sqrt(a.var(ddof=1) / n_chains + b.var(ddof=1) / n_chains))
            d = float(abs(a.mean() - b.mean()))
            if se > 0 and d > 3.0 * se:
                ok = False
            detail.append(f"k={k}: {a.mean():+.3f} vs {b.mean():+.3f} "
                          f"(|d|={d:.3f}, 3SE={3 * se:.3f})")
        _check(f"invarianza: {label} — single vs blocco entro 3 SE",
               ok, " | ".join(detail))

    _two_sample(ms, mb, "media del path")
    _two_sample(ss, sb, "sd del path")

    kw = dict(b=tg["b"], S=tg["S"], has_obs=tg["has"], Qinv_half=tg["Qinv_half"],
              phi=tg["phi"], sigma2=tg["sigma2"], rho=tg["rho"])
    Xs = _run_chain("single", tg, n_single, seed=11)
    Xl = _run_chain("laplace", tg, n_block, seed=12)
    lp_s = float(np.mean([_logpost_A_common(x, **kw) for x in Xs[::20]]))
    lp_l = float(np.mean([_logpost_A_common(x, **kw) for x in Xl[::10]]))
    _check("invarianza: stesso livello medio del log-target esatto",
           abs(lp_s - lp_l) < 0.05 * abs(lp_s),
           f"single {lp_s:.1f} vs blocco {lp_l:.1f}")


def test_block_beats_single():
    """Il *punto* della riparazione, misurato: a parità di sweep il kernel a blocco
    muove il path molto di più.  Autocorrelazione di lag 1 del livello medio."""
    tg = _toy_target()

    def _ac1(kernel, n=600, seed=5):
        X = _run_chain(kernel, tg, n, seed, keep_from=n // 3)
        v = X[:, :, 0].mean(axis=1)
        v = v - v.mean()
        return float((v[:-1] * v[1:]).sum() / (v * v).sum())

    ac_s = _ac1("single")
    ac_b = _ac1("laplace")
    _check("il blocco batte il single-move (autocorr lag 1)",
           ac_b < ac_s, f"blocco {ac_b:.3f} vs single {ac_s:.3f}")


def test_no_warm_seed_needed():
    """P3 muore per costruzione: dallo start PIATTO il kernel a blocco atterra su un
    path non degenere in **uno** sweep — senza alcun warm seed KSC."""
    tg = _toy_target()
    rng = np.random.default_rng(7)
    x0 = np.zeros((tg["T"], tg["r"]))
    x1, n_acc, n_prop = _sweep("laplace", x0, tg, rng)
    _check("P3: un accept/reject a blocco per fattore", n_prop == tg["r"],
           f"n_prop={n_prop}, atteso {tg['r']}")
    _check("P3: dal warm start piatto il path esce non degenere in 1 sweep",
           n_acc >= 1 and np.std(x1) > 0.1,
           f"n_acc={n_acc}, sd(path)={np.std(x1):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# [2] QML sotto leverage
# ─────────────────────────────────────────────────────────────────────────────

def test_qml_constants():
    """Ri-deriva ``(QML_A, QML_B)`` in Monte Carlo invece di fidarsi dell'algebra:
    sono il miglior predittore lineare di ``|z|`` dato ``xi = log z^2``, ``z~N(0,1)``
    — la controparte a gaussiana singola dei ``(a_j, b_j)`` di Omori."""
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


def test_guards(theta, fl, bm, oc, r):
    """I guard dicono ciò che dice la teoria: il passo accoppiato esiste sotto
    Branch B (QML), **mai** sotto Branch A (non c'è nulla da accoppiare: il suo
    target è esatto), e **mai** in forma 'literal' sotto leverage (non fattorizza)."""
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


def test_qml_leverage_instability(theta0, fl, bm, oc, r):
    r"""**Congela il fallimento**, non lo nasconde.

    Il passo accoppiato sotto leverage è implementato e **non funziona**, proprio nel
    regime che lo giustificherebbe.  A ``corr(Q) = 0.8``:

        decoupled   phi = [0.925, 0.972, 0.878]   (regge)
        qml         phi = [0.930, 0.895, 0.421]   rho_2 = +0.97  (bordo)

    e coi segni per-componente va **peggio** (``phi_2 = -0.29``), quindi non è
    l'inconsistenza segno/magnitudine: è il **coupling della misura in sé**.  La
    ragione è quella che il .tex dà per il coupling in generale — log-square
    positivamente correlate sono **ridondanti**, quindi accoppiarle *allarga* il
    posterior di ciascuna volatilità.  Senza leverage quell'allargamento è il punto
    (calibrazione onesta); **con** il leverage lascia il path sotto-determinato e la
    **deriva** di leverage riempie il vuoto: rho al bordo, phi che collassa per
    assorbirla — lo stesso modo di fallimento della forma ``literal``.

    Il test asserisce il **divario**: decoupled regge dove qml collassa.  Se un
    giorno qualcuno derivasse la mistura sign-augmented congiunta r-dimensionale che
    il .tex dichiara non derivata, questo test diventerebbe rosso — ed è esattamente
    il segnale che si vorrebbe.
    """
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

    # il default non deve poter arrivare al passo accoppiato per sbaglio
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
    _check("qml COLLASSA a corr(Q)=0.8 (un phi crolla) — instabilità congelata",
           bool(np.any(phi_q < 0.5)), f"phi={np.round(phi_q, 3)}")
    _check("qml spinge un rho al bordo — instabilità congelata",
           bool(np.max(np.abs(rho_q)) > 0.85), f"rho={np.round(rho_q, 3)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slow", action="store_true", help="include il recovery QML (lento)")
    a = p.parse_args()

    print("\n[1] Branch A riparato — blocco Laplace + Metropolis esatto")
    test_invariance()
    test_block_beats_single()
    test_no_warm_seed_needed()

    print("\n[2] QML sotto leverage (Branch B)")
    test_qml_constants()
    w = load_warm_init("small")
    theta, fl, bm, oc, r = (w["theta"], w["freq_list"], w["block_map"],
                            w["ordered_cols"], w["r"])
    test_guards(theta, fl, bm, oc, r)
    if a.slow:
        test_qml_leverage_instability(theta, fl, bm, oc, r)
    else:
        print("  [skip] test_qml_leverage_instability (usa --slow)")

    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
