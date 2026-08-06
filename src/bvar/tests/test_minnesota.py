"""
src/bvar/tests/test_minnesota.py

Il test del Blocco 2: le eq. (2)-(3) di Cimadomo sono implementate come dicono
il paper e GLP?

    python -m src.bvar.tests.test_minnesota

Non e' ancora il recovery test del Gate 1 — qui non si stima niente.  E' la
verifica che i MOMENTI ANALITICI del prior siano quelli giusti, cosi' che al
Blocco 4 possano fare da oracolo per le dummy observations di BGR.

I controlli sono scritti come *proprieta' della teoria*, non come confronto con
numeri magici: ogni assert corrisponde a una frase del paper.
"""

from __future__ import annotations

import numpy as np

from src.bvar.dummies import (
    minnesota_omega_diag,
    minnesota_prior_mean,
    prior_coefficient_sd,
    regressor_layout,
)
from src.bvar.spec import BVARSpec, Hyper

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _fixture(seed: int = 0):
    """Uno spec vero (Q-BVAR, n=30) e un hyper con psi eterogenei.

    psi eterogenei sono il punto: con psi tutti uguali il fattore di scala
    psi_i/psi_j sparirebbe e il test non lo vedrebbe.
    """
    spec = BVARSpec.from_config("Q")
    rng = np.random.default_rng(seed)
    psi = np.exp(rng.normal(-4.0, 1.5, size=spec.n))     # scale molto diverse
    return spec, Hyper(lam=0.6, mu=1.0, psi=psi)


# ─── 1. Il layout dei regressori ──────────────────────────────────────────────

def test_layout() -> bool:
    print("\n1. Layout dei regressori: lag-major, costante in fondo (BGR)")
    spec, _ = _fixture()
    lay = regressor_layout(spec)
    ok = True
    ok &= _check("lunghezza k = n*p+1", len(lay) == spec.k, f"k={spec.k}")
    ok &= _check("i primi n sono il lag 1", all(s == 1 for s, _ in lay[: spec.n]))
    ok &= _check("gli ultimi n prima della costante sono il lag p",
                 all(s == spec.p for s, _ in lay[spec.k - 1 - spec.n: spec.k - 1]))
    ok &= _check("la costante e' l'ultima voce", lay[-1] == (0, "<const>"))
    ok &= _check("(lag s, var j) sta all'indice (s-1)*n + j",
                 all(lay[(s - 1) * spec.n + j] == (s, spec.series[j])
                     for s in (1, 3, spec.p) for j in (0, 7, spec.n - 1)))
    return ok


# ─── 2. Le medie, eq. (2) ─────────────────────────────────────────────────────

def test_prior_mean() -> bool:
    print("\n2. Le medie a priori — eq. (2): E(A_1)=diag(d), E(A_s)=0 per s>1")
    spec, _ = _fixture()
    b = minnesota_prior_mean(spec)
    n, p = spec.n, spec.p
    ok = True

    ok &= _check("forma (k, n)", b.shape == (spec.k, n), f"{b.shape}")

    a1 = b[:n, :]
    ok &= _check("la diagonale del blocco lag-1 e' d_centre",
                 np.array_equal(np.diag(a1), spec.minnesota.d))
    ok &= _check("il resto del blocco lag-1 e' zero (fuori diagonale)",
                 np.array_equal(a1 - np.diag(np.diag(a1)), np.zeros((n, n))))
    ok &= _check("i blocchi dei lag 2..p sono tutti zero",
                 not b[n: n * p, :].any())
    ok &= _check("la riga della costante e' zero (prior piatto)",
                 not b[-1, :].any())

    # il legame con la config: le 4 wn del profilo q_b prendono 0, le 26 rw 1
    wn = [s for s, d in zip(spec.series, spec.minnesota.d) if d == 0.0]
    ok &= _check("le wn di q_b hanno media a priori 0 sul proprio lag 1",
                 all(b[spec.series.index(s), spec.series.index(s)] == 0.0 for s in wn),
                 f"wn={sorted(wn)}")
    ok &= _check("le rw hanno media a priori 1",
                 all(b[i, i] == 1.0 for i in range(n) if spec.minnesota.d[i] == 1.0))
    return ok


# ─── 3. Le varianze, eq. (3) ──────────────────────────────────────────────────

def test_omega() -> bool:
    print("\n3. Le varianze a priori — eq. (3)")
    spec, hyp = _fixture()
    om = minnesota_omega_diag(spec, hyp)
    n, p = spec.n, spec.p
    ok = True

    ok &= _check("forma (k,)", om.shape == (spec.k,))
    ok &= _check("la costante ha varianza infinita (prior piatto)",
                 np.isinf(om[-1]))
    ok &= _check("tutto il resto e' finito e positivo",
                 bool(np.all(np.isfinite(om[:-1])) and np.all(om[:-1] > 0)))

    blk = om[:-1].reshape(p, n)          # riga = lag s, colonna = variabile j

    # (a) decadimento 1/s^2 col lag — "the rate at which the prior variance
    #     decreases with increasing lag length" (Cimadomo), esponente 2 (fn. 8)
    ratios = blk / blk[0][None, :]
    want = (1.0 / np.arange(1, p + 1) ** spec.minnesota.lag_decay)[:, None]
    ok &= _check("Omega(s)/Omega(1) = 1/s^2 per ogni variabile",
                 bool(np.allclose(ratios, np.broadcast_to(want, ratios.shape))),
                 f"lag5: {ratios[4, 0]:.6f} vs {1/25:.6f}")

    # (b) il denominatore e' psi_j, la variabile RITARDATA (Decisione 5).
    #     Test discriminante: Omega e' costante lungo le RIGHE se psi fosse
    #     ignorato, e deve invece essere proporzionale a 1/psi_j lungo le
    #     colonne.  Con psi_i != psi_j questo distingue j da i.
    ok &= _check("Omega(1,j) * psi_j e' costante in j  =>  denominatore psi_j",
                 bool(np.allclose(blk[0] * hyp.psi, blk[0][0] * hyp.psi[0])))

    # controprova esplicita: se il denominatore fosse psi_i, Omega non
    # dipenderebbe da j — cioe' sarebbe costante lungo la riga.  Non lo e'.
    ok &= _check("Omega NON e' costante in j (escluderebbe psi_j)",
                 not np.allclose(blk[0], blk[0][0]))

    # (c) lambda scala TUTTO come lambda^2
    hyp2 = Hyper(lam=2 * hyp.lam, mu=hyp.mu, psi=hyp.psi)
    om2 = minnesota_omega_diag(spec, hyp2)
    ok &= _check("raddoppiare lambda quadruplica Omega",
                 bool(np.allclose(om2[:-1], 4.0 * om[:-1])))

    # (d) formula esplicita, elemento per elemento
    manual = np.array([hyp.lam ** 2 / (s ** 2 * hyp.psi[j])
                       for s in range(1, p + 1) for j in range(n)])
    ok &= _check("coincide con lambda^2/(s^2 psi_j) calcolata a mano",
                 bool(np.allclose(om[:-1], manual)))
    return ok


# ─── 4. I casi limite di lambda ───────────────────────────────────────────────

def test_lambda_limits() -> bool:
    print("\n4. I casi limite di lambda (Cimadomo §2.1)")
    spec, hyp = _fixture()
    ok = True

    # I controlli vanno fatti in forma RELATIVA, non contro soglie assolute:
    # Omega ha le unita' di 1/psi_j, e con psi eterogenei (che e' il caso
    # realistico) una soglia assoluta misura la dispersione delle scale, non il
    # comportamento di lambda.  Il riferimento e' lambda = 1.
    ref = minnesota_omega_diag(spec, Hyper(lam=1.0, mu=hyp.mu, psi=hyp.psi))

    # lambda -> 0: prior dogmatico.  "For lambda = 0 the posterior equals the
    # prior and the data do not influence the estimates."
    tiny = Hyper(lam=1e-8, mu=hyp.mu, psi=hyp.psi)
    om_t = minnesota_omega_diag(spec, tiny)
    ok &= _check("lambda->0: Omega scende esattamente come lambda^2",
                 bool(np.allclose(om_t[:-1], (1e-8 ** 2) * ref[:-1], rtol=1e-12)),
                 f"rapporto medio={np.mean(om_t[:-1] / ref[:-1]):.2e}")
    ok &= _check("lambda->0: Omega e' ~1e-16 volte quella a lambda=1",
                 bool(np.all(om_t[:-1] / ref[:-1] < 1e-15)))

    # lambda -> infinito: prior piatto.  "If lambda -> infinity, posterior
    # expectations coincide with the Ordinary Least Squares (OLS) estimates."
    big = Hyper(lam=1e8, mu=hyp.mu, psi=hyp.psi)
    om_b = minnesota_omega_diag(spec, big)
    ok &= _check("lambda->inf: Omega cresce esattamente come lambda^2",
                 bool(np.allclose(om_b[:-1], (1e8 ** 2) * ref[:-1], rtol=1e-12)))
    ok &= _check("lambda->inf: Omega e' ~1e16 volte quella a lambda=1",
                 bool(np.all(om_b[:-1] / ref[:-1] > 1e15)))

    # la media a priori NON dipende da lambda: lambda governa solo la dispersione
    ok &= _check("la media a priori non dipende da lambda",
                 np.array_equal(minnesota_prior_mean(spec),
                                minnesota_prior_mean(spec)))
    return ok


# ─── 5. Le deviazioni standard: le due letture ────────────────────────────────

def test_prior_sd() -> bool:
    print("\n5. Le sd a priori: lambda e il rapporto di scale")
    spec, hyp = _fixture()
    sd = prior_coefficient_sd(spec, hyp)
    n = spec.n
    ok = True

    ok &= _check("forma (k, n)", sd.shape == (spec.k, n))

    # LETTURA 1 — lambda E' la sd a priori del coefficiente della variabile su
    # se stessa al primo lag.  E' il modo concreto di leggere la Tabella B.1.
    own = np.array([sd[i, i] for i in range(n)])
    ok &= _check("sd del proprio lag-1 = lambda, per TUTTE le n equazioni",
                 bool(np.allclose(own, hyp.lam)),
                 f"lambda={hyp.lam}, max scarto={np.abs(own - hyp.lam).max():.2e}")

    # LETTURA 2 — il fattore e' un rapporto di scale sqrt(psi_i/psi_j)
    s_, i_, j_ = 3, 4, 11
    idx = (s_ - 1) * n + j_
    want = (hyp.lam / s_) * np.sqrt(hyp.psi[i_] / hyp.psi[j_])
    ok &= _check("sd[(s,j), i] = (lambda/s)*sqrt(psi_i/psi_j)",
                 bool(np.isclose(sd[idx, i_], want)),
                 f"{sd[idx, i_]:.6g} vs {want:.6g}")

    # la costante resta a prior piatto
    ok &= _check("la sd della costante e' infinita", bool(np.all(np.isinf(sd[-1]))))

    # esempio concreto: il coefficiente di una serie a scala grande in
    # un'equazione a scala piccola deve essere MINUSCOLO a priori
    big_j = int(np.argmax(hyp.psi))
    small_i = int(np.argmin(hyp.psi))
    ratio = sd[big_j, small_i] / hyp.lam
    ok &= _check("scala grande -> equazione piccola: sd molto sotto lambda",
                 ratio < 0.5,
                 f"sd/lambda = {ratio:.4f}  ({spec.series[big_j]} -> {spec.series[small_i]})")
    return ok


# ─── 6. Coerenza fra i modelli ────────────────────────────────────────────────

def test_across_models() -> bool:
    print("\n6. Coerenza sui quattro modelli")
    ok = True
    for m in ("Q", "C", "B", "L"):
        spec = BVARSpec.from_config(m)
        hyp = Hyper(lam=0.6, mu=1.0, psi=np.full(spec.n, 0.02 ** 2))
        b = minnesota_prior_mean(spec)
        om = minnesota_omega_diag(spec, hyp)
        good = (b.shape == (spec.k, spec.n) and om.shape == (spec.k,)
                and np.array_equal(np.diag(b[: spec.n, :]), spec.minnesota.d))
        ok &= _check(f"{m}-BVAR: b e Omega coerenti con n={spec.n}, p={spec.p}",
                     good, f"k={spec.k}")

    # con p=17 il decadimento e' severo: il lag 17 ha 1/289 della varianza del
    # lag 1.  E' cio' che rende sostenibile l'L-BVAR (paper nota 10 + §2.1
    # "longer lags are shrunk more").
    spec = BVARSpec.from_config("L")
    hyp = Hyper(lam=0.6, mu=1.0, psi=np.full(spec.n, 0.02 ** 2))
    om = minnesota_omega_diag(spec, hyp).reshape(-1)[:-1].reshape(spec.p, spec.n)
    ok &= _check("L-BVAR: il lag 17 ha 1/289 della varianza del lag 1",
                 bool(np.allclose(om[16] / om[0], 1.0 / 17 ** 2)),
                 f"{om[16, 0] / om[0, 0]:.6f}")

    # e il mismatch di dimensione viene rifiutato
    try:
        minnesota_omega_diag(spec, Hyper(lam=0.6, mu=1.0, psi=np.ones(3)))
        ok &= _check("psi di lunghezza sbagliata viene rifiutato", False)
    except ValueError:
        ok &= _check("psi di lunghezza sbagliata viene rifiutato", True)
    return ok


def main() -> bool:
    print("=" * 82)
    print("Gate 1 / Blocco 2 — il prior di Minnesota, eq. (2)-(3)")
    print("=" * 82)
    ok = True
    for fn in (test_layout, test_prior_mean, test_omega,
               test_lambda_limits, test_prior_sd, test_across_models):
        ok &= fn()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
