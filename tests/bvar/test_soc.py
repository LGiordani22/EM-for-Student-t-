"""
core/bvar/tests/test_soc.py

Il test del Blocco 3: il sum-of-coefficients e il meccanismo delle dummy
observations.

    python -m core.bvar.tests.test_soc

Come al Blocco 2, i controlli sono scritti come PROPRIETA' DELLA TEORIA: ogni
assert corrisponde a una frase del paper, non a un numero magico.

Il controllo centrale e' il primo: la dummy del soc deve essere soddisfatta
ESATTAMENTE quando sum_s A_s = diag(d) e violata altrimenti.  E'
quello che dimostra che le righe fittizie codificano davvero il postulato
"somma dei propri lag = 1, somma dei lag altrui = 0" e non qualcos'altro.
"""

from __future__ import annotations

import numpy as np

from core.bvar import data as bdata
from core.bvar.dummies import (
    implied_prior_moments,
    initial_observation_mean,
    minnesota_omega_diag,
    minnesota_prior_mean,
    sum_of_coefficients_dummy,
)
from core.bvar.spec import BVARSpec, Hyper

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<64} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _fixture(n_small: int = 4, seed: int = 0):
    """Un sistema piccolo, per poter guardare le matrici a occhio."""
    spec = BVARSpec.from_config("Q")
    rng = np.random.default_rng(seed)
    psi = np.exp(rng.normal(-4.0, 1.5, size=spec.n))
    hyp = Hyper(lam=0.6, mu=1.0, psi=psi)
    y0 = np.abs(rng.normal(10.0, 4.0, size=spec.n)) + 1.0
    return spec, hyp, y0


def _B_from_lag_sum(spec, lag_sum: np.ndarray, split: np.ndarray | None = None):
    """
    Costruisce un B (k, n) i cui coefficienti sommano su tutti i lag a `lag_sum`.

    `lag_sum` e' (n, n) con la convenzione dell'equazione: lag_sum[h, j] e' la
    somma sui lag del coefficiente della variabile j nell'equazione h.  Nel
    layout di B (righe = regressori, colonne = equazioni) va trasposta.
    `split` distribuisce la somma fra i p lag (default: tutto sul primo).
    """
    n, p, k = spec.n, spec.p, spec.k
    if split is None:
        split = np.zeros(p)
        split[0] = 1.0
    B = np.zeros((k, n))
    for s in range(p):
        B[s * n: (s + 1) * n, :] = split[s] * lag_sum.T
    return B


# ─── 1. y0_bar ────────────────────────────────────────────────────────────────

def test_y0_bar() -> bool:
    print("\n1. y0_bar: la media delle prime p osservazioni (GLP §III)")
    spec = BVARSpec.from_config("Q")
    ok = True

    panel = np.arange(60, dtype=float).reshape(20, 3)
    y0 = initial_observation_mean(panel, p=5)
    ok &= _check("shape (n,)", y0.shape == (3,))
    ok &= _check("e' la media delle PRIME p righe, non di tutto il campione",
                 bool(np.allclose(y0, panel[:5].mean(axis=0))),
                 f"{y0} vs media totale {panel.mean(axis=0)}")
    ok &= _check("e infatti differisce dalla media di tutto il campione",
                 not np.allclose(y0, panel.mean(axis=0)))

    # rifiuta i NaN: e' l'invariante del core applicato anche qui
    bad = panel.copy()
    bad[2, 1] = np.nan
    try:
        initial_observation_mean(bad, p=5)
        ok &= _check("rifiuta NaN nelle prime p righe", False)
    except ValueError:
        ok &= _check("rifiuta NaN nelle prime p righe", True)
    try:
        initial_observation_mean(panel[:3], p=5)
        ok &= _check("rifiuta un campione piu' corto di p", False)
    except ValueError:
        ok &= _check("rifiuta un campione piu' corto di p", True)

    # sui dati veri
    est = bdata.build_panel(spec, end=bdata.estimation_end(spec))
    qe = est.index[est.index.month.isin((3, 6, 9, 12))]
    y0r = initial_observation_mean(est.loc[qe], spec.p)
    ok &= _check("funziona sul pannello vero (q_b, trimestrale)",
                 y0r.shape == (spec.n,) and np.all(np.isfinite(y0r)),
                 f"GDPC1 y0_bar = {y0r[spec.series.index('GDPC1')]:.4f}")
    return ok


# ─── 2. La forma delle dummy ──────────────────────────────────────────────────

def test_soc_shape() -> bool:
    print("\n2. La forma delle dummy soc: y+ = diag(y0/mu), x+ = [y+,...,y+, 0]")
    spec, hyp, y0 = _fixture()
    Yd, Xd = sum_of_coefficients_dummy(spec, hyp, y0)
    n, p, k = spec.n, spec.p, spec.k
    ok = True

    ok &= _check("Yd e' (n, n)", Yd.shape == (n, n), f"{Yd.shape}")
    ok &= _check("Xd e' (n, k)", Xd.shape == (n, k), f"{Xd.shape}")
    ok &= _check("una riga per variabile (n righe)", Yd.shape[0] == n)
    d = spec.minnesota.d
    atteso = np.diag(y0 / hyp.mu)
    atteso[d == 0, d == 0] = 0.0            # `ydnoc(pos,pos) = 0` degli autori
    ok &= _check("Yd = diag(y0_bar/mu) con le righe wn AZZERATE",
                 np.allclose(Yd, atteso),
                 f"{int((d == 0).sum())} serie wn su {n}")
    ok &= _check("...e le righe wn sono davvero zero",
                 np.allclose(np.diag(Yd)[d == 0], 0.0) if (d == 0).any() else True)
    ok &= _check("...mentre le rw valgono y0_bar/mu",
                 np.allclose(np.diag(Yd)[d == 1], (y0 / hyp.mu)[d == 1]))
    ok &= _check("Yd e' diagonale", bool(np.allclose(Yd, np.diag(np.diag(Yd)))))

    # p copie identiche di y+ nelle colonne dei lag
    copies = [Xd[:, s * n: (s + 1) * n] for s in range(p)]
    y_plus_full = np.diag(y0 / hyp.mu)   # Xd NON e' azzerata: solo Yd lo e'
    ok &= _check(f"Xd contiene {p} copie di y+ NON azzerata (solo Yd lo e')",
                 all(np.allclose(c, y_plus_full) for c in copies))
    ok &= _check("l'ultima colonna (costante) e' ZERO: il soc tace sulla costante",
                 bool(np.allclose(Xd[:, -1], 0.0)))

    # mu <= 0 rifiutato dal costruttore di Hyper stesso
    try:
        sum_of_coefficients_dummy(spec, hyp, y0[:3])
        ok &= _check("y0_bar di lunghezza sbagliata viene rifiutato", False)
    except ValueError:
        ok &= _check("y0_bar di lunghezza sbagliata viene rifiutato", True)
    return ok


# ─── 3. IL CONTROLLO CENTRALE: che cosa afferma la dummy ──────────────────────

def test_soc_postulate() -> bool:
    print("\n3. IL POSTULATO: residuo nullo se e solo se sum_s A_s = I  (Pi = 0)")
    spec, hyp, y0 = _fixture()
    Yd, Xd = sum_of_coefficients_dummy(spec, hyp, y0)
    n = spec.n
    ok = True

    # (a) Pi = 0 esatto -> residuo esattamente nullo.
    B_unit = _B_from_lag_sum(spec, np.diag(spec.minnesota.d))
    res = Yd - Xd @ B_unit
    ok &= _check("sum_s A_s = diag(d)  =>  residuo nullo",
                 bool(np.allclose(res, 0.0, atol=1e-10)),
                 f"max|res| = {np.abs(res).max():.2e}")

    # (b) e vale comunque si distribuisca la somma fra i lag: il soc parla della
    #     SOMMA, non dei singoli coefficienti.  E' il punto (ii) dell'header.
    split = np.array([0.5, -0.3, 0.4, 0.2, 0.2])          # somma 1.0, p=5
    B_split = _B_from_lag_sum(spec, np.diag(spec.minnesota.d), split=split)
    res2 = Yd - Xd @ B_split
    ok &= _check("...anche con la somma distribuita diversamente fra i lag",
                 bool(np.allclose(res2, 0.0, atol=1e-10)),
                 f"split={split}, max|res| = {np.abs(res2).max():.2e}")

    # (c) somma sui propri lag != 1 -> residuo NON nullo
    ls = np.diag(spec.minnesota.d).copy()
    ls[3, 3] = 0.8
    res3 = Yd - Xd @ _B_from_lag_sum(spec, ls)
    ok &= _check("somma dei PROPRI lag = 0.8 invece di 1  =>  residuo non nullo",
                 float(np.abs(res3).max()) > 1e-6,
                 f"max|res| = {np.abs(res3).max():.4g}")

    # (d) somma sui lag ALTRUI != 0 -> residuo NON nullo
    ls = np.diag(spec.minnesota.d).copy()
    ls[2, 7] = 0.15
    res4 = Yd - Xd @ _B_from_lag_sum(spec, ls)
    ok &= _check("somma dei lag ALTRUI = 0.15 invece di 0  =>  residuo non nullo",
                 float(np.abs(res4).max()) > 1e-6,
                 f"max|res| = {np.abs(res4).max():.4g}")

    # (e) la costante non conta: il soc tace su di essa
    B_c = _B_from_lag_sum(spec, np.diag(spec.minnesota.d))
    B_c[-1, :] = 999.0
    ok &= _check("cambiare la costante non tocca il residuo (soc tace)",
                 bool(np.allclose(Yd - Xd @ B_c, 0.0, atol=1e-10)))
    return ok


# ─── 4. mu: intensita', non posizione ─────────────────────────────────────────

def test_mu() -> bool:
    print("\n4. mu governa l'INTENSITA', non il punto verso cui si tira")
    spec, hyp, y0 = _fixture()
    n = spec.n
    ok = True

    Yd1, Xd1 = sum_of_coefficients_dummy(spec, Hyper(0.6, 1.0, hyp.psi), y0)
    Yd2, Xd2 = sum_of_coefficients_dummy(spec, Hyper(0.6, 2.0, hyp.psi), y0)

    ok &= _check("dimezzare mu raddoppia la magnitudine della dummy",
                 bool(np.allclose(Yd1, 2.0 * Yd2) and np.allclose(Xd1, 2.0 * Xd2)))

    # il PUNTO verso cui si tira (Pi = 0) e' lo stesso per ogni mu: il residuo
    # resta nullo.  E' la nota nel docstring di sum_of_coefficients_dummy.
    B_unit = _B_from_lag_sum(spec, np.diag(spec.minnesota.d))
    same = all(
        np.allclose(
            *(lambda Y, X: (Y, X @ B_unit))(
                *sum_of_coefficients_dummy(spec, Hyper(0.6, m, hyp.psi), y0)
            ),
            atol=1e-9,
        )
        for m in (0.01, 0.5, 1.0, 10.0, 100.0)
    )
    ok &= _check("il residuo resta nullo a diag(d) per OGNI mu (il punto non cambia)",
                 same)

    # mu -> 0 : dummy esplode  (prior dogmatico)
    Yd_t, _ = sum_of_coefficients_dummy(spec, Hyper(0.6, 1e-6, hyp.psi), y0)
    ok &= _check("mu->0: la dummy esplode (prior dogmatico, Pi=0 esatto)",
                 float(np.abs(Yd_t).max()) > 1e5,
                 f"max = {np.abs(Yd_t).max():.3g}")

    # mu -> inf : dummy svanisce (prior inesistente, resta la sola Minnesota)
    Yd_b, _ = sum_of_coefficients_dummy(spec, Hyper(0.6, 1e6, hyp.psi), y0)
    ok &= _check("mu->inf: la dummy svanisce (resta la sola Minnesota)",
                 float(np.abs(Yd_b).max()) < 1e-3,
                 f"max = {np.abs(Yd_b).max():.3g}")

    # il verso: mu piccolo = prior stretto, come lambda
    ok &= _check("verso confermato: mu piccolo = prior STRETTO (come lambda)",
                 float(np.abs(Yd_t).max()) > float(np.abs(Yd_b).max()))
    return ok


# ─── 5. Il meccanismo al contrario ────────────────────────────────────────────

def test_implied_moments() -> bool:
    print("\n5. implied_prior_moments: da dummy a momenti (identita' di BGR)")
    spec, hyp, y0 = _fixture()
    ok = True

    # (a) sul SOLO blocco soc deve FALLIRE: n righe per k colonne, rango <= n
    Yd, Xd = sum_of_coefficients_dummy(spec, hyp, y0)
    try:
        implied_prior_moments(Yd, Xd)
        ok &= _check("il solo blocco soc e' singolare -> errore esplicito", False)
    except np.linalg.LinAlgError as exc:
        ok &= _check("il solo blocco soc e' singolare -> errore esplicito",
                     "identificano il prior solo" in str(exc),
                     f"rango {spec.n} su k={spec.k}")

    # (b) round-trip su un caso sintetico a rango pieno: costruisco delle dummy
    #     DA un prior noto (b, Omega) e verifico di riottenerlo.
    #     Se Xd = L^-1 con Omega = L L' ... piu' semplice: scelgo Xd invertibile
    #     qualsiasi, pongo Omega = (Xd'Xd)^-1 e Yd = Xd @ b; allora le identita'
    #     di BGR devono restituire esattamente (b, Omega).
    rng = np.random.default_rng(7)
    k_small, n_small = 6, 3
    Xd_s = rng.normal(size=(k_small, k_small))
    b_true = rng.normal(size=(k_small, n_small))
    Yd_s = Xd_s @ b_true
    omega_true = np.linalg.inv(Xd_s.T @ Xd_s)

    b0, om0 = implied_prior_moments(Yd_s, Xd_s)
    ok &= _check("round-trip: B0 ritrova la media a priori",
                 bool(np.allclose(b0, b_true)),
                 f"max|err| = {np.abs(b0 - b_true).max():.2e}")
    ok &= _check("round-trip: Omega0 = (Xd'Xd)^-1",
                 bool(np.allclose(om0, omega_true)),
                 f"max|err| = {np.abs(om0 - omega_true).max():.2e}")

    # (c) piu' righe che colonne va benissimo (e' il caso dello stack completo)
    Xd_t = rng.normal(size=(20, k_small))
    Yd_t = rng.normal(size=(20, n_small))
    b0t, om0t = implied_prior_moments(Yd_t, Xd_t)
    ok &= _check("funziona con Td > k (il caso dello stack del Blocco 4)",
                 b0t.shape == (k_small, n_small) and om0t.shape == (k_small, k_small))

    # (d) righe incoerenti rifiutate
    try:
        implied_prior_moments(Yd_t[:5], Xd_t)
        ok &= _check("Yd e Xd con righe diverse vengono rifiutati", False)
    except ValueError:
        ok &= _check("Yd e Xd con righe diverse vengono rifiutati", True)
    return ok


# ─── 6. Minnesota e soc sono additivi, non alternativi ────────────────────────

def test_additivity() -> bool:
    print("\n6. Minnesota e soc: due prior DISTINTI, su oggetti diversi")
    spec, hyp, y0 = _fixture()
    n, p = spec.n, spec.p
    Yd, Xd = sum_of_coefficients_dummy(spec, hyp, y0)
    ok = True

    # (i) il soc NON dice nulla sui singoli coefficienti: due B con la stessa
    #     somma ma singoli diversissimi gli sono indifferenti...
    B_a = _B_from_lag_sum(spec, np.diag(spec.minnesota.d), split=np.array([1.0, 0, 0, 0, 0]))
    B_b = _B_from_lag_sum(spec, np.diag(spec.minnesota.d), split=np.array([1.4, -0.4, 0, 0, 0]))
    ok &= _check("il soc e' indifferente a come la somma si distribuisce sui lag",
                 bool(np.allclose(Yd - Xd @ B_a, 0.0, atol=1e-10)
                      and np.allclose(Yd - Xd @ B_b, 0.0, atol=1e-10)))

    # ...mentre la Minnesota li distingue eccome: B_b e' lontano dal suo centro.
    b_minn = minnesota_prior_mean(spec)
    om = minnesota_omega_diag(spec, hyp)[:-1]            # esclusa la costante
    dev_a = ((B_a - b_minn)[:n * p] ** 2 / om[:, None]).sum()
    dev_b = ((B_b - b_minn)[:n * p] ** 2 / om[:, None]).sum()
    ok &= _check("...ma la Minnesota li distingue: (1.4, -0.4) e' penalizzato",
                 dev_b > dev_a,
                 f"penalita' Minnesota: {dev_a:.3g} vs {dev_b:.3g}")

    # (ii) e viceversa: un B gradito alla Minnesota puo' non piacere al soc.
    #      A_1 = diag(0.9), resto zero -> somma 0.9, il soc protesta.
    B_c = _B_from_lag_sum(spec, 0.9 * np.eye(n))
    res_c = Yd - Xd @ B_c
    dev_c = ((B_c - b_minn)[:n * p] ** 2 / om[:, None]).sum()
    ok &= _check("un B vicino al centro Minnesota puo' violare il soc",
                 float(np.abs(res_c).max()) > 1e-6 and dev_c < dev_b,
                 f"residuo soc = {np.abs(res_c).max():.4g}, penalita' Minn = {dev_c:.3g}")

    # (iii) i due blocchi hanno righe diverse: sono additivi per costruzione
    ok &= _check("il blocco soc ha n righe, distinte da quelle della Minnesota",
                 Xd.shape[0] == n, f"{n} righe soc")
    return ok


# ─── 7. Sui dati veri ─────────────────────────────────────────────────────────

def test_on_real_data() -> bool:
    print("\n7. Sui dati veri (q_b, campione di stima)")
    spec = BVARSpec.from_config("Q")
    est = bdata.build_panel(spec, end=bdata.estimation_end(spec))
    qe = est.index[est.index.month.isin((3, 6, 9, 12))]
    panel = est.loc[qe]
    ok = True

    y0 = initial_observation_mean(panel, spec.p)
    hyp = Hyper(lam=0.6, mu=1.0, psi=np.full(spec.n, 0.02 ** 2))
    Yd, Xd = sum_of_coefficients_dummy(spec, hyp, y0)

    ok &= _check("le dummy si costruiscono senza NaN",
                 bool(np.all(np.isfinite(Yd)) and np.all(np.isfinite(Xd))))
    ok &= _check("sum_s A_s = diag(d) annulla il residuo anche coi y0_bar veri",
                 bool(np.allclose(
                     Yd - Xd @ _B_from_lag_sum(spec, np.diag(spec.minnesota.d)),
                     0.0, atol=1e-9)))
    # e la controprova che il test distingue: con I al posto di diag(d) il
    # residuo NON e' nullo, proprio per le 4 serie wn del profilo q_b.
    res_I = Yd - Xd @ _B_from_lag_sum(spec, np.eye(spec.n))
    ok &= _check("...mentre sum_s A_s = I NON lo annulla (ci sono serie wn)",
                 not np.allclose(res_I, 0.0, atol=1e-9),
                 f"max|res| = {np.abs(res_I).max():.3g} su "
                 f"{int((spec.minnesota.d == 0).sum())} serie wn")

    # con mu ~ 1 la dummy ha grandezza confrontabile con i dati: "vale piu' o
    # meno un'osservazione in piu' per variabile" (header, sezione MU)
    scale_dummy = float(np.abs(np.diag(Yd)).mean())
    scale_data = float(np.abs(panel.to_numpy()).mean())
    ratio = scale_dummy / scale_data
    ok &= _check("a mu=1 la dummy e' dello stesso ordine dei dati",
                 0.1 < ratio < 10.0,
                 f"dummy {scale_dummy:.3g} vs dati {scale_data:.3g}  (rapporto {ratio:.2f})")

    i_gdp = spec.series.index("GDPC1")
    print(f"     GDPC1: y0_bar = {y0[i_gdp]:.4f} (log-livello)  ->  "
          f"riga dummy a mu=1: {Yd[i_gdp, i_gdp]:.4f}")
    return ok


def main() -> bool:
    print("=" * 84)
    print("Gate 1 / Blocco 3 — il sum-of-coefficients e le dummy observations")
    print("=" * 84)
    ok = True
    for fn in (test_y0_bar, test_soc_shape, test_soc_postulate, test_mu,
               test_implied_moments, test_additivity, test_on_real_data):
        ok &= fn()
    print("\n" + "=" * 84)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 84)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
