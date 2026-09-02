"""
src/forecast/test_guardia_pca.py

LA GUARDIA SULLA RIPARTENZA PCA: le vie d'uscita, e il contatore.

    python -m src.forecast.test_guardia_pca

`guardia_pca` decide se ADOTTARE il theta appena ripartito a freddo, dopo che
`theta_problema()` ha gia' detto che e' valido.  Sono due controlli in serie:
il primo chiede "e' utilizzabile?", il secondo "e' meglio di quello che
avevamo?".

QUESTO TEST NON STIMA NIENTE.  Sostituisce `obiettivo_a_theta_fermo` con una
funzione finta che ritorna il numero che gli si dice: cio' che si verifica e' la
LOGICA DELLA DECISIONE, non il valore dell'obiettivo — quello e' l'ELBO dell'EM
ed e' gia' provato altrove.  Cosi' il test gira in millisecondi e non dipende
dal pannello.

LE COSE CHE DEVE PROTEGGERE, e perche' ognuna e' li':

  1. la guardia puo' solo NON adottare un theta nuovo; non deve MAI poter far
     fallire una passata.  Se il confronto stesso solleva, si adotta la PCA;
  2. sui PARI vince la PCA: il default resta il comportamento di prima della
     guardia, e ci si scosta solo con una prova a carico;
  3. alla prima stima della cella non c'e' niente da confrontare;
  4. dopo `_MAX_RIFIUTI_PCA` rifiuti CONSECUTIVI la PCA si adotta d'ufficio.
     Senza, per assurdo si potrebbe tenere lo stesso theta per sempre;
  5. il contatore si azzera appena la catena riprende a muoversi.
"""

from __future__ import annotations

import sys

from src.forecast import weekly_nowcast as wn

_A = {"A": [[0.5]]}          # theta finti: alla guardia serve solo `A` per r
_B = {"A": [[0.5]]}


def _con_obiettivi(prec: float, pca: float):
    """Sostituisce il calcolo dell'obiettivo con due numeri dati."""
    def finto(as_of, q, spec, variant, theta, panel, metadata):
        return prec if theta is _A else pca
    return finto


def _chiama(prec, pca, theta_prec=_A, rifiuti=0, esplode=False):
    vero = wn.obiettivo_a_theta_fermo
    if esplode:
        def boom(*a, **k):
            raise RuntimeError("il giudice e' rotto")
        wn.obiettivo_a_theta_fermo = boom
    else:
        wn.obiettivo_a_theta_fermo = _con_obiettivi(prec, pca)
    try:
        return wn.guardia_pca("2021-01-01", "2020Q4", "diag3", "student_t_ar1",
                              theta_prec, _B, None, None,
                              rifiuti_consecutivi=rifiuti)
    finally:
        wn.obiettivo_a_theta_fermo = vero


def check() -> int:
    fails = 0
    print(f"{'=' * 78}\n  GUARDIA PCA\n{'=' * 78}")

    casi = [
        ("PCA peggiore -> si rifiuta",
         dict(prec=-9358.0, pca=-9546.0), False),
        ("PCA migliore -> si adotta",
         dict(prec=-9401.0, pca=-9377.0), True),
        ("pari -> vince la PCA (default di prima della guardia)",
         dict(prec=-9400.0, pca=-9400.0), True),
        ("prima stima della cella -> si adotta",
         dict(prec=-9358.0, pca=-9546.0, theta_prec=None), True),
        ("il confronto solleva -> si adotta, la passata non si ferma",
         dict(prec=0.0, pca=0.0, esplode=True), True),
        (f"{wn._MAX_RIFIUTI_PCA} rifiuti consecutivi -> adottata d'ufficio",
         dict(prec=-9358.0, pca=-9546.0, rifiuti=wn._MAX_RIFIUTI_PCA), True),
        (f"{wn._MAX_RIFIUTI_PCA - 1} rifiuti -> ancora si rifiuta",
         dict(prec=-9358.0, pca=-9546.0, rifiuti=wn._MAX_RIFIUTI_PCA - 1),
         False),
    ]
    for nome, kw, atteso in casi:
        adotta, perche = _chiama(**kw)
        ok = adotta is atteso
        fails += not ok
        print(f"  {'OK ' if ok else 'ROTTA'}  {nome}")
        print(f"           -> {'adotta' if adotta else 'rifiuta'}: {perche}")

    # L'etichetta d'ufficio deve DIRSI: e' l'unico modo in cui un theta fermo da
    # un trimestre diventa visibile senza aprire i .npz a mano.
    _, perche = _chiama(prec=-9358.0, pca=-9546.0,
                        rifiuti=wn._MAX_RIFIUTI_PCA)
    ok = "D'UFFICIO" in perche
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  l'adozione d'ufficio si dichiara nel "
          f"motivo")

    # ── Il contatore: la regola di azzeramento, sulle due strade ─────────────
    print(f"\n{'=' * 78}\n  IL CONTATORE\n{'=' * 78}")

    def avanza(rifiuti: int, origine: str | None, adottato: bool = True) -> int:
        """La regola scritta nei due orchestratori, in un posto solo per il test."""
        if adottato:
            return rifiuti + 1 if origine == "ripiego" else 0
        return rifiuti

    prove = [
        ("un ripiego incrementa", 0, "ripiego", True, 1),
        ("due ripieghi di fila", 1, "ripiego", True, 2),
        ("una PCA adottata azzera", 2, "pca", True, 0),
        ("una stima calda azzera", 2, "warm", True, 0),
        ("una settimana congelata non tocca niente", 2, None, False, 2),
    ]
    for nome, da, origine, adottato, atteso in prove:
        got = avanza(da, origine, adottato)
        ok = got == atteso
        fails += not ok
        print(f"  {'OK ' if ok else 'ROTTA'}  {nome}: {da} -> {got} "
              f"(atteso {atteso})")

    # Il referto non deve stampare niente quando non e' successo niente: e' la
    # differenza fra una riga che significa qualcosa e rumore a ogni cella.
    ok = wn.referto_guardia("diag3", "gaussian", []) == []
    fails += not ok
    print(f"\n  {'OK ' if ok else 'ROTTA'}  nessuna ripartenza -> referto muto")

    righe = wn.referto_guardia("diag3", "student_t_ar1", [
        ("2021-01-01", "2020Q4", "ripiego", "obiettivo pca=-9546 < prec=-9358"),
        ("2021-02-05", "2021Q1", "pca", "obiettivo pca=-9377 >= prec=-9401"),
    ])
    ok = any("2 ripartenze valutate, 1 rifiutate" in r for r in righe)
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  il referto conta le ripartenze e i "
          f"rifiuti")
    for r in righe:
        print("   " + r)
    return fails


if __name__ == "__main__":
    n = check()
    print("\nGUARDIA OK" if not n else f"\n{n} CONTROLLI ROTTI")
    sys.exit(1 if n else 0)
