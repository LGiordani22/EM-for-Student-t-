"""
GUARDIA: i CSV raccolti del DFM contengono davvero dei nowcast?

PERCHE' ESISTE QUESTO FILE
--------------------------
La passata del server ha prodotto quindici CSV del DFM formalmente perfetti —
nome giusto, intestazione giusta, 2277 righe per cella, una riga per ogni
settimana attesa — e completamente vuoti: `nowcast_bea` assente ovunque,
`n_iter=-1` su tutte e 34 155 le righe, cioe' un'eccezione a ogni singola
chiamata di `nowcast()`.

La fase 4 di `run_all.sh` controllava una cosa sola:

    [[ -f "$src" ]] || fail "manca il CSV della cella ..."

Il file c'era.  Quindi la raccolta e' passata, e con lei le fasi 5, 6 e 7, che
hanno lavorato su CSV senza un solo numero dentro.  L'ESISTENZA DI UN FILE NON
E' LA PRESENZA DI UN RISULTATO, e questa guardia e' la differenza fra le due.

COSA CONTROLLA, E COSA DELIBERATAMENTE NON CONTROLLA
-----------------------------------------------------
Controlla che ogni cella (spec x variante) abbia prodotto dei nowcast, con la
stessa regola di `weekly_nowcast.cell_health`:

  ROTTA        zero nowcast, oppure piu' di `_MAX_ERROR_FRAC` di righe con
               `n_iter=-1` (l'eccezione: nessun numero prodotto).

  NON rotta    righe con `converged=False` ma `n_iter>=0`.  L'EM ha girato e
               ha prodotto un nowcast, esaurendo `max_iter` o fermandosi su un
               massimo locale.  E' un RISULTATO da leggere nel merito — magari
               deludente — non un guasto d'impianto, e non deve fermare una
               passata da giorni.  Si conta e si stampa; basta.

Non controlla la QUALITA' dei numeri (RMSE, bias, plausibilita'): non e' il
suo mestiere e non e' quello che e' andato storto.  Se ne occupano
`metrics_tables` e `compare_nyfed`, a valle.

USO
---
    python -m src.forecast.test_cells_produced                 # i CSV raccolti
    python -m src.forecast.test_cells_produced --dir <cartella>

Esce 0 se tutte le celle trovate hanno prodotto, 1 altrimenti.  Con `--expect
N` pretende anche di TROVARE N celle: senza, una cella il cui CSV non e' mai
stato scritto passerebbe per assenza invece che per merito.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from src.forecast.weekly_nowcast import cell_health


def check_dir(csv_dir: str, expect: int | None = None) -> list[dict]:
    """
    Referto su tutti i CSV di `csv_dir`.  Ritorna le celle rotte.

    Legge i file uno per uno e non concatenati: un CSV illeggibile e' esso
    stesso un guasto da riportare con il suo nome, non un'eccezione anonima a
    meta' di una `pd.concat`.
    """
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        print(f"  NESSUN CSV in {csv_dir}")
        return [{"spec": "-", "variant": "-", "n": 0, "n_ok": 0,
                 "n_err": 0, "n_noconv": 0, "frac_err": 1.0, "rotta": True}]

    health: list[dict] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ILLEGGIBILE  {os.path.basename(p)}: "
                  f"{type(exc).__name__}: {exc}")
            health.append({"spec": os.path.basename(p), "variant": "-",
                           "n": 0, "n_ok": 0, "n_err": 0, "n_noconv": 0,
                           "frac_err": 1.0, "rotta": True})
            continue
        health.extend(cell_health(df))

    print(f"\n  {'cella':<38} {'righe':>6} {'con nowcast':>12} "
          f"{'ERRORE':>8} {'non conv.':>10}")
    for h in sorted(health, key=lambda x: (x["spec"], x["variant"])):
        marca = "  <-- ROTTA" if h["rotta"] else ""
        print(f"  {h['spec'] + '/' + h['variant']:<38} {h['n']:>6} "
              f"{h['n_ok']:>12} {h['n_err']:>8} {h['n_noconv']:>10}{marca}")

    rotte = [h for h in health if h["rotta"]]

    # Una cella che non ha mai scritto il suo CSV non compare fra le rotte: non
    # c'e' niente da giudicare.  `--expect` e' cio' che la fa emergere.
    if expect is not None and len(health) < expect:
        print(f"\n  MANCANO {expect - len(health)} celle: "
              f"trovate {len(health)}, attese {expect}")
        rotte = rotte + [{"spec": "(celle mancanti)", "variant": "-",
                          "n": 0, "n_ok": 0, "n_err": 0, "n_noconv": 0,
                          "frac_err": 1.0, "rotta": True}]
    return rotte


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guardia: i CSV del DFM contengono dei nowcast?")
    p.add_argument("--dir", default=None,
                   help="cartella dei CSV (default: quella di output_layout)")
    p.add_argument("--expect", type=int, default=None,
                   help="quante celle ci si aspetta di trovare (es. 15)")
    a = p.parse_args()

    if a.dir:
        csv_dir = a.dir
    else:
        from src import output_layout as _layout
        csv_dir = _layout.dfm_csv_dir()

    print("=" * 78)
    print(f"  GUARDIA CELLE PRODOTTE — {csv_dir}")
    print("=" * 78)

    rotte = check_dir(csv_dir, expect=a.expect)

    if rotte:
        print(f"\n  {'!' * 74}")
        print("  !!  IL DFM NON HA PRODOTTO.  Le figure e le tabelle a valle")
        print("  !!  lavorerebbero su CSV senza numeri dentro.")
        for h in rotte:
            motivo = ("nessun nowcast" if h["n_ok"] == 0
                      else f"{h['n_err']} righe su {h['n']} in errore "
                           f"({h['frac_err']:.1%})")
            print(f"  !!    {h['spec']}/{h['variant']}: {motivo}")
        print("  !!")
        print("  !!  I messaggi d'eccezione stanno nei log della fase 3")
        print("  !!  (output/_logs/dfm_<spec>_<variante>.log).")
        print(f"  {'!' * 74}")
        sys.exit(1)

    print("\n  Tutte le celle hanno prodotto nowcast.")
    print("  ('non conv.' non e' un guasto: l'EM ha girato e ha dato un numero,")
    print("   esaurendo max_iter o fermandosi su un massimo locale.)")


if __name__ == "__main__":
    main()
