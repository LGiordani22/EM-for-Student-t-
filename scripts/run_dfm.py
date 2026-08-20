#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_dfm.py — UNA CELLA DFM, dall'inizio alla fine.

    python scripts/run_dfm.py --spec diag3 --variant student_t

L'UNITA' ATOMICA E' LA CELLA (spec x variante), e non e' una scelta di comodo:
dentro una cella i venerdi' sono SEQUENZIALI, perche' ogni ri-stima parte dal
theta del vintage precedente.  Non si spezza una cella per data senza spezzare
quella catena.  Fra celle invece non c'e' nessun legame: le quindici si possono
lanciare tutte insieme.

    python scripts/run_dfm.py --list        le quindici celle, una per riga

QUANTE, E QUANTO COSTANO
------------------------
3 strutture di caricamento (diag3, diag4, fed_overlap) x 5 varianti (gaussian,
gaussian_ar1, student_t, student_t_ar1, student_t_ar1_shared) = 15 celle sul
2007-01-01 .. 2025-12-31, 991 venerdi' ciascuna.  Non costano uguale: le
varianti `_ar1` stanno un ordine di grandezza sopra le altre, e la piu' lenta
(fed_overlap/student_t_ar1) da sola ha preso 149 ore in un processo solo.
Il tempo di parete di una passata parallela e' quello della cella piu' lenta.

CHE COSA SCRIVE
---------------
    output/forecast_weekly/csv/_cells/<spec>_<variante>/weekly_nowcast_*.csv
        il risultato intermedio, e insieme lo stato di ripresa della cella
    output/forecast_weekly/csv/dfm/weekly_nowcast_<spec>_<variante>_*.csv
        la copia pubblicata, che figure e tabelle leggono

Si copia e non si sposta: spostando il primo, un rilancio ripartirebbe da zero.

RIPRESA
-------
Rilanciare lo stesso comando.  Niente viene cancellato all'avvio: le settimane
gia' nel CSV si saltano e si riprende dalla prima che manca.  Per rifare una
cella davvero da capo si cancella la sua cartella in `csv/_cells/`.

I BENCHMARK LI CALCOLA UNA CELLA SOLA
-------------------------------------
AR(2) e media espandente sono univariati: non dipendono ne' dalla spec ne'
dalla variante, e quindici copie identiche gonfierebbero il conteggio `n`
delle tabelle senza cambiare un solo RMSE.  Li prende la PRIMA cella
dell'ordine canonico (diag3/gaussian), da sola e senza doverlo dire — cosi'
l'invocazione di uno shard non ha bisogno di sapere che cosa fanno gli altri.
`--benchmarks` / `--no-benchmarks` scavalcano la regola.

CODICE D'USCITA
---------------
0 se la cella ha prodotto nowcast, 1 se e' rotta — cioe' se ha scritto righe
senza un solo numero dentro (`n_iter=-1`: un'eccezione a ogni settimana).
Un EM che ha solo FATICATO (`converged=False` con un nowcast in mano) non e'
un guasto e non ferma niente: e' un risultato da leggere nel merito.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import output_layout as layout
from core.forecast.collect import publish_cell


def cells() -> list[tuple[str, str]]:
    """Le quindici celle nell'ordine canonico. La prima porta i benchmark."""
    return [(s, v) for s in layout.SPECS for v in layout.VARIANTS]


def main() -> int:
    start_def, end_def = layout.FULL_SPAN
    p = argparse.ArgumentParser(
        description="Una cella DFM (spec x variante) sul calendario settimanale.")
    p.add_argument("--spec", choices=layout.SPECS,
                   help="struttura di caricamento")
    p.add_argument("--variant", choices=layout.VARIANTS,
                   help="variante del modello")
    p.add_argument("--start", default=start_def, help=f"default: {start_def}")
    p.add_argument("--end", default=end_def, help=f"default: {end_def}")
    p.add_argument("--list", action="store_true",
                   help="stampa le celle ('spec variante' per riga) ed esce")
    p.add_argument("--benchmarks", dest="benchmarks", action="store_true",
                   default=None, help="calcola ar2 e mean in questa cella")
    p.add_argument("--no-benchmarks", dest="benchmarks", action="store_false",
                   help="non calcolarli (li porta un'altra cella)")
    p.add_argument("--em-frequency", choices=["weekly", "monthly"],
                   default="monthly", help="quando ri-stimare l'EM")
    p.add_argument("--n-ahead", type=int, default=1,
                   help="trimestri oltre quello corrente")
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--verbose-em", action="store_true")
    a = p.parse_args()

    if a.list:
        for spec, variant in cells():
            print(f"{spec} {variant}")
        return 0

    if not a.spec or not a.variant:
        p.error("servono --spec e --variant (oppure --list).")

    # La prima cella dell'ordine canonico porta i benchmark, se non si dice altro.
    benchmarks = (cells()[0] == (a.spec, a.variant)) if a.benchmarks is None \
        else a.benchmarks

    cell_dir = layout.dfm_cell_dir(a.spec, a.variant)
    os.makedirs(cell_dir, exist_ok=True)

    print("=" * 78)
    print(f"  DFM  {a.spec} / {a.variant}   {a.start} .. {a.end}")
    print(f"  benchmark: {'si' if benchmarks else 'no'}       cartella: {cell_dir}")
    print("=" * 78)

    from core.forecast.weekly_nowcast import cell_health, run_weekly_nowcast

    df = run_weekly_nowcast(
        a.start, a.end, specs=(a.spec,), variants=(a.variant,),
        em_frequency=a.em_frequency, n_ahead=a.n_ahead, max_iter=a.max_iter,
        benchmarks=benchmarks, output_dir=cell_dir, save=True,
        verbose_em=a.verbose_em,
    )

    rotte = [h for h in cell_health(df) if h["rotta"]]
    if rotte:
        print("\nLa cella non ha prodotto nowcast: il referto e' qui sopra.")
        print("PRIMA di rilanciare, cancellare il CSV vuoto — altrimenti la "
              "ripresa lo considera 'gia' fatto' e lo salta:")
        print(f"    {cell_dir}")
        return 1

    # ── La pubblicazione ─────────────────────────────────────────────────────
    # Si COPIA, non si sposta: il file nella cartella della cella e' lo stato
    # di ripresa.  Lo stesso passo lo rifa' `run_outputs.py` su tutto l'albero,
    # per le celle arrivate da un'altra macchina; il codice sta in un posto solo.
    try:
        dst = publish_cell(a.spec, a.variant, a.start, a.end)
    except FileNotFoundError as e:
        print("\n" + str(e))
        return 1
    print("\n  pubblicato -> " + dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
