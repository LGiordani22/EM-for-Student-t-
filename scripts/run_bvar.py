#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_bvar.py — UN BLOCCO BVAR, dall'inizio alla fine.

    python scripts/run_bvar.py --start 2007-01-05 --end 2007-04-27

L'UNITA' ATOMICA E' IL BLOCCO: una stima completa piu' le settimane che ne
riusano il risultato.  E' indipendente dalle altre PER COSTRUZIONE — ogni
stima parte dal pannello a `as_of`, non dallo stato della precedente — quindi
i blocchi si possono lanciare tutti insieme, in qualunque ordine.  E' la
differenza col DFM, dove la catena dei theta impone la sequenza.

I CONFINI NON SI SCRIVONO A MANO
--------------------------------
    python scripts/run_bvar.py --list-blocks       'inizio fine' per riga

Li taglia `evaluate.parallel_blocks`, sulle settimane di STIMA PIENA che
seguono le release BEA: e' l'unico confine che non costa una stima in piu' e
non cambia una riga rispetto a una passata continua.  Sul 2007-2025 sono 77.
Il vecchio taglio annuale scritto a mano cadeva a Capodanno — in mezzo a un
trimestre — e promuoveva diciannove settimane da riuso a stima fresca: numeri
che dipendevano da come era stato affettato il lavoro, non dal modello.

Senza `--start/--end` si percorre l'intero 2007-2025 in un processo solo:
corretto, ma e' la passata seriale.  Il modo parallelo e' un blocco per
processo, presi da `--list-blocks`.

CHE COSA SCRIVE
---------------
    output/forecast_weekly/bvar/csv/     i nowcast, i quantili .npz, i log score
                                         (un file per blocco, il periodo nel nome)
    output/_checkpoint/                  lo stato di ripresa, fuori dalla consegna

I quattro modelli — qbvar, cbvar, bbvar, lbvar — girano tutti dentro lo stesso
blocco: condividono il pannello e la cache di quella settimana.

RIPRESA
-------
Rilanciare lo stesso comando: ogni settimana conclusa ha il suo marcatore e si
salta.  `--fresh` non e' esposto qui apposta — cancellerebbe il checkpoint, e
con esso la ripresa.  Chi lo vuole passa dal modulo:
`python -m src.bvar.evaluate --fresh ...`.

PRIMA DI PAGARE LA PASSATA
--------------------------
`--dry-run` percorre il calendario e stampa lo schema — quante stime complete,
quante settimane di riuso, quali date — senza stimare nulla.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import output_layout as layout


def main() -> int:
    start_def, end_def = layout.FULL_SPAN
    p = argparse.ArgumentParser(
        description="Un blocco BVAR sul calendario settimanale real-time.")
    p.add_argument("--start", default=start_def, help=f"default: {start_def}")
    p.add_argument("--end", default=end_def, help=f"default: {end_def}")
    p.add_argument("--models", default=",".join(layout.BVAR_MODELS),
                   help="quali dei quattro stimare (default: tutti)")
    p.add_argument("--list-blocks", action="store_true",
                   help="stampa i blocchi ('inizio fine' per riga) ed esce")
    p.add_argument("--dry-run", action="store_true",
                   help="percorre il calendario, non stima nulla")
    p.add_argument("--no-benchmarks", action="store_true",
                   help="niente righe ar2/mean (sconsigliato: senza, "
                        "RMSE_rel_ar2 esce vuoto)")
    p.add_argument("--draws", type=int, default=None,
                   help="scavalca S per tutti i modelli: e' la leva di un "
                        "pilota, NON un'opzione della passata vera")
    p.add_argument("--output-root", default=None,
                   help="radice alternativa per csv/ e csv/logscore/. UN "
                        "PILOTA VA SEMPRE LANCIATO CON QUESTO: i suoi CSV a "
                        "estrazioni ridotte, lasciati fra gli artefatti veri, "
                        "finirebbero nelle tabelle insieme alla passata buona")
    p.add_argument("--keep-cache", action="store_true",
                   help="non cancellare le cache_*.pkl a blocco finito")
    p.add_argument("--max-cache-mb", type=float, default=None,
                   help="tetto per una cache su disco; oltre, la ripresa "
                        "riavvolge all'ultima stima piena (default: quello "
                        "di src.bvar.evaluate)")
    a = p.parse_args()

    from src.bvar.evaluate import MAX_CACHE_MB, parallel_blocks, run_realtime

    if a.list_blocks:
        for bs, be in parallel_blocks(a.start, a.end):
            print(f"{bs} {be}")
        return 0

    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    sconosciuti = [m for m in models if m not in layout.BVAR_MODELS]
    if sconosciuti:
        p.error(f"modelli sconosciuti: {sconosciuti}. "
                f"Noti: {list(layout.BVAR_MODELS)}")

    print("=" * 78)
    print(f"  BVAR  {', '.join(models)}   {a.start} .. {a.end}")
    print("=" * 78)

    run_realtime(
        a.start, a.end, models,
        output_root=a.output_root,
        dry_run=a.dry_run,
        benchmarks=not a.no_benchmarks,
        fresh=False,
        max_cache_mb=(MAX_CACHE_MB if a.max_cache_mb is None else a.max_cache_mb),
        keep_cache=a.keep_cache,
        n_draws=({m: a.draws for m in models} if a.draws else None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
