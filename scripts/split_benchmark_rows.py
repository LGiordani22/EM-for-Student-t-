#!/usr/bin/env python3
"""
scripts/split_benchmark_rows.py — UNA TANTUM: i benchmark fuori dalle celle.

Fino a oggi AR(2) e media espandente si calcolavano dentro la PRIMA cella
dell'ordine canonico (`diag3/gaussian`) e le loro righe finivano nel CSV di
quella cella, sotto la pseudo-spec `benchmark`.  Quel file conteneva percio'
TRE serie sotto il nome di UNA — 6831 righe dove ne dichiarava 2277 — e chi lo
apriva a mano leggeva tre passate sovrapposte.

Ora i benchmark sono un lavoro a se' (`dfm/_cells/benchmark/` ->
`dfm/csv/benchmark/`).  Questo script porta l'albero che sta gia' sul disco nella
forma nuova SENZA ri-stimare niente: sposta le righe, non le ricalcola.

    python scripts/split_benchmark_rows.py            dice cosa farebbe
    python scripts/split_benchmark_rows.py --apply    lo fa

E' IDEMPOTENTE: su un albero gia' separato non trova righe da spostare e non
tocca niente.  Serve una volta per macchina — poi si puo' cancellare.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import output_layout as layout
from src.forecast.benchmarks import BENCHMARK_SPEC


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sposta le righe di benchmark dalle celle alla loro cartella.")
    p.add_argument("--apply", action="store_true",
                   help="scrive davvero (senza, dice solo cosa farebbe)")
    a = p.parse_args()

    root = layout.dfm_cells_root()
    if not os.path.isdir(root):
        print(f"Non c'e' niente da separare: manca {root}.")
        return 0

    dest_dir = layout.benchmark_cell_dir()
    n_mossi = 0

    for cella in sorted(os.listdir(root)):
        if cella == layout.BENCHMARK_JOB:
            continue
        for src in sorted(glob.glob(os.path.join(root, cella,
                                                 "weekly_nowcast_*.csv"))):
            d = pd.read_csv(src)
            if "spec" not in d.columns:
                continue
            mask = d["spec"] == BENCHMARK_SPEC
            if not mask.any():
                continue

            bench, cell = d[mask].copy(), d[~mask].copy()
            dst = os.path.join(dest_dir, os.path.basename(src))
            print(f"  {cella}: {len(d)} righe -> {len(cell)} nella cella, "
                  f"{len(bench)} in {os.path.relpath(dst, os.getcwd())}")

            # Se il file dei benchmark esiste gia', si fondono e si tengono le
            # righe uniche: due celle potrebbero averne portate una copia.
            if a.apply:
                os.makedirs(dest_dir, exist_ok=True)
                if os.path.exists(dst):
                    bench = pd.concat([pd.read_csv(dst), bench],
                                      ignore_index=True)
                    bench = bench.drop_duplicates(
                        ["as_of", "target_quarter", "spec", "variant"],
                        keep="last")
                bench.sort_values(["as_of", "target_quarter", "variant"],
                                  kind="stable").to_csv(dst, index=False)
                cell.to_csv(src, index=False)
            n_mossi += len(bench)

    if n_mossi == 0:
        print("  nessuna riga di benchmark dentro le celle: gia' separati.")
    elif not a.apply:
        print("\n  (prova: rilancia con --apply per scrivere)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
