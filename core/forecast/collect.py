"""
core/forecast/collect.py — DA `csv/_cells/` A `csv/dfm/`: la pubblicazione.

Un passo solo, e sta qui perche' lo fanno in due: `scripts/run_dfm.py` alla
fine della sua cella (cosi' uno shard e' subito leggibile) e
`scripts/run_outputs.py` all'inizio della catena ex post (cosi' un albero di
celle arrivato dal server, o rimasto da una passata precedente, si pubblica
senza ri-stimare nulla).  Scritto due volte, sarebbe divergiuto una volta.

PERCHE' DUE CARTELLE
--------------------
`weekly_nowcast` scrive un file che si chiama `weekly_nowcast_<inizio>_<fine>`
e basta: per quindici celle sullo stesso periodo e' lo STESSO nome.  Ogni
cella ha percio' la sua sottocartella in `csv/_cells/`, dove i quindici file
convivono; la pubblicazione li ricopia in `csv/dfm/` mettendo la cella nel
nome, che e' la forma che figure, metriche e tabelle si aspettano.

SI COPIA, NON SI SPOSTA
-----------------------
Il file in `csv/_cells/` e' lo stato di ripresa della cella: e' da li' che un
rilancio riparte.  Spostandolo, ogni ripresa ricomincerebbe da zero.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

from core import output_layout as layout

#: `weekly_nowcast_2007-01-01_2025-12-31.csv` -> ('2007-01-01', '2025-12-31')
_SPAN = re.compile(r"weekly_nowcast_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def published_name(spec: str, variant: str, start: str, end: str) -> str:
    """Il nome con cui una cella compare in `csv/dfm/`."""
    return f"weekly_nowcast_{spec}_{variant}_{start}_{end}.csv"


def publish_cell(spec: str, variant: str, start: str, end: str) -> str:
    """
    Copia il CSV di UNA cella da `csv/_cells/<cella>/` a `csv/dfm/`.

    Torna il percorso pubblicato.  Solleva `FileNotFoundError` se la cella non
    ha scritto: e' un guasto, non un caso al contorno.
    """
    src = os.path.join(layout.dfm_cell_dir(spec, variant),
                       f"weekly_nowcast_{start}_{end}.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"la cella {spec}/{variant} non ha un CSV su {start}..{end}:\n"
            f"    {src}")
    dst_dir = layout.dfm_csv_dir()
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, published_name(spec, variant, start, end))
    shutil.copyfile(src, dst)
    return dst


def collect_cells(verbose: bool = True) -> list[str]:
    """
    Pubblica TUTTE le celle che hanno scritto qualcosa in `csv/_cells/`.

    Idempotente: si puo' rilanciare, ricopia e basta.  Non pretende che le
    quindici ci siano — a dirlo se manca qualcosa e' la guardia
    `tests/forecast/test_cells_produced.py`, che guarda anche DENTRO i file.
    Qui si guarda solo l'albero.
    """
    root = os.path.dirname(layout.dfm_cell_dir("x", "y"))
    pubblicati: list[str] = []
    for cella in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for src in sorted(glob.glob(os.path.join(root, cella,
                                                 "weekly_nowcast_*.csv"))):
            m = _SPAN.search(os.path.basename(src))
            if not m:
                continue
            # Il nome della cartella e' '<spec>_<variante>': la spec e' il
            # prefisso noto, il resto e' la variante (che contiene '_').
            spec = next((s for s in layout.SPECS
                         if cella.startswith(s + "_")), None)
            if spec is None:
                if verbose:
                    print(f"  [salto] {cella}: non e' una cella nota")
                continue
            variant = cella[len(spec) + 1:]
            dst = publish_cell(spec, variant, m.group(1), m.group(2))
            pubblicati.append(dst)
            if verbose:
                print(f"  {spec}/{variant} -> {os.path.basename(dst)}")
    if verbose:
        print(f"  {len(pubblicati)} CSV in {layout.dfm_csv_dir()}")
    return pubblicati


def main() -> int:
    print("=" * 78)
    print("  RACCOLTA — csv/_cells/ -> csv/dfm/")
    print("=" * 78)
    return 0 if collect_cells() else 1


if __name__ == "__main__":
    raise SystemExit(main())
