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

UN PERIODO SOLO PER VOLTA, E QUESTA E' UNA GUARDIA
--------------------------------------------------
`csv/dfm/` e' l'input di TUTTO il resto, e chi lo legge — `figures`,
`compute_metrics`, `metrics_tables` — prende ogni file che ci trova e li
CONCATENA.  Due periodi diversi nella stessa cartella (una prova sul 2016
rimasta accanto alla passata 2007-2025) non danno errore: danno righe
doppie sugli stessi (trimestre, settimana), cioe' RMSE calcolati due volte
sugli stessi punti.  `run_all.sh` non poteva incapparci perche' copiava un
nome di file esatto; qui si raccoglie tutto, quindi la guardia serve e sta
sotto, in `_un_periodo_solo`.
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


def benchmark_published_name(start: str, end: str) -> str:
    """Il nome con cui i benchmark compaiono in `csv/benchmark/`."""
    return f"weekly_nowcast_{layout.BENCHMARK_JOB}_{start}_{end}.csv"


def cell_parts(nome_cartella: str) -> tuple[str, str] | None:
    """
    `'fed_overlap_student_t_ar1'` -> `('fed_overlap', 'student_t_ar1')`.

    `None` se non e' una cella nota: la spec deve stare in `layout.SPECS` e la
    variante in `layout.VARIANTS`.  Serve perche' la variante contiene degli
    underscore quanto il nome della cartella, quindi lo spezzone non si puo'
    indovinare — si riconosce.
    """
    for spec in layout.SPECS:
        if nome_cartella.startswith(spec + "_"):
            variant = nome_cartella[len(spec) + 1:]
            if variant in layout.VARIANTS:
                return spec, variant
    return None


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


def publish_benchmark(start: str, end: str) -> str:
    """
    Copia il CSV dei benchmark da `csv/_cells/benchmark/` a `csv/benchmark/`.

    Stesso gesto di `publish_cell`, altra destinazione: i benchmark non sono una
    cella e non devono comparire in `csv/dfm/`, dove chi legge si aspetta un
    file per cella e conta i file per sapere quante celle ci sono.
    """
    src = os.path.join(layout.benchmark_cell_dir(),
                       f"weekly_nowcast_{start}_{end}.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"il lavoro dei benchmark non ha un CSV su {start}..{end}:\n"
            f"    {src}")
    dst_dir = layout.benchmark_csv_dir()
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, benchmark_published_name(start, end))
    shutil.copyfile(src, dst)
    return dst


def _un_periodo_solo(paths: list[str]) -> None:
    """
    Solleva se fra i CSV pubblicati convivono periodi diversi.

    Guarda le cartelle intere — `csv/dfm/` E `csv/benchmark/` — non solo quel
    che si e' appena copiato: un file rimasto da prima e' pericoloso quanto uno
    appena scritto, e nessuno dei due si annuncia.  Le due cartelle si guardano
    insieme perche' insieme vengono lette: un benchmark sul 2016 accanto a celle
    sul 2007-2025 darebbe le stesse righe doppie di due celle discordi.
    """
    spans = {}
    for p in paths:
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$",
                      os.path.basename(p))
        if m:
            spans.setdefault(m.groups(), []).append(os.path.basename(p))
    if len(spans) > 1:
        righe = "\n".join(
            f"    {a} .. {b}   ({len(f)} file, es. {f[0]})"
            for (a, b), f in sorted(spans.items()))
        raise SystemExit(
            f"Fra {layout.dfm_csv_dir()} e {layout.benchmark_csv_dir()} "
            f"convivono {len(spans)} periodi:\n"
            f"{righe}\n"
            f"Figure e tabelle concatenano TUTTO quello che trovano: cosi' "
            f"gli stessi\n(trimestre, settimana) verrebbero contati piu' "
            f"volte.  Cancellare i file\ndel periodo che non serve e "
            f"rilanciare.")


def collect_cells(verbose: bool = True) -> list[str]:
    """
    Pubblica TUTTE le celle che hanno scritto qualcosa in `csv/_cells/`.

    Idempotente: si puo' rilanciare, ricopia e basta.  Non pretende che le
    quindici ci siano — a dirlo se manca qualcosa e' la guardia
    `tests/forecast/test_cells_produced.py`, che guarda anche DENTRO i file.
    Qui si guarda solo l'albero, piu' la guardia sul periodo unico.
    """
    root = layout.dfm_cells_root()
    if not os.path.isdir(root):
        raise SystemExit(
            f"Non c'e' niente da raccogliere: manca {root}.\n"
            f"Le celle si stimano con:  python scripts/run_dfm.py "
            f"--spec <spec> --variant <variante>")

    pubblicati: list[str] = []
    for cella in sorted(os.listdir(root)):
        # I benchmark stanno fra le celle ma non SONO una cella: stesso gesto,
        # altra destinazione (`csv/benchmark/`), e nessuna coppia spec/variante
        # da riconoscere — il nome della cartella e' esatto.
        if cella == layout.BENCHMARK_JOB:
            for src in sorted(glob.glob(os.path.join(root, cella,
                                                     "weekly_nowcast_*.csv"))):
                m = _SPAN.search(os.path.basename(src))
                if not m:
                    continue
                dst = publish_benchmark(m.group(1), m.group(2))
                pubblicati.append(dst)
                if verbose:
                    print(f"  {layout.BENCHMARK_JOB} -> {os.path.basename(dst)}")
            continue
        parti = cell_parts(cella)
        if parti is None:
            if verbose and os.path.isdir(os.path.join(root, cella)):
                print(f"  [salto] {cella}: non e' una cella nota")
            continue
        spec, variant = parti
        for src in sorted(glob.glob(os.path.join(root, cella,
                                                 "weekly_nowcast_*.csv"))):
            m = _SPAN.search(os.path.basename(src))
            if not m:
                continue
            dst = publish_cell(spec, variant, m.group(1), m.group(2))
            pubblicati.append(dst)
            if verbose:
                print(f"  {spec}/{variant} -> {os.path.basename(dst)}")

    if not pubblicati:
        raise SystemExit(
            f"Nessuna cella ha scritto un CSV sotto {root}.\n"
            f"Le celle si stimano con:  python scripts/run_dfm.py "
            f"--spec <spec> --variant <variante>")

    # La guardia guarda le CARTELLE, non la lista: conta anche cio' che c'era.
    _un_periodo_solo(
        sorted(glob.glob(os.path.join(layout.dfm_csv_dir(),
                                      "weekly_nowcast_*.csv")))
        + sorted(glob.glob(os.path.join(layout.benchmark_csv_dir(),
                                        "weekly_nowcast_*.csv"))))
    if verbose:
        print(f"  {len(pubblicati)} CSV in {layout.dfm_csv_dir()} "
              f"e {layout.benchmark_csv_dir()}")
    return pubblicati


def main() -> int:
    print("=" * 78)
    print("  RACCOLTA — csv/_cells/ -> csv/dfm/")
    print("=" * 78)
    collect_cells()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
