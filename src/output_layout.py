"""
src/output_layout.py

THE SINGLE PLACE where two things are defined: the evaluation WINDOWS and the
OUTPUT TREE.  Nothing else in the project should hardcode either.

WHY ONE MODULE FOR BOTH
-----------------------
A window and a directory are the same decision seen twice: "the 2024-2025
forecast figures" is a date range *and* a place on disk.  Keeping them apart is
how the old tree drifted — BVAR figures landed in `output/bvar/`, the RMSE
figures in a NY-Fed folder, the DFM ones in `figures/<spec>/` — with no single
file to read to find out where anything goes.

WINDOWS ARE WHOLE YEARS, INCLUSIVE
----------------------------------
"2024-2025" means 2024-01-01 .. 2025-12-31.  Every window below is written as an
explicit pair of dates so there is nothing to infer.  The weekly grid is
Friday-anchored (`release_calendar.weekly_grid`), so the first `as_of` of a
window is its first Friday and the last is its last Friday — the bounds are
calendar dates, not week counts.

ONE PASS FEEDS EVERYTHING
-------------------------
Every window here is a subset of `2007-01-01 .. 2025-12-31`.  A single weekly
pass over that span produces all four forecast figures and all three RMSE
passes by slicing; running the seven windows separately would re-estimate the
thirteen years they share.  `FULL_SPAN` is that envelope.

THE TREE
--------
    output/forecast_weekly/
      dfm/_cells/<spec>_<variant>/  resume state of one cell (working, not output)
      dfm/_cells/benchmark/         resume state of the benchmark job
      dfm/csv/                      the fifteen published cell CSVs
      dfm/csv/benchmark/            the benchmark CSV
      dfm/<spec>/<variant>/         forecast figures, one per window
      dfm/<spec>/rmse/              RMSE figures (all passes + zooms) + tables
      dfm/<spec>/mda/               directional accuracy, one figure
      bvar/csv/                     the per-block nowcasts and their .npz quantiles
      bvar/csv/logscore/            the per-block log scores
      bvar/<model>/                 forecast figures, one per window
      bvar/rmse/                    RMSE figures + tables
      bvar/logscore/                log predictive score
      bvar/mda/                     directional accuracy, one figure
      comparison/                   BVAR-vs-DFM tables

`spec` is one of the three loading structures, `variant` one of the five model
flags, `model` one of the four BVARs.  The accessors below are the only
supported way to build these paths.

THREE FOLDERS, AND THE RAW CSVs LIVE INSIDE THEM
------------------------------------------------
`forecast_weekly/` holds exactly `dfm/`, `bvar/` and `comparison/`.  The raw
CSVs used to sit in a fourth, family-neutral `csv/` at the top — `csv/dfm/`,
`csv/bvar/`, `csv/_cells/`, `csv/benchmark/` — which split every family in two
places: to see everything the DFM produced you opened `csv/dfm/` AND `dfm/`.
Now each family owns its whole chain, from the CSV it writes to the figure that
reads it, and `comparison/` is the only thing outside because it is the only
thing that belongs to neither.

Inside a family the raw CSVs stay under a `csv/` of their own: they are the
INPUT of everything else, and mixing them with the figures would lose the one
distinction that matters when a number looks wrong — is the CSV bad, or is the
figure reading it badly?

FILE NAMES ARE ENGLISH
----------------------
Directories, file names and the stems inside them (`metrics_`, `report_`,
`rmse_by_horizon_`, `nyfed_`, `logscore_`) are English throughout.  Prose,
printed reports and column names are Italian — this project writes in Italian
— but a file name is an address, and an address that switches language between
two sibling folders (`rmse_per_orizzonte_bvar_2007-2025.csv` next to
`rmse_by_horizon_fed_overlap.csv`) cannot be predicted, only looked up.
"""

from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Root of every output this module governs.
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "output", "forecast_weekly")


# ─── The windows ──────────────────────────────────────────────────────────────

#: Cascaldi-Garcia forecast figures: one figure per cell per window.
FORECAST_WINDOWS: dict[str, tuple[str, str]] = {
    "2007-2010": ("2007-01-01", "2010-12-31"),
    "2014-2016": ("2014-01-01", "2016-12-31"),
    "2019-2021": ("2019-01-01", "2021-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
}

#: Full RMSE passes: cumulative samples, all starting in 2007.
RMSE_PASSES: dict[str, tuple[str, str]] = {
    "2007-2019": ("2007-01-01", "2019-12-31"),
    "2007-2021": ("2007-01-01", "2021-12-31"),
    "2007-2025": ("2007-01-01", "2025-12-31"),
}

#: The NY Fed publishes no comparable nowcast over the last span, so the
#: head-to-head is limited to the first two passes.
NYFED_COMPARISON_PASSES: tuple[str, ...] = ("2007-2019", "2007-2021")

#: Zoomed RMSE figures: the same metric read on a single regime.
RMSE_ZOOM_WINDOWS: dict[str, tuple[str, str]] = {
    "2007-2010": FORECAST_WINDOWS["2007-2010"],
    "2019-2021": FORECAST_WINDOWS["2019-2021"],
    "2024-2025": FORECAST_WINDOWS["2024-2025"],
}

#: The envelope of every window above — the one estimation pass to run.
FULL_SPAN: tuple[str, str] = ("2007-01-01", "2025-12-31")


# ─── The cells ────────────────────────────────────────────────────────────────

#: Loading structures, in the order they are discussed.
SPECS: tuple[str, ...] = ("diag3", "diag4", "fed_overlap")

#: Model flags, from the lightest to the most parameterised.
VARIANTS: tuple[str, ...] = ("gaussian", "gaussian_ar1", "student_t",
                             "student_t_ar1", "student_t_ar1_shared")

#: The four BVARs, from the blindest to the most informed.
BVAR_MODELS: tuple[str, ...] = ("qbvar", "cbvar", "bbvar", "lbvar")

#: Univariate references. They are a yardstick in the tables, not a trajectory
#: to look at, so they get no forecast figure of their own.
BENCHMARKS: tuple[str, ...] = ("ar2", "mean")

#: Il nome del lavoro che li calcola: una cartella sotto `dfm/_cells/` e una
#: sotto `dfm/csv/`, esattamente come una cella, perche' e' un'unita' di lavoro
#: indipendente quanto una cella. Coincide con la pseudo-spec `BENCHMARK_SPEC`
#: che i benchmark portano nella colonna `spec`, cosi' il nome della cartella e
#: quello delle righe che ci stanno dentro sono la stessa parola.
BENCHMARK_JOB: str = "benchmark"


# ─── The tree ─────────────────────────────────────────────────────────────────

def dfm_forecast_dir(spec: str, variant: str) -> str:
    """`dfm/<spec>/<variant>/` — the four forecast figures of one cell."""
    return os.path.join(OUTPUT_ROOT, "dfm", spec, variant)


def dfm_rmse_dir(spec: str) -> str:
    """`dfm/<spec>/rmse/` — RMSE figures and tables of one spec."""
    return os.path.join(OUTPUT_ROOT, "dfm", spec, "rmse")


def dfm_mda_dir(spec: str) -> str:
    """`dfm/<spec>/mda/` — directional accuracy, one figure per spec."""
    return os.path.join(OUTPUT_ROOT, "dfm", spec, "mda")


def bvar_forecast_dir(model: str) -> str:
    """`bvar/<model>/` — the four forecast figures of one BVAR."""
    return os.path.join(OUTPUT_ROOT, "bvar", model)


def bvar_rmse_dir() -> str:
    return os.path.join(OUTPUT_ROOT, "bvar", "rmse")


def bvar_mda_dir() -> str:
    """`bvar/mda/` — directional accuracy, the four models together."""
    return os.path.join(OUTPUT_ROOT, "bvar", "mda")


def bvar_logscore_dir() -> str:
    return os.path.join(OUTPUT_ROOT, "bvar", "logscore")


def comparison_dir() -> str:
    """`comparison/` — the BVAR-vs-DFM tables."""
    return os.path.join(OUTPUT_ROOT, "comparison")


# ─── I CSV grezzi: l'input di tutto il resto ──────────────────────────────────

def dfm_csv_dir() -> str:
    """
    `dfm/csv/` — i nowcast settimanali del DFM, formato lungo, uno per cella.

    DENTRO `dfm/`, non in un `csv/` in cima all'albero.  Stavano fuori, in
    `csv/dfm/`, e la conseguenza era che il DFM viveva in due posti: chi voleva
    vedere tutto quello che il DFM ha prodotto doveva aprire `csv/dfm/` E
    `dfm/`, e chi rinominava una cella doveva ricordarsi di entrambi.  Una
    famiglia, una cartella, dal CSV che scrive alla figura che lo legge.

    ATTENZIONE: qui stanno solo i CSV GREZZI, l'input.  Le metriche NON vanno
    qui: vivono accanto alle figure che commentano, in `dfm/<spec>/rmse/`.

    Chi conta i file `weekly_nowcast_*.csv` di questa cartella conta le CELLE:
    per questo i benchmark, che cella non sono, stanno un livello sotto in
    `benchmark_csv_dir()` e non qui in mezzo.
    """
    return os.path.join(OUTPUT_ROOT, "dfm", "csv")


def bvar_csv_dir() -> str:
    """
    `bvar/csv/` — i nowcast dei BVAR per blocco e i quantili `.npz`.

    Simmetrico a `dfm_csv_dir()`: ogni famiglia tiene i propri CSV grezzi sotto
    la propria cartella.  I log score per blocco, che sono l'altra uscita
    grezza della passata BVAR, stanno in `bvar_logscore_csv_dir()`.
    """
    return os.path.join(OUTPUT_ROOT, "bvar", "csv")


def bvar_logscore_csv_dir() -> str:
    """
    `bvar/csv/logscore/` — i log score GREZZI, un CSV per blocco.

    Da non confondere con `bvar_logscore_dir()` (`bvar/logscore/`), che e' la
    sua LETTURA: tabelle e figure aggregate.  La regola dell'albero li tiene
    distinti senza doverci pensare — sotto `csv/` c'e' l'input, fuori c'e' cio'
    che qualcuno ha calcolato leggendolo.

    Esiste come accessor perche' prima era un `os.path.join(csv_dir,
    "logscore")` ricostruito a mano in due moduli: due copie della stessa
    decisione, che e' esattamente cio' che questo file esiste per evitare.
    """
    return os.path.join(bvar_csv_dir(), "logscore")


def benchmark_csv_dir() -> str:
    """
    `dfm/csv/benchmark/` — i nowcast dell'AR(2) e della media espandente.

    PERCHE' NON DENTRO UNA CELLA.  I benchmark non dipendono ne' dalla spec ne'
    dalla variante: si calcolano una volta per (settimana, target), non quindici.
    Prima quel "una volta" era realizzato dando `benchmarks=True` alla PRIMA
    cella dell'ordine canonico, e le loro righe finivano nel CSV di
    `diag3/gaussian` sotto la pseudo-spec `benchmark`: 6831 righe in un file che
    ne dichiara 2277, e tre serie diverse mescolate in un file che porta il nome
    di una sola.  Chi lo apriva a mano leggeva tre passate sovrapposte.

    PERCHE' UN LIVELLO SOTTO E NON ACCANTO.  Sono del DFM (li produce la stessa
    passata, sono il suo metro di paragone), quindi stanno sotto `dfm/`; ma non
    sono una cella, e `dfm/csv/*.csv` deve continuare a contenere quindici file
    e basta — quel conteggio e' la verifica piu' rapida che la passata sia
    completa.  Sottocartella: dentro la famiglia, fuori dal conteggio.
    """
    return os.path.join(dfm_csv_dir(), BENCHMARK_JOB)


def dfm_cells_root() -> str:
    """
    `dfm/_cells/` — la radice sotto cui sta una cartella per cella.

    L'underscore dice che e' LAVORO, non consegna: e' lo stato da cui una cella
    riprende, non un risultato da leggere.  Stessa convenzione di `_logs/` e
    `_checkpoint/`, che pero' stanno del tutto fuori dall'albero perche'
    pesano ordini di grandezza di piu'.
    """
    return os.path.join(OUTPUT_ROOT, "dfm", "_cells")


def benchmark_cell_dir() -> str:
    """
    `dfm/_cells/benchmark/` — lo stato di ripresa del lavoro dei benchmark.

    Sta fra le celle e non accanto perche' il lavoro ha la stessa forma: scrive
    `weekly_nowcast_<inizio>_<fine>.csv`, riprende da quel file, e viene poi
    pubblicato in `benchmark_csv_dir()`.  `collect.cell_parts` lo riconosce per
    nome esatto, non provando a spezzarlo in spec e variante.

    NON E' UN DOPPIONE di `benchmark_csv_dir()`, benche' i due file si
    somiglino: questo e' lo stato di ripresa (si riscrive a ogni settimana
    calcolata), quello e' la copia pubblicata (si riscrive a fine passata, col
    nome del lavoro dentro).  Vale per ogni cella allo stesso modo.
    """
    return os.path.join(dfm_cells_root(), BENCHMARK_JOB)


def dfm_cell_dir(spec: str, variant: str) -> str:
    """
    `dfm/_cells/<spec>_<variant>/` — il RISULTATO INTERMEDIO di una cella DFM,
    e insieme il suo stato di ripresa.

    Una cartella per cella, e non e' cosmesi: il nome del file che
    `weekly_nowcast` scrive e' `weekly_nowcast_<inizio>_<fine>.csv`, che per
    quindici celle sullo stesso periodo sarebbe lo STESSO file — quindici
    processi che si sovrascrivono a vicenda.  Separate, i quindici CSV
    convivono, e `run_dfm.py` ne pubblica una copia in `dfm_csv_dir()` con il
    nome della cella dentro.

    Si COPIA, non si sposta: questo file e' cio' da cui la cella riprende.
    """
    return os.path.join(dfm_cells_root(), f"{spec}_{variant}")


def dfm_benchmark_figure_dir(name: str) -> str:
    """
    `dfm/benchmark/<ar2|mean>/` — dove finirebbe la traiettoria di un benchmark.

    Esiste per non lasciare quel percorso cablato dentro `figures.py`, ma
    `BENCHMARKS` dice che i benchmark sono un metro nelle tabelle e non una
    traiettoria da guardare: la scoperta automatica dei CSV non li passa alle
    figure, e questa cartella si riempie solo se qualcuno chiama `figures.py`
    con `--csv` puntato a mano sul file dei benchmark.

    Percio' NON sta in `all_dirs()` e di norma non esiste.  Le otto figure che
    ci si trovavano erano un residuo dell'epoca in cui le righe dei benchmark
    stavano dentro il CSV di `diag3/gaussian`: nessuna passata le rigenerava
    piu', quindi mostravano numeri di una parametrizzazione superata senza
    niente che lo dicesse.  Cancellate.
    """
    return os.path.join(OUTPUT_ROOT, "dfm", "benchmark", name)


def logs_dir() -> str:
    """
    `output/_logs/` — l'uscita a video di ogni passo, un file per passo.

    Fuori dall'albero di consegna come `checkpoint_dir()`: e' diario di bordo,
    non risultato.  `run_outputs.py` ci scrive un log per passo, cosi' un passo
    fallito si legge senza rilanciare tutta la catena.
    """
    return os.path.join(_PROJECT_ROOT, "output", "_logs")


def checkpoint_dir() -> str:
    """
    `output/_checkpoint/` — lo stato di lavoro della passata, FUORI dall'albero
    di consegna.

    Non e' un risultato: e' la cache che permette di riprendere una run
    interrotta, pesa ordini di grandezza piu' dei CSV (215 MB per il solo
    pilota di quattro settimane) e non va ne' consegnata ne' sincronizzata.
    Dentro `forecast_weekly/` significherebbe spedire gigabyte di cache
    insieme ai risultati.
    """
    return os.path.join(_PROJECT_ROOT, "output", "_checkpoint")


def all_dirs() -> list[str]:
    """
    Every directory of the tree, in creation order.

    TUTTE, comprese quelle dei CSV e delle MDA.  Prima ne mancavano cinque —
    `csv/`, `mda/` — e non davano errore: il modulo che ci scriveva faceva
    `makedirs` per conto suo.  Il risultato era che questa lista, che dovrebbe
    essere l'inventario dell'albero, ne descriveva i tre quarti, e
    `build_tree()` costruiva una cosa diversa da quella che la passata poi
    riempiva.
    """
    out: list[str] = [dfm_csv_dir(), benchmark_csv_dir(), dfm_cells_root()]
    for spec in SPECS:
        out.append(dfm_rmse_dir(spec))
        out.append(dfm_mda_dir(spec))
        out.extend(dfm_forecast_dir(spec, v) for v in VARIANTS)
    out.append(bvar_csv_dir())
    out.append(bvar_logscore_csv_dir())
    out.extend(bvar_forecast_dir(m) for m in BVAR_MODELS)
    out.append(bvar_rmse_dir())
    out.append(bvar_logscore_dir())
    out.append(bvar_mda_dir())
    out.append(comparison_dir())
    return out


def build_tree() -> list[str]:
    """Create the whole tree. Idempotent, and the only way it gets created."""
    for d in all_dirs():
        os.makedirs(d, exist_ok=True)
    return all_dirs()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def window(name: str) -> tuple[str, str]:
    """The (start, end) of any named window, whichever family it belongs to."""
    for table in (FORECAST_WINDOWS, RMSE_PASSES, RMSE_ZOOM_WINDOWS):
        if name in table:
            return table[name]
    raise KeyError(
        f"Unknown window {name!r}. Known: "
        f"{sorted(set(FORECAST_WINDOWS) | set(RMSE_PASSES) | set(RMSE_ZOOM_WINDOWS))}")


def slice_window(df, name: str, column: str = "as_of_dt"):
    """
    The rows of `df` falling inside the named window, bounds included.

    This is how one estimation pass becomes seven outputs: slice, never re-run.
    """
    import pandas as pd
    start, end = window(name)
    src = column if column in df.columns else "as_of"
    # Sempre convertita: `as_of` arriva come stringa dal CSV e come datetime
    # dai frame gia' preparati, e confrontare una stringa con un Timestamp
    # non fallisce in modo ovvio — solleva a meta' pipeline.
    col = pd.to_datetime(df[src])
    return df[(col >= pd.Timestamp(start)) & (col <= pd.Timestamp(end))]


__all__ = [
    "OUTPUT_ROOT", "FORECAST_WINDOWS", "RMSE_PASSES", "RMSE_ZOOM_WINDOWS",
    "NYFED_COMPARISON_PASSES", "FULL_SPAN", "SPECS", "VARIANTS", "BVAR_MODELS",
    "BENCHMARKS", "BENCHMARK_JOB",
    "dfm_forecast_dir", "dfm_rmse_dir", "dfm_mda_dir", "bvar_forecast_dir",
    "bvar_rmse_dir", "bvar_mda_dir", "bvar_logscore_dir", "comparison_dir",
    "dfm_csv_dir", "bvar_csv_dir", "bvar_logscore_csv_dir", "benchmark_csv_dir",
    "dfm_cells_root", "dfm_cell_dir", "benchmark_cell_dir",
    "dfm_benchmark_figure_dir",
    "logs_dir",
    "checkpoint_dir", "all_dirs",
    "build_tree", "window", "slice_window",
]
