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
      dfm/<spec>/rmse/            RMSE figures (all passes + zooms) + tables
      dfm/<spec>/<variant>/       forecast figures, one per window
      bvar/<model>/               forecast figures, one per window
      bvar/rmse/                  RMSE figures + tables
      bvar/logscore/              log predictive score
      comparison/                 BVAR-vs-DFM tables

`spec` is one of the three loading structures, `variant` one of the five model
flags, `model` one of the four BVARs.  The accessors below are the only
supported way to build these paths.
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

#: Il nome del lavoro che li calcola: una cartella sotto `csv/_cells/` e una
#: sotto `csv/`, esattamente come una cella, perche' e' un'unita' di lavoro
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


def bvar_forecast_dir(model: str) -> str:
    """`bvar/<model>/` — the four forecast figures of one BVAR."""
    return os.path.join(OUTPUT_ROOT, "bvar", model)


def bvar_rmse_dir() -> str:
    return os.path.join(OUTPUT_ROOT, "bvar", "rmse")


def bvar_logscore_dir() -> str:
    return os.path.join(OUTPUT_ROOT, "bvar", "logscore")


def comparison_dir() -> str:
    """`comparison/` — the BVAR-vs-DFM tables."""
    return os.path.join(OUTPUT_ROOT, "comparison")


# ─── I CSV grezzi: l'input di tutto il resto ──────────────────────────────────

def dfm_csv_dir() -> str:
    """
    `csv/dfm/` — i nowcast settimanali del DFM, formato lungo.

    Simmetrico a `csv/bvar/`: con i file del DFM sfusi in `csv/` e quelli del
    BVAR in una sottocartella, l'albero diceva che una famiglia e' il default
    e l'altra un'aggiunta.  Sono due stimatori dello stesso esperimento.

    ATTENZIONE: qui stanno solo i CSV GREZZI, l'input.  Le metriche NON vanno
    qui: vivono accanto alle figure che commentano, in `dfm/<spec>/rmse/`.
    """
    return os.path.join(OUTPUT_ROOT, "csv", "dfm")


def bvar_csv_dir() -> str:
    """
    `csv/bvar/` — i nowcast dei BVAR, i quantili `.npz` e i log score per blocco.

    Sotto lo STESSO `csv/` del DFM e non in un `output/bvar/` parallelo: sono
    l'input della stessa catena di tabelle e figure, e tenerli in due alberi
    era il motivo per cui le uscite finivano sparse.
    """
    return os.path.join(OUTPUT_ROOT, "csv", "bvar")


def benchmark_csv_dir() -> str:
    """
    `csv/benchmark/` — i nowcast dell'AR(2) e della media espandente.

    PERCHE' NON DENTRO UNA CELLA.  I benchmark non dipendono ne' dalla spec ne'
    dalla variante: si calcolano una volta per (settimana, target), non quindici.
    Prima quel "una volta" era realizzato dando `benchmarks=True` alla PRIMA
    cella dell'ordine canonico, e le loro righe finivano nel CSV di
    `diag3/gaussian` sotto la pseudo-spec `benchmark`: 6831 righe in un file che
    ne dichiara 2277, e tre serie diverse mescolate in un file che porta il nome
    di una sola.  Chi lo apriva a mano leggeva tre passate sovrapposte.

    Qui hanno la loro cartella e il loro stato di ripresa, come una cella —
    perche' e' quello che sono: un lavoro indipendente, che ora puo' anche
    girare in parallelo agli altri invece che saldato al primo.
    """
    return os.path.join(OUTPUT_ROOT, "csv", "benchmark")


def dfm_cells_root() -> str:
    """`csv/_cells/` — la radice sotto cui sta una cartella per cella."""
    return os.path.join(OUTPUT_ROOT, "csv", "_cells")


def benchmark_cell_dir() -> str:
    """
    `csv/_cells/benchmark/` — lo stato di ripresa del lavoro dei benchmark.

    Sta fra le celle e non accanto perche' il lavoro ha la stessa forma: scrive
    `weekly_nowcast_<inizio>_<fine>.csv`, riprende da quel file, e viene poi
    pubblicato in `benchmark_csv_dir()`.  `collect.cell_parts` lo riconosce per
    nome esatto, non provando a spezzarlo in spec e variante.
    """
    return os.path.join(dfm_cells_root(), BENCHMARK_JOB)


def dfm_cell_dir(spec: str, variant: str) -> str:
    """
    `csv/_cells/<spec>_<variant>/` — il RISULTATO INTERMEDIO di una cella DFM,
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
    """Every directory of the tree, in creation order."""
    out: list[str] = []
    for spec in SPECS:
        out.append(dfm_rmse_dir(spec))
        out.extend(dfm_forecast_dir(spec, v) for v in VARIANTS)
    out.extend(bvar_forecast_dir(m) for m in BVAR_MODELS)
    out.append(bvar_rmse_dir())
    out.append(bvar_logscore_dir())
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
    "dfm_forecast_dir", "dfm_rmse_dir", "bvar_forecast_dir",
    "bvar_rmse_dir", "bvar_logscore_dir", "comparison_dir",
    "dfm_csv_dir", "bvar_csv_dir", "benchmark_csv_dir",
    "dfm_cells_root", "dfm_cell_dir", "benchmark_cell_dir",
    "logs_dir",
    "checkpoint_dir", "all_dirs",
    "build_tree", "window", "slice_window",
]
