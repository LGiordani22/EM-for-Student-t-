"""
src/forecast/test_parallel_cell.py

LE DUE STRADE DEVONO DARE LO STESSO CSV.

    python -m src.forecast.test_parallel_cell

Una cella DFM si puo' percorrere in tre modi, e il progetto sostiene che diano
numeri IDENTICI, non «vicini»:

  1. `run_weekly_nowcast`  — il ciclo sequenziale storico;
  2. `run_cell_pipeline(workers=1)`  — la pipeline in-process, il riferimento
     contro cui si misura il parallelismo;
  3. `run_cell_pipeline(workers=3)`  — le settimane a theta congelato in un
     pool di processi.

La proprieta' era DICHIARATA (nel docstring di `run_dfm.py`: «verificato bit per
bit su diag3/student_t_ar1») ma **nessun test la eseguiva**: era una verifica
fatta una volta a mano, non una rete.  Questo file la rende una rete.

PERCHE' SERVE DAVVERO, e non e' zelo
------------------------------------
`esegui_settimana` e' pura e vive in un posto solo, ma lo STATO DELLA CELLA —
il theta corrente, il mese dell'ultima ri-stima, il contatore dei rifiuti della
guardia PCA — e' tenuto da DUE orchestratori diversi, uno per strada.  Due
copie della stessa logica di stato sono due copie che possono divergere, e la
divergenza si vedrebbe solo nei numeri, in silenzio, a passata finita.  E' lo
stesso rischio contro cui mette in guardia il docstring di `esegui_settimana`.

Qui l'uguaglianza e' ESATTA, non a tolleranza: le tre strade partono dallo
stesso punto e percorrono la stessa catena di theta nello stesso ordine.  Se
qualcosa si muove anche di 1e-12 e' un difetto, non innesco.  (Il caso della
RIPRESA, che invece diverge per costruzione ed e' a tolleranza, sta in
`test_resume.py`: sono due proprieta' diverse e non vanno confuse.)

Gira su una finestra CORTA con la cella piu' rapida (`diag3/gaussian`): la
meccanica dello scheduling non dipende dalla lunghezza della finestra, e il
test deve poter girare in un paio di minuti.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from src.forecast.pipeline import run_cell_pipeline
from src.forecast.weekly_nowcast import _KEY, COLUMNS, run_weekly_nowcast

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

#: Finestra di prova: corta, ma con piu' di un mese, altrimenti la cadenza
#: mensile delle ri-stime non viene esercitata e il test non prova niente.
_START, _END = "2008-01-01", "2008-06-30"
_SPEC, _VARIANT = "diag3", "gaussian"

#: Dove scrive il test.  MAI fra gli artefatti veri.
_OUT = os.path.join(_PROJECT_ROOT, "output", "_test_parallel_cell")

#: Le colonne numeriche su cui l'uguaglianza dev'essere esatta.
_NUM = ["nowcast_bea", "nowcast_livello", "nowcast_z", "sd_z",
        "realizzato_bea", "realizzato_livello", "errore_bea"]


def _ordina(df: pd.DataFrame) -> pd.DataFrame:
    return (df.reindex(columns=COLUMNS)
              .sort_values(list(_KEY), kind="stable")
              .reset_index(drop=True))


def _confronta(nome_a: str, a: pd.DataFrame,
               nome_b: str, b: pd.DataFrame) -> int:
    """Zero se identici; stampa dove differiscono e ritorna il numero di guasti."""
    fails = 0
    a, b = _ordina(a), _ordina(b)

    ok = len(a) == len(b)
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  stesso numero di righe "
          f"({len(a)} vs {len(b)})")
    if not ok:
        return fails

    chiavi_a = list(map(tuple, a[list(_KEY)].to_numpy()))
    chiavi_b = list(map(tuple, b[list(_KEY)].to_numpy()))
    ok = chiavi_a == chiavi_b
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  stesse chiavi, nello stesso ordine")
    if not ok:
        return fails

    da = a[_NUM].to_numpy(float)
    db = b[_NUM].to_numpy(float)
    # NaN in entrambi conta come uguale: le righe in errore hanno nowcast NaN.
    diverse = ~((da == db) | (np.isnan(da) & np.isnan(db)))
    n_righe = int(diverse.any(axis=1).sum())
    peggio = float(np.nanmax(np.abs(da - db))) if diverse.any() else 0.0
    ok = n_righe == 0
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  numeri IDENTICI bit per bit "
          f"({n_righe} righe diverse, scarto max {peggio:.3e})")
    if not ok:
        i = int(np.argmax(diverse.any(axis=1)))
        print(f"           prima riga diversa: {chiavi_a[i]}")
        print(f"             {nome_a}: {da[i]}")
        print(f"             {nome_b}: {db[i]}")

    for col in ("n_iter", "converged", "reestimated"):
        uguali = (a[col].astype(str).to_numpy()
                  == b[col].astype(str).to_numpy()).all()
        fails += not uguali
        print(f"  {'OK ' if uguali else 'ROTTA'}  colonna {col!r} identica")
    return fails


def check() -> int:
    shutil.rmtree(_OUT, ignore_errors=True)
    fails = 0
    esiti: dict[str, pd.DataFrame] = {}

    prove = (
        ("sequenziale", lambda d: run_weekly_nowcast(
            _START, _END, specs=(_SPEC,), variants=(_VARIANT,),
            benchmarks=False, output_dir=d, save=True)),
        ("pipeline w=1", lambda d: run_cell_pipeline(
            _SPEC, _VARIANT, _START, _END, out_dir=d, workers=1,
            save=True, verbose=False)),
        ("pipeline w=3", lambda d: run_cell_pipeline(
            _SPEC, _VARIANT, _START, _END, out_dir=d, workers=3,
            save=True, verbose=False)),
    )

    print(f"{'=' * 78}\n  LE DUE STRADE, STESSO CSV\n{'=' * 78}")
    print(f"  cella    : {_SPEC}/{_VARIANT}")
    print(f"  finestra : {_START} .. {_END}\n")

    for nome, fn in prove:
        cartella = os.path.join(_OUT, nome.replace(" ", "_").replace("=", ""))
        t0 = time.perf_counter()
        df = fn(cartella)
        # Il ciclo sequenziale produce anche altre celle se glielo si chiede:
        # qui gliene abbiamo chiesta una, ma il filtro rende il test robusto
        # a un cambio di default.
        df = df[(df["spec"] == _SPEC) & (df["variant"] == _VARIANT)]
        esiti[nome] = df
        print(f"  {nome:14s} {len(df):5d} righe in "
              f"{time.perf_counter() - t0:6.1f}s")

    for nome_b in ("pipeline w=1", "pipeline w=3"):
        print(f"\n  sequenziale  vs  {nome_b}")
        fails += _confronta("sequenziale", esiti["sequenziale"],
                            nome_b, esiti[nome_b])

    # Il test vale solo se la finestra ha DAVVERO esercitato piu' di una
    # ri-stima: con una sola, la catena dei theta non viene percorsa e le tre
    # strade coinciderebbero per banalita'.
    n_em = int(esiti["sequenziale"]["reestimated"]
               .astype(str).str.lower().isin(["true", "1"]).sum())
    ok = n_em >= 2
    fails += not ok
    print(f"\n  {'OK ' if ok else 'ROTTA'}  la finestra esercita la catena: "
          f"{n_em} ri-stime"
          + ("" if ok else "   <- troppo poche: il test non prova niente"))

    shutil.rmtree(_OUT, ignore_errors=True)
    return fails


if __name__ == "__main__":
    n = check()
    print("\nLE DUE STRADE COINCIDONO" if not n else f"\n{n} CONTROLLI ROTTI")
    sys.exit(1 if n else 0)
