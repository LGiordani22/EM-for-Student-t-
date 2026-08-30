"""
src/bvar/figures.py

LE FIGURE DEL GATE 6 — un involucro, non un disegnatore.

    python -m src.bvar.figures

Questo modulo NON disegna: chiama `src/forecast/figures.py`, che e' gia' la
figura in stile Cascaldi-Garcia del lavoro DFM, e si limita a puntarla sul CSV
del BVAR e sull'albero di uscita giusto.  E' la ragione per cui al Gate 6 il
contratto CSV e' `weekly_nowcast.COLUMNS` e non una sua variante: cambiare
schema avrebbe significato riscrivere questo file invece di scriverlo in
cinquanta righe.

L'ALBERO DI USCITA LO DECIDE `output_layout`
--------------------------------------------
`forecast.figures.make_trajectories` accetta un `dir_for_cell(cella) -> cartella`, e
qui si passa `_bvar_dir_for_cell`, che manda 'cbvar/authors' in

    output/forecast_weekly/bvar/cbvar/

Le figure del BVAR stanno quindi dentro `forecast_weekly/` accanto a quelle
del DFM, non in un `output/bvar/` parallelo: sono lo stesso esperimento letto
con due stimatori, e tenerle in due alberi diversi era la ragione per cui le
figure finivano sparse.  I CSV grezzi restano dove la passata li scrive
(`evaluate.OUTPUT_ROOT`); qui si governa solo l'uscita leggibile.

LE DUE FIGURE, E PERCHE' SERVONO ENTRAMBE
------------------------------------------
Sono due tagli della stessa passata, e la differenza e' l'ASSE X.

  trajectories  x = CALENDARIO (`as_of`).  La traiettoria di ogni trimestre che
         si aggiorna settimana per settimana, i pallini sui rilasci.  E' la
         figura del lavoro DFM, identica: si legge l'EPISODIO — quando il
         modello ha girato, quanto e' rimasto indietro nel 2008Q4.

  rmse per orizzonte   x = `horizon_week`, con le TRE FASI come bande di
         sfondo (forecast h<1, nowcast 1-13, backcast h>13).  E' la Figura 1
         del paper: non il tempo, ma la distanza dal trimestre obiettivo, con
         i trimestri MEDIATI.  Si legge l'APPRENDIMENTO — l'errore scende man
         mano che l'informazione arriva?
         Vive in `src/bvar/metrics.py` perche' e' una figura di metrica (la
         disegna `compare_nyfed.figure_rmse_by_horizon`, sempre riusata).

La convenzione delle fasi e' una sola per tutto il progetto, e sta in
`release_calendar.horizon_week`: non e' ridefinita qui.

NESSUN ADATTAMENTO RESIDUO
--------------------------
Serviva, quando il DFM aveva `_YLIM=(-15,+6)` cablato e la legenda inchiodata
in basso a destra: le bande del BVAR sono piu' larghe, un blocco 2020 usciva
dal riquadro e la legenda copriva il pallino del realizzato.  Ora la scala e'
automatica per finestra e la legenda si posiziona verificando di non coprire i
dati per ENTRAMBI, quindi questo involucro non deroga piu' su niente: sceglie
solo il CSV, le celle e la cartella.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from src import output_layout as layout
from src.bvar.evaluate import OUTPUT_ROOT
from src.forecast import figures as dfm

#: I quattro modelli, nell'ordine del paper: dal piu' cieco al piu' informato.
MODELS = layout.BVAR_MODELS


def _bvar_dir_for_cell(cell: str, root: str | None = None) -> str:
    """
    'cbvar/authors' -> output/forecast_weekly/bvar/cbvar/

    A differenza del DFM, la variante NON diventa una cartella: i quattro BVAR
    hanno una parametrizzazione sola ciascuno ('-' o 'authors'), quindi un
    livello in piu' sarebbe una cartella per un file.

    `root` VA ONORATO.  Senza, questa funzione cablava `bvar_forecast_dir` e
    scavalcava l'`output_root` che `make_all` le passava — perche' in
    `make_trajectories` il router ha la precedenza sulla cartella.
    Risultato: una sonda lanciata con `--output-root` creava la cartella
    richiesta e scriveva le figure fra gli artefatti veri lo stesso.  E' il
    gemello del difetto chiuso in `evaluate._paths`.
    """
    model = cell.split("/", 1)[0]
    if root and os.path.abspath(root) != os.path.abspath(layout.OUTPUT_ROOT):
        return os.path.join(root, "bvar", model)
    return layout.bvar_forecast_dir(model)


def _comparison_dir(root: str | None = None) -> str:
    """La cartella del confronto, con la stessa regola su `root`."""
    if root and os.path.abspath(root) != os.path.abspath(layout.OUTPUT_ROOT):
        return os.path.join(root, "comparison")
    return layout.comparison_dir()


def discover_csvs(paths: list[str] | None = None) -> list[str]:
    """I CSV del Gate 6: quelli dati, o tutti quelli sul disco."""
    if paths:
        return list(paths)
    d = layout.bvar_csv_dir()
    found = sorted(glob.glob(os.path.join(d, "bvar_realtime_*.csv")))
    if not found:
        raise SystemExit(
            f"Nessun CSV in {d}.\n"
            f"Generane uno con:  python -m src.bvar.evaluate "
            f"--start YYYY-MM-DD --end YYYY-MM-DD"
        )
    return found


def load(paths: list[str] | None = None) -> pd.DataFrame:
    """
    I CSV del BVAR, uniti, nella forma che `forecast.figures` si aspetta.

    Qui i blocchi-trimestre sono tanti file (l'unita' di parallelismo della
    passata), quindi si concatena prima e si riusa la preparazione di colonne
    di `dfm` dopo — che oggi concatena a sua volta, una cella per file.
    """
    frames = [pd.read_csv(p) for p in discover_csvs(paths)]
    df = pd.concat(frames, ignore_index=True)
    df["target_quarter"] = df["target_quarter"].astype(str)
    df["as_of_dt"] = pd.to_datetime(df["as_of"])
    df["release_dt"] = pd.to_datetime(df["gdp_release_date"])
    df["cella"] = (df["spec"] + "/" + df["variant"]).where(
        df["spec"] != dfm._BENCHMARK_SPEC, df["variant"])
    return df.sort_values(["cella", "target_quarter", "as_of_dt"])


def make_all(df: pd.DataFrame, output_root: str | None = None,
             models: tuple[str, ...] = MODELS,
             ylim: tuple[float, float] | None = None,
             compare_quarters: list[str] | None = None,
             window_label: str | None = None) -> list[str]:
    """
    Una figura di traiettorie per modello in `output/bvar/<modello>/`,
    piu' — se richiesto — la
    figura di confronto su un trimestre.

    I benchmark (`ar2`, `mean`) restano fuori dalle figure per scelta: sono
    piatti fra un rilascio e l'altro e servono come METRO nelle tabelle, non
    come traiettoria da guardare.  Nella figura di confronto invece entrano,
    perche' li' il contrasto e' il punto.
    """
    root = output_root or layout.OUTPUT_ROOT
    cells = [c for c in sorted(df["cella"].unique())
             if c.split("/")[0] in models]
    if not cells:
        raise SystemExit(
            f"Nessuna cella dei modelli {models} nel CSV; presenti: "
            f"{sorted(df['cella'].unique())}")

    # Non c'e' piu' nessuna divergenza dal DFM: la legenda si posiziona da
    # sola verificando di non coprire i dati (`forecast.figures._place_legend`)
    # e la scala e' automatica per finestra su entrambi i lati.  Prima qui
    # serviva `legend_loc="best"` perche' il default DFM era cablato in basso
    # a destra e col 2008Q4 (realizzato -8.47) copriva il pallino.
    written = dfm.make_trajectories(
        df, root, ylim=ylim, cells=cells,
        dir_for_cell=lambda c: _bvar_dir_for_cell(c, root),
        window_label=window_label, family="bvar")
    for q in compare_quarters or []:
        written += dfm.make_compare(df[df["target_quarter"] == q],
                                    _comparison_dir(root), q, ylim=ylim,
                                    window_label=window_label)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Figure del Gate 6 — involucro su src/forecast/figures.py")
    p.add_argument("--csv", nargs="*", default=None,
                   help="CSV da leggere (default: tutti quelli in output/bvar/csv)")
    p.add_argument("--output-root", default=None)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--compare", nargs="*", default=None,
                   help="trimestri per la figura di confronto, es. 2008Q4")
    p.add_argument("--ylim", nargs=2, type=float, default=None,
                   metavar=("MIN", "MAX"),
                   help="scala fissa (default: automatica sulla finestra)")
    p.add_argument("--window", default=None,
                   help="nome di una finestra di output_layout, es. 2014-2016")
    a = p.parse_args()

    df = load(a.csv)
    if a.window:
        df = layout.slice_window(df, a.window)
        if df.empty:
            raise SystemExit(f"Nessuna riga nella finestra {a.window} "
                             f"{layout.window(a.window)}.")
    print(f"  {len(df)} righe, {df['target_quarter'].nunique()} trimestri, "
          f"celle: {sorted(df['cella'].unique())}")
    n_real = int(df["realizzato_bea"].notna().sum())
    print(f"  realizzato presente su {n_real}/{len(df)} righe"
          + ("" if n_real else "   <- SENZA, NIENTE PALLINI: CSV da rifare"))

    written = make_all(df, a.output_root, tuple(a.models.split(",")),
                       ylim=tuple(a.ylim) if a.ylim else None,
                       compare_quarters=a.compare, window_label=a.window)
    print(f"\n{len(written)} figura/e scritte.")


if __name__ == "__main__":
    main()
