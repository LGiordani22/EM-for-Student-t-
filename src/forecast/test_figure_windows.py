"""
src/forecast/test_figure_windows.py

LA GUARDIA SUL BORDO DELLE FIGURE.  Verifica che in ogni finestra forecast si
disegnino trimestri INTERI, e solo quelli.

    python -m src.forecast.test_figure_windows

Esiste per un difetto che e' stato in tesi per mesi senza che nessun test lo
vedesse: la finestra ritagliava sui VINTAGE (`as_of` fra i due bordi) mentre la
figura disegna per TRIMESTRE TARGET.  L'ultima linea finiva percio' tagliata a
fine dicembre — amputata proprio del backcast — col pallino staccato di quattro
settimane invece di una; i trimestri gia' in volo al bordo entravano come
monconi; e il pallino di un trimestre quasi tutto fuori finestra restava in
figura da solo, mesi oltre l'ultima linea.  Nessuno dei tre e' un errore di
calcolo, e proprio per questo nessuna guardia numerica poteva accorgersene: era
l'inquadratura a essere sbagliata, e si vedeva solo guardando la figura.

Le tre condizioni, per ogni cella e ogni trimestre disegnato:

  1. la FINE del trimestre cade dentro la finestra dichiarata;
  2. la linea arriva al rilascio — l'ultima as_of dista al piu' una settimana
     da `gdp_release_date`, quindi il vuoto fino al pallino resta una settimana;
  3. la linea comincia da dove il trimestre e' entrato in volo, cioe' dall'inizio
     del trimestre precedente: niente monconi.

Senza CSV sul disco non e' un guasto: prima della passata non ci sono per
costruzione, e il test lo dice ed esce bene.
"""

from __future__ import annotations

import sys

import pandas as pd

from src import output_layout as layout
from src.forecast import figures as fig
from src.forecast.release_calendar import quarter_end


def _frames() -> list[tuple[str, pd.DataFrame]]:
    """I due alberi, quando ci sono: (nome, frame gia' preparato)."""
    out: list[tuple[str, pd.DataFrame]] = []
    try:
        out.append(("DFM", fig.load()))
    except FileNotFoundError as e:
        print(f"  [assente] DFM: {e.args[0].splitlines()[0]}")
    try:
        from src.bvar import figures as bfig
        out.append(("BVAR", bfig.load()))
    except SystemExit as e:
        print(f"  [assente] BVAR: {str(e).splitlines()[0]}")
    return out


def check() -> int:
    frames = _frames()
    if not frames:
        print("\nNessun CSV sul disco: niente da verificare.")
        return 0

    tolleranza = fig._TOLLERANZA
    failures = 0

    for nome, df in frames:
        print(f"--- {nome} ---")
        for finestra in layout.FORECAST_WINDOWS:
            start, end = (pd.Timestamp(x) for x in layout.window(finestra))
            d = fig.slice_target_quarters(df, finestra, verbose=False)
            if d.empty:
                print(f"  VUOTA  {finestra}")
                failures += 1
                continue

            rotti: list[str] = []
            for (cella, q), rows in d.groupby(["cella", "target_quarter"]):
                qe = quarter_end(str(q))
                release = pd.Timestamp(rows["release_dt"].iloc[0])
                if not (start <= qe <= end):
                    rotti.append(f"{cella}/{q}: finisce fuori finestra")
                if release - rows["as_of_dt"].max() > tolleranza:
                    rotti.append(f"{cella}/{q}: linea tagliata prima del rilascio")
                if rows["as_of_dt"].min() - fig._prima_as_of_attesa(str(q)) > tolleranza:
                    rotti.append(f"{cella}/{q}: moncone, non parte da quando e' in volo")

            trimestri = sorted(d["target_quarter"].unique(), key=fig._quarter_key)
            ok = not rotti
            failures += not ok
            print(f"  {'OK ' if ok else 'ROTTA'}  {finestra:12s} "
                  f"{len(trimestri):2d} trimestri  "
                  f"{trimestri[0]}..{trimestri[-1]}  "
                  f"asse {d['as_of_dt'].min().date()} .. "
                  f"{d['release_dt'].max().date()}")
            for r in rotti[:5]:
                print(f"          {r}")
        print()

    print("=" * 78)
    print("TUTTO OK" if not failures else f"{failures} finestra/e ROTTE")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(check())
