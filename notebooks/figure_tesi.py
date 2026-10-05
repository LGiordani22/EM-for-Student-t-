"""
notebooks/figure_tesi.py

DALLA FIGURA DELLA PASSATA ALLA FIGURA DELLA TESI.

    python notebooks/figure_tesi.py --check    dice cosa toglierebbe, non tocca
    python notebooks/figure_tesi.py            ritaglia le figure in tesi

Si lancia DOPO `notebooks/tabelle_risultati.py`, che e' quello che copia (e
dove puo' rigenera) le figure della passata in `docs/tesi/figures/`.  Questo
script non ridisegna niente e non tocca `src/`: lavora sul PNG gia' scritto.

PERCHE' ESISTE
--------------
Una figura buona per una passata e una figura buona per una tesi non sono la
stessa figura.  Nella passata il titolo dentro l'immagine e' quello che dice
di quale cella si sta guardando il risultato, e la nota in fondo e' il
promemoria di come si legge.  In tesi quelle due cose stanno nella DIDASCALIA:
ripeterle dentro l'immagine le stampa due volte, e la seconda volta a un corpo
che alla scala della pagina (~0.53) diventa illeggibile.

Tre interventi, tutti per sottrazione:

  TITOLO      Va via sempre.  Quello che diceva deve finire nella didascalia,
              ed e' un VINCOLO: la didascalia deve dire metodo, finestra e
              numero di trimestri, altrimenti l'informazione si perde.  Le
              stringhe giuste sono in `IDENT` qui sotto, lette dai titoli
              originali; `--check` le stampa gia' pronte da incollare.

  NOTA        Va via dalle figure che ce l'hanno ancora.  Le due RMSE del DFM
              escono gia' senza (`tabelle_risultati` le rigenera con
              `note=False`); le due dei BVAR sono copiate dalla passata e la
              nota ce l'hanno.  Qui vanno via lo stesso, senza dover passare
              per il codice che le disegna.

  LEGENDA     Nelle coppie di pannelli (a)/(b) ne resta UNA SOLA.  Le due
              curve RMSE hanno gli stessi metodi in entrambi i pannelli: due
              legende identiche una sopra l'altra sono rumore.  Si TIENE
              QUELLA DEL PANNELLO (a) e non quella del (b), e non e' una
              scelta di gusto: nel pannello superiore c'e' anche la NY Fed,
              che nella finestra piena non c'e', quindi la legenda di (a) e'
              un soprainsieme di quella di (b) e le copre entrambe.
              Le TRAIETTORIE fanno eccezione e tengono la loro: li' la legenda
              dice quale colore e' Q1 e quale Q3, cambia da pannello a
              pannello, ed e' dentro il riquadro.

COME TROVA I PEZZI
------------------
Non per coordinate fisse, che si romperebbero al primo cambio di layout, ma
per BANDE: si guardano le righe di pixel non bianche e si raggruppano in
blocchi separati da spazio bianco.  Il riquadro del grafico e' la banda piu'
alta; quello che sta SOPRA e' il titolo; sotto vengono in fila le etichette
degli assi, poi la legenda, poi la nota.  La legenda si riconosce perche' e'
l'unica banda bassa alta piu' di `H_LEGENDA` pixel (le etichette stanno sotto
i 30, la legenda sta sugli 80).  La nota e' tutto cio' che viene dopo di lei.

L'operazione e' IDEMPOTENTE: su una figura gia' trattata non trova banda sopra
il grafico ne' banda alta sotto, e non fa niente.  Si puo' rilanciare senza
pensarci dopo ogni passata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "tesi" / "figures"
CSV_DIR = ROOT / "output" / "forecast_weekly"

#: Una riga di pixel e' "contenuto" se qualche pixel si scosta dal bianco piu'
#: di questo.  12 su 255 tiene fuori l'antialiasing e dentro le campiture
#: chiare delle fasi (FORECAST/NOWCAST/BACKCAST), che sono contenuto vero.
TOL = 12

#: Sotto il grafico, una banda piu' alta di cosi' e' la legenda.  Le etichette
#: degli assi stanno fra 10 e 27 pixel, la legenda fra 79 e 80: la soglia sta
#: comoda in mezzo e non si appoggia a nessun numero esatto.
H_LEGENDA = 60

#: Bianco lasciato sopra il riquadro dopo aver tolto il titolo, e sotto
#: l'ultima banda tenuta.  Senza, il bordo del riquadro tocca il margine.
PAD = 18


#: Cosa fare, figura per figura, e cosa deve dire la didascalia.
#:   legenda=True   la figura tiene la sua legenda
#:   legenda=False  la legenda va via: e' il pannello (b) di una coppia
#:   ident          l'identificazione OBBLIGATORIA in didascalia, presa dal
#:                  titolo originale prima di toglierlo
#:   csv            file della passata su cui ricontrollare il numero di
#:                  trimestri, cosi' la cifra in didascalia non puo' invecchiare
FIGURES: dict[str, dict] = {
    # Le due del DFM non hanno un CSV per finestra su cui ricontrollare: la
    # passata ne scrive uno solo, sul campione pieno, e le due figure le
    # rigenera `tabelle_risultati` tagliando la finestra al volo.  Il numero
    # in didascalia e' quello che quella rigenerazione stampa (46 e 66) ed e'
    # lo stesso di Table~\ref{tab:rmse-fed-overlap}.
    "fig_rmse_2007_2019.png": dict(
        legenda=True,
        ident="RMSE by horizon, \\texttt{fed\\_overlap}, "
              "2007Q2--2019Q3 (46 target quarters)",
        csv=None, quarters=46),
    "fig_rmse_2007_2025.png": dict(
        legenda=False,
        ident="RMSE by horizon, \\texttt{fed\\_overlap}, "
              "2007Q2--2025Q3 (66 target quarters)",
        csv=None, quarters=66),
    "fig_bvar_rmse_2007_2019.png": dict(
        legenda=True,
        ident="RMSE by horizon, the four Bayesian VARs, "
              "2007Q2--2019Q3 (46 target quarters)",
        csv="bvar/rmse/rmse_by_horizon_bvar_2007-2019.csv",
        quarters=46),
    "fig_bvar_rmse_2007_2025.png": dict(
        legenda=False,
        ident="RMSE by horizon, the four Bayesian VARs, "
              "2007Q2--2025Q3 (66 target quarters)",
        csv="bvar/rmse/rmse_by_horizon_bvar_2007-2025.csv",
        quarters=66),
    "fig_traj_2007_2010.png": dict(
        legenda=True,
        ident="\\texttt{fed\\_overlap} Student-$t$, the financial crisis: "
              "target quarters 2006Q4--2011Q1",
        csv=None, quarters=None),
    "fig_traj_2019_2021.png": dict(
        legenda=True,
        ident="\\texttt{fed\\_overlap} Student-$t$, the pandemic: "
              "target quarters 2018Q4--2022Q1",
        csv=None, quarters=None),
    "fig_bvar_traj_2007_2010.png": dict(
        legenda=True,
        ident="L-BVAR, the financial crisis: target quarters 2006Q4--2011Q1",
        csv=None, quarters=None),
    "fig_bvar_traj_2019_2021.png": dict(
        legenda=True,
        ident="L-BVAR, the pandemic: target quarters 2018Q4--2022Q1",
        csv=None, quarters=None),
}


def bande(arr: np.ndarray) -> list[tuple[int, int]]:
    """I blocchi di righe non bianche, dall'alto in basso."""
    contenuto = (np.abs(arr - 255).max(axis=2) > TOL).any(axis=1)
    fuori: list[tuple[int, int]] = []
    inizio = None
    for i, v in enumerate(contenuto):
        if v and inizio is None:
            inizio = i
        elif not v and inizio is not None:
            fuori.append((inizio, i - 1))
            inizio = None
    if inizio is not None:
        fuori.append((inizio, len(contenuto) - 1))
    return fuori


def piano(arr: np.ndarray, tieni_legenda: bool) -> tuple[int, int, list[str]]:
    """Da dove a dove tenere l'immagine, e cosa si sta togliendo.

    Ritorna `(y0, y1, cosa)` con `y1` ESCLUSO, come una fetta.
    """
    b = bande(arr)
    if not b:
        return 0, arr.shape[0], []
    grafico = max(b, key=lambda t: t[1] - t[0])
    sotto = [t for t in b if t[0] > grafico[1]]
    legenda = next((t for t in sotto if (t[1] - t[0]) >= H_LEGENDA), None)
    cosa: list[str] = []

    y0 = 0
    if grafico[0] > PAD:
        y0 = max(0, grafico[0] - PAD)
        cosa.append("titolo")

    y1 = arr.shape[0]
    if legenda is None:
        # Nessuna legenda sotto il riquadro: le traiettorie ce l'hanno dentro.
        # Non si tocca niente in fondo, le etichette degli assi restano.
        return y0, y1, cosa

    if tieni_legenda:
        if legenda[1] + PAD < arr.shape[0] - 1:
            y1 = min(arr.shape[0], legenda[1] + PAD)
            if any(t[0] > legenda[1] for t in sotto):
                cosa.append("nota")
    else:
        prima = [t for t in sotto if t[1] < legenda[0]]
        fine = prima[-1][1] if prima else grafico[1]
        y1 = min(arr.shape[0], fine + PAD)
        cosa.append("legenda")
        if any(t[0] > legenda[1] for t in sotto):
            cosa.append("nota")
    return y0, y1, cosa


def conta_trimestri(csv_rel: str | None) -> int | None:
    """Il numero di trimestri su cui la curva e' disegnata, dalla passata.

    Le righe con `pieno` falso sono le settimane che la regola di copertura
    esclude: il campione della figura e' quello delle righe piene.
    """
    if csv_rel is None:
        return None
    p = CSV_DIR / csv_rel
    if not p.exists():
        return None
    import pandas as pd
    d = pd.read_csv(p)
    if "pieno" not in d.columns or "n_trimestri" not in d.columns:
        return None
    pieno = d[d["pieno"].astype(str).str.lower().isin(("true", "1"))]
    if pieno.empty:
        return None
    return int(pieno["n_trimestri"].max())


def lavora(nome: str, cfg: dict, solo_check: bool) -> None:
    p = FIG_DIR / nome
    if not p.exists():
        print(f"  MANCA   {nome}")
        return
    im = Image.open(p)
    arr = np.asarray(im.convert("RGB")).astype(int)
    y0, y1, cosa = piano(arr, cfg["legenda"])

    atteso = cfg.get("quarters")
    visto = conta_trimestri(cfg.get("csv"))
    avviso = ""
    if atteso is not None and visto is not None and visto != atteso:
        avviso = (f"  ATTENZIONE: la didascalia dice {atteso} trimestri, "
                  f"la passata ne ha {visto}")

    if not cosa:
        print(f"  a posto  {nome}{avviso}")
        return
    print(f"  {'toglierei' if solo_check else 'tolto'} "
          f"{', '.join(cosa):<18} {nome}"
          f"   ({arr.shape[0]} -> {y1 - y0} px){avviso}")
    if solo_check:
        return
    im.crop((0, y0, im.width, y1)).save(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--check", action="store_true",
                    help="dice cosa farebbe senza scrivere niente")
    a = ap.parse_args()

    print(f"figure in {FIG_DIR}\n")
    for nome, cfg in FIGURES.items():
        lavora(nome, cfg, a.check)

    print("\nDA METTERE IN DIDASCALIA (vincolo: senza queste righe la figura "
          "non si identifica piu'):")
    for nome, cfg in FIGURES.items():
        print(f"  {nome:32s} {cfg['ident']}")


if __name__ == "__main__":
    main()
