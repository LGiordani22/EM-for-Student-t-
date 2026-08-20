"""
src/forecast/nyfed_all.py

IL CONFRONTO COL NY FED PER TUTTE LE SPEC E TUTTE LE FINESTRE, IN POCHI FILE.

    python -m src.forecast.nyfed_all

PERCHE' ESISTE QUESTO MODULO
----------------------------
`compare_nyfed.main` fa una coppia (spec, finestra) per invocazione e SCRIVE
subito: nove file ogni volta.  Tre spec per sei finestre sono diciotto
invocazioni, cioe' CENTOSESSANTADUE file — di cui solo diciotto sono figure,
che e' quello che si voleva.  Le altre centoquarantaquattro sono la stessa
tabella in ventiquattro copie, una per combinazione.

Qui il ciclo sta dentro un processo solo, le tabelle si IMPILANO con una
colonna `window` in testa, e si scrivono in fondo.  Le figure restano una per
finestra — quelle sono il prodotto, non il disordine.

    per cartella `dfm/<spec>/rmse/`:
        prima                          dopo
         6  nyfed_report_<fin>.txt      1  confronto_nyfed.txt
        30  nyfed_<tabella>_<fin>.csv   5  nyfed_<tabella>.csv
         6  nyfed_panel_finale_<fin>    1  nyfed_panel_finale.csv
         6  rmse_by_horizon_<fin>.csv   1  rmse_by_horizon.csv
         6  PNG                         6  PNG
        ──                             ──
        54                             14

DOVE VANNO: DOVE STAVANO
------------------------
Non si accorpa FRA le spec.  Le tabelle di `fed_overlap` servono a leggere le
figure di `fed_overlap` e restano nella sua cartella — `output_layout` lo dice
esplicitamente: le metriche vivono accanto alle figure che commentano.  Si
impila per FINESTRA, che e' la dimensione che moltiplicava i file, non per
spec, che e' la dimensione che li organizza.

E' la stessa forma gia' adottata due volte: le tabelle di famiglia in
`metrics_tables` e il confronto NY Fed del BVAR in `bvar/metrics`.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src import output_layout as layout
from src.forecast import compare_nyfed as cn
from src.forecast import compute_metrics as cm

#: Le sei finestre in cui si riporta l'RMSE: tre passate cumulate e tre zoom.
WINDOWS: list[str] = list(layout.RMSE_PASSES) + list(layout.RMSE_ZOOM_WINDOWS)


def run_spec(mine_all: pd.DataFrame, spec: str,
             windows: list[str] = WINDOWS,
             out_dir: str | None = None) -> list[str]:
    """Una spec, tutte le finestre: figure per finestra, tabelle impilate."""
    out_dir = out_dir or layout.dfm_rmse_dir(spec)
    os.makedirs(out_dir, exist_ok=True)

    tabs: dict[str, list[pd.DataFrame]] = {}
    testi: list[str] = []
    horiz: list[pd.DataFrame] = []
    written: list[str] = []

    for w in windows:
        mine = layout.slice_window(mine_all, w, column="as_of")
        if mine.empty:
            print(f"  [{spec}/{w}] nessuna riga nella finestra — salto.")
            continue

        # ── le tabelle del confronto ─────────────────────────────────────────
        try:
            panel, sample, registro = cn.build_panel(mine, [spec])
        except (FileNotFoundError, KeyError) as exc:
            print(f"  [{spec}/{w}] NY Fed non confrontabile "
                  f"({type(exc).__name__}) — solo la figura.")
            sample = []
        if sample:
            report, t = cn.build_report(panel, sample, registro)
            testi.append(cm._section(f"SPEC {spec}  —  FINESTRA {w}") + "\n" + report)
            for name, df in t.items():
                df = df.copy()
                df.insert(0, "window", w)
                tabs.setdefault(name, []).append(df)
            p = panel.copy()
            p.insert(0, "window", w)
            tabs.setdefault("panel_finale", []).append(p)
            print(f"  [{spec}/{w}] {len(sample)} trimestri allineati col NY Fed")

        # ── la figura, che NON dipende dalla Fed ─────────────────────────────
        # La Fed entra come curva quando copre gli stessi trimestri
        # (`horizon_panel` la carica da se'), ma se non li copre la figura esce
        # lo stesso: l'RMSE per orizzonte e' una metrica mia.
        # TUTTI i trimestri della finestra: il campione lo sceglie
        # `horizon_panel`, che applica la regola DOPO aver ristretto ai metodi
        # di questa spec.  Pre-filtrare qui era sbagliato due volte: la regola
        # era l'unione esatta (vedi `core_coverage_quarters`), e si applicava
        # al frame di TUTTE le spec — cosi' le due celle guaste di diag3
        # tagliavano il campione anche a diag4 e fed_overlap, che sono sane.
        qs = sorted(mine["target_quarter"].unique())
        ph, sample_fig = cn.horizon_panel(mine, qs, spec)
        if not sample_fig or ph.empty:
            print(f"  [{spec}/{w}] copertura incompleta: niente figura.")
            continue
        h = ph.copy()
        h.insert(0, "window", w)
        horiz.append(h)
        written.append(cn.figure_rmse_by_horizon(
            ph, sample_fig, spec,
            os.path.join(out_dir, f"rmse_by_horizon_{spec}_{w}.png")))

    # ── scrittura, UNA VOLTA ────────────────────────────────────────────────
    for name, parts in tabs.items():
        p = os.path.join(out_dir, f"nyfed_{name}.csv")
        pd.concat(parts, ignore_index=True).to_csv(p, index=False)
        written.append(p)
    if horiz:
        p = os.path.join(out_dir, f"rmse_by_horizon_{spec}.csv")
        pd.concat(horiz, ignore_index=True).to_csv(p, index=False)
        written.append(p)
    if testi:
        p = os.path.join(out_dir, "confronto_nyfed.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(f"CONFRONTO COL NY FED STAFF NOWCAST — spec {spec}\n"
                     f"Un blocco per finestra.  I CSV accanto sono in formato "
                     f"lungo: si filtra sulla\ncolonna `window` invece di "
                     f"aprire un file diverso per periodo.\n"
                     + "\n".join(testi) + "\n")
        written.append(p)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Confronto NY Fed per tutte le spec e finestre, in pochi file.")
    p.add_argument("--csv", nargs="*", default=None)
    p.add_argument("--spec", nargs="*", default=None,
                   help=f"default: {list(layout.SPECS)}")
    p.add_argument("--window", nargs="*", default=None,
                   help=f"default: {WINDOWS}")
    a = p.parse_args()

    mine, _, _ = cn.load_mine(a.csv)
    specs = a.spec or list(layout.SPECS)
    windows = a.window or WINDOWS
    print(f"{len(mine)} righe, {mine['target_quarter'].nunique()} trimestri; "
          f"spec {specs}, finestre {windows}")

    tot = []
    for s in specs:
        if not any(m.split("/")[0] == s for m in mine["metodo"].unique()):
            print(f"  [{s}] assente dal CSV — salto.")
            continue
        print(cm._section(f"SPEC {s}"))
        tot += run_spec(mine, s, windows)
    print(f"\n{len(tot)} file scritti "
          f"(erano {9 * len(specs) * len(windows)} col vecchio percorso).")


if __name__ == "__main__":
    main()
