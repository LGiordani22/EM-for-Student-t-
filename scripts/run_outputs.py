#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_outputs.py — TUTTO CIO' CHE VIENE DOPO LE STIME.

    python scripts/run_outputs.py

Figure, metriche, tabelle e confronti, letti EX POST dai CSV che `run_dfm.py` e
`run_bvar.py` hanno lasciato su disco.  Qui non si stima niente: se un numero
non torna, si ripara e si rilancia questo script — non la passata.  Dura
minuti, non ore.

QUANDO LANCIARLO
----------------
Quando le quindici celle DFM e i blocchi BVAR sono finiti.  Il passo 3 lo
verifica e si ferma se non e' vero: figure e tabelle su CSV senza numeri
dentro sono passate senza lamentarsi per sette fasi, una volta.

Non serve che le stime siano girate QUI: il passo 2 pubblica le celle che
trova in `csv/_cells/`, quindi un albero copiato dal server basta e avanza.

I PASSI, NELL'ORDINE (che e' obbligato)
---------------------------------------
    1  guardie        le finestre tagliano il periodo che dichiarano, e il
                      vincolo di campione comune scatta davvero
    2  raccolta       le celle di `csv/_cells/` diventano i CSV di `csv/dfm/`
    3  celle          i quindici CSV del DFM contengono nowcast, non eccezioni
    4  figure         traiettorie DFM e BVAR, una per finestra forecast
    5  nyfed          confronto con la Fed + figure RMSE per orizzonte (DFM)
    6  bvar-metrics   tabelle BVAR + LA figura coi quattro modelli insieme
    7  tabelle        famiglie, matrici di confronto, il PER FASE, il backcast
    8  allineamento   la guardia severa: DFM e BVAR si incontrano sulle chiavi
                      del join, quindi le tabelle di confronto sono affidabili

    python scripts/run_outputs.py --list          i passi, uno per riga
    python scripts/run_outputs.py --only tabelle  rifa' un passo solo

UNA FINESTRA SENZA DATI SI SALTA
--------------------------------
`figures` esce con errore quando nessuna riga cade nella finestra.  Su una
passata piena tutte e quattro hanno dati; su una ripresa a meta' no, e buttare
via il lavoro fatto per una figura che non si puo' disegnare e' il
comportamento sbagliato.  Si distingue il vuoto (si salta, e lo si dice) dal
guasto (ferma tutto).

DOVE FINISCE L'USCITA A VIDEO
-----------------------------
Un file per passo in `output/_logs/`.  A schermo resta la traccia dei passi;
il dettaglio si legge li' senza rilanciare la catena.

DA GUARDARE PER PRIMO, A FINE CORSA
-----------------------------------
    output/forecast_weekly/comparison/summary.txt
        le matrici di confronto, la sezione PER FASE e il pannello BACKCAST
    output/forecast_weekly/bvar/rmse/rmse_per_orizzonte_bvar_*.png
        i quattro BVAR insieme, col NY Fed in nero
    output/forecast_weekly/dfm/fed_overlap/rmse/metrics_fed_overlap.txt
        le tabelle del DFM, con n/n_com affiancati

Leggendo, ricorda: SOLO le colonne `_com` confrontano le righe fra loro.  Sono
i punti (trimestre, settimana) punteggiati da TUTTI i metodi della tabella.  Un
metodo che parte piu' tardi, o che si ferma alla settimana 13 invece che alla
17, e' mediato su bersagli diversi.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core import output_layout as layout

#: Le finestre forecast con una figura di traiettoria ciascuna.
_FIG_WINDOWS = tuple(layout.FORECAST_WINDOWS)

#: Cio' che, in un log di `figures`, significa "niente da disegnare" e non
#: "qualcosa si e' rotto".
_VUOTO = ("Nessuna riga", "Nessun CSV")


def _steps(n_cells: int) -> list[dict]:
    """
    La catena, un passo per voce.

    `tollera`: sottostringhe che nel log declassano un'uscita non nulla da
    guasto a "non c'era niente da fare".  `fatale=False`: il passo puo'
    fallire senza fermare la catena.
    """
    figure: list[list[str]] = []
    for w in _FIG_WINDOWS:
        figure.append(["-m", "core.forecast.figures", "--window", w])
        figure.append(["-m", "core.bvar.figures", "--window", w])

    return [
        {"nome": "guardie", "cmds": [
            ["-m", "tests.forecast.test_windows", "--pre-run"],
            ["-m", "tests.forecast.test_common_sample"],
        ]},
        {"nome": "raccolta", "cmds": [
            ["-m", "core.forecast.collect"],
        ]},
        {"nome": "celle", "cmds": [
            ["-m", "tests.forecast.test_cells_produced", "--expect", str(n_cells)],
        ]},
        {"nome": "figure", "cmds": figure, "tollera": _VUOTO},
        {"nome": "nyfed", "cmds": [
            ["-m", "core.forecast.nyfed_all", "--spec", s] for s in layout.SPECS
        ], "fatale": False},
        {"nome": "bvar-metrics", "cmds": [["-m", "core.bvar.metrics"]]},
        {"nome": "tabelle", "cmds": [["-m", "core.forecast.metrics_tables"]]},
        {"nome": "allineamento", "cmds": [["-m", "tests.forecast.test_windows"]]},
    ]


def _slug(argv: list[str]) -> str:
    """Un nome di file leggibile a partire dagli argomenti del sottoprocesso."""
    pezzi = [p.replace("core.", "").replace("tests.", "").replace(".", "_")
             for p in argv if not p.startswith("-")]
    return "_".join(pezzi) or "passo"


def _run(argv: list[str], log_path: str, tollera: tuple[str, ...]) -> int:
    """Esegue un sottoprocesso, scrive il log, torna il codice d'uscita."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call([sys.executable] + argv, cwd=_ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    if rc and tollera:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            testo = fh.read()
        if any(m in testo for m in tollera):
            return 0
    return rc


def main() -> int:
    passi = _steps(len(layout.SPECS) * len(layout.VARIANTS))
    nomi = [p["nome"] for p in passi]

    p = argparse.ArgumentParser(
        description="Figure, metriche e tabelle, lette ex post dai CSV.")
    p.add_argument("--list", action="store_true", help="i passi, ed esce")
    p.add_argument("--only", choices=nomi, default=None,
                   help="esegue un passo solo")
    p.add_argument("--from", dest="da", choices=nomi, default=None,
                   help="riprende da questo passo in poi")
    a = p.parse_args()

    if a.list:
        for nome in nomi:
            print(nome)
        return 0

    if a.only:
        passi = [q for q in passi if q["nome"] == a.only]
    elif a.da:
        passi = passi[nomi.index(a.da):]

    logs = layout.logs_dir()
    os.makedirs(logs, exist_ok=True)
    layout.build_tree()

    print("=" * 78)
    print(f"  USCITE — {len(passi)} passi, log in {logs}")
    print("=" * 78)

    t0 = time.perf_counter()
    for i, passo in enumerate(passi, 1):
        nome = passo["nome"]
        tollera = tuple(passo.get("tollera", ()))
        fatale = passo.get("fatale", True)
        print(f"\n[{i}/{len(passi)}]  {nome}")
        for argv in passo["cmds"]:
            log_path = os.path.join(logs, f"{nome}_{_slug(argv)}.log")
            rc = _run(argv, log_path, tollera)
            etichetta = " ".join(argv[1:])
            if rc == 0:
                print(f"    ok      {etichetta}")
            elif not fatale:
                print(f"    saltato {etichetta}  (vedi {log_path})")
            else:
                print(f"    ROTTO   {etichetta}")
                print(f"\n  Il passo '{nome}' e' fallito.  Il perche' sta in:")
                print(f"    {log_path}")
                print("  I CSV delle stime restano su disco: si ripara la causa")
                print(f"  e si rilancia da qui — `--from {nome}` — non la passata.")
                return 1

    print(f"\n{'=' * 78}")
    print(f"  FATTO in {time.perf_counter() - t0:.0f}s.")
    print(f"  Da guardare per primo: "
          f"{os.path.join(layout.comparison_dir(), 'summary.txt')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
