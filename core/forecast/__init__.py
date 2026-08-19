"""
core/forecast/ — nowcast settimanale del PIL sul pannello `final` (37 serie).

    release_calendar.py  il calendario di pubblicazione come servizio
    scale.py             la catena di scala, andata e ritorno
    panel_builder.py     il pannello com'era noto a una data
    nowcast_engine.py    una cella (spec x variante) -> un nowcast
    benchmarks.py        i metri di paragone univariati
    weekly_nowcast.py    il ciclo settimanale su tutte le celle
    compute_metrics.py   accuratezza
    figures.py           la figura in stile Cascaldi-Garcia 8a
    nyfed_nowcast.py     il nowcast NY Fed, allineato al mio metro di orizzonte
    compare_nyfed.py     io contro la NY Fed e i benchmark (valutazione a valle)

IL FLUSSO
---------
`release_calendar` (calendario) -> `panel_builder` (vintage a una data) ->
`nowcast_engine` (una cella spec x variante -> un nowcast, via `em.fit_dfm` o
`filter_only` + `scale`) -> `weekly_nowcast` (ciclo su ogni venerdi', 15 celle
+ 2 benchmark) -> CSV -> `compute_metrics` (RMSE relativo vs AR(2)/media
espandente, MDA, SignAcc) e `figures` (Cascaldi-Garcia 8a).

A valle e in parallelo, senza toccare il motore: `nyfed_nowcast` legge il NY Fed
Staff Nowcast dagli Excel in `data/raw/` e gli applica la stessa `horizon_week`
del mio ciclo; `compare_nyfed` mette io, Fed e benchmark sulla stessa riga
all'ultima stima prima del rilascio del PIL.  I dati NY Fed non entrano nella
figura 8a.

Tre spec (`diag3`, `diag4`, `fed_overlap`), cinque varianti, due benchmark
(AR(2) e media espandente).  Il target e' il PIL in QoQ annualizzato.
"""

import os
import sys

# Il motore EM vive in `core/dfm/` e i suoi moduli si importano fra loro come
# `em.*`: perche' quelle importazioni si risolvano, `core/` deve stare sul path.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
