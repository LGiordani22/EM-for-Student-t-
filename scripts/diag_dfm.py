"""
DIAGNOSI IN CINQUE SECONDI: perche' il DFM non produce nowcast su questa macchina.

    cd <radice del repo>
    python scripts/diag_dfm.py

PERCHE' ESISTE
--------------
La passata sul server ha prodotto quindici CSV con 34 155 righe e nessun
numero dentro: `nowcast()` ha sollevato un'eccezione a OGNI chiamata, e
`weekly_nowcast` la cattura riga per riga scrivendo `n_iter=-1` e tirando
avanti.  Il CSV registra CHE e' fallito, mai PERCHE': il messaggio finisce
solo a video, in `output/_logs/dfm_*.log`, che e' gitignorato e quindi non
viaggia col push.

Questo script rifa' UNA SOLA chiamata, fuori dal ciclo e senza rete di
protezione, e stampa il traceback completo — la catena di chiamate, che nel
log non c'e'.  Costa pochi secondi: una sola settimana, la cella piu' leggera
(diag3/gaussian).

Stampa PRIMA il contorno, perche' quasi sempre e' li' che sta la differenza
fra due macchine: versioni di Python e delle librerie numeriche, forma del
pannello, copertura delle serie, presenza dei file di configurazione.  Il DFM
li usa tutti; i benchmark AR(2) e media espandente — che sul server hanno
funzionato — non usano che la colonna del PIL.  Se il pannello arriva mutilo,
i benchmark passano e il DFM muore: esattamente il quadro osservato.
"""

from __future__ import annotations

import os
import sys
import traceback

# Lanciando `python scripts/diag_dfm.py`, Python mette sul path la cartella
# DELLO SCRIPT (`scripts/`), non quella da cui lo si lancia: senza questa
# riga `import src...` fallisce anche stando nella radice del repo.  Stessa
# medicina che usa `src/forecast/__init__.py` per far risolvere `em.*`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SEP = "=" * 78


def main() -> int:
    print(SEP)
    print("  DIAGNOSI DFM")
    print(SEP)

    print(f"  cwd    : {os.getcwd()}")
    print(f"  python : {sys.version.splitlines()[0]}")
    print(f"  exe    : {sys.executable}")

    try:
        import numpy
        import pandas
        import scipy
        print(f"  numpy  : {numpy.__version__}   pandas: {pandas.__version__}"
              f"   scipy: {scipy.__version__}")
    except Exception:                                        # noqa: BLE001
        print("\n>>> NON RIESCO NEMMENO A IMPORTARE numpy/pandas/scipy:")
        traceback.print_exc()
        return 1

    # ── 1. Gli import del motore ────────────────────────────────────────────
    # Se cade qui, il guasto e' d'ambiente (path, dipendenze, versione di
    # Python) e non arriva nemmeno a guardare un dato.
    try:
        from src.forecast.nowcast_engine import nowcast
        from src.forecast.release_calendar import load_metadata, load_panel
    except Exception:                                        # noqa: BLE001
        print("\n>>> FALLITO ALL'IMPORT DEL MOTORE:")
        traceback.print_exc()
        print("\n  (se e' un ModuleNotFoundError su 'src', lancia lo script")
        print("   dalla RADICE del repo, non da scripts/)")
        return 1
    print("  import : ok")

    # ── 2. I file di configurazione ─────────────────────────────────────────
    # Il DFM li legge, i benchmark no.  Un file assente o illeggibile spiega
    # da solo un fallimento sistematico che risparmia ar2 e mean.
    for rel in ("config/series_final.json", "config/factor_specs.json"):
        stato = f"{os.path.getsize(rel)} byte" if os.path.exists(rel) else "ASSENTE"
        print(f"  config : {rel:<32} {stato}")

    # ── 3. Il pannello ──────────────────────────────────────────────────────
    try:
        panel, meta = load_panel(), load_metadata()
    except Exception:                                        # noqa: BLE001
        print("\n>>> FALLITO NEL COSTRUIRE IL PANNELLO:")
        traceback.print_exc()
        return 1

    print(f"  pannello: {panel.shape[0]} mesi x {panel.shape[1]} serie"
          f"   ({panel.index[0].date()} .. {panel.index[-1].date()})")

    # Le serie VUOTE sono la spia di un `data/raw` incompleto: il pannello ha
    # la forma giusta ma le colonne non hanno dentro niente.  I benchmark
    # userebbero comunque la sola colonna del PIL e non se ne accorgerebbero.
    vuote = [c for c in panel.columns if panel[c].notna().sum() == 0]
    scarse = [c for c in panel.columns
              if 0 < panel[c].notna().sum() < 0.20 * len(panel)]
    print(f"  serie vuote : {len(vuote)}"
          + (f"  -> {vuote[:8]}" if vuote else ""))
    print(f"  serie scarse: {len(scarse)} (meno del 20% di osservazioni)"
          + (f"  -> {scarse[:8]}" if scarse else ""))

    # ── 4. LA chiamata ──────────────────────────────────────────────────────
    print(f"\n{SEP}\n  chiamata di prova: diag3/gaussian, as_of 2015-01-02, "
          f"target 2015Q1\n{SEP}")
    try:
        r = nowcast("2015-01-02", "2015Q1", "diag3", "gaussian",
                    panel=panel, metadata=meta)
    except Exception:                                        # noqa: BLE001
        print(">>> ECCO L'ECCEZIONE CHE SVUOTA I CSV DEL DFM:\n")
        traceback.print_exc()
        return 1

    print(f">>> OK — nowcast = {r['nowcast_bea']:+.4f}%   "
          f"n_iter = {r['n_iter']}   converged = {r['converged']}")
    print("\n  Il DFM su questa macchina FUNZIONA: il guasto e' altrove")
    print("  (o e' gia' stato riparato).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
