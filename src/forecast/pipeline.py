"""
src/forecast/pipeline.py — LE STESSE SETTIMANE, IN UN ALTRO ORDINE.

Una cella DFM eseguita in un pool di processi invece che in fila.  Non cambia
un numero: cambia chi esegue e quando.

    from src.forecast.pipeline import run_cell_pipeline
    df = run_cell_pipeline("diag3", "student_t_ar1", "2007-01-01", "2025-12-31",
                           out_dir=..., workers=4)

PERCHE' SI PUO' FARE
--------------------
Un venerdi' e' di due tipi.

  RI-STIMA   l'EM riparte dal theta del venerdi' prima.  Le ~228 ri-stime di
             una cella sono una CATENA: la n-esima ha bisogno del risultato
             della (n-1)-esima.  Non si parallelizzano, mai.

  CONGELATA  theta fermo, si ricalcola solo lo stato.  E' una FUNZIONE PURA di
             (as_of, target, theta): non produce theta, non legge altro, non
             influenza nessun'altra riga.  Le ~763 congelate di una cella sono
             FOGLIE, e le foglie si colgono in qualsiasi ordine.

Oggi le foglie stanno nella stessa fila delle ri-stime, quindi ognuna blocca la
ri-stima che viene dopo pur non servendole a niente.  Qui no: lo scheduler
scorre i venerdi' in avanti mettendo in coda le congelate e si FERMA sulla
prima ri-stima; la cella resta bloccata finche' quella non torna, e intanto le
congelate girano nel pool.

Il cammino critico diventa la sola catena delle ri-stime.  Misurato su
`diag3/student_t_ar1`, 991 venerdi': **155,4 min in fila contro 62,1 min con
quattro processi**, con il CSV identico bit per bit (2026-09-01).

PERCHE' I NUMERI NON POSSONO CAMBIARE
-------------------------------------
1. `esegui_settimana` e' PURA e sta in `weekly_nowcast.py`: la chiamano sia il
   ciclo sequenziale sia questo modulo.  Una copia sola della cascata di
   ripieghi, quindi le due strade non possono divergere per trascrizione.
2. Una settimana congelata NON produce theta: `nowcast` con `theta=` dato
   restituisce lo stesso oggetto, quindi la catena non la vede passare.
3. La catena delle ri-stime resta nello stesso ORDINE, perche' `_release` si
   blocca finche' la stima non ha fatto commit.
4. Nel percorso DFM non c'e' nessun RNG che dipenda dall'ordine: i semi sono
   costanti fisse (42 nel riempimento del bordo frastagliato, 0 in `scale.py`
   e in `em_main.py`).  Nessun `seed + i` con `i` indice di ciclo — che e'
   precisamente il difetto che aveva il BVAR, dove spezzare CAMBIAVA i numeri.

CIO' CHE DEVE RESTARE VERO
--------------------------
* un thread BLAS per processo (`_init_worker`): con piu' thread l'ordine delle
  riduzioni nei prodotti cambia, e con esso l'ultimo bit;
* la settimana e' ATOMICA — tutti i target dello stesso venerdi' in UN job,
  perche' il 2o e il 3o riusano il theta appena stimato dal 1o;
* la cascata di ripieghi sta DENTRO il job, intera: se il livello PCA fosse
  deciso dallo scheduler, l'esito di una settimana dipenderebbe da quando e'
  stata schedulata;
* un solo scrittore sul CSV: lo scheduler.

QUANTI WORKER SERVONO: QUATTRO
------------------------------
Non di piu', ed e' un tetto strutturale.  Dopo una ri-stima alla settimana `i`,
`_release` emette le ~3 congelate del mese e la ri-stima di `i+4`, poi si
ferma: **quattro lavori in volo**, e il quinto dipende da un theta che non
esiste ancora.  Il guadagno massimo per cella e' quindi

    tempo totale / tempo della sola catena = 2,45 h / 0,81 h ~ 3,0x

e oltre i quattro processi i worker restano fermi.  Chi ha molti core li usa
lanciando piu' CELLE insieme, che e' quel che `run_all.py` gia' fa: 15 celle
x 4 worker = 60 processi.

UN JOB CHE MUORE NON PORTA VIA GLI ALTRI
-----------------------------------------
`lg_nowcasting`, da cui viene il disegno, su qualsiasi eccezione di un worker fa
`raise` e ammazza l'intero studio.  Qui no: la settimana fallita produce le sue
righe con `n_iter=-1` — che e' il segnale che `cell_health` gia' sa leggere — e
la corsa continua.  E' la regola di `run_all.py`, e non la si perde per un
cambio di scheduling.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast.release_calendar import (
    load_metadata, load_panel, releases_between, targets_in_flight, weekly_grid,
)
from src.forecast.weekly_nowcast import (
    COLUMNS, _KEY, _atomic_write, _csv_path, _load_existing, _row_key,
    _save_theta, esegui_settimana, report_health, riga_di_stato,
)


#: Il tetto oltre il quale i worker restano fermi (vedi l'intestazione).
WORKER_UTILI = 4


# ─── I worker: il pannello si legge UNA VOLTA per processo ────────────────────

_PANEL: pd.DataFrame | None = None
_META: pd.DataFrame | None = None
_TARGET: str | None = None


def _init_worker() -> None:
    """
    Inizializzatore del pool: fissa il BLAS a un thread e carica il pannello.

    IL PANNELLO NON ATTRAVERSA IL CONFINE DI PROCESSO.  Passarlo a ogni job
    vorrebbe dire serializzare l'intero pannello per ciascuna delle ~991
    settimane; caricato qui, si legge una volta per processo e basta.  Il theta
    invece viaggia, ed e' giusto cosi': e' l'unica cosa che cambia di settimana
    in settimana, e in pickle un `float64` e' esatto.
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _PANEL, _META, _TARGET
    _PANEL, _META = load_panel(), load_metadata()
    _TARGET = list(_PANEL.columns)[-1]


def _lavora(as_of, targets, spec, variant, theta, reestimate, n_rel,
            max_iter, verbose_em, da_saltare):
    """Il job che gira nel worker: solo un ponte verso `esegui_settimana`."""
    global _PANEL, _META, _TARGET
    if _PANEL is None:                      # esecuzione in-process (workers=1)
        _init_worker()
    return esegui_settimana(as_of, targets, spec, variant, theta, reestimate,
                            n_rel, _PANEL, _META, _TARGET, max_iter=max_iter,
                            verbose_em=verbose_em, da_saltare=da_saltare)


# ─── La firma: perche' un CSV vecchio non si riusi per sbaglio ────────────────

def firma_corsa(spec: str, variant: str, start: str, end: str,
                em_frequency: str, n_ahead: int, max_iter: int,
                panel: pd.DataFrame) -> dict:
    """
    Cosa deve coincidere perche' le righe gia' su disco siano RIUSABILI.

    PERCHE' ESISTE.  La ripresa salta le righe gia' presenti nel CSV, e finche'
    il codice e la configurazione sono gli stessi e' un risparmio.  Ma se il
    file viene da un'altra passata, riprendere ci MESCOLA dentro righe prodotte
    da un altro modello, e il file risultante non e' ne' l'una ne' l'altra cosa
    — in silenzio, perche' un CSV non dice da dove viene.

    Non e' un timore teorico: il 2026-09-01 abbiamo misurato che il codice
    corrente NON riproduce i CSV della passata del 27-8 (tutte le 2277 righe di
    `diag3/student_t_ar1` diverse, fino a 11,3 punti BEA), per via del commit
    `9564fae` del 28-8 sul miglior iterato dell'EM.  Un rilancio "di ripresa"
    su quei file avrebbe cucito insieme due modelli diversi senza un avviso.

    Nel confronto entrano la configurazione e il DATO; il codice no, perche' un
    digest dei sorgenti scatterebbe anche per un commento e renderebbe la
    ripresa inservibile.  Al suo posto si registra `git HEAD`, che non blocca
    ma si stampa: se e' cambiato, chi legge lo sa.
    """
    return {
        "spec": spec, "variant": variant, "start": start, "end": end,
        "em_frequency": em_frequency, "n_ahead": int(n_ahead),
        "max_iter": int(max_iter),
        "panel": _digest_pannello(panel),
    }


def _digest_pannello(panel: pd.DataFrame) -> str:
    from hashlib import sha256
    h = sha256()
    h.update(",".join(map(str, panel.columns)).encode())
    h.update(str(panel.shape).encode())
    h.update(np.ascontiguousarray(panel.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:16]


def _git_head() -> str:
    import subprocess
    try:
        radice = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=radice, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def _percorso_firma(csv_path: str) -> str:
    return csv_path[:-4] + ".firma.json"


def _leggi_righe_riusabili(csv_path: str, firma: dict) -> tuple[list[dict], str]:
    """
    Le righe gia' fatte, SOLO se provengono dalla stessa corsa.

    Torna `(righe, motivo)`.  `righe` vuoto significa "si ricalcola tutto", e
    `motivo` dice perche' — che va stampato, mai taciuto.
    """
    if not os.path.exists(csv_path):
        return [], "nessun CSV precedente"
    p_firma = _percorso_firma(csv_path)
    if not os.path.exists(p_firma):
        return [], (f"c'e' un CSV ma non la sua firma ({os.path.basename(p_firma)}): "
                    f"non si sa da quale corsa venga, quindi NON si riusa")
    try:
        vecchia = json.load(open(p_firma, encoding="utf-8"))
    except Exception as exc:
        return [], f"firma illeggibile ({type(exc).__name__}): non si riusa"
    diverse = [k for k in firma if vecchia.get(k) != firma[k]]
    if diverse:
        return [], (f"la firma non coincide su {', '.join(diverse)}: "
                    f"il CSV viene da un'altra corsa, NON si riusa")
    return _load_existing(csv_path), (
        f"ripresa dal CSV precedente (firma coincide; git {vecchia.get('git', '?')}"
        + (f" -> {firma.get('git', '?')}, IL CODICE E' CAMBIATO"
           if vecchia.get("git") != _git_head() else "")
        + ")")


# ─── Lo scheduler ─────────────────────────────────────────────────────────────

@dataclass
class _Lavoro:
    tipo: str            # 'stima' | 'congelata'
    i: int               # indice del venerdi' nel piano
    theta: dict | None


def run_cell_pipeline(spec: str, variant: str, start: str, end: str, *,
                      out_dir: str, workers: int = WORKER_UTILI,
                      n_ahead: int = 1, em_frequency: str = "monthly",
                      max_iter: int = 250, save: bool = True,
                      verbose_em: bool = False,
                      verbose: bool = True) -> pd.DataFrame:
    """
    UNA cella sul calendario settimanale, con le settimane congelate in parallelo.

    Stessa uscita di `run_weekly_nowcast` per una cella sola: lo stesso CSV,
    nella stessa cartella, con gli stessi theta accanto.  `workers=1` esegue
    tutto in-process ed e' il RIFERIMENTO contro cui si misura il resto.
    """
    t_start = time.perf_counter()
    if workers < 1:
        raise ValueError("workers dev'essere >= 1")
    if em_frequency not in ("weekly", "monthly"):
        raise ValueError(f"em_frequency {em_frequency!r}: 'weekly' o 'monthly'.")

    panel, meta = load_panel(), load_metadata()
    target_series = list(panel.columns)[-1]
    grid = weekly_grid(start, end)
    if not grid:
        raise SystemExit(f"nessun venerdi' fra {start} e {end}.")

    piano = [(d, targets_in_flight(d, n_ahead=n_ahead,
                                   target_series=target_series, metadata=meta))
             for d in grid]
    # `n_releases` guarda il venerdi' PRECEDENTE della griglia: e' una proprieta'
    # del calendario, non dell'esecuzione, quindi si precalcola tutta qui e non
    # dipende dall'ordine in cui le settimane vengono eseguite.
    n_rel = [np.nan] + [len(releases_between(grid[i - 1], grid[i], panel, meta))
                        for i in range(1, len(grid))]

    os.makedirs(out_dir, exist_ok=True)
    path = _csv_path(out_dir, start, end)
    firma = firma_corsa(spec, variant, start, end, em_frequency, n_ahead,
                        max_iter, panel)
    firma_su_disco = dict(firma, git=_git_head())

    rows, motivo = ([], "salvataggio disattivato") if not save else \
        _leggi_righe_riusabili(path, firma)
    done = {_row_key(r) for r in rows}

    if verbose:
        print(f"\n{'=' * 78}\n  PIPELINE  {spec} / {variant}  "
              f"workers={workers}\n{'=' * 78}")
        print(f"  griglia    : {len(grid)} settimane, "
              f"{grid[0].date()} .. {grid[-1].date()}")
        print(f"  ripresa    : {motivo}")
        print(f"  gia' fatti : {len(rows)} righe")
        if workers > WORKER_UTILI:
            print(f"  NOTA       : oltre {WORKER_UTILI} worker una cella non ha "
                  f"lavoro da dare; il resto resta fermo.")
        if save:
            print(f"  CSV        : {path}\n")

    # ── Lo stato della cella: vive SOLO qui ──────────────────────────────────
    theta: dict | None = None
    last_em_month: tuple[int, int] | None = None
    prossimo = 0
    in_volo = 0
    per_settimana: dict[int, list[dict]] = {}
    n_em = 0
    err_msgs: dict[tuple[str, str], Counter] = {}

    pronte_stime: deque[_Lavoro] = deque()
    pronte_congelate: deque[_Lavoro] = deque()

    save_state = {"n": len(rows), "t": time.perf_counter()}

    def _persist(force: bool = False) -> None:
        if not save:
            return
        from src.forecast.weekly_nowcast import (
            _SAVE_EVERY_ROWS, _SAVE_EVERY_SECONDS,
        )
        now = time.perf_counter()
        nuove = len(rows) - save_state["n"]
        if nuove <= 0:
            return
        if not (force or nuove >= _SAVE_EVERY_ROWS
                or now - save_state["t"] >= _SAVE_EVERY_SECONDS):
            return
        _atomic_write(rows, path)
        json.dump(firma_su_disco, open(_percorso_firma(path), "w",
                                       encoding="utf-8"), indent=1)
        save_state["n"], save_state["t"] = len(rows), now

    def _release() -> None:
        """Emette lavori finche' non incontra una ri-stima, poi si ferma."""
        nonlocal prossimo, in_volo
        while prossimo < len(piano):
            i = prossimo
            as_of, targets = piano[i]
            as_of_iso = str(as_of.date())
            if all((as_of_iso, q, spec, variant) in done for q in targets):
                per_settimana[i] = []      # gia' in CSV: niente da calcolare
                prossimo += 1
                continue
            mese = (as_of.year, as_of.month)
            due = (em_frequency == "weekly") or (mese != last_em_month)
            stima = due or (theta is None)
            (pronte_stime if stima else pronte_congelate).append(
                _Lavoro("stima" if stima else "congelata", i, theta))
            in_volo += 1
            prossimo += 1
            if stima:
                return      # BLOCCA: tutto il resto dipende da questo theta

    def _commit(i: int, esito) -> None:
        nonlocal theta, last_em_month, n_em
        as_of, _ = piano[i]
        as_of_iso = str(as_of.date())
        theta = esito.theta
        if esito.adottato:
            last_em_month = (as_of.year, as_of.month)
            n_em += 1
            if save:
                _save_theta(out_dir, as_of_iso, spec, variant, theta,
                            esito.n_iter, esito.converged, esito.origine)
        for exc in esito.errori:
            err_msgs.setdefault((spec, variant), Counter())[
                f"{type(exc).__name__}: {exc}"] += 1
        per_settimana[i] = esito.rows
        for riga, (q, status, dt) in zip(esito.rows, esito.stati):
            rows.append(riga)
            done.add((as_of_iso, q, spec, variant))
            _persist()
            if verbose:
                print(riga_di_stato(as_of_iso, q, as_of, riga, status, dt))

    def _argomenti(lav: _Lavoro) -> tuple:
        as_of, targets = piano[lav.i]
        as_of_iso = str(as_of.date())
        return (as_of, targets, spec, variant, lav.theta,
                lav.tipo == "stima", n_rel[lav.i], max_iter, verbose_em,
                frozenset(q for q in targets
                          if (as_of_iso, q, spec, variant) in done))

    try:
        _release()
        if workers == 1:
            # LA VIA SEQUENZIALE, il riferimento.  Stesse funzioni, stesso
            # ordine di `run_weekly_nowcast`: se questa e la parallela
            # divergono, e' lo scheduling, e non c'e' altro posto dove cercare.
            _init_worker()
            while pronte_stime or pronte_congelate:
                lav = (pronte_stime.popleft() if pronte_stime
                       else pronte_congelate.popleft())
                in_volo -= 1
                _commit(lav.i, _lavora(*_argomenti(lav)))
                if lav.tipo == "stima":
                    _release()
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker) as pool:
                futuri: dict = {}
                while pronte_stime or pronte_congelate or futuri:
                    # LE STIME PER PRIME: sono il cammino critico, e una stima
                    # che aspetta un posto libero allunga la catena di tutti.
                    while len(futuri) < workers and (pronte_stime
                                                     or pronte_congelate):
                        lav = (pronte_stime.popleft() if pronte_stime
                               else pronte_congelate.popleft())
                        futuri[pool.submit(_lavora, *_argomenti(lav))] = lav
                    finiti, _ = wait(futuri, return_when=FIRST_COMPLETED)
                    for f in finiti:
                        lav = futuri.pop(f)
                        in_volo -= 1
                        _commit(lav.i, f.result())
                        if lav.tipo == "stima":
                            _release()
    finally:
        _persist(force=True)

    mancano = sorted(set(range(len(piano))) - set(per_settimana))
    if mancano:
        raise RuntimeError(
            f"{spec}/{variant}: {len(mancano)} settimane non eseguite "
            f"(la prima e' {piano[mancano[0]][0].date()}).")

    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.sort_values(list(_KEY), kind="stable").reset_index(drop=True)
    if save:
        _atomic_write(rows, path)
        json.dump(firma_su_disco, open(_percorso_firma(path), "w",
                                       encoding="utf-8"), indent=1)
    dt = time.perf_counter() - t_start
    if verbose:
        print(f"\n  {len(df)} righe, {df['as_of'].nunique()} venerdi', "
              f"{n_em} stime EM in {dt:.1f}s ({dt / 60:.1f} min)")
        report_health(df, err_msgs)
    return df


__all__ = ["run_cell_pipeline", "firma_corsa", "WORKER_UTILI"]
