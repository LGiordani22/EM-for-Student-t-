"""
src/forecast/weekly_nowcast.py

Il CICLO SETTIMANALE: per ogni venerdi', per ogni cella, per ogni trimestre in
volo, un nowcast del PIL.

LA GRIGLIA
----------
`as_of` avanza di sette giorni per volta (venerdi': chiude la settimana
lavorativa, quindi tutto cio' che e' uscito nella settimana e' dentro).  A ogni
data, `release_calendar` dice quali serie erano pubblicate e quali trimestri non
avevano ancora il loro PIL: quelli sono i bersagli, tipicamente tre vivi insieme
— precedente, corrente e prossimo.

Fra un venerdi' e l'altro il pannello cambia solo dove e' uscito qualcosa.  E'
per questo che le traiettorie della figura 8a sono a gradini: piatte fra due
rilasci, poi saltano.  Il gradino non e' un artefatto da lisciare, e' il segnale.

QUANDO SI RI-STIMA (--em-frequency)
-----------------------------------
Ri-stimare l'EM ogni settimana su quindici celle costa ~170 ore per cinque anni
di storia.  Il default e' quindi `monthly`: l'EM gira sul primo venerdi' di ogni
mese, e nelle settimane intermedie i parametri restano congelati mentre lo stato
viene ricalcolato sull'informazione nuova (`nowcast_engine.filter_only`).  I
parametri di un fattore dinamico si muovono lentamente; lo stato no, ed e' lo
stato a portare la notizia.

    --em-frequency monthly   default, sviluppo e debug
    --em-frequency weekly    ri-stima ogni settimana, per la run finale

Quando si ri-stima, il punto di partenza e' il theta del vintage PRECEDENTE
invece dell'inizializzazione PCA.  Non e' look-ahead: e' lo stesso ottimo con un
innesco migliore, e nessuna informazione futura entra.  (Partire dal theta di
campione pieno lo sarebbe, e infatti non si fa.)

RIPRESA
-------
Ogni nowcast e' identificato da (as_of, target_quarter, spec, variant).  Il CSV
del periodo viene riscritto in modo atomico a BLOCCHI — ogni 100 righe o 120
secondi, quello che viene prima — piu' un salvataggio forzato alla fine di ogni
cella e all'uscita, anche se l'uscita e' un'interruzione.

Non dopo OGNI riga, come faceva prima: `_atomic_write` riscrive il file intero,
quindi salvare a ogni riga costa O(n^2).  Su 2007-2025 sono ~38 000 righe per
~7,6 MB, cioe' centinaia di GB di I/O — e su una cartella sincronizzata ogni
riscrittura e' anche un evento di sync.

Rilanciare lo stesso comando riprende da dove si era fermato.  La ripresa e'
corretta anche se il processo muore FRA due salvataggi: `done` si ricostruisce
leggendo il CSV, quindi le righe non ancora scritte semplicemente non risultano
fatte e vengono ricalcolate.  Nessun buco (chi manca si rifa'), nessun duplicato
(la chiave e' unica).  Si perde al piu' il lavoro di un blocco.

La catena di theta e la ripresa convivono: se una settimana da calcolare arriva
senza un theta in memoria (perche' le precedenti erano gia' in cache), quella
settimana ri-stima invece di riusare.  Non si eredita mai un theta che non si e'
prodotto in questa sessione.

  CONSEGUENZA DA DICHIARARE: una run RIPRESA non e' bit-identica a una run
  pulita.  Nel punto di ripresa `reestimated` passa da False a True e da li' la
  catena dei theta diverge.  Misurato su 2008-01/2008-06, cella diag3/gaussian:
  scarto massimo 2.4e-03 punti BEA su 53 righe di 60 — tre ordini di grandezza
  sotto le differenze di RMSE fra metodi (0.1-1.0), quindi irrilevante per le
  conclusioni ma non zero.  Il realizzato, che non dipende dal modello, resta
  identico bit per bit.  Lo verifica `src/forecast/test_resume.py`.

UN CSV PER PERIODO, UN PROCESSO PER CSV
---------------------------------------
Ogni processo riscrive il file INTERO dalla propria copia in memoria, e il nome
del file dipende SOLO dal periodo (`weekly_nowcast_<inizio>_<fine>.csv`).  La
regola percio' non e' "mai due celle insieme": e' che DUE PROCESSI NON DEVONO
MAI CONDIVIDERE LO STESSO FILE.  Da qui due modi leciti di parallelizzare:

  - sul PERIODO, celle in sequenza dentro ogni processo: i file sono gia'
    distinti perche' il periodo e' diverso;
  - sulla CELLA sullo stesso periodo, dando a ciascuna un `--output-dir` suo:
    i file tornano distinti perche' e' distinta la cartella.  E' come lancia
    `scripts/run_all.sh`, e in piu' tiene la catena dei theta continua su
    tutto lo span invece di spezzarla a ogni confine di blocco.

Quello che NON si puo' fare e' due celle sullo stesso periodo E sulla stessa
cartella: li' si sovrascrivono, e l'ultimo a scrivere fa sparire il lavoro
dell'altro.

Uso
---
  python -m src.forecast.weekly_nowcast --start 2008-01-01 --end 2009-12-31 \\
      --spec diag3 --variant student_t
  python -m src.forecast.weekly_nowcast --start 2008-01-01 --end 2009-12-31 \\
      --all-specs --all-variants --em-frequency weekly
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.forecast import scale
from src.forecast.benchmarks import BENCHMARKS, BENCHMARK_SPEC
from src.forecast.nowcast_engine import SPECS, VARIANTS, nowcast
from src.forecast.release_calendar import (
    gdp_release_date,
    horizon_week,
    load_metadata,
    load_panel,
    quarter_end,
    releases_between,
    targets_in_flight,
    weekly_grid,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMNS = [
    "as_of", "target_quarter", "horizon_week", "spec", "variant",
    "nowcast_bea", "nowcast_livello", "nowcast_z", "sd_z",
    "realizzato_bea", "realizzato_livello", "errore_bea",
    "gdp_release_date", "n_releases", "n_iter", "converged", "reestimated",
]

#: Chiave d'identita' di un nowcast.
_KEY = ("as_of", "target_quarter", "spec", "variant")

#: Ogni quante righe, e ogni quanti secondi, si riscrive il CSV.  Vedi il
#: blocco "Il salvataggio: a BLOCCHI" in `run_weekly` per il perche' servono
#: entrambi i limiti e perche' la ripresa resta corretta.
#:
#: Sovrascrivibili da ambiente: servono al test di ripresa, che deve provocare
#: un salvataggio parziale in pochi secondi, e danno una manopola sul server
#: senza toccare il codice (disco lento -> alzarli; run fragile -> abbassarli).
_SAVE_EVERY_ROWS = int(os.environ.get("WEEKLY_SAVE_EVERY_ROWS", "100"))
_SAVE_EVERY_SECONDS = float(os.environ.get("WEEKLY_SAVE_EVERY_SECONDS", "120"))


# ─── Persistenza atomica + ripresa ────────────────────────────────────────────

def _csv_path(out_dir: str, start: str, end: str) -> str:
    return os.path.join(out_dir, f"weekly_nowcast_{start}_{end}.csv")


def _row_key(r: dict) -> tuple:
    return tuple(str(r[k]) for k in _KEY)


def _load_existing(path: str) -> list[dict]:
    """Rilegge un CSV di periodo.  Un file illeggibile degrada a 'riparto da zero'."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        print(f"  [attenzione] CSV esistente illeggibile ({type(exc).__name__}: {exc}); "
              f"riparto da zero: {path}")
        return []
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS].to_dict("records")


def _atomic_write(rows: list[dict], path: str, retries: int = 6,
                  delay: float = 0.7) -> bool:
    """
    Scrive su file temporaneo e poi `os.replace` (atomico su Windows): un file
    scritto a meta' non e' mai osservabile.  Il progetto sta su OneDrive, che
    puo' tenere il file bloccato per qualche istante: si riprova, e se proprio
    non si riesce si avvisa senza abortire — le righe restano in memoria e il
    prossimo salvataggio riscrive tutto.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.sort_values(list(_KEY), kind="stable").reset_index(drop=True)
    tmp = f"{path}.tmp"
    for attempt in range(1, retries + 1):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return True
        except (PermissionError, OSError) as exc:
            if attempt == retries:
                print(f"  [attenzione] scrittura fallita dopo {retries} tentativi "
                      f"({type(exc).__name__}: {exc}); riprovo al prossimo nowcast.")
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                return False
            time.sleep(delay)
    return False


# ─── Il realizzato, per il punteggio ──────────────────────────────────────────

def _realised(panel: pd.DataFrame, quarter: str, target_series: str) -> tuple[float, float]:
    """(livello, BEA) realizzato del trimestre, o (nan, nan) se non ancora uscito."""
    qe = quarter_end(quarter)
    if qe not in panel.index or pd.isna(panel.loc[qe, target_series]):
        return float("nan"), float("nan")
    lvl = float(panel.loc[qe, target_series])
    return lvl, float(scale.to_bea(lvl))


# ─── Il ciclo ─────────────────────────────────────────────────────────────────

def run_weekly_nowcast(
    start: str,
    end: str,
    specs: tuple[str, ...] = SPECS,
    variants: tuple[str, ...] = tuple(VARIANTS),
    em_frequency: str = "monthly",
    n_ahead: int = 1,
    max_iter: int = 250,
    benchmarks: bool = True,
    output_dir: str | None = None,
    save: bool = True,
    verbose_em: bool = False,
) -> pd.DataFrame:
    """
    Percorre [start, end] a passo settimanale su ogni cella spec x variante.

    Ogni cella e' una catena temporale a se': i suoi theta si passano di settimana
    in settimana, quindi le celle si percorrono una alla volta, dall'inizio alla
    fine, e non si mescolano.
    """
    if em_frequency not in ("weekly", "monthly"):
        raise ValueError(f"em_frequency {em_frequency!r}: attesi 'weekly' o 'monthly'.")

    from src import output_layout as _layout
    out_dir = output_dir or _layout.dfm_csv_dir()
    path = _csv_path(out_dir, start, end)

    panel, meta = load_panel(), load_metadata()
    target_series = list(panel.columns)[-1]
    grid = weekly_grid(start, end)
    if not grid:
        raise SystemExit(f"nessun venerdi' nell'intervallo {start} .. {end}.")

    rows: list[dict] = []
    done: set[tuple] = set()
    if save and os.path.exists(path):
        rows = _load_existing(path)
        done = {_row_key(r) for r in rows}

    # Quante righe ci si aspetta, per il conteggio e la ripresa.
    plan: list[tuple[pd.Timestamp, list[str]]] = [
        (d, targets_in_flight(d, n_ahead=n_ahead, target_series=target_series,
                              metadata=meta))
        for d in grid
    ]
    cells = [(s, v) for s in specs for v in variants]
    if benchmarks:
        cells += [(BENCHMARK_SPEC, name) for name in BENCHMARKS]
    n_expected = sum(len(t) for _, t in plan) * len(cells)
    n_done = sum(
        1 for d, tgts in plan for (s, v) in cells for q in tgts
        if (str(d.date()), q, s, v) in done
    )

    print(f"  griglia    : {len(grid)} settimane, {grid[0].date()} .. {grid[-1].date()}")
    print(f"  celle      : {len(specs)} spec x {len(variants)} varianti = "
          f"{len(specs) * len(variants)}")
    print(f"  ri-stima   : {em_frequency}"
          + ("  (EM ogni settimana)" if em_frequency == "weekly"
             else "  (EM sul primo venerdi' del mese, poi solo stato)"))
    print(f"  nowcast    : {n_expected} previsti, {n_done} gia' fatti, "
          f"{n_expected - n_done} da calcolare")
    if save:
        print(f"  CSV        : {path}\n")

    # ── Il salvataggio: a BLOCCHI, non a ogni riga ────────────────────────────
    # `_atomic_write` riscrive il CSV INTERO.  Chiamarlo dopo ogni riga costa
    # O(n^2): su 2007-2025 sono ~38 000 righe per ~7,6 MB di file, cioe'
    # centinaia di GB di I/O — e su una cartella sincronizzata ogni riscrittura
    # e' anche un evento di sync.  Si salva quindi ogni `_SAVE_EVERY_ROWS`
    # righe OPPURE ogni `_SAVE_EVERY_SECONDS`, quello che viene prima.
    #
    # Il limite di TEMPO non e' ridondante: una singola riga con EM sotto
    # `student_t_ar1_shared` a fine campione costa ~280 s, quindi cento righe
    # possono valere ore.  Il tempo mette un tetto a quanto lavoro si perde.
    #
    # PERCHE' E' SICURO PER LA RIPRESA: `done` viene ricostruito da `_load_existing`
    # sul CSV, non dalla memoria.  Le righe calcolate dopo l'ultimo salvataggio
    # non sono nel file, quindi alla ripresa non sono in `done` e vengono
    # ricalcolate — nessun buco.  E la chiave `_KEY` e' unica, quindi non
    # possono neanche finirci due volte — nessun duplicato.  Si perde al piu'
    # il lavoro di un blocco, mai la coerenza.
    save_state = {"n": len(rows), "t": time.perf_counter()}

    def _persist(force: bool = False):
        if not save:
            return
        now = time.perf_counter()
        new = len(rows) - save_state["n"]
        if new <= 0:
            return          # niente di nuovo: riscrivere sarebbe I/O a vuoto
        due = (force
               or new >= _SAVE_EVERY_ROWS
               or now - save_state["t"] >= _SAVE_EVERY_SECONDS)
        if not due:
            return
        _atomic_write(rows, path)
        save_state["n"], save_state["t"] = len(rows), now

    n_em = 0
    t_start = time.perf_counter()

    # Un'interruzione (Ctrl-C, o il gestore di coda del server) non deve
    # buttare via il blocco in corso: il `finally` scrive quel che c'e' in
    # memoria prima di uscire.  Non e' cio' che rende corretta la RIPRESA —
    # quella regge anche su un kill brutale, perche' `done` si ricostruisce
    # dal CSV — ma evita di rifare fino a due minuti di EM per una fermata
    # volontaria.
    try:
        # ── I benchmark ───────────────────────────────────────────────────────────
        # Non dipendono da spec ne' da variante: si calcolano una volta per
        # (settimana, target), non quindici.  Vivono nello stesso CSV sotto la
        # pseudo-spec "benchmark", cosi' metriche e figure li leggono senza sapere
        # che vengono da un'altra parte.
        if benchmarks:
            print(f"\n{'=' * 78}\n  benchmark: {', '.join(BENCHMARKS)}\n{'=' * 78}")
            prev_week = None
            for as_of, targets in plan:
                as_of_iso = str(as_of.date())
                n_rel = (len(releases_between(prev_week, as_of, panel, meta))
                         if prev_week is not None else np.nan)
                prev_week = as_of
                for q in targets:
                    real_lvl, real_bea = _realised(panel, q, target_series)
                    for name, fn in BENCHMARKS.items():
                        key = (as_of_iso, q, BENCHMARK_SPEC, name)
                        if key in done:
                            continue
                        try:
                            b = fn(as_of, q, target_series=target_series,
                                   panel=panel, metadata=meta)
                            bea, liv, z = b["nowcast_bea"], b["nowcast_livello"], b["nowcast_z"]
                            conv = b["converged"]
                        except Exception as exc:
                            bea = liv = z = float("nan")
                            conv = False
                            print(f"  [errore] {name} {as_of_iso} {q}: "
                                  f"{type(exc).__name__}: {exc}")
                        rows.append({
                            "as_of": as_of_iso, "target_quarter": q,
                            "horizon_week": horizon_week(as_of, q),
                            "spec": BENCHMARK_SPEC, "variant": name,
                            "nowcast_bea": bea, "nowcast_livello": liv,
                            "nowcast_z": z, "sd_z": float("nan"),
                            "realizzato_bea": real_bea, "realizzato_livello": real_lvl,
                            "errore_bea": bea - real_bea,
                            "gdp_release_date": str(gdp_release_date(
                                q, target_series, meta).date()),
                            "n_releases": n_rel, "n_iter": 0,
                            "converged": conv, "reestimated": False,
                        })
                        done.add(key)
            _persist(force=True)      # fine dei benchmark: confine naturale
            print(f"  {sum(1 for r in rows if r['spec'] == BENCHMARK_SPEC)} righe di benchmark")

        for spec in specs:
            for variant in variants:
                theta = None          # theta corrente della cella (None = da stimare)
                last_em_month = None
                prev_week = None

                print(f"\n{'=' * 78}\n  {spec} / {variant}\n{'=' * 78}")

                for as_of, targets in plan:
                    as_of_iso = str(as_of.date())
                    keys = [(as_of_iso, q, spec, variant) for q in targets]

                    # Quante pubblicazioni sono cadute dall'ultimo venerdi': e' cio'
                    # che puo' far muovere il nowcast, e spiega i gradini.
                    n_rel = (len(releases_between(prev_week, as_of, panel, meta))
                             if prev_week is not None else np.nan)
                    prev_week = as_of

                    if all(k in done for k in keys):
                        continue      # gia' in CSV: niente da calcolare

                    # Ri-stimare o riusare?  Si ri-stima quando la cadenza lo chiede,
                    # e comunque sempre quando non si ha un theta in mano (prima
                    # settimana della cella, o ripresa a meta' periodo).
                    month = (as_of.year, as_of.month)
                    due = (em_frequency == "weekly") or (month != last_em_month)
                    reestimate = due or (theta is None)

                    for q in targets:
                        key = (as_of_iso, q, spec, variant)
                        if key in done:
                            continue

                        t0 = time.perf_counter()
                        try:
                            r = nowcast(
                                as_of, q, spec, variant,
                                theta=(None if reestimate else theta),
                                theta_warm=(theta if reestimate else None),
                                max_iter=max_iter, verbose=verbose_em,
                                target_series=target_series,
                                panel=panel, metadata=meta,
                            )
                            theta = r["theta"]
                            if reestimate:
                                last_em_month = month
                                n_em += 1
                                reestimate = False   # gli altri target della stessa
                                #                      settimana riusano questo theta:
                                #                      stesso pannello, stessa stima
                            liv, bea, z = r["nowcast_livello"], r["nowcast_bea"], r["nowcast_z"]
                            sd, n_it = r["sd_z"], r["n_iter"]
                            conv, reest = r["converged"], r["reestimated"]
                            status = "OK" if conv else "non converge"
                        except Exception as exc:
                            # Ricaduta: se e' fallita la RI-STIMA EM ma abbiamo un
                            # theta da una stima precedente, filtriamo con quello
                            # invece di emettere NaN — la traiettoria resta continua
                            # e i parametri restano dell'ultima stima riuscita.  La
                            # ri-stima NON viene segnata come fatta (last_em_month
                            # invariato), quindi si ritenta la settimana dopo.
                            if reestimate and theta is not None:
                                try:
                                    r = nowcast(
                                        as_of, q, spec, variant, theta=theta,
                                        max_iter=max_iter, verbose=verbose_em,
                                        target_series=target_series,
                                        panel=panel, metadata=meta,
                                    )
                                    liv = r["nowcast_livello"]
                                    bea, z = r["nowcast_bea"], r["nowcast_z"]
                                    sd, n_it = r["sd_z"], r["n_iter"]
                                    conv, reest = r["converged"], False
                                    status = (f"EM fallito, filtro col theta "
                                              f"precedente ({type(exc).__name__})")
                                except Exception as exc2:
                                    liv = bea = z = sd = float("nan")
                                    n_it, conv, reest = -1, False, False
                                    status = f"ERRORE ({type(exc2).__name__}: {exc2})"
                            else:
                                liv = bea = z = sd = float("nan")
                                n_it, conv, reest = -1, False, False
                                status = f"ERRORE ({type(exc).__name__}: {exc})"

                        real_lvl, real_bea = _realised(panel, q, target_series)
                        rows.append({
                            "as_of": as_of_iso,
                            "target_quarter": q,
                            "horizon_week": horizon_week(as_of, q),
                            "spec": spec,
                            "variant": variant,
                            "nowcast_bea": bea,
                            "nowcast_livello": liv,
                            "nowcast_z": z,
                            "sd_z": sd,
                            "realizzato_bea": real_bea,
                            "realizzato_livello": real_lvl,
                            "errore_bea": bea - real_bea,
                            "gdp_release_date": str(gdp_release_date(
                                q, target_series, meta).date()),
                            "n_releases": n_rel,
                            "n_iter": n_it,
                            "converged": conv,
                            "reestimated": reest,
                        })
                        done.add(key)
                        _persist()

                        dt = time.perf_counter() - t0
                        flag = "EM " if reest else "   "
                        print(f"  {as_of_iso}  {q}  h{horizon_week(as_of, q):+3d}  "
                              f"{flag}{dt:6.1f}s  {status:<14} "
                              f"nowcast={bea:+7.3f}%  realizzato={real_bea:+7.3f}%")

                # Fine di una cella: confine naturale, e la piu' lunga distanza
                # possibile fra due salvataggi forzati.
                _persist(force=True)

    finally:
        _persist(force=True)

    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.sort_values(list(_KEY), kind="stable").reset_index(drop=True)
    if save:
        _atomic_write(rows, path)
        print(f"\n  salvate {len(df)} righe -> {path}")
    print(f"  {n_em} stime EM in {time.perf_counter() - t_start:.1f}s")
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Nowcast settimanale del PIL.")
    p.add_argument("--start", required=True, help="prima data 'YYYY-MM-DD'")
    p.add_argument("--end", required=True, help="ultima data 'YYYY-MM-DD'")
    p.add_argument("--spec", choices=SPECS, default=None)
    p.add_argument("--all-specs", action="store_true")
    p.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    p.add_argument("--all-variants", action="store_true")
    p.add_argument("--em-frequency", choices=["weekly", "monthly"], default="monthly",
                   help="quando ri-stimare l'EM (default: monthly)")
    p.add_argument("--n-ahead", type=int, default=1,
                   help="trimestri da prevedere oltre quello corrente (default: 1)")
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--no-benchmarks", action="store_true",
                   help="non calcolare ar2 e mean (sono gia' nel CSV)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose-em", action="store_true")
    a = p.parse_args()

    specs = SPECS if a.all_specs else ((a.spec,) if a.spec else (SPECS[0],))
    variants = (tuple(VARIANTS) if a.all_variants
                else ((a.variant,) if a.variant else ("gaussian",)))

    print("\n" + "=" * 78)
    print(f"  NOWCAST SETTIMANALE  {a.start} .. {a.end}")
    print("=" * 78)
    run_weekly_nowcast(
        a.start, a.end, specs=specs, variants=variants,
        em_frequency=a.em_frequency, n_ahead=a.n_ahead, max_iter=a.max_iter,
        benchmarks=not a.no_benchmarks,
        output_dir=a.output_dir, save=not a.no_save, verbose_em=a.verbose_em,
    )


if __name__ == "__main__":
    main()
