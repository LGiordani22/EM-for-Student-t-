"""
src/bvar/tests/test_resume.py

GATE 6 — LA PASSATA E' INTERROMPIBILE.  Il test della ripresa.

    python -m src.bvar.tests.test_resume

La passata 2007-2010 sui quattro modelli sta nell'ordine delle decine di ore.
In quel tempo il PC si riavvia, e la domanda a cui questo file risponde non e'
"la ripresa funziona?" ma la sola che conti davvero:

    UNA PASSATA RIPRESA DA' GLI STESSI IDENTICI NUMERI DI UNA NON INTERROTTA?

Se la risposta fosse no, il checkpoint sarebbe peggio di niente: produrrebbe
risultati che dipendono da QUANDO e' andata via la corrente, cioe' non
riproducibili, e non ci sarebbe modo di accorgersene guardando il CSV.  La
risposta e' si' per costruzione: il seme deriva dalla data assoluta della
settimana, non dal suo indice locale nella griglia. Ripresa e blocchi
paralleli usano quindi lo stesso stream della passata continua. I §1-§2
verificano la ripresa; il §0 verifica l'invarianza rispetto allo shard.

I QUATTRO TEST, che sono i quattro modi in cui il checkpoint puo' tradire:

  §1  IDENTITA'.  Passata intera contro passata interrotta+ripresa: i CSV
      devono coincidere cella per cella, non "essere simili".
  §2  NIENTE DOPPIONI.  La ripresa rilegge il CSV e ci riscrive dentro: una
      chiave duplicata passerebbe inosservata fino al calcolo delle metriche,
      dove pesa due volte.
  §3  LA CACHE SERVE A QUALCOSA.  Riprendere a meta' trimestre NON deve
      rifare la stima piena — e' l'unica ragione per cui la cache esiste.  Si
      contano le chiamate allo stimatore.  Poi si CANCELLA la cache e si
      pretende il contrario: il riavvolgimento deve scattare, altrimenti la
      ripresa userebbe parametri che non ha.
  §4  IL MANIFESTO.  Riprendere una passata con S diverso deve FERMARSI.  E'
      la guardia contro l'unico errore silenzioso di questo impianto: righe a
      S=100 e righe a S=1000 nello stesso CSV, che nessuna colonna distingue.

Il modello e' il Q-BVAR a S=20 su quattro settimane: qui non si valuta nulla,
si esercita la MECCANICA, e il modello piu' economico la esercita tutta.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pandas as pd

from src.bvar import evaluate

START, END = "2008-01-01", "2008-01-25"       # 4 venerdi', 1 sola stima piena
MODELS = ("qbvar",)
DRAWS = {"qbvar": 20}


class _Interrompi(RuntimeError):
    """L'interruzione simulata.  Non e' un errore del modello: e' la corrente."""


def _run(root: str, *, stop_after: int | None = None, fresh: bool = False,
         draws: dict | None = None) -> pd.DataFrame:
    """Una passata, eventualmente interrotta dopo `stop_after` settimane."""
    vero = evaluate.run_model
    stato = {"n": 0}

    def _spia(*a, **kw):
        stato["n"] += 1
        if stop_after is not None and stato["n"] > stop_after:
            raise _Interrompi
        return vero(*a, **kw)

    evaluate.run_model = _spia
    try:
        return evaluate.run_realtime(
            START, END, MODELS, output_root=root, verbose=False, fresh=fresh,
            n_draws=draws or DRAWS)
    finally:
        evaluate.run_model = vero


def _stima_fallita_alla_prima_settimana(root: str) -> tuple[pd.DataFrame, list]:
    """Una passata in cui la stima piena della PRIMA settimana non regge.

    E' il caso del blocco Covid: `parallel_blocks` taglia sulle settimane
    piene, quindi quando fallisce la prima settimana di un blocco la cache e'
    vuota per costruzione e non c'e' niente da ereditare.  Il rimedio e' la
    stima di RISCALDAMENTO: il blocco si ricalcola da solo l'ultima stima
    piena precedente, che sta fuori dal blocco.
    """
    vero = evaluate._full_estimate
    chiamate: list = []
    prima = pd.Timestamp(evaluate.weekly_grid(START, END)[0])

    def _spia(model, as_of, **kw):
        chiamate.append(pd.Timestamp(as_of))
        if pd.Timestamp(as_of) == prima:
            raise FloatingPointError("stima non convergente (simulata)")
        return vero(model, as_of, **kw)

    evaluate._full_estimate = _spia
    try:
        evaluate.run_realtime(START, END, MODELS, output_root=root,
                              verbose=False, fresh=True, n_draws=DRAWS)
    finally:
        evaluate._full_estimate = vero
    return _csv(root), chiamate


def _csv(root: str) -> pd.DataFrame:
    p = evaluate._paths(root, START, END)["csv"]
    return pd.read_csv(p).sort_values(list(evaluate._KEY)).reset_index(drop=True)


def _n_stime(root: str, **kw) -> int:
    """Quante STIME PIENE ha fatto una passata.  E' il §3."""
    vero = evaluate.run_model
    stato = {"n": 0}

    def _spia(model, as_of, quarters, cache, *, full, **kw2):
        if full or not cache.ready:
            stato["n"] += 1
        return vero(model, as_of, quarters, cache, full=full, **kw2)

    evaluate.run_model = _spia
    try:
        evaluate.run_realtime(START, END, MODELS, output_root=root,
                              verbose=False, n_draws=DRAWS, **kw)
    finally:
        evaluate.run_model = vero
    return stato["n"]


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="gate6_resume_")
    intero = os.path.join(tmp, "intero")
    rotto = os.path.join(tmp, "rotto")
    esiti = []
    try:
        # ── §0 il seed non dipende dal confine del blocco ────────────────────
        D = pd.Timestamp("2008-01-18")
        a = evaluate._rng_for_week(20260801, D).standard_normal(16)
        b = evaluate._rng_for_week(20260801, D).standard_normal(16)
        c = evaluate._rng_for_week(
            20260801, D + pd.Timedelta(days=7)).standard_normal(16)
        esiti.append(("0   seed stabile per data, non per shard",
                      bool((a == b).all() and not (a == c).all()),
                      "stessa data = stesso stream; data diversa = stream diverso"))

        print("  passata di riferimento (non interrotta) ...")
        _run(intero)
        rif = _csv(intero)

        print("  passata interrotta dopo 2 settimane ...")
        try:
            _run(rotto, stop_after=2)
        except _Interrompi:
            pass
        else:
            raise AssertionError("l'interruzione simulata non e' scattata")
        parziale = _csv(rotto)
        assert len(parziale) < len(rif), "l'interruzione non ha tolto niente"

        # ── §3 la cache serve: la ripresa non rifa' la stima piena ────────────
        print("  ripresa ...")
        n_stime = _n_stime(rotto)
        esiti.append(("3   la ripresa NON rifa' la stima piena", n_stime == 0,
                      f"{n_stime} stime piene (attese 0)"))
        ripreso = _csv(rotto)

        # ── §1 identita' ─────────────────────────────────────────────────────
        stessa_forma = ripreso.shape == rif.shape
        uguali = stessa_forma and ripreso.equals(rif)
        if stessa_forma and not uguali:
            # dove differiscono, per poterlo dire invece di limitarsi a "no"
            diff = (ripreso != rif) & ~(ripreso.isna() & rif.isna())
            cols = [c for c in diff.columns if diff[c].any()]
            det = f"differiscono le colonne {cols}"
        else:
            det = (f"{ripreso.shape} vs {rif.shape}" if not stessa_forma
                   else "identici cella per cella")
        esiti.append(("1   ripresa == passata intera", uguali, det))

        # ── §2 niente doppioni ───────────────────────────────────────────────
        dup = int(ripreso.duplicated(subset=list(evaluate._KEY)).sum())
        esiti.append(("2   nessuna riga in doppio", dup == 0,
                      f"{dup} chiavi duplicate"))

        # ── §3b senza cache si RIAVVOLGE ─────────────────────────────────────
        # Si torna allo stato "interrotto" e si toglie la cache: la stima piena
        # deve tornare, altrimenti si starebbe riusando una cache che non c'e'.
        shutil.rmtree(rotto)
        try:
            _run(rotto, stop_after=2)
        except _Interrompi:
            pass
        os.remove(evaluate._cache_path(evaluate._paths(rotto, START, END), "qbvar"))
        n_stime2 = _n_stime(rotto)
        esiti.append(("3b senza cache si riavvolge alla stima", n_stime2 == 1,
                      f"{n_stime2} stime piene (attesa 1)"))
        esiti.append(("3b il riavvolgimento non cambia i numeri",
                      _csv(rotto).equals(rif), "confronto col riferimento"))

        # ── §4 il manifesto ferma le passate diverse ─────────────────────────
        try:
            _run(rotto, draws={"qbvar": 21})
        except SystemExit:
            ok4, det4 = True, "SystemExit, come deve"
        except Exception as exc:
            ok4, det4 = False, f"{type(exc).__name__} invece di SystemExit"
        else:
            ok4, det4 = False, "la passata e' proseguita con S diverso"
        esiti.append(("4   S diverso -> la ripresa si ferma", ok4, det4))

        # ── §5 una stima che fallisce non porta giu' il blocco ───────────────
        # E' il caso Covid, esercitato sulla meccanica invece che sulle sei ore
        # e mezza del blocco vero: la stima piena della PRIMA settimana non
        # regge, la cache e' vuota, e il blocco deve procurarsi i parametri da
        # solo con la stima di riscaldamento.
        print("  stima fallita alla prima settimana ...")
        caduto = os.path.join(tmp, "caduto")
        try:
            df5, chiamate = _stima_fallita_alla_prima_settimana(caduto)
            ok5, det5 = True, ""
        except Exception as exc:                                   # noqa: BLE001
            df5, chiamate = None, []
            ok5, det5 = False, f"{type(exc).__name__}: {exc}"
        esiti.append(("5   una stima fallita non uccide il blocco", ok5,
                      det5 or f"{len(df5)} righe scritte, blocco completo"))

        if df5 is not None:
            # tutte le settimane ci sono, come nella passata sana
            esiti.append(("5b  nessuna settimana persa",
                          set(df5["as_of"]) == set(rif["as_of"]),
                          f"{df5['as_of'].nunique()} settimane "
                          f"(attese {rif['as_of'].nunique()})"))
            # e la riga DICHIARA di non essere stata ristimata
            esiti.append(("5c  le righe dichiarano il riuso",
                          not df5["reestimated"].any(),
                          "reestimated=False su tutte, come dev'essere quando "
                          "l'unica stima piena della finestra e' fallita"))
            # il riscaldamento e' andato a prendere una data PRECEDENTE
            prima = pd.Timestamp(evaluate.weekly_grid(START, END)[0])
            riscaldamento = [d for d in chiamate if d < prima]
            esiti.append(("5d  la stima di riscaldamento e' fuori dal blocco",
                          len(riscaldamento) == 1
                          and riscaldamento[0] == evaluate.previous_full_week(prima),
                          f"{[str(d.date()) for d in riscaldamento]} "
                          f"(prima settimana del blocco: {prima.date()})"))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    for nome, ok, det in esiti:
        print(f"  {'OK  ' if ok else 'ROTTO'}  {nome:45s}  {det}")
    if not all(ok for _, ok, _ in esiti):
        raise SystemExit("\n  Il checkpoint NON e' affidabile: non lanciare la passata.")
    print("\n  La passata si puo' interrompere: riprenderla da' gli stessi numeri.")


if __name__ == "__main__":
    main()
