"""
src/forecast/test_resume.py

LA PROVA DELLA RIPRESA: si uccide il processo a meta' e si controlla che il
rilancio riprenda pulito.

    python -m src.forecast.test_resume

Non e' un test simbolico: lancia `weekly_nowcast` come SOTTOPROCESSO vero, lo
ammazza con `kill()` (SIGKILL su POSIX, TerminateProcess su Windows — nessun
`finally`, nessun flush di cortesia: il caso peggiore), poi rilancia lo stesso
comando e verifica tre cose sul CSV finale:

  1. NESSUN DUPLICATO sulla chiave (as_of, target_quarter, spec, variant);
  2. NESSUNA RIGA PERSA rispetto a una passata pulita di riferimento;
  3. VALORI COERENTI col riferimento entro `_TOL_BEA`.

PERCHE' LA (3) E' UNA TOLLERANZA E NON UN'UGUAGLIANZA — misurato, non assunto.
Alla ripresa la prima settimana da calcolare non ha un theta in memoria e
quindi RI-STIMA invece di riusare quello congelato (`reestimated` passa da
False a True, `n_iter` da 0 a ~34).  E' la regola voluta: non si eredita mai un
theta che non si e' prodotto in questa sessione, perche' ereditarlo darebbe una
irriproducibilita' peggiore.  Da quel punto la catena dei theta diverge di
poco e tutte le righe successive si spostano.

Misura su questa finestra: scarto massimo 2.4e-03 punti BEA su nowcast fra +1
e -8, cioe' tre ordini di grandezza sotto qualunque differenza che conti (le
differenze di RMSE fra metodi valgono 0.1-1.0).  `_TOL_BEA` e' tarata li' in
mezzo: abbastanza larga da non rompersi sulla divergenza di innesco, abbastanza
stretta da intercettare una riga corrotta, che sbaglierebbe di punti interi.

Un CSV troncato a meta' riga passerebbe il punto (1) e cadrebbe qui: e' questo
che giustifica `_atomic_write` con `os.replace`.

Gira su una finestra CORTA con una cella sola (`diag3/gaussian`, la piu'
rapida) — la meccanica della ripresa non dipende dalla lunghezza, e il test
deve poter girare in un paio di minuti.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pandas as pd

from src import output_layout as layout
from src.forecast.weekly_nowcast import _KEY, COLUMNS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Finestra di prova: corta, ma abbastanza da avere piu' di un salvataggio.
_START, _END = "2008-01-01", "2008-06-30"

#: Dove scrive il test.  MAI fra gli artefatti veri: una passata di prova che
#: sovrascrive il CSV della run buona e' un modo silenzioso di perdere ore.
_OUT = os.path.join(_PROJECT_ROOT, "output", "_test_resume")


#: Soglie di salvataggio abbassate SOLO per il test: con i 100 righe / 120 s
#: di produzione, una finestra corta finisce prima di salvare anche una volta
#: e il kill troverebbe il disco vuoto — la ripresa ripartirebbe da zero senza
#: mai esercitare il caso che interessa (CSV parziale + completamento).
_ENV = {"WEEKLY_SAVE_EVERY_ROWS": "10", "WEEKLY_SAVE_EVERY_SECONDS": "3"}

#: Scarto massimo tollerato fra passata pulita e ripresa, in punti BEA.  Vedi
#: il docstring: la divergenza vera misurata e' ~2.4e-03, le differenze che
#: contano valgono 0.1-1.0, una riga corrotta sbaglierebbe di punti interi.
_TOL_BEA = 0.05


def _cmd() -> list[str]:
    return [sys.executable, "-m", "src.forecast.weekly_nowcast",
            "--start", _START, "--end", _END,
            "--spec", "diag3", "--variant", "gaussian",
            "--no-benchmarks", "--output-dir", _OUT]


def _env() -> dict:
    return {**os.environ, **_ENV}


def _csv() -> str:
    return os.path.join(_OUT, f"weekly_nowcast_{_START}_{_END}.csv")


def _read() -> pd.DataFrame:
    return pd.read_csv(_csv()) if os.path.exists(_csv()) else pd.DataFrame(columns=COLUMNS)


def _run(tag: str) -> None:
    print(f"  [{tag}] {' '.join(_cmd()[2:])}")
    subprocess.run(_cmd(), cwd=_PROJECT_ROOT, check=True, env=_env(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_and_kill(seconds: float) -> int:
    """Lancia e ammazza dopo `seconds`.  Ritorna quante righe erano su disco."""
    p = subprocess.Popen(_cmd(), cwd=_PROJECT_ROOT, env=_env(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Si aspetta che il CSV parziale ESISTA prima di uccidere: e' l'unico modo
    # di garantire che il test provi la ripresa da un file a meta' invece che
    # da zero.  Se dopo `seconds` non c'e' ancora nulla, si ammazza lo stesso e
    # il test lo dichiara, invece di fingere di aver provato qualcosa.
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if p.poll() is not None:
            print("  [kill] il processo e' finito da solo prima del kill")
            return len(_read())
        if len(_read()):
            break
        time.sleep(0.3)
    if p.poll() is not None:
        print("  [kill] il processo e' finito da solo prima del kill")
        return len(_read())
    p.kill()                      # brutale: nessun finally, nessun flush
    p.wait()
    n = len(_read())
    print(f"  [kill] ucciso con {n} righe sul disco")
    return n


def check() -> int:
    import shutil
    shutil.rmtree(_OUT, ignore_errors=True)
    os.makedirs(_OUT, exist_ok=True)

    # ── Riferimento: una passata pulita, senza interruzioni ───────────────────
    print("1. passata di riferimento (pulita)")
    t0 = time.perf_counter()
    _run("ref")
    ref = _read().sort_values(list(_KEY)).reset_index(drop=True)
    dt = time.perf_counter() - t0
    print(f"   {len(ref)} righe in {dt:.0f}s\n")
    if ref.empty:
        print("   nessuna riga: test inconcludente")
        return 1

    # ── Interrotta + ripresa ──────────────────────────────────────────────────
    shutil.rmtree(_OUT, ignore_errors=True)
    os.makedirs(_OUT, exist_ok=True)
    print(f"2. passata interrotta a meta' (~{dt / 2:.0f}s) e ripresa")
    n_partial = _run_and_kill(max(dt / 2.0, 5.0))
    _run("resume")
    got = _read().sort_values(list(_KEY)).reset_index(drop=True)
    print(f"   {len(got)} righe dopo la ripresa\n")

    # ── I tre controlli ───────────────────────────────────────────────────────
    fails = 0

    dup = got.duplicated(subset=list(_KEY)).sum()
    ok = dup == 0
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  nessun duplicato sulla chiave "
          f"({dup} trovati)")

    kref = set(map(tuple, ref[list(_KEY)].astype(str).to_numpy()))
    kgot = set(map(tuple, got[list(_KEY)].astype(str).to_numpy()))
    ok = kref == kgot
    fails += not ok
    print(f"  {'OK ' if ok else 'ROTTA'}  nessuna riga persa "
          f"(mancano {len(kref - kgot)}, in piu' {len(kgot - kref)})")

    if kref == kgot:
        # Il realizzato NON dipende dal modello: quello deve essere identico
        # bit per bit, e se non lo fosse sarebbe corruzione, non innesco.
        exact = float((got["realizzato_bea"].to_numpy(float)
                       - ref["realizzato_bea"].to_numpy(float)).__abs__().max())
        ok = exact < 1e-9
        fails += not ok
        print(f"  {'OK ' if ok else 'ROTTA'}  realizzato identico bit per bit "
              f"(scarto {exact:.2e})")

        num = ["nowcast_bea", "nowcast_z"]
        d = (got[num].to_numpy(float) - ref[num].to_numpy(float))
        worst = float(pd.DataFrame(d).abs().max().max())
        n_diff = int((pd.DataFrame(d).abs() > 1e-12).any(axis=1).sum())
        ok = worst < _TOL_BEA
        fails += not ok
        print(f"  {'OK ' if ok else 'ROTTA'}  nowcast entro tolleranza "
              f"{_TOL_BEA} (scarto max {worst:.2e} su {n_diff}/{len(got)} righe)")
        print(f"           divergenza attesa: alla ripresa la prima settimana "
              f"ri-stima invece di\n           riusare il theta congelato — "
              f"vedi il docstring del modulo")
    else:
        print("  [salto]  confronto dei valori: le chiavi non coincidono")
        fails += 1

    # Il test vale solo se il kill ha davvero trovato un CSV a meta': con zero
    # righe sul disco si sarebbe verificata una passata pulita travestita da
    # ripresa.  Si dichiara rotto invece di passare per finta.
    partial = 0 < n_partial < len(ref)
    fails += not partial
    print(f"\n  {'OK ' if partial else 'ROTTA'}  il kill ha trovato un CSV "
          f"PARZIALE: {n_partial} righe su {len(ref)}"
          + ("" if partial else
             "   <- ripresa non esercitata: il test non prova niente"))
    shutil.rmtree(_OUT, ignore_errors=True)
    return fails


if __name__ == "__main__":
    n = check()
    print("\nRIPRESA OK" if not n else f"\n{n} CONTROLLI ROTTI")
    sys.exit(1 if n else 0)
