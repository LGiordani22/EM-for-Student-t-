"""
core/check_final_artifacts.py

Verifica di coerenza delle 15 celle di `data/processed/final/<spec>/<variante>/`.

PERCHE' ESISTE
--------------
`run_final_artifacts.run_one` scrive `theta_initial.npz` **prima** di lanciare
`fit_dfm`, e `fit_dfm_result.npz` **dopo**. Fra i due momenti possono passare
minuti (una cella `student_t_ar1` ne costa un paio), e qualunque interruzione
in mezzo — Ctrl-C, crash, kill — lascia nella stessa cartella un theta^(0)
NUOVO accanto a un fit VECCHIO.

Il punto delicato e' che quella mescolanza e' **invisibile**: nessuno dei due
file e' corrotto, entrambi si aprono, e la cartella sembra a posto. E' gia'
successo il 2026-07-19 su `fed_overlap/student_t` (theta^(0) delle 11:56,
fit delle 16:30 del giorno prima).

Questo script non previene il disallineamento — lo rende **visibile**. E' la
stessa regola che vale per i self-test: gli artefatti veri devono essere
leggibili dall'esterno, e un risultato incoerente deve dichiararsi tale invece
di passare per buono.

COSA CONTROLLA, per ogni cella
------------------------------
1. **Presenza**    — i 4 file attesi ci sono? (la cartella vuota conta come
                     cella assente, non come cella rotta: `resolve_final_path`
                     crea directory come effetto collaterale)
2. **Ordine**      — `fit_dfm_result` e' piu' recente di `theta_initial`?
                     Se no, il fit non appartiene a quel theta^(0).
3. **Etichette**   — `spec` e `variant` dentro i JSON corrispondono al
                     PERCORSO in cui il file si trova?
4. **Convergenza** — `converged` e' vero, e l'ELBO e' monotono?
                     Una cella non convergiuta non e' un errore di coerenza,
                     ma non va confusa con una convergiuta.

USO
---
    python core/check_final_artifacts.py
    python core/check_final_artifacts.py --spec diag3
    python core/check_final_artifacts.py --quiet     # solo i problemi

Esce con codice 1 se almeno una cella e' INCOERENTE (non se e' assente: una
cella non ancora calcolata e' uno stato legittimo).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FINAL = os.path.join(_ROOT, "data", "processed", "final")

SPECS = ("fed_overlap", "diag4", "diag3")
VARIANTS = ("gaussian", "gaussian_ar1", "student_t", "student_t_ar1",
            "student_t_ar1_shared")

_THETA_JSON = "theta_initial_metadata.json"
_THETA_NPZ = "theta_initial.npz"
_FIT_JSON = "fit_dfm_result.json"
_FIT_NPZ = "fit_dfm_result.npz"

# Esiti. "assente" e "non convergiuta" NON sono incoerenze: sono stati leciti
# che vanno solo distinti da "tutto a posto".
OK, ASSENTE, PARZIALE, INCOERENTE, DA_GUARDARE = (
    "ok", "assente", "parziale", "INCOERENTE", "da guardare")


def _mtime(path: str) -> float | None:
    return os.path.getmtime(path) if os.path.isfile(path) else None


def _hhmm(ts: float | None) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "--"


def check_cell(spec: str, variant: str) -> dict:
    """Controlla una cella. Restituisce esito + motivo + qualche dato utile."""
    d = os.path.join(_FINAL, spec, variant)
    paths = {n: os.path.join(d, n)
             for n in (_THETA_JSON, _THETA_NPZ, _FIT_JSON, _FIT_NPZ)}
    present = {n: os.path.isfile(p) for n, p in paths.items()}

    out = {"spec": spec, "variant": variant, "dir": d,
           "t_theta": _mtime(paths[_THETA_JSON]),
           "t_fit": _mtime(paths[_FIT_JSON]),
           "n_iter": None, "loglik": None, "converged": None}

    if not any(present.values()):
        # Cartella inesistente o vuota: cella semplicemente non calcolata.
        out["esito"], out["motivo"] = ASSENTE, "nessun artefatto"
        return out

    mancanti = [n for n, ok in present.items() if not ok]
    if mancanti:
        # Un fit senza theta^(0) (o viceversa) e' un mezzo risultato: va
        # segnalato, ma se manca SOLO il fit e' verosimilmente una run in corso.
        solo_fit_manca = set(mancanti) <= {_FIT_JSON, _FIT_NPZ}
        out["esito"] = PARZIALE if solo_fit_manca else INCOERENTE
        out["motivo"] = "manca " + ", ".join(sorted(mancanti))
        return out

    # ── Ordine temporale: il fit deve venire DOPO il theta^(0) che lo genera ──
    if out["t_fit"] < out["t_theta"]:
        delta = (out["t_theta"] - out["t_fit"]) / 60.0
        out["esito"] = INCOERENTE
        out["motivo"] = (f"fit precede theta^(0) di {delta:.0f} min "
                         f"-> il fit non appartiene a questo theta^(0)")
        return out

    # ── Etichette: il contenuto dichiara la cella in cui si trova? ───────────
    try:
        with open(paths[_FIT_JSON], encoding="utf-8") as fh:
            fit = json.load(fh)
        with open(paths[_THETA_JSON], encoding="utf-8") as fh:
            th0 = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        out["esito"], out["motivo"] = INCOERENTE, f"JSON illeggibile: {exc}"
        return out

    out["n_iter"] = fit.get("n_iter")
    out["loglik"] = fit.get("loglik_last")
    out["converged"] = fit.get("converged")

    for nome, blob in ((_FIT_JSON, fit), (_THETA_JSON, th0)):
        for campo, atteso in (("spec", spec), ("variant", variant)):
            got = blob.get(campo)
            if got is not None and got != atteso:
                out["esito"] = INCOERENTE
                out["motivo"] = (f"{nome} dichiara {campo}={got!r} "
                                 f"ma sta in .../{spec}/{variant}/")
                return out

    # ── Qualita' della stima (non e' incoerenza, ma non va confusa con OK) ───
    if fit.get("converged") is False:
        out["esito"] = DA_GUARDARE
        out["motivo"] = f"NON convergiuta dopo {fit.get('n_iter')} iterazioni"
        return out
    if fit.get("elbo_monotone") is False:
        # ATTENZIONE: qui convivono DUE controlli con soglie diverse, e vanno
        # riportati insieme o si leggono come una contraddizione.
        #   `monotonicity_violations` (da fit_dfm) usa una soglia RELATIVA,
        #       tol_outer * |L| — su |L| ~ 9700 vale ~0.097;
        #   `elbo_monotone` (da run_final_artifacts) usa una soglia ASSOLUTA,
        #       1e-6, insensibile alla scala di L.
        # Un calo di 0.02 sta in mezzo: zero violazioni per il primo, non
        # monotono per il secondo. Non e' un bug di nessuno dei due, e' una
        # differenza di convenzione — ma va detta, altrimenti "ELBO non
        # monotono (0 violazioni)" sembra un errore del check.
        md = fit.get("elbo_min_delta")
        md_s = f"{md:.4g}" if isinstance(md, (int, float)) else "?"
        out["esito"] = DA_GUARDARE
        out["motivo"] = (f"convergiuta, ma l'ELBO cala di {md_s} in un passo "
                         f"(soglia assoluta 1e-6; sotto la soglia relativa di "
                         f"fit_dfm, che infatti segna 0 violazioni)")
        return out

    out["esito"], out["motivo"] = OK, ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Coerenza degli artefatti in data/processed/final/.")
    ap.add_argument("--spec", choices=SPECS, default=None,
                    help="controlla una sola spec (default: tutte)")
    ap.add_argument("--quiet", action="store_true",
                    help="stampa solo le celle problematiche")
    a = ap.parse_args()

    specs = [a.spec] if a.spec else list(SPECS)
    righe = [check_cell(s, v) for s in specs for v in VARIANTS]

    if not a.quiet:
        print(f"{'cella':30s} {'theta^(0)':>12s} {'fit':>12s} "
              f"{'iter':>5s} {'loglik':>11s}  esito")
        print("-" * 92)
    for r in righe:
        if a.quiet and r["esito"] in (OK,):
            continue
        cella = f"{r['spec']}/{r['variant']}"
        ll = f"{r['loglik']:.2f}" if isinstance(r["loglik"], (int, float)) else "--"
        it = str(r["n_iter"]) if r["n_iter"] is not None else "--"
        coda = r["esito"] + (f"  ({r['motivo']})" if r["motivo"] else "")
        print(f"{cella:30s} {_hhmm(r['t_theta']):>12s} {_hhmm(r['t_fit']):>12s} "
              f"{it:>5s} {ll:>11s}  {coda}")

    conteggi = {e: sum(1 for r in righe if r["esito"] == e)
                for e in (OK, ASSENTE, PARZIALE, DA_GUARDARE, INCOERENTE)}
    print("\n" + "  ".join(f"{k}: {v}" for k, v in conteggi.items() if v))

    rotte = conteggi[INCOERENTE]
    if rotte:
        print(f"\n{rotte} cella/e INCOERENTE/I: ricalcolale con\n"
              f"  python core/run_final_artifacts.py --spec <spec> --variant <variante>")
        return 1
    print("\nNessuna incoerenza.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
