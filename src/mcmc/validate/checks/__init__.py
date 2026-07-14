"""
mcmc/validate/checks/
=====================

Gli helper condivisi dai quattro moduli di check.

**Seconda regola non negoziabile, resa strutturale:** :func:`summarize` restituisce
sempre ``ess`` e ``r_hat`` insieme alla stima.  Non è disciplina, è tipo: non esiste un
modo di ottenere una media a posteriori da questo modulo *senza* il suo ESS.  Il motivo è
storico e concreto — il ``rho_hat = -0.51`` in cui abbiamo creduto per settimane veniva da
catene con ESS 3–23 su 2000 draw, cioè da una manciata di draw indipendenti travestita da
misura.
"""

from __future__ import annotations

import numpy as np

from mcmc.diagnostics import ess, split_r_hat


def summarize(chains: np.ndarray, truth=None, *, q=0.05) -> dict:
    """``chains`` è ``(n_chains, n_keep)`` per uno scalare.  Restituisce media, CI, ESS
    (pooled), split-R-hat e — se ``truth`` è dato — se il CI **copre il vero**.

    ``covers`` è la quantità che distingue «stima incerta» da «stima confidente e
    sbagliata», ed è quella che ha smascherato l'attenuazione di ``rho``: un CI al 90%
    deve coprire il vero ~90% delle volte, e ne copriva il 25%.
    """
    ch = np.asarray(chains, float)
    if ch.ndim == 1:
        ch = ch[None, :]
    flat = ch.ravel()
    lo, hi = float(np.quantile(flat, q)), float(np.quantile(flat, 1 - q))
    out = {
        "mean": float(flat.mean()),
        "lo": lo, "hi": hi,
        "ess": float(ess(ch)),
        "r_hat": float(split_r_hat(ch)),
        "n": int(flat.size),
    }
    if truth is not None:
        out["truth"] = float(truth)
        out["covers"] = bool(lo <= truth <= hi)
        out["ratio"] = float(out["mean"] / truth) if truth != 0 else float("nan")
    return out


def relerr(est, truth) -> float:
    """Errore relativo su una matrice/vettore (norma di Frobenius)."""
    e, t = np.asarray(est, float), np.asarray(truth, float)
    den = np.linalg.norm(t)
    return float(np.linalg.norm(e - t) / den) if den > 0 else float("nan")


def say(msg: str) -> None:
    print(f"    {msg}", flush=True)
