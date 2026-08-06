"""
src/forecast/benchmarks.py

I METRI DI PARAGONE univariati del nowcast.

Due, entrambi sul solo PIL trimestrale: non vedono nessuno dei 36 indicatori,
quindi non possono reagire alle notizie dentro il trimestre.  E' il punto.  Se
il modello a fattori non li batte, tutta la macchina mensile-settimanale non sta
aggiungendo niente.

  ar2   AR(2) con costante sul PIL disponibile a `as_of`, previsto fino al
        trimestre target.  Il metro principale: cattura la persistenza della
        crescita, che e' quanto si puo' sapere senza guardare altro.
  mean  la media storica del PIL, calcolata sulle sole osservazioni pubblicate
        a `as_of`.  Espandente, mai di campione pieno — usare la media
        dell'intero campione sarebbe look-ahead, e in una tesi che parla di code
        sarebbe il look-ahead peggiore: includerebbe il 2020.

COSA VUOL DIRE "SETTIMANALE" PER UN BENCHMARK
---------------------------------------------
Entrambi si valutano sulla stessa griglia settimanale dei modelli, ma cambiano
solo quando esce un PIL nuovo: fra un rilascio e l'altro la loro traiettoria e'
piatta per costruzione.  Non e' un difetto dell'implementazione, e' la loro
natura, ed e' il contrasto che rende leggibile la figura — il modello a fattori
si muove ogni settimana, il benchmark a scalini trimestrali.

L'UNITA'
--------
Il PIL del pannello e' in log-differenza annualizzata (`QoQ_log_ar`).  I
benchmark stimano in quell'unita' e riportano anche il valore in punti BEA
passando dalla stessa catena dei modelli (`scale.to_bea`), cosi' i numeri sono
confrontabili riga per riga senza conversioni sparse per il codice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.forecast import scale
from src.forecast.release_calendar import (
    known_at,
    load_metadata,
    load_panel,
    quarter_end,
)

#: Ordine dell'AR: due ritardi con costante.
_AR_ORDER = (2, 0, 0)

#: Etichette con cui i benchmark compaiono nel CSV lungo.
AR2 = "ar2"
MEAN = "mean"
BENCHMARK_SPEC = "benchmark"


# ─── Il PIL noto a una data ───────────────────────────────────────────────────

def available_target(as_of_date, target_series: str | None = None,
                     panel: pd.DataFrame | None = None,
                     metadata: pd.DataFrame | None = None) -> pd.Series:
    """
    La storia della serie target pubblicata a `as_of_date`, senza NaN.

    Passa dallo stesso `known_at` dei modelli: un solo calendario per tutti, cosi'
    modello e benchmark vedono per costruzione la stessa informazione.
    """
    panel = load_panel() if panel is None else panel
    metadata = load_metadata() if metadata is None else metadata
    target = target_series or list(panel.columns)[-1]
    return known_at(panel, as_of_date, metadata=metadata)[target].dropna()


def _quarters_ahead(last_qe: pd.Timestamp, target_qe: pd.Timestamp) -> int:
    """Quanti trimestri separano l'ultimo pubblicato dal target."""
    a = last_qe.year * 4 + (last_qe.month - 1) // 3
    b = target_qe.year * 4 + (target_qe.month - 1) // 3
    return b - a


def _package(level: float, history: pd.Series, as_of_date, target_quarter: str,
             variant: str, converged: bool, extra: dict | None = None) -> dict:
    """Le stesse chiavi che produce `nowcast_engine.nowcast`, per unire i CSV."""
    mean_h, std_h = float(history.mean()), float(history.std(ddof=1))
    out = {
        "as_of": str(pd.Timestamp(as_of_date).date()),
        "target_quarter": target_quarter,
        "spec": BENCHMARK_SPEC,
        "variant": variant,
        "nowcast_bea": float(scale.to_bea(level)),
        "nowcast_livello": float(level),
        "nowcast_z": (level - mean_h) / std_h if std_h else float("nan"),
        "sd_z": float("nan"),
        "n_iter": 0,
        "converged": bool(converged),
        "reestimated": False,
        "n_obs_used": int(history.size),
        "ultimo_target_noto": str(history.index[-1].date()),
    }
    if extra:
        out.update(extra)
    return out


# ─── (i) AR(2) ────────────────────────────────────────────────────────────────

def ar2_nowcast(as_of_date, target_quarter: str, target_series: str | None = None,
                panel: pd.DataFrame | None = None,
                metadata: pd.DataFrame | None = None) -> dict:
    """
    AR(2) con costante sul PIL noto a `as_of_date`, previsto fino al trimestre
    target.

    Se il target risulta gia' pubblicato a questa data, la previsione onesta e'
    il valore pubblicato stesso: non c'e' niente da prevedere.  Non capita nel
    ciclo settimanale — il calendario esclude i trimestri gia' usciti dai
    bersagli — ma la funzione e' chiamabile da sola e non deve mentire.

    Rete di sicurezza: se la stima non converge o fallisce, si ricade sulla media
    della storia disponibile, e `converged` lo dichiara.
    """
    from statsmodels.tsa.arima.model import ARIMA

    history = available_target(as_of_date, target_series, panel, metadata)
    target_qe = quarter_end(target_quarter)
    last_qe = history.index[-1]

    if target_qe <= last_qe:
        return _package(float(history.loc[target_qe]), history, as_of_date,
                        target_quarter, AR2, converged=True)

    steps = _quarters_ahead(last_qe, target_qe)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(history.to_numpy(), order=_AR_ORDER, trend="c").fit()
        fc = np.asarray(res.forecast(steps), dtype=float)
        if fc.size == steps and np.all(np.isfinite(fc)):
            converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            return _package(float(fc[-1]), history, as_of_date, target_quarter,
                            AR2, converged=converged)
    except Exception:
        pass

    return _package(float(history.mean()), history, as_of_date, target_quarter,
                    AR2, converged=False)


# ─── (ii) Media espandente ────────────────────────────────────────────────────

def mean_nowcast(as_of_date, target_quarter: str, target_series: str | None = None,
                 panel: pd.DataFrame | None = None,
                 metadata: pd.DataFrame | None = None) -> dict:
    """
    La media storica del PIL sulle sole osservazioni pubblicate a `as_of_date`.

    Espandente: la finestra cresce con la data, e non contiene mai un trimestre
    che a quella data non era uscito.  E' il metro che certifica se un modello
    aggiunge qualcosa — batterlo e' il minimo sindacale, e in campioni con code
    grosse non e' scontato.
    """
    history = available_target(as_of_date, target_series, panel, metadata)
    return _package(float(history.mean()), history, as_of_date, target_quarter,
                    MEAN, converged=True)


#: I benchmark, per nome, come li chiama il ciclo settimanale.
BENCHMARKS = {AR2: ar2_nowcast, MEAN: mean_nowcast}

__all__ = ["ar2_nowcast", "mean_nowcast", "available_target",
           "BENCHMARKS", "AR2", "MEAN", "BENCHMARK_SPEC"]


# ─── Verifica ─────────────────────────────────────────────────────────────────
# Esegui:  python -m src.forecast.benchmarks

if __name__ == "__main__":
    def _hr(t: str) -> None:
        print("\n" + "=" * 76)
        print(t)
        print("=" * 76)

    full, meta = load_panel(), load_metadata()
    target = list(full.columns)[-1]

    _hr("benchmarks.py — AR(2) e media espandente")

    for as_of, tq, scenario in (("2008-11-14", "2008Q4", "CRISI"),
                                ("2015-05-15", "2015Q2", "CALMA")):
        _hr(f"{scenario}:  as_of = {as_of}   target = {tq}")
        realised_lvl = float(full.loc[quarter_end(tq), target])
        realised_bea = float(scale.to_bea(realised_lvl))

        print(f"  {'metodo':<8}{'livello':>11}{'BEA %':>11}{'errore BEA':>13}"
              f"{'n oss.':>9}{'ultimo noto':>14}")
        for name, fn in BENCHMARKS.items():
            r = fn(as_of, tq, panel=full, metadata=meta)
            print(f"  {name:<8}{r['nowcast_livello']:>11.4f}{r['nowcast_bea']:>11.4f}"
                  f"{r['nowcast_bea'] - realised_bea:>13.4f}{r['n_obs_used']:>9d}"
                  f"{r['ultimo_target_noto']:>14}")
        print(f"  {'realizz.':<8}{realised_lvl:>11.4f}{realised_bea:>11.4f}"
              f"{0.0:>13.4f}")

    _hr("Nessun look-ahead: la finestra cresce con la data")
    for as_of in ("1995-06-15", "2008-11-14", "2020-05-15", "2024-06-14"):
        h = available_target(as_of, panel=full, metadata=meta)
        print(f"  as_of {as_of}: {h.size:3d} trimestri, ultimo {h.index[-1].date()}, "
              f"media espandente {h.mean():+.4f} (log ann.)")
    print("  -> la media si muove nel tempo: se fosse di campione pieno sarebbe "
          "una costante,\n     e includerebbe il 2020 gia' nel 1995.")

    _hr("Fra due rilasci il benchmark e' piatto (e il modello no)")
    prev = None
    for as_of in ("2008-11-07", "2008-11-14", "2008-11-21", "2008-11-28",
                  "2009-02-06"):
        r = ar2_nowcast(as_of, "2009Q1", panel=full, metadata=meta)
        flag = "" if prev is None else ("  <- invariato" if abs(r["nowcast_bea"] - prev) < 1e-9
                                        else "  <- SALTO (nuovo PIL pubblicato)")
        print(f"  as_of {as_of}: ar2 = {r['nowcast_bea']:+.4f}%{flag}")
        prev = r["nowcast_bea"]

    _hr("Fatto.")
