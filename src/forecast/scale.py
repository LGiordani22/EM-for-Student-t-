"""
src/forecast/scale.py

La CATENA DI SCALA del nowcast, andata e ritorno, in un posto solo.

Il modello non vede punti di PIL: vede una variabile standardizzata di una
variabile trasformata.  Fra il numero che esce dal filtro di Kalman e il numero
che si scrive su un asse ci sono due cambi di unita', e sbagliarne uno non
produce alcun errore visibile — produce una figura plausibile e sbagliata.
Questo modulo li tiene insieme, entrambe le direzioni, con un test che chiude
il giro.

    livello BEA          100*((X_t/X_{t-1})^4 - 1)      cio' che dice il giornale
        ^  |  to_bea / from_bea
        |  v
    log annualizz.       400*log(X_t/X_{t-1})           l'unita' del pannello
        ^  |  to_level / to_z
        |  v
    standardizzata       (x - mean) / std               l'unita' del modello

LA MEDIA E LA DEVIAZIONE SONO QUELLE DELLA DATA `as_of`, MAI DEL CAMPIONE PIENO
------------------------------------------------------------------------------
`standardize_asof` prende in ingresso il pannello GIA' mascherato al `as_of`
(vedi `release_calendar.known_at`), quindi media e deviazione sono calcolate
sulle sole osservazioni pubblicate a quella data: espandenti per costruzione.
Usare media e deviazione del campione pieno sarebbe look-ahead — e non un
look-ahead innocuo: la deviazione del PIL calcolata includendo il 2020 e' molto
piu' grande di quella nota nel 2008, quindi lo stesso z tornerebbe un livello
diverso.  E' il Paradosso della Volatilita': in una finestra di stima calma
`std` e' piccola, e il livello de-standardizzato sembra compresso.  E' corretto
cosi' in tempo reale.

PERCHE' IL LOG E POI L'ESPONENZIALE
-----------------------------------
Il pannello tiene il PIL in log-differenza annualizzata perche' l'aggregatore di
Mariano-Murasawa e' esatto solo in log (l'argomento sta in
`data_loader_final.apply_final_transform`).  Il ritorno in unita' BEA e'
`100*(exp(x/100) - 1)`, che e' MONOTONA: preserva l'ordine, quindi si applica
ai quantili uno a uno, senza ricampionare nulla.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from em.em_initialization import standardize as _standardize

#: Le trasformazioni per cui questo modulo sa tornare in unita' BEA.
#: `QoQ_log_ar` e' quella del PIL (400*log del rapporto trimestrale).
_ANNUALISED_LOG = "QoQ_log_ar"


# ─── Andata: pannello -> unita' del modello ───────────────────────────────────

def standardize_asof(panel_asof: pd.DataFrame):
    """
    Standardizza il pannello di UNA data `as_of`.

    Parameters
    ----------
    panel_asof : pd.DataFrame
        Pannello gia' mascherato al `as_of` (`release_calendar.known_at`).
        Che sia mascherato e' il punto: media e deviazione escono espandenti.

    Returns
    -------
    (Y_std, mean, std) : (pd.DataFrame, pd.Series, pd.Series)
    """
    return _standardize(panel_asof)


# ─── Ritorno: unita' del modello -> punti di PIL ──────────────────────────────

def to_level(z, mean: float, std: float):
    """Da standardizzata a log-differenza annualizzata: `x = z*std + mean`."""
    return np.asarray(z, dtype=float) * float(std) + float(mean)


def to_z(x, mean: float, std: float):
    """L'inversa di `to_level`."""
    return (np.asarray(x, dtype=float) - float(mean)) / float(std)


def to_bea(x, transform: str = _ANNUALISED_LOG):
    """
    Da log-differenza annualizzata a tasso di crescita BEA:
    `g = 100*(exp(x/100) - 1)`.

    Monotona crescente, quindi applicabile ai quantili uno a uno.
    """
    if transform != _ANNUALISED_LOG:
        raise ValueError(
            f"to_bea sa invertire solo {_ANNUALISED_LOG!r}, non {transform!r}. "
            f"La serie target del pannello `final` e' in {_ANNUALISED_LOG}."
        )
    return 100.0 * (np.exp(np.asarray(x, dtype=float) / 100.0) - 1.0)


def from_bea(g, transform: str = _ANNUALISED_LOG):
    """L'inversa di `to_bea`: `x = 100*log(1 + g/100)`."""
    if transform != _ANNUALISED_LOG:
        raise ValueError(
            f"from_bea sa invertire solo {_ANNUALISED_LOG!r}, non {transform!r}."
        )
    return 100.0 * np.log1p(np.asarray(g, dtype=float) / 100.0)


# ─── La catena intera ─────────────────────────────────────────────────────────

def unscale(z, mean: float, std: float, transform: str = _ANNUALISED_LOG) -> dict:
    """
    La catena di ritorno completa, in una chiamata: da cio' che esce dal modello
    a cio' che si mette su un asse.

    Returns
    -------
    dict
        `z`      valore standardizzato (com'e' entrato)
        `level`  log-differenza annualizzata (l'unita' del pannello)
        `bea`    tasso di crescita BEA, %  (l'unita' del giornale)
    """
    level = to_level(z, mean, std)
    return {
        "z": float(np.asarray(z, dtype=float)),
        "level": float(level),
        "bea": float(to_bea(level, transform)),
    }


__all__ = ["standardize_asof", "to_level", "to_z", "to_bea", "from_bea", "unscale"]


# ─── Test di andata e ritorno ─────────────────────────────────────────────────
# Esegui:  python -m src.forecast.scale
# Verifica che la catena chiuda: partendo da un valore noto e tornando indietro
# si deve riottenere lo stesso numero, bit per bit entro la tolleranza numerica.

def _roundtrip_checks() -> bool:
    ok = True

    # 1. standardizzazione: (x - mean)/std e ritorno.
    rng = np.random.default_rng(0)
    x = rng.normal(2.5, 3.0, size=500)
    mean, std = float(x.mean()), float(x.std(ddof=1))
    back = to_level(to_z(x, mean, std), mean, std)
    err = float(np.abs(back - x).max())
    ok &= err < 1e-10
    print(f"  standardizza -> de-standardizza : max|err| = {err:.2e}  "
          f"{'OK' if err < 1e-10 else 'FAIL'}")

    # 2. BEA: log annualizzato -> tasso -> log annualizzato.
    x = np.array([-32.82, -10.0, -1.5, 0.0, 2.0, 7.5, 29.90])   # include 2020Q2/Q3
    back = from_bea(to_bea(x))
    err = float(np.abs(back - x).max())
    ok &= err < 1e-10
    print(f"  to_bea -> from_bea              : max|err| = {err:.2e}  "
          f"{'OK' if err < 1e-10 else 'FAIL'}")

    # 3. La catena intera su un valore noto.
    mean, std = 2.1, 3.4
    z0 = -1.75
    chain = unscale(z0, mean, std)
    z_back = to_z(from_bea(chain["bea"]), mean, std)
    err = abs(z_back - z0)
    ok &= err < 1e-10
    print(f"  catena intera (z -> BEA -> z)   : max|err| = {err:.2e}  "
          f"{'OK' if err < 1e-10 else 'FAIL'}")
    print(f"    z={z0:+.4f}  ->  livello={chain['level']:+.4f} (log ann.)  "
          f"->  BEA={chain['bea']:+.4f}%")

    # 4. Il segno e l'ordine si conservano.
    q = np.array([-8.0, -2.0, 0.0, 1.0, 4.0])
    mono = bool(np.all(np.diff(to_bea(q)) > 0))
    ok &= mono
    print(f"  monotonia di to_bea             : {'OK' if mono else 'FAIL'}")

    # 5. Ordine di grandezza: in log annualizzato e BEA i piccoli valori quasi
    #    coincidono, i grandi no. Se qualcuno confondesse le due unita', un
    #    -32.82 diventerebbe -28.0: e' li' che si vede l'errore.
    print(f"  riferimento 2020Q2: {-32.82:+.2f} log ann.  ->  "
          f"{float(to_bea(-32.82)):+.2f}% BEA")
    return ok


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("scale.py — test di andata e ritorno della catena di scala")
    print("=" * 72)
    good = _roundtrip_checks()
    print("=" * 72)
    print("TUTTO OK" if good else "QUALCOSA NON TORNA")
    print("=" * 72)
