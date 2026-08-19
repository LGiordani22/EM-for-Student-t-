"""
core/forecast/release_calendar.py

Il CALENDARIO DI PUBBLICAZIONE come servizio interrogabile.

Il pannello `final` porta con se' il proprio calendario: ogni serie ha un
`publication_delay_days` in `config/series_final.json` (riportato in
`data/processed/final/metadata_final.csv`), misurato in giorni di calendario
dalla FINE del periodo di riferimento.  La regola di disponibilita' e' una sola
e vive in `data_loader_final.as_of`:

    il periodo p della serie i e' noto alla data D  <=>  fine_mese(p) + delay_i <= D

Questo modulo NON riscrive quella regola: la importa.  Aggiunge le tre domande
che il nowcast settimanale pone e che il mascheramento da solo non risponde.

  1. `known_at(panel, D)`      il pannello come si vedeva a D (ragged edge)
  2. `releases_between(a, b)`  quali serie escono nell'intervallo (a, b]
  3. `gdp_release_date(q)`     quando esce il PIL del trimestre q

La (2) serve a due cose: sapere se fra due venerdi' e' cambiato qualcosa (se
non e' cambiato niente, il nowcast non puo' muoversi e il fit si salta), e
dare all'asse x della figura 8a i suoi punti di salto.  La (3) posiziona i
pallini dei rilasci nella stessa figura.

LE QUATTRO ISM HANNO LE DATE VERE
---------------------------------
`data/processed/final/ism_release_dates.csv` contiene la data di rilascio
EFFETTIVA di ISM_PMI, ISM_PRICES, ISM_EMP e ISM_NMI, mese per mese dal 1979.
Per quelle quattro serie non si approssima con il delay: si usa il giorno
esatto.  Dove il file non copre un mese si ricade sul delay, cosi' il
comportamento non ha buchi.

PSEUDO-REAL-TIME
----------------
Il calendario si applica ai valori CORRENTI (revisionati) del pannello: si
riproduce la tempistica di pubblicazione, non le revisioni dei dati.  Un
nowcast del 2008 vede quindi i dati del 2008 nella loro versione di oggi.  E'
una semplificazione dichiarata, non un difetto nascosto.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from data_loader_final import as_of as _as_of_by_delay

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FINAL_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "final")

#: Serie con date di rilascio reali (non approssimate dal delay).
_EXACT_RELEASE_SERIES = ("ISM_PMI", "ISM_PRICES", "ISM_EMP", "ISM_NMI")

#: Giorno-ancora della griglia settimanale: il venerdi' chiude la settimana
#: lavorativa, quindi tutto cio' che esce nella settimana e' dentro.
_WEEK_ANCHOR = "W-FRI"


# ─── Caricamento ──────────────────────────────────────────────────────────────

def load_metadata() -> pd.DataFrame:
    """Metadati delle 37 serie: `series_id`, `freq`, `publication_delay_days`, ..."""
    path = os.path.join(_FINAL_DIR, "metadata_final.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Metadati non trovati: {path}\n"
            f"Generali con:  python core/data_loader_final.py"
        )
    return pd.read_csv(path)


def load_panel() -> pd.DataFrame:
    """Il pannello `final` trasformato (499 x 37), indice month-end."""
    path = os.path.join(_FINAL_DIR, "dataset_final.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pannello non trovato: {path}\n"
            f"Generalo con:  python core/data_loader_final.py"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_exact_releases() -> pd.DataFrame:
    """
    Date di rilascio reali delle ISM: colonne `series_id`, `ref_month`,
    `release_date`.  DataFrame vuoto se il file non c'e' (si ricade sui delay).
    """
    path = os.path.join(_FINAL_DIR, "ism_release_dates.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["series_id", "ref_month", "release_date"])
    df = pd.read_csv(path, parse_dates=["ref_month", "release_date"])
    return df.dropna(subset=["release_date"])


# ─── 1. Cosa e' noto a una data ───────────────────────────────────────────────

def known_at(panel: pd.DataFrame, D, metadata: pd.DataFrame | None = None,
             exact: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Il pannello come si vedeva alla data `D`: copia con NaN su tutto cio' che a
    `D` non era ancora pubblicato.

    Base: la regola dei delay di `data_loader_final.as_of`.  Sopra, le quattro
    ISM vengono ricalcolate con la loro data di rilascio vera dove disponibile.

    Parameters
    ----------
    panel : pd.DataFrame
        Pannello month-end (tipicamente `load_panel()`).
    D : str | pd.Timestamp
        La data di osservazione.
    metadata, exact : pd.DataFrame, optional
        Gia' caricati, per non rileggere il disco a ogni settimana.

    Returns
    -------
    pd.DataFrame
        Stessa forma di `panel`, bordo frastagliato al fondo.
    """
    D = pd.Timestamp(D)
    meta = load_metadata() if metadata is None else metadata
    out = _as_of_by_delay(panel, meta, D)

    exact = load_exact_releases() if exact is None else exact
    if exact.empty:
        return out

    for sid in _EXACT_RELEASE_SERIES:
        if sid not in out.columns:
            continue
        rel = exact[exact["series_id"] == sid].set_index("ref_month")["release_date"]
        covered = out.index.intersection(rel.index)
        # Dove la data vera esiste decide lei; fuori copertura resta il delay.
        published = rel.reindex(covered) <= D
        out.loc[covered[~published.to_numpy()], sid] = np.nan
        out.loc[covered[published.to_numpy()], sid] = panel.loc[
            covered[published.to_numpy()], sid
        ]
    return out


# ─── 2. Quando cambia l'informazione ──────────────────────────────────────────

def release_date(series_id: str, ref_period: pd.Timestamp,
                 metadata: pd.DataFrame | None = None,
                 exact: pd.DataFrame | None = None) -> pd.Timestamp:
    """
    Data di pubblicazione del periodo `ref_period` (month-end) della serie
    `series_id`: la data vera se nota, altrimenti `ref_period + delay`.
    """
    meta = load_metadata() if metadata is None else metadata
    ref_period = pd.Timestamp(ref_period)

    if series_id in _EXACT_RELEASE_SERIES:
        exact = load_exact_releases() if exact is None else exact
        hit = exact[(exact["series_id"] == series_id)
                    & (exact["ref_month"] == ref_period)]
        if len(hit):
            return pd.Timestamp(hit["release_date"].iloc[0])

    row = meta[meta["series_id"] == series_id]
    if not len(row):
        raise KeyError(f"Serie {series_id!r} assente dai metadati.")
    delay = int(row["publication_delay_days"].iloc[0])
    return ref_period + pd.Timedelta(days=delay)


def releases_between(a, b, panel: pd.DataFrame | None = None,
                     metadata: pd.DataFrame | None = None,
                     exact: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Le pubblicazioni cadute nell'intervallo semiaperto (a, b].

    Restituisce un DataFrame `series_id`, `ref_period`, `release_date`, ordinato
    per data.  Vuoto = fra `a` e `b` non e' uscito nulla: il nowcast non puo'
    essersi mosso, e chi chiama puo' saltare il fit.
    """
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    panel = load_panel() if panel is None else panel
    meta = load_metadata() if metadata is None else metadata
    exact = load_exact_releases() if exact is None else exact

    rows = []
    for _, m in meta.iterrows():
        sid = m["series_id"]
        if sid not in panel.columns:
            continue
        # Solo i periodi che il pannello contiene davvero (le trimestrali
        # esistono solo a fine trimestre).
        periods = panel.index[panel[sid].notna()]
        # Finestra stretta attorno a (a, b]: nessun bisogno di scorrere 40 anni.
        lo = a - pd.Timedelta(days=int(m["publication_delay_days"]) + 40)
        for p in periods[(periods >= lo) & (periods <= b)]:
            rd = release_date(sid, p, meta, exact)
            if a < rd <= b:
                rows.append({"series_id": sid, "ref_period": p, "release_date": rd})

    out = pd.DataFrame(rows, columns=["series_id", "ref_period", "release_date"])
    return out.sort_values(["release_date", "series_id"]).reset_index(drop=True)


# ─── 3. Il PIL: rilascio e trimestri in volo ──────────────────────────────────

def quarter_end(quarter: str) -> pd.Timestamp:
    """'2008Q4' -> Timestamp('2008-12-31')."""
    s = str(quarter).upper().replace(" ", "")
    if "Q" not in s:
        raise ValueError(f"trimestre {quarter!r} non e' nella forma 'YYYYQn'.")
    y, q = s.split("Q")
    y, q = int(y), int(q)
    if not 1 <= q <= 4:
        raise ValueError(f"trimestre fuori range in {quarter!r}: Q{q}.")
    return pd.Timestamp(y, q * 3, 1) + pd.offsets.MonthEnd(0)


def quarter_label(qe: pd.Timestamp) -> str:
    """Timestamp('2008-12-31') -> '2008Q4'."""
    qe = pd.Timestamp(qe)
    return f"{qe.year}Q{(qe.month - 1) // 3 + 1}"


def gdp_release_date(quarter: str, target_series: str = "GDPC1",
                     metadata: pd.DataFrame | None = None) -> pd.Timestamp:
    """
    Quando esce il PIL del trimestre `quarter`: `fine_trimestre + delay`, col
    delay LETTO dai metadati (non assunto).  E' la x dei pallini della 8a.
    """
    return release_date(target_series, quarter_end(quarter), metadata)


def targets_in_flight(D, n_ahead: int = 1, target_series: str = "GDPC1",
                      metadata: pd.DataFrame | None = None) -> list[str]:
    """
    I trimestri "in volo" alla data `D`: quelli il cui PIL non e' ancora uscito,
    dal primo non pubblicato fino a `n_ahead` trimestri oltre quello che
    contiene `D`.

    Con `n_ahead=1` (default) ne restano tipicamente tre vivi insieme —
    precedente, corrente e prossimo — che e' cio' che nella figura 8a fa vedere
    piu' linee colorate contemporaneamente.

    Il target non puo' mai essere un trimestre gia' pubblicato: la stessa regola
    di calendario decide sia cosa entra nel pannello sia cosa e' un bersaglio,
    quindi il PIL del target non e' nel pannello per costruzione.
    """
    D = pd.Timestamp(D)
    meta = load_metadata() if metadata is None else metadata

    current_qe = D + pd.offsets.QuarterEnd(0)
    last_qe = current_qe + pd.offsets.QuarterEnd(n_ahead)

    out: list[str] = []
    qe = current_qe
    # Indietro finche' il PIL non risulta gia' pubblicato.
    while release_date(target_series, qe, meta) > D:
        out.append(quarter_label(qe))
        qe = qe - pd.offsets.QuarterEnd(1)
    # Avanti di n_ahead trimestri.
    qe = current_qe + pd.offsets.QuarterEnd(1)
    while qe <= last_qe:
        out.append(quarter_label(qe))
        qe = qe + pd.offsets.QuarterEnd(1)

    return sorted(set(out), key=lambda q: quarter_end(q))


def horizon_week(D, quarter: str) -> int:
    """
    Settimane rispetto all'INIZIO del trimestre target: 1 = prima settimana del
    trimestre, 13 = ultima, negative = prima che il trimestre cominci
    (forecast), oltre 13 = il trimestre e' finito ma il PIL non e' uscito
    (backcast).  E' la convenzione dell'asse x di Cascaldi-Garcia.
    """
    D = pd.Timestamp(D)
    # MonthBegin(-3) e non (-2): partendo da una data di FINE mese, il primo
    # MonthBegin all'indietro cade sul primo del mese stesso.  Da 2008-12-31
    # servono quindi tre passi per arrivare a 2008-10-01, l'inizio del
    # trimestre; con due si arriverebbe a 2008-11-01 e tutta la scala sarebbe
    # sfasata di un mese (as_of=2008-10-03 su 2008Q4 dava h-4 invece di h+1).
    q_start = quarter_end(quarter) + pd.offsets.MonthBegin(-3)
    return int(np.floor((D - q_start).days / 7.0)) + 1


# ─── La griglia settimanale ───────────────────────────────────────────────────

def weekly_grid(start: str, end: str) -> list[pd.Timestamp]:
    """Le date `as_of`: tutti i venerdi' in [start, end] ('YYYY-MM-DD' o 'YYYY-MM')."""
    return list(pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end),
                              freq=_WEEK_ANCHOR))


__all__ = [
    "load_metadata", "load_panel", "load_exact_releases",
    "known_at", "release_date", "releases_between",
    "quarter_end", "quarter_label", "gdp_release_date", "targets_in_flight",
    "horizon_week", "weekly_grid",
]
