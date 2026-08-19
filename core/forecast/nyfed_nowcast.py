"""
core/forecast/nyfed_nowcast.py

IL NOWCAST DELLA NEW YORK FED come serie confrontabile con la mia.

Lettore puro: legge i due Excel storici in `data/raw/` e li riduce a una tabella
lunga con la STESSA nozione di orizzonte che usa `weekly_nowcast.py`.  Non stima
niente, non tocca il motore, non entra nella figura 8a.

I DUE FILE
----------
    New-York-Fed-Staff-Nowcast_data_2002-2021.xlsx   2002-01-04 -> 2021-08-27
    New-York-Fed-Staff-Nowcast_download_data.xlsx    2022-12-02 -> oggi

In mezzo c'e' un buco: fra il 2021-09 e il 2022-11 la Fed sospese il nowcast.
Non e' un difetto dei file ed e' inutile interpolarlo — i due blocchi si
concatenano e basta, e `coverage_gaps()` lo rende esplicito a chi legge.

Il foglio utile e' "Forecasts By Horizon" in entrambi.  Le intestazioni non
stanno alla stessa riga (13 nel primo, 5 nel secondo) e sopra ci sono righe di
testo libero: il numero di riga NON e' cablato, si cerca la riga che comincia
con "Forecast date".  Il trimestre e' scritto in due modi diversi — `2002Q1`
nel primo file, `2022:Q4` nel secondo — e viene uniformato a `YYYYQn`.

LA STRUTTURA DEL FOGLIO (dedotta, non assunta)
----------------------------------------------
Ogni riga e' un venerdi' di pubblicazione con un `Reference quarter` che e' il
trimestre CONTENENTE quella data, e fino a tre valori:

    Backcast (previous quarter)   stima di  refq - 1
    Nowcast  (current quarter)    stima di  refq
    Forecast (next quarter)       stima di  refq + 1

Le tre colonne non sono mai piene tutte insieme: il backcast esiste solo nelle
prime settimane del trimestre nuovo (finche' il PIL del precedente non esce), il
forecast solo nelle ultime settimane del trimestre corrente.  Il target di ogni
cella si RICAVA da `Reference quarter` con l'offset -1/0/+1: e' l'unico modo
per non sbagliare, perche' la colonna dice l'orizzonte, non il trimestre.

L'ALLINEAMENTO DEGLI ORIZZONTI — il punto dove il confronto sbaglia in silenzio
------------------------------------------------------------------------------
Il rischio non e' scrivere una formula sbagliata: e' confrontare due numeri che
si riferiscono a istanti diversi senza accorgersene.  Tre fatti lo evitano.

1.  STESSA GRIGLIA.  La Fed pubblica di venerdi'; `release_calendar.weekly_grid`
    ancora la mia griglia al venerdi' (`W-FRI`).  Le due serie cadono quindi
    sulle stesse date di calendario e non serve nessun ricampionamento — che e'
    proprio il passaggio in cui un confronto si corrompe.  Su 1026 righe del
    file storico, 1023 sono venerdi' e 3 giovedi' (settimane di festivita'):
    `off_grid_dates()` le elenca invece di nasconderle.

2.  STESSO METRO.  L'orizzonte NON viene mappato con una tabella
    backcast->"fase finale" scritta a mano.  Si applica alle date della Fed la
    STESSA funzione che etichetta le mie righe,
    `release_calendar.horizon_week(data, trimestre_target)`, che conta le
    settimane dall'INIZIO del trimestre target:

        <= 0   il trimestre non e' ancora cominciato   (il loro "Forecast")
        1..13  il trimestre e' in corso                (il loro "Nowcast")
        > 13   il trimestre e' chiuso, il PIL non e' uscito  (il loro "Backcast")

    Le tre etichette della Fed cadono cosi' da sole nelle mie tre fasi, e se un
    giorno non ci cadessero — un file diverso, una convenzione cambiata — lo si
    vede subito perche' i numeri non tornerebbero, invece di restare sepolti in
    una mappatura fissa.  `check_alignment()` verifica esattamente questo.

3.  STESSO ESTREMO DESTRO.  Il confronto "ultima stima prima del rilascio" ha
    senso solo se il rilascio e' lo stesso per tutti: si usa la data del MIO
    calendario, `gdp_release_date(q) = fine_trimestre + 28g`, la stessa regola
    che governa il mio pannello.  Serve davvero: la Fed pubblica talvolta un
    backcast il primo venerdi' del mese nuovo, DOPO che il PIL e' uscito
    (2007-11-02 porta ancora un backcast di 2007Q3).  `pre_release` lo marca e
    `last_before_release()` lo esclude.

L'ESEMPIO CHE TIENE INSIEME TUTTO: 2008Q4, GENNAIO 2009
-------------------------------------------------------
Un solo blocco di righe fa vedere sia che i trimestri sono allineati bene, sia
perche' qualcosa viene escluso.  Nel file, tutte queste righe portano
`Reference quarter = 2009Q1`: il trimestre che CONTIENE il venerdi', non il
trimestre a cui il numero si riferisce.

    COME STA NELL'EXCEL                    COME LO LEGGE QUESTO MODULO
    forecast_date  refq    Backcast        target   settimana  pre_release
    2009-01-02     2009Q1   -2.69     ->   2008Q4      +14         True
    2009-01-09     2009Q1   -2.71     ->   2008Q4      +15         True
    2009-01-16     2009Q1   -3.40     ->   2008Q4      +16         True
    2009-01-23     2009Q1   -3.47     ->   2008Q4      +17         True   <- usata
    2009-01-30     2009Q1   -3.59     ->   2008Q4      +18         False  <- scartata

Tre letture, tutte necessarie.

(a) LO SPOSTAMENTO DI UN TRIMESTRE.  La colonna dice l'ORIZZONTE ("backcast"),
    non il trimestre: il target e' `refq - 1`, cioe' 2008Q4, non 2009Q1.  Senza
    questo `-1` si confronterebbe la stima di 2008Q4 con il PIL di 2009Q1 —
    l'errore piu' facile da fare e il piu' difficile da vedere, perche' produce
    numeri plausibili.  Lo stesso venerdi' 2009-01-23 genera anche un `nowcast`
    con target 2009Q1 alla settimana +4: stessa riga, due trimestri, due
    orizzonti.  Lo spostamento e' DEDOTTO dall'offset della colonna, mai scritto
    a mano trimestre per trimestre.

(b) NESSUN RISCALAMENTO.  -3.47 nell'Excel resta -3.47 qui.  La Fed pubblica la
    crescita del PIL reale in tasso trimestrale annualizzato (SAAR, punti
    percentuali), che e' l'unita' di `nowcast_bea` e di GDPC1.  Applicare un
    fattore introdurrebbe un errore invece di toglierlo.  Si sposta il
    TRIMESTRE, non la SCALA.

(c) L'ESCLUSIONE DEL 30 GENNAIO.  Il mio calendario colloca il rilascio del PIL
    di 2008Q4 a `fine_trimestre + 28g = 2009-01-28`.  Il mio ultimo nowcast su
    quel trimestre e' quindi quello del venerdi' 23 gennaio, settimana +17: dopo
    quella data, per me 2008Q4 non e' piu' un bersaglio.  La Fed invece pubblica
    ancora il 30 gennaio, settimana +18, e quel -3.59 e' piu' vicino al vero del
    -3.47.  Tenerlo significherebbe confrontare la loro stima del 30 con la mia
    del 23: sette giorni di dati mensili in piu' da una parte sola.  Il numero
    che ne uscirebbe sembrerebbe un vantaggio di modello e sarebbe un vantaggio
    di calendario.

    E' esattamente il senso di "stesso orizzonte": non le stesse etichette, ma
    lo stesso ISTANTE rispetto al rilascio.  Il valore non e' sbagliato — e'
    fuori finestra.  `pre_release` lo marca e nessuna tabella e nessuna figura
    lo usa.

PERCHE' SI SCARTA QUEL BACKCAST (e perche' NON per il motivo che sembra)
-----------------------------------------------------------------------
La spiegazione istintiva e' che quel backcast incorpori gia' la stima advance
del BEA — loro avrebbero il numero e io no.  I dati dicono di NO, e conviene
saperlo prima di scriverlo in tesi.

Su 67 trimestri in cui la Fed pubblica un backcast anche dopo il rilascio, il
valore rispetto all'ultimo pre-rilascio si muove di +0.003 in media (sd 0.107,
scarto massimo 0.52) e in 15 casi su 67 non si muove affatto.  Se l'advance
entrasse nel modello il salto sarebbe grande e sistematicamente verso il numero
pubblicato.  Il controesempio piu' netto e' 2007Q3: backcast 1.99 il 26 ottobre,
backcast 1.99 il 2 novembre — identico — mentre il PIL era gia' uscito nel
frattempo.  Se avessero usato l'advance, il 2 novembre non avrebbero ripetuto
1.99.

Il motivo vero e' piu' semplice e non richiede di sapere cosa c'e' dentro il
loro modello: e' ASIMMETRIA DELL'INSIEME INFORMATIVO.  Quel venerdi' in piu'
porta una settimana in piu' di dati mensili (occupazione, ISM, vendite) che al
mio ultimo nowcast non erano arrivati.  Confrontarlo con il mio significherebbe
dare a loro sette giorni di dati che io non ho, e chiamare "modello migliore"
cio' che e' solo "modello piu' aggiornato".  Che l'effetto medio sia piccolo non
cambia l'argomento: e' piccolo ma a senso unico, e non c'e' ragione di regalarlo.

Nota sulla soglia: `fine_trimestre + 28g` e' la convenzione del MIO calendario e
approssima il calendario BEA, non lo riproduce (per 2008Q4 il taglio cade il 28
gennaio, mentre l'advance vero usci' negli ultimi giorni del mese).  L'argomento
non dipende da quale sia la data vera: la riga del 30 gennaio va comunque fuori
perche' ha visto una settimana di dati mensili che il mio ultimo nowcast su quel
trimestre non aveva.  E' proprio la convenzione che rende il taglio difendibile:
la stessa regola,
applicata a me e a loro, invece di due calendari diversi.  Chi vuole le date BEA
effettive deve cambiarle in `metadata_final.csv`, e cambieranno per entrambi.

UNITA'
------
La Fed pubblica la crescita del PIL reale in tasso trimestrale ANNUALIZZATO
(SAAR, punti percentuali): la stessa unita' della mia colonna `nowcast_bea` e
del realizzato `GDPC1`.  Nessuna conversione, nessun riscalamento.

RICOSTRUZIONI, NON REAL-TIME PURO
---------------------------------
Il file storico dichiara (riga di testo sopra l'intestazione) che i trimestri
2002Q1-2015Q4 sono RICOSTRUZIONI su dati real-time, non pubblicazioni avvenute
davvero; dal 2016Q1 in poi sono real-time veri.  Il campione 2007-2010 cade
quindi interamente nella parte ricostruita.  E' la stessa natura del mio
esercizio (pseudo-real-time: calendario vero, dati nella versione di oggi),
quindi il confronto e' fra due pseudo-real-time e non fra un pseudo e un vero —
ma va detto, non lasciato intendere.  `is_reconstruction` porta la bandierina
riga per riga.

Uso
---
    python -m core.forecast.nyfed_nowcast              # ispezione + controlli
    python -m core.forecast.nyfed_nowcast --quarters 2007Q4 2008Q4
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from core.forecast.release_calendar import (
    gdp_release_date,
    horizon_week,
    quarter_end,
    quarter_label,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Cartella dei due Excel.  I dati NY Fed stanno in `data/`, non fra gli output.
_NYFED_DIR = os.path.join(_PROJECT_ROOT, "data", "raw",
                          "Hystorical data NYFED Staff Nowcast")

_SHEET = "Forecasts By Horizon"

#: Prima cella dell'intestazione: la si cerca invece di cablare il numero di riga
#: (13 in un file, 5 nell'altro).
_HEADER_KEY = "forecast date"

#: Quante righe in cima scandire in cerca dell'intestazione.
_HEADER_SCAN = 40

#: Le tre colonne di valore -> (etichetta, offset in trimestri rispetto a `refq`).
#: L'offset e' cio' che trasforma "orizzonte" in "trimestre target".
_HORIZON_COLUMNS = {
    "backcast": -1,
    "nowcast": 0,
    "forecast": +1,
}

#: Ultimo trimestre ricostruito a posteriori dalla Fed; da 2016Q1 e' real-time.
_LAST_RECONSTRUCTED = "2015Q4"

#: Il giorno-ancora atteso (venerdi' = 4): serve solo alla diagnostica.
_EXPECTED_WEEKDAY = 4


# ─── Lettura dei due file ─────────────────────────────────────────────────────

def _default_paths() -> list[str]:
    """I due Excel, in ordine cronologico.  Errore parlante se mancano."""
    if not os.path.isdir(_NYFED_DIR):
        raise FileNotFoundError(f"Cartella NY Fed non trovata: {_NYFED_DIR}")
    # I `~$...xlsx` sono i lock che Excel crea quando il file e' aperto: non
    # sono cartelle di lavoro e aprirli da' PermissionError.
    found = sorted(f for f in os.listdir(_NYFED_DIR)
                   if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    if not found:
        raise FileNotFoundError(f"Nessun .xlsx in {_NYFED_DIR}")
    return [os.path.join(_NYFED_DIR, f) for f in found]


def _find_header_row(path: str) -> int:
    """
    La riga dell'intestazione: la prima la cui prima cella e' "Forecast date".

    I due file la mettono a righe diverse e sopra hanno testo libero (titoli,
    URL, la nota sulle ricostruzioni).  Cercarla invece di contarla e' cio' che
    permette di aggiungere un terzo file domani senza toccare il codice.
    """
    head = pd.read_excel(path, sheet_name=_SHEET, header=None, nrows=_HEADER_SCAN)
    first = head.iloc[:, 0].astype(str).str.strip().str.lower()
    hit = np.flatnonzero(first.to_numpy() == _HEADER_KEY)
    if not hit.size:
        raise ValueError(
            f"Intestazione non trovata in {os.path.basename(path)}: nessuna riga "
            f"fra le prime {_HEADER_SCAN} comincia con {_HEADER_KEY!r}."
        )
    return int(hit[0])


def normalise_quarter(raw) -> str:
    """
    `'2002Q1'`, `'2022:Q4'`, `'2022 q4'` -> `'2022Q4'`.

    I due file usano due convenzioni diverse; senza uniformarle il `groupby`
    per trimestre spezzerebbe in due lo stesso trimestre.
    """
    s = str(raw).strip().upper().replace(":", "").replace(" ", "").replace("-", "")
    if "Q" not in s:
        raise ValueError(f"trimestre non riconosciuto: {raw!r}")
    y, q = s.split("Q")
    return f"{int(y)}Q{int(q)}"


def _shift_quarter(q: str, k: int) -> str:
    """Trimestre spostato di `k` posizioni: `_shift_quarter('2008Q1', -1)`."""
    return quarter_label(quarter_end(q) + pd.offsets.QuarterEnd(k))


def read_sheet(path: str) -> pd.DataFrame:
    """
    Un file -> tabella larga grezza: `forecast_date`, `reference_quarter`,
    `backcast`, `nowcast`, `forecast`, `source_file`.

    Le celle non numeriche (righe di commento in coda, testo residuo) diventano
    NaN: e' `pd.to_numeric(errors='coerce')`, non un silenziamento, perche' le
    righe senza data vengono comunque scartate subito dopo.
    """
    raw = pd.read_excel(path, sheet_name=_SHEET, header=_find_header_row(path))
    if raw.shape[1] < 5:
        raise ValueError(f"{os.path.basename(path)}: attese 5 colonne, trovate "
                         f"{raw.shape[1]}.")
    df = raw.iloc[:, :5].copy()
    df.columns = ["forecast_date", "reference_quarter",
                  "backcast", "nowcast", "forecast"]

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df = df[df["forecast_date"].notna()].copy()

    df["reference_quarter"] = df["reference_quarter"].map(normalise_quarter)
    for c in _HORIZON_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["source_file"] = os.path.basename(path)
    return df.sort_values("forecast_date").reset_index(drop=True)


def load_wide(paths: list[str] | None = None) -> pd.DataFrame:
    """
    I due file concatenati, una riga per venerdi'.

    Sovrapposizioni: se una stessa `forecast_date` comparisse in entrambi i file
    vince l'ULTIMO letto (il download corrente, che incorpora le revisioni della
    Fed).  Oggi non succede — i due blocchi sono disgiunti — ma la regola deve
    esistere prima di servire, non dopo.
    """
    frames = [read_sheet(p) for p in (paths or _default_paths())]
    df = pd.concat(frames, ignore_index=True)
    df = (df.drop_duplicates(subset="forecast_date", keep="last")
            .sort_values("forecast_date").reset_index(drop=True))
    return df


# ─── Da larga a lunga, con l'orizzonte allineato al mio ───────────────────────

def load_long(paths: list[str] | None = None) -> pd.DataFrame:
    """
    La struttura pulita: una riga per (`forecast_date`, orizzonte) con il
    trimestre target gia' risolto e l'orizzonte gia' espresso nel MIO metro.

    Colonne
    -------
    forecast_date      il venerdi' di pubblicazione
    reference_quarter  il trimestre che contiene `forecast_date`
    horizon_label      'backcast' | 'nowcast' | 'forecast'  (etichetta Fed)
    target_quarter     il trimestre a cui il valore si riferisce (refq -1/0/+1)
    nowcast_bea        il valore, in punti BEA annualizzati
    horizon_week       settimane dall'inizio del trimestre target (mio metro)
    gdp_release_date   fine_trimestre + 28g, dal MIO calendario
    pre_release        True se la stima precede il rilascio BEA del target
    is_reconstruction  True se il target e' <= 2015Q4 (ricostruzione Fed)
    """
    wide = load_wide(paths)

    rows = []
    for label, offset in _HORIZON_COLUMNS.items():
        sub = wide[wide[label].notna()][
            ["forecast_date", "reference_quarter", label, "source_file"]
        ].copy()
        sub = sub.rename(columns={label: "nowcast_bea"})
        sub["horizon_label"] = label
        # Il target si DEDUCE dall'offset della colonna: la colonna dice
        # l'orizzonte, non il trimestre.
        sub["target_quarter"] = sub["reference_quarter"].map(
            lambda q, k=offset: _shift_quarter(q, k))
        rows.append(sub)

    df = pd.concat(rows, ignore_index=True)

    # ── L'allineamento, in due righe: la MIA funzione applicata alle LORO date.
    df["horizon_week"] = [
        horizon_week(d, q)
        for d, q in zip(df["forecast_date"], df["target_quarter"])
    ]
    rel = {q: gdp_release_date(q) for q in df["target_quarter"].unique()}
    df["gdp_release_date"] = df["target_quarter"].map(rel)
    df["pre_release"] = df["forecast_date"] < df["gdp_release_date"]

    last_rec = quarter_end(_LAST_RECONSTRUCTED)
    df["is_reconstruction"] = df["target_quarter"].map(
        lambda q: quarter_end(q) <= last_rec)

    return (df.sort_values(["target_quarter", "forecast_date"])
              .reset_index(drop=True)[
                  ["forecast_date", "reference_quarter", "horizon_label",
                   "target_quarter", "nowcast_bea", "horizon_week",
                   "gdp_release_date", "pre_release", "is_reconstruction",
                   "source_file"]])


# ─── L'aggregazione per il confronto ──────────────────────────────────────────

def last_before_release(df: pd.DataFrame | None = None,
                        quarters: list[str] | None = None) -> pd.DataFrame:
    """
    Per ogni trimestre, l'ULTIMA stima della Fed anteriore al rilascio BEA: la
    loro stima piu' informata, quella con cui ha senso confrontare la mia.

    Nella pratica e' sempre un backcast (il trimestre e' chiuso, il PIL non e'
    ancora uscito), ma non lo si impone: si prende l'ultima riga con
    `pre_release`, qualunque sia l'etichetta, e si RIPORTA quale etichetta era
    (`horizon_label`) e a che settimana cadeva (`horizon_week`).  Se un
    trimestre non avesse backcast — succede quando la Fed salta le settimane
    fra la chiusura del trimestre e il rilascio — il confronto userebbe l'ultimo
    nowcast e la colonna lo direbbe, invece di far sparire il trimestre.

    Il filtro `pre_release` non e' cosmetico: senza, il 2007Q3 entrerebbe col
    backcast del 2007-11-02, che ha visto una settimana di dati mensili in piu'
    di qualunque mio nowcast su quel trimestre.  Il perche' — e perche' NON e'
    la contaminazione da advance BEA che sembra — sta nel docstring del modulo.
    """
    df = load_long() if df is None else df
    sub = df[df["pre_release"]].copy()
    if quarters is not None:
        sub = sub[sub["target_quarter"].isin(quarters)]
    if sub.empty:
        return sub

    idx = sub.groupby("target_quarter")["forecast_date"].idxmax()
    out = sub.loc[idx].copy()
    out = out.rename(columns={"forecast_date": "as_of"})
    return (out[["target_quarter", "as_of", "horizon_week", "horizon_label",
                 "nowcast_bea", "gdp_release_date", "is_reconstruction"]]
            .sort_values("target_quarter").reset_index(drop=True))


def at_horizon(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """
    La stima della Fed a una settimana-orizzonte fissata, un trimestre per riga.

    E' il mattone della curva RMSE-per-orizzonte (fase 2).  Se in quella
    settimana esistono piu' righe per lo stesso trimestre — non dovrebbe, ma un
    venerdi' spostato di festivita' puo' farne cadere due nello stesso bucket —
    vince la piu' recente, coerentemente con `last_before_release`.
    """
    sub = df[(df["horizon_week"] == int(week)) & df["pre_release"]]
    if sub.empty:
        return sub
    idx = sub.groupby("target_quarter")["forecast_date"].idxmax()
    return df.loc[idx].sort_values("target_quarter").reset_index(drop=True)


# ─── Diagnostica: che l'allineamento si veda, non si creda ────────────────────

def coverage_gaps(df: pd.DataFrame | None = None,
                  min_days: int = 21) -> pd.DataFrame:
    """
    I buchi nella griglia settimanale: intervalli fra due venerdi' consecutivi
    piu' lunghi di `min_days`.  Il buco 2021-2022 (sospensione del nowcast) deve
    comparire qui, e nient'altro di grosso.
    """
    df = load_long() if df is None else df
    dates = pd.Series(sorted(df["forecast_date"].unique()))
    gap = dates.diff().dt.days
    hit = gap > min_days
    return pd.DataFrame({
        "da": dates[hit.shift(-1, fill_value=False).to_numpy()].to_numpy(),
        "a": dates[hit.to_numpy()].to_numpy(),
        "giorni": gap[hit].to_numpy().astype(int),
    })


def off_grid_dates(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Le date che NON cadono di venerdi'.  Sono poche (settimane di festivita') e
    sfasano l'orizzonte di al massimo un giorno, quindi non spostano la
    settimana; restano elencate perche' un'eccezione taciuta e' un'eccezione che
    prima o poi diventa un errore.
    """
    df = load_long() if df is None else df
    d = df.drop_duplicates("forecast_date")
    off = d[d["forecast_date"].dt.dayofweek != _EXPECTED_WEEKDAY]
    return (off[["forecast_date", "reference_quarter"]]
            .assign(giorno=off["forecast_date"].dt.day_name())
            .sort_values("forecast_date").reset_index(drop=True))


#: L'esempio del docstring, in forma verificabile: `(data, target, settimana,
#: valore, pre_release)` per le cinque righe di gennaio 2009 che nel file
#: portano tutte `Reference quarter = 2009Q1`.
_WORKED_EXAMPLE = (
    ("2009-01-02", "2008Q4", 14, -2.69, True),
    ("2009-01-09", "2008Q4", 15, -2.71, True),
    ("2009-01-16", "2008Q4", 16, -3.40, True),
    ("2009-01-23", "2008Q4", 17, -3.47, True),
    ("2009-01-30", "2008Q4", 18, -3.59, False),
)


def check_worked_example(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Il caso 2008Q4 del docstring, ricalcolato sui dati veri.

    Non e' decorazione: e' l'esempio che dimostra le tre cose insieme — lo
    spostamento di un trimestre (`refq - 1` porta a 2008Q4 righe scritte sotto
    2009Q1), l'assenza di riscalamento (-3.47 nell'Excel resta -3.47) e
    l'esclusione del 30 gennaio, fuori dalla finestra del mio calendario.  Se un
    domani il file cambiasse convenzione, o l'offset venisse toccato, questo
    controllo diventa rosso prima che una tabella sbagliata finisca in tesi.

    Restituisce una riga per caso con `atteso`, `ottenuto` e `ok`.
    """
    df = load_long() if df is None else df
    rows = []
    for date, target, week, value, pre in _WORKED_EXAMPLE:
        hit = df[(df["forecast_date"] == pd.Timestamp(date))
                 & (df["horizon_label"] == "backcast")]
        got = None if hit.empty else hit.iloc[0]
        ok = (got is not None
              and got["target_quarter"] == target
              and int(got["horizon_week"]) == week
              and abs(float(got["nowcast_bea"]) - value) < 1e-9
              and bool(got["pre_release"]) is pre)
        rows.append({
            "forecast_date": date,
            "atteso": f"{target} sett{week:+d} {value:+.2f} "
                      f"{'usata' if pre else 'SCARTATA'}",
            "ottenuto": ("(riga assente)" if got is None else
                         f"{got['target_quarter']} "
                         f"sett{int(got['horizon_week']):+d} "
                         f"{float(got['nowcast_bea']):+.2f} "
                         f"{'usata' if got['pre_release'] else 'SCARTATA'}"),
            "ok": ok,
        })
    return pd.DataFrame(rows)


def check_alignment(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Il controllo che il confronto non sbagli in silenzio: le tre etichette della
    Fed devono cadere nelle tre fasi del MIO metro senza essere state mappate a
    mano.

    Atteso, se le due convenzioni coincidono davvero:

        forecast  -> horizon_week <= 0        (trimestre non cominciato)
        nowcast   -> 1 <= horizon_week <= 13  (trimestre in corso)
        backcast  -> horizon_week >= 14       (trimestre chiuso)

    Restituisce, per etichetta, il range osservato e la quota di righe nella
    fase attesa.  Sui file attuali `quota_attesa` sta a 0.99 e le 16 righe che
    sforano sono di due tipi, entrambi innocui e entrambi fuori dal campione
    2007-2010:

      bordo di calendario (8 righe)  un venerdi' cade esattamente sull'ultimo
          giorno del trimestre (2004-12-31, 2010-12-31, 2016-09-30, ...) e
          finisce nella settimana 14, o un backcast esce il giorno dopo la
          chiusura (2005-04-01) e cade nella 13.  Sbaglia il secchiello di una
          settimana, non il trimestre target.
      coda del file corrente (8 righe)  a luglio 2026 la Fed non ha ancora
          fatto avanzare `Reference quarter` da 2026Q2 a 2026Q3, quindi il suo
          "nowcast" corre fino alla settimana 17.  Il codice segue le etichette
          del file e non le corregge d'ufficio: si sistemera' da se' al
          prossimo download, e nel frattempo si vede qui.

    `quota_pre_release` del backcast e' ~0.82, e quel 18% e' il motivo per cui
    `last_before_release()` filtra: sono backcast usciti DOPO la stima advance
    del BEA, che rispetto al mio nowcast guardano il futuro.
    """
    df = load_long() if df is None else df
    expected = {"forecast": lambda w: w <= 0,
                "nowcast": lambda w: (w >= 1) & (w <= 13),
                "backcast": lambda w: w >= 14}
    rows = []
    for label, ok in expected.items():
        g = df[df["horizon_label"] == label]
        w = g["horizon_week"].to_numpy()
        rows.append({
            "horizon_label": label,
            "n": len(g),
            "week_min": int(w.min()) if w.size else np.nan,
            "week_max": int(w.max()) if w.size else np.nan,
            "quota_attesa": float(np.mean(ok(w))) if w.size else np.nan,
            "quota_pre_release": float(g["pre_release"].mean()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _section(t: str) -> str:
    return "\n" + "=" * 84 + "\n" + t + "\n" + "=" * 84


def main() -> None:
    p = argparse.ArgumentParser(
        description="Loader del NY Fed Staff Nowcast + controlli di allineamento.")
    p.add_argument("--quarters", nargs="*", default=None,
                   help="trimestri da mostrare in dettaglio (es. 2007Q4 2008Q4)")
    a = p.parse_args()

    df = load_long()

    print(f"NY FED STAFF NOWCAST — struttura lunga")
    print(f"righe: {len(df)}   date: {df['forecast_date'].nunique()}   "
          f"trimestri target: {df['target_quarter'].nunique()}")
    print(f"campione: {df['forecast_date'].min():%Y-%m-%d} -> "
          f"{df['forecast_date'].max():%Y-%m-%d}")
    print(f"file: {', '.join(sorted(df['source_file'].unique()))}")
    print("\nrighe per etichetta:")
    print(df["horizon_label"].value_counts().to_string())

    print(_section("BUCHI NELLA GRIGLIA SETTIMANALE (atteso: solo 2021-2022)"))
    print(coverage_gaps(df).to_string(index=False) or "(nessuno)")

    print(_section("DATE FUORI GRIGLIA (non venerdi')"))
    off = off_grid_dates(df)
    print(off.to_string(index=False) if len(off) else "(nessuna)")

    print(_section("ALLINEAMENTO: etichetta Fed -> settimana del mio metro"))
    print(check_alignment(df).to_string(index=False))

    print(_section("ESEMPIO VERIFICATO: 2008Q4 (righe scritte sotto 2009Q1)\n"
                   "   lo spostamento di un trimestre, nessun riscalamento, e "
                   "il 30 gennaio fuori finestra"))
    we = check_worked_example(df)
    print(we.to_string(index=False))
    print("ESITO:", "tutto come atteso" if we["ok"].all()
          else "DISCORDANZA — l'allineamento non e' piu' quello documentato")

    print(_section("ULTIMA STIMA PRIMA DEL RILASCIO (ultime 12)"))
    lb = last_before_release(df)
    print(lb.tail(12).to_string(index=False))

    if a.quarters:
        for q in a.quarters:
            print(_section(f"DETTAGLIO {q}"))
            sub = df[df["target_quarter"] == q]
            print(sub[["forecast_date", "horizon_label", "horizon_week",
                       "nowcast_bea", "pre_release"]].to_string(index=False))


__all__ = [
    "normalise_quarter", "read_sheet", "load_wide", "load_long",
    "last_before_release", "at_horizon",
    "coverage_gaps", "off_grid_dates", "check_alignment", "check_worked_example",
]


if __name__ == "__main__":
    main()
