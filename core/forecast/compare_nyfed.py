"""
core/forecast/compare_nyfed.py

I MIEI MODELLI CONTRO LA NEW YORK FED (e contro i benchmark).

Lettore puro, a valle di tutto: legge il CSV lungo di `weekly_nowcast.py` e la
tabella lunga di `nyfed_nowcast.py`, e li mette sulla stessa riga.  Non stima
niente, non tocca il motore, non entra nella figura 8a — i dati NY Fed restano
in `data/` e non sporcano la 8a.

IL CONFRONTO SI FA SULL'ULTIMA STIMA PRIMA DEL RILASCIO
-------------------------------------------------------
Per ogni trimestre target si prende, di ciascun metodo, la stima piu' informata
che esisteva PRIMA che il BEA pubblicasse: il fondo del backcast.  E' l'unico
punto in cui i due esercizi rispondono alla stessa domanda — "cosa sapevi del
PIL il venerdi' prima che uscisse" — e quindi l'unico in cui la differenza fra
i numeri e' una differenza di modello e non di informazione.

Per i miei metodi e' la riga con `horizon_week` massima (17: l'ultimo venerdi'
prima di fine_trimestre + 28g).  Per la Fed e' `last_before_release()`, che
scarta i backcast pubblicati dopo l'advance del BEA.  Sul campione 2007-2010 le
due cose cadono sullo STESSO venerdi', e la tabella 4 lo fa vedere data per
data invece di chiederlo per fede: se un giorno divergessero, la colonna
`stessa_data` lo direbbe prima che la conclusione sia gia' stata scritta.

IL RIFERIMENTO
--------------
Il realizzato e' `realizzato_bea`, cioe' GDPC1 nella versione corrente del mio
dataset, gia' allegato a ogni riga del CSV settimanale.  Prenderlo da li' e non
da una fonte terza garantisce che io, la Fed, l'AR(2) e la media espandente
siamo punteggiati contro lo STESSO numero: un realizzato diverso per la Fed
renderebbe il confronto incomparabile senza che si veda.

Nota: il realizzato e' l'ultima revisione disponibile, non la stima advance che
la Fed cercava di indovinare in tempo reale.  Penalizza tutti allo stesso modo,
ma piu' chi era vicino all'advance e lontano dalla revisione finale — nel 2008
le revisioni sono state grandi.

PERCHE' `fed_overlap` NON DEVE COINCIDERE CON LORO (e perche' e' un bene)
------------------------------------------------------------------------
La spec `fed_overlap` e' costruita per sovrapporsi alla loro: quattro fattori,
un globale su cui carica ogni serie piu' tre locali (soft/real/labor), con ogni
serie che carica su G + il suo blocco e i sette prezzi solo su G.  E' la
Tabella 3 di Bok, Caratelli, Giannone, Sbordone e Tambalotti (2018), e anche il
pannello `final` nasce da li': 37 serie, 34 mensili e 3 trimestrali.

Detto questo, i due esercizi NON possono dare lo stesso numero, e aspettarselo
sarebbe un errore di lettura.  Le differenze che restano sono almeno quattro:

  1. LE SERIE non coincidono una a una.  Il pannello replica la loro tabella,
     ma la ricostruisce da FRED (32 serie) e da un file ISM (4): dove una serie
     non e' disponibile o e' definita diversamente si e' scelto il sostituto
     piu' vicino, non l'identico.
  2. LE QUATTRO ISM SONO DI SECONDA MANO, ed e' la differenza di qualita' piu'
     concreta che questo esercizio si porta dietro.  ISM_PMI, ISM_PRICES,
     ISM_EMP e ISM_NMI (`source = manual_xlsx` nei metadati) non vengono dal
     sito ISM, che le distribuisce a pagamento, ma da una ricostruzione via
     investing.com raccolta in `data/raw/ISM_series_pulite.xlsx`.  Sono quindi
     quasi certamente diverse dalle loro nelle cifre decimali e forse nelle
     revisioni.  Non e' un dettaglio marginale: le quattro ISM sono l'intero
     blocco Soft (`GS`), cioe' l'informazione tempestiva che nelle prime
     settimane del trimestre fa muovere il nowcast quando i dati hard non sono
     ancora usciti.  Un handicap sull'input, dichiarato.
  3. IL VINTAGE.  Io lavoro in pseudo-real-time: calendario di pubblicazione
     vero, valori nella versione corrente (revisionata).  Sul loro lato la nota
     del file dice, alla lettera:

         "Forecasts for GDP growth in quarters 2002Q1 through 2015Q4 are
          historical reconstructions based on real-time data"

     "based on real-time data" suggerisce vintage veri, ma resta il fatto —
     indipendente dal vintage — che si tratta di un esercizio RETROSPETTIVO:
     specificazione, selezione delle serie e parametrizzazione sono state
     fissate dopo aver visto il periodo, quindi quei numeri non sono un track
     record out-of-sample per nessuna definizione utile.  Prima di scrivere in
     tesi che usarono dati revisionati serve una fonte, perche' il file loro
     afferma il contrario: la nota qui sopra e' l'unica evidenza che questo
     repo contiene.
  4. LA STIMA.  Dinamica idiosincratica, trattamento del bordo frastagliato,
     inizializzazione e criterio di arresto sono i miei, non i loro.
  5. LE VARIANTI.  `gaussian`, `student_t`, `_ar1`, `_shared` non hanno
     corrispettivo nel loro modello: sono il contributo di questa tesi, non una
     replica.

Da qui la lettura corretta dei numeri.  Che `gaussian_ar1` non riproduca la
curva della Fed non e' un difetto da spiegare: con serie diverse, vintage
diversi e stima diversa, riprodurla sarebbe stato sospetto.  Quello che conta e'
l'ORDINE DI GRANDEZZA dell'errore, ed e' li' che il risultato e' informativo:
sul campione 2007Q4-2010Q1 le celle a quattro fattori correlano 0.94-0.96 con
loro e stanno entro l'8% del loro RMSE.  Due pipeline costruite in modo
indipendente, che partono da dati non identici, arrivano quasi nello stesso
punto: e' la prova che la struttura a quattro fattori porta il segnale e che la
catena vintage -> pannello -> EM -> nowcast non ha una falla grossa.  E' un test
di validita' esterna, e va scritto come tale.

COME SCRIVERE IL DIVARIO SENZA SBAGLIARE
----------------------------------------
La frase difendibile e' quella del punto 2, ed e' circoscritta: le quattro ISM
sono replicate da una fonte secondaria e sono quasi certamente diverse dalle
originali, quindi il blocco Soft entra nel modello degradato.  E' verificabile
nei metadati (`source = manual_xlsx`) e non richiede assunzioni su di loro.

La frase da NON scrivere e' quella larga — "loro hanno dati proprietari migliori
dei nostri" — riferita all'intero pannello: le altre 33 serie sono FRED
pubbliche, prese dalla stessa tabella che loro documentano, e li' non c'e'
nessun vantaggio informativo da invocare.  Allargare l'argomento dalle 4 ISM a
tutte e 37 le serie e' il modo piu' facile di farsi smontare la conclusione da
chi legge, perche' basta aprire la loro Tabella 3 per vedere che sono pubbliche.

CHE COSA SI LEGGE
-----------------
    1. RMSE/MAE/bias/correlazione di ciascuno contro il realizzato
    2. io contro la Fed: differenza media, correlazione, quante volte la batto
    3. la Fed contro il realizzato, trimestre per trimestre
    4. verifica dell'allineamento: le date usate, mie contro loro

AVVERTENZA SUL CAMPIONE
-----------------------
Con una dozzina di trimestri — e per giunta tutti dentro la Grande Recessione —
queste sono differenze descrittive, non risultati con un errore standard che
valga la pena scrivere.  Nessun Diebold-Mariano: su 12 osservazioni non
distinguerebbe niente.  La tabella serve a vedere l'ordine di grandezza e la
direzione, e a controllare che la meccanica funzioni prima della run lunga.

Inoltre i valori Fed fino a 2015Q4 sono RICOSTRUZIONI su dati real-time, non
pubblicazioni avvenute davvero (lo dichiara il loro file); il mio esercizio e'
pseudo-real-time.  Sono due pseudo-real-time, il che rende il confronto equo ma
non lo rende un confronto con "la previsione che la Fed fece quel giorno".

Uso
---
    python -m core.forecast.compare_nyfed
    python -m core.forecast.compare_nyfed --spec fed_overlap
    python -m core.forecast.compare_nyfed --csv output/forecast_weekly/csv/xxx.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.forecast.compute_metrics import (
    core_coverage_quarters, load_long as load_mine, standard_axis,
)
from core import output_layout as layout
from core.forecast import figures as fg
from core.forecast.nyfed_nowcast import last_before_release, load_long as load_nyfed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Dove finiscono report, CSV e figura RMSE di questo confronto: nella cartella
#: `rmse/` DELLA SPEC disegnata, non piu' in una cartella `nyfed/` a parte.
#: Il confronto con la Fed e' una lettura dell'RMSE di quella spec, e teneva
#: separate cose che si guardano insieme — era una delle due sorgenti di
#: dispersione delle figure.  Il percorso lo decide `output_layout`.

#: Il nome con cui la Fed compare fra i metodi.
_NYFED = "nyfed"

#: I due benchmark, in coda alle tabelle.
_BENCH = ("ar2", "mean")


# ─── Le stime finali, tutte sulla stessa riga ─────────────────────────────────

def my_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Di ogni (metodo, trimestre), la stima all'orizzonte piu' profondo: l'ultimo
    venerdi' prima del rilascio del PIL.

    Non serve filtrare per `pre_release`: il ciclo settimanale genera solo
    target in volo (`targets_in_flight` taglia a `gdp_release_date`), quindi la
    settimana massima e' gia' l'ultima pre-rilascio per costruzione.
    """
    idx = df.groupby(["metodo", "target_quarter"])["horizon_week"].idxmax()
    out = df.loc[idx, ["metodo", "target_quarter", "as_of", "horizon_week",
                       "nowcast_bea", "realizzato_bea"]].copy()
    out["as_of"] = pd.to_datetime(out["as_of"])
    return out.reset_index(drop=True)


def nyfed_final(quarters: list[str], realised: dict[str, float]) -> pd.DataFrame:
    """La stessa cosa per la Fed, con il realizzato preso dal MIO dataset."""
    fed = last_before_release(load_nyfed(), quarters=quarters).copy()
    fed["metodo"] = _NYFED
    fed["realizzato_bea"] = fed["target_quarter"].map(realised)
    return fed[["metodo", "target_quarter", "as_of", "horizon_week",
                "nowcast_bea", "realizzato_bea",
                "gdp_release_date"]].reset_index(drop=True)


def _last_week_before_release(panel: pd.DataFrame) -> pd.Series:
    """
    Per ogni trimestre, l'inizio della SETTIMANA DI RILASCIO: la finestra
    `[rilascio - 7g, rilascio)` dentro cui deve cadere l'ultima stima di
    chiunque voglia entrare nel confronto.
    """
    rel = (panel.dropna(subset=["gdp_release_date"])
           .groupby("target_quarter")["gdp_release_date"].first())
    return pd.to_datetime(rel) - pd.Timedelta(days=7)


def aligned_sample(panel: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    I trimestri su cui il confronto e' lecito, e il registro di quelli scartati.

    Due condizioni, entrambe necessarie:

    1.  TUTTI hanno una stima finale e un realizzato.  Un RMSE della Fed
        calcolato su un sottoinsieme diverso dal mio non e' confrontabile col
        mio: se la Fed saltasse il 2008Q4 e io no, il suo RMSE scenderebbe per
        il motivo sbagliato.

    2.  Tutte le stime finali cadono nella SETTIMANA DEL RILASCIO, cioe' in
        `[rilascio - 7g, rilascio)`.  E' la condizione che il campione grezzo
        NON garantisce, ed e' quella che conta: la mia run si ferma al
        2010-06-30, quindi per 2010Q2 la mia ultima stima e' del 25 giugno
        (settimana 13) mentre la Fed arriva al 23 luglio (settimana 17).
        Metterle sulla stessa riga confronterebbe un nowcast di meta' trimestre
        con un backcast maturo — quattro settimane di dati in piu' da una parte
        sola — e il numero che ne esce sembra un risultato di modello mentre e'
        un artefatto di calendario.  E' esattamente il "sbagliare in silenzio"
        che la tabella 4 esiste per rendere rumoroso.

    La finestra e' una settimana e non l'uguaglianza esatta delle date perche'
    la Fed pubblica di giovedi' nelle settimane di festivita' (3 casi su 1226):
    pretendere lo stesso giorno butterebbe via trimestri validi per uno scarto
    di 24 ore, che a livello di settimana-orizzonte non esiste.
    """
    n_metodi = panel["metodo"].nunique()
    counts = panel.groupby("target_quarter")["metodo"].nunique()
    floor = _last_week_before_release(panel)
    last = panel.groupby(["target_quarter", "metodo"])["as_of"].max()

    rows = []
    for q in sorted(panel["target_quarter"].unique()):
        completo = int(counts.get(q, 0)) == n_metodi
        lo = floor.get(q, pd.NaT)
        dates = last.loc[q] if q in last.index.get_level_values(0) else pd.Series(dtype="datetime64[ns]")
        allineato = bool(pd.notna(lo)) and bool((dates >= lo).all())
        spread = int((dates.max() - dates.min()).days) if len(dates) else 0
        rows.append({
            "trimestre": q,
            "n_metodi": int(counts.get(q, 0)),
            "completo": completo,
            "allineato": allineato,
            "spread_gg": spread,
            "tenuto": completo and allineato,
            "motivo": ("" if completo and allineato
                       else "metodi mancanti" if not completo
                       else "stime finali fuori dalla settimana di rilascio"),
        })
    registro = pd.DataFrame(rows)
    return registro.loc[registro["tenuto"], "trimestre"].tolist(), registro


def build_panel(df_mine: pd.DataFrame,
                specs: list[str] | None = None
                ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Il pannello del confronto: una riga per (metodo, trimestre), con la Fed
    dentro come un metodo qualsiasi, ristretto al campione allineato.
    """
    mine = my_final(df_mine)
    if specs:
        keep = mine["metodo"].str.split("/").str[0].isin(list(specs))
        keep |= mine["metodo"].isin(_BENCH)
        mine = mine[keep]

    realised = (df_mine.dropna(subset=["realizzato_bea"])
                .groupby("target_quarter")["realizzato_bea"].first().to_dict())
    fed = nyfed_final(sorted(mine["target_quarter"].unique()), realised)

    panel = pd.concat([mine, fed], ignore_index=True)
    panel = panel[panel["nowcast_bea"].notna() & panel["realizzato_bea"].notna()]
    # La data di rilascio vale per il trimestre, non per il metodo: la porta
    # solo la Fed, e va propagata a tutte le righe dello stesso trimestre.
    rel = (panel.dropna(subset=["gdp_release_date"])
           .groupby("target_quarter")["gdp_release_date"].first())
    panel["gdp_release_date"] = panel["target_quarter"].map(rel)

    sample, registro = aligned_sample(panel)
    panel = panel[panel["target_quarter"].isin(sample)].copy()

    panel["errore"] = panel["nowcast_bea"] - panel["realizzato_bea"]
    return panel.reset_index(drop=True), sample, registro


# ─── Le quattro tabelle ───────────────────────────────────────────────────────

def _corr(a, b) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _order(methods) -> list[str]:
    """Modelli in ordine alfabetico, poi la Fed, poi i due benchmark."""
    m = list(methods)
    tail = [x for x in (_NYFED,) + _BENCH if x in m]
    return sorted(x for x in m if x not in tail) + tail


def table_accuracy(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Tabella 1 — ciascun metodo contro il realizzato, sull'ultima stima
    pre-rilascio.  `RMSE_rel_fed` < 1 significa battere la NY Fed.
    """
    rows = []
    for m, g in panel.groupby("metodo"):
        e = g["errore"].to_numpy(float)
        rows.append({
            "metodo": m,
            "n": len(g),
            "settimana": float(g["horizon_week"].mean()),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAE": float(np.mean(np.abs(e))),
            "Bias": float(np.mean(e)),
            "corr": _corr(g["nowcast_bea"], g["realizzato_bea"]),
        })
    tab = pd.DataFrame(rows)
    for name, col in ((_NYFED, "RMSE_rel_fed"), ("ar2", "RMSE_rel_ar2")):
        ref = tab.loc[tab["metodo"] == name, "RMSE"]
        tab[col] = tab["RMSE"] / float(ref.iloc[0]) if len(ref) else float("nan")
    tab["__o"] = tab["metodo"].map({m: i for i, m in enumerate(_order(tab["metodo"]))})
    return tab.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def table_vs_fed(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Tabella 2 — io contro la Fed, trimestre per trimestre appaiato.

    `diff_media`   media di (mio - fed): dove sto sistematicamente rispetto a
                   loro.  Positivo = sono piu' ottimista.
    `corr_fed`     quanto le due serie si muovono insieme.  Alta con RMSE simile
                   = stesso segnale, stesso errore; bassa = due modelli che
                   guardano cose diverse, e allora la media dei due varrebbe
                   piu' di ciascuno.
    `quota_meglio` in quanti trimestri il mio errore assoluto e' minore del loro.
                   Su una dozzina di trimestri e' un conteggio, non un test.
    """
    fed = panel[panel["metodo"] == _NYFED].set_index("target_quarter")
    if fed.empty:
        return pd.DataFrame()

    rows = []
    for m, g in panel.groupby("metodo"):
        if m == _NYFED:
            continue
        g = g.set_index("target_quarter").reindex(fed.index)
        d = g["nowcast_bea"] - fed["nowcast_bea"]
        rows.append({
            "metodo": m,
            "n": int(d.notna().sum()),
            "diff_media": float(d.mean()),
            "diff_sd": float(d.std(ddof=1)),
            "corr_fed": _corr(g["nowcast_bea"], fed["nowcast_bea"]),
            "RMSE": float(np.sqrt(np.mean(g["errore"].to_numpy(float) ** 2))),
            "RMSE_fed": float(np.sqrt(np.mean(fed["errore"].to_numpy(float) ** 2))),
            "quota_meglio": float(np.mean(
                np.abs(g["errore"].to_numpy(float))
                < np.abs(fed["errore"].to_numpy(float)))),
        })
    tab = pd.DataFrame(rows)
    tab["__o"] = tab["metodo"].map({m: i for i, m in enumerate(_order(tab["metodo"]))})
    return tab.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def table_fed_by_quarter(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Tabella 3 — quanto vale il metro: la Fed trimestre per trimestre contro il
    realizzato, con accanto il migliore e il peggiore dei miei per quel
    trimestre.  E' qui che si vede se un RMSE aggregato nasce da un errore
    diffuso o da un solo trimestre andato male.
    """
    fed = panel[panel["metodo"] == _NYFED].set_index("target_quarter")
    mine = panel[panel["metodo"] != _NYFED]
    mine = mine[~mine["metodo"].isin(_BENCH)]

    rows = []
    for q in fed.index:
        g = mine[mine["target_quarter"] == q]
        best = g.loc[g["errore"].abs().idxmin()] if len(g) else None
        worst = g.loc[g["errore"].abs().idxmax()] if len(g) else None
        rows.append({
            "trimestre": q,
            "realizzato": float(fed.loc[q, "realizzato_bea"]),
            "fed": float(fed.loc[q, "nowcast_bea"]),
            "err_fed": float(fed.loc[q, "errore"]),
            "mio_migliore": None if best is None else best["metodo"],
            "err_migliore": None if best is None else float(best["errore"]),
            "mio_peggiore": None if worst is None else worst["metodo"],
            "err_peggiore": None if worst is None else float(worst["errore"]),
        })
    return pd.DataFrame(rows)


def table_alignment(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Tabella 4 — LA VERIFICA.  Per ogni trimestre: la data della mia ultima
    stima, la data dell'ultima stima Fed, la data di rilascio del PIL, e se le
    prime due coincidono.

    Se `stessa_data` fosse False da qualche parte, il confronto di quel
    trimestre non e' fra due modelli ma fra due istanti diversi, e i giorni di
    scarto dicono quanto vale la differenza.  E' il controllo che rende la
    tabella 1 leggibile: senza, un RMSE piu' basso potrebbe voler dire soltanto
    "ha guardato una settimana in piu' di dati".
    """
    fed = panel[panel["metodo"] == _NYFED].set_index("target_quarter")
    mine = panel[panel["metodo"] != _NYFED]

    rows = []
    for q, g in mine.groupby("target_quarter"):
        if q not in fed.index:
            continue
        d_mine = g["as_of"].max()
        d_fed = pd.Timestamp(fed.loc[q, "as_of"])
        rows.append({
            "trimestre": q,
            "mia_as_of": d_mine.date(),
            "mia_sett": int(g.loc[g["as_of"].idxmax(), "horizon_week"]),
            "fed_as_of": d_fed.date(),
            "fed_sett": int(fed.loc[q, "horizon_week"]),
            "rilascio_pil": pd.Timestamp(fed.loc[q, "gdp_release_date"]).date(),
            "scarto_gg": int((d_mine - d_fed).days),
            "stessa_data": bool(d_mine == d_fed),
        })
    return pd.DataFrame(rows)


# ─── FASE 2: l'RMSE per orizzonte ─────────────────────────────────────────────
#
# LA CURVA CHE SI VUOLE VEDERE
# ----------------------------
# Un metodo che sta imparando dai dati ha una curva DECRESCENTE: piu' ci si
# avvicina al rilascio, piu' e' uscita informazione, piu' l'RMSE scende.  Una
# curva piatta e' un metodo che non ascolta il pannello (l'AR(2) e la media
# espandente devono essere piatti: non hanno un pannello da ascoltare, e la loro
# piattezza e' il controllo che la meccanica del grafico funzioni).  Una curva
# che RISALE avvicinandosi al rilascio e' un sintomo, non un risultato: vuol
# dire che l'ultima informazione entrata ha peggiorato la stima.
#
# L'ASSE X: SI RAGGRUPPA SU `horizon_week`, SI ETICHETTA IN SETTIMANE AL RILASCIO
# -------------------------------------------------------------------------------
# Le due ancore non sono la stessa cosa e vale la pena dirlo dove sta il codice.
#
#   `horizon_week`      settimane dall'INIZIO del trimestre target.  E' discreta,
#                       intera, e vale identica per me e per la Fed: e' quindi
#                       l'unica su cui i due RMSE si possono appaiare punto per
#                       punto.  E' la variabile di RAGGRUPPAMENTO.
#   settimane al rilascio  (gdp_release_date - as_of)/7.  E' quella che si legge
#                       ("quanto manca al numero vero"), ma NON e' costante a
#                       parita' di `horizon_week`: i trimestri hanno 90, 91 o 92
#                       giorni e i venerdi' ci cadono dentro in posizioni diverse,
#                       quindi alla settimana 17 mancano 5 giorni al rilascio per
#                       2008Q4 e 1 solo per 2007Q2.  E' la variabile di ETICHETTA.
#
# Il jitter fra le due sta sotto la settimana e finisce in didascalia.  Usare la
# seconda per raggruppare spargerebbe i punti su ascisse leggermente diverse per
# ogni trimestre e il grafico diventerebbe una nuvola invece di otto curve.
#
# LO STESSO CAMPIONE A OGNI ORIZZONTE
# -----------------------------------
# Un punto viene disegnato solo se quel metodo, a quella settimana, ha TUTTI i
# trimestri del campione.  Senza questa regola la curva scenderebbe anche solo
# perche' agli orizzonti profondi sopravvivono i trimestri facili: sarebbe un
# effetto di composizione travestito da apprendimento.  Il prezzo e' che la curva
# della Fed comincia alla settimana -4 invece che alla -12, perche' loro non
# pubblicano un forecast prima di ~5 settimane dall'inizio del trimestre.  Non e'
# un dato mancante, e' il loro prodotto che comincia li'.
#
# I DUE FILTRI, E COSA FANNO QUI
# ------------------------------
#   `pre_release`       ATTIVO e vincolante.  I backcast NY Fed pubblicati dopo
#                       il rilascio del PIL restano fuori anche dalla figura,
#                       esattamente come dalle tabelle: altrimenti il punto piu'
#                       a destra della loro curva — proprio quello che si guarda
#                       per primo — sarebbe l'unico calcolato su una settimana
#                       di dati mensili che io non avevo.
#   `is_reconstruction` ATTIVO come BANDIERINA, non come esclusione, e la
#                       differenza va detta.  Tutti e dieci i trimestri disegnati
#                       sono <= 2015Q4, quindi ricostruzioni Fed su dati
#                       real-time: filtrarli via svuoterebbe la figura invece di
#                       ripulirla.  Il flag viaggia con ogni riga
#                       (`nyfed_nowcast.load_long`) e la natura ricostruita del
#                       campione sta scritta nel report, dove serve per leggere
#                       il confronto — non e' un difetto da nascondere ma un
#                       fatto da dichiarare.  Sulla run lunga, quando il campione
#                       arrivera' a coprire anche il post-2016, quel flag
#                       diventera' una partizione utile: ricostruito contro
#                       real-time vero, e si vedra' se la Fed perde qualcosa
#                       passando dall'uno all'altro.

#: I colori dei modelli: l'ordine MATLAB, lo stesso della 8a.
_MODEL_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"]

#: Fed e benchmark non sono modelli miei e non devono sembrarlo: neri/grigi, e
#: distinti dal tratto piu' che dal colore.
_REF_STYLE = {
    _NYFED: {"color": "black", "linestyle": "-", "linewidth": 2.6,
             "label": "NY Fed Staff Nowcast"},
    "ar2": {"color": "#7A7A7A", "linestyle": "--", "linewidth": 1.8,
            "label": "AR(2)"},
    "mean": {"color": "#7A7A7A", "linestyle": ":", "linewidth": 1.8,
             "label": "Expanding mean"},
}

#: Le tre fasi sulla scala `horizon_week`: (da, a, etichetta, colore).
#: Gli estremi sono INCLUSIVI; il disegno li allarga di mezza settimana per
#: lato, cosi' le bande si toccano fra i punti interi e non sopra di essi.
_PHASE_BANDS = [
    (-12, 0, "forecast\n(quarter not started)", "#EAF0F6"),
    (1, 13, "nowcast\n(current quarter)", "#F5EFE4"),
    (14, 17, "backcast\n(quarter closed,\nGDP not released)", "#EDE6F0"),
]

_FIG_SIZE = (11.0, 6.4)
_TICK_EVERY = 3


def _serif() -> dict:
    return {"font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "mathtext.fontset": "dejavuserif"}


def _apply_axes_style(ax) -> None:
    """Riquadro chiuso, tick interni sui quattro lati, niente griglia."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=9, width=0.8)
    ax.grid(False)
    ax.set_facecolor("white")


def horizon_panel(df_mine: pd.DataFrame, sample: list[str],
                  spec: str = "fed_overlap") -> tuple[pd.DataFrame, list[str]]:
    """
    Il pannello lungo che sta sotto la figura: una riga per (metodo, settimana)
    con l'RMSE, piu' il campione effettivamente usato.

    Colonne: `metodo`, `horizon_week`, `n_trimestri`, `RMSE`, `sett_al_rilascio`
    (media, per l'etichetta dell'asse), `pieno` (True se `n_trimestri` copre
    tutto il campione: solo questi punti vengono disegnati).

    IL CAMPIONE DELLA FIGURA E' PIU' STRETTO DI QUELLO DELLE TABELLE
    ---------------------------------------------------------------
    Le tabelle guardano un punto solo — l'ultima stima prima del rilascio — e
    per quello bastano 12 trimestri: 2007Q2 e 2007Q3 hanno un backcast finale
    valido anche se la mia run, partendo dal 2007-07-01, non ne ha mai visto la
    fase di forecast.  La figura invece percorre TUTTO l'asse, e un trimestre
    che esiste solo da meta' asse in poi farebbe scendere la curva a sinistra
    per composizione e non per apprendimento.  Si tengono quindi solo i
    trimestri con la copertura piena sul NUCLEO dell'asse
    (`core_coverage_quarters`, che spiega perche' il nucleo e non l'unione), e
    quanti sono lo si dichiara nel titolo invece di lasciarlo indovinare.

    I miei metodi arrivano dal CSV settimanale, che porta gia' `horizon_week` e
    il realizzato riga per riga.  La Fed arriva da `nyfed_nowcast.load_long()`
    con il filtro `pre_release` attivo — gli stessi backcast scartati nelle
    tabelle restano scartati qui, altrimenti il punto piu' a destra della loro
    curva sarebbe l'unico calcolato su un'informazione che io non avevo.
    """
    # IL FILTRO PER SPEC VIENE PRIMA DEL CAMPIONE, e l'ordine non e' un
    # dettaglio.  Calcolando il campione sul frame di TUTTE le spec, un
    # trimestre risultava coperto perche' lo copriva QUALCHE metodo: entrava in
    # `n_target`, e poi i metodi che non ce l'avevano fallivano `pieno` a ogni
    # settimana e sparivano dal grafico.  Sul 2024-2025 restava disegnato un
    # metodo solo su otto.  Il campione va deciso sui metodi che si disegnano.
    keep = df_mine["metodo"].str.split("/").str[0] == spec
    keep |= df_mine["metodo"].isin(_BENCH)
    mine = df_mine[keep].copy()

    sample = core_coverage_quarters(mine, sample)
    n_target = len(sample)
    if not sample:
        vuoto = pd.DataFrame(columns=["metodo", "horizon_week", "n_trimestri",
                                      "RMSE", "sett_al_rilascio", "pieno"])
        return vuoto, []

    mine = mine[mine["target_quarter"].isin(sample)]
    mine["as_of"] = pd.to_datetime(mine["as_of"])
    mine["gdp_release_date"] = pd.to_datetime(mine["gdp_release_date"])

    realised = (df_mine.dropna(subset=["realizzato_bea"])
                .groupby("target_quarter")["realizzato_bea"].first().to_dict())

    fed = load_nyfed()
    fed = fed[fed["target_quarter"].isin(sample) & fed["pre_release"]].copy()
    # LA FED SI RITAGLIA SULLA STESSA FINESTRA, e prima non succedeva: `df_mine`
    # arriva gia' tagliato dal chiamante, la Fed no.  Sul 2024-2025 la loro
    # curva aveva percio' dieci trimestri per settimana dove i miei metodi ne
    # avevano otto — punti calcolati su un campione piu' largo del mio, cioe'
    # esattamente il confronto per composizione che tutto il resto del modulo
    # evita.  I confini si prendono da `mine`, che E' la finestra.
    lo, hi = mine["as_of"].min(), mine["as_of"].max()
    fed = fed[(fed["forecast_date"] >= lo) & (fed["forecast_date"] <= hi)]
    # Un venerdi' spostato di festivita' puo' far cadere due righe dello stesso
    # trimestre nella stessa settimana: vince la piu' recente, come altrove.
    fed = (fed.sort_values("forecast_date")
              .drop_duplicates(["target_quarter", "horizon_week"], keep="last"))
    fed["metodo"] = _NYFED
    fed["realizzato_bea"] = fed["target_quarter"].map(realised)
    fed["errore"] = fed["nowcast_bea"] - fed["realizzato_bea"]
    fed = fed.rename(columns={"forecast_date": "as_of"})

    cols = ["metodo", "target_quarter", "horizon_week", "as_of",
            "gdp_release_date", "errore"]
    both = pd.concat([mine[cols], fed[cols]], ignore_index=True)
    both = both[both["errore"].notna()]

    rows = []
    for (m, w), g in both.groupby(["metodo", "horizon_week"]):
        e = g["errore"].to_numpy(float)
        n = int(g["target_quarter"].nunique())
        gg = (g["gdp_release_date"] - g["as_of"]).dt.days.to_numpy(float)
        rows.append({
            "metodo": m,
            "horizon_week": int(w),
            "n_trimestri": n,
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "sett_al_rilascio": float(np.mean(gg) / 7.0),
            "pieno": n == n_target,
        })
    return (pd.DataFrame(rows)
            .sort_values(["metodo", "horizon_week"]).reset_index(drop=True)), sample


def figure_rmse_by_horizon(panel_h: pd.DataFrame, sample: list[str],
                           spec: str, out_path: str) -> str:
    """
    Il grafico: RMSE medio sui trimestri del campione, settimana per settimana,
    con le tre fasi come bande di sfondo.

    QUANTI TRIMESTRI CI SONO SOTTO SI LEGGE NEL TITOLO, e cambia la lettura.
    Con dieci ogni punto e' un RMSE su dieci errori: l'ordinamento fra curve
    vicine puo' ribaltarsi per un solo trimestre andato male, e le oscillazioni
    da una settimana all'altra sono in buona parte rumore di campionamento.
    Con quaranta e passa — le tre passate lunghe — il confronto fra curve
    regge; sugli zoom corti resta indicativo, e la didascalia lo dichiara.
    Quello che si guarda comunque per primo: che AR(2) e media espandente siano
    piatti (non hanno pannello da ascoltare) e che le curve del pannello
    scendano verso destra.
    """
    plotted = panel_h[panel_h["pieno"]]
    models = sorted(m for m in plotted["metodo"].unique()
                    if m not in _REF_STYLE)
    weeks = np.sort(plotted["horizon_week"].unique())

    # Etichetta dell'asse: media, sul campione, delle settimane che mancavano al
    # rilascio a quella `horizon_week`.  E' la conversione fra le due ancore.
    lab = (plotted.groupby("horizon_week")["sett_al_rilascio"].mean().round().astype(int))

    with plt.rc_context(_serif()):
        fig, ax = plt.subplots(figsize=_FIG_SIZE)
        _apply_axes_style(ax)

        # Scala y sui DATI, non ancorata a zero.  Con `ylim(0, ...)` e RMSE fra
        # 2.4 e 5.5 il 40% del riquadro restava vuoto e le curve si
        # schiacciavano l'una sull'altra proprio dove si devono distinguere.
        # In cima resta la fascia che ospita le etichette delle tre fasi.
        y_lo = float(plotted["RMSE"].min())
        y_hi = float(plotted["RMSE"].max())
        span = (y_hi - y_lo) or max(abs(y_hi), 1.0)
        y0 = max(0.0, y_lo - 0.10 * span)      # sotto zero l'RMSE non esiste
        ymax = y_hi + 0.22 * span              # spazio per le etichette di fase

        for lo, hi, name, colour in _PHASE_BANDS:
            ax.axvspan(lo - 0.5, hi + 0.5, color=colour, zorder=0, linewidth=0)
            ax.text((lo + hi) / 2.0, ymax - 0.015 * (ymax - y0), name,
                    ha="center", va="top", fontsize=8.5, color="#4A4A4A",
                    linespacing=1.25, zorder=1)
        for _, hi, _, _ in _PHASE_BANDS[:-1]:
            ax.axvline(hi + 0.5, color="white", linewidth=1.2, zorder=1)

        for i, m in enumerate(models):
            g = plotted[plotted["metodo"] == m].sort_values("horizon_week")
            ax.plot(g["horizon_week"], g["RMSE"],
                    color=_MODEL_COLORS[i % len(_MODEL_COLORS)],
                    linewidth=1.9, marker="o", markersize=3.2,
                    label=m.split("/", 1)[-1], zorder=3)
        for m, st in _REF_STYLE.items():
            g = plotted[plotted["metodo"] == m].sort_values("horizon_week")
            if g.empty:
                continue
            ax.plot(g["horizon_week"], g["RMSE"], zorder=4, **st)

        ax.set_xlim(weeks.min() - 0.5, weeks.max() + 0.5)
        ax.set_ylim(y0, ymax)
        ticks = [w for w in weeks if (weeks.max() - w) % _TICK_EVERY == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(lab[w]) for w in ticks])
        ax.set_xlabel("Weeks to the GDP release  "
                      "(release on the right; below, the week of the quarter)",
                      fontsize=10)
        ax.set_ylabel("RMSE  (BEA percentage points)", fontsize=10)

        # Seconda scala, sotto: la variabile su cui si e' RAGGRUPPATO davvero.
        ax2 = ax.secondary_xaxis(-0.13)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{w:+d}" for w in ticks])
        ax2.tick_params(labelsize=8, direction="in", colors="#4A4A4A", width=0.8)
        for side in ax2.spines:
            ax2.spines[side].set_visible(False)

        ax.set_title(f"RMSE by horizon — spec {spec} — "
                     f"{sample[0]}–{sample[-1]} ({len(sample)} quarters)",
                     fontsize=12, pad=12)
        # Tolto lo zero dall'asse, l'angolo in basso a sinistra non e' piu'
        # vuoto per costruzione: la posizione si verifica invece di assumerla.
        # Il riquadro opaco resta — qui sotto la legenda ci sono le bande
        # colorate, e senza riquadro il testo si legge male.
        leg = fg._place_legend(fig, ax, ax.get_legend_handles_labels()[0],
                               ncol=2, fallback_anchor=(0.5, -0.22),
                               fallback_ncol=5)
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_edgecolor("#B0B0B0")
        leg.set_zorder(5)

        fed = plotted.loc[plotted["metodo"] == _NYFED, "horizon_week"]
        note = (
            f"RMSE over {len(sample)} quarters"
            + ("  —  SMALL SAMPLE: differences between neighbouring weeks are "
               "largely sampling noise." if len(sample) < 20 else ".")
            + "\nHorizon axis grouped on horizon_week (lower row), labelled in "
              "weeks to release (upper row): at a given week the release can be "
              "up to ~1 week nearer or further,\nbecause quarters have 90/91/92 "
              "days.  Each point uses all "
            f"{len(sample)} quarters, otherwise it is not drawn."
        )
        if len(fed):
            note += (f"  The NY Fed curve starts at week {int(fed.min()):+d}: "
                     "before that they publish no forecast.\nNY Fed backcasts "
                     "released after the GDP release are excluded.")
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.4,
                 color="#3A3A3A", linespacing=1.6)

        # bottom=0.34 e non 0.30: sotto il riquadro stanno TRE cose in fila —
        # asse secondario (-0.13), legenda di ripiego (-0.22) e nota a pie' di
        # figura — e con 0.30 la legenda finiva sulla prima riga della nota.
        fig.subplots_adjust(bottom=0.34, top=0.92, left=0.075, right=0.98)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)
    return out_path


# ─── Formattazione e CLI ──────────────────────────────────────────────────────

_FMT = {
    "RMSE": "{:.3f}", "RMSE_fed": "{:.3f}", "MAE": "{:.3f}", "Bias": "{:+.3f}",
    "corr": "{:+.3f}", "corr_fed": "{:+.3f}", "settimana": "{:.1f}",
    "RMSE_rel_fed": "{:.3f}", "RMSE_rel_ar2": "{:.3f}",
    "diff_media": "{:+.3f}", "diff_sd": "{:.3f}", "quota_meglio": "{:.2f}",
    "realizzato": "{:+.2f}", "fed": "{:+.2f}", "err_fed": "{:+.2f}",
    "err_migliore": "{:+.2f}", "err_peggiore": "{:+.2f}",
}


def _fmt(tab: pd.DataFrame) -> str:
    out = tab.copy()
    for col, f in _FMT.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    return out.to_string(index=False)


def _section(t: str) -> str:
    return "\n" + "=" * 92 + "\n" + t + "\n" + "=" * 92


def build_report(panel: pd.DataFrame, sample: list[str],
                 registro: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    tabs = {
        "accuracy": table_accuracy(panel),
        "vs_fed": table_vs_fed(panel),
        "fed_by_quarter": table_fed_by_quarter(panel),
        "alignment": table_alignment(panel),
        "campione": registro,
    }
    fuori = registro[~registro["tenuto"]]
    header = (
        "IO CONTRO LA NEW YORK FED — ultima stima prima del rilascio del PIL\n"
        "errore in punti BEA (nowcast - realizzato), realizzato = GDPC1 versione corrente\n"
        f"trimestri nel campione ({len(sample)}): {', '.join(sample)}\n"
        + ("esclusi: " + "; ".join(f"{r.trimestre} ({r.motivo})"
                                   for r in fuori.itertuples()) + "\n"
           if len(fuori) else "")
        + f"metodi ({panel['metodo'].nunique()}): "
          f"{', '.join(_order(panel['metodo'].unique()))}\n"
        "\n"
        "AVVERTENZA  campione corto e tutto dentro la Grande Recessione: sono\n"
        "differenze descrittive, non risultati con un errore standard.  I valori\n"
        "Fed fino a 2015Q4 sono ricostruzioni su dati real-time, non pubblicazioni\n"
        "avvenute davvero; il mio esercizio e' pseudo-real-time.  Confronto equo\n"
        "fra due pseudo-real-time, non fra un pseudo e un vero."
    )
    blocks = [
        header,
        _section("1. ACCURATEZZA contro il realizzato  "
                 "(RMSE_rel_fed < 1 batte la NY Fed)"),
        _fmt(tabs["accuracy"]),
        _section("2. IO CONTRO LA FED  "
                 "(diff = mio - fed; quota_meglio = trimestri in cui sbaglio meno)"),
        _fmt(tabs["vs_fed"]),
        _section("3. LA FED TRIMESTRE PER TRIMESTRE  (quanto vale il metro)"),
        _fmt(tabs["fed_by_quarter"]),
        _section("4. VERIFICA DELL'ALLINEAMENTO  "
                 "(stessa_data False = si confrontano istanti diversi)"),
        _fmt(tabs["alignment"]),
        _section("5. COMPOSIZIONE DEL CAMPIONE  (cosa e' entrato e cosa no)"),
        _fmt(registro),
    ]
    return "\n".join(blocks), tabs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Confronto fra i miei nowcast, la NY Fed e i benchmark.")
    p.add_argument("--csv", nargs="*", default=None,
                   help="CSV settimanali da leggere (default: tutti)")
    p.add_argument("--spec", nargs="*", default=None,
                   help="limita alle spec date (es. fed_overlap diag4)")
    p.add_argument("--out-dir", default=None,
                   help="default: output/forecast_weekly/dfm/<spec-figura>/rmse")
    p.add_argument("--figura", action="store_true",
                   help="disegna anche l'RMSE per orizzonte (fase 2)")
    p.add_argument("--spec-figura", default="fed_overlap",
                   help="la spec disegnata nella figura (default: fed_overlap)")
    p.add_argument("--window", default=None,
                   help="nome di una finestra di output_layout, es. 2007-2019: "
                        "ritaglia il campione e nomina i file con quel nome")
    a = p.parse_args()

    out_dir = a.out_dir or layout.dfm_rmse_dir(a.spec_figura)
    tag = f"_{a.window}" if a.window else ""

    mine, _, _ = load_mine(a.csv)
    if a.window:
        mine = layout.slice_window(mine, a.window, column="as_of")
        if mine.empty:
            raise SystemExit(f"Nessuna riga nella finestra {a.window} "
                             f"{layout.window(a.window)}.")
    panel, sample, registro = build_panel(mine, a.spec)
    if not sample:
        raise SystemExit("Nessun trimestre allineato: confronto impossibile.\n"
                         + registro.to_string(index=False))

    report, tabs = build_report(panel, sample, registro)
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"nyfed_report{tag}.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nscritto: {txt}")
    for name, t in tabs.items():
        path = os.path.join(out_dir, f"nyfed_{name}{tag}.csv")
        t.to_csv(path, index=False)
        print(f"scritto: {path}")
    path = os.path.join(out_dir, f"nyfed_panel_finale{tag}.csv")
    panel.to_csv(path, index=False)
    print(f"scritto: {path}")

    if a.figura:
        ph, sample_fig = horizon_panel(mine, sample, a.spec_figura)
        print(_section(f"FIGURA — RMSE per orizzonte, spec {a.spec_figura}"))
        print(f"campione della figura ({len(sample_fig)}): "
              f"{', '.join(sample_fig)}")
        if len(sample_fig) < len(sample):
            print("piu' stretto delle tabelle: esclusi "
                  f"{', '.join(q for q in sample if q not in sample_fig)} "
                  "(copertura settimanale incompleta)")
        csv = os.path.join(out_dir, f"rmse_by_horizon_{a.spec_figura}{tag}.csv")
        ph.to_csv(csv, index=False)
        png = figure_rmse_by_horizon(
            ph, sample_fig, a.spec_figura,
            # Nome stabile, non il campione: vedi la nota in `bvar.metrics`.
            # Un nome che porta dentro il campione lascia un file orfano ogni
            # volta che il campione cambia.
            os.path.join(out_dir, f"rmse_by_horizon_{a.spec_figura}"
                                  f"{tag or '_completo'}.png"))
        scartati = ph[~ph["pieno"]]
        print(f"\nscritto: {csv}")
        print(f"scritto: {png}")
        if len(scartati):
            print(f"punti non disegnati (campione incompleto): {len(scartati)}")
            print(scartati[["metodo", "horizon_week", "n_trimestri"]]
                  .to_string(index=False))


__all__ = [
    "my_final", "nyfed_final", "build_panel", "aligned_sample",
    "table_accuracy", "table_vs_fed", "table_fed_by_quarter", "table_alignment",
    "build_report", "horizon_panel", "figure_rmse_by_horizon",
]


if __name__ == "__main__":
    main()
