"""
src/bvar/evaluate.py

GATE 6 — LA VALUTAZIONE REAL-TIME.  §3 del paper.

    python -m src.bvar.evaluate --help

Questo modulo non contiene modellistica: i quattro modelli sono chiusi ai Gate
2-5.  Contiene il CALENDARIO — chi vede che cosa, quando, e quando si ristima —
e il ciclo che ne esce.  E' il gate in cui l'errore possibile non e' un numero
sbagliato ma un numero **troppo buono**.


================================================================================
CHE COSA VALUTA LA §3
================================================================================
Il paper non chiede "quanto e' bravo il modello": chiede **come migliora il
nowcast al crescere dell'insieme informativo**.  E' una differenza sostanziale.
La quantita' obiettivo e' sempre la stessa — la crescita annualizzata del PIL di
un trimestre fissato — mentre cio' che cambia e' l'informazione disponibile:

    y_{t+h|Omega_D}      con D che scorre di settimana in settimana

Da cui l'asse x della loro Figura 1, che non e' il tempo ma la **settimana
rispetto al trimestre obiettivo**.  Un modello puo' essere mediocre in assoluto
e ottimo nel modo in cui incorpora le nuove pubblicazioni, o viceversa; la §3
misura la seconda cosa, ed e' quella che interessa a chi fa nowcasting.

LE TRE FASI, che sono la stessa quantita' vista da tre distanze
(`release_calendar.horizon_week`, la convenzione di Cascaldi-Garcia):

    h < 1        FORECAST   il trimestre non e' ancora cominciato
    1 <= h <= 13 NOWCAST    il trimestre e' in corso, i dati arrivano
    h > 13       BACKCAST   il trimestre e' finito, il PIL non e' ancora uscito

Il backcast e' la fase che distingue il nowcasting dalla previsione ordinaria:
c'e' un trimestre **gia' accaduto** di cui non si conosce ancora il valore
ufficiale, e l'informazione mensile lo copre quasi tutto.  E' li' che ci si
aspetta l'errore minimo, ed e' li' che il confronto col NY Fed e' piu' duro.

RMSE **E** LOG PREDICTIVE SCORE, e servono entrambi.  Il primo valuta la
mediana; il secondo valuta la **densita' intera**, cioe' se il modello sa
quanto e' incerto.  E' l'unica metrica in cui un BVAR bayesiano puo' battere un
point forecast ben tarato, e quindi la ragione per cui produciamo estrazioni e
non solo la mediana.  Formalmente, per il realizzato y*:

    LS_D = log p(y* | Omega_D)

stimata dalle estrazioni con un kernel gaussiano sui quantili (vedi
`log_score`).  Piu' alto e' meglio.


================================================================================
LO SCHEMA A DUE VELOCITA' — ED E' LORO, NON NOSTRO
================================================================================
`STEP1_BBVAR.m` r.113 e `STEP3_LBVAR.m` r.87 hanno lo stesso identico `if`, col
loro commento sopra:

    % Estimate each quarter after new GDP release
    if (yy == 4 && ww == 1) || (ww>2 && <#GDP osservati questa settimana>
                                        > <#GDP osservati la scorsa>)
        tempRes = bbvar(...)                       % STIMA COMPLETA
    else   % "otherwise just use existing parameter estimates on new vintage"
        run_Ksmoother_FM(betaHat, sigmaHat, ...)   % punto
        run_Ssmoother_block(beta, sigma, ...)      % DENSITA', estrazioni conservate
    end

LA LOGICA, che vale la pena aver capito: i parametri (lambda, mu, psi, B, Sigma)
sono trattati come costanti **entro il trimestre**.  Si ristima quando arriva
informazione che li cambia — cioe' il PIL, la variabile che il VAR sta imparando
a spiegare — non quando arriva informazione che cambia lo **stato**.  Un dato
mensile nuovo sposta il bordo, non la legge di moto.

E' un'approssimazione, e va dichiarata come tale: fra una release e l'altra i
parametri sarebbero leggermente diversi se ristimati.  Ma e' l'approssimazione
DEGLI AUTORI, ed e' quella che rende il gate eseguibile: costa ~45 min (B) e
~57 min (L) per stima completa contro ~25-80 s per settimana di riuso.

    IL TRIGGER, tradotto in pseudo-real-time.  Loro rilevano il rilascio
    contando i non-NaN della colonna PIL fra due vintage consecutive.  Noi non
    abbiamo vintage vere (§ semplificazione dichiarata, sotto), quindi
    l'equivalente fedele e' il calendario: c'e' stata una release di `GDPC1`
    fra la settimana scorsa e questa?  Lo dice
    `release_calendar.releases_between`, la stessa infrastruttura del DFM.

    ANCHE IL RAMO DI RIUSO PRODUCE LA DENSITA'.  Si legge male nel loro codice
    perche' la prima riga del blocco `else` e' un filtro puntuale, ma subito
    dopo c'e' `run_Ssmoother_block` sulle estrazioni conservate.  Le bande ci
    sono tutte le settimane, non solo in quelle di ristima.


CHE COSA SI CONSERVA, MODELLO PER MODELLO
------------------------------------------
    Q-BVAR    B, Sigma
    C-BVAR    B, Sigma + I SISTEMI MENSILI GIA' MAPPATI  <- vedi sotto
    B-BVAR    B, Sigma  (nello spazio bloccato)
    L-BVAR    B, Sigma  ->  `lbvar.fit_reuse`, che era gia' scritto al Gate 5

LA CACHE DEL C-BVAR E' UNA DIVERGENZA DI EFFICIENZA, DICHIARATA.  Lo stadio 2
del C-BVAR — la mappa cube-root — prende `(Phi, Sigma_eps, c)` e restituisce
`(Phi_m, Sigma_epsm, c_m)`: **non tocca i dati**.  Quindi, fissate le estrazioni
del Q-BVAR, il sistema mensile e' identico in ogni settimana del trimestre, e
con lui la radice cubica di una companion 150x150, la (A.10), la proiezione di
Higham e P0 — cioe' tutta la parte cara.  Cambia solo la finestra di bordo.
Gli autori non hanno questa cache **perche' in `crbvar.m` il ramo di riuso non
esiste affatto**: `STEP2_CRBVAR.m` ristima a ogni vintage.  Noi estendiamo al C
lo schema a due velocita' che loro usano per B e L, e la cache ne e' la
conseguenza.  E' ESATTA: stesse estrazioni, stesso seme, uscita bit-identica.
Da dichiarare in tesi accanto alle altre divergenze paper<->codice.


================================================================================
IL RISCHIO DI QUESTO GATE: IL LOOK-AHEAD
================================================================================
Lo stimatore ha gia' i recovery test ai Gate 1 (core sampler) e 2 (wrapper
trimestrale), e il Gate 6 **non introduce stima nuova**.  Il rischio nuovo e'
solo il calendario: che il pannello a un dato `as_of` contenga informazione che
a quella data non esisteva.

E' il tipo di errore peggiore che possa capitare a una tesi, perche' non si
manifesta come crash ne' come numero assurdo: si manifesta come **performance
troppo buona**.  Un RMSE che batte il NY Fed di tre volte non fa sospettare
nessuno finche' non lo si va a cercare.

E' gia' successo qui.  Costruendo questo modulo si e' scoperto che `cbvar.fit`
chiamava `estimation_panel(spec, raw=raw)` **senza `as_of`**: il campione di
stima arrivava a `estimation_end` (2025-09-30) qualunque fosse la data del
nowcast, quindi un nowcast del 2008 avrebbe stimato su dati fino al 2025.
Corretto (`cbvar.py`, stadio 1) e `estimation_panel` ora taglia anche la coda
all'ultimo trimestre pienamente osservato — l'`endEstimT` degli autori.

Da cui l'oracolo di `tests/test_gate6.py`, che e' un test **esatto** e non di
plausibilita': intercetta ogni pannello che il codice costruisce e verifica, per
ogni cella osservata, che la sua data di rilascio sia <= `as_of`.  Con un
controllo negativo che ri-introduce il baco e pretende che il test fallisca —
senza quello, un test verde non dimostrerebbe niente.


PSEUDO-REAL-TIME: LA SEMPLIFICAZIONE DICHIARATA
------------------------------------------------
Si maschera un unico file corrente col calendario di pubblicazione: si riproduce
la **tempistica** dei rilasci, non le **revisioni**.  Un nowcast del 2008 vede i
dati del 2008 nella loro versione di oggi.  Il paper (§3.1) ricostruisce invece
vintage settimanali vere.  E' una semplificazione dichiarata, ereditata dal
lavoro DFM e coerente con esso — il confronto fra i nostri modelli resta interno
e leale; il confronto col NY Fed va letto sapendo questo.


================================================================================
IL CONTRATTO CSV — IDENTICO A QUELLO DEL DFM
================================================================================
Le colonne sono `forecast.weekly_nowcast.COLUMNS`, non una loro variante.  E' una
scelta di progetto, non una comodita': `compute_metrics.py`, `figures.py` e
`compare_nyfed.py` girano sull'output del BVAR **senza modifiche**, e con loro
tutta l'infrastruttura di confronto NY Fed.  Cambiare schema significherebbe
perderla.

    spec        'qbvar' | 'cbvar' | 'bbvar' | 'lbvar' | 'benchmark'
    variant     la formula di accoppiamento per il C ('authors'), '-' altrove;
                per i benchmark il nome del metro ('ar2', 'mean')
    nowcast_bea mediana della crescita annualizzata
    sd_z        sd delle estrazioni -> e' cio' che alimenta il log score
    n_iter      S, le estrazioni tenute
    reestimated True nella settimana di release, False nelle intermedie

AVERE LE STESSE COLONNE NON BASTA: VANNO RIEMPITE
-------------------------------------------------
Le colonne coincidevano gia', ma sei di esse uscivano vuote — fra cui
`realizzato_bea`.  E' la colonna contro cui TUTTI i lettori punteggiano: senza,
`figures.py` non disegna i pallini, `compute_metrics.py` scarta il 100% delle
righe come non punteggiabili e `compare_nyfed.py` non ha riferimento.  Il
contratto sarebbe stato formalmente rispettato e sostanzialmente vuoto, e lo si
sarebbe scoperto dopo la passata.  Ora si riempiono qui:

    realizzato_bea / realizzato_livello   dal pannello `final` del DFM, cioe'
        LA STESSA fonte del lavoro DFM.  E' la condizione perche' io, la Fed,
        l'AR(2) e la media espandente siamo punteggiati contro lo STESSO
        numero (l'argomento sta nell'header di `compare_nyfed.py`).
    errore_bea      nowcast - realizzato, in punti BEA.
    nowcast_livello `scale.from_bea` della mediana: la log-differenza
        annualizzata, l'unita' del pannello.
    n_releases      quante pubblicazioni ci sono state dalla settimana scorsa.
    nowcast_z       resta NaN, ed e' corretto: e' l'unita' STANDARDIZZATA del
        DFM, e il BVAR non standardizza niente — lavora sui livelli.  Nessun
        lettore la usa; riempirla con un numero inventato sarebbe peggio.

I BENCHMARK VIAGGIANO NELLO STESSO CSV
--------------------------------------
`compute_metrics.py` calcola `RMSE_rel_ar2` cercando le righe `spec ==
'benchmark'` NELLO STESSO file; `compare_nyfed.py` fa lo stesso con `_BENCH`.
Senza, le colonne relative escono NaN e il confronto col metro principale non
esiste.  Costano millisecondi (un ARIMA(2,0,0) sul solo PIL) e si calcolano
qui, una volta per settimana e non una per modello.

Accanto al CSV, e SOLO accanto, un `.npz` coi quantili per riga (5,10,25,50,75,
90,95): servono alle fan chart e non entrano nel contratto.

IL LOG SCORE SI CALCOLA QUI O NON SI CALCOLA PIU'
--------------------------------------------------
`log_score` vuole le ESTRAZIONI, e le estrazioni non sopravvivono alla
settimana: sette quantili non bastano a ricostruire la densita'.  O lo si
valuta mentre le si ha in mano, o va rifatta la passata.  Esce quindi in un CSV
suo, `output/bvar/logscore/`, che non tocca il contratto.


================================================================================
LA PASSATA E' INTERROMPIBILE — E DEVE ESSERLO
================================================================================
Una passata quadriennale sui quattro modelli sta nell'ordine delle decine di ore
di calcolo su una macchina sola.  In quel tempo il PC si riavvia, OneDrive
blocca un file, la batteria finisce.  Una passata che tiene tutto in memoria e
scrive solo alla fine perde tutto: non e' un rischio accettabile per l'unico
esercizio empirico della tesi.

Quindi si scrive DOPO OGNI SETTIMANA, e si riprende da dove si era rimasti.  E'
lo stesso schema del ciclo DFM (`forecast/weekly_nowcast.py`), con una cosa in
piu' che qui e' obbligatoria e li' non serviva.

    1. GLI ARTEFATTI SONO SEMPRE CORRENTI.  CSV, `.npz` dei quantili e CSV del
       log score si riscrivono per intero a ogni settimana, su file temporaneo
       + `os.replace` — atomico anche su Windows, quindi un file scritto a meta'
       non e' mai osservabile.  Si possono aprire le figure a passata in corso.

    2. IL MARCATORE DI SETTIMANA.  `checkpoint/<start>_<end>/weeks/<data>.done`,
       scritto DOPO che gli artefatti sono a posto.  Alla ripresa si salta ogni
       settimana marcata.  Serve un marcatore esplicito e non basta guardare le
       righe del CSV: una settimana puo' legittimamente produrre MENO righe del
       previsto (`_growth_rows` restituisce None se la finestra non arriva al
       trimestre), e dedurre "fatta" dal conteggio la rifarebbe in eterno.

    3. LA CACHE SU DISCO — ed e' questa la differenza col DFM.  Lo schema a due
       velocita' fa dipendere ogni settimana di riuso dalla STIMA COMPLETA che
       apre il trimestre: e' li' che stanno le ore.  Senza la cache su disco,
       un'interruzione a meta' trimestre costringerebbe a rifare quella stima —
       cioe' la parte cara — per poter ricalcolare settimane di riuso da 30
       secondi.  Con la cache, si riprende dalla settimana esatta.

       La cache si mette su disco SOLO alle settimane di stima piena (nelle
       altre non cambia), una `pickle` per modello, sempre con `os.replace`.

    QUANDO LA CACHE NON SI PUO' USARE, SI RIAVVOLGE.  Il pickle di un modello
    puo' mancare (prima esecuzione), essere piu' grande del tetto
    (`--max-cache-mb`) o venire da un commit diverso.  In tutti e tre i casi la
    ripresa RIAVVOLGE all'ultima settimana di stima piena e la rifa': le
    settimane di riuso gia' marcate restano saltate, quindi si paga la stima e
    non il trimestre.  Il degrado e' graduale, non un ritorno a zero.

    IL MANIFESTO IMPEDISCE DI MESCOLARE PASSATE DIVERSE.  `--draws` esiste per
    il pilota, e un pilota a S=100 che riprende una passata a S=1000 scriverebbe
    nello stesso CSV righe non confrontabili, senza che nulla lo segnali.  Il
    manifesto registra (start, end, modelli, seme, orizzonte, n_ahead, S) e la
    ripresa si ferma se non coincidono: si riparte con `--fresh` o si scrive in
    un `--output-root` diverso.  Il commit invece NON blocca le righe (si puo'
    committare a passata in corso) ma invalida le CACHE, che sono oggetti
    Python dipendenti dal codice che li ha prodotti.

    RIPRENDERE NON CAMBIA I NUMERI.  Il seme e' `seed + i` con `i` l'indice
    della settimana nella griglia, non un contatore di quante ne sono state
    fatte: una settimana ricalcolata dopo un'interruzione riparte dallo stesso
    stato del generatore e da' la stessa riga.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import output_layout as _layout
from src.bvar import bbvar, cbvar, lbvar, qbvar
from src.bvar.data import load_raw_levels
from src.bvar.spec import BVARSpec
from src.forecast import scale
from src.forecast.benchmarks import BENCHMARKS, BENCHMARK_SPEC
from src.forecast.release_calendar import (
    gdp_release_date,
    horizon_week,
    load_metadata,
    load_panel,
    quarter_end,
    quarter_label,
    releases_between,
    targets_in_flight,
    weekly_grid,
)
from src.forecast.weekly_nowcast import COLUMNS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: La serie obiettivo.  Una sola, e non e' un parametro: la §3 valuta il PIL.
TARGET = "GDPC1"

#: Mesi di previsione oltre l'ultimo dato — `STEP1_BBVAR.m` r.98.
HORIZON_MONTHS = 24

#: Le estrazioni per modello.  L'asimmetria e' misurata, non arbitraria: nel
#: B-BVAR `find_mode` e' il 97% del costo e S e' quasi gratis, nell'L-BVAR la
#: catena e' il grosso e S e' LA leva.  Errore Monte Carlo a S=250: ~0.12 pp
#: sulla mediana, contro RMSE tipici di 1-2 pp.
DRAWS = {"qbvar": 1000, "cbvar": 1000, "bbvar": 1000, "lbvar": 250}

#: I quantili salvati accanto al CSV.
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: La radice dell'output del Gate 6.  NON piu' `output/bvar/` accanto a
#: `forecast_weekly/`: i CSV del BVAR sono l'input della stessa catena di
#: tabelle e figure del DFM, e tenerli in un albero parallelo era il motivo
#: per cui le uscite finivano sparse.  I percorsi li decide `output_layout`.
OUTPUT_ROOT = _layout.OUTPUT_ROOT

#: Chiave d'identita' di una riga — la stessa di `weekly_nowcast._KEY`.
_KEY = ("as_of", "target_quarter", "spec", "variant")

#: Tetto di default per una cache su disco.  Oltre, non si salva e la ripresa
#: riavvolge alla stima piena: meglio pagare una stima che riempire il disco.
MAX_CACHE_MB = 1500.0


# ─── 0. Il realizzato, il metro di tutti ──────────────────────────────────────

def realised_bea(quarter: str, panel: pd.DataFrame | None = None
                 ) -> tuple[float, float]:
    """
    `(livello, BEA)` del PIL realizzato del trimestre, dal pannello `final`.

    E' LA STESSA FONTE DEL LAVORO DFM — `weekly_nowcast._realised` legge da qui
    con la stessa catena `scale.to_bea` — e deve restarlo: un realizzato diverso
    fra i due esercizi renderebbe incomparabili i loro RMSE senza che si veda.

    Il pannello e' quello PIENO, non mascherato a `as_of`: il realizzato non e'
    un input del nowcast, e' il metro con cui lo si punteggia dopo.  Nessun
    look-ahead — nessun modello lo vede mai.
    """
    panel = load_panel() if panel is None else panel
    qe = quarter_end(quarter)
    if qe not in panel.index or pd.isna(panel.loc[qe, TARGET]):
        return float("nan"), float("nan")
    lvl = float(panel.loc[qe, TARGET])
    return lvl, float(scale.to_bea(lvl))


# ─── 1. Il calendario: quando si ristima ──────────────────────────────────────

def gdp_released_between(a, b, metadata: pd.DataFrame | None = None) -> bool:
    """
    C'e' stata una release di `GDPC1` in (a, b]?

    E' la traduzione in pseudo-real-time del trigger degli autori, che invece
    conta i non-NaN della colonna PIL fra due vintage consecutive
    (`STEP1_BBVAR.m` r.113).  Senza vintage vere quel conteggio non e'
    riproducibile; il calendario si'.
    """
    rel = releases_between(a, b, metadata=metadata)
    if rel is None or len(rel) == 0:
        return False
    col = "series_id" if "series_id" in rel.columns else rel.columns[0]
    return bool((rel[col] == TARGET).any())


def estimation_weeks(grid: list[pd.Timestamp],
                     metadata: pd.DataFrame | None = None) -> list[bool]:
    """
    Per ogni settimana della griglia: e' una settimana di STIMA COMPLETA?

    La prima lo e' sempre — e' il `(yy == 4 && ww == 1)` degli autori, cioe' il
    bootstrap: senza una stima iniziale non c'e' niente da riusare.
    """
    out = [True]
    for prev, cur in zip(grid[:-1], grid[1:]):
        out.append(gdp_released_between(prev, cur, metadata=metadata))
    return out


def parallel_blocks(start, end, metadata: pd.DataFrame | None = None
                    ) -> list[tuple[str, str]]:
    """
    Spezza [start, end] nei blocchi da dare a processi separati.

    IL TAGLIO CADE SULLE SETTIMANE DI STIMA PIENA, non sul calendario, ed e'
    l'unica scelta che non costa niente.  `run()` lo dice gia': l'unita'
    indipendente e' "1 stima completa + le sue settimane di riuso", perche' ogni
    stima parte dal pannello a `as_of` e non dallo stato della precedente.
    Tagliare li' significa che la prima settimana di ogni blocco — che
    `estimation_weeks` forza comunque a piena — E' una settimana che sarebbe
    stata piena lo stesso.

    Tagliare altrove no.  Un confine a Capodanno (i blocchi annuali di prima)
    cade in mezzo a un trimestre, perche' le stime vere seguono le release BEA e
    cadono a febbraio/maggio/agosto/novembre: quel venerdi' veniva promosso da
    riuso a stima fresca, e con lui cambiava il suo nowcast.  Diciannove
    settimane su 991 avevano un numero che dipendeva da come avevamo affettato
    il lavoro per il cluster, non dal modello.  Qui non piu': la passata spezzata
    da' le STESSE righe di una passata continua.

    Sul 2007-2025 sono 77 blocchi di 13 settimane mediane (4 il piu' corto, 14
    il piu' lungo) — e 77 unita' indipendenti stanno in una sola ondata su una
    macchina da qualche decina di core in su.

    Returns
    -------
    [(inizio, fine), ...] con le date in ISO, estremi INCLUSI, contigui e senza
    buchi: la fine di un blocco e' il venerdi' prima dell'inizio del successivo.
    """
    grid = weekly_grid(start, end)
    if not grid:
        return []
    is_full = estimation_weeks(grid, metadata=metadata if metadata is not None
                               else load_metadata())
    cuts = [i for i, f in enumerate(is_full) if f]
    out = []
    for k, i in enumerate(cuts):
        j = cuts[k + 1] - 1 if k + 1 < len(cuts) else len(grid) - 1
        out.append((str(grid[i].date()), str(grid[j].date())))
    return out


def target_quarter(as_of) -> str:
    """
    Il trimestre obiettivo a `as_of`: quello di cui il PIL non e' ancora uscito.

    E' il `nowcastM` degli autori (`STEP1_BBVAR.m` r.106), letto dal calendario
    invece che dal conteggio dei NaN.  Si scorre in avanti dal trimestre che
    contiene `as_of` finche' la sua release non e' ancora avvenuta.

    ATTENZIONE: restituisce UN SOLO trimestre, ed e' sempre quello corrente o il
    prossimo.  Non basta per la §3 — vedi `targets(as_of)` qui sotto, che e'
    quello che il ciclo usa.  Resta perche' e' la traduzione fedele del loro
    `nowcastM`, ed e' la funzione giusta se si vuole sapere "qual e' IL
    trimestre che stanno nowcastando".
    """
    D = pd.Timestamp(as_of)
    q = pd.Timestamp(D).to_period("Q").to_timestamp("Q")
    for _ in range(6):
        lab = quarter_label(q)
        rel = gdp_release_date(lab, target_series=TARGET)
        if rel is None or pd.Timestamp(rel) > D:
            return lab
        q = (q + pd.offsets.QuarterEnd(1))
    return quarter_label(q)


def targets(as_of, n_ahead: int = 1, metadata: pd.DataFrame | None = None
            ) -> list[str]:
    """
    TUTTI i trimestri in volo a `as_of` — ed e' la differenza fra avere le tre
    fasi e averne una sola.

    Un trimestre e' in volo finche' il suo PIL non e' pubblicato, e a ogni data
    ce n'e' piu' d'uno: il precedente non ancora uscito (BACKCAST), quello in
    corso (NOWCAST) e il prossimo (FORECAST).  Sono la stessa quantita' vista da
    tre distanze, che e' letteralmente la §3.

        2009-01-02  ->  2008Q4 (h=+14, backcast)
                        2009Q1 (h= +1, nowcast)
                        2009Q2 (h=-12, forecast)

    Con `target_quarter` — un obiettivo solo — `horizon_week` non usciva mai da
    1..13: le bande "forecast" e "backcast" delle figure sarebbero rimaste
    VUOTE, e `compare_nyfed` avrebbe messo sulla stessa riga la mia stima di
    meta' trimestre (settimana 13) e il loro backcast maturo (settimana 17),
    che e' esattamente il confronto sbilanciato contro cui il suo
    `aligned_sample` esiste per protestare.

    E' `release_calendar.targets_in_flight`, la stessa funzione del ciclo
    settimanale del DFM: un calendario solo per i due esercizi, altrimenti i
    loro RMSE non sono confrontabili.

    IL COSTO NON TRIPLICA.  La stima e' UNA per settimana e per modello; i tre
    obiettivi sono tre righe della stessa posteriore, estratte da
    `res.growth(TARGET)`.  Quel che cambia e' il numero di righe del CSV, non
    il numero di catene.
    """
    return list(targets_in_flight(pd.Timestamp(as_of), n_ahead=n_ahead,
                                  target_series=TARGET, metadata=metadata))


# ─── 2. La cache: che cosa sopravvive alla settimana ──────────────────────────

@dataclass
class ModelCache:
    """
    Quel che si porta avanti fra una stima completa e la successiva.

    `extra` e' il posto dove ogni modello mette cio' che gli serve in piu': per
    il C-BVAR i sistemi mensili gia' mappati, che sono la parte cara e non
    dipendono da `as_of` (vedi l'header, "LA CACHE DEL C-BVAR").
    """
    B: np.ndarray | None = None
    Sigma: np.ndarray | None = None
    lam: np.ndarray | None = None
    mu: np.ndarray | None = None
    psi: np.ndarray | None = None
    draws_obj: object = field(default=None, repr=False)
    extra: dict = field(default_factory=dict, repr=False)
    estimated_at: pd.Timestamp | None = None

    @property
    def ready(self) -> bool:
        return self.B is not None or bool(self.extra)


# ─── 3. I quattro adattatori ──────────────────────────────────────────────────
#
# Ogni adattatore fa due cose e nient'altro: chiama il modello (ramo pieno o
# ramo di riuso) e restituisce le estrazioni di crescita annualizzata del PIL,
# indicizzate per trimestre.  La modellistica sta nei moduli dei Gate 2-5.

def _growth_rows(g: pd.DataFrame, quarter: str) -> np.ndarray | None:
    """Le estrazioni per il trimestre chiesto, o None se la finestra non ci arriva."""
    target = pd.Period(quarter, freq="Q").to_timestamp("Q").normalize()
    idx = pd.DatetimeIndex(g.index).normalize()
    hit = idx == target
    if not hit.any():
        return None
    return np.asarray(g.to_numpy(dtype=float)[hit][0])


def qbvar_growth(fit, quarters: list[str], as_of, *,
                 rng: np.random.Generator, raw: pd.DataFrame | None = None,
                 horizon_q: int = HORIZON_MONTHS // 3
                 ) -> dict[str, np.ndarray | None]:
    """
    Il nowcast del Q-BVAR: il filtro sul bordo TRIMESTRALE, `qbvar.nowcast`.

    ERA UN'ITERAZIONE DELLA COMPANION dal campione di stima, e sbagliava per
    difetto.  E' vero che il Q-BVAR non forma mai una media parziale, quindi nel
    trimestre APERTO non ha nulla da aggiungere; ma appena il trimestre CHIUDE —
    settimana 14 — le sue celle si completano serie per serie mentre il PIL
    manca ancora, e li' c'e' informazione vera su cui condizionare.  Iterando la
    companion non entrava mai: il nostro Q-BVAR non poteva avere il "catching up
    ... in week 14" del §3.2 nemmeno in linea di principio.

    Gli autori ci fanno girare un Kalman smoother (`run_Ksmoother_FM`,
    STEP4_QBVAR.m r.149) sulla finestra ancorata a `endEstimT`, righe di
    previsione comprese.  La motivazione completa e la misura di quante serie
    entrano settimana per settimana stanno in `qbvar.py`, sezione 4.

    UNA STIMA, PIU' OBIETTIVI: la posteriore del bordo si calcola una volta e i
    trimestri in volo sono sue righe — come per C, B e L.
    """
    res = qbvar.nowcast(fit, as_of=as_of, horizon_q=horizon_q, raw=raw, rng=rng)
    g = res.growth(TARGET)
    return {q: _growth_rows(g, q) for q in quarters}


def run_model(model: str, as_of, quarters: list[str], cache: ModelCache, *,
              full: bool, spec: BVARSpec, raw: pd.DataFrame,
              rng: np.random.Generator, horizon: int = HORIZON_MONTHS,
              verbose: bool = False,
              n_draws: int | None = None
              ) -> tuple[dict[str, np.ndarray | None], ModelCache]:
    """
    Una settimana, un modello, TUTTI i trimestri in volo.  `full=True` = stima
    completa, `False` = riuso.

    UNA STIMA, PIU' OBIETTIVI.  I trimestri in volo sono tre righe della stessa
    posteriore — `res.growth(TARGET)` le contiene tutte — quindi si stima una
    volta sola e si estrae piu' volte.  Passare i trimestri uno per uno avrebbe
    triplicato il costo della passata per niente.

    `n_draws` scavalca `DRAWS` — serve al pilota, dove si vuole la MECCANICA a
    costo ridotto e non il numero definitivo.  Chi non lo passa ha i valori
    tarati al Gate 5.

    Returns
    -------
    ({trimestre: estrazioni della crescita annualizzata, o None}, cache)
    """
    S = DRAWS[model] if n_draws is None else int(n_draws)
    kw = dict(as_of=as_of, raw=raw, rng=rng, verbose=verbose)

    if model == "qbvar":
        # La baseline.  I parametri si ristimano solo alle settimane piene; il
        # BORDO invece si ricostruisce ogni settimana, perche' e' li' che il
        # flusso dati entra — e' la stessa divisione di C, B e L.
        if full or not cache.ready:
            panel = qbvar.estimation_panel(spec, as_of=as_of, raw=raw)
            fit = qbvar.fit(spec, panel, n_draws=S, burn=S // 2, rng=rng)
            cache = ModelCache(B=fit.draws.B, Sigma=fit.draws.Sigma,
                               lam=fit.draws.lam, mu=fit.draws.mu,
                               psi=fit.draws.psi, draws_obj=fit.draws,
                               estimated_at=pd.Timestamp(as_of),
                               extra={"fit": fit, "panel": panel})
        # nel riuso il campione di stima resta quello dell'ultima stima piena:
        # e' la stessa approssimazione degli autori, applicata al Q.
        return (qbvar_growth(cache.extra["fit"], quarters, as_of, rng=rng,
                             raw=raw, horizon_q=horizon // 3), cache)

    if model == "cbvar":
        if full or not cache.ready:
            res = cbvar.fit(spec, n_draws=S, burn=S // 2, horizon=horizon, **kw)
            cache = ModelCache(draws_obj=res, estimated_at=pd.Timestamp(as_of),
                               extra={"systems": res.systems})
        else:
            res = cbvar.fit_reuse(cache.extra["systems"], spec, horizon=horizon, **kw)

    elif model == "bbvar":
        if full or not cache.ready:
            res = bbvar.fit(spec, n_draws=S, burn=S // 2, horizon=horizon, **kw)
            cache = ModelCache(B=res.draws.B, Sigma=res.draws.Sigma,
                               lam=res.draws.lam, mu=res.draws.mu,
                               psi=res.draws.psi, draws_obj=res.draws,
                               estimated_at=pd.Timestamp(as_of))
        else:
            res = bbvar.fit_reuse(cache.B, cache.Sigma, spec, horizon=horizon,
                                  draws_obj=cache.draws_obj, **kw)

    elif model == "lbvar":
        if full or not cache.ready:
            res = lbvar.fit(spec, n_draws=S, burn=S // 5, horizon=horizon, **kw)
            # LO SPEC RIDOTTO, non quello di configurazione: l'L-BVAR scarta le
            # serie che a `as_of` non esistono ancora, e (B, Sigma) sono
            # indicizzati su quelle rimaste.  Il ramo di riuso deve vedere lo
            # STESSO insieme, altrimenti a meta' trimestre una serie che entra
            # allarga il pannello sotto parametri che non la contemplano.
            cache = ModelCache(B=res.B, Sigma=res.Sigma, lam=res.lam,
                               mu=res.mu, psi=res.psi,
                               estimated_at=pd.Timestamp(as_of),
                               extra={"spec": res.spec})
        else:
            res = lbvar.fit_reuse(cache.B, cache.Sigma,
                                  cache.extra.get("spec", spec), horizon=horizon,
                                  lam=cache.lam, mu=cache.mu, psi=cache.psi, **kw)
    else:
        raise ValueError(f"modello ignoto {model!r}")

    g = res.growth(TARGET)                      # una volta, non una per obiettivo
    return {q: _growth_rows(g, q) for q in quarters}, cache


# ─── 4. Le metriche di densita' ───────────────────────────────────────────────

def log_score(draws: np.ndarray, realised: float) -> float:
    r"""
    Log predictive score con kernel gaussiano — la seconda meta' della Fig. 1.

        LS = log( (1/S) sum_s N(y* ; x_s, h^2) )

    con `h` la finestra di Silverman.  E' la stima non parametrica standard
    quando la predittiva e' rappresentata da estrazioni e non ha forma chiusa —
    e qui non ce l'ha, perche' la densita' del nowcast e' un misto sugli
    iperparametri.

    Perche' non basta una normale con media e sd delle estrazioni: la
    predittiva del BVAR e' ASIMMETRICA e a code spesse, e schiacciarla su una
    gaussiana butterebbe via proprio cio' che il modello ha da dire sulla coda.
    """
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    h = 0.9 * min(sd, iqr / 1.349 if iqr > 0 else sd) * x.size ** (-0.2)
    if h <= 0:
        return float("nan")
    z = (realised - x) / h
    m = float(np.max(-0.5 * z ** 2))
    lse = m + np.log(np.mean(np.exp(-0.5 * z ** 2 - m)))
    return float(lse - np.log(h) - 0.5 * np.log(2 * np.pi))


def summarise(draws: np.ndarray) -> dict:
    """Mediana, sd e quantili di una predittiva."""
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "sd": np.nan,
                "q": {q: np.nan for q in QUANTILES}}
    return {"median": float(np.median(x)), "sd": float(np.std(x, ddof=1)),
            "q": {q: float(np.quantile(x, q)) for q in QUANTILES}}


# ─── 5. Persistenza incrementale e ripresa ────────────────────────────────────
#
# Vedi l'header, "LA PASSATA E' INTERROMPIBILE".  Niente qui riguarda la
# modellistica: e' solo il contratto con il disco.

def _paths(root: str | None, start: str, end: str) -> dict[str, str]:
    """
    I quattro posti in cui una passata scrive.

    Risultati (`csv`, `npz`, `ls`) sotto `<root>/csv/`, che di default e'
    `forecast_weekly/csv/bvar/`; il CHECKPOINT invece fuori dall'albero di
    consegna, perche' e' cache di ripresa e pesa ordini di grandezza piu' dei
    risultati.

    `root` VA ONORATO, e la riga qui sotto e' il fix di un difetto vero: la
    funzione riceveva `root` e lo ignorava, cablando `bvar_csv_dir()`.  Il
    parametro esisteva, era documentato nella CLI ("radice di csv/ e
    logscore/") e non faceva niente — quindi un pilota lanciato con
    `--output-root` scriveva i suoi CSV a ridotte estrazioni IN MEZZO agli
    artefatti veri, dove `metrics_tables.load_bvar` li avrebbe poi raccolti
    con il glob `bvar_realtime_*.csv` senza modo di distinguerli dalla passata
    buona.  Silenzioso in entrambi i passaggi: la sonda non protestava e la
    tabella nemmeno.
    """
    csv_dir = (os.path.join(root, "csv") if root and root != OUTPUT_ROOT
               else _layout.bvar_csv_dir())
    ck = os.path.join(_layout.checkpoint_dir(), "bvar", f"{start}_{end}")
    return {
        "csv": os.path.join(csv_dir, f"bvar_realtime_{start}_{end}.csv"),
        "npz": os.path.join(csv_dir, f"bvar_realtime_{start}_{end}_quantiles.npz"),
        "ls": os.path.join(csv_dir, "logscore", f"logscore_{start}_{end}.csv"),
        "ckpt": ck,
        "weeks": os.path.join(ck, "weeks"),
        "manifest": os.path.join(ck, "manifest.json"),
    }


def _atomic(write_tmp, path: str, retries: int = 6, delay: float = 0.7) -> bool:
    """
    `write_tmp(tmp)` poi `os.replace(tmp, path)` — atomico anche su Windows.

    Il progetto sta su OneDrive, che puo' tenere un file bloccato per qualche
    istante: si riprova.  Se proprio non si riesce si AVVISA e si prosegue —
    perdere la passata per un file bloccato sarebbe la cura peggiore della
    malattia, e il salvataggio della settimana dopo riscrive tutto comunque.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    for attempt in range(1, retries + 1):
        try:
            write_tmp(tmp)
            os.replace(tmp, path)
            return True
        except (PermissionError, OSError) as exc:
            if attempt == retries:
                print(f"  [attenzione] scrittura fallita dopo {retries} tentativi "
                      f"({type(exc).__name__}: {exc}): {path}")
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                return False
            time.sleep(delay)
    return False


def _git_head() -> str:
    """Il commit corrente, o `''` fuori da un repo.  Invalida le cache, non le righe."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _manifest(start: str, end: str, models, seed: int, horizon: int,
              n_ahead: int, S_of: dict) -> dict:
    """L'identita' della passata: due passate con manifesti diversi non si mescolano."""
    return {"start": start, "end": end, "models": list(models), "seed": int(seed),
            "horizon": int(horizon), "n_ahead": int(n_ahead),
            "draws": {m: int(S_of[m]) for m in models}}


def _check_manifest(paths: dict, man: dict, fresh: bool) -> None:
    """
    Scrive il manifesto, o si ferma se ne trova uno diverso.

    E' la guardia contro l'errore silenzioso piu' facile di questo gate: un
    pilota a `--draws 100` che riprende la passata vera e le infila righe con
    S diverso, che nessuna colonna del CSV distinguerebbe da sole.
    """
    old = None
    if os.path.exists(paths["manifest"]) and not fresh:
        try:
            with open(paths["manifest"], encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:
            old = None
    if old is not None:
        diff = {k: (old.get(k), man[k]) for k in man if old.get(k) != man[k]}
        if diff:
            righe = "\n".join(f"      {k}: sul disco {a!r}, chiesto {b!r}"
                              for k, (a, b) in diff.items())
            raise SystemExit(
                "  Il checkpoint su disco viene da una passata DIVERSA:\n"
                f"{righe}\n"
                "  Riprendere mescolerebbe righe non confrontabili.  O si riparte\n"
                "  da zero (--fresh), o si scrive altrove (--output-root).")
    def _json(t: str) -> None:
        with open(t, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)

    _atomic(_json, paths["manifest"])


def _week_marker(paths: dict, D) -> str:
    return os.path.join(paths["weeks"], f"{pd.Timestamp(D).date()}.done")


def _done_weeks(paths: dict) -> set[str]:
    """Le settimane gia' concluse, dal marcatore.  Vedi l'header, punto 2."""
    if not os.path.isdir(paths["weeks"]):
        return set()
    return {f[:-5] for f in os.listdir(paths["weeks"]) if f.endswith(".done")}


def _cache_path(paths: dict, model: str) -> str:
    return os.path.join(paths["ckpt"], f"cache_{model}.pkl")


def _save_cache(paths: dict, model: str, cache: "ModelCache", *,
                max_mb: float, git: str) -> bool:
    """
    La cache dell'ultima stima piena su disco.  Un fallimento non ferma la
    passata: costa un riavvolgimento alla ripresa, non i risultati.
    """
    try:
        blob = pickle.dumps({"git": git, "cache": cache}, protocol=5)
    except Exception as exc:
        print(f"  [attenzione] cache {model} non serializzabile "
              f"({type(exc).__name__}: {exc}); la ripresa riavvolgera' alla stima.")
        return False
    mb = len(blob) / 1e6
    if mb > max_mb:
        print(f"  [attenzione] cache {model} {mb:.0f} MB > tetto {max_mb:.0f} MB: "
              f"non salvata; la ripresa riavvolgera' alla stima.")
        return False
    def _pkl(t: str) -> None:
        with open(t, "wb") as fh:
            fh.write(blob)

    return _atomic(_pkl, _cache_path(paths, model))


def _drop_caches(paths: dict, verbose: bool = False) -> int:
    """
    Cancella le `cache_*.pkl` di un blocco COMPLETO, tenendo tutto il resto.

    E' la sola parte pesante del checkpoint — il pilota faceva 215 MB di cache
    contro 40 KB di risultati — e una volta che tutte le settimane sono marcate
    non serve piu' a nessuno: la cache esiste per riprendere un blocco a meta',
    e un blocco finito non si riprende.

    NON si cancella la cartella intera, e la differenza non e' cosmetica: i
    marcatori `weeks/*.done` sono ZERO byte ma sono l'unica cosa che dice
    "questa settimana e' fatta".  `_done_weeks` li legge, NON conta le righe
    del CSV (una settimana puo' legittimamente produrne meno del previsto).
    Cancellarli farebbe rifare per intero una passata gia' completa.
    """
    freed = 0
    d = paths["ckpt"]
    if not os.path.isdir(d):
        return 0
    for f in os.listdir(d):
        if not (f.startswith("cache_") and f.endswith(".pkl")):
            continue
        p = os.path.join(d, f)
        try:
            n = os.path.getsize(p)
            os.remove(p)
            freed += n
        except OSError as exc:
            if verbose:
                print(f"  [attenzione] cache {f} non rimovibile: {exc}")
    return freed


def purge_caches(root: str | None = None, verbose: bool = True) -> int:
    """
    Cancella le cache di TUTTI i blocchi sotto `output/_checkpoint/bvar/`,
    tenendo i marcatori.  Per i blocchi interrotti che nessuno riprendera'.

    Usarla su un blocco che si vuole ancora riprendere non perde risultati: la
    ripresa RIAVVOLGE all'ultima stima piena e la rifa'.  Si paga una stima,
    non il trimestre — e' lo stesso degrado graduale del caso "cache di un
    commit diverso".
    """
    base = os.path.join(root or _layout.checkpoint_dir(), "bvar")
    if not os.path.isdir(base):
        return 0
    freed = 0
    for block in sorted(os.listdir(base)):
        d = os.path.join(base, block)
        if not os.path.isdir(d):
            continue
        n = _drop_caches({"ckpt": d}, verbose=verbose)
        freed += n
        if verbose and n:
            print(f"  {block}: {n / 1e6:.0f} MB")
    if verbose:
        print(f"  totale liberato: {freed / 1e6:.0f} MB")
    return freed


def _load_cache(paths: dict, model: str, *, expect_at, git: str) -> "ModelCache | None":
    """
    La cache dell'ultima stima piena, se e' QUELLA giusta.

    Tre motivi per rifiutarla, e tutti e tre portano al riavvolgimento: il file
    non c'e', viene da un commit diverso (e' un oggetto Python, non un dato), o
    e' stata prodotta a una data che non e' la stima piena che ci serve.
    """
    p = _cache_path(paths, model)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as exc:
        print(f"  [attenzione] cache {model} illeggibile ({type(exc).__name__}: {exc}).")
        return None
    if git and blob.get("git") and blob["git"] != git:
        print(f"  [attenzione] cache {model} da un altro commit "
              f"({blob['git'][:8]} vs {git[:8]}): scartata.")
        return None
    cache = blob.get("cache")
    at = getattr(cache, "estimated_at", None)
    if at is None or pd.Timestamp(at) != pd.Timestamp(expect_at):
        return None
    return cache


def _persist(paths: dict, rows: list[dict], quants: dict, ls_rows: list[dict]) -> None:
    """
    Riscrive i tre artefatti per intero.  Costa millisecondi su qualche migliaio
    di righe, e in cambio a passata in corso il CSV e' sempre leggibile dalle
    figure.
    """
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    if len(df):
        df = df.sort_values(list(_KEY), kind="stable").reset_index(drop=True)
    _atomic(lambda t: df.to_csv(t, index=False), paths["csv"])

    keys = np.array([f"{a}|{b}|{c}" for a, b, c in quants]) if quants else np.array([])
    vals = (np.array(list(quants.values())) if quants
            else np.zeros((0, len(QUANTILES))))

    def _npz(t: str) -> None:
        # SUL FILE APERTO, non sul nome: `savez_compressed` appiccica `.npz` a
        # un percorso che non ce l'ha, e scriverebbe `....npz.tmp.npz` — un file
        # che poi `os.replace` non troverebbe.
        with open(t, "wb") as fh:
            np.savez_compressed(fh, keys=keys, values=vals,
                                quantiles=np.array(QUANTILES))

    _atomic(_npz, paths["npz"])

    ls = pd.DataFrame(ls_rows)
    _atomic(lambda t: ls.to_csv(t, index=False), paths["ls"])


def _reload(paths: dict) -> tuple[list[dict], dict, list[dict]]:
    """Rilegge quel che c'e'.  Un file illeggibile degrada a 'quella parte si rifa'."""
    rows, quants, ls_rows = [], {}, []
    try:
        if os.path.exists(paths["csv"]):
            df = pd.read_csv(paths["csv"], on_bad_lines="skip")
            for c in COLUMNS:
                if c not in df.columns:
                    df[c] = np.nan
            rows = df[COLUMNS].to_dict("records")
    except Exception as exc:
        print(f"  [attenzione] CSV illeggibile ({type(exc).__name__}: {exc}): riparto da zero.")
    try:
        if os.path.exists(paths["npz"]):
            z = np.load(paths["npz"], allow_pickle=False)
            for k, v in zip(z["keys"], z["values"]):
                a, b, c = str(k).split("|")
                quants[(a, b, c)] = np.asarray(v, dtype=float)
    except Exception as exc:
        print(f"  [attenzione] quantili illeggibili ({type(exc).__name__}: {exc}).")
    try:
        if os.path.exists(paths["ls"]):
            ls_rows = pd.read_csv(paths["ls"], on_bad_lines="skip").to_dict("records")
    except Exception as exc:
        print(f"  [attenzione] log score illeggibile ({type(exc).__name__}: {exc}).")
    return rows, quants, ls_rows


# ─── 6. Il ciclo real-time ────────────────────────────────────────────────────

def run_realtime(
    start: str,
    end: str,
    models: tuple[str, ...] = ("qbvar", "cbvar", "bbvar", "lbvar"),
    *,
    output_root: str | None = None,
    horizon: int = HORIZON_MONTHS,
    seed: int = 20260801,
    verbose: bool = True,
    dry_run: bool = False,
    benchmarks: bool = True,
    n_ahead: int = 1,
    n_draws: dict[str, int] | None = None,
    resume: bool = True,
    fresh: bool = False,
    max_cache_mb: float = MAX_CACHE_MB,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """
    Percorre [start, end] a passo settimanale, con lo schema a due velocita'.

    `dry_run=True` non stima nulla: percorre il calendario e stampa lo schema —
    quante stime complete, quante settimane di riuso, quali date.  E' il modo di
    controllare l'impianto senza pagarlo, e quello che si esegue PRIMA della
    passata vera.

    `resume=True` (default) riprende una passata interrotta: si salta ogni
    settimana marcata come conclusa e si ricarica la cache dell'ultima stima
    piena.  `fresh=True` ignora e sovrascrive quel che trova.  Il perche' e il
    comportamento nei casi al contorno stanno nell'header, "LA PASSATA E'
    INTERROMPIBILE".

    UNITA' DI PARALLELISMO: il blocco-trimestre (1 stima completa + le sue
    settimane di riuso) e' indipendente dagli altri per costruzione, perche'
    ogni stima parte dal pannello a `as_of` e non dallo stato della precedente.
    Il ciclo qui e' seriale di proposito — il parallelismo si mette FUORI, un
    processo per blocco, dopo aver misurato il guadagno netto su un pilota:
    OpenBLAS e' gia' multi-thread, quindi lo speedup non e' il numero di core.
    """
    # CSV e log score sotto la STESSA radice: un pilota che scrive il CSV
    # altrove non deve lasciare il suo log score fra gli artefatti veri.
    root = output_root or OUTPUT_ROOT
    grid = weekly_grid(start, end)
    if not grid:
        raise SystemExit(f"nessun venerdi' in {start} .. {end}")
    meta = load_metadata()
    is_full = estimation_weeks(grid, metadata=meta)

    if verbose or dry_run:
        n_full = sum(is_full)
        n_rows = sum(len(targets(D, n_ahead=n_ahead, metadata=meta)) for D in grid)
        print(f"  {len(grid)} settimane, {n_full} stime complete, "
              f"{len(grid) - n_full} settimane di riuso")
        print(f"  {n_rows} nowcast per modello "
              f"({n_rows / max(len(grid), 1):.1f} obiettivi in volo per settimana)")
        for D, f in zip(grid, is_full):
            if f:
                print(f"    STIMA  {D.date()}   in volo: "
                      f"{', '.join(targets(D, n_ahead=n_ahead, metadata=meta))}")
    if dry_run:
        return pd.DataFrame(columns=COLUMNS)

    raw = load_raw_levels()
    panel = load_panel()                       # solo per il realizzato e i metri
    S_of = dict(DRAWS, **(n_draws or {}))
    specs = {m: BVARSpec.from_config(m[0].upper()) for m in models}
    caches = {m: ModelCache() for m in models}

    # ── Il checkpoint: che cosa c'e' gia' e da dove si riparte ────────────────
    paths = _paths(root, start, end)
    git = _git_head()
    if fresh or not resume:
        # Il checkpoint va CANCELLATO, non solo ignorato: lasciarne i marcatori
        # significherebbe che la prima ripresa dopo una `--fresh` piu' corta
        # salta settimane le cui righe non sono piu' nel CSV.
        shutil.rmtree(paths["ckpt"], ignore_errors=True)
    _check_manifest(paths, _manifest(start, end, models, seed, horizon, n_ahead, S_of),
                    fresh=fresh or not resume)
    os.makedirs(paths["weeks"], exist_ok=True)

    done: set[str] = set() if (fresh or not resume) else _done_weeks(paths)
    rows, quants, ls_rows = ([], {}, []) if (fresh or not resume) else _reload(paths)
    seen = {tuple(str(r[k]) for k in _KEY) for r in rows}
    seen_ls = {tuple(str(r[k]) for k in _KEY) for r in ls_rows}

    # La prima settimana da fare, e la stima piena che la governa.  Se la cache
    # di quella stima non e' recuperabile, si RIAVVOLGE a lei: si ripaga la
    # stima, non le settimane di riuso gia' fatte (header, punto 3).
    todo = [i for i, D in enumerate(grid) if str(D.date()) not in done]
    rebuild_at = None
    if done and todo:
        # `estimation_weeks` marca sempre la prima settimana come piena, quindi
        # una stima che governa la ripresa esiste sempre.
        last_full = max(j for j in range(todo[0] + 1) if is_full[j])
        for m in models:
            c = _load_cache(paths, m, expect_at=grid[last_full], git=git)
            if c is None or not c.ready:
                rebuild_at = last_full
                caches = {mm: ModelCache() for mm in models}
                break
            caches[m] = c

    if done:
        print(f"  ripresa: {len(done)}/{len(grid)} settimane gia' fatte, "
              f"{len(todo)} da fare")
        if rebuild_at is not None:
            print(f"  cache non recuperabile: si rifa' la stima piena del "
                  f"{grid[rebuild_at].date()} (le settimane di riuso restano)")
        elif todo:
            print("  cache dell'ultima stima piena ricaricata da disco")

    t_start = time.perf_counter()
    n_done_now = 0

    for i, (D, full) in enumerate(zip(grid, is_full)):
        if str(D.date()) in done and i != rebuild_at:
            continue
        qs = targets(D, n_ahead=n_ahead, metadata=meta)
        prev = grid[i - 1] if i else D - pd.Timedelta(days=7)
        _rel = releases_between(prev, D, metadata=meta)
        n_rel = 0 if _rel is None else len(_rel)   # non `or []`: un DataFrame
        #                                            vuoto non e' falso, e' ambiguo
        info = {q: (horizon_week(D, q), *realised_bea(q, panel),
                    gdp_release_date(q, target_series=TARGET)) for q in qs}

        def _common(q: str, nowcast_bea: float) -> dict:
            """Le colonne che non dipendono dal modello — realizzato compreso."""
            h, rel_lvl, rel_bea, rel_date = info[q]
            return {
                "as_of": D.date().isoformat(), "target_quarter": q,
                "horizon_week": h,
                "nowcast_bea": nowcast_bea,
                "nowcast_livello": (float(scale.from_bea(nowcast_bea))
                                    if np.isfinite(nowcast_bea) else np.nan),
                "nowcast_z": np.nan,           # il BVAR non standardizza: vedi header
                "realizzato_bea": rel_bea, "realizzato_livello": rel_lvl,
                "errore_bea": nowcast_bea - rel_bea,
                # STRINGA, come nel CSV del DFM (`weekly_nowcast` r.278) e non
                # `Timestamp`: alla ripresa le righe rilette dal CSV sono
                # stringhe e le nuove sarebbero Timestamp: la colonna
                # diventerebbe `object` mista e `to_csv` renderebbe le due meta'
                # in due formati diversi nello stesso file.
                #
                # `.date()` e non `str(Timestamp)`: il DFM scrive '2009-01-28',
                # questo scriveva '2009-01-28 00:00:00'.  Stesso istante, due
                # stringhe diverse — invisibile finche' qualcuno non confronta
                # o unisce le due famiglie SULLA COLONNA invece che sulla data
                # convertita, e a quel punto non trova nessuna corrispondenza
                # senza sbagliare in modo rumoroso.
                "gdp_release_date": ("" if rel_date is None
                                     else str(pd.Timestamp(rel_date).date())),
                "n_releases": n_rel,
            }

        def _add(r: dict, target: list, keyed: set) -> None:
            """
            Aggiunge una riga se non c'e' gia'.  Serve al riavvolgimento: la
            settimana di stima piena rifatta produce le STESSE righe (stesso
            seme, stesso indice), e senza questo il CSV le avrebbe in doppio.
            """
            k = tuple(str(r[c]) for c in _KEY)
            if k in keyed:
                return
            keyed.add(k)
            target.append(r)

        for m in models:
            rng = np.random.default_rng(seed + i)
            per_q, caches[m] = run_model(
                m, D, qs, caches[m], full=full, spec=specs[m], raw=raw,
                rng=rng, horizon=horizon, verbose=False, n_draws=S_of[m])
            if full:
                # SOLO qui: nelle settimane di riuso la cache non cambia, quindi
                # riscriverla sarebbe un pickle da centinaia di MB per niente.
                _save_cache(paths, m, caches[m], max_mb=max_cache_mb, git=git)
            for q, draws in per_q.items():
                if draws is None:
                    continue
                s = summarise(draws)
                rel_bea = info[q][2]
                _add({
                    **_common(q, s["median"]), "spec": m,
                    "variant": "authors" if m == "cbvar" else "-",
                    "sd_z": s["sd"], "n_iter": S_of[m],
                    "converged": True, "reestimated": full,
                }, rows, seen)
                quants[(D.date().isoformat(), m, q)] = np.array(
                    [s["q"][x] for x in QUANTILES])
                # Il log score ORA, finche' le estrazioni esistono.
                _add({
                    "as_of": D.date().isoformat(), "target_quarter": q,
                    "horizon_week": info[q][0], "spec": m,
                    "variant": "authors" if m == "cbvar" else "-",
                    "log_score": log_score(draws, rel_bea),
                    "realizzato_bea": rel_bea, "n_draws": int(np.size(draws)),
                    "reestimated": full,
                }, ls_rows, seen_ls)

        # I metri, una volta per settimana e per obiettivo: non dipendono dal
        # modello, quindi si calcolano fuori dal ciclo sui modelli.
        if benchmarks:
            for q in qs:
                for name, fn in BENCHMARKS.items():
                    try:
                        b = fn(D, q, target_series=TARGET, panel=panel,
                               metadata=meta)
                    except Exception as exc:          # un metro non ferma la passata
                        print(f"  [attenzione] benchmark {name} a {D.date()} "
                              f"su {q}: {type(exc).__name__}: {exc}")
                        continue
                    _add({
                        **_common(q, b["nowcast_bea"]), "spec": BENCHMARK_SPEC,
                        "variant": name,
                        "nowcast_livello": b["nowcast_livello"],
                        "nowcast_z": b["nowcast_z"], "sd_z": np.nan,
                        "n_iter": 0, "converged": b["converged"],
                        "reestimated": False,
                    }, rows, seen)

        # ── La settimana e' finita: prima gli artefatti, POI il marcatore ─────
        # In quest'ordine, e non nell'altro: un'interruzione fra i due fa
        # rifare una settimana (costo noto), l'ordine inverso la darebbe per
        # fatta senza che i suoi numeri siano su disco.
        _persist(paths, rows, quants, ls_rows)
        _atomic(lambda t: open(t, "w").close(), _week_marker(paths, D))
        done.add(str(D.date()))

        n_done_now += 1
        if verbose:
            fasi = "  ".join(f"{q}:{info[q][0]:+d}" for q in qs)
            el = time.perf_counter() - t_start
            left = len([j for j in range(len(grid)) if str(grid[j].date()) not in done])
            eta = (f"   ~{el / n_done_now * left / 3600:.1f} h alla fine"
                   if left else "")
            print(f"  {D.date()}  {'STIMA' if full else 'riuso'}   {fasi}"
                  f"   [{el / n_done_now:.0f} s/sett{eta}]")

    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    if len(df):
        df = df.sort_values(list(_KEY), kind="stable").reset_index(drop=True)
    _persist(paths, rows, quants, ls_rows)

    # ── La passata e' finita: le cache non servono piu' ───────────────────────
    if not keep_cache and len(done) >= len(grid):
        freed = _drop_caches(paths, verbose=verbose)
        if verbose and freed:
            print(f"  cache     : {freed / 1e6:.0f} MB liberati "
                  f"(passata completa: la ripresa non ne ha piu' bisogno)")

    if verbose:
        print(f"\n  CSV       : {paths['csv']}")
        print(f"  quantili  : {paths['npz']}")
        print(f"  log score : {paths['ls']}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 6 — nowcast real-time del BVAR")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--models", default="qbvar,cbvar,bbvar,lbvar")
    ap.add_argument("--dry-run", action="store_true",
                    help="percorre solo il calendario, non stima nulla")
    ap.add_argument("--print-blocks", action="store_true",
                    help="stampa i blocchi paralleli ('inizio fine' per riga) "
                         "e esce. E' cio' che `scripts/run_all.sh` legge per "
                         "generare i processi della fase 5: i confini si "
                         "calcolano QUI, dove sta `estimation_weeks` che li "
                         "genera, e non si riscrivono nello script.")
    ap.add_argument("--draws", type=int, default=None,
                    help="scavalca S per TUTTI i modelli: e' la leva del pilota, "
                         "non un'opzione della passata vera")
    ap.add_argument("--no-benchmarks", action="store_true",
                    help="niente righe ar2/mean (sconsigliato: senza, "
                         "RMSE_rel_ar2 esce vuoto)")
    ap.add_argument("--output-root", default=None,
                    help="radice sotto cui finiscono csv/ e csv/logscore/ "
                         "(default: output/forecast_weekly/csv/bvar). "
                         "UN PILOTA O UNA SONDA VA SEMPRE LANCIATA CON QUESTO: "
                         "i suoi CSV a estrazioni ridotte, lasciati fra gli "
                         "artefatti veri, verrebbero raccolti dalle tabelle "
                         "insieme alla passata buona.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignora il checkpoint e ricomincia da capo "
                         "(default: si RIPRENDE da dove ci si era interrotti)")
    ap.add_argument("--purge-caches", action="store_true",
                    help="cancella le cache di TUTTI i blocchi e esce. Per "
                         "ripulire dopo una passata interrotta.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="non cancellare le cache_*.pkl a passata completa. "
                         "Di default si cancellano: sono la sola parte pesante "
                         "del checkpoint e a blocco finito non servono. I "
                         "marcatori weeks/*.done restano comunque, quindi un "
                         "rilancio salta tutto lo stesso.")
    ap.add_argument("--max-cache-mb", type=float, default=MAX_CACHE_MB,
                    help="tetto per una cache su disco; oltre, la ripresa "
                         "riavvolge all'ultima stima piena")
    a = ap.parse_args()
    if a.print_blocks:
        for bs, be in parallel_blocks(a.start, a.end):
            print(f"{bs} {be}")
        return
    if a.purge_caches:
        print("Pulizia delle cache di checkpoint (i marcatori restano):")
        purge_caches()
        return
    models = tuple(a.models.split(","))
    run_realtime(a.start, a.end, models, dry_run=a.dry_run,
                 output_root=a.output_root,
                 benchmarks=not a.no_benchmarks,
                 fresh=a.fresh, max_cache_mb=a.max_cache_mb,
                 keep_cache=a.keep_cache,
                 n_draws=({m: a.draws for m in models} if a.draws else None))


if __name__ == "__main__":
    main()
