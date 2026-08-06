"""
src/bvar/data.py

IL WRAPPER DATI del BVAR: dai livelli grezzi alle unita' del modello.

Fa tre cose, in quest'ordine, e nient'altro:

    raw_levels_final.csv  ->  known_at(D)  ->  log / identita'

Cioe': legge i livelli grezzi gia' su disco, li maschera con il calendario di
pubblicazione esistente, e applica il logaritmo dove la config dice `log` e
l'identita' dove dice `level`.  **Nessuna standardizzazione.**  Nessun
ri-download.  Nessuna modifica a `data_loader_final.py`, `scale.py` o
`data_loader.py`: quelli servono la pipeline DFM e restano dove sono.


PERCHE' RIPARTIAMO DAI LIVELLI GREZZI E NON DAL PANNELLO DFM
============================================================
Il BVAR di Cimadomo NON differenzia per rendere stazionario.  Tiene i
(log-)livelli e lascia che la non-stazionarieta' sia domata dal centraggio
random-walk del Minnesota e dal sum-of-coefficients.  E' il vantaggio esplicito
dei VAR sui DFM che il paper argomenta (§2, p. 502):

    "factor models generally require the data to be made stationary, while
     VARs can be easily estimated also on non-stationary data
     (Sims et al., 1990)"

La catena DFM ha percio' un DOPPIO mismatch con quello che serve qui:

  1. 31 serie su 37 sono DIFFERENZIATE (22 MoM_log, 3 QoQ_log_ar, 4 diff_level,
     2 diff_ppt).  Il BVAR le vuole in log-livello.
  2. C'e' anche la STANDARDIZZAZIONE as-of (`src/forecast/scale.py`).  Il BVAR
     non la vuole: lo scaling fra variabili lo fa il PRIOR, tramite psi.
     Cimadomo eq. (3) dice che il fattore Sigma/Psi "accounts for the different
     scale and variability of the data"; GLP (2015) §III dice che l'iperprior
     per psi "should be loosely related to the scale of the variables in the
     model".  Standardizzare a monte significherebbe farlo due volte.

Il calendario invece si riusa tale e quale, ed e' lecito: livello e differenza
della stessa serie hanno lo stesso periodo di riferimento e lo stesso
publication delay, quindi la maschera as-of e' identica.  (E il log commuta con
la maschera, perche' log(NaN) = NaN.)


I DUE PROFILI DI VARIABILI: PERCHE' 30 PER Q/C/B E 37 PER L
============================================================
E' la parte teorica che conta di piu' in questo modulo, e la ragione per cui
`build_panel` chiede uno `spec` e non un elenco di colonne.

LA DOMANDA: perche' Q-BVAR, C-BVAR e B-BVAR non possono usare tutte e 37 le
serie, mentre l'L-BVAR si'?

LA RISPOSTA NON E' "questi modelli non usano il Kalman".  Il Kalman c'e' in
tutti e quattro i modelli.  La differenza e' DOVE agisce nella pipeline: a
monte (nella stima) o a valle (nella previsione).

**Q-BVAR / C-BVAR / B-BVAR — il Kalman agisce A VALLE.**  La stima dei
parametri e' in forma chiusa sul posterior Normal-Inverse-Wishart, e richiede
un pannello pieno e rettangolare.  Il paper e' esplicito, §2.1:

    "When all variables in the vector x_tq are available, the model can be
     readily estimated with standard Bayesian methods"

— la condizione e' *"when all variables are available"*.  Il B-BVAR (§2.3)
stima allo stesso modo, in forma chiusa.  In questi modelli il Kalman entra
DOPO la stima, solo per produrre i nowcast come previsioni condizionate ai
diversi information set (§2.3):

    "Given the model parameters, the nowcasts can be viewed as forecasts
     conditional on different information sets.  We compute these using the
     Kalman filtering techniques."

Due fasi distinte: prima stimi su dati pieni, poi il Kalman gestisce il bordo
in previsione.

**L-BVAR — il Kalman/smoother agisce A MONTE.**  Il simulation smoother di
Durbin-Koopman opera DENTRO il ciclo MCMC (§2.2): si estrae il dataset mensile
completo *"including draws of the latent missing values, conditional on the
model parameters"*.  Qui i dati mancanti vengono riempiti come parte
integrante della stima.

LA DISTINZIONE CHE RENDE TUTTO CHIARO — ci sono DUE tipi diversi di dato
mancante, e vanno trattati diversamente:

  * BORDO FRASTAGLIATO (ragged edge) = dati che mancano IN FONDO al campione,
    perche' non ancora pubblicati.  Lo gestisce il Kalman in TUTTI i modelli,
    Q/B-BVAR inclusi.  Nessun problema: e' previsione.

  * PARTENZE TARDIVE (late starts) = dati che mancano IN CIMA al campione,
    perche' la serie non esisteva ancora (PPIFIS prima del 2009, ecc.).  Questo
    e' un buco nel CAMPIONE DI STIMA, e solo l'L-BVAR lo puo' riempire, perche'
    solo lui ha lo smoother dentro la stima.

CONCLUSIONE OPERATIVA: il Q-BVAR e il B-BVAR scartano le 7 serie a storia corta
NON perche' "non sanno usare il Kalman", ma perche' il loro Kalman interviene
troppo tardi (in fase di previsione) per salvare i buchi in cima al campione.
La stima conjugate a monte vuole un pannello pieno.  Ecco perche' l'invariante
"il core non vede mai un NaN" impone n=30 per Q/C/B: le 7 serie tardive
porterebbero NaN in cima, e l'unico modo di dare al core una matrice densa e'
escluderle.  Per l'L-BVAR, invece, il riempimento di quei buchi E' parte della
stima, quindi le 37 serie vanno bene.

Non e' un limite arbitrario ne' una scelta di comodo: e' la conseguenza diretta
di come il paper struttura i due tipi di stima (conjugate closed-form vs MCMC
con stati latenti).


DOVE FINISCE QUESTO MODULO
==========================
Qui il pannello resta MENSILE.  L'aggregazione mensile->trimestrale (medie
mobili a 3 mesi sui LIVELLI, poi log, poi campionate a fine trimestre; paper
nota 17, convenzione di Giannone, Reichlin & Small 2008) serve al Q-BVAR e al
C-BVAR ed e' materia del Gate 2/3, non di qui — vive in `qbvar.py`, che la
espone anche al C-BVAR.  Non riguarda affatto B ed L, che i mensili li vogliono
grezzi.  L'impilamento del B-BVAR e' il Gate 4.

Questo modulo si ferma al punto in cui il calendario e' stato applicato.  Con
`model_units=True` (default) applica anche log/identita' e il pannello e' nelle
unita' del modello; con `model_units=False` si ferma un passo prima e
restituisce i livelli, perche' la media mobile deve stare PRIMA del logaritmo.


PSEUDO-REAL-TIME
================
Convenzione ereditata dal lavoro DFM e confermata per il BVAR: si maschera un
unico file corrente con il calendario.  Si riproduce la TEMPISTICA dei rilasci,
non le REVISIONI.  Un nowcast del 2008 vede i dati del 2008 nella loro versione
di oggi.  Semplificazione dichiarata, non difetto nascosto.  Il paper invece
ricostruisce vintage settimanali veri (§3.1).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.bvar.spec import BVARSpec
from src.forecast.release_calendar import known_at, load_metadata

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FINAL_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "final")

#: I livelli GREZZI, scritti da `data_loader_final.build_dataset_final`.  Stesso
#: indice month-end e stesso ordine di colonne di `dataset_final.csv`, ma senza
#: nessuna trasformazione applicata.  E' il punto di ingresso del BVAR.
RAW_LEVELS_PATH = os.path.join(_FINAL_DIR, "raw_levels_final.csv")

#: I mesi in cui le serie trimestrali portano un valore.  Fuori da questi sono
#: NaN *per costruzione* (placement a fine trimestre deciso dal loader), il che
#: non e' un dato mancante e non va contato come tale.
_QUARTER_END_MONTHS = (3, 6, 9, 12)


# ─── 1. Caricamento ───────────────────────────────────────────────────────────

def load_raw_levels(path: str | None = None) -> pd.DataFrame:
    """
    I livelli grezzi delle 37 serie: 499 x 37, indice month-end.

    Non e' un file che generiamo noi: lo scrive gia' il loader DFM
    (`data_loader_final.py`, riga ~390) per poter invertire le trasformazioni.
    Il BVAR lo riusa come input primario — ed e' la ragione per cui il wrapper
    e' piccolo invece di essere una seconda pipeline di download.
    """
    p = RAW_LEVELS_PATH if path is None else path
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"livelli grezzi non trovati: {p}\n"
            f"Generali con:  python src/data_loader_final.py"
        )
    return pd.read_csv(p, index_col=0, parse_dates=True)


# ─── 2. La trasformazione ─────────────────────────────────────────────────────

def to_model_units(panel: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    Da livelli a unita' del modello: log dove la config dice `log`, identita'
    dove dice `level`.

    E' tutta qui la "trasformazione" del BVAR — perche' il BVAR non
    differenzia.  Confronta con `data_loader_final.apply_final_transform`, che
    per la pipeline DFM applica cinque verbi diversi e differenzia 31 serie su
    37.

    Perche' il log e perche' non ovunque
    ------------------------------------
    Il log stabilizza la varianza di serie che crescono in modo
    MOLTIPLICATIVO e rende additive le relazioni di crescita, senza pero'
    differenziare: la non-stazionarieta' resta, e la gestisce il prior.  Non si
    applica a:
      * gli indici di diffusione con segno (Empire, Philly), dove sarebbe
        aritmeticamente IMPOSSIBILE — assumono regolarmente valori negativi;
      * le quattro ISM, dove sarebbe calcolabile ma concettualmente sbagliato,
        perche' non sono quantita' che crescono moltiplicativamente;
      * i tassi in percentuale (UNRATE, TCU), che il paper tiene in livello
        (Tabella 1: UNRATE, FEDFUNDS, spread sono tutti "Level").

    Raises
    ------
    ValueError
        Se una serie marcata `log` contiene valori non positivi.  E' un errore
        vero, non un caso da aggirare: significa che la mappa log/livello nella
        config non descrive piu' i dati.
    """
    out = panel.copy()
    tmap = dict(zip(spec.series, spec.transform))
    for col in out.columns:
        if tmap.get(col) != "log":
            continue
        vals = out[col].to_numpy(dtype=float)
        obs = vals[~np.isnan(vals)]
        if obs.size and not np.all(obs > 0):
            n_bad = int((obs <= 0).sum())
            raise ValueError(
                f"{col} e' marcata 'log' nella config ma ha {n_bad} valori <= 0 "
                f"(minimo {obs.min():.6g}).  Il logaritmo non e' definito: "
                f"controlla `transform` in config/bvar_series.json."
            )
        out[col] = np.log(out[col])
    return out


# ─── 3. Il pannello ───────────────────────────────────────────────────────────

def build_panel(
    spec: BVARSpec,
    as_of=None,
    *,
    end=None,
    raw: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    model_units: bool = True,
) -> pd.DataFrame:
    """
    Il pannello del BVAR a una data: mensile, in unita' del modello.

    I tre passi, nell'ordine in cui li fa:

      1. seleziona le colonne del profilo di `spec` e taglia a
         `spec.sample_start`;
      2. se `as_of` e' dato, maschera con `release_calendar.known_at` — cioe'
         mette a NaN tutto cio' che a quella data non era ancora pubblicato;
      3. applica log/identita' secondo la config.

    Parameters
    ----------
    spec : BVARSpec
        Decide colonne, ordine, inizio campione e mappa delle trasformazioni.
    as_of : date-like | None
        La data di osservazione.  `None` = nessun mascheramento (si usa il
        pannello com'e' su disco, che ha comunque il proprio bordo frastagliato
        dovuto ai ritardi di pubblicazione reali del file).
    end : date-like | None
        Taglio superiore dell'indice, utile per fissare la finestra di stima.
    raw, metadata : pd.DataFrame | None
        Gia' caricati, per non rileggere il disco a ogni settimana nel ciclo
        real-time.
    model_units : bool
        True (default) esegue anche il passo 3.  False si FERMA AL PASSO 2 e
        restituisce i LIVELLI mascherati, lasciando la trasformazione al
        chiamante.  Vedi sotto: serve al Q/C-BVAR, ed e' l'unico modo di
        rispettare l'ordine imposto dalla nota 17 del paper.

    Returns
    -------
    pd.DataFrame
        Indice month-end, colonne = `spec.series` nell'ordine della config.
        Le serie trimestrali portano un valore solo nei mesi 3/6/9/12: e' il
        placement del loader, non un dato mancante.

    Nota sull'ordine dei passi
    --------------------------
    Mascheratura e logaritmo COMMUTANO (log(NaN) = NaN), quindi l'ordine fra 2
    e 3 e' indifferente per il risultato.  Si maschera prima perche' e' piu'
    economico ed e' l'ordine in cui si legge la pipeline.

    PERCHE' ESISTE `model_units=False`
    ----------------------------------
    Perche' l'aggregazione mensile->trimestrale di Q e C **non commuta** con il
    logaritmo, e va infilata FRA il passo 2 e il passo 3.

    La nota 17 del paper lo impone alla lettera: i mensili sono trasformati in
    quantita' trimestrali "BEFORE TAKING LOGS".  Cioe' livelli -> media mobile a
    3 mesi -> log.  Applicare la media al pannello gia' in log darebbe la MEDIA
    DEI LOG invece del LOG DELLA MEDIA — che e' il log della media geometrica,
    un'altra quantita' (misurato sulle nostre serie: fino a 1.7% di scarto sul
    livello e 166 punti base su una crescita trimestrale, per l'edilizia).

    Senza questo flag l'unico modo di avere i livelli mascherati sarebbe
    riscrivere qui dentro `qbvar.py` i passi 1-2, duplicandoli.  Il flag e'
    retrocompatibile: chi non lo passa ottiene esattamente il comportamento di
    prima.  Vedi `bvar/qbvar.py`, sezione "1. L'ORDINE DELLE OPERAZIONI".
    """
    raw = load_raw_levels() if raw is None else raw

    missing = [s for s in spec.series if s not in raw.columns]
    if missing:
        raise KeyError(f"serie assenti dai livelli grezzi: {missing}")

    panel = raw.loc[:, list(spec.series)]
    panel = panel.loc[panel.index >= spec.sample_start]
    if end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]

    if as_of is not None:
        meta = load_metadata() if metadata is None else metadata
        panel = known_at(panel, as_of, metadata=meta)

    return to_model_units(panel, spec) if model_units else panel


def drop_empty_series(panel: pd.DataFrame, spec: BVARSpec
                      ) -> tuple[pd.DataFrame, BVARSpec]:
    """
    Toglie le serie che a questa `as_of` **non hanno nemmeno un'osservazione**.

    Restituisce `(pannello ridotto, spec ristretto)`, allineati.  Serve
    all'L-BVAR in real time, dove l'insieme delle serie esistenti dipende dalla
    data: misurato, `n` vale 35 all'inizio del 2007, 36 dal 2007-06 al 2010-01, e 37
    da 2010-04 in poi.

    ================================================================================
    PERCHE' SERVE — e NON e' lo smoother che si rompe
    ================================================================================
    E' il punto in cui e' facile confondersi rileggendo, quindi vale la pena
    scriverlo per esteso.

    La domanda naturale e': *l'L-BVAR ha il simulation smoother DENTRO la stima,
    quindi le partenze tardive dovrebbe trattarle come stati latenti — perche'
    si rompe?*

    Non si rompe lo smoother.  Lo smoother sa benissimo fare il suo lavoro: e'
    esattamente per quello che il profilo `l` ha 37 serie mentre Q/C/B ne hanno
    30 (vedi `notes.perche_due_profili` nella config).  Cio' che manca e'
    un'altra cosa: **l'ANCORAGGIO**.

        lastFull = find(~isnan(sum(x,2)),1,'last') - 3      (`lbvar.m` r.18)

    `last_full_row` cerca l'ultima riga in cui **tutte** le serie sono
    osservate, e quell'indice ancora la fine del campione di STIMA — la
    marginal likelihood e l'estrazione di (beta, Su) girano su
    `xx(1:lastFull,:)`.  E' una convenzione sul campione, non un requisito
    dello smoother.  Se una colonna e' interamente vuota, nessuna riga e' mai
    piena e l'ancoraggio non esiste: il modello resta senza campione di stima,
    e solleva `ValueError`.

    Da cui: si toglie la colonna vuota, non si rilassa l'ancoraggio.

    ================================================================================
    PERCHE' SCARTARE E' MEGLIO CHE RILASSARE L'ANCORAGGIO TENENDOLE
    ================================================================================
    La strada alternativa sarebbe tenere le colonne vuote e definire `lastFull`
    su un sottoinsieme di serie, lasciando che lo smoother "riempia" le altre.
    E' peggio, per tre ragioni indipendenti:

      1. **UNA COLONNA INTERAMENTE NaN NON E' LATENTE, E' PURO PRIOR.**  Uno
         stato latente si stima perche' i dati lo informano — attraverso le sue
         osservazioni passate, o attraverso le correlazioni con le altre serie
         nei periodi in cui e' osservato.  Una serie che a quella data non ha
         MAI avuto un'osservazione non ha ne' l'uno ne' l'altro canale: quel
         che ne uscirebbe e' la simulazione del prior, e nient'altro.  Zero
         informazione, presentata come un dato;

      2. **AUMENTA d, E IL MIXING PEGGIORA COME 1/d.**  Ogni serie in piu' e'
         un `psi` in piu' nella proposta del Metropolis.  Non e' un timore
         generico: e' la legge misurata in `tests/test_mixing` (pendenza -1.10
         contro il -1.00 teorico).  Pagare dimensione per colonne che non
         portano informazione e' il peggior scambio possibile;

      3. **NON SAREBBE REAL-TIME.**  Nel 2008 PPIFIS non esisteva.  Un
         previsore a quella data non poteva usarla in nessun modo, nemmeno come
         latente: non era nel suo insieme informativo.  Scartarla ricostruisce
         l'insieme informativo vero.

    E' anche la scelta **coerente con quel che gia' facciamo**: le 7 serie a
    partenza tardiva sono escluse da Q/C/B proprio perche' non esistono ancora.
    Qui l'L-BVAR le include quando esistono e le scarta quando non esistono —
    la stessa regola, applicata alla data invece che al profilo.

    DA DICHIARARE IN TESI: `n` varia nel tempo (35 -> 36 -> 37 fra il 2007 e il
    2010, poi stabile).  Non
    e' un compromesso ne' un ripiego, e' la ricostruzione onesta del real-time.

    ================================================================================
    CHE COSA **RESTA**
    ================================================================================
    Le serie PARZIALMENTE osservate restano, e il loro passato lo riempie lo
    smoother: quello e' il disegno dell'L-BVAR e non si tocca.  PCEC96 (prima
    osservazione 2007-01) a `as_of = 2008-06` ha 17 mesi di dati e ~260 mesi
    latenti — resta, ed e' proprio il caso che il profilo `l` esiste per
    trattare.

    ================================================================================
    LE DUE FASI — e perche' la seconda non e' un'eccezione ma la stessa regola
    ================================================================================
    FASE 1, la regola nella sua forma pulita: esce chi ha **zero** osservazioni.

    FASE 2, il caso al contorno che la fase 1 da sola non copre.  Trovato
    misurando, a `as_of = 2010-01-01`: PPIFIS ha la prima osservazione a
    2009-11, quindi NON e' vuota e la fase 1 la tiene — ma con due soli mesi di
    storia non esiste ancora nessuna riga in cui **tutte** le serie siano
    osservate *e* gia' pubblicate, e l'ancoraggio manca lo stesso.

    (Perche' le righe piene possono essere solo i quarter-end: le trimestrali
    portano un valore solo nei mesi 3/6/9/12.  A `as_of = 2010-01-01` il
    quarter-end 2009-09 non ha PPIFIS, e il 2009-12 non ha ancora il PIL —
    esce il 2010-01-29.  Nessuno dei due e' pieno.)

    Il rimedio e' la STESSA regola portata alla sua conseguenza: si scarta la
    serie con la prima osservazione **piu' recente** e si riprova, finche'
    l'ancoraggio non esiste.  Cioe' **`last_full_row` e' definito sull'insieme
    delle serie effettivamente trattenute, per costruzione e non come caso
    particolare**.  Una serie entra nel modello quando ha accumulato abbastanza
    storia da poter partecipare al campione di stima — che e' esattamente cio'
    che un previsore reale avrebbe potuto fare con lei.

    La procedura e' deterministica e termina: a ogni passo l'insieme si riduce
    di uno, e con una sola serie l'ancoraggio esiste sempre.
    """
    obs = panel.notna().any(axis=0)
    keep = [c for c in panel.columns if bool(obs.get(c, False))]
    if not keep:
        raise ValueError("nessuna serie ha osservazioni a questa data")

    # FASE 2: si toglie la serie che comincia piu' tardi finche' non esiste una
    # riga pienamente osservata sull'insieme trattenuto.
    while len(keep) > 1:
        sub = panel.loc[:, keep]
        if bool(sub.notna().all(axis=1).any()):
            break
        first = {c: sub[c].first_valid_index() for c in keep}
        tardiva = max(keep, key=lambda c: first[c])
        keep = [c for c in keep if c != tardiva]

    if len(keep) == len(panel.columns):
        return panel, spec
    return panel.loc[:, keep], spec.restrict(keep)


def append_forecast_rows(monthly: pd.DataFrame, horizon: int, *,
                         align_quarters: bool = False) -> pd.DataFrame:
    """
    Le righe di PREVISIONE — `STEP1_BBVAR.m` r.98: `X = X(1:nowcastM+horizon,:)`.

    Gli autori troncano il pannello a `nowcastM + horizon` con `horizon = 24`
    mesi.  Le righe oltre l'ultimo dato sono **tutte-NaN**, e lo smoother, non
    avendo nulla da cui aggiornare, le riempie **iterando la transizione**:
    cioe' produce il sentiero di previsione a 8 trimestri, con bande vere, nello
    STESSO passaggio del nowcast.  Non serve un ramo di codice separato per la
    previsione — e' il nowcast con piu' righe vuote.

    E' il punto 4 del registro del README, aperto dal Gate 4: senza queste righe
    `fit()` da' il nowcast e basta, e le fan chart forecast/nowcast/backcast non
    si possono disegnare.

    Parameters
    ----------
    align_quarters : bool
        True impone `horizon` multiplo di 3.  Serve al B-BVAR, dove il blocking
        vuole trimestri interi; l'L-BVAR e il C-BVAR lavorano in mesi e non ne
        hanno bisogno.
    """
    horizon = int(horizon)
    if horizon <= 0:
        return monthly
    if align_quarters and horizon % 3:
        raise ValueError(f"horizon={horizon} non e' multiplo di 3: il blocking "
                         "vuole trimestri interi")
    extra = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(1),
                          periods=horizon, freq="ME")
    tail = pd.DataFrame(np.nan, index=extra, columns=monthly.columns)
    return pd.concat([monthly, tail])


# ─── 4. Diagnostica dei buchi ─────────────────────────────────────────────────

def gaps_report(panel: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    Dove sono i NaN, e di che tipo sono.

    Distingue le tre cose che il docstring del modulo tiene separate, perche'
    contarle insieme e' esattamente l'errore da non fare:

      * `late_start`  buchi IN CIMA (la serie non era ancora nata).  Fatali per
                      Q/C/B, riempibili dallo smoother per L.
      * `interior`    buchi IN MEZZO (serie nata e non finita, ma un'osservazione
                      manca).  Fatali per Q/C/B e basta: nessun profilo li
                      risolve escludendo serie.
      * `ragged_edge` buchi IN FONDO (non ancora pubblicato).  Fisiologici: li
                      gestisce il Kalman a valle in tutti i modelli.

    Le trimestrali sono valutate solo sui mesi 3/6/9/12: fuori di li' il NaN e'
    il placement, non un buco.

    Returns
    -------
    pd.DataFrame
        Una riga per serie con almeno un buco; colonne `series_id`, `freq`,
        `late_start`, `interior`, `ragged_edge`, `first_obs`, `interior_dates`.
        DataFrame vuoto (con le colonne giuste) se il pannello e' pieno.
    """
    fmap = dict(zip(spec.series, spec.freq))
    rows = []
    for col in panel.columns:
        s = panel[col]
        if fmap.get(col) == "Q":
            s = s[s.index.month.isin(_QUARTER_END_MONTHS)]
        first, last = s.first_valid_index(), s.last_valid_index()
        if first is None:
            rows.append({
                "series_id": col, "freq": fmap.get(col, "?"),
                "late_start": int(s.size), "interior": 0, "ragged_edge": 0,
                "first_obs": None, "interior_dates": "",
            })
            continue
        interior = s.loc[first:last]
        holes = interior.index[interior.isna()]
        n_late = int((s.index < first).sum())
        n_edge = int((s.index > last).sum())
        if n_late or n_edge or len(holes):
            rows.append({
                "series_id": col,
                "freq": fmap.get(col, "?"),
                "late_start": n_late,
                "interior": int(len(holes)),
                "ragged_edge": n_edge,
                "first_obs": first.date(),
                "interior_dates": ", ".join(str(d.date()) for d in holes[:6]),
            })
    cols = ["series_id", "freq", "late_start", "interior", "ragged_edge",
            "first_obs", "interior_dates"]
    return pd.DataFrame(rows, columns=cols)


def last_dense_date(panel: pd.DataFrame, spec: BVARSpec) -> pd.Timestamp | None:
    """
    L'ultima data fino a cui il pannello e' pieno, cioe' l'ultimo month-end tale
    che tutte le mensili siano osservate e tutte le trimestrali lo siano su ogni
    quarter-end precedente.

    Serve a rispondere alla domanda pratica "dove posso chiudere il campione di
    stima senza dare NaN al core?".
    """
    fmap = dict(zip(spec.series, spec.freq))
    mon = [c for c in panel.columns if fmap.get(c) == "M"]
    qrt = [c for c in panel.columns if fmap.get(c) == "Q"]

    ok_m = ~panel[mon].isna().any(axis=1) if mon else pd.Series(True, index=panel.index)
    qe = panel.index.month.isin(_QUARTER_END_MONTHS)
    ok_q = pd.Series(True, index=panel.index)
    if qrt:
        ok_q[qe] = ~panel.loc[qe, qrt].isna().any(axis=1)

    ok = ok_m & ok_q
    if not ok.any():
        return None
    # il primo fallimento chiude la finestra
    bad = ok[~ok]
    if bad.empty:
        return panel.index[-1]
    first_bad = bad.index[0]
    before = panel.index[panel.index < first_bad]
    return before[-1] if len(before) else None


def estimation_end(spec: BVARSpec) -> pd.Timestamp | None:
    """
    La fine del campione di STIMA decisa per questo profilo, se ce n'e' una.

    Per `q_b` vale **2025-09-30**, ed e' una decisione presa, non un default
    tecnico: e' l'ultimo month-end prima del buco da shutdown federale di
    ottobre 2025 (vedi `notes.known_data_gaps` nella config).  Chiudere li'
    costa quasi nulla — restano T=135 trimestri, ancora vicini ai ~130 del
    paper — ed evita di interpolare o inventare un dato per darlo in pasto al
    posterior coniugato, che non ammette NaN.

    DISTINZIONE DA NON PERDERE: chiudere la STIMA a settembre 2025 **non**
    limita il NOWCASTING di trimestri successivi.  Sono due fasi diverse — e'
    esattamente il "Kalman a valle" descritto nell'header di questo modulo: i
    parametri si stimano sulla finestra densa, poi il filtro proietta in avanti
    su qualunque orizzonte.  La finestra di VALUTAZIONE e' un'altra decisione
    ancora, e appartiene al Gate 6.

    Per questo `build_panel` NON applica da solo questo taglio: il pannello di
    nowcast deve poter arrivare oltre.  Chi costruisce il campione di stima
    passa esplicitamente `end=estimation_end(spec)`.

    Returns
    -------
    pd.Timestamp | None
        None se il profilo non fissa una fine (e' il caso di `l`).
    """
    prof = spec.config.get("profiles", {}).get(spec.profile, {})
    end = prof.get("estimation_end")
    return None if end is None else pd.Timestamp(end)


def assert_dense(panel: pd.DataFrame, spec: BVARSpec) -> None:
    """
    L'INVARIANTE, applicato: il core non deve mai vedere un NaN.

    Questa e' la porta che lo fa rispettare.  Va chiamata da chi sta per
    passare una matrice al core sampler (dal Gate 1 in poi), non da chi
    costruisce un pannello per il nowcast — li' il bordo frastagliato ci deve
    essere.

    Solleva `ValueError` con un messaggio che dice *quale* dei tre tipi di buco
    ha trovato, perche' i rimedi sono diversi: una partenza tardiva si risolve
    cambiando profilo, un buco interno no.

    IL BUCO DA SHUTDOWN 2025-10: DECISIONE PRESA
    --------------------------------------------
    Il profilo `q_b` contiene un buco INTERNO a 2025-10 — cinque serie
    (CPIAUCSL, UNRATE, CPILFESL, IR, IQ) non hanno l'osservazione di ottobre
    2025, non raccolta durante lo shutdown federale USA.  E' permanente: non si
    riempira' mai.

    **Deciso: il campione di stima si chiude al 2025-09-30**, cioe' subito
    prima del buco; nessuna interpolazione.  Il taglio e' registrato in
    `profiles.q_b.estimation_end` nella config e si legge con
    `estimation_end(spec)`.  Un campione di stima costruito cosi' passa questo
    controllo.

    Se qualcuno passa comunque un pannello che oltrepassa quella data, questa
    funzione fallisce — ed e' voluto: meglio fallire qui che scoprire il NaN
    dentro il posterior.
    """
    rep = gaps_report(panel, spec)
    if rep.empty:
        return

    late = rep[rep["late_start"] > 0]
    inte = rep[rep["interior"] > 0]
    edge = rep[rep["ragged_edge"] > 0]

    msgs = []
    if not late.empty:
        msgs.append(
            "PARTENZE TARDIVE (buchi in cima) su "
            + ", ".join(f"{r.series_id} ({r.late_start} periodi)"
                        for r in late.itertuples())
            + ".  Rimedio: escludere queste serie (profilo q_b) oppure usare "
              "un modello con lo smoother nella stima (L-BVAR)."
        )
    if not inte.empty:
        msgs.append(
            "BUCHI INTERNI su "
            + "; ".join(f"{r.series_id} ({r.interior} a {r.interior_dates})"
                        for r in inte.itertuples())
            + ".  Rimedio: NON si risolve cambiando profilo — o si accorcia la "
              "finestra di stima, o si tratta il buco esplicitamente."
        )
    if not edge.empty:
        msgs.append(
            "BORDO FRASTAGLIATO su "
            + ", ".join(f"{r.series_id} ({r.ragged_edge})"
                        for r in edge.itertuples())
            + ".  Rimedio: chiudere il campione a "
            + f"{last_dense_date(panel, spec)} — il bordo e' materia del Kalman "
              "a valle, non del core."
        )

    raise ValueError(
        f"il pannello passato al core non e' denso ({spec.model}-BVAR, "
        f"profilo {spec.profile}).\n  " + "\n  ".join(msgs)
    )


__all__ = [
    "load_raw_levels",
    "to_model_units",
    "build_panel",
    "gaps_report",
    "last_dense_date",
    "estimation_end",
    "assert_dense",
    "RAW_LEVELS_PATH",
]
