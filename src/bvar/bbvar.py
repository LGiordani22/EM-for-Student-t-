"""
src/bvar/bbvar.py

GATE 4 — L'L-BVAR "a blocchi": il B-BVAR.  §2.3 del paper.


================================================================================
LA TEORIA — CHE COSA VUOL DIRE "BLOCKING"
================================================================================

Il problema e' sempre quello: il PIL e' trimestrale, i predittori mensili, e un
VAR vuole una frequenza sola.  I quattro modelli del paper sono quattro risposte
diverse alla stessa domanda, e vale la pena vedere dove si colloca questa.

    Q-BVAR   butta via la frequenza mensile (medie mobili a 3 mesi, nota 17)
    C-BVAR   scende a mensile trasformando i PARAMETRI (Phi^(1/3), Appendice A)
    L-BVAR   scende a mensile trattando le trimestrali come LATENTI (§2.2)
    B-BVAR   SALE a trimestrale trattando i mesi come VARIABILI DIVERSE  <-- qui

L'idea del blocking, e la ragione per cui e' elegante: se `INDPRO` esiste a
gennaio, febbraio e marzo, invece di comprimere quei tre numeri in uno (Q-BVAR)
o di inventare un processo mensile latente (C, L), li si dichiara **tre
variabili trimestrali distinte** — `INDPRO_m1`, `INDPRO_m2`, `INDPRO`.  Ogni
trimestre porta un'osservazione di ciascuna, e il sistema torna bilanciato.

    mese      1     2     3        ->      variabili trimestrali
    INDPRO   x_1   x_2   x_3               INDPRO_m1, INDPRO_m2, INDPRO
    GDPC1     .     .    y_3               GDPC1

NON SI PERDE INFORMAZIONE, e questa e' la differenza col Q-BVAR: la media mobile
comprime tre numeri in uno, il blocking li tiene tutti e tre e lascia che sia il
VAR a stimare come pesarli.  Il prezzo lo si paga altrove — vedi sotto.


IL PREZZO: IL SISTEMA TRIPLICA
-------------------------------
Con `n_m` mensili e `n` serie totali, il sistema blocked ha

    n_b = 2*n_m + n     variabili

perche' ogni mensile contribuisce m1 e m2 (i mesi 1 e 2) e poi rientra nel terzo
blocco insieme alle trimestrali.  Sul profilo `q_b` (n = 30, n_m = 27) fa
n_b = 84, e con p = 5 il numero di coefficienti per equazione e' k = 421.
Il documento di contesto lo segnala: il B-BVAR **triplica** le variabili mensili
nello stack, «a much wider VAR».

Ed e' esattamente il caso in cui la Minnesota serve davvero: 421 coefficienti per
equazione su ~135 trimestri e' un problema in cui l'OLS non esiste nemmeno, e
tutto il lavoro lo fa lo shrinkage gerarchico del Gate 1.

Nota di simmetria con l'L-BVAR: li' il sistema e' n = 37 con p = 17 (stato 629),
qui e' n_b = 84 con p = 5 (stato 420).  Sono due modi di spendere lo stesso
budget di parametri — l'L-BVAR in profondita' temporale, il B-BVAR in ampiezza.


DOVE ENTRA IL KALMAN, E PERCHE' SOLO A VALLE
---------------------------------------------
Come Q e C, e a differenza di L: la stima e' in forma chiusa sul posterior NIW e
vuole un pannello pieno, quindi il Kalman interviene **dopo**, solo per il bordo
frastagliato.  Il documento di contesto lo dice esplicitamente — «B-BVAR uses the
core on the stacked (wider) system» + «Kalman (filter only)» — e rimanda a
Banbura, Giannone & Lenza (2015) per le equazioni del bordo.

Da cui, di nuovo, l'invariante del pacchetto:

    >>> il core sampler non vede mai un NaN.

Il pannello di stima si ferma a `endEstimT` (l'ultimo trimestre con la riga del
mese 3 completa) e i buchi residui si riempiono prima di entrare nel core.


================================================================================
I DETTAGLI CHE STANNO SOLO NEL CODICE  (`bbvar.m`, `STEP1_BBVAR.m`)
================================================================================

--- 1. IL TERZO BLOCCO PRENDE **TUTTE** LE COLONNE ------------------------

    month1 = X(1:3:end, mSeries);    % solo le MENSILI
    month2 = X(2:3:end, mSeries);    % solo le MENSILI
    month3 = X(3:3:end, :);          % TUTTE, mensili + trimestrali

E' la riga che si sbaglia leggendo il paper.  Verrebbe da impilare tre volte le
sole mensili e aggiungere le trimestrali a parte; invece le trimestrali stanno
gia' dentro `month3`, perche' sono osservate proprio nel terzo mese.

ATTENZIONE a come si controlla, ed e' il Gate 4 ad averlo scoperto: la lettura
sbagliata **non si vede nella dimensione**, perche' `3*n_m + n_q` e
`2*n_m + n` sono lo stesso numero (n_q = n - n_m).  Si vede nell'**ordine delle
colonne**: qui il terzo blocco tiene ogni trimestrale al suo posto originale,
mentre l'altra lettura le accoda in fondo.  Stesso n_b, colonne scambiate, e
nessun errore sollevato — e' l'ordine che `STEP1_BBVAR.m` r.79 assume per
rileggere i risultati.  Lo verifica `test_gate4.py` §1.

--- 2. `pos` (il nostro `d_centre`) SI REPLICA COL BLOCCO -----------------

    [~,pos] = find([pos(mSeries) pos(mSeries) pos] == 1);

il centraggio rw/wn dell'eq. (2) segue lo stesso impilamento: una serie `wn`
resta `wn` in tutti e tre i suoi blocchi.  Ovvio a dirsi, ma e' il genere di cosa
che se sbagliata non solleva errori — produce solo un prior storto.

--- 3. LA SPLINE E' IL METODO **1**, NON IL 5 ----------------------------

    optNaN.method = 1;   optNaN.k = 3;

`remNaNs_spline` metodo 1: ogni buco -> mediana della serie, poi sostituito con
una media mobile centrata a 2k+1 = 7 termini.  **Niente spline cubica sui buchi
interni**, che e' invece il metodo 5 usato dall'L-BVAR.  Divergenza voluta fra i
due modelli, non una svista: qui il pannello di stima e' gia' tagliato a
`endEstimT`, quindi i buchi che restano sono partenze tardive, dove una spline
non avrebbe nulla su cui interpolare.

--- 4. `MCMCconst = 0.14`, NON 1.6 --------------------------------------

Riga 34 contro l'1.6 di `lbvar.m` (e l'1 di `qbvar.m`, lo 0.5 di `crbvar.m`).
Piu' di dieci volte piu' stretta, ed e' coerente: il sistema blocked ha
n_b = 84 variabili e quindi **d = 86 iperparametri**, contro i 39 dell'L-BVAR —
in dimensione piu' alta la proposta va accorciata per tenere l'accettazione
ragionevole.  E' la stessa logica per cui `tune` esiste.

E VALE LA PENA LEGGERLO PER QUEL CHE E': la sequenza 1 / 0.5 / 0.14 / 1.6
cablata a mano nei loro quattro modelli e' **la taratura 1/sqrt(d) fatta a
occhio**.  Gli autori hanno incontrato il problema della dimensione — e' scritto
nei loro numeri — ma nel codice non c'e' nessun ESS, nessun R-hat, nessuna
autocorrelazione: solo `ACCrate`, calcolata a posteriori e mai usata per agire.
Con la sola accettazione a bersaglio il campionatore SEMBRA sano, ed e'
esattamente la situazione del B-BVAR: accettazione 20.8%, ESS/iterazione 0.015.
Misura e trattazione in `tests/test_mixing` e nell'header di `hyper.py`,
sezione "IL MESCOLAMENTO".

--- 5. IL FILTRO GIRA SULLA FINESTRA TERMINALE --------------------------

    [X_sm,~] = run_Ksmoother_FM(betaHat, sigmaHat, lags, xQ(endEstimT-lags:end,:));
    X_sm = [xQ(lags+1:endEstimT-1,:); X_sm];

Identico a `crbvar.m` (C-BVAR) e a `lbvar_NE.m` (riuso dell'L-BVAR): si filtra
**solo il bordo** e la storia si prende dai dati osservati.  Il Gate 3 aveva gia'
misurato perche' — la ricorsione lunga degrada — e il Gate 5 l'ha ritrovato nel
ramo di riuso.  Tre modelli su quattro fanno la stessa cosa.

--- 6. L'INIZIALIZZAZIONE: LORO USANO IL LYAPUNOV, NOI NON POSSIAMO -------

    initX = <le prime `lags` righe di yQ, piu' recente per prima>
    initX(isnan(initX)) = 0;
    initV = lyapunov_symm(A, Q, 1, 1e-6);

Loro inizializzano con la varianza NON CONDIZIONATA dello stato.  Da noi non
esiste: **rho(A) = 1.014 - 1.027**, il VAR blocked stimato e' lievemente
esplosivo — come il Q-BVAR (rho ~ 1.012, gia' documentato nel README) e per la
stessa ragione, dati in log-livelli.  Siamo fuori dalle ipotesi del loro codice
esattamente come il C-BVAR al Gate 3.

DECISIONE: `P0 = 0`, cioe' la convenzione dell'ALTRO loro modello (`lbvar.m`
riga 52).  E' legittima perche' le prime `p` righe della finestra sono dati
OSSERVATI (riempiti), quindi trattarle come note non inventa informazione.

ERRORE FATTO E CORRETTO, tenuto perche' e' istruttivo.  Una prima versione
ripiegava su una diffusa `1e4*I` quando il Lyapunov falliva.  Sembrava prudente
ed era il contrario: il nowcast del trimestre obiettivo usciva **-97.74%** con
banda superiore 6.7e15.  Sondando le due scelte sospette a (B, Sigma) fissi:

    P0        finestra 5 (2 lisciati)   12 (9)   24 (21)
    0                        1.36        1.44     1.59
    1e2*I                  -29.54        1.44     1.59
    1e4*I                  -97.74        1.44     1.59

Il guasto e' l'INTERAZIONE fra P0 diffusa e finestra corta: con 2 soli periodi
lisciati la diffusione iniziale non viene mai consumata dai dati e lo stato al
bordo resta non identificato (|max| dello stato 331 contro 84.6).  E' lo stesso
meccanismo per cui `state_space.py` esclude le prime osservazioni dalla
verosimiglianza (`n_diffuse`), visto dal lato opposto.

Da cui anche il secondo rimedio, adottato insieme al primo perche' e' gratis:
`edge_lags` di default 24 trimestri invece di `p`.  Con la finestra minima degli
autori (`lags`, cioe' 1-2 periodi lisciati) il sistema e' sano solo con P0 = 0,
e non conviene dipendere da una condizione sola.

--- 7. LA CRESCITA SI CALCOLA SU t-1, NON t-3 ---------------------------

    X_transf(2:end,qSeries) = 100*((X_transf(2:end,qSeries)./X_transf(1:end-1,qSeries)).^4-1);

perche' il pannello blocked e' **gia' trimestrale**: ogni riga e' un trimestre.
Nel C/L-BVAR il pannello e' mensile e la stessa crescita si fa su t-3.  Sbagliare
qui darebbe un tasso plausibile ma sfasato di due trimestri.

--- 8. I CI DEGENERI SONO CORRETTI, E SONO ANCHE LORO ---------------------

Leggendo l'output del fit si vede questo, e la prima reazione e' che sia rotto:

    2025-12-31      0.48   [   0.48,    0.48]      <- banda nulla
    2026-03-31      2.09   [   2.09,    2.09]      <- banda nulla
    2026-06-30      1.37   [  -1.17,    3.93]      <- l'unica banda vera

Non e' rotto, e non va risolto.  **Un solo trimestre ha una densita' perche' un
solo trimestre ha il PIL latente** — gli altri il PIL ce l'hanno osservato, e un
dato osservato non ha distribuzione posteriore.  Le due righe che lo producono
sono deliberate e stanno anche nel codice degli autori, in forma piu' stretta
della nostra:

  a) `run_Ssmoother_block.m` r.64 passa **`zeros(n)`** come covarianza di misura
     (il quinto argomento di `runKF_DK`; idem `run_Ksmoother_FM.m` r.37).  R = 0
     esattamente: una cella osservata e' vincolata al suo valore in OGNI
     estrazione, e `runKF_DK` inverte `F = Z*P*Z'` cosi' com'e'.  Noi ci mettiamo
     un nugget `1e-12`: **QUI e' scelta nostra**, e resta l'unica ragione per cui
     le nostre bande storiche sono larghe ~1e-7 invece di 0, il [4.37, 4.38] qui
     sopra.  Gli autori sono quindi PIU' degeneri di noi, non meno.

     PRECISAZIONE (2026-08-01): "scelta nostra" vale per il B-BVAR e basta.
     Nell'L-BVAR gli autori usano `1e-12 * I`, identico al nostro
     (`lbvar.m` r.88).  La divergenza e' quindi di questo modulo, non del
     pacchetto — vedi la nota in `lbvar.build_state_space`.

  b) `bbvar.m` r.52-56 riempie ogni estrazione `ss` con lo stesso `xQ`
     osservato, per tutte le righe prima della finestra terminale.  E' il
     `np.repeat(hist[None], S, axis=0)` della sezione 4 qui sotto.

Che loro sappiano che la storia e' degenere si vede da come la usano:
`STEP1_BBVAR.m` r.172 costruisce le distribuzioni predittive `res.pd` solo per
`hh = -1:horizon/3`, cioe' dal trimestre del nowcast in avanti.  La parte
storica non entra mai come densita'.

L'IPOTESI ALTERNATIVA, FALSIFICATA, perche' e' quella che viene in mente e non
val la pena rifarla.  Si potrebbe pensare che il blocking crei identita'
deterministiche fra i blocchi (m1, m2, m3 della stessa serie), e che certe
combinazioni lineari abbiano varianza nulla per costruzione.  Misurato sul
pannello di stima vero (137 x 84):

    valori singolari   max 86.586   min 2.799e-03   cond 3.1e4
    rango numerico (tol 1e-8)   84   = pieno

Un'identita' darebbe min(s) ~ 0 e cond ~ 1e16.  E non puo' esserci: e' proprio
la differenza col Q-BVAR — la media mobile a 3 mesi COMPRIME tre numeri in uno e
crea la relazione, il blocking li tiene separati apposta per non crearla.

Quello che c'e' davvero e' una CRESTA, non un'identita', e va detto perche'
conta altrove: fra i tre blocchi della stessa serie la correlazione arriva a
1.0000 a quattro decimali su PCEPILFE, CPILFESL, PCEPI (indici di prezzo in
log-livelli: tre mesi consecutivi sono quasi lo stesso numero).  Non tocca i CI
del nowcast, ma e' la ragione per cui qui il prior Minnesota non e' un
accessorio: 421 coefficienti per equazione su regressori quasi collineari.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.bvar.core import BVARDraws, sample
from src.bvar.data import append_forecast_rows, build_panel, load_raw_levels
from src.bvar.hyper import HyperPrior
from src.bvar.lbvar import build_state_space
from src.bvar.simsmoother import simulation_smoother
from src.bvar.spec import BVARSpec, MinnesotaSpec

#: `optNaN.k` di `bbvar.m` riga 30 — media mobile a 2k+1 = 7 termini.
FILL_K = 3

#: `MCMCconst` di `bbvar.m` riga 34.  Vedi il punto 4 dell'header.
MCMC_CONST = 0.14

#: I mesi in cui una serie trimestrale porta un'osservazione.
QUARTER_END_MONTHS = (3, 6, 9, 12)


# ─── 1. Lo spec del sistema impilato ──────────────────────────────────────────

def blocked_spec(spec: BVARSpec) -> BVARSpec:
    """
    Lo `BVARSpec` del sistema blocked — `bbvar.m` righe 18-27.

    L'ordine delle colonne e' quello degli autori e va rispettato, perche' e' lo
    stesso che usano per rileggere i risultati (`STEP1_BBVAR.m` riga 79):

        [ <mensili>_m1 , <mensili>_m2 , <tutte le serie> ]

    `d_centre` segue lo stesso impilamento (punto 2 dell'header).
    """
    m = [i for i, f in enumerate(spec.freq) if f == "M"]
    d = spec.minnesota.d_centre
    names = (tuple(f"{spec.series[i]}_m1" for i in m)
             + tuple(f"{spec.series[i]}_m2" for i in m)
             + tuple(spec.series))
    freq = tuple(["Q"] * (2 * len(m)) + list(spec.freq))
    transform = (tuple(spec.transform[i] for i in m) * 2 + tuple(spec.transform))
    d_b = tuple([d[i] for i in m] * 2 + list(d))
    return BVARSpec(
        model="B", profile=spec.profile, series=names, freq=freq,
        transform=transform, sample_start=spec.sample_start,
        minnesota=MinnesotaSpec(d_centre=d_b, p=spec.minnesota.p,
                                **{k: v for k, v in vars(spec.minnesota).items()
                                   if k not in ("d_centre", "p")}),
        config=spec.config,
    )


def monthly_mask(spec: BVARSpec) -> np.ndarray:
    """Maschera booleana (n,) delle serie mensili — il `mSeries` degli autori."""
    return np.array([f == "M" for f in spec.freq])


# ─── 2. Il blocking ───────────────────────────────────────────────────────────

def block_panel(panel: pd.DataFrame, spec: BVARSpec) -> pd.DataFrame:
    """
    Da pannello MENSILE (T, n) a pannello TRIMESTRALE blocked (T/3, 2*n_m + n).

    `bbvar.m` righe 23-27.  Il terzo blocco prende TUTTE le colonne — punto 1
    dell'header, ed e' l'errore piu' facile da fare leggendo il solo paper.

    L'indice del pannello mensile deve iniziare al PRIMO mese di un trimestre
    (gennaio, aprile, luglio, ottobre) e contenere trimestri interi: e'
    l'allineamento che il taglio `X(1:3:end)` assume implicitamente, e che qui
    si verifica invece di sperarci.
    """
    idx = panel.index
    if idx[0].month not in (1, 4, 7, 10):
        raise ValueError(
            f"il pannello inizia a {idx[0].date()}: il blocking vuole il primo "
            f"mese di un trimestre (1, 4, 7, 10).  Sistemare `sample_start`."
        )
    T = len(idx)
    if T % 3:
        panel, idx = panel.iloc[: T - T % 3], idx[: T - T % 3]

    X = panel.to_numpy(dtype=float)
    m = monthly_mask(spec)
    blocked = np.column_stack([X[0::3][:, m], X[1::3][:, m], X[2::3]])
    return pd.DataFrame(blocked, index=idx[2::3],
                        columns=list(blocked_spec(spec).series))


def last_full_quarter(blocked: pd.DataFrame, spec: BVARSpec) -> int:
    """
    `endEstimT` di `bbvar.m` riga 26, in indice 0-based.

    E' l'ultimo trimestre in cui la riga del MESE 3 e' completamente osservata —
    calcolato sulle sole colonne del terzo blocco, come fanno loro
    (`find(~isnan(sum(X(3:3:end,:),2)),1,'last')`), non su tutta la riga blocked.
    """
    n_third = len(spec.series)
    third = blocked.to_numpy(dtype=float)[:, -n_third:]
    pieno = ~np.isnan(third).any(axis=1)
    if not pieno.any():
        raise ValueError("nessun trimestre con la riga del mese 3 completa")
    return int(np.flatnonzero(pieno)[-1])


def fill_median_ma(X: np.ndarray, *, k: int = FILL_K) -> np.ndarray:
    """
    `remNaNs_spline` metodo **1** — punto 3 dell'header.

    Ogni buco -> mediana della serie, poi sostituito con una media mobile
    CENTRATA a 2k+1 termini della serie cosi' riempita.  Niente spline cubica:
    quella e' il metodo 5, ed e' dell'L-BVAR.
    """
    X = np.asarray(X, dtype=float).copy()
    for i in range(X.shape[1]):
        x = X[:, i]
        na = np.isnan(x)
        if not na.any():
            continue
        if na.all():
            X[:, i] = 0.0
            continue
        x[na] = np.nanmedian(x)
        pad = np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])])
        ma = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        x[na] = ma[na]
        X[:, i] = x
    return X


#: Finestra terminale, in trimestri.  Gli autori usano `lags`; noi 24 — vedi il
#: punto 6 dell'header, e la tabella che lo motiva.
EDGE_LAGS = 24


def spectral_radius(A: np.ndarray) -> float:
    """rho(A) della companion.  >= 1 significa varianza non condizionata assente."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


# ─── 3. Il risultato ──────────────────────────────────────────────────────────

@dataclass
class BBVARDraws:
    """Le estrazioni del B-BVAR sul pannello blocked."""

    panels: np.ndarray            # (S, W, n_b)  il pannello blocked completato
    draws: BVARDraws = field(repr=False, default=None)
    index: pd.DatetimeIndex = field(repr=False, default=None)
    spec: BVARSpec = field(repr=False, default=None)        # lo spec BLOCKED
    base_spec: BVARSpec = field(repr=False, default=None)   # quello originale
    rho: float = float("nan")

    @property
    def S(self) -> int:
        return int(self.panels.shape[0])

    def growth(self, series_id: str = "GDPC1") -> pd.DataFrame:
        """
        Crescita trimestrale annualizzata `100*((x_t/x_{t-1})^4 - 1)`.

        **t-1, non t-3** — punto 7 dell'header: qui ogni riga E' un trimestre.
        """
        j = list(self.spec.series).index(series_id)
        lv = self.panels[:, :, j]
        if self.spec.transform[j] == "log":
            lv = np.exp(lv)
        g = 100.0 * ((lv[:, 1:] / lv[:, :-1]) ** 4 - 1.0)
        return pd.DataFrame(g.T, index=self.index[1:])

    def summary(self, series_id: str = "GDPC1") -> str:
        """
        Mediana e banda 5-95 degli ultimi 4 trimestri.

        ATTESO: solo l'ultimo trimestre ha una banda larga.  Gli altri hanno il
        PIL osservato, e un dato osservato non ha posteriore — non e' un bug,
        e gli autori lo hanno in forma ancora piu' netta (R = 0 invece del
        nostro nugget).  Punto 8 dell'header.
        """
        q = self.growth(series_id).quantile([0.05, 0.5, 0.95], axis=1).T
        righe = "\n".join(
            f"    {d.date()}   {q.loc[d, 0.5]:7.2f}   "
            f"[{q.loc[d, 0.05]:7.2f}, {q.loc[d, 0.95]:7.2f}]" for d in q.index[-4:])
        return (f"B-BVAR  S={self.S}  n_b={self.spec.n}  p={self.spec.p}  "
                f"T={self.panels.shape[1]}  rho(A)={self.rho:.4f}\n"
                f"  {self.draws.summary()}\n"
                f"  {series_id}, crescita annualizzata:\n{righe}")


# ─── 4. Il ciclo ──────────────────────────────────────────────────────────────

def fit(
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    n_draws: int = 1000,
    burn: int = 500,
    horizon: int = 0,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    edge_lags: int | None = None,
    p0: str | float | None = None,
    verbose: bool = True,
) -> BBVARDraws:
    """
    Il B-BVAR end-to-end — `bbvar.m`.

    `horizon` : mesi di previsione oltre l'ultimo dato, multiplo di 3.  Vedi
    `append_forecast_rows`.  Default 0 = solo il nowcast; gli autori usano 24.

    I tre stadi:
      1. blocking del pannello mensile  ->  VAR trimestrale a 2*n_m + n variabili
      2. core sampler sul pannello di stima (fino a `endEstimT`, riempito)
      3. simulation smoother sulla FINESTRA TERMINALE per il bordo frastagliato,
         con la storia presa dai dati osservati (punto 5 dell'header)
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("B") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    bspec = blocked_spec(spec)
    p = bspec.p
    edge_lags = EDGE_LAGS if edge_lags is None else int(edge_lags)

    # ── 1. blocking
    monthly = build_panel(spec, as_of, raw=raw)
    monthly = append_forecast_rows(monthly, horizon, align_quarters=True)
    blocked = block_panel(monthly, spec)
    Xb = blocked.to_numpy(dtype=float)
    end = last_full_quarter(blocked, spec)
    if verbose:
        print(f"  blocked {Xb.shape[0]} trimestri x {bspec.n} variabili "
              f"(n_m={int(monthly_mask(spec).sum())}, n={spec.n}), "
              f"k={bspec.k}\n  endEstimT = {blocked.index[end].date()}, "
              f"NaN nel pannello di stima "
              f"{int(np.isnan(Xb[: end + 1]).sum())}")

    # ── 2. il core, sul pannello riempito fino a endEstimT
    est = fill_median_ma(Xb[: end + 1])
    if verbose:
        print("  stima del sistema impilato...")
    draws = sample(bspec, est, n_draws=n_draws, burn=burn, rng=rng,
                   hyperprior=HyperPrior(), c=MCMC_CONST, verbose=verbose)

    # ── 3. lo smoother sulla finestra terminale
    start = max(0, end - edge_lags)
    Yw = Xb[start:]
    head = fill_median_ma(Xb[start:start + p])
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])
    Y_in = Yw[p:]
    S = draws.B.shape[0]
    W = p + Y_in.shape[0]
    panels = np.empty((S, W, bspec.n))
    rho = float("nan")
    if verbose:
        print(f"  smoother sulla finestra terminale: {Yw.shape[0]} trimestri "
              f"({blocked.index[start].date()} - {blocked.index[-1].date()})")
    for s in range(S):
        # P0 = 0: le prime p righe della finestra sono dati OSSERVATI, e la
        # varianza non condizionata non esiste (rho(A) > 1).  Punto 6.
        ss = build_state_space(draws.B[s], draws.Sigma[s], bspec.n, p, a0,
                               p0=p0)
        if s == 0:
            rho = spectral_radius(ss.A)
        panels[s] = np.vstack([head, simulation_smoother(ss, Y_in, rng)[:, :bspec.n]])

    # la storia PRIMA della finestra: dato osservato, replicato (punto 5).
    # Qui la banda posteriore e' nulla per COSTRUZIONE, ed e' la riga 47-51 di
    # `bbvar.m` — non un difetto del campionatore.  Punto 8 dell'header.
    hist = Xb[:start]
    if hist.shape[0]:
        panels = np.concatenate(
            [np.repeat(hist[None, :, :], S, axis=0), panels], axis=1)
    index = blocked.index[: start + W]

    return BBVARDraws(panels=panels, draws=draws, index=index, spec=bspec,
                      base_spec=spec, rho=rho)


def fit_reuse(
    B_draws: np.ndarray,
    Sigma_draws: np.ndarray,
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    horizon: int = 0,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    edge_lags: int | None = None,
    draws_obj=None,
    p0: str | float | None = None,
    verbose: bool = False,
) -> BBVARDraws:
    """
    IL RAMO DI RIUSO — `STEP1_BBVAR.m` righe 132-155, il blocco `else`.

    Gli autori lo commentano da soli: *"otherwise just use existing parameter
    estimates on new vintage"*.  Nessuna stima: si riprendono `beta` e `sigma`
    dall'ultima stima completa e si rifa' **solo lo smoother sulla finestra
    terminale**, che e' l'unica cosa che dipende da `as_of`.

        [X_sm,~] = run_Ksmoother_FM(betaHat, sigmaHat, lags, xQ(endEstimT-lags:end,:));
        X_transf_MCMC(...) = run_Ssmoother_block(beta, sigma, nDraws/nSave, lags, ...);

    Nota che ANCHE il ramo di riuso produce la densita', non solo il punto: la
    seconda riga gira sulle estrazioni conservate.  Le bande ci sono tutte le
    settimane.

    E' il gemello di `lbvar.fit_reuse` (`lbvar_NE.m`).  La differenza sta solo
    in dove passa il bordo: li' mesi, qui trimestri del sistema impilato.

    Parameters
    ----------
    B_draws, Sigma_draws : (S, k_b, n_b) e (S, n_b, n_b)
        Le estrazioni dell'ultima stima completa, gia' nello spazio BLOCCATO.
    horizon : int
        Mesi di previsione oltre l'ultimo dato: si appendono `horizon/3`
        trimestri tutti-NaN, che lo smoother riempie iterando la transizione.
        `STEP1_BBVAR.m` r.98 usa `horizon = 24`.
    draws_obj : BVARDraws | None
        L'oggetto del core dell'ultima stima completa, trasportato tale e quale
        perche' l'uscita abbia la stessa forma.  Non viene usato per stimare.
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("B") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    bspec = blocked_spec(spec)
    p = bspec.p
    edge_lags = EDGE_LAGS if edge_lags is None else int(edge_lags)
    S = int(B_draws.shape[0])

    monthly = build_panel(spec, as_of, raw=raw)
    monthly = append_forecast_rows(monthly, horizon, align_quarters=True)
    blocked = block_panel(monthly, spec)
    Xb = blocked.to_numpy(dtype=float)
    end = last_full_quarter(blocked, spec)

    start = max(0, end - edge_lags)
    Yw = Xb[start:]
    head = fill_median_ma(Xb[start:start + p])
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])
    Y_in = Yw[p:]
    W = p + Y_in.shape[0]
    if verbose:
        print(f"  riuso B: finestra {Yw.shape[0]} trimestri "
              f"({blocked.index[start].date()} - {blocked.index[-1].date()}), S={S}")

    panels = np.empty((S, W, bspec.n))
    rho = float("nan")
    for s in range(S):
        ss = build_state_space(B_draws[s], Sigma_draws[s], bspec.n, p, a0,
                               p0=p0)
        if s == 0:
            rho = spectral_radius(ss.A)
        panels[s] = np.vstack([head, simulation_smoother(ss, Y_in, rng)[:, :bspec.n]])

    hist = Xb[:start]
    if hist.shape[0]:
        panels = np.concatenate(
            [np.repeat(hist[None, :, :], S, axis=0), panels], axis=1)
    index = blocked.index[: start + W]

    return BBVARDraws(panels=panels, draws=draws_obj, index=index, spec=bspec,
                      base_spec=spec, rho=rho)


__all__ = ["blocked_spec", "monthly_mask", "block_panel", "last_full_quarter",
           "fill_median_ma", "spectral_radius", "BBVARDraws", "fit", "fit_reuse",
           "MCMC_CONST", "FILL_K", "EDGE_LAGS"]
