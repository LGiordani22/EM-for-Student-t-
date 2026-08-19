"""
core/bvar/cbvar.py

GATE 3, BLOCCO 3 — IL C-BVAR: la colla dei tre stadi.  §2.4 del paper.

Questo modulo e' SOTTILE di proposito: non contiene teoria propria.  La sostanza
sta gia' dove vive il codice che la implementa, e duplicarla qui sarebbe il modo
piu' rapido di farla divergere:

    la mappa cube-root, le TRE letture dell'App. A, A_0, GMR   -> `cube_root.py`
    la finestra del filtro, Higham, l'inizializzazione         -> `state_space.py`
    il simulation smoother di Durbin-Koopman                   -> `simsmoother.py`
    l'aggregazione mensile->trimestrale (nota 17)              -> `qbvar.py`

Qui c'e' solo l'orchestrazione, piu' l'unica cosa che nessuno degli altri fa:
tradurre gli stati latenti in un NOWCAST leggibile.


================================================================================
COSA DICE IL PAPER
================================================================================
§2.4, in fondo: "In summary, the first step to obtain the C-BVAR is to estimate
a quarterly VAR(p) model, like the one in Section 2.1.  Given estimates of the
parameters of the quarterly model (4), Phi and Omega, we define a monthly model
(6) with parameters Phi_m and Omega_m, which can be recovered from Eqs. (8) and
(10).  Finally, as for the B-BVAR, we compute the distributions of forecasts
conditional on the real-time data flow, exploiting the Kalman filtering methods."

Tre stadi, in quest'ordine.  Il paper li elenca e basta; tutto il resto —
quale finestra, quale inizializzazione, filtro o smoother — sta nel codice di
replica degli autori, non nel testo.


================================================================================
COSA SIGNIFICA
================================================================================
Il C-BVAR non stima niente di mensile: stima trimestrale e DEDUCE la legge
mensile.  Il nowcast e' quindi il prodotto di due oggetti diversi:

    l'INCERTEZZA SUI PARAMETRI   dalle estrazioni MCMC del Q-BVAR
    l'INCERTEZZA SUGLI STATI     dal simulation smoother, per ogni estrazione

Tenerli entrambi e' cio' che rende il nowcast una DENSITA' e non un punto.  Se
si prendesse la media smussata invece di un'estrazione, la densita' risultante
sottostimerebbe l'incertezza esattamente come l'imputazione singola in un Gibbs.


================================================================================
COME SI TRADUCE IN CODICE — e i DUE dettagli presi dal codice degli autori
================================================================================
La catena, per ogni estrazione s del Q-BVAR:

    (Phi[s], Sigma[s], A_0[s])            stadio 1, gia' fatto da `qbvar.fit`
      -> cube_root.quarterly_to_monthly   stadio 2, variante "authors"
      -> state_space.build_state_space    Higham + nugget
      -> simsmoother.simulation_smoother  stadio 3, SULLA FINESTRA
      -> livelli -> crescita annualizzata

--- DETTAGLIO 1: si SMUSSA, non si filtra --------------------------------

`runKF_DK.m` degli autori non e' un filtro: contiene `FIS`, "Fixed interval
smoother (see Durbin and Koopman, 2001, p. 64-71)", con la ricorsione
all'indietro.  E per la densita' `run_Ssmoother.m` e' il simulation smoother DK
completo.

La differenza conta: filtrando, la stima del mese t usa solo l'informazione fino
a t; smussando usa tutta la finestra.  Per i mesi latenti dentro il trimestre in
corso — che sono esattamente quelli che ci interessano — e' la differenza fra
usare o buttare via i dati dei mesi successivi gia' pubblicati.

Riusiamo quindi `simsmoother.py` del Gate 5: e' lo stesso algoritmo, gia'
validato contro l'oracolo esatto (media a 1e-15, covarianza entro l'1%).

--- DETTAGLIO 2: l'uscita e' la CRESCITA ANNUALIZZATA ---------------------

`STEP2_CRBVAR.m`:

    X_transf(:,toLog) = exp(X_transf(:,toLog));
    X_transf(4:end,qSeries) = 100*((X_transf(4:end,qSeries)./X_transf(1:end-3,qSeries)).^4-1);

cioe': si torna ai livelli con l'esponenziale, poi per le serie TRIMESTRALI si
calcola 100*((x_t/x_{t-3})^4 - 1) — il tasso di crescita trimestrale
ANNUALIZZATO, con x_{t-3} tre MESI indietro sull'indice mensile, cioe' un
trimestre.  E' il numero che si confronta col dato BEA e col NY Fed.

Nota sull'ordine: l'esponenziale PRIMA del rapporto.  Fare il rapporto in log e
poi esponenziare darebbe la stessa cosa solo per il rapporto puro, non dopo
l'elevamento alla quarta e il -1.


LA CONVENZIONE DEI DUE PANNELLI, da non confondere
---------------------------------------------------
    pannello di STIMA      trimestrale, denso, chiuso a `estimation_end`
                           -> stadio 1
    pannello di NOWCAST    mensile, con bordo frastagliato, mascherato ad `as_of`
                           -> stadio 3

Sono due oggetti diversi e servono a due cose diverse: il primo produce i
parametri, il secondo l'informazione su cui condizionare.  E' la stessa
separazione descritta in `data.estimation_end`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.bvar.cube_root import DEFAULT_COUPLING, quarterly_to_monthly
from core.bvar.data import load_raw_levels
from core.bvar.qbvar import QUARTER_END_MONTHS, build_monthly_panel, estimation_panel
from core.bvar.qbvar import fit as fit_qbvar
from core.bvar.simsmoother import LinearGaussianSS, simulation_smoother
from core.bvar.state_space import (
    MonthlyStateSpace,
    P0_DEFAULT,
    PSDReport,
    build_state_space,
    edge_window,
    initial_covariance,
    psd_summary,
)
from core.bvar.spec import BVARSpec


def _as_linear_gaussian(ss: MonthlyStateSpace, f0: np.ndarray,
                        P0: np.ndarray) -> LinearGaussianSS:
    """
    Ponte fra i due contenitori: `MonthlyStateSpace` (Gate 3) e
    `LinearGaussianSS` (Gate 5).  Descrivono lo stesso oggetto con nomi diversi
    — il primo e' specifico del C-BVAR, il secondo e' generico perche' doveva
    essere testabile su modelli minuscoli.
    """
    return LinearGaussianSS(A=ss.Phi_m, Q=ss.Q, Z=ss.Lambda, R=ss.R,
                            c=ss.c_m, a0=f0, P0=P0)


def _initial_covariance(ss: MonthlyStateSpace, *,
                        p0: str | float = P0_DEFAULT) -> tuple[np.ndarray, str]:
    """
    P_0 per la finestra — la scelta sta in `state_space.initial_covariance`, che
    e' condivisa dai quattro modelli.  Il default e' `lyapunov_symm`, la riga di
    `run_Ksmoother.m`; il C e' anche il modello in cui P_0 = 0 NON e' innocuo,
    perche' il suo `initX` e' vecchio di due mesi (vedi li').
    """
    return initial_covariance(ss.Phi_m, ss.Q, kind=p0)


def _edge(spec: BVARSpec, as_of, *, horizon: int, months: int | None,
          raw: pd.DataFrame):
    """
    Il pannello di NOWCAST, la finestra, lo stato iniziale e la storia osservata
    — in un posto solo perche' `fit` e `fit_reuse` devono vedere lo STESSO bordo.

    IL TAGLIO IN FONDO, e perche' non e' opzionale.  `known_at` MASCHERA ma non
    ACCORCIA: l'indice arriva comunque alla fine del file, quindi dopo `as_of`
    restano centinaia di righe tutte NaN.  Gli autori tagliano esplicito —
    `X = X(1:nowcastM+horizon,:)`, STEP2_CRBVAR.m r.112/141 — dove `nowcastM` e'
    l'ultimo mese del trimestre obiettivo, qui quello che contiene `as_of`.

    E LE `horizon` RIGHE DI NAN CI VANNO DENTRO.  Sono la previsione: il filtro
    non ha nulla da aggiornare e proietta.  Vedi `state_space.edge_window`, che
    ancora la finestra all'ultimo quarter-end OSSERVATO proprio perche' la coda
    e' vuota per costruzione.

    Returns
    -------
    (Y_win, f0, idx, history, hist_index)
    """
    if months is not None and horizon:
        raise ValueError(
            "months e horizon insieme no: `months` ancora la finestra alla fine "
            "dell'indice, e con `horizon` righe di NaN appese l'ancora finisce "
            "nel vuoto (era il bloccante del C-BVAR).  `months` e' rimasto solo "
            "per la diagnostica di `edge_window`."
        )
    p = spec.p
    mon = build_monthly_panel(spec, as_of, raw=raw)
    target = _target_quarter_end(mon.index, as_of)
    mon = mon.loc[:target + pd.offsets.MonthEnd(horizon)] if horizon else mon.loc[:target]

    qe = mon.index.month.isin(QUARTER_END_MONTHS)
    Y = mon.to_numpy(dtype=float)
    Y_win, f0, i0 = edge_window(Y, qe, p, months=months)
    if np.isnan(f0).any():
        raise ValueError(
            "lo stato iniziale contiene NaN: i quarter-end che precedono la "
            "finestra non sono pienamente osservati.  Di solito significa che "
            "`as_of` e' troppo vicino all'inizio del campione."
        )
    # la storia: `X(3*lags+1:endEstimT-1,:)` degli autori.  Le prime 3p righe
    # cadono perche' nel loro ramo pieno sono l'avvio dello smoother; qui sono
    # solo dato, ma si taglia lo stesso per avere lo stesso indice di uscita.
    h0 = min(3 * p, i0)
    return Y_win, f0, mon.index[i0:], Y[h0:i0], mon.index[h0:i0]


def _target_quarter_end(index: pd.DatetimeIndex, as_of) -> pd.Timestamp:
    """
    L'ultimo mese del trimestre OBIETTIVO — il `nowcastM` di STEP2_CRBVAR.m.

    E' il quarter-end del trimestre che contiene `as_of`; con `as_of=None` e'
    l'ultimo quarter-end dell'indice.  Serve a tagliare il pannello di nowcast:
    vedi il commento in `fit`.
    """
    qe = index[index.month.isin(QUARTER_END_MONTHS)]
    if as_of is None:
        return qe[-1]
    t = pd.Timestamp(as_of)
    dopo = qe[qe >= t]
    return dopo[0] if len(dopo) else qe[-1]


@dataclass
class CBVARNowcast:
    """
    L'uscita del C-BVAR: estrazioni del pannello mensile latente sulla finestra,
    la storia osservata che la precede, e gli strumenti per leggerne il nowcast.

    I DUE PEZZI, e perche' non sono lo stesso pezzo.  `STEP2_CRBVAR.m` r.150:

        X_transf = [X(3*lags+1:endEstimT-1,:); X_sm];

    prima di `endEstimT` c'e' IL DATO, dopo c'e' quel che ha prodotto il filtro.
    Tenerli separati invece di replicare la storia S volte non e' solo memoria
    (a S = 1000 sarebbero ~100 MB di righe tutte uguali): e' che il lettore deve
    poter sapere quale trimestre e' stato stimato e quale e' stato letto.

    Attributes
    ----------
    draws : (S, W, n)     livelli mensili in UNITA' DEL MODELLO (log dove serve),
                          SOLO la finestra — da `endEstimT` alla fine del bordo
    index : DatetimeIndex lungo W, le date della finestra
    history : (H, n)      il pannello OSSERVATO che precede la finestra, uguale
                          per tutte le estrazioni.  Puo' contenere NaN: le
                          trimestrali fuori dai quarter-end, e le serie che
                          iniziano tardi.
    hist_index : DatetimeIndex lungo H
    spec  : BVARSpec
    psd   : list[PSDReport]   una per estrazione, per il conto di Higham
    init  : dict              quante estrazioni hanno usato Lyapunov vs fallback
    systems : list[(ss, P0)]  I SISTEMI MENSILI GIA' MAPPATI, uno per estrazione.
                              Non dipendono da `as_of`: sono la cache che rende
                              quasi gratuito il ramo di riuso.  Vedi `fit_reuse`.
    """

    draws: np.ndarray
    index: pd.DatetimeIndex
    spec: BVARSpec = field(repr=False)
    psd: list = field(repr=False, default_factory=list)
    init: dict = field(default_factory=dict)
    systems: list = field(repr=False, default_factory=list)
    history: np.ndarray | None = field(repr=False, default=None)
    hist_index: pd.DatetimeIndex | None = field(repr=False, default=None)

    @property
    def S(self) -> int:
        return int(self.draws.shape[0])

    @property
    def full_index(self) -> pd.DatetimeIndex:
        """Storia osservata + finestra: l'indice di `X_transf`."""
        if self.hist_index is None or len(self.hist_index) == 0:
            return self.index
        return self.hist_index.append(self.index)

    def _column(self, series_id: str) -> np.ndarray:
        """
        (S, H+W) — una serie sola, in LIVELLI, storia inclusa.

        Per colonna e non per pannello: `growth` ne usa una, e materializzare
        (S, H+W, n) per buttarne via 29 trentesimi costa un ordine di grandezza
        di memoria in piu' senza servire a nessuno.
        """
        j = list(self.spec.series).index(series_id)
        col = self.draws[:, :, j]
        if self.history is not None and len(self.history):
            hist = np.broadcast_to(self.history[:, j], (self.S, len(self.history)))
            col = np.concatenate([hist, col], axis=1)
        return np.exp(col) if self.spec.transform[j] == "log" else col

    def levels(self) -> np.ndarray:
        """
        (S, H+W, n) — i livelli, con l'esponenziale dove la config dice `log`.

        Materializza la storia S volte: a S grande e' un oggetto pesante.  Chi
        vuole una serie sola usi `_column`, che e' quel che fa `growth`.
        """
        if self.history is not None and len(self.history):
            hist = np.broadcast_to(self.history, (self.S, *self.history.shape))
            out = np.concatenate([hist, self.draws], axis=1)   # gia' una copia
        else:
            out = self.draws.copy()
        is_log = np.array([t == "log" for t in self.spec.transform])
        out[:, :, is_log] = np.exp(out[:, :, is_log])
        return out

    def growth(self, series_id: str) -> pd.DataFrame:
        """
        La crescita trimestrale ANNUALIZZATA, `100*((x_t/x_{t-3})^4 - 1)`,
        campionata ai quarter-end — l'uscita di `STEP2_CRBVAR.m`.

        Sui trimestri PRIMA della finestra le estrazioni sono identiche fra loro
        (e' il dato osservato, non una posteriore); la banda si apre da
        `endEstimT` in poi.  Non e' un difetto: e' la riga 150 di STEP2.

        Returns
        -------
        pd.DataFrame  righe = quarter-end, colonne = estrazioni.
        """
        lv = self._column(series_id)                      # (S, H+W)
        if lv.shape[1] <= 3:
            raise ValueError(
                f"finestra troppo corta ({lv.shape[1]} mesi) per una crescita "
                f"su 3 mesi: servono almeno 4 mesi."
            )
        g = 100.0 * ((lv[:, 3:] / lv[:, :-3]) ** 4 - 1.0)
        idx = self.full_index[3:]
        qe = idx.month.isin(QUARTER_END_MONTHS)
        return pd.DataFrame(g[:, qe].T, index=idx[qe])

    def summary(self, series_id: str = "GDPC1") -> str:
        g = self.growth(series_id)
        q = g.quantile([0.05, 0.5, 0.95], axis=1).T
        righe = "\n".join(
            f"    {d.date()}   {q.loc[d, 0.5]:7.2f}   "
            f"[{q.loc[d, 0.05]:7.2f}, {q.loc[d, 0.95]:7.2f}]"
            for d in q.index[-4:]
        )
        s = psd_summary(self.psd)
        h = 0 if self.history is None else len(self.history)
        return (
            f"C-BVAR  S={self.S}  finestra {len(self.index)} mesi "
            f"({self.index[0].date()} - {self.index[-1].date()})"
            f" + {h} mesi di storia osservata\n"
            f"  init: {self.init}\n"
            f"  Higham: mediana {s.get('rel_frobenius_median', 0):.1%}, "
            f"p90 {s.get('rel_frobenius_p90', 0):.1%}\n"
            f"  {series_id}, crescita annualizzata (mediana e banda 90%):\n{righe}"
        )


def fit(
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    n_draws: int = 200,
    burn: int = 300,
    months: int | None = None,
    horizon: int = 0,
    variant: str = DEFAULT_COUPLING,
    p0: str | float = P0_DEFAULT,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    verbose: bool = False,
) -> CBVARNowcast:
    """
    I tre stadi del C-BVAR, dall'inizio alla fine.

    Parameters
    ----------
    as_of : date-like | None   la data di osservazione per il pannello di nowcast
    months : int | None        SOLO diagnostica, e incompatibile con `horizon`:
                               vedi `state_space.edge_window`.  None = la
                               convenzione degli autori (ancora all'ultimo
                               quarter-end osservato).
    horizon : int              mesi di previsione OLTRE il trimestre obiettivo,
                               tutti NaN, che il filtro riempie proiettando.
                               Gli autori usano 24; default 0 = solo il nowcast.
    variant : str              la formula di accoppiamento (vedi `cube_root`)
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("C") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    n = spec.n

    # ── stadio 1: il Q-BVAR sul pannello di STIMA ──
    #
    # `as_of` VA PROPAGATO QUI DENTRO.  Fino al Gate 5 non lo era, e il
    # campione di stima arrivava sempre a `estimation_end` (2025-09-30) anche
    # per un nowcast del 2008: in un run real-time e' look-ahead puro.  Preso
    # dall'oracolo del Gate 6 (`tests/test_gate6.py`); `estimation_panel` ora
    # taglia la coda all'ultimo trimestre pienamente osservato a `as_of`.
    qfit = fit_qbvar(spec, estimation_panel(spec, as_of=as_of, raw=raw),
                     n_draws=n_draws, burn=burn, rng=rng, verbose=verbose)

    # ── il pannello di NOWCAST: mensile, con il bordo ──
    Y_win, f0, idx, hist, hidx = _edge(spec, as_of, horizon=horizon,
                                       months=months, raw=raw)

    # ── stadi 2 e 3, estrazione per estrazione ──
    W = Y_win.shape[0]
    draws = np.empty((qfit.S, W, n))
    reps: list[PSDReport] = []
    systems, inits = [], []          # la cache del ramo di riuso, vedi `fit_reuse`
    come = {"lyapunov": 0, "kappa": 0, "fallback": 0}
    for s in range(qfit.S):
        mm = quarterly_to_monthly(qfit.companion(s), qfit.Sigma[s], n=n,
                                  const=qfit.const[s], variant=variant)
        ss = build_state_space(mm)
        reps.append(ss.psd)
        P0, how = _initial_covariance(ss, p0=p0)
        come[how] += 1
        systems.append(ss)
        inits.append(P0)
        lg = _as_linear_gaussian(ss, f0, P0)
        alpha = simulation_smoother(lg, Y_win, rng)          # (W, ns)
        draws[s] = alpha[:, :n]
        if verbose and (s + 1) % max(1, qfit.S // 5) == 0:
            print(f"    C-BVAR draw {s + 1}/{qfit.S}")

    return CBVARNowcast(draws=draws, index=idx, spec=spec, psd=reps, init=come,
                        systems=list(zip(systems, inits)),
                        history=hist, hist_index=hidx)


# ─── Il ramo di riuso, e la cache che lo rende quasi gratis ───────────────────

def fit_reuse(
    systems: list,
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    months: int | None = None,
    horizon: int = 0,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    verbose: bool = False,
) -> CBVARNowcast:
    """
    IL RAMO DI RIUSO del C-BVAR, e **una divergenza di efficienza dichiarata**.

    CHE COSA NON DIPENDE DA `as_of`, ed e' il punto.  Lo stadio 2 — la mappa
    cube-root — prende in ingresso `(Phi, Sigma_eps, c)` del Q-BVAR e restituisce
    `(Phi_m, Sigma_epsm, c_m)`.  **Non tocca i dati.**  Quindi, fissate le
    estrazioni del Q-BVAR, il sistema mensile e' lo stesso in ogni settimana del
    trimestre, e con lui:

        * la radice cubica di una companion (n*p x n*p)   — autovalori, la parte cara
        * la (A.9)/(A.10) per Sigma_epsm
        * la proiezione di Higham sulla PSD piu' vicina    — l'altra parte cara
        * P0, che dipende solo dal sistema

    L'unica cosa che cambia da una settimana all'altra e' la **finestra di
    bordo** `Y_win` e il suo stato iniziale `f0`.  Da cui: si cachano i sistemi
    (`CBVARNowcast.systems`) e si rifa' solo il simulation smoother.

    DIVERGENZA DA DICHIARARE IN TESI.  Gli autori non hanno questa cache — ma
    non perche' abbiano deciso diversamente: in `crbvar.m` **il ramo di riuso
    non esiste proprio**, `STEP2_CRBVAR.m` ristima a ogni vintage.  Noi
    adottiamo lo schema a due velocita' che loro usano per B e L (`STEP1` r.113,
    `STEP3` r.87) anche per il C, e la cache e' la conseguenza.  E' **esatta**:
    a parita' di estrazioni del Q-BVAR e di seme, l'uscita e' bit-identica a
    quella che darebbe ricalcolare la mappa.  Nessuna approssimazione.

    Parameters
    ----------
    systems : list[tuple[MonthlyStateSpace, np.ndarray]]
        `(ss, P0)` per estrazione — il campo `systems` di un `CBVARNowcast`
        prodotto da `fit`.
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("C") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    n = spec.n

    Y_win, f0, idx, hist, hidx = _edge(spec, as_of, horizon=horizon,
                                       months=months, raw=raw)

    S = len(systems)
    W = Y_win.shape[0]
    draws = np.empty((S, W, n))
    if verbose:
        print(f"  riuso C: finestra {W} mesi ({idx[0].date()} - {idx[-1].date()}), S={S}")
    for s, (ss, P0) in enumerate(systems):
        lg = _as_linear_gaussian(ss, f0, P0)
        draws[s] = simulation_smoother(lg, Y_win, rng)[:, :n]

    return CBVARNowcast(draws=draws, index=idx, spec=spec,
                        psd=[ss.psd for ss, _ in systems], systems=systems,
                        history=hist, hist_index=hidx)


__all__ = ["CBVARNowcast", "fit", "fit_reuse"]


# ─── Il controllo di gate, eseguibile ─────────────────────────────────────────

def _gate_check(as_of: str = "2019-05-15", n_draws: int = 40, seed: int = 1,
                horizon: int = 0) -> bool:
    """
    IL CONTROLLO CHE CHIUDE IL GATE 3, e la domanda a cui risponde e' precisa:
    dove il dato c'e', il C-BVAR deve RIPRODURLO, non approssimarlo.

    Li' lo smoother non ha niente da stimare — il livello e' noto — quindi uno
    scarto rivelerebbe un errore in un punto qualsiasi della catena: media
    mobile, ordine del logaritmo, radice cubica, proiezione PSD, smoothing, o la
    trasformazione finale in crescita annualizzata.  E' un test end-to-end che
    non puo' passare per caso.

    DOVE STA LA RIPRODUZIONE, dopo il cambio di ancoraggio.  Con la finestra
    ancorata a `endEstimT` (l'ultimo quarter-end osservato) i trimestri gia'
    pubblicati stanno quasi tutti nella STORIA, che e' dato copiato: confrontarli
    col BEA verifica la catena delle trasformazioni ma non tocca il filtro.  Il
    controllo si sdoppia quindi in due, e sono due cose diverse:

      (a) le CELLE OSSERVATE DENTRO LA FINESTRA — il primo quarter-end e tutti i
          mensili gia' usciti.  Li' lo smoother e' inchiodato dal dato (R = il
          nugget) e deve restituirlo: e' il controllo che passa DAVVERO per lo
          stato-spazio;
      (b) i trimestri, contro il BEA: quelli di storia devono tornare esatti
          (catena delle trasformazioni), quelli in volo devono essere contenuti
          nella banda.

    Sul trimestre OBIETTIVO non si controlla il livello ma la FORMA: la banda
    dev'essere larga (a meta' trimestre l'informazione e' poca) e deve contenere
    il realizzato.

    Non sta in `tests/` perche' richiede una stima vera (~2-3 minuti) e
    rallenterebbe la suite.  Si esegue:

        python -m core.bvar.cbvar
    """
    rng = np.random.default_rng(seed)
    spec = BVARSpec.from_config("C")
    raw_lv = load_raw_levels()
    r = fit(spec, as_of=as_of, n_draws=n_draws, burn=200, rng=rng,
            horizon=horizon, raw=raw_lv)
    g = r.growth("GDPC1")

    raw = raw_lv["GDPC1"].dropna()
    vero = 100.0 * ((raw / raw.shift(1)) ** 4 - 1.0)

    print(r.summary("GDPC1"))
    ok = True

    # (a) le celle OSSERVATE dentro la finestra: il filtro deve restituirle
    Y_win, _, idx, _, _ = _edge(spec, as_of, horizon=horizon, months=None,
                                raw=raw_lv)
    vista = np.isfinite(Y_win)
    med = np.median(r.draws, axis=0)
    err = np.abs(med[vista] - Y_win[vista])
    peggio = float(err.max()) if err.size else 0.0
    buono = err.size > 0 and peggio < 1e-4        # unita' di log-livello
    ok &= buono
    print(f"\n  celle osservate nella finestra ({idx[0].date()} - "
          f"{idx[-1].date()}): {int(vista.sum())} su {vista.size}, "
          f"scarto max {peggio:.2e}  {'OK' if buono else 'FAIL'}")

    print("\n  confronto coi livelli grezzi:")
    q = g.quantile([0.05, 0.5, 0.95], axis=1).T
    inizio = pd.Timestamp(r.index[0])
    for d in g.index:
        if d not in vero.index:
            continue
        m, atteso = q.loc[d, 0.5], float(vero.loc[d])
        dove = "storia " if d < inizio else "finestra"
        if d < pd.Timestamp(as_of):
            b = abs(m - atteso) < 0.10
            ok &= b
            print(f"    {d.date()}  osservato {dove}  {m:6.2f} vs {atteso:6.2f}   "
                  f"scarto {abs(m - atteso):.3f}  {'OK' if b else 'FAIL'}")
        else:
            dentro = q.loc[d, 0.05] <= atteso <= q.loc[d, 0.95]
            # oltre il trimestre obiettivo e' PREVISIONE a piu' trimestri: si
            # riporta, non si giudica (una banda che non contiene il realizzato
            # a due anni non e' un difetto di implementazione).
            in_volo = d <= _target_quarter_end(r.full_index, as_of)
            ok &= dentro or not in_volo
            etichetta = "NOWCAST " if in_volo else "PREVISTO"
            print(f"    {d.date()}  {etichetta}   {m:6.2f} vs {atteso:6.2f}   "
                  f"banda [{q.loc[d, 0.05]:.2f}, {q.loc[d, 0.95]:.2f}]  "
                  f"{'contiene' if dentro else 'NON contiene'}"
                  f"{'' if in_volo else '  (non giudicato)'}")
    print("\n" + ("GATE 3 OK" if ok else "QUALCOSA NON TORNA"))
    return ok


if __name__ == "__main__":
    import sys
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(0 if _gate_check(horizon=h) else 1)
