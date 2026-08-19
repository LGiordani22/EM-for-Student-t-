"""
core/bvar/state_space.py

GATE 3, BLOCCO 2 — LO STATO-SPAZIO MENSILE DEL C-BVAR e il suo filtro.

Prende il gemello mensile prodotto da `cube_root.py` e lo mette in forma
stato-spazio, poi ci fa girare sopra un ciclo di Kalman costruito sulle
PRIMITIVE IMPORTATE da `core/kalman.py`.  Il filtro non viene riscritto: si
importano `kalman_predict`, `kalman_update` e `build_selection_matrix`, che sono
pura algebra matriciale, e si scrive qui il solo ciclo.  `core/kalman.py` NON si
tocca — e' condiviso da `em/`, `mcmc/` e `forecast/` (vincolo del documento di
contesto).


================================================================================
COSA DICE IL PAPER
================================================================================
Cimadomo §2.4, ultima riga: "Finally, as for the B-BVAR, we compute the
distributions of forecasts conditional on the real-time data flow, exploiting
the Kalman filtering methods."  E §2.3, a cui rimanda: "Given the model
parameters, the nowcasts can be viewed as forecasts conditional on different
information sets.  We compute these using the Kalman filtering techniques
described in Banbura et al. (2015)."

La forma dello stato-spazio e' l'Appendice A.  Lo stato e' (§2.4)

    X_tm = (x'_tm, x'_tm-3, ..., x'_tm-3p+3)'

cioe' p blocchi distanziati di TRE MESI, non di uno: e' cio' che lo fa
coincidere con lo stato trimestrale X_tq quando tm e' l'ultimo mese del
trimestre.  La transizione e' (A.3), X_tm = Phi_m X_tm-1 + nu_m,tm, e i mensili
osservabili sono le prime n componenti dello stato.

Per il dato mancante seguiamo GMR §2.2, eq. (12), che a sua volta segue
Giannone, Reichlin & Small (2008): "V_tm = (v1,tm, ..., vk,tm) is such that
var(vi,tm) = 0 if yi,tm is available and var(vi,tm) = infinity otherwise".  Noi
otteniamo lo stesso effetto in modo esatto invece che con un infinito numerico,
con la matrice di selezione W_t di `kalman.py` — la stessa tecnica gia' usata
dalla pipeline DFM.


================================================================================
COSA SIGNIFICA
================================================================================
Il C-BVAR non stima niente qui: i parametri arrivano gia' fatti dal Q-BVAR
(stadio 1) attraverso la mappa cube-root (stadio 2).  Questo modulo e' lo
stadio 3, e serve a UNA cosa sola: leggere il flusso dati mensile man mano che
esce e proiettare in avanti.

E' qui che il C-BVAR guadagna sul Q-BVAR.  Il Gate 2 ha misurato la cecita' del
Q-BVAR: al 10 aprile 2019 condizionava su dati fino al 2018Q4, due trimestri
indietro.  Il C-BVAR vede invece i mensili di gennaio, febbraio e marzo alle
loro date, perche' il suo modello E' mensile — e la quantita' di fine trimestre
la produce il filtro prevedendo i mesi che mancano.  Nessuna media parziale
viene mai formata (vedi il punto 3 nell'header di `qbvar.py`).


================================================================================
I TRE PUNTI DELICATI
================================================================================

--- 1. LA FINESTRA: si filtra il BORDO, non tutto il campione ---------------

E' la scelta piu' importante del modulo e la meno ovvia.  Sta in `edge_window`,
con la tabella che la giustifica.  In una riga: Phi_m e' cosi' non normale che
una ricorsione lunga accumula errore fino a rendere il filtro inservibile, e la
storia non serve filtrarla perche' e' osservata.

--- 2. L'INIZIALIZZAZIONE: dai dati, e CONTA ---------------------------------

Lo stato iniziale si costruisce dalle p osservazioni TRIMESTRALI che precedono
la finestra (`edge_window`), come `initX` in `run_Ksmoother.m` degli autori.
Fin qui non e' cambiato nulla.  P_0 invece si', e questa nota diceva il falso.

DICEVA: "la varianza non condizionata NON ESISTE, `solve_discrete_lyapunov` sul
nostro Phi_m da' autovalore minimo -1.14e+06 (il sistema e' esplosivo), quindi
il `lyapunov_symm` degli autori fallirebbe allo stesso modo".  La prima meta' e'
vera, la conclusione no: `lyapunov_symm.m` NON e' `solve_discrete_lyapunov`.
E' la routine di Dynare, e risolve **solo il sottosistema stabile** — Schur
ordinato, via le direzioni con |lambda| > 1, Lyapunov sul resto, riproiezione.
Su un sistema esplosivo e' esattamente il caso per cui e' scritta.  Sul nostro
il blocco stabile e' 135-142 direzioni su 150.  Vedi `lyapunov_symm` qui sotto.

DICEVA ANCHE: "l'init non cambia nulla, P_0 da 1e-2*I a 1e6*I danno lo stesso
errore".  Era vero SU TUTTO IL CAMPIONE, dove sono le centinaia di osservazioni
a fissare lo stato, ed e' li' che era stato misurato.  Sulla finestra ancorata a
`endEstimT` le righe informative sono poche — una riga piena piu' il bordo — e
P_0 torna a essere un ingrediente del nowcast.  Misurato su 2019Q2 (BEA 3.38),
una sola stima e un solo seme: fra le righe cambia SOLO P_0.

                         horizon = 0                 horizon = 24
    P_0 = 1e0 * I    7.68 [-73.68, +314.44]   34.87 [-75.87, +207.21]   <- ripiego
    P_0 = 1e-2 * I   2.50 [-15.93,  +17.32]    4.36 [-16.30,  +15.50]
    P_0 = 1e-4 * I   1.16 [ -3.05,   +4.71]    0.77 [ -3.74,   +5.58]
    lyapunov_symm    1.58 [ -0.81,   +5.38]    1.82 [ -1.23,   +4.50]   <- autori

A horizon = 24 il ripiego non allargava solo la banda: spostava la MEDIANA a
34.87.  A quell'orizzonte il nowcast non era impreciso, era finto.

Il ripiego kappa*I non era sbagliato per caso: era invisibile finche' la
finestra sbagliata era lunga.

`diffuse_init` resta come fallback documentato quando il blocco stabile e' vuoto
o non ci sono p quarter-end prima della finestra.

--- 3. Sigma_m e la proiezione di Higham -------------------------------------

CAMBIATA DI SENSO rispetto a una versione precedente di questa nota, e vale la
pena dire perche'.

Con la formula di accoppiamento "exact_a15" Sigma_m non era MAI semidefinita
positiva e Higham ne sostituiva il 75.8% in norma di Frobenius: non era
un'approssimazione dichiarabile.  Con la formula "authors" — quella del codice
di replica, ora il default (vedi `cube_root.coupling_matrix`) — la mediana dello
spostamento e' 0.0%, gli autovalori negativi sono ~2 su 30 e stanno a
lambda_min/lambda_max ~ -1e-06.

Quindi Higham QUI E' una correzione di arrotondamento, ed e' esattamente cio'
che fanno gli autori in `build_monthly_ss.m`:

    catch
        [V,D] = eig(qq);  D = diag(D);  D(D<0) = 1e-10;
        qq = V*(V'.*D);   qqFlag = 1;

con tanto di flag che registra l'evento e un riferimento commentato alla
"Cheng and N. J. Higham approximation".

MA LA CODA NON E' TRASCURABILE e va riportata: a seconda del run il p90 dello
spostamento sta fra 2% e 14%, con massimi vicini al 90%.  Per questo
`nearest_psd` restituisce SEMPRE `rel_frobenius`, `n_neg` e i rapporti prima e
dopo, e `psd_summary` li aggrega: serve a poter scrivere in tesi *quanto* la
proiezione sposta, in mediana E nella coda, invece di un generico "abbiamo
proiettato".


================================================================================
CHE COSA SUCCEDE SUL SISTEMA VERO — e le tre cause ESCLUSE
================================================================================
Il C-BVAR FUNZIONA, ma solo sulla finestra terminale.  Ci sono volute tre misure
per capire perche', e le tre cause che sembravano ovvie sono tutte FALSE.

LA CAUSA VERA: LA LUNGHEZZA DELLA RICORSIONE.  Phi_m ha raggio spettrale 1.005
ma entrate fino a 1.8e+04 e cond(V) ~ 1e+06 — e' fortemente NON NORMALE, quindi
ogni passo amplifica anche se gli autovalori sono miti.  Su 16 passi non si
vede, su 400 domina.  Il degrado e' MONOTONO nella finestra (tabella in
`edge_window`), che e' la firma dell'accumulo e non di un guasto.

Il rimedio e' quello degli autori: filtrare solo il bordo e prendere la storia
dai dati osservati.  Vedi `edge_window`.

LE TRE CAUSE ESCLUSE, tutte misurate e tutte a vuoto:

  (a) Sigma_m NON PSD.  Era vero, ma era un ARTEFATTO DELLA FORMULA DI
      ACCOPPIAMENTO, non del metodo.  Con la variante "authors" (default) lo
      spostamento di Higham ha mediana 0.0% e ~2 autovalori negativi su 30, con
      lambda_min/lambda_max ~ -1e-6: e' una vera correzione di arrotondamento.
      Con "exact_a15" era 75.8% mediano.  Vedi `cube_root.coupling_matrix`.
      >>> La proiezione di Higham RESTA, ma cambia di senso: da "sostituzione
      >>> sostanziale da dichiarare" a "correzione numerica".  La coda resta
      >>> non trascurabile (p90 fra 2% e 14% a seconda del run) e va riportata.

  (b) L'INIZIALIZZAZIONE.  Provata su OTTO ordini di grandezza, P_0 da 1e-2*I a
      1e6*I: l'errore sui mesi osservati resta 1.59e-02, identico a tre cifre.
      Non e' un transitorio da diffusione.
      E la varianza non condizionata NON ESISTE: `solve_discrete_lyapunov` sul
      nostro Phi_m restituisce una matrice con autovalore minimo -1.14e+06,
      perche' il sistema e' esplosivo.  Il `lyapunov_symm` che usano gli autori
      fallirebbe allo stesso modo qui.  Su una finestra di 16 mesi la questione
      e' comunque marginale: lo stato iniziale viene dai DATI.

  (c) LA DIMENSIONE n.  Una versione precedente di questa nota sosteneva una
      soglia ("PD a n<=5, indefinita da n>=10").  Era misurata su VAR sintetici
      troppo benigni.  Su profili di serie vere il comportamento non cambia
      qualitativamente fra n=6 e n=30.

NUMERI SULLA FINESTRA DEGLI AUTORI (16 mesi, n=30, formula "authors"):

    errore sui mesi osservati        2.67e-04   (su log-livelli ~10)
    GDPC1 nei mesi latenti           [10.06, 10.08]
    |f|max                           824

Contro il campione pieno: errore 1.59e-02 e |f|max 4.6e+04.

ATTENZIONE A COME SI LEGGONO I TEST.  `test_gate3` §8 fa girare il filtro su un
caso SINTETICO a n=4, dove Sigma_m e' definita positiva e la proiezione non si
attiva.  Verifica che il ciclo sia scritto bene — NON che il C-BVAR funzioni
sul sistema vero.  Le due cose sono diverse e il test lo dice.


================================================================================
COME SI TRADUCE IN CODICE
================================================================================
    nearest_psd(S)                 Higham 1988 + rapporto sulla correzione
    build_state_space(mmap, ...)   MonthlyMap -> matrici dello stato-spazio
    diffuse_init(ns, kappa)        f_0, P_0
    filter_monthly(ss, Y)          il ciclo, sulle primitive importate

Cosa si importa da `core/kalman.py`, e nient'altro:
    kalman_predict, kalman_update, build_selection_matrix

L'UNICA COSA CHE LE PRIMITIVE NON COPRONO e' la costante: `kalman_predict`
calcola f_pred = A f_prev, senza intercetta, perche' il DFM non ne ha una nello
stato.  Il C-BVAR ce l'ha (c_m, vedi `cube_root.monthly_constant`).  Si somma
dopo la chiamata — e' esatto, perche' una costante sposta la media e non tocca
la covarianza:

    f_pred = kalman_predict(...)[0] + c_m

Alternativa scartata: aumentare lo stato con una componente costante pari a 1.
Sarebbe piu' "pulito" formalmente ma allargherebbe uno stato gia' di 150
dimensioni e renderebbe singolare P per costruzione, complicando proprio
l'inizializzazione diffusa del punto 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.bvar.cube_root import MonthlyMap
from core.kalman import build_selection_matrix, kalman_predict, kalman_update

#: Varianza dell'inizializzazione diffusa.  Grande rispetto alla scala dei dati
#: (log-livelli, ordine 1-10), non infinita: un infinito in P_0 propagherebbe
#: NaN al primo update.
DEFAULT_KAPPA = 1e6


# ─── 1. Higham (1988): la matrice PSD piu' vicina ─────────────────────────────

@dataclass
class PSDReport:
    """
    Quanto la proiezione ha spostato la matrice.  E' il numero che va in tesi.

    Attributes
    ----------
    n_neg : int            autovalori negativi trovati
    lam_min, lam_max : float   estremi dello spettro PRIMA
    ratio_before : float   lam_min / lam_max prima (negativo se indefinita)
    rel_frobenius : float  ||S - S_+||_F / ||S||_F, la distanza relativa della
                           proiezione.  E' esattamente
                           sqrt(sum_{lam<0} lam^2) / ||S||_F.
    projected : bool       se e' stato fatto qualcosa
    """

    n_neg: int
    lam_min: float
    lam_max: float
    ratio_before: float
    rel_frobenius: float
    projected: bool

    def __str__(self) -> str:
        if not self.projected:
            return "PSD, nessuna proiezione"
        return (f"{self.n_neg} autoval. negativi, lam_min/lam_max = "
                f"{self.ratio_before:+.3f}, spostamento relativo di Frobenius "
                f"{self.rel_frobenius:.1%}")


# ─── L'update di Kalman veloce ────────────────────────────────────────────────
#
# PERCHE' ESISTE UN UPDATE LOCALE, E PERCHE' NON E' UNA VIOLAZIONE
# =================================================================
# `core/kalman.py` NON si tocca: e' condiviso con `em/`, `mcmc/`, `forecast/`, e
# regge i risultati della tesi DFM.  Ma il README prevede da sempre che
# «`bvar/state_space.py` scrivera' il proprio ciclo `predict`/`update` sulle
# primitive importate»: questo E' quel ciclo.  `kalman_update` resta l'ORACOLO
# contro cui si verifica (vedi `tests/test_gate5.py`).
#
#
# IL PROBLEMA: 982 VOLTE IL PAVIMENTO DI BANDA
# ---------------------------------------------
# Profilato lo smoother dell'L-BVAR (stato companion n*p = 629, n = 37):
# `kalman_update` era 43.4 s su 48.8 (89%), e `kalman_predict` solo il 7.8%.
# Dentro l'update, il colpevole e' UNA riga:
#
#     WL @ P_pred      (37,629) @ (629,629)     91.35 ms
#     (P_pred @ WL.T)  (629,629) @ (629,37)      0.28 ms      <- identica
#
# stessi 1.46e7 flop, stessa matrice letta, 326 volte di differenza.  E NON e'
# banda: misurato, leggere P una volta costa 0.093 ms, quindi la forma buona sta
# a 3x il pavimento (quasi ottimale) e quella cattiva a 982x.  E' un percorso
# patologico di scipy-openblas per la forma "corta e larga per quadrata".
#
# Distinzione che conta, perche' e' l'errore gia' fatto una volta: al Gate 5
# `companion_predict` diede guadagno NULLO proprio perche' li' il regime era di
# banda, non di flop (vedi la sua docstring).  Qui e' l'opposto, ed e' misurato,
# non supposto.
#
#
# LA FORMA GIUSTA E' QUELLA DEGLI AUTORI
# ---------------------------------------
# Non e' un accorgimento nostro: `runKF_DK.m` righe 74-81 fa gia' esattamente
# questo, e non forma MAI `Z*P`.
#
#     PZ  = P*Z_tt;          % P @ Z'    <- la forma veloce, UNA volta
#     F   = (Z_t*PZ + R_t);
#     ZF_t = Z_tt*Finv;   PZF = P*ZF_t;  % guadagno
#     Pu  = P - PZF*PZ';     % niente Z*P da nessuna parte
#
# L'unica differenza che teniamo e' `inv2(F)` -> `np.linalg.solve`: risolvere e'
# numericamente preferibile a invertire, e il resto dell'algebra e' identico.
#
# Misurato end-to-end sullo smoother vero: 34.7 s -> 5.7 s, **6.1x**, con
# scarto massimo 1.6e-11 in relativo contro `kalman_update`.

def kalman_update_fast(f_pred: np.ndarray, P_pred: np.ndarray, y_t: np.ndarray,
                       W_t: np.ndarray, Lambda_tilde: np.ndarray,
                       R_tilde: np.ndarray) -> dict:
    """
    `core.kalman.kalman_update` riscritto nell'ordine di `runKF_DK.m`.

    Stessa firma, stesse chiavi in uscita, stesso risultato entro l'errore di
    macchina.  L'unica differenza e' l'ORDINE delle moltiplicazioni: si calcola
    `PW = P_pred @ WL'` una volta e non si forma mai `WL @ P_pred`.

    Identita' usate (P_pred simmetrica):

        S       = WL P WL' + WR        = WL @ PW + WR
        K       = P WL' inv(S)         = PW @ inv(S)
        K WL P  = P WL' inv(S) WL P    = PW @ K'      (simmetrica per costruzione)
    """
    m_t = W_t.shape[0]
    if m_t == 0:                                   # nessuna osservazione a t
        ns = f_pred.shape[0]
        return {"f_filt": f_pred.copy(), "P_filt": P_pred.copy(),
                "eta": np.empty(0), "S": np.empty((0, 0)), "loglik_t": 0.0,
                "K": np.zeros((ns, 0)),
                "WL": np.zeros((0, Lambda_tilde.shape[1]))}

    y_obs = W_t @ np.where(np.isnan(y_t), 0.0, y_t)
    WL = W_t @ Lambda_tilde                        # (m, ns)
    WR = W_t @ R_tilde @ W_t.T                     # (m, m)
    eta = y_obs - WL @ f_pred

    PW = P_pred @ WL.T                             # (ns, m)  <- la forma veloce
    S = WL @ PW + WR                               # (m, m)
    K_T = np.linalg.solve(S, PW.T)                 # (m, ns) = inv(S) WL P
    K = K_T.T

    f_filt = f_pred + K @ eta
    P_filt = P_pred - PW @ K_T
    P_filt = 0.5 * (P_filt + P_filt.T)

    _, logdet_S = np.linalg.slogdet(S)
    loglik_t = -0.5 * (m_t * np.log(2.0 * np.pi) + logdet_S
                       + float(eta @ np.linalg.solve(S, eta)))
    return {"f_filt": f_filt, "P_filt": P_filt, "eta": eta, "S": S,
            "loglik_t": loglik_t, "K": K, "WL": WL}


def nearest_psd(S: np.ndarray, *, floor: float = 0.0) -> tuple[np.ndarray, PSDReport]:
    r"""
    La matrice semidefinita positiva piu' vicina a `S` in norma di Frobenius —
    Higham (1988), "Computing a nearest symmetric positive semidefinite matrix".

    PER UNA MATRICE GIA' SIMMETRICA la soluzione e' in forma chiusa e NON
    richiede iterazioni: si decompone S = V diag(lam) V' e si azzerano gli
    autovalori negativi,

        S_+ = V diag(max(lam, 0)) V'

    ed e' esattamente la proiezione sul cono PSD.  (L'algoritmo iterativo di
    Higham 2002, con le proiezioni alternate, serve alla matrice di
    CORRELAZIONE piu' vicina — li' il vincolo di diagonale unitaria rompe la
    forma chiusa.  Non e' il nostro caso: vogliamo una covarianza, senza vincoli
    sulla diagonale.)

    La distanza vale ||S - S_+||_F = sqrt(sum_{lam < 0} lam^2), il che rende il
    rapporto relativo calcolabile esattamente e non stimato.

    Parameters
    ----------
    floor : float
        Valore a cui portare gli autovalori negativi.  0.0 (default) da' la
        proiezione esatta, ed e' cio' che vogliamo: la Q del Kalman puo' essere
        singolare senza problemi, `kalman_predict` la somma e basta.  Un floor
        > 0 servirebbe solo se a valle qualcuno pretendesse una Cholesky di Q.

    Returns
    -------
    (S_psd, report)
    """
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)                      # simmetria esatta prima di eigh
    lam, V = np.linalg.eigh(S)

    neg = lam < floor
    n_neg = int((lam < 0.0).sum())
    lam_min, lam_max = float(lam.min()), float(lam.max())
    denom = max(abs(lam_max), 1e-300)
    norm_S = float(np.linalg.norm(S, "fro"))

    if not neg.any():
        return S, PSDReport(n_neg=0, lam_min=lam_min, lam_max=lam_max,
                            ratio_before=lam_min / denom, rel_frobenius=0.0,
                            projected=False)

    shift = np.sqrt(float(np.sum(lam[lam < 0.0] ** 2)))
    lam_c = np.where(neg, floor, lam)
    S_psd = (V * lam_c) @ V.T
    S_psd = 0.5 * (S_psd + S_psd.T)

    return S_psd, PSDReport(
        n_neg=n_neg, lam_min=lam_min, lam_max=lam_max,
        ratio_before=lam_min / denom,
        rel_frobenius=shift / max(norm_S, 1e-300),
        projected=True,
    )


# ─── 2. Lo stato-spazio ───────────────────────────────────────────────────────

@dataclass
class MonthlyStateSpace:
    """
    Il modello mensile del C-BVAR in forma stato-spazio.

        X_tm = c_m + Phi_m X_tm-1 + nu_m,tm      nu_m ~ N(0, Q)
        y_tm = Lambda X_tm                        (osservazione ESATTA)

    dove Q = blkdiag(Sigma_m, 0) e Lambda = [I_n, 0 ... 0].

    OSSERVAZIONE ESATTA, R = 0: i mensili non sono misurati con errore, SONO le
    prime n componenti dello stato.  Il dato mancante non si modella con una
    varianza infinita (GMR eq. 12) ma si toglie con W_t, che e' esatto.
    """

    Phi_m: np.ndarray            # (ns, ns)
    Q: np.ndarray                # (ns, ns)  blkdiag(Sigma_m proiettata, 0)
    Lambda: np.ndarray           # (n, ns)
    R: np.ndarray                # (n, n)    zeri
    c_m: np.ndarray              # (ns,)
    n: int
    p: int
    psd: PSDReport = field(repr=False, default=None)

    @property
    def ns(self) -> int:
        return self.n * self.p


def build_state_space(mmap: MonthlyMap, *, project: bool = True,
                      nugget: float = 1e-8) -> MonthlyStateSpace:
    """
    Da `MonthlyMap` (uscita di `cube_root.quarterly_to_monthly`) alle matrici
    dello stato-spazio, applicando la proiezione di Higham a Sigma_m.

    Parameters
    ----------
    project : bool
        True (default) proietta.  False lascia Sigma_m com'e': serve SOLO a
        misurare il fenomeno nei test, perche' con una Q indefinita il filtro
        puo' produrre covarianze non PSD e verosimiglianze prive di senso.
    nugget : float
        Varianza di osservazione RELATIVA, sulla diagonale di R.  Vedi sotto.

    IL NUGGET SU R, E PERCHE' NON E' ZERO
    -------------------------------------
    Il modello dice R = 0: i mensili non sono misurati con errore, SONO le
    prime n componenti dello stato (GMR eq. 12, "var(v_i,tm) = 0 if y_i,tm is
    available").  Numericamente pero' R = 0 esatto NON E' REALIZZABILE qui, e il
    motivo e' una conseguenza diretta della proiezione di Higham.

    Con R = 0 e osservazione esatta, dopo un aggiornamento in cui tutte le n
    serie sono osservate il blocco (1,1) di P_filt diventa esattamente zero: lo
    stato e' noto senza incertezza.  Al passo dopo

        S = Lambda P_pred Lambda' = [Phi_m P_filt Phi_m']_11 + Sigma_m

    e Sigma_m dopo la proiezione E' SINGOLARE PER COSTRUZIONE — Higham azzera gli
    autovalori negativi, quindi Q ha un nucleo di dimensione pari al numero di
    autovalori corretti (mediana 10 su 30 a n=30).  Se il termine propagato non
    riempie quel nucleo, S e' singolare e `np.linalg.solve` fallisce.  Succede
    davvero: misurato a n=6, LinAlgError al primo aggiornamento pieno.

    Il nugget e' quindi una REGOLARIZZAZIONE NUMERICA, non una scelta di
    modellazione: R = nugget * media(diag(Sigma_m)) * I.  Con il default 1e-8 la
    deviazione standard dell'errore di misura e' 1e-4 volte quella
    dell'innovazione — cinque ordini di grandezza sotto la scala dei dati (log-
    livelli di ordine 1-100).  Non puo' spostare una conclusione economica; serve
    solo a rendere S invertibile.

    Va comunque DICHIARATO, perche' e' uno scostamento dalla specifica del
    paper, e perche' la sua NECESSITA' e' essa stessa un sintomo: senza la
    proiezione di Higham, Sigma_m non sarebbe singolare e il nugget non
    servirebbe.
    """
    n, p = mmap.n, mmap.p
    ns = n * p

    Sigma_m, rep = (nearest_psd(mmap.Sigma_m) if project
                    else (mmap.Sigma_m, PSDReport(0, 0.0, 0.0, 0.0, 0.0, False)))

    Q = np.zeros((ns, ns))
    Q[:n, :n] = Sigma_m

    Lambda = np.zeros((n, ns))
    Lambda[:, :n] = np.eye(n)

    scale = float(np.mean(np.abs(np.diag(Sigma_m))))
    R = nugget * max(scale, 1e-300) * np.eye(n)

    c_m = np.zeros(ns) if mmap.const_m is None else np.asarray(mmap.const_m, float)

    return MonthlyStateSpace(Phi_m=mmap.Phi_m, Q=Q, Lambda=Lambda,
                             R=R, c_m=c_m, n=n, p=p, psd=rep)


# ─── 3. L'inizializzazione ────────────────────────────────────────────────────

def lyapunov_symm(A: np.ndarray, Q: np.ndarray, *, qz_criterium: float = 1.0
                  ) -> tuple[np.ndarray, int]:
    r"""
    P_0 come lo costruiscono gli autori: `lyapunov_symm(aa, qqKF, 1, 1e-6)` in
    `run_Ksmoother.m`.  E NON e' la varianza non condizionata.

    COSA SI ERA CAPITO MALE.  `lyapunov_symm.m` e' la routine di Dynare, e la
    sua intestazione lo dice in una riga: *"If a has some unit roots, the
    function computes only the solution of the stable subsystem."*  Il corpo:

        [U,T] = schur(a);
        e1 = abs(ordeig(T)) > 2-qz_criterium;   % con qz_criterium = 1: |lambda| > 1
        k = sum(e1);                            % le direzioni da BUTTARE
        [U,T] = ordschur(U,T,e1);  T = T(k+1:end,k+1:end);
        B = U(:,k+1:end)'*b*U(:,k+1:end);
        ... risolve x - T x T' = B sul solo blocco stabile ...
        x = U(:,k+1:end)*x*U(:,k+1:end)';

    cioe': Schur ordinato, si scartano le direzioni con |lambda| > 1, si risolve
    Lyapunov solo dove converge, e si riproietta.  Il risultato e' PSD e ha
    varianza ESATTAMENTE ZERO lungo le direzioni esplosive.

    Quindi su un sistema esplosivo la routine NON fallisce: e' scritta apposta.
    Qui si era usato `scipy.linalg.solve_discrete_lyapunov`, che invece risolve
    l'equazione piena e su Phi_m (raggio 1.0038) restituisce una matrice con
    autovalori a -1e+06 — da cui il ripiego su kappa*I, che degli autori non e'.
    Sul nostro sistema il blocco stabile e' 135-142 direzioni su 150.

    Non era una differenza visibile finche' la finestra era lunga: erano le
    osservazioni a determinare lo stato.  Con la finestra ancorata a `endEstimT`
    (vedi `edge_window`) le righe informative sono poche e P_0 torna a pesare —
    ed e' li' che kappa*I apriva la banda del nowcast a [-74, +314].

    Returns
    -------
    (P0, n_stabili)  `P0` (ns, ns) PSD, `n_stabili` la dimensione del blocco
                     stabile: 0 significa che non c'era nulla da risolvere.
    """
    from scipy.linalg import schur, solve_discrete_lyapunov

    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    soglia = 2.0 - float(qz_criterium)
    T, Z, sdim = schur(A, output="real",
                       sort=lambda wr, wi: (wr * wr + wi * wi) <= soglia ** 2)
    if sdim == 0:
        return np.zeros_like(A), 0
    Z1 = Z[:, :sdim]
    B = Z1.T @ Q @ Z1
    x = solve_discrete_lyapunov(T[:sdim, :sdim], 0.5 * (B + B.T))
    P = Z1 @ (0.5 * (x + x.T)) @ Z1.T
    return 0.5 * (P + P.T), int(sdim)


#: La scelta di P_0 per i modelli che filtrano una finestra di bordo.
#: "lyapunov" = `lyapunov_symm` (run_Ksmoother*.m, bbvar.m); un float = kappa*I,
#: dove 0.0 significa "lo stato iniziale e' dato osservato, quindi noto"
#: (`lbvar.m` r.52).  Vedi `initial_covariance`.
P0_DEFAULT: str | float = "lyapunov"


def initial_covariance(A: np.ndarray, Q: np.ndarray, *,
                       kind: str | float = P0_DEFAULT,
                       kappa_fallback: float = 1.0) -> tuple[np.ndarray, str]:
    """
    P_0 per una finestra di bordo, in UN POSTO SOLO per tutti e quattro i modelli.

    PERCHE' CENTRALIZZATA.  Gli autori stessi non sono coerenti: `lbvar.m` r.52
    azzera P_0, `bbvar.m` r.145 e `run_Ksmoother*.m` usano `lyapunov_symm`.  Da
    noi la scelta e' arrivata modello per modello — il B a zero (con la misura
    del punto 6 di `bbvar.py`: con `1e4*I` e finestra corta il nowcast usciva
    -97.74%), l'L a zero (`lbvar.py` punto 3), C e Q al Lyapunov — e tre
    convenzioni diverse per la stessa domanda non sono difendibili in tesi.

    L'ARGOMENTO A PRIORI, e perche' NON decide.  Verrebbe da dire: `initX` sono
    righe OSSERVATE, quindi lo stato iniziale e' noto, quindi P_0 = 0 e il
    Lyapunov mette varianza dove non ce n'e'.  E' l'argomento con cui il B-BVAR
    ha scelto lo zero (punto 6 di `bbvar.py`).  Misurato, perde:

        C-BVAR, S=150, due date, cambia solo P_0
                          celle osservate   1o trim. vs BEA   banda obiettivo
            lyapunov          4.8e-05        0.008/0.009 pp    [-2.9,+7.6]
            P_0 = 0           3.5e-04        0.076/0.079 pp    [-6.0,+7.7]

        Q-BVAR, RMSE per fase, pannello bilanciato (8 trimestri, 2008-2010)
                          forecast   M1      M2      M3     backcast
            lyapunov        4.332   4.374   4.343   4.219    3.134
            P_0 = 0         4.307   4.526   4.393   4.253    3.364

    Nel C lo zero fa FALLIRE il controllo del Gate 6 (0.076 contro 0.03 di
    tolleranza) e ALLARGA la banda; nel Q costa 0.23 pp proprio nel backcast,
    cioe' nella fase che il filtro sul bordo esiste per migliorare.

    PERCHE' l'argomento a priori sbaglia.  Perche' "lo stato iniziale e' noto"
    e' vero solo dove i blocchi sono periodi ADIACENTI (B, L, Q).  Nel C no: lo
    stato mensile ha i blocchi a 3 mesi (`[x_t, x_{t-3}, ...]`) ma la finestra
    parte a `endEstimT`, mentre `initX` porta gli ULTIMI QUARTER-END — cioe' lo
    stato di `endEstimT-3` usato come stato di `endEstimT-1`.  Due mesi di
    gioco, che sono nel codice degli autori prima che nel nostro.  Azzerare P_0
    li' non dichiara conoscenza: congela un valore vecchio, che poi litiga con
    la riga osservata all'ancora.  Nel Q il gioco non c'e' e infatti la
    differenza e' molto piu' piccola — ma resta, e va nella stessa direzione.

    E SU B E L?  Confronto fatto (as_of 2018-11-16, S=40, stessa stima, stesso
    seme, trimestre in volo 2018Q4), ed e' l'unico posto dove la risposta e'
    "dipende dalla FINESTRA, non dal modello":

        B    P_0 = 0    3.35 [1.86, 5.36]      |  IDENTICI a quattro decimali
             lyapunov   3.35 [1.86, 5.36]      |  (riproduzione BEA 0.0002 pp entrambi)

        L    P_0 = 0    3.19 [1.35, 4.62]   ampiezza 3.27
             lyapunov   2.77 [1.20, 5.71]   ampiezza 4.51   <- +38%

    Il Lyapunov NON e' degenerato in nessuno dei due (B: 387 direzioni stabili
    su 420, sd media 2.4; L: 615 su 629, sd 2.1), quindi l'uguaglianza del B e'
    un risultato, non un ripiego silenzioso.  La causa e' la lunghezza della
    finestra informativa:

        B    `edge_lags = 24` TRIMESTRI di dati osservati dopo l'ancora
             -> P_0 e' lavata via dalle osservazioni, la scelta e' immateriale
        L    la finestra parte a `lf_full - p - 3` e le prime p righe sono
             condizioni iniziali fisse: al filtro restano ~3 righe informative
             prima del bordo -> P_0 pesa, come nel C e nel Q

    E' la stessa legge trovata nel C: P_0 conta quando la finestra e' corta.  Su
    quale sia MEGLIO per l'L non c'e' evidenza — un trimestre non decide
    un'accuratezza — quindi l'L resta a 0, che e' letteralmente il suo
    `lbvar.m` r.52.

    STATO DELLA SCELTA NEI QUATTRO MODELLI:

        C, Q    "lyapunov"   misurato meglio di 0 (tabelle qui sopra)
        B       0            immateriale (misurato): finestra da 24 trimestri
        L       0            loro `lbvar.m` r.52; la scelta PESA (banda +38%)
                             ma serve una griglia per dire quale sia migliore

    Returns
    -------
    (P0, come)  con `come` in {"lyapunov", "kappa", "fallback"}.
    """
    ns = A.shape[0]
    if kind == "lyapunov":
        try:
            P, n_stab = lyapunov_symm(A, Q)
            if n_stab > 0 and np.isfinite(P).all():
                return P, "lyapunov"
        except Exception:
            pass
        return kappa_fallback * np.eye(ns), "fallback"
    return float(kind) * np.eye(ns), "kappa"


def diffuse_init(ns: int, *, kappa: float = DEFAULT_KAPPA,
                 f0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    f_0 = 0 (o quello che si passa), P_0 = kappa I.

    E' il ripiego, non la strada principale: la strada principale e'
    `lyapunov_symm`, che regge anche i sistemi esplosivi.  Resta perche' il
    filtro va fatto girare anche su sistemi in cui il blocco stabile e' vuoto,
    e per i test in cui l'inizializzazione dev'essere deliberatamente ignorante.
    """
    f = np.zeros(ns) if f0 is None else np.asarray(f0, dtype=float).copy()
    return f, kappa * np.eye(ns)


# ─── 4. Il ciclo ──────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """L'uscita del filtro."""

    f_filt: np.ndarray           # (T, ns)   stati filtrati
    P_filt: np.ndarray           # (T, ns, ns)
    f_pred: np.ndarray           # (T, ns)   previsione a un passo
    loglik: float                # somma dei contributi, diffusi esclusi
    loglik_t: np.ndarray         # (T,)
    n_obs: np.ndarray            # (T,) quante serie osservate a ogni t
    psd: PSDReport = field(repr=False, default=None)
    _n: int = field(repr=False, default=0)

    @property
    def observables(self) -> np.ndarray:
        """
        (T, n) — le prime n componenti dello stato filtrato, cioe' x_tm.

        E' il nowcast: dove il pannello ha un NaN questa e' la stima del filtro,
        dove ha un dato coincide con il dato (l'osservazione e' esatta, R = 0).
        """
        return self.f_filt[:, : self._n]


def edge_window(Y: np.ndarray, quarter_end: np.ndarray, p: int,
                *, months: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    r"""
    La FINESTRA TERMINALE su cui va fatto girare il filtro, e lo stato iniziale
    preso dai dati.  E' la scelta implementativa degli autori, ed e' necessaria.

    L'ANCORA STA IN FONDO AL DATO, NON IN FONDO ALL'INDICE
    -------------------------------------------------------
    `STEP2_CRBVAR.m` (r.144-150) fa questo, e l'ordine delle due righe e' il
    contenuto:

        endEstimT = 3*find(~isnan(sum(X(3:3:end,:),2)),1,'last');
        [X_sm,logLik] = run_Ksmoother(betaHat, sigmaHat, lags,
                                      X(endEstimT-3*lags:end,:), yQ(end-lags:end,:));

    `endEstimT` e' **l'ultimo quarter-end PIENAMENTE OSSERVATO**, e la finestra
    parte da li' e arriva alla fine di `X` — righe di previsione NaN comprese.
    Non e' un dettaglio: `X` a quel punto e' gia' stato esteso di `horizon = 24`
    mesi di NaN (r.112/141, `X = X(1:nowcastM+horizon,:)`), e sono proprio quelle
    righe che il filtro riempie proiettando.  E' cosi' che si ottiene la
    previsione: NON si itera la companion a parte, si mette il NaN nel pannello e
    si lascia lavorare il passo di predizione.

    QUESTA FUNZIONE ANCORAVA ALLA FINE DELL'INDICE (`i0 = T - (3p+1)`), che
    coincide con la loro convenzione **solo se non ci sono righe appese**.  Con
    `horizon = 24` l'ancora finiva 24 mesi dentro il vuoto: finestra tutta NaN,
    `f0` costruito da quarter-end mascherati.  Era il bloccante del C-BVAR.

    Il conto di quante righe si filtrano non e' quindi piu' una costante: e'
    `T - endEstimT`, cioe' il bordo frastagliato (0-2 trimestri) piu' `horizon`.
    Con `horizon = 24` sono 25-31 mesi — lo stesso ordine di grandezza dei 16
    della vecchia convenzione, e dentro la zona sicura della tabella qui sotto.

    PERCHE' NON SI FILTRA TUTTO IL CAMPIONE
    ----------------------------------------
    Sempre `STEP2_CRBVAR.m`, la riga subito dopo:

        X_transf = [X(3*lags+1:endEstimT-1,:); X_sm];

    cioe' per tutta la storia precedente ANTEPONE I DATI OSSERVATI invece di
    smussarli.  Nel `crbvar.m` la versione che filtra l'intero campione c'e' ed
    e' COMMENTATA — ci sono passati e hanno ripiegato.  Chi chiama questa
    funzione deve fare la stessa cosa: la finestra non e' il risultato, e' solo
    la parte del risultato che il filtro produce (vedi `cbvar.CBVARNowcast`).

    Non e' un espediente: per il nowcasting il modello mensile serve a
    interpolare il trimestre in corso e il bordo frastagliato.  La storia e'
    osservata, filtrarla non aggiunge informazione.

    E SUL NOSTRO SISTEMA E' OBBLIGATORIO.  Misurato, stessa estrazione, stesso
    stato iniziale dai dati, al variare della sola lunghezza della finestra:

        finestra    |f|max    errore sui mesi OSSERVATI
          16 mesi      824              2.67e-04
          24 mesi     1.2e3             1.05e-03
          48 mesi    1.59e3             2.06e-03
          96 mesi     2.8e3             4.58e-03
         200 mesi    5.75e3             1.28e-02
         403 mesi    4.6e4              1.59e-02

    Il degrado e' MONOTONO nella lunghezza: e' la firma dell'accumulo attraverso
    una matrice fortemente non normale (Phi_m ha raggio 1.005 ma entrate fino a
    1.8e4 e cond(V) ~ 1e6), non di un guasto.  Su 16 passi non si vede, su 400
    domina.  E' anche il motivo per cui l'inizializzazione non conta: provata su
    otto ordini di grandezza di P_0, l'errore resta 1.59e-02 sul campione pieno.

    Il paper non spiega mai perche' il filtro giri su una finestra corta.  Questa
    tabella e' la risposta, ed e' un risultato nostro.

    Parameters
    ----------
    Y : (T, n)              il pannello mensile completo, righe di previsione
                            (tutte NaN) comprese
    quarter_end : (T,) bool quali righe sono fine trimestre
    p : int                 ritardi trimestrali del VAR
    months : int | None
        **Solo diagnostica.**  Se dato, la finestra torna ad essere ancorata
        alla FINE DELL'INDICE e lunga `months` righe: e' la vecchia convenzione,
        conservata perche' e' quella con cui si e' misurata la tabella qui
        sopra, e riproducibile solo cosi'.  Corretta unicamente su un pannello
        senza righe appese; chi la usa con `horizon > 0` riapre il bloccante.
        `None` (default) = la convenzione degli autori.

    Returns
    -------
    (Y_win, f0, i0)
        `Y_win` la finestra da `endEstimT` in poi, `f0` lo stato iniziale
        costruito dalle p osservazioni trimestrali che la precedono (piu'
        recente per prima, come `initX` in `run_Ksmoother.m`), `i0` = `endEstimT`,
        l'indice di inizio in `Y`.  Le righe `Y[:i0]` NON sono nel risultato: sono
        il dato osservato, e chi chiama le antepone.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    qe = np.flatnonzero(np.asarray(quarter_end, dtype=bool))

    if months is None:
        # `endEstimT`: l'ultimo quarter-end con TUTTE le serie osservate.  E'
        # `find(~isnan(sum(X(3:3:end,:),2)),1,'last')` — la somma per riga
        # propaga il NaN, quindi basta un buco perche' il trimestre non conti.
        pieni = qe[np.isfinite(Y[qe]).all(axis=1)] if len(qe) else qe
        if len(pieni) == 0:
            raise ValueError(
                "nessuna riga completamente osservata fra i quarter-end: il "
                "C-BVAR non ha un punto a cui agganciare la finestra."
            )
        i0 = int(pieni[-1])
        W = T - i0
    else:
        W = min(int(months), T)
        i0 = T - W

    prima = qe[qe < i0][-p:]
    if len(prima) < p:
        raise ValueError(
            f"servono {p} quarter-end prima della finestra, trovati {len(prima)}: "
            f"la finestra ({W} mesi su {T}, ancorata a {i0}) e' troppo lunga, "
            f"oppure il pannello troppo corto."
        )
    f0 = np.concatenate([Y[j] for j in prima[::-1]])
    return Y[i0:], f0, i0


def filter_monthly(ss: MonthlyStateSpace, Y: np.ndarray, *,
                   kappa: float = DEFAULT_KAPPA,
                   n_diffuse: int | None = None,
                   f0: np.ndarray | None = None) -> FilterResult:
    """
    Il ciclo di Kalman sul modello mensile, scritto sulle primitive importate.

    Parameters
    ----------
    Y : (T, n)
        Il pannello MENSILE in unita' del modello.  I NaN sono ammessi e sono
        anzi il punto: le trimestrali sono NaN nei mesi 1 e 2 di ogni trimestre,
        e il bordo frastagliato mette NaN in fondo.  `build_selection_matrix` li
        toglie a ogni t.
    n_diffuse : int | None
        Quante osservazioni iniziali ESCLUDERE dalla verosimiglianza, perche'
        servono a consumare la diffusione di P_0 e i loro contributi dipendono
        da `kappa` invece che dal modello.  None = `p` (un ciclo completo di
        blocchi dello stato).

    Returns
    -------
    FilterResult
    """
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    if n != ss.n:
        raise ValueError(f"Y ha {n} colonne, lo stato-spazio ne attende {ss.n}")
    ns = ss.ns
    n_diffuse = ss.p if n_diffuse is None else int(n_diffuse)

    f, P = diffuse_init(ns, kappa=kappa, f0=f0)

    f_filt = np.empty((T, ns))
    P_filt = np.empty((T, ns, ns))
    f_pred_all = np.empty((T, ns))
    ll_t = np.zeros(T)
    n_obs = np.zeros(T, dtype=int)

    for t in range(T):
        # --- predizione.  La costante non e' coperta dalla primitiva: si somma
        #     dopo, ed e' esatto (sposta la media, non la covarianza).
        f_pred, P_pred = kalman_predict(f, P, ss.Phi_m, ss.Q)
        f_pred = f_pred + ss.c_m
        f_pred_all[t] = f_pred

        # --- aggiornamento con la selezione dei soli osservati
        W_t = build_selection_matrix(Y[t])
        out = kalman_update(f_pred, P_pred, Y[t], W_t, ss.Lambda, ss.R)

        f, P = out["f_filt"], out["P_filt"]
        f_filt[t], P_filt[t] = f, P
        ll_t[t] = out["loglik_t"]
        n_obs[t] = W_t.shape[0]

    loglik = float(ll_t[n_diffuse:].sum())
    return FilterResult(f_filt=f_filt, P_filt=P_filt, f_pred=f_pred_all,
                        loglik=loglik, loglik_t=ll_t, n_obs=n_obs,
                        psd=ss.psd, _n=n)


# ─── 5. L'aggregato sulle estrazioni — il numero per la tesi ──────────────────

def psd_summary(reports: list[PSDReport]) -> dict:
    """
    Riassume la correzione di Higham su un insieme di estrazioni.

    E' la funzione che produce il numero da riportare: non "abbiamo proiettato"
    ma quanto, in mediana e nei casi peggiori.
    """
    if not reports:
        return {}
    rf = np.array([r.rel_frobenius for r in reports])
    rb = np.array([r.ratio_before for r in reports])
    nn = np.array([r.n_neg for r in reports])
    return {
        "n_draws": len(reports),
        "frac_projected": float(np.mean([r.projected for r in reports])),
        "n_neg_median": float(np.median(nn)),
        "n_neg_max": int(nn.max()),
        "ratio_before_median": float(np.median(rb)),
        "ratio_before_worst": float(rb.min()),
        "rel_frobenius_median": float(np.median(rf)),
        "rel_frobenius_p90": float(np.percentile(rf, 90)),
        "rel_frobenius_max": float(rf.max()),
    }


def format_psd_summary(s: dict) -> str:
    """Il riassunto in forma leggibile, per i log e per la tesi."""
    if not s:
        return "(nessuna estrazione)"
    return (
        f"proiezione di Higham su {s['n_draws']} estrazioni\n"
        f"  proiettate                        {s['frac_projected']:.0%}\n"
        f"  autovalori negativi               mediana {s['n_neg_median']:.0f}, "
        f"max {s['n_neg_max']}\n"
        f"  lambda_min/lambda_max prima       mediana {s['ratio_before_median']:+.3f}, "
        f"peggiore {s['ratio_before_worst']:+.3f}\n"
        f"  spostamento rel. di Frobenius     mediana {s['rel_frobenius_median']:.1%}, "
        f"p90 {s['rel_frobenius_p90']:.1%}, max {s['rel_frobenius_max']:.1%}"
    )


__all__ = [
    "PSDReport",
    "MonthlyStateSpace",
    "FilterResult",
    "nearest_psd",
    "kalman_update_fast",
    "build_state_space",
    "diffuse_init",
    "filter_monthly",
    "psd_summary",
    "format_psd_summary",
    "DEFAULT_KAPPA",
]
