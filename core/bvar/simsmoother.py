"""
core/bvar/simsmoother.py

GATE 5, BLOCCO 1 — IL SIMULATION SMOOTHER DI DURBIN-KOOPMAN.

Codice NUOVO scritto in `bvar/`, appoggiato alle PRIMITIVE importate da
`core/kalman.py`.  `core/kalman.py` non si tocca (condiviso da `em/`, `mcmc/`,
`forecast/`), e non se ne riusa lo smoother: vedi "PERCHE' NON `kalman_smoother`"
piu' sotto — non e' una preferenza di stile, e' una questione di 3 GB di RAM.


################################################################################
#                                                                              #
#   PRIMA PARTE — L'L-BVAR, §2.2.  La teoria del Gate 5.                       #
#                                                                              #
#   Questo e' il modello per cui lo smoother esiste.  Il ciclo MCMC vero sta   #
#   in `lbvar.py` (blocco 2); qui la mappa, perche' il blocco 1 si capisce     #
#   solo sapendo dove va a finire.                                            #
#                                                                              #
################################################################################

COSA DICE IL PAPER
==================
Il VAR e' MENSILE e le trimestrali sono variabili mensili con due osservazioni
mancanti su tre:

    "The first approach for dealing with mixed-frequency treats the quarterly
     variables as monthly variables, with missing observations in the first two
     months of the quarter.  The VAR model is thus defined at monthly frequency,
     and Kalman filtering techniques are employed to estimate the latent monthly
     processes."                                                         (§2.2)

    "We assume that the (log-)levels of our n variables (collected in the
     n-dimensional vector x_tm) are described by a monthly vector autoregressive
     process, but otherwise similar the one in Eq. (1), with p = 17 lags."

I 17 ritardi non sono una scelta libera: la nota 10 li DERIVA dall'equivalenza
degli insiemi informativi con B e C, che usano 5 ritardi trimestrali.

E il prior e' lo stesso di sempre — e' cio' che autorizza a non ramificare:

    "We adopt a Normal-Inverse Wishart prior with THE SAME PARAMETRIZATION AS
     THE BASELINE CASE, which combines the Minnesota prior with the
     sum-of-coefficients prior."

IL CICLO, che il paper detta passo per passo:

    "Starting with the parameters set at their prior mean, we iterate the
     following steps: [1] Using the simulation smoother of Durbin and Koopman
     (2001), we draw the complete monthly dataset (i.e., including draws of the
     latent missing values) conditional on the model parameters A_m's and
     Sigma_m; then, [2] using the posterior sampler of Giannone et al. (2015),
     we draw the hyperparameters lambda, mu and psi conditional on the complete
     monthly dataset, and finally, [3] we draw the model parameters conditional
     on the hyperparameters and the complete monthly dataset.  This process
     naturally also yields draws of the nowcast/forecast conditional on the
     dataset used for estimation."

L'INIZIALIZZAZIONE, che il paper specifica per intero e va seguita alla lettera:

    "We interpolate quarterly data using splines to obtain a preliminary
     complete monthly dataset, which we use to specify the initial conditions.
     The latter are assumed to be Normally-distributed with mean equal to the
     first p months in the complete dataset, and with variance equal to zero or
     equal to the prior variance Psi_ii depending on whether the data is
     observed or estimated."

E la nota 11, sulla natura della variabile latente:

    "We treat quarterly data as monthly data available only in the last month of
     the quarter.  Hence, the latent variable we estimate INHERITS THE FEATURES
     OF THE QUARTERLY VARIABLE (e.g. in the case of GDP it is still defined
     approximately as the sum of three consecutive monthly levels)."

— e, importante, hanno provato a imporre esplicitamente l'aggregazione nello
state space e "we do not find improvements given the very general lag structure
of the model".  Quindi NON si impone: la trimestrale e' semplicemente una serie
mensile osservata un mese su tre.


COSA SIGNIFICA
==============
E' l'unica delle quattro varianti in cui il dato mancante sta DENTRO la stima
invece che prima (Q, B: aggregazione a monte) o dopo (C: filtro a valle).  Tre
conseguenze, tutte importanti:

1. E' L'UNICO CHE PUO' USARE TUTTE E 37 LE SERIE.  Le 7 a partenza tardiva
   (PPIFIS dal 2009, PCEC96 dal 2007, GACDISA dal 2001, JTSJOL dal 2000,
   ISM_NMI dal 1997, TTLCONS dal 1993, DGORDER dal 1992) sono buchi IN CIMA al
   campione, e solo uno smoother dentro la stima li puo' riempire.  Q, C e B ne
   usano 30 perche' il posterior coniugato pretende un pannello denso.  Vedi
   l'header di `data.py` per la distinzione fra bordo frastagliato e partenza
   tardiva.

2. IL DATO MANCANTE DIVENTA UN PARAMETRO.  Non c'e' nessuna trasformazione che
   lo elimina (la media mobile di Q e C) e nessuna ricodifica che lo aggira (le
   tre colonne di B): il pannello mensile completo e' una VARIABILE CASUALE che
   si estrae a ogni iterazione.  Da cui il nome del passo 1.

3. IL CORE VIENE CHIAMATO CON UN PANNELLO DIVERSO OGNI VOLTA.  E' esattamente il
   requisito d'interfaccia fissato al Gate 0 e implementato al Gate 1:
   `core.step(state, rng, panel=...)` ricostruisce il target sul nuovo pannello,
   rivaluta la log-posterior corrente e fa UNA spazzata, mantenendo la catena
   degli iperparametri da dove stava.  I passi [2] e [3] del ciclo di §2.2 SONO
   `core.step`.  Il blocco 2 non dovra' inventare niente: dovra' solo alternare
   `step` con questo modulo.


COME SI TRADUCE IN CODICE
=========================
    simsmoother.py   (blocco 1, QUESTO FILE)  il passo [1]: estrarre il pannello
    lbvar.py         (blocco 2)               il ciclo che alterna [1] e [2]+[3]

Lo stato-spazio dell'L-BVAR e' la companion mensile del VAR(17):

    alpha_t = (x_t', x_{t-1}', ..., x_{t-16}')'        ns = n*p = 37*17 = 629
    alpha_t = c + A alpha_{t-1} + eta_t                eta = (eps', 0...0)'
    y_t     = Z alpha_t                                Z = [I_n, 0 ... 0]

L'osservazione e' ESATTA (le variabili SONO le prime n componenti dello stato) e
il dato mancante si toglie con la matrice di selezione W_t, non con una varianza
infinita.

>>> DIMENSIONAMENTO, da tenere presente e non da risolvere adesso: ns = 629 e
>>> T ~ 490 mesi.  E' il sistema piu' pesante dei quattro, ed e' atteso.  Le
>>> scelte di questo modulo sono guidate da quel numero — vedi la sezione sulla
>>> memoria.


################################################################################
#                                                                              #
#   SECONDA PARTE — questo modulo: il simulation smoother.                     #
#                                                                              #
################################################################################

COSA DICE IL PAPER
==================
Una riga sola: "Using the simulation smoother of Durbin and Koopman (2001), we
draw the complete monthly dataset".  Cimadomo cita il LIBRO (DK 2001), che e' la
citazione per la tesi; l'algoritmo che si implementa e' quello dell'articolo
DK (2002), "A simple and efficient simulation smoother for state space time
series analysis", Biometrika 89, 603-616.  Il libro non e' in `docs/` perche' e'
un libro; l'algoritmo e' standard e si costruisce qui.


COSA SIGNIFICA
==============
LA DISTINZIONE CHE FA TUTTO, e che il nome nasconde: uno SMOOTHER e un
SIMULATION SMOOTHER non fanno la stessa cosa.

    smoother              restituisce  E[alpha | y]        — UNA media
    simulation smoother   restituisce  alpha ~ p(alpha|y)  — UN'ESTRAZIONE

Per il Gibbs serve la seconda.  Se si mettesse la media al posto
dell'estrazione, il pannello "completato" sarebbe piu' liscio del vero e la
catena sottostimerebbe sistematicamente l'incertezza dei parametri: si
tratterebbero i valori latenti come noti quando non lo sono.  E' l'errore
classico dell'imputazione singola, e in un Gibbs corrompe il posterior di TUTTO,
non solo dei latenti.

L'IDEA DI DK 2002, che e' bellissima e vale la pena averla capita.  Estrarre da
p(alpha|y) sembrerebbe richiedere la covarianza congiunta di TUTTI gli stati —
una matrice (T*ns) x (T*ns), da noi 308.210 x 308.210.  Impensabile.  DK
osservano che in un modello lineare gaussiano l'ERRORE di smoothing

    alpha - E[alpha | y]

ha una distribuzione che NON DIPENDE DA y.  Quindi lo si puo' campionare da un
dataset FINTO, generato da noi, di cui conosciamo la verita':

    1. si simula dal modello, senza condizionare: alpha+ e y+;
    2. si calcola alpha_hat* = E[alpha | y - y+] con un normale smoother;
    3. alpha_tilde = alpha_hat* + alpha+  e' un'estrazione da p(alpha | y).

Al posto di una fattorizzazione gigantesca: una simulazione in avanti e UNA
passata di smoother.  L'errore che ci serve lo prendiamo in prestito dal mondo
finto, dove lo sappiamo misurare, perche' e' distribuito come quello vero.


COME SI TRADUCE IN CODICE
=========================
PERCHE' NON `kalman_smoother` DI `kalman.py`
--------------------------------------------
Quello e' uno smoother RTS (Rauch-Tung-Striebel): la ricorsione all'indietro usa
`P_pred[t+1]` e `P_filt[t]`, quindi PRETENDE che tutte le covarianze siano state
conservate.  Alla nostra taglia:

    2 x T x ns^2 x 8 byte = 2 x 490 x 629^2 x 8  =  3.1 GB

per UNA passata, e ne serve una per iterazione MCMC.  Non e' praticabile.
(Ha anche la firma legata all'uscita di `kalman_filter`, che cabla la struttura
DFM — ma il problema vero e' la memoria.)

Si usa quindi la forma a DISTURBI di DK, dove la ricorsione all'indietro e' su
un VETTORE r_t invece che su una matrice:

    r_{t-1} = Z_t' S_t^-1 eta_t + (I - K_t Z_t)' A' r_t          r_T = 0

e la media smussata si ricostruisce con una passata IN AVANTI:

    alpha_hat_1     = a_1 + P_1 r_0
    alpha_hat_{t+1} = c + A alpha_hat_t + Q r_t

Serve solo `P_1` — la covarianza iniziale, che abbiamo — e non tutte le `P_t`.
Memoria: i vettori r_t sono 490 x 629 x 8 = 2.5 MB; i guadagni K_t, che servono
al termine (I - K_t Z_t)', sono 490 x 629 x 37 x 8 ~ 91 MB.  Due ordini di
grandezza sotto i 3.1 GB.

COSA SI IMPORTA DA `kalman.py`, e nient'altro:
    kalman_predict, kalman_update, build_selection_matrix

`kalman_update` restituisce gia' `eta`, `S`, `K` e `WL`: sono esattamente i
quattro oggetti che la ricorsione all'indietro consuma.  Il ciclo in avanti e'
quindi riuso puro; l'unica cosa nuova e' la ricorsione all'indietro e la
ricostruzione, che e' il codice di questo modulo.

LA COSTANTE.  `kalman_predict` non ha intercetta (il DFM non ne ha una nello
stato).  Si somma dopo, come gia' fatto in `state_space.py`: e' esatto, perche'
sposta la media e non tocca la covarianza.  Nella simulazione in avanti e nella
ricostruzione la costante c'e'; nella passata su `y - y+` NON c'e', perche' la
differenza di due processi con la stessa intercetta non ha intercetta.  E' il
punto in cui e' piu' facile sbagliare, ed e' verificato dal test.

IL TEST DI ISOLAMENTO, che e' il motivo per cui questo blocco esiste da solo.
Su un modello piccolo la congiunta di (alpha, y) e' una gaussiana che si puo'
costruire DENSAMENTE, e quindi p(alpha|y) ha media e covarianza calcolabili in
forma chiusa.  Si confrontano media e covarianza empiriche di molte estrazioni
dello smoother contro quelle esatte.  E' un test che prende qualunque errore di
convenzione — un'intercetta di troppo, una trasposta, un indice sfasato — cosa
che un test "gira e non esplode" non farebbe.  Vedi `test_gate5`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.kalman import build_selection_matrix, kalman_predict, kalman_update  # noqa: F401
# L'update VELOCE, locale a bvar/ — vedi la nota in `state_space.py`.
# `kalman_update` resta importato: e' l'oracolo di `tests/test_gate5.py`.
from core.bvar.state_space import kalman_update_fast


@dataclass
class LinearGaussianSS:
    """
    Un modello lineare gaussiano generico, nella convenzione di `kalman.py`.

        alpha_t = c + A alpha_{t-1} + eta_t        eta_t ~ N(0, Q)
        y_t     = Z alpha_t + eps_t                eps_t ~ N(0, R)

    con inizializzazione alpha_0 ~ N(a0, P0).

    Generico di proposito: cosi' il simulation smoother si testa contro modelli
    minuscoli di cui si sa tutto, prima di vedere le 629 dimensioni dell'L-BVAR.

    Attributes
    ----------
    A : (ns, ns)      transizione
    Q : (ns, ns)      covarianza dello stato.  Per un VAR in companion e'
                      blkdiag(Sigma, 0) — SINGOLARE, ed e' normale.
    Z : (n, ns)       osservazione
    R : (n, n)        covarianza di misura.  Zero (o un nugget) quando le
                      variabili SONO componenti dello stato.
    c : (ns,)         intercetta di stato
    a0 : (ns,)        media iniziale
    P0 : (ns, ns)     covarianza iniziale
    """

    A: np.ndarray
    Q: np.ndarray
    Z: np.ndarray
    R: np.ndarray
    c: np.ndarray = None
    a0: np.ndarray = None
    P0: np.ndarray = None
    companion_n: int | None = None

    def __post_init__(self) -> None:
        ns = self.A.shape[0]
        if self.c is None:
            self.c = np.zeros(ns)
        if self.a0 is None:
            self.a0 = np.zeros(ns)
        if self.P0 is None:
            self.P0 = np.eye(ns)
        for nome, M, shape in (("A", self.A, (ns, ns)), ("Q", self.Q, (ns, ns)),
                               ("P0", self.P0, (ns, ns))):
            if M.shape != shape:
                raise ValueError(f"{nome} ha shape {M.shape}, attesa {shape}")
        if self.Z.shape[1] != ns:
            raise ValueError(f"Z ha {self.Z.shape[1]} colonne, attese {ns}")

    _chol: tuple = field(repr=False, default=None, compare=False)

    @property
    def ns(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.Z.shape[0])

    def factors(self) -> tuple:
        """
        I fattori (L_Q, L_R, L_P0) con L L' = M, calcolati UNA VOLTA SOLA.

        PERCHE' LA CACHE NON E' UN'OTTIMIZZAZIONE PREMATURA.  `simulate_forward`
        serve a ogni iterazione MCMC, e queste tre decomposizioni non cambiano
        mai fra un'iterazione e l'altra a parametri fissi.  Alla taglia
        dell'L-BVAR (ns = 629) una `eigh` costa ~7 s: ricalcolarne tre a ogni
        estrazione buttava via ~20 s su 77, cioe' un quarto del tempo totale,
        per riottenere numeri identici.  Misurato, non stimato.

        ATTENZIONE per il Gate 5 blocco 2: i parametri CAMBIANO a ogni
        iterazione del Gibbs (il passo 3 estrae A e Sigma nuovi), quindi lo
        state-space va RICOSTRUITO — non mutato in place — altrimenti la cache
        servirebbe fattori vecchi.  Costruire un nuovo `LinearGaussianSS` e' la
        via giusta e la cache si rigenera da sola.
        """
        if self._chol is None:
            object.__setattr__(self, "_chol",
                               (_psd_chol(self.Q), _psd_chol(self.R),
                                _psd_chol(self.P0)))
        return self._chol


# ─── 1. La passata in avanti ──────────────────────────────────────────────────

@dataclass
class ForwardPass:
    """
    Quel che la passata in avanti conserva per la ricorsione all'indietro.

    NON conserva `P_filt` ne' `P_pred`: e' tutto il punto (vedi l'header).
    """

    u: np.ndarray                 # (T, ns)   Z' S^-1 eta, gia' proiettato
    K: list                       # T elementi (ns, m_t)
    WL: list                      # T elementi (m_t, ns)
    a_pred: np.ndarray            # (T, ns)   f_pred, per la ricostruzione
    P1: np.ndarray = field(repr=False, default=None)   # (ns,ns) P_pred al t=0
    loglik: float = 0.0


def forward_pass(ss: LinearGaussianSS, Y: np.ndarray, *,
                 with_const: bool = True) -> ForwardPass:
    """
    Il filtro in avanti, sulle primitive importate, conservando il minimo
    necessario allo smoother a disturbi.

    Parameters
    ----------
    Y : (T, n)   NaN ammessi: sono il punto.
    with_const : bool
        Se False l'intercetta `c` viene ignorata.  Serve alla passata su
        `y - y+` del simulation smoother, dove l'intercetta si cancella.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    ns = ss.ns
    c = ss.c if with_const else np.zeros(ns)

    u = np.zeros((T, ns))
    Ks: list = []
    WLs: list = []
    a_pred = np.empty((T, ns))
    ll = 0.0
    P1 = None

    cn = ss.companion_n
    f, P = ss.a0.copy(), ss.P0.copy()
    for t in range(T):
        if cn is None:
            f_pred, P_pred = kalman_predict(f, P, ss.A, ss.Q)
        else:
            f_pred, P_pred = companion_predict(f, P, ss.A, ss.Q, cn)
        f_pred = f_pred + c
        a_pred[t] = f_pred
        if t == 0:
            P1 = P_pred.copy()

        W_t = build_selection_matrix(Y[t])
        out = kalman_update_fast(f_pred, P_pred, Y[t], W_t, ss.Z, ss.R)

        WL = out["WL"]
        if WL.shape[0]:
            # u_t = Z' S^-1 eta_t
            u[t] = WL.T @ np.linalg.solve(out["S"], out["eta"])
        Ks.append(out["K"])
        WLs.append(WL)
        ll += out["loglik_t"]
        f, P = out["f_filt"], out["P_filt"]

    return ForwardPass(u=u, K=Ks, WL=WLs, a_pred=a_pred, P1=P1, loglik=float(ll))


# ─── 2. La ricorsione all'indietro e la ricostruzione ─────────────────────────

def smoothed_mean(ss: LinearGaussianSS, fp: ForwardPass, *,
                  with_const: bool = True) -> np.ndarray:
    r"""
    E[alpha_t | y_{1:T}] per ogni t, in forma a DISTURBI.

    All'indietro, su un vettore:

        r_{t-1} = u_t + (I - K_t Z_t)' A' r_t,      r_{T} = 0
                = u_t + A' r_t - Z_t' K_t' A' r_t

    In avanti, per ricostruire la media:

        alpha_hat_0     = a_pred_0 + P_1 r_{-1}
        alpha_hat_{t}   = c + A alpha_hat_{t-1} + Q r_{t-1}

    Returns
    -------
    (T, ns)
    """
    T, ns = fp.u.shape
    c = ss.c if with_const else np.zeros(ns)

    # --- all'indietro: r[t] = r_t, piu' r_prev per t = -1
    r = np.zeros((T + 1, ns))          # r[T] = 0
    for t in range(T - 1, -1, -1):
        At_r = ss.A.T @ r[t + 1]
        corr = fp.WL[t].T @ (fp.K[t].T @ At_r) if fp.WL[t].shape[0] else 0.0
        r[t] = fp.u[t] + At_r - corr

    # --- in avanti: ricostruzione
    alpha = np.empty((T, ns))
    alpha[0] = fp.a_pred[0] + fp.P1 @ r[0]
    for t in range(1, T):
        alpha[t] = c + ss.A @ alpha[t - 1] + ss.Q @ r[t]
    return alpha


# ─── 3. La simulazione in avanti (passo 1 di DK) ──────────────────────────────

def simulate_forward(ss: LinearGaussianSS, T: int, rng: np.random.Generator,
                     *, pattern: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera `alpha+` e `y+` dal modello, SENZA condizionare — il mondo finto di
    cui conosciamo la verita'.

    Parameters
    ----------
    pattern : (T, n) | None
        Se dato, i NaN di `pattern` vengono riprodotti in `y+`.  E' necessario:
        `y - y+` deve avere lo STESSO insieme di osservati di `y`, altrimenti
        lo smoother condizionerebbe su informazione che non abbiamo.

    Returns
    -------
    (alpha_plus, y_plus) : (T, ns) e (T, n)
    """
    ns, n = ss.ns, ss.n
    L_Q, L_R, L_P0 = ss.factors()

    alpha = np.empty((T, ns))
    y = np.empty((T, n))

    a_prev = ss.a0 + L_P0 @ rng.standard_normal(ns)
    for t in range(T):
        alpha[t] = ss.c + ss.A @ a_prev + L_Q @ rng.standard_normal(ns)
        y[t] = ss.Z @ alpha[t] + L_R @ rng.standard_normal(n)
        a_prev = alpha[t]

    if pattern is not None:
        y[np.isnan(np.asarray(pattern, dtype=float))] = np.nan
    return alpha, y


# ─── 3b. La predizione COMPANION-AWARE ────────────────────────────────────────

def companion_predict(f: np.ndarray, P: np.ndarray, A: np.ndarray,
                      Q: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Lo stesso risultato di `kalman_predict`, sfruttando la struttura companion.

    LA STRUTTURA.  Per un VAR(p) in companion form

        A = [ A_1  A_2  ...  A_p ]        <- n righe PIENE
            [  I    0   ...   0  ]        <- ns-n righe che SPOSTANO e basta
            [  0    I   ...   0  ]
            [        ...         ]

    e Q = blkdiag(Sigma, 0): solo il blocco (1,1) e' non nullo.

    IL RISPARMIO.  Nel prodotto A P A' le righe inferiori di A non moltiplicano
    niente: selezionano.  Scrivendo A = [A_top ; S] con S = [I_{ns-n}, 0],

        (A P)  righe alte  = A_top @ P           costa n * ns^2
               righe basse = P[:ns-n, :]         GRATIS, e' una fetta

        (A P A')  colonne a sinistra = (A P) @ A_top'    costa n * ns^2
                  colonne a destra   = (A P)[:, :ns-n]   GRATIS

    e sommare Q tocca solo n^2 elementi.  Totale 2 n ns^2 invece di 2 ns^3:
    un fattore ns/n = p.  A p = 17 e' il margine che il Gate 5 cercava.

    Lo stesso vale per la media: f_pred = A_top @ f (n*ns) piu' uno scorrimento.

    CORRETTEZZA ANCORATA ALLA PRIMITIVA CONDIVISA.  Questa non e' una
    riscrittura del filtro: e' una SPECIALIZZAZIONE che deve dare, entro
    l'errore di macchina, cio' che da' `kalman_predict`.  Il test lo verifica
    confrontando le due strade — `kalman_predict` resta l'ORACOLO, e il vincolo
    "riuso per import, non riscrittura" e' rispettato nella forma che conta: la
    correttezza continua a essere definita dalla primitiva condivisa.

    L'UNICA differenza deliberata e' la simmetrizzazione finale, che
    `kalman_predict` non fa: su una ricorsione lunga la deriva di arrotondamento
    fra P e P' si accumula, e `kalman_update` gia' simmetrizza per lo stesso
    motivo.  Non e' un cambio di semantica.

    ACCURATEZZA MISURATA, e i due numeri vanno letti insieme:

        su UNA chiamata, n da 2 a 12, p da 3 a 6      scarto 0.0 - 1.4e-16
        sullo smoothing COMPLETO a ns = 629, T = 490  scarto relativo 3.4e-08

    Il secondo non e' un errore dell'una o dell'altra strada: e' arrotondamento
    accumulato su 490 passi con ordini di operazione diversi, in un sistema con
    P0 diffusa.  Nessuna delle due e' "quella giusta" — ma vanno riportati
    entrambi, perche' il primo da solo suggerirebbe un'equivalenza a precisione
    di macchina che alla taglia vera non c'e'.

    >>> NON USARE QUESTA STRADA, PER ORA.  Il default e' `companion_n=None`,
    >>> cioe' `kalman_predict`.  Motivo: MISURATO, il guadagno e' NULLO.
    >>>
    >>>     generico        forward 16.28 s   estrazione 15.53 s
    >>>     companion-aware forward 15.75 s   estrazione 16.60 s   ->  x1.0
    >>>
    >>> La stima a priori era x17 (il rapporto dei flop, ns/n = p) ed era
    >>> SBAGLIATA: a ns = 629 una matrice pesa 3.2 MB, ben oltre la cache, quindi
    >>> il regime e' limitato dalla BANDA DI MEMORIA e non dai flop.  I flop
    >>> risparmiati vengono rimpiazzati da copie di fette larghe, e il gemm
    >>> 629^3 del BLAS — multithread — non era il collo di bottiglia.
    >>> (Spiegazione plausibile, non misurata: il micro-profilo che l'avrebbe
    >>> confermata e' uscito internamente incoerente su macchina contesa.)
    >>>
    >>> Il codice resta perche' e' corretto e perche' il regime potrebbe cambiare
    >>> con n o p diversi — ma va riattivato solo dopo averlo RImisurato.

    Parameters
    ----------
    n : int
        Dimensione del blocco.  `A` deve essere la companion di un VAR con
        blocchi n x n; `Q` deve avere solo il blocco (1,1) non nullo.
    """
    ns = A.shape[0]
    A_top = A[:n]                                  # (n, ns)

    # media: la parte alta si calcola, la bassa scorre
    f_pred = np.empty(ns)
    f_pred[:n] = A_top @ f
    f_pred[n:] = f[: ns - n]

    # A P: righe alte calcolate, righe basse = fetta di P
    AP = np.empty((ns, ns))
    AP[:n] = A_top @ P
    AP[n:] = P[: ns - n]

    # (A P) A': colonne di sinistra calcolate, colonne di destra = fetta di AP
    P_pred = np.empty((ns, ns))
    P_pred[:, :n] = AP @ A_top.T
    P_pred[:, n:] = AP[:, : ns - n]

    P_pred[:n, :n] += Q[:n, :n]                    # Q e' zero altrove
    P_pred = 0.5 * (P_pred + P_pred.T)             # come fa kalman_update
    return f_pred, P_pred


def _psd_chol(M: np.ndarray) -> np.ndarray:
    """
    Un fattore L con L L' = M, che regge anche M SINGOLARE.

    Serve perche' la Q di un VAR in companion e' blkdiag(Sigma, 0): singolare
    per costruzione, quindi `np.linalg.cholesky` fallisce.  Si passa per la
    decomposizione spettrale, che con gli autovalori a zero non ha problemi.
    """
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    M = 0.5 * (M + M.T)
    lam, V = np.linalg.eigh(M)
    lam = np.clip(lam, 0.0, None)
    return V * np.sqrt(lam)


# ─── 4. Il simulation smoother ────────────────────────────────────────────────

def simulation_smoother(ss: LinearGaussianSS, Y: np.ndarray,
                        rng: np.random.Generator, *,
                        n_draws: int = 1,
                        numerical_guard: bool = False) -> np.ndarray:
    r"""
    Estrazioni da p(alpha_{1:T} | y_{1:T}) — Durbin & Koopman (2002).

    L'algoritmo, nei tre passi dell'header:

        1.  (alpha+, y+)  simulati dal modello, con lo stesso pattern di NaN
        2.  alpha_hat*  =  E[alpha | y - y+]      <- smoother SENZA intercetta
        3.  alpha_tilde =  alpha_hat* + alpha+

    Il passo 2 gira SENZA intercetta e con media iniziale nulla: `y - y+` e' la
    differenza di due processi con la stessa `c` e la stessa `a0`, quindi
    entrambe si cancellano.  Sbagliare qui produce un bias costante che un test
    "gira e non esplode" non vedrebbe — per questo il test confronta media e
    covarianza contro la congiunta esatta.

    Returns
    -------
    (n_draws, T, ns), oppure (T, ns) se `n_draws == 1`.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]

    # lo smoother del passo 2 non ha intercetta: si costruisce una volta sola
    ss0 = LinearGaussianSS(A=ss.A, Q=ss.Q, Z=ss.Z, R=ss.R,
                           c=np.zeros(ss.ns), a0=np.zeros(ss.ns), P0=ss.P0,
                           companion_n=ss.companion_n)

    out = np.empty((n_draws, T, ss.ns))
    for d in range(n_draws):
        alpha_p, y_p = simulate_forward(ss, T, rng, pattern=Y)
        fp = forward_pass(ss0, Y - y_p, with_const=False)
        correction = smoothed_mean(ss0, fp, with_const=False)
        draw = correction + alpha_p

        if numerical_guard:
            # DK obtains the conditional draw by cancelling an unconditional
            # auxiliary path with its smoothed correction.  For an explosive
            # transition both terms can be enormous even when their sum is
            # ordinary.  A finite sum is then not sufficient: it may have lost
            # every meaningful digit before it reaches the next Gibbs step.
            #
            # This is an accuracy test, not an economic bound on the draw.  A
            # genuinely large conditional draw passes when it does not rely on
            # catastrophic cancellation.  ``sqrt(eps)`` asks for roughly half
            # of double precision to remain; ``ns * eps`` is a conservative
            # first-order bound for the state-sized matrix operations above.
            block = slice(0, ss.n)
            finite_y = np.abs(Y[np.isfinite(Y)])
            data_scale = float(finite_y.max(initial=1.0))
            draw_scale = float(np.abs(draw[:, block]).max(initial=0.0))
            reference = max(1.0, data_scale, draw_scale)
            cancellation_scale = float(
                (np.abs(correction[:, block]) + np.abs(alpha_p[:, block])).max(
                    initial=0.0))
            error_bound = ss.ns * np.finfo(float).eps * cancellation_scale
            if (not np.isfinite(error_bound)
                    or error_bound > np.sqrt(np.finfo(float).eps) * reference):
                raise FloatingPointError(
                    "simulation smoother: cancellazione numericamente non affidabile")

        out[d] = draw

    return out[0] if n_draws == 1 else out


__all__ = [
    "LinearGaussianSS",
    "ForwardPass",
    "forward_pass",
    "smoothed_mean",
    "simulate_forward",
    "simulation_smoother",
]
