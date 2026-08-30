"""
src/bvar/lbvar.py

GATE 5, BLOCCO 2 — L'L-BVAR: il ciclo MCMC a tre passi.  §2.2 del paper.

La teoria della §2.2 sta nell'header di `simsmoother.py` (prima parte): il
modello mensile, i 17 ritardi della nota 10, il ciclo che il paper detta passo
per passo, l'inizializzazione con le spline, la nota 11 sulla natura della
variabile latente.  Qui non si ripete: qui c'e' il ciclo, e i dettagli che
stanno SOLO nel codice di replica degli autori (`functionsGMS/lbvar.m`).


================================================================================
IL CICLO, E DOVE OGNI PEZZO ESISTE GIA'
================================================================================
    passo 1   estrai il pannello mensile completo   -> `simsmoother.py`  (Gate 5.1)
    passo 2   estrai gli iperparametri | pannello   -> `core.step()`     (Gate 1)
    passo 3   estrai (B, Sigma) | iperparametri     -> `core.step()`     (Gate 1)

I passi 2 e 3 SONO `core.step`, e non per comodita': e' il requisito
d'interfaccia fissato al Gate 0 proprio per questo modello.  `step(state, rng,
panel=...)` ricostruisce il target sul nuovo pannello, RIVALUTA la log-posterior
corrente e fa una spazzata, mantenendo la catena degli iperparametri.

Quel "rivaluta" e' essenziale e si vede in `lbvar.m` riga 110-111: `logML_old` e
`logML_new` sono ricalcolati ENTRAMBI a ogni iterazione, sul pannello CORRENTE.
Il pannello cambia, quindi la verosimiglianza del punto "vecchio" cambia con
lui: riusare il valore dell'iterazione precedente sarebbe un errore silenzioso
che invaliderebbe la catena.


================================================================================
I DETTAGLI CHE STANNO SOLO NEL CODICE  (`lbvar.m`)
================================================================================

--- 1. IL PANNELLO E' GREZZO, NON MEDIATO --------------------------------

L'L-BVAR NON usa le medie mobili a 3 mesi della nota 17: quelle sono di Q e C.
Qui i mensili entrano come sono, e le trimestrali sono mensili osservati solo
nell'ultimo mese del trimestre (nota 11).  Vedi la sezione finale dell'header
del pacchetto.

--- 2. L'INIZIALIZZAZIONE: spline, poi si buttano 3 righe ----------------

    [xx,indNaN] = remNaNs_spline(x,options);   % options.k=3, options.method=5
    xx = xx(4:end,:);   x = x(4:end,:);

`remNaNs_spline` metodo 5, per ogni serie: i buchi INTERNI si riempiono con una
spline cubica; quel che resta (bordi) si mette alla mediana e poi si sostituisce
con una media mobile CENTRATA a 2k+1 = 7 termini della serie cosi' riempita.

Serve a due cose sole — il pannello su cui si cerca il modo, e le prime p righe
che restano poi FISSE per tutta la catena — non entra mai nella verosimiglianza
delle iterazioni.

--- 3. P0 = 0: le condizioni iniziali sono DETERMINISTICHE ---------------

Il paper dice: "with variance equal to zero or equal to the prior variance
Psi_ii depending on whether the data is observed or estimated".  Il codice dice
altro:

    temp = kron(ones(1,p),Psi)*0;  temp(initxMiss==0)=0;  P0 = diag(temp);

quel `*0` azzera TUTTO, e la riga successiva rimette a zero anche gli osservati.
Quindi P0 = 0 e `a0 = mvnrnd(M0,P0)` = M0, deterministico.  DIVERGENZA
paper<->codice: seguiamo il codice, e la dichiariamo.

--- 4. LE PRIME p RIGHE NON SI ESTRAGGONO MAI ---------------------------

    xx = [flipud(reshape(M0,N,p)'); atilda(1:N,:)'] ...

`M0` sono le prime p righe del pannello riempito con le spline, e vengono
RIMESSE identiche a ogni iterazione: il pannello estratto e' "prime p righe
fisse + cammino smussato".  Non sono stati latenti, sono condizioni iniziali.

--- 5. LA STIMA USA `xx(1:lastFull,:)`, NON TUTTO IL PANNELLO ------------

    lastFull = find(~isnan(sum(x,2)),1,'last')-3;

e sia la marginal likelihood (riga 110-111) sia l'estrazione di (beta,Su)
(riga 133) girano su `xx(1:lastFull,:)`.  Il bordo frastagliato viene ESTRATTO
dallo smoother ma NON entra nella stima.  Il `-3` compensa esattamente le tre
righe buttate al punto 2, cosi' `lastFull` indicizza la stessa riga prima e dopo
il taglio.

--- 6. LO STATO-SPAZIO SI RICOSTRUISCE, NON SI MUTA ---------------------

Righe 69-71: AA, c2 e QQ sono riassegnate da (beta, Su) all'inizio di OGNI
iterazione.  E' anche il caveat documentato in `LinearGaussianSS.factors()`: i
fattori di Cholesky sono in cache, quindi un oggetto mutato in place servirebbe
fattori vecchi.  Si costruisce un `LinearGaussianSS` nuovo ogni volta.

--- 7. LA CONVENZIONE DELLA COSTANTE NEL DK -----------------------------

Riga 81-88: la simulazione in avanti usa `c2` e parte da `a0`; lo smoother su
`ystar` gira con intercetta ZERO e media iniziale ZERO.  E' la stessa
convenzione di `simsmoother.simulation_smoother`, gia' validata contro l'oracolo
esatto.  (Nel loro `run_Ssmoother.m` — il ramo C-BVAR — la ripartizione e'
opposta ma equivalente: e' il tipo di asimmetria che, sbagliata, darebbe un bias
costante invisibile.)

--- 8. I LIMITI SUGLI IPERPARAMETRI ------------------------------------

Righe 31-36 e 100: proposta fuori dai limiti = rifiuto immediato.

    lambda in [1e-4, 5]      miu in [1e-4, 5]
    psi    in [SSar1/100, SSar1*100]

Noi proponiamo in scala LOG (vedi `hyper.py`), quindi la positivita' e' gratis e
i limiti INFERIORI sono automatici.  I superiori si applicano mettendo -inf
nell'IPERPRIOR (`HyperPrior.lam_max` / `mu_max`), che e' esattamente equivalente
al loro rifiuto secco.

ATTENZIONE, errore fatto e corretto: una prima versione di questo modulo si
limitava a CONTARE le violazioni a valle, lasciandole entrare nella catena.  Lo
smoke test l'ha reso visibile subito — lambda fino a 7.6 con lam_max = 5 — ed e'
il tipo di bug che su un sistema grande sarebbe passato per "mescolamento
lento".

--- 9. IN REAL TIME, n VARIA NEL TEMPO ----------------------------------

Trovato dall'oracolo del Gate 6 a `as_of = 2008-06-20`: l'L-BVAR non partiva.

**NON E' LO SMOOTHER CHE SI ROMPE**, ed e' il punto in cui e' facile
confondersi.  Verrebbe da dire: l'L-BVAR ha il simulation smoother dentro la
stima, quindi le partenze tardive dovrebbe trattarle come stati latenti — e
infatti e' proprio per questo che il profilo `l` ha 37 serie mentre Q/C/B ne
hanno 30.  Lo smoother sa fare il suo lavoro.

Quel che manca e' **l'ANCORAGGIO**.  `last_full_row` (il `lastFull` del punto 5)
cerca l'ultima riga con **tutte** le serie osservate, e quell'indice fissa la
fine del campione di STIMA.  E' una convenzione sul campione, non un requisito
dello smoother.  Con una colonna interamente vuota nessuna riga e' mai piena,
l'ancoraggio non esiste, e il modello resta senza campione.

RIMEDIO: si scarta la colonna vuota, **non si rilassa l'ancoraggio**.
`data.drop_empty_series` toglie le serie con ZERO osservazioni a quella data e
restringe lo spec di conseguenza, cosi' `last_full_row` e' definito
sull'insieme delle serie EFFETTIVAMENTE TRATTENUTE — non come caso particolare,
ma per costruzione.  Le tre ragioni per cui scartare batte rilassare (una
colonna tutta NaN e' puro prior, non uno stato latente; aumenta d e il mixing
peggiora come 1/d; e non sarebbe real-time) stanno per esteso li'.

Le serie PARZIALMENTE osservate restano — PCEC96 a `as_of = 2008-06` ha 17 mesi
di dati e ~260 latenti, ed e' esattamente il caso che il profilo `l` esiste per
trattare.  La soglia e' netta: zero osservazioni si scarta, una o piu' si tiene.

**Da dichiarare in tesi: `n` varia nel tempo.**  Misurato sul profilo `l`:

    as_of        n    serie non ancora disponibili
    2007-01     35    PCEC96, PPIFIS
    2007-06     36    PPIFIS
    2010-01     36    PPIFIS
    2010-04     37    -

Monotono e senza salti: ogni serie entra quando ha accumulato abbastanza storia.
Non e' un compromesso, e' la ricostruzione onesta dell'insieme informativo.

--- 10. IL DK NON REGGE SUL BLOCCO COVID, E C'E' UN RIPIEGO -------------

Il blocco `2020-07-31 .. 2020-10-23` e' stato l'unico a fallire, in tutte le
passate e solo qui.  La causa non e' l'esplosivita' — quella e' voluta, il
punto 8 lo dice — ma la CANCELLAZIONE: DK forma `alpha+ + E[alpha | y - y+]`,
il primo addendo cresce come rho^T e il risultato resta alla scala del dato,
quindi l'errore relativo e' ~ ns * eps * rho^T / scala.  Su T ~ 500 mesi e una
companion sopra il cerchio unitario le cifre finiscono.

Il rimedio sta in `_finite_smoother` (che lo spiega per esteso) e in
`precision_smoother.py` (che spiega l'algoritmo).  Due corollari che chiudono
il caso e che vale la pena avere qui:

  * il ramo di RIUSO non falliva perche' la sua finestra e' di ~45 mesi e non
    ~500 — stesso DK, altro esponente;
  * gli autori questo caso lo hanno EVITATO: `STEP3a_LBVAR_covid.m` r.88
    commenta il trigger di ri-stima al rilascio del PIL, quindi nell'anno del
    Covid l'L-BVAR si stima UNA volta sola (prima settimana) e tutto il resto
    va al riuso.  E il loro driver ha `X_draws(X_draws==inf) = 1e16;`.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.bvar.core import CoreState, step
from src.bvar.data import (
    append_forecast_rows,
    build_panel,
    drop_empty_series,
    load_raw_levels,
    truncate_at_vintage,
)
from src.bvar.hyper import HyperPrior, build_target, init_metropolis
from src.bvar.precision_smoother import precision_draw
from src.bvar.simsmoother import LinearGaussianSS, simulation_smoother
from src.bvar.spec import BVARSpec

#: Ampiezza del filtro di `remNaNs_spline` (options.k = 3 -> media a 2k+1 = 7).
SPLINE_K = 3

#: Righe iniziali scartate dopo il riempimento, come `xx = xx(4:end,:)`.
DROP_HEAD = 3

#: `MCMCconst` di `lbvar.m`: il fattore di scala della proposta Metropolis.
MCMC_CONST = 1.6

MAX_CONSECUTIVE_NUMERICAL_REJECTIONS = 25

#: Di quanto il cammino estratto puo' superare la scala del dato OSSERVATO
#: prima di essere dichiarato fuori dal supporto computazionale.
#:
#: NON E' UN VINCOLO ECONOMICO.  Non dice "questa crescita e' implausibile":
#: dice che un cammino cosi' grande non e' rappresentabile in cio' che viene
#: dopo.  Il passo successivo stima (B, Sigma) su quel pannello e poi ne
#: inverte la Sigma; un cammino 1e24 volte il dato da' `diag(Sigma)` a 1e37 e
#: `Sigma^-1` a 1e-37, e a quel punto la matrice di precisione non e' piu'
#: definita positiva.  E' lo stesso criterio della guardia di cancellazione.
#:
#: IL CORRIDOIO E' LARGO DI PROPOSITO, perche' deve lasciar passare le
#: estrazioni legittime e fermare solo la fuga.  Misurato sul pannello
#: 2020-07-31, dove il dato osservato sta a 91:
#:
#:     rho = 1.015-1.047, tratti ciechi da 278 mesi   max|x| = 91.5   (x1.0)
#:     l'estrazione che ha avvelenato la catena       max|x| = 1.3e26 (x1.4e24)
#:
#: fra i due ci sono ventiquattro ordini di grandezza: qualunque soglia in
#: mezzo separa gli stessi due casi, e 1e6 lascia comodo perfino a un cammino
#: che cresca di mille volte sopra il dato senza motivo apparente.
MAX_PATH_OVER_DATA = 1e6


def _path_within_support(alpha: np.ndarray, Y: np.ndarray, n: int) -> bool:
    """Il cammino estratto e' rappresentabile a valle?

    Vale per ENTRAMBE le strade — DK e precisione — e non e' un dettaglio: la
    prima versione del ripiego controllava solo `isfinite`, cosi' un cammino a
    1e26 (finito!) veniva accettato, entrava nella stima e da li' in poi ogni
    estrazione di (B, Sigma) era degenere.  La catena non si e' fermata dove
    si e' rotta: si e' rotta molto prima, in silenzio.

    Il ramo DK aveva si' un controllo, ma con soglia `sqrt(float_max / T)`
    ~1.9e152: non rifiutava niente di cio' che conta.
    """
    if not np.isfinite(alpha).all():
        return False
    Y = np.asarray(Y, dtype=float)
    finite_y = np.abs(Y[np.isfinite(Y)])
    scale = float(finite_y.max(initial=1.0))
    peak = float(np.max(np.abs(alpha[:, :n]), initial=0.0))
    return peak <= MAX_PATH_OVER_DATA * max(scale, 1.0)


def head_from_a0(a0: np.ndarray, n: int, p: int) -> np.ndarray:
    """Le p righe di testa, in ordine CRONOLOGICO, da `a0`.

    `fit` costruisce `a0 = concat([head[p-1-j] for j in range(p)])`, cioe' il
    piu' recente per primo (e' l'ordine della companion).  Il campionatore a
    precisione ragiona in tempo, non in companion, e vuole l'ordine opposto.
    """
    return np.stack([a0[(p - 1 - i) * n:(p - i) * n] for i in range(p)])


def _finite_smoother(state: CoreState, Y: np.ndarray, rng: np.random.Generator,
                     *, n: int, p: int, a0: np.ndarray
                     ) -> tuple[np.ndarray, bool]:
    """Il cammino latente: Durbin-Koopman, e il ripiego quando il DK non regge.

    IL PROBLEMA, misurato sul blocco 2020-07-31.  L'L-BVAR ammette
    deliberatamente estrazioni esplosive del VAR — `lbvar.m` non ha nessun
    controllo di stabilita', verificato — e a quel vintage la posterior sta
    SOPRA il cerchio unitario in TUTTE le estrazioni (raggio 1.006-1.028).  Il
    cammino ausiliario di DK, che e' simulato dalla PRIOR, cresce allora come
    rho^T su T ~ 500 mesi, mentre il risultato resta alla scala del dato:
    l'estrazione condizionale e' la differenza di due numeri enormi e le cifre
    finiscono.  Non e' un overflow — e' cancellazione, e un risultato FINITO
    puo' essere gia' privo di senso.  Trattazione completa nell'header di
    `precision_smoother.py`.

    IL RAMO NORMALE NON CAMBIA.  Dove il DK regge — cioe' ovunque tranne il
    Covid — si usa il DK, con le stesse chiamate all'RNG di prima: i risultati
    delle passate esistenti restano identici bit per bit.

    IL RIPIEGO.  Quando la guardia di cancellazione scatta, la stessa
    condizionale si estrae dalla matrice di PRECISIONE, che non forma mai il
    cammino ausiliario e quindi non ha niente da cancellare.  Non e' un'altra
    posterior e non e' un'approssimazione: `test_precision_smoother` verifica
    contro un oracolo denso che media e covarianza siano quelle esatte, e §6
    che i due algoritmi coincidano dove il DK e' sano.

    COSA SI E' TOLTO, e perche'.  Prima di qui c'era un ciclo che ri-estraeva
    `(B, Sigma)` fino a 100 volte sperando in un'estrazione rappresentabile.
    Non poteva funzionare: a quel vintage il 100% della posterior e' esplosivo,
    quindi ogni ri-estrazione ricade nella stessa trappola — e infatti la
    catena si fermava dopo 11 mosse.  Rifiutare le estrazioni esplosive
    sarebbe stato peggio: devia dagli autori e rifiuterebbe il 100% delle
    proposte.  Il difetto non e' nell'estrazione, e' nell'algoritmo con cui la
    si trasforma in un cammino.

    Returns
    -------
    (alpha, fallback) con `fallback` True se ha risposto il campionatore a
    precisione.
    """
    ss = build_state_space(state.B, state.Sigma, n, p, a0)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            alpha = simulation_smoother(ss, Y, rng, numerical_guard=True)
    except (FloatingPointError, np.linalg.LinAlgError):
        alpha = None
    if alpha is not None and _path_within_support(alpha, Y, n):
        return alpha, False

    alpha = precision_draw(state.B, state.Sigma, Y,
                           head_from_a0(a0, n, p), rng, n=n, p=p)
    # LA STESSA GUARDIA, e non "anche": il ripiego non e' meno soggetto alla
    # fuga del DK.  Il campionatore a precisione la produce anzi con piu'
    # facilita', perche' e' l'unico dei due che sopravvive abbastanza a lungo
    # da estrarre davvero su un tratto cieco di 278 mesi con rho > 1.
    if not _path_within_support(alpha, Y, n):
        raise FloatingPointError(
            "L-BVAR: il cammino estratto e' fuori dal supporto "
            f"computazionale (max {np.max(np.abs(alpha[:, :n])):.3e} contro "
            f"una scala del dato di "
            f"{np.abs(Y[np.isfinite(Y)]).max(initial=1.0):.3e})")
    return alpha, True


def _candidate_transition(state: CoreState, panel: np.ndarray, Y: np.ndarray,
                          rng: np.random.Generator, *, n: int, p: int,
                          a0: np.ndarray, n_metro: int
                          ) -> tuple[CoreState, np.ndarray | None, bool, bool]:
    """Propose one complete Gibbs sweep without corrupting the valid state.

    A numerically unrepresentable smoother draw rejects the *whole* candidate
    sweep.  The original state is untouched, so the chain makes an explicit
    self-transition instead of being stranded at the last unusable ``(B, Sigma)``.
    This is the finite-precision analogue of rejecting a proposal outside the
    computational support; no stationarity restriction or replacement data are
    introduced.

    Da quando `_finite_smoother` ha il ripiego a precisione questa rete quasi
    non serve piu': ci si arriva solo se ANCHE il campionatore a precisione
    fallisce (Cholesky non definita positiva), cioe' se il sistema e' rotto
    davvero e non solo mal condizionato.  Si tiene apposta: e' il confine fra
    "il DK non regge qui" — che ora si attraversa — e "questo stato non e' un
    modello", che deve restare un errore.
    """
    candidate = copy.deepcopy(state)
    try:
        candidate = step(candidate, rng, panel=panel, n_metro=n_metro)
        alpha, fallback = _finite_smoother(candidate, Y, rng, n=n, p=p, a0=a0)
    except (FloatingPointError, np.linalg.LinAlgError):
        return state, None, False, False
    return candidate, alpha, fallback, True


# ─── 1. L'inizializzazione con le spline ──────────────────────────────────────

def spline_fill(panel: np.ndarray, *, k: int = SPLINE_K
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    `remNaNs_spline` metodo 5, serie per serie.

        1. i buchi INTERNI (fra la prima e l'ultima osservazione) -> spline cubica
        2. quel che resta (bordi) -> mediana della serie
        3. poi sostituito con una media mobile CENTRATA a 2k+1 termini della
           serie cosi' riempita

    Returns
    -------
    (filled, nan_mask)  con `nan_mask` la maschera ORIGINALE dei NaN.
    """
    from scipy.interpolate import CubicSpline

    X = np.asarray(panel, dtype=float).copy()
    T, n = X.shape
    mask = np.isnan(X)

    for i in range(n):
        x = X[:, i]
        obs = ~np.isnan(x)
        if not obs.any():
            X[:, i] = 0.0
            continue
        t1, t2 = int(np.argmax(obs)), int(T - 1 - np.argmax(obs[::-1]))
        interno = np.arange(t1, t2 + 1)
        if obs.sum() >= 2:
            cs = CubicSpline(np.flatnonzero(obs), x[obs])
            x[interno] = cs(interno)

        ancora = np.isnan(x)
        if ancora.any():
            x[ancora] = np.nanmedian(x)
            # media mobile centrata a 2k+1, con padding costante agli estremi
            pad = np.concatenate([np.full(k, x[0]), x, np.full(k, x[-1])])
            ma = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
            x[ancora] = ma[ancora]
        X[:, i] = x

    return X, mask


def last_full_row(panel: np.ndarray) -> int:
    """
    `lastFull` di `lbvar.m` riga 15, in indice 0-based sul pannello GIA' tagliato
    delle prime `DROP_HEAD` righe.

    In MATLAB e' `find(~isnan(sum(x,2)),1,'last') - 3` calcolato PRIMA del taglio
    `x = x(4:end,:)`: il -3 compensa esattamente le tre righe buttate, cosi'
    l'indice punta alla stessa riga prima e dopo.  Qui si calcola direttamente
    sul pannello tagliato, che e' la stessa cosa detta piu' semplicemente.
    """
    pieno = ~np.isnan(np.asarray(panel, dtype=float)).any(axis=1)
    if not pieno.any():
        raise ValueError("nessuna riga completamente osservata nel pannello")
    return int(np.flatnonzero(pieno)[-1])


# ─── 2. Lo stato-spazio dell'L-BVAR ───────────────────────────────────────────

def build_state_space(B: np.ndarray, Sigma: np.ndarray, n: int, p: int,
                      a0: np.ndarray, *, nugget: float = 1e-12,
                      P0: np.ndarray | None = None,
                      p0: str | float | None = None) -> LinearGaussianSS:
    """
    Da (B, Sigma) del core alla companion mensile — `lbvar.m` righe 38-47, 69-71.

        AA(1:N,1:N*p) = beta(2:end,:)'      i coefficienti autoregressivi
        AA(N+1:end, 1:N*(p-1)) = I          lo scorrimento
        c2 = [beta(1,:)'; 0...]             la costante, solo nel primo blocco
        CC(:,1:N) = I                       i mensili SONO le prime n componenti
        QQ(1:N,1:N) = Su                    l'innovazione, solo nel primo blocco

    RICOSTRUITO, non mutato: vedi il punto 6 dell'header.

    `nugget` — R non e' esattamente zero, ed e' **la stessa scelta degli
    autori**, non nostra.  `lbvar.m` riga 88:

        ahatstar = runKF_DK(ystar, AA, CC, QQ, diag(ones(1,N)*1e-12), ...)

    cioe' R = 1e-12 * I, identico al nostro default.

    ATTRIBUZIONE, due correzioni successive — la seconda del 2026-08-01:

      * NON viene da `runKF_DK` riga 88, dove non c'e' nessun `1e-12` (quella
        riga e' un `end`, e la costante non compare nel file).  La citazione
        giusta e' `lbvar.m` r.88 — stesso numero di riga, altro file, ed e'
        probabilmente cosi' che nacque l'errore;
      * e NON e' una scelta nostra.  La versione precedente di questa nota lo
        dichiarava tale citando `run_Ssmoother_block.m` r.64 e
        `run_Ksmoother_FM.m` r.37: quelle due righe passano davvero `zeros(n)`,
        ma sono **le vie del B-BVAR**, non dell'L-BVAR.

    Quindi la mappa vera e':

        L-BVAR   R = 1e-12 * I     autori (`lbvar.m` r.88)   == noi
        B-BVAR   R = 0             autori                     != noi (nugget)

    Una divergenza da dichiarare in meno per l'L-BVAR, e una che resta solo sul
    B-BVAR.  Conseguenza visibile del nugget: le celle OSSERVATE tornano larghe
    ~1e-7 invece che esatte.  Vedi il punto 8 dell'header di `bbvar.py`.

    `P0` None = zeri, cioe' il `temp*0` di `lbvar.m` riga 52 (punto 3).  Il
    B-BVAR passa invece la varianza non condizionata, perche' `run_Ksmoother_FM`
    riga 35 usa `lyapunov_symm`: e' la stessa companion, ma con un'altra
    inizializzazione.  Vedi l'header di `bbvar.py`.

    `p0` — il commutatore CONDIVISO con C e Q (`state_space.initial_covariance`),
    aggiunto per poter misurare 0 contro "lyapunov" con lo stesso disegno usato
    li'.  `None` (default) lascia il comportamento storico: `P0` se dato, zeri
    altrimenti.  Il ripiego quando il blocco stabile e' vuoto e' **0**, non
    kappa*I: su questa companion una diffusa isotropa e' il caso patologico
    misurato al punto 6 di `bbvar.py` (nowcast -97.74%), e non va reintrodotto
    per sbaglio da un `except`.
    """
    ns = n * p
    A = np.zeros((ns, ns))
    A[:n] = B[:n * p, :].T                       # (n, n*p), lag-major come noi
    if p > 1:
        A[n:, : ns - n] = np.eye(ns - n)

    Q = np.zeros((ns, ns))
    Q[:n, :n] = Sigma

    Z = np.zeros((n, ns))
    Z[:, :n] = np.eye(n)

    c = np.zeros(ns)
    c[:n] = B[-1, :]                             # la costante e' l'ultima riga

    if p0 is not None:
        from src.bvar.state_space import initial_covariance
        P0 = initial_covariance(A, Q, kind=p0, kappa_fallback=0.0)[0]
    return LinearGaussianSS(A=A, Q=Q, Z=Z, R=nugget * np.eye(n), c=c,
                            a0=np.asarray(a0, dtype=float),
                            P0=(np.zeros((ns, ns)) if P0 is None    # punto 3
                                else np.asarray(P0, dtype=float)))


# ─── 3. Il risultato ──────────────────────────────────────────────────────────

@dataclass
class LBVARDraws:
    """Le estrazioni dell'L-BVAR."""

    panels: np.ndarray            # (S, T, n)   il pannello mensile completo
    B: np.ndarray                 # (S, k, n)
    Sigma: np.ndarray             # (S, n, n)
    lam: np.ndarray               # (S,)
    mu: np.ndarray                # (S,)
    psi: np.ndarray               # (S, n)
    index: pd.DatetimeIndex = field(repr=False, default=None)
    spec: BVARSpec = field(repr=False, default=None)
    acceptance: float = 0.0

    #: True se le estrazioni vengono dal ramo di RIUSO (`fit_reuse`), dove
    #: (B, Sigma) sono ereditati e non ristimati.
    reused: bool = False
    numerical_rejections: int = 0

    #: Quanti cammini sono stati estratti dalla matrice di precisione invece
    #: che dal DK.  Zero ovunque tranne dove la companion e' esplosiva su
    #: campione lungo — nella passata 2007-2025 significa: solo il Covid.
    precision_fallbacks: int = 0

    @property
    def S(self) -> int:
        return int(self.B.shape[0])

    def growth(self, series_id: str = "GDPC1") -> pd.DataFrame:
        """
        Crescita trimestrale ANNUALIZZATA `100*((x_t/x_{t-3})^4 - 1)` ai
        quarter-end.  Stessa convenzione di `cbvar.CBVARDraws.growth`, cosi' i
        due modelli — e i due rami dell'L-BVAR — sono confrontabili senza
        riscrivere la trasformazione ogni volta.
        """
        j = list(self.spec.series).index(series_id)
        lv = self.panels[:, :, j]
        if self.spec.transform[j] == "log":
            lv = np.exp(lv)
        if lv.shape[1] <= 3:
            raise ValueError(f"finestra troppo corta ({lv.shape[1]} mesi)")
        g = 100.0 * ((lv[:, 3:] / lv[:, :-3]) ** 4 - 1.0)
        idx = self.index[3:]
        qe = idx.month.isin((3, 6, 9, 12))
        return pd.DataFrame(g[:, qe].T, index=idx[qe])

    def summary(self) -> str:
        q = lambda a: np.percentile(a, [5, 50, 95])                  # noqa: E731
        ql, qm = q(self.lam), q(self.mu)
        return (f"L-BVAR  S={self.S}  n={self.spec.n}  p={self.spec.p}  "
                f"T={self.panels.shape[1]}\n"
                f"  accettazione {self.acceptance:.1%}\n"
                f"  lambda  mediana {ql[1]:.4f}   [90%: {ql[0]:.4f}, {ql[2]:.4f}]\n"
                f"  mu      mediana {qm[1]:.4f}   [90%: {qm[0]:.4f}, {qm[2]:.4f}]")


# ─── 4. Il ciclo ──────────────────────────────────────────────────────────────

def fit(
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    horizon: int = 0,
    n_draws: int = 1200,
    burn: int = 200,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    lam_max: float = 5.0,
    mu_max: float = 5.0,
    n_metro: int = 1,
    verbose: bool = True,
) -> LBVARDraws:
    """
    Il ciclo a tre passi di §2.2 — `lbvar.m`.

    Parameters
    ----------
    spec : BVARSpec | None   None costruisce il profilo `l` (37 serie, p=17)
    lam_max, mu_max : float  i limiti superiori di `lbvar.m` righe 33-34
    n_draws, burn : int
        Default 1200 e 200, cioe' **1000 estrazioni tenute** — la taglia degli
        autori (`STEP3_LBVAR.m` riga 21: `nDraws = 1100`, `nBurn = 100`).  Il
        burn e' doppio del loro, per prudenza; e' l'unico scostamento.

        E' anche la taglia che serve: misurato per bootstrap sulla catena
        definitiva, con 1100 tenute l'errore Monte Carlo sui quantili del
        nowcast e' 0.128 / 0.058 / 0.081 pp su q05 / q50 / q95, cioe' l'1-3%
        dell'ampiezza della banda 90%.  E gli ESS su cio' che FORMA la densita'
        sono ~1000 su 1100 (B mediana 999.7, pannello mediana 1064.1).
    n_metro : int
        Spazzate di Metropolis per ogni estrazione dello smoother (vedi
        `core.step`).  **Default 1 = il ciclo degli autori.**  Vedi sotto
        perche' alzarlo non serve.


    IL MESCOLAMENTO DEGLI IPERPARAMETRI: UN RISULTATO, NON UN BUG
    =============================================================
    Il ciclo mostra due popolazioni nettamente separate, ed e' strutturale:

        B, pannello latente   ESS/iterazione ~0.95   (coniugate, quasi iid)
        lambda, mu, psi       ESS/iterazione ~0.03   (Metropolis)

    TENTATIVO FATTO E FALLITO, tenuto qui perche' il percorso e' istruttivo.
    Si erano aggiunte due cose: 10 spazzate di Metropolis per iterazione e la
    taratura di `c` nel burn-in verso il 20% di accettazione di GLP App. B.
    Misurato su 1300 iterazioni (178 min):

        ESS/iterazione di lambda   0.053  ->  0.030      PEGGIORATO
        accettazione               5.8% -> 1.6% -> 1.4% -> 2.8%
        c                          1.6 -> 1.204 -> 0.833 -> 0.575 -> 0.407

    RESTRINGERE la proposta ha ABBASSATO l'accettazione: il contrario di come si
    comporta un Metropolis proposal-limited.  Con l'autocorrelazione a lag 1 di
    0.939 su lambda, la catena non e' "rifiutata troppo" — e' TRASCINATA.

Il target si muove sotto la catena: ogni iterazione ricostruisce la
    condizionale degli iperparametri su un PANNELLO LATENTE DIVERSO, e
    lambda/mu sono agganciati a quel pannello.  Le spazzate multiple equilibrano
    gli iperparametri sul pannello corrente, poi il pannello cambia e sono di
    nuovo fuori posto.  E' una cresta nella posteriore congiunta, non una
    taratura sbagliata, e da li' non si esce ne' con piu' iterazioni ne' con una
    proposta piu' stretta.

    ATTENZIONE — CORREZIONE DEL 2026-07-29, dall'esperimento di controllo.
    Quanto sopra spiega bene perche' le spazzate multiple PEGGIORARONO, e resta
    valido per quello.  Ma NON e' la causa dell'ESS basso in se': il B-BVAR ha
    lo stesso core e lo stesso Metropolis con il Kalman A VALLE, quindi il
    target degli iperparametri e' FISSO per costruzione — e mescola PEGGIO.

        catena definitiva 1000/500     L-BVAR        B-BVAR (target fisso)
        ESS/iterazione di lambda        0.053         0.015
        autocorr lag 1 di lambda        0.939         0.986
        accettazione                    5.8%          20.8%  (a bersaglio!)
        dimensione della proposta       39            86

    Nel B-BVAR l'accettazione e' centrata sul 20% dell'App. B e l'ESS resta
    all'1.5%: un Metropolis *proposal-limited* si sbloccherebbe, questo no.

    CHIUSO IL 2026-08-01 — E' LA DIMENSIONE, MISURATA.  `tests/test_mixing`
    isola d a modello fermo (Q-BVAR, sottoinsiemi annidati, stessa T, stesso
    seme, c tarato al 20% in ogni cella): ESS/iterazione x d resta piatto fra
    d=2 e d=32 e la pendenza di log(ESS/it) su log(d) e' **-1.10** contro il
    -1.00 che la teoria del random-walk Metropolis prevede (Roberts-Gelman-Gilks
    1997).  E' la legge 1/d, cioe' il costo noto dell'algoritmo dell'Appendice B
    in dimensione alta — non una patologia dei nostri modelli.

    Cade anche la lettura "e' una cresta": la posteriore degli iperparametri e'
    MITE (|corr| mediana 0.028, cond(W) = 58.8), e la proposta c*W e' gia'
    allineata alle correlazioni per costruzione, perche' W E' la forma locale
    della posteriore.  Cio' che W non puo' correggere e' d.

    Trattazione completa per la tesi: header di `hyper.py`, sezione
    "IL MESCOLAMENTO"; sintesi nel README, sezione omonima.

    DECISIONE: si resta sul ciclo degli autori.  Dove si replica fedelmente, quel
    comportamento E' il risultato — da riportare, non da superare.  Il criterio
    che conta e' il nowcast, e quello e' sano.

    PERCHE' NON CONTAMINA IL DELIVERABLE, misurato: corr(nowcast, mu) = -0.045,
    e separando le estrazioni con mu alto da quelle con mu basso la mediana del
    nowcast si sposta di -0.14 pp, dentro l'errore Monte Carlo.  Il prezzo del
    cattivo mescolamento e' la rappresentazione dell'incertezza sullo shrinkage
    — la rivendicazione di Cimadomo §2 — non il numero.

    MU CONTRO IL TETTO  <<< da portare al relatore
    ----------------------------------------------
    Sulla catena definitiva il 15.2% delle estrazioni ha mu > 4.9 e il 9.0%
    > 4.95, con `mu_max = 5.0` (lambda invece sta comodo, 0.42-0.49).  mu sta al
    DENOMINATORE della dummy, quindi mu grande = sum-of-coefficients DEBOLE: i
    dati chiedono di spegnere il soc e il limite glielo impedisce.

    Il limite e' fedele — e' `miu_max = 5` di `lbvar.m` riga 34 — ma NON e'
    verificabile dal loro codice se anche nei loro run mu ci si appoggi: non
    pubblicano i valori stimati.  Resta quindi aperto se sia una proprieta' del
    nostro pannello a n=37 o un segnale sul soc.
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("L") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    p = spec.p

    # ── il pannello: mensile, GREZZO (niente medie mobili), in unita' del modello
    # `X(1:nowcastM+horizon,:)` degli autori: il pannello si ferma al
    # vintage, poi si aggiungono le righe di previsione.  Senza il taglio
    # la coda cieca e' di 96 righe invece di 26, e su un'estrazione
    # esplosiva questo cambia la scala del cammino di cento ordini di
    # grandezza.  Vedi `data.truncate_at_vintage`.
    panel = append_forecast_rows(
        truncate_at_vintage(build_panel(spec, as_of, raw=raw), as_of),
        horizon)
    # REAL-TIME: le serie che a questa data non hanno NEMMENO
    # un'osservazione escono dal modello, e con loro dallo spec.  Non e' un
    # ripiego per far funzionare `last_full_row`: e' la ricostruzione
    # dell'insieme informativo disponibile a `as_of`.  Trattazione completa
    # in `data.drop_empty_series`, e il punto 9 dell'header.
    panel, spec = drop_empty_series(panel, spec)
    # `n` SI LEGGE QUI, DOPO IL DROP, e non insieme a `p` prima del pannello.
    # Leggerlo prima lo congelava a 37 mentre lo spec scendeva a 36, e il
    # disallineamento non si vedeva fino alla companion di `build_state_space`
    # (`A[:n] = B[:n*p, :].T`), che alzava
    #     could not broadcast (36,613) into (37,629)
    # con 613 = 36*17+1 = k nuovo e 629 = 37*17 = n*p vecchio.  Nel 2007-2010
    # l'L-BVAR non partiva affatto: PCEC96 e PPIFIS non esistono ancora.
    n = spec.n
    idx_full = panel.index
    X = panel.to_numpy(dtype=float)

    # ── spline + taglio delle prime DROP_HEAD righe (punto 2)
    XX, _ = spline_fill(X)
    XX, X = XX[DROP_HEAD:], X[DROP_HEAD:]
    index = idx_full[DROP_HEAD:]
    lf = last_full_row(X)
    if verbose:
        print(f"  pannello {X.shape[0]} x {n}, lastFull = {lf} "
              f"({index[lf].date()}), NaN {int(np.isnan(X).sum())}")

    # ── il modo, UNA VOLTA SOLA, sul pannello riempito (punto della §2.2)
    if verbose:
        print("  cerco il modo a posteriori (una tantum)...")
    # I limiti superiori entrano nell'IPERPRIOR (-inf fuori), che e' esattamente
    # il rifiuto secco di `lbvar.m` riga 100.  Contarli a valle, come faceva una
    # prima versione di questo modulo, li lasciava invece ENTRARE nella catena:
    # lo smoke test l'ha reso visibile (lambda fino a 7.6 con lam_max=5).
    hp = HyperPrior(lam_max=lam_max, mu_max=mu_max)
    target0 = build_target(spec, XX[: lf + 1], hp)
    state = CoreState(target=target0,
                      metro=init_metropolis(target0, rng, c=MCMC_CONST))

    # ── M0: le prime p righe, FISSE per tutta la catena (punto 4)
    head = XX[:p]                                    # (p, n)
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])   # piu' recente prima
    Y_in = X[p:]                                     # yinput, con i NaN
    T_in = Y_in.shape[0]

    S_keep = n_draws - burn
    panels = np.empty((S_keep, p + T_in, n))
    Bs = np.empty((S_keep, spec.k, n))
    Ss = np.empty((S_keep, n, n))
    lams = np.empty(S_keep); mus = np.empty(S_keep); psis = np.empty((S_keep, n))
    B_cur, S_cur = state.B, state.Sigma
    if B_cur is None:                                # la prima spazzata li crea
        state = step(state, rng)
        B_cur, S_cur = state.B, state.Sigma

    # Il primo smoother e poi quello dell'iterazione successiva sono calcolati
    # in pipeline.  Dove non serve un retry, l'ordine delle chiamate RNG e tutti
    # i risultati restano identici al ciclo precedente.  Il look-ahead finale
    # garantisce inoltre che nessun (B, Sigma) non verificato entri nella cache.
    alpha, fallback = _finite_smoother(state, Y_in, rng, n=n, p=p, a0=a0)
    n_fallback = int(fallback)
    if fallback and verbose:
        print("    DK non affidabile: estrazione dalla matrice di precisione")

    numerical_rejections = 0
    consecutive_rejections = 0

    for it in range(n_draws):
        # passo 1 — il cammino e' gia' stato estratto dalla pipeline
        drawn = np.vstack([head, alpha[:, :n]])                  # punto 4

        # Passi 2 e 3, piu' il smoother successivo, sono una sola transizione
        # candidata.  Se la parte DK non e' rappresentabile in doppia
        # precisione, la transizione e' un self-loop e lo stato valido resta
        # intatto.  Questo evita che un pannello numericamente corrotto avveleni
        # tutte le condizionali successive.
        candidate, next_alpha, fallback, accepted = _candidate_transition(
            state, drawn[: lf + 1], Y_in, rng,
            n=n, p=p, a0=a0, n_metro=n_metro)
        if accepted:
            state = candidate
            alpha = next_alpha
            n_fallback += int(fallback)
            consecutive_rejections = 0
        else:
            numerical_rejections += 1
            consecutive_rejections += 1
            if consecutive_rejections >= MAX_CONSECUTIVE_NUMERICAL_REJECTIONS:
                raise FloatingPointError(
                    "L-BVAR: catena ferma dopo "
                    f"{consecutive_rejections} transizioni numeriche rifiutate; "
                    "ultimo stato valido preservato")

        hyp = state.hyper
        B_cur, S_cur = state.B, state.Sigma
        if accepted and fallback and verbose:
            print(f"    iterazione {it + 1}: DK non affidabile, "
                  "estrazione dalla matrice di precisione")

        if it == burn - 1:
            state.metro.n_accept = state.metro.n_prop = 0   # statistiche pulite

        if it >= burn:
            j = it - burn
            panels[j], Bs[j], Ss[j] = drawn, B_cur, S_cur
            lams[j], mus[j], psis[j] = hyp.lam, hyp.mu, hyp.psi
        if verbose and (it + 1) % max(1, n_draws // 10) == 0:
            print(f"    iterazione {it + 1}/{n_draws}  "
                  f"lambda {hyp.lam:.3f}  mu {hyp.mu:.3f}  "
                  f"acc {state.metro.acceptance:.1%}")

    if numerical_rejections:
        warnings.warn(
            f"L-BVAR: {numerical_rejections}/{n_draws} transizioni numeriche "
            "rifiutate e sostituite da self-loop espliciti",
            RuntimeWarning, stacklevel=2)
    if n_fallback:
        warnings.warn(
            f"L-BVAR: {n_fallback}/{n_draws + 1} cammini estratti dalla "
            "matrice di precisione perche' il DK non era affidabile "
            "(companion esplosiva su campione lungo)",
            RuntimeWarning, stacklevel=2)

    return LBVARDraws(panels=panels, B=Bs, Sigma=Ss, lam=lams, mu=mus, psi=psis,
                      index=index[: p + T_in], spec=spec,
                      acceptance=state.metro.acceptance,
                      numerical_rejections=numerical_rejections,
                      precision_fallbacks=n_fallback)


# ─── 5. Il ramo di RIUSO — `lbvar_NE.m` ───────────────────────────────────────
#
# LO SCHEMA A DUE VELOCITA' DEGLI AUTORI  (`STEP3_LBVAR.m` righe 88-111)
# =======================================================================
# Il run completo NON si rifa' ogni settimana.  Il trigger, riga ~90, e':
#
#     if (yy == 4 && ww == 1) || (ww>2 && <la colonna del PIL nel vintage della
#                                 settimana ww ha PIU' osservazioni di ww-1>)
#
# cioe' al PRIMO vintage, e poi **a ogni nuovo rilascio del PIL** — una volta
# per trimestre.  Tutte le altre settimane vanno a `lbvar_NE`, che:
#
#     - NON cerca il modo,  NON fa Metropolis,  NON estrae (B, Sigma);
#     - riusa le S estrazioni di (B, Sigma) dell'ultimo run completo;
#     - per ognuna fa UNA passata di simulation smoother.
#
# E le due economie che rendono il ramo davvero leggero:
#
#   1. LA FINESTRA E' CORTA.  Il chiamante passa `X(lastFull-p-3:end,:)`, non
#      tutto il pannello: si filtra il solo BORDO.  Misurato da noi, lo smoother
#      costa ~66-72 ms per mese di finestra, quindi ~3.4 s su 47 mesi contro
#      ~32 s su 479.
#   2. LA STORIA NON SI RICAMPIONA AFFATTO.  Riga 111:
#          X_draws = cat(1, repmat(X(4:lastFull-1,:),1,1,nDraws-nBurn), ...)
#      il dato OSSERVATO viene replicato identico su tutte le estrazioni.  Non
#      e' un'approssimazione: dove il dato c'e', lo smoother restituirebbe il
#      dato stesso (R = 1e-12).  E' la stessa logica di `state_space.edge_window`
#      gia' validata al Gate 3 per il C-BVAR.
#
# NOTA su `lastFull`: qui si calcola SENZA il `-3` di `lbvar.m` riga 15, perche'
# la finestra viene tagliata prima e l'indice si riferisce gia' al pannello
# tagliato.  E' la differenza fra le righe 8 di `lbvar_NE.m` e 15 di `lbvar.m`,
# facile da leggere come una svista e invece voluta.

def fit_reuse(
    B_draws: np.ndarray,
    Sigma_draws: np.ndarray,
    spec: BVARSpec | None = None,
    *,
    as_of=None,
    horizon: int = 0,
    rng: np.random.Generator | None = None,
    raw: pd.DataFrame | None = None,
    lam: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    psi: np.ndarray | None = None,
    p0: str | float | None = None,
    verbose: bool = True,
) -> LBVARDraws:
    """
    Il ramo di riuso: `lbvar_NE.m`.  Nessuna stima, solo lo smoother sul bordo.

    Parameters
    ----------
    B_draws, Sigma_draws : (S, k, n) e (S, n, n)
        Le estrazioni dell'ultimo run completo — `BETA`, `SU` di `lbvar.m`.
    lam, mu, psi
        Gli iperparametri di quel run, trasportati tali e quali per tenere
        l'oggetto di uscita della stessa forma.  Non vengono usati: nel ramo di
        riuso non si stima nulla.
    """
    rng = np.random.default_rng() if rng is None else rng
    spec = BVARSpec.from_config("L") if spec is None else spec
    raw = load_raw_levels() if raw is None else raw
    p = spec.p
    S = int(B_draws.shape[0])

    # `X(1:nowcastM+horizon,:)` degli autori: il pannello si ferma al
    # vintage, poi si aggiungono le righe di previsione.  Senza il taglio
    # la coda cieca e' di 96 righe invece di 26, e su un'estrazione
    # esplosiva questo cambia la scala del cammino di cento ordini di
    # grandezza.  Vedi `data.truncate_at_vintage`.
    panel = append_forecast_rows(
        truncate_at_vintage(build_panel(spec, as_of, raw=raw), as_of),
        horizon)
    # REAL-TIME: le serie che a questa data non hanno NEMMENO
    # un'osservazione escono dal modello, e con loro dallo spec.  Non e' un
    # ripiego per far funzionare `last_full_row`: e' la ricostruzione
    # dell'insieme informativo disponibile a `as_of`.  Trattazione completa
    # in `data.drop_empty_series`, e il punto 9 dell'header.
    panel, spec = drop_empty_series(panel, spec)
    n = spec.n                                   # DOPO il drop — vedi `fit`

    # CONFORMITA' COL RUN CHE SI STA RIUSANDO.  (B, Sigma) sono indicizzati
    # sulle serie della stima piena: applicarli a un insieme diverso non e' una
    # approssimazione, e' un errore di algebra.  Il caso si presenta davvero —
    # una serie che entra A META' TRIMESTRE (PCEC96 nel 2007, PPIFIS a fine
    # 2009) allarga il pannello mentre i parametri restano quelli di prima —
    # ed e' la ragione per cui `evaluate` passa qui lo spec RIDOTTO della stima,
    # non quello di configurazione.  Se la guardia scatta, il chiamante sta
    # riusando la cache sbagliata: meglio fermarsi che produrre numeri.
    if B_draws.shape[1] != spec.k or Sigma_draws.shape[1] != n:
        raise ValueError(
            f"riuso non conforme: le estrazioni hanno k={B_draws.shape[1]}, "
            f"n={Sigma_draws.shape[1]}, il pannello a {as_of} ha k={spec.k}, "
            f"n={n}.  Passa a `fit_reuse` lo spec della stima piena "
            f"(`LBVARDraws.spec`), non `BVARSpec.from_config('L')`.")
    X_full = panel.to_numpy(dtype=float)
    lf_full = last_full_row(X_full)                 # senza -3, vedi la nota

    # la finestra corta: `X(lastFull-p-3:end,:)`
    start = max(0, lf_full - p - DROP_HEAD)
    Xw = X_full[start:]
    idx_w = panel.index[start:]

    XXw, _ = spline_fill(Xw)
    XXw, Xw = XXw[DROP_HEAD:], Xw[DROP_HEAD:]
    idx_w = idx_w[DROP_HEAD:]
    head = XXw[:p]
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])
    Y_in = Xw[p:]
    if verbose:
        print(f"  riuso: finestra {Xw.shape[0]} mesi "
              f"({idx_w[0].date()} - {idx_w[-1].date()}), S={S}")

    W = p + Y_in.shape[0]
    panels = np.empty((S, W, n))
    n_fallback = 0
    for s in range(S):
        # Stessa guardia del ramo pieno, e per la stessa ragione.  Qui il DK
        # regge quasi sempre — la finestra e' di ~45 mesi contro ~500, e
        # rho^45 non e' rho^500: e' PROPRIO per questo che il Covid falliva
        # solo nelle settimane di stima piena.  Ma le estrazioni esplosive
        # ereditate sono le stesse, quindi la rete va tesa anche qui.
        ss = build_state_space(B_draws[s], Sigma_draws[s], n, p, a0, p0=p0)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                alpha = simulation_smoother(ss, Y_in, rng, numerical_guard=True)
            if not np.isfinite(alpha).all():
                raise FloatingPointError("cammino non finito")
        except (FloatingPointError, np.linalg.LinAlgError):
            alpha = precision_draw(B_draws[s], Sigma_draws[s], Y_in,
                                   head, rng, n=n, p=p)
            n_fallback += 1
        panels[s] = np.vstack([head, alpha[:, :n]])
    if n_fallback:
        warnings.warn(
            f"L-BVAR (riuso): {n_fallback}/{S} cammini estratti dalla matrice "
            "di precisione perche' il DK non era affidabile",
            RuntimeWarning, stacklevel=2)

    # la storia PRIMA della finestra: dato osservato, replicato (riga 111)
    hist = X_full[:start + DROP_HEAD]
    if hist.shape[0]:
        panels = np.concatenate(
            [np.repeat(hist[None, :, :], S, axis=0), panels], axis=1)
    index = panel.index[: start + DROP_HEAD + W]

    z = lambda a, sh: (np.zeros(sh) if a is None else np.asarray(a))   # noqa: E731
    return LBVARDraws(panels=panels, B=np.asarray(B_draws),
                      Sigma=np.asarray(Sigma_draws),
                      lam=z(lam, S), mu=z(mu, S), psi=z(psi, (S, n)),
                      index=index, spec=spec, acceptance=float("nan"),
                      reused=True, precision_fallbacks=n_fallback)


__all__ = ["spline_fill", "last_full_row", "build_state_space",
           "head_from_a0", "LBVARDraws", "fit", "fit_reuse",
           "MCMC_CONST", "SPLINE_K", "DROP_HEAD"]
