"""
src/bvar/niw.py

IL POSTERIOR CONIUGATO Normal-Inverse-Wishart, e la marginal likelihood.


================================================================================
IL VAR E LE DUE INCOGNITE  —  da qui parte tutto
================================================================================
Il modello e' l'eq. (1) di Cimadomo, un VAR di ordine p:

    x_t = A_0 + A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + eps_t

dove x_t e' il vettore n x 1 delle variabili al tempo t, le A_s sono matrici
n x n di coefficienti, A_0 e' il vettore delle costanti, e

    eps_t ~ N(0, Sigma)

Impilando le T osservazioni una sotto l'altra, questo diventa una REGRESSIONE
MULTIVARIATA — la forma con cui lavora tutto il pacchetto:

        Y      =      X      B      +      U
     (T x n)      (T x k) (k x n)       (T x n)

    riga t di Y  =  x_t'                     la variabile dipendente
    riga t di X  =  (x_{t-1}', ..., x_{t-p}', 1)     i regressori
    colonna i di B  =  i coefficienti dell'EQUAZIONE i
    k = n*p + 1     coefficienti per equazione (l'1 e' la costante)

Cioe': n regressioni lineari che condividono gli stessi regressori, e i cui
errori sono correlati fra loro.

LE INCOGNITE SONO DUE, E VANNO TENUTE DISTINTE:

  B  (k x n)   I COEFFICIENTI.  Come il passato di ciascuna variabile entra
               nell'equazione di ciascun'altra.  E' l'oggetto su cui parla il
               prior di Minnesota (dummies.py) e su cui il sum-of-coefficients
               impone la restrizione di lungo periodo.

  Sigma (n x n)  LA MATRICE DI VARIANZA-COVARIANZA DEGLI ERRORI VERI DEL VAR.
               Non e' un artificio tecnico: e' la covarianza contemporanea degli
               shock eps_t.  La diagonale dice quanto e' rumorosa ciascuna
               equazione; i fuori-diagonale dicono quali shock arrivano
               insieme.  Sono proprio quelle correlazioni contemporanee che ai
               Gate 3-4 faranno muovere il nowcast del PIL quando esce l'ISM.

E' importante fissare questo adesso, perche' l'iperparametro `psi` e' UN PRIOR
SU Sigma — sulla sua diagonale, per l'esattezza.  Vedi la sezione conclusiva
di `hyper.py` ("CHE COSA SONO DAVVERO I TRE IPERPARAMETRI").

Il problema che rende necessario tutto il resto: nel nostro Q-BVAR
k = 30*5+1 = 151 e n = 30, quindi B ha 4530 elementi, stimati su T = 130
osservazioni trimestrali.  L'OLS qui non e' "impreciso": e' letteralmente
indefinito (X'X e' singolare, rango 130 su 151 — misurato in test_dummies §6).
E' il prior a rendere il problema risolubile.
================================================================================

Qui si raccoglie il frutto del Blocco 1: dati gli iperparametri (lambda, psi,
mu), i parametri del VAR si aggiornano IN FORMA CHIUSA e la densita' dei dati
integrando via (B, Sigma) e' anch'essa in forma chiusa.  Nessun MCMC a questo
livello.  L'MCMC (Blocco 5, `hyper.py`) sta uno strato sopra, sugli
iperparametri.

Fonte vincolante: GLP (2015), Appendice A — che abbiamo nella versione ECB
Working Paper 1494 in docs/BVARs/.

    (A.8)   Sigma | Y ~ IW( Psi + eps'eps + (B-b)' Omega^-1 (B-b),  T-p+d )
    (A.9)   beta | Sigma, Y ~ N( beta_hat,  Sigma (x) (x'x + Omega^-1)^-1 )
    (A.13)  la ML in forma chiusa
    (A.14)  la stessa, in forma NUMERICAMENTE STABILE
    (A.3)   con dummy observations:  ML = p(Y+) / p(Y*)


COME SI INCASTRANO I BLOCCHI 2, 4 E 5 — leggere prima di modificare
====================================================================
C'e' una scelta architetturale che va capita, perche' i due paper implementano
la stessa cosa in due modi diversi e noi usiamo l'uno per stimare e l'altro per
verificare.

BGR (2010) mette TUTTO nelle dummy: Minnesota, covarianza, costante.  E' la
strada del Blocco 4.
GLP (2015) invece tiene la Minnesota in FORMA ANALITICA — l'eq. (3) da'
direttamente Omega — e usa le dummy SOLO per cio' che una forma analitica
comoda non ce l'ha: il sum-of-coefficients (e il dummy-initial-observation).
A.3 lo dice esplicitamente: "It is common in the literature to implement SOME
conjugate priors using dummy observations (e.g. the sum-of-coefficients and the
dummy-initial-observation priors)".

Le due strade sono equivalenti, ma A.13/A.14 sono PARAMETRIZZATE su
(Omega, Psi, b, d): sono scritte per la strada di GLP.  Siccome per il
campionatore l'autorita' e' GLP, seguiamo GLP:

    Omega, b   <- forma analitica, dal Blocco 2 (minnesota_omega_diag / _mean)
    Psi, d     <- diag(psi), n+2
    dummy      <- SOLO il blocco sum-of-coefficients del Blocco 3
    ML         <- A.14 valutata due volte, rapporto secondo A.3

IL BLOCCO 4 NON E' SPRECATO, anzi: e' cio' che ci ha DIMOSTRATO che l'Omega
analitica del Blocco 2 e' quella giusta (l'oracolo, con errore 1e-16).  Senza
quella verifica staremmo usando qui una formula creduta corretta invece che
provata corretta.

Una conseguenza pratica importante: **Omega e' DIAGONALE** (eq. 3), quindi
Omega^-1 e' banale e non serve mai invertire una k x k per il prior.  Il solo
sistema k x k da risolvere e' (x'x + Omega^-1).


NUMERICA: PERCHE' A.14 E NON A.13
==================================
A.13 contiene |Omega|^(-n/2) e |Psi|^(d/2) presi separatamente.  Coi nostri
numeri veri sono catastrofici: Omega ha elementi da ~1e-4 a ~1e8 (la costante),
quindi |Omega| su k=151 dimensioni va in overflow/underflow immediato, molto
prima che il rapporto finale — che e' un numero ragionevole — venga calcolato.

GLP A.2 lo dice: "For large systems, (A.13) is numerically unstable and it is
convenient to replace it with the equivalent expression (A.14)".  A.14 combina i
determinanti a coppie PRIMA di prendere i logaritmi:

    |Omega|^(-n/2) |x'x + Omega^-1|^(-n/2)  =  |D_Omega' x'x D_Omega + I_k|^(-n/2)
    |Psi|^(d/2) |Psi + S|^(-(T-p+d)/2)      =  |Psi|^(-(T-p)/2) |I_n + D_Psi' S D_Psi|^(-(T-p+d)/2)

con D_Omega D_Omega' = Omega e D_Psi D_Psi' = Psi^-1.  Gli argomenti dei due
determinanti sono ora (I + qualcosa di semidefinito positivo), quindi i loro
autovalori sono >= 1 e i logaritmi si sommano senza mai uscire dal range.
GLP: "The last two determinants can be computed as the product of one plus the
eigenvalues ... which is numerically stable."

L'implementiamo da subito, non "dopo se serve": a n=30, k=151 serve davvero.
La verifica di equivalenza A.13 == A.14 e' nel test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigvalsh, solve_triangular
from scipy.special import multigammaln

from src.bvar.dummies import minnesota_omega_diag, minnesota_prior_mean
from src.bvar.spec import BVARSpec, Hyper


# ─── 1. Il prior ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class NIWPrior:
    """
    Il prior (3)-(4) di GLP, nella parametrizzazione di A.13.

        Sigma ~ IW(Psi, dof)          Psi = diag(psi)
        vec(B) | Sigma ~ N(vec(b), Sigma (x) Omega)

    Attributes
    ----------
    b : (k, n)          medie a priori — eq. (2), dal Blocco 2
    omega_diag : (k,)   diagonale di Omega — eq. (3), dal Blocco 2.
                        Omega e' diagonale, quindi basta il vettore.
    psi : (n,)          diagonale della scala IW
    dof : int           n + 2 (Blocco 1)
    """

    b: np.ndarray
    omega_diag: np.ndarray
    psi: np.ndarray
    dof: int

    @property
    def n(self) -> int:
        return int(self.psi.shape[0])

    @property
    def k(self) -> int:
        return int(self.omega_diag.shape[0])


def build_prior(spec: BVARSpec, hyper: Hyper, *, const_var: float | None = None) -> NIWPrior:
    """
    Il prior NIW a partire dallo spec e da un punto negli iperparametri.

    `const_var` e' la varianza a priori sulla costante: `minnesota_omega_diag`
    la mette a `inf` (la verita' matematica del prior piatto), ma A.13/A.14
    vogliono un numero FINITO — un infinito in una matrice non e' calcolabile.
    Default: 1/eps^2 con l'eps del Blocco 4, cioe' 1e8, che rispetto a dati in
    log-livelli dell'ordine di 10 e' gia' piattissimo.

    ATTENZIONE — LA ML *DIPENDE* DA const_var, ED E' CORRETTO COSI'
    ---------------------------------------------------------------
    Misurato: log ML cala esattamente di n/2 * log(const_var).  Non e' un bug,
    e' il PARADOSSO DI BARTLETT (o di Lindley): con un prior che si appiattisce
    la marginal likelihood tende a zero, perche' la massa a priori si spalma su
    un supporto sempre piu' largo.  Con un prior davvero improprio sulla
    costante la ML non sarebbe nemmeno definita.

    PERCHE' NON CI DANNEGGIA: quel termine e' una COSTANTE ADDITIVA che non
    dipende da (lambda, mu, psi).  Nel rapporto di accettazione del Metropolis
    si cancella, quindi il posterior degli iperparametri e' INALTERATO.
    Verificato numericamente: la differenza di log ML fra due valori di
    iperparametri e' identica a 10 decimali al variare di const_var su sei
    ordini di grandezza (test_gate1 §2).

    Corollario da ricordare: i valori ASSOLUTI di log ML non sono confrontabili
    fra specifiche con const_var diversi, e non vanno usati per confronto fra
    modelli senza fissare quel valore.  Le DIFFERENZE a const_var fisso si'.
    """
    from src.bvar.dummies import DEFAULT_CONST_EPS

    if const_var is None:
        const_var = 1.0 / DEFAULT_CONST_EPS ** 2
    return NIWPrior(
        b=minnesota_prior_mean(spec),
        omega_diag=minnesota_omega_diag(spec, hyper, const_var=const_var),
        psi=np.asarray(hyper.psi, dtype=float),
        dof=spec.n + 2,
    )


# ─── 2. Il posterior (A.8)-(A.9) ──────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class NIWPosterior:
    """
    Il posterior, nella stessa famiglia del prior: e' tutto il punto della
    coniugatezza (Blocco 1).

        Sigma | Y ~ IW(psi_bar, dof_bar)
        vec(B) | Sigma, Y ~ N(vec(b_bar), Sigma (x) omega_bar)
    """

    b_bar: np.ndarray          # (k, n)
    omega_bar: np.ndarray      # (k, k)
    psi_bar: np.ndarray        # (n, n)   NON piu' diagonale: il posterior
    dof_bar: int               #          impara le correlazioni fra residui
    resid: np.ndarray          # (T, n)   residui a posteriori, per diagnostica
    #: Radice ESATTA di `omega_bar`: L_om con ``L_om @ L_om.T == omega_bar``,
    #: calcolata in `niw_posterior` dalla Cholesky di `prec` riequilibrata.
    #: Vedi la sezione "LA RADICE DI OMEGA_BAR" nella docstring di
    #: `niw_posterior` per perche' non si passa piu' da `robust_sqrt` qui.
    #: `None` solo se costruito a mano (test) o se la Cholesky e' fallita anche
    #: dopo il riequilibrio: allora `draw` ricade su `robust_sqrt`.
    l_omega: np.ndarray | None = None   # (k, k)


#: Soglia RELATIVA sotto la quale un autovalore si considera nullo in
#: `robust_sqrt`.  Vedi la sua docstring per perche' relativa e non assoluta
#: come negli autori.
_EIG_RTOL = 1e-12


def robust_sqrt(S: np.ndarray, *, rtol: float = _EIG_RTOL) -> np.ndarray:
    """
    Radice quadrata simmetrica: ritorna C con ``C @ C.T == S``.

    STESSO CONTRATTO di ``cholesky(S, lower=True)``, che sostituisce: chi la
    usa non cambia una riga.  Ma non puo' fallire, ed e' tutta la differenza.

    PERCHE' NON LA CHOLESKY
    -----------------------
    La matrice che serve qui — ``prec = X'X + Omega^-1`` e le sue derivate —
    e' definita positiva PER COSTRUZIONE: ``X'X`` e' semidefinita e
    ``Omega^-1`` e' diagonale strettamente positiva.  In doppia precisione
    pero' non lo resta: l'header di questo file segnala che Omega ha elementi
    da ~1e-4 a ~1e8, dodici ordini di grandezza, e le osservazioni del 2020
    gonfiano ``X'X`` di altri due.  Il condizionamento sfonda il doppio e la
    Cholesky si arresta a meta' matrice:

        LinAlgError: 122-th leading minor of the array is not positive definite

    Non e' un caso di scuola: e' cio' che ha ucciso i due blocchi BVAR
    2020-07-31 e 2020-10-30 della passata, i primi due il cui campione di
    stima contiene il crollo e il rimbalzo del PIL Covid.

    GLI AUTORI CI SONO ARRIVATI PRIMA.  In `logMLVAR_formcmc.m` (pacchetto di
    replica CGLMS, `functionsGMS/fromGLP/`) le due righe con `chol` ci sono
    ancora, COMMENTATE, e sotto c'e' `cholred` — eig con gli autovalori
    tagliati.  Hanno sbattuto sullo stesso muro e hanno cambiato strada.

    NON SI PERDE NIENTE IN CORRETTEZZA.  La radice di una covarianza non e'
    unica: per estrarre da N(0,S) serve una qualsiasi C con ``C C' = S``.  La
    Cholesky ne da' una triangolare, questa una simmetrica; dove S e' sana
    sono entrambe esatte e le estrazioni hanno la stessa distribuzione.  La
    Cholesky e' piu' VELOCE, non piu' CORRETTA.

    UNA DEVIAZIONE DAGLI AUTORI, VOLUTA: LA SOGLIA E' RELATIVA
    ----------------------------------------------------------
    `cholred` taglia in ASSOLUTO (``dd(dd<1e-12)=0``).  Qui no, e la ragione
    e' che le nostre matrici non vivono tutte sulla stessa scala.
    ``omega_bar`` e' l'inversa di ``prec``, che con i numeri veri e' enorme:
    i suoi autovalori possono stare legittimamente sotto 1e-12 senza essere
    affatto nulli.  Una soglia assoluta li azzererebbe e butterebbe via
    incertezza a posteriori VERA, in silenzio — un errore peggiore di quello
    che stiamo riparando, perche' non si vede.  Si taglia quindi rispetto
    all'autovalore piu' grande della matrice stessa, che e' anche cio' che fa
    `numpy.linalg.pinv` col suo `rcond`.

    Il taglio resta un'approssimazione e va dichiarato: le direzioni quasi
    nulle vengono scartate invece di far fermare il conto.  Su una matrice
    ben condizionata non ne viene tagliata nessuna, ed e' cio' che verifica
    `test_niw_robust.py`.
    """
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)                    # simmetria numerica, poi eigh
    d, V = np.linalg.eigh(S)
    if d.size:
        # `eigh` ordina crescente: l'ultimo e' il massimo.  Un massimo <= 0
        # vuol dire matrice nulla o tutta negativa: si azzera tutto e si
        # lascia che sia il chiamante a trovarsi una C nulla, invece di
        # inventare una scala di riferimento.
        d = np.where(d < rtol * max(d[-1], 0.0), 0.0, d)
    return V * np.sqrt(d)                  # == V @ diag(sqrt(d))


def _omega_bar_fallback(prec: np.ndarray, rhs: np.ndarray, k: int):
    """
    La vecchia via: LU sulla `prec` grezza, `omega_bar` esplicita, radice
    lasciata a `robust_sqrt` (segnalata con ``l_omega=None``).  Resta come rete
    per il caso in cui nemmeno il riequilibrio renda fattorizzabile la matrice.
    """
    b_bar = np.linalg.solve(prec, rhs)
    omega_bar = np.linalg.solve(prec, np.eye(k))
    omega_bar = 0.5 * (omega_bar + omega_bar.T)
    return None, omega_bar, b_bar


def _omega_bar_factor(prec: np.ndarray, rhs: np.ndarray, k: int):
    """
    Da `prec = X'X + Omega^-1` a ``(L_om, omega_bar, b_bar)`` in un colpo solo.

    IL PROBLEMA ERA LA SCALA, NON LA MATRICE
    ----------------------------------------
    `prec` e' definita positiva per costruzione (``X'X`` semidefinita piu' una
    diagonale strettamente positiva) e smetteva di esserlo solo in doppia
    precisione.  La causa e' documentata nell'header: ``Omega`` spazia da ~1e-4
    a ~1e8, dodici ordini di grandezza, quindi la DIAGONALE di `prec` e' gia'
    sbilanciata di suo, prima ancora che i dati ci mettano del loro.  E' cattivo
    condizionamento **di unita' di misura**, il caso da manuale per il
    riequilibrio di Jacobi:

        D = diag(sqrt(diag(prec)))       P = D^-1 prec D^-1

    `P` ha diagonale unitaria, e il suo condizionamento riflette la correlazione
    vera fra regressori invece del capriccio delle unita'.  Su `P` la Cholesky
    passa dove su `prec` si arrestava.  **Non e' un'approssimazione**: e'
    l'identita' ``prec = D P D`` riscritta, esatta in aritmetica esatta e molto
    piu' stabile in quella finita.

    LA RADICE DI OMEGA_BAR, ESATTA E SENZA AUTOVALORI
    -------------------------------------------------
    Serve `L_om` con ``L_om L_om' = omega_bar = prec^-1``.  Da ``P = L L'``:

        prec^-1 = D^-1 P^-1 D^-1 = D^-1 L^-T L^-1 D^-1 = (D^-1 L^-T)(D^-1 L^-T)'

    quindi ``L_om = D^-1 L^-T``, un'inversa triangolare per sostituzione.

    PERCHE' CONTA.  La via precedente formava `omega_bar` esplicita (un'inversa
    k x k) e poi ne prendeva `robust_sqrt` (una `eigh` k x k).  Sono fuori dal
    ciclo interno di `draw`, ma NON fuori da quello che conta: `core.step`
    chiama `niw_posterior` + ``draw(n_draws=1)`` **a ogni iterazione Gibbs**,
    quindi le due fattorizzazioni k x k si pagano per iterazione.  Misurato,
    a k=201: 161 ms contro 1,65 ms, ~98x.  Ma il guadagno vero e' l'altro:
    il taglio relativo degli autovalori spariva.  Gli autovalori di
    `omega_bar` sono legittimamente sparsi su ~12 ordini — la varianza a
    posteriori dei coefficienti DEVE differire cosi' fra regressori — e col
    taglio a `rtol=1e-12` le direzioni piu' piccole finivano azzerate proprio
    dove ``X'X`` e' gonfiata (2020).  Una `Sigma` cosi' mutilata mandava NaN a
    valle: e' il ``il pannello contiene NaN`` che uccideva il blocco
    2020-07-31 attraverso il simulation smoother dell'L-BVAR.  La via
    triangolare non guarda gli autovalori, quindi non ha niente da troncare:
    il problema non si ripresenta per costruzione.

    `robust_sqrt` NON sparisce: resta su `psi_bar` e su `Sigma` dentro `draw`,
    dove le matrici sono n x n e ben educate, e resta come rete qui sotto.
    """
    diag = np.diag(prec)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
        return _omega_bar_fallback(prec, rhs, k)

    inv_d = 1.0 / np.sqrt(diag)
    P = prec * np.outer(inv_d, inv_d)
    P = 0.5 * (P + P.T)
    try:
        L = cholesky(P, lower=True)
    except LinAlgError:
        # Il riequilibrio non e' bastato: la matrice e' malata davvero, non
        # solo mal scalata.  Si torna alla via robusta.
        return _omega_bar_fallback(prec, rhs, k)

    L_inv = solve_triangular(L, np.eye(k), lower=True)
    l_omega = inv_d[:, None] * L_inv.T                   # D^-1 L^-T
    omega_bar = l_omega @ l_omega.T
    omega_bar = 0.5 * (omega_bar + omega_bar.T)
    b_bar = l_omega @ (l_omega.T @ rhs)                  # prec^-1 rhs
    return l_omega, omega_bar, b_bar


def niw_posterior(Y: np.ndarray, X: np.ndarray, prior: NIWPrior) -> NIWPosterior:
    """
    Il posterior coniugato — GLP (A.8)-(A.9).

        B_hat    = (x'x + Omega^-1)^-1 (x'y + Omega^-1 b)
        eps_hat  = y - x B_hat
        psi_bar  = Psi + eps'eps + (B_hat - b)' Omega^-1 (B_hat - b)
        dof_bar  = T + dof            (T = righe di y, dummy incluse)

    Parameters
    ----------
    Y : (T, n)      variabile dipendente, GIA' comprensiva delle eventuali
                    righe di dummy (soc) impilate sopra.
    X : (T, k)      regressori, stesso trattamento.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Y e X hanno righe diverse: {Y.shape[0]} vs {X.shape[0]}")
    if X.shape[1] != prior.k:
        raise ValueError(f"X ha {X.shape[1]} colonne ma il prior ne attende {prior.k}")

    om_inv = 1.0 / prior.omega_diag                      # Omega e' diagonale
    xtx = X.T @ X
    prec = xtx + np.diag(om_inv)                         # (k, k)
    rhs = X.T @ Y + om_inv[:, None] * prior.b

    l_omega, omega_bar, b_bar = _omega_bar_factor(prec, rhs, prior.k)

    resid = Y - X @ b_bar
    dev = b_bar - prior.b
    psi_bar = np.diag(prior.psi) + resid.T @ resid + dev.T @ (om_inv[:, None] * dev)
    psi_bar = 0.5 * (psi_bar + psi_bar.T)

    return NIWPosterior(
        b_bar=b_bar,
        omega_bar=omega_bar,
        psi_bar=psi_bar,
        dof_bar=int(Y.shape[0]) + prior.dof,
        resid=resid,
        l_omega=l_omega,
    )


def draw(post: NIWPosterior, rng: np.random.Generator, n_draws: int = 1):
    """
    Estrae (B, Sigma) dal posterior.  Nessuna catena: e' esatto.

        Sigma ~ IW(psi_bar, dof_bar)
        B | Sigma = b_bar + L_Omega Z L_Sigma'      Z ~ N(0,1) di forma (k, n)

    dove L_Omega L_Omega' = omega_bar e L_Sigma L_Sigma' = Sigma.  Si verifica
    che vec(B - b_bar) abbia covarianza Sigma (x) omega_bar, come vuole (A.9).

    Returns
    -------
    (B, Sigma) : (S, k, n) e (S, n, n)
    """
    n, k = post.psi_bar.shape[0], post.b_bar.shape[0]
    # La radice di `omega_bar` arriva GIA' FATTA da `niw_posterior`, esatta e
    # triangolare (vedi `_omega_bar_factor`): qui non si fattorizza piu' nulla
    # di k x k, ne' una volta ne' per draw.  `robust_sqrt` resta il ripiego se
    # il posterior e' stato costruito a mano o se la Cholesky e' fallita.
    L_om = post.l_omega if post.l_omega is not None else robust_sqrt(post.omega_bar)
    # IW: Sigma = (W)^-1 con W ~ Wishart(dof, psi_bar^-1).  Si estrae via
    # Bartlett sulla scala inversa, che e' numericamente piu' stabile.
    c_psi = robust_sqrt(post.psi_bar)

    B = np.empty((n_draws, k, n))
    S = np.empty((n_draws, n, n))
    for s in range(n_draws):
        # Wishart(dof_bar, I) via Bartlett
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = np.sqrt(rng.chisquare(post.dof_bar - i))
            if i:
                A[i, :i] = rng.standard_normal(i)
        # Sigma = psi_bar^{1/2} (A A')^{-1} psi_bar^{T/2}
        Linv = np.linalg.solve(A, c_psi.T)               # A^-1 c_psi'
        Sigma = Linv.T @ Linv
        Sigma = 0.5 * (Sigma + Sigma.T)
        S[s] = Sigma
        L_sig = robust_sqrt(Sigma)
        Z = rng.standard_normal((k, n))
        B[s] = post.b_bar + L_om @ Z @ L_sig.T
    return B, S


# ─── 3. La marginal likelihood (A.13) e (A.14) ────────────────────────────────

def log_ml(Y: np.ndarray, X: np.ndarray, prior: NIWPrior, *, stable: bool = True,
           xtx: np.ndarray | None = None, xty: np.ndarray | None = None) -> float:
    """
    Log della marginal likelihood p(Y) — GLP (A.13), per default nella forma
    numericamente stabile (A.14).

    E' la densita' dei dati DOPO aver integrato via B e Sigma: la funzione che
    il Metropolis del Blocco 5 valutera' per ogni proposta di iperparametri.

    Parameters
    ----------
    stable : bool
        True  -> (A.14), i determinanti combinati a coppie.  DEFAULT.
        False -> (A.13) letterale.  Serve solo al test di equivalenza: su
                 sistemi grandi va in overflow, ed e' esattamente il motivo per
                 cui GLP scrive A.2.
    xtx, xty : np.ndarray | None
        `X'X` e `X'Y` gia' calcolati.  Sono gli UNICI due oggetti di questa
        funzione che NON dipendono dagli iperparametri: passarli evita di
        rifarli a ogni valutazione del Metropolis.  Vedi la nota in `hyper.py`
        (`HyperTarget`) e `log_ml_with_dummies`.  Se `None` si calcolano qui.

        ATTENZIONE: il risultato e' algebricamente identico ma NON bit a bit.
        Sommare `Xd'Xd + X'X` accumula in un ordine diverso da `X_plus'X_plus`:
        MISURATO sul profilo `l`, lo scarto e' ~2e-4 in valore assoluto su una
        log-posterior di ~2.8e4, cioe' ~6e-9 in relativo.  Irrilevante per il
        rapporto di accettazione (le differenze in gioco sono di ordine 1-100
        nat) e per il modo (riverificato: 0 direzioni che migliorano), ma va
        saputo — chi cercasse la riproducibilita' bit a bit con una versione
        precedente non la trovera'.

        NB: `eps = Y - X b_hat` resta calcolato per via diretta e NON dai
        momenti.  La forma `Y'Y - 2 b' X'Y + b' X'X b` sarebbe algebricamente
        equivalente ma richiede il prodotto `(n,k) @ (k,k)`, che su questa
        macchina cade nel percorso BLAS patologico documentato in `simsmoother`
        — costerebbe piu' di quanto fa risparmiare.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, n = Y.shape
    k = prior.k
    d = prior.dof

    om = prior.omega_diag
    om_inv = 1.0 / om
    psi = prior.psi

    xtx = X.T @ X if xtx is None else xtx
    prec = xtx + np.diag(om_inv)
    # Solve LU e non Cholesky, per la stessa ragione di `niw_posterior`: qui
    # `prec` e' la stessa matrice, e la stessa perdita di definita positivita'
    # fermerebbe anche il calcolo della verosimiglianza marginale — cioe' il
    # Metropolis sugli iperparametri, non solo l'estrazione di (B, Sigma).
    b_hat = np.linalg.solve(prec, (X.T @ Y if xty is None else xty)
                            + om_inv[:, None] * prior.b)
    eps = Y - X @ b_hat
    dev = b_hat - prior.b
    S = eps.T @ eps + dev.T @ (om_inv[:, None] * dev)          # (n, n)
    S = 0.5 * (S + S.T)

    # il termine di normalizzazione, comune alle due forme
    out = (
        -0.5 * n * T * np.log(np.pi)
        + multigammaln(0.5 * (T + d), n)
        - multigammaln(0.5 * d, n)
    )

    if stable:
        # (A.14).  D_Omega D_Omega' = Omega  ->  D_Omega = diag(sqrt(om))
        #          D_Psi  D_Psi'  = Psi^-1   ->  D_Psi  = diag(1/sqrt(psi))
        sq_om = np.sqrt(om)
        M1 = (sq_om[:, None] * xtx) * sq_om[None, :]           # D' x'x D
        ev1 = eigvalsh(M1)
        ev1 = np.clip(ev1, 0.0, None)
        out += -0.5 * n * np.sum(np.log1p(ev1))

        inv_sq_psi = 1.0 / np.sqrt(psi)
        M2 = (inv_sq_psi[:, None] * S) * inv_sq_psi[None, :]   # D' S D
        ev2 = eigvalsh(M2)
        ev2 = np.clip(ev2, 0.0, None)
        out += -0.5 * T * np.sum(np.log(psi))
        out += -0.5 * (T + d) * np.sum(np.log1p(ev2))
    else:
        # (A.13) letterale — instabile su sistemi grandi, tenuta per il test
        out += -0.5 * n * np.sum(np.log(om))
        out += 0.5 * d * np.sum(np.log(psi))
        out += -0.5 * n * np.linalg.slogdet(prec)[1]
        out += -0.5 * (T + d) * np.linalg.slogdet(np.diag(psi) + S)[1]

    return float(out)


def log_ml_with_dummies(
    Y: np.ndarray,
    X: np.ndarray,
    Yd: np.ndarray,
    Xd: np.ndarray,
    prior: NIWPrior,
    *,
    stable: bool = True,
    moments: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """
    La ML quando una parte dei prior e' imposta via dummy observations — GLP A.3:

        ML = p(Y+) / p(Y*)

    dove `p(.)` e' la funzione (A.13)/(A.14), `Y*` sono le sole dummy e `Y+` e'
    l'insieme esteso (dummy + dati veri).

    L'INTUIZIONE.  Le dummy sono pseudo-osservazioni: `p(Y+)` e' la densita' di
    "dummy + dati" sotto il prior analitico, `p(Y*)` quella delle sole dummy.
    Il rapporto toglie di mezzo il contributo delle dummy e lascia la densita'
    dei DATI VERI sotto il prior che le dummy inducono.  E' la stessa logica di
    p(A,B)/p(A) = p(B|A).

    Nel nostro caso `Yd, Xd` e' il solo blocco sum-of-coefficients (vedi la nota
    architetturale in cima al modulo): Minnesota, covarianza e costante sono
    gia' dentro `prior` in forma analitica.

    LA CACHE DEI MOMENTI
    --------------------
    `moments = (X'X, X'Y)` del blocco dei DATI VERI, che non dipende dagli
    iperparametri.  I momenti impilati si ottengono per semplice somma,

        X_plus' X_plus = Xd'Xd + X'X          (Xd ha solo n righe: costa nulla)
        X_plus' Y_plus = Xd'Yd + X'Y

    perche' impilare righe somma i prodotti incrociati.  E' un'identita' esatta,
    non un'approssimazione.  Misurato sul profilo `l` (n=37, p=17, k=630):
    `X'X` da solo e' il 26% di una valutazione, e in `find_mode` la stessa
    matrice veniva ricalcolata a ogni singola valutazione sullo stesso pannello.
    """
    Y_plus = np.vstack([np.asarray(Yd, dtype=float), np.asarray(Y, dtype=float)])
    X_plus = np.vstack([np.asarray(Xd, dtype=float), np.asarray(X, dtype=float)])
    xtx_p = xty_p = None
    if moments is not None:
        XtX, XtY = moments
        Xd = np.asarray(Xd, dtype=float)
        xtx_p = Xd.T @ Xd + XtX
        xty_p = Xd.T @ np.asarray(Yd, dtype=float) + XtY
    return (log_ml(Y_plus, X_plus, prior, stable=stable, xtx=xtx_p, xty=xty_p)
            - log_ml(Yd, Xd, prior, stable=stable))


__all__ = [
    "NIWPrior",
    "NIWPosterior",
    "build_prior",
    "niw_posterior",
    "draw",
    "robust_sqrt",
    "log_ml",
    "log_ml_with_dummies",
]
