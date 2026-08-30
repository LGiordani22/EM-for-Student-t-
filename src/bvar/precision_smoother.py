"""
src/bvar/precision_smoother.py

Il campionatore del pannello latente dell'L-BVAR scritto sulla MATRICE DI
PRECISIONE, invece che sul cammino ausiliario di Durbin-Koopman.

Non e' un secondo modello e non e' un'approssimazione: e' la STESSA
condizionale `p(x_{1:T} | y)` di `simsmoother.simulation_smoother`, ottenuta
per una strada che non puo' perdere cifre.  Serve dove il DK non regge, e
`lbvar.py` lo chiama solo li'.


================================================================================
PERCHE' SERVE: DOVE IL DK PERDE LE CIFRE
================================================================================
Durbin-Koopman ottiene l'estrazione condizionale come

    alpha~  =  alpha+  +  E[alpha | y - y+]

dove `alpha+` e' un cammino simulato SENZA condizionare, cioe' dalla PRIOR.
L'identita' e' esatta in aritmetica reale.  In doppia precisione no: l'errore
del risultato e' proporzionale alla scala dei due addendi, non a quella della
loro somma.

Nell'L-BVAR i due addendi hanno scale diversissime.  La companion e' un VAR(17)
su 37 serie e la posterior, a certi vintage, sta SOPRA il cerchio unitario:
allora `alpha+` cresce come rho^T, con T dell'ordine dei 500 mesi.  Il
risultato invece resta alla scala del dato, perche' R = 1e-12 inchioda le celle
osservate al valore osservato.  Il rapporto fra i due e' quanto si perde:

    errore relativo del DK  ~  ns * eps * rho^T / (scala del dato)

e la companion non e' normale, quindi il transitorio ||A^t|| supera rho(A)^t di
parecchi ordini prima di seguirlo.  Le cifre finiscono, e la guardia di
`simulation_smoother` lo dice.

QUI NON C'E' NIENTE DA CANCELLARE.  Le voci di Omega (sotto) sono combinazioni
di A_j e Sigma^-1: O(||A||^2), senza nessun rho^T dentro.  Il costo e la
precisione non dipendono dal raggio spettrale.


================================================================================
IL MODELLO, SCRITTO PER QUEL CHE E'
================================================================================
Lo stato-spazio che `lbvar.build_state_space` costruisce e' degenere nel modo
piu' comodo possibile: lo stato E' il pannello mensile (Z = [I 0]), le prime p
righe sono FISSE (P0 = 0, punto 3 dell'header di `lbvar.py`) e la misura e'
l'identita' con R = 1e-12, cioe' zero a meno di 1e-6 in deviazione standard.
Togliendo la companion, che li' e' solo contabilita', resta un VAR(p):

    x_t = c + sum_{j=1..p} A_j x_{t-j} + eps_t        eps_t ~ N(0, Sigma)
    x_0, ..., x_{1-p}  =  le p righe di testa, deterministiche
    x_{t,i} = y_{t,i}  dove il dato c'e'

Impilando X = (x_1', ..., x_T')' si ha  H X = k + E,  E ~ N(0, I_T (x) Sigma),
con H triangolare a blocchi (I sulla diagonale, -A_j sulla j-esima
sottodiagonale) e k che raccoglie la costante e il contributo delle righe di
testa.  Quindi

    precisione   Omega = H' (I (x) Sigma^-1) H
    media        Omega mu = H' (I (x) Sigma^-1) k =: b

e condizionare sulle celle osservate e' una PARTIZIONE, non un filtro:

    x_M | x_O  ~  N(m, Omega_MM^-1),   Omega_MM m = b_M - Omega_MO x_O

Riferimenti: Rue (2001), Chan & Jeliazkov (2009), McCausland, Miller &
Pelletier (2011) — il campionamento gaussiano su precisione bandata.


================================================================================
DUE PROPRIETA' CHE RENDONO LA COSA ECONOMICA
================================================================================

--- 1. Omega e' BANDATA, e resta bandata dopo la partizione ------------------

Due celle interagiscono solo se distano al piu' p periodi, cioe' al piu'
p*n + (n-1) posizioni nell'indice piatto (t, i).  Cancellare le righe e le
colonne osservate non puo' ALLONTANARE due indici, quindi Omega_MM ha
semiampiezza <= n*(p+1)-1 nell'indice compattato: si fattorizza con
`cholesky_banded`, non con una Cholesky densa da 6000^3.

Omega e' anche quasi Toeplitz a blocchi: Omega[a, a+d] dipende da `a` solo
attraverso il troncamento di coda `e = min(p, T-1-a)`.  Bastano (p+1)^2
blocchi n x n, che `_omega_blocks` calcola una volta sola.

--- 2. LA CODA DI PREVISIONE SI STACCA -------------------------------------

Il modello e' ricorsivo in avanti, quindi le righe dopo l'ultima osservazione
non dicono niente su quelle prima: la congiunta fattorizza in

    p(x_{1:t*} | y)  *  p(x_{t*+1:T} | x_{1:t*})

e il secondo fattore E' la simulazione in avanti del VAR.  Separarli non e'
un'approssimazione, ed e' NECESSARIO: la varianza condizionale della coda
cresce come rho^(2H) con H la lunghezza del tratto cieco, e infilarla dentro
Omega ne farebbe saltare il numero di condizionamento.  Misurato: con
rho = 1.37 e H = 96 la Cholesky bandata fallisce; fuori da Omega quella stessa
crescita e' semplicemente il ventaglio di previsione, calcolato dove non fa
danno.

E' anche il motivo per cui la condizione del sistema dipende dal PIU' LUNGO
TRATTO CIECO e non da T: dentro i dati i buchi sono di due mesi (le
trimestrali), e li' rho^2 non e' niente.


================================================================================
COSA COSTA, MISURATO
================================================================================
Alla taglia vera dell'L-BVAR (n = 37, p = 17, T = 503, ~6800 celle mancanti):

    simulation_smoother (DK)      ~32 s
    precision_draw                 ~0.26 s

e le celle osservate tornano ESATTE (scarto 0.0), non larghe 1e-7 come con il
nugget: qui non si filtra, si condiziona.

ATTENZIONE.  Il divario di velocita' NON e' un invito a sostituire il DK
ovunque.  Il DK e' l'algoritmo degli autori e dove regge va tenuto: `lbvar.py`
prova quello per primo e scende qui solo quando la guardia di cancellazione
scatta.  Cambiare il default e' una decisione separata, che rifarebbe tutti i
risultati.


================================================================================
CONTRO COSA E' STATO VERIFICATO
================================================================================
`tests/test_precision_smoother.py`, oracolo DENSO costruito a mano (H, Omega e
la condizionale gaussiana esplicite, senza riusare niente di qui):

    media analitica      scarto ~1e-15   (esatta a precisione di macchina)
    matrice Omega        scarto ~1e-16   sull'assemblaggio a blocchi
    covarianza empirica  dentro l'errore Monte Carlo

piu' il confronto con il DK stesso sui vintage dove il DK e' sano.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded


def _omega_blocks(A_list: list[np.ndarray], Sig_inv: np.ndarray,
                  p: int) -> np.ndarray:
    r"""I blocchi distinti di Omega.

    Con G_0 = I e G_j = -A_j,

        Omega[a, b] = sum_t G_{t-a}' Sigma^-1 G_{t-b}

    e ponendo d = b - a, j = t - a, la somma va da max(0, d) a min(p, T-1-a).
    L'estremo superiore e' l'unico posto in cui `a` entra, quindi i blocchi
    distinti sono (p+1)^2:

        M[e, d] = sum_{j=max(0,d)}^{e} G_j' Sigma^-1 G_{j-d}

    Il bordo di SINISTRA non ha correzione (G_m = 0 per m < 0 la assorbe gia');
    quello di DESTRA si', ed e' `e`.

    Returns
    -------
    (p+1, p+1, n, n)
    """
    n = Sig_inv.shape[0]
    G = [np.eye(n)] + [-np.asarray(A, dtype=float) for A in A_list]
    SG = [Sig_inv @ g for g in G]
    M = np.zeros((p + 1, p + 1, n, n))
    for e in range(p + 1):
        for d in range(p + 1):
            acc = np.zeros((n, n))
            for j in range(max(0, d), e + 1):
                if 0 <= j - d <= p:
                    acc += G[j].T @ SG[j - d]
            M[e, d] = acc
    return M


def _apply_H(v: np.ndarray, A_list: list[np.ndarray]) -> np.ndarray:
    """(H v)_t = v_t - sum_j A_j v_{t-j}, con v_{<=0} = 0."""
    out = v.copy()
    for j, A in enumerate(A_list, start=1):
        if j < len(v):
            out[j:] -= v[:-j] @ A.T
    return out


def _apply_Ht(w: np.ndarray, A_list: list[np.ndarray]) -> np.ndarray:
    """(H' w)_t = w_t - sum_j A_j' w_{t+j}."""
    out = w.copy()
    for j, A in enumerate(A_list, start=1):
        if j < len(w):
            out[:-j] -= w[j:] @ A
    return out


def _chol(M: np.ndarray) -> np.ndarray:
    """Cholesky con ripiego spettrale: Sigma arriva da una NIW ed e' definita
    positiva, ma un'estrazione al limite non deve far cadere il ripiego."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        lam, V = np.linalg.eigh(0.5 * (M + M.T))
        return V * np.sqrt(np.clip(lam, 0.0, None))


def last_informative_row(Y: np.ndarray) -> int:
    """L'ultima riga con almeno un'osservazione; -1 se non ce n'e' nessuna."""
    rows = np.flatnonzero(~np.isnan(np.asarray(Y, dtype=float)).all(axis=1))
    return int(rows[-1]) if rows.size else -1


def precision_draw(B: np.ndarray, Sigma: np.ndarray, Y: np.ndarray,
                   head: np.ndarray, rng: np.random.Generator, *,
                   n: int, p: int, return_full: bool = True) -> np.ndarray:
    """Una estrazione da p(x_{1:T} | y), esatta e senza cancellazione.

    Parameters
    ----------
    B : (n*p+1, n)
        Come nel core: le prime n*p righe danno [A_1 ... A_p] per trasposizione
        (`A_j = B[(j-1)*n : j*n, :].T`), l'ULTIMA riga e' la costante.
    Sigma : (n, n)
    Y : (T, n)
        Il pannello osservato, NaN dove manca.
    head : (p, n)
        Le p righe di testa in ordine CRONOLOGICO: `head[p-1]` e' x_0.
    return_full : bool
        True restituisce (T, n*p) in forma companion, come il DK, cosi' il
        chiamante non deve sapere quale dei due l'ha prodotto.

    Returns
    -------
    (T, n*p) se `return_full`, altrimenti (T, n).
    """
    Y_all = np.asarray(Y, dtype=float)
    T_all = Y_all.shape[0]
    head = np.asarray(head, dtype=float)
    A_list = [np.asarray(B[j * n:(j + 1) * n, :], dtype=float).T
              for j in range(p)]
    c = np.asarray(B[-1, :], dtype=float)

    Sig = np.asarray(Sigma, dtype=float)
    Sig = 0.5 * (Sig + Sig.T)
    Sig_inv = np.linalg.inv(Sig)
    Sig_inv = 0.5 * (Sig_inv + Sig_inv.T)

    # ── la coda di previsione si stacca (punto 2 dell'header)
    t_star = last_informative_row(Y_all)
    T = t_star + 1
    Yc = Y_all[:T]

    if T == 0:
        draw = np.empty((0, n))
    else:
        # k_t = c + sum_{j >= t} A_j x_{t-j}, con le righe di testa
        k = np.tile(c, (T, 1))
        for j, A in enumerate(A_list, start=1):
            for t in range(min(j, T)):
                k[t] = k[t] + A @ head[p + t - j]
        b = _apply_Ht(k @ Sig_inv, A_list)          # H' (I (x) Sigma^-1) k

        miss = np.isnan(Yc).ravel()
        g_miss = np.flatnonzero(miss)
        m = g_miss.size
        x_pad = np.where(np.isnan(Yc), 0.0, Yc)     # x_O al suo posto, 0 altrove

        if m == 0:
            draw = x_pad
        else:
            # Omega x_O per via di H: esatto, O(T p n^2), niente matrici grandi
            om_x = _apply_Ht(_apply_H(x_pad, A_list) @ Sig_inv, A_list).ravel()
            rhs = b.ravel()[g_miss] - om_x[g_miss]

            M = _omega_blocks(A_list, Sig_inv, p)
            bw = min(n * (p + 1) - 1, m - 1)
            ab = np.zeros((bw + 1, m))              # storage bandato SUPERIORE
            ta, ra = np.divmod(g_miss, n)
            e_idx = np.minimum(p, T - 1 - ta)
            for delta in range(bw + 1):
                i0 = np.arange(m - delta)
                j0 = i0 + delta
                d = ta[j0] - ta[i0]
                ok = d <= p
                vals = np.zeros(i0.size)
                if ok.any():
                    vals[ok] = M[e_idx[i0][ok], d[ok], ra[i0][ok], ra[j0][ok]]
                ab[bw - delta, j0] = vals

            U = cholesky_banded(ab, lower=False)    # Omega_MM = U' U
            mean_M = cho_solve_banded((U, False), rhs)
            # Cov = Omega_MM^-1 = U^-1 U^-T, quindi la perturbazione e' U^-1 z
            pert = solve_banded((0, bw), U, rng.standard_normal(m))
            draw = x_pad.copy()
            draw.ravel()[g_miss] = mean_M + pert

    # ── la coda: iterazione della transizione con shock veri
    if T_all > T:
        L_S = _chol(Sig)
        stack = np.vstack([head, draw]) if T else head.copy()
        tail = np.empty((T_all - T, n))
        for s in range(T_all - T):
            hist = stack[-1:-p - 1:-1]              # x_{t-1}, ..., x_{t-p}
            nxt = c + sum(A_list[j] @ hist[j] for j in range(p))
            tail[s] = nxt + L_S @ rng.standard_normal(n)
            stack = np.vstack([stack, tail[s][None, :]])
        draw = np.vstack([draw, tail]) if T else tail

    if not return_full:
        return draw
    stacked = np.vstack([head, draw])               # (p + T_all, n)
    full = np.empty((T_all, n * p))
    for j in range(p):
        full[:, j * n:(j + 1) * n] = stacked[p - j: p - j + T_all]
    return full


__all__ = ["precision_draw", "last_informative_row"]
