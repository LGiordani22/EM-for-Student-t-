r"""
src/em/idio_ar1.py

Asse B: errori idiosincratici AR(1) nello stato.

L'ESTENSIONE
------------
Nel baseline gli errori idiosincratici sono i.i.d. nel tempo:

    y_t = Lambda f_t + eps_t,     eps_{i,t} | w ~ N(0, sigma_i^2 / w_eps)

e vivono nell'EQUAZIONE DI OSSERVAZIONE (matrice R). Qui ciascuna serie
riceve una persistenza propria,

    eps_{i,t} = rho_i eps_{i,t-1} + u^eps_{i,t},
    u^eps_{i,t} | w ~ N(0, sigma_i^2 / w_eps_t),

e l'unico modo di rappresentarla in forma stato-spazio e' spostare eps_t
DENTRO LO STATO. L'osservazione diventa (quasi) deterministica: R_tilde -> 0,
tutta la stocasticita' e' nell'innovazione di stato. E' il device standard per
gli errori di misura autocorrelati.

Persistenza e code pesanti restano ORTOGONALI: rho_i governa *quanto dura* una
deviazione, w_eps_t *quanto puo' essere estrema* l'innovazione fresca. Percio'
`idio_ar1` e `gaussian` sono due flag indipendenti: 4 combinazioni tutte valide.

IL PUNTO DELICATO: LE SERIE TRIMESTRALI
---------------------------------------
Per una trimestrale l'osservazione e' l'aggregato MM della sua controparte
mensile latente. Se la serie latente e' y*_{i,t} = Lambda_i f_t + eps_{i,t},
allora

    y^Q_{i,t} = sum_l c_l (Lambda_i f_{t-l} + eps_{i,t-l})
              = Lambda_i . phi_t  +  sum_l c_l eps_{i,t-l}
                ^ comune AGGREGATO   ^ idio AGGREGATO

cioe' l'idiosincratico va aggregato **esattamente come** la parte comune, con
gli stessi pesi c = {1/3, 2/3, 1, 2/3, 1/3}. Usare un blocco identita' I_M
(come fa la versione attuale della sezione AR(1) del .tex, eq:ar1-check-Lambda)
spalmerebbe la parte comune sui 5 lag e non l'idio: due meta' della stessa
equazione trattate in modo diverso, con la distorsione concentrata proprio sul
target trimestrale. Banbura & Modugno (2014) aggregano entrambe.

Conseguenza sul layout: le trimestrali portano nello stato 5 lag del proprio
eps, le mensili uno solo.

LAYOUT DELLO STATO
------------------
    check_f_t = [ f_tilde_t (5r) | e_t (n_e) ]

dove il blocco idiosincratico e' la concatenazione per serie, nell'ordine delle
colonne del pannello:

    serie i mensile     -> 1 slot :  [ eps_{i,t} ]
    serie i trimestrale -> 5 slot :  [ eps_{i,t}, eps_{i,t-1}, ..., eps_{i,t-4} ]

    n_e = M_m + 5 * M_q

Sul dataset 'final' (M=37, di cui 3 trimestrali, r=4): n_e = 34 + 15 = 49, e lo
stato passa da 5r = 20 a 5r + n_e = 69.

NON si usa un layout uniforme a 5 lag per tutte le serie (che darebbe un blocco
companion identico a quello dei fattori, piu' elegante) perche' costerebbe
5M = 185 componenti invece di 49: il Kalman e' cubico nella dimensione dello
stato, e sarebbero ~20x di costo per rappresentare lag che per le mensili non
servono a nulla.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_WEIGHTS = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])


@dataclass(frozen=True)
class IdioLayout:
    """
    Mappa serie -> slot nel blocco idiosincratico dello stato.

    Attributes
    ----------
    n_e : int
        Dimensione totale del blocco idiosincratico (M_m + 5*M_q).
    start : np.ndarray, shape (M,), int
        `start[i]` = primo slot della serie i DENTRO il blocco idio
        (offset relativo, non assoluto nello stato).
    n_lags : np.ndarray, shape (M,), int
        1 per le mensili, 5 per le trimestrali.
    is_quarterly : np.ndarray, shape (M,), bool
    M : int
    """
    n_e: int
    start: np.ndarray
    n_lags: np.ndarray
    is_quarterly: np.ndarray
    M: int

    def slot(self, i: int) -> int:
        """Slot CORRENTE (eps_{i,t}) della serie i, relativo al blocco idio."""
        return int(self.start[i])

    def slots(self, i: int) -> np.ndarray:
        """Tutti gli slot della serie i (1 per mensile, 5 per trimestrale)."""
        s = int(self.start[i])
        return np.arange(s, s + int(self.n_lags[i]))

    def current_slots(self) -> np.ndarray:
        """Slot correnti di TUTTE le serie: dove vivono gli eps_{i,t}."""
        return self.start.copy()


def build_idio_layout(freq_list: list[str]) -> IdioLayout:
    """Costruisce il layout dal solo `freq_list` (l'ordine e' quello di Y)."""
    M = len(freq_list)
    is_q = np.array([f == "quarterly" for f in freq_list], dtype=bool)
    for i, f in enumerate(freq_list):
        if f not in ("monthly", "quarterly"):
            raise ValueError(
                f"freq_list[{i}] = {f!r}; atteso 'monthly' o 'quarterly'.")
    n_lags = np.where(is_q, 5, 1).astype(int)
    start = np.zeros(M, dtype=int)
    acc = 0
    for i in range(M):
        start[i] = acc
        acc += int(n_lags[i])
    return IdioLayout(n_e=int(acc), start=start, n_lags=n_lags,
                      is_quarterly=is_q, M=M)


# ─── Costruttori delle matrici dello stato aumentato ──────────────────────────

def build_P_idio(rho: np.ndarray, layout: IdioLayout) -> np.ndarray:
    r"""
    Blocco di transizione idiosincratico P (n_e x n_e), diagonale a blocchi.

    Per una MENSILE il blocco e' lo scalare [rho_i].
    Per una TRIMESTRALE e' la companion 5x5 dello stesso AR(1):

        [ rho_i  0  0  0  0 ]      eps_{i,t}   = rho_i eps_{i,t-1} + u
        [   1    0  0  0  0 ]      eps_{i,t-1} = eps_{i,t-1}         (shift)
        [   0    1  0  0  0 ]      ...
        [   0    0  1  0  0 ]
        [   0    0  0  1  0 ]

    Le righe di shift non hanno innovazione: Q_idio e' singolare, esattamente
    come il blocco companion dei fattori.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    if rho.size != layout.M:
        raise ValueError(f"rho ha {rho.size} elementi, attesi M={layout.M}.")
    P = np.zeros((layout.n_e, layout.n_e))
    for i in range(layout.M):
        s = layout.slot(i)
        P[s, s] = rho[i]
        for l in range(1, int(layout.n_lags[i])):
            P[s + l, s + l - 1] = 1.0       # shift dei lag
    return P


def build_Q_idio(sigma2: np.ndarray, layout: IdioLayout,
                 w_eps_t: float | np.ndarray = 1.0) -> np.ndarray:
    r"""
    Covarianza dell'innovazione idiosincratica (n_e x n_e), singolare.

    Solo gli slot CORRENTI ricevono varianza `sigma_i^2 / w_eps_t`; le righe di
    shift hanno varianza zero. E' qui — e solo qui — che il layer dei pesi
    Student-t tocca l'Asse B: `w_eps_t` scala la varianza dell'innovazione
    fresca, senza toccare la struttura AR(1). Accetta uno scalare (peso
    condiviso, come nel baseline) o un vettore (M,) di pesi per-serie.
    """
    sigma2 = np.asarray(sigma2, dtype=float).ravel()
    if sigma2.size != layout.M:
        raise ValueError(f"sigma2 ha {sigma2.size} elementi, attesi M={layout.M}.")
    w = np.asarray(w_eps_t, dtype=float)
    w_vec = np.full(layout.M, float(w)) if w.ndim == 0 else w.ravel()
    if w_vec.size != layout.M:
        raise ValueError(f"w_eps_t ha {w_vec.size} elementi, attesi M={layout.M}.")

    Q = np.zeros((layout.n_e, layout.n_e))
    idx = layout.current_slots()
    Q[idx, idx] = sigma2 / w_vec
    return Q


def build_C_idio(layout: IdioLayout,
                 mm_weights: np.ndarray | None = None) -> np.ndarray:
    r"""
    Blocco di caricamento idiosincratico C (M x n_e): come eps entra in y_t.

    Riga i MENSILE      : 1 sullo slot corrente.
    Riga i TRIMESTRALE  : i pesi MM {1/3, 2/3, 1, 2/3, 1/3} sui suoi 5 slot —
                          la stessa aggregazione applicata alla parte comune.

    E' questo il blocco che sostituisce l'`I_M` della versione attuale del .tex.
    """
    c = MM_WEIGHTS if mm_weights is None else np.asarray(mm_weights, float)
    if c.size != 5:
        raise ValueError(f"mm_weights deve avere 5 elementi, ne ha {c.size}.")
    C = np.zeros((layout.M, layout.n_e))
    for i in range(layout.M):
        sl = layout.slots(i)
        if layout.is_quarterly[i]:
            C[i, sl] = c
        else:
            C[i, sl[0]] = 1.0
    return C


def build_Sigma0_idio(rho: np.ndarray, sigma2: np.ndarray,
                      layout: IdioLayout) -> np.ndarray:
    r"""
    Covarianza iniziale del blocco idiosincratico: la STAZIONARIA dell'AR(1).

    Per la serie i, gamma_i(k) = rho_i^{|k|} sigma_i^2 / (1 - rho_i^2), che per
    le trimestrali riempie anche le covarianze fra i 5 lag dello stesso eps.
    Usare l'identita' (come per il blocco dei fattori) sarebbe incoerente: la
    varianza incondizionata di un AR(1) e' sigma^2/(1-rho^2), non 1, e su serie
    molto persistenti lo scarto e' grande.
    """
    rho = np.asarray(rho, float).ravel()
    sigma2 = np.asarray(sigma2, float).ravel()
    S = np.zeros((layout.n_e, layout.n_e))
    for i in range(layout.M):
        sl = layout.slots(i)
        r_i = float(np.clip(rho[i], -0.999, 0.999))
        g0 = float(sigma2[i]) / (1.0 - r_i ** 2)
        for a, sa in enumerate(sl):
            for b, sb in enumerate(sl):
                S[sa, sb] = g0 * r_i ** abs(a - b)
    return S


# ─── Assemblaggio dello stato aumentato ───────────────────────────────────────

def build_augmented_A(A_tilde: np.ndarray, rho: np.ndarray,
                      layout: IdioLayout) -> np.ndarray:
    """check_A = blkdiag(A_tilde, P): fattori e idio evolvono indipendenti."""
    d_f, n_e = A_tilde.shape[0], layout.n_e
    out = np.zeros((d_f + n_e, d_f + n_e))
    out[:d_f, :d_f] = A_tilde
    out[d_f:, d_f:] = build_P_idio(rho, layout)
    return out


def build_augmented_Q(Q_tilde: np.ndarray, sigma2: np.ndarray,
                      layout: IdioLayout,
                      w_eps_t: float | np.ndarray = 1.0) -> np.ndarray:
    """check_Q = blkdiag(Q_tilde, Q_idio). Entrambi singolari (companion)."""
    d_f, n_e = Q_tilde.shape[0], layout.n_e
    out = np.zeros((d_f + n_e, d_f + n_e))
    out[:d_f, :d_f] = Q_tilde
    out[d_f:, d_f:] = build_Q_idio(sigma2, layout, w_eps_t)
    return out


def build_augmented_Lambda(Lambda_tilde: np.ndarray, layout: IdioLayout,
                           mm_weights: np.ndarray | None = None) -> np.ndarray:
    """check_Lambda = [Lambda_tilde | C]  (M x (5r + n_e))."""
    return np.hstack([np.asarray(Lambda_tilde, float),
                      build_C_idio(layout, mm_weights)])


def build_augmented_Sigma0(Sigma0_f: np.ndarray, rho: np.ndarray,
                           sigma2: np.ndarray, layout: IdioLayout) -> np.ndarray:
    """check_Sigma_0 = blkdiag(Sigma_0 dei fattori, stazionaria dell'idio)."""
    d_f, n_e = Sigma0_f.shape[0], layout.n_e
    out = np.zeros((d_f + n_e, d_f + n_e))
    out[:d_f, :d_f] = Sigma0_f
    out[d_f:, d_f:] = build_Sigma0_idio(rho, sigma2, layout)
    return out


# ─── Estrazione dei momenti idiosincratici dallo smoother ─────────────────────

def extract_idio_moments(f_smooth: np.ndarray, P_smooth: np.ndarray,
                         P_lag: np.ndarray, layout: IdioLayout,
                         d_f: int) -> dict:
    r"""
    Dai momenti dello stato aumentato ai momenti che servono a update_rho e
    update_R, per serie:

        E[eps_{i,t} | Y],  E[eps_{i,t}^2 | Y],  E[eps_{i,t} eps_{i,t-1} | Y].

    Il termine incrociato usa DUE sorgenti, a seconda della serie:

      * TRIMESTRALI: eps_{i,t-1} e' gia' nello stato al tempo t (slot+1), quindi
        E[eps_t eps_{t-1}] si legge dentro P_smooth[t] — nessun bisogno di P_lag.
      * MENSILI: c'e' un solo slot, quindi serve la cross-covarianza fra t e
        t-1, cioe' P_lag[t] (Cov[check_f_t, check_f_{t-1} | Y]).

    Restituisce dict con array (T, M): `E_e`, `E_e2`, `E_e_lag` (= E[e_t e_{t-1}]),
    `E_e2_lag` (= E[e_{t-1}^2]).
    """
    T = f_smooth.shape[0]
    M = layout.M
    E_e = np.zeros((T, M))
    E_e2 = np.zeros((T, M))
    E_e_lag = np.zeros((T, M))
    E_e2_lag = np.zeros((T, M))

    for i in range(M):
        s = d_f + layout.slot(i)
        e_t = f_smooth[:, s]
        v_t = P_smooth[:, s, s]
        E_e[:, i] = e_t
        E_e2[:, i] = v_t + e_t ** 2

        if layout.is_quarterly[i]:
            # eps_{i,t-1} vive nello stesso stato, slot successivo.
            s1 = s + 1
            e_tm1 = f_smooth[:, s1]
            v_tm1 = P_smooth[:, s1, s1]
            cov = P_smooth[:, s, s1]
            E_e_lag[:, i] = cov + e_t * e_tm1
            E_e2_lag[:, i] = v_tm1 + e_tm1 ** 2
        else:
            # Mensile: un solo slot -> serve la cross-covarianza temporale.
            e_tm1 = np.empty(T)
            e_tm1[0] = 0.0
            e_tm1[1:] = f_smooth[:-1, s]
            v_tm1 = np.empty(T)
            v_tm1[0] = 0.0
            v_tm1[1:] = P_smooth[:-1, s, s]
            cov = P_lag[:, s, s]          # Cov[e_t, e_{t-1} | Y]
            E_e_lag[:, i] = cov + e_t * e_tm1
            E_e2_lag[:, i] = v_tm1 + e_tm1 ** 2

    return {"E_e": E_e, "E_e2": E_e2, "E_e_lag": E_e_lag, "E_e2_lag": E_e2_lag}


def idio_innovation_second_moment(mom: dict, rho: np.ndarray) -> np.ndarray:
    r"""
    E[(eps_{i,t} - rho_i eps_{i,t-1})^2 | Y]  =  E[e_t^2] - 2 rho E[e_t e_{t-1}]
                                                 + rho^2 E[e_{t-1}^2].

    E' il secondo momento dell'innovazione FRESCA: quello che entra sia in
    update_R (stima di sigma_i^2) sia nel residuo di Mahalanobis d_eps.
    """
    rho = np.asarray(rho, float).ravel()
    return (mom["E_e2"] - 2.0 * rho[None, :] * mom["E_e_lag"]
            + (rho ** 2)[None, :] * mom["E_e2_lag"])


# ─── Aggiornamenti M-step ─────────────────────────────────────────────────────

def update_rho(mom: dict, w_eps: np.ndarray | None = None,
               rho_max: float = 0.98) -> np.ndarray:
    r"""
    M-step per rho: OLS pesato per serie sui momenti posteriori,

        rho_i = sum_t w_t E[eps_{i,t} eps_{i,t-1}] / sum_t w_t E[eps_{i,t-1}^2].

    Il primo periodo e' escluso (eps_{i,-1} non e' definito). Il risultato e'
    troncato a |rho| <= rho_max per garantire la stazionarieta' del blocco.
    """
    E_lag = mom["E_e_lag"][1:, :]
    E_sq = mom["E_e2_lag"][1:, :]
    if w_eps is None:
        w = np.ones(E_lag.shape[0])[:, None]
    else:
        w = np.asarray(w_eps, float)
        w = w[1:, None] if w.ndim == 1 else w[1:, :]
    num = (w * E_lag).sum(axis=0)
    den = (w * E_sq).sum(axis=0)
    rho = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.clip(rho, -rho_max, rho_max)


def update_sigma2(mom: dict, rho: np.ndarray,
                  w_eps: np.ndarray | None = None) -> np.ndarray:
    r"""
    M-step per sigma_i^2: varianza pesata dell'innovazione FRESCA,

        sigma_i^2 = (1/(T-1)) sum_{t>=1} w_t E[(eps_{i,t} - rho_i eps_{i,t-1})^2].

    Nel baseline (rho = 0) si riduce esattamente alla varianza media di
    E[eps^2] (l'update i.i.d. della scala idiosincratica).
    """
    S = idio_innovation_second_moment(mom, rho)[1:, :]
    if w_eps is None:
        w = np.ones(S.shape[0])[:, None]
    else:
        w = np.asarray(w_eps, float)
        w = w[1:, None] if w.ndim == 1 else w[1:, :]
    return np.maximum((w * S).sum(axis=0) / S.shape[0], 1e-10)
