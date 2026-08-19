"""
core/bvar/simulate.py

IL DGP per i recovery test: si simula da parametri NOTI, si stima, si verifica
di ritrovarli dentro l'intervallo di credibilita'.

E' il gate che il documento di contesto impone a ogni passo ("Recovery test at
every gate before moving on").  La domanda a cui risponde e' diversa da quella
dei test unitari: quelli chiedono "il codice fa quello che dice la matematica?",
questo chiede "la matematica recupera la verita'?".

DIMENSIONAMENTO
---------------
Per il test di CORRETTEZZA del core si usa un VAR piccolo (n=3..5) con T
generoso, dove il recovery e' netto e la run dura secondi.  Il regime vero
large-n (n=30 per Q/C/B, n=37 e p=17 per L) si stressa ai gate dei modelli
specifici, dove e' il punto; qui servirebbe solo a rendere il test lento e la
diagnosi ambigua.
"""

from __future__ import annotations

import numpy as np


def stable_var_coefficients(
    n: int,
    p: int,
    rng: np.random.Generator,
    *,
    d_centre: np.ndarray | None = None,
    persistence: float = 0.9,
    cross: float = 0.05,
    max_eig: float = 0.98,
) -> np.ndarray:
    """
    Coefficienti VAR STABILI, con struttura simile a quella che il prior si
    aspetta (diagonale dominante al primo lag, spillover piccoli).

    Costruisce A_1 attorno a `persistence * diag(d_centre)` piu' rumore fuori
    diagonale, e lag successivi via via piu' piccoli; poi RISCALA finche' il
    raggio spettrale della companion form scende sotto `max_eig`.  Il rescaling
    e' necessario perche' un VAR simulato a caso e' quasi sempre esplosivo, e un
    campione esplosivo renderebbe il recovery test privo di significato.

    Returns
    -------
    A : (p, n, n)     A[s-1] e' la matrice del lag s
    """
    d = np.ones(n) if d_centre is None else np.asarray(d_centre, dtype=float)
    A = np.zeros((p, n, n))
    A[0] = persistence * np.diag(d) + cross * rng.normal(size=(n, n))
    for s in range(1, p):
        A[s] = (cross / (s + 1)) * rng.normal(size=(n, n))

    for _ in range(200):
        comp = np.zeros((n * p, n * p))
        comp[:n] = np.hstack([A[s] for s in range(p)])
        if p > 1:
            comp[n:, :-n] = np.eye(n * (p - 1))
        if np.abs(np.linalg.eigvals(comp)).max() < max_eig:
            break
        A *= 0.95
    return A


def simulate_var(
    n: int,
    p: int,
    T: int,
    rng: np.random.Generator,
    *,
    A: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    const: np.ndarray | None = None,
    burn: int = 200,
    d_centre: np.ndarray | None = None,
    sigma_scale: float = 0.05,
) -> dict:
    """
    Simula un VAR(p) gaussiano e restituisce il pannello piu' i parametri veri.

    Returns
    -------
    dict con
        'panel' (T, n)   il pannello simulato
        'A'     (p,n,n)  le matrici vere
        'Sigma' (n,n)    la covarianza vera
        'const' (n,)     la costante vera
        'B'     (k,n)    gli stessi coefficienti nel layout del core
                         (lag-major, costante in fondo) — e' la forma con cui
                         confrontare i draw.
    """
    if A is None:
        A = stable_var_coefficients(n, p, rng, d_centre=d_centre)
    if Sigma is None:
        L = np.tril(rng.normal(size=(n, n)) * 0.3)
        np.fill_diagonal(L, np.abs(np.diag(L)) + 1.0)
        Sigma = sigma_scale ** 2 * (L @ L.T) / n
    if const is None:
        const = np.zeros(n)

    L = np.linalg.cholesky(Sigma)
    total = T + burn
    y = np.zeros((total + p, n))
    for t in range(p, total + p):
        m = const.copy()
        for s in range(p):
            m += A[s] @ y[t - 1 - s]
        y[t] = m + L @ rng.standard_normal(n)
    panel = y[p + burn:]

    B = np.zeros((n * p + 1, n))
    for s in range(p):
        B[s * n:(s + 1) * n, :] = A[s].T          # righe = regressori
    B[-1, :] = const
    return {"panel": panel, "A": A, "Sigma": Sigma, "const": const, "B": B}


__all__ = ["stable_var_coefficients", "simulate_var"]
