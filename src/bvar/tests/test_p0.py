"""
src/bvar/tests/test_p0.py

P_0 — LA SCELTA CENTRALIZZATA, CABLATA IN UN TEST.

    python -m src.bvar.tests.test_p0

Il rischio che questo test copre non e' un baco: e' una REGRESSIONE SILENZIOSA
di convenzione.  `state_space.initial_covariance` esiste per avere una sola
risposta alla domanda «che cosa vale P_0 sulla finestra di bordo», ma la
risposta NON e' la stessa nei quattro modelli — e' la stessa FUNZIONE con
default diversi, decisi sulla misura:

    C, Q    "lyapunov"   misurato meglio di 0 (banda del C, RMSE del backcast
                         del Q: le tabelle stanno in `initial_covariance`)
    B       0            immateriale, MISURATO: con `EDGE_LAGS = 24` trimestri
                         P_0 e' lavata via dalle osservazioni (identici a 4
                         decimali, e il Lyapunov NON era degenerato)
    L       0            e' il loro `lbvar.m` r.52.  Qui la scelta PESA
                         (ampiezza della banda +38%), quindi e' l'unica che
                         un domani si potrebbe voler cambiare — e va cambiata
                         APPOSTA, non per effetto collaterale

Da cui il §3, che e' il controllo che conta: B e L NON devono seguire
`P0_DEFAULT`.  Chi domani cambiasse quella costante per il C o per il Q
ribalterebbe anche l'L, e l'unico segnale sarebbe una banda piu' larga del 38%
in mezzo a una passata da ore.  Il §3 lo fa fallire subito.

  §1  `initial_covariance`: le tre vie e il ripiego.
  §2  I default dei quattro modelli, letti dalle firme.
  §3  B e L sono INSENSIBILI a `P0_DEFAULT`.  IL CONTROLLO CENTRALE,
      con controllo negativo (`p0` esplicito DEVE cambiare il risultato).
  §4  Il ripiego dell'L e' 0, non kappa*I — anti-regressione sul -97.74%.
"""

from __future__ import annotations

import inspect
import re

import numpy as np

from src.bvar import bbvar, cbvar, lbvar, qbvar
from src.bvar import state_space as ss_mod
from src.bvar.state_space import P0_DEFAULT, initial_covariance, lyapunov_symm

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _lbvar_B(lag1: np.ndarray, p: int = 1) -> np.ndarray:
    """(B, n, p) per `lbvar.build_state_space`: (n*p + 1, n), costante in fondo."""
    n = lag1.shape[0]
    B = np.zeros((n * p + 1, n))
    B[:n] = lag1.T                                # lag 1, lag-major come loro
    return B


# ─── §1  Le tre vie e il ripiego ──────────────────────────────────────────────

def test_vie() -> bool:
    print("\n§1  initial_covariance: lyapunov / kappa / ripiego")
    ok = True

    # (a) sistema stabile: deve essere ESATTAMENTE lyapunov_symm, non una copia
    #     riscritta che un giorno diverge.
    A = np.diag([0.5, 0.9])
    Q = np.eye(2)
    P, come = initial_covariance(A, Q, kind="lyapunov")
    P_rif, _ = lyapunov_symm(A, Q)
    ok &= _check("stabile -> e' lyapunov_symm, cella per cella",
                 come == "lyapunov" and np.allclose(P, P_rif, atol=0, rtol=0))
    ok &= _check("...e vale 1/(1-phi^2) sulla diagonale",
                 np.allclose(np.diag(P), [1 / 0.75, 1 / 0.19]),
                 f"{np.diag(P).round(4)}")

    # (b) esplosivo con sottospazio stabile: la riga degli autori NON fallisce,
    #     scarta le direzioni con |lambda| > 1.  E' il caso del C e del B.
    A = np.diag([1.5, 0.5])
    P, come = initial_covariance(A, Q, kind="lyapunov")
    ok &= _check("esplosivo+stabile -> risolve sul solo blocco stabile",
                 come == "lyapunov" and abs(P[0, 0]) < 1e-12
                 and np.isclose(P[1, 1], 1 / 0.75),
                 f"P00 {P[0, 0]:.1e}  P11 {P[1, 1]:.4f}")
    ok &= _check("...ed e' PSD", np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1e-9)

    # (c) kappa*I, con lo zero come caso particolare e non come ramo a parte.
    P, come = initial_covariance(A, Q, kind=0.0)
    ok &= _check("kind = 0.0 -> P_0 = 0", come == "kappa" and not P.any())
    P, come = initial_covariance(A, Q, kind=1e4)
    ok &= _check("kind = 1e4 -> kappa*I", come == "kappa"
                 and np.allclose(P, 1e4 * np.eye(2)))

    # (d) il ripiego: tutto esplosivo, non c'e' blocco stabile da risolvere.
    #     `come` deve DIRLO, perche' e' l'unico caso in cui il numero non e' loro.
    A = np.diag([1.5, 2.0])
    P, come = initial_covariance(A, Q, kind="lyapunov", kappa_fallback=1.0)
    ok &= _check("tutto esplosivo -> ripiego dichiarato",
                 come == "fallback" and np.allclose(P, np.eye(2)))
    P, come = initial_covariance(A, Q, kind="lyapunov", kappa_fallback=0.0)
    ok &= _check("...e con kappa_fallback = 0 il ripiego e' 0",
                 come == "fallback" and not P.any())
    return ok


# ─── §2  I default dei quattro modelli ────────────────────────────────────────

def test_default() -> bool:
    print("\n§2  I default, letti dalle firme (la tabella dell'header)")
    ok = True

    def _dflt(fn, nome: str = "p0"):
        return inspect.signature(fn).parameters[nome].default

    ok &= _check("P0_DEFAULT == 'lyapunov'", P0_DEFAULT == "lyapunov",
                 repr(P0_DEFAULT))
    ok &= _check("C: cbvar.fit  p0 = 'lyapunov'", _dflt(cbvar.fit) == "lyapunov",
                 repr(_dflt(cbvar.fit)))
    ok &= _check("Q: qbvar.nowcast  p0 = None -> il condiviso",
                 _dflt(qbvar.nowcast) is None)
    ok &= _check("B: bbvar.fit / fit_reuse  p0 = None -> zeri",
                 _dflt(bbvar.fit) is None and _dflt(bbvar.fit_reuse) is None)
    ok &= _check("L: lbvar.fit_reuse  p0 = None -> zeri",
                 _dflt(lbvar.fit_reuse) is None)
    ok &= _check("L: build_state_space  p0 = None, P0 = None",
                 _dflt(lbvar.build_state_space) is None
                 and _dflt(lbvar.build_state_space, "P0") is None)

    # Il C non espone `p0` nel ramo di riuso, e va bene cosi': `fit` calcola
    # (ss, P0) una volta e `fit_reuse` li riceve gia' fatti in `systems`.  Se un
    # domani comparisse un `p0` anche li', sarebbero due strade per la stessa
    # scelta — cioe' il problema che questo modulo esiste per aver chiuso.
    ok &= _check("C: fit_reuse NON ha un suo p0 (P_0 arriva da `systems`)",
                 "p0" not in inspect.signature(cbvar.fit_reuse).parameters)

    # Lo smoother DENTRO la stima dell'L e' il `temp*0` di lbvar.m r.52 e non
    # deve diventare parametrico: li' le prime p righe sono dati osservati.
    # Si guarda la CHIAMATA, non la riga: cosi' una riformattazione non fa
    # fallire il test, ma un `p0=` che comparisse li' si'.
    #
    # SI GUARDA IN `_finite_smoother`, NON IN `fit`.  La chiamata e' li' da
    # quando la stima ha una rete numerica intorno allo smoother, e questo
    # test cercava ancora in `fit`: trovava zero chiamate e falliva senza che
    # niente fosse rotto — un test che guarda il posto sbagliato non e' un
    # test.  Si cerca in ENTRAMBE le funzioni, cosi' regge anche se la
    # chiamata torna a spostarsi.
    sorgente = inspect.getsource(lbvar.fit) + inspect.getsource(
        lbvar._finite_smoother)
    chiamate = re.findall(r"build_state_space\(([^)]*)\)", sorgente)
    ok &= _check("L: lo smoother della stima resta a P_0 = 0",
                 len(chiamate) == 1 and "p0" not in chiamate[0],
                 f"{len(chiamate)} chiamata/e")
    return ok


# ─── §3  B e L non seguono P0_DEFAULT.  IL CONTROLLO CENTRALE ─────────────────

def test_insensibilita() -> bool:
    print("\n§3  B e L sono insensibili a P0_DEFAULT (con controllo negativo)")
    ok = True

    n, p = 2, 2
    B = _lbvar_B(np.diag([0.5, 0.9]), p=p)
    Sigma = np.eye(n)
    a0 = np.zeros(n * p)

    base = lbvar.build_state_space(B, Sigma, n, p, a0).P0
    ok &= _check("p0 = None -> P_0 = 0 (l'unica companion di B e di L)",
                 not base.any())

    # Il B usa LA STESSA funzione: `bbvar` importa `build_state_space` da
    # `lbvar`.  Se un domani si sdoppiassero, questo controllo lo direbbe.
    ok &= _check("...ed e' la stessa che usa il B", bbvar.build_state_space
                 is lbvar.build_state_space)

    # Il controllo negativo: se `p0` esplicito NON cambiasse nulla, il test
    # sopra non starebbe distinguendo niente.
    esplicito = lbvar.build_state_space(B, Sigma, n, p, a0, p0="lyapunov").P0
    ok &= _check("controllo negativo: p0 esplicito CAMBIA P_0",
                 esplicito.any(), f"max {np.abs(esplicito).max():.4f}")

    # E ora la regressione vera: si sposta la costante condivisa.
    vecchio = ss_mod.P0_DEFAULT
    try:
        ss_mod.P0_DEFAULT = 1e4
        dopo = lbvar.build_state_space(B, Sigma, n, p, a0).P0
        ok &= _check("P0_DEFAULT = 1e4 -> B e L NON si muovono",
                     not dopo.any(), f"max {np.abs(dopo).max():.1e}")
        # ...mentre il Q la legge a ogni chiamata, ed e' il comportamento voluto:
        # la stessa riga di `qbvar.nowcast`, `P0_DEFAULT if p0 is None else p0`.
        P_q, _ = initial_covariance(np.diag([0.5, 0.9]), np.eye(2),
                                    kind=ss_mod.P0_DEFAULT)
        ok &= _check("...mentre il Q segue la costante (e' il suo default)",
                     np.allclose(P_q, 1e4 * np.eye(2)))
    finally:
        ss_mod.P0_DEFAULT = vecchio
    ok &= _check("costante ripristinata", ss_mod.P0_DEFAULT == "lyapunov")
    return ok


# ─── §4  Il ripiego dell'L e' 0, non kappa*I ──────────────────────────────────

def test_ripiego_L() -> bool:
    print("\n§4  Il ripiego di B/L e' 0 - anti-regressione sul -97.74%")
    ok = True

    # p = 1: nessun blocco di scorrimento, quindi la companion e' esplosiva DEL
    # TUTTO e `lyapunov_symm` non ha niente da risolvere.  E' il solo modo di
    # esercitare l'`except` senza inventare una matrice patologica.
    n, p = 2, 1
    B = _lbvar_B(np.diag([1.5, 2.0]), p=p)
    P0 = lbvar.build_state_space(B, np.eye(n), n, p, np.zeros(n),
                                 p0="lyapunov").P0
    ok &= _check("companion esplosiva + p0 = 'lyapunov' -> P_0 = 0",
                 not P0.any(), f"max {np.abs(P0).max():.1e}")

    # Che sia una SCELTA e non il default della funzione: `initial_covariance`
    # da sola ripiegherebbe su I.  E' il `kappa_fallback = 0.0` di lbvar.py.
    P_std, come = initial_covariance(np.diag([1.5, 2.0]), np.eye(2),
                                     kind="lyapunov")
    ok &= _check("...e non e' il default: la funzione da sola darebbe I",
                 come == "fallback" and np.allclose(P_std, np.eye(2)))
    ok &= _check("il kappa_fallback = 0.0 e' scritto in lbvar",
                 "kappa_fallback=0.0" in inspect.getsource(lbvar.build_state_space))
    return ok


def main() -> bool:
    print("=" * 82)
    print("P_0 - la scelta centralizzata (state_space.initial_covariance)")
    print("=" * 82)
    ok = True
    for t in (test_vie, test_default, test_insensibilita, test_ripiego_L):
        ok &= t()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
