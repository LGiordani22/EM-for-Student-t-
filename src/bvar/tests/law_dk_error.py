"""
src/bvar/tests/law_dk_error.py

ESPERIMENTO, non un test di gate: qui non c'e' niente da «passare», produce
**numeri**.  Sta in `tests/` perche' vada rieseguito quando serve, come
`test_mixing`.

    python -m src.bvar.tests.law_dk_error

LA DOMANDA.  Il blocco Covid dell'L-BVAR falliva, e la spiegazione proposta e'
che l'errore del simulation smoother di Durbin-Koopman cresce come rho^T
mentre quello del campionatore a precisione non cresce affatto.  Qui la
spiegazione si MISURA, su un sistema piccolo (n = 6, p = 4) ma con la stessa
struttura dell'L-BVAR: lo stato E' il pannello, Z = [I 0], R = 1e-12, P0 = 0,
buchi sparsi piu' una coda di previsione.

IL CRITERIO E' UN RESIDUO, NON UN CONFRONTO FRA I DUE.  Dove il dato c'e', la
condizionale dice che il cammino latente deve riprodurlo entro ~1e-6 (R =
1e-12, cioe' 1e-6 in deviazione standard).  Chi non lo riproduce ha perso le
cifre, e lo si vede senza conoscere la risposta giusta.

COSA SI E' TROVATO (T = 500):

    rho^T        residuo DK     residuo precisione
    3.7e+05        8.6e-07            0
    4.1e+09        1.0e-05            0
    4.1e+11        2.3e-04            0
    4.0e+13        5.2e-02            0
    3.7e+15        2.7e+00            0

cioe' esattamente il limite di prim'ordine  ns * eps * rho^T / scala: il
residuo del DK e' proporzionale a rho^T una volta superato il pavimento del
nugget, e all'ultima riga il pannello estratto non riproduce piu' il dato.
Quello del campionatore a precisione e' zero a ogni rho e a ogni T, perche' li'
non c'e' nessuna differenza di numeri grandi da formare.

⚠️  Il rho stampato e' quello EFFETTIVO della companion, non il bersaglio
nominale: il rumore su A_1 lo sposta, e va letto quello.
"""
from __future__ import annotations

import numpy as np

from src.bvar.lbvar import build_state_space
from src.bvar.precision_smoother import precision_draw
from src.bvar.simsmoother import simulation_smoother

N, P = 6, 4


def _case(rho: float, T: int, seed: int = 0):
    """Un VAR(p) con la struttura dell'L-BVAR e raggio spettrale ~`rho`."""
    rng = np.random.default_rng(seed)
    A = ([np.eye(N) * rho + 0.02 * rng.standard_normal((N, N))]
         + [0.01 * rng.standard_normal((N, N)) for _ in range(P - 1)])
    B = np.vstack([np.vstack([a.T for a in A]),
                   0.01 * rng.standard_normal((1, N))])
    L = np.tril(rng.standard_normal((N, N))) * 0.2
    Sigma = L @ L.T + np.eye(N)
    head = rng.standard_normal((P, N))
    Y = rng.standard_normal((T, N))
    Y[rng.random((T, N)) < 0.3] = np.nan          # buchi sparsi
    Y[-6:] = np.nan                               # coda di previsione
    return B, Sigma, Y, head


def main() -> None:
    print("=" * 62)
    print("L'errore del DK contro rho e T — il residuo sulle celle osservate")
    print("=" * 62)
    print(f"{'rho':>7} {'T':>5} {'rho^T':>10} | {'res DK':>10} {'res prec':>10}")
    print("-" * 62)
    for rho in (0.98, 1.00, 1.01, 1.02, 1.03):
        for T in (100, 250, 500):
            B, Sigma, Y, head = _case(rho, T)
            a0 = np.concatenate([head[P - 1 - j] for j in range(P)])
            ss = build_state_space(B, Sigma, N, P, a0)
            r_eff = float(np.max(np.abs(np.linalg.eigvals(ss.A))))
            obs = ~np.isnan(Y)
            sc = float(np.max(np.abs(Y[obs])))
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    x = simulation_smoother(ss, Y, np.random.default_rng(1))
                e_dk = float(np.max(np.abs(x[:, :N][obs] - Y[obs]))) / sc
            except Exception:                                  # noqa: BLE001
                e_dk = float("inf")
            xp = precision_draw(B, Sigma, Y, head, np.random.default_rng(1),
                                n=N, p=P)
            e_pr = float(np.max(np.abs(xp[:, :N][obs] - Y[obs]))) / sc
            print(f"{r_eff:7.4f} {T:5d} {r_eff ** T:10.2e} | "
                  f"{e_dk:10.2e} {e_pr:10.2e}")
    print("-" * 62)
    print("res = residuo relativo massimo sulle celle OSSERVATE (atteso <= ~1e-6)")


if __name__ == "__main__":
    main()
