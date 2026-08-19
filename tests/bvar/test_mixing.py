"""
core/bvar/tests/test_mixing.py

IL MESCOLAMENTO DEGLI IPERPARAMETRI — da sospetto a misura.

    python -m core.bvar.tests.test_mixing

Non e' un test di gate: e' un ESPERIMENTO, e sta qui perche' vada rieseguito.
Non c'e' niente da "passare": produce numeri, e i numeri decidono se
l'ipotesi regge.  L'unica cosa che puo' fallire e' la meccanica (una catena
degenere, un modo non trovato).


================================================================================
LA DOMANDA, E PERCHE' SI FA QUESTO TEST E NON UN ALTRO
================================================================================
Il Metropolis sugli iperparametri mescola male in TUTTI i modelli:

    modello   dimensione della proposta d     ESS/iterazione di lambda
    L-BVAR              39                          0.053
    B-BVAR              86                          0.015

La prima spiegazione scritta era il TARGET MOBILE: nell'L-BVAR il pannello
latente cambia a ogni iterazione, quindi la condizionale degli iperparametri si
muove sotto la catena.  **Falsificata** dal B-BVAR, che ha il Kalman a valle —
target FISSO per costruzione — e mescola PEGGIO, con l'accettazione centrata sul
20% dell'Appendice B.

Sopravvive un solo sospetto: **la DIMENSIONE della proposta**.  Ma 39 e 86 sono
DUE PUNTI, non una misura, e i due modelli differiscono anche in altro (dati,
frequenza, blocking).  Serve variare d **a modello fermo**.

Il Q-BVAR e' lo strumento giusto: forma chiusa, nessuno smoother, una
valutazione del target costa ~8 ms, e d = 2 + n si muove semplicemente
togliendo colonne al pannello.  Stesso modello, stessa T, stesso seme, stesso
Metropolis: l'unica cosa che cambia e' d.

  §1  LA CRESTA — la correlazione a posteriori fra iperparametri, letta da W.
  §2  ESPERIMENTO A — d che cresce: sottoinsiemi annidati n = 6 ... 30.
  §3  ESPERIMENTO B — blocking dei psi: si campiona (lambda, mu), d = 2.
  §4  IL VERDETTO — ESS/iterazione x d e' costante?

La PREVISIONE FALSIFICABILE (Roberts-Gelman-Gilks): per un random-walk
Metropolis in dimensione d, anche con la scala ottimale, l'efficienza per
iterazione scala come 1/d, cioe'

    ESS/iterazione  x  d  ~=  costante.

Se il prodotto e' piatto lungo il §2, il mescolamento lento e' il costo noto
dell'algoritmo dell'Appendice B in dimensione alta — una proprieta' del metodo,
da descrivere.  Se crolla piu' di 1/d, c'e' dell'altro e va cercato.


================================================================================
L'ESITO (run del 2026-08-01, 5000 estrazioni per cella, 2 semi, 388 s)
================================================================================
    n     d    acc      c      ESS/it lam   rho1 lam   ESS/it x d
    6     8   17.3%   1.091      0.0355      0.938        0.28
    12   14   18.2%   0.514      0.0269      0.955        0.38
    18   20   20.5%   0.307      0.0117      0.973        0.23
    24   26   19.5%   0.285      0.0104      0.976        0.27
    30   32   19.9%   0.220      0.0087      0.977        0.28
    psi congelati:
          2   19.9%   9.403      0.1099      0.781        0.22

    pendenza log(ESS/it) ~ log(d):  -1.10   (previsione teorica -1.00)
    prodotto ESS/it x d: 0.22 ... 0.38, spread 1.6x

**LA PREVISIONE E' CONFERMATA.**  Tre letture:

  1. il prodotto e' piatto su un fattore 4 di dimensione, la pendenza e' -1.10
     contro -1.00 previsto: e' la legge 1/d;
  2. l'accettazione e' a bersaglio in OGNI cella (17-21%) e c scende da 1.09 a
     0.22 al crescere di d — la taratura 1/sqrt(d) che la procedura trova da
     sola.  Non e' un passo mal tarato;
  3. congelare i psi NON e' speciale: l'ESS salta 13x (previsti 32/2 = 16x) ma
     il prodotto resta 0.22, in linea con tutti gli altri.  I psi sono 30
     coordinate come le altre, non un blocco patologico.

SORPRESA UTILE — la CRESTA c'e' ma NON e' la causa.  Il §1 misura corr(lambda,
mu) = -0.410, |corr| massima 0.496, ma |corr| MEDIANA 0.028 e cond(W) = 58.8.
La posteriore e' mite, non una cresta stretta.  E c'e' una ragione strutturale:
la proposta dell'Appendice B usa gia' W, l'Hessiana inversa al modo, che E' la
forma locale della posteriore — quindi propone gia' allineata alle correlazioni.
Cio' che W non puo' correggere e' la DIMENSIONE.

Conclusione: il mescolamento lento e' il costo noto di un random-walk Metropolis
in dimensione alta, non una patologia dei nostri modelli.  Si descrive.
Trattazione completa per la tesi nell'header di `hyper.py`, sezione
"IL MESCOLAMENTO".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from core.bvar import qbvar
from core.bvar.hyper import (
    HyperTarget,
    MetropolisState,
    build_target,
    find_mode,
    metropolis_step,
)
from core.bvar.spec import BVARSpec, MinnesotaSpec
from core.bvar.diagnostics import ess

_OK, _FAIL = "OK", "FAIL"

#: I due punti veri, dai run definitivi dei Gate 4 e 5.  Non si ricalcolano
#: qui (costano ore): servono da termine di paragone per il §4.
REAL_MODELS = (("L-BVAR", 39, 0.053, 0.939, 0.058),
               ("B-BVAR", 86, 0.015, 0.986, 0.208))


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


# ─── La meccanica: una catena di Metropolis, sola ─────────────────────────────

@dataclass
class ChainStats:
    """Quel che si misura su una catena."""
    d: int
    n: int
    acceptance: float
    c: float
    ess_lam: float          # ESS/iterazione
    ess_mu: float
    ess_psi: float          # mediana sulle n coordinate
    rho1_lam: float         # autocorrelazione a lag 1
    seconds: float


def _autocorr1(x: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    den = float(x @ x)
    return float(x[:-1] @ x[1:] / den) if den > 0 else np.nan


def run_chain(target, theta0: np.ndarray, W: np.ndarray, rng, *,
              n_draws: int = 5000, burn: int = 1500, c: float = 0.5,
              target_acceptance: float = 0.20, tune: bool = True):
    """
    Il Metropolis nudo, senza l'estrazione di (B, Sigma).

    E' la stessa procedura di `core.sample` — stessa regola di taratura di `c`
    nel burn-in, stesso congelamento dopo — ma senza il passo coniugato, che
    qui non serve e costerebbe: l'oggetto in esame e' la catena degli
    iperparametri, non il posterior dei coefficienti.
    """
    state = MetropolisState(theta=np.asarray(theta0, float).copy(),
                            logpost=target.log_posterior_log_scale(theta0),
                            W=W, c=c)
    win = max(50, burn // 10)
    for i in range(burn):
        metropolis_step(target, state, rng)
        if tune and (i + 1) % win == 0:
            acc = state.acceptance
            state.c *= float(np.exp((acc - target_acceptance) * 2.0))
            state.c = float(np.clip(state.c, 1e-6, 1e3))
            state.n_accept = state.n_prop = 0
    state.n_accept = state.n_prop = 0

    keep = np.empty((n_draws, state.theta.size))
    for s in range(n_draws):
        metropolis_step(target, state, rng)
        keep[s] = state.theta
    return np.exp(keep), state


def measure(target, theta0, W, rng, n: int, *, n_draws: int, burn: int,
            c: float = 0.5) -> ChainStats:
    t0 = time.time()
    g, state = run_chain(target, theta0, W, rng, n_draws=n_draws, burn=burn, c=c)
    d = g.shape[1]
    e = lambda col: ess(g[:, col][None, :]) / n_draws          # noqa: E731
    psi = ([e(j) for j in range(2, d)] if d > 2 else [np.nan])
    return ChainStats(d=d, n=n, acceptance=state.acceptance, c=state.c,
                      ess_lam=e(0), ess_mu=e(1), ess_psi=float(np.median(psi)),
                      rho1_lam=_autocorr1(g[:, 0]), seconds=time.time() - t0)


def _subset(spec: BVARSpec, n: int) -> BVARSpec:
    """Le prime `n` colonne del profilo, con d_centre affettato di conseguenza."""
    minn = MinnesotaSpec(p=spec.minnesota.p,
                         d_centre=tuple(spec.minnesota.d_centre[:n]),
                         use_dio=spec.minnesota.use_dio)
    return replace(spec, series=spec.series[:n], freq=spec.freq[:n],
                   transform=spec.transform[:n], minnesota=minn)


# ─── §1  La cresta ────────────────────────────────────────────────────────────

def test_ridge(W: np.ndarray, names: list[str]) -> bool:
    """
    La geometria della posteriore degli iperparametri, letta al modo.

    `W` e' l'Hessiana inversa in scala log: e' l'approssimazione gaussiana
    della posteriore attorno al picco, cioe' esattamente la forma che la
    proposta dell'Appendice B usa.  La sua matrice di CORRELAZIONE dice se le
    coordinate sono allineate agli assi (mescola bene) o giacciono su una
    cresta obliqua (mescola male).
    """
    print("\n[1] LA CRESTA - correlazione a posteriori al modo (da W)")
    sd = np.sqrt(np.diag(W))
    R = W / np.outer(sd, sd)
    off = R - np.eye(len(R))
    i, j = np.unravel_index(np.abs(off).argmax(), off.shape)
    ev = np.linalg.eigvalsh(R)

    print(f"    corr(lambda, mu)                       {R[0, 1]:+.3f}")
    print(f"    |corr| massima fuori diagonale         {off[i, j]:+.3f}"
          f"   ({names[i]} vs {names[j]})")
    print(f"    |corr| mediana fuori diagonale         {np.median(np.abs(off[~np.eye(len(R), dtype=bool)])):.3f}")
    print(f"    autovalori di R: min {ev[0]:.4f}  max {ev[-1]:.4f}"
          f"   -> anisotropia {ev[-1] / max(ev[0], 1e-12):.0f}x")
    print(f"    cond(W)                                {np.linalg.cond(W):.1f}")
    print("\n    Come si legge: se questa fosse una cresta stretta, la |corr|"
          "\n    mediana sarebbe alta e cond(W) grande.  Se invece e' mite, la"
          "\n    correlazione NON puo' essere la causa del mescolamento lento —"
          "\n    tanto piu' che la proposta c*W e' gia' allineata a W, cioe' alla"
          "\n    forma locale della posteriore.  Resterebbe solo la dimensione.")
    return _check("W e' definita positiva e non degenere", ev[0] > 0,
                  f"autovalore minimo {ev[0]:.4f}")


# ─── §2  Esperimento A — la dimensione ────────────────────────────────────────

def test_dimension(panel, spec, grid, *, n_draws: int, burn: int,
                   seeds: tuple[int, ...]) -> list[ChainStats]:
    print("\n[2] ESPERIMENTO A - d che cresce a modello fermo")
    print(f"    Q-BVAR, p={spec.p}, T={panel.shape[0]} trimestri, sottoinsiemi"
          f" annidati; d = 2 + n\n")
    print(f"    {'n':>3} {'d':>4} {'acc':>7} {'c':>7} "
          f"{'ESS/it lam':>11} {'ESS/it mu':>10} {'ESS/it psi':>11} "
          f"{'rho1 lam':>9} {'sec':>6}")
    out: list[ChainStats] = []
    for n in grid:
        sub = _subset(spec, n)
        tgt = build_target(sub, panel.iloc[:, :n].to_numpy(float))
        gm, W = find_mode(tgt)                    # una volta per cella
        theta0 = np.log(gm)
        cells = [measure(tgt, theta0, W, np.random.default_rng(sd), n,
                         n_draws=n_draws, burn=burn) for sd in seeds]
        # media sui semi: l'ESS e' esso stesso una stima rumorosa
        st = ChainStats(d=cells[0].d, n=n,
                        acceptance=float(np.mean([c.acceptance for c in cells])),
                        c=float(np.mean([c.c for c in cells])),
                        ess_lam=float(np.mean([c.ess_lam for c in cells])),
                        ess_mu=float(np.mean([c.ess_mu for c in cells])),
                        ess_psi=float(np.mean([c.ess_psi for c in cells])),
                        rho1_lam=float(np.mean([c.rho1_lam for c in cells])),
                        seconds=float(np.sum([c.seconds for c in cells])))
        out.append(st)
        print(f"    {st.n:>3} {st.d:>4} {st.acceptance:>6.1%} {st.c:>7.3f} "
              f"{st.ess_lam:>11.4f} {st.ess_mu:>10.4f} {st.ess_psi:>11.4f} "
              f"{st.rho1_lam:>9.3f} {st.seconds:>6.0f}")
    return out


# ─── §3  Esperimento B — il blocking dei psi ──────────────────────────────────

class FrozenPsiTarget:
    """
    Il target ristretto a (lambda, mu), con i `psi` CONGELATI al modo.

    E' il "blocking dei psi" nella sua forma estrema: invece di aggiornare 84
    coordinate insieme, se ne aggiornano 2 e le altre stanno ferme.  Non e' un
    campionatore valido per la posteriore congiunta — e non deve esserlo: serve
    a separare l'effetto della DIMENSIONE da quello del resto.

    Nota che l'interruttore esiste anche da loro: `bvarGLP_fixedhyp` con
    `'MNpsi', 0` (riga commentata di `bbvar.m` r.35), dove i psi non sono
    iperparametri ma restano fissi a `SS`, il residuo AR(1).
    """

    def __init__(self, full: HyperTarget, psi_mode: np.ndarray):
        self.full = full
        self.log_psi = np.log(np.asarray(psi_mode, float))

    def log_posterior_log_scale(self, theta2: np.ndarray) -> float:
        return self.full.log_posterior_log_scale(
            np.concatenate([np.asarray(theta2, float), self.log_psi]))


def test_blocking(panel, spec, W_full, gamma_mode, *, n_draws: int, burn: int,
                  seeds: tuple[int, ...]) -> ChainStats:
    print("\n[3] ESPERIMENTO B - blocking dei psi: si campiona (lambda, mu), d = 2")
    tgt = build_target(spec, panel.to_numpy(float))
    frozen = FrozenPsiTarget(tgt, gamma_mode[2:])

    # La proposta giusta per il blocco condizionale non e' il sotto-blocco 2x2
    # di W (quella e' la MARGINALE), ma l'inversa del sotto-blocco 2x2
    # dell'Hessiana: Var(lambda, mu | psi).
    H = np.linalg.inv(W_full)
    W2 = np.linalg.inv(H[:2, :2])

    cells = [measure(frozen, np.log(gamma_mode[:2]), W2,
                     np.random.default_rng(sd), spec.n,
                     n_draws=n_draws, burn=burn) for sd in seeds]
    st = ChainStats(d=2, n=spec.n,
                    acceptance=float(np.mean([c.acceptance for c in cells])),
                    c=float(np.mean([c.c for c in cells])),
                    ess_lam=float(np.mean([c.ess_lam for c in cells])),
                    ess_mu=float(np.mean([c.ess_mu for c in cells])),
                    ess_psi=np.nan,
                    rho1_lam=float(np.mean([c.rho1_lam for c in cells])),
                    seconds=float(np.sum([c.seconds for c in cells])))
    print(f"    n={st.n}  d={st.d}  acc={st.acceptance:.1%}  c={st.c:.3f}  "
          f"ESS/it lam={st.ess_lam:.4f}  ESS/it mu={st.ess_mu:.4f}  "
          f"rho1={st.rho1_lam:.3f}  ({st.seconds:.0f}s)")
    return st


# ─── §4  Il verdetto ──────────────────────────────────────────────────────────

def test_verdict(cells: list[ChainStats], blocked: ChainStats) -> bool:
    print("\n[4] IL VERDETTO - ESS/iterazione x d")
    print("\n    Se il mescolamento e' governato dalla DIMENSIONE, il prodotto"
          "\n    ESS/iterazione x d dev'essere ~piatto (Roberts-Gelman-Gilks).\n")
    print(f"    {'modello':<12} {'d':>4} {'ESS/it lam':>11} {'x d':>7}")
    rows = [(f"Q n={c.n}", c.d, c.ess_lam) for c in cells]
    rows.append((f"Q psi bloc.", blocked.d, blocked.ess_lam))
    for name, d, e in rows:
        print(f"    {name:<12} {d:>4} {e:>11.4f} {e * d:>7.2f}")
    print()
    for name, d, e, rho, acc in REAL_MODELS:
        print(f"    {name:<12} {d:>4} {e:>11.4f} {e * d:>7.2f}"
              f"     (rho1 {rho:.3f}, acc {acc:.1%})")

    prod = np.array([c.ess_lam * c.d for c in cells])
    spread = prod.max() / max(prod.min(), 1e-12)
    slope = np.polyfit(np.log([c.d for c in cells]),
                       np.log([c.ess_lam for c in cells]), 1)[0]
    print(f"\n    prodotto: min {prod.min():.2f}  max {prod.max():.2f}"
          f"  -> spread {spread:.2f}x")
    print(f"    pendenza log(ESS/it) su log(d):  {slope:+.2f}"
          f"   (la previsione 1/d e' -1.00)")

    ok = _check("il prodotto ESS x d resta nello stesso ordine (spread < 4x)",
                spread < 4.0, f"{spread:.2f}x")
    ok &= _check("la pendenza e' negativa e vicina a -1 (fra -2.0 e -0.4)",
                 -2.0 < slope < -0.4, f"{slope:+.2f}")
    ok &= _check("congelare i psi (d=2) fa saltare l'ESS di almeno 5x",
                 blocked.ess_lam > 5 * cells[-1].ess_lam,
                 f"{blocked.ess_lam:.3f} contro {cells[-1].ess_lam:.4f}"
                 f"  ({blocked.ess_lam / max(cells[-1].ess_lam, 1e-12):.0f}x)")
    return ok


# ─── main ─────────────────────────────────────────────────────────────────────

def main(n_draws: int = 5000, burn: int = 1500,
         grid: tuple[int, ...] = (6, 12, 18, 24, 30),
         seeds: tuple[int, ...] = (11, 23)) -> bool:
    print("=" * 82)
    print("IL MESCOLAMENTO DEGLI IPERPARAMETRI — la dimensione della proposta")
    print("=" * 82)
    t0 = time.time()

    spec = BVARSpec.from_config("Q")
    panel = qbvar.estimation_panel(spec)
    print(f"  {spec.summary()}")
    print(f"  pannello {panel.shape[0]} x {panel.shape[1]}, "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")

    tgt = build_target(spec, panel.to_numpy(float))
    gamma_mode, W = find_mode(tgt)
    names = ["lambda", "mu"] + [f"psi[{s}]" for s in spec.series]

    ok = True
    ok &= test_ridge(W, names)
    cells = test_dimension(panel, spec, grid, n_draws=n_draws, burn=burn, seeds=seeds)
    blocked = test_blocking(panel, spec, W, gamma_mode, n_draws=n_draws,
                            burn=burn, seeds=seeds)
    ok &= test_verdict(cells, blocked)

    print("\n" + "=" * 82)
    print(("MECCANICA OK" if ok else "QUALCOSA NON TORNA") +
          f"   ({time.time() - t0:.0f}s)")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
