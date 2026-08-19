"""
core/bvar/core.py

IL CORE SAMPLER — l'unica porta d'ingresso per Q, C, B e L.

Assembla i due strati costruiti nei blocchi precedenti:

    strato 2   gamma = (lambda, mu, psi)   Metropolis     (hyper.py)
    strato 1   (B, Sigma) | gamma          forma chiusa   (niw.py)

L'INVARIANTE (Gate 0): il core non vede mai un NaN.  Riceve una matrice (T, n)
DENSA e restituisce estrazioni di (B, Sigma).  Il dato mancante e' gestito
FUORI: dal Kalman a valle (Q, C, B) o dal simulation smoother a monte (L).

I QUATTRO MODELLI DIFFERISCONO IN COSA GLI PASSANO, NON IN QUALE CORE CHIAMANO
------------------------------------------------------------------------------
    Q-BVAR   pannello trimestrale, n=30, p=5     -> sample() una volta
    C-BVAR   identico a Q (e' il suo stadio 1)   -> sample() una volta
    B-BVAR   pannello impilato, n piu' grande    -> sample() una volta
    L-BVAR   pannello mensile completo, p=17     -> step() a OGNI iterazione,
                                                    intrecciato con lo smoother


PERCHE' ESISTONO SIA sample() SIA step()
=========================================
`step()` non e' una comodita': e' un REQUISITO D'INTERFACCIA fissato al Gate 0.

Cimadomo §2.2 descrive l'L-BVAR come un ciclo che alterna: "Using the simulation
smoother of Durbin and Koopman (2001), we draw the complete monthly dataset ...
then, using the posterior sampler of Giannone et al. (2015), we draw the
hyperparameters lambda, mu and psi conditional on the complete monthly dataset,
and finally, we draw the model parameters conditional on the hyperparameters and
the complete monthly dataset."

Se il core sapesse fare solo run completi, l'L-BVAR dovrebbe reimplementarlo —
ed e' esattamente il branching che il documento di contesto vieta.  `step()`
esegue UNA spazzata (un passo di Metropolis sugli iperparametri + un'estrazione
coniugata dei parametri) su un pannello che puo' cambiare da una chiamata
all'altra: e' la primitiva che al Gate 5 si intreccia con lo smoother.

`sample()` e' semplicemente `step()` in un ciclo, con burn-in e raccolta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.bvar.dummies import sum_of_coefficients_dummy
from core.bvar.hyper import (
    HyperPrior,
    HyperTarget,
    MetropolisState,
    build_target,
    init_metropolis,
    metropolis_step,
)
from core.bvar.niw import build_prior, draw, niw_posterior
from core.bvar.spec import BVARSpec, Hyper


@dataclass
class CoreState:
    """
    Lo stato del campionatore fra una spazzata e l'altra.

    E' cio' che l'L-BVAR si portera' dietro nel proprio ciclo MCMC: contiene
    tutto il necessario per riprendere, cosi' la catena degli iperparametri non
    riparte da zero a ogni cambio del pannello latente.
    """

    target: HyperTarget
    metro: MetropolisState
    B: np.ndarray | None = None          # (k, n)  ultimo draw
    Sigma: np.ndarray | None = None      # (n, n)

    @property
    def hyper(self) -> Hyper:
        return self.target.unpack(self.metro.gamma)


@dataclass
class BVARDraws:
    """Le estrazioni raccolte da `sample()`."""

    B: np.ndarray                 # (S, k, n)
    Sigma: np.ndarray             # (S, n, n)
    lam: np.ndarray               # (S,)
    mu: np.ndarray                # (S,)
    psi: np.ndarray               # (S, n)
    logpost: np.ndarray           # (S,)
    acceptance: float
    spec: BVARSpec = field(repr=False, default=None)

    def summary(self) -> str:
        q = lambda a: np.percentile(a, [5, 50, 95])                  # noqa: E731
        ql, qm = q(self.lam), q(self.mu)
        return (f"S={len(self.lam)}  accettazione={self.acceptance:.1%}\n"
                f"  lambda  mediana {ql[1]:.4f}   [90%: {ql[0]:.4f}, {ql[2]:.4f}]\n"
                f"  mu      mediana {qm[1]:.4f}   [90%: {qm[0]:.4f}, {qm[2]:.4f}]")


# ─── Una spazzata ─────────────────────────────────────────────────────────────

def step(state: CoreState, rng: np.random.Generator, *,
         panel: np.ndarray | None = None, draw_params: bool = True,
         n_metro: int = 1) -> CoreState:
    """
    UNA spazzata del core: un passo di Metropolis sugli iperparametri, poi (se
    richiesto) un'estrazione coniugata di (B, Sigma).

    Corrisponde ai passi 2-4 dell'Appendice B di GLP.

    Parameters
    ----------
    panel : (T, n) or None
        Se dato, il pannello viene RICOSTRUITO prima della spazzata.  E' il
        gancio per l'L-BVAR: a ogni iterazione lo smoother produce un pannello
        mensile completo diverso, e il core deve girare su quello, mantenendo
        pero' la catena degli iperparametri (theta, W, c) da dove stava.
    draw_params : bool
        False salta l'estrazione di (B, Sigma) — utile quando servono solo gli
        iperparametri (es. durante il burn-in).
    n_metro : int
        Quante spazzate di Metropolis fare sullo STESSO pannello prima di
        estrarre (B, Sigma).  Default 1 = il ciclo del paper.

        PERCHE' PUO' CONVENIRE PIU' DI UNA.  Ripetere un kernel che lascia
        invariata la condizionale la lascia invariata comunque: n spazzate sono
        un kernel di Metropolis valido quanto una, e il target non cambia.  E'
        una deviazione ALGORITMICA dagli autori, della stessa natura della
        parametrizzazione log — da dichiarare, non da nascondere.

        Conviene quando il passo costoso e' l'ALTRO.  Nell'L-BVAR il simulation
        smoother costa ~8 s e una valutazione di marginal likelihood 0.148 s:
        con una sola spazzata si paga lo smoother per aggiornare gli
        iperparametri una volta sola, ed e' proprio la' che il mescolamento e'
        pessimo (ESS/iterazione 0.05 su lambda contro 0.93 sul pannello).
    """
    if panel is not None:
        old = state.target
        state.target = build_target(old.spec, panel, old.hyperprior)
        # il target e' cambiato: la log-posterior corrente va rivalutata
        state.metro.logpost = state.target.log_posterior_log_scale(state.metro.theta)

    for _ in range(max(1, int(n_metro))):
        state.metro = metropolis_step(state.target, state.metro, rng)

    if draw_params:
        tgt, hyp = state.target, state.hyper
        prior = build_prior(tgt.spec, hyp)
        Yd, Xd = sum_of_coefficients_dummy(tgt.spec, hyp, tgt.y0_bar)
        post = niw_posterior(np.vstack([Yd, tgt.Y]), np.vstack([Xd, tgt.X]), prior)
        B, S = draw(post, rng, n_draws=1)
        state.B, state.Sigma = B[0], S[0]
    return state


# ─── Il run completo ──────────────────────────────────────────────────────────

def sample(
    spec: BVARSpec,
    panel: np.ndarray,
    *,
    n_draws: int = 1000,
    burn: int = 500,
    rng: np.random.Generator | None = None,
    hyperprior: HyperPrior | None = None,
    c: float = 0.5,
    tune: bool = True,
    target_acceptance: float = 0.20,
    verbose: bool = False,
) -> BVARDraws:
    """
    Il campionatore completo: burn-in + raccolta.

    Parameters
    ----------
    panel : (T, n)
        DENSO, in unita' del modello.  Solleva se contiene NaN.
    tune : bool
        Durante il burn-in adatta `c` per avvicinare l'accettazione a
        `target_acceptance`.  L'Appendice B lo autorizza esplicitamente: "c is a
        scaling constant CHOSEN TO OBTAIN an acceptance rate of approximately 20
        percent".  Dopo il burn-in `c` e' congelato, cosi' la catena raccolta e'
        un Metropolis omogeneo e valido.
    """
    rng = np.random.default_rng() if rng is None else rng
    target = build_target(spec, panel, hyperprior)
    state = CoreState(target=target, metro=init_metropolis(target, rng, c=c))

    # ── burn-in, con taratura di c ──
    win = max(50, burn // 10)
    for i in range(burn):
        step(state, rng, draw_params=False)
        if tune and (i + 1) % win == 0:
            acc = state.metro.acceptance
            state.metro.c *= float(np.exp((acc - target_acceptance) * 2.0))
            state.metro.c = float(np.clip(state.metro.c, 1e-6, 1e3))
            state.metro.n_accept = state.metro.n_prop = 0
            if verbose:
                print(f"    burn {i+1}/{burn}  acc={acc:.1%}  c={state.metro.c:.4g}")
    state.metro.n_accept = state.metro.n_prop = 0          # statistiche pulite

    # ── raccolta ──
    k, n = spec.k, spec.n
    B = np.empty((n_draws, k, n))
    S = np.empty((n_draws, n, n))
    lam = np.empty(n_draws)
    mu = np.empty(n_draws)
    psi = np.empty((n_draws, n))
    lps = np.empty(n_draws)

    for s in range(n_draws):
        step(state, rng)
        B[s], S[s] = state.B, state.Sigma
        g = state.metro.gamma
        lam[s], mu[s], psi[s] = g[0], g[1], g[2:]
        lps[s] = state.metro.logpost
        if verbose and (s + 1) % max(1, n_draws // 5) == 0:
            print(f"    draw {s+1}/{n_draws}  acc={state.metro.acceptance:.1%}")

    return BVARDraws(B=B, Sigma=S, lam=lam, mu=mu, psi=psi, logpost=lps,
                     acceptance=state.metro.acceptance, spec=spec)


__all__ = ["CoreState", "BVARDraws", "step", "sample"]
