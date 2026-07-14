"""
mcmc/validate/dgp.py
====================

**Un solo posto dove vive la verità.**  Ogni check legge il DGP da qui; nessuno se lo
costruisce per conto suo.  È la lezione della collisione ``sigma`` / ``sigma^2``: quando
la verità è definita in tre posti diversi, prima o poi due di essi divergono e il
confronto restituisce numeri *plausibili e sbagliati*.

Il DGP canonico
---------------
Spec II + Option A, ``r = 3`` canali deliberatamente **disuguali**:

    rho^u   = (-0.70, -0.15, +0.45)      dominante negativo / debole / positivo
    sd(log h^u_k) = (1.03, 0.46, 0.70)   forte / DEBOLE / forte

Il canale 1 è **debole di proposito**.  Sotto Spec II ogni ``h^u_k`` è letto attraverso
**una sola** log-square per periodo (non c'è la media su ``r`` canali che il caso
scalare-comune regalava), quindi identificarlo richiede volatilità *e* lunghezza: a
``T = 600`` il canale debole **non è identificato**, e con lui non lo è il suo ``rho``.
Questa non è una scomodità da aggirare: è il **regime operativo** del nowcast in tempo
reale (T effettivo corto), quindi il validatore lo misura invece di scegliere un DGP
compiacente.

``Q`` è **diagonale** nel DGP canonico.  È una scelta, e va detta: a ``Q`` diagonale il
blocco decoupled è *esatto* (nessun effetto P1/P4) e il whitening di magnitudine di
Branch B è *esatto* (nessuna attenuazione P5, ``lambda(c)=1``).  Così ciò che la tabella
misura sul leverage è il **leverage**, non il coupling.  Il coupling ha il suo DGP
dedicato (``dgp_correlated_Q``), usato solo dai check su QML.

Il blocco idiosincratico ha **SV e leverage propri** (``rho^eps = -0.3``): esistono nel
modello, girano a ogni sweep, entrano nella densità predittiva — e non erano mai stati
validati.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmc.gibbs import fit_dfm_mcmc, load_warm_init
from mcmc.simulate_sv import simulate_dfm_sv


# ── La verita', in un posto solo ────────────────────────────────────────────────
RHO_U_TRUE = np.array([-0.70, -0.15, 0.45])
RHO_EPS_TRUE = -0.30
#: (mu, phi, SIGMA) — attenzione: il simulatore prende la DEVIAZIONE STANDARD in
#: ingresso e restituisce la VARIANZA (vedi simulate_sv, avvertimento nel docstring).
SV_U_PF = np.array([[0.0, 0.97, 0.25],       # sd incondizionata 1.03  — forte
                    [0.0, 0.92, 0.18],       #                   0.46  — DEBOLE
                    [0.0, 0.95, 0.22]])      #                   0.70  — forte
SV_EPS = (0.0, 0.94, 0.12)

STRONG = [0, 2]      # i canali comuni identificabili a T = 600
WEAK = 1             # il canale che NON lo e' — e il validatore lo dichiara


@dataclass
class Mode:
    """Le tre modalita'.  La tabella dichiara in quale e' stata prodotta: **un OK su
    ``rho`` ottenuto in --quick non vale**, e deve portarselo scritto."""
    name: str
    T: int
    n_iter: int
    burn_in: int
    chains: int
    n_reps: int = 0          # repliche per la modalita' coverage

    @property
    def is_probative(self) -> bool:
        """Ha diritto di pronunciarsi su rho e sui quantili di coda?  Solo --full:
        l'ESS di rho e' 40-122 su 3400 draw ANCHE col prior half-Normal, quindi una
        catena corta non misura, indovina."""
        return self.name in ("full", "coverage")


QUICK = Mode("quick", T=300, n_iter=400, burn_in=150, chains=2)
FULL = Mode("full", T=600, n_iter=1200, burn_in=400, chains=2)


def coverage_mode(n_reps: int) -> Mode:
    # Catene piu' corte ma MOLTE repliche: la copertura non ha bisogno di un ESS enorme
    # per replica, ha bisogno di N indipendenti.  E' l'unico modo di distinguere
    # "sfortuna" da "bias": su un dataset rho_hat = -0.50 sembrava rumore, su 12 la
    # copertura al 25% l'ha smascherato.
    return Mode("coverage", T=600, n_iter=1200, burn_in=400, chains=1, n_reps=n_reps)


def _diagonal(theta: dict) -> dict:
    th = dict(theta)
    th["Q"] = np.diag(np.diag(np.asarray(theta["Q"], float)))
    return th


def build(mode: Mode, *, branch: str = "B", seed: int = 22,
          config_name: str = "small", correlated_Q: float = 0.0) -> dict:
    """Il DGP canonico.  ``branch`` sceglie il **timing del DGP**: ogni ramo gira sul
    DGP del *proprio* timing, perche' A e B sono due modelli, non due approssimazioni
    dello stesso.  ``correlated_Q > 0`` costruisce il DGP a ``Q`` correlata usato dai
    soli check su QML."""
    w = load_warm_init(config_name)
    theta = _diagonal(w["theta"])
    r = w["r"]
    if correlated_Q > 0.0:
        sd = np.sqrt(np.diag(np.asarray(theta["Q"], float)))
        C = np.full((r, r), correlated_Q) + (1.0 - correlated_Q) * np.eye(r)
        theta["Q"] = (sd[:, None] * C) * sd[None, :]

    timing = "contemporaneous" if branch == "A" else "lagged"
    sim = simulate_dfm_sv(theta, T=mode.T, freq_list=w["freq_list"],
                          block_map=w["block_map"], ordered_cols=w["ordered_cols"],
                          r=r, seed=seed, sv_u=(0.0, 0.95, 0.0),
                          sv_u_perfactor=SV_U_PF, sv_eps=SV_EPS,
                          rho_u=RHO_U_TRUE[:r], rho_eps=RHO_EPS_TRUE, timing=timing)
    return {"sim": sim, "theta": theta, "w": w, "r": r, "branch": branch,
            "timing": timing, "mode": mode, "seed": seed}


#: **La cache dei fit.**  Senza di essa il validatore rifa' la stessa catena in ogni
#: modulo — `linear`, `volatility` e `leverage` chiedono tutti e tre il fit di default —
#: e in modalita' --full questo significa ore di calcolo buttate.  La chiave e' l'intero
#: contenuto della richiesta (DGP + seme + kwargs), quindi due chiamate identiche
#: restituiscono la *stessa* catena: nessun rischio di confrontare draw diversi credendoli
#: uguali.
_CACHE: dict = {}


def _key(dgp: dict, seed: int, kw: dict) -> tuple:
    m = dgp["mode"]
    return (dgp["branch"], dgp["seed"], m.name, m.T, m.n_iter, m.burn_in, m.chains,
            seed, tuple(sorted((k, repr(v)) for k, v in kw.items())))


def fit(dgp: dict, *, seed: int = 700, n_chains: int | None = None, **kw) -> list[dict]:
    """Gira ``mode.chains`` catene e restituisce i risultati.  **Un solo fit serve molti
    check** — ed e' cio' che rende il validatore eseguibile invece di un'ora per
    parametro: i risultati sono in cache per (DGP, seme, kwargs).

    Default: prior half-Normal e griddy su ``rho`` — la configurazione che spediremmo.
    Ogni check che voglia variarli lo fa **esplicitamente**, e la sua ragione lo dice.
    """
    mode, w = dgp["mode"], dgp["w"]
    kw.setdefault("sv_sigma_prior", "half_normal")
    kw.setdefault("rho_sampler", "griddy")
    kw.setdefault("store_states", True)      # servono i draw di f_t per validarli
    kw.setdefault("store_vol", True)         # e i path h^u / h^eps

    nc = mode.chains if n_chains is None else n_chains
    k = _key(dgp, seed, kw) + (nc,)
    if k in _CACHE:
        return _CACHE[k]

    out = []
    for c in range(nc):
        out.append(fit_dfm_mcmc(
            dgp["sim"]["Y"], dgp["theta"], w["freq_list"], w["block_map"],
            w["ordered_cols"], n_iter=mode.n_iter, burn_in=mode.burn_in,
            seed=seed + c, sv=True, leverage=True, timing=dgp["timing"],
            verbose=False, **kw))
    _CACHE[k] = out
    return out


def stack(chains: list[dict], key: str) -> np.ndarray:
    """I draw di ``key`` impilati come ``(n_chains, n_keep, ...)`` — la forma che ESS e
    split-R-hat vogliono."""
    return np.stack([ch["draws"][key] for ch in chains])
