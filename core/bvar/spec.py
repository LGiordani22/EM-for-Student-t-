"""
core/bvar/spec.py

LE SPECIFICHE: che cosa E' un modello BVAR, prima ancora di stimarlo.

Questo modulo non calcola niente.  Contiene i tre dataclass che descrivono un
modello — quali serie, quanti lag, su che cosa si centra il prior — e il
costruttore che li riempie leggendo `config/bvar_series.json`.

    MinnesotaSpec   la forma del prior sui coefficienti autoregressivi
    Hyper           un punto nello spazio degli iperparametri (lambda, mu, psi)
    BVARSpec        un modello completo = serie + campione + MinnesotaSpec

REGOLA CHE VALE PER TUTTO IL PACCHETTO
--------------------------------------
Nessuna scelta di modellazione e' cablata qui dentro.  Quali serie entrano in
quale modello, quali vanno in log e quali in livello, quali si centrano su un
random walk e quali su un white noise: tutto sta in `config/bvar_series.json`,
in un file leggibile, con la motivazione accanto.  Questo modulo lo legge e lo
traduce in oggetti Python.  Se una scelta va cambiata si cambia la config, non
il codice — e la traccia resta nel file, che e' anche il documento da cui si
scrive la tesi.

In particolare `d_centre` (il vettore 1=rw / 0=wn dell'eq. 2 del paper) e'
COSTRUITO DALLA CONFIG, mai scritto a mano in un array.


================================================================================
k = n*p + 1 — IL NUMERO DA CUI DISCENDE TUTTO IL RESTO
================================================================================
`BVARSpec.k` sembra un dettaglio contabile.  Non lo e': e' la dimensione da cui
discendono, nell'ordine, il prior, il mescolamento della catena e il costo della
passata real-time.  Vale la pena averlo in testa prima di leggere il resto del
pacchetto, perche' tre risultati sperimentali che sembrano indipendenti sono in
realta' lo stesso fatto visto da tre lati.

OGNI equazione del VAR regredisce una variabile su TUTTE le variabili a TUTTI i
lag, piu' l'intercetta:

    x_{i,t} = c_i + sum_{s=1..p} sum_{j=1..n} a^{(s)}_{ij} x_{j,t-s} + eps_{i,t}

da cui `k = n*p + 1` coefficienti per equazione, e `k*n` in tutto — cioe' la
forma di `B`, che e' (k, n): una colonna per equazione.

                    n     p      k = n*p+1     k*n
    Q / C / B      30     5          151      4 530
    L              37    17          630     23 310

I NUMERI DEL PAPER SONO PIU' PICCOLI, e la differenza non e' cosmetica: loro
hanno n=18, quindi k=91 e companion 18*17=306 contro la nostra 37*17=629.  Ogni
volta che un risultato di mescolamento o di costo non combacia col loro, il
primo sospetto e' questo — vedi il campo `dimensionamento` della config, scritto
apposta perche' i recovery test fossero dimensionati sui NOSTRI numeri.

LE TRE CONSEGUENZE

1. IL PRIOR NON E' UN RAFFINAMENTO, E' LA CONDIZIONE DI STIMABILITA'.  L'L-BVAR
   stima 23 310 coefficienti su ~600 osservazioni mensili.  Nessun campione al
   mondo identifica quel modello ai minimi quadrati: il sistema e' clamorosamente
   sottodeterminato.  E' tutto il senso del Minnesota — non serve a "migliorare"
   una stima OLS che esisterebbe comunque, serve a farne esistere una.  Da cui
   anche il fatto che lambda (la stretta del prior) non sia un parametro di
   contorno ma la quantita' che decide quanto modello c'e'.

2. IL MESCOLAMENTO PEGGIORA COME 1/d, E NON E' UN BACO.  L'ESS per iterazione
   degli iperparametri cala come l'inverso della dimensione — misurato in
   `tests/test_mixing`: pendenza -1.10 contro il -1.00 che la teoria del
   random-walk Metropolis prevede (Roberts-Gelman-Gilks 1997).  Con d di ordine
   630 contro 151, l'L mescola strutturalmente peggio del Q, e nessuna taratura
   della proposta lo sposta.  Vedi l'header di `lbvar.py`, che documenta il
   tentativo fatto e fallito.

3. IL COSTO DELLA PASSATA.  Lo stato companion e' `n*p` (629 per l'L), e il
   simulation smoother ci gira dentro A OGNI ITERAZIONE MCMC.  E' letteralmente
   il 629 che compare nella companion di `lbvar.build_state_space`
   (`A[:n] = B[:n*p, :].T`).  Da qui le decine di ore della valutazione real-time
   del Gate 6, e il fatto che B e L pesino per due terzi del totale.

E LO SANNO GLI AUTORI.  In nota 9 (p. 503) dichiarano di aver scelto p=5 invece
di p=10 — pur avendo verificato che con p=10 i risultati sono qualitativamente
simili — «to ensure that the monthly models that are consistent with it, and the
L-BVAR in particular, are not too computationally burdensome».  Il p mensile
DISCENDE da quello trimestrale (nota 10, vedi `LAGS` sotto): con p=10 trimestrali
l'L sarebbe andato a p=32, cioe' una companion da 1 184 dimensioni con n=37.  La
scelta di specificazione e' stata pagata al costo computazionale, e questo
paragrafo e' il posto in cui i due fatti si toccano.


Riferimento
-----------
Cimadomo, Giannone, Lenza, Monti & Sokol (2022), "Nowcasting with large
Bayesian vector autoregressions", J. Econometrics 231, 500-519, §2.1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: La singola fonte di verita' per dati e prior del BVAR.
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "bvar_series.json")

#: I quattro modelli del paper.  Q e C condividono lo stesso profilo di
#: variabili perche' il Q-BVAR *e'* lo stadio 1 del C-BVAR (paper, nota 20).
MODELS: dict[str, str] = {
    "Q": "q_b",   # §2.1  baseline trimestrale
    "C": "q_b",   # §2.4  cube root: parte dal Q-BVAR
    "B": "q_b",   # §2.3  blocking / stacking
    "L": "l",     # §2.2  latent monthly VAR
}

#: Numero di lag per modello.  Q/C/B sono trimestrali con p=5 (paper §2.1 e
#: §2.3); l'L-BVAR e' mensile con p=17.  La nota 10 del paper spiega il 17:
#: serve a dare all'L-BVAR lo stesso insieme informativo di B e C, che con 5
#: lag trimestrali arrivano indietro fino a 17 mesi.
LAGS: dict[str, int] = {"Q": 5, "C": 5, "B": 5, "L": 17}


# ─── 1. Il prior Minnesota ────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class MinnesotaSpec:
    r"""
    La forma del prior Minnesota sui coefficienti autoregressivi.

    Teoria (Cimadomo §2.1, eq. 2-3)
    -------------------------------
    Condizionatamente a Sigma_eps, il prior sui coefficienti e' Normale con

        E(A_1) = diag(d),   E(A_2) = ... = E(A_p) = 0                   (eq. 2)

        Cov[(A_s)_ij, (A_r)_hm | Sigma_eps]
              = lambda^2 * Sigma_eps,ih / (s^2 * Psi_jj)                (eq. 3)
          se m = j e r = s, zero altrimenti.

    Cioe': il primo lag di ogni variabile su se stessa e' centrato su `d_i`
    (1 = random walk, 0 = white noise), tutto il resto su zero; e la varianza
    del prior si stringe come 1/s^2 al crescere del lag s.

    Parametri
    ---------
    p : int
        Numero di lag.  5 per Q/C/B (trimestrali), 17 per L (mensile).
    d_centre : tuple[int, ...]
        Il vettore `d` dell'eq. (2), lungo n: 1 = random walk, 0 = white noise.
        COSTRUITO DALLA CONFIG (campo `minnesota_centre`), mai a mano.
        Tenuto come tuple e non come np.ndarray perche' il dataclass e'
        `frozen`: una tuple e' immutabile davvero, un array no.  Usa la
        proprieta' `.d` per averlo come array di float pronto per l'algebra.
    lag_decay : float
        L'esponente del decadimento 1/s^lag_decay.  Il paper lo fissa a 2
        (nota 8: "As it is standard in the BVAR literature, we set the
        parameter governing this decay, s, to 2").  Non toccarlo senza motivo.
    use_soc : bool
        Sum-of-coefficients prior (Doan, Litterman & Sims 1984), governato da
        `mu`.  Il paper lo usa sempre: default True.
    use_dio : bool
        Dummy-initial-observation prior (Sims 1993), governato da `delta`.
        **Default False, ed e' una decisione deliberata.**  Cimadomo §2.1 usa
        solo Minnesota + sum-of-coefficients ed elenca TRE iperparametri
        (lambda, psi, mu), non quattro; la Tabella B.1 riporta solo lambda e
        mu.  Restiamo fedeli alla fonte-verita'.  GLP (2015) §III invece lo
        include: la porta resta aperta a costo zero, si accende qui senza
        cambiare nessuna firma.  Vedi `notes.dummy_initial_observation` nella
        config.

    Nota su `eq=False`
    ------------------
    Il confronto fra dataclass che contengono array/tuple lunghe non serve a
    niente qui e genererebbe solo ambiguita'.  Disattivato.
    """

    p: int
    d_centre: tuple[int, ...]
    lag_decay: float = 2.0
    use_soc: bool = True
    use_dio: bool = False

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError(f"p deve essere >= 1, ricevuto {self.p}")
        if not self.d_centre:
            raise ValueError("d_centre e' vuoto")
        bad = sorted({v for v in self.d_centre} - {0, 1})
        if bad:
            raise ValueError(
                f"d_centre ammette solo 0 (white noise) e 1 (random walk); "
                f"trovati anche {bad}"
            )
        if self.lag_decay <= 0:
            raise ValueError(f"lag_decay deve essere > 0, ricevuto {self.lag_decay}")

    @property
    def n(self) -> int:
        """Numero di variabili del sistema."""
        return len(self.d_centre)

    @property
    def k(self) -> int:
        """
        Coefficienti per equazione: n*p + 1 (l'1 e' la costante).

        E' la dimensione delle righe di `B`, che e' (k, n).  Il perche' questo
        numero governi prior, mescolamento e costo della passata sta nell'header
        del modulo, sezione «k = n*p + 1».
        """
        return self.n * self.p + 1

    @property
    def d(self) -> np.ndarray:
        """`d_centre` come array (n,) di float, pronto per `np.diag(d)`."""
        return np.asarray(self.d_centre, dtype=float)


# ─── 2. Gli iperparametri ─────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Hyper:
    r"""
    Un punto nello spazio degli iperparametri: gamma = (lambda, mu, psi[, delta]).

    Teoria (Cimadomo §2.1, GLP 2015 §II-III)
    ----------------------------------------
    Il punto dell'approccio gerarchico di Giannone, Lenza & Primiceri (2015),
    che Cimadomo adotta, e' che questi NON sono costanti da fissare a occhio ma
    variabili casuali da estrarre dal loro posterior.  Cimadomo §2.1: "As in
    Giannone et al. (2015), we treat these hyperparameters as random variables,
    and we draw them from their posterior distributions."

    Il campionatore vero e' materia del Gate 1 (`bvar/hyper.py`); qui c'e' solo
    il contenitore, perche' e' l'oggetto che il core sampler si passa a ogni
    iterazione.

    Parametri
    ---------
    lam : float
        Tightness complessiva del Minnesota.  Cimadomo §2.1: "The key
        hyperparameter is lambda, which controls the scale of all prior
        variances and covariances, and effectively determines the overall
        tightness of the prior.  For lambda = 0 the posterior equals the prior
        and the data do not influence the estimates.  If lambda -> infinity,
        posterior expectations coincide with the OLS estimates."
        Banda di plausibilita' dal paper (Tab. B.1): ~0.59-0.75.
    mu : float
        Intensita' del sum-of-coefficients.  Banda dal paper: ~0.97-1.72.
    psi : np.ndarray, shape (n,)
        La diagonale della matrice di scala Psi dell'Inverse-Wishart su
        Sigma_eps.  E' QUI che entra la scala delle variabili: e' il motivo per
        cui i dati NON vanno standardizzati a monte (vedi
        `notes.perche_i_livelli_e_non_le_differenze` nella config).
    delta : float | None
        Dummy-initial-observation.  `None` nel default, coerente con
        `MinnesotaSpec.use_dio = False`.

    Nota sull'immutabilita'
    -----------------------
    `psi` e' un array e il dataclass e' `frozen`: si puo' bloccare il binding
    ma non il contenuto.  `__post_init__` mette quindi il flag di sola lettura
    sull'array, cosi' un `hyper.psi[0] = ...` per sbaglio dentro il ciclo
    Metropolis fallisce subito invece di corrompere in silenzio le estrazioni.
    """

    lam: float
    mu: float
    psi: np.ndarray
    delta: float | None = None

    def __post_init__(self) -> None:
        psi = np.asarray(self.psi, dtype=float)
        if psi.ndim != 1:
            raise ValueError(f"psi deve essere 1-D (n,), ricevuto shape {psi.shape}")
        if not np.all(psi > 0):
            raise ValueError("psi deve essere strettamente positivo (e' una varianza)")
        if self.lam <= 0:
            raise ValueError(f"lam deve essere > 0, ricevuto {self.lam}")
        if self.mu <= 0:
            raise ValueError(f"mu deve essere > 0, ricevuto {self.mu}")
        if self.delta is not None and self.delta <= 0:
            raise ValueError(f"delta deve essere > 0 o None, ricevuto {self.delta}")
        psi = psi.copy()
        psi.setflags(write=False)
        object.__setattr__(self, "psi", psi)

    @property
    def n(self) -> int:
        return int(self.psi.shape[0])


# ─── 3. Il modello completo ───────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class BVARSpec:
    """
    Un modello BVAR completo: quali serie, che campione, che prior.

    E' l'oggetto che `bvar/data.py` consuma per costruire il pannello e che dal
    Gate 1 in poi il core sampler consuma per costruire le dummy observations.

    Parametri
    ---------
    model : str
        'Q' | 'C' | 'B' | 'L'.
    profile : str
        'q_b' | 'l'.  Deriva da `model` via `MODELS`; vedi
        `notes.perche_due_profili` nella config per il perche' teorico.
    series : tuple[str, ...]
        Gli id delle serie, nell'ordine delle colonne del pannello.
    freq : tuple[str, ...]
        'M' | 'Q' per ciascuna serie, stesso ordine.
    transform : tuple[str, ...]
        'log' | 'level' per ciascuna serie, stesso ordine.
    sample_start : pd.Timestamp
        Inizio del campione di stima (month-end).
    minnesota : MinnesotaSpec
    """

    model: str
    profile: str
    series: tuple[str, ...]
    freq: tuple[str, ...]
    transform: tuple[str, ...]
    sample_start: pd.Timestamp
    minnesota: MinnesotaSpec
    config: dict = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.series)
        for name, seq in (("freq", self.freq), ("transform", self.transform)):
            if len(seq) != n:
                raise ValueError(
                    f"{name} ha lunghezza {len(seq)} ma le serie sono {n}"
                )
        if self.minnesota.n != n:
            raise ValueError(
                f"MinnesotaSpec.d_centre ha lunghezza {self.minnesota.n} "
                f"ma le serie sono {n}"
            )

    # ── proprieta' di comodo ────────────────────────────────────────────────
    @property
    def n(self) -> int:
        """Numero di variabili."""
        return len(self.series)

    @property
    def p(self) -> int:
        return self.minnesota.p

    @property
    def k(self) -> int:
        """Coefficienti per equazione, n*p + 1."""
        return self.minnesota.k

    @property
    def monthly(self) -> tuple[str, ...]:
        return tuple(s for s, f in zip(self.series, self.freq) if f == "M")

    @property
    def quarterly(self) -> tuple[str, ...]:
        return tuple(s for s, f in zip(self.series, self.freq) if f == "Q")

    @property
    def log_series(self) -> tuple[str, ...]:
        return tuple(s for s, t in zip(self.series, self.transform) if t == "log")

    @property
    def level_series(self) -> tuple[str, ...]:
        return tuple(s for s, t in zip(self.series, self.transform) if t == "level")

    # ── il costruttore vero ─────────────────────────────────────────────────
    @classmethod
    def from_config(
        cls,
        model: str,
        config: dict | str | None = None,
        *,
        p: int | None = None,
        sample_start=None,
        use_dio: bool = False,
    ) -> "BVARSpec":
        """
        Costruisce la specifica di un modello leggendo la config.

        E' l'UNICO modo previsto di ottenere un `BVARSpec`: garantisce che
        `d_centre`, l'ordine delle colonne e la mappa log/livello vengano dal
        file e non da un array scritto a mano.

        Parameters
        ----------
        model : str
            'Q' | 'C' | 'B' | 'L'.
        config : dict | str | None
            La config gia' caricata, un path, o None per il default
            (`config/bvar_series.json`).
        p : int | None
            Override del numero di lag.  None = il default del modello
            (5 per Q/C/B, 17 per L).
        sample_start : optional
            Override dell'inizio campione.  None = quello del profilo.
        use_dio : bool
            Accende il dummy-initial-observation.  Default False: vedi
            `MinnesotaSpec.use_dio`.
        """
        model = model.upper()
        if model not in MODELS:
            raise ValueError(f"modello ignoto {model!r}; attesi {sorted(MODELS)}")

        cfg = load_config(config)
        profile = MODELS[model]
        prof = cfg["profiles"][profile]
        wanted = set(prof["series_ids"])

        # L'ordine delle colonne e' quello di `cfg['series']`, non quello di
        # `series_ids`: cosi' i due profili condividono lo stesso ordinamento e
        # una colonna significa la stessa cosa in entrambi.
        chosen = [e for e in cfg["series"] if e["series_id"] in wanted]
        if len(chosen) != len(wanted):
            missing = wanted - {e["series_id"] for e in chosen}
            raise ValueError(f"serie del profilo {profile!r} assenti dalla config: {missing}")

        minn = MinnesotaSpec(
            p=LAGS[model] if p is None else p,
            d_centre=tuple(1 if e["minnesota_centre"] == "rw" else 0 for e in chosen),
            use_dio=use_dio,
        )
        return cls(
            model=model,
            profile=profile,
            series=tuple(e["series_id"] for e in chosen),
            freq=tuple(e["freq"] for e in chosen),
            transform=tuple(e["transform"] for e in chosen),
            sample_start=pd.Timestamp(
                prof["sample_start"] if sample_start is None else sample_start
            ),
            minnesota=minn,
            config=cfg,
        )

    def restrict(self, series) -> "BVARSpec":
        """
        Lo stesso modello su un SOTTOINSIEME di serie, nell'ordine originale.

        Affetta insieme `series`, `freq`, `transform` e `d_centre`, che devono
        restare allineati: e' l'unico modo previsto di ridurre un profilo, e
        serve al real-time dell'L-BVAR, dove l'insieme delle serie disponibili
        dipende dalla data (`data.drop_empty_series`).

        L'ordine e' quello dello spec di partenza, non quello di `series`: una
        colonna deve significare la stessa cosa prima e dopo.
        """
        want = set(series)
        keep = [i for i, s in enumerate(self.series) if s in want]
        if len(keep) != len(want):
            mancanti = want - set(self.series)
            raise ValueError(f"serie non presenti nello spec: {sorted(mancanti)}")
        if not keep:
            raise ValueError("il sottoinsieme e' vuoto")
        pick = lambda seq: tuple(seq[i] for i in keep)              # noqa: E731
        return replace(
            self,
            series=pick(self.series),
            freq=pick(self.freq),
            transform=pick(self.transform),
            minnesota=MinnesotaSpec(
                p=self.minnesota.p,
                d_centre=pick(self.minnesota.d_centre),
                lag_decay=self.minnesota.lag_decay,
                use_soc=self.minnesota.use_soc,
                use_dio=self.minnesota.use_dio,
            ),
        )

    def summary(self) -> str:
        """Una riga di descrizione, per i log e per i test."""
        return (
            f"{self.model}-BVAR  profilo={self.profile}  n={self.n}  p={self.p}  "
            f"k={self.k}  campione da {self.sample_start.date()}  "
            f"({len(self.monthly)}M + {len(self.quarterly)}Q, "
            f"{len(self.log_series)} log + {len(self.level_series)} livello, "
            f"{int(self.minnesota.d.sum())} rw + "
            f"{int(self.n - self.minnesota.d.sum())} wn)"
        )


# ─── 4. Caricamento della config ──────────────────────────────────────────────

def load_config(config: dict | str | None = None) -> dict:
    """
    Carica `config/bvar_series.json` (o quello che gli si passa).

    Parameters
    ----------
    config : dict | str | None
        Un dict gia' caricato viene restituito com'e'; una stringa e' trattata
        come path; None usa `CONFIG_PATH`.
    """
    if isinstance(config, dict):
        return config
    path = CONFIG_PATH if config is None else str(config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config BVAR non trovata: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("schema") != "bvar-v1":
        raise ValueError(
            f"schema atteso 'bvar-v1', trovato {cfg.get('schema')!r} in {path}"
        )
    return cfg


__all__ = [
    "MinnesotaSpec",
    "Hyper",
    "BVARSpec",
    "load_config",
    "CONFIG_PATH",
    "MODELS",
    "LAGS",
]
