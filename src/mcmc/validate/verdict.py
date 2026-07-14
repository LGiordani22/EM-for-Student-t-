"""
mcmc/validate/verdict.py
========================

**Quattro esiti, non pass/fail.**  È la prima delle tre regole non negoziabili del
validatore, e nasce da un errore che abbiamo fatto davvero.

Un `assert` binario costringe a scegliere una soglia; e quando un parametro *non è
identificato*, qualunque soglia si scelga è arbitraria.  La tentazione — inevitabile —
è allargarla finché il test passa.  È esattamente così che l'attenuazione di ``rho^u``
(35% sul canale dominante) è rimasta invisibile per mesi **con la suite verde**: i test
asserivano il *segno* e l'*ordinamento*, mai la *magnitudine*, e l'unico che guardava il
valore usava una tolleranza abbastanza larga da non accorgersi di nulla.

Qui un parametro non identificato viene **dichiarato tale** — e il validatore resta
verde.  Il rosso è riservato a una cosa sola: **il sampler è rotto**.

Gli esiti
---------
``RECOVERED``          il parametro è confrontato con il **vero** e lo recupera.
``RECOVERED_BIASED``   lo recupera ma con un **bias noto e misurato**: la stima è
                       utilizzabile *sapendo di cosa*.  Non è un fallimento — è un
                       risultato, e va scritto in tesi.
``NOT_IDENTIFIED``     i dati **non lo pinzano**.  Non è un difetto del campionatore: è
                       quanto segnale c'è.  Il check afferma il *limite* (per esempio:
                       «il CI copre lo zero ed entrambi i segni») e **passa**
                       asserendolo.
``UNTESTED``           nessun check lo copre ancora.  Compare in tabella come **riga
                       vuota dichiarata**, non come assenza silenziosa.
``BROKEN``             🔴 **l'unico rosso**: il campionatore fa una cosa sbagliata (una
                       legge invariante spostata, un guard che non scatta, un'identità
                       algebrica violata).  Va riparato, non annotato.

Regola d'oro
------------
Se, scrivendo un check, ti trovi a **rilassare una tolleranza per farlo passare**:
fermati.  O il verdetto giusto è ``NOT_IDENTIFIED`` / ``RECOVERED_BIASED`` — e allora
dillo — oppure il sampler è ``BROKEN``.  Allargare la soglia è il modo in cui un bug di
identificazione torna a nascondersi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    RECOVERED = "recuperato"
    RECOVERED_BIASED = "recuperato con bias noto"
    NOT_IDENTIFIED = "non identificato"
    UNTESTED = "mai testato"
    BROKEN = "ROTTO"

    @property
    def is_red(self) -> bool:
        """Il rosso e' UNO SOLO: il sampler sbaglia.  Un parametro non identificato non
        e' un fallimento del codice — e' una proprieta' del problema."""
        return self is Outcome.BROKEN

    @property
    def mark(self) -> str:
        return {
            Outcome.RECOVERED: "OK ",
            Outcome.RECOVERED_BIASED: "BIAS",
            Outcome.NOT_IDENTIFIED: "N.I.",
            Outcome.UNTESTED: "--- ",
            Outcome.BROKEN: "ROTTO",
        }[self]


@dataclass
class Verdict:
    """Una riga della tabella finale: **un parametro, un contesto, un verdetto**.

    ``ess`` / ``r_hat`` non sono decorativi.  Seconda regola non negoziabile: **nessuna
    stima senza il suo ESS e il suo R-hat**.  Un ``rho_hat`` senza ESS non e' una
    misura — e infatti il ``rho_hat = -0.51`` che abbiamo creduto per settimane veniva
    da catene con ESS 3-23 su 2000 draw.
    """
    param: str                     # simbolo, es. "rho^u_0"
    block: str                     # "fattori" | "osservazioni"
    branch: str                    # "A" | "B" | "A+B" | "-"
    outcome: Outcome
    reason: str                    # la RAGIONE, in prosa: perche' questo verdetto
    truth: float | None = None
    estimate: float | None = None
    ci: tuple[float, float] | None = None
    ess: float | None = None
    r_hat: float | None = None
    coverage: float | None = None  # frazione di repliche il cui CI copre il vero
    n_reps: int | None = None
    mode: str = "quick"            # in quale modalita' e' stato prodotto
    detail: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float | None:
        if self.truth in (None, 0.0) or self.estimate is None:
            return None
        return self.estimate / self.truth

    #: Sotto queste soglie la catena **non ha esplorato**, e nessuna sua media e' una
    #: misura.  Non e' pedanteria: la prima run --full ha dichiarato ``Q`` "recuperato"
    #: sulla base di un errore relativo calcolato da una catena con **R-hat = 2.3 ed ESS
    #: = 6**.  Un numero del genere non e' una stima sbagliata: **non e' una stima**.
    ESS_MIN = 30.0
    R_HAT_MAX = 1.15

    def __post_init__(self):
        # Terza regola, applicata al tipo: un verdetto "recuperato" su un parametro
        # delicato non puo' poggiare su un solo dataset.  Non lo VIETIAMO qui (sarebbe
        # una trappola), ma lo RENDIAMO VISIBILE: la tabella stampa la modalita', e un
        # OK prodotto in --quick porta la sua etichetta con se'.
        if self.outcome is Outcome.RECOVERED and self.mode == "quick":
            self.detail.setdefault("caveat", "verdetto da --quick: non probante")

        # ── IL CANCELLO DI CONVERGENZA ────────────────────────────────────────
        # Un verdetto POSITIVO (recuperato / bias noto) afferma qualcosa sulla posterior.
        # Se la catena non e' convergente su quel parametro, quell'affermazione non ha
        # fondamento — e va sospesa, non annacquata.  Il verdetto diventa "non
        # identificato" con una ragione che distingue esplicitamente le due cose, perche'
        # sono diverse e la differenza conta:
        #   * "il DATO non lo pinza"      -> limite informativo, nessun rimedio nel sampler;
        #   * "la CATENA non converge"    -> limite del CAMPIONATORE, e un rimedio esiste.
        positive = self.outcome in (Outcome.RECOVERED, Outcome.RECOVERED_BIASED)
        bad_ess = self.ess is not None and self.ess < self.ESS_MIN
        bad_rhat = self.r_hat is not None and self.r_hat > self.R_HAT_MAX
        if positive and (bad_ess or bad_rhat):
            self.detail["suspended_from"] = self.outcome.value
            self.detail["ess"] = self.ess
            self.detail["r_hat"] = self.r_hat
            self.outcome = Outcome.NOT_IDENTIFIED
            self.reason = (
                f"**VERDETTO SOSPESO — la CATENA non converge su questo parametro** "
                f"(ESS {self.ess:.0f}, R-hat {self.r_hat:.2f}). Attenzione: NON e' il dato "
                f"che non lo pinza, e' il campionatore che non lo esplora — sono due cose "
                f"diverse e il rimedio e' diverso. Qualunque media a posteriori calcolata "
                f"da questa catena non e' una stima sbagliata: **non e' una stima**. "
                f"(Sarebbe stato: {self.detail['suspended_from']} — {self.reason})")
