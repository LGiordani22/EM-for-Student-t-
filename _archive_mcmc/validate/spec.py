"""
mcmc/validate/spec.py
=====================

**L'inventario dei parametri come DATO** — la §B di ``docs/MCMC_ENGINES_AND_TESTS.md``,
resa eseguibile.  Ogni riga della tabella finale nasce da una riga di qui: se un
parametro è in ``PARAMS`` e nessun check lo tocca, compare come ``UNTESTED``.  Un buco
non può più essere un'assenza silenziosa.

---

Il modello, in breve — la teoria che serve per leggere la tabella
----------------------------------------------------------------

**Specification II (volatilità *outside*).**  La covarianza condizionale degli shock
dei fattori è

    Var(u_t | ·) = sqrt(H^u_t) · Q · sqrt(H^u_t) / w^u_t ,   H^u_t = diag(h^u_{1,t}, …, h^u_{r,t})

cioè la volatilità sta **fuori** dalla radice di ``Q``.  La conseguenza è la ragione
della scelta: ``h^u_k`` è la volatilità **del fattore k**, un oggetto interpretabile
(«quanto è volatile il fattore reale», «quanto il finanziario»), non la volatilità di una
direzione ortogonalizzata che cambia significato ogni volta che ``Q`` si muove.  L'altra
collocazione (*inside*, ``sqrt(Q) H sqrt(Q)``) darebbe volatilità di direzioni ruotate,
inutili da leggere.  Le due coincidono solo sotto la restrizione ``H = h·I``, che **non**
è "Spec I": è un caso particolare di entrambe.

**Option A (leverage sugli shock grezzi).**  Il leverage accoppia l'innovazione di
log-volatilità del fattore ``k`` al **suo** shock di livello, misurato sullo shock
*grezzo* pienamente sbiancato

    z^u_t = sqrt(w) · Q^{-1/2} · (sqrt H^u_t)^{-1} · u_t   ~ N(0, I),

con ``r`` correlazioni **scalari** ``rho^u_k`` (una per canale), non un vettore né una
direzione dominante.  Sotto Spec II questa è l'unica forma in cui i ``(phi_k, sigma_k,
rho_k)`` per fattore siano separatamente identificati.

**Branch A (contemporaneo) e Branch B (laggato)** — non sono due approssimazioni della
stessa cosa: sono due **timing** diversi, cioè due modelli.
* **A**: ``eta_t`` (che guida ``log h_t``) correla con ``z_t`` — lo shock *dello stesso*
  periodo.  Il drift ``rho·sigma·z_t`` contiene ``exp(-x_t/2)``, quindi la transizione
  **non è lineare-gaussiana** e non esiste FFBS: il path si estrae per Metropolis.  Il
  target è però **esatto** — nessuna mistura, nessuna linearizzazione, la ``Q`` piena
  trattata esattamente.  Per questo A è la **controparte esatta**, il metro contro cui si
  misura B.
* **B**: ``eta_{t+1}`` correla con ``z_t`` (Yu 2005).  L'identità del segno
  ``z_t = d_t · exp(xi_t/2)`` più la mistura a 10 componenti di Omori rendono il sistema
  condizionatamente lineare-gaussiano ⇒ il path è un **draw FFBS diretto**, accettazione
  1 per costruzione.  Il prezzo è la **linearizzazione**.

**Le famiglie di full-conditional** (da cui viene recuperato ciascun parametro):

| Famiglia | Parametri | Kernel |
|---|---|---|
| **A**  | ``A``, ``Q`` (+ ``hw_a``) | two-step per-fattore / MNIW; IW o Huang–Wand |
| **A'** | ``Lambda``, ``R``         | NIG per equazione |
| **B**  | ``phi``, ``sigma^2_eta``  | Gauss + Metropolis leggero (il fattore ``1-rho^2`` rende ``sigma^2`` non coniugato) |
| **C**  | ``rho``                   | griddy-Gibbs sul supporto compatto ``(-1,1)`` (default) o RW |
| **D**  | ``nu``, ``w``             | griddy-Gibbs per ``nu``; Gamma per i pesi |

``mu`` (media della log-vol) **non è un parametro stimato**: è fissato a 0 per
identificazione — con ``mu`` libero la scala di ``h`` e quella di ``Q`` non sarebbero
separabili.  Compare in tabella come ``FIXED`` perché la sua *assenza* è una scelta, e
va dichiarata.
"""

from __future__ import annotations

from dataclasses import dataclass


FACTORS = "fattori"
OBS = "osservazioni"


@dataclass(frozen=True)
class ParamSpec:
    key: str            # chiave stabile, usata dai check e dalla tabella
    tex: str            # simbolo nel .tex
    code: str           # dove vive nel codice
    block: str          # FACTORS | OBS
    family: str         # "A" | "A'" | "B" | "C" | "D" | "-"
    where: str          # da quale funzione e' campionato
    note: str = ""


#: L'inventario.  **Esaustivo**: se un parametro esiste nel modello e non e' qui, la
#: tabella non lo vedra' mai — quindi questa lista e' il contratto.
PARAMS: list[ParamSpec] = [
    # ── Blocco FATTORI / STATI (shock u_t) ───────────────────────────────────
    ParamSpec("f", "f_t", 'draws["F"]', FACTORS, "-",
              "sample_states.ffbs_sample_states", "stati latenti (blocco (a))"),
    ParamSpec("A", "A", 'draws["A"]', FACTORS, "A",
              "shared.draw_A_Q_perfactor (SV) / sample_params.draw_A_Q_block (no-SV)"),
    ParamSpec("Q", "Q", 'draws["Q"]', FACTORS, "A",
              "idem", "IW, oppure Huang-Wand con le r scale ausiliarie hw_a"),
    ParamSpec("mu_u", "mu^u_k", "sv_u[:, 0]", FACTORS, "-",
              "NON CAMPIONATO", "fissato a 0: identificazione (eq:sv-mu-identification)"),
    ParamSpec("phi_u", "phi^u_k", "sv_u[:, 1]", FACTORS, "B",
              "_draw_phi_lev / draw_ar1_params", "r canali"),
    ParamSpec("sigma2_eta_u", "sigma^2_{eta,k}", "sv_u[:, 2]", FACTORS, "B",
              "_draw_sigma2_lev / draw_ar1_params", "r canali; vol-of-vol"),
    ParamSpec("rho_u", "rho^u_k", 'draws["rho_u"]', FACTORS, "C",
              "draw_rho (griddy)", "r canali scalari — Option A. DA' LO SKEW ALLA DENSITA'"),
    ParamSpec("h_u", "h^u_k", 'draws["h_u"]', FACTORS, "-",
              "blocco (b)", "path latente, r canali"),
    ParamSpec("w_u", "w^u_t", "w_u", FACTORS, "D",
              "shared.draw_weights", "scalare per periodo (condiviso fra i fattori)"),
    ParamSpec("nu_u", "nu_u", 'draws["nu_u"]', FACTORS, "D",
              "sample_params.draw_nu_griddy"),

    # ── Blocco OSSERVAZIONI / IDIOSINCRATICO (shock eps_t) ────────────────────
    ParamSpec("Lambda", "Lambda", 'draws["Lambda"]', OBS, "A'",
              "sample_params.draw_Lambda_R_block"),
    ParamSpec("R", "R", 'draws["R"]', OBS, "A'",
              "idem", "diagonale"),
    ParamSpec("mu_eps", "mu^eps_i", "sv_eps[:, 0]", OBS, "-",
              "NON CAMPIONATO", "fissato a 0"),
    ParamSpec("phi_eps", "phi^eps_i", "sv_eps[:, 1]", OBS, "B",
              "_draw_phi_lev / draw_ar1_params", "M canali — UNO PER SERIE"),
    ParamSpec("sigma2_eta_eps", "sigma^2_{eps,i}", "sv_eps[:, 2]", OBS, "B",
              "_draw_sigma2_lev / draw_ar1_params", "M canali"),
    ParamSpec("rho_eps", "rho^eps_i", 'draws["rho_eps"]', OBS, "C",
              "draw_rho (griddy)",
              "M canali — IL LEVERAGE IDIOSINCRATICO ESISTE, e non era mai stato testato"),
    ParamSpec("h_eps", "h^eps_i", 'draws["h_eps"]', OBS, "-",
              "blocco (b)", "path latente, M canali; spento da sv_idio=False (D2-a)"),
    ParamSpec("w_eps", "w^eps_t", "w_eps", OBS, "D",
              "shared.draw_weights", "scalare per periodo (condiviso fra le serie)"),
    ParamSpec("nu_eps", "nu_eps", 'draws["nu_eps"]', OBS, "D",
              "sample_params.draw_nu_griddy", "distinto da nu_u"),
]

BY_KEY: dict[str, ParamSpec] = {p.key: p for p in PARAMS}


def params_of(block: str) -> list[ParamSpec]:
    return [p for p in PARAMS if p.block == block]
