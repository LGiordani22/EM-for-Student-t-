"""
mcmc/validate/checks/variants.py
================================

Le due varianti algoritmiche — **ASIS** e **QML** — che nessun test copriva in modo
sistematico.

ASIS
----
**Cosa riparametrizza.**  La cresta *path ↔ scala*: il path ``x_t = log h_t`` e la sua
scala ``sigma_eta`` sono fortemente correlati a posteriori, e ``(phi, sigma^2_eta)``
mescolano male di conseguenza.  ASIS alterna due parametrizzazioni della stessa coppia:
**CP** (si estrae ``(phi, sigma^2)`` dato il path) e **NCP** (si standardizza
``x̃ = x/sigma_eta``, così ``sigma_eta`` finisce *nell'equazione di misura* e si ri-estrae
come coefficiente di regressione **con segno**; poi il path viene riscalato).

**Il confondimento (`gibbs.py:498`).**  ``use_asis=True`` **forza**
``sv_sigma_prior="half_normal"``.  I due flag **non sono ortogonali**, e ogni esperimento
che vari ``use_asis`` sta variando **anche il prior**.

Ma — e qui va corretta la premessa naturale — **non è (solo) un bug: è un vincolo
matematico.**  Perché l'interweaving campioni il giunto esatto, CP e NCP devono esprimere
lo **stesso** prior su ``sigma_eta``; e il gaussiano sul ``sigma_eta`` *segnato* è ciò che
rende **coniugato** il draw NCP.  Renderli ortogonali *tout court* romperebbe ASIS.

Due vie, e adottiamo la seconda:
1. **Ortogonalizzazione vera** — implementare il draw NCP **non coniugato** sotto IG (un
   passo di Metropolis in più, ~30 righe).  Solo così i flag diventano indipendenti.
   **Non implementata**, documentata qui come opzione.
2. **Disciplina sperimentale** — tenere il vincolo (che è corretto) e **vietare** agli
   esperimenti di variare entrambi.  :func:`_guard_asis_comparison` fa **fallire il
   validatore** se un confronto su ASIS varia anche il prior.

Il confondimento ha già prodotto un errore reale: la conclusione «ASIS aiuta ``rho``» era
in realtà **il prior** che aiutava ``rho``.  E quel prior, ora lo sappiamo, **gonfia
``sigma^2_eta``**.

**Correttezza ≠ efficienza**, e vanno testate separatamente: ASIS on/off devono campionare
**lo stesso target** (se divergono, ASIS è *rotto*); e poi, separatamente, deve **alzare
l'ESS** di ``phi``/``sigma^2`` (se non lo alza, è inutile — non rotto).

QML
---
**Test di dominio di validità, non di successo.**  La QML sostituisce la mistura con UNA
gaussiana a covarianza costante esatta ``(pi^2/2) R_xi``, il che è ciò che permette a una
``R_xi`` **piena** di passare (la mistura non la fa passare: non fattorizza sugli
indicatori — ed è il motivo per cui la forma ``literal`` è instabile).

* **senza leverage**: stabile a ``corr(Q)`` alta ⇒ ✓
* **sotto leverage**: la versione corretta (``z = M·eps`` esatto,
  ``M = Q^{-1/2} diag(sqrt q_kk)``, transizione FFBS piena) **recupera ``rho``** — a
  ``corr(Q)=0.8`` dà ``rho_hat ≈ -0.47`` contro il ``-0.15`` del decoupled ⇒ **P5
  eliminata** — ma ``phi`` di un fattore **collassa ancora**.  **Progresso, non
  soluzione** ⇒ verdetto *recuperato con bias noto*, con entrambi i fatti congelati.
* **sotto Branch A**: solleva ``ValueError`` **per costruzione** — A non forma mai una
  covarianza di misura (nessuna mistura, nessuna linearizzazione).  **Non è una lacuna, è
  una proprietà**, e va asserita come tale invece di lasciare la cella vuota.
* ``recommend_coupling``: la soglia è sulla **sovra-confidenza** (≤5% ⇒ ``decoupled``),
  non su ``corr(Q)``.  Sul pannello reale (0.4%) ⇒ ``decoupled``.
"""

from __future__ import annotations

import numpy as np

from mcmc.diagnostics import recommend_coupling
from mcmc.gibbs import fit_dfm_mcmc
from mcmc.validate import dgp as D
from mcmc.validate.checks import say, summarize
from mcmc.validate.verdict import Outcome, Verdict


def _guard_asis_comparison(cells: list[tuple[bool, str]]) -> str | None:
    """**La disciplina sperimentale, resa eseguibile.**  Un confronto su ASIS e' valido
    solo **a prior fissato**.  Se le celle da confrontare variano ``use_asis`` *e*
    ``sv_sigma_prior`` insieme, il risultato e' ininterpretabile — ed e' esattamente
    l'errore che ci ha fatto credere per settimane che ASIS aiutasse ``rho``."""
    asis = {c[0] for c in cells}
    prior = {c[1] for c in cells}
    if len(asis) > 1 and len(prior) > 1:
        return ("confronto INVALIDO: varia use_asis E sv_sigma_prior insieme. "
                "Il forzamento in gibbs.py:498 li lega; ogni confronto su ASIS va fatto "
                "A PRIOR FISSATO.")
    return None


def _asis_cells(mode: D.Mode, branch: str) -> list[Verdict]:
    g = D.build(mode, branch=branch)
    V: list[Verdict] = []

    # Le quattro celle.  NB: (asis=True, IG) e' IRRAGGIUNGIBILE — gibbs forza half_normal.
    # Non e' un bug del validatore: e' il vincolo.  Lo registriamo invece di fingere.
    cells = {}
    for asis in (False, True):
        for prior in ("inverse_gamma", "half_normal"):
            eff_prior = "half_normal" if asis else prior     # cio' che gibbs FARA' davvero
            key = (asis, prior)
            if asis and prior == "inverse_gamma":
                cells[key] = None                            # cella irraggiungibile
                continue
            # (off, half_normal) e' il fit di DEFAULT: stesso seme => cache hit, gratis.
            nc = None if (not asis and prior == "half_normal") else 1
            ch = D.fit(g, seed=700, use_asis=asis, sv_sigma_prior=prior, n_chains=nc)
            sv = D.stack(ch, "sv_u")
            cells[key] = {
                "phi": summarize(sv[:, :, 0, 1]),
                "s2": summarize(sv[:, :, 0, 2]),
                "rho": summarize(D.stack(ch, "rho_u")[:, :, 0],
                                 truth=float(D.RHO_U_TRUE[0])),
                "eff_prior": eff_prior,
            }
            say(f"  asis={asis!s:<5} prior={prior:<14} -> phi {cells[key]['phi']['mean']:.3f} "
                f"(ESS {cells[key]['phi']['ess']:.0f})  "
                f"rho {cells[key]['rho']['mean']:+.3f} (ESS {cells[key]['rho']['ess']:.0f})")

    V.append(Verdict(
        "ASIS x prior (cella off-IG)", "fattori", branch, Outcome.NOT_IDENTIFIED,
        "la cella (use_asis=True, prior=IG) e' **IRRAGGIUNGIBILE**: gibbs.py:498 forza il "
        "half-Normal. Non e' un bug — CP e NCP devono esprimere lo STESSO prior perche' "
        "l'interweaving campioni il giunto esatto, e il gaussiano sul sigma segnato e' cio' "
        "che rende coniugato il draw NCP. Via 1 (draw NCP non coniugato sotto IG, ~30 righe) "
        "documentata e NON implementata; adottata la via 2: disciplina sperimentale.",
        mode=mode.name, detail={"spec_key": "_asis_cells"}))

    # (1) CORRETTEZZA — a prior FISSATO (half-Normal): on/off devono dare lo STESSO target
    off, on = cells[(False, "half_normal")], cells[(True, "half_normal")]
    err = _guard_asis_comparison([(False, "half_normal"), (True, "half_normal")])
    assert err is None, err
    d_phi = abs(off["phi"]["mean"] - on["phi"]["mean"])
    d_s2 = abs(off["s2"]["mean"] - on["s2"]["mean"]) / max(off["s2"]["mean"], 1e-9)
    broken = d_phi > 0.15 or d_s2 > 0.60
    V.append(Verdict(
        "ASIS correttezza", "fattori", branch,
        Outcome.BROKEN if broken else Outcome.RECOVERED,
        (f"a prior FISSATO (half-Normal), ASIS on/off campionano lo stesso target: "
         f"|d phi| = {d_phi:.3f}, |d sigma^2|/sigma^2 = {d_s2:.0%}. Se divergessero, ASIS "
         f"sarebbe ROTTO (una riparametrizzazione non puo' spostare la legge invariante)."),
        estimate=d_phi, mode=mode.name, detail={"spec_key": "_asis_correct"}))

    # (2) EFFICIENZA — separata dalla correttezza
    eff = lambda d: d["ess"] / max(d["n"], 1)      # ESS PER DRAW: l'unica misura
    gain_phi = eff(on["phi"]) / max(eff(off["phi"]), 1e-12)   # confrontabile fra fit con
                                                              # numero di catene diverso
    V.append(Verdict(
        "ASIS efficienza (ESS phi)", "fattori", branch,
        Outcome.RECOVERED if gain_phi > 1.15 else Outcome.RECOVERED_BIASED,
        (f"ESS(phi) x{gain_phi:.2f} con ASIS. E' cio' per cui ASIS e' costruito — la cresta "
         f"(phi, sigma^2). Non alzarlo lo renderebbe INUTILE, non rotto."),
        estimate=gain_phi, mode=mode.name, detail={"spec_key": "_asis_ess"}))

    # (3) ASIS NON e' la leva su rho: e' il PRIOR — a ASIS fissato (off), IG vs half-Normal
    err = _guard_asis_comparison([(False, "inverse_gamma"), (False, "half_normal")])
    assert err is None, err
    ig = cells[(False, "inverse_gamma")]
    hn = cells[(False, "half_normal")]
    gain_prior = eff(hn["rho"]) / max(eff(ig["rho"]), 1e-12)
    gain_asis = eff(on["rho"]) / max(eff(off["rho"]), 1e-12)

    # Un guadagno di ESS non e' misurabile da catene corte: l'ESS stesso ha un errore
    # standard enorme quando e' dell'ordine di 20.  In --quick questo confronto NON PUO'
    # pronunciarsi, e lo dice invece di fingere un verdetto.  (Tentazione da evitare:
    # allargare la soglia finche' il numero di --quick "conferma" cio' che sappiamo da
    # --full.  Sarebbe esattamente il vizio che questo validatore esiste per estirpare.)
    if not mode.is_probative:
        V.append(Verdict(
            "leva su rho: prior o ASIS?", "fattori", branch, Outcome.NOT_IDENTIFIED,
            (f"--{mode.name} NON PUO' decidere: un guadagno di ESS non e' misurabile da "
             f"catene corte (qui ESS ~ 20, con un errore standard dello stesso ordine). "
             f"I numeri grezzi — prior x{gain_prior:.2f}, ASIS x{gain_asis:.2f} — sono "
             f"rumore. Serve --full."),
            estimate=gain_prior, mode=mode.name,
            detail={"spec_key": "_asis_vs_prior", "gain_prior": gain_prior,
                    "gain_asis": gain_asis, "inconclusive": True}))
        return V

    V.append(Verdict(
        "leva su rho: prior, non ASIS", "fattori", branch,
        Outcome.RECOVERED if gain_prior > gain_asis else Outcome.RECOVERED_BIASED,
        (f"a ASIS FISSATO (off), cambiare il prior IG -> half-Normal moltiplica ESS(rho) per "
         f"x{gain_prior:.1f}. A prior FISSATO (half-Normal), accendere ASIS lo moltiplica per "
         f"x{gain_asis:.1f}. **La leva e' il prior, non l'interweaving** — la catena causale "
         f"'meglio sigma => meglio rho' del .tex e' falsificata. Ogni confronto qui e' fatto "
         f"A PRIOR FISSATO o A ASIS FISSATO, mai variando entrambi (il guard lo impone)."),
        estimate=gain_prior, mode=mode.name, detail={"spec_key": "_asis_vs_prior",
                                                     "gain_prior": gain_prior,
                                                     "gain_asis": gain_asis}))
    return V


def _qml(mode: D.Mode) -> list[Verdict]:
    V: list[Verdict] = []

    # (a) Branch A: solleva PER COSTRUZIONE — e' una proprieta', non una lacuna
    gA = D.build(mode, branch="A")
    raised = False
    try:
        fit_dfm_mcmc(gA["sim"]["Y"], gA["theta"], gA["w"]["freq_list"],
                     gA["w"]["block_map"], gA["w"]["ordered_cols"], n_iter=2, burn_in=0,
                     sv=True, leverage=True, timing="contemporaneous",
                     common_vol_coupling="qml", allow_experimental=True, verbose=False)
    except ValueError:
        raised = True
    V.append(Verdict(
        "QML sotto Branch A", "fattori", "A",
        Outcome.RECOVERED if raised else Outcome.BROKEN,
        ("solleva ValueError **per costruzione**: Branch A non forma MAI una covarianza di "
         "misura (nessuna mistura, nessuna linearizzazione; usa Q^{-1/2} pieno in modo "
         "esatto), quindi non c'e' nulla da accoppiare. **Non e' una lacuna, e' una "
         "proprieta'** — ed e' la ragione per cui A e' la controparte esatta."),
        mode=mode.name, detail={"spec_key": "_qml_A"}))

    # (b) recommend_coupling: la soglia e' sulla SOVRA-CONFIDENZA, non su corr(Q)
    g = D.build(mode, branch="B")
    rec_diag = recommend_coupling(np.asarray(g["theta"]["Q"], float))
    Qc = D.build(mode, branch="B", correlated_Q=0.8)["theta"]["Q"]
    rec_corr = recommend_coupling(np.asarray(Qc, float))
    ok = rec_diag["recommend"] == "decoupled" and rec_corr["recommend"] == "qml"
    V.append(Verdict(
        "recommend_coupling", "fattori", "B",
        Outcome.RECOVERED if ok else Outcome.BROKEN,
        (f"la soglia e' sulla SOVRA-CONFIDENZA indotta (<=5%), non su corr(Q): a Q diagonale "
         f"{rec_diag['overconfidence']:.1%} -> '{rec_diag['recommend']}'; a corr(Q)=0.8 "
         f"{rec_corr['overconfidence']:.0%} -> '{rec_corr['recommend']}'. Sul pannello reale "
         f"(~0.4%) => decoupled."),
        mode=mode.name, detail={"spec_key": "_qml_rec"}))

    # (c) Sotto leverage a corr(Q)=0.8: PROGRESSO e COLLASSO RESIDUO, entrambi congelati
    gq = D.build(mode, branch="B", correlated_Q=0.8)
    out = {}
    for cp in ("decoupled", "qml"):
        kw = {"allow_experimental": True} if cp == "qml" else {}
        ch = D.fit(gq, seed=700, common_vol_coupling=cp, n_chains=1, **kw)
        out[cp] = {
            "phi": D.stack(ch, "sv_u")[:, :, :, 1].reshape(-1, gq["r"]).mean(axis=0),
            "rho": summarize(D.stack(ch, "rho_u")[:, :, 0],
                             truth=float(D.RHO_U_TRUE[0])),
        }
        say(f"  corr(Q)=0.8 {cp:<10}: phi {np.round(out[cp]['phi'], 3)}  "
            f"rho_0 {out[cp]['rho']['mean']:+.3f}")

    better_rho = abs(out["qml"]["rho"]["mean"]) > abs(out["decoupled"]["rho"]["mean"]) + 0.10
    still_collapses = bool(np.any(out["qml"]["phi"] < 0.5))
    V.append(Verdict(
        "QML sotto leverage", "fattori", "B", Outcome.RECOVERED_BIASED,
        (f"**progresso, non soluzione** — e il check congela ENTRAMBI i fatti. "
         f"A corr(Q)=0.8 la deriva corretta (z = M eps esatto) recupera rho: "
         f"{out['qml']['rho']['mean']:+.2f} contro il {out['decoupled']['rho']['mean']:+.2f} "
         f"del decoupled (vero {D.RHO_U_TRUE[0]:+.2f}) ⇒ **P5 eliminata** "
         f"[{'confermato' if better_rho else 'NON confermato'}]. Ma phi di un fattore "
         f"{'COLLASSA ancora' if still_collapses else 'non collassa piu'}: "
         f"{np.round(out['qml']['phi'], 2)}. Resta dietro allow_experimental."),
        truth=float(D.RHO_U_TRUE[0]), estimate=out["qml"]["rho"]["mean"],
        ess=out["qml"]["rho"]["ess"], r_hat=out["qml"]["rho"]["r_hat"],
        mode=mode.name, detail={"spec_key": "_qml_lev",
                                "rho_gain": better_rho, "phi_collapses": still_collapses}))
    return V


def run(mode: D.Mode, *, coverage_reps: int = 0) -> list[Verdict]:
    V: list[Verdict] = []
    say("ASIS — quattro celle (use_asis x prior), branch B")
    V += _asis_cells(mode, "B")
    if mode.name != "quick":
        say("ASIS — branch A (mai coperto prima)")
        V += _asis_cells(mode, "A")
    say("QML — dominio di validita'")
    V += _qml(mode)
    return V
