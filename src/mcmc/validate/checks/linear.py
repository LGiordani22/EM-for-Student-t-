"""
mcmc/validate/checks/linear.py
==============================

**Recupera:** ``A``, ``Q`` [Fam. A]; ``Lambda``, ``R`` [Fam. A']; ``f_t`` (stati, blocco
(a)); ``nu_u``, ``nu_eps`` [Fam. D]; e i pesi ``w_u``/``w_eps`` [Fam. D] — immagazzinati
via ``store_weights`` e confrontati col vero sulla traccia (come gli ``h``).

Il blocco lineare — **sotto SV e leverage**, che è la novità.

I parametri di Famiglia **A** (``A``, ``Q``) e **A'** (``Lambda``, ``R``) erano validati
solo contro l'EM, e **senza volatilità stocastica** (``test_passo1``).  Ma la
configurazione che spediremmo ha SV *e* leverage accesi, e in quel regime nessuno li
guardava.  Non è un dettaglio: sotto SV gli shock sono deflazionati per ``h_t``, la
verosimiglianza cambia forma, e ``A``/``Q``/``Lambda``/``R`` sono stimati da un
conditional diverso (per ``A,Q`` letteralmente un kernel diverso —
``shared.draw_A_Q_perfactor`` invece del MNIW).

Le famiglie, e da dove viene ciascun parametro
----------------------------------------------
* **Famiglia A** — ``A`` (VAR dei fattori) e ``Q`` (covarianza delle innovazioni).  Sotto
  SV il draw è un *two-step per-fattore*: gli shock vengono sbiancati per ``sqrt(H_t)`` e
  pesati per ``w_t``, poi ``A`` esce dalla sua gaussiana condizionale e ``Q`` da una
  inverse-Wishart (o dalla costruzione gerarchica half-t di Huang–Wand, se richiesta).
* **Famiglia A'** — ``Lambda`` (loadings) e ``R`` (varianze idiosincratiche): una NIG per
  equazione, cioè una regressione bayesiana per serie del dato osservato sui fattori.
* **Famiglia D** — ``nu_u`` e ``nu_eps`` (gradi di libertà) da un griddy-Gibbs sul
  supporto compatto, e i pesi ``w`` da una Gamma.  Sono **due** ``nu`` distinti: le code
  del blocco fattori e quelle del blocco osservazioni non sono lo stesso oggetto.
* **Stati** ``f_t``: FFBS di Kalman (blocco (a)), confrontati col vero per correlazione —
  sono un *path*, non un parametro, quindi il verdetto è sulla traccia, non sul valore.
"""

from __future__ import annotations

import numpy as np

from mcmc.diagnostics import innovation_correlation, loadings_unit_factor
from mcmc.validate import dgp as D
from mcmc.validate.checks import relerr, say, summarize
from mcmc.validate.verdict import Outcome, Verdict


def _corr(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def run(mode: D.Mode, *, coverage_reps: int = 0) -> list[Verdict]:
    V: list[Verdict] = []
    for branch in ("B", "A"):
        g = D.build(mode, branch=branch)
        chains = D.fit(g, seed=700)
        sim, r = g["sim"], g["r"]
        say(f"branch {branch}: {mode.chains} catene x {mode.n_iter} iter")

        # ── Famiglia A: A ─────────────────────────────────────────────────────
        A_true = np.asarray(g["theta"]["A"], float)
        A_hat = D.stack(chains, "A").reshape(-1, *A_true.shape).mean(axis=0)
        a00 = D.stack(chains, "A")[:, :, 0, 0]
        sA = summarize(a00)
        eA = relerr(A_hat, A_true)
        V.append(Verdict(
            "A", "fattori", branch,
            Outcome.RECOVERED if eA < 0.35 else Outcome.RECOVERED_BIASED,
            (f"err. rel. {eA:.0%} contro il vero, **sotto SV + leverage** (prima era "
             f"validato solo contro l'EM e senza SV)"),
            estimate=float(A_hat[0, 0]), truth=float(A_true[0, 0]),
            ess=sA["ess"], r_hat=sA["r_hat"], mode=mode.name,
            detail={"spec_key": "A", "relerr": eA}))
        say(f"  A: err.rel. {eA:.1%}  (ESS {sA['ess']:.0f}, R-hat {sA['r_hat']:.3f})")

        # ── Famiglia A: Q come CORRELAZIONE identificata ──────────────────────
        # Q = D^{1/2} R_Q D^{1/2}.  La scala D (diagonale) e' la stessa non-identificazione
        # di Lambda (assorbita da Lambda, ancorata da mu_h) — non un fallimento del sampler.
        # L'oggetto identificato e' la correlazione R_Q, che mescola bene (ESS ~500).  Nel
        # DGP canonico Q e' diagonale => R_Q = I: si verifica che le off-diagonali stiano
        # a zero.  Le correlazioni VERE (Q non diagonale) sono testate in coupling.py.
        Q_true = np.asarray(g["theta"]["Q"], float)
        Rq = innovation_correlation(D.stack(chains, "Q"))
        Rq_true = innovation_correlation(Q_true)
        r01 = Rq[:, :, 0, 1]
        sR = summarize(r01, truth=float(Rq_true[0, 1]))
        offmax = float(np.max(np.abs(Rq.reshape(-1, r, r).mean(axis=0) - np.eye(r))))
        V.append(Verdict(
            "Q", "fattori", branch,
            Outcome.RECOVERED if offmax < 0.15 else Outcome.RECOVERED_BIASED,
            (f"riportata come CORRELAZIONE R_Q (oggetto identificato): la scala (diagonale) "
             f"e' la stessa non-identificazione di Lambda, assorbita da Lambda. DGP diagonale "
             f"=> R_Q = I: off-diag max {offmax:.3f} (vero 0), ESS {sR['ess']:.0f}. La scala "
             f"grezza di Q NON e' un parametro — le correlazioni vere sono in coupling.py."),
            estimate=float(sR["mean"]), truth=float(Rq_true[0, 1]),
            ess=sR["ess"], r_hat=sR["r_hat"], mode=mode.name,
            detail={"spec_key": "Q", "offdiag_max": offmax}))
        say(f"  Q (corr R_Q): off-diag max {offmax:.3f}  (ESS {sR['ess']:.0f}, "
            f"R-hat {sR['r_hat']:.3f})")

        # ── Famiglia A': Lambda, R ────────────────────────────────────────────
        # Lambda nella normalizzazione a fattore a varianza unitaria (loadings_unit_factor):
        # l'oggetto identificato.  Anche il VERO va normalizzato con lo stesso criterio,
        # altrimenti si confrontano scale diverse.  A, Q, R non toccati.
        L_true = loadings_unit_factor(np.asarray(g["theta"]["Lambda"], float),
                                      np.asarray(sim["F"], float)[:, :r])
        R_true = np.asarray(g["theta"]["R"], float).ravel()
        L_n = loadings_unit_factor(D.stack(chains, "Lambda"), D.stack(chains, "F"))
        L_hat = L_n.reshape(-1, *L_true.shape).mean(axis=0)
        R_hat = D.stack(chains, "R").reshape(-1, R_true.size).mean(axis=0)
        l00 = L_n[:, :, 0, 0]
        r00 = D.stack(chains, "R")[:, :, 0]
        lam_why = (f"err. rel. {relerr(L_hat, L_true):.0%} contro il vero, nella "
                   f"normalizzazione a fattore a varianza unitaria (Lambda*sd(f)). "
                   f"**Buco chiuso.**")
        r_why = ("err. rel. {e:.0%} contro il vero, **sotto SV + leverage** (buco chiuso)")
        for name, hat, true, s, why in (
                ("Lambda", L_hat, L_true, summarize(l00), lam_why),
                ("R", R_hat, R_true, summarize(r00), None)):
            e = relerr(hat, true)
            ok = e < 0.35
            V.append(Verdict(
                name, "osservazioni", branch,
                Outcome.RECOVERED if ok else Outcome.RECOVERED_BIASED,
                why if why is not None else r_why.format(e=e),
                estimate=float(np.asarray(hat).ravel()[0]),
                truth=float(np.asarray(true).ravel()[0]),
                ess=s["ess"], r_hat=s["r_hat"], mode=mode.name,
                detail={"spec_key": name, "relerr": e}))
            say(f"  {name}: err.rel. {e:.1%}  (ESS {s['ess']:.0f}, R-hat {s['r_hat']:.3f})")

        # ── Stati: sono un path, quindi il verdetto e' sulla TRACCIA ──────────
        F_true = np.asarray(sim["F"], float)[:, :r]
        F_hat = D.stack(chains, "F").reshape(-1, *F_true.shape).mean(axis=0)
        c = float(np.mean([_corr(F_hat[:, k], F_true[:, k]) for k in range(r)]))
        V.append(Verdict(
            "f_t", "fattori", branch,
            Outcome.RECOVERED if c > 0.8 else Outcome.RECOVERED_BIASED,
            f"corr. media col vero {c:.2f} (un path si giudica sulla traccia, non sul valore)",
            estimate=c, mode=mode.name, detail={"spec_key": "f"}))
        say(f"  f_t: corr {c:.3f}")

        # ── Famiglia D: nu_u, nu_eps ─────────────────────────────────────────
        # nu_u non era MAI stato confrontato col vero (solo "resta sano").  Nota: nu e'
        # notoriamente difficile — la verosimiglianza e' piatta per nu grandi (una t con
        # nu=30 e una con nu=60 sono quasi indistinguibili), quindi il verdetto atteso e'
        # spesso "non identificato", e va DETTO, non aggirato con una soglia larga.
        for key, tex, blk in (("nu_u", "nu_u", "fattori"), ("nu_eps", "nu_eps", "osservazioni")):
            true = float(g["theta"][key])
            s = summarize(D.stack(chains, key), truth=true)
            if s["covers"] and abs(s["ratio"] - 1) < 0.5:
                oc, why = Outcome.RECOVERED, f"CI copre il vero, rapporto {s['ratio']:.2f}"
            elif s["covers"]:
                oc, why = (Outcome.NOT_IDENTIFIED,
                           "il CI copre il vero ma e' larghissimo: la verosimiglianza di nu "
                           "e' piatta per nu grandi (t con nu=30 e nu=60 quasi indistinguibili)")
            else:
                oc, why = (Outcome.RECOVERED_BIASED,
                           f"il CI ESCLUDE il vero (rapporto {s['ratio']:.2f})")
            V.append(Verdict(tex, blk, branch, oc, why, truth=true, estimate=s["mean"],
                             ci=(s["lo"], s["hi"]), ess=s["ess"], r_hat=s["r_hat"],
                             mode=mode.name, detail={"spec_key": key}))
            say(f"  {key}: vero {true:.1f}, stima {s['mean']:.1f} "
                f"[{s['lo']:.1f},{s['hi']:.1f}]  (ESS {s['ess']:.0f})")

        # ── Famiglia D: i pesi.  Sono un path per periodo (uno per ogni t), non uno
        # scalare: si giudicano sulla TRACCIA, come gli h — la posterior media dei pesi
        # traccia i pesi veri se il campionatore vede gli outlier che il DGP ha messo.
        # (store_weights=True in dgp.fit li espone nei draws.)
        for key, tex, blk in (("w_u", "w^u_t", "fattori"), ("w_eps", "w^eps_t", "osservazioni")):
            true_w = np.asarray(sim[f"{key}_true"], float)
            w_hat = D.stack(chains, key).reshape(-1, true_w.size).mean(axis=0)
            c = _corr(w_hat, true_w)
            V.append(Verdict(
                tex, blk, branch,
                Outcome.RECOVERED if c > 0.5 else Outcome.NOT_IDENTIFIED,
                f"i pesi sono un path per periodo: corr. media della posterior media col "
                f"vero {c:.2f} (giudicati sulla traccia, come gli h).",
                estimate=c, mode=mode.name, detail={"spec_key": key}))
            say(f"  {key}: corr {c:.3f}")

        if mode.name == "quick":
            break   # in --quick un solo ramo: serve a vedere se gira, non a decidere
    return V
