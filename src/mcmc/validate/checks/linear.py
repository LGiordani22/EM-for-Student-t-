"""
mcmc/validate/checks/linear.py
==============================

**Recupera:** ``A``, ``Q`` [Fam. A]; ``Lambda``, ``R`` [Fam. A']; ``f_t`` (stati, blocco
(a)); ``nu_u``, ``nu_eps`` [Fam. D].  I pesi ``w_u``/``w_eps`` NON sono immagazzinati nei
draw → restano ``UNTESTED`` (il loro conditional è verificato in ``test_shared``).

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

        # ── Famiglia A: A, Q ──────────────────────────────────────────────────
        A_true = np.asarray(g["theta"]["A"], float)
        Q_true = np.asarray(g["theta"]["Q"], float)
        A_hat = D.stack(chains, "A").reshape(-1, *A_true.shape).mean(axis=0)
        Q_hat = D.stack(chains, "Q").reshape(-1, *Q_true.shape).mean(axis=0)
        # ESS/R-hat su una coordinata rappresentativa (il primo elemento diagonale):
        # una matrice non ha un ESS, ma una sua coordinata sì, e serve a dire se la
        # catena su quel blocco si e' mossa.
        a00 = D.stack(chains, "A")[:, :, 0, 0]
        q00 = D.stack(chains, "Q")[:, :, 0, 0]
        sA, sQ = summarize(a00), summarize(q00)

        for name, hat, true, s in (("A", A_hat, A_true, sA), ("Q", Q_hat, Q_true, sQ)):
            e = relerr(hat, true)
            ok = e < 0.35
            V.append(Verdict(
                name, "fattori", branch,
                Outcome.RECOVERED if ok else Outcome.RECOVERED_BIASED,
                (f"err. rel. {e:.0%} contro il vero, **sotto SV + leverage** (prima era "
                 f"validato solo contro l'EM e senza SV)"),
                estimate=float(hat[0, 0]), truth=float(true[0, 0]),
                ess=s["ess"], r_hat=s["r_hat"], mode=mode.name,
                detail={"spec_key": name, "relerr": e}))
            say(f"  {name}: err.rel. {e:.1%}  (ESS {s['ess']:.0f}, R-hat {s['r_hat']:.3f})")

        # ── Famiglia A': Lambda, R ────────────────────────────────────────────
        L_true = np.asarray(g["theta"]["Lambda"], float)
        R_true = np.asarray(g["theta"]["R"], float).ravel()
        L_hat = D.stack(chains, "Lambda").reshape(-1, *L_true.shape).mean(axis=0)
        R_hat = D.stack(chains, "R").reshape(-1, R_true.size).mean(axis=0)
        l00 = D.stack(chains, "Lambda")[:, :, 0, 0]
        r00 = D.stack(chains, "R")[:, :, 0]
        for name, hat, true, s in (("Lambda", L_hat, L_true, summarize(l00)),
                                   ("R", R_hat, R_true, summarize(r00))):
            e = relerr(hat, true)
            ok = e < 0.35
            V.append(Verdict(
                name, "osservazioni", branch,
                Outcome.RECOVERED if ok else Outcome.RECOVERED_BIASED,
                f"err. rel. {e:.0%} contro il vero, **sotto SV + leverage** (buco chiuso)",
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
