"""
mcmc/validate/checks/volatility.py
==================================

**Recupera:** ``phi^u_k``, ``sigma^2_{eta,k}``, ``h^u_k`` (r canali comuni) e
``phi^eps``, ``sigma^2_eps``, ``h^eps`` (media sulle M serie) — [Fam. B + i path ``h``].

Famiglia **B** — ``(phi, sigma^2_eta)`` — e i path ``h``, su **entrambi i blocchi**.

Il blocco idiosincratico ha una SV **per serie** (``h^eps_i``, con i suoi ``phi^eps_i`` e
``sigma^2_{eps,i}``): esiste, gira a ogni sweep, entra nella densità predittiva del PIL —
e **non era mai stata validata**.  Qui viene chiusa.

Due cose che questo modulo misura
---------------------------------

**Il tetto informativo.**  Sotto Spec II ogni ``h^u_k`` è letto attraverso **una
sola** log-square per periodo: la media su ``r`` canali che il caso scalare-comune
regalava non c'è più.  Su un canale a bassa volatilità il segnale *per periodo* non basta,
e ``corr(ĥ, h)`` **resta limitata anche a T grande e con i parametri veri**.  Non è
un difetto del campionatore: è quanto segnale c'è.  Il verdetto giusto è **non
identificato**, e va affermato — non aggirato allargando una soglia.

**Il prior guida ``sigma^2_eta``.**  ``sigma^2_eta`` è debolmente identificato, quindi la
risposta la dà in buona parte il prior.  Questo modulo MISURA, per ciascun prior (il
half-Normal adottato per il mixing di ``rho`` e l'inverse-Gamma), il rapporto fra la stima
e il vero — senza assumerne l'esito: il verdetto lo decide il dato.
"""

from __future__ import annotations

import numpy as np

from mcmc.validate import dgp as D
from mcmc.validate.checks import say, summarize
from mcmc.validate.verdict import Outcome, Verdict


def _track(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    """corr fra il path stimato (media a posteriori di log h) e il vero."""
    a = np.log(np.asarray(h_hat, float) + 1e-12).ravel()
    b = np.asarray(h_true, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _sigma_prior_bias(g, mode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """La misura che mancava: ``sigma^2_eta`` contro il vero, sotto i DUE prior.  Non e'
    una curiosita': abbiamo curato il mixing di rho agendo su questo prior senza mai controllare cosa
    facesse alla STIMA del parametro su cui agisce."""
    true = np.asarray(g["sim"]["sv_u"], float)[:, 2]          # gia' in varianza
    hn = D.fit(g, seed=700, sv_sigma_prior="half_normal")   # = il fit di default (cache)
    ig = D.fit(g, seed=700, sv_sigma_prior="inverse_gamma", n_chains=1)
    m_hn = D.stack(hn, "sv_u")[:, :, :, 2].reshape(-1, true.size).mean(axis=0)
    m_ig = D.stack(ig, "sv_u")[:, :, :, 2].reshape(-1, true.size).mean(axis=0)
    return true, m_hn, m_ig


def run(mode: D.Mode, *, coverage_reps: int = 0) -> list[Verdict]:
    V: list[Verdict] = []
    branches = ("B",) if mode.name == "quick" else ("B", "A")

    for branch in branches:
        g = D.build(mode, branch=branch)
        chains = D.fit(g, seed=700)
        sim, r = g["sim"], g["r"]
        M = np.asarray(sim["Y"]).shape[1]
        say(f"branch {branch}")

        # ══ BLOCCO FATTORI ═══════════════════════════════════════════════════
        sv_u = D.stack(chains, "sv_u")                     # (nc, nk, r, 3)
        true_sv = np.asarray(sim["sv_u"], float)           # (r, 3), varianza
        h_u = D.stack(chains, "h_u").reshape(-1, mode.T, r).mean(axis=0)
        logh_true = np.asarray(sim["logh_u_true"], float)

        for k in range(r):
            weak = (k == D.WEAK)
            low_vol = " (canale a bassa volatilita', una sola log-square per periodo)" if weak else ""
            # φ — il verdetto lo decide il dato (copertura del CI + rapporto), non l'etichetta
            s = summarize(sv_u[:, :, k, 1], truth=float(true_sv[k, 1]))
            if s["covers"] and abs(s["ratio"] - 1.0) < 0.15:
                oc, why = Outcome.RECOVERED, f"phi = {s['mean']:.2f} (vero {true_sv[k,1]:.2f}), il CI copre il vero{low_vol}"
            elif not s["covers"]:
                oc, why = Outcome.RECOVERED_BIASED, f"phi = {s['mean']:.2f} contro {true_sv[k,1]:.2f}: il CI esclude il vero{low_vol}"
            else:
                oc, why = Outcome.NOT_IDENTIFIED, f"phi = {s['mean']:.2f}: il CI copre il vero ma e' largo{low_vol}"
            V.append(Verdict(f"phi^u_{k}", "fattori", branch, oc, why,
                             truth=float(true_sv[k, 1]), estimate=s["mean"],
                             ci=(s["lo"], s["hi"]), ess=s["ess"], r_hat=s["r_hat"],
                             mode=mode.name, detail={"spec_key": "phi_u"}))
            # h path — si giudica sulla traccia (corr col vero)
            c = _track(h_u[:, k], logh_true[:, k])
            if c > 0.55:
                oc, why = Outcome.RECOVERED, f"corr(h_hat, h) = {c:.2f} sul proprio path"
            else:
                oc, why = Outcome.NOT_IDENTIFIED, f"corr(h_hat, h) = {c:.2f}: segnale per periodo insufficiente{low_vol}"
            V.append(Verdict(f"h^u_{k}", "fattori", branch, oc, why, estimate=c,
                             mode=mode.name, detail={"spec_key": "h_u"}))
            say(f"  k={k}{' (DEBOLE)' if weak else ''}: phi {s['mean']:.2f} "
                f"(vero {true_sv[k,1]:.2f}), corr(h) {c:.2f}")

        # σ²_eta — il bias del prior, misurato
        true_s2, m_hn, m_ig = _sigma_prior_bias(g, mode)
        for k in range(r):
            s = summarize(sv_u[:, :, k, 2], truth=float(true_s2[k]))
            ratio_hn = float(m_hn[k] / true_s2[k])
            ratio_ig = float(m_ig[k] / true_s2[k])
            why = (f"sigma^2_eta e' debolmente identificato, quindi il prior GUIDA la stima: "
                   f"rapporto sul vero half-Normal x{ratio_hn:.1f}, inverse-Gamma x{ratio_ig:.1f}. "
                   f"I due prior danno stime diverse — il check lo MISURA.")
            if not s["covers"]:
                oc = Outcome.RECOVERED_BIASED       # il CI esclude il vero
            elif abs(s["ratio"] - 1.0) < 0.5:
                oc = Outcome.RECOVERED
            else:
                oc = Outcome.NOT_IDENTIFIED         # copre il vero ma il CI e' largo
            if k == D.WEAK:
                why = "canale a bassa volatilita': " + why
            V.append(Verdict(f"sigma^2_eta,{k}", "fattori", branch, oc, why,
                             truth=float(true_s2[k]), estimate=s["mean"],
                             ci=(s["lo"], s["hi"]), ess=s["ess"], r_hat=s["r_hat"],
                             mode=mode.name,
                             detail={"spec_key": "sigma2_eta_u",
                                     "ratio_half_normal": ratio_hn, "ratio_IG": ratio_ig}))
        say(f"  sigma^2_eta: vero {np.round(true_s2,4)} | HN {np.round(m_hn,4)} "
            f"| IG {np.round(m_ig,4)}")

        # ══ BLOCCO OSSERVAZIONI — MAI VALIDATO PRIMA ═════════════════════════
        sv_e = D.stack(chains, "sv_eps")                   # (nc, nk, M, 3)
        true_e = np.asarray(sim["sv_eps"], float)          # (M, 3), varianza
        h_e = D.stack(chains, "h_eps").reshape(-1, mode.T, M).mean(axis=0)
        loge_true = np.asarray(sim["logh_eps_true"], float)

        phis = summarize(sv_e[:, :, :, 1].mean(axis=2), truth=float(true_e[:, 1].mean()))
        corr_e = float(np.mean([_track(h_e[:, i], loge_true[:, i]) for i in range(M)]))
        s2s = summarize(sv_e[:, :, :, 2].mean(axis=2), truth=float(true_e[:, 2].mean()))

        oc_phi = Outcome.RECOVERED if abs(phis["ratio"] - 1) < 0.25 else Outcome.RECOVERED_BIASED
        V.append(Verdict("phi^eps (media M)", "osservazioni", branch, oc_phi,
                         f"media sulle M serie: {phis['mean']:.2f} contro {phis['truth']:.2f}. "
                         f"**Buco chiuso**: la SV idiosincratica esiste (una per serie) e non "
                         f"era mai stata confrontata col vero.",
                         truth=phis["truth"], estimate=phis["mean"],
                         ess=phis["ess"], r_hat=phis["r_hat"], mode=mode.name,
                         detail={"spec_key": "phi_eps"}))
        oc_s2e = (Outcome.RECOVERED if (s2s["covers"] and abs(s2s["ratio"] - 1) < 0.5)
                  else Outcome.RECOVERED_BIASED)
        V.append(Verdict("sigma^2_eps (media M)", "osservazioni", branch, oc_s2e,
                         f"media sulle M serie: {s2s['mean']:.4f} contro {s2s['truth']:.4f} "
                         f"(x{s2s['ratio']:.1f}). Stessa sensibilita' al prior del blocco comune.",
                         truth=s2s["truth"], estimate=s2s["mean"],
                         ess=s2s["ess"], r_hat=s2s["r_hat"], mode=mode.name,
                         detail={"spec_key": "sigma2_eta_eps"}))
        oc_h = Outcome.RECOVERED if corr_e > 0.4 else Outcome.NOT_IDENTIFIED
        V.append(Verdict("h^eps (media M)", "osservazioni", branch, oc_h,
                         f"corr media col vero {corr_e:.2f} sulle M serie. La SV "
                         f"idiosincratica e' letta attraverso UNA log-square per serie per "
                         f"periodo: stesso tetto informativo del blocco comune.",
                         estimate=corr_e, mode=mode.name, detail={"spec_key": "h_eps"}))
        say(f"  idio: phi {phis['mean']:.2f} (vero {phis['truth']:.2f}), "
            f"corr(h^eps) {corr_e:.2f}, sigma^2 x{s2s['ratio']:.1f}")
    return V
