"""
mcmc/validate/checks/coupling.py
================================

**Recupero a Q NON diagonale** — il coupling fra i fattori.

Il DGP canonico ha ``Q`` diagonale: un esperimento controllato, perché a ``Q`` diagonale
il blocco di volatilità comune disaccoppiato è *esatto* e il whitening di magnitudine di
Branch B è *esatto*.  Ma quello isola il leverage dal coupling e lascia una domanda
aperta: **quando ``Q`` è piena, si recuperano ancora i parametri?**  Questo modulo la
chiude.  Costruisce lo stesso DGP con un ``Q`` a off-diagonale non nullo e ripete la
batteria di recupero.

**Recupera** (tutti a ``Q`` non diagonale, un solo fit): ``A``, ``Q`` (off-diagonale
incluso); ``Lambda``, ``R``; e — dove la ``Q`` piena conta di più, perché il blocco
disaccoppiato non è più esatto — ``phi^u_k``, ``sigma^2_{eta,k}``, ``h^u_k`` e ``rho^u_k``
(r canali comuni).  E — per completezza, su **entrambi i blocchi** — anche
l'idiosincratico (``phi^eps``, ``sigma^2_eps``, ``h^eps``, ``rho^eps``, media sulle M
serie): l'off-diagonale di ``Q`` accoppia i *fattori*, non le serie, quindi lì non
dovrebbe cambiare nulla rispetto al caso diagonale — e se cambia, lo si vede.  Ogni riga
porta ``[Q corr]`` per distinguerla dalla stessa a ``Q`` diagonale: la tabella le mostra
affiancate.
"""

from __future__ import annotations

import numpy as np

from mcmc.validate import dgp as D
from mcmc.validate.checks import relerr, say, summarize
from mcmc.validate.verdict import Outcome, Verdict

#: off-diagonale di Q nel DGP di questo modulo (correlazione fra gli shock dei fattori)
_CORR = 0.5


def _track(h_hat, h_true) -> float:
    a = np.log(np.asarray(h_hat, float) + 1e-12).ravel()
    b = np.asarray(h_true, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _oc_ratio(s, tol=0.15) -> Outcome:
    """Verdetto data-driven da (copertura del CI, rapporto): lo decide il dato."""
    if s["covers"] and abs(s["ratio"] - 1.0) < tol:
        return Outcome.RECOVERED
    if not s["covers"]:
        return Outcome.RECOVERED_BIASED
    return Outcome.NOT_IDENTIFIED


def run(mode: D.Mode, *, coverage_reps: int = 0) -> list[Verdict]:
    if mode.name == "quick":
        return []   # un fit in più, non probante in quick: si salta
    V: list[Verdict] = []
    g = D.build(mode, branch="B", correlated_Q=_CORR)
    chains = D.fit(g, seed=700)
    sim, r = g["sim"], g["r"]
    say(f"Q NON diagonale (off-diag {_CORR}), branch B — un solo fit, tutta la batteria")

    # ── A, Q (l'off-diagonale, che a Q diagonale vale 0 e non e' mai messo alla prova) ──
    A_true = np.asarray(g["theta"]["A"], float)
    Q_true = np.asarray(g["theta"]["Q"], float)
    A_hat = D.stack(chains, "A").reshape(-1, *A_true.shape).mean(axis=0)
    Q_hat = D.stack(chains, "Q").reshape(-1, *Q_true.shape).mean(axis=0)
    sQ = summarize(D.stack(chains, "Q")[:, :, 0, 1], truth=float(Q_true[0, 1]))
    eA, eQ = relerr(A_hat, A_true), relerr(Q_hat, Q_true)
    V.append(Verdict(
        "A [Q corr]", "fattori", "B",
        Outcome.RECOVERED if eA < 0.35 else Outcome.RECOVERED_BIASED,
        f"err. rel. A {eA:.0%} a Q non diagonale.",
        estimate=float(A_hat[0, 0]), truth=float(A_true[0, 0]), mode=mode.name,
        detail={"spec_key": "A"}))
    V.append(Verdict(
        "Q off-diag [Q corr]", "fattori", "B",
        Outcome.RECOVERED if (sQ["covers"] and abs(sQ["ratio"] - 1) < 0.5) else Outcome.RECOVERED_BIASED,
        f"off-diag vero {Q_true[0,1]:+.3f}, stima {Q_hat[0,1]:+.3f}; err. rel. Q (matrice) {eQ:.0%}.",
        truth=float(Q_true[0, 1]), estimate=float(Q_hat[0, 1]), ci=(sQ["lo"], sQ["hi"]),
        ess=sQ["ess"], r_hat=sQ["r_hat"], mode=mode.name, detail={"spec_key": "Q"}))

    # ── Lambda, R ──────────────────────────────────────────────────────────────
    L_true = np.asarray(g["theta"]["Lambda"], float)
    R_true = np.asarray(g["theta"]["R"], float).ravel()
    L_hat = D.stack(chains, "Lambda").reshape(-1, *L_true.shape).mean(axis=0)
    R_hat = D.stack(chains, "R").reshape(-1, R_true.size).mean(axis=0)
    for nm, hat, tru, key, blk in (("Lambda", L_hat, L_true, "Lambda", "osservazioni"),
                                   ("R", R_hat, R_true, "R", "osservazioni")):
        e = relerr(hat, tru)
        V.append(Verdict(
            f"{nm} [Q corr]", blk, "B",
            Outcome.RECOVERED if e < 0.35 else Outcome.RECOVERED_BIASED,
            f"err. rel. {e:.0%} a Q non diagonale.",
            estimate=float(np.asarray(hat).ravel()[0]), truth=float(np.asarray(tru).ravel()[0]),
            mode=mode.name, detail={"spec_key": key}))

    # ── phi, sigma^2, h, rho per fattore — il blocco dove la Q piena conta di piu' ──
    sv = D.stack(chains, "sv_u")
    true_sv = np.asarray(sim["sv_u"], float)
    h_u = D.stack(chains, "h_u").reshape(-1, mode.T, r).mean(axis=0)
    logh_true = np.asarray(sim["logh_u_true"], float)
    ru = D.stack(chains, "rho_u")
    true_rho = np.asarray(sim["rho_u_true"], float)
    for k in range(r):
        sph = summarize(sv[:, :, k, 1], truth=float(true_sv[k, 1]))
        V.append(Verdict(
            f"phi^u_{k} [Q corr]", "fattori", "B", _oc_ratio(sph),
            f"phi = {sph['mean']:.2f} (vero {true_sv[k,1]:.2f}) a Q non diagonale.",
            truth=float(true_sv[k, 1]), estimate=sph["mean"], ci=(sph["lo"], sph["hi"]),
            ess=sph["ess"], r_hat=sph["r_hat"], mode=mode.name, detail={"spec_key": "phi_u"}))
        ss2 = summarize(sv[:, :, k, 2], truth=float(true_sv[k, 2]))
        V.append(Verdict(
            f"sigma^2_eta,{k} [Q corr]", "fattori", "B", _oc_ratio(ss2, tol=0.5),
            f"sigma^2 = {ss2['mean']:.4f} (vero {true_sv[k,2]:.4f}) a Q non diagonale.",
            truth=float(true_sv[k, 2]), estimate=ss2["mean"], ci=(ss2["lo"], ss2["hi"]),
            ess=ss2["ess"], r_hat=ss2["r_hat"], mode=mode.name, detail={"spec_key": "sigma2_eta_u"}))
        cc = _track(h_u[:, k], logh_true[:, k])
        V.append(Verdict(
            f"h^u_{k} [Q corr]", "fattori", "B",
            Outcome.RECOVERED if cc > 0.55 else Outcome.NOT_IDENTIFIED,
            f"corr(h_hat, h) = {cc:.2f} a Q non diagonale.",
            estimate=cc, mode=mode.name, detail={"spec_key": "h_u"}))
        srh = summarize(ru[:, :, k], truth=float(true_rho[k]))
        if srh["lo"] < 0 < srh["hi"]:
            oc = Outcome.NOT_IDENTIFIED
        else:
            oc = _oc_ratio(srh, tol=0.2)
        V.append(Verdict(
            f"rho^u_{k} [Q corr]", "fattori", "B", oc,
            f"rho = {srh['mean']:+.2f} (vero {true_rho[k]:+.2f}) a Q non diagonale — "
            f"a Q piena il whitening di magnitudine non e' piu' esatto: si vede qui.",
            truth=float(true_rho[k]), estimate=srh["mean"], ci=(srh["lo"], srh["hi"]),
            ess=srh["ess"], r_hat=srh["r_hat"], mode=mode.name, detail={"spec_key": "rho_u"}))
        say(f"  k={k}: phi {sph['mean']:.2f}  sigma^2 {ss2['mean']:.4f}  "
            f"corr(h) {cc:.2f}  rho {srh['mean']:+.2f} (vero {true_rho[k]:+.2f})")

    # ── blocco IDIOSINCRATICO a Q non diagonale ──────────────────────────────
    # L'off-diagonale di Q accoppia i FATTORI, non le serie (R e' diagonale), quindi
    # non tocca DIRETTAMENTE l'idio: lo verifichiamo per completezza (entrambi i blocchi),
    # e uno scostamento dal caso diagonale segnalerebbe un accoppiamento inatteso.
    M = np.asarray(sim["Y"]).shape[1]
    sv_e = D.stack(chains, "sv_eps")
    true_e = np.asarray(sim["sv_eps"], float)
    h_e = D.stack(chains, "h_eps").reshape(-1, mode.T, M).mean(axis=0)
    loge_true = np.asarray(sim["logh_eps_true"], float)
    re_ = D.stack(chains, "rho_eps")
    true_re = float(np.asarray(sim["rho_eps_true"], float).mean())

    phis = summarize(sv_e[:, :, :, 1].mean(axis=2), truth=float(true_e[:, 1].mean()))
    s2se = summarize(sv_e[:, :, :, 2].mean(axis=2), truth=float(true_e[:, 2].mean()))
    corr_e = float(np.mean([_track(h_e[:, i], loge_true[:, i]) for i in range(M)]))
    srhe = summarize(re_.mean(axis=2), truth=true_re)
    V.append(Verdict(
        "phi^eps (media M) [Q corr]", "osservazioni", "B", _oc_ratio(phis, tol=0.25),
        f"media M: phi = {phis['mean']:.2f} (vero {phis['truth']:.2f}) a Q non diagonale.",
        truth=phis["truth"], estimate=phis["mean"], ci=(phis["lo"], phis["hi"]),
        ess=phis["ess"], r_hat=phis["r_hat"], mode=mode.name, detail={"spec_key": "phi_eps"}))
    V.append(Verdict(
        "sigma^2_eps (media M) [Q corr]", "osservazioni", "B", _oc_ratio(s2se, tol=0.5),
        f"media M: sigma^2 = {s2se['mean']:.4f} (vero {s2se['truth']:.4f}) a Q non diagonale.",
        truth=s2se["truth"], estimate=s2se["mean"], ci=(s2se["lo"], s2se["hi"]),
        ess=s2se["ess"], r_hat=s2se["r_hat"], mode=mode.name, detail={"spec_key": "sigma2_eta_eps"}))
    V.append(Verdict(
        "h^eps (media M) [Q corr]", "osservazioni", "B",
        Outcome.RECOVERED if corr_e > 0.4 else Outcome.NOT_IDENTIFIED,
        f"media M: corr(h_hat, h) = {corr_e:.2f} a Q non diagonale.",
        estimate=corr_e, mode=mode.name, detail={"spec_key": "h_eps"}))
    oc_re = Outcome.NOT_IDENTIFIED if (srhe["lo"] < 0 < srhe["hi"]) else _oc_ratio(srhe, tol=0.25)
    V.append(Verdict(
        "rho^eps (media M) [Q corr]", "osservazioni", "B", oc_re,
        f"media M: rho = {srhe['mean']:+.2f} (vero {true_re:+.2f}) a Q non diagonale.",
        truth=true_re, estimate=srhe["mean"], ci=(srhe["lo"], srhe["hi"]),
        ess=srhe["ess"], r_hat=srhe["r_hat"], mode=mode.name, detail={"spec_key": "rho_eps"}))
    say(f"  idio: phi {phis['mean']:.2f}  sigma^2 {s2se['mean']:.4f}  "
        f"corr(h) {corr_e:.2f}  rho {srhe['mean']:+.2f} (vero {true_re:+.2f})")
    return V
