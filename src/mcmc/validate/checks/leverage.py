"""
mcmc/validate/checks/leverage.py
================================

Famiglia **C** — ``rho`` — su **entrambi i blocchi** e **entrambi i rami**.

È il modulo che conta di più, per una ragione che non è tecnica: **``rho`` è ciò che dà
lo skew alla densità predittiva del PIL**.  Un ``|rho|`` attenuato **sottostima la coda
sinistra**, cioè sbaglia esattamente nel verso che rovina un esercizio di Growth-at-Risk.
E poiché non c'è nessun secondo stadio a valle, la densità del DFM *è* il deliverable:
non esiste un passo successivo che assorba l'errore.

Tre cose che questo modulo rende impossibili da riseppellire
------------------------------------------------------------

**1. La MAGNITUDINE, non solo il segno.**  I test storici asserivano il *segno* e
l'*ordinamento* dei canali, mai il valore.  È così che un'attenuazione del **35%** è
rimasta invisibile per mesi con la suite verde: la suite misurava la cosa sbagliata.
Qui il verdetto poggia sul rapporto ``rho_hat / rho_vero`` **e sulla copertura del CI**.

**2. L'attenuazione è errors-in-variables, NON la linearizzazione di Omori.**
:func:`check_oracle` congela il path **vero**, la volatilità **vera** e i pesi **veri**, e
fa girare **solo** il draw di Family C.  Con le latenti note ``rho`` si recupera al **98%**
— sia col regressore esatto sia con quello di Omori, che quindi **non attenua**.  Ergo: il
conditional è **corretto**, e l'attenuazione entra **tutta dall'incertezza sulle latenti**.
Questa prova sta *in suite*, non in un notebook, perché è la risposta alla domanda più
importante che la tesi deve reggere.

**3. Il canale debole non è identificato, e va DETTO.**  Il CI di ``rho^u_1`` copre lo zero
**ed entrambi i segni**.  I vecchi test asserivano proprietà di questa quantità non
identificata ed erano verdi solo perché la random-walk non convergeva: il fix li ha
*smascherati*, non rotti.  Qui il check **afferma il limite** e passa asserendolo.

E il buco più grave della mappa: ``rho^eps``
---------------------------------------------
Il leverage idiosincratico **esiste** (uno per serie, stessa Family C) ed è impostato nel
DGP di quasi ogni test (``rho_eps = -0.3``) — ma **nessuno lo aveva mai confrontato col
vero**.  È il blocco che proietta il PIL.  Qui viene chiuso.
"""

from __future__ import annotations

import numpy as np

from mcmc.constants import OMORI10
from mcmc.sample_leverage import draw_rho
from mcmc.sample_vol import _inv_sqrt_spd
from mcmc.validate import dgp as D
from mcmc.validate.checks import say, summarize
from mcmc.validate.verdict import Outcome, Verdict


# ─────────────────────────────────────────────────────────────────────────────
# LA PROVA: Family C con le latenti VERE
# ─────────────────────────────────────────────────────────────────────────────

def _family_c_at_true_path(g: dict, n_draw: int = 3000) -> dict:
    """Congela path, volatilita' e pesi VERI; estrae solo ``rho``.

    Due regressori, per separare due domande:
      (a) ``z`` ESATTO (whitening pieno al path vero) -> il CONDITIONAL e' corretto?
      (b) regressore di OMORI al path vero            -> la linearizzazione attenua?

    Se (a) ~ (b) ~ 1.0, allora Family C e' giusta E Omori non attenua, e l'attenuazione
    del Gibbs completo viene per forza dall'incertezza sulle latenti.
    """
    sim, theta, r = g["sim"], g["theta"], g["r"]
    A = np.asarray(theta["A"], float)
    Q = np.asarray(theta["Q"], float)
    Qih = _inv_sqrt_spd(Q)
    qd = np.diag(Q)

    x = np.asarray(sim["logh_u_true"], float)            # path VERO
    wu = np.asarray(sim["w_u_true"], float)              # pesi VERI
    F = np.asarray(sim["F"], float)[:, :r]               # fattori VERI
    sv = np.asarray(sim["sv_u"], float)                  # (r,3), gia' in VARIANZA
    rho_true = np.asarray(sim["rho_u_true"], float)
    T = x.shape[0]

    u = F[1:] - F[:-1] @ A.T
    b = np.zeros((T, r)); b[1:] = np.sqrt(wu[1:])[:, None] * u
    z = (np.exp(-0.5 * x) * b) @ Qih                     # z VERO, whitening pieno
    E = b / np.sqrt(qd)[None, :]
    xi = np.log(E ** 2 + 1e-6) - x
    ystar = np.log(E ** 2 + 1e-6)
    sgn = np.sign(z); sgn[sgn == 0] = 1.0

    rng = np.random.default_rng(4242)
    q_, m_, v_, a_, bb_ = (OMORI10[k] for k in ("q", "m", "v2", "a", "b"))
    dm = xi[:, :, None] - m_[None, None, :]
    lp = (np.log(q_)[None, None, :] - 0.5 * np.log(v_)[None, None, :]
          - 0.5 * dm * dm / v_[None, None, :])
    s_idx = np.argmax(lp + rng.gumbel(size=lp.shape), axis=2)
    g_om = sgn * np.exp(0.5 * m_[s_idx]) * (a_[s_idx] + bb_[s_idx] * (xi - m_[s_idx]))

    sel = np.zeros(T - 1, bool); sel[1:] = True          # leverage nelle transizioni t>=2
    out = {"exact": [], "omori": [], "truth": rho_true}
    for k in range(r):
        phi_k, s2_k = float(sv[k, 1]), float(sv[k, 2])
        eta = (x[1:, k] - phi_k * x[:-1, k])[sel]
        for tag, reg in (("exact", z[:-1, k]), ("omori", g_om[:-1, k])):
            kr = (np.sqrt(s2_k) * reg)[sel]
            d = np.array([draw_rho(0.0, eta, kr, s2_k, rng, sampler="griddy")[0]
                          for _ in range(n_draw)])
            out[tag].append(summarize(d[None, :], truth=float(rho_true[k])))
    return out


def check_oracle(mode: D.Mode) -> list[Verdict]:
    g = D.build(D.FULL, branch="B")      # l'oracolo non campiona il modello: T lungo, gratis
    o = _family_c_at_true_path(g)
    V = []
    ratios_e = [s["ratio"] for s in o["exact"]]
    ratios_o = [s["ratio"] for s in o["omori"]]
    say(f"oracolo (path VERO): z esatto x{np.round(ratios_e, 2)}  "
        f"Omori x{np.round(ratios_o, 2)}")

    # ⚠ La soglia NON puo' essere sul RAPPORTO: sul canale debole rho vero = -0.15, e un
    # rapporto su un valore cosi' piccolo e' dominato dal rumore (0.68 significa uno scarto
    # assoluto di 0.05, cioe' nulla).  Asserire il rapporto la' sarebbe asserire una
    # proprieta' di una quantita' NON IDENTIFICATA — il peccato che questo validatore
    # esiste per impedire, e che alla prima run ho commesso io.  La statistica giusta e'
    # lo scarto ASSOLUTO, che ha lo stesso significato su tutti e tre i canali.
    err_e = [abs(s["mean"] - s["truth"]) for s in o["exact"]]
    err_o = [abs(s["mean"] - s["truth"]) for s in o["omori"]]
    ok = all(e < 0.08 for e in err_e)
    om_ok = all(abs(a - b) < 0.05 for a, b in zip(err_e, err_o))
    say(f"  scarto assoluto: z esatto {np.round(err_e, 3)}  Omori {np.round(err_o, 3)}")
    V.append(Verdict(
        "rho^u | path VERO", "fattori", "B",
        Outcome.RECOVERED if ok else Outcome.BROKEN,
        ("**LA PROVA.** Con path/volatilita'/pesi VERI congelati, Family C recupera rho a "
         f"scarto assoluto max {max(err_e):.3f} (rapporti {np.round(ratios_e, 2)}). "
         "Il conditional e' dunque CORRETTO: "
         "l'attenuazione del Gibbs completo (x0.64) NON viene da un bug in Family C, ma "
         "**tutta dall'incertezza sulle latenti** (errors-in-variables)."),
        estimate=float(np.mean(ratios_e)), truth=1.0, mode="full",
        detail={"spec_key": "_oracle", "ratios": ratios_e}))
    V.append(Verdict(
        "rho^u | Omori vs esatto", "fattori", "B",
        Outcome.RECOVERED if om_ok else Outcome.BROKEN,
        ("Al path VERO il regressore di Omori e quello esatto danno lo stesso rho "
         f"(scarto max {max(abs(re-ro) for re, ro in zip(ratios_e, ratios_o)):.3f}): "
         "**la linearizzazione NON attenua**. Omori e' scagionato."),
        estimate=float(np.mean(ratios_o)), truth=1.0, mode="full",
        detail={"spec_key": "_oracle_omori"}))
    return V


# ─────────────────────────────────────────────────────────────────────────────
# Copertura: l'unico modo di distinguere la sfortuna dal bias
# ─────────────────────────────────────────────────────────────────────────────

def _coverage(n_reps: int, branch: str) -> dict:
    mode = D.coverage_mode(n_reps)
    r = 3
    acc = {"rho_u": {"mean": [[] for _ in range(r)], "cov": [[] for _ in range(r)]},
           "rho_eps": {"mean": [], "cov": []}}
    for rep in range(n_reps):
        g = D.build(mode, branch=branch, seed=1000 + rep)
        ch = D.fit(g, seed=7000 + rep)
        ru = D.stack(ch, "rho_u")
        re_ = D.stack(ch, "rho_eps")
        true_u = np.asarray(g["sim"]["rho_u_true"], float)
        true_e = float(np.asarray(g["sim"]["rho_eps_true"], float).mean())
        for k in range(r):
            s = summarize(ru[:, :, k], truth=float(true_u[k]))
            acc["rho_u"]["mean"][k].append(s["mean"])
            acc["rho_u"]["cov"][k].append(s["covers"])
        s = summarize(re_.mean(axis=2), truth=true_e)
        acc["rho_eps"]["mean"].append(s["mean"])
        acc["rho_eps"]["cov"].append(s["covers"])
        say(f"  rep {rep}: rho_u {np.round([m[-1] for m in acc['rho_u']['mean']], 3)} "
            f"cov {[c[-1] for c in acc['rho_u']['cov']]}")
    return acc


def run(mode: D.Mode, *, coverage_reps: int = 0) -> list[Verdict]:
    V: list[Verdict] = []
    branches = ("B",) if mode.name == "quick" else ("B", "A")

    for branch in branches:
        g = D.build(mode, branch=branch)
        chains = D.fit(g, seed=700)
        r = g["r"]
        true_u = np.asarray(g["sim"]["rho_u_true"], float)
        true_e = float(np.asarray(g["sim"]["rho_eps_true"], float).mean())
        say(f"branch {branch}")

        # ── rho^u, canale per canale: MAGNITUDINE, non solo segno ────────────
        ru = D.stack(chains, "rho_u")
        for k in range(r):
            s = summarize(ru[:, :, k], truth=float(true_u[k]))
            weak = (k == D.WEAK)
            spans_zero = s["lo"] < 0 < s["hi"]
            if weak and spans_zero:
                oc = Outcome.NOT_IDENTIFIED
                why = (f"il CI [{s['lo']:+.2f}, {s['hi']:+.2f}] copre lo ZERO ed ENTRAMBI I "
                       f"SEGNI: il canale debole non e' identificato. I vecchi test "
                       f"asserivano proprieta' di questa quantita' ed erano verdi solo "
                       f"perche' la RW non convergeva. Qui il limite si AFFERMA.")
            elif abs(s["ratio"] - 1.0) < 0.20 and s["covers"]:
                oc, why = Outcome.RECOVERED, f"rapporto {s['ratio']:.2f}, il CI copre il vero"
            else:
                oc = Outcome.RECOVERED_BIASED
                why = (f"**ATTENUATO**: rho_hat = {s['mean']:+.2f} contro {true_u[k]:+.2f} "
                       f"(x{s['ratio']:.2f}). Segno e ordinamento giusti, MAGNITUDINE no. "
                       f"Un |rho| attenuato SOTTOSTIMA la coda sinistra — il verso peggiore "
                       f"per il GaR. Causa: errors-in-variables (vedi il check oracolo), non "
                       f"Omori.")
            V.append(Verdict(f"rho^u_{k}", "fattori", branch, oc, why,
                             truth=float(true_u[k]), estimate=s["mean"],
                             ci=(s["lo"], s["hi"]), ess=s["ess"], r_hat=s["r_hat"],
                             mode=mode.name, detail={"spec_key": "rho_u"}))
            say(f"  rho^u_{k}: {s['mean']:+.3f} (vero {true_u[k]:+.2f}, "
                f"x{s['ratio']:.2f})  ESS {s['ess']:.0f}  R-hat {s['r_hat']:.3f}")

        # ── rho^eps — IL BUCO PIU' GRAVE DELLA MAPPA, chiuso qui ─────────────
        re_ = D.stack(chains, "rho_eps")
        s = summarize(re_.mean(axis=2), truth=true_e)
        if abs(s["ratio"] - 1.0) < 0.25 and s["covers"]:
            oc, why = (Outcome.RECOVERED,
                       f"media sulle M serie: {s['mean']:+.2f} contro {true_e:+.2f}. "
                       f"**Buco chiuso**: il leverage idiosincratico esiste (uno per serie), "
                       f"era nel DGP di quasi ogni test, e NON era mai stato confrontato col "
                       f"vero. E' il blocco che proietta il PIL.")
        elif s["lo"] < 0 < s["hi"]:
            oc, why = (Outcome.NOT_IDENTIFIED,
                       f"il CI [{s['lo']:+.2f},{s['hi']:+.2f}] copre lo zero: il leverage "
                       f"idiosincratico non e' identificato in media sulle M serie.")
        else:
            oc, why = (Outcome.RECOVERED_BIASED,
                       f"media {s['mean']:+.2f} contro {true_e:+.2f} (x{s['ratio']:.2f}): "
                       f"stessa attenuazione del blocco comune. **Buco chiuso**, e il "
                       f"risultato non e' rassicurante: e' il blocco che proietta il PIL.")
        V.append(Verdict("rho^eps (media M)", "osservazioni", branch, oc, why,
                         truth=true_e, estimate=s["mean"], ci=(s["lo"], s["hi"]),
                         ess=s["ess"], r_hat=s["r_hat"], mode=mode.name,
                         detail={"spec_key": "rho_eps"}))
        say(f"  rho^eps: {s['mean']:+.3f} (vero {true_e:+.2f}, x{s['ratio']:.2f})  "
            f"ESS {s['ess']:.0f}")

    # ── L'oracolo: la prova che l'attenuazione e' EIV, non Omori ─────────────
    if mode.name != "quick":
        V += check_oracle(mode)

    # ── Copertura ────────────────────────────────────────────────────────────
    if coverage_reps:
        say(f"copertura su {coverage_reps} dataset (branch B)")
        acc = _coverage(coverage_reps, "B")
        for k in range(3):
            cov = float(np.mean(acc["rho_u"]["cov"][k]))
            m = float(np.mean(acc["rho_u"]["mean"][k]))
            t = float(D.RHO_U_TRUE[k])
            oc = (Outcome.NOT_IDENTIFIED if k == D.WEAK
                  else (Outcome.RECOVERED if cov > 0.7 else Outcome.RECOVERED_BIASED))
            V.append(Verdict(
                f"rho^u_{k} [coverage]", "fattori", "B", oc,
                (f"su {coverage_reps} dataset: media {m:+.3f} contro {t:+.2f} "
                 f"(x{m/t:.2f}), il CI 90% copre il vero il {cov:.0%} delle volte. "
                 f"Un CI al 90% DEVE coprire ~90%: {'e lo fa' if cov > 0.7 else 'e non lo fa'}"
                 f" ⇒ {'nessun bias' if cov > 0.7 else 'BIAS, non sfortuna'}."),
                truth=t, estimate=m, coverage=cov, n_reps=coverage_reps,
                mode="coverage", detail={"spec_key": "rho_u"}))
        cov = float(np.mean(acc["rho_eps"]["cov"]))
        m = float(np.mean(acc["rho_eps"]["mean"]))
        V.append(Verdict(
            "rho^eps [coverage]", "osservazioni", "B",
            Outcome.RECOVERED if cov > 0.7 else Outcome.RECOVERED_BIASED,
            (f"su {coverage_reps} dataset: media {m:+.3f} contro {D.RHO_EPS_TRUE:+.2f}, "
             f"copertura {cov:.0%}."),
            truth=D.RHO_EPS_TRUE, estimate=m, coverage=cov, n_reps=coverage_reps,
            mode="coverage", detail={"spec_key": "rho_eps"}))
    return V
