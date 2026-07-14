"""
src/mcmc/test_leverage_common.py
================================

**Una funzione, parametrizzata per ramo** — al posto della coppia gemella
``test_passo3.test_end_to_end`` / ``test_passo4.test_end_to_end``, che asserivano le
stesse cose su Branch A e Branch B con il codice copiato due volte.

La duplicazione non era innocua: quando il griddy è stato cablato su Family C, la
semantica dell'``acceptance`` di ``rho`` è cambiata (**accetta per costruzione**, quindi
1.0), e la correzione andava fatta in **due** posti — di cui uno è stato dimenticato
finché il test non è diventato rosso.  Una sola funzione, due chiamate.

Cosa cambia fra i due rami — ed è il punto della parametrizzazione
------------------------------------------------------------------
* **Branch A** (contemporaneo): il path si estrae per **Metropolis** (il drift
  ``rho·sigma·z_t`` contiene ``exp(-x_t/2)``, quindi la transizione non è
  lineare-gaussiana e non esiste FFBS).  La sua acceptance è quindi una **vera** tasso di
  accettazione, e deve stare in un intervallo sano.
* **Branch B** (laggato): la mistura di Omori rende il sistema condizionatamente
  lineare-gaussiano, il path è un **draw FFBS diretto**, e la sua acceptance è **1.0 per
  costruzione**.  Non è un tasso: è la registrazione del vantaggio strutturale.

In **entrambi**, l'acceptance di ``rho`` è **1.0** (il griddy estrae dalla full
conditional).  ⚠ **Non è una diagnostica di mixing**: si legge l'**ESS**, non
l'accettazione.  Questo è il punto su cui il vecchio test si è rotto.

``T`` differisce perché la SV per-fattore paga il costo informativo di leggere ogni
``h^u_k`` da **una sola** log-square per periodo (P2): Branch B ha bisogno di un pannello
più lungo per recuperare il leverage dominante.
"""

from __future__ import annotations

import numpy as np

from mcmc.gibbs import fit_dfm_mcmc
from mcmc.simulate_sv import simulate_dfm_sv


#: I due rami, e cio' che li distingue *operativamente*.
BRANCH_CFG = {
    "A": {"timing": "contemporaneous", "T": 220, "n_iter": 800, "burn_in": 300,
          "path_is_metropolis": True},
    "B": {"timing": "lagged", "T": 500, "n_iter": 600, "burn_in": 250,
          "path_is_metropolis": False},
}

RHO_U_TRUE = np.array([-0.6, -0.3, -0.2])


def leverage_end_to_end(branch: str, theta, fl, bm, oc, r, check) -> dict:
    """Il gate end-to-end del leverage, **uno per ramo**.  ``check(nome, ok, dettaglio)``
    è il reporter del file chiamante, così il conteggio pass/fail resta suo."""
    cfg = BRANCH_CFG[branch]
    rho_true = RHO_U_TRUE[:r]

    sim = simulate_dfm_sv(theta, T=cfg["T"], freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=9, sv_u=(0.0, 0.97, 0.22), sv_eps=(0.0, 0.95, 0.15),
                          rho_u=rho_true, rho_eps=-0.3, timing=cfg["timing"])
    res = fit_dfm_mcmc(sim["Y"], {**theta, "Sigma_0": np.asarray(theta["Sigma_0"])},
                       fl, bm, oc, n_iter=cfg["n_iter"], burn_in=cfg["burn_in"], thin=1,
                       seed=4, sv=True, leverage=True, timing=cfg["timing"],
                       store_vol=True, verbose=False)
    acc = res["meta"]["acceptance"]

    # ── il path: Metropolis sotto A, draw diretto sotto B ────────────────────
    if cfg["path_is_metropolis"]:
        mh = [k for k in acc if not k.startswith("rho_")]
        check(f"[{branch}] acceptance Metropolis (path, sigma2) in (0.05, 0.95)",
              all(0.05 < acc[k] < 0.95 for k in mh),
              f"{ {k: round(acc[k], 2) for k in mh} }")
    else:
        check(f"[{branch}] il path FFBS e' un draw DIRETTO (acceptance == 1.0)",
              acc["path_u"] == 1.0 and acc["path_eps"] == 1.0, f"{acc}")
        check(f"[{branch}] sigma2 resta un Metropolis (acceptance in (0.05, 0.95))",
              0.05 < acc["sigma2"] < 0.95, f"sigma2={acc['sigma2']:.2f}")

    # ── Family C: il griddy accetta per costruzione, su ENTRAMBI i rami ──────
    check(f"[{branch}] il griddy accetta per costruzione (rho acc == 1) — NON e' una "
          f"diagnostica di mixing: si legge l'ESS",
          acc["rho_u"] == 1.0 and acc["rho_eps"] == 1.0,
          f"rho_u={acc['rho_u']}, rho_eps={acc['rho_eps']}")

    # ── nessun rho_hat senza il suo ESS ──────────────────────────────────────
    d = res["diagnostics"]["rho_u"]
    check(f"[{branch}] l'ESS di rho e' riportato (nessun rho_hat senza il suo ESS)",
          d["ess"].shape == (r,) and np.all(d["ess"] > 0),
          f"ESS={np.round(d['ess'], 1)}")

    # ── il segno dominante.  NB: e' il SEGNO, non la MAGNITUDINE — quest'ultima e'
    #    attenuata (x0.64) e la misura il validatore (mcmc.validate), non questo gate.
    rho_hat = res["theta_mean"]["rho_u"]
    j = int(np.argmax(np.abs(rho_true)))
    check(f"[{branch}] il canale dominante ha segno negativo",
          rho_hat[j] < 0, f"rho_hat={np.round(rho_hat, 3)}")

    # ── i path per-fattore tracciano la volatilita' vera ─────────────────────
    lhu = np.log(res["draws"]["h_u"].mean(axis=0))
    truth = sim["logh_u_true"]
    if lhu.ndim == 2:
        if np.asarray(truth).ndim == 1:                 # DGP scalare-comune
            c = float(np.mean([np.corrcoef(lhu[:, k], truth)[0, 1]
                               for k in range(lhu.shape[1])]))
        else:
            c = float(np.mean([np.corrcoef(lhu[:, k], truth[:, k])[0, 1]
                               for k in range(lhu.shape[1])]))
        thr = 0.3 if branch == "A" else 0.4
        check(f"[{branch}] i path h^u per fattore tracciano il vero (corr media > {thr})",
              c > thr, f"corr={c:.3f}")
    return res
