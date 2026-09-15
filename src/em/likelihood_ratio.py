"""
src/em/likelihood_ratio.py — il test del rapporto di verosimiglianza fra celle.

PERCHE' UN TEST DENTRO CAMPIONE, VISTO CHE CE N'E' GIA' UNO FUORI
------------------------------------------------------------------
Perche' lo suggerisce chi ha inventato quello fuori.  Diebold (2015, sez.
3.1.1) osserva che per confronti ANNIDATI il paradigma pseudo-out-of-sample e'
sub-ottimale, e che "the optimal model comparison procedure is based on
full-sample residuals, not (pseudo-) out-of-sample forecast errors".

Le due cose rispondono a domande diverse e vanno lette in coppia:

    LR   le code ci SONO nel pannello?          (37 serie, 15.341 osservazioni)
    CW   modellarle MIGLIORA il nowcast del PIL? (1 serie, 46-66 trimestri)

Il contrasto fra le due risposte e' il risultato: un rifiuto schiacciante del
gaussiano dentro campione accanto a un vantaggio al limite fuori.  E' la
versione misurata di "robustezza alle osservazioni estreme non e' robustezza
agli eventi estremi".  Per i test fuori campione vedi
`src/forecast/predictive_accuracy.py`.

CHE COSA SI CONFRONTA, ESATTAMENTE
-----------------------------------
`loglik_history` dentro `fit_dfm_result.npz` contiene il FULL ELBO, non la sola
log-verosimiglianza di Kalman: `em_main.run_em` somma a `e_step_output["loglik"]`
la correzione `compute_elbo_correction` quando `use_full_elbo=True`, che e' il
default (em_main.py, righe ~2068-2071).  Cioe'

    ELBO = loglik_Kalman + E_q[log p(W|nu)] + H[q(W)] = loglik - KL(q_W || p_W).

DUE ASIMMETRIE, E VANNO NELLA STESSA DIREZIONE — IL TEST E' CONSERVATIVO
-------------------------------------------------------------------------
  * Per le celle GAUSSIANE la correzione e' ESATTAMENTE ZERO.  La docstring di
    `compute_elbo_correction` lo dice: "When nu_s = inf (Gaussian limit) the
    correction for that component is 0: weights are identically 1, the Kalman
    log-lik IS the full log-lik".  Li' l'ELBO e' la verosimiglianza vera.
  * Per le celle STUDENT-t l'E-step e' a campo medio fra fattori e pesi, quindi
    l'ELBO e' un LIMITE INFERIORE STRETTO della marginale.

Dunque   LR_calcolato = 2 (ELBO_grande - loglik_piccolo)  <=  LR_vero:
il modello grande e' sotto-rappresentato, mai sopra.  Se questo LR rifiuta, il
vero rifiuta a maggior ragione.  L'asimmetria e' una garanzia, non un difetto.

LA DISTRIBUZIONE NULLA NON E' UN CHI-QUADRO
--------------------------------------------
`nu -> inf` non e' un punto interno dello spazio parametrico: riparametrizzando
`delta = 1/nu >= 0` l'ipotesi nulla e' `delta = 0`, che sta sul BORDO.  Li' il
teorema di Wilks non vale, e la nulla e' la mistura di Chernoff (1954) /
Self & Liang (1987):

    sum_{j=0}^{k} C(k,j) 2^-k chi^2_j        (chi^2_0 = massa in zero)

Per k=1 il critico al 5% e' 2.706 (cioe' il chi^2_1 al 10%), per k=2 e' 4.231.
Per le restrizioni INTERNE (rho_i = 0, con rho in (-1,1)) resta chi^2_k.

DUE COSE CHE NON SI AFFERMANO
------------------------------
  * La mistura con k=2 vale se i due parametri sul bordo sono asintoticamente
    ORTOGONALI.  `nu_u` e `nu_eps` non e' detto che lo siano.  Con LR nell'ordine
    delle migliaia contro un critico di 4.2 la questione e' immateriale, ma la
    si dichiara invece di tacerla.
  * Si confrontano ELBO RAGGIUNTI, non massimi globali garantiti: le iterazioni
    vanno da 44 a 187 a seconda della cella.  Il RIFIUTO del gaussiano e' al
    sicuro (la cella con meno iterazioni lo batte comunque di migliaia); la
    GRADUATORIA fra le varianti Student-t no, e non va usata per scegliere.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import special, stats

__all__ = ["chernoff_sf", "chernoff_crit", "parametri_liberi", "tabella", "main"]

_FIT = os.path.join("data", "processed", "final", "%s", "%s",
                    "fit_dfm_result.npz")
_DATASET = os.path.join("data", "processed", "final", "dataset_final.csv")

VARIANTI = ["gaussian", "gaussian_ar1", "student_t", "student_t_ar1",
            "student_t_ar1_shared"]

#: I confronti annidati che hanno senso, con il numero e il TIPO di restrizione.
#:
#: `bordo` = le due 1/nu portate a zero (mistura di Chernoff).
#: `interno` = i 37 rho portati a zero, che stanno dentro (-1,1) (chi-quadro).
#: Il numero di restrizioni interne non si scrive a mano: si conta dai theta.
#:
#: OGNI RIGA ACCENDE UN INTERRUTTORE SOLO, e questo vincola quali coppie si
#: possono mettere.  `student_t` ha i pesi idiosincratici CONDIVISI
#: (`per_series_weights: False` in `forecast/nowcast_engine.py`), `student_t_ar1`
#: li ha PER SERIE: la coppia `student_t -> student_t_ar1` cambierebbe insieme
#: la persistenza e lo schema dei pesi, mentre i gradi contano i soli 37 rho e
#: il chi-quadro presume solo quelli.  Peggio: i pesi per-serie non costano
#: parametri (nu_eps resta scalare), quindi il modello grande guadagna
#: flessibilita' senza pagarla nella penalita' e il critico e' troppo
#: indulgente.  La coppia pulita e' `student_t -> student_t_ar1_shared`: pesi
#: condivisi da una parte e dall'altra, cambiano i soli rho.  Il verdetto non
#: cambia (LR 3435.7 invece di 5452.0, contro lo stesso 52.2), cambia che la
#: riga diventa quello che la colonna dice di essere.
#:
#: Fuori restano `gaussian -> student_t_ar1` e `gaussian -> student_t_ar1_shared`,
#: che accendono i due interruttori insieme: l'LR e' additivo lungo la catena
#: (5035.2 + 5452.0 = 10487.2, l'LR diretto), quindi non direbbero niente di
#: nuovo, e la loro nulla sarebbe una terza distribuzione (restrizioni miste
#: bordo + interno) per una riga che e' la somma di due gia' presenti.
COPPIE: list[tuple[str, str, int | None, str]] = [
    ("gaussian", "student_t", 2, "bordo"),
    ("gaussian_ar1", "student_t_ar1", 2, "bordo"),
    ("gaussian_ar1", "student_t_ar1_shared", 2, "bordo"),
    ("gaussian", "gaussian_ar1", None, "interno"),
    ("student_t", "student_t_ar1_shared", None, "interno"),
]


# ─── La distribuzione nulla al bordo ─────────────────────────────────────────

def chernoff_sf(lr: float, k: int) -> float:
    """
    P(mistura >= lr) per la nulla di Chernoff / Self & Liang con `k` parametri
    sul bordo:  sum_j C(k,j) 2^-k chi^2_j.

    La componente `chi^2_0` e' una massa puntuale in zero: contribuisce 1 se
    `lr <= 0` e 0 altrimenti.  E' la meta' della distribuzione che rende il test
    MENO severo di un chi^2_k — ed e' la ragione per cui usare il chi^2_k
    sarebbe conservativo a sproposito.
    """
    if lr <= 0.0:
        return 1.0
    p = 0.0
    for j in range(1, int(k) + 1):                 # j = 0 non contribuisce
        p += float(special.comb(k, j)) * 0.5 ** k * float(stats.chi2.sf(lr, j))
    return float(p)


def chernoff_crit(k: int, alpha: float) -> float:
    """Il quantile (1-alpha) della mistura, per bisezione."""
    lo, hi = 0.0, 500.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if chernoff_sf(mid, k) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ─── Conteggio dei parametri ─────────────────────────────────────────────────

def parametri_liberi(fit: dict) -> int:
    """
    I parametri liberi di una cella, per AIC/BIC e per contare le restrizioni.

        Lambda   i caricamenti NON azzerati dalla mask di `factor_specs`
        A        r^2
        Q        r(r+1)/2, perche' simmetrica
        R        uno per serie
        rho      uno per serie, solo nelle varianti `_ar1`
        nu       due, solo nelle varianti Student-t

    I caricamenti si contano dai valori non nulli e non dalla mask: sotto
    `fed_overlap` la mask e' a blocchi e uno zero esatto la' dentro e' un
    vincolo, non una stima che e' venuta zero.
    """
    Lam = np.asarray(fit["Lambda"], dtype=float)
    r = Lam.shape[1]
    k = int((np.abs(Lam) > 1e-12).sum())
    k += r * r
    k += r * (r + 1) // 2
    k += int(np.asarray(fit["R"], dtype=float).size)
    rho = fit.get("rho")
    if rho is not None and np.ndim(rho) > 0:
        k += int(np.asarray(rho, dtype=float).size)
    if np.isfinite(float(fit["nu_u"])):
        k += 2
    return k


def _carica(spec: str, variante: str, radice: str = ".") -> dict:
    d = np.load(os.path.join(radice, _FIT % (spec, variante)), allow_pickle=True)
    return {k: d[k] for k in d.files}


# ─── La tabella ──────────────────────────────────────────────────────────────

def tabella(spec: str, radice: str = ".") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Per una spec: `(adattamento, confronti, n_osservazioni)`.

    `adattamento` porta ELBO, parametri, AIC e BIC di ogni variante;
    `confronti` porta LR, gradi, p-value e quale nulla si e' usata.
    """
    Y = pd.read_csv(os.path.join(radice, _DATASET), index_col=0)
    n_oss = int(Y.notna().to_numpy().sum())

    fits = {v: _carica(spec, v, radice) for v in VARIANTI}
    ll = {v: float(np.asarray(f["loglik_history"], dtype=float)[-1])
          for v, f in fits.items()}
    kk = {v: parametri_liberi(f) for v, f in fits.items()}

    adatt = pd.DataFrame({
        "variante": VARIANTI,
        "ELBO": [ll[v] for v in VARIANTI],
        "n_par": [kk[v] for v in VARIANTI],
        "AIC": [-2 * ll[v] + 2 * kk[v] for v in VARIANTI],
        "BIC": [-2 * ll[v] + kk[v] * np.log(n_oss) for v in VARIANTI],
        "nu_u": [float(fits[v]["nu_u"]) for v in VARIANTI],
        "nu_eps": [float(fits[v]["nu_eps"]) for v in VARIANTI],
    })

    righe = []
    for piccolo, grande, k, tipo in COPPIE:
        lr = 2.0 * (ll[grande] - ll[piccolo])
        gradi = k if k is not None else kk[grande] - kk[piccolo]
        if tipo == "bordo":
            p, nulla = chernoff_sf(lr, gradi), "Chernoff k=%d" % gradi
        else:
            p, nulla = float(stats.chi2.sf(lr, gradi)), "chi2_%d" % gradi
        righe.append({"piccolo": piccolo, "grande": grande, "LR": lr,
                      "gradi": gradi, "p": p, "nulla": nulla,
                      "nat_per_oss": (ll[grande] - ll[piccolo]) / n_oss})
    return adatt, pd.DataFrame(righe), n_oss


def main(radice: str = ".", out_dir: str | None = None) -> str:
    """Il rapporto sulle tre spec.  Torna il testo."""
    R: list[str] = []
    P = R.append
    P("=" * 100)
    P("RAPPORTO DI VEROSIMIGLIANZA, CAMPIONE PIENO")
    P("LR = 2 (ELBO_grande - ELBO_piccolo).  Per lo Student-t l'ELBO e' un")
    P("LIMITE INFERIORE della marginale, quindi il test e' CONSERVATIVO.")
    P("=" * 100)
    P("")
    P("valori critici della mistura di Chernoff con k=2 sul bordo:"
      "   5%%: %.3f    1%%: %.3f" % (chernoff_crit(2, .05), chernoff_crit(2, .01)))

    for spec in ("fed_overlap", "diag4", "diag3"):
        try:
            adatt, conf, n_oss = tabella(spec, radice)
        except FileNotFoundError:
            continue
        P("")
        P("#" * 100)
        P("# spec %s   —   pannello con %d osservazioni non mancanti" % (spec, n_oss))
        P("#" * 100)
        P("%-24s%13s%8s%13s%13s" % ("variante", "ELBO", "n.par", "AIC", "BIC"))
        for _, r in adatt.iterrows():
            P("%-24s%13.1f%8d%13.1f%13.1f"
              % (r["variante"], r["ELBO"], r["n_par"], r["AIC"], r["BIC"]))
        P("")
        P("%-46s%11s%7s%12s   %s"
          % ("confronto  (piccolo -> grande)", "LR", "gradi", "p", "nulla"))
        for _, r in conf.iterrows():
            ps = "<1e-300" if r["p"] < 1e-300 else "%.3g" % r["p"]
            P("%-46s%11.1f%7d%12s   %s"
              % (r["piccolo"] + " -> " + r["grande"], r["LR"], r["gradi"],
                 ps, r["nulla"]))
        P("")
        P("guadagno per osservazione non mancante (nat):")
        for _, r in conf.iterrows():
            if r["nulla"].startswith("Chernoff"):
                P("   %-42s%8.4f" % (r["piccolo"] + " -> " + r["grande"],
                                     r["nat_per_oss"]))

    testo = "\n".join(R)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "likelihood_ratio.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(testo)
    return testo


# ─── Self-test ───────────────────────────────────────────────────────────────
#
# Solo verifiche analitiche sulla distribuzione nulla e sul conteggio dei
# parametri: niente lettura di artefatti, niente scrittura.

if __name__ == "__main__":
    ok = True

    def check(nome: str, cond: bool, extra: str = "") -> None:
        global ok
        ok &= bool(cond)
        print("  [%s] %s%s" % ("OK " if cond else "FAIL", nome,
                               ("  — " + extra) if extra else ""))

    print("1. la mistura di Chernoff a k=1 e' mezza massa in zero + mezzo chi2_1")
    for x in (0.5, 2.0, 2.706, 10.0):
        check("x=%.3f" % x,
              abs(chernoff_sf(x, 1) - 0.5 * stats.chi2.sf(x, 1)) < 1e-14)

    print("2. il critico al 5% con k=1 e' il chi2_1 al 10% (= 2.7055)")
    c1 = chernoff_crit(1, 0.05)
    check("2.7055", abs(c1 - stats.chi2.ppf(0.90, 1)) < 1e-6, "%.5f" % c1)

    print("3. il critico al 5% con k=2 vale 4.231, e ci si arriva a mano")
    c2 = chernoff_crit(2, 0.05)
    a_mano = 0.5 * stats.chi2.sf(c2, 1) + 0.25 * stats.chi2.sf(c2, 2)
    check("la sf al critico vale 0.05", abs(a_mano - 0.05) < 1e-8,
          "critico=%.4f  sf=%.6f" % (c2, a_mano))

    print("4. la mistura e' MENO severa del chi2_k (meta' massa sta in zero)")
    for k in (1, 2, 3):
        check("k=%d: crit_Chernoff < crit_chi2" % k,
              chernoff_crit(k, 0.05) < stats.chi2.ppf(0.95, k),
              "%.3f < %.3f" % (chernoff_crit(k, 0.05), stats.chi2.ppf(0.95, k)))

    print("5. monotonia e bordi della sf")
    check("sf(0) = 1", abs(chernoff_sf(0.0, 2) - 1.0) < 1e-14)
    check("sf decrescente",
          all(chernoff_sf(x, 2) > chernoff_sf(x + 0.5, 2)
              for x in (0.1, 1.0, 5.0, 20.0)))
    check("sf(molto grande) ~ 0", chernoff_sf(500.0, 2) < 1e-100)

    print("6. conteggio dei parametri su un theta finto")
    r, M = 4, 37
    Lam = np.ones((M, r)); Lam[:, 1:] = 0.0          # un solo fattore caricato
    finto_g = {"Lambda": Lam, "R": np.ones(M), "nu_u": np.inf,
               "nu_eps": np.inf}
    atteso = M + r * r + r * (r + 1) // 2 + M
    check("gaussiano: Lambda + A + Q + R",
          parametri_liberi(finto_g) == atteso, "%d" % parametri_liberi(finto_g))
    finto_t = dict(finto_g, nu_u=5.0, nu_eps=7.0)
    check("Student-t: + 2 per i nu",
          parametri_liberi(finto_t) == atteso + 2)
    finto_ar = dict(finto_t, rho=np.zeros(M))
    check("_ar1: + M per i rho",
          parametri_liberi(finto_ar) == atteso + 2 + M)
    check("gli zeri della mask NON si contano",
          parametri_liberi(finto_g) < M * r + r * r + r * (r + 1) // 2 + M)

    print("\n%s" % ("TUTTI I CONTROLLI PASSATI" if ok
                    else "*** QUALCOSA NON TORNA ***"))
    raise SystemExit(0 if ok else 1)
