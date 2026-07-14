"""
src/mcmc/test_sigma_eta.py
==========================

Il buco di test che ha nascosto una mina.

`sigma_eta^2` — la *volatilità della volatilità* — non era confrontato con il vero
da **nessun** test: la suite verificava che fosse finito, positivo, e che ASIS non ne
spostasse la posterior.  Tutte cose vere e tutte irrilevanti alla domanda «lo stimiamo
bene?».  Ed è un parametro che pesa: è quello di cui abbiamo cambiato il **prior**
(half-Normal) per curare il mixing di `rho`, ed è quello che governa quanto la
volatilità stimata può muoversi da un periodo all'altro.

Il buco non era innocuo.  Nascondeva una **collisione di convenzione**:

    simulate_dfm_sv   ingresso  sv_u = (mu, phi, SIGMA)     <- deviazione standard
    fit_dfm_mcmc      draws     sv_u = (mu, phi, SIGMA^2)   <- varianza

Stesso nome di campo, due significati.  Finché nessuno confrontava i due, la
collisione non poteva emergere: un confronto scritto contro di essa avrebbe paragonato
`sigma` a `sigma^2` restituendo numeri **plausibili e sbagliati** — il tipo di errore
peggiore.  Ora `simulate_dfm_sv` restituisce `sv_u` **nella convenzione del sampler** e
riecheggia gli ingressi come `sv_u_sigma`; questi test congelano entrambe le cose.

Cosa si asserisce
-----------------
[1] **La convenzione**, per prima: la sd incondizionata *empirica* del path vero deve
    valere ``sigma / sqrt(1 - phi^2)``.  È il test che avrebbe fatto emergere la mina
    da solo — non guarda il sampler, guarda solo il DGP, e distingue le due letture
    (a phi = 0.97 le due ipotesi danno 1.03 contro 2.06: non c'è modo di confonderle).

[2] **Il round-trip**: ``sim["sv_u"][:, 2] == sim["sv_u_sigma"][:, 2] ** 2``.

[3] **Il recovery vero e proprio**: la media a posteriori di ``sigma_eta^2`` contro la
    verità, canale per canale.  L'aspettativa è deliberatamente **asimmetrica** fra
    canali forti e canale debole: P2 dice che dove non c'è volatilità da leggere il
    path non è identificato, e allora nemmeno la sua innovazione può esserlo.  Il test
    misura quella frontiera invece di girarci intorno.

[4] **Il prior conta**: half-Normal contro inverse-Gamma sullo stesso dato.  Serve a
    sapere se la scelta che abbiamo fatto per P6 *sposta la stima*, non solo il mixing.

Run
---
    python src/mcmc/test_sigma_eta.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.bench_p6_rho import RHO_TRUE, SV_U_PF, _diagonal_theta          # noqa: E402
from mcmc.gibbs import fit_dfm_mcmc, load_warm_init                       # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                              # noqa: E402

_PASS = 0
_FAIL = 0

T_SIM = 600
N_ITER, BURN_IN = 1200, 400

# sd incondizionata di ciascun canale: sigma_k / sqrt(1 - phi_k^2) = [1.03, 0.46, 0.70].
# Il canale 1 e' deliberatamente il debole (P2): la' il path non e' identificato a
# T = 600, quindi nemmeno sigma_eta^2 puo' esserlo.  Non e' un difetto: e' il tetto
# informativo, e il test lo asserisce invece di nasconderlo.
STRONG = [0, 2]
WEAK = 1


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1; print(f"  [PASS] {name}")
    else:
        _FAIL += 1; print(f"  [FAIL] {name}   {detail}")


def _simulate(seed=22):
    w = load_warm_init("small")
    theta = _diagonal_theta(w["theta"])
    sim = simulate_dfm_sv(theta, T=T_SIM, freq_list=w["freq_list"],
                          block_map=w["block_map"], ordered_cols=w["ordered_cols"],
                          r=w["r"], seed=seed, sv_u=(0.0, 0.95, 0.0),
                          sv_u_perfactor=SV_U_PF, sv_eps=(0.0, 0.94, 0.12),
                          rho_u=RHO_TRUE, rho_eps=-0.3, timing="lagged")
    return sim, theta, w


# ─────────────────────────────────────────────────────────────────────────────
# [1]-[2] La convenzione — il test che avrebbe trovato la mina da solo
# ─────────────────────────────────────────────────────────────────────────────

def test_convention(sim):
    print("\n[1] sigma in, sigma^2 out — la convenzione, congelata")
    sig = np.asarray(sim["sv_u_sigma"], float)          # (r, 3): (mu, phi, SIGMA)
    var = np.asarray(sim["sv_u"], float)                # (r, 3): (mu, phi, SIGMA^2)
    x = np.asarray(sim["logh_u_true"], float)           # (T, r) path VERO

    _check("round-trip: sv_u[:,2] == sv_u_sigma[:,2]**2",
           np.allclose(var[:, 2], sig[:, 2] ** 2),
           f"{np.round(var[:, 2], 4)} vs {np.round(sig[:, 2] ** 2, 4)}")

    # La prova indipendente dal codice: la sd EMPIRICA del path vero.  Le due letture
    # danno numeri lontanissimi (a phi=0.97: 1.03 contro 2.06), quindi il dato decide.
    # NB: e' un test di DISCRIMINAZIONE, non di precisione.  La sd incondizionata
    # *empirica* di un AR(1) con phi ~ 0.96 su T = 600 ha una variabilita' campionaria
    # enorme (l'ESS per stimarla e' una manciata di osservazioni indipendenti), quindi
    # pretendere un accordo stretto sarebbe una soglia arbitraria che diventa rossa a
    # caso.  Cio' che il dato *puo'* decidere, e decide senza ambiguita', e' QUALE delle
    # due letture sia quella giusta: le loro predizioni differiscono di un fattore ~2.
    emp = x.std(axis=0)
    as_sd = sig[:, 2] / np.sqrt(1.0 - sig[:, 1] ** 2)            # se la colonna e' sigma
    as_var = np.sqrt(sig[:, 2]) / np.sqrt(1.0 - sig[:, 1] ** 2)  # se fosse sigma^2
    err_sd = float(np.max(np.abs(emp - as_sd) / as_sd))
    err_var = float(np.max(np.abs(emp - as_var) / as_var))
    _check("la sd empirica del path vero identifica l'INPUT come sigma, non sigma^2",
           err_sd < 0.5 * err_var,
           f"empirica={np.round(emp, 3)}, come-sigma={np.round(as_sd, 3)} "
           f"(err {err_sd:.0%}), come-sigma^2={np.round(as_var, 3)} (err {err_var:.0%})")


# ─────────────────────────────────────────────────────────────────────────────
# [3] Recovery: sigma_eta^2 contro il vero
# ─────────────────────────────────────────────────────────────────────────────

def _fit(sim, theta, w, *, sigma_prior="half_normal", seed=5):
    res = fit_dfm_mcmc(sim["Y"], theta, w["freq_list"], w["block_map"],
                       w["ordered_cols"], n_iter=N_ITER, burn_in=BURN_IN, seed=seed,
                       sv=True, leverage=True, timing="lagged",
                       sv_sigma_prior=sigma_prior, rho_sampler="griddy", verbose=False)
    d = res["draws"]["sv_u"][:, :, 2]                   # (n_keep, r) = sigma_eta^2
    return d.mean(axis=0), np.quantile(d, 0.05, axis=0), np.quantile(d, 0.95, axis=0)


def test_recovery(sim, theta, w):
    print("\n[3] recovery di sigma_eta^2 contro il vero (prior half-Normal)")
    true_s2 = np.asarray(sim["sv_u"], float)[:, 2]      # gia' in varianza
    mean, lo, hi = _fit(sim, theta, w)

    for k in STRONG:
        ratio = float(mean[k] / true_s2[k])
        covers = bool(lo[k] <= true_s2[k] <= hi[k])
        _check(f"canale FORTE k={k}: sigma_eta^2 recuperato entro un fattore 2",
               0.5 < ratio < 2.0,
               f"vero={true_s2[k]:.4f}, stimato={mean[k]:.4f} (x{ratio:.2f}), "
               f"CI=[{lo[k]:.4f},{hi[k]:.4f}], copre={covers}")

    k = WEAK
    ratio = float(mean[k] / true_s2[k])
    print(f"  [info] canale DEBOLE k={k} (P2): vero={true_s2[k]:.4f}, "
          f"stimato={mean[k]:.4f} (x{ratio:.2f}), CI=[{lo[k]:.4f},{hi[k]:.4f}]")
    _check("canale DEBOLE: la stima resta finita e positiva (non collassa a 0)",
           bool(np.isfinite(mean[k]) and mean[k] > 1e-6),
           f"stimato={mean[k]:.6f}")


def test_prior_moves_the_estimate(sim, theta, w):
    """P6 e' stato curato cambiando il prior di sigma_eta.  Quel prior sposta anche la
    STIMA, o solo il mixing?  E' una domanda che va posta esplicitamente: se half-Normal
    e inverse-Gamma dessero risposte molto diverse sullo stesso dato, la scelta fatta per
    ragioni di campionamento avrebbe conseguenze sull'inferenza, e andrebbe dichiarata."""
    print("\n[4] il prior sposta la stima, o solo il mixing?")
    true_s2 = np.asarray(sim["sv_u"], float)[:, 2]
    m_hn, _, _ = _fit(sim, theta, w, sigma_prior="half_normal")
    m_ig, _, _ = _fit(sim, theta, w, sigma_prior="inverse_gamma")

    for k in STRONG:
        rel = float(abs(m_hn[k] - m_ig[k]) / max(m_ig[k], 1e-9))
        _check(f"canale FORTE k={k}: half-Normal e IG concordano entro il 50%",
               rel < 0.50,
               f"vero={true_s2[k]:.4f}, HN={m_hn[k]:.4f}, IG={m_ig[k]:.4f} "
               f"(scarto {rel:.0%})")
    print(f"  [info] vero  = {np.round(true_s2, 4)}")
    print(f"  [info] HN    = {np.round(m_hn, 4)}")
    print(f"  [info] IG    = {np.round(m_ig, 4)}")


def main() -> int:
    sim, theta, w = _simulate()
    test_convention(sim)
    test_recovery(sim, theta, w)
    test_prior_moves_the_estimate(sim, theta, w)
    print("\n" + "=" * 72)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
