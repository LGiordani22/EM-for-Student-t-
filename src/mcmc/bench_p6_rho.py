"""
src/mcmc/bench_p6_rho.py
========================

Benchmark del mixing di ``rho`` sotto **Branch B** (laggato) sul **DGP per-fattore**
Spec II + Option A — il numero che decide il fix P6 (``docs/audit_P1-P5.md`` §P6,
``docs/fix_P6_map.md``).

Perché un modulo dedicato e non ``diagnostics.run_recovery_mcmc_leverage``: quella
harness passa ``sv_u`` come **tupla** a ``simulate_dfm_sv`` (``diagnostics.py:786``),
quindi genera il DGP **scalar-common** ``H = h·I``, non il per-fattore; e rifà l'EM, il
controllo ``rho=0`` e la skewness, che qui sono rumore. Questo bench misura una cosa
sola, con catene lunghe e più semi, e serve **due volte**: baseline RW (PASSO 2) e
confronto col griddy (GATE 3).

Cosa riporta, per canale ``k``:

* ``ESS(rho_k)`` — per catena e **pooled** (le catene sono indipendenti, gli ESS si
  sommano);
* ``split-R-hat(rho_k)`` — su tutte le catene insieme (``(n_chains, n_draws)``), che
  è più severo del split-R-hat da catena singola: coglie anche una catena bloccata in
  un modo diverso dalle altre;
* ``rho_hat_k`` con intervallo posteriore, e l'errore Monte Carlo
  ``MCSE = sd(rho) / sqrt(ESS_pooled)`` — l'unica cifra che dice se ``rho_hat`` sia
  una misura o un rumore.

Uso::

    python src/mcmc/bench_p6_rho.py                 # baseline (RW), come da PASSO 2
    python src/mcmc/bench_p6_rho.py --n-iter 4000 --chains 2
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcmc.diagnostics import ess, split_r_hat                                # noqa: E402
from mcmc.gibbs import fit_dfm_mcmc, load_warm_init                          # noqa: E402
from mcmc.simulate_sv import simulate_dfm_sv                                 # noqa: E402

# Il DGP dell'audit: tre canali distinti, uno dominante negativo, uno debole, uno
# positivo (quindi l'ordinamento, non solo il segno, deve essere recuperato).
RHO_TRUE = np.array([-0.70, -0.15, 0.45])
SV_U_PF = np.array([[0.0, 0.97, 0.25],
                    [0.0, 0.92, 0.18],
                    [0.0, 0.95, 0.22]])
T_SIM = 600


def _diagonal_theta(theta: dict) -> dict:
    """Q diagonale: il whitening di magnitudine di Branch B e' esatto (nessun effetto
    P5), e il blocco di volatilita' disaccoppiato e' esatto (nessun effetto P1/P4).
    Cosi' il bench misura il mixing di rho, e nient'altro."""
    th = dict(theta)
    th["Q"] = np.diag(np.diag(np.asarray(theta["Q"], float)))
    th["Sigma_0"] = np.asarray(theta["Sigma_0"], float)
    return th


def run_bench(*, n_iter: int = 4000, burn_in: int = 1500, chains: int = 2,
              thin: int = 1, config_name: str = "small", seed0: int = 700,
              verbose: bool = True, **fit_kwargs) -> dict:
    """Gira ``chains`` catene di Branch B sul DGP per-fattore e riassume ``rho``.

    ``fit_kwargs`` e' passato a ``fit_dfm_mcmc``: e' il gancio con cui il GATE 3
    selezionera' ``rho_sampler="griddy"`` senza duplicare questo codice.
    """
    w = load_warm_init(config_name)
    theta = _diagonal_theta(w["theta"])
    fl, bm, oc, r = w["freq_list"], w["block_map"], w["ordered_cols"], w["r"]
    if r != len(RHO_TRUE):
        raise ValueError(f"il bench e' calibrato su r=3, la config ha r={r}")

    sim = simulate_dfm_sv(theta, T=T_SIM, freq_list=fl, block_map=bm, ordered_cols=oc,
                          r=r, seed=22, sv_u=(0.0, 0.95, 0.0), sv_u_perfactor=SV_U_PF,
                          sv_eps=(0.0, 0.94, 0.12), rho_u=RHO_TRUE, rho_eps=-0.3,
                          timing="lagged")

    t0 = time.time()
    rho_chains = []
    sig2_chains = []
    ess_per_chain = []
    for c in range(chains):
        if verbose:
            print(f"  [bench] catena {c + 1}/{chains} (n_iter={n_iter}) ...", flush=True)
        res = fit_dfm_mcmc(sim["Y"], theta, fl, bm, oc,
                           n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed0 + c,
                           sv=True, leverage=True, timing="lagged",
                           verbose=False, **fit_kwargs)
        rho_chains.append(res["draws"]["rho_u"])                 # (n_keep, r)
        sig2_chains.append(res["draws"]["sv_u"][:, :, 2])        # (n_keep, r)
        ess_per_chain.append(res["diagnostics"]["rho_u"]["ess"])  # (r,)

    R = np.stack(rho_chains)                                     # (chains, n_keep, r)
    S = np.stack(sig2_chains)                                    # (chains, n_keep, r)
    n_keep = R.shape[1]

    out = {"rho_true": RHO_TRUE, "n_iter": n_iter, "burn_in": burn_in,
           "chains": chains, "n_keep": n_keep, "T": T_SIM,
           "wall_clock_s": time.time() - t0,
           "acceptance_rho_u": res["meta"]["acceptance"]["rho_u"],
           "ess_per_chain": np.stack(ess_per_chain)}             # (chains, r)

    pooled_ess, rhat, mean, lo, hi, mcse = [], [], [], [], [], []
    ess_s2, rhat_s2 = [], []
    for k in range(r):
        ch = R[:, :, k]                                          # (chains, n_keep)
        pooled_ess.append(float(ess(ch)))
        rhat.append(float(split_r_hat(ch)))
        flat = ch.ravel()
        mean.append(float(flat.mean()))
        lo.append(float(np.quantile(flat, 0.05)))
        hi.append(float(np.quantile(flat, 0.95)))
        mcse.append(float(flat.std() / np.sqrt(max(pooled_ess[-1], 1.0))))
        # sigma_eta^2: e' la coordinata con cui rho forma la cresta (audit P6). Se un
        # rimedio agisce via il mixing di sigma, deve vedersi prima qui.
        s2 = S[:, :, k]
        ess_s2.append(float(ess(s2)))
        rhat_s2.append(float(split_r_hat(s2)))
    out.update(ess=np.array(pooled_ess), r_hat=np.array(rhat), mean=np.array(mean),
               lo=np.array(lo), hi=np.array(hi), mcse=np.array(mcse),
               ess_sigma2=np.array(ess_s2), r_hat_sigma2=np.array(rhat_s2))
    return out


def print_bench(out: dict, label: str = "") -> None:
    r = len(out["rho_true"])
    hdr = f"  {label}" if label else ""
    print(f"\n{'=' * 78}")
    print(f"  BENCH rho — Branch B, DGP per-fattore, T={out['T']},"
          f" n_iter={out['n_iter']}, burn={out['burn_in']}, catene={out['chains']}{hdr}")
    print(f"  draw tenuti per catena: {out['n_keep']}   "
          f"totali: {out['n_keep'] * out['chains']}   "
          f"wall-clock: {out['wall_clock_s']:.0f}s")
    print(f"{'=' * 78}")
    print(f"  {'canale':<8}{'rho vero':>10}{'rho_hat':>10}{'CI 90%':>20}"
          f"{'ESS':>9}{'R-hat':>9}{'MCSE':>9}")
    for k in range(r):
        ci = f"[{out['lo'][k]:+.3f}, {out['hi'][k]:+.3f}]"
        print(f"  k={k:<6}{out['rho_true'][k]:>10.2f}{out['mean'][k]:>10.3f}{ci:>20}"
              f"{out['ess'][k]:>9.1f}{out['r_hat'][k]:>9.3f}{out['mcse'][k]:>9.3f}")
    print(f"\n  ESS per catena:\n{out['ess_per_chain']}")
    print(f"  acceptance(rho_u) = {out['acceptance_rho_u']:.3f}")
    n_tot = out["n_keep"] * out["chains"]
    print(f"  efficienza ESS/draw: {np.round(out['ess'] / n_tot, 4)}")
    print(f"  ESS(sigma_eta^2):    {np.round(out['ess_sigma2'], 1)}"
          f"   R-hat: {np.round(out['r_hat_sigma2'], 3)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn-in", type=int, default=1500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--rho-sampler", type=str, default="griddy",
                   choices=("griddy", "rw"))
    a = p.parse_args()
    out = run_bench(n_iter=a.n_iter, burn_in=a.burn_in, chains=a.chains,
                    rho_sampler=a.rho_sampler)
    print_bench(out, a.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
