"""
src/bvar/tests/test_gate1.py

IL RECOVERY TEST DEL GATE 1: il core sampler recupera i parametri veri?

    python -m src.bvar.tests.test_gate1

I test dei blocchi 2-4 chiedevano "il codice fa quello che dice la matematica?".
Questo chiede la domanda scientifica: "la matematica recupera la verita'?".
Si simula da parametri e iperparametri noti, si stima tutto, si verifica di
ritrovarli dentro l'intervallo di credibilita'.

Dimensionato su VAR piccoli, come concordato: qui serve che il recovery sia
NETTO e la diagnosi non ambigua.  Il regime large-n si stressa ai gate dei
modelli specifici.

I DUE CONTROLLI PIU' IMPORTANTI
-------------------------------
  §2  la ML in forma chiusa contro l'identita' di Chib — ESATTA, non stocastica:
      valida A.13/A.14 contro cio' che la ML significa;
  §5  l'equivalenza fra parametrizzazione log e naturale del Metropolis — rende
      DIMOSTRATA, e non asserita, la fedelta' all'Appendice B (stesso spirito
      dell'oracolo del Blocco 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.special import multigammaln

from src.bvar.core import CoreState, sample, step
from src.bvar.hyper import (
    HyperPrior,
    build_target,
    gamma_from_mode_sd,
    init_metropolis,
    metropolis_step,
)
from src.bvar.niw import build_prior, draw, log_ml, niw_posterior
from src.bvar.simulate import simulate_var
from src.bvar.spec import BVARSpec, Hyper, MinnesotaSpec

_OK, _FAIL = "OK", "FAIL"


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {label:<62} {_OK if cond else _FAIL}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def _spec(n: int, p: int, d_centre=None) -> BVARSpec:
    d = tuple(np.ones(n, dtype=int)) if d_centre is None else tuple(int(x) for x in d_centre)
    return BVARSpec(model="Q", profile="q_b",
                    series=tuple(f"v{i}" for i in range(n)),
                    freq=("M",) * n, transform=("log",) * n,
                    sample_start=pd.Timestamp("1990-01-31"),
                    minnesota=MinnesotaSpec(p=p, d_centre=d), config={})


# ─── 1. Gli iperprior ─────────────────────────────────────────────────────────

def test_hyperprior() -> bool:
    print("\n1. Gli iperprior di GLP §III")
    ok = True
    for name, mode, sd in (("lambda", 0.2, 0.4), ("mu", 1.0, 1.0)):
        a, th = gamma_from_mode_sd(mode, sd)
        got_mode, got_sd = (a - 1.0) * th, np.sqrt(a) * th
        ok &= _check(f"Gamma per {name}: moda {mode} e sd {sd} ritrovate",
                     np.isclose(got_mode, mode) and np.isclose(got_sd, sd),
                     f"shape={a:.4f} scale={th:.4f}")

    hp = HyperPrior()
    ok &= _check("psi: inverse-Gamma con shape=scale=(0.02)^2",
                 np.isclose(hp.psi_shape, 0.0004) and np.isclose(hp.psi_scale, 0.0004))
    # "peaks at approximately (0.02)^2": moda IG = scale/(shape+1)
    ok &= _check("...e ha la moda a ~(0.02)^2 come dice GLP",
                 np.isclose(hp.psi_scale / (hp.psi_shape + 1.0), 0.0004, rtol=1e-3))
    ok &= _check("l'iperprior rifiuta valori non positivi",
                 not np.isfinite(hp.log_pdf(-1.0, 1.0, np.array([0.1]))))
    ok &= _check("ed e' finito su valori ammissibili",
                 np.isfinite(hp.log_pdf(0.5, 1.0, np.array([0.01, 0.02]))))
    return ok


# ─── 2. La ML: A.13 vs A.14 vs Chib ───────────────────────────────────────────

def _log_iw(S, Psi, dof):
    n = S.shape[0]
    Ls, Lp = cholesky(S, lower=True), cholesky(Psi, lower=True)
    return (0.5 * dof * 2 * np.sum(np.log(np.diag(Lp))) - 0.5 * dof * n * np.log(2.0)
            - multigammaln(0.5 * dof, n)
            - 0.5 * (n + dof + 1) * 2 * np.sum(np.log(np.diag(Ls)))
            - 0.5 * float(np.trace(np.linalg.solve(S, Psi))))


def _log_mn(B, b, S, Omega):
    k, n = B.shape
    Lo, Ls = cholesky(Omega, lower=True), cholesky(S, lower=True)
    ld = 2 * n * np.sum(np.log(np.diag(Lo))) + 2 * k * np.sum(np.log(np.diag(Ls)))
    D = B - b
    quad = float(np.trace(np.linalg.solve(Omega, D) @ np.linalg.solve(S, D.T)))
    return -0.5 * (k * n * np.log(2 * np.pi) + ld + quad)


def _loglik(Y, X, B, S):
    E = Y - X @ B
    L = cholesky(S, lower=True)
    Z = np.linalg.solve(L, E.T)
    T_, n = Y.shape
    return -0.5 * (T_ * n * np.log(2 * np.pi)
                   + T_ * 2 * np.sum(np.log(np.diag(L))) + float((Z ** 2).sum()))


def test_ml() -> bool:
    print("\n2. La marginal likelihood: A.13, A.14 e l'identita' di Chib")
    rng = np.random.default_rng(1)
    ok = True

    for (n, p, T) in [(2, 1, 25), (3, 2, 60), (4, 3, 80)]:
        spec = _spec(n, p)
        hyp = Hyper(lam=0.5, mu=1.0, psi=np.exp(rng.normal(-1.0, 0.6, size=n)))
        prior = build_prior(spec, hyp)
        Y = rng.normal(size=(T, n))
        X = np.column_stack([rng.normal(size=(T, n * p)), np.ones(T)])

        a13 = log_ml(Y, X, prior, stable=False)
        a14 = log_ml(Y, X, prior, stable=True)
        post = niw_posterior(Y, X, prior)
        Sg = post.psi_bar / post.dof_bar
        Bp = post.b_bar
        chib = (_loglik(Y, X, Bp, Sg)
                + _log_iw(Sg, np.diag(prior.psi), prior.dof)
                + _log_mn(Bp, prior.b, Sg, np.diag(prior.omega_diag))
                - _log_iw(Sg, post.psi_bar, post.dof_bar)
                - _log_mn(Bp, post.b_bar, Sg, post.omega_bar))
        # tolleranza RELATIVA: sono log-ML dell'ordine di 1e2-1e3, e le due
        # forme prendono strade numeriche diverse (autovalori vs slogdet)
        ok &= _check(f"n={n} p={p} T={T}: A.13 == A.14",
                     np.isclose(a13, a14, rtol=1e-9, atol=1e-6),
                     f"|d|={abs(a13-a14):.2e}")
        ok &= _check(f"n={n} p={p} T={T}: A.14 == ML vera (Chib, esatta)",
                     np.isclose(a14, chib, rtol=1e-9, atol=1e-6),
                     f"|d|={abs(a14-chib):.2e}")

    # ---- const_var e il PARADOSSO DI BARTLETT -------------------------------
    # La ML DIPENDE da const_var, ed e' corretto: appiattendo il prior sulla
    # costante la massa a priori si spalma e la ML cala.  Il punto che conta e'
    # che sia una COSTANTE ADDITIVA indipendente dagli iperparametri, cosi' si
    # cancella nel rapporto di accettazione del Metropolis e il posterior di
    # (lambda, mu, psi) resta inalterato.  E' questo che va verificato.
    spec = _spec(3, 2)
    Y = rng.normal(size=(60, 3))
    X = np.column_stack([rng.normal(size=(60, 6)), np.ones(60)])
    h1 = Hyper(lam=0.3, mu=1.0, psi=np.full(3, 0.01))
    h2 = Hyper(lam=0.9, mu=1.5, psi=np.full(3, 0.02))

    vals = [log_ml(Y, X, build_prior(spec, h1, const_var=cv)) for cv in (1e6, 1e8, 1e10)]
    steps = np.diff(vals)
    ok &= _check("log ML cala di esattamente -n/2*log(const_var)  [Bartlett]",
                 bool(np.allclose(steps, -3 / 2 * np.log(100.0), atol=1e-5)),
                 f"passi {steps.round(5)} vs {-3/2*np.log(100):.5f}")

    diffs = [log_ml(Y, X, build_prior(spec, h1, const_var=cv))
             - log_ml(Y, X, build_prior(spec, h2, const_var=cv))
             for cv in (1e6, 1e8, 1e10, 1e12)]
    ok &= _check("...ma le DIFFERENZE fra iperparametri sono invarianti",
                 max(diffs) - min(diffs) < 1e-6,
                 f"escursione = {max(diffs)-min(diffs):.2e}  -> il posterior non cambia")
    return ok


# ─── 3. Il posterior coniugato ────────────────────────────────────────────────

def test_posterior() -> bool:
    print("\n3. Il posterior (A.8)-(A.9)")
    rng = np.random.default_rng(2)
    n, p = 3, 2
    spec = _spec(n, p)
    hyp = Hyper(lam=0.5, mu=1.0, psi=np.full(n, 0.02))
    prior = build_prior(spec, hyp)
    ok = True

    # T = 0: il posterior DEVE coincidere col prior.  E' il controllo che
    # smaschera gli errori di segno e di trasposizione nell'aggiornamento.
    post0 = niw_posterior(np.zeros((0, n)), np.zeros((0, spec.k)), prior)
    ok &= _check("T=0: b_bar == b del prior",
                 bool(np.allclose(post0.b_bar, prior.b)))
    ok &= _check("T=0: omega_bar == diag(omega) del prior",
                 bool(np.allclose(np.diag(post0.omega_bar), prior.omega_diag)))
    ok &= _check("T=0: psi_bar == diag(psi), dof_bar == dof",
                 bool(np.allclose(post0.psi_bar, np.diag(prior.psi)))
                 and post0.dof_bar == prior.dof)

    # prior vago + T grande: b_bar -> OLS
    sim = simulate_var(n, p, 3000, rng)
    pan = sim["panel"]
    Y, X = pan[p:], np.column_stack([pan[p - s: len(pan) - s] for s in range(1, p + 1)]
                                    + [np.ones(len(pan) - p)])
    ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    vague = build_prior(spec, Hyper(lam=1e4, mu=1.0, psi=np.full(n, 0.02)))
    ok &= _check("prior vago + T grande: b_bar -> OLS",
                 bool(np.allclose(niw_posterior(Y, X, vague).b_bar, ols, atol=1e-6)))

    # draw(): la covarianza empirica di vec(B) deve essere Sigma (x) omega_bar
    # ATTENZIONE ALL'ORDINE DI vec: la convenzione di Kronecker Sigma (x) Omega
    # corrisponde a vec che impila le COLONNE (Fortran order).  Un flatten
    # row-major va confrontato con kron(Omega, Sigma), non con kron(Sigma, Omega).
    post = niw_posterior(Y, X, prior)
    S_DRAWS = 20000
    B, S = draw(post, np.random.default_rng(3), n_draws=S_DRAWS)
    dev = np.stack([(B[s] - post.b_bar).flatten(order="F") for s in range(S_DRAWS)])
    emp = np.cov(dev.T)
    theo = np.kron(S.mean(axis=0), post.omega_bar)
    rel = np.abs(emp - theo).max() / np.abs(theo).max()
    ok &= _check("draw(): cov(vec(B)) == Sigma (x) omega_bar  [vec col-major]",
                 rel < 0.06, f"scarto rel max = {rel:.4f}")
    ok &= _check("draw(): E[Sigma] == psi_bar/(dof_bar-n-1)",
                 bool(np.allclose(S.mean(axis=0),
                                  post.psi_bar / (post.dof_bar - n - 1), rtol=0.1)))
    return ok


# ─── 4. Il recovery vero ──────────────────────────────────────────────────────

def test_recovery() -> bool:
    print("\n4. RECOVERY: si ritrovano i parametri veri?")
    ok = True
    # La COPERTURA e' una frequenza: va misurata SU PIU' REPLICHE e messa
    # insieme.  Su una replica sola con n=3 gli elementi di Sigma sono 9, e
    # 5/9 contro 9/9 e' pura oscillazione binomiale, non un segnale.
    pooled_B: list[float] = []
    pooled_S: list[float] = []

    for (n, p, T, seed) in [(3, 2, 400, 10), (3, 2, 400, 13), (4, 1, 500, 11),
                            (4, 1, 500, 15), (5, 2, 600, 12)]:
        rng = np.random.default_rng(seed)
        d = np.ones(n)
        d[-1] = 0.0                                    # una serie wn, come da noi
        sim = simulate_var(n, p, T, rng, d_centre=d)
        spec = _spec(n, p, d_centre=d)
        dr = sample(spec, sim["panel"], n_draws=500, burn=400, rng=rng)

        lo, hi = np.percentile(dr.B, [2.5, 97.5], axis=0)
        cov = float(((sim["B"] >= lo) & (sim["B"] <= hi)).mean())
        corr = float(np.corrcoef(sim["B"].ravel(), dr.B.mean(axis=0).ravel())[0, 1])
        slo, shi = np.percentile(dr.Sigma, [2.5, 97.5], axis=0)
        scov = float(((sim["Sigma"] >= slo) & (sim["Sigma"] <= shi)).mean())
        pooled_B.append(cov)
        pooled_S.append(scov)

        ok &= _check(f"n={n} p={p} T={T} (seed {seed}): corr(B vero, B stimato)",
                     corr > 0.95, f"{corr:.4f}  |  cop.B {cov:.0%}  cop.Sigma {scov:.0%}")
        ok &= _check(f"n={n} p={p} T={T} (seed {seed}): accettazione nella banda",
                     0.10 <= dr.acceptance <= 0.50, f"{dr.acceptance:.1%}")

    mb, ms = float(np.mean(pooled_B)), float(np.mean(pooled_S))
    ok &= _check("copertura CI 95% su B, messa insieme su 5 repliche",
                 0.85 <= mb <= 1.0, f"{mb:.1%}")
    ok &= _check("copertura CI 95% su Sigma, messa insieme su 5 repliche",
                 0.80 <= ms <= 1.0, f"{ms:.1%}")
    return ok


# ─── 5. L'EQUIVALENZA fra le due parametrizzazioni ────────────────────────────

def test_parameterisation_equivalence() -> bool:
    print("\n5. Metropolis: log-parametrizzazione == parametrizzazione naturale")
    print("     (rende DIMOSTRATA la fedelta' all'Appendice B, non asserita)")
    n, p, T = 3, 1, 300
    rng = np.random.default_rng(20)
    sim = simulate_var(n, p, T, rng)
    spec = _spec(n, p)
    target = build_target(spec, sim["panel"])
    ok = True

    # (a) il target su scala log e' quello naturale piu' lo Jacobiano
    g = np.concatenate([[0.4, 1.1], np.full(n, 0.01)])
    ok &= _check("log p(theta) = log p(gamma) + sum(log gamma)  [Jacobiano]",
                 np.isclose(target.log_posterior_log_scale(np.log(g)),
                            target.log_posterior(g) + np.sum(np.log(g))))

    # (b) due catene, stessa partenza, parametrizzazioni diverse
    draws = {}
    for name, log_scale in (("log", True), ("naturale", False)):
        r = np.random.default_rng(77)
        st = init_metropolis(target, r, c=0.4)
        for _ in range(600):
            metropolis_step(target, st, r, log_scale=log_scale)
        acc, keep = [], []
        for _ in range(2500):
            metropolis_step(target, st, r, log_scale=log_scale)
            keep.append(st.gamma.copy())
        draws[name] = np.array(keep)
        acc.append(st.acceptance)
        print(f"       {name:<9} accettazione {st.acceptance:5.1%}")

    a, b = draws["log"], draws["naturale"]
    for i, nm in enumerate(["lambda", "mu"]):
        ma, mb = np.median(a[:, i]), np.median(b[:, i])
        sd = np.std(np.concatenate([a[:, i], b[:, i]]))
        ok &= _check(f"mediana di {nm} coincide fra le due parametrizzazioni",
                     abs(ma - mb) < 0.35 * sd,
                     f"log {ma:.4f} vs naturale {mb:.4f}  (sd {sd:.4f})")
    return ok


# ─── 6. step(): il requisito d'interfaccia dell'L-BVAR ────────────────────────

def test_step_interface() -> bool:
    print("\n6. step(): la primitiva che al Gate 5 si intreccia con lo smoother")
    n, p, T = 3, 1, 300
    rng = np.random.default_rng(30)
    sim = simulate_var(n, p, T, rng)
    spec = _spec(n, p)
    target = build_target(spec, sim["panel"])
    state = CoreState(target=target, metro=init_metropolis(target, rng, c=0.4))
    ok = True

    for _ in range(30):
        step(state, rng)
    ok &= _check("step() produce (B, Sigma) delle forme giuste",
                 state.B.shape == (spec.k, n) and state.Sigma.shape == (n, n))
    ok &= _check("...e iperparametri validi",
                 state.hyper.lam > 0 and state.hyper.mu > 0
                 and np.all(state.hyper.psi > 0))

    # IL PUNTO: cambiare il pannello a meta' catena (cio' che fa l'L-BVAR a ogni
    # iterazione, quando lo smoother estrae un nuovo dataset mensile completo)
    theta_before = state.metro.theta.copy()
    n_before = state.metro.n_prop
    sim2 = simulate_var(n, p, T, np.random.default_rng(31))
    step(state, rng, panel=sim2["panel"])
    ok &= _check("step(panel=...) accetta un pannello NUOVO senza ricominciare",
                 state.metro.n_prop == n_before + 1)
    ok &= _check("...e la catena degli iperparametri prosegue (non si azzera)",
                 state.metro.theta.shape == theta_before.shape
                 and np.isfinite(state.metro.logpost))
    ok &= _check("...e il target e' stato ricostruito sul nuovo pannello",
                 not np.allclose(state.target.Y, target.Y))

    for _ in range(20):
        step(state, rng, panel=sim2["panel"])
    ok &= _check("30 spazzate col pannello che cambia: nessun degrado",
                 np.isfinite(state.metro.logpost) and state.B is not None)
    return ok


def main() -> bool:
    print("=" * 82)
    print("Gate 1 — RECOVERY TEST del core sampler")
    print("=" * 82)
    ok = True
    for fn in (test_hyperprior, test_ml, test_posterior, test_recovery,
               test_parameterisation_equivalence, test_step_interface):
        ok &= fn()
    print("\n" + "=" * 82)
    print("TUTTO OK" if ok else "QUALCOSA NON TORNA")
    print("=" * 82)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
