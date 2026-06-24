"""
diagnostics/density_nowcast_feasibility.py

VERIFICA DI FATTIBILITA' (non l'apparato completo) della soluzione
"density nowcasting": generare la densita' predittiva del GDP simulando
DIRETTAMENTE dal DFM Student-t stimato, invece della quantile regression
separata.

Domanda critica
---------------
La densita' predittiva del GDP cosi' generata e' abbastanza FAT-TAILED /
larga da catturare il rischio di coda (i crolli cadono DENTRO, in una coda
con massa non trascurabile), o eredita la COMPRESSIONE degli estremi del
nowcast puntuale (Volatility Paradox)?

Le code grasse dello Student-t vivono su DUE canali:
  - innovazioni del fattore  u_t ~ t_{nu_u}(0, Q)      (nu_u ~ 4.08)
  - idiosincratico del GDP    eps_t ~ t_{nu_eps}(0,R)   (nu_eps ~ 4.40)
La domanda e' se questi si PROPAGANO alla densita' del GDP.

Cosa fa lo script (solo diagnosi, NESSUN fan-chart / IRF / shock-decomp)
-----------------------------------------------------------------------
Usa il fit FULL-SAMPLE gia' su disco (data/processed/small/fit_dfm_result.npz:
theta completo + f_smooth/P_smooth augmented 15-dim).  Per alcuni trimestri
(calmi + crisi 2008Q4, 2020Q2) costruisce la densita' predittiva del GDP via
Monte Carlo in DUE varianti, piu' il controfattuale gaussiano:

  V1  "full-info / canale idiosincratico":  GDP | fattori-smussati(t*).
      Campiona lo stato augmented ~ N(f_smooth[t*], P_smooth[t*]) e aggiunge
      l'idiosincratico Student-t.  Isola: noti i fattori, la coda
      idiosincratica + incertezza di stato bastano a raggiungere il crollo?

  V2  "forward / nowcast vero":  condiziona sullo stato a fine trimestre
      PRECEDENTE (t* - 3) e SIMULA in avanti i 3 mesi del trimestre target
      propagando il VAR(1) con pesi Gamma + shock-t, poi MM-aggrega e
      genera il GDP.  E' la densita' predittiva "generativa" descritta:
      estrai fattori, pesi Gamma, shock t, propaga, genera GDP.  Qui vive il
      Volatility Paradox: la proiezione A^h f shrinka verso 0; gli shock-t
      bastano ad allargare la densita' fino al crollo?

  GAUSS  controfattuale: la STESSA V2 ma con nu_u=nu_eps=inf (pesi=1).
      Confronto V2-studentt vs V2-gauss = "quanto allargano le code grasse".

Per ogni trimestre/variante: media, std, skew, EXCESS KURTOSIS, quantili
(1/5/10/50/90/95/99 %), realizzato, e dove cade il realizzato:
  PIT = P_sim(GDP <= realizzato)   (= percentile del realizzato nella densita')
  confronto con il PIT GAUSSIANO   (stessa media/std ma normale): le code
  grasse danno ORDINI DI GRANDEZZA piu' massa al crollo?

Punto 4: confronto con la quantile regression del fattore reale (gia'
diagnosticata) sugli stessi trimestri: q05 e ampiezza q95-q05.

Run:
  python -m diagnostics.density_nowcast_feasibility
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

from src.data_loader import load_config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "diagnostics", "out")
os.makedirs(_OUT, exist_ok=True)

_R = 3
_MM = np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])   # (f_t, f_{t-1}, .., f_{t-4})
_S = 200_000                                         # draws Monte Carlo per densita'
_TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# trimestri da testare: due calmi + le due crisi richieste + contorno 2020
_TARGETS = [
    ("2006-12-31", "2006Q4 (calmo)"),
    ("2014-12-31", "2014Q4 (calmo)"),
    ("2008-12-31", "2008Q4 CRISI"),
    ("2009-03-31", "2009Q1 (coda crisi)"),
    ("2020-03-31", "2020Q1 (inizio Covid)"),
    ("2020-06-30", "2020Q2 CRISI Covid"),
]


def gamma_weights(nu: float, size, rng) -> np.ndarray:
    """w ~ Gamma(nu/2, 2/nu): media 1, var 2/nu. nu=inf -> 1 (limite gaussiano)."""
    if np.isinf(nu):
        return np.ones(size)
    return rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=size)


def simulate_density_V1(theta, fs, Ps, t_star, gi, S, rng, gaussian=False):
    """Densita' predittiva del GDP (std units) condizionata ai fattori SMUSSATI
    a t* (full info).  Stato augmented ~ N(f_smooth[t*], P_smooth[t*]) +
    idiosincratico Student-t.  Isola il canale idiosincratico."""
    Lamt = theta["Lambda_tilde"][gi]          # (15,)
    R_gdp = theta["R"][gi]
    nu_eps = np.inf if gaussian else theta["nu_eps"]
    mean = fs[t_star]                          # (15,)
    cov = 0.5 * (Ps[t_star] + Ps[t_star].T)
    # campiona lo stato augmented dalla posterior smoothing
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    aug = mean[None, :] + rng.standard_normal((S, mean.size)) @ L.T
    signal = aug @ Lamt                        # (S,)
    w_eps = gamma_weights(nu_eps, S, rng)
    eps = rng.standard_normal(S) * np.sqrt(R_gdp / w_eps)
    return signal + eps


def simulate_density_V2(theta, fs, Ps, t_star, gi, S, rng, gaussian=False,
                        w_u_oracle=None, w_eps_oracle=None):
    """Densita' predittiva FORWARD: condiziona sullo stato a t0=t*-3 (fine
    trimestre precedente) e simula in avanti i 3 mesi del trimestre target,
    propagando il VAR(1) con shock-t, poi MM-aggrega e genera GDP.
    E' la densita' 'nowcast generativa'.

    ORACOLO (opzione b): se w_u_oracle (3 valori, mesi t*-2,t*-1,t*) e/o
    w_eps_oracle (scalare, mese t*) sono forniti, lo SCALE degli shock viene
    FISSATO al valore SMOOTHED che il modello stima in-sample per quel
    trimestre (regime-dipendente), invece di estrarre w dalla prior Gamma.
    Conditional-on-w: la mistura sparisce (no curtosi extra) ma la SCALA e'
    quella del regime -> e' il test 'e se il modello sapesse il w giusto?'."""
    A = theta["A"]; Q = theta["Q"]; Lam = theta["Lambda"][gi]   # (3,)
    R_gdp = theta["R"][gi]
    nu_u = np.inf if gaussian else theta["nu_u"]
    nu_eps = np.inf if gaussian else theta["nu_eps"]
    LQ = np.linalg.cholesky(0.5 * (Q + Q.T))

    t0 = t_star - 3
    mean0 = fs[t0]; cov0 = 0.5 * (Ps[t0] + Ps[t0].T)
    L0 = np.linalg.cholesky(cov0 + 1e-12 * np.eye(cov0.shape[0]))
    aug0 = mean0[None, :] + rng.standard_normal((S, mean0.size)) @ L0.T
    f_tm3 = aug0[:, 0:3]        # f_{t*-3}  (lag0 dello stato a t0)
    f_tm4 = aug0[:, 3:6]        # f_{t*-4}  (lag1 dello stato a t0)

    # propaga 3 passi avanti: f_{t*-2}, f_{t*-1}, f_{t*}
    f_prev = f_tm3
    fs_fwd = []
    for h in range(3):
        if w_u_oracle is not None:
            w_u = np.full(S, w_u_oracle[h])      # SCALE fissato al regime
        else:
            w_u = gamma_weights(nu_u, S, rng)
        u = (rng.standard_normal((S, 3)) @ LQ.T) / np.sqrt(w_u)[:, None]
        f_new = f_prev @ A.T + u
        fs_fwd.append(f_new)
        f_prev = f_new
    f_tm2, f_tm1, f_t = fs_fwd                  # ognuno (S,3)

    # MM aggregate: c0 f_t + c1 f_{t-1} + c2 f_{t-2} + c3 f_{t-3} + c4 f_{t-4}
    Phi = (_MM[0] * f_t + _MM[1] * f_tm1 + _MM[2] * f_tm2
           + _MM[3] * f_tm3 + _MM[4] * f_tm4)    # (S,3)
    signal = Phi @ Lam                           # (S,)
    if w_eps_oracle is not None:
        w_eps = np.full(S, w_eps_oracle)         # SCALE idiosinc. al regime
    else:
        w_eps = gamma_weights(nu_eps, S, rng)
    eps = rng.standard_normal(S) * np.sqrt(R_gdp / w_eps)
    return signal + eps


def describe(sim_std, mu, sd, realized_nat):
    """Statistiche di coda di una densita' (campioni in std units) in unita'
    NATURALI, + dove cade il realizzato."""
    sim = sim_std * sd + mu
    q = np.quantile(sim, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    m, s = sim.mean(), sim.std()
    pit = float(np.mean(sim <= realized_nat))
    pit_gauss = float(stats.norm.cdf((realized_nat - m) / s))
    return {
        "mean": m, "std": s,
        "skew": float(stats.skew(sim)),
        "exkurt": float(stats.kurtosis(sim)),   # eccesso (0 = gaussiano)
        "q01": q[0], "q05": q[1], "q10": q[2], "q50": q[3],
        "q90": q[4], "q95": q[5], "q99": q[6],
        "pit": pit, "pit_gauss": pit_gauss,
        "_sim": sim,
    }


def qreg_predictive(panel, fs, idx, gi, mu, sd, target_ts):
    """Punto 4: quantile regression FULL-SAMPLE del GDP (naturale) sul fattore
    reale MM-trimestrale; predice q05/q50/q95 al trimestre target.
    Restituisce dict per trimestre."""
    # fattore reale MM trimestrale a ogni quarter-end
    qe = idx[idx.month.isin([3, 6, 9, 12])]
    lags_real = np.array([lag * _R + 0 for lag in range(5)])    # blocco real=0
    freal = pd.Series(
        [float(fs[idx.get_loc(d), lags_real] @ _MM) for d in qe], index=qe
    )
    gdp = panel["GDPC1"].dropna()
    common = qe[qe.isin(gdp.index)]
    train = pd.DataFrame({"f_real": freal.loc[common].to_numpy(),
                          "gdp": gdp.loc[common].to_numpy()}, index=common)
    X = sm.add_constant(train[["f_real"]])
    preds = {}
    for ts in target_ts:
        if ts not in freal.index:
            continue
        x_t = np.array([1.0, freal.loc[ts]])
        qs = {}
        for tau in _TAUS:
            res = QuantReg(train["gdp"], X).fit(q=tau, max_iter=5000)
            qs[tau] = float(res.params.iloc[0] + res.params.iloc[1] * x_t[1])
        # rearrangement anti-crossing
        vals = np.sort(np.array([qs[t] for t in _TAUS]))
        qs = dict(zip(_TAUS, vals))
        preds[ts] = qs
    return preds


def main():
    report = []

    def log(s=""):
        print(s); report.append(s)

    # ---- carica fit full-sample e dati ----
    z = np.load(os.path.join(_ROOT, "data", "processed", "small",
                             "fit_dfm_result.npz"), allow_pickle=True)
    theta = {
        "Lambda": z["theta__Lambda"], "A": z["theta__A"], "Q": z["theta__Q"],
        "R": z["theta__R"], "nu_u": float(z["theta__nu_u"]),
        "nu_eps": float(z["theta__nu_eps"]),
        "Lambda_tilde": z["estep__Lambda_tilde"],
    }
    fs = z["f_smooth"]; Ps = z["P_smooth"]

    cfg = load_config("small")
    cols = cfg["ORDERED_COLS"]
    gi = cols.index("GDPC1")
    panel = pd.read_csv(os.path.join(_ROOT, "data", "processed",
                                     "dataset_small.csv"), index_col=0)
    panel.index = pd.to_datetime(panel.index)
    idx = panel.index
    gdp = panel["GDPC1"].dropna()
    mu, sd = gdp.mean(), gdp.std(ddof=0)

    log("=" * 78)
    log("DENSITY NOWCASTING - VERIFICA FATTIBILITA' (tail risk)")
    log("=" * 78)
    log(f"fit full-sample small student_t  T={len(idx)}  "
        f"({idx[0].date()}..{idx[-1].date()})")
    log(f"GDP naturale: mean={mu:+.3f} std={sd:.3f} (100*dlog, SAAR-non-annual)")
    log(f"Lambda_GDP (solo blocco reale) = {theta['Lambda'][gi,0]:.4f}   "
        f"R_GDP(var idiosinc, std units) = {theta['R'][gi]:.4f}")
    log(f"nu_u = {theta['nu_u']:.3f}   nu_eps = {theta['nu_eps']:.3f}   "
        f"(curtosi finita solo per nu>4 -> code MOLTO grasse)")
    log(f"rho(A) = {max(abs(np.linalg.eigvals(theta['A']))):.4f}   "
        f"S = {_S} draws/densita'")
    log("NOTA: lo stato di partenza usa P_smooth (full-sample) -> condiziona")
    log("      su info che gia' 'vede' un po' la crisi; il test reale-time")
    log("      (stato filtrato) sarebbe PIU' severo. Quindi e' un test")
    log("      CONSERVATIVO-favorevole: se qui non cattura, real-time peggio.")

    rng = np.random.default_rng(12345)
    target_ts = [pd.Timestamp(d) for d, _ in _TARGETS]
    qr = qreg_predictive(panel, fs, idx, gi, mu, sd, target_ts)

    rows = []
    dens_store = {}
    for dstr, label in _TARGETS:
        ts = pd.Timestamp(dstr)
        if ts not in gdp.index:
            log(f"\n[skip] {label}: GDP non disponibile")
            continue
        t_star = idx.get_loc(ts)
        realized = float(gdp.loc[ts])
        fitted_std = float(theta["Lambda_tilde"][gi] @ fs[t_star])
        fitted_nat = fitted_std * sd + mu

        v1 = describe(simulate_density_V1(theta, fs, Ps, t_star, gi, _S, rng),
                      mu, sd, realized)
        v2 = describe(simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng),
                      mu, sd, realized)
        v2g = describe(simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng,
                                           gaussian=True), mu, sd, realized)
        dens_store[label] = {"V1": v1, "V2": v2, "V2g": v2g,
                             "realized": realized, "fitted": fitted_nat}

        log("\n" + "-" * 78)
        log(f"{label}   [{dstr}]")
        log(f"  realizzato = {realized:+.3f} nat   |  fitted (smoother) = "
            f"{fitted_nat:+.3f} nat   (Volatility Paradox: |fitted| << |realizz|)")
        log(f"  {'variante':22s} {'mean':>7s} {'std':>6s} {'exkurt':>7s} "
            f"{'q05':>7s} {'q01':>7s} {'PIT':>9s} {'PIT_gauss':>11s}")
        for nm, d in [("V1 full-info (idio)", v1),
                      ("V2 forward (t-DFM)", v2),
                      ("V2 forward GAUSS", v2g)]:
            log(f"  {nm:22s} {d['mean']:+7.2f} {d['std']:6.2f} "
                f"{d['exkurt']:7.2f} {d['q05']:+7.2f} {d['q01']:+7.2f} "
                f"{d['pit']:9.5f} {d['pit_gauss']:11.2e}")
        # interpretazione coda per V2 student-t
        if qr.get(ts):
            qq = qr[ts]
            log(f"  QuantReg(f_real) : q05={qq[0.05]:+.2f}  q50={qq[0.50]:+.2f}  "
                f"q95={qq[0.95]:+.2f}  ampiezza(q95-q05)={qq[0.95]-qq[0.05]:.2f}")
            log(f"  V2 t-DFM         : q05={v2['q05']:+.2f}  q50={v2['q50']:+.2f}  "
                f"q95={v2['q95']:+.2f}  ampiezza(q95-q05)={v2['q95']-v2['q05']:.2f}")
        rows.append({
            "trimestre": label, "realizzato": realized, "fitted": fitted_nat,
            "V1_std": v1["std"], "V1_exkurt": v1["exkurt"], "V1_PIT": v1["pit"],
            "V2_std": v2["std"], "V2_exkurt": v2["exkurt"], "V2_PIT": v2["pit"],
            "V2_q05": v2["q05"], "V2_q01": v2["q01"],
            "V2g_std": v2g["std"], "V2g_PIT": v2g["pit"],
            "qr_q05": qr.get(ts, {}).get(0.05, np.nan),
        })

    # ============================================================ TEST ORACOLO
    # Opzione (b): scale REGIME-DIPENDENTE. Invece di estrarre w dalla prior
    # Gamma in forward, FORZA w al valore SMOOTHED stimato in-sample per quel
    # trimestre. "E se il modello sapesse il w giusto?" (oracolo: usa info sul
    # realizzato -> upper bound di fattibilita').
    w_eps_sm = z["theta__w_eps"]; w_u_sm = z["theta__w_u"]
    log("\n" + "=" * 78)
    log("TEST ORACOLO (b): w SMOOTHED (regime) vs w da PRIOR Gamma")
    log("=" * 78)
    log("Oracolo = forza lo scale degli shock al w smoothed in-sample del")
    log("trimestre (usa info sul realizzato -> upper bound). w_u sui 3 mesi")
    log("forward, w_eps al mese quarter-end.")
    oracle_rows = []
    for dstr, label in _TARGETS:
        if "CRISI" not in label and "Covid" not in label:
            continue
        ts = pd.Timestamp(dstr)
        if ts not in gdp.index:
            continue
        t_star = idx.get_loc(ts)
        realized = float(gdp.loc[ts])
        # w smoothed dei 3 mesi forward (t*-2,t*-1,t*) e w_eps al quarter-end
        wu_or = [w_u_sm[t_star - 2], w_u_sm[t_star - 1], w_u_sm[t_star]]
        weps_or = w_eps_sm[t_star]
        mm = [(idx[t_star - 2].date(), w_u_sm[t_star - 2]),
              (idx[t_star - 1].date(), w_u_sm[t_star - 1]),
              (idx[t_star].date(), w_u_sm[t_star])]

        prior = dens_store[label]["V2"]               # gia' calcolato (w-prior)
        orac = describe(simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng,
                                            w_u_oracle=wu_or,
                                            w_eps_oracle=weps_or),
                        mu, sd, realized)
        log("\n" + "-" * 78)
        log(f"{label}   realizzato = {realized:+.3f} nat")
        log(f"  w_u smoothed forward: " +
            "  ".join(f"{d}={w:.4f}" for d, w in mm))
        log(f"  w_eps smoothed quarter-end ({idx[t_star].date()}) = {weps_or:.4f}"
            f"  -> idio std = {np.sqrt(theta['R'][gi]/weps_or):.3f} std units")
        log(f"  {'':14s} {'std(nat)':>9s} {'q05':>8s} {'q01':>8s} "
            f"{'exkurt':>8s} {'PIT':>10s}")
        log(f"  {'w-PRIOR':14s} {prior['std']:9.3f} {prior['q05']:+8.2f} "
            f"{prior['q01']:+8.2f} {prior['exkurt']:8.2f} {prior['pit']:10.5f}")
        log(f"  {'w-SMOOTHED':14s} {orac['std']:9.3f} {orac['q05']:+8.2f} "
            f"{orac['q01']:+8.2f} {orac['exkurt']:8.2f} {orac['pit']:10.5f}")
        log(f"  --> std x{orac['std']/prior['std']:.1f}   "
            f"PIT {prior['pit']:.5f} -> {orac['pit']:.5f}   "
            f"(realizzato {'DENTRO' if 0.005 <= orac['pit'] <= 0.5 else 'ai bordi' if orac['pit']>1e-4 else 'ancora FUORI'})")
        dens_store[label]["V2_oracle"] = orac
        oracle_rows.append({
            "trimestre": label, "realizzato": realized,
            "std_prior": prior["std"], "std_oracle": orac["std"],
            "PIT_prior": prior["pit"], "PIT_oracle": orac["pit"],
            "q05_oracle": orac["q05"], "q01_oracle": orac["q01"],
        })

    log("\n" + "-" * 78)
    log("SINTESI ORACOLO (crolli):")
    log(pd.DataFrame(oracle_rows).set_index("trimestre").to_string(
        float_format=lambda v: f"{v:+.5f}"))

    # ---- verdetto sintetico ----
    log("\n" + "=" * 78)
    log("VERDETTO")
    log("=" * 78)
    df = pd.DataFrame(rows).set_index("trimestre")
    log(df.to_string(float_format=lambda v: f"{v:+.4f}"))
    log("")
    log("Lettura PIT (= percentile del realizzato nella densita' predittiva V2):")
    log("  PIT ~ 0.05-0.95 = realizzato BEN dentro la densita' (coda usabile)")
    log("  PIT < 0.01 o > 0.99 = realizzato nella coda ESTREMA (quasi fuori)")
    log("  PIT ~ 0 (e.g. 1e-4) = densita' troppo stretta -> COMPRESSIONE")
    log("Confronto PIT (t-DFM) vs PIT_gauss quantifica quanto le code grasse")
    log("spostano massa verso il crollo: se PIT >> PIT_gauss, le code-t SI")
    log("propagano; se realizzato resta a PIT~0 comunque, la coda-t non basta.")

    # ---- figura ----
    crisis = [l for _, l in _TARGETS if "CRISI" in l or "Covid" in l]
    fig, axes = plt.subplots(1, len(crisis), figsize=(6 * len(crisis), 5),
                             squeeze=False)
    for ax, label in zip(axes[0], crisis):
        if label not in dens_store:
            continue
        st = dens_store[label]
        curves = [("V2 t-DFM (w-prior)", "#1f77b4", st["V2"]),
                  ("V2 Gauss", "#7f7f7f", st["V2g"])]
        if "V2_oracle" in st:
            curves.append(("V2 ORACOLO (w-smoothed)", "#d62728", st["V2_oracle"]))
        for nm, col, d in curves:
            xs = d["_sim"]
            lo, hi = np.quantile(xs, [0.001, 0.999])
            grid = np.linspace(min(lo, st["realized"]) - 0.5,
                               max(hi, st["realized"]) + 0.5, 500)
            kde = stats.gaussian_kde(xs)
            ax.plot(grid, kde(grid), color=col, lw=1.6, label=nm)
        ax.axvline(st["realized"], color="red", lw=2,
                   label=f"realizzato {st['realized']:+.2f}")
        ax.axvline(st["fitted"], color="black", ls="--", lw=1,
                   label=f"fitted {st['fitted']:+.2f}")
        ax.set_title(label); ax.set_xlabel("GDP (100*dlog, naturale)")
        ax.set_ylabel("densita'"); ax.legend(fontsize=7); ax.grid(alpha=0.2)
    fig.suptitle("Densita' predittiva del GDP simulata dal DFM Student-t "
                 "vs realizzato (crisi)", fontsize=11)
    fig.tight_layout()
    figpath = os.path.join(_OUT, "density_nowcast_feasibility.png")
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"\nfigura -> {figpath}")

    with open(os.path.join(_OUT, "density_nowcast_feasibility.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"report -> {os.path.join(_OUT, 'density_nowcast_feasibility.txt')}")


if __name__ == "__main__":
    main()
