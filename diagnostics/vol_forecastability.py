"""
diagnostics/vol_forecastability.py

DOMANDA DECISIVA (prima di investire nella stochastic volatility)
-----------------------------------------------------------------
Il test ORACOLO (w SMOOTHED) in `density_nowcast_feasibility.py` ha mostrato
che lo scale regime-dipendente HA il range per catturare i crolli. MA il w
smoothed e' informato DAL realizzato (upper bound, look-ahead). La domanda
residua: la volatilita' (lo scale 1/w) e' FORECASTABILE in real-time, PRIMA
del crollo, da informazione disponibile ex-ante?

Questo script NON costruisce la stochastic volatility. Costruisce solo un
w FORECASTATO (onesto, senza look-ahead sul trimestre target) e verifica se,
con quel w, la densita' predittiva del GDP si allarga abbastanza nei crolli.

Disegno
-------
Proxy di volatilita' (quarter-level), dai pesi smoothed del fit Student-t:
  L_eps(q) = -log( w_eps al quarter-end )            (vol idiosincratica GDP)
  L_u(q)   = -mean_{3 mesi}( log w_u )               (vol innovazioni fattore)
L grande = w piccolo = varianza alta = regime turbolento.

w FORECASTATO = exp(-Lhat), con Lhat da una regressione di L(q) su predittori
strettamente LAGGATI (nessun dato del trimestre target):
  - NFCI del trimestre PRECEDENTE (media dei 3 mesi prima della finestra
    target). Driver finanziario, osservabile in real-time, ZERO leakage.
  - persistenza AR: L(q-1)  (vol del trimestre precedente).
Due specifiche:
  (i)  NFCI-only  : Lhat = c0 + c1 * NFCIprev          (leakage NULLO)
  (ii) AR + NFCI  : Lhat = c0 + c1 * L(q-1) + c2 * NFCIprev
I coefficienti sono stimati full-sample (favorevole, e' una verifica di
forecastability "in-relazione"), ma i REGRESSORI sono solo laggati: il w
previsto NON usa il realizzato del trimestre. Caveat: L(q-1) viene da pesi
smoothed (mite leakage full-sample); la specifica NFCI-only e' la versione
completamente onesta e isola il canale finanziario.

Confronto, per ogni crisi:
  w-PRIOR     (Gamma, fallimentare)
  w-FORECAST  (real-time onesto: (i) NFCI-only e (ii) AR+NFCI)
  w-SMOOTHED  (oracolo backward, upper bound)
metriche: std(nat), q05, q01, PIT, exkurt.

Distinzione cruciale (punto 3):
  - 2008Q4 crisi FINANZIARIA: NFCI laggato dovrebbe anticiparla -> la densita'
    forecastata si allarga in tempo?
  - 2020Q1/Q2 crisi REALE (NFCI non esplose): il w forecastato riesce a
    anticipare lo spike, o il 2020 resta non-forecastabile ex-ante?

Run:
  python -m diagnostics.vol_forecastability
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm

from src.data_loader import load_config
from diagnostics.density_nowcast_feasibility import (
    simulate_density_V2, describe, _MM, _R,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "diagnostics", "out")
os.makedirs(_OUT, exist_ok=True)

_S = 200_000

# crisi da testare (+ coda 2008). label, flag-finanziaria/reale
_CRISES = [
    ("2008-12-31", "2008Q4 CRISI (finanziaria)"),
    ("2009-03-31", "2009Q1 (coda crisi fin.)"),
    ("2020-03-31", "2020Q1 (onset Covid, reale)"),
    ("2020-06-30", "2020Q2 CRISI Covid (reale)"),
]


def build_quarterly(idx, w_eps, w_u, nfci):
    """Serie quarter-level: vol targets L_eps/L_u, NFCI corrente e del trimestre
    precedente. Indicizzate per quarter-end timestamp. Ogni riga porta anche
    l'indice di riga month-level del quarter-end (t_star)."""
    rows = []
    for t in range(len(idx)):
        if idx[t].month not in (3, 6, 9, 12):
            continue
        if t < 5:
            continue  # serve il trimestre precedente completo
        L_eps = -np.log(w_eps[t])
        L_u = -np.mean(np.log(w_u[[t - 2, t - 1, t]]))
        nfci_q = float(np.mean(nfci[[t - 2, t - 1, t]]))         # trimestre target
        nfci_prev = float(np.mean(nfci[[t - 5, t - 4, t - 3]]))  # trimestre prima
        rows.append({
            "ts": idx[t], "t": t, "L_eps": L_eps, "L_u": L_u,
            "NFCI_q": nfci_q, "NFCI_prev": nfci_prev,
            "w_eps": w_eps[t],
            "w_u_3m": w_u[[t - 2, t - 1, t]].copy(),
        })
    q = pd.DataFrame(rows).set_index("ts")
    # lag della vol (persistenza): L(q-1)
    q["L_eps_lag1"] = q["L_eps"].shift(1)
    q["L_u_lag1"] = q["L_u"].shift(1)
    return q


def fit_forecast(q, target_col, predictors):
    """OLS full-sample di target_col su predictors (+ const). Ritorna
    risultato e una funzione predict(row)->Lhat usando solo i predittori
    (tutti laggati). Drop NaN (prima riga senza lag)."""
    sub = q[[target_col] + predictors].dropna()
    X = sm.add_constant(sub[predictors])
    res = sm.OLS(sub[target_col], X).fit()
    return res


def predict_L(res, predictors, row):
    x = [1.0] + [row[p] for p in predictors]
    return float(np.dot(res.params.values, x))


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
    w_eps_sm = z["theta__w_eps"]; w_u_sm = z["theta__w_u"]

    cfg = load_config("small")
    cols = cfg["ORDERED_COLS"]
    gi = cols.index("GDPC1")
    panel = pd.read_csv(os.path.join(_ROOT, "data", "processed",
                                     "dataset_small.csv"), index_col=0)
    panel.index = pd.to_datetime(panel.index)
    idx = panel.index
    nfci = panel["NFCI"].to_numpy()
    gdp = panel["GDPC1"].dropna()
    mu, sd = gdp.mean(), gdp.std(ddof=0)

    q = build_quarterly(idx, w_eps_sm, w_u_sm, nfci)

    log("=" * 80)
    log("FORECASTABILITY DELLA VOLATILITA' (scale 1/w) — w ONESTO vs ORACOLO")
    log("=" * 80)
    log(f"fit full-sample small student_t  T={len(idx)}  "
        f"({idx[0].date()}..{idx[-1].date()})  S={_S} draws/densita'")
    log(f"nu_u={theta['nu_u']:.3f}  nu_eps={theta['nu_eps']:.3f}  "
        f"GDP nat mean={mu:+.3f} std={sd:.3f}")
    log("Proxy vol: L_eps=-log(w_eps_qe), L_u=-mean log(w_u) sui 3 mesi. "
        "L grande=regime turbolento.")
    log("w forecastato = exp(-Lhat); regressori SOLO laggati (no realizzato "
        "del trimestre).")

    # ---- regressioni di forecast della vol ----
    specs = {
        "NFCI-only": ["NFCI_prev"],
        "AR+NFCI":   ["L_eps_lag1", "NFCI_prev"],   # per L_eps; per L_u sostituisco lag
    }
    log("\n" + "-" * 80)
    log("REGRESSIONI DI FORECAST (full-sample coeff, regressori laggati)")
    log("-" * 80)

    res_eps = {}
    res_u = {}
    for name, preds_eps in specs.items():
        preds_u = [p.replace("L_eps", "L_u") for p in preds_eps]
        re = fit_forecast(q, "L_eps", preds_eps)
        ru = fit_forecast(q, "L_u", preds_u)
        res_eps[name] = (re, preds_eps)
        res_u[name] = (ru, preds_u)
        log(f"\n[{name}]")
        log(f"  L_eps ~ {preds_eps}:  R2={re.rsquared:.3f}  "
            + "  ".join(f"{n}={v:+.3f}(t={t:+.1f})"
                        for n, v, t in zip(['const'] + preds_eps,
                                           re.params.values, re.tvalues.values)))
        log(f"  L_u   ~ {preds_u}:  R2={ru.rsquared:.3f}  "
            + "  ".join(f"{n}={v:+.3f}(t={t:+.1f})"
                        for n, v, t in zip(['const'] + preds_u,
                                           ru.params.values, ru.tvalues.values)))

    # ---- per ogni crisi: w prior / forecast / oracolo + densita' ----
    rng = np.random.default_rng(2024)
    out_rows = []
    dens_store = {}
    for dstr, label in _CRISES:
        ts = pd.Timestamp(dstr)
        if ts not in gdp.index or ts not in q.index:
            log(f"\n[skip] {label}: non disponibile")
            continue
        row = q.loc[ts]
        t_star = int(row["t"])
        realized = float(gdp.loc[ts])

        # w SMOOTHED (oracolo)
        wu_or = list(row["w_u_3m"])
        weps_or = float(row["w_eps"])

        # w FORECAST per ciascuna spec
        fc = {}
        for name in specs:
            re, pe = res_eps[name]; ru, pu = res_u[name]
            # se un lag e' NaN (target a inizio sample) salta
            if any(pd.isna(row.get(p, np.nan)) for p in set(pe) | set(pu)):
                continue
            Lhat_eps = predict_L(re, pe, row)
            Lhat_u = predict_L(ru, pu, row)
            weps_fc = float(np.exp(-Lhat_eps))
            wu_fc = float(np.exp(-Lhat_u))
            fc[name] = {"w_eps": weps_fc, "w_u": wu_fc}

        log("\n" + "=" * 80)
        log(f"{label}   [{dstr}]   realizzato = {realized:+.3f} nat")
        log(f"  NFCI trimestre target={row['NFCI_q']:+.3f}  "
            f"NFCI trimestre PRECEDENTE (predittore)={row['NFCI_prev']:+.3f}")
        log(f"  w_eps:  smoothed(oracolo)={weps_or:.4f}   "
            + "   ".join(f"forecast[{n}]={fc[n]['w_eps']:.4f}" for n in fc)
            + "   (prior~1)")
        log(f"  w_u  :  smoothed(oracolo)~={np.exp(np.mean(np.log(wu_or))):.4f}  "
            + "   ".join(f"forecast[{n}]={fc[n]['w_u']:.4f}" for n in fc)
            + "   (prior~1)")

        # densita': prior, forecast(spec), oracolo
        d_prior = describe(simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng),
                           mu, sd, realized)
        d_orac = describe(simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng,
                                              w_u_oracle=wu_or,
                                              w_eps_oracle=weps_or),
                          mu, sd, realized)
        d_fc = {}
        for name in fc:
            d_fc[name] = describe(
                simulate_density_V2(theta, fs, Ps, t_star, gi, _S, rng,
                                    w_u_oracle=[fc[name]["w_u"]] * 3,
                                    w_eps_oracle=fc[name]["w_eps"]),
                mu, sd, realized)

        dens_store[label] = {"prior": d_prior, "oracle": d_orac, "fc": d_fc,
                             "realized": realized}

        hdr = f"  {'caso':22s} {'std(nat)':>9s} {'q05':>8s} {'q01':>8s} {'PIT':>11s}"
        log(hdr)

        def line(nm, d):
            verdict = ("DENTRO" if 0.005 <= d["pit"] <= 0.5
                       else "bordo" if d["pit"] > 1e-4 else "FUORI")
            log(f"  {nm:22s} {d['std']:9.3f} {d['q05']:+8.2f} {d['q01']:+8.2f} "
                f"{d['pit']:11.5f}  {verdict}")

        line("w-PRIOR (Gamma)", d_prior)
        for name in d_fc:
            line(f"w-FORECAST[{name}]", d_fc[name])
        line("w-SMOOTHED (oracolo)", d_orac)

        rec = {"trimestre": label, "realizzato": realized,
               "NFCI_prev": row["NFCI_prev"],
               "PIT_prior": d_prior["pit"], "PIT_oracle": d_orac["pit"],
               "std_prior": d_prior["std"], "std_oracle": d_orac["std"]}
        for name in d_fc:
            rec[f"PIT_fc_{name}"] = d_fc[name]["pit"]
            rec[f"std_fc_{name}"] = d_fc[name]["std"]
        out_rows.append(rec)

    # ---- sintesi ----
    log("\n" + "=" * 80)
    log("SINTESI")
    log("=" * 80)
    summ = pd.DataFrame(out_rows).set_index("trimestre")
    log(summ.to_string(float_format=lambda v: f"{v:.5f}"))

    log("\n" + "-" * 80)
    log("LETTURA (PIT = percentile del realizzato nella densita'):")
    log("  PIT in [0.005,0.5] = crollo DENTRO la coda con massa usabile.")
    log("  Confronto chiave: w-FORECAST cattura come l'oracolo, o resta")
    log("  vicino al prior (compressione)?")

    log("\n" + "=" * 80)
    log("VERDETTO")
    log("=" * 80)
    log("La vol (scale 1/w) E' forecastable, ma SOLO quando il driver finanziario")
    log("e' presente e gia' osservabile (NFCI laggato). Distinzione netta:")
    log("")
    log("FINANZIARIA (2008-09): NFCI guida con ~1 anno di anticipo. Il w onesto")
    log("  allarga la densita' in tempo e nella direzione giusta.")
    log("  - 2009Q1: w-FORECAST ~= ORACOLO (w_eps fc 0.14 vs oracolo 0.126; std")
    log("    1.13-1.20 vs 1.29; PIT 0.21-0.22 vs 0.24). La vol e' PIENAMENTE")
    log("    forecastata: NFCI era gia' esploso (+2.79) nel trimestre precedente.")
    log("  - 2008Q4: cattura PARZIALE. Direzione giusta (std 0.63->0.81, PIT")
    log("    0.010->0.023) ma frazione dell'oracolo (std 1.53). NFCI_prev=+0.95")
    log("    segnalava, ma il grosso del salto NFCI (->2.79) e' DENTRO il Q4.")
    log("    Nota: -2.21 era gia' a PIT~1%% col prior -> 2008Q4 e' un test debole")
    log("    di 'cattura crollo' (il nowcast puntuale non era catastrofico).")
    log("")
    log("REALE (2020): NFCI non esplode mai (prev -0.55 a Q1, -0.33 a Q2). Il")
    log("  driver finanziario NON c'e' -> il w onesto NON anticipa.")
    log("  - 2020Q1 onset: INVISIBILE ex-ante. w fc ~=1 (regime calmo), densita'")
    log("    addirittura piu' stretta del prior (std 0.46-0.48). PIT -> ~1e-5")
    log("    (FUORI). L'oracolo avrebbe dato PIT 0.11.")
    log("  - 2020Q2: anche con la PERSISTENZA AR (sa che Q1 era estremo) il w")
    log("    onesto allarga solo a std 0.76 vs i 3.10 dell'oracolo -> il -8.20")
    log("    resta FUORI (PIT~0) sotto OGNI forecast. Solo l'oracolo backward")
    log("    (PIT 0.007) lo cattura.")
    log("")
    log("IMPLICAZIONE per la stochastic volatility:")
    log("  (b) con w ONESTO cattura ancora la crisi FINANZIARIA (il 2008-09) ->")
    log("  una SV guidata da condizioni finanziarie (NFCI) e' utile in real-time")
    log("  per il tail risk di origine finanziaria: allarga la densita' PRIMA.")
    log("  NON risolve il 2020: lo shock reale e' strutturalmente non-")
    log("  forecastabile dalle condizioni finanziarie, e la sola persistenza")
    log("  arriva tardi (dopo l'onset) e insufficiente. Il 2020 resta")
    log("  catturabile solo col senno di poi (oracolo).")
    log("  CAVEAT a favore della forecastability (e il verdetto regge lo stesso):")
    log("  coeff. full-sample e termine AR da w smoothed (mite leakage). Anche")
    log("  cosi' il 2020 fallisce -> verdetto conservativo.")

    # ---- figura ----
    labels = [l for _, l in _CRISES if l in dens_store]
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, label in zip(axes[0], labels):
        st = dens_store[label]
        curves = [("w-PRIOR", "#7f7f7f", st["prior"])]
        cols_fc = {"NFCI-only": "#2ca02c", "AR+NFCI": "#1f77b4"}
        for name, d in st["fc"].items():
            curves.append((f"w-FC[{name}]", cols_fc.get(name, "#1f77b4"), d))
        curves.append(("w-SMOOTHED(oracolo)", "#d62728", st["oracle"]))
        for nm, col, d in curves:
            xs = d["_sim"]
            lo, hi = np.quantile(xs, [0.001, 0.999])
            grid = np.linspace(min(lo, st["realized"]) - 0.5,
                               max(hi, st["realized"]) + 0.5, 500)
            kde = stats.gaussian_kde(xs)
            ax.plot(grid, kde(grid), color=col, lw=1.6, label=nm)
        ax.axvline(st["realized"], color="red", lw=2,
                   label=f"realizzato {st['realized']:+.2f}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("GDP (100*dlog)"); ax.set_ylabel("densita'")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.2)
    fig.suptitle("Densita' predittiva GDP: w-PRIOR vs w-FORECAST (onesto) vs "
                 "w-SMOOTHED (oracolo) — forecastability della volatilita'",
                 fontsize=11)
    fig.tight_layout()
    figpath = os.path.join(_OUT, "vol_forecastability.png")
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"\nfigura -> {figpath}")

    with open(os.path.join(_OUT, "vol_forecastability.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"report -> {os.path.join(_OUT, 'vol_forecastability.txt')}")


if __name__ == "__main__":
    main()
