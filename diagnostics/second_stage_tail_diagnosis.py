"""
diagnostics/second_stage_tail_diagnosis.py

Diagnosi del risultato del Second-Stage prototype:
  "il fattore REALE domina la coda della distribuzione del GDP, il FINANZIARIO no"
  (contrario al Growth-at-Risk di Adrian-Boyarchenko-Giannone, dove le condizioni
   finanziarie - NFCI - guidano la coda sinistra).

Testa SEPARATAMENTE le ipotesi:
  A  strutturale: Lambda block-diagonal -> il fattore reale e' "costruito" per
     correlare col GDP (che sta nel blocco reale); il finanziario no.
  B  il FATTORE finanziario e' un cattivo proxy delle condizioni finanziarie:
     confronto potere-di-coda fattore-fin vs NFCI diretto (grezzo dal panel).
  C  finestra del prototipo (<=2008Q3) senza crolli: rifaccio su finestra che
     INCLUDE 2008/2020.
  D  reale + GLOBALE-proxy (PC1 di tutte le serie, o media dei 3 fattori):
     il globale porta il segnale finanziario nella coda meglio del fattore-fin?
     O e' solo "piu' reale"?
  SIM  recupero: dati sintetici dove la coda e' guidata PER COSTRUZIONE dal
     fattore finanziario -> il Second Stage la recupera?

Output: report testuale a sezioni in diagnostics/ + figure di confronto.

Run:
  python -m diagnostics.second_stage_tail_diagnosis
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

from src.data_loader import load_config
from src.forecast.panel_builder import build_panel

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from em_initialization import (  # noqa: E402
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from em_main import fit_dfm  # noqa: E402

_ROOT = os.path.dirname(_SRC_DIR)
_OUT = os.path.join(_ROOT, "diagnostics", "out")
os.makedirs(_OUT, exist_ok=True)

_R = 3
_MM = np.array([1/3, 2/3, 1.0, 2/3, 1/3])
_FNAMES = ["f_real", "f_fin", "f_nom"]
_TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
_REF = {"real": "PAYEMS", "financial": "S&P 500", "other": "UMCSENTx"}


# ───────────────────────── helpers ──────────────────────────────────────────

def quarterly_factors(f_smooth: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """MM-weighted quarterly factor per block at quarter-end months (= il bridge
    del first stage). col augmented = lag*3 + j ; blocchi real=0/fin=1/nom=2."""
    qe = index[index.month.isin([3, 6, 9, 12])]
    out = {}
    for j, name in enumerate(_FNAMES):
        idx_lags = np.array([lag * _R + j for lag in range(5)])
        out[name] = [float(f_smooth[index.get_loc(d), idx_lags] @ _MM) for d in qe]
    return pd.DataFrame(out, index=qe)


def fit_fullsample_factors(config="small", estimator="student_t", refit=False) -> dict:
    """Fit DFM una volta sull'intero campione rivisto (dataset_<config>.csv).
    Restituisce factor MM-trimestrali, GDP, NFCI, e il panel standardizzato."""
    cache = os.path.join(_OUT, f"fullsample_{config}_{estimator}.npz")
    cfg = load_config(config)
    freq, block, cols = cfg["FREQ"], cfg["BLOCK"], cfg["ORDERED_COLS"]
    panel = pd.read_csv(os.path.join(_ROOT, "data", "processed", f"dataset_{config}.csv"),
                        index_col=0)
    panel.index = pd.to_datetime(panel.index)

    Y_std, mean, std = standardize(panel)

    if os.path.exists(cache) and not refit:
        z = np.load(cache, allow_pickle=True)
        f_smooth = z["f_smooth"]
        print(f"  [cache] fullsample factors {os.path.basename(cache)}")
    else:
        Y_mm = Y_std.copy()
        for c in Y_std.columns:
            if freq.get(c) == "quarterly":
                Y_mm[c] = mm_fill_quarterly(Y_std[c])
        Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
        F0, _ = pca_initialization(Y_filled, block)
        theta0 = compute_theta_initial(Y_filled, F0, block)
        print(f"  fitting fullsample DFM ({estimator},{config}) T={len(Y_std)} ... (slow)")
        fit = fit_dfm(Y=Y_std.to_numpy(), theta_init=theta0,
                      freq_list=[freq[c] for c in cols], block_map=block,
                      ordered_cols=cols, ref_series=_REF,
                      gaussian=(estimator == "gaussian"), use_full_elbo=True,
                      max_iter=250, verbose=False, save_path=None)
        f_smooth = np.asarray(fit["f_smooth"])
        np.savez(cache, f_smooth=f_smooth, n_iter=fit["n_iter"], converged=fit["converged"])
        print(f"  fit done n_iter={fit['n_iter']} converged={fit['converged']}")

    qf = quarterly_factors(f_smooth, Y_std.index)
    gdp = panel["GDPC1"].dropna()
    nfci = panel["NFCI"]
    # global proxy = PC1 di TUTTE le serie standardizzate (riempite); influenzato
    # da tutte le variabili incluse le finanziarie.
    Y_mm = Y_std.copy()
    for c in Y_std.columns:
        if freq.get(c) == "quarterly":
            Y_mm[c] = mm_fill_quarterly(Y_std[c])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    Xall = Y_filled.to_numpy()
    Xall = (Xall - Xall.mean(0)) / Xall.std(0)
    u, s, vt = np.linalg.svd(Xall, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    # orienta PC1 in modo che correli + col blocco reale (PAYEMS)
    if np.corrcoef(pc1, Y_filled["PAYEMS"].to_numpy())[0, 1] < 0:
        pc1 = -pc1
    pc1_s = pd.Series(pc1, index=Y_std.index)
    return {"qf": qf, "gdp": gdp, "nfci": nfci, "pc1": pc1_s,
            "Y_std": Y_std, "block": block, "cols": cols}


def qreg_table(train: pd.DataFrame, regressors: list[str], taus=_TAUS) -> pd.DataFrame:
    """QuantReg di 'gdp' su regressors per ogni tau. Coefficienti + pseudo-R1."""
    X = sm.add_constant(train[regressors])
    y = train["gdp"].to_numpy()
    rows = []
    for tau in taus:
        res = QuantReg(train["gdp"], X).fit(q=tau, max_iter=5000)
        # pseudo-R1 (Koenker-Machado) vs intercetta-sola (quantile empirico)
        resid = y - res.predict(X)
        V_full = np.sum(resid * (tau - (resid < 0)))
        q_emp = np.quantile(y, tau)
        r0 = y - q_emp
        V_rest = np.sum(r0 * (tau - (r0 < 0)))
        r1 = 1.0 - V_full / V_rest if V_rest > 0 else np.nan
        row = {"tau": tau, "b0": res.params["const"], "R1": r1}
        for rg in regressors:
            row[f"b_{rg}"] = res.params[rg]
        rows.append(row)
    return pd.DataFrame(rows).set_index("tau")


def tail_summary(tab: pd.DataFrame, key: str) -> dict:
    """Sintetizza il comportamento di coda del regressore `key`."""
    return {
        "b05": tab.loc[0.05, f"b_{key}"],
        "b10": tab.loc[0.10, f"b_{key}"],
        "b50": tab.loc[0.50, f"b_{key}"],
        "steep": tab.loc[0.05, f"b_{key}"] - tab.loc[0.50, f"b_{key}"],
        "R1_05": tab.loc[0.05, "R1"],
        "R1_10": tab.loc[0.10, "R1"],
    }


def hr(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


# ───────────────────────── MAIN ─────────────────────────────────────────────

def main():
    report = []

    def log(s=""):
        print(s)
        report.append(s)

    # ============ 0. riproduzione prototipo: cache real-time big (as_of 2008-11-15)
    hr("0. RIPRODUZIONE PROTOTIPO  (cache real-time big student_t, as_of 2008-11-15)")
    cache = os.path.join(_ROOT, "_archive_second_stage", "second_stage_cache",
                         "factors_big_student_t_20081115.npz")
    z = np.load(cache, allow_pickle=True)
    idx_rt = pd.DatetimeIndex(z["index"])
    qf_rt = quarterly_factors(z["f_smooth"], idx_rt)
    gdp_rt = pd.Series(z["gdp_natural"], index=pd.DatetimeIndex(z["gdp_index"]))
    last_gdp = gdp_rt.index[-1]  # 2008Q3
    tr_rt = qf_rt.loc[(qf_rt.index <= last_gdp) & qf_rt.index.isin(gdp_rt.index)].copy()
    tr_rt["gdp"] = gdp_rt.loc[tr_rt.index].to_numpy()
    tr_rt = tr_rt.dropna()
    log(f"n training = {len(tr_rt)}  ({tr_rt.index[0].date()} .. {tr_rt.index[-1].date()})")
    log(f"GDP range training: [{tr_rt['gdp'].min():+.2f}, {tr_rt['gdp'].max():+.2f}]")
    tab_rt = qreg_table(tr_rt, _FNAMES)
    log("\nQuantReg coefficienti (factor reale/fin/nom), cache real-time:")
    log(tab_rt.to_string(float_format=lambda v: f"{v:+.4f}"))
    log("\n-> conferma prototipo: a tau=0.05 il dominante e' "
        f"{max(_FNAMES, key=lambda n: abs(tab_rt.loc[0.05, 'b_'+n]))}")

    # ============ full-sample fit (small, clean) per A/B/C/D/SIM
    hr("FIT FULL-SAMPLE  (config small, 1985-2026) - base per A/B/C/D/SIM")
    fs = fit_fullsample_factors(config="small", estimator="student_t")
    qf, gdp, nfci, pc1 = fs["qf"], fs["gdp"], fs["nfci"], fs["pc1"]
    # tabella unificata ai quarter-end con GDP realizzato
    qe = qf.index[qf.index.isin(gdp.index)]
    D = qf.loc[qe].copy()
    D["gdp"] = gdp.loc[qe].to_numpy()
    D["nfci"] = nfci.reindex(qe).to_numpy()
    D["pc1"] = pc1.reindex(qe).to_numpy()
    D = D.dropna()
    log(f"campione completo quarter-ends con GDP: n={len(D)}  "
        f"({D.index[0].date()} .. {D.index[-1].date()})")
    log(f"GDP range full: [{D['gdp'].min():+.2f}, {D['gdp'].max():+.2f}]  "
        f"(min = {D['gdp'].idxmin().date()})")

    # finestre
    pre08 = D[D.index <= pd.Timestamp("2008-09-30")]    # finestra prototipo
    full = D                                            # include 2008 + 2020
    log(f"\nfinestra PROTOTIPO (<=2008Q3): n={len(pre08)}  "
        f"GDP range [{pre08['gdp'].min():+.2f}, {pre08['gdp'].max():+.2f}]")
    log(f"finestra CON CROLLI (full):    n={len(full)}  "
        f"GDP range [{full['gdp'].min():+.2f}, {full['gdp'].max():+.2f}]")

    # ============ IPOTESI A — correlazioni fattori-GDP (block diagonal)
    hr("IPOTESI A - strutturale: quanto correlano i 3 fattori col GDP?")
    for label, W in [("finestra prototipo <=2008Q3", pre08), ("full con crolli", full)]:
        log(f"\n[{label}]  corr(fattore, GDP):")
        for n in _FNAMES:
            c = np.corrcoef(W[n], W["gdp"])[0, 1]
            log(f"   {n:8s} : {c:+.3f}")
        log(f"   corr(NFCI, GDP)   : {np.corrcoef(W['nfci'], W['gdp'])[0,1]:+.3f}")
        log(f"   corr(pc1,  GDP)   : {np.corrcoef(W['pc1'],  W['gdp'])[0,1]:+.3f}")
    log("\nLettura A: con Lambda block-diagonal il GDP carica SOLO sul blocco reale,")
    log("quindi f_real e' estratto per co-muoversi col GDP; f_fin (da sole serie")
    log("finanziarie) non 'vede' il GDP in estrazione -> piu' debolmente correlato.")

    # ============ IPOTESI B — fattore-fin vs NFCI diretto, potere di coda
    hr("IPOTESI B - fattore finanziario vs NFCI DIRETTO sulla coda")
    log(f"\ncorr(f_fin, NFCI) full = {np.corrcoef(full['f_fin'], full['nfci'])[0,1]:+.3f}  "
        f"| pre08 = {np.corrcoef(pre08['f_fin'], pre08['nfci'])[0,1]:+.3f}")
    log("(f_fin e' normalizzato col segno di S&P 500: f_fin alto = mercati forti =")
    log(" condizioni BUONE; NFCI alto = condizioni TESE -> attesa corr negativa)")

    for label, W in [("PROTOTIPO <=2008Q3", pre08), ("FULL con crolli", full)]:
        log(f"\n--- [{label}] ---")
        t_real = qreg_table(W, ["f_real"])
        t_fin = qreg_table(W, ["f_real", "f_fin"])
        t_nfci = qreg_table(W, ["f_real", "nfci"])
        log("real-only        R1(.05)={:.3f} R1(.10)={:.3f}".format(
            t_real.loc[0.05, "R1"], t_real.loc[0.10, "R1"]))
        s_fin = tail_summary(t_fin, "f_fin")
        s_nfci = tail_summary(t_nfci, "nfci")
        log("real+f_fin   : b_fin(.05)={b05:+.3f} b_fin(.50)={b50:+.3f} "
            "steep={steep:+.3f} | R1(.05)={R1_05:.3f} R1(.10)={R1_10:.3f}".format(**s_fin))
        log("real+NFCI    : b_nfci(.05)={b05:+.3f} b_nfci(.50)={b50:+.3f} "
            "steep={steep:+.3f} | R1(.05)={R1_05:.3f} R1(.10)={R1_10:.3f}".format(**s_nfci))
        # NFCI da sola sulla coda
        t_nfci_only = qreg_table(W, ["nfci"])
        log("NFCI-only    : b_nfci(.05)={:+.3f} b_nfci(.50)={:+.3f} R1(.05)={:.3f}".format(
            t_nfci_only.loc[0.05, "b_nfci"], t_nfci_only.loc[0.50, "b_nfci"],
            t_nfci_only.loc[0.05, "R1"]))

    # ============ IPOTESI C — finestra con crolli
    hr("IPOTESI C - con crolli nel campione il finanziario emerge nella coda?")
    log("Confronto coefficiente di coda di f_fin: prototipo vs full.")
    cp = qreg_table(pre08, _FNAMES)
    cf = qreg_table(full, _FNAMES)
    log("\nf_fin coeff:")
    log(f"   prototipo  b_fin(.05)={cp.loc[0.05,'b_f_fin']:+.3f}  b_fin(.50)={cp.loc[0.50,'b_f_fin']:+.3f}")
    log(f"   full       b_fin(.05)={cf.loc[0.05,'b_f_fin']:+.3f}  b_fin(.50)={cf.loc[0.50,'b_f_fin']:+.3f}")
    log("\nf_real coeff:")
    log(f"   prototipo  b_real(.05)={cp.loc[0.05,'b_f_real']:+.3f}  b_real(.50)={cp.loc[0.50,'b_f_real']:+.3f}")
    log(f"   full       b_real(.05)={cf.loc[0.05,'b_f_real']:+.3f}  b_real(.50)={cf.loc[0.50,'b_f_real']:+.3f}")
    log("\ncoeff completi full (con crolli):")
    log(cf.to_string(float_format=lambda v: f"{v:+.4f}"))
    dom_full = max(_FNAMES, key=lambda n: abs(cf.loc[0.05, 'b_'+n]))
    log(f"-> a tau=0.05 (full) il dominante e': {dom_full}")

    # ============ IPOTESI D — reale + globale-proxy
    hr("IPOTESI D - reale + GLOBALE-proxy (PC1 di tutte le serie)")
    log(f"\ncorr(pc1, f_real) full = {np.corrcoef(full['pc1'], full['f_real'])[0,1]:+.3f}")
    log(f"corr(pc1, NFCI)   full = {np.corrcoef(full['pc1'], full['nfci'])[0,1]:+.3f}")
    log(f"corr(pc1, f_fin)  full = {np.corrcoef(full['pc1'], full['f_fin'])[0,1]:+.3f}")
    log("(se pc1 correla molto col reale e poco con NFCI -> e' 'piu' reale', non sente il fin)")

    log("\nConfronto TRE setup sulla coda (full, con crolli) - guadagno R1 vs real-only:")
    base = qreg_table(full, ["f_real"])
    setups = {
        "(i)  real+f_fin": ["f_real", "f_fin"],
        "(ii) real+pc1  ": ["f_real", "pc1"],
        "(iii)real+NFCI ": ["f_real", "nfci"],
    }
    log(f"   real-only baseline  R1(.05)={base.loc[0.05,'R1']:.3f} R1(.10)={base.loc[0.10,'R1']:.3f}")
    for name, regs in setups.items():
        t = qreg_table(full, regs)
        sec = regs[1]
        d05 = t.loc[0.05, "R1"] - base.loc[0.05, "R1"]
        d10 = t.loc[0.10, "R1"] - base.loc[0.10, "R1"]
        log(f"   {name}: R1(.05)={t.loc[0.05,'R1']:.3f} (dR1={d05:+.3f}) "
            f"R1(.10)={t.loc[0.10,'R1']:.3f} (dR1={d10:+.3f}) "
            f"b_{sec}(.05)={t.loc[0.05,'b_'+sec]:+.3f}")
    # ripeti su prototipo
    log("\nStesso confronto sulla finestra PROTOTIPO (<=2008Q3):")
    base_p = qreg_table(pre08, ["f_real"])
    log(f"   real-only baseline  R1(.05)={base_p.loc[0.05,'R1']:.3f} R1(.10)={base_p.loc[0.10,'R1']:.3f}")
    for name, regs in setups.items():
        t = qreg_table(pre08, regs)
        sec = regs[1]
        d05 = t.loc[0.05, "R1"] - base_p.loc[0.05, "R1"]
        log(f"   {name}: R1(.05)={t.loc[0.05,'R1']:.3f} (dR1={d05:+.3f}) "
            f"b_{sec}(.05)={t.loc[0.05,'b_'+sec]:+.3f}")

    # ============ SIMULAZIONE — recupero del segnale finanziario di coda
    hr("SIMULAZIONE - dati sintetici con coda guidata PER COSTRUZIONE dal fin")
    rng = np.random.default_rng(0)
    # uso i fattori reali stimati (full) come regressori realistici, standardizzati
    F = full[_FNAMES].copy()
    Fz = (F - F.mean()) / F.std()
    n = len(Fz)

    def run_sim(gdp_synth, label):
        S = Fz.copy()
        S["gdp"] = gdp_synth
        t = qreg_table(S, _FNAMES)
        b_real = qreg_table(S, ["f_real"])
        t_rf = qreg_table(S, ["f_real", "f_fin"])
        dR1 = t_rf.loc[0.05, "R1"] - b_real.loc[0.05, "R1"]
        log(f"\n[{label}]")
        log("  coeff per tau:")
        log("  " + t[["b_f_real", "b_f_fin", "b_f_nom", "R1"]].to_string(
            float_format=lambda v: f"{v:+.3f}").replace("\n", "\n  "))
        log(f"  f_fin: b(.05)={t.loc[0.05,'b_f_fin']:+.3f} b(.50)={t.loc[0.50,'b_f_fin']:+.3f} "
            f"steep={t.loc[0.05,'b_f_fin']-t.loc[0.50,'b_f_fin']:+.3f}")
        log(f"  dR1(.05) aggiungendo f_fin a real-only = {dR1:+.3f}")
        return t

    # (1) coda guidata dal FINANZIARIO: scala cresce quando f_fin scende (condizioni tese)
    sigma = np.exp(0.0 - 0.55 * Fz["f_fin"].to_numpy())   # f_fin basso -> sigma alta -> coda sx
    eps = rng.standard_normal(n)
    gdp_fin = 0.0 + 0.6 * Fz["f_real"].to_numpy() + sigma * eps
    t_fin = run_sim(gdp_fin, "SIM-1 coda guidata dal FINANZIARIO (per costruzione)")
    rec_fin = (t_fin.loc[0.05, "b_f_fin"] - t_fin.loc[0.50, "b_f_fin"])

    # (2) placebo: coda guidata dal REALE
    sigma_r = np.exp(0.0 - 0.55 * Fz["f_real"].to_numpy())
    gdp_real = 0.0 + 0.6 * Fz["f_fin"].to_numpy() + sigma_r * rng.standard_normal(n)
    t_realtail = run_sim(gdp_real, "SIM-2 placebo: coda guidata dal REALE")

    log("\n--- esito simulazione ---")
    log(f"SIM-1 (fin guida): steepening f_fin = {rec_fin:+.3f} ; "
        f"il metodo {'RECUPERA' if abs(rec_fin) > 0.15 else 'NON recupera'} il segnale finanziario di coda")
    log(f"SIM-2 (real guida): steepening f_real = "
        f"{t_realtail.loc[0.05,'b_f_real']-t_realtail.loc[0.50,'b_f_real']:+.3f} ; "
        f"steepening f_fin = {t_realtail.loc[0.05,'b_f_fin']-t_realtail.loc[0.50,'b_f_fin']:+.3f}")

    # ============ FIGURA riassuntiva: coeff di coda nei tre setup (full)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    taus = np.array(_TAUS)
    # pannello 1: coeff del secondo regressore (standardizzato per confronto scala)
    ax = axes[0]
    for name, regs, col in [("real+f_fin", ["f_real", "f_fin"], "#1f77b4"),
                            ("real+pc1", ["f_real", "pc1"], "#2ca02c"),
                            ("real+NFCI", ["f_real", "nfci"], "#d62728")]:
        W = full.copy()
        sec = regs[1]
        W[sec + "_z"] = (W[sec] - W[sec].mean()) / W[sec].std()
        t = qreg_table(W, ["f_real", sec + "_z"])
        ax.plot(taus, t[f"b_{sec}_z"], "o-", color=col, label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Coeff. del 2o regressore (z-score) per tau\n(full, con crolli)")
    ax.set_xlabel("tau"); ax.set_ylabel("coeff (per 1 sd)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    # pannello 2: R1 per tau
    ax = axes[1]
    ax.plot(taus, qreg_table(full, ["f_real"])["R1"], "k--", label="real-only")
    for name, regs, col in [("real+f_fin", ["f_real", "f_fin"], "#1f77b4"),
                            ("real+pc1", ["f_real", "pc1"], "#2ca02c"),
                            ("real+NFCI", ["f_real", "nfci"], "#d62728")]:
        ax.plot(taus, qreg_table(full, regs)["R1"], "o-", color=col, label=name)
    ax.set_title("Pseudo-R1 (Koenker-Machado) per tau\n(full, con crolli)")
    ax.set_xlabel("tau"); ax.set_ylabel("R1"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    figpath = os.path.join(_OUT, "tail_diagnosis_setups.png")
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"\nfigura -> {figpath}")

    with open(os.path.join(_OUT, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"\nreport -> {os.path.join(_OUT, 'report.txt')}")


if __name__ == "__main__":
    main()
