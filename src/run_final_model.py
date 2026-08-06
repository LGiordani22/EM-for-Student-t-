"""
src/run_final_model.py

Runner del DFM (First Stage) sul dataset FINAL (37 serie NY Fed) con una delle
spec di config/factor_specs.json (fed_overlap / diag4 / diag3), Lambda
mask-driven.

Il modello gira solo sulle tre strutture di fattore sopra
(fed_overlap / diag4 / diag3).

Per ogni run: costruisce standardizzazione + init PCA mask-driven + theta^(0),
poi fit_dfm (gaussiano o Student-t) e stampa una diagnostica del GATE Asse A
(convergenza, ELBO monotono, varianze totali dei fattori ~1, leak G⊥locali,
segni, nu per lo Student-t).

Uso:
    python src/run_final_model.py --spec fed_overlap --estimator gaussian
    python src/run_final_model.py --spec diag4 --estimator student_t
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from em.em_initialization import (  # noqa: E402
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from em.factor_structure import build_loading_mask  # noqa: E402
from em.em_main import fit_dfm  # noqa: E402


def _prep_final(spec: str):
    df = pd.read_csv(os.path.join(_ROOT, "data", "processed", "final", "dataset_final.csv"),
                     index_col=0, parse_dates=True)
    cfg = json.load(open(os.path.join(_ROOT, "config", "series_final.json"), encoding="utf-8"))
    freq_of = {s["series_id"]: ("monthly" if s["freq"] == "M" else "quarterly")
               for s in cfg["series"]}
    cols = list(df.columns)
    freq_list = [freq_of[c] for c in cols]
    structure = build_loading_mask(spec, cols)
    return df, cols, freq_list, structure


def run(spec: str, estimator: str, max_iter: int, verbose: bool):
    df, cols, freq_list, structure = _prep_final(spec)
    label = f"final:{spec}"

    # ── standardizzazione + init PCA mask-driven ──────────────────────────────
    Y_std_df, mean, std = standardize(df)
    Y_mm = Y_std_df.copy()
    for c, fr in zip(cols, freq_list):
        if fr == "quarterly":
            Y_mm[c] = mm_fill_quarterly(Y_std_df[c])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    F0, _info = pca_initialization(Y_filled, structure)
    theta0 = compute_theta_initial(Y_filled, F0, structure)
    r = theta0["A"].shape[0]

    Y = Y_std_df.to_numpy()

    print(f"\n{'='*70}\n  RUN  {label}  |  estimator={estimator}  |  r={r}  |  "
          f"T×M = {Y.shape[0]}×{Y.shape[1]}\n{'='*70}")

    out = fit_dfm(
        Y, theta0,
        freq_list=freq_list,
        block_map=structure,
        ordered_cols=cols,
        gaussian=(estimator == "gaussian"),
        max_iter=max_iter,
        verbose=verbose,
    )

    # ── diagnostica GATE ──────────────────────────────────────────────────────
    ll = np.asarray(out["loglik_history"], dtype=float)
    dll = np.diff(ll)
    f = out["f_smooth"][:, :r]
    fvar = f.var(axis=0)                      # varianza campionaria (post convention_1)
    print(f"\n  --- DIAGNOSTICA {label} / {estimator} ---")
    print(f"  converged           : {out['converged']}  (n_iter={out['n_iter']})")
    print(f"  loglik  primo→ultimo: {ll[0]:.2f} → {ll[-1]:.2f}")
    print(f"  ELBO monotono       : {(dll >= -1e-6).all()}  "
          f"(min Δ = {dll.min():.2e}, violazioni={out['monotonicity_violations']})")
    print(f"  var(f) [~1 attesa]  : {np.array2string(fvar, precision=3)}")
    if out.get("leak_coef"):
        print(f"  leak G⊥locali (c_l) : {{{', '.join(f'{k}:{v:+.3f}' for k,v in out['leak_coef'].items())}}}")
    print(f"  sign_flips          : {out['sign_flips']}")
    th = out["theta"]
    if estimator == "student_t":
        print(f"  nu_u, nu_eps        : {float(th['nu_u']):.2f}, {float(th['nu_eps']):.2f}")
    # Fit della serie TARGET (per contratto l'ultima colonna del config):
    # riga target di Lambda x fattori, de-standardizzata. Il nome non e'
    # cablato — viene dall'ordine delle colonne.
    tgt = cols[-1]
    ti = len(cols) - 1
    tgt_fit_std = f @ th["Lambda"][ti, :]
    tgt_fit = tgt_fit_std * std[tgt] + mean[tgt]
    print(f"  fit '{tgt}' (log-diff annualizz., ultimo mese): "
          f"{np.array2string(tgt_fit[-1:], precision=2)} (mensile; ~stato)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["fed_overlap", "diag4", "diag3"],
                    default="fed_overlap")
    ap.add_argument("--estimator", choices=["gaussian", "student_t"], default="gaussian")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(a.spec, a.estimator, a.max_iter, verbose=not a.quiet)


if __name__ == "__main__":
    main()
