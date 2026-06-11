"""
src/forecast/extract_weps.py

Diagnostic helper (NOT part of the rolling pipeline): re-fit the Student-t DFM
for ONE real-time vintage exactly as nowcast_engine.nowcast_gdp does, then pull
the per-month measurement-equation mixing weights w_eps_t out of the fit's
e_step_output and align them to the panel month index.

A single fit at a late-crisis as_of (e.g. 2009-01-15) covers every month up to
that vintage, so its w_eps trajectory documents the down-weighting of the crisis
months (2008-09..2008-12) without re-running anything.  w_eps_t < 1 means month
t was treated as a (common) outlier and down-weighted; w_eps_t ~ 1 is a normal
month under the Student-t scale mixture.

Run
---
  python -m src.forecast.extract_weps --as-of 2009-01-15 --target 2008Q4 \
      --window 2008-01:2009-12
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data_loader import BLOCK, FREQ, ORDERED_COLS, load_config
from src.config_utils import parse_config_args
from src.forecast.panel_builder import build_panel
from src.forecast.nowcast_engine import _quarter_end, _REF_SERIES
from em_initialization import (
    standardize, mm_fill_quarterly, gaussian_fill_ragged,
    pca_initialization, compute_theta_initial,
)
from em_main import fit_dfm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_weps(as_of, target_quarter: str, config_name: str = "small",
                 max_iter: int = 250) -> pd.DataFrame:
    """Re-fit the Student-t DFM at `as_of` and return a DataFrame indexed by
    panel month with columns w_eps, w_u (the E-step mixing weights)."""
    cfg = load_config(config_name)
    _block = cfg["BLOCK"]
    _freq = cfg["FREQ"]
    _ordered_cols = cfg["ORDERED_COLS"]

    target_qe = _quarter_end(target_quarter)
    panel = build_panel(as_of, config_name=config_name)
    if target_qe > panel.index[-1]:
        panel = panel.reindex(pd.date_range(panel.index[0], target_qe, freq="ME"))
    panel.index.name = None

    Y_std, mean, std = standardize(panel)
    Y_mm = Y_std.copy()
    for col in Y_std.columns:
        if _freq.get(col) == "quarterly":
            Y_mm[col] = mm_fill_quarterly(Y_std[col])
    Y_filled = gaussian_fill_ragged(Y_mm, random_state=42)
    F0, _info = pca_initialization(Y_filled, _block)
    theta0 = compute_theta_initial(Y_filled, F0, _block)

    freq_list = [_freq[c] for c in _ordered_cols]
    fit = fit_dfm(
        Y=Y_std.to_numpy(), theta_init=theta0, freq_list=freq_list,
        block_map=_block, ordered_cols=_ordered_cols, ref_series=_REF_SERIES,
        gaussian=False, use_full_elbo=True, max_iter=max_iter,
        verbose=False, save_path=None,
    )
    eso = fit["e_step_output"]
    w_eps = np.asarray(eso["w_eps"]).ravel()
    w_u = np.asarray(eso["w_u"]).ravel()
    idx = Y_std.index[: len(w_eps)]
    out = pd.DataFrame({"w_eps": w_eps, "w_u": w_u}, index=idx)
    out.index.name = "month"
    out.attrs["nu_eps"] = float(fit["theta"]["nu_eps"])
    out.attrs["nu_u"] = float(fit["theta"]["nu_u"])
    out.attrs["n_iter"] = int(fit["n_iter"])
    out.attrs["converged"] = bool(fit["converged"])
    return out


def main() -> None:
    def _extra(p: argparse.ArgumentParser) -> None:
        p.add_argument("--as-of", required=True, help="vintage date 'YYYY-MM-DD'")
        p.add_argument("--target", required=True, help="target quarter 'YYYYQn'")
        p.add_argument("--window", default=None,
                       help="optional 'YYYY-MM:YYYY-MM' month range to print/save")
        p.add_argument("--save", default=None, help="optional CSV output path")

    args = parse_config_args("Extract per-month w_eps/w_u from a single DFM fit.", extra=_extra)

    df = extract_weps(pd.Timestamp(args.as_of), args.target, config_name=args.config)
    print(f"\nfit: nu_eps={df.attrs['nu_eps']:.3f}  nu_u={df.attrs['nu_u']:.3f}  "
          f"n_iter={df.attrs['n_iter']}  converged={df.attrs['converged']}")
    print(f"w_eps: median={df['w_eps'].median():.3f}  min={df['w_eps'].min():.3f} "
          f"at {df['w_eps'].idxmin().date()}")

    view = df
    if args.window:
        lo, hi = args.window.split(":")
        view = df.loc[lo:hi]
    print("\nPer-month weights" + (f" ({args.window})" if args.window else "") + ":")
    with pd.option_context("display.max_rows", None):
        print(view.to_string(float_format=lambda x: f"{x:.4f}"))

    save = args.save
    if save is None and args.window:
        out_dir = os.path.join(_PROJECT_ROOT, "output", "forecast_realtime", "csv", args.config)
        save = os.path.join(out_dir, f"weps_{args.as_of}_{args.target}.csv")
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        view.to_csv(save)
        print(f"\nsaved -> {save}")


if __name__ == "__main__":
    main()
