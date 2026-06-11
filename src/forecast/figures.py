"""
src/forecast/figures.py

NOWCAST-TRAJECTORY FIGURES for the nowcasting pipeline.

Pure plotter: reads the long CSV produced by rolling_nowcast.py.

TWO FIGURE TYPES
----------------
1. --style cg   (default)
   2×2 grid, one panel per method (Student-t / Gaussian / ARMA(2,2) / Random Walk).
   Each panel is a Cascaldi-Garcia continuous timeline: one coloured trajectory
   per target quarter, filled-circle release dots, short "YYQn" labels.
   Produces two files per run: level and z-score.

2. --style compare  (--target YYYYQn required)
   Single panel, all 4 methods overlaid for one target quarter.
   For zooming in on a specific episode (e.g. 2008Q4).

Config-aware: --small / --big.

Run
---
  python -m src.forecast.figures                         # CG 4-panel, newest CSV
  python -m src.forecast.figures --csv path/to/file.csv
  python -m src.forecast.figures --style compare --target 2008Q4
"""

from __future__ import annotations

import argparse
import glob
import os

from src.config_utils import parse_config_args

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ─── Cascaldi-Garcia quarter palette ─────────────────────────────────────────
_CG_COLORS = [
    "#1f77b4",  # blu
    "#ff7f0e",  # arancione
    "#d62728",  # rosso
    "#9467bd",  # viola
    "#2ca02c",  # verde
    "#8c564b",  # marrone
    "#e377c2",  # rosa
    "#17becf",  # azzurro
    "#bcbd22",  # ocra
    "#7f7f7f",  # grigio
]

# ─── Per-method styles and display names ─────────────────────────────────────
_METHOD_STYLE: dict[str, dict] = {
    "student_t":   dict(color="#1f77b4", label="Student-t DFM"),
    "gaussian":    dict(color="#ff7f0e", label="Gaussian DFM"),
    "arma22":      dict(color="#2ca02c", label="ARMA(2,2)"),
    "random_walk": dict(color="#9467bd", label="Random Walk"),
}
_METHOD_DISPLAY = {
    "student_t":   "Student-t DFM",
    "gaussian":    "Gaussian DFM",
    "arma22":      "ARMA(2,2)",
    "random_walk": "Random Walk",
}
_METHOD_ORDER = ["student_t", "gaussian", "arma22", "random_walk"]
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_RELEASE_OFFSET_DAYS = 20   # advance GDP release ~2 weeks after last as_of
_MIN_TRAJ_POINTS = 2        # suppress edge quarters with < 2 nowcast vintages

# ─── Figure styles ────────────────────────────────────────────────────────────
# Each entry maps a subfolder name to the rendering parameters.
# "zoomate"  = auto y-scale, all 4 horizons (original style)
# "figure_-10_5" = fixed ylim (-10, 5), only h2-h4 per quarter (CG clean style)
_FIGURE_STYLES: dict[str, dict] = {
    "figure_zoomate": dict(min_horizon=1, ylim_fixed=None),
    "figure_-10_5":   dict(min_horizon=2, ylim_fixed=(-10.0, 5.0)),
}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _newest_csv(config_name: str = "small") -> str:
    csv_dir = os.path.join(_PROJECT_ROOT, "output", "forecast_realtime", "csv", config_name)
    cands = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not cands:
        raise FileNotFoundError(
            f"No rolling-nowcast CSV found in {csv_dir}.\n"
            f"Run: python -m src.forecast.rolling_nowcast --{config_name} "
            f"--start YYYY-MM --end YYYY-MM"
        )
    return max(cands, key=os.path.getmtime)


def _resolve_csv(csv_arg: str | None, config: str) -> str:
    if csv_arg:
        return csv_arg
    return _newest_csv(config)


def _quarter_key(label: str) -> tuple[int, int]:
    y, q = label.upper().split("Q")
    return int(y), int(q)


def _period_str(df: pd.DataFrame) -> str:
    months = pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m")
    return f"{months.min()}_{months.max()}"


def _methods_present(df: pd.DataFrame) -> list[str]:
    present = [m for m in _METHOD_ORDER if m in set(df["method"])]
    extra = [m for m in df["method"].unique() if m not in _METHOD_ORDER]
    return present + sorted(extra)


def _style_for(method: str, fallback_idx: int = 0) -> dict:
    if method in _METHOD_STYLE:
        return _METHOD_STYLE[method]
    color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
    return dict(color=color, label=method)


def _method_display_name(method: str) -> str:
    return _METHOD_DISPLAY.get(method, method)


def _quarter_short_label(q: str) -> str:
    """'2008Q3' -> '08Q3'"""
    year, num = q.upper().split("Q")
    return year[2:] + "Q" + num


def _apply_clean_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(True, alpha=0.18, linewidth=0.6, color="gray", which="major")
    ax.set_facecolor("white")


# ─── Z-score recovery ─────────────────────────────────────────────────────────

def _recover_realised_z(sub: pd.DataFrame, realised_level: float) -> float:
    """
    Fit  level = std * z + mean  across available (level, z) pairs and return
    standardised realised_level.  Returns NaN if not identifiable.
    """
    z = sub["nowcast_z"].to_numpy(dtype=float)
    lv = sub["nowcast_livello"].to_numpy(dtype=float)
    ok = np.isfinite(z) & np.isfinite(lv)
    z, lv = z[ok], lv[ok]
    if z.size < 2 or np.unique(np.round(z, 9)).size < 2:
        return float("nan")
    std, mean = np.polyfit(z, lv, 1)
    if std == 0:
        return float("nan")
    return (realised_level - mean) / std


# ─── Single CG panel (one method, shared colour-map) ─────────────────────────

def _draw_cg_panel(
    ax: plt.Axes,
    df_method: pd.DataFrame,       # already filtered to one method, as_of_dt added
    quarters: list[str],
    color_map: dict[str, str],
    value_col: str,
    method_name: str,
    ylabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
    min_horizon: int = 1,
    ylim_fixed: tuple[float, float] | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Draw one CG-style panel.  Returns (xmin, xmax) across plotted data for
    the caller to set a consistent shared xlim.

    min_horizon: skip horizons below this value (1 = all, 2 = skip h1, …)
    ylim_fixed:  if set, apply this fixed (ymin, ymax) instead of auto-scaling
    """
    all_asof: list[pd.Timestamp] = []
    all_release: list[pd.Timestamp] = []

    for q in quarters:
        qrows = df_method[df_method["target_quarter"] == q].sort_values("as_of_dt")
        if min_horizon > 1:
            qrows = qrows[qrows["horizon_month"] >= min_horizon]
        if len(qrows) < _MIN_TRAJ_POINTS:
            continue

        color = color_map[q]
        all_asof.extend(qrows["as_of_dt"].tolist())

        ax.plot(
            qrows["as_of_dt"], qrows[value_col],
            color=color, linewidth=1.8,
            solid_capstyle="round", solid_joinstyle="round",
        )

        last_asof = qrows["as_of_dt"].max()
        release_date = last_asof + pd.Timedelta(days=_RELEASE_OFFSET_DAYS)
        all_release.append(release_date)

        realised_level = float(qrows["gdp_realizzato"].iloc[0])
        if value_col == "nowcast_livello":
            release_y = realised_level
        else:
            # Use all horizons (h1-h4) for z-recovery: methods like ARMA/RW have
            # constant z across h2-h4 but a distinct value at h1, so we need h1
            # to get a second unique point for the linear fit.
            qrows_all_h = df_method[df_method["target_quarter"] == q]
            release_y = _recover_realised_z(qrows_all_h, realised_level)

        if np.isfinite(release_y):
            ax.scatter([release_date], [release_y],
                       color=color, s=45, zorder=5, linewidths=0,
                       clip_on=False)
            ax.annotate(
                _quarter_short_label(q), (release_date, release_y),
                textcoords="offset points", xytext=(4, 0),
                ha="left", va="center",
                fontsize=6.5, color=color, fontweight="bold",
                annotation_clip=False,
            )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    if ylim_fixed is not None:
        ax.set_ylim(*ylim_fixed)
    _apply_clean_style(ax)

    ax.set_title(_method_display_name(method_name), fontsize=10, fontweight="bold", pad=6)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.yaxis.set_tick_params(labelsize=8)

    # x-axis tick formatting — only show labels on bottom row
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    if show_xlabel:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_tick_params(labelsize=7.5)
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    xmin = min(all_asof) - pd.Timedelta(days=10) if all_asof else None
    xmax = max(all_release) + pd.Timedelta(days=35) if all_release else None
    return xmin, xmax


# ══════════════════════════════════════════════════════════════════════════════
# TIPO 1 — 4-panel Cascaldi-Garcia figure
# ══════════════════════════════════════════════════════════════════════════════

def _make_cg4_figure(
    df: pd.DataFrame,
    value_col: str,
    suptitle: str,
    ylabel: str,
    min_horizon: int = 1,
    ylim_fixed: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    2×2 grid: one CG-style panel per method.  Shared x-axis range and y-axis.

    min_horizon: skip horizons below this value (passed to _draw_cg_panel)
    ylim_fixed:  if set, fix the y-axis range (overrides sharey auto-scale)
    """
    quarters = sorted(df["target_quarter"].unique(), key=_quarter_key)
    color_map = {q: _CG_COLORS[i % len(_CG_COLORS)] for i, q in enumerate(quarters)}

    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    # pad to 4 slots so the grid is always 2×2
    while len(methods) < 4:
        methods.append(None)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8),
        sharex=True, sharey=True,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    xmins: list[pd.Timestamp] = []
    xmaxs: list[pd.Timestamp] = []

    for idx, ax in enumerate(axes.flatten()):
        meth = methods[idx]
        if meth is None:
            ax.set_visible(False)
            continue

        row, col = divmod(idx, 2)
        show_xlabel = (row == 1)
        show_ylabel = (col == 0)

        df_m = df[df["method"] == meth].copy()
        df_m["as_of_dt"] = pd.to_datetime(df_m["as_of"])

        xmin, xmax = _draw_cg_panel(
            ax, df_m, quarters, color_map,
            value_col, meth, ylabel,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            min_horizon=min_horizon,
            ylim_fixed=ylim_fixed,
        )
        if xmin is not None:
            xmins.append(xmin)
        if xmax is not None:
            xmaxs.append(xmax)

    # apply consistent shared xlim
    if xmins and xmaxs:
        axes[0, 0].set_xlim(min(xmins), max(xmaxs))

    # shared legend at the bottom centre
    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=1.8, label="Nowcast evolution"),
        Line2D([0], [0], color="gray", marker="o", markersize=6,
               linewidth=0, label="GDP release"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    return fig


def make_figures_cg(
    csv_path: str,
    output_dir: str,
    config_name: str = "small",
) -> list[str]:
    """Write the 4-panel CG figure for LEVEL and Z-SCORE in both styles.

    Generates two subfolders inside output_dir:
      figure_zoomate/  — auto y-scale, all 4 horizons (original style)
      figure_-10_5/    — fixed ylim (-10, 5), h2-h4 only (CG clean style)

    Returns all PNG paths written.
    """
    df = pd.read_csv(csv_path)
    df["target_quarter"] = df["target_quarter"].astype(str)
    period = _period_str(df)

    views: list[tuple[str, str, str, str]] = [
        (
            "nowcast_livello",
            "Nowcast trajectories (GDP growth, quarterly %)",
            "GDP growth (%)",
            "level",
        )
    ]

    written: list[str] = []
    for style_folder, style_params in _FIGURE_STYLES.items():
        out_dir = os.path.join(output_dir, style_folder)
        os.makedirs(out_dir, exist_ok=True)
        for value_col, suptitle, ylabel, tag in views:
            fig = _make_cg4_figure(
                df, value_col, suptitle, ylabel,
                min_horizon=style_params["min_horizon"],
                ylim_fixed=style_params["ylim_fixed"],
            )
            fname = f"cg4_{tag}_{period}.png"
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
            print(f"  wrote: {path}")

    return written


# ══════════════════════════════════════════════════════════════════════════════
# TIPO 2 — Method comparison for one target quarter
# ══════════════════════════════════════════════════════════════════════════════

def _make_compare_figure(
    df: pd.DataFrame,
    target_quarter: str,
    value_col: str,
    title: str,
    ylabel: str,
) -> plt.Figure:
    """All methods overlaid on one panel for a single target quarter."""
    sub = df[df["target_quarter"] == target_quarter].copy()
    sub["as_of_dt"] = pd.to_datetime(sub["as_of"])
    methods = _methods_present(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    last_asof: pd.Timestamp | None = None

    for i, meth in enumerate(methods):
        st = _style_for(meth, i)
        mrows = sub[sub["method"] == meth].sort_values("as_of_dt")
        if mrows.empty:
            continue
        ax.plot(
            mrows["as_of_dt"], mrows[value_col],
            color=st["color"], linewidth=1.8, label=st["label"],
            solid_capstyle="round", solid_joinstyle="round",
        )
        candidate = mrows["as_of_dt"].max()
        if last_asof is None or candidate > last_asof:
            last_asof = candidate

    # single release dot
    if last_asof is not None:
        release_date = last_asof + pd.Timedelta(days=_RELEASE_OFFSET_DAYS)
        realised_level = float(sub["gdp_realizzato"].iloc[0])
        if value_col == "nowcast_livello":
            release_y = realised_level
        else:
            last_h = sub["horizon_month"].max()
            release_y = _recover_realised_z(
                sub[sub["horizon_month"] == last_h], realised_level
            )
        if np.isfinite(release_y):
            ax.scatter([release_date], [release_y],
                       color="black", s=70, zorder=6, linewidths=0,
                       label="GDP release")
            ax.annotate(
                "realised", (release_date, release_y),
                textcoords="offset points", xytext=(6, 0),
                ha="left", va="center", fontsize=8.5, color="black",
            )
        ax.set_xlim(
            sub["as_of_dt"].min() - pd.Timedelta(days=8),
            release_date + pd.Timedelta(days=30),
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.45)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_tick_params(labelsize=8)
    _apply_clean_style(ax)
    fig.patch.set_facecolor("white")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def make_figures_compare(
    csv_path: str,
    output_dir: str,
    config_name: str = "small",
    target_quarter: str | None = None,
) -> list[str]:
    """Write method-comparison figure(s).  One quarter or all quarters."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["target_quarter"] = df["target_quarter"].astype(str)
    period = _period_str(df)
    has_z = df["nowcast_z"].notna().any()

    quarters = ([target_quarter] if target_quarter
                else sorted(df["target_quarter"].unique(), key=_quarter_key))

    written: list[str] = []
    for q in quarters:
        if q not in df["target_quarter"].values:
            print(f"  WARNING: {q} not in CSV — skipping")
            continue
        qlabel = _quarter_short_label(q)
        views: list[tuple[str, str, str, str]] = [
            (
                "nowcast_livello",
                f"Nowcast trajectories — {q} (all methods)",
                "GDP growth (%)",
                "level",
            )
        ]
        if has_z:
            views.append((
                "nowcast_z",
                f"Nowcast trajectories (standardised) — {q} (all methods)",
                "Nowcast z-score",
                "zscore",
            ))
        for value_col, title, ylabel, tag in views:
            fig = _make_compare_figure(df, q, value_col, title, ylabel)
            fname = f"compare_{qlabel}_{tag}_{period}.png"
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
            print(f"  wrote: {path}")

    return written


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    def _extra(p: argparse.ArgumentParser) -> None:
        p.add_argument("--csv", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument(
            "--style", choices=["cg", "compare"], default="cg",
            help="cg = 4-panel CG timeline (default); "
                 "compare = all methods on one quarter (needs --target)",
        )
        p.add_argument("--target", default=None,
                       help="compare style: target quarter, e.g. 2008Q4")

    args = parse_config_args("Plot rolling-nowcast trajectory figures.", extra=_extra)

    csv_path = _resolve_csv(args.csv, args.config)
    out_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "output", "forecast_realtime", "figures", args.config
    )

    print(f"reading: {csv_path}")
    df_preview = pd.read_csv(csv_path)
    n_q = df_preview["target_quarter"].nunique()
    print(f"  {len(df_preview)} rows, {n_q} quarters, "
          f"methods: {_methods_present(df_preview)}")

    if args.style == "cg":
        written = make_figures_cg(csv_path, out_dir, args.config)
    else:
        written = make_figures_compare(csv_path, out_dir, args.config, args.target)

    print(f"\nwrote {len(written)} figure(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
