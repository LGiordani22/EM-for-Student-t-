"""
src/run_experiment_c.py
=======================
Experiment C — Contamination robustness: idiosyncratic outliers (pi > 0).

Objective
---------
Experiments A and B vary the *tails of the DGP* (heavy vs Gaussian) but keep
the data "clean": every observation is drawn from the same generative model.
Experiment C instead keeps the model fixed and *injects additive outliers*
into a fraction ``pi`` of the periods.  Concretely, for each period an
independent Bernoulli(pi) indicator ``z_t`` decides whether that period is
contaminated; at a contaminated period the idiosyncratic shock is REPLACED by
an inflated heavy-tailed draw from ``t_{nu_contam}(0, kappa^2 R)``.  The factor
signal ``Lambda f_t`` is left untouched — the contamination lives entirely in
the idiosyncratic component (see :func:`simulate_dfm` / ``apply_contamination``).

The economic question is robustness: when a handful of periods carry large
idiosyncratic shocks (data glitches, one-off spikes, a COVID-like month), does
the estimator's factor still track the truth, or does it chase the outliers?

The expected outcome:
  - The Student-t estimator DOWN-WEIGHTS the contaminated periods (its
    idiosyncratic weights ``w_eps_hat`` collapse there), so its factor stays
    close to the true factor.
  - The Gaussian estimator has no such mechanism (weights identically 1), so
    the outliers pull its factor off course even though the *true* factor was
    never contaminated.
  - As ``pi -> 0`` Experiment C collapses onto Experiment B (clean Gaussian-
    tail data); as the contamination intensifies the panel looks more like the
    heavy-tailed regime of Experiment A.

Thesis reference: section "Experiment C — contamination robustness",
line ~12071+.

Construction of the contaminated DGP
------------------------------------
Experiment C's BASE DGP is Gaussian: it takes the calibrated ``theta_star``
and sends ``nu_u = nu_eps = inf`` (exactly :func:`_build_theta_star_C`, the
same transform as Experiment B's ``_build_theta_star_B``).  The heavy tails do
NOT live in ``theta`` here — the only departure from Gaussianity is the
additive-outlier OVERLAY injected by the simulator through ``pi`` / ``kappa`` /
``nu_contam``.  This makes ``pi`` a clean single knob that interpolates between
the two reference experiments:

  - at ``pi = 0`` the overlay is empty and the panel is bit-identical to
    Experiment B (clean Gaussian DGP, ``contam_mask`` all-False) — so the
    ``pi = 0`` column of every table reproduces B and is the uncontaminated
    baseline against which the degradation is read;
  - as ``pi`` grows a Bernoulli(``pi``) fraction of periods has its
    idiosyncratic shock replaced by an inflated ``t_{nu_contam}(0, kappa^2 R)``
    draw, pushing the panel toward the heavy-tailed, outlier-ridden regime of
    Experiment A.  The factor signal ``Lambda f_t`` is never touched: the
    contamination is purely idiosyncratic.

Keeping the tails OUT of ``theta`` and IN the overlay is what isolates the
contamination effect: any change across the ``pi`` grid is attributable to the
additive outliers alone, not to a different calibration.

The three thesis predictions (made visible in the reporting)
------------------------------------------------------------
  1. GAUSSIAN DEGRADES — lacking any down-weighting, the Gaussian estimator
     treats the inflated idiosyncratic spikes as signal: its Lambda / Q / R
     relative errors GROW with ``pi``.
  2. STUDENT-T STAYS STABLE — its idiosyncratic ``nu_eps_hat`` ADAPTS
     (decreases with ``pi`` as the data look heavier-tailed) and absorbs the
     outliers, so its recovery metrics stay flat across the ``pi`` grid.
  3. DETECTION — the Student-t down-weighting locks onto the TRUE contaminated
     periods: ``detection_rate_natural`` is high and ``detection_lift_natural``
     stays well above 1 as ``pi`` grows.  The Gaussian estimator has a constant
     ``w_eps_hat`` (no ranking), so its detection metrics are ``nan`` by design.

Usage
-----
    python src/run_experiment_c.py [--S 20] [--full]   # Monte Carlo grid + report
                                                        # (also emits the 5 figures)

    --S N   : replications per scenario (default 20 for validation)
    --full  : full design (S=1000, T_grid=[100,200,400,800,497])
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import sys

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_utils import parse_config_args                      # noqa: E402
from data_loader import BLOCK, FREQ, ORDERED_COLS, load_config  # noqa: E402
from em_main import fit_dfm, load_dfm_fit                       # noqa: E402
from simulate_dfm import simulate_dfm                           # noqa: E402
from monte_carlo_recovery import (                              # noqa: E402
    init_theta_from_synthetic,
    align_sign_per_factor,
)
from monte_carlo import (                                       # noqa: E402
    run_grid,
    _scenario_filename,
    _load_scenario,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _tee_file(path: pathlib.Path):
    """Tee stdout to *path* for the duration of this context."""
    buf = io.StringIO()
    real_stdout = sys.stdout
    class _T:
        encoding = getattr(real_stdout, "encoding", "utf-8")
        errors   = getattr(real_stdout, "errors",   "replace")
        def write(self, s):   real_stdout.write(s); buf.write(s)
        def flush(self):      real_stdout.flush()
        def isatty(self):     return False
    sys.stdout = _T()
    try:
        yield
    finally:
        sys.stdout = real_stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Saved txt report: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Module constants
# ──────────────────────────────────────────────────────────────────────────────

# Fixed seed for ALL figure panels.  Every figure re-derives its synthetic
# panel from this single seed so that the plots are exactly reproducible and so
# that panels that are meant to be compared (e.g. different kappa or pi on
# "the same series") share an identical clean backbone — only the contamination
# overlay changes (the simulator draws the contamination from a disjoint RNG
# stream ``seed + 2``, independent of the factor/observation streams).
_DEFAULT_FIG_SEED: int = 20240601

# One representative MONTHLY series per block, used as the "reference series"
# for the per-block residual plots.  All three are monthly FRED-MD series, so
# their idiosyncratic residual eps_{t,i} = y_{t,i} - lambda_i^T f_t is a clean
# one-to-one signal (no mixed-frequency aggregation to unwind).
#   - real      : INDPRO    (industrial production; the canonical activity proxy)
#   - financial : S&P 500   (equity index; the financial-block headline series)
#   - other     : CPIAUCSL  (headline CPI; the price/other-block series)
_REF_SERIES_PER_BLOCK: dict[str, str] = {
    "real":      "INDPRO",
    "financial": "S&P 500",
    "other":     "CPIAUCSL",
}

# Colour scheme shared across the figures (sober, colour-blind friendly).
_C_TRUE = "C0"      # true / clean signal
_C_GAUSS = "C3"     # Gaussian estimator (the one that chases outliers)
_C_STUDT = "C2"     # Student-t estimator (the robust one)
_C_OUTLIER = "C3"   # contaminated-period highlight


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers (simulation, fitting, residuals, sign alignment)
# ──────────────────────────────────────────────────────────────────────────────

def _panel_meta(theta_star: dict, cfg_data: dict | None = None) -> dict:
    """
    Assemble the panel metadata that both the simulator and the estimator need.

    These are exactly the defaults used by the Monte Carlo driver
    (:func:`monte_carlo.run_one_replication`): the canonical column order, the
    block membership, the per-column frequency list, and the factor dimension
    ``r`` read off ``theta_star``.  Bundling them in one dict keeps the figure
    functions uncluttered.
    cfg_data, if provided, overrides the module-level ORDERED_COLS/BLOCK/FREQ.
    """
    oc = cfg_data["ORDERED_COLS"] if cfg_data is not None else ORDERED_COLS
    bm = cfg_data["BLOCK"]        if cfg_data is not None else BLOCK
    fq = cfg_data["FREQ"]         if cfg_data is not None else FREQ
    return {
        "ordered_cols": oc,
        "block_map":    bm,
        "freq_list":    [fq[c] for c in oc],
        "r":            int(np.asarray(theta_star["A"]).shape[0]),
    }


def _simulate_panel(
    theta_star: dict,
    meta: dict,
    *,
    T: int,
    seed: int,
    pi: float,
    kappa: float,
    nu_contam: float,
) -> dict:
    """
    Draw one synthetic panel with the requested contamination settings.

    Thin wrapper around :func:`simulate_dfm` that passes the shared panel
    metadata.  Returns the full simulator dict, of which the figures use:
      - ``Y``           : the masked panel (fed to the estimator);
      - ``Y_complete``  : the pre-mask panel (used to read clean per-series
                          idiosyncratic residuals without NaN gaps);
      - ``F``           : the TRUE monthly factor (never contaminated);
      - ``contam_mask`` : the (T,) boolean ground-truth outlier indicator.
    """
    return simulate_dfm(
        theta=theta_star,
        T=T,
        freq_list=meta["freq_list"],
        block_map=meta["block_map"],
        ordered_cols=meta["ordered_cols"],
        r=meta["r"],
        seed=seed,
        pi=pi,
        nu_contam=nu_contam,
        kappa=kappa,
    )


def _refit_panel(Y: np.ndarray, meta: dict, *, gaussian: bool) -> dict:
    """
    Re-estimate the DFM on a (contaminated) panel from a blind PCA start.

    Mirrors the Monte Carlo pipeline exactly: a fresh PCA initialisation on the
    synthetic panel (no leakage of the true theta), then the full EM via
    :func:`fit_dfm`.  ``gaussian=False`` is the Student-t estimator (with the
    weight-attenuation mechanism); ``gaussian=True`` is the Gaussian estimator
    (weights pinned to 1).  ``use_full_elbo=True`` is forced for parity with
    Experiments A/B.  Returns the full fit dict; the figures read
    ``fit["theta"]``, ``fit["f_smooth"]`` and ``fit["e_step_output"]["w_eps"]``.
    """
    theta_0, _ = init_theta_from_synthetic(
        Y,
        ordered_cols=meta["ordered_cols"],
        block_map=meta["block_map"],
        freq_map=FREQ,
    )
    return fit_dfm(
        Y=Y,
        theta_init=theta_0,
        freq_list=meta["freq_list"],
        block_map=meta["block_map"],
        ordered_cols=meta["ordered_cols"],
        verbose=False,
        save_path=None,
        gaussian=gaussian,
        use_full_elbo=True,
    )


def _idiosyncratic_residual(
    Y_complete: np.ndarray,
    F: np.ndarray,
    Lambda: np.ndarray,
    col_idx: int,
) -> np.ndarray:
    """
    Reconstruct the TRUE idiosyncratic residual of one monthly series.

    For a monthly series the observation equation is simply
    ``y_{t,i} = lambda_i^T f_t + eps_{t,i}``, so subtracting the (known) factor
    signal recovers the idiosyncratic shock that actually entered the panel:

        eps_{t,i} = y_{t,i} - lambda_i^T f_t .

    At a contaminated period this ``eps`` IS the inflated heavy-tailed draw, so
    plotting it over time makes the outliers visible exactly where they live —
    in the idiosyncratic component, not in the factor.  ``Y_complete`` (the
    pre-mask panel) is used so the residual has no mixed-frequency / ragged-edge
    gaps (only the first few MM-boundary periods may be NaN, which matplotlib
    renders as a small gap at the very start).
    """
    return Y_complete[:, col_idx] - F @ Lambda[col_idx, :]


def _sign_align_factor(
    f_smooth_hat: np.ndarray,
    Lambda_hat: np.ndarray,
    Lambda_star: np.ndarray,
    r: int,
) -> np.ndarray:
    """
    Sign-align an estimated factor matrix to the true factor's orientation.

    Both fits are already sign-normalised against their own reference series,
    but a residual per-factor sign flip can remain (PCA initialisation noise).
    We reuse the recovery module's rule — pick the sign of
    ``Lambda_hat[:, j]^T Lambda_star[:, j]`` per factor — so that the plotted
    estimated factor and the true factor share orientation and the visual
    comparison is meaningful.  Only the contemporaneous block ``[:, :r]`` of the
    augmented smoothed state is returned (that is the actual factor; the higher
    lag blocks are not plotted).  The scale needs no adjustment: ``theta_star``
    and every fit live in the same Convention-1 (unit total variance) frame.
    """
    d_sign = align_sign_per_factor(Lambda_hat, Lambda_star)   # (r,) in {-1,+1}
    f_now = np.asarray(f_smooth_hat)[:, :r].copy()
    return f_now * d_sign[None, :]


def _mark_contaminated(ax, contam_mask: np.ndarray, series: np.ndarray | None = None,
                       *, band_label: str | None = None,
                       dot_label: str | None = None) -> None:
    """
    Highlight the contaminated periods on a time-axis plot.

    Draws faint vertical bands at every contaminated period and, if a ``series``
    is supplied, overlays red dots on that series at the contaminated periods.
    Both cues point at the SAME (T,) boolean ``contam_mask`` — the simulator's
    ground truth — so the reader can see at a glance which spikes are injected
    outliers rather than ordinary noise.
    """
    idx = np.where(contam_mask)[0]
    if idx.size == 0:
        return
    ymin, ymax = ax.get_ylim()
    ax.vlines(idx, ymin=ymin, ymax=ymax, colors=_C_OUTLIER, alpha=0.15, lw=0.8,
              label=band_label)
    ax.set_ylim(ymin, ymax)
    if series is not None:
        ax.plot(idx, np.asarray(series)[idx], "o", color=_C_OUTLIER, ms=3.5,
                label=dot_label)


# ──────────────────────────────────────────────────────────────────────────────
# The five figures
# ──────────────────────────────────────────────────────────────────────────────

def _fig_a_overview(plt, theta_star, meta, *, seed, T, pi, kappa, nu_contam,
                    out_path) -> pathlib.Path:
    """
    FIGURE A — "contamination_overview": where the outliers live.

    One panel per block (real / financial / other), each showing the TRUE
    idiosyncratic residual eps_{t,i} of that block's reference monthly series
    over time, with the contaminated periods highlighted (vertical bands + red
    dots).

    What it shows / what to conclude
    --------------------------------
    The outliers sit in the IDIOSYNCRATIC component, not in the factor: the
    spikes appear in eps = y - Lambda f.  About ``pi`` (~5%) of the periods are
    flagged, and because the contamination is a per-PERIOD event (the Bernoulli
    indicator z_t is shared across series), the SAME periods light up in all
    three blocks simultaneously — a contaminated month hits every series at
    once, each through its own inflated idiosyncratic draw.
    """
    Lambda = np.asarray(theta_star["Lambda"])
    sim = _simulate_panel(theta_star, meta, T=T, seed=seed, pi=pi,
                          kappa=kappa, nu_contam=nu_contam)
    Yc, F, mask = sim["Y_complete"], sim["F"], sim["contam_mask"]

    blocks = ["real", "financial", "other"]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(11, 8), sharex=True)
    for ax, blk in zip(axes, blocks):
        name = _REF_SERIES_PER_BLOCK[blk]
        col = meta["ordered_cols"].index(name)
        eps = _idiosyncratic_residual(Yc, F, Lambda, col)
        ax.plot(eps, lw=0.8, color=_C_TRUE, label=f"eps (idiosyncratic)")
        ax.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
        _mark_contaminated(ax, mask, series=eps,
                           band_label="contaminated period",
                           dot_label="outlier")
        ax.set_ylabel(f"{blk}\n{name}", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("simulated time index")
    n_contam = int(mask.sum())
    fig.suptitle(
        f"Experiment C — contamination overview  |  T={T}, seed={seed}, "
        f"pi={pi:.2f} ({n_contam}/{T} periods), kappa={kappa:.0f}, "
        f"nu_contam={nu_contam:.0f}\n"
        f"Outliers live in the idiosyncratic residual eps = y - Lambda f and "
        f"hit all blocks in the same periods",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _fig_b_kappa(plt, theta_star, meta, *, seed, T, pi, nu_contam,
                 kappa_list, out_path) -> pathlib.Path:
    """
    FIGURE B — "contamination_kappa": kappa controls the AMPLITUDE of outliers.

    Same financial reference series, same seed, three stacked panels:
    a clean baseline (pi = 0) and two contaminated panels at the SAME pi but
    increasing kappa (e.g. 3 then 5).  Because pi and the seed are fixed, the
    contaminated PERIODS are identical across the two contaminated panels — only
    the size of the spikes changes.  All panels share a common y-range so the
    growth in amplitude is read off directly.

    What it shows / what to conclude
    --------------------------------
    kappa is the scale-inflation of the contaminated idiosyncratic covariance
    (the outlier draw is t_{nu_contam}(0, kappa^2 R)).  The variance scales with
    kappa^2, so going from the clean baseline to kappa=5 inflates the outlier
    variance by ~25x: the spikes tower over the ordinary noise.  kappa sets HOW
    BIG the outliers are, independently of HOW OFTEN they occur (that is pi,
    Figure C).
    """
    Lambda = np.asarray(theta_star["Lambda"])
    name = _REF_SERIES_PER_BLOCK["financial"]
    col = meta["ordered_cols"].index(name)

    # Panel specs: (label, pi, kappa).  Baseline first, then increasing kappa.
    specs = [("baseline (pi=0)", 0.0, kappa_list[0])]
    specs += [(f"kappa={k:.0f}", pi, float(k)) for k in kappa_list]

    # Pre-compute every residual so we can fix a common y-range from the most
    # contaminated panel (largest kappa), making amplitudes visually comparable.
    panels = []
    for label, pi_v, kappa_v in specs:
        sim = _simulate_panel(theta_star, meta, T=T, seed=seed, pi=pi_v,
                              kappa=kappa_v, nu_contam=nu_contam)
        eps = _idiosyncratic_residual(sim["Y_complete"], sim["F"], Lambda, col)
        panels.append((label, pi_v, eps, sim["contam_mask"]))
    finite = np.concatenate([p[2][np.isfinite(p[2])] for p in panels])
    ylim = (1.05 * finite.min(), 1.05 * finite.max())

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8), sharex=True)
    for ax, (label, pi_v, eps, mask) in zip(axes, panels):
        ax.plot(eps, lw=0.8, color=_C_TRUE)
        ax.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
        ax.set_ylim(ylim)
        if pi_v > 0:
            _mark_contaminated(ax, mask, series=eps,
                               band_label="contaminated period")
            # Legend only where there is something to label: the clean baseline
            # (pi=0) has no contaminated period / outlier, so no legend there.
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
    axes[-1].set_xlabel("simulated time index")
    fig.suptitle(
        f"Experiment C — kappa = outlier AMPLITUDE  |  {name}, seed={seed}, "
        f"pi={pi:.2f}, nu_contam={nu_contam:.0f}\n"
        f"Same periods, larger spikes as kappa grows (variance ~ kappa^2 R; "
        f"kappa=5 ~ 25x baseline)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _fig_c_pi(plt, theta_star, meta, *, seed, T, kappa, nu_contam,
              pi_list, out_path) -> pathlib.Path:
    """
    FIGURE C — "contamination_pi": pi controls the FREQUENCY of outliers.

    Same financial reference series, same seed, fixed kappa; three stacked
    panels at increasing pi (e.g. 0.01, 0.05, 0.10).  As pi grows, more periods
    are flagged as contaminated — the number of spikes increases while their
    typical amplitude (set by kappa) stays the same.

    What it shows / what to conclude
    --------------------------------
    pi is the contamination FREQUENCY (the Bernoulli probability that any given
    period is an outlier).  The two limits bracket the other experiments:
      - pi -> 0  recovers Experiment B (clean data, no outliers);
      - larger pi  pushes the panel toward the heavy-tailed regime of
        Experiment A, where a non-negligible share of periods carry fat-tailed
        shocks.
    Together with Figure B this separates the two axes of contamination:
    pi = how OFTEN, kappa = how BIG.
    """
    Lambda = np.asarray(theta_star["Lambda"])
    name = _REF_SERIES_PER_BLOCK["financial"]
    col = meta["ordered_cols"].index(name)

    panels = []
    for pi_v in pi_list:
        sim = _simulate_panel(theta_star, meta, T=T, seed=seed, pi=float(pi_v),
                              kappa=kappa, nu_contam=nu_contam)
        eps = _idiosyncratic_residual(sim["Y_complete"], sim["F"], Lambda, col)
        panels.append((float(pi_v), eps, sim["contam_mask"]))
    finite = np.concatenate([p[1][np.isfinite(p[1])] for p in panels])
    ylim = (1.05 * finite.min(), 1.05 * finite.max())

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8), sharex=True)
    for ax, (pi_v, eps, mask) in zip(axes, panels):
        ax.plot(eps, lw=0.8, color=_C_TRUE)
        ax.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
        ax.set_ylim(ylim)
        _mark_contaminated(ax, mask, series=eps, band_label="contaminated period")
        n_contam = int(mask.sum())
        ax.set_ylabel(f"pi={pi_v:.2f}\n({n_contam}/{T})", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("simulated time index")
    fig.suptitle(
        f"Experiment C — pi = outlier FREQUENCY  |  {name}, seed={seed}, "
        f"kappa={kappa:.0f}, nu_contam={nu_contam:.0f}\n"
        f"More spikes as pi grows (pi->0 = Experiment B; larger pi -> toward "
        f"the heavy-tailed regime of Experiment A)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _fig_d_weights(plt, theta_star, meta, *, seed, T, pi, kappa, nu_contam,
                   out_path) -> pathlib.Path:
    """
    FIGURE D — "contamination_weights": the Student-t down-weighting at work.

    Two time-aligned panels sharing the x-axis:
      - TOP    : the idiosyncratic residual eps of the financial reference
                 series, outliers highlighted (same as Figure A, financial row);
      - BOTTOM : the idiosyncratic weights ``w_eps_hat`` ESTIMATED by the
                 Student-t model, re-fitted on this very contaminated panel.

    What it shows / what to conclude
    --------------------------------
    The Student-t E-step assigns each period a weight that shrinks toward zero
    as the period's residual grows in the Mahalanobis metric.  The bottom panel
    should therefore show ``w_eps_hat`` COLLAPSING precisely at the contaminated
    periods marked in the top panel: the model recognises the outliers as low-
    weight observations and discounts them when it updates Lambda, Q, R and the
    factor.  This is the direct visual evidence of the robustness mechanism that
    Figure E then shows paying off at the factor level.  (The Gaussian estimator
    is not plotted here: its weights are pinned to 1 by construction, a flat line
    that would down-weight nothing.)
    """
    Lambda = np.asarray(theta_star["Lambda"])
    name = _REF_SERIES_PER_BLOCK["financial"]
    col = meta["ordered_cols"].index(name)

    sim = _simulate_panel(theta_star, meta, T=T, seed=seed, pi=pi,
                          kappa=kappa, nu_contam=nu_contam)
    eps = _idiosyncratic_residual(sim["Y_complete"], sim["F"], Lambda, col)
    mask = sim["contam_mask"]

    # Re-fit the Student-t estimator on the contaminated panel and read the
    # estimated idiosyncratic weights (one scalar per period).
    fit = _refit_panel(sim["Y"], meta, gaussian=False)
    w_eps_hat = np.asarray(fit["e_step_output"]["w_eps"])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_top.plot(eps, lw=0.8, color=_C_TRUE, label="eps (idiosyncratic)")
    ax_top.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
    _mark_contaminated(ax_top, mask, series=eps,
                       band_label="contaminated period", dot_label="outlier")
    ax_top.set_ylabel(f"residual\n{name}", fontsize=9)
    ax_top.legend(loc="upper right", fontsize=8)

    ax_bot.plot(w_eps_hat, lw=0.9, color=_C_STUDT, label="w_eps_hat (Student-t)")
    ax_bot.axhline(1.0, color="grey", lw=0.5, alpha=0.5,
                   label="w = 1 (no down-weighting)")
    # Mark the SAME contaminated periods here, and overlay red dots ON the
    # weights (same style as the top-panel outlier dots) so the eye connects
    # vertically "outlier above -> low weight below": those red dots should sit
    # near the bottom of the panel.
    _mark_contaminated(ax_bot, mask, series=w_eps_hat, band_label=None,
                       dot_label="contaminated period (weight)")
    ax_bot.set_ylabel("estimated\nweight", fontsize=9)
    ax_bot.set_xlabel("simulated time index")
    ax_bot.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Experiment C — Student-t down-weighting  |  {name}, seed={seed}, "
        f"pi={pi:.2f}, kappa={kappa:.0f}, nu_contam={nu_contam:.0f}\n"
        f"w_eps_hat collapses exactly at the contaminated periods: the model "
        f"discounts the outliers",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _fig_e_factor_robustness(plt, theta_star, meta, *, seed, T, pi, kappa,
                             nu_contam, factor_idx, out_path) -> pathlib.Path:
    """
    FIGURE E — "factor_robustness": the payoff (the most important figure).

    One factor over time, three lines:
      - F_true       : the TRUE factor (never contaminated);
      - F_hat gauss  : the Gaussian estimator's factor, re-fitted on the
                       contaminated panel;
      - F_hat stud-t : the Student-t estimator's factor, re-fitted on the same
                       contaminated panel.
    Both estimated factors are sign-aligned to the true factor's orientation
    (the recovery rule, based on the loadings' inner product); the scale needs
    no adjustment since all three live in the same Convention-1 frame.  A higher
    contamination level (pi ~ 0.10) is used so the effect is unmistakable.

    What it shows / what to conclude
    --------------------------------
    This is the practical reason for the Student-t model.  The crucial subtlety:
    the TRUE factor is NEVER contaminated — the outliers were injected only into
    the idiosyncratic component.  Yet the Gaussian estimator still gets dragged
    off course: lacking any down-weighting, it treats every inflated
    idiosyncratic spike as if it were signal, and to fit those large residuals
    it bends the factor toward them (and inflates Q, R).  The Student-t
    estimator instead recognises the contaminated periods as low-weight
    observations (Figure D) and discounts them, so its factor holds the line and
    stays close to F_true.  In other words: idiosyncratic outliers leak into the
    Gaussian factor but are filtered out of the Student-t factor — which is
    exactly what we need for a credible Growth-at-Risk signal.
    """
    Lambda_star = np.asarray(theta_star["Lambda"])
    r = meta["r"]

    sim = _simulate_panel(theta_star, meta, T=T, seed=seed, pi=pi,
                          kappa=kappa, nu_contam=nu_contam)
    F_true = sim["F"]
    mask = sim["contam_mask"]

    # Re-fit BOTH estimators on the identical contaminated panel.
    fit_g = _refit_panel(sim["Y"], meta, gaussian=True)
    fit_t = _refit_panel(sim["Y"], meta, gaussian=False)

    Fg = _sign_align_factor(fit_g["f_smooth"], np.asarray(fit_g["theta"]["Lambda"]),
                            Lambda_star, r)
    Ft = _sign_align_factor(fit_t["f_smooth"], np.asarray(fit_t["theta"]["Lambda"]),
                            Lambda_star, r)

    j = int(factor_idx)
    block_name = ["real", "financial", "other"][j] if j < 3 else f"factor {j}"

    # Correlation with the TRUE factor — the scale-invariant proof of
    # robustness.  Computed on the raw (sign-aligned) factors; we print these
    # numbers ON the figure so the evidence is self-contained.  The Student-t
    # correlation should be high (~1) and the Gaussian one near zero on the
    # contaminated financial block.
    ft, fg, f0 = Ft[:, j], Fg[:, j], np.asarray(F_true)[:, j]
    corr_t = float(np.corrcoef(ft, f0)[0, 1])
    corr_g = float(np.corrcoef(fg, f0)[0, 1])

    # GRAPHICAL-ONLY rescaling: F_true is plotted at its realised sample
    # amplitude, while the estimates are pinned to total variance 1 by
    # Convention-1, so the Student-t line can look ~2x wider than F_true even
    # though corr ~ 1.  To make the three lines comparable in amplitude we
    # standardise EACH series to unit sample std before plotting.  This is
    # purely cosmetic — correlation is scale-invariant, so the substance is
    # unchanged; it only lets the eye see Student-t (green) sitting on top of
    # F_true (blue) while the Gaussian (red) peels off at the contaminated
    # periods.
    def _unit_std(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s = np.std(x)
        return x / s if s > 0 else x

    f0_p, fg_p, ft_p = _unit_std(f0), _unit_std(fg), _unit_std(ft)

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.plot(f0_p, lw=1.4, color=_C_TRUE, label="F_true (uncontaminated)")
    ax.plot(fg_p, lw=1.0, color=_C_GAUSS, alpha=0.9,
            label=f"F_hat Gaussian (chases outliers), corr={corr_g:+.3f}")
    ax.plot(ft_p, lw=1.0, color=_C_STUDT, alpha=0.9,
            label=f"F_hat Student-t (robust), corr={corr_t:+.3f}")
    ax.axhline(0.0, color="grey", lw=0.5, alpha=0.5)
    _mark_contaminated(ax, mask, band_label="contaminated period")
    ax.set_xlabel("simulated time index")
    ax.set_ylabel(f"factor {j} ({block_name}) — standardised (graphical)")
    ax.legend(loc="upper right", fontsize=8)
    # Title split over THREE short lines so it never overflows the 11-inch
    # figure width (the previous single-line variant ran off both edges and was
    # clipped on save).  The correlations get their own dedicated line so they
    # stay legible, and bbox_inches="tight" on savefig guarantees the full
    # suptitle is included in the exported PNG even if it is a touch wide.
    fig.suptitle(
        f"Experiment C — factor robustness  |  factor {j} ({block_name})  |  "
        f"seed={seed}, pi={pi:.2f}, kappa={kappa:.0f}, nu_contam={nu_contam:.0f}\n"
        f"Student-t corr={corr_t:+.3f}   vs   Gaussian corr={corr_g:+.3f}   "
        f"(vs the true, uncontaminated factor)\n"
        f"The true factor is clean, yet the Gaussian estimate chases the "
        f"idiosyncratic outliers while the Student-t holds the line",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point: generate the five figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_contamination_figures(
    theta_star: dict,
    *,
    cfg_data: dict | None = None,
    seed: int = _DEFAULT_FIG_SEED,
    out_dir: "str | pathlib.Path | None" = None,
    pi: float = 0.05,
    kappa: float = 5.0,
    nu_contam: float = 3.0,
    T: int = 200,
    factor_idx: int = 1,
) -> list[pathlib.Path]:
    """
    Generate the five Experiment-C contamination figures as separate PNGs.

    All panels are derived from a single fixed ``seed`` so the whole set is
    exactly reproducible.  Figures A and D use the ``pi`` / ``kappa`` passed in
    (the "baseline" contamination); Figure B sweeps kappa in {3, 5} at this pi,
    Figure C sweeps pi in {0.01, 0.05, 0.10} at this kappa, and Figure E uses a
    stronger pi = 0.10 so the factor distortion is unmistakable.

    Parameters
    ----------
    theta_star : dict
        The calibrated DGP (same as Experiment A).  Must carry ``A``, ``Q``,
        ``Lambda``, ``R``, ``nu_u``, ``nu_eps``.
    seed : int
        Master seed shared by every panel.
    out_dir : path-like or None
        Output directory; defaults to ``<project>/output/figures``.
    pi : float
        Baseline contamination frequency (Figures A, D; the pivot of B/C).
    kappa : float
        Baseline outlier amplitude factor (Figures A, C, D, E; the pivot of B).
    nu_contam : float
        Degrees of freedom of the inflated contaminating Student-t.
    T : int
        Synthetic panel length.
    factor_idx : int
        Which factor to display in Figure E (default 1 = the financial factor,
        the block where the outliers live: the Gaussian-vs-Student-t contrast is
        largest and visually unambiguous there.  On the real factor (idx 0) the
        Gaussian distortion arrives by leakage and reads as "flat + crash",
        which is misleading; the correlations confirm both behave analogously).

    Returns
    -------
    list[pathlib.Path]
        The five saved PNG paths, in order A, B, C, D, E.
    """
    # Headless backend so the figures render without a display (project idiom).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = (pathlib.Path(out_dir) if out_dir is not None
               else _PROJECT_ROOT / "output" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _panel_meta(theta_star, cfg_data)
    paths: list[pathlib.Path] = []

    print(f"Generating Experiment-C contamination figures "
          f"(seed={seed}, T={T}) -> {out_dir}")

    paths.append(_fig_a_overview(
        plt, theta_star, meta, seed=seed, T=T, pi=pi, kappa=kappa,
        nu_contam=nu_contam, out_path=out_dir / "contamination_overview.png"))

    paths.append(_fig_b_kappa(
        plt, theta_star, meta, seed=seed, T=T, pi=pi, nu_contam=nu_contam,
        kappa_list=[3.0, 5.0], out_path=out_dir / "contamination_kappa.png"))

    paths.append(_fig_c_pi(
        plt, theta_star, meta, seed=seed, T=T, kappa=kappa, nu_contam=nu_contam,
        pi_list=[0.01, 0.05, 0.10], out_path=out_dir / "contamination_pi.png"))

    paths.append(_fig_d_weights(
        plt, theta_star, meta, seed=seed, T=T, pi=pi, kappa=kappa,
        nu_contam=nu_contam, out_path=out_dir / "contamination_weights.png"))

    paths.append(_fig_e_factor_robustness(
        plt, theta_star, meta, seed=seed, T=T, pi=0.10, kappa=kappa,
        nu_contam=nu_contam, factor_idx=factor_idx,
        out_path=out_dir / "factor_robustness.png"))

    print("\nSaved figures:")
    for p in paths:
        print(f"  {p}")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo configuration: the contaminated DGP
# ──────────────────────────────────────────────────────────────────────────────

def _build_theta_star_C(theta_star: dict) -> dict:
    """
    Construct the Experiment-C BASE DGP by setting nu_u = nu_eps = np.inf.

    This is byte-for-byte the same transform as Experiment B's
    ``_build_theta_star_B``: it copies Lambda, A, Q, R, Sigma_0 unchanged and
    only sends the two tail parameters to infinity, so the base data-generating
    process is GAUSSIAN.  The heavy tails of Experiment C do NOT live in theta —
    they are injected by the simulator as an additive-outlier OVERLAY, period by
    period, through the ``pi`` / ``kappa`` / ``nu_contam`` arguments of
    :func:`run_grid` (Bernoulli(pi) indicator z_t; at a contaminated period the
    idiosyncratic shock is replaced by an inflated ``t_{nu_contam}(0, kappa^2 R)``
    draw; the factor signal ``Lambda f_t`` is never touched).

    Why nu = inf in theta (and not the heavy theta_star of Experiment A):
      - it makes ``pi`` the SINGLE knob of the experiment.  At ``pi = 0`` the
        overlay is empty and the panel is bit-identical to Experiment B (clean
        Gaussian DGP), so the ``pi = 0`` column reproduces B exactly and is the
        uncontaminated baseline;
      - as ``pi`` grows the panel interpolates toward the heavy-tailed,
        outlier-ridden regime of Experiment A.
    Keeping the only non-Gaussianity in the overlay isolates the contamination
    effect: any change across the ``pi`` grid is attributable to the additive
    outliers alone, not to a different calibration.

    Distinction DGP vs estimator (as in B): here nu = inf lives in the DGP (the
    clean backbone is generated tail-free), which is distinct from the
    ``gaussian=True`` ESTIMATOR (nu = inf imposed on the fitted model).  At each
    ``pi`` Experiment C runs BOTH estimators on the identical contaminated
    panels: the Student-t estimator must adapt its ``nu_eps_hat`` downward to
    absorb the outliers, while the Gaussian estimator has no such mechanism.
    """
    theta_C = {k: np.asarray(v).copy() for k, v in theta_star.items()}
    theta_C["nu_u"]   = np.inf
    theta_C["nu_eps"] = np.inf
    return theta_C


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
#
# Two levels of reporting (the brief: print EVERYTHING; prune later in the
# thesis).  (a) a FULL per-pi table — student_t vs gaussian on every aggregate
# metric available — and (b) a TREND table — key metrics across the pi grid,
# one row per estimator — which is the headline artefact of Experiment C: it is
# where the degradation of the Gaussian estimator and the stability of the
# Student-t one become visible as pi grows.

# Curated, ordered metric spec for the FULL table.  (label, aggregate_key,
# is_pct).  Mirrors monte_carlo.print_aggregate_table's grouping but flattened
# to the per-block expanded keys produced by aggregate_replications.  Any
# aggregate key NOT listed here is still printed by the "uncurated" catch-all
# at the bottom of the table, so nothing is omitted.
_METRIC_SECTIONS: list[tuple[str, list[tuple[str, str, bool]]]] = [
    ("Heavy-tail parameters (Student-t estimator; Gaussian = inf by design)", [
        ("nu_u_hat   (estimate)",        "nu_u_hat",        False),
        ("nu_eps_hat (estimate)",        "nu_eps_hat",      False),
        ("nu_u  rel.err",                "nu_u_relerr",     True),
        ("nu_eps rel.err",               "nu_eps_relerr",   True),
    ]),
    ("Spectral radius of A", [
        ("rho(A) (estimate)",            "rho_A_hat",       False),
        ("rho(A) rel.err",               "rho_A_relerr",    True),
        ("|eig(A_hat)-eig(A*)| Euclid",  "eig_A_err_norm",  False),
    ]),
    ("Loading matrix Lambda", [
        ("Lambda relerr Procrustes-block [PRIMARY]",
         "lambda_relerr_procrustes_blockdiag", True),
        ("Lambda relerr (sign-normalised only)",
         "lambda_relerr_normalised",      True),
    ]),
    ("Block-diagonal Procrustes scale factors", [
        ("h_real",                       "H_block_diag_real",  False),
        ("h_financial",                  "H_block_diag_fin",   False),
        ("h_other",                      "H_block_diag_other", False),
    ]),
    ("Q diagonal per block", [
        ("diag(Q) estimate [real]",      "diagQ_hat_real",     False),
        ("diag(Q) estimate [financial]", "diagQ_hat_fin",      False),
        ("diag(Q) estimate [other]",     "diagQ_hat_other",    False),
        ("diag(Q) rel.err [real]",       "diagQ_relerr_real",  True),
        ("diag(Q) rel.err [financial]",  "diagQ_relerr_fin",   True),
        ("diag(Q) rel.err [other]",      "diagQ_relerr_other", True),
    ]),
    ("R (idiosyncratic variances)", [
        ("R rel.err (median over series)", "R_median_relerr", True),
        ("R rel.err (max over series)",    "R_max_relerr",    True),
    ]),
    ("Factor recovery", [
        ("|corr| factor [real]",         "factor_abscorr_real",   False),
        ("|corr| factor [financial]",    "factor_abscorr_fin",    False),
        ("|corr| factor [other]",        "factor_abscorr_other",  False),
        ("|cross-corr| real-financial",  "factor_crosscorr_RF",   False),
        ("|cross-corr| real-other",      "factor_crosscorr_RX",   False),
        ("|cross-corr| financial-other", "factor_crosscorr_FX",   False),
        ("RMSE trajectory [real]",       "factor_rmse_traj_real", False),
        ("RMSE trajectory [financial]",  "factor_rmse_traj_fin",  False),
        ("RMSE trajectory [other]",      "factor_rmse_traj_other",False),
    ]),
    ("Weight recovery (Student-t specific)", [
        ("corr(w_u_hat,   w_u_true)",    "w_u_corr",          False),
        ("corr(w_eps_hat, w_eps_true)",  "w_eps_corr",        False),
        ("w_u   overlap@5%",             "w_u_overlap_5pct",  True),
        ("w_eps overlap@5%",             "w_eps_overlap_5pct",True),
        ("w_u   lift@5% (overlap/chance)",  "w_u_lift_5pct",  False),
        ("w_eps lift@5% (overlap/chance)",  "w_eps_lift_5pct",False),
    ]),
    ("Contamination detection (Experiment C; Gaussian = nan by design)", [
        ("# contaminated periods (mean)",   "detection_n_contam",       False),
        ("effective T (mean)",              "detection_T_eff",          False),
        ("detection rate  (natural k=n_c)", "detection_rate_natural",   True),
        ("detection lift  (natural k=n_c)", "detection_lift_natural",   False),
        ("detection recall    @5%",         "detection_rate_5pct",      True),
        ("detection precision @5%",         "detection_precision_5pct", True),
        ("detection lift      @5%",         "detection_lift_5pct",      False),
        ("detection recall    @10%",        "detection_rate_10pct",     True),
        ("detection precision @10%",        "detection_precision_10pct",True),
        ("detection lift      @10%",        "detection_lift_10pct",     False),
    ]),
    ("Likelihood", [
        ("loglik_final (mean over reps)", "loglik_final",   False),
        ("loglik_initial (mean)",         "loglik_initial", False),
    ]),
    ("Algorithmic reliability", [
        ("convergence rate",              "converged",       True),
        ("iterations (mean)",             "n_iter",          False),
        ("monotonicity violations (mean)","n_monotonicity_violations", False),
    ]),
]

# Keys that the catch-all should never echo (true constants / bookkeeping that
# carry no estimator contrast).  Any key ending in "_star" is also skipped.
_CATCHALL_SKIP: frozenset[str] = frozenset({"seed", "T", "pi", "nu_frozen"})


def _load_agg(out_dir: pathlib.Path, estimator: str, T: int, pi: float,
              S: int) -> dict | None:
    """Load the ``aggregates`` block of one scenario JSON, or None if absent."""
    path = out_dir / _scenario_filename(estimator, T, pi, S)
    if not path.exists():
        return None
    return _load_scenario(path).get("aggregates", {})


def _full_metric_table(agg_st: dict | None, agg_g: dict | None,
                       *, T: int, S: int, pi: float, kappa: float,
                       nu_contam: float) -> None:
    """
    FULL per-pi table: student_t vs gaussian on EVERY aggregate metric.

    Modelled on Experiment B's ``_loglik_gap_table`` but extended to the entire
    aggregate dictionary (curated sections + an uncurated catch-all), so no
    metric is dropped.  The third column is the gap (student_t - gaussian); on a
    contaminated panel a negative gap on the error metrics (relerr) is the
    Student-t advantage — it recovers theta with smaller error than the Gaussian.
    """
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"  Experiment C — FULL metric table  |  T={T}, S={S}, pi={pi:.2f}")
    print(f"  DGP: Gaussian base (nu=inf) + contamination overlay "
          f"(kappa={kappa:.0f}, nu_contam={nu_contam:.0f})")
    print(f"  Two estimators on identical synthetic panels")
    print(bar)
    print(f"  {'metric':<46s}  {'student_t':>13s}  {'gaussian':>13s}  "
          f"{'gap (t-g)':>13s}")
    print("  " + "-" * 98)

    def _cell(agg: dict | None, key: str, pct: bool) -> tuple[float, str]:
        if agg is None or key not in agg:
            return float("nan"), f"{'N/A':>13s}"
        v = agg[key]["mean"]
        if not np.isfinite(v):
            return float("nan"), f"{'nan':>13s}"
        return v, (f"{v:>13.2%}" if pct else f"{v:>13.4f}")

    def _row(label: str, key: str, pct: bool) -> None:
        vt, st = _cell(agg_st, key, pct)
        vg, sg = _cell(agg_g, key, pct)
        if np.isfinite(vt) and np.isfinite(vg):
            gap = vt - vg
            gs = f"{gap:>+13.2%}" if pct else f"{gap:>+13.4f}"
        else:
            gs = f"{'—':>13s}"
        print(f"  {label:<46s}  {st}  {sg}  {gs}")

    listed: set[str] = set()
    for title, items in _METRIC_SECTIONS:
        for _, k, _ in items:
            listed.add(k)
        present = [(lbl, k, p) for (lbl, k, p) in items
                   if (agg_st and k in agg_st) or (agg_g and k in agg_g)]
        if not present:
            continue
        print(f"\n  [{title}]")
        for label, key, pct in present:
            _row(label, key, pct)

    # Uncurated catch-all: any aggregate key not already listed (so the table
    # is genuinely exhaustive — "non omettere nulla").
    keys: set[str] = set()
    for a in (agg_st, agg_g):
        if a:
            keys |= set(a)
    extra = sorted(k for k in (keys - listed)
                   if k not in _CATCHALL_SKIP and not k.endswith("_star"))
    if extra:
        print(f"\n  [other aggregate keys (uncurated)]")
        for k in extra:
            pct = any(t in k for t in ("relerr", "rate", "precision", "converged"))
            _row(k, k, pct)
    print(bar)


# Trend metrics: rows of the across-pi table.  (label, aggregate_key, is_pct).
# Deliberately the small set the brief calls out — the ones that make the three
# predictions legible — plus the two detection summaries.
_TREND_METRICS: list[tuple[str, str, bool]] = [
    ("Lambda relerr Procrustes-block", "lambda_relerr_procrustes_blockdiag", True),
    ("R median rel.err",               "R_median_relerr",        True),
    ("diag(Q) rel.err [real]",         "diagQ_relerr_real",      True),
    ("diag(Q) rel.err [financial]",    "diagQ_relerr_fin",       True),
    ("diag(Q) rel.err [other]",        "diagQ_relerr_other",     True),
    ("nu_eps_hat (adapts down)",       "nu_eps_hat",             False),
    ("|corr| factor [financial]",      "factor_abscorr_fin",     False),
    ("detection rate (natural)",       "detection_rate_natural", True),
    ("detection lift (natural)",       "detection_lift_natural", False),
]


def _trend_table(aggs: dict, *, T: int, S: int,
                 pi_grid: list[float]) -> None:
    """
    TREND table: key metrics across the pi grid, one row per estimator.

    ``aggs`` maps ``(pi, estimator) -> aggregate-dict | None``.  For each metric
    two rows are printed (student_t above gaussian) with one column per pi, so
    the degradation / stability is read HORIZONTALLY and the estimator contrast
    VERTICALLY.  This is the most important Experiment-C artefact.
    """
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"  Experiment C — TREND across pi  |  T={T}, S={S}")
    print(f"  Read each metric LEFT->RIGHT (effect of pi) and student_t-vs-gaussian "
          f"TOP-vs-BOTTOM.")
    print(f"  Expected: gaussian error metrics GROW with pi; student_t stays flat; "
          f"nu_eps_hat falls.")
    print(bar)
    pi_hdr = "".join(f"{('pi=' + format(p, '.2f')):>12s}" for p in pi_grid)
    print(f"  {'metric':<34s}  {'estimator':<10s}{pi_hdr}")
    print("  " + "-" * (48 + 12 * len(pi_grid)))

    for label, key, pct in _TREND_METRICS:
        for est in ("student_t", "gaussian"):
            cells = ""
            for p in pi_grid:
                a = aggs.get((p, est))
                if a is None or key not in a or not np.isfinite(a[key]["mean"]):
                    cells += f"{'—':>12s}"
                else:
                    v = a[key]["mean"]
                    cells += (f"{v:>12.2%}" if pct else f"{v:>12.4f}")
            print(f"  {label:<34s}  {est:<10s}{cells}")
        print()
    print(bar)


def _print_predictions(aggs: dict, *, T: int, pi_grid: list[float]) -> None:
    """
    Spell out the three thesis predictions against the numbers just tabulated.

    Pulls the relevant aggregate means out of ``aggs`` and prints, for each
    prediction, the pi = 0 baseline vs the pi = max endpoint so the reader sees
    immediately whether the prediction holds in this run.
    """
    def g(pi: float, est: str, key: str) -> float:
        a = aggs.get((pi, est))
        if a is None or key not in a:
            return float("nan")
        return float(a[key]["mean"])

    p0, pmax = pi_grid[0], pi_grid[-1]
    pos_pi = [p for p in pi_grid if p > 0.0]
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"  Experiment C — the three predictions  |  T={T}  "
          f"(pi: {p0:.2f} -> {pmax:.2f})")
    print(bar)

    # (1) Gaussian degrades.
    print("  (1) GAUSSIAN DEGRADES: Lambda / Q / R rel.err should GROW with pi.")
    for lbl, key in [("Lambda relerr", "lambda_relerr_procrustes_blockdiag"),
                     ("diag(Q) relerr [real]", "diagQ_relerr_real"),
                     ("R median relerr", "R_median_relerr")]:
        v0, vM = g(p0, "gaussian", key), g(pmax, "gaussian", key)
        trend = "GROWS" if (np.isfinite(v0) and np.isfinite(vM) and vM > v0) else "?"
        print(f"      {lbl:<24s} [gaussian]:  "
              f"pi={p0:.2f} -> {v0:.2%}   pi={pmax:.2f} -> {vM:.2%}   [{trend}]")

    # (2) Student-t stable; nu_eps adapts down.
    print("\n  (2) STUDENT-T STABLE: recovery metrics flat, nu_eps_hat ADAPTS "
          "(decreases with pi).")
    for lbl, key in [("Lambda relerr", "lambda_relerr_procrustes_blockdiag"),
                     ("R median relerr", "R_median_relerr")]:
        v0, vM = g(p0, "student_t", key), g(pmax, "student_t", key)
        print(f"      {lbl:<24s} [student_t]: "
              f"pi={p0:.2f} -> {v0:.2%}   pi={pmax:.2f} -> {vM:.2%}   [expect ~flat]")
    nu0, nuM = g(p0, "student_t", "nu_eps_hat"), g(pmax, "student_t", "nu_eps_hat")
    falls = "FALLS" if (np.isfinite(nu0) and np.isfinite(nuM) and nuM < nu0) else "?"
    print(f"      {'nu_eps_hat':<24s} [student_t]: "
          f"pi={p0:.2f} -> {nu0:.1f}   pi={pmax:.2f} -> {nuM:.1f}   [{falls}]")

    # (3) Detection.
    print("\n  (3) DETECTION: Student-t flags the TRUE contaminated periods "
          "(rate high, lift > 1);")
    print("      Gaussian has constant w_eps_hat -> nan by design.")
    for p in pos_pi:
        rate = g(p, "student_t", "detection_rate_natural")
        lift = g(p, "student_t", "detection_lift_natural")
        print(f"      pi={p:.2f}:  student_t detection rate={rate:.2%}  "
              f"lift={lift:.2f}     gaussian rate=nan (no down-weighting)")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE F — the robustness curve (a RESULTS figure, not a single-replica figure)
# ──────────────────────────────────────────────────────────────────────────────
#
# Figures A–E above each illustrate ONE example replication (a single synthetic
# panel from a fixed seed) — they are pedagogical pictures of the mechanism.
# Figure F is categorically different: it is a RESULTS figure.  It consumes the
# Monte-Carlo AGGREGATES produced by run_grid (the per-scenario JSONs under
# ``mc_results_expC``, each averaged over S replications) and plots how the
# recovery metrics move as the contamination frequency ``pi`` grows, for BOTH
# estimators side by side.
#
# The reading of the figure is the "robustness margin":
#   - at pi = 0 the panel is the clean Gaussian DGP (== Experiment B), so the
#     two estimators are NESTED and the curves COINCIDE — there is nothing to be
#     robust against yet;
#   - as pi grows the additive idiosyncratic outliers kick in: the Student-t
#     curve should stay flat (it down-weights the contaminated periods), while
#     the Gaussian curve degrades (it chases the outliers).  The VERTICAL GAP
#     between the two curves at each pi IS the robustness margin — how much
#     accuracy the tail-robust estimator buys over the Gaussian one at that
#     contamination level.  Panel 1 (factor recovery) answers directly "at what
#     pi does the Gaussian estimator lose the factor".
#
# Because this is an aggregate figure it must be generated AFTER run_grid and
# AFTER the textual reporting (it reads the very same saved scenarios).  It is
# defensive by construction: any scenario that has not been computed yet (e.g. a
# T not in the grid, or an interrupted run) is simply skipped with a warning
# rather than crashing — exactly the contract of ``_compare_A_vs_B`` in
# Experiment B.

# Spec of the (up to) three robustness panels.  Each tuple is
#   (aggregate_key, y-axis label, is_pct, plot_gaussian?, panel title).
# ``plot_gaussian=False`` marks the detection panel, where the Gaussian
# estimator is nan by design (constant w_eps_hat, no ranking) and so contributes
# no line.
_ROBUSTNESS_PANELS: list[tuple[str, str, bool, bool, str]] = [
    ("factor_abscorr_fin", "|corr(F_hat, F_true)|  [financial]", False, True,
     "Panel 1 — factor recovery (financial block): the GAP = robustness margin"),
    ("R_median_relerr", "R median rel.err", True, True,
     "Panel 2 — parameter recovery (idiosyncratic R): Gaussian degrades, Student-t stable"),
    ("detection_lift_natural", "detection lift (natural)", False, False,
     "Panel 3 — detection: Student-t flags the true outliers (lift > 1); Gaussian = nan"),
]


def _series_across_pi(
    aggs: dict, estimator: str, key: str, pi_grid: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Pull one aggregate metric across the pi grid for a single estimator.

    Walks ``pi_grid`` and, for every pi whose scenario is present and whose
    metric ``key`` is finite, collects the point (pi, mean) plus the per-pi
    dispersion band (q05, q95) computed over the S replications.  Scenarios that
    are missing, or that lack the key, or whose mean is non-finite (e.g. the
    Gaussian estimator's detection metrics, nan by design) are silently dropped
    from THIS series — they leave a gap rather than crashing the plot.  Returns
    four equally-long lists (xs, means, lo, hi); empty lists mean "this
    estimator has no plottable point for this metric".
    """
    xs: list[float] = []
    means: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for p in pi_grid:
        a = aggs.get((p, estimator))
        if a is None or key not in a:
            continue
        stats = a[key]
        m = stats.get("mean", float("nan"))
        if not np.isfinite(m):
            continue
        xs.append(float(p))
        means.append(float(m))
        # q05/q95 give the across-replication spread; fall back to the mean
        # (degenerate band) if a quantile is missing or non-finite.
        q05 = stats.get("q05", m)
        q95 = stats.get("q95", m)
        lo.append(float(q05) if np.isfinite(q05) else float(m))
        hi.append(float(q95) if np.isfinite(q95) else float(m))
    return xs, means, lo, hi


def plot_robustness_curve(
    out_dir_expC: "str | pathlib.Path",
    *,
    T: int = 497,
    pi_grid: "list[float] | None" = None,
    S: int = 1000,
    fig_out_dir: "str | pathlib.Path | None" = None,
) -> "pathlib.Path | None":
    """
    FIGURE F — "robustness_curve": the robustness margin across the pi grid.

    This is a RESULTS figure (NOT one of the single-replication illustrations
    A–E): it reads the Monte-Carlo AGGREGATES that ``run_grid`` saved under
    ``out_dir_expC`` — one JSON per (estimator, T, pi, S), each already averaged
    over the S replications — and draws how the recovery metrics evolve as the
    contamination frequency ``pi`` increases, for the Student-t and the Gaussian
    estimators together.  It must therefore be called AFTER ``run_grid`` and
    AFTER the textual reporting, since it consumes exactly those saved scenarios
    (it uses the same ``_scenario_filename`` / ``_load_scenario`` plumbing via
    the local ``_load_agg`` helper, mirroring ``_compare_A_vs_B`` in
    Experiment B).

    What the reader takes away — the "robustness margin"
    ----------------------------------------------------
    At ``pi = 0`` the DGP is the clean Gaussian panel of Experiment B, where the
    two estimators are NESTED, so the two curves COINCIDE (the nesting check).
    As ``pi`` grows the additive idiosyncratic outliers bite:
      - Panel 1 (factor recovery, the headline): |corr(F_hat, F_true)| of the
        financial block stays ~1 for the Student-t estimator (it down-weights the
        contaminated periods) while the Gaussian estimator DEGRADES (it chases
        the outliers).  The vertical GAP between the two lines at each pi is the
        robustness margin, and the pi where the Gaussian line falls away answers
        "at what contamination level does the Gaussian estimator lose the
        factor".
      - Panel 2 (parameter recovery): the idiosyncratic-variance error
        ``R_median_relerr`` grows for the Gaussian estimator and stays flat for
        the Student-t — the same story on a second front.
      - Panel 3 (detection): the Student-t ``detection_lift_natural`` stays well
        above 1 at every pi > 0 (it locks onto the TRUE contaminated periods);
        the Gaussian estimator has a constant weight (no ranking) so its
        detection is nan by design and contributes NO line — it is annotated
        instead.

    Robustness / graceful degradation
    ---------------------------------
    Every lookup is defensive.  If an entire scenario is missing (a T not yet
    computed, an interrupted run) its points simply do not appear; if a whole
    panel's metric is absent from the aggregates for both estimators, that panel
    is dropped and a warning is printed; if NO panel has any data the function
    returns ``None`` without writing a file.  Nothing here raises — exactly the
    contract honoured by ``_compare_A_vs_B``.

    Parameters
    ----------
    out_dir_expC : path-like
        Directory of the saved Experiment-C scenarios (``mc_results_expC``).
    T : int
        Panel length to slice the grid at (default 497, the real-panel length);
        parametrisable so any computed T can be reported.
    pi_grid : list[float] or None
        The pi values to place on the x-axis; defaults to ``[0.0, 0.01, 0.05,
        0.10]`` (the grid the main driver runs).
    S : int
        Replications per scenario — used only to resolve the scenario filename
        (it is part of the JSON name), so it must match the run that produced
        the aggregates.
    fig_out_dir : path-like or None
        Output directory for the PNG; defaults to ``<project>/output/figures``.

    Returns
    -------
    pathlib.Path or None
        The saved PNG path, or ``None`` if no scenario had any plottable metric.
    """
    # Headless backend so the figure renders without a display (project idiom).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    out_dir_expC = pathlib.Path(out_dir_expC)
    if pi_grid is None:
        pi_grid = [0.0, 0.01, 0.05, 0.10]
    fig_out_dir = (pathlib.Path(fig_out_dir) if fig_out_dir is not None
                   else _PROJECT_ROOT / "output" / "figures")
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_out_dir / "robustness_curve.png"

    estimators = ["student_t", "gaussian"]

    # ── Load every (pi, estimator) aggregate once, warning on missing files ──
    # A missing scenario is not fatal: it just leaves a gap in the curves (the
    # _series_across_pi helper skips absent points), so the figure still draws
    # whatever HAS been computed.  We surface the gaps as warnings, the same way
    # _compare_A_vs_B reports a missing file instead of crashing.
    aggs: dict[tuple[float, str], dict | None] = {}
    n_missing = 0
    for p in pi_grid:
        for est in estimators:
            a = _load_agg(out_dir_expC, est, T, p, S)
            aggs[(p, est)] = a
            if a is None:
                n_missing += 1
                print(f"[WARN] robustness_curve: missing scenario "
                      f"{est} T={T} pi={p:.2f} S={S} in {out_dir_expC}; "
                      f"that point will be skipped.")
    if n_missing == len(pi_grid) * len(estimators):
        print(f"[WARN] robustness_curve: NO scenarios found for T={T} "
              f"(looked for S={S}); skipping Figure F.")
        return None

    # ── Decide which panels actually have data, so we never draw an empty axis ──
    # A panel is kept only if at least one of its estimators contributes at least
    # one finite point.  This also handles the case where an aggregate key (e.g.
    # factor_abscorr_fin or detection_lift_natural) is entirely absent from the
    # saved metrics: that panel is dropped with a warning, the others still draw.
    drawable: list[tuple] = []   # (key, ylabel, is_pct, plot_gaussian, title, series_by_est)
    for key, ylabel, is_pct, plot_gaussian, title in _ROBUSTNESS_PANELS:
        ests_here = estimators if plot_gaussian else ["student_t"]
        series_by_est = {
            est: _series_across_pi(aggs, est, key, pi_grid) for est in ests_here
        }
        has_any = any(len(s[0]) > 0 for s in series_by_est.values())
        if not has_any:
            print(f"[WARN] robustness_curve: metric '{key}' absent from the "
                  f"aggregates at T={T}; skipping that panel.")
            continue
        drawable.append((key, ylabel, is_pct, plot_gaussian, title, series_by_est))

    if not drawable:
        print(f"[WARN] robustness_curve: none of the robustness metrics are "
              f"present at T={T}; skipping Figure F.")
        return None

    # ── Draw: panels stacked in a COLUMN, all sharing the same x-axis = pi ──
    # A column makes the three metrics read against one common pi axis (the
    # cleanest way to see "as pi grows, the Gaussian degrades on every front").
    style = {"student_t": (_C_STUDT, "Student-t (robust)"),
             "gaussian":  (_C_GAUSS, "Gaussian (chases outliers)")}
    n = len(drawable)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.1 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (key, ylabel, is_pct, plot_gaussian, title, series_by_est) in zip(axes, drawable):
        for est, (xs, means, lo, hi) in series_by_est.items():
            if not xs:
                continue
            colour, lbl = style[est]
            ax.plot(xs, means, "o-", color=colour, lw=1.6, ms=5, label=lbl)
            # Faint q05–q95 band: this is a RESULTS figure, so showing the
            # across-replication spread (not just the mean) makes the gap's
            # significance legible.  Drawn only when the band is non-degenerate.
            if any(h > l for l, h in zip(lo, hi)):
                ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, lw=0)
        # The detection panel: chance level is lift = 1 (a flagged period is no
        # more likely than random to be a true outlier); the Student-t lift must
        # sit ABOVE it.  For the other panels no such reference line applies.
        if not plot_gaussian:
            ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.7,
                       label="lift = 1 (chance)")
            # Spell out WHY the Gaussian line is missing (nan by design), so the
            # single-line panel is not misread as a plotting bug.
            ax.text(0.99, 0.04, "Gaussian: no down-weighting (nan)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=_C_GAUSS, alpha=0.8)
        if is_pct:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _pos: f"{v:.0%}"))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    # Mark the pi grid explicitly on the (shared) x-axis; pi=0 is the nesting
    # baseline where the two estimators must coincide.
    axes[-1].set_xticks(pi_grid)
    axes[-1].set_xlabel("contamination frequency  pi   "
                        "(pi = 0 -> clean Gaussian DGP = Experiment B)")

    fig.suptitle(
        f"Experiment C — robustness margin  |  T={T}, S={S}\n"
        f"Student-t stays robust while the Gaussian degrades as contamination "
        f"(pi) grows; the GAP between the curves is the robustness margin\n"
        f"(at pi = 0 the curves coincide — nesting, Experiment B — and diverge "
        f"as pi increases)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    # bbox_inches='tight' so the multi-line suptitle is never clipped on save.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved robustness curve (Figure F): {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main:  run_grid over the pi grid  ->  full + trend reporting  ->  figures
# ──────────────────────────────────────────────────────────────────────────────

def main(S: int = 20, full: bool = False, config: str = "small", *,
         fig_seed: int = _DEFAULT_FIG_SEED,
         make_figures: bool = True) -> None:
    # Force UTF-8 on Windows to avoid em-dash encoding errors (A/B idiom).
    try:
        sys.stdout.reconfigure(encoding="utf-8")              # type: ignore
    except Exception:
        pass

    cfg_data     = load_config(config)
    ordered_cols = cfg_data["ORDERED_COLS"]
    block_map    = cfg_data["BLOCK"]
    freq_list    = [cfg_data["FREQ"][c] for c in ordered_cols]
    fit_path = _PROJECT_ROOT / "data" / "processed" / config / "fit_dfm_result.npz"
    print(f"Loading calibrated theta_star from: {fit_path}")
    real_fit   = load_dfm_fit(fit_path)
    theta_star = real_fit["theta"]
    nu_u_A     = float(theta_star["nu_u"])
    nu_eps_A   = float(theta_star["nu_eps"])
    rho_A      = float(np.max(np.abs(np.linalg.eigvals(
        np.asarray(theta_star["A"])))))
    print(f"  theta_star: rho(A)={rho_A:.4f}  nu_u={nu_u_A:.3f}  nu_eps={nu_eps_A:.3f}")

    # ── Build theta_star^C (Gaussian base DGP; contamination via the overlay) ──
    theta_star_C = _build_theta_star_C(theta_star)
    print(f"\ntheta_star^C (Gaussian base DGP; contamination injected via pi):")
    print(f"  nu_u   = {theta_star_C['nu_u']}   (was {nu_u_A:.3f} in theta_star)")
    print(f"  nu_eps = {theta_star_C['nu_eps']}  (was {nu_eps_A:.3f} in theta_star)")
    print(f"  Lambda, A, Q, R, Sigma_0: UNCHANGED")
    print(f"  -> at pi=0 the DGP IS Experiment B; contamination grows with pi")

    # ── Grid configuration ─────────────────────────────────────────────────────
    # T_grid as A/B.  pi_grid includes 0 (= full interpolation from B toward A):
    # the pi=0 column is the uncontaminated Gaussian baseline (reproduces B), and
    # 0.01/0.05/0.10 ramp the additive-outlier intensity.  kappa / nu_contam set
    # the amplitude / heaviness of each injected spike and are propagated through
    # run_grid -> run_monte_carlo -> _mc_worker -> run_one_replication.
    if full:
        T_GRID = [100, 200, 400, 800, 497]
        S_RUN  = 1000
        print(f"\n[FULL RUN] S={S_RUN}, T_grid={T_GRID}  (thesis-quality)")
    else:
        T_GRID = [100, 200, 400, 800, 497]
        S_RUN  = S
        print(f"\n[TEST RUN] S={S_RUN}, T_grid={T_GRID}  (validation; use --full for S=1000)")

    ESTIMATORS = ["student_t", "gaussian"]
    PI_GRID    = [0.0, 0.01, 0.05, 0.10]
    KAPPA      = 5.0
    NU_CONTAM  = 3.0
    OUT_DIR_C  = _PROJECT_ROOT / "output" / "monte_carlo" / config / "expC"

    print(f"\nestimators  : {ESTIMATORS}")
    print(f"pi_grid     : {PI_GRID}  (0 = Experiment B baseline)")
    print(f"kappa       : {KAPPA}   nu_contam : {NU_CONTAM}")
    print(f"output_dir  : {OUT_DIR_C}")
    print(f"config      : {config}")
    print(f"resume      : True  (safe to interrupt and re-run)")

    # ── Run grid ────────────────────────────────────────────────────────────────
    grid_result = run_grid(
        theta_star=theta_star_C,
        S=S_RUN,
        T_grid=T_GRID,
        estimators=ESTIMATORS,
        pi_grid=PI_GRID,
        nu_contam=NU_CONTAM,
        kappa=KAPPA,
        freq_list=freq_list,
        block_map=block_map,
        ordered_cols=ordered_cols,
        output_dir=OUT_DIR_C,
        resume=True,
    )
    print(f"\nrun_grid done: {grid_result['n_computed']} computed, "
          f"{grid_result['n_skipped']} skipped")

    # ── Reporting per T:  full per-pi tables  +  trend table  +  predictions ────
    with _tee_file(OUT_DIR_C / "results.txt"):
        for T in T_GRID:
            # Pre-load every (pi, estimator) aggregate for this T once.
            aggs: dict[tuple[float, str], dict | None] = {}
            for p in PI_GRID:
                for est in ESTIMATORS:
                    aggs[(p, est)] = _load_agg(OUT_DIR_C, est, T, p, S_RUN)

            # (a) FULL table for each pi (everything, student_t vs gaussian).
            for p in PI_GRID:
                agg_st = aggs[(p, "student_t")]
                agg_g  = aggs[(p, "gaussian")]
                if agg_st is None and agg_g is None:
                    print(f"\n[WARN] no scenario files for T={T}, pi={p:.2f}; skipping.")
                    continue
                _full_metric_table(agg_st, agg_g, T=T, S=S_RUN, pi=p,
                                    kappa=KAPPA, nu_contam=NU_CONTAM)

            # (b) TREND table across pi (the headline Experiment-C artefact).
            _trend_table(aggs, T=T, S=S_RUN, pi_grid=PI_GRID)

            # The three predictions, evaluated against this T's numbers.
            _print_predictions(aggs, T=T, pi_grid=PI_GRID)

    # ── FIGURE F: the robustness curve (a RESULTS figure on the aggregates) ─────
    # Generated AFTER run_grid AND AFTER the textual reporting, because it reads
    # the very same saved scenario aggregates (mc_results_expC).  We pin it to
    # the real-panel length T=497 (falling back to the longest computed T if 497
    # is not in the grid), at the same S as the run so the scenario filenames
    # resolve.  plot_robustness_curve is defensive: any missing scenario / metric
    # is skipped with a warning instead of crashing.
    if make_figures:
        T_robust = 497 if 497 in T_GRID else max(T_GRID)
        print("\n" + "=" * 100)
        print(f"  Generating Figure F — robustness curve (RESULTS figure, "
              f"aggregates over S={S_RUN} reps, T={T_robust})")
        print("=" * 100)
        plot_robustness_curve(
            OUT_DIR_C, T=T_robust, pi_grid=PI_GRID, S=S_RUN,
            fig_out_dir=OUT_DIR_C,
        )

    # ── Diagnostic figures (once, fixed seed, separate from the grid) ───────────
    # The five contamination figures illustrate the SAME contaminated DGP used by
    # the grid (Gaussian base theta_star^C + overlay), at the figure-default pi /
    # kappa / nu_contam, from a single fixed seed for exact reproducibility.
    if make_figures:
        print("\n" + "=" * 100)
        print("  Generating the five Experiment-C diagnostic figures "
              "(fixed seed, separate from the grid)")
        print("=" * 100)
        plot_contamination_figures(
            theta_star_C, cfg_data=cfg_data, seed=fig_seed, kappa=KAPPA,
            nu_contam=NU_CONTAM, out_dir=OUT_DIR_C,
        )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _extra(p):
        p.add_argument("--S", type=int, default=20,
                       help="Replications per scenario (default 20 for test)")
        p.add_argument("--full", action="store_true",
                       help="Full run: S=1000, T_grid=[100,200,400,800,497]")
        p.add_argument("--no-figures", action="store_true",
                       help="Skip the five diagnostic figures (grid + report only)")
        p.add_argument("--fig-seed", type=int, default=_DEFAULT_FIG_SEED,
                       help="Master seed for the diagnostic figures")

    args = parse_config_args(
        "Run Experiment C (contamination robustness): "
        "Monte Carlo grid over pi + full/trend reporting + figures",
        extra=_extra,
    )
    S = 1000 if args.full else args.S
    main(S=S, full=args.full, config=args.config,
         fig_seed=args.fig_seed, make_figures=not args.no_figures)
