"""
src/test_monte_carlo_engine.py
==============================

FAST, STRUCTURED self-test of the Monte Carlo engine (``monte_carlo.py``).

This is the regression harness for the engine machinery — NOT an experiment.
It recovers the engine self-checks that were scattered/lost in the
reorganisation (reproducibility, parallel-vs-serial invariance, resume,
anti-zombie fingerprinting, aggregation) and adds the Experiment-C
contamination path (``pi > 0`` injection + Student-t detection metrics) that
was implemented afterwards.

Design for speed
----------------
Everything runs at ``S = 3`` replications and ``T = 150``–``200`` with a small
EM budget (``max_iter`` ≈ 50, ``tol`` ≈ 1e-4).  We are testing the *plumbing*
(determinism, I/O, resume, key presence, the sign of the detection lift), not
the statistical quality of the estimator, so a coarse fit is sufficient and the
whole suite finishes in a few minutes.

All artefacts go into a DEDICATED TEMPORARY directory
``data/processed/_engine_test/`` which is wiped at the start (so the resume
tests always begin from a clean slate) and removed again at the end.  The real
experiment output dirs (``mc_results_expA`` …) are never touched.

Run
---
    python src/test_monte_carlo_engine.py

Each test prints ``[PASS]`` / ``[FAIL]`` with a one-line message; a final line
reports ``N/M tests passed`` and the process exits non-zero if any failed.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import traceback
from typing import Any, Callable

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config_utils import parse_config_args                      # noqa: E402
from data_loader import load_config                             # noqa: E402
from em_main import load_dfm_fit, _theta_fingerprint            # noqa: E402
from monte_carlo import (                                        # noqa: E402
    run_one_replication,
    run_monte_carlo,
    run_grid,
    aggregate_replications,
    _scenario_filename,
    _load_scenario,
)

# ── Fast-test configuration (small everywhere — NOT an experiment) ────────────
S_FAST     = 3            # replications per scenario
T_FAST     = 150          # panel length for the classic-engine tests
T_DETECT   = 200          # slightly longer for the contamination tests (more
                          # contaminated periods -> steadier detection signal)
MAX_ITER   = 50           # coarse EM budget — we test plumbing, not accuracy
TOL        = 1e-4         # looser outer tolerance -> fewer iterations -> faster
BASE_SEED  = 1000
PI_TEST    = 0.05         # contamination fraction for the single-rep C tests

_TEST_DIR  = _PROJECT_ROOT / "data" / "processed" / "_engine_test"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   Comparison helpers                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _values_equal(a: Any, b: Any) -> bool:
    """Exact equality for metric values (arrays / strings / bool / numbers).

    ``nan == nan`` is treated as ``True`` (so two reproducible runs that both
    produce a NaN for a by-design-undefined metric still compare equal), and
    ``inf`` compares by its IEEE bit-value as usual.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        return aa.shape == bb.shape and bool(np.array_equal(aa, bb, equal_nan=True))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if np.isnan(fa) and np.isnan(fb):
        return True
    return fa == fb


def _metrics_equal(m1: dict, m2: dict) -> tuple[bool, str]:
    """Bit-compare two per-replication metric dicts."""
    if set(m1) != set(m2):
        only1 = set(m1) - set(m2)
        only2 = set(m2) - set(m1)
        return False, f"key sets differ (only in 1: {only1}, only in 2: {only2})"
    for k in m1:
        if not _values_equal(m1[k], m2[k]):
            return False, f"mismatch at '{k}': {m1[k]!r} != {m2[k]!r}"
    return True, ""


def _reps_equal(r1: list, r2: list) -> tuple[bool, str]:
    """Compare two ``per_replication`` lists element-by-element."""
    if len(r1) != len(r2):
        return False, f"length differs ({len(r1)} != {len(r2)})"
    for i, (a, b) in enumerate(zip(r1, r2)):
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            return False, f"None-ness differs at replica {i}"
        ok, msg = _metrics_equal(a, b)
        if not ok:
            return False, f"replica {i}: {msg}"
    return True, ""


def _clean_dir(path: pathlib.Path) -> None:
    """Remove ``path`` (if present) and recreate it empty."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   GROUP A — Classic engine (pi = 0)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def test_A1_reproducibility(ctx: dict) -> str:
    """Same seed twice -> bit-identical per-replication metrics."""
    ok, msg = _reps_equal(ctx["serial_a"]["per_replication"],
                          ctx["serial_b"]["per_replication"])
    assert ok, f"two identical serial runs differ: {msg}"
    return (f"{S_FAST} replicas reproduced bit-for-bit across two runs "
            f"(same base_seed={BASE_SEED}, n_jobs=1)")


def test_A2_parallel_vs_serial(ctx: dict) -> str:
    """n_jobs=1 vs n_jobs>1 -> identical results (seeds are per-replica)."""
    ok, msg = _reps_equal(ctx["serial_a"]["per_replication"],
                          ctx["parallel"]["per_replication"])
    assert ok, f"serial (n_jobs=1) and parallel results differ: {msg}"
    return ("parallel pool (n_jobs=2) reproduced the serial (n_jobs=1) "
            "results exactly — parallelism does not perturb the outcome")


def test_A3_resume(ctx: dict) -> str:
    """Full resume (skip all) and partial resume (recompute one only)."""
    theta = ctx["theta_star"]
    out = _TEST_DIR / "A3_resume"
    _clean_dir(out)

    mc_kw = ctx.get("mc_kw", {})
    grid_kw = dict(
        theta_star=theta, S=S_FAST, T_grid=[T_FAST, T_FAST + 50],
        estimators=["student_t"], pi_grid=[0.0],
        n_jobs=2, base_seed=BASE_SEED, max_iter=MAX_ITER, tol=TOL,
        output_dir=out, resume=True,
        **mc_kw,
    )

    # (1) Cold run — both scenarios computed.
    g1 = run_grid(**grid_kw)
    assert g1["n_computed"] == 2 and g1["n_skipped"] == 0, (
        f"cold run should compute 2 / skip 0; got "
        f"{g1['n_computed']} / {g1['n_skipped']}")

    # (2) Resume — everything cached, nothing recomputed.
    g2 = run_grid(**grid_kw)
    assert g2["n_computed"] == 0 and g2["n_skipped"] == 2, (
        f"resume should compute 0 / skip 2; got "
        f"{g2['n_computed']} / {g2['n_skipped']}")

    # …and the resumed payloads match the cold-run payloads bit-for-bit.
    for s1, s2 in zip(g1["scenarios"], g2["scenarios"]):
        ok, msg = _reps_equal(s1["per_replication"], s2["per_replication"])
        assert ok, f"resumed scenario differs from cold run: {msg}"

    # (3) Partial resume — delete ONE scenario file, only that one recomputes.
    victim = out / _scenario_filename("student_t", T_FAST, 0.0, S_FAST)
    assert victim.exists(), f"expected scenario file missing: {victim.name}"
    victim.unlink()
    g3 = run_grid(**grid_kw)
    assert g3["n_computed"] == 1 and g3["n_skipped"] == 1, (
        f"partial resume should compute 1 / skip 1; got "
        f"{g3['n_computed']} / {g3['n_skipped']}")

    return ("cold=2/0, resume=0/2 (payloads identical), partial=1/1 "
            "after deleting one scenario file")


def test_A4_fingerprint_anti_zombie(ctx: dict) -> str:
    """A changed theta_star invalidates the cache (fingerprint mismatch)."""
    theta = ctx["theta_star"]
    out = _TEST_DIR / "A4_fingerprint"
    _clean_dir(out)

    mc_kw = ctx.get("mc_kw", {})
    grid_kw = dict(
        S=S_FAST, T_grid=[T_FAST], estimators=["student_t"], pi_grid=[0.0],
        n_jobs=1, base_seed=BASE_SEED, max_iter=MAX_ITER, tol=TOL,
        output_dir=out, resume=True,
        **mc_kw,
    )

    # A perturbed DGP: scale A by 0.95 (keeps it stationary, changes the hash).
    theta_pert = {k: np.asarray(v).copy() for k, v in theta.items()}
    theta_pert["A"] = theta_pert["A"] * 0.95
    fp_orig = _theta_fingerprint(theta)
    fp_pert = _theta_fingerprint(theta_pert)
    assert fp_orig != fp_pert, (
        "sanity: perturbing A must change the fingerprint "
        f"(both are {fp_orig})")

    # (1) Cold run with the original theta -> cache stamped with fp_orig.
    g1 = run_grid(theta_star=theta, **grid_kw)
    assert g1["n_computed"] == 1, f"cold run should compute 1; got {g1['n_computed']}"

    # (2) Resume with the PERTURBED theta -> fingerprint mismatch -> recompute
    #     (the anti-zombie guard must refuse the stale cache).
    g2 = run_grid(theta_star=theta_pert, **grid_kw)
    assert g2["n_computed"] == 1 and g2["n_skipped"] == 0, (
        f"stale cache (theta changed) must be recomputed, not skipped; got "
        f"{g2['n_computed']} computed / {g2['n_skipped']} skipped")

    # (3) Resume again with the SAME perturbed theta -> now it skips (the cache
    #     was re-stamped with fp_pert), proving step (2) really recomputed.
    g3 = run_grid(theta_star=theta_pert, **grid_kw)
    assert g3["n_computed"] == 0 and g3["n_skipped"] == 1, (
        f"second resume with the matching theta should skip; got "
        f"{g3['n_computed']} computed / {g3['n_skipped']} skipped")

    return (f"fingerprint {fp_orig}->{fp_pert}: stale cache recomputed (1/0), "
            f"matching cache then skipped (0/1)")


def test_A5_aggregation(ctx: dict) -> str:
    """aggregate_replications exposes the expected stat keys, all finite."""
    agg = aggregate_replications(
        [r for r in ctx["serial_a"]["per_replication"] if r is not None]
    )
    assert agg, "aggregate dict is empty"

    base_keys = {"mean", "std", "median", "q05", "q95"}
    headline = [
        "rho_A_relerr",
        "lambda_relerr_procrustes_blockdiag",
        "lambda_relerr_normalised",
        "factor_abscorr_real", "factor_abscorr_fin", "factor_abscorr_other",
    ]
    for key in headline:
        assert key in agg, f"expected aggregate key '{key}' missing"
        present = base_keys & set(agg[key])
        assert present == base_keys, (
            f"'{key}' missing stat fields: {base_keys - set(agg[key])}")
        assert np.isfinite(agg[key]["mean"]), f"'{key}'.mean is not finite"
        assert np.isfinite(agg[key]["median"]), f"'{key}'.median is not finite"

    return (f"{len(agg)} aggregated metrics; mean/std/median/q05/q95 present "
            f"and finite for {len(headline)} headline keys")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   GROUP B — Contamination path (pi > 0, Experiment C)                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_DETECTION_KEYS = (
    "detection_rate_natural", "detection_lift_natural",
    "detection_rate_5pct",  "detection_precision_5pct",  "detection_lift_5pct",
    "detection_rate_10pct", "detection_precision_10pct", "detection_lift_10pct",
    "detection_n_contam", "detection_T_eff",
)


def test_B1_contamination_runs(ctx: dict) -> str:
    """pi>0 no longer raises; the metrics carry the detection_* keys."""
    m = run_one_replication(
        seed=BASE_SEED, theta_star=ctx["theta_star"], T=T_DETECT,
        estimator="student_t", pi=PI_TEST,
        max_iter=MAX_ITER, tol_outer=TOL,
        **ctx.get("mc_kw", {}),
    )
    missing = [k for k in _DETECTION_KEYS if k not in m]
    assert not missing, f"detection keys missing from metrics: {missing}"
    assert m["detection_n_contam"] > 0, (
        f"expected some contaminated periods at pi={PI_TEST}, T={T_DETECT}; "
        f"got n_contam={m['detection_n_contam']}")
    ctx["B_student_t"] = m          # reuse in B2
    return (f"pi={PI_TEST} ran without error; "
            f"{int(m['detection_n_contam'])} contaminated periods, "
            f"all {len(_DETECTION_KEYS)} detection keys present")


def test_B2_detection_signal(ctx: dict) -> str:
    """Student-t detects the TRUE contaminated periods better than chance."""
    m = ctx.get("B_student_t")
    if m is None:                   # run standalone if B1 was skipped
        m = run_one_replication(
            seed=BASE_SEED, theta_star=ctx["theta_star"], T=T_DETECT,
            estimator="student_t", pi=PI_TEST,
            max_iter=MAX_ITER, tol_outer=TOL,
            **ctx.get("mc_kw", {}),
        )
    rate = m["detection_rate_natural"]
    lift = m["detection_lift_natural"]
    expected_random = float(m["detection_n_contam"]) / float(m["detection_T_eff"])
    print(f"        detection_rate_natural = {rate:.3f}  "
          f"(random baseline {expected_random:.3f})")
    print(f"        detection_lift_natural = {lift:.2f}x  "
          f"detection_rate_5pct = {m['detection_rate_5pct']:.3f}")
    assert np.isfinite(lift) and lift > 1.0, (
        f"Student-t lift must beat chance (>1); got {lift}")
    assert rate > expected_random, (
        f"detection rate {rate:.3f} not above random baseline "
        f"{expected_random:.3f}")
    return (f"Student-t down-weighting locks onto true outliers: "
            f"rate={rate:.2f}, lift={lift:.1f}x (>1)")


def test_B3_gaussian_detection_nan(ctx: dict) -> str:
    """Gaussian estimator: detection keys present but NaN (no down-weighting)."""
    m = run_one_replication(
        seed=BASE_SEED, theta_star=ctx["theta_star"], T=T_DETECT,
        estimator="gaussian", pi=PI_TEST,
        max_iter=MAX_ITER, tol_outer=TOL,
        **ctx.get("mc_kw", {}),
    )
    # The mask exists (pi>0), so the keys are present…
    assert "detection_rate_natural" in m, "detection keys absent for Gaussian"
    assert m["detection_n_contam"] > 0, "Gaussian run had no contaminated periods"
    # …but every rate/precision/lift is NaN by design (constant w_eps_hat -> no
    # ranking to score).
    nan_keys = [
        "detection_rate_natural", "detection_lift_natural",
        "detection_rate_5pct", "detection_precision_5pct", "detection_lift_5pct",
        "detection_rate_10pct", "detection_precision_10pct", "detection_lift_10pct",
    ]
    bad = [k for k in nan_keys if not np.isnan(m[k])]
    assert not bad, f"Gaussian detection metrics should be NaN; non-NaN: {bad}"
    return ("Gaussian: detection keys present, all rate/precision/lift = NaN "
            "by design (constant w_eps_hat, nothing to down-weight)")


def test_B4_pi0_untouched(ctx: dict) -> str:
    """pi=0 (A/B path) produces NO detection keys (contam_mask=None)."""
    m = run_one_replication(
        seed=BASE_SEED, theta_star=ctx["theta_star"], T=T_FAST,
        estimator="student_t", pi=0.0,
        max_iter=MAX_ITER, tol_outer=TOL,
        **ctx.get("mc_kw", {}),
    )
    present = [k for k in _DETECTION_KEYS if k in m]
    assert not present, (
        f"pi=0 must not emit detection keys (A/B path unchanged); "
        f"found {present}")
    return ("pi=0 path unchanged: contam_mask=None -> zero detection keys "
            "(Experiments A/B unaffected)")


def test_B5_pi_trend(ctx: dict) -> str:
    """As pi grows: Student-t lift stays >1, Gaussian Q rel.err inflates."""
    theta = ctx["theta_star"]
    pis = [0.0, 0.10]

    def _q_relerr_mean(agg: dict) -> float:
        vals = [agg[k]["mean"] for k in
                ("diagQ_relerr_real", "diagQ_relerr_fin", "diagQ_relerr_other")
                if k in agg]
        return float(np.mean(vals)) if vals else float("nan")

    st_lifts: dict[float, float] = {}
    g_qrel:   dict[float, float] = {}
    mc_kw = ctx.get("mc_kw", {})
    for pi in pis:
        r_st = run_monte_carlo(
            theta_star=theta, S=S_FAST, T=T_DETECT, estimator="student_t",
            pi=pi, n_jobs=2, base_seed=BASE_SEED, max_iter=MAX_ITER, tol=TOL,
            **mc_kw,
        )
        r_g = run_monte_carlo(
            theta_star=theta, S=S_FAST, T=T_DETECT, estimator="gaussian",
            pi=pi, n_jobs=2, base_seed=BASE_SEED, max_iter=MAX_ITER, tol=TOL,
            **mc_kw,
        )
        st_lift = r_st["aggregates"].get(
            "detection_lift_natural", {}).get("mean", float("nan"))
        st_lifts[pi] = st_lift
        g_qrel[pi] = _q_relerr_mean(r_g["aggregates"])
        print(f"        pi={pi:.2f}:  student_t lift={st_lift if np.isnan(st_lift) else round(st_lift,2)}"
              f"   gaussian Q rel.err={g_qrel[pi]:.3f}")

    # Student-t: lift must stay > 1 at every contaminated pi (pi=0 has no
    # contamination, so its lift is NaN — skip it).
    for pi in pis:
        if pi > 0.0:
            assert np.isfinite(st_lifts[pi]) and st_lifts[pi] > 1.0, (
                f"Student-t lift at pi={pi} should exceed 1; got {st_lifts[pi]}")

    # Gaussian: covariance inflation grows with contamination (it cannot
    # down-weight the spikes), so Q rel.err at the top pi exceeds pi=0.
    assert g_qrel[pis[-1]] > g_qrel[pis[0]], (
        f"Gaussian Q rel.err should grow with contamination; "
        f"pi={pis[0]}:{g_qrel[pis[0]]:.3f} -> pi={pis[-1]}:{g_qrel[pis[-1]]:.3f}")

    return (f"Student-t lift stays >1 across pi {pis}; Gaussian Q rel.err "
            f"inflates {g_qrel[pis[0]]:.3f} -> {g_qrel[pis[-1]]:.3f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   Harness                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _run(name: str, fn: Callable[[dict], str], ctx: dict,
         results: list[tuple[str, bool]]) -> None:
    """Execute one test, print [PASS]/[FAIL], record the outcome."""
    print(f"\n>>> {name}")
    try:
        msg = fn(ctx)
        print(f"    [PASS] {msg}")
        results.append((name, True))
    except AssertionError as exc:
        print(f"    [FAIL] {exc}")
        results.append((name, False))
    except Exception as exc:                                    # noqa: BLE001
        print(f"    [FAIL] unexpected {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results.append((name, False))


def main(config: str = "small") -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")                # type: ignore[attr-defined]
    except Exception:
        pass

    bar = "=" * 78
    print(bar)
    print("  MONTE CARLO ENGINE — FAST SELF-TEST  (monte_carlo.py)")
    print(f"  S={S_FAST}, T={T_FAST}/{T_DETECT}, max_iter={MAX_ITER}, tol={TOL}")
    print(f"  config={config}")
    print(bar)

    # ── Fresh temp workspace (resume tests must start from zero) ──────────────
    _clean_dir(_TEST_DIR)

    cfg_data  = load_config(config)
    mc_kw: dict = {
        "ordered_cols": cfg_data["ORDERED_COLS"],
        "block_map":    cfg_data["BLOCK"],
        "freq_list":    [cfg_data["FREQ"][c] for c in cfg_data["ORDERED_COLS"]],
    }
    fit_path = _PROJECT_ROOT / "data" / "processed" / config / "fit_dfm_result.npz"
    print(f"Loading calibrated theta_star from: {fit_path}")
    theta_star = load_dfm_fit(fit_path)["theta"]

    # ── Shared runs for Group A (computed once, reused across A1/A2/A5) ────────
    print("\nPre-computing shared Group-A runs (3 x run_monte_carlo, S=%d) ..."
          % S_FAST)
    common = dict(theta_star=theta_star, S=S_FAST, T=T_FAST,
                  estimator="student_t", pi=0.0,
                  base_seed=BASE_SEED, max_iter=MAX_ITER, tol=TOL,
                  **mc_kw)
    serial_a = run_monte_carlo(n_jobs=1, **common)
    serial_b = run_monte_carlo(n_jobs=1, **common)
    parallel = run_monte_carlo(n_jobs=2, **common)

    ctx: dict[str, Any] = {
        "theta_star": theta_star,
        "mc_kw":      mc_kw,
        "serial_a":   serial_a,
        "serial_b":   serial_b,
        "parallel":   parallel,
    }

    tests: list[tuple[str, Callable[[dict], str]]] = [
        ("A1  reproducibility (same seed -> identical)", test_A1_reproducibility),
        ("A2  parallel vs serial (n_jobs invariant)",    test_A2_parallel_vs_serial),
        ("A3  resume (full skip + partial recompute)",   test_A3_resume),
        ("A4  fingerprint anti-zombie (stale cache)",    test_A4_fingerprint_anti_zombie),
        ("A5  aggregation (stat keys finite)",           test_A5_aggregation),
        ("B1  pi>0 runs, detection keys present",        test_B1_contamination_runs),
        ("B2  Student-t detection beats chance",         test_B2_detection_signal),
        ("B3  Gaussian detection = NaN by design",       test_B3_gaussian_detection_nan),
        ("B4  pi=0 leaves A/B path untouched",           test_B4_pi0_untouched),
        ("B5  pi trend (lift>1 / Gaussian Q inflates)",  test_B5_pi_trend),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in tests:
        _run(name, fn, ctx, results)

    # ── Cleanup the temporary workspace ───────────────────────────────────────
    shutil.rmtree(_TEST_DIR, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(ok for _, ok in results)
    n_tot  = len(results)
    print("\n" + bar)
    print(f"  SUMMARY: {n_pass}/{n_tot} tests passed")
    if n_pass != n_tot:
        for name, ok in results:
            if not ok:
                print(f"    FAILED: {name}")
    print(bar)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    args = parse_config_args("Monte Carlo Engine — Fast Self-Test")
    sys.exit(main(config=args.config))
