"""Regression guards for model sharding and L-BVAR numerical redraws."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np

from core import output_layout as layout
from core.bvar import evaluate, lbvar, simsmoother
from core.forecast.weekly_nowcast import COLUMNS


START = END = "2008-01-04"


def _row(spec: str, value: float, variant: str = "-") -> dict:
    row = {column: np.nan for column in COLUMNS}
    row.update({"as_of": START, "target_quarter": "2008Q1", "spec": spec,
                "variant": variant, "nowcast_bea": value,
                "converged": True, "reestimated": True})
    return row


def _ls(spec: str, value: float, variant: str = "-") -> dict:
    return {"as_of": START, "target_quarter": "2008Q1", "horizon_week": -8,
            "spec": spec, "variant": variant, "log_score": value,
            "realizzato_bea": 1.0, "n_draws": 10, "reestimated": True}


def check_merge_is_verbatim() -> None:
    with tempfile.TemporaryDirectory(prefix="bvar_model_merge_") as tmp:
        roots = {}
        rows, quants, logs = [], {}, []
        for i, model in enumerate(layout.BVAR_MODELS):
            root = os.path.join(tmp, "shards", model)
            roots[model] = root
            model_rows = [_row(model, i + 0.125,
                               "authors" if model == "cbvar" else "-")]
            if model == "qbvar":
                model_rows.append(_row(evaluate.BENCHMARK_SPEC, 9.125, "ar2"))
            model_quants = {(START, model, "2008Q1"):
                            np.arange(len(evaluate.QUANTILES), dtype=float) + i}
            model_logs = [_ls(model, -i - 0.25,
                              "authors" if model == "cbvar" else "-")]
            evaluate._persist(evaluate._paths(root, START, END), model_rows,
                              model_quants, model_logs)
            paths = evaluate._paths(root, START, END)
            os.makedirs(paths["weeks"], exist_ok=True)
            open(evaluate._week_marker(paths, START), "w").close()
            rows.extend(model_rows)
            quants.update(model_quants)
            logs.extend(model_logs)

        reference = os.path.join(tmp, "reference")
        merged = os.path.join(tmp, "merged")
        evaluate._persist(evaluate._paths(reference, START, END),
                          rows, quants, logs)
        evaluate.merge_model_runs(START, END, roots, output_root=merged)

        ref_paths = evaluate._paths(reference, START, END)
        out_paths = evaluate._paths(merged, START, END)
        for artifact in ("csv", "npz", "ls"):
            with (open(ref_paths[artifact], "rb") as left,
                  open(out_paths[artifact], "rb") as right):
                assert left.read() == right.read(), artifact


def check_nonfinite_draw_is_replaced() -> None:
    state = SimpleNamespace(B=np.array([[0.0]]), Sigma=np.eye(1))
    original_build = lbvar.build_state_space
    original_smoother = lbvar.simulation_smoother
    original_redraw = lbvar.draw_parameters

    lbvar.build_state_space = lambda B, Sigma, n, p, a0: float(B[0, 0])
    lbvar.simulation_smoother = lambda ss, Y, rng, **kw: (
        np.full((2, 1), np.nan) if ss == 0.0 else np.ones((2, 1)))

    def redraw(current, rng):
        current.B = np.array([[1.0]])
        return current

    lbvar.draw_parameters = redraw
    try:
        alpha, retries = lbvar._finite_smoother(
            state, np.zeros((2, 1)), np.random.default_rng(1),
            n=1, p=1, a0=np.zeros(1))
    finally:
        lbvar.build_state_space = original_build
        lbvar.simulation_smoother = original_smoother
        lbvar.draw_parameters = original_redraw

    assert retries == 1
    assert np.isfinite(alpha).all()
    assert state.B[0, 0] == 1.0


def check_failed_sweep_preserves_last_valid_state() -> None:
    state = SimpleNamespace(B=np.array([[1.0]]), Sigma=np.eye(1))
    original_step = lbvar.step
    original_finite = lbvar._finite_smoother

    def mutate_candidate(candidate, rng, **kw):
        candidate.B[0, 0] = 99.0
        return candidate

    lbvar.step = mutate_candidate
    lbvar._finite_smoother = lambda *a, **kw: (_ for _ in ()).throw(
        FloatingPointError("synthetic overflow"))
    try:
        returned, alpha, _, accepted = lbvar._candidate_transition(
            state, np.zeros((2, 1)), np.zeros((2, 1)),
            np.random.default_rng(2), n=1, p=1, a0=np.zeros(1), n_metro=1)
    finally:
        lbvar.step = original_step
        lbvar._finite_smoother = original_finite

    assert not accepted and alpha is None
    assert returned is state
    assert state.B[0, 0] == 1.0


def check_catastrophic_cancellation_is_rejected() -> None:
    ss = simsmoother.LinearGaussianSS(
        A=np.zeros((1, 1)), Q=np.eye(1), Z=np.eye(1), R=np.eye(1))
    original_simulate = simsmoother.simulate_forward
    original_forward = simsmoother.forward_pass
    original_mean = simsmoother.smoothed_mean

    simsmoother.simulate_forward = lambda *a, **kw: (
        np.full((2, 1), 1e20), np.zeros((2, 1)))
    simsmoother.forward_pass = lambda *a, **kw: object()
    simsmoother.smoothed_mean = lambda *a, **kw: np.full((2, 1), -1e20)
    rejected = False
    try:
        simsmoother.simulation_smoother(
            ss, np.ones((2, 1)), np.random.default_rng(3),
            numerical_guard=True)
    except FloatingPointError:
        rejected = True
    finally:
        simsmoother.simulate_forward = original_simulate
        simsmoother.forward_pass = original_forward
        simsmoother.smoothed_mean = original_mean

    assert rejected


def main() -> None:
    check_merge_is_verbatim()
    check_nonfinite_draw_is_replaced()
    check_failed_sweep_preserves_last_valid_state()
    check_catastrophic_cancellation_is_rejected()
    print("  OK  serial artifacts == model shards + merge, byte for byte")
    print("  OK  a non-finite L-BVAR parameter draw is replaced conditionally")
    print("  OK  a failed numerical sweep preserves the last valid chain state")
    print("  OK  catastrophic DK cancellation is rejected before contaminating MCMC")


if __name__ == "__main__":
    main()
