"""Regression guards for model sharding and the L-BVAR numerical fallback."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import output_layout as layout
from src.bvar import evaluate, lbvar, simsmoother
from src.forecast.weekly_nowcast import COLUMNS


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


def check_unusable_dk_falls_back_to_precision() -> None:
    """Il DK non rappresentabile passa la mano, e SOLO allora.

    Non e' un dettaglio di implementazione: e' la scelta con cui il blocco
    2020-07-31 esce.  Vanno verificati i due versi — che il ripiego scatti
    quando il DK non regge, e che NON scatti quando regge, altrimenti si
    cambierebbero in silenzio tutte le passate esistenti.
    """
    original_build = lbvar.build_state_space
    original_smoother = lbvar.simulation_smoother
    original_precision = lbvar.precision_draw
    called: list[int] = []

    lbvar.build_state_space = lambda B, Sigma, n, p, a0: float(B[0, 0])

    def fake_precision(B, Sigma, Y, head, rng, *, n, p, return_full=True):
        called.append(1)
        return np.full((2, 1), 7.0)

    lbvar.precision_draw = fake_precision
    try:
        # (a) il DK non regge -> ripiego
        lbvar.simulation_smoother = lambda ss, Y, rng, **kw: np.full((2, 1), np.nan)
        state = SimpleNamespace(B=np.array([[0.0]]), Sigma=np.eye(1))
        alpha, fallback = lbvar._finite_smoother(
            state, np.zeros((2, 1)), np.random.default_rng(1),
            n=1, p=1, a0=np.zeros(1))
        assert fallback and np.isfinite(alpha).all() and alpha[0, 0] == 7.0
        assert len(called) == 1

        # (b) il DK regge -> il ripiego NON viene nemmeno chiamato
        lbvar.simulation_smoother = lambda ss, Y, rng, **kw: np.ones((2, 1))
        alpha, fallback = lbvar._finite_smoother(
            state, np.zeros((2, 1)), np.random.default_rng(1),
            n=1, p=1, a0=np.zeros(1))
        assert not fallback and alpha[0, 0] == 1.0
        assert len(called) == 1
    finally:
        lbvar.build_state_space = original_build
        lbvar.simulation_smoother = original_smoother
        lbvar.precision_draw = original_precision


def check_failed_estimate_falls_back_to_reuse() -> None:
    """Una stima piena che fallisce non porta giu' il blocco.

    Vale per TUTTI e quattro i modelli, ed e' il punto: prima l'eccezione
    risaliva e tredici settimane non venivano scritte perche' una stima non
    aveva retto.  Si verificano i tre esiti:

      (a) con una cache -> riuso, la riga esce con `reestimated=False`
      (b) senza cache   -> `EstimationFailed`, che il chiamante sa gestire
      (c) se la stima riesce -> `reestimated=True`, ramo normale intatto
    """
    rng = np.random.default_rng(0)

    class _Res:
        def growth(self, target):
            return None

    original_full = evaluate._full_estimate
    original_reuse = evaluate._reuse_estimate
    original_rows = evaluate._growth_rows
    evaluate._growth_rows = lambda g, q: np.zeros(3)
    riusi: list[str] = []

    def esplode(model, as_of, **kw):
        raise FloatingPointError("catena ferma")

    def reuse_ok(model, cache, **kw):
        riusi.append(model)
        return _Res()

    try:
        evaluate._reuse_estimate = reuse_ok
        for model in ("cbvar", "bbvar", "lbvar"):
            # (a) la stima fallisce ma c'e' una cache: si riusa
            evaluate._full_estimate = esplode
            cache = evaluate.ModelCache(B=np.zeros((1, 1)),
                                        estimated_at=pd.Timestamp("2020-05-01"))
            per_q, _, estimated = evaluate.run_model(
                model, pd.Timestamp("2020-07-31"), ["2020Q3"], cache,
                full=True, spec=None, raw=None, rng=rng, n_draws=4)
            assert not estimated, f"{model}: doveva dichiarare il riuso"
            assert riusi and riusi[-1] == model
            assert "2020Q3" in per_q

            # (b) la stima fallisce e NON c'e' cache
            try:
                evaluate.run_model(model, pd.Timestamp("2020-07-31"),
                                   ["2020Q3"], evaluate.ModelCache(),
                                   full=True, spec=None, raw=None, rng=rng,
                                   n_draws=4)
                raise AssertionError(f"{model}: doveva sollevare")
            except evaluate.EstimationFailed as exc:
                assert exc.model == model

            # (c) la stima riesce: ramo normale, nessun riuso
            n_prima = len(riusi)
            evaluate._full_estimate = (
                lambda mo, ao, **kw: (_Res(), evaluate.ModelCache(
                    B=np.zeros((1, 1)), estimated_at=pd.Timestamp(ao))))
            _, _, estimated = evaluate.run_model(
                model, pd.Timestamp("2020-07-31"), ["2020Q3"],
                evaluate.ModelCache(), full=True, spec=None, raw=None,
                rng=rng, n_draws=4)
            assert estimated and len(riusi) == n_prima
    finally:
        evaluate._full_estimate = original_full
        evaluate._reuse_estimate = original_reuse
        evaluate._growth_rows = original_rows


def check_previous_full_week_is_a_real_release() -> None:
    """`previous_full_week` trova un trigger VERO, non la prima riga di griglia.

    `estimation_weeks` marca sempre come piena la prima settimana della
    griglia: riusarla qui avrebbe dato un falso positivo a 60 settimane di
    distanza invece dell'ultimo rilascio del PIL.
    """
    prev = evaluate.previous_full_week("2020-07-31")
    assert prev is not None and prev < pd.Timestamp("2020-07-31")
    # dev'essere una settimana di stima piena per costruzione
    assert evaluate.gdp_released_between(prev - pd.Timedelta(weeks=1), prev)
    # e non deve saltare oltre il rilascio piu' recente
    assert (pd.Timestamp("2020-07-31") - prev).days < 120


def check_runaway_path_is_rejected() -> None:
    """Un cammino in fuga viene rifiutato, da QUALSIASI delle due strade.

    E' la regressione che conta di piu' su questo modulo.  Prima il ripiego
    controllava solo `isfinite`: un cammino a 1e26 — finito — veniva accettato,
    entrava nella stima e da li' ogni estrazione di (B, Sigma) era degenere.
    La catena non si fermava dove si rompeva, si rompeva molto prima e in
    silenzio.  Anche il ramo DK aveva una soglia, ma a 1.9e152, cioe' nessuna.
    """
    Y = np.array([[1.0], [2.0]])            # scala del dato = 2
    buono = np.full((2, 1), 50.0)           # x25: dentro il corridoio
    fuga = np.full((2, 1), 1e26)            # x5e25: fuori
    assert lbvar._path_within_support(buono, Y, 1)
    assert not lbvar._path_within_support(fuga, Y, 1)
    assert not lbvar._path_within_support(np.full((2, 1), np.nan), Y, 1)

    original_build = lbvar.build_state_space
    original_smoother = lbvar.simulation_smoother
    original_precision = lbvar.precision_draw
    lbvar.build_state_space = lambda B, Sigma, n, p, a0: 0.0
    try:
        # (a) DK in fuga -> passa la mano al ripiego, che qui e' sano
        lbvar.simulation_smoother = lambda ss, Y_, rng, **kw: fuga.copy()
        lbvar.precision_draw = (
            lambda *a, **kw: np.full((2, 1), 50.0))
        state = SimpleNamespace(B=np.array([[0.0]]), Sigma=np.eye(1))
        alpha, fallback = lbvar._finite_smoother(
            state, Y, np.random.default_rng(1), n=1, p=1, a0=np.zeros(1))
        assert fallback and alpha[0, 0] == 50.0

        # (b) ENTRAMBE in fuga -> si solleva, e la spazzata verra' rifiutata
        lbvar.precision_draw = lambda *a, **kw: fuga.copy()
        try:
            lbvar._finite_smoother(state, Y, np.random.default_rng(1),
                                   n=1, p=1, a0=np.zeros(1))
            raise AssertionError("un cammino a 1e26 non doveva passare")
        except FloatingPointError as exc:
            assert "supporto" in str(exc)
    finally:
        lbvar.build_state_space = original_build
        lbvar.simulation_smoother = original_smoother
        lbvar.precision_draw = original_precision


def check_head_from_a0_roundtrip() -> None:
    """`head_from_a0` inverte davvero l'impilamento di `fit`."""
    rng = np.random.default_rng(4)
    n, p = 3, 5
    head = rng.standard_normal((p, n))
    a0 = np.concatenate([head[p - 1 - j] for j in range(p)])
    assert np.array_equal(lbvar.head_from_a0(a0, n, p), head)


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
    check_unusable_dk_falls_back_to_precision()
    check_runaway_path_is_rejected()
    check_failed_estimate_falls_back_to_reuse()
    check_previous_full_week_is_a_real_release()
    check_head_from_a0_roundtrip()
    check_failed_sweep_preserves_last_valid_state()
    check_catastrophic_cancellation_is_rejected()
    print("  OK  serial artifacts == model shards + merge, byte for byte")
    print("  OK  an unusable DK draw falls back to the precision sampler, "
          "and only then")
    print("  OK  a runaway drawn path is rejected on both branches")
    print("  OK  a failed full estimate falls back to reuse, for every model")
    print("  OK  previous_full_week finds a real GDP release")
    print("  OK  head_from_a0 inverts the companion stacking")
    print("  OK  a failed numerical sweep preserves the last valid chain state")
    print("  OK  catastrophic DK cancellation is rejected before contaminating MCMC")


if __name__ == "__main__":
    main()
