#!/usr/bin/env python3
"""Produce all weekly forecast results. There are no command-line options.

Run with ``python run_all.py``. The 15 DFM cells and every BVAR model/block
pair run in parallel, followed by deterministic BVAR merges and all outputs.

If one job fails, the other independent jobs continue. Every job has a log in
``output/_logs/run_all/`` and failures are listed at the end.
"""

from __future__ import annotations

import concurrent.futures
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "output" / "_logs" / "run_all"
# One row per job: what it cost and whether it held. It sits with the results
# and not with the logs because it is their provenance -- `output/_logs*/` is
# gitignored, `output/forecast_weekly/` is not. The per-job timings are
# otherwise printed only on this process's stdout, so a closed terminal loses
# them while the per-job logs survive.
TIMES_CSV = ROOT / "output" / "forecast_weekly" / "run_times.csv"
TIMES_HEADER = ("job", "ok", "returncode", "minutes", "detail", "log")
START = "2007-01-01"
END = "2025-12-31"
MAX_PROCESSES = 224

# The codebase's measured policy is parallel processes with one numerical
# thread each. More BLAS threads make the optimization-heavy jobs slower.
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "WEEKLY_SAVE_EVERY_ROWS": "100",
    "WEEKLY_SAVE_EVERY_SECONDS": "120",
}


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]


@dataclass
class Result:
    name: str
    ok: bool
    returncode: int | None
    seconds: float
    detail: str = ""


results: list[Result] = []
print_lock = threading.Lock()


def record(result: Result) -> None:
    """Appende una riga a `TIMES_CSV`.  Va chiamata sotto `print_lock`.

    Si appende job per job invece di scrivere tutto in fondo: una passata dura
    ore, e se muore a meta' i tempi dei job finiti sono comunque su disco.
    """
    try:
        with TIMES_CSV.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                (result.name, result.ok, result.returncode,
                 round(result.seconds / 60.0, 2), result.detail,
                 (LOG_DIR / f"{result.name}.log").relative_to(ROOT)))
    except OSError as error:
        # Il registro dei tempi non deve poter fermare una passata da ore.
        print(f"  [attenzione] run_times.csv non scritto: {error}", flush=True)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(CHILD_ENV)
    return env


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run(job: Job) -> Result:
    """Run one child process, log it, and return instead of raising."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{job.name}.log"
    started = time.perf_counter()

    with print_lock:
        print(f"  START   {job.name}", flush=True)
    try:
        with log.open("w", encoding="utf-8", errors="replace") as stream:
            completed = subprocess.run(
                [sys.executable, *job.command], cwd=ROOT, env=environment(),
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        returncode = completed.returncode
        ok = returncode == 0
        detail = "" if ok else f"exit code {returncode}"
    except Exception as error:
        returncode = None
        ok = False
        detail = f"{type(error).__name__}: {error}"
        log.write_text(detail + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    result = Result(job.name, ok, returncode, elapsed, detail)
    results.append(result)
    with print_lock:
        state = "OK" if ok else "FAILED"
        print(f"  {state:7s} {job.name} ({elapsed / 60:.1f} min)", flush=True)
        record(result)
    return result


def discover(job: Job) -> list[tuple[str, str]]:
    """Read a two-column job list from --list or --list-blocks.

    One-column lines are ignored on purpose: ``run_dfm.py --list`` ends with a
    bare ``benchmark`` line, which is a job without a spec or a variant and so
    has no second column to give.  It is added explicitly in ``main`` instead of
    being parsed here -- silently dropping it would leave the run without AR(2)
    and the expanding mean, and nothing downstream would say so.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, *job.command], cwd=ROOT, env=environment(),
        capture_output=True, text=True, check=False)
    (LOG_DIR / f"{job.name}.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8")
    rows = [tuple(line.split()) for line in completed.stdout.splitlines()
            if len(line.split()) == 2]
    result = Result(job.name, completed.returncode == 0 and bool(rows),
                    completed.returncode, 0.0,
                    f"{len(rows)} jobs" if rows else "no jobs discovered")
    results.append(result)
    print(f"  {job.name}: {result.detail}")
    record(result)
    return rows if result.ok else []


def main() -> int:
    workers = min(MAX_PROCESSES, available_cpus())
    # Si riparte da un file vuoto: senza, una seconda passata si appenderebbe
    # alla prima e le due sarebbero indistinguibili riga per riga.
    TIMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TIMES_CSV.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(TIMES_HEADER)
    print("=" * 78)
    print("WEEKLY DFM + BVAR: ALL RESULTS")
    print(f"period={START}..{END}, process slots={workers}, threads/process=1")
    print("A failed job is reported but does not stop independent jobs.")
    print("=" * 78)

    # Cheap correctness guards before paying for the estimates. Their failure
    # is recorded but, like every other failure, does not abort independent work.
    print("\nPREFLIGHT")
    for job in (
        Job("preflight_windows",
            ("-m", "src.forecast.test_windows", "--pre-run")),
        Job("preflight_common_sample",
            ("-m", "src.forecast.test_common_sample")),
    ):
        run(job)

    print("\nDISCOVERY")
    cells = discover(Job("discover_dfm_cells",
                         ("scripts/run_dfm.py", "--list")))
    blocks = discover(Job(
        "discover_bvar_blocks",
        ("scripts/run_bvar.py", "--start", START, "--end", END,
         "--list-blocks")))

    jobs = [
        Job(f"dfm_{spec}_{variant}",
            ("scripts/run_dfm.py", "--spec", spec, "--variant", variant,
             "--start", START, "--end", END))
        for spec, variant in cells
    ]
    # I benchmark sono la sedicesima unita' del DFM, e non arrivano da
    # `discover`: la loro riga ha un campo solo (vedi il docstring li' sopra).
    # Vanno prima delle celle perche' non fanno EM e finiscono in minuti,
    # liberando subito lo slot.
    jobs.insert(0, Job("dfm_benchmark",
                       ("scripts/run_dfm.py", "--benchmark",
                        "--start", START, "--end", END)))
    # Longest jobs first minimizes the tail once fewer than 224 jobs remain.
    for model in ("lbvar", "bbvar", "cbvar", "qbvar"):
        for start, end in blocks:
            shard = ROOT / "output" / "_bvar_shards" / f"{start}_{end}" / model
            command = ["scripts/run_bvar.py", "--start", start, "--end", end,
                       "--models", model, "--output-root", str(shard)]
            if model != "qbvar":
                command.append("--no-benchmarks")
            jobs.append(Job(f"bvar_{model}_{start}", tuple(command)))

    print(f"\nESTIMATION: {len(jobs)} independent jobs")
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, jobs))

    print(f"\nBVAR MERGE: {len(blocks)} blocks")
    # `--allow-missing` e' coerente con la regola di questo file: "un lavoro
    # fallito viene riportato ma non ferma gli indipendenti".  Senza, un solo
    # modello morto si porterebbe via anche i tre che hanno finito, e la loro
    # riga di fallimento c'e' gia' nel riepilogo qui sotto.
    merges = [
        Job(f"merge_bvar_{start}",
            ("scripts/merge_bvar_models.py", "--start", start, "--end", end,
             "--allow-missing"))
        for start, end in blocks
    ]
    if merges:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, merges))

    # Always attempt post-processing. If estimates are missing, its existing
    # guards explain the problem in forecast_outputs.log.
    print("\nOUTPUTS")
    run(Job("forecast_outputs", ("scripts/run_outputs.py",)))

    failed = [result for result in results if not result.ok]
    print("\n" + "=" * 78)
    print(f"COMPLETE: {len(results) - len(failed)} ok, {len(failed)} failed")
    if failed:
        print("\nFAILED JOBS (all independent jobs were allowed to finish):")
        for result in failed:
            print(f"  - {result.name}: {result.detail}; "
                  f"log: {LOG_DIR / (result.name + '.log')}")
    print(f"\nAll logs: {LOG_DIR}")
    print(f"Times:    {TIMES_CSV}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
