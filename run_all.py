#!/usr/bin/env python3
"""Produce all weekly forecast results. There are no command-line options.

Run with ``python run_all.py``. The 15 DFM cells and every BVAR model/block
pair run in parallel, followed by deterministic BVAR merges and all outputs.

If one job fails, the other independent jobs continue. Every job has a log in
``output/_logs/run_all/`` and failures are listed at the end.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "output" / "_logs" / "run_all"
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
    return result


def discover(job: Job) -> list[tuple[str, str]]:
    """Read a two-column job list from --list or --list-blocks."""
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
    return rows if result.ok else []


def main() -> int:
    workers = min(MAX_PROCESSES, available_cpus())
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
            ("-m", "tests.forecast.test_windows", "--pre-run")),
        Job("preflight_common_sample",
            ("-m", "tests.forecast.test_common_sample")),
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
    merges = [
        Job(f"merge_bvar_{start}",
            ("scripts/merge_bvar_models.py", "--start", start, "--end", end))
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
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
