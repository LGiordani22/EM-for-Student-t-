#!/usr/bin/env python3
"""Live, read-only dashboard for ``python run_all.py``.

Run from the repository root with ``python check_progress.py``. It refreshes
automatically; scroll with the mouse wheel, arrows, or Page Up/Down, and stop
with ``q`` or Ctrl-C. When redirected to a file it prints one snapshot and exits.
"""

from __future__ import annotations

from collections import defaultdict, deque
import bisect
import csv
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "output" / "_logs" / "run_all"
SHARDS = ROOT / "output" / "_bvar_shards"
CELLS_ROOT = ROOT / "output" / "forecast_weekly" / "csv" / "_cells"
START, END = date(2007, 1, 1), date(2025, 12, 31)
MODELS = ("lbvar", "bbvar", "cbvar", "qbvar")
REFRESH_SECONDS = 10
FAILURE = re.compile(r"^Traceback|MemoryError|FloatingPointError|No space left|Killed",
                     re.MULTILINE)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def discovered_blocks() -> list[tuple[str, str]]:
    rows = []
    for line in read_text(LOGS / "discover_bvar_blocks.log").splitlines():
        fields = line.split()
        if len(fields) == 2 and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", x)
                                    for x in fields):
            rows.append((fields[0], fields[1]))
    return rows


#: Il lavoro dei benchmark: una cella per il pool, ma senza spec ne' variante.
#: `run_dfm.py --list` lo stampa come una riga di UN campo solo, che percio' non
#: passa il filtro a due colonne qui sotto -- e prima di questa riga spariva dal
#: cruscotto, che mostrava quindici lavori invece di sedici.  Si rappresenta con
#: la variante vuota: `cell_dir` e `cell_log` sanno che vuol dire.
BENCHMARK_JOB = "benchmark"


def discovered_cells() -> list[tuple[str, str]]:
    rows = []
    for line in read_text(LOGS / "discover_dfm_cells.log").splitlines():
        fields = line.split()
        if len(fields) == 2:
            rows.append((fields[0], fields[1]))
        elif len(fields) == 1 and fields[0] == BENCHMARK_JOB:
            rows.append((BENCHMARK_JOB, ""))
    return rows


def cell_dir(spec: str, variant: str) -> Path:
    """La cartella della cella, o quella dei benchmark se la variante e' vuota."""
    return CELLS_ROOT / (spec if not variant else f"{spec}_{variant}")


def cell_log(spec: str, variant: str) -> Path:
    return LOGS / (f"dfm_{spec}.log" if not variant
                   else f"dfm_{spec}_{variant}.log")


def cell_label(spec: str, variant: str) -> str:
    return spec if not variant else f"{spec}/{variant}"


def fridays(start: date, end: date) -> list[date]:
    first = start + timedelta(days=(4 - start.weekday()) % 7)
    out = []
    day = first
    while day <= end:
        out.append(day)
        day += timedelta(days=7)
    return out


ALL_FRIDAYS = fridays(START, END)
_CSV_CACHE: dict[Path, tuple[int, int, date | None]] = {}
_THETA_CACHE: dict[Path, tuple[int, date | None]] = {}


def failed(path: Path) -> bool:
    return bool(FAILURE.search(read_text(path)))


def latest_dfm_date(spec: str, variant: str) -> date | None:
    latest = None
    csv_path = cell_dir(spec, variant) / f"weekly_nowcast_{START}_{END}.csv"
    try:
        stat = csv_path.stat()
        cached = _CSV_CACHE.get(csv_path)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            latest = cached[2]
        else:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    # I benchmark sono due varianti (ar2, mean) sotto la stessa
                    # pseudo-spec: si filtra sulla sola spec, o non passerebbe
                    # nessuna riga.
                    if row.get("spec") != spec:
                        continue
                    if variant and row.get("variant") != variant:
                        continue
                    try:
                        value = date.fromisoformat(row["as_of"][:10])
                    except (KeyError, TypeError, ValueError):
                        continue
                    latest = value if latest is None or value > latest else latest
            _CSV_CACHE[csv_path] = (stat.st_mtime_ns, stat.st_size, latest)
    except OSError:
        pass

    # I benchmark non fanno EM e non hanno una cartella `theta/`: l'OSError qui
    # sotto e' la loro strada normale, non un guasto.
    theta = cell_dir(spec, variant) / "theta"
    try:
        stamp = theta.stat().st_mtime_ns
        cached = _THETA_CACHE.get(theta)
        if cached and cached[0] == stamp:
            theta_latest = cached[1]
        else:
            theta_latest = None
            for path in theta.glob("theta_*.npz"):
                try:
                    value = date.fromisoformat(path.stem.removeprefix("theta_")[:10])
                except ValueError:
                    continue
                theta_latest = (value if theta_latest is None or value > theta_latest
                                else theta_latest)
            _THETA_CACHE[theta] = (stamp, theta_latest)
        if theta_latest is not None:
            latest = theta_latest if latest is None or theta_latest > latest else latest
    except OSError:
        pass
    return latest


def dfm_fraction(latest: date | None) -> float:
    if latest is None:
        return 0.0
    return min(1.0, bisect.bisect_right(ALL_FRIDAYS, latest) / len(ALL_FRIDAYS))


def run_start() -> float | None:
    manifests = list(SHARDS.glob("*/*/_checkpoint/bvar/*/manifest.json"))
    stamps = []
    for path in manifests:
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            pass
    if stamps:
        return min(stamps)
    discovery = [LOGS / "discover_dfm_cells.log", LOGS / "discover_bvar_blocks.log"]
    stamps = [p.stat().st_mtime for p in discovery if p.exists()]
    return min(stamps) if stamps else None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not (seconds < float("inf")):
        return "waiting for data"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def checkpoint_eta(stamps: list[float], done: int, total: int,
                   now: float) -> float | None:
    """Bootstrap a family ETA from marker timestamps already on disk."""
    # Ten events are enough to avoid the one-marker absurdity while keeping
    # the bootstrap responsive as many parallel shards begin finishing.
    recent = sorted(stamps)[-10:]
    if len(recent) < 10:
        return None
    span = now - recent[0]                 # includes silence since last marker
    if span < 30:
        return None
    rate = (len(recent) - 1) / span
    return None if rate <= 0 else max(0, total - done) / rate


def bar(value: float, width: int = 24) -> str:
    value = min(1.0, max(0.0, value))
    filled = int(value * width)
    return "█" * filled + "·" * (width - filled)


def memory() -> tuple[int, int] | None:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.split()[0]) * 1024
        return values["MemTotal"], values["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return None


class Rates:
    def __init__(self) -> None:
        self.history: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=240))

    def eta(self, name: str, done: float, total: float, now: float,
            started: float | None, *, min_change: float = 0.0) -> float | None:
        points = self.history[name]
        points.append((now, done))
        rate = None
        for old_t, old_done in points:
            if now - old_t >= 30 and done - old_done >= min_change:
                rate = (done - old_done) / (now - old_t)
                break
        return None if not rate else max(0.0, total - done) / rate


def snapshot(rates: Rates) -> str:
    now = time.time()
    started = run_start()
    blocks = discovered_blocks()
    cells = discovered_cells()

    family = {m: {"started": 0, "complete": 0, "failed": 0, "progressing": 0,
                  "weeks": 0, "total_weeks": 0, "marker_times": []}
              for m in MODELS}
    failure_names = []
    bvar_started = bvar_complete = bvar_failed = 0

    for model in MODELS:
        for block_start, block_end in blocks:
            expected = len(fridays(date.fromisoformat(block_start),
                                    date.fromisoformat(block_end)))
            family[model]["total_weeks"] += expected
            log = LOGS / f"bvar_{model}_{block_start}.log"
            is_failed = failed(log)
            if log.exists():
                family[model]["started"] += 1
                bvar_started += 1
            if is_failed:
                family[model]["failed"] += 1
                bvar_failed += 1
                failure_names.append(log.stem)
            weeks_dir = (SHARDS / f"{block_start}_{block_end}" / model /
                         "_checkpoint" / "bvar" / f"{block_start}_{block_end}" /
                         "weeks")
            try:
                markers = list(weeks_dir.glob("*.done"))
            except OSError:
                markers = []
            done = len(markers)
            for marker in markers:
                try:
                    family[model]["marker_times"].append(marker.stat().st_mtime)
                except OSError:
                    pass
            family[model]["weeks"] += min(done, expected)
            if done:
                family[model]["progressing"] += 1
            if done >= expected and expected and not is_failed:
                family[model]["complete"] += 1
                bvar_complete += 1

    dfm_rows = []
    dfm_started = dfm_complete = dfm_failed = 0
    dfm_work = 0.0
    for spec, variant in cells:
        log = cell_log(spec, variant)
        text = read_text(log)
        is_failed = bool(FAILURE.search(text))
        latest = latest_dfm_date(spec, variant)
        # A restart truncates the per-process log before the resumed worker
        # writes its final line.  The cell CSV/theta checkpoints are persistent,
        # so full date coverage is also authoritative evidence of completion.
        is_complete = (("pubblicato ->" in text) or
                       (latest is not None and latest >= ALL_FRIDAYS[-1])) \
                      and not is_failed
        if log.exists():
            dfm_started += 1
        if is_complete:
            dfm_complete += 1
        if is_failed:
            dfm_failed += 1
            failure_names.append(log.stem)
        fraction = 1.0 if is_complete else dfm_fraction(latest)
        dfm_work += fraction
        dfm_rows.append((cell_label(spec, variant), fraction,
                         latest.isoformat() if latest else "starting"))

    total_jobs = len(blocks) * len(MODELS) + len(cells)
    started_jobs = bvar_started + dfm_started
    complete_jobs = bvar_complete + dfm_complete
    failed_jobs = bvar_failed + dfm_failed
    running_jobs = max(0, started_jobs - complete_jobs - failed_jobs)
    queued_jobs = max(0, total_jobs - started_jobs)

    elapsed = None if started is None else now - started
    lines = [
        "RUN ALL — LIVE PROGRESS",
        f"updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}   "
        f"elapsed {fmt_duration(elapsed)}   refresh {REFRESH_SECONDS}s   Ctrl-C exits",
        "",
        f"ESTIMATION JOBS  [{bar(complete_jobs / max(1, total_jobs))}] "
        f"{complete_jobs}/{total_jobs} complete   {running_jobs} running   "
        f"{queued_jobs} queued   {failed_jobs} failed",
        "",
        "BVAR MODEL SHARDS",
        f"  {'model':<7} {'done':>4} {'run':>4} {'queue':>5}   {'weeks':<17} "
        f"{'progress':<27} ETA at current rate",
    ]

    family_etas = {}
    for model in MODELS:
        item = family[model]
        active = max(0, item["started"] - item["complete"] - item["failed"])
        queued = max(0, len(blocks) - item["started"])
        frac = item["weeks"] / max(1, item["total_weeks"])
        calibrated = item["progressing"] >= 3 and item["weeks"] >= 10
        observed_eta = (rates.eta(model, item["weeks"], item["total_weeks"], now,
                                  started, min_change=5.0)
                        if calibrated else None)
        disk_eta = (checkpoint_eta(item["marker_times"], item["weeks"],
                                   item["total_weeks"], now)
                    if calibrated else None)
        eta = observed_eta if observed_eta is not None else disk_eta
        family_etas[model] = eta
        if item["complete"] == len(blocks):
            eta_text = "complete"
            family_etas[model] = 0.0
        elif not calibrated and item["weeks"]:
            eta_text = f"calibrating ({item['progressing']} shard"
            eta_text += "s)" if item["progressing"] != 1 else ")"
        elif calibrated and eta is None:
            eta_text = "learning rate"
        elif observed_eta is None and disk_eta is not None:
            eta_text = "~" + fmt_duration(eta) + " (provisional)"
        else:
            eta_text = fmt_duration(eta)
        lines.append(
            f"  {model:<7} {item['complete']:>4} {active:>4} {queued:>5}   "
            f"{item['weeks']:>4}/{item['total_weeks']:<10} "
            f"[{bar(frac, 18)}] {frac:6.1%}  {eta_text}")

    dfm_eta = rates.eta("dfm", dfm_work, len(cells), now, started)
    dfm_frac = dfm_work / max(1, len(cells))
    lines.extend([
        "",
        f"DFM CELLS  [{bar(dfm_frac)}] average {dfm_frac:6.1%}   "
        f"jobs {dfm_complete}/{len(cells)} complete   ETA {fmt_duration(dfm_eta)}",
    ])
    for name, fraction, latest in dfm_rows:
        lines.append(f"  {name:<39} {fraction:6.1%}   through {latest}")

    merge_logs = list(LOGS.glob("merge_bvar_*.log"))
    merge_complete = sum("merged " in read_text(path) and not failed(path)
                         for path in merge_logs)
    output_log = LOGS / "forecast_outputs.log"
    output_state = ("not started" if not output_log.exists() else
                    "failed" if failed(output_log) else "running/finished")

    critical = {"DFM": dfm_eta, **{m.upper(): family_etas[m] for m in MODELS}}
    unavailable = [name for name, eta in critical.items() if eta is None]
    if unavailable:
        overall = ("pending — no measurable progress yet for " +
                   ", ".join(unavailable))
    else:
        overall = "~" + fmt_duration(max(critical.values()))

    total, used, free = shutil.disk_usage(ROOT)
    mem = memory()
    resource = f"disk free {free / 2**40:.2f} TiB"
    if mem:
        resource += f"   RAM available {mem[1] / 2**30:.0f}/{mem[0] / 2**30:.0f} GiB"

    lines.extend([
        "",
        f"PUBLICATION  BVAR merges {merge_complete}/{len(blocks)}   outputs {output_state}",
        f"OVERALL ETA  {overall}",
        "             ETA learns from 30s+ of observed checkpoint movement;"
        " first full BVAR estimations are invisible until their first marker.",
        f"RESOURCES    {resource}",
    ])
    if failure_names:
        lines.append("FAILURES     " + ", ".join(sorted(failure_names)[:8]))
        if len(failure_names) > 8:
            lines.append(f"             and {len(failure_names) - 8} more")
    else:
        lines.append("FAILURES     none detected")
    return "\n".join(lines)


def main() -> None:
    rates = Rates()
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        print(snapshot(rates))
        return

    import curses

    def dashboard(screen) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.keypad(True)
        screen.nodelay(True)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
        except curses.error:
            pass

        lines: list[str] = []
        offset = 0
        next_update = 0.0
        running = True
        while running:
            now = time.monotonic()
            if now >= next_update or not lines:
                lines = snapshot(rates).splitlines()
                next_update = now + REFRESH_SECONDS

            height, width = screen.getmaxyx()
            visible = max(1, height - 1)
            maximum = max(0, len(lines) - visible)
            offset = min(maximum, max(0, offset))

            screen.erase()
            for row, line in enumerate(lines[offset:offset + visible]):
                try:
                    screen.addnstr(row, 0, line, max(1, width - 1))
                except curses.error:
                    pass
            footer = (f" lines {offset + 1}-{min(len(lines), offset + visible)}"
                      f"/{len(lines)}  ↑↓/wheel scroll  PgUp/PgDn  Home/End  "
                      f"q exits  refresh {REFRESH_SECONDS}s ")
            try:
                screen.addnstr(height - 1, 0, footer.ljust(max(1, width - 1)),
                               max(1, width - 1), curses.A_REVERSE)
            except curses.error:
                pass
            screen.refresh()

            key = screen.getch()
            if key == -1:
                time.sleep(0.1)
                continue
            if key in (ord("q"), ord("Q"), 3):
                running = False
            elif key == curses.KEY_UP:
                offset -= 1
            elif key == curses.KEY_DOWN:
                offset += 1
            elif key == curses.KEY_PPAGE:
                offset -= visible
            elif key == curses.KEY_NPAGE:
                offset += visible
            elif key == curses.KEY_HOME:
                offset = 0
            elif key == curses.KEY_END:
                offset = maximum
            elif key == curses.KEY_MOUSE:
                try:
                    _, _, _, _, state = curses.getmouse()
                    if state & getattr(curses, "BUTTON4_PRESSED", 0):
                        offset -= 3
                    if state & getattr(curses, "BUTTON5_PRESSED", 0):
                        offset += 3
                except curses.error:
                    pass

    try:
        curses.wrapper(dashboard)
    except KeyboardInterrupt:
        pass
    print("progress monitor stopped")


if __name__ == "__main__":
    main()
