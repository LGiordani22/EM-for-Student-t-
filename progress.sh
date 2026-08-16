#!/usr/bin/env bash
# Read-only progress summary for scripts/run_all.sh and scripts/run_all_par.sh.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

readonly START="2007-01-01"
readonly END="2025-12-31"
readonly EXPECTED_ROWS=2277

heading() {
  printf '\n\033[1;36m%s\033[0m\n' "$1"
}

count_processes() {
  local needle="$1"
  ps -eo comm=,args= | awk -v needle="$needle" \
    '$1 == "python" && index($0, needle) { count += 1 } END { print count + 0 }'
}

printf '\033[1mForecast Pipeline Progress\033[0m\n'
printf 'Checked: %s\n' "$(date '+%F %T %Z')"

heading 'Active Workers'
orchestrators="$(ps -eo comm=,args= | awk '$1 == "bash" && index($0, "scripts/run_all_par.sh") { count += 1 } END { print count + 0 }')"
printf '  Orchestrators : %s\n' "$orchestrators"
printf '  DFM workers  : %s\n' "$(count_processes 'src.forecast.weekly_nowcast')"
printf '  BVAR workers : %s\n' "$(count_processes 'src.bvar.evaluate --start')"
ps -eo pid,etime,%cpu,args | rg 'src\.forecast\.weekly_nowcast|src\.bvar\.evaluate' || true

heading 'DFM Cells'
printf '  %-38s %10s %9s  %s\n' 'cell' 'rows' 'progress' 'last checkpoint'
complete=0
total=0
for directory in output/forecast_weekly/csv/_cells/*; do
  [[ -d "$directory" ]] || continue
  file="$directory/weekly_nowcast_${START}_${END}.csv"
  [[ -f "$file" ]] || continue
  cell="$(basename "$directory")"
  rows=$(( $(wc -l < "$file") - 1 ))
  (( rows >= EXPECTED_ROWS )) && ((complete += 1))
  ((total += 1))
  progress=$(( rows * 1000 / EXPECTED_ROWS ))
  (( progress > 1000 )) && progress=1000
  printf '  %-38s %5d/%-4d %7.1f%%  %s\n' \
    "$cell" "$rows" "$EXPECTED_ROWS" "$progress"e-1 \
    "$(stat -c '%y' "$file" | cut -d. -f1)"
done
printf '\n  Completed: %d/%d cells\n' "$complete" "$total"

heading 'BVAR'
bvar_files=$(find output/forecast_weekly/csv/bvar -type f 2>/dev/null | wc -l)
mapfile -t failures < <(rg -l 'Traceback|ValueError|ERROR|ERRORE' output/_logs/bvar_*.log 2>/dev/null || true)
printf '  Output files : %s\n' "$bvar_files"
printf '  Failed blocks: %d\n' "${#failures[@]}"
for log in "${failures[@]}"; do
  printf '    - %s\n' "$(basename "$log" .log)"
done

heading 'Reporting'
dfm_csvs=$(find output/forecast_weekly/csv/dfm -type f 2>/dev/null | wc -l)
comparison_files=$(find output/forecast_weekly/comparison -type f 2>/dev/null | wc -l)
printf '  Collected DFM CSVs : %s\n' "$dfm_csvs"
printf '  Comparison files   : %s\n' "$comparison_files"

heading 'Machine'
uptime
free -h | awk 'NR == 2 {printf "  Memory: %s used / %s total / %s available\n", $3, $2, $7}'