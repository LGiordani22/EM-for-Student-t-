#!/usr/bin/env bash
# =============================================================================
# scripts/run_all_par.sh -- full DFM + BVAR pass with concurrent estimation
#
# Unlike run_all.sh, phases 3 (DFM) and 5 (BVAR) begin together.  Their outputs
# and checkpoints are disjoint; collection, figures, and metrics wait for both.
# Do not run this alongside scripts/run_all.sh on the same output directory.
# =============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export WEEKLY_SAVE_EVERY_ROWS=100
export WEEKLY_SAVE_EVERY_SECONDS=120

if [[ "${SMOKE:-0}" == "1" ]]; then
  readonly START="2009-01-02"
  readonly END="2009-02-27"
  readonly SMOKE_DRAWS=50
else
  readonly START="2007-01-01"
  readonly END="2025-12-31"
  readonly SMOKE_DRAWS=""
fi

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OUT="$ROOT/output/forecast_weekly"
readonly CELLS_DIR="$OUT/csv/_cells"
readonly LOGS="$ROOT/output/_logs"
readonly N_CORES="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
readonly SPECS=(diag3 diag4 fed_overlap)
readonly VARIANTS=(gaussian gaussian_ar1 student_t student_t_ar1 student_t_ar1_shared)
PYTHON="${PYTHON:-python3}"
WORKERS="${MAX_WORKERS:-$N_CORES}"
(( WORKERS < 1 )) && WORKERS=1

log() { printf '\n\033[1m[%s]  %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[1;31m[ERRORE]  %s\033[0m\n' "$*" >&2; exit 1; }
secs_to_hm() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

run_pool() {
  local max="$1"; shift
  local -a commands=("$@")
  local running=0 rc=0
  local command
  for command in "${commands[@]}"; do
    while (( running >= max )); do
      if wait -n; then :; else rc=1; fi
      running=$((running - 1))
    done
    eval "$command" &
    running=$((running + 1))
  done
  while (( running > 0 )); do
    if wait -n; then :; else rc=1; fi
    running=$((running - 1))
  done
  return "$rc"
}

cd "$ROOT"
command -v "$PYTHON" >/dev/null || fail "python non trovato: esporta PYTHON=/percorso/python"
"$PYTHON" -c "import src.output_layout" 2>/dev/null \
  || fail "i moduli non si importano da $ROOT"

# The sequential orchestrator has no lock.  Refuse an overlap that would write
# the same DFM cells or BVAR checkpoints concurrently.
if pgrep -af 'bash scripts/run_all\.sh' >/dev/null; then
  fail "scripts/run_all.sh e' gia' in esecuzione: attendilo o fermalo prima di lanciare run_all_par.sh"
fi

mkdir -p "$LOGS" "$CELLS_DIR" "$OUT/csv/dfm" "$OUT/csv/bvar"
exec 9>"$OUT/.run_all_par.lock"
flock -n 9 || fail "run_all_par.sh e' gia' in esecuzione per questo output"

mapfile -t BLOCKS < <("$PYTHON" -m src.bvar.evaluate --start "$START" --end "$END" --print-blocks) \
  || fail "non riesco a calcolare i blocchi BVAR"
(( ${#BLOCKS[@]} > 0 )) || fail "nessun blocco BVAR in $START .. $END"

T0=$(date +%s)
printf '%s\n' \
  "================================================================================" \
  "  PASSATA PARALLELA -- DFM + BVAR, $START .. $END" \
  "  macchina: $N_CORES core; massimo processi: $WORKERS" \
  "  DFM (15 celle) e BVAR (${#BLOCKS[@]} blocchi) partono insieme." \
  "================================================================================"

log "FASE 1/6 -- guardie"
"$PYTHON" -m src.forecast.test_windows --pre-run \
  || fail "test_windows fallito"
"$PYTHON" -m src.forecast.test_common_sample \
  || fail "test_common_sample fallito"

run_dfm() {
  # Sedici unita' indipendenti: le quindici celle piu' il lavoro dei benchmark.
  # I benchmark non sono piu' attaccati alla prima cella — hanno cartella e
  # ripresa proprie, quindi partono insieme alle altre invece che dentro una.
  local -a commands=()
  local spec variant directory
  for spec in "${SPECS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      directory="$CELLS_DIR/${spec}_${variant}"
      mkdir -p "$directory"
      commands+=("\"$PYTHON\" -m src.forecast.weekly_nowcast --start $START --end $END --spec $spec --variant $variant --no-benchmarks --output-dir '$directory' > '$LOGS/dfm_${spec}_${variant}.log' 2>&1")
    done
  done
  directory="$CELLS_DIR/benchmark"
  mkdir -p "$directory"
  commands+=("\"$PYTHON\" -m src.forecast.weekly_nowcast --start $START --end $END --only-benchmarks --output-dir '$directory' > '$LOGS/dfm_benchmark.log' 2>&1")
  run_pool "$WORKERS" "${commands[@]}"
}

run_bvar() {
  local -a commands=()
  local block block_start block_end draws_flag=""
  [[ -n "$SMOKE_DRAWS" ]] && draws_flag="--draws $SMOKE_DRAWS"
  for block in "${BLOCKS[@]}"; do
    block_start="${block%% *}"
    block_end="${block##* }"
    commands+=("\"$PYTHON\" -m src.bvar.evaluate --start $block_start --end $block_end $draws_flag > '$LOGS/bvar_${block_start}.log' 2>&1")
  done
  run_pool "$WORKERS" "${commands[@]}"
}

log "FASE 2/6 -- DFM e BVAR in parallelo"
run_dfm &
dfm_pid=$!
run_bvar &
bvar_pid=$!

dfm_rc=0
bvar_rc=0
wait "$dfm_pid" || dfm_rc=$?
wait "$bvar_pid" || bvar_rc=$?
(( dfm_rc == 0 )) || fail "una o piu' celle DFM sono fallite; vedi $LOGS/dfm_*.log"
(( bvar_rc == 0 )) || fail "uno o piu' blocchi BVAR sono falliti; vedi $LOGS/bvar_*.log"

log "FASE 3/6 -- raccolta e verifica DFM"
for spec in "${SPECS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    source_csv="$CELLS_DIR/${spec}_${variant}/weekly_nowcast_${START}_${END}.csv"
    [[ -f "$source_csv" ]] || fail "manca il CSV DFM: $source_csv"
    cp -f "$source_csv" "$OUT/csv/dfm/weekly_nowcast_${spec}_${variant}_${START}_${END}.csv"
  done
done
"$PYTHON" -m src.forecast.test_cells_produced --expect 15 \
  || fail "il DFM non ha prodotto nowcast validi; vedi $LOGS/dfm_*.log"

log "FASE 4/6 -- figure di traiettoria"
for window in 2007-2010 2014-2016 2019-2021 2024-2025; do
  if ! "$PYTHON" -m src.forecast.figures --window "$window" > "$LOGS/fig_dfm_$window.log" 2>&1; then
    grep -q "Nessuna riga" "$LOGS/fig_dfm_$window.log" || fail "figure DFM $window: vedi $LOGS/fig_dfm_$window.log"
  fi
  if ! "$PYTHON" -m src.bvar.figures --window "$window" > "$LOGS/fig_bvar_$window.log" 2>&1; then
    grep -q "Nessuna riga\|Nessun CSV" "$LOGS/fig_bvar_$window.log" || fail "figure BVAR $window: vedi $LOGS/fig_bvar_$window.log"
  fi
done

log "FASE 5/6 -- NY Fed, metriche e tabelle"
nyfed_commands=()
for spec in "${SPECS[@]}"; do
  nyfed_commands+=("\"$PYTHON\" -m src.forecast.nyfed_all --spec $spec > '$LOGS/nyfed_${spec}.log' 2>&1 || true")
done
run_pool "$WORKERS" "${nyfed_commands[@]}"
"$PYTHON" -m src.bvar.metrics > "$LOGS/bvar_metrics.log" 2>&1 \
  || fail "metriche BVAR: vedi $LOGS/bvar_metrics.log"
"$PYTHON" -m src.forecast.metrics_tables > "$LOGS/metrics_tables.log" 2>&1 \
  || fail "tabelle finali: vedi $LOGS/metrics_tables.log"

log "FASE 6/6 -- guardia di allineamento"
"$PYTHON" -m src.forecast.test_windows \
  || fail "allineamento DFM/BVAR rotto dopo la passata"

elapsed=$(( $(date +%s) - T0 ))
printf '\nFATTO. durata effettiva: %s\n' "$(secs_to_hm "$elapsed")"