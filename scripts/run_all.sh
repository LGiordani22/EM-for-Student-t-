#!/usr/bin/env bash
# =============================================================================
#  scripts/run_all.sh — LA PASSATA COMPLETA, DAL DATO ALLE TABELLE
#
#  Uso:      ./scripts/run_all.sh          (dalla radice del repo)
#  Ripresa:  ./scripts/run_all.sh          (lo stesso comando: riprende)
#
#  E' lo script del SERVER (Linux, molti core, esecuzione diretta), ed e'
#  l'unico orchestratore rimasto: il vecchio `run_forecasts.ps1` e' stato
#  cancellato perche' portava periodi che non si usano piu' e non copriva i
#  BVAR.  Per girare a mano una cella sola bastano i moduli
#  (`python -m src.forecast.weekly_nowcast --spec ... --variant ...`).
#
# =============================================================================
#  CHE COSA FA QUESTO SCRIPT
# =============================================================================
#
#  Stima il nowcast settimanale del PIL USA con due famiglie di modelli sullo
#  stesso identico calendario real-time, e ne confronta l'accuratezza.
#
#    DFM   15 celle = 3 strutture di caricamento (diag3, diag4, fed_overlap)
#                     x 5 varianti (gaussian, gaussian_ar1, student_t,
#                     student_t_ar1, student_t_ar1_shared)
#    BVAR   4 modelli (qbvar, cbvar, bbvar, lbvar)
#    piu' 2 benchmark univariati (AR(2) e media espandente) come metro.
#
#  Il periodo e' UNO SOLO: 2007-01-01 .. 2025-12-31, ogni venerdi'.  Tutte le
#  finestre sotto sono RITAGLI di questa passata, non passate separate: girare
#  le sei finestre a parte ri-stimerebbe tredici volte gli anni che condividono.
#
# =============================================================================
#  CHE COSA PRODUCE, E DOVE
# =============================================================================
#
#  output/forecast_weekly/
#    csv/dfm/          i nowcast settimanali del DFM, formato lungo (INPUT di
#                      tutto il resto: se si cancellano, si rifa' la passata)
#    csv/bvar/         gli stessi per i BVAR, piu' i quantili .npz e i log score
#    dfm/<spec>/<variante>/    4 figure di traiettoria per cella, una per
#                      finestra forecast
#    dfm/<spec>/rmse/  tabelle di accuratezza + figure RMSE per orizzonte
#    bvar/<modello>/   4 figure di traiettoria per modello
#    bvar/rmse/        tabelle + LA figura RMSE con i quattro BVAR INSIEME
#    bvar/logscore/    log predictive score (tabelle + una figura sola)
#    comparison/       BVAR-vs-DFM: matrici aggregate e confronto PER FASE
#
#  LE FINESTRE, e perche' sono di tre tipi diversi
#  -----------------------------------------------
#    4 finestre FORECAST — figure di traiettoria, un regime ciascuna:
#        2007-2010 (Grande Recessione)   2014-2016 (espansione)
#        2019-2021 (Covid)               2024-2025 (coda del campione)
#
#    3 passate RMSE — campioni CUMULATI, tutti dal 2007, per vedere se la
#      graduatoria regge man mano che si aggiungono anni:
#        2007-2019    2007-2021    2007-2025
#
#    3 zoom RMSE — la stessa metrica su un regime solo:
#        2007-2010    2019-2021    2024-2025
#
#  COSA ASPETTARSI IN USCITA
#  --------------------------
#    - 15 celle x 4 finestre = 60 figure DFM di traiettoria, piu' 16 BVAR
#    - per ogni spec DFM e per la famiglia BVAR: 6 figure RMSE per orizzonte
#      (3 passate + 3 zoom), con le tre fasi come bande e il NY Fed come curva
#      nera dove pubblica
#    - tabelle di accuratezza con RMSE, MAE, Bias, corr, MDA, SignAcc
#    - un log score per i soli BVAR (i point forecast non hanno densita')
#    - il confronto BVAR-vs-DFM per fase, con il BACKCAST come pannello a se'
#
#  COME SI LEGGONO LE TABELLE (la cosa che si sbaglia piu' facilmente)
#  -------------------------------------------------------------------
#  Ogni metrica compare DUE VOLTE, affiancata:
#      n     / RMSE     / MDA        tutto cio' che quel metodo ha punteggiato
#      n_com / RMSE_com / MDA_com    solo i punti (trimestre, settimana)
#                                    punteggiati da TUTTI i metodi della tabella
#  SOLO le colonne `_com` confrontano le righe fra loro.  Un metodo che parte
#  piu' tardi, o che si ferma alla settimana 13 invece che alla 17, e' mediato
#  su bersagli diversi: sul campione libero puo' sembrare migliore per ragioni
#  che col modello non c'entrano.  La distanza fra RMSE e RMSE_com misura
#  esattamente quanto pesa quella differenza.
#
# =============================================================================
#  RIPRESA
# =============================================================================
#  Lo script NON CANCELLA NIENTE all'avvio.  Se il run si interrompe — di
#  qualunque morte — basta rilanciare lo stesso comando: ogni fase riprende dai
#  suoi file gia' su disco (i CSV parziali per il DFM, i checkpoint per il
#  BVAR) e ricalcola solo cio' che manca.  Per ricominciare davvero da zero
#  bisogna cancellare a mano output/forecast_weekly/csv/ e output/_checkpoint/.
#
# =============================================================================
#  PARALLELISMO — MISURATO, NON STIMATO
# =============================================================================
#  Le 15 celle DFM sono indipendenti fra loro, e cosi' i blocchi temporali del
#  BVAR: si possono girare tutte insieme.  Ma "quanti thread per processo" NON
#  si decide a tavolino, e su questo progetto le stime a priori hanno sbagliato
#  ripetutamente: dentro lo stesso codice convivono regimi opposti — la ricerca
#  della moda vuole 1 thread (il multi-thread le fa perdere tempo in
#  sincronizzazione), lo smoother dell'L-BVAR ne vuole molti (e' algebra densa).
#
#  L'ASSE E' LA CELLA, NON IL PERIODO, ed e' voluto.  Il nome del CSV dipende
#  solo dal periodo, quindi due celle sullo stesso periodo E sulla stessa
#  cartella si sovrascriverebbero.  Qui si parallelizza sulla CELLA dando a
#  ciascuna una cartella sua (fase 3): la collisione sparisce, e in piu' si
#  guadagna una cosa che il taglio per periodo perde — ogni cella percorre
#  il 2007-2025 in un processo solo, quindi la sua catena di theta resta
#  continua per tutti e diciannove gli anni.  Spezzando per periodo, la catena
#  si interrompe a ogni confine di blocco.
#
#  Percio' la FASE 2 qui sotto MISURA: gira lo stesso blocco corto con 1, 2, 4,
#  8, 16 thread e sceglie il punto che minimizza i CORE-SECONDI per blocco,
#  cioe' il miglior rendimento per core — che con 224 core e ~92 unita'
#  indipendenti (77 blocchi BVAR + 15 celle DFM) e' la quantita' che conta,
#  non il tempo del singolo blocco.
#  Il risultato finisce in un file e non viene rimisurato alle riprese.
# =============================================================================

set -euo pipefail

# ─── Manopole: FISSE QUI, non da toccare ─────────────────────────────────────
# Ogni processo prende N thread BLAS; il parallelismo vero e' fra processi.
# Il valore viene deciso dalla fase 2 e riesportato prima di ogni fase.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Salvataggio incrementale: ogni 100 righe o 120 secondi, quello che viene
# prima.  Servono entrambi i limiti — le righe da sole non bastano quando una
# cella e' lenta, i secondi da soli non bastano quando e' veloce.  E' anche
# cio' che rende la ripresa fine: si perde al massimo un blocco di lavoro.
export WEEKLY_SAVE_EVERY_ROWS=100
export WEEKLY_SAVE_EVERY_SECONDS=120

# ─── MODALITA' SMOKE: la prova dell'impianto ─────────────────────────────────
#   SMOKE=1 ./scripts/run_all.sh
#
# Gira TUTTE E SETTE LE FASI su un calendario di due mesi e con estrazioni
# ridotte, cosi' che finisca in circa un'ora invece che in giorni.  Non serve a
# stimare niente: serve a rispondere alla sola domanda che conta prima di
# lanciare la passata vera — le fasi si concatenano, i file finiscono dove il
# codice li cerca, la raccolta trova cio' che deve, le tabelle si scrivono?
#
# I NUMERI CHE PRODUCE NON SONO RISULTATI.  Con estrazioni ridotte e due mesi
# di calendario le catene non sono convergute e i campioni sono minuscoli: le
# tabelle usciranno con gli avvisi di campione parziale, ed e' giusto cosi'.
#
# SCRIVE NELL'ALBERO VERO, e va ripulito dopo: `nyfed_all`, `bvar.metrics` e
# `metrics_tables` non hanno una manopola per la cartella di uscita — decidono
# i percorsi da `output_layout`.  A fine corsa lo script stampa il comando di
# pulizia; finche' non lo si lancia, in output/ ci sono numeri da buttare.
if [[ "${SMOKE:-0}" == "1" ]]; then
  readonly START="2009-01-02"
  readonly END="2009-02-27"
  readonly SMOKE_DRAWS=50
else
  readonly START="2007-01-01"
  readonly END="2025-12-31"
  readonly SMOKE_DRAWS=""
fi
# `nproc` NUDO QUI SAREBBE SEMPRE 1, ed e' il bug piu' costoso che questo script
# abbia avuto.  GNU coreutils fa onorare a `nproc` le variabili OMP — e noi
# abbiamo appena esportato OMP_NUM_THREADS=1 poche righe sopra, per dare un
# thread a processo.  Il risultato era `N_CORES=1`, quindi `WORKERS=1`, quindi
# TUTTO IN FILA: i 77 blocchi BVAR uno dopo l'altro su una macchina da 224 core,
# e la fase 2 che scarta ogni prova con t>1 (`(( t > N_CORES )) && continue`)
# senza misurare niente.  In silenzio: nessun errore, solo giorni di calcolo.
#
# `env -u` e non `nproc --all`: `--all` conta i processori INSTALLATI e ignora
# anche l'affinita' di CPU, quindi su un nodo con cgroup o allocazione parziale
# prometterebbe core che non abbiamo.  Togliendo solo le due OMP si rispetta
# l'allocazione vera e si ignorano le manopole che ci siamo messi da soli.
readonly N_CORES="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"

readonly SPECS=(diag3 diag4 fed_overlap)
readonly VARIANTS=(gaussian gaussian_ar1 student_t student_t_ar1 student_t_ar1_shared)

# La radice del repo e' il PADRE di scripts/, non la cartella dello script.
# Si ricava dal percorso dello script e non dalla directory corrente, cosi'
# `./scripts/run_all.sh` e `cd scripts && ./run_all.sh` fanno la stessa cosa.
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OUT="$ROOT/output/forecast_weekly"
readonly CELLS_DIR="$OUT/csv/_cells"          # una sottocartella per cella DFM
readonly LOGS="$ROOT/output/_logs"
readonly SIZING="$ROOT/output/_sizing.env"    # la misura della fase 2, cachata

PYTHON="${PYTHON:-python3}"

mkdir -p "$LOGS" "$CELLS_DIR" "$OUT/csv/dfm" "$OUT/csv/bvar"

log()  { printf '\n\033[1m[%s]  %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[1;31m[ERRORE]  %s\033[0m\n' "$*" >&2; exit 1; }

# Pool di processi paralleli: lancia al massimo $1 job insieme.
# `wait -n` ritorna appena UNO qualsiasi finisce, quindi il pool resta pieno
# invece di svuotarsi aspettando il piu' lento del gruppo.
run_pool() {
  local max="$1"; shift
  local -a cmds=("$@")
  local running=0 rc=0
  for c in "${cmds[@]}"; do
    while (( running >= max )); do
      if wait -n; then :; else rc=1; fi
      running=$((running - 1))
    done
    eval "$c" &
    running=$((running + 1))
  done
  while (( running > 0 )); do
    if wait -n; then :; else rc=1; fi
    running=$((running - 1))
  done
  return $rc
}

secs_to_hm() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

T0=$(date +%s)
MODO="PASSATA COMPLETA"
[[ -n "$SMOKE_DRAWS" ]] && MODO="*** SMOKE — PROVA DELL'IMPIANTO, I NUMERI SI BUTTANO ***"

cd "$ROOT"
command -v "$PYTHON" >/dev/null || fail "python non trovato: esporta PYTHON=/percorso/python"
"$PYTHON" -c "import src.output_layout" 2>/dev/null \
  || fail "i moduli non si importano da $ROOT — lancia lo script dalla radice del repo"

cat <<BANNER

================================================================================
  ${MODO:-PASSATA COMPLETA} — DFM + BVAR, $START .. $END
  macchina: $N_CORES core        radice: $ROOT
  ripresa : rilanciare questo stesso comando; niente viene cancellato all'avvio
================================================================================
BANNER

# =============================================================================
#  FASE 1 — LE GUARDIE.  Passano prima di iniziare, o non si inizia.
# =============================================================================
#  Costano due secondi e proteggono ore.  La prima verifica che le sei finestre
#  taglino il periodo che dichiarano; la seconda che il vincolo di campione
#  comune scatti davvero — cioe' che le tabelle di confronto non mettano
#  accanto RMSE calcolati su bersagli diversi.  Se una fallisce, il run non
#  parte: scoprirlo alla fine significherebbe buttare via la passata.
# =============================================================================
log "FASE 1/7 — guardie"
"$PYTHON" -m src.forecast.test_windows --pre-run \
  || fail "test_windows fallito: le finestre non tagliano il periodo dichiarato."
"$PYTHON" -m src.forecast.test_common_sample \
  || fail "test_common_sample fallito: il vincolo di campione comune non scatta."

# =============================================================================
#  FASE 2 — DIMENSIONAMENTO MISURATO
# =============================================================================
log "FASE 2/7 — dimensionamento (misura, non stima)"

if [[ -n "$SMOKE_DRAWS" ]]; then
  THREADS_PER_JOB=1; SECS_PER_BLOCK=600
  echo "  SMOKE: dimensionamento saltato (1 thread). Misurarlo su due mesi"
  echo "  non direbbe niente sulla passata vera."
elif [[ -f "$SIZING" ]]; then
  # shellcheck disable=SC1090
  source "$SIZING"
  echo "  gia' misurato in una passata precedente (cancella $SIZING per rifarlo):"
  echo "    thread per processo : $THREADS_PER_JOB"
  echo "    secondi per blocco  : $SECS_PER_BLOCK"
else
  echo "  giro lo stesso blocco corto con 1, 2, 4, 8, 16 thread e misuro."
  echo "  criterio: CORE-SECONDI per blocco (thread x tempo), cioe' il"
  echo "  rendimento per core — con 224 core e ~92 unita' indipendenti e'"
  echo "  quello che decide il tempo totale, non il tempo del singolo blocco."
  best_t=1; best_cost=""; best_wall=""
  for t in 1 2 4 8 16; do
    (( t > N_CORES )) && continue
    probe="$ROOT/output/_pilot_sizing/t$t"
    rm -rf "$probe"; mkdir -p "$probe"
    s=$(date +%s)
    OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t \
      "$PYTHON" -m src.bvar.evaluate \
        --start 2009-01-02 --end 2009-01-23 --draws 100 --fresh \
        --output-root "$probe" > "$LOGS/sizing_t$t.log" 2>&1 \
      || fail "il blocco di dimensionamento e' fallito con $t thread; vedi $LOGS/sizing_t$t.log"
    wall=$(( $(date +%s) - s ))
    cost=$(( wall * t ))
    printf '    %2d thread  ->  %4d s di parete   %5d core-secondi\n' "$t" "$wall" "$cost"
    if [[ -z "$best_cost" || $cost -lt $best_cost ]]; then
      best_cost=$cost; best_t=$t; best_wall=$wall
    fi
  done
  rm -rf "$ROOT/output/_pilot_sizing"
  # Il blocco di prova e' a 100 estrazioni; la passata vera ne usa il pieno.
  # Il fattore si legge dai DRAWS del modulo invece di inventarlo qui.
  full_draws=$("$PYTHON" -c "from src.bvar.evaluate import DRAWS; print(max(DRAWS.values()))")
  scale=$(( full_draws / 100 )); (( scale < 1 )) && scale=1
  {
    echo "THREADS_PER_JOB=$best_t"
    echo "SECS_PER_BLOCK=$(( best_wall * scale ))"
    echo "PROBE_SECS=$best_wall"
    echo "DRAWS_SCALE=$scale"
  } > "$SIZING"
  # shellcheck disable=SC1090
  source "$SIZING"
  echo "  scelto: $THREADS_PER_JOB thread per processo (minimo core-secondi)"
fi

# ─── LA VALVOLA SULLA MEMORIA ────────────────────────────────────────────────
# `WORKERS` esce dai CORE, e QUI NESSUNO GUARDA LA RAM: non c'e' un solo punto
# in tutto lo script che legga la memoria disponibile.  Non e' una svista, e'
# che non esiste un numero giusto da cablare — dipende dal modello (l'L-BVAR
# tiene uno stato companion da n*p dimensioni nel simulation smoother), dalle
# estrazioni e dal tetto della cache (`--max-cache-mb`, default 1500).
#
# Il picco vero e' piu' basso di `WORKERS`: `run_pool` non lancia mai piu'
# processi di quanti siano i lavori, quindi la fase 5 arriva al massimo al
# numero di BLOCCHI (77 sul 2007-2025) e la fase 3 a quello delle CELLE (15).
#
# `MAX_WORKERS` e' la valvola: si abbassa senza toccare lo script, e non costa
# niente perche' la ripresa e' garantita — se la passata muore per memoria
# esaurita al blocco 50, si rilancia con un numero piu' basso e i 50 finiti
# restano.  I blocchi restano 77 in ogni caso: cambia solo in quante ondate.
#
#     ./scripts/run_all.sh                  tutti i core
#     MAX_WORKERS=60 ./scripts/run_all.sh   al massimo 60 processi insieme
WORKERS="${MAX_WORKERS:-$(( N_CORES / THREADS_PER_JOB ))}"
(( WORKERS < 1 )) && WORKERS=1
N_DFM_CELLS=$(( ${#SPECS[@]} * ${#VARIANTS[@]} ))

# I BLOCCHI BVAR LI DECIDE IL MODULO, NON QUESTO SCRIPT.  Il taglio cade sulle
# settimane di STIMA PIENA (`evaluate.parallel_blocks`), che seguono le release
# BEA: e' l'unico confine che non costa una stima in piu' e non cambia una riga
# rispetto a una passata continua.  Il vecchio taglio annuale, scritto qui a
# mano con `seq 2007 2025`, cadeva a Capodanno — in mezzo a un trimestre — e
# promuoveva diciannove settimane da riuso a stima fresca: numeri che
# dipendevano da come avevamo affettato il lavoro, non dal modello.
#
# Serve in FASE 2 e non solo in fase 5, perche' il preventivo conta le ondate.
# Costa un attraversamento di calendario, nessuna stima.
mapfile -t BLOCKS < <("$PYTHON" -m src.bvar.evaluate \
                        --start "$START" --end "$END" --print-blocks) \
  || fail "non riesco a calcolare i blocchi BVAR"
N_BVAR_BLOCKS=${#BLOCKS[@]}
(( N_BVAR_BLOCKS > 0 )) || fail "nessun blocco BVAR in $START .. $END"

export OMP_NUM_THREADS=$THREADS_PER_JOB
export OPENBLAS_NUM_THREADS=$THREADS_PER_JOB
export MKL_NUM_THREADS=$THREADS_PER_JOB

# ─── PREVENTIVO ──────────────────────────────────────────────────────────────
bvar_waves=$(( (N_BVAR_BLOCKS + WORKERS - 1) / WORKERS ))
est_bvar=$(( bvar_waves * SECS_PER_BLOCK ))
# Il DFM non ha un blocco di prova proprio: si dichiara che la sua stima e'
# GROSSOLANA invece di darle una precisione che non ha.  Il consuntivo a fine
# script dice il vero, ed e' quello da citare la prossima volta.
est_dfm=$(( SECS_PER_BLOCK * 3 ))
cat <<PREVENTIVO

  ── PREVENTIVO ────────────────────────────────────────────────────────────
    core disponibili      : $N_CORES
    thread per processo   : $THREADS_PER_JOB   (misurati, non assunti)
    processi in parallelo : $WORKERS$([[ -n "${MAX_WORKERS:-}" ]] && echo "   (limitati a mano con MAX_WORKERS)")
    picco vero            : $(( WORKERS < N_BVAR_BLOCKS ? WORKERS : N_BVAR_BLOCKS )) processi in fase 5, $(( WORKERS < N_DFM_CELLS ? WORKERS : N_DFM_CELLS )) in fase 3
                            (nessuno guarda la RAM: se serve, MAX_WORKERS=N)
    celle DFM             : $N_DFM_CELLS  (tutte in una sola ondata)
    blocchi BVAR          : $N_BVAR_BLOCKS   -> $bvar_waves ondata/e
    durata stimata        : ~$(secs_to_hm $(( est_dfm + est_bvar ))) piu' figure e tabelle
                            (STIMA GROSSOLANA, estrapolata da un blocco corto a
                             estrazioni ridotte: il consuntivo in fondo e' il
                             numero vero)
  ──────────────────────────────────────────────────────────────────────────

PREVENTIVO

# =============================================================================
#  FASE 3 — DFM, le 15 celle in parallelo
# =============================================================================
#  OGNI CELLA SCRIVE IN UNA CARTELLA SUA, e non e' un dettaglio: il nome del
#  CSV e' `weekly_nowcast_<inizio>_<fine>.csv`, che per quindici celle sullo
#  STESSO periodo sarebbe lo stesso identico file — quindici processi che si
#  sovrascrivono a vicenda.  Con una cartella per cella i CSV non si toccano,
#  e la fase 4 li raccoglie con nomi distinti.
#
#  I benchmark (AR(2), media espandente) li calcola UNA cella sola: sono
#  univariati e non dipendono ne' dalla spec ne' dalla variante, quindi
#  quindici copie identiche gonfierebbero soltanto il conteggio `n` delle
#  tabelle senza cambiare un solo RMSE.
# =============================================================================
log "FASE 3/7 — DFM: $N_DFM_CELLS celle in parallelo, $START .. $END"
cmds=(); first=1
for s in "${SPECS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    d="$CELLS_DIR/${s}_${v}"; mkdir -p "$d"
    bench=""; (( first )) || bench="--no-benchmarks"; first=0
    cmds+=("\"$PYTHON\" -m src.forecast.weekly_nowcast --start $START --end $END \
            --spec $s --variant $v $bench --output-dir '$d' \
            > '$LOGS/dfm_${s}_${v}.log' 2>&1")
  done
done
run_pool "$WORKERS" "${cmds[@]}" \
  || fail "una o piu' celle DFM sono fallite; i log stanno in $LOGS/dfm_*.log"

# =============================================================================
#  FASE 4 — raccolta dei CSV DFM
# =============================================================================
#  Si COPIA, non si sposta: il file nella cartella della cella e' lo stato di
#  ripresa di quella cella.  Spostandolo, un rilancio ripartirebbe da zero.
# =============================================================================
log "FASE 4/7 — raccolta dei CSV del DFM"
n=0
for s in "${SPECS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    src="$CELLS_DIR/${s}_${v}/weekly_nowcast_${START}_${END}.csv"
    [[ -f "$src" ]] || fail "manca il CSV della cella $s/$v: $src"
    cp -f "$src" "$OUT/csv/dfm/weekly_nowcast_${s}_${v}_${START}_${END}.csv"
    n=$((n + 1))
  done
done
echo "  $n CSV raccolti in $OUT/csv/dfm/"

# =============================================================================
#  FASE 5 — BVAR, un blocco per STIMA PIENA in parallelo
# =============================================================================
#  I blocchi sono indipendenti: ognuno ha il suo checkpoint e il suo CSV, con
#  il periodo nel nome, quindi non si sovrappongono.  Non passare MAI --fresh
#  qui: cancellerebbe il checkpoint e con esso la ripresa.
#
#  I CONFINI NON SI SCRIVONO QUI.  Li da' `evaluate --print-blocks`, che taglia
#  sulle settimane di stima piena — l'unita' che il docstring di `run()` dichiara
#  indipendente per costruzione ("1 stima completa + le sue settimane di riuso",
#  perche' ogni stima parte dal pannello a `as_of` e non dallo stato della
#  precedente).  Tagliare li' e' gratis: la prima settimana di un blocco,
#  che `estimation_weeks` forza comunque a piena, sarebbe stata piena lo stesso.
#
#  SMOKE PASSA DI QUI ANCHE LUI, e prima no: aveva un ramo suo con un blocco
#  unico sull'intero span, quindi il ciclo dei blocchi — proprio la parte che
#  si voleva provare — non veniva mai eseguito.  Ora la modalita' cambia solo
#  `--draws`, e sui due mesi dello SMOKE i blocchi naturali sono due: il
#  percorso multi-blocco viene esercitato davvero, e per giunta in parallelo.
# =============================================================================
log "FASE 5/7 — BVAR: $N_BVAR_BLOCKS blocchi in parallelo, $START .. $END"
cmds=()
DRAWS_FLAG=""
[[ -n "$SMOKE_DRAWS" ]] && DRAWS_FLAG="--draws $SMOKE_DRAWS"
for b in "${BLOCKS[@]}"; do
  bs="${b%% *}"; be="${b##* }"
  cmds+=("\"$PYTHON\" -m src.bvar.evaluate --start $bs --end $be $DRAWS_FLAG \
          > '$LOGS/bvar_${bs}.log' 2>&1")
done
run_pool "$WORKERS" "${cmds[@]}" \
  || fail "uno o piu' blocchi BVAR sono falliti; i log stanno in $LOGS/bvar_*.log"

# =============================================================================
#  FASE 6 — FIGURE, una per finestra forecast
# =============================================================================
log "FASE 6/7 — figure di traiettoria"
# UNA FINESTRA SENZA DATI SI SALTA, NON UCCIDE LA PASSATA.  `figures` esce con
# errore quando nessuna riga cade nella finestra, e qui quell'errore era fatale:
# lo SMOKE (due mesi di calendario) moriva alla prima finestra vuota — 2014-2016
# — dopo aver gia' fatto un'ora e mezza di stime buone.  Sulla passata piena
# tutte e quattro hanno dati, ma una passata RIPRESA a meta' no, e buttare via
# il lavoro fatto per una figura che non si puo' disegnare e' il comportamento
# sbagliato.  Le altre fasi gia' saltano da sole: questa si allinea.
for w in 2007-2010 2014-2016 2019-2021 2024-2025; do
  echo "  finestra $w"
  if ! "$PYTHON" -m src.forecast.figures --window "$w" \
       > "$LOGS/fig_dfm_$w.log" 2>&1; then
    if grep -q "Nessuna riga" "$LOGS/fig_dfm_$w.log"; then
      echo "    [DFM $w] nessun dato in questa finestra — saltata."
    else
      fail "figure DFM $w: vedi $LOGS/fig_dfm_$w.log"
    fi
  fi
  if ! "$PYTHON" -m src.bvar.figures --window "$w" \
       > "$LOGS/fig_bvar_$w.log" 2>&1; then
    if grep -q "Nessuna riga\|Nessun CSV" "$LOGS/fig_bvar_$w.log"; then
      echo "    [BVAR $w] nessun dato in questa finestra — saltata."
    else
      fail "figure BVAR $w: vedi $LOGS/fig_bvar_$w.log"
    fi
  fi
done

# =============================================================================
#  FASE 7 — NY FED, METRICHE, TABELLE PER FASE
# =============================================================================
#  Ordine obbligato: prima il confronto con la Fed e le figure RMSE per
#  orizzonte (una per spec DFM e una per la famiglia BVAR, su ciascuna delle
#  sei finestre), poi le tabelle che leggono tutti i CSV insieme e producono
#  il confronto BVAR-vs-DFM per fase, backcast compreso.
# =============================================================================
log "FASE 7/7 — NY Fed, metriche, tabelle"

# 7a. DFM: le figure RMSE per (spec, finestra) e le tabelle del confronto.
#     Una invocazione per spec, non una per coppia: `nyfed_all` cicla sulle
#     finestre dentro un processo solo e IMPILA le tabelle con una colonna
#     `window`, invece di scrivere nove file per combinazione.  Diciotto
#     invocazioni della vecchia CLI facevano 162 file, di cui solo 18 figure.
cmds=()
for s in "${SPECS[@]}"; do
  cmds+=("\"$PYTHON\" -m src.forecast.nyfed_all --spec $s \
          > '$LOGS/nyfed_${s}.log' 2>&1 || true")
done
run_pool "$WORKERS" "${cmds[@]}"
echo "  confronto NY Fed + figure RMSE per finestra: fatto"
echo "  (le finestre senza trimestri allineati si saltano da sole — e' un dato,"
echo "   non un errore; i dettagli in $LOGS/nyfed_*.log)"

# 7b. BVAR: tabelle + LA figura RMSE con i quattro modelli insieme, su tutte
#     le finestre in una sola invocazione, piu' il log score.
"$PYTHON" -m src.bvar.metrics > "$LOGS/bvar_metrics.log" 2>&1 \
  || fail "metriche BVAR: vedi $LOGS/bvar_metrics.log"
echo "  metriche BVAR + figura RMSE dei quattro modelli: fatto"

# 7c. Le tabelle finali: famiglie, matrici di confronto, e il PER FASE con il
#     backcast come pannello dedicato.
"$PYTHON" -m src.forecast.metrics_tables > "$LOGS/metrics_tables.log" 2>&1 \
  || fail "tabelle finali: vedi $LOGS/metrics_tables.log"
echo "  tabelle di confronto (aggregate e per fase): fatto"

# 7d. La guardia di allineamento, ORA IN VERSIONE SEVERA.  Prima della passata
#     l'assenza di una famiglia era lecita (--pre-run); adesso non lo e' piu':
#     se DFM e BVAR non si incontrano sulle chiavi del join, il confronto non
#     e' verificato e va detto, non lasciato passare in silenzio.
"$PYTHON" -m src.forecast.test_windows \
  || fail "allineamento DFM/BVAR rotto DOPO la passata: le tabelle di confronto
           non sono affidabili.  I CSV restano su disco: si indaga e si
           rigenerano solo le tabelle, non la passata."

# =============================================================================
#  CONSUNTIVO
# =============================================================================
ELAPSED=$(( $(date +%s) - T0 ))
cat <<FINE

================================================================================
  FATTO.  durata effettiva: $(secs_to_hm $ELAPSED)   ($ELAPSED secondi)

  Preventivo dichiarato all'inizio: ~$(secs_to_hm $(( est_dfm + est_bvar ))).
  Se le due cifre divergono molto, il numero da credere e' QUESTO: il
  preventivo estrapola da un blocco corto a estrazioni ridotte.

  Da guardare per primo:
    $OUT/comparison/summary.txt
        le matrici di confronto, la sezione PER FASE e il pannello BACKCAST
    $OUT/comparison/backcast_matrix.csv
        il backcast di ogni metodo per finestra, sul campione comune
    $OUT/bvar/rmse/rmse_per_orizzonte_bvar_*.png
        i quattro BVAR insieme, con il NY Fed in nero
    $OUT/dfm/fed_overlap/rmse/metrics_fed_overlap.txt
        le tabelle del DFM, con n/n_com affiancati

  Ricorda leggendo: solo le colonne _com confrontano le righe fra loro.
  I log di ogni fase stanno in $LOGS/
================================================================================
FINE

if [[ -n "$SMOKE_DRAWS" ]]; then
  cat <<SMOKEFINE

  !!  ERA UNA PROVA: i numeri qui sopra NON sono risultati (due mesi di
  !!  calendario, estrazioni ridotte).  Se sei arrivato fin qui senza errori,
  !!  l'impianto regge e si puo' lanciare la passata vera.
  !!
  !!  PULISCI PRIMA DI LANCIARLA, o i numeri finti si mescolano ai veri:
  !!
  !!    rm -rf output/forecast_weekly/csv output/forecast_weekly/bvar \
  !!           output/forecast_weekly/comparison output/_checkpoint
  !!    rm -rf output/forecast_weekly/dfm/*/rmse

SMOKEFINE
fi
