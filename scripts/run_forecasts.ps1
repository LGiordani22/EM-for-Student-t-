# scripts/run_forecasts.ps1
#
# Orchestrates rolling_nowcast runs across multiple periods and configurations,
# with controlled parallelism and automatic resume.
#
# HARDWARE: Ryzen 5 5600G (6c/12t), 32 GB RAM.
# MaxParallel DEFAULT = 3 for "small" (20 series):
#   Each Student-t DFM fit is sequential at the Python level; BLAS parallelism
#   kicks in for matrix ops but small (20x20) matrices don't spawn many threads.
#   3 concurrent processes sit well within the 12 logical threads and stay within
#   RAM budget (~1-2 GB per small run). For "big" (50 series, ~3x heavier), use
#   -MaxParallel 2 or 1 to avoid memory pressure and CPU contention.
#
# Config order when -Config both: ALL small runs first, then ALL big.
#   Rationale: small (20 series) is lighter and finishes faster, giving early
#   validation before committing CPU hours to the heavier big fits.
#
# Usage examples -- see bottom of this file for a full reference.

param(
    [ValidateSet("small", "big", "both")]
    [string]$Config = "small",
    [int]$MaxParallel = 3,
    [switch]$Force,
    [switch]$Figures
)

# ══════════════════════════════════════════════════════════════════════════════
# PERIODI -- aggiungere un periodo = aggiungere una riga qui sotto.
# Tutti devono stare in 1999–2024: il 2025 richiede vintage non ancora scaricati.
# Per girare un solo periodo: commentare gli altri con #.
# ══════════════════════════════════════════════════════════════════════════════
$Periods = @(
    @{ Start="2008-01"; End="2009-12"; Label="crisi_2008"      },
    @{ Start="2015-01"; End="2015-12"; Label="calma_2015"      },
    @{ Start="2020-01"; End="2020-12"; Label="covid_2020"      },
    @{ Start="2001-01"; End="2001-12"; Label="dotcom_2001"     },
    @{ Start="2011-01"; End="2012-12"; Label="debito_eu_2011"  },
    @{ Start="2018-01"; End="2019-12"; Label="dazi_2018"       },
    @{ Start="2022-01"; End="2023-12"; Label="inflazione_2022" }
    # Per aggiungere un periodo:
    # @{ Start="YYYY-MM"; End="YYYY-MM"; Label="nome_breve" },
)
# ══════════════════════════════════════════════════════════════════════════════

$root   = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"

$configList   = switch ($Config) {
    "small" { @("small") }
    "big"   { @("big") }
    "both"  { @("small", "big") }
}
$bigRequested = $configList -contains "big"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Get-NowcastCsvPath([string]$cfg, [string]$start, [string]$end) {
    return Join-Path $root "output\forecast_realtime\csv\$cfg\rolling_nowcast_${start}_${end}.csv"
}

function Get-CsvDataRowCount([string]$path) {
    if (-not (Test-Path $path)) { return 0 }
    try {
        $n = (Get-Content $path -ErrorAction Stop | Measure-Object -Line).Lines
        return [math]::Max(0, $n - 1)   # exclude header
    } catch { return 0 }
}

# ── Build ordered run list (small first, then big) ────────────────────────────
$plannedRuns = [System.Collections.Generic.List[hashtable]]::new()
foreach ($cfg in $configList) {
    foreach ($p in $Periods) {
        $plannedRuns.Add(@{ Cfg=$cfg; Start=$p.Start; End=$p.End; Label=$p.Label })
    }
}

$bigRunCount   = ($plannedRuns | Where-Object { $_.Cfg -eq "big" }).Count
$smallRunCount = $plannedRuns.Count - $bigRunCount

# Count runs that already have CSV data (pre-Force state, for the header display)
$csvHaveData = 0
foreach ($r in $plannedRuns) {
    if ((Get-CsvDataRowCount (Get-NowcastCsvPath $r.Cfg $r.Start $r.End)) -gt 0) {
        $csvHaveData++
    }
}

# ── Header ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== run_forecasts.ps1 ===" -ForegroundColor Cyan
Write-Host "Config       : $Config"
Write-Host "Periodi      : $($Periods.Count)"
Write-Host "Run small    : $smallRunCount"
if ($bigRequested) {
    Write-Host "Run big      : $bigRunCount"
}
Write-Host "CSV esistenti: $csvHaveData  (verranno ripresi o saltati se gia' completi)"
Write-Host "Da avviare   : $([math]::Max(0, ($plannedRuns.Count) - $csvHaveData))"
Write-Host "MaxParallel  : $MaxParallel  (pool: max $MaxParallel run contemporanei)"
Write-Host "Force        : $(if ($Force) {'SI -- CSV rimossi prima del lancio, ricalcolo da zero'} else {'NO -- resume automatico (tripla as_of/target/method)'})"
Write-Host "Figures      : $(if ($Figures) {'SI -- generate al termine di ogni run completato'} else {'NO -- usa figures.py separatamente'})"
Write-Host ""

# Note on series coverage for early periods (big config)
if ($bigRequested) {
    $earlyBig = $Periods | Where-Object { [int]($_.Start.Split("-")[0]) -lt 2003 }
    if ($earlyBig) {
        Write-Host "  [nota-big] Periodi che iniziano prima del 2003:" -ForegroundColor DarkGray
        foreach ($p in $earlyBig) {
            Write-Host "    '$($p.Label)' ($($p.Start)): T10YIE disponibile dal 2003, JTSJOL da dic-2000." -ForegroundColor DarkGray
            Write-Host "    Le prime osservazioni avranno NaN -- il filtro di Kalman li gestisce," -ForegroundColor DarkGray
            Write-Host "    ma aspettarsi stime meno precise nelle primissime date." -ForegroundColor DarkGray
        }
        Write-Host "  [nota-big] WPSFD49207 disponibile dal 2016-03: allNaN nei run pre-2016 (atteso)." -ForegroundColor DarkGray
        Write-Host ""
    }
}

# ── Populate run queue ────────────────────────────────────────────────────────
$runQueue = [System.Collections.Generic.Queue[hashtable]]::new()
foreach ($r in $plannedRuns) {
    $runQueue.Enqueue($r)
}

if ($runQueue.Count -eq 0) {
    Write-Host "Nessun run da eseguire (tutti saltati o config non disponibile)." -ForegroundColor Yellow
    exit 0
}

# ── Job pool state ────────────────────────────────────────────────────────────
$activeJobs = @{}    # int JobId -> hashtable with metadata
$results    = [System.Collections.Generic.List[pscustomobject]]::new()

# ── Launch one job from the queue ─────────────────────────────────────────────
function Invoke-StartJob {
    param([System.Collections.Generic.Queue[hashtable]]$Queue)
    if ($Queue.Count -eq 0) { return }

    $run = $Queue.Dequeue()
    $cfg = $run.Cfg
    $lbl = $run.Label
    $st  = $run.Start
    $en  = $run.End
    $csv = Get-NowcastCsvPath $cfg $st $en

    $rowsBefore = Get-CsvDataRowCount $csv

    if ($script:Force) {
        if (Test-Path $csv) {
            Remove-Item $csv -Force
            Write-Host "  [force] $lbl ($cfg): rimosso CSV ($rowsBefore righe)" -ForegroundColor Yellow
        }
        $rowsBefore = 0
    }

    $statusMsg = if ($rowsBefore -gt 0) {
        "ripreso ($rowsBefore righe gia' nel CSV)"
    } else {
        "avviato da zero"
    }
    Write-Host "  [avvio] $lbl ($cfg)  $st -> $en   $statusMsg" -ForegroundColor Green

    $job = Start-Job -Name "${cfg}_${lbl}" -ScriptBlock {
        param($py, $root_, $cfg_, $start_, $end_)
        Push-Location $root_
        & $py -m src.forecast.rolling_nowcast --start $start_ --end $end_ "--$cfg_" 2>&1
        Pop-Location
    } -ArgumentList $script:python, $script:root, $cfg, $st, $en

    $script:activeJobs[$job.Id] = @{
        Job        = $job
        Label      = $lbl
        Cfg        = $cfg
        Start      = $st
        End        = $en
        CsvPath    = $csv
        StartTime  = Get-Date
        RowsBefore = $rowsBefore
    }
}

# ── Start initial batch ───────────────────────────────────────────────────────
$initial = [math]::Min($MaxParallel, $runQueue.Count)
for ($i = 0; $i -lt $initial; $i++) { Invoke-StartJob $runQueue }
Write-Host ""

# ── Main pool loop ────────────────────────────────────────────────────────────
while ($activeJobs.Count -gt 0) {

    $jobIds = [int[]]@($activeJobs.Keys)
    $done   = Wait-Job -Id $jobIds -Any -Timeout 30

    if ($null -eq $done) {
        $running = ($activeJobs.Values | ForEach-Object { "$($_.Label)($($_.Cfg))" }) -join "  "
        Write-Host "  [watch] $(Get-Date -Format 'HH:mm:ss')  in corso: $running"
        continue
    }

    $meta = $activeJobs[$done.Id]
    $activeJobs.Remove($done.Id)

    $elapsed   = [int]((Get-Date) - $meta.StartTime).TotalSeconds
    $output    = @(Receive-Job $done 2>&1)
    Remove-Job $done -Force

    $rowsAfter = Get-CsvDataRowCount $meta.CsvPath
    $added     = $rowsAfter - $meta.RowsBefore

    if ($done.State -eq "Completed") {
        if ($added -eq 0 -and $meta.RowsBefore -gt 0) {
            Write-Host "  [gia' completo] $($meta.Label) ($($meta.Cfg))  nessuna riga aggiunta ($rowsAfter nel CSV)" -ForegroundColor DarkGreen
        } else {
            Write-Host "  [completato] $($meta.Label) ($($meta.Cfg))  ${elapsed}s   +$added righe  (totale CSV: $rowsAfter)" -ForegroundColor Green
        }

        if ($Figures -and $rowsAfter -gt 0) {
            Write-Host "  [figure] Generazione figure per $($meta.Label)..." -ForegroundColor DarkGray
            Push-Location $root
            & $python -m src.forecast.figures "--$($meta.Cfg)" --csv $meta.CsvPath 2>&1 |
                ForEach-Object { Write-Host "    [fig] $_" }
            Pop-Location
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [warn] figures.py exit $LASTEXITCODE per $($meta.Label)" -ForegroundColor Yellow
            }
        }

        $results.Add([pscustomobject]@{
            Label   = $meta.Label; Cfg = $meta.Cfg
            Status  = "OK"; Elapsed = $elapsed; Rows = $rowsAfter
        })

    } else {
        Write-Host "  [ERRORE] $($meta.Label) ($($meta.Cfg))  stato=$($done.State)  dopo ${elapsed}s" -ForegroundColor Red
        $results.Add([pscustomobject]@{
            Label   = $meta.Label; Cfg = $meta.Cfg
            Status  = "FAIL"; Elapsed = $elapsed; Rows = $rowsAfter
        })
    }

    # Print captured Python output (indented, after status line)
    if ($output.Count -gt 0) {
        Write-Host ""
        $output | ForEach-Object { Write-Host "    $_" }
    }
    Write-Host ""

    # Refill pool
    $free = $MaxParallel - $activeJobs.Count
    for ($i = 0; $i -lt $free -and $runQueue.Count -gt 0; $i++) {
        Invoke-StartJob $runQueue
    }
}

# ── Final summary ─────────────────────────────────────────────────────────────
Write-Host "=== Riepilogo run_forecasts ===" -ForegroundColor Cyan
$ok   = @($results | Where-Object { $_.Status -eq "OK" })
$fail = @($results | Where-Object { $_.Status -eq "FAIL" })
Write-Host "Completati : $($ok.Count)  |  Falliti: $($fail.Count)"

if ($ok.Count -gt 0) {
    Write-Host ""
    Write-Host "Dettaglio run OK:" -ForegroundColor DarkGreen
    foreach ($r in $ok) {
        Write-Host "  $($r.Label) ($($r.Cfg))  $($r.Elapsed)s  $($r.Rows) righe nel CSV" -ForegroundColor DarkGreen
    }
}

if ($fail.Count -gt 0) {
    Write-Host ""
    Write-Host "Run falliti (riesegui: il resume riparte da dove si e' fermato):" -ForegroundColor Red
    foreach ($f in $fail) {
        Write-Host "  - $($f.Label) ($($f.Cfg))" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Tutti i run completati." -ForegroundColor Green

# ══════════════════════════════════════════════════════════════════════════════
# ESEMPI D'USO (copia-incolla dalla root del progetto)
#
# Small, tutti i periodi (default):
#   .\scripts\run_forecasts.ps1
#
# Solo un periodo (commentare gli altri in $Periods, poi):
#   .\scripts\run_forecasts.ps1
#
# Solo small, MaxParallel esplicito:
#   .\scripts\run_forecasts.ps1 -MaxParallel 2
#
# Ricalcolo da zero (rimuove i CSV prima di ogni lancio):
#   .\scripts\run_forecasts.ps1 -Force
#
# Con generazione figure al termine di ogni run:
#   .\scripts\run_forecasts.ps1 -Figures
#
# Config big, tutti i periodi (MaxParallel 2 consigliato per RAM):
#   .\scripts\run_forecasts.ps1 -Config big -MaxParallel 2
#
# Solo un periodo big (commentare gli altri in $Periods, poi):
#   .\scripts\run_forecasts.ps1 -Config big -MaxParallel 1
#
# Small + big, MaxParallel ridotto, con figure:
#   .\scripts\run_forecasts.ps1 -Config both -MaxParallel 2 -Figures
#
# MaxParallel consigliato per Ryzen 5 5600G (6c/12t, 32 GB):
#   -MaxParallel 3  per small (default)  -- 20 serie, ~85s/fit student_t, RAM ok
#   -MaxParallel 2  per big              -- 50 serie, ~40-90s/fit, piu' RAM
#   -MaxParallel 1  se si vuole output sequenziale leggibile in tempo reale
# ══════════════════════════════════════════════════════════════════════════════
