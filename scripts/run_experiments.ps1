# scripts/run_experiments.ps1
#
# Lancia gli esperimenti A, B, C rispettando la dipendenza B->A.
#
# MODALITA':
#   NON-FULL (default, S=20): sequenziale A->B->C per semplicita'.
#         Il guadagno del parallelo e' inferiore al rumore sui run brevi.
#   FULL (-full, S=1000):     A e C in PARALLELO, B parte solo dopo A.
#         A e C sono indipendenti; B richiede i JSON di A per la tabella
#         A-vs-B a T=497. Schema: (A||C) -> B -> attendi C se ancora attivo.
#         Risparmio atteso: ~30-40% sul wall-clock (es. 3h->2h su Ryzen 5600G).
#
# Resume sicuro (default, senza -Force):
#   Ogni scenario MC contiene il fingerprint SHA1 di theta_star
#   (Lambda,A,Q,R,Sigma_0,nu_u,nu_eps + base_seed + nu_contam + kappa).
#   Se il JSON esiste e il fingerprint combacia: "skipped (cached)".
#   Se il fingerprint diverge (theta o config cambiati): ricalcola.
#   -Force: rimuove i JSON cached prima di girare (ricalcolo completo).
#
# Uso:
#   .\scripts\run_experiments.ps1              -> small, S=20 (test rapido, sequenziale)
#   .\scripts\run_experiments.ps1 -big         -> big,   S=20
#   .\scripts\run_experiments.ps1 -full        -> small, S=1000 (A||C parallelo)
#   .\scripts\run_experiments.ps1 -big -full   -> big,   S=1000
#   .\scripts\run_experiments.ps1 -both        -> small + big, S=20
#   .\scripts\run_experiments.ps1 -both -full  -> small + big, S=1000 (run tesi)
#   .\scripts\run_experiments.ps1 -Force       -> ignora cache, ricalcola tutto
#   .\scripts\run_experiments.ps1 -full -Force -> full + cache azzerata

param(
    [switch]$big,
    [switch]$both,
    [switch]$full,
    [switch]$Force
)

$root   = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"

if ($both)    { $configs = @("small", "big") }
elseif ($big) { $configs = @("big") }
else          { $configs = @("small") }

$modeLabel = if ($full) { "FULL S=1000 — A||C parallelo, B dopo A" } `
             else       { "RAPIDA S=20 — A->B->C sequenziale" }

Write-Host ""
Write-Host "=== Run esperimenti ===" -ForegroundColor Cyan
Write-Host "Config  : $($configs -join ' + ')"
Write-Host "Modalita: $modeLabel"
Write-Host "Force   : $(if ($Force) {'SI - cache azzerata'} else {'NO - fingerprint resume'})"
Write-Host ""

# ── Helper: rimuovi JSON cached di uno scenario ───────────────────────────────
function Clear-ScenarioCache {
    param([string]$outDir, [string]$label)
    if (Test-Path $outDir) {
        $jsons = @(Get-ChildItem -Path $outDir -Filter "mc_*.json" -ErrorAction SilentlyContinue)
        if ($jsons.Count -gt 0) {
            $jsons | Remove-Item -Force
            Write-Host "  [force] $label : rimossi $($jsons.Count) JSON cached" -ForegroundColor Yellow
        } else {
            Write-Host "  [force] $label : nessun JSON cached" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [force] $label : cartella assente" -ForegroundColor Yellow
    }
}

# ── Helper: verifica prerequisito A per B ────────────────────────────────────
function Test-ExpAOutput {
    param([string]$cfg)
    $aDir   = Join-Path $root "output\monte_carlo\$cfg\expA"
    $aJsons = @(Get-ChildItem -Path $aDir -Filter "mc_*.json" -ErrorAction SilentlyContinue)
    if ($aJsons.Count -eq 0) {
        Write-Host "  [ERRORE] Nessun JSON in output\monte_carlo\$cfg\expA" -ForegroundColor Red
        Write-Host "           Exp B richiede l'output di Exp A (tabella A-vs-B)." -ForegroundColor Red
        return $false
    }
    Write-Host "  [ok] Prerequisito A: $($aJsons.Count) JSON trovati -> procedo con B" -ForegroundColor DarkGreen
    return $true
}

# ── Helper: lancia esperimento in foreground e controlla exit code ────────────
function Invoke-Experiment {
    param([string]$label, [string]$script, [string]$cfg, [bool]$isFull)
    Write-Host ""
    Write-Host "  [calcolo] Avvio $label ..." -ForegroundColor Green
    $t0 = Get-Date
    $pyArgs = @("--$cfg"); if ($isFull) { $pyArgs += "--full" }
    & $python $script @pyArgs
    $ec      = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    if ($ec -ne 0) {
        Write-Host "  [ERRORE] $label fallito (exit $ec) dopo ${elapsed}s" -ForegroundColor Red
        return $false
    }
    Write-Host "  [ok] $label completato in ${elapsed}s" -ForegroundColor Green
    return $true
}


# ══════════════════════════════════════════════════════════════════════════════
foreach ($cfg in $configs) {
    Write-Host "━━━ Config: $cfg ━━━" -ForegroundColor Cyan
    Write-Host ""

    $scriptA = Join-Path $root "src\run_experiment_a.py"
    $scriptB = Join-Path $root "src\run_experiment_b.py"
    $scriptC = Join-Path $root "src\run_experiment_c.py"
    $outA    = Join-Path $root "output\monte_carlo\$cfg\expA"
    $outB    = Join-Path $root "output\monte_carlo\$cfg\expB"
    $outC    = Join-Path $root "output\monte_carlo\$cfg\expC"

    # ── MODALITA' NON-FULL: A->B->C sequenziale ───────────────────────────────
    if (-not $full) {
        foreach ($pair in @(
            @{ Label="Exp_A_$cfg"; Script=$scriptA; OutDir=$outA },
            @{ Label="Exp_B_$cfg"; Script=$scriptB; OutDir=$outB },
            @{ Label="Exp_C_$cfg"; Script=$scriptC; OutDir=$outC }
        )) {
            if ($Force) { Clear-ScenarioCache $pair.OutDir $pair.Label }

            if ($pair.Label -like "Exp_B_*") {
                if (-not (Test-ExpAOutput $cfg)) { exit 1 }
            }

            if (-not (Invoke-Experiment $pair.Label $pair.Script $cfg $false)) { exit 1 }
            Write-Host ""
        }
        Write-Host "━━━ Config $cfg completata ━━━" -ForegroundColor Cyan
        Write-Host ""
        continue
    }

    # ── MODALITA' FULL: A||C in parallelo, B dopo A ───────────────────────────
    Write-Host "  [parallel] Avvio A e C in parallelo (A||C)..." -ForegroundColor Cyan

    if ($Force) {
        Clear-ScenarioCache $outA "Exp_A_$cfg"
        Clear-ScenarioCache $outC "Exp_C_$cfg"
    }

    $fullFlag = $true
    $jobA = Start-Job -Name "Exp_A_$cfg" -ScriptBlock {
        param($py, $sc, $cfg_)
        & $py $sc "--$cfg_" "--full" 2>&1
    } -ArgumentList $python, $scriptA, $cfg

    $jobC = Start-Job -Name "Exp_C_$cfg" -ScriptBlock {
        param($py, $sc, $cfg_)
        & $py $sc "--$cfg_" "--full" 2>&1
    } -ArgumentList $python, $scriptC, $cfg

    Write-Host "  Avviati in background: $($jobA.Name) (Job $($jobA.Id))  |  $($jobC.Name) (Job $($jobC.Id))" -ForegroundColor DarkGray
    Write-Host "  In attesa di A (necessario per B)..." -ForegroundColor DarkGray
    Write-Host ""

    # Aspetta A, monitora ogni 60s
    while ($jobA.State -eq "Running") {
        Start-Sleep -Seconds 60
        $stateC = $jobC.State
        Write-Host "  [watch] $(Get-Date -Format 'HH:mm:ss')  A=$($jobA.State)  C=$stateC"
    }

    # Raccoglie output di A e verifica
    $outA_log = Receive-Job $jobA
    $outA_log | ForEach-Object { Write-Host "  [A] $_" }
    Write-Host ""

    if ($jobA.State -ne "Completed") {
        Write-Host "  [ERRORE] Exp_A_$cfg fallito (stato: $($jobA.State))" -ForegroundColor Red
        Write-Host "           Interruzione: fermo C e esco." -ForegroundColor Red
        Stop-Job $jobC; Remove-Job $jobA, $jobC -Force
        exit 1
    }

    # Prerequisito: verifica JSON di A prima di avviare B
    if (-not (Test-ExpAOutput $cfg)) {
        Stop-Job $jobC; Remove-Job $jobA, $jobC -Force
        exit 1
    }

    # Lancia B in foreground (C continua in background)
    if ($Force) { Clear-ScenarioCache $outB "Exp_B_$cfg" }
    if (-not (Invoke-Experiment "Exp_B_$cfg" $scriptB $cfg $true)) {
        Stop-Job $jobC; Remove-Job $jobA, $jobC -Force
        exit 1
    }
    Write-Host ""

    # Aspetta C (se non ha gia' finito durante B)
    if ($jobC.State -eq "Running") {
        Write-Host "  [wait] B completato. C ancora in esecuzione, attendo..." -ForegroundColor DarkGray
        while ($jobC.State -eq "Running") {
            Start-Sleep -Seconds 60
            Write-Host "  [watch] $(Get-Date -Format 'HH:mm:ss')  C=$($jobC.State)"
        }
    }

    $outC_log = Receive-Job $jobC
    $outC_log | ForEach-Object { Write-Host "  [C] $_" }
    Write-Host ""

    if ($jobC.State -ne "Completed") {
        Write-Host "  [ERRORE] Exp_C_$cfg fallito (stato: $($jobC.State))" -ForegroundColor Red
        Remove-Job $jobA, $jobC -Force
        exit 1
    }

    Remove-Job $jobA, $jobC -Force
    Write-Host "━━━ Config $cfg completata (A||C + B) ━━━" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "=== Tutti gli esperimenti completati ===" -ForegroundColor Green
