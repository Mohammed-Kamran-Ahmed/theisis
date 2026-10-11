# run_multi_seed_evaluation.ps1 (fixed)
# ========================================
# Fixes from the previous version:
# 1. Removed `$ErrorActionPreference = "Stop"` — on PowerShell 7.3+, this
#    combines with native-command stderr handling to turn a Python
#    traceback (written to stderr) into a PowerShell-level terminating
#    error, corrupting the rest of the script's execution. We rely on
#    $LASTEXITCODE checks instead, which is the correct, version-safe way
#    to detect failures from external programs.
# 2. Added a short delay + automatic retry around each Python invocation,
#    to work around an intermittent Windows issue where a DLL a process
#    just finished using (torch's c10.dll here) gets briefly re-locked —
#    commonly by antivirus real-time scanning — right as the next process
#    tries to load it. If this keeps happening, the more permanent fix is
#    adding a Windows Defender (or your AV) exclusion for your venv folder
#    (thesis_env) and/or the project folder, since large DLLs like torch's
#    are the most common trigger for this.
#
# Usage:
#   .\run_multi_seed_evaluation.ps1
#   .\run_multi_seed_evaluation.ps1 -Seeds 42,1,7,123,2024

param(
    [int[]]$Seeds = @(42, 1, 7)
)

# Avoids PowerShell 7.3+ treating native stderr output as a terminating error.
$PSNativeCommandUseErrorActionPreference = $false

New-Item -ItemType Directory -Force -Path ".\multi_seed_runs" | Out-Null

function Run-PythonStep {
    param(
        [string[]]$Arguments,
        [string]$Label,
        [int]$MaxRetries = 2
    )

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Write-Host ""
        Write-Host "--- $Label (attempt $attempt/$MaxRetries) ---"
        & python @Arguments
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        Write-Host "Attempt $attempt failed (exit code $LASTEXITCODE)."
        if ($attempt -lt $MaxRetries) {
            Write-Host "Waiting 5 seconds before retry (helps if this is the DLL-lock race condition)..."
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "FAILED: $Label did not succeed after $MaxRetries attempts."
    return $false
}

foreach ($seed in $Seeds) {
    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "  RUNNING FULL PIPELINE FOR SEED $seed"
    Write-Host "======================================================================"

    $ok = Run-PythonStep -Arguments @("utils/data_split.py", "--seed", $seed) -Label "Data split (seed=$seed)"
    if (-not $ok) { continue }
    Start-Sleep -Seconds 2

    $ok = Run-PythonStep -Arguments @("stage4_contrastive_learning.py") -Label "Stage 4 encoder training (seed=$seed)"
    if (-not $ok) { continue }
    Start-Sleep -Seconds 2

    $ok = Run-PythonStep -Arguments @("stage5_extract_and_evaluate.py") -Label "Stage 5 embedding extraction (seed=$seed)"
    if (-not $ok) { continue }
    Start-Sleep -Seconds 2

    $ok = Run-PythonStep -Arguments @("stage6_downstream_evaluation.py") -Label "Stage 6 evaluation (seed=$seed)"
    if (-not $ok) { continue }

    Copy-Item ".\stage6_metrics_summary.json" ".\multi_seed_runs\seed_${seed}_summary.json" -Force
    Write-Host "Saved .\multi_seed_runs\seed_${seed}_summary.json"
}

Write-Host ""
Write-Host "All seeds attempted. Run: python aggregate_multi_seed_results.py"