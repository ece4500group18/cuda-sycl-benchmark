param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment,

    [string]$Case = "vectorAdd",

    [string]$PreflightOutput = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $PreflightOutput) {
    $experimentName = [System.IO.Path]::GetFileNameWithoutExtension($Experiment)
    $PreflightOutput = Join-Path $RepoRoot "artifacts\stage2\preflight-$experimentName.json"
}

Write-Host "== PLAN ==" -ForegroundColor Cyan
& python (Join-Path $RepoRoot "tools\stage2\cli.py") plan --experiment $Experiment
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 plan failed."
}

Write-Host "== PREFLIGHT ==" -ForegroundColor Cyan
& python (Join-Path $RepoRoot "tools\stage2\cli.py") preflight --experiment $Experiment --output $PreflightOutput
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 preflight failed."
}

Write-Host "== RUN ==" -ForegroundColor Cyan
& python (Join-Path $RepoRoot "tools\stage2\cli.py") run --experiment $Experiment --case $Case
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 run failed."
}

Write-Host "Done." -ForegroundColor Green
