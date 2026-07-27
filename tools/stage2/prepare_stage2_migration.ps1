param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment,

    [string]$Case = "vectorAdd",

    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Cli = Join-Path $RepoRoot "tools\stage2\cli.py"

function Invoke-RepoPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    & py -3 @Args
    return $LASTEXITCODE
}

if (-not $env:STAGE2_SSH_TARGET) {
    $env:STAGE2_SSH_TARGET = "placeholder@invalid.invalid"
    Write-Warning "STAGE2_SSH_TARGET was not set. Using placeholder@invalid.invalid so the model phase can run and produce sandbox/main.sycl.cpp before evaluator SSH fails."
}

Write-Host "== PLAN ==" -ForegroundColor Cyan
Invoke-RepoPython $Cli plan --experiment $Experiment --case $Case
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 plan failed."
}

Write-Host "== MODEL RUN (skip preflight) ==" -ForegroundColor Cyan
$runArgs = @(
    $Cli, "run",
    "--experiment", $Experiment,
    "--case", $Case,
    "--skip-preflight"
)
if ($Overwrite) {
    $runArgs += "--overwrite"
}
Invoke-RepoPython @runArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Stage 2 run returned non-zero. This can still be useful if sandbox/main.sycl.cpp was produced."
    exit $LASTEXITCODE
}

Write-Host "Model phase finished." -ForegroundColor Green
