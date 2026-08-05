param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment,

    [Parameter(Mandatory = $true)]
    [string]$Result
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

Write-Host "== REEVALUATE ==" -ForegroundColor Cyan
Invoke-RepoPython $Cli reevaluate --experiment $Experiment --result $Result
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 reevaluate failed."
}

Write-Host "Evaluator-only pass finished." -ForegroundColor Green
