# Run one batch of cases from a case-list file against a Stage 2 experiment.
#
# Artifacts land in the experiment's own directory, so every batch accumulates
# into the same report. Cases already finished are skipped automatically.
#
#   .\tools\stage2\run_batch.ps1 `
#       -Experiment benchmark\stage2\experiments\codebuddy_minimax-m3_full250.json `
#       -CaseFile benchmark\stage2\batches\hard50.txt
#
# Pass extra CLI flags after --, e.g. ... -CaseFile x.txt -- --skill oob

param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$CaseFile,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

if (-not $env:STAGE2_SSH_TARGET) {
    Write-Error 'STAGE2_SSH_TARGET is not set in this window; the run would fail at preflight.'
}

$cases = Get-Content $CaseFile |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith('#') }

if (-not $cases) { Write-Error "no case ids found in $CaseFile" }

$argv = @('tools/stage2/cli.py', 'run', '--experiment', $Experiment)
foreach ($case in $cases) { $argv += '--case'; $argv += $case }
if ($ExtraArgs) { $argv += $ExtraArgs | Where-Object { $_ -ne '--' } }

Write-Host "batch: $($cases.Count) cases from $CaseFile" -ForegroundColor Cyan
& python @argv
exit $LASTEXITCODE
