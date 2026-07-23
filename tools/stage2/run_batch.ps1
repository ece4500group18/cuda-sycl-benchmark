# Run one batch of cases from a case-list file against a Stage 2 experiment.
#
# Artifacts land in the experiment's own directory, so every batch accumulates
# into the same report. Cases already finished are skipped automatically.
#
#   .\tools\stage2\run_batch.ps1 `
#       -Experiment benchmark\stage2\experiments\codebuddy_minimax-m3_full250.json `
#       -CaseFile benchmark\stage2\batches\hard40.txt
#
# Extra cli.py flags go through -ExtraArgs. Pass them as an explicit array;
# PowerShell tries to bind a bare `--skip-preflight` to a parameter of this
# script and fails before the value ever reaches cli.py:
#
#   ... -CaseFile x.txt -ExtraArgs '--skill','oob'
#
# Cases run one at a time so a dropped VPN cannot poison the whole batch. The
# build worker is reachable over VPN only, and `sycl_build.sh` failing with an
# SSH timeout is indistinguishable to the runner from a compiler rejecting the
# code: it records funnel="compile_error", a scored result, which is then
# skipped on resume and stays in the pass rate forever. Worse, the agent reads
# those timeouts as compiler diagnostics and burns its whole repair loop on
# them -- the observed cost was ~1.4M tokens per cell against ~0.9M for a real
# one, for a session that measures nothing.
#
# So: check the worker before each case, and check again after it. If the link
# dropped, delete that case's cells and stop the batch immediately.

param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$CaseFile,
    [int]$SshTimeout = 8,
    [switch]$NoGuard,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

if (-not $env:STAGE2_SSH_TARGET) {
    Write-Error 'STAGE2_SSH_TARGET is not set in this window; the run would fail at preflight.'
}

# Windows PowerShell 5.1 wraps a native command's stderr in an ErrorRecord, so
# with $ErrorActionPreference = 'Stop' an expected ssh timeout would abort the
# script with a NativeCommandError instead of returning a code we can act on.
# Run these probes with the preference relaxed and judge them by exit code.
function Invoke-Quietly {
    param([scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command 2>$null | Out-Null
        return $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
}

function Test-Worker {
    # BatchMode stops ssh from blocking on a password prompt when the tunnel is
    # up but authentication is not.
    $code = Invoke-Quietly {
        & ssh -o BatchMode=yes -o ConnectTimeout=$SshTimeout -o StrictHostKeyChecking=accept-new `
            $env:STAGE2_SSH_TARGET 'true'
    }
    return ($code -eq 0)
}

function Remove-PoisonedCells {
    param([string]$Case)
    # Reuse the same detector the standalone purge tool uses, scoped to one case.
    # Exit 0 = clean, 1 = found something, 2 = error.
    $code = Invoke-Quietly {
        & python (Join-Path $repoRoot 'tools/stage2/purge_network_cells.py') `
            --experiment $Experiment --case $Case
    }
    if ($code -eq 1) {
        Write-Host "  network failure recorded for $Case; deleting its cells" -ForegroundColor Yellow
        & python (Join-Path $repoRoot 'tools/stage2/purge_network_cells.py') `
            --experiment $Experiment --case $Case --purge
        $archiveDir = Join-Path (Split-Path $repoRoot -Parent) 'artifacts/stage2_purged'
        if (Test-Path $archiveDir) {
            $latest = Get-ChildItem -Directory $archiveDir |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($latest) {
                Write-Host ("  archived to: {0}" -f $latest.FullName) -ForegroundColor DarkGray
            }
        }
        return $true
    }
    return $false
}

$cases = Get-Content $CaseFile |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith('#') }

if (-not $cases) { Write-Error "no case ids found in $CaseFile" }

if (-not $NoGuard) {
    Write-Host "checking build worker $env:STAGE2_SSH_TARGET ..." -ForegroundColor Cyan
    if (-not (Test-Worker)) {
        Write-Error 'build worker unreachable; connect the VPN before starting a batch.'
    }
}

function Format-Span {
    param([TimeSpan]$Span)
    if ($Span.TotalHours -ge 1) { return ('{0}h{1:00}m' -f [int]$Span.TotalHours, $Span.Minutes) }
    if ($Span.TotalMinutes -ge 1) { return ('{0}m{1:00}s' -f [int]$Span.TotalMinutes, $Span.Seconds) }
    return ('{0}s' -f [int]$Span.TotalSeconds)
}

Write-Host "batch: $($cases.Count) cases from $CaseFile" -ForegroundColor Cyan

$batchStart = Get-Date
$done = 0
# Already-finished cases return in under a second. Averaging those into the
# estimate would make the remaining time look far shorter than it is, so only
# cases that actually ran feed the projection.
$ranCount = 0
$ranSeconds = 0.0

foreach ($case in $cases) {
    $done++
    $header = "[$done/$($cases.Count)] $case"
    if ($ranCount -gt 0) {
        $remaining = $cases.Count - $done + 1
        $eta = [TimeSpan]::FromSeconds(($ranSeconds / $ranCount) * $remaining)
        $header += "   elapsed $(Format-Span ((Get-Date) - $batchStart))   eta ~$(Format-Span $eta)"
    }
    Write-Host $header -ForegroundColor Cyan
    $caseStart = Get-Date

    if (-not $NoGuard) {
        if (-not (Test-Worker)) {
            Write-Host ''
            Write-Host "build worker went unreachable before $case. Stopping." -ForegroundColor Red
            Write-Host 'Reconnect the VPN and rerun this same batch; finished cases are skipped.' -ForegroundColor Red
            exit 1
        }
    }

    $argv = @('tools/stage2/cli.py', 'run', '--experiment', $Experiment, '--case', $case)
    if ($ExtraArgs) { $argv += $ExtraArgs }
    & python @argv
    $runExit = $LASTEXITCODE

    $caseSeconds = ((Get-Date) - $caseStart).TotalSeconds
    if ($caseSeconds -gt 5) {
        $ranCount++
        $ranSeconds += $caseSeconds
    }

    if (-not $NoGuard) {
        # The link can drop mid-case, so judge by what the case actually wrote
        # rather than by whether ssh answers now.
        if (Remove-PoisonedCells -Case $case) {
            Write-Host ''
            Write-Host "VPN dropped during $case. Its results were deleted and the batch stopped." -ForegroundColor Red
            Write-Host 'Reconnect the VPN and rerun this same batch; finished cases are skipped.' -ForegroundColor Red
            exit 1
        }
    }

    # What the session actually did: funnel, the turn count --max-turns governs,
    # tokens, and the campaign tally so far.
    & python (Join-Path $repoRoot 'tools/stage2/case_status.py') `
        --experiment $Experiment --case $case --totals

    # cli.py run returns nonzero for a verification failure (wrong_output,
    # compile_error, etc.). That's a scored result for the cell, not a batch
    # problem -- the pre-case Test-Worker / mid-case Remove-PoisonedCells
    # guards above are what actually guard the link. Keep going.
    if ($runExit -ne 0) {
        Write-Host "cli.py run exited $runExit on $case; recorded as a failure, continuing the batch." -ForegroundColor Yellow
    }
}

Write-Host "batch complete: $($cases.Count) cases in $(Format-Span ((Get-Date) - $batchStart))" -ForegroundColor Green
& python (Join-Path $repoRoot 'tools/stage2/case_status.py') --experiment $Experiment
exit 0
