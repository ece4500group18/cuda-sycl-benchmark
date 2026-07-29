param(
    [string[]]$ExperimentFiles = @(),
    [string]$ArtifactRoot = '',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-RepoPython {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    & py -3 @Args
    return $LASTEXITCODE
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Cli = Join-Path $RepoRoot 'tools\stage2\cli.py'

if (-not $ArtifactRoot) {
    $ArtifactRoot = Join-Path $RepoRoot 'artifacts\stage2'
}

if (-not $ExperimentFiles -or $ExperimentFiles.Count -eq 0) {
    $ExperimentFiles = @(
        (Join-Path $RepoRoot 'benchmark\stage2\experiments\trae_deepseek-v4pro.json'),
        (Join-Path $RepoRoot 'benchmark\stage2\experiments\trae_glm-5.2.json'),
        (Join-Path $RepoRoot 'benchmark\stage2\experiments\trae_kimi-v3.json'),
        (Join-Path $RepoRoot 'benchmark\stage2\experiments\trae_minimax-m3.json')
    )
}

$experimentById = @{}
foreach ($experimentFile in $ExperimentFiles) {
    if (-not (Test-Path $experimentFile)) {
        Write-Warning "Skipping missing experiment file: $experimentFile"
        continue
    }
    $experiment = Get-Content $experimentFile -Raw | ConvertFrom-Json
    $experimentById[$experiment.experiment_id] = $experimentFile
}

$errorFiles = Get-ChildItem $ArtifactRoot -Recurse -Filter harness_error.json -ErrorAction SilentlyContinue
if (-not $errorFiles) {
    Write-Host 'No harness_error.json files found to rerun.' -ForegroundColor Green
    exit 0
}

$groups = @{}
foreach ($file in $errorFiles) {
    $record = Get-Content $file.FullName -Raw | ConvertFrom-Json
    if (-not $experimentById.ContainsKey($record.experiment_id)) {
        continue
    }
    $key = "$($record.experiment_id)||$($record.model)||$($record.skill_condition)"
    if (-not $groups.ContainsKey($key)) {
        $groups[$key] = [PSCustomObject]@{
            experiment_id = [string]$record.experiment_id
            experiment_file = [string]$experimentById[$record.experiment_id]
            model = [string]$record.model
            skill = [string]$record.skill_condition
            cases = New-Object System.Collections.Generic.List[string]
        }
    }
    if (-not $groups[$key].cases.Contains([string]$record.case_id)) {
        $groups[$key].cases.Add([string]$record.case_id)
    }
}

if ($groups.Count -eq 0) {
    Write-Host 'Found harness_error files, but none belong to the configured TRAE experiments.' -ForegroundColor Yellow
    exit 0
}

foreach ($group in ($groups.Values | Sort-Object experiment_id, skill, model)) {
    $caseList = $group.cases | Sort-Object
    Write-Host ''
    Write-Host ("=== Rerun {0} / {1} / {2} ===" -f $group.experiment_id, $group.model, $group.skill) -ForegroundColor Cyan
    Write-Host ('cases: ' + ($caseList -join ', '))

    $args = @(
        $Cli, 'run',
        '--experiment', $group.experiment_file,
        '--model', $group.model,
        '--skill', $group.skill
    )
    foreach ($caseId in $caseList) {
        $args += @('--case', $caseId)
    }

    if ($DryRun) {
        $pretty = @('py','-3') + $args
        Write-Host ('DRY RUN: ' + ($pretty -join ' '))
        continue
    }

    Invoke-RepoPython @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning ("Rerun did not fully succeed: {0} / {1} / {2}" -f $group.experiment_id, $group.model, $group.skill)
    }
}
