param(
    [string[]]$ExperimentFiles = @(
        'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\benchmark\stage2\experiments\trae_deepseek-v4pro.json',
        'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\benchmark\stage2\experiments\trae_glm-5.2.json',
        'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\benchmark\stage2\experiments\trae_kimi-v3.json',
        'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\benchmark\stage2\experiments\trae_minimax-m3.json'
    ),
    [string]$ArtifactRoot = 'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\artifacts\stage2',
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

$Cli = 'E:\SJTU Courses\Senior Summer\capstone\cuda-sycl-benchmark\tools\stage2\cli.py'

$experimentById = @{}
foreach ($experimentFile in $ExperimentFiles) {
    if (-not (Test-Path $experimentFile)) {
        Write-Warning "跳过不存在的 experiment 文件：$experimentFile"
        continue
    }
    $experiment = Get-Content $experimentFile -Raw | ConvertFrom-Json
    $experimentById[$experiment.experiment_id] = $experimentFile
}

$errorFiles = Get-ChildItem $ArtifactRoot -Recurse -Filter harness_error.json -ErrorAction SilentlyContinue
if (-not $errorFiles) {
    Write-Host '没有发现需要补跑的 harness_error.json。' -ForegroundColor Green
    exit 0
}

$groups = @{}
foreach ($file in $errorFiles) {
    $record = Get-Content $file.FullName -Raw | ConvertFrom-Json
    if (-not $experimentById.ContainsKey($record.experiment_id)) {
        continue
    }
    $key = '{0}|{1}|{2}' -f $record.experiment_id, $record.model, $record.skill_condition
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
    Write-Host '发现的 harness_error 不属于当前这四个 TRAE experiment，无需补跑。' -ForegroundColor Yellow
    exit 0
}

foreach ($group in ($groups.Values | Sort-Object experiment_id, skill, model)) {
    $caseList = $group.cases | Sort-Object
    Write-Host ''
    Write-Host ('=== 补跑 {0} / {1} / {2} ===' -f $group.experiment_id, $group.model, $group.skill) -ForegroundColor Cyan
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
        Write-Warning ('补跑未完全成功：{0} / {1} / {2}' -f $group.experiment_id, $group.model, $group.skill)
    }
}
