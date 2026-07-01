param(
  [string]$Category,
  [string]$Case,
  [int]$Repeat = 5,
  [int]$Warmup = 1
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$FilterArgs = @()
if ($Category) { $FilterArgs += @("--category", $Category) }
if ($Case) { $FilterArgs += @("--case", $Case) }

Write-Host "### collect cases ###"
& $Python tools/collect_cases.py @FilterArgs
Write-Host "### build CUDA ###"
& $Python tools/build_cuda.py @FilterArgs
Write-Host "### run CUDA ###"
& $Python tools/run_case.py --variant cuda @FilterArgs
Write-Host "### verify CUDA ###"
& $Python tools/verify_case.py --variant cuda @FilterArgs
Write-Host "### benchmark CUDA ###"
& $Python tools/benchmark_case.py --variant cuda --repeat $Repeat --warmup $Warmup @FilterArgs
Write-Host "### generate reports ###"
& $Python tools/generate_report.py
& $Python tools/generate_performance_report.py
Write-Host "Done."
