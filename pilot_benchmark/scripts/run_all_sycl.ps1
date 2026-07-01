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

Write-Host "### build SYCL ###"
& $Python tools/build_sycl.py @FilterArgs
Write-Host "### run SYCL ###"
& $Python tools/run_case.py --variant sycl @FilterArgs
Write-Host "### verify SYCL ###"
& $Python tools/verify_case.py --variant sycl @FilterArgs
Write-Host "### benchmark SYCL ###"
& $Python tools/benchmark_case.py --variant sycl --repeat $Repeat --warmup $Warmup @FilterArgs
Write-Host "### generate reports ###"
& $Python tools/generate_report.py
& $Python tools/generate_performance_report.py
Write-Host "Done."
