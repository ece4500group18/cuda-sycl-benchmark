param(
  [string]$Category,
  [string]$Case
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$FilterArgs = @()
if ($Category) { $FilterArgs += @("--category", $Category) }
if ($Case) { $FilterArgs += @("--case", $Case) }

Write-Host "### verify CUDA ###"
& $Python tools/verify_case.py --variant cuda @FilterArgs
Write-Host "### verify SYCL ###"
& $Python tools/verify_case.py --variant sycl @FilterArgs
Write-Host "### generate reports ###"
& $Python tools/generate_report.py
& $Python tools/generate_performance_report.py
Write-Host "Done."
