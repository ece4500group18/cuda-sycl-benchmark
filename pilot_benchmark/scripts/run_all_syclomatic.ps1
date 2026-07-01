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

Write-Host "### SYCLomatic migration ###"
& $Python tools/run_syclomatic.py @FilterArgs
Write-Host "### generate report ###"
& $Python tools/generate_report.py
Write-Host "Done."
