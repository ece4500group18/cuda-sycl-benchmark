$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "Repo root: $Root"
Write-Host ""

function Check-Tool {
  param([string]$Name, [string[]]$ToolArgs = @())
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    $line = (& $Name @ToolArgs 2>$null | Select-Object -First 1)
    Write-Host ("  [present] {0,-12} {1}" -f $Name, $line)
  } else {
    Write-Host ("  [MISSING] {0,-12}" -f $Name)
  }
}

function Find-Vcvars64 {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" 2>$null |
      Select-Object -First 1
    if ($found -and (Test-Path $found)) {
      return $found
    }
  }

  $roots = @(${env:ProgramFiles(x86)}, $env:ProgramFiles, "D:\Tools") | Where-Object { $_ -and (Test-Path $_) }
  foreach ($root in $roots) {
    $found = Get-ChildItem "$root\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
      Sort-Object FullName -Descending |
      Select-Object -First 1
    if ($found) {
      return $found.FullName
    }
  }
  return $null
}

Write-Host "=== CUDA toolchain ==="
Check-Tool "nvcc" "--version"
Check-Tool "cl" "/?"
$Vcvars = Find-Vcvars64
if ($Vcvars) {
  Write-Host ("  [present] {0,-12} {1}" -f "vcvars64", $Vcvars)
}
Check-Tool "nvidia-smi" "--version"
Write-Host ""

Write-Host "=== SYCLomatic / DPCT ==="
Check-Tool "c2s" "--version"
Check-Tool "dpct" "--version"
Write-Host ""

Write-Host "=== SYCL compiler / runtime ==="
Check-Tool "icpx" "--version"
Check-Tool "icx" "--version"
Check-Tool "clang++" "--version"
Check-Tool "sycl-ls"
Write-Host ""

Write-Host "=== build / scripting ==="
Check-Tool "cmake" "--version"
Check-Tool "make" "--version"
Check-Tool $Python "--version"
try {
  $numpy = & $Python -c "import numpy; print(numpy.__version__)" 2>$null
  Write-Host ("  [present] {0,-12} {1}" -f "numpy", $numpy)
} catch {
  Write-Host ("  [MISSING] {0,-12} {1}" -f "numpy", "(pip install numpy)")
}

Write-Host ""
Write-Host "If cl.exe is missing from PATH but vcvars64 is present, the benchmark tools can bootstrap MSVC automatically."
