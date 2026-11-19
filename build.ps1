# Build WarpDB natively on Windows (MSVC + CUDA + Ninja).
# Usage: .\build.ps1 [-Config Release|Debug] [-Clean] [-Test]
param(
  [string]$Config = "Release",
  [switch]$Clean,
  [switch]$Test
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv\Scripts"
$buildDir = Join-Path $root "build\$($Config.ToLower())"

# Load the MSVC x64 environment (needed by nvcc + Ninja).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
$vsPath = & $vswhere -latest -products * -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$tmpBat = Join-Path $env:TEMP "warpdb_vcvars.bat"
Set-Content -Path $tmpBat -Encoding ASCII -Value "@echo off`r`ncall `"$vcvars`" >nul`r`nset"
$ErrorActionPreference = "Continue"   # vcvars prints harmless stderr noise
cmd /c $tmpBat 2>$null | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}
$env:PATH = "$venv;$env:PATH"

if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }

# Pin the interpreter used for the pybind11 module to THIS project's venv (CMake would otherwise honour whatever
# VIRTUAL_ENV happens to be active in the shell).
$py = Join-Path $venv "python.exe"
$pyArg = @()
if (Test-Path $py) { $pyArg = @("-DPython3_EXECUTABLE=$py", "-DPython3_FIND_VIRTUALENV=ONLY") }
cmake -S $root -B $buildDir -G Ninja "-DCMAKE_BUILD_TYPE=$Config" @pyArg
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build $buildDir
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if ($Test) {
  ctest --test-dir $buildDir --output-on-failure
  exit $LASTEXITCODE
}
