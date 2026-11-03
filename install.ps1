# WeightLang installer for Windows (PowerShell 5.1+).
#
#   irm https://github.com/WeightLang/WeightLang/releases/latest/download/install.ps1 | iex
#
# Environment variables:
#   WEIGHTLANG_SOURCE    local checkout to build/install wgt (requires Cargo; skips wheel)
#   WEIGHTLANG_BIN_DIR   native wgt launcher directory (default: install root\bin)
#   WEIGHTLANG_BASE_URL  release download mirror, including SHA256SUMS
#   WEIGHTLANG_VERSION   release tag to install, e.g. v0.1.0 (default: latest)
#   WEIGHTLANG_HOME      install root when uv is not available (default: %LOCALAPPDATA%\weightlang)
#   WEIGHTLANG_NO_UV=1   do not use uv even if it is installed
#
# Downloads the weightlang wheel and SHA256SUMS from the GitHub release, verifies the
# wheel's sha256, then installs it with `uv tool install` if uv is present, otherwise
# into a private virtual environment, and adds its launcher directory to the user PATH.
# PyTorch from PyPI is CPU-only on Windows. No administrator rights are needed.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Say($msg) { Write-Host "weightlang-install: $msg" }
function Die($msg) { throw "weightlang-install: error: $msg" }

$Repo = 'WeightLang/WeightLang'
$Version = if ($env:WEIGHTLANG_VERSION) { $env:WEIGHTLANG_VERSION } else { 'latest' }
$HomeDir = if ($env:WEIGHTLANG_HOME) { $env:WEIGHTLANG_HOME } else { Join-Path $env:LOCALAPPDATA 'weightlang' }
function Add-UserPath($Directory) {
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($UserPath -split ';') -contains $Directory)) {
        [Environment]::SetEnvironmentVariable('Path', ($(if ($UserPath) { "$UserPath;" } else { '' }) + $Directory), 'User')
        Say "added $Directory to your user PATH (open a new terminal)"
    }
}

if ($env:WEIGHTLANG_SOURCE) {
    $Source = (Resolve-Path -LiteralPath $env:WEIGHTLANG_SOURCE).Path
    $Manifest = Join-Path $Source 'crates\wgt\Cargo.toml'
    if (-not (Test-Path $Manifest) -or -not (Test-Path (Join-Path $Source 'pyproject.toml'))) {
        Die 'WEIGHTLANG_SOURCE must be a WeightLang checkout'
    }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Die 'source installation requires Cargo' }
    $TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Source 'target' }
    Say "building wgt from $Source"
    & cargo install --path (Split-Path $Manifest) --locked --force --root $HomeDir --target-dir $TargetDir
    if ($LASTEXITCODE -ne 0) { Die 'cargo install failed' }
    $Native = Join-Path $HomeDir 'bin\wgt.exe'
    & $Native --version
    if ($LASTEXITCODE -ne 0) { Die 'installed wgt failed verification' }
    $BinDir = if ($env:WEIGHTLANG_BIN_DIR) { $env:WEIGHTLANG_BIN_DIR } else { Join-Path $HomeDir 'bin' }
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $Destination = Join-Path $BinDir 'wgt.exe'
    if ([IO.Path]::GetFullPath($Native) -ne [IO.Path]::GetFullPath($Destination)) {
        Copy-Item -LiteralPath $Native -Destination $Destination -Force
    }
    Add-UserPath $BinDir
    Say "installed wgt at $Destination"
    Say 'next: wgt new demo; cd demo; wgt sync'
    return
}

$Base = if ($env:WEIGHTLANG_BASE_URL) {
    $env:WEIGHTLANG_BASE_URL.TrimEnd('/')
} elseif ($Version -eq 'latest') {
    "https://github.com/$Repo/releases/latest/download"
} else {
    "https://github.com/$Repo/releases/download/$Version"
}

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$CacheDir = if ($env:WEIGHTLANG_CACHE) { $env:WEIGHTLANG_CACHE } else { Join-Path $env:LOCALAPPDATA 'weightlang\cache' }
$Tmp = Join-Path $CacheDir ("install-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
    Say "fetching release metadata ($Version)"
    $SumsPath = Join-Path $Tmp 'SHA256SUMS'
    Invoke-WebRequest -UseBasicParsing -Uri "$Base/SHA256SUMS" -OutFile $SumsPath
    $Wheel = $null; $Expected = $null
    foreach ($line in Get-Content $SumsPath) {
        $parts = $line -split '\s+'
        if ($parts.Count -ge 2 -and $parts[1] -match '^weightlang-.*-py3-none-any\.whl$') {
            $Expected = $parts[0].ToLower(); $Wheel = $parts[1]; break
        }
    }
    if (-not $Wheel) { Die 'no weightlang wheel listed in SHA256SUMS' }

    Say "downloading $Wheel"
    $WheelPath = Join-Path $Tmp $Wheel
    Invoke-WebRequest -UseBasicParsing -Uri "$Base/$Wheel" -OutFile $WheelPath
    $Actual = (Get-FileHash -Algorithm SHA256 $WheelPath).Hash.ToLower()
    if ($Actual -ne $Expected) { Die "sha256 mismatch for ${Wheel}: expected $Expected, got $Actual" }
    Say "sha256 ok ($Actual)"

    $Uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($Uv -and $env:WEIGHTLANG_NO_UV -ne '1') {
        Say 'installing with uv tool'
        & uv tool install --force --python '>=3.10' $WheelPath
        if ($LASTEXITCODE -ne 0) { Die 'uv tool install failed' }
        $BinDir = & uv tool dir --bin
        if ($LASTEXITCODE -ne 0) { Die 'uv tool dir failed' }
        $BinDir = $BinDir.Trim()
        & uv tool update-shell | Out-Null
        if ($LASTEXITCODE -ne 0) { Die 'uv tool update-shell failed' }
    } else {
        $Py = $null
        foreach ($cand in @(@('py', '-3'), @('python'))) {
            if (Get-Command $cand[0] -ErrorAction SilentlyContinue) {
                $args0 = @($cand | Select-Object -Skip 1)
                & $cand[0] @args0 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) { $Py = $cand; break }
            }
        }
        if (-not $Py) { Die 'need Python >= 3.10 (https://www.python.org/downloads/) or uv (https://docs.astral.sh/uv/)' }
        $Venv = Join-Path $HomeDir 'venv'
        Say "installing into $Venv"
        if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
        New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
        $pyArgs = @($Py | Select-Object -Skip 1)
        & $Py[0] @pyArgs -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Die 'python -m venv failed' }
        $VPy = Join-Path $Venv 'Scripts\python.exe'
        & $VPy -m pip install -q --upgrade pip
        if ($LASTEXITCODE -ne 0) { Die 'pip upgrade failed' }
        & $VPy -m pip install -q $WheelPath
        if ($LASTEXITCODE -ne 0) { Die 'pip install failed' }
        $BinDir = Join-Path $Venv 'Scripts'
        $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not (($UserPath -split ';') -contains $BinDir)) {
            [Environment]::SetEnvironmentVariable('Path', ($(if ($UserPath) { "$UserPath;" } else { '' }) + $BinDir), 'User')
            Say "added $BinDir to your user PATH (open a new terminal)"
        }
    }

    $Weightc = Join-Path $BinDir 'weightc.exe'
    $Ver = & $Weightc --version
    if ($LASTEXITCODE -ne 0) { Die "installed, but $Weightc --version failed" }
    Say "installed $Ver at $Weightc"
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
