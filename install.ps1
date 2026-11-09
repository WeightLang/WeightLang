# Install uv, Rust/Cargo, weightc and wgt for the current user.
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Use install.sh on Linux and macOS.' }

function Add-ToolPath([string]$Directory, [string]$Scope = 'Process') {
    $Current = [Environment]::GetEnvironmentVariable('Path', $Scope)
    $OtherPaths = @($Current -split ';' | Where-Object { $_ -and $_ -ine $Directory })
    $Value = (@($Directory) + $OtherPaths) -join ';'
    if ($Current -cne $Value) {
        [Environment]::SetEnvironmentVariable('Path', $Value, $Scope)
    }
}

function Test-Tool([string]$Name, [string[]]$Arguments = @('--version')) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { return $false }
    try {
        & $Name @Arguments *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Invoke-Tool([string]$Command, [string[]]$Arguments, [string]$Failure) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Failure (exit $LASTEXITCODE)" }
}

$CargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $HOME '.cargo' }
$InstallRoot = if ($env:CARGO_INSTALL_ROOT) { $env:CARGO_INSTALL_ROOT } else { $CargoHome }
$UvBin = if ($env:UV_INSTALL_DIR) { $env:UV_INSTALL_DIR } else { Join-Path $HOME '.local/bin' }
$CargoHome = [IO.Path]::GetFullPath($CargoHome)
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$UvBin = [IO.Path]::GetFullPath($UvBin)
$ToolPaths = @($UvBin, (Join-Path $CargoHome 'bin'), (Join-Path $InstallRoot 'bin')) | Select-Object -Unique
# Keep working tools already selected by PATH ahead of stale default installs.
$ExistingBins = @('uv', 'cargo', 'rustc', 'rustup' | ForEach-Object {
    if (Test-Tool $_) {
        $Path = (Get-Command $_).Source
        if ($Path) { Split-Path -Parent $Path }
    }
})
$ExistingBins = @($env:Path -split ';' | Where-Object { $ExistingBins -contains $_ -and $ToolPaths -notcontains $_ } | Select-Object -Unique)
[Array]::Reverse($ExistingBins)
$ToolPaths = @($ToolPaths) + $ExistingBins
foreach ($Directory in $ToolPaths) { Add-ToolPath $Directory }

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) 'weightlang'
$Temporary = Join-Path $TemporaryRoot ([Guid]::NewGuid().ToString('N'))
$PreviousUvDir = $env:UV_INSTALL_DIR
$PreviousUvPath = $env:UV_NO_MODIFY_PATH
$PreviousProbe = $env:_WGT_VERSION_PROBE
$PreviousTarget = $env:CARGO_TARGET_DIR
try {
    if (-not (Test-Tool 'cargo') -or -not (Test-Tool 'rustc')) {
        if (Test-Tool 'rustup') {
            Invoke-Tool 'rustup' @('toolchain', 'install', 'stable', '--profile', 'minimal') 'Rust installation failed.'
            if (-not (Test-Tool 'rustup' @('default'))) {
                Invoke-Tool 'rustup' @('default', 'stable') 'Could not select the stable Rust toolchain.'
            }
        } else {
            Write-Output 'Installing Rust/Cargo with rustup (stable, minimal profile)...'
            $Architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
            $Target = switch ($Architecture) {
                'AMD64' { 'x86_64-pc-windows-msvc' }
                'ARM64' { 'aarch64-pc-windows-msvc' }
                'x86' { 'i686-pc-windows-msvc' }
                default { throw "Unsupported Windows architecture: $Architecture" }
            }
            New-Item -ItemType Directory -Force -Path $Temporary | Out-Null
            $Installer = Join-Path $Temporary 'rustup-init.exe'
            Invoke-WebRequest -UseBasicParsing -Uri "https://static.rust-lang.org/rustup/dist/$Target/rustup-init.exe" -OutFile $Installer
            Invoke-Tool $Installer @('-y', '--profile', 'minimal', '--default-toolchain', 'stable', '--no-modify-path') 'Rust installation failed.'
        }
        if (-not (Test-Tool 'cargo') -or -not (Test-Tool 'rustc')) {
            throw 'Rust/Cargo is unavailable. Check CARGO_HOME, RUSTUP_HOME and any project toolchain override.'
        }
    }

    if (-not (Test-Tool 'uv')) {
        Write-Output 'Installing uv...'
        New-Item -ItemType Directory -Force -Path $Temporary | Out-Null
        $Installer = Join-Path $Temporary 'uv-install.ps1'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1' -OutFile $Installer
        $env:UV_INSTALL_DIR = $UvBin
        $env:UV_NO_MODIFY_PATH = '1'
        $ShellName = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh.exe' } else { 'powershell.exe' }
        Invoke-Tool (Join-Path $PSHOME $ShellName) @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Installer) 'uv installation failed.'
        if (-not (Test-Tool 'uv')) { throw 'uv installation did not produce a working uv executable.' }
    }

    if (-not $env:CARGO_TARGET_DIR) {
        $Cache = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME 'AppData/Local' }
        $env:CARGO_TARGET_DIR = Join-Path $Cache 'WeightLang/cache/install/target'
    }
    $CargoArguments = @('install', '--locked', '--force', '--root', $InstallRoot)
    if ($env:WEIGHTLANG_SOURCE) {
        $CargoArguments += @('--path', (Join-Path $env:WEIGHTLANG_SOURCE 'crates/wgt'))
    } elseif (Test-Path 'crates/wgt/Cargo.toml') {
        $CargoArguments += @('--path', 'crates/wgt')
    } elseif ($env:WEIGHTLANG_VERSION) {
        $CargoArguments += @('--git', 'https://github.com/WeightLang/WeightLang.git', '--tag', $env:WEIGHTLANG_VERSION)
    } else {
        $CargoArguments += @('--git', 'https://github.com/WeightLang/WeightLang.git', '--branch', 'main')
    }
    $CargoArguments += @('--bin', 'weightc', '--bin', 'wgt')
    if ($CargoArguments -contains '--git') { $CargoArguments += 'weightlang-wgt' }
    Invoke-Tool 'cargo' $CargoArguments 'WeightLang installation failed. Check the build output; Windows source builds need Visual Studio C++ Build Tools and a Windows SDK. For an outdated Rust toolchain, run rustup update stable.'

    $env:_WGT_VERSION_PROBE = '1'
    foreach ($Name in @('wgt', 'weightc')) {
        Invoke-Tool (Join-Path $InstallRoot "bin/$Name.exe") @('--version') "Installed $Name failed its version check."
    }
    Invoke-Tool 'uv' @('--version') 'uv failed its version check.'
    Invoke-Tool 'cargo' @('--version') 'Cargo failed its version check.'
    Invoke-Tool 'rustc' @('--version') 'Rust failed its version check.'
    foreach ($Directory in $ToolPaths) { Add-ToolPath $Directory 'User' }
    # Custom rustup homes must also survive a new terminal session.
    foreach ($Name in @('CARGO_HOME', 'RUSTUP_HOME', 'CARGO_INSTALL_ROOT')) {
        $Value = [Environment]::GetEnvironmentVariable($Name, 'Process')
        if ($Value) { [Environment]::SetEnvironmentVariable($Name, [IO.Path]::GetFullPath($Value), 'User') }
    }
    Write-Output 'Installed uv, Cargo, wgt and weightc. PATH is ready in this PowerShell session and saved for new terminals.'
} finally {
    $env:UV_INSTALL_DIR = $PreviousUvDir
    $env:UV_NO_MODIFY_PATH = $PreviousUvPath
    $env:_WGT_VERSION_PROBE = $PreviousProbe
    $env:CARGO_TARGET_DIR = $PreviousTarget
    if (Test-Path $Temporary) { Remove-Item -Recurse -Force $Temporary }
}
