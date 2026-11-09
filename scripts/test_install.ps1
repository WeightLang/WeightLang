# Offline installer acceptance checks. Run with Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'These tests require Windows.' }
$Installer = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../install.ps1'))
$Root = Join-Path ([IO.Path]::GetTempPath()) ('weightlang/install-tests-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Fixture = Join-Path $Root 'fixture.exe'
$Names = @('Path', 'CARGO_HOME', 'RUSTUP_HOME', 'CARGO_INSTALL_ROOT', 'CARGO_TARGET_DIR', 'UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH', '_WGT_VERSION_PROBE', 'WEIGHTLANG_SOURCE', 'WEIGHTLANG_VERSION', 'TEST_INSTALL_ROOT', 'TEST_INSTALL_FIXTURE', 'TEST_INSTALL_FAIL')
$Saved = @{}
foreach ($Scope in @('Process', 'User')) {
    foreach ($Name in $Names) { $Saved["$Scope/$Name"] = [Environment]::GetEnvironmentVariable($Name, $Scope) }
}

function Assert([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

# Real subprocesses model tool presence and native exit statuses without network/builds.
$FixtureSource = @'
using System;
using System.IO;
using System.Reflection;
public static class WeightLangInstallerFixture {
    public static int Main(string[] args) {
        string self = Assembly.GetExecutingAssembly().Location;
        string name = Path.GetFileNameWithoutExtension(self);
        string root = Environment.GetEnvironmentVariable("TEST_INSTALL_ROOT");
        File.AppendAllText(Path.Combine(root, "calls.txt"), name + "\t" + String.Join("\t", args) + "\n");
        string failure = Environment.GetEnvironmentVariable("TEST_INSTALL_FAIL");
        if (name == "rustup-init") {
            string bin = Path.Combine(Environment.GetEnvironmentVariable("CARGO_HOME"), "bin");
            Directory.CreateDirectory(bin);
            foreach (string tool in new[] { "cargo", "rustc", "rustup" }) File.Copy(self, Path.Combine(bin, tool + ".exe"), true);
            File.WriteAllText(Path.Combine(root, "rust-ready"), "1");
            File.WriteAllText(Path.Combine(root, "rust-default"), "1");
        } else if (name == "rustup" && args[0] == "toolchain") {
            File.WriteAllText(Path.Combine(root, "rust-ready"), "1");
        } else if (name == "rustup" && args[0] == "default") {
            if (args.Length > 1) File.WriteAllText(Path.Combine(root, "rust-default"), "1");
            if (!File.Exists(Path.Combine(root, "rust-default"))) { Console.Error.WriteLine("no default toolchain"); return 1; }
        } else if ((name == "cargo" || name == "rustc") && !File.Exists(Path.Combine(root, "rust-ready"))) {
            Console.Error.WriteLine("no active toolchain"); return 1;
        } else if (name == "cargo" && args[0] == "install") {
            if (failure == "cargo") return 17;
            int flag = Array.IndexOf(args, "--root");
            if (flag < 0 || String.IsNullOrEmpty(Environment.GetEnvironmentVariable("CARGO_TARGET_DIR"))) return 18;
            string bin = Path.Combine(args[flag + 1], "bin");
            Directory.CreateDirectory(bin);
            foreach (string tool in new[] { "wgt", "weightc" }) File.Copy(self, Path.Combine(bin, tool + ".exe"), true);
        } else if (name == "wgt" || name == "weightc") {
            if (Environment.GetEnvironmentVariable("_WGT_VERSION_PROBE") != "1") return 19;
            if (failure == "probe") return 20;
        }
        Console.WriteLine(name + " fixture");
        return 0;
    }
}
'@

function Reset-Case([string]$Name, [string]$Tools = 'ready') {
    $script:Case = Join-Path $Root $Name
    New-Item -ItemType Directory -Path $Case | Out-Null
    Set-Location $Case
    $env:Path = Join-Path $env:SystemRoot 'System32'
    $env:CARGO_HOME = Join-Path $Case 'cargo home'
    $env:RUSTUP_HOME = Join-Path $Case 'rustup home'
    $env:CARGO_INSTALL_ROOT = Join-Path $Case 'cli root'
    $env:UV_INSTALL_DIR = Join-Path $Case 'uv bin'
    $env:UV_NO_MODIFY_PATH = 'original'
    $env:_WGT_VERSION_PROBE = 'original'
    $env:CARGO_TARGET_DIR = $null
    $env:WEIGHTLANG_SOURCE = $null
    $env:WEIGHTLANG_VERSION = $null
    $env:TEST_INSTALL_FAIL = $null
    $env:TEST_INSTALL_ROOT = $Case
    $env:TEST_INSTALL_FIXTURE = $Fixture
    $script:Downloads = New-Object 'System.Collections.Generic.List[string]'
    $script:DownloadFailure = $null
    [Environment]::SetEnvironmentVariable('Path', (Join-Path $Case 'unrelated'), 'User')
    if ($Tools -ne 'missing') {
        $Bin = Join-Path $env:CARGO_HOME 'bin'
        New-Item -ItemType Directory -Force -Path $Bin, $env:UV_INSTALL_DIR | Out-Null
        foreach ($Tool in @('cargo', 'rustc', 'rustup')) { Copy-Item $Fixture (Join-Path $Bin "$Tool.exe") }
        Copy-Item $Fixture (Join-Path $env:UV_INSTALL_DIR 'uv.exe')
        if ($Tools -eq 'ready') {
            Set-Content (Join-Path $Case 'rust-ready') '1'
            Set-Content (Join-Path $Case 'rust-default') '1'
        }
    }
}

function Invoke-WebRequest([switch]$UseBasicParsing, [string]$Uri, [string]$OutFile) {
    $Downloads.Add($Uri)
    if ($DownloadFailure) { throw 'fixture download failed' }
    if ($Uri -like 'https://static.rust-lang.org/rustup/dist/*/rustup-init.exe') {
        Copy-Item $Fixture $OutFile
    } elseif ($Uri -eq 'https://astral.sh/uv/install.ps1') {
        Set-Content -LiteralPath $OutFile -Value @'
$ErrorActionPreference = 'Stop'
if ($env:UV_NO_MODIFY_PATH -ne '1') { exit 21 }
if ($env:TEST_INSTALL_FAIL -eq 'uv') { exit 22 }
New-Item -ItemType Directory -Force -Path $env:UV_INSTALL_DIR | Out-Null
Copy-Item $env:TEST_INSTALL_FIXTURE (Join-Path $env:UV_INSTALL_DIR 'uv.exe')
'@
    } else { throw "Unexpected download: $Uri" }
}

function Run-Installer([string]$ExpectedError) {
    $Failure = $null
    try { $script:Output = & $Installer 2>&1 | Out-String } catch { $Failure = $_ }
    if ($ExpectedError) {
        Assert ($null -ne $Failure -and "$Failure" -like "*$ExpectedError*") "Expected '$ExpectedError', got '$Failure'"
    } elseif ($Failure) { throw $Failure }
    Assert ($env:UV_NO_MODIFY_PATH -eq 'original') 'uv control flag was not restored'
    Assert ($env:_WGT_VERSION_PROBE -eq 'original') 'version probe flag was not restored'
    Assert (-not $env:CARGO_TARGET_DIR) 'temporary Cargo target setting leaked'
}

$OriginalLocation = Get-Location
try {
    Add-Type -TypeDefinition $FixtureSource -OutputAssembly $Fixture -OutputType ConsoleApplication

    Reset-Case 'fresh' 'missing'
    Run-Installer
    Assert ($Downloads.Count -eq 2) 'fresh install must bootstrap Rust and uv'
    $Calls = Get-Content -Raw (Join-Path $Case 'calls.txt')
    Assert ($Calls.Contains("--default-toolchain`tstable")) 'Rust must install stable'
    Assert ($Calls.Contains("--branch`tmain")) 'default source must be main'
    Assert ($Output.Contains('Installed uv, Cargo, wgt and weightc.')) 'success summary is missing'
    Assert ([Environment]::GetEnvironmentVariable('RUSTUP_HOME', 'User') -eq $env:RUSTUP_HOME) 'custom Rust home was not saved'

    Reset-Case 'reuse'
    Run-Installer
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    Run-Installer
    Assert ($Downloads.Count -eq 0) 'existing tools must not be downloaded again'
    Assert ([Environment]::GetEnvironmentVariable('Path', 'User') -eq $UserPath) 'rerun must not duplicate PATH entries'
    Assert ($UserPath.Contains((Join-Path $Case 'unrelated'))) 'unrelated PATH was removed'

    Reset-Case 'external-tools'
    $External = Join-Path $Case 'existing tools'
    New-Item -ItemType Directory -Path $External | Out-Null
    foreach ($Tool in @('cargo', 'rustc', 'rustup', 'uv')) { Copy-Item $Fixture (Join-Path $External "$Tool.exe") }
    foreach ($Tool in @('cargo', 'rustc')) { Set-Content (Join-Path $env:CARGO_HOME "bin/$Tool.exe") 'broken executable' }
    Set-Content (Join-Path $env:UV_INSTALL_DIR 'uv.exe') 'broken executable'
    $env:Path = "$External;$env:Path"
    Run-Installer
    Assert ($Downloads.Count -eq 0) 'stale default tools shadowed working existing tools'
    Assert ($env:Path.StartsWith("$External;")) 'working tool precedence was not preserved'

    Reset-Case 'rustup-without-default' 'rustup'
    Run-Installer
    $Calls = Get-Content -Raw (Join-Path $Case 'calls.txt')
    Assert ($Calls.Contains("rustup`ttoolchain`tinstall`tstable`t--profile`tminimal")) 'missing toolchain was not installed'
    Assert ($Calls.Contains("rustup`tdefault`tstable")) 'missing Rust default was not selected'
    Assert ($Downloads.Count -eq 0) 'existing rustup must be reused'

    Reset-Case 'tag'
    $env:WEIGHTLANG_VERSION = 'v0.2.0'
    Run-Installer
    Assert ((Get-Content -Raw (Join-Path $Case 'calls.txt')).Contains("--tag`tv0.2.0")) 'release tag was not forwarded'

    Reset-Case 'explicit-source'
    $env:WEIGHTLANG_SOURCE = Join-Path $Case 'source tree'
    $env:WEIGHTLANG_VERSION = 'ignored-tag'
    Run-Installer
    $Calls = Get-Content -Raw (Join-Path $Case 'calls.txt')
    Assert ($Calls.Contains((Join-Path $env:WEIGHTLANG_SOURCE 'crates/wgt'))) 'explicit source was not forwarded as one argument'
    Assert (-not $Calls.Contains('ignored-tag')) 'explicit source must take priority over tag'

    Reset-Case 'local-checkout'
    New-Item -ItemType Directory -Path 'crates/wgt' -Force | Out-Null
    Set-Content 'crates/wgt/Cargo.toml' '[package]'
    Run-Installer
    Assert ((Get-Content -Raw (Join-Path $Case 'calls.txt')).Contains("--path`tcrates/wgt")) 'checkout source was not selected'

    Reset-Case 'download-failure' 'missing'
    $script:DownloadFailure = $true
    Run-Installer 'fixture download failed'
    Assert (-not (Test-Path (Join-Path $Case 'calls.txt'))) 'failed download must not execute an installer'

    Reset-Case 'uv-failure'
    Remove-Item (Join-Path $env:UV_INSTALL_DIR 'uv.exe')
    $env:TEST_INSTALL_FAIL = 'uv'
    Run-Installer 'uv installation failed'
    Assert (-not (Get-Content -Raw (Join-Path $Case 'calls.txt')).Contains("cargo`tinstall")) 'Cargo ran after uv failed'

    Reset-Case 'cargo-failure'
    $env:TEST_INSTALL_FAIL = 'cargo'
    Run-Installer 'WeightLang installation failed'
    Assert (-not (Get-Content -Raw (Join-Path $Case 'calls.txt')).Contains("wgt`t--version")) 'success probes ran after Cargo failed'

    Reset-Case 'binary-failure'
    $env:TEST_INSTALL_FAIL = 'probe'
    Run-Installer 'Installed wgt failed its version check'
    Write-Output 'Passed 11 Windows installer scenarios.'
} finally {
    Set-Location $OriginalLocation
    foreach ($Scope in @('Process', 'User')) {
        foreach ($Name in $Names) { [Environment]::SetEnvironmentVariable($Name, $Saved["$Scope/$Name"], $Scope) }
    }
    Remove-Item -Recurse -Force $Root
}
