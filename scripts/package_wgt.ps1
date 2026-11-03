param(
    [Parameter(Mandatory=$true)][string]$Binary,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Weightc
)
$ErrorActionPreference = 'Stop'
if (-not $Weightc) { $Weightc = Join-Path (Split-Path $Binary) 'weightc.exe' }
$PreviousProbe = $env:_WGT_VERSION_PROBE
try {
    $env:_WGT_VERSION_PROBE = '1'
    $VersionText = (& $Binary --version)
    if ($LASTEXITCODE -ne 0 -or $VersionText -notmatch '^wgt ((0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*))$') { throw 'Invalid wgt version' }
    $Version = $Matches[1]
    $CompilerVersion = (& $Weightc --version)
    if ($LASTEXITCODE -ne 0 -or $CompilerVersion -ne "weightc $Version") { throw 'weightc and wgt versions must match' }
} finally { $env:_WGT_VERSION_PROBE = $PreviousProbe }
$Arch = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()) {
    'X64' { 'x86_64' }
    'Arm64' { 'aarch64' }
    default { throw 'Unsupported architecture' }
}
if (Test-Path -LiteralPath $Output) { throw 'Output already exists' }
New-Item -ItemType Directory -Path $Output | Out-Null
try {
    Copy-Item -LiteralPath $Binary -Destination (Join-Path $Output "wgt-windows-$Arch.exe")
    Copy-Item -LiteralPath $Weightc -Destination (Join-Path $Output "weightc-windows-$Arch.exe")
    $Sums = foreach ($Name in @("wgt-windows-$Arch.exe", "weightc-windows-$Arch.exe")) {
        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Output $Name)).Hash.ToLowerInvariant()
        "$Hash  $Name"
    }
    [System.IO.File]::WriteAllText((Join-Path $Output 'SHA256SUMS'), ($Sums -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))
} catch {
    Remove-Item -LiteralPath $Output -Recurse -Force
    throw
}
Write-Output "Prepared $Output for release wgt-v$Version"
