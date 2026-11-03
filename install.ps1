# Install native weightc and wgt. Python is not part of this installation.
$ErrorActionPreference = 'Stop'
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'Install Rust/Cargo before running this installer.'
}
if ($env:WEIGHTLANG_SOURCE) {
    & cargo install --locked --force --path (Join-Path $env:WEIGHTLANG_SOURCE 'crates/wgt') --bin weightc --bin wgt
} elseif (Test-Path 'crates/wgt/Cargo.toml') {
    & cargo install --locked --force --path crates/wgt --bin weightc --bin wgt
} elseif ($env:WEIGHTLANG_VERSION) {
    & cargo install --locked --force --git https://github.com/WeightLang/WeightLang.git --tag $env:WEIGHTLANG_VERSION --bin weightc --bin wgt weightlang-wgt
} else {
    & cargo install --locked --force --git https://github.com/WeightLang/WeightLang.git --branch main --bin weightc --bin wgt weightlang-wgt
}
if ($LASTEXITCODE -ne 0) { throw 'Native installation failed.' }
Write-Output 'Installed native weightc and wgt in Cargo bin directory. Add it to PATH.'
