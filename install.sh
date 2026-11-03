#!/bin/sh
# Install native weightc and wgt. Python is not part of this installation.
set -eu
command -v cargo >/dev/null 2>&1 || { echo 'Install Rust/Cargo before running this installer.' >&2; exit 1; }
if [ -n "${WEIGHTLANG_SOURCE:-}" ]; then
    cargo install --locked --force --path "$WEIGHTLANG_SOURCE/crates/wgt" --bin weightc --bin wgt
elif [ -f crates/wgt/Cargo.toml ]; then
    cargo install --locked --force --path crates/wgt --bin weightc --bin wgt
elif [ -n "${WEIGHTLANG_VERSION:-}" ]; then
    cargo install --locked --force --git https://github.com/WeightLang/WeightLang.git --tag "$WEIGHTLANG_VERSION" --bin weightc --bin wgt weightlang-wgt
else
    cargo install --locked --force --git https://github.com/WeightLang/WeightLang.git --branch main --bin weightc --bin wgt weightlang-wgt
fi
printf '%s\n' 'Installed native weightc and wgt in Cargo bin directory. Add it to PATH.'
