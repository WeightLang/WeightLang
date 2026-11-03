#!/bin/sh
# WeightLang installer for Linux and macOS.
#
#   curl -fsSL https://github.com/WeightLang/WeightLang/releases/latest/download/install.sh | sh
#
# Environment variables:
#   WEIGHTLANG_SOURCE    local checkout to build/install wgt (requires Cargo; skips wheel)
#   WEIGHTLANG_VERSION   release tag to install, e.g. v0.1.0 (default: latest)
#   WEIGHTLANG_TORCH     "cpu" (default on Linux) installs CPU-only PyTorch, which is much
#                        smaller; "default" uses PyPI's torch (CUDA build on Linux)
#   WEIGHTLANG_HOME      install root when uv is not available (default: ~/.local/share/weightlang)
#   WEIGHTLANG_BIN_DIR   where the weightc launcher goes (default: ~/.local/bin)
#   WEIGHTLANG_NO_UV=1   do not use uv even if it is installed
#   WEIGHTLANG_BASE_URL  download from this URL instead of the GitHub release (mirrors, testing)
#
# What it does: downloads the weightlang wheel and SHA256SUMS from the GitHub release,
# verifies the wheel's sha256, then installs it with `uv tool install` if uv is present,
# otherwise into a private virtual environment with a launcher in WEIGHTLANG_BIN_DIR.
# It does not use sudo and does not download any other installer scripts.
set -eu

REPO="WeightLang/WeightLang"
VERSION="${WEIGHTLANG_VERSION:-latest}"
TORCH="${WEIGHTLANG_TORCH:-}"
HOME_DIR="${WEIGHTLANG_HOME:-$HOME/.local/share/weightlang}"
BIN_DIR="${WEIGHTLANG_BIN_DIR:-$HOME/.local/bin}"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

say() { printf 'weightlang-install: %s\n' "$*"; }
die() { printf 'weightlang-install: error: %s\n' "$*" >&2; exit 1; }

# Source mode installs the native CLI, including its matching compiler sources.
# Keep Cargo build intermediates on the checkout filesystem, not in /tmp.
if [ -n "${WEIGHTLANG_SOURCE:-}" ]; then
    SOURCE="$(cd "$WEIGHTLANG_SOURCE" && pwd)" || die "invalid WEIGHTLANG_SOURCE"
    [ -f "$SOURCE/crates/wgt/Cargo.toml" ] && [ -f "$SOURCE/pyproject.toml" ] || die "WEIGHTLANG_SOURCE must be a WeightLang checkout"
    command -v cargo >/dev/null 2>&1 || die "source installation requires Cargo"
    say "building wgt from $SOURCE"
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$SOURCE/target}" cargo install --path "$SOURCE/crates/wgt" --locked --force --root "$HOME_DIR"
    "$HOME_DIR/bin/wgt" --version || die "installed wgt failed verification"
    mkdir -p "$BIN_DIR"
    if [ "$BIN_DIR" != "$HOME_DIR/bin" ]; then
        ln -sf "$HOME_DIR/bin/wgt" "$BIN_DIR/wgt"
    fi
    say "installed wgt at $BIN_DIR/wgt"
    case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "add $BIN_DIR to PATH" ;; esac
    say "next: wgt new demo; cd demo; wgt sync"
    exit 0
fi

if [ -z "$TORCH" ]; then
    case "$(uname -s)" in
        Linux) TORCH=cpu ;;
        *) TORCH=default ;;
    esac
fi
case "$TORCH" in cpu | default) ;; *) die "WEIGHTLANG_TORCH must be 'cpu' or 'default'" ;; esac

if [ -n "${WEIGHTLANG_BASE_URL:-}" ]; then
    BASE="${WEIGHTLANG_BASE_URL%/}"
elif [ "$VERSION" = latest ]; then
    BASE="https://github.com/$REPO/releases/latest/download"
else
    BASE="https://github.com/$REPO/releases/download/$VERSION"
fi

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q -O "$2" "$1"; }
else
    die "need curl or wget"
fi

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "need sha256sum or shasum"
fi

# Release downloads use disk-backed cache; only this invocation's scratch is removed.
CACHE_DIR="${WEIGHTLANG_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/weightlang}"
mkdir -p "$CACHE_DIR"
TMP="$(mktemp -d "$CACHE_DIR/install.XXXXXX")"
trap 'rm -rf "$TMP"' 0
trap 'exit 130' INT
trap 'exit 143' TERM

say "fetching release metadata ($VERSION)"
fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS" || die "cannot download $BASE/SHA256SUMS"
WHEEL="$(awk '$2 ~ /^weightlang-.*-py3-none-any\.whl$/ { print $2; exit }' "$TMP/SHA256SUMS")"
EXPECTED="$(awk -v w="$WHEEL" '$2 == w { print $1; exit }' "$TMP/SHA256SUMS")"
[ -n "$WHEEL" ] && [ -n "$EXPECTED" ] || die "no weightlang wheel listed in SHA256SUMS"

say "downloading $WHEEL"
fetch "$BASE/$WHEEL" "$TMP/$WHEEL" || die "cannot download $WHEEL"
ACTUAL="$(sha256 "$TMP/$WHEEL")"
[ "$ACTUAL" = "$EXPECTED" ] || die "sha256 mismatch for $WHEEL: expected $EXPECTED, got $ACTUAL"
say "sha256 ok ($ACTUAL)"

if [ "${WEIGHTLANG_NO_UV:-}" != 1 ] && command -v uv >/dev/null 2>&1; then
    say "installing with uv tool (torch: $TORCH)"
    if [ "$TORCH" = cpu ]; then
        uv tool install --force --python '>=3.10' --index "$CPU_INDEX" "$TMP/$WHEEL"
    else
        uv tool install --force --python '>=3.10' "$TMP/$WHEEL"
    fi
    WEIGHTC="$(uv tool dir --bin)/weightc"
else
    PY=""
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 &&
            "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY="$cand"
            break
        fi
    done
    [ -n "$PY" ] || die "need Python >= 3.10 (or install uv: https://docs.astral.sh/uv/)"
    say "installing into $HOME_DIR/venv with $($PY --version) (torch: $TORCH)"
    rm -rf "$HOME_DIR/venv"
    mkdir -p "$HOME_DIR" "$BIN_DIR"
    "$PY" -m venv "$HOME_DIR/venv" || die "python venv failed (Debian/Ubuntu: apt install python3-venv)"
    "$HOME_DIR/venv/bin/python" -m pip install -q --upgrade pip
    if [ "$TORCH" = cpu ]; then
        "$HOME_DIR/venv/bin/python" -m pip install -q --extra-index-url "$CPU_INDEX" "$TMP/$WHEEL"
    else
        "$HOME_DIR/venv/bin/python" -m pip install -q "$TMP/$WHEEL"
    fi
    ln -sf "$HOME_DIR/venv/bin/weightc" "$BIN_DIR/weightc"
    WEIGHTC="$BIN_DIR/weightc"
fi

"$WEIGHTC" --version >/dev/null || die "installed, but $WEIGHTC --version failed"
say "installed $("$WEIGHTC" --version) at $WEIGHTC"
case ":$PATH:" in
    *":$(dirname "$WEIGHTC"):"*) ;;
    *) say "add $(dirname "$WEIGHTC") to PATH to run weightc" ;;
esac
