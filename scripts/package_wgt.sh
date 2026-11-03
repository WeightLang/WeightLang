#!/bin/sh
# Package a matching native wgt/weightc pair for a wgt-vX.Y.Z release.
set -eu
BINARY= COMPILER= OUTPUT=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary) BINARY=$2; shift 2 ;;
        --weightc) COMPILER=$2; shift 2 ;;
        --output) OUTPUT=$2; shift 2 ;;
        *) echo 'Usage: package_wgt.sh --binary PATH [--weightc PATH] --output NEW_DIRECTORY' >&2; exit 2 ;;
    esac
done
[ -n "$BINARY" ] && [ -n "$OUTPUT" ] || { echo '--binary and --output are required' >&2; exit 2; }
[ -n "$COMPILER" ] || COMPILER="$(dirname "$BINARY")/weightc"
VERSION=$(_WGT_VERSION_PROBE=1 "$BINARY" --version)
printf '%s\n' "$VERSION" | LC_ALL=C grep -Eq '^wgt (0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' || { echo 'Invalid wgt version' >&2; exit 1; }
VERSION=${VERSION#wgt }
[ "$(_WGT_VERSION_PROBE=1 "$COMPILER" --version)" = "weightc $VERSION" ] || { echo 'weightc and wgt versions must match' >&2; exit 1; }
case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) echo 'Unsupported host OS' >&2; exit 1 ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; arm64|aarch64) ARCH=aarch64 ;; *) echo 'Unsupported host architecture' >&2; exit 1 ;; esac
# Never overwrite an existing release directory, even on a packaging error.
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
trap 'rm -f "$OUTPUT/wgt-$OS-$ARCH" "$OUTPUT/weightc-$OS-$ARCH" "$OUTPUT/SHA256SUMS"; rmdir "$OUTPUT" 2>/dev/null || true' EXIT
trap 'exit 1' HUP INT TERM
cp "$BINARY" "$OUTPUT/wgt-$OS-$ARCH"
cp "$COMPILER" "$OUTPUT/weightc-$OS-$ARCH"
chmod 755 "$OUTPUT/wgt-$OS-$ARCH" "$OUTPUT/weightc-$OS-$ARCH"
(
    cd "$OUTPUT"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "wgt-$OS-$ARCH" "weightc-$OS-$ARCH" > SHA256SUMS
    else
        shasum -a 256 "wgt-$OS-$ARCH" "weightc-$OS-$ARCH" > SHA256SUMS
    fi
)
trap - EXIT HUP INT TERM
printf 'Prepared %s for release wgt-v%s\n' "$OUTPUT" "$VERSION"
