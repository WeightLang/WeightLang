#!/usr/bin/env python3
"""Package a native wgt binary for the wgt-vX.Y.Z GitHub release contract."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    version = subprocess.check_output(
        [str(binary), "--version"], env={**os.environ, "_WGT_VERSION_PROBE": "1"}, text=True
    ).strip()
    if not re.fullmatch(r"wgt (0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        parser.error(f"unexpected executable version: {version!r}")
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system())
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(
        platform.machine().lower()
    )
    if not system or not arch:
        parser.error("unsupported platform")
    name = f"wgt-{system}-{arch}" + (".exe" if system == "windows" else "")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / name
    shutil.copy2(binary, destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    (args.output / "SHA256SUMS").write_text(f"{digest.hexdigest()}  {name}\n", encoding="utf-8")
    print(f"Prepared {args.output} for release wgt-v{version.removeprefix('wgt ')}")


if __name__ == "__main__":
    main()
