"""Publish tested SDK archives, rejecting version collisions and supporting retries."""

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CRATES = ("wgtime", "wgt-language", "wgt-model-macros", "wgt-model")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_version(name, version):
    request = Request(
        f"https://crates.io/api/v1/crates/{name}/{version}",
        headers={"User-Agent": "WeightLang-release (https://github.com/WeightLang/WeightLang)"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            record = json.load(response)["version"]
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    if record["crate"] != name or record["num"] != version:
        raise ValueError(f"Unexpected registry identity for {name} {version}")
    return record


def matches_registry(name, version, expected):
    record = registry_version(name, version)
    if record is None:
        return False
    if record["yanked"] or record["checksum"] != expected:
        raise ValueError(f"{name} {version} already exists with different bytes or is yanked")
    return True


def publish(directory):
    version = tomllib.loads(Path("Cargo.toml").read_text())["workspace"]["package"]["version"]
    # Inspect every version before publishing anything, including on partial retries.
    expected = {name: checksum(directory / f"{name}-{version}.crate") for name in CRATES}
    pending = [name for name in CRATES if not matches_registry(name, version, expected[name])]
    if not pending:
        print(f"All Rust SDK crates {version} already match the tested archives.")
        return
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], text=True,
    ))
    packages = Path(metadata["target_directory"]) / "package"
    command = ["cargo", "publish", "--registry", "crates-io", "--locked", "--allow-dirty"]
    for name in pending:
        command.extend(["--package", name])
    # Cargo resolves the dependency order and verifies packaged sources. Repacking
    # must produce the exact archives that passed CI before any registry write.
    subprocess.run([*command, "--dry-run"], check=True)
    for name in pending:
        if checksum(packages / f"{name}-{version}.crate") != expected[name]:
            raise ValueError(f"{name}: publication archive differs from the tested artifact")
    subprocess.run(command, check=True)
    for name in pending:
        if not matches_registry(name, version, expected[name]):
            raise ValueError(f"{name} {version} is not visible after publication; retry this job")
    print(f"Published and verified all Rust SDK crates {version}.")


if __name__ == "__main__":
    try:
        publish(Path(sys.argv[1]))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"publish_rust_sdk: {error}")
