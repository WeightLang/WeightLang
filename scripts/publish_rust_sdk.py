"""Publish tested SDK archives, rejecting version collisions and supporting retries."""

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CRATES = ("wgtime", "wgt-language", "wgt-model-macros", "wgt-model")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_record(path):
    request = Request(
        f"https://crates.io/api/v1/crates/{path}",
        headers={"User-Agent": "WeightLang-release (https://github.com/WeightLang/WeightLang)"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def registry_version(name, version):
    response = registry_record(f"{name}/{version}")
    if response is None:
        return None
    record = response["version"]
    if record["crate"] != name or record["num"] != version:
        raise ValueError(f"Unexpected registry identity for {name} {version}")
    return record


def check_publishing():
    if os.environ.get("CARGO_REGISTRY_TOKEN", ""):
        print("CARGO_REGISTRY_TOKEN is configured; crates.io will check its permissions during publication.")
        return
    missing = []
    for name in CRATES:
        response = registry_record(name)
        if response is None:
            missing.append(name)
        elif response["crate"]["id"] != name:
            raise ValueError(f"Unexpected registry identity for {name}")
    if missing:
        raise ValueError(
            f"First crates.io publication requires CARGO_REGISTRY_TOKEN for: {', '.join(missing)}. "
            "Add a token with permission to publish these new crates to the GitHub crates-io "
            "environment or repository secrets. Trusted Publishing requires existing crates. "
            "See docs/releases.md#首次配置包仓库."
        )
    print("All Rust SDK crates exist; the workflow will now check Trusted Publishing authentication.")


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
    package_command = ["cargo", "package", "--locked", "--allow-dirty"]
    command = ["cargo", "publish", "--registry", "crates-io", "--locked", "--allow-dirty"]
    for name in pending:
        package_command.extend(["--package", name])
        command.extend(["--package", name])
    # `cargo publish --dry-run` does not retain archives in target/package.
    # Package the complete dependency closure and check its bytes before upload.
    subprocess.run(package_command, check=True)
    for name in pending:
        if checksum(packages / f"{name}-{version}.crate") != expected[name]:
            raise ValueError(f"{name}: publication archive differs from the tested artifact")
    subprocess.run(command, check=True)
    for name in pending:
        if not matches_registry(name, version, expected[name]):
            raise ValueError(f"{name} {version} is not visible after publication; retry this job")
    print(f"Published and verified all Rust SDK crates {version}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: publish_rust_sdk.py <artifact-directory> | --check-publishing")
    try:
        if sys.argv[1] == "--check-publishing":
            check_publishing()
        else:
            publish(Path(sys.argv[1]))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"publish_rust_sdk: {error}")
