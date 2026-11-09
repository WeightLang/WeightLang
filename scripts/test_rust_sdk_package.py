"""Build a downstream Rust application using only the released crate contents."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CRATES = ("wgtime", "wgt-language", "wgt-model-macros", "wgt-model")
MODEL = '''wgt 0.5
import activation from "std/activation"
model PackagedModel
input x:f32[2]
output y:f32[2]
# Two independent features; the packaged standard library squares positive values.
fn forward(x){return activation.squared_relu(x);}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, default=ROOT / "target/package")
    parser.add_argument("--daemon", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".cache/weightlang/sdk-consumer")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    socket_root = Path("/tmp/weightlang")
    socket_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir) as work, tempfile.TemporaryDirectory(
        prefix="sdk-", dir=socket_root
    ) as service_directory:
        work = Path(work).resolve()
        packages = {}
        for name in CRATES:
            with tarfile.open(args.packages / f"{name}-{version}.crate") as archive:
                # Cargo must flatten the repository's shared-resource symlinks.
                assert not any(member.issym() or member.islnk() for member in archive), name
                archive.extractall(work, filter="data")
            package = work / f"{name}-{version}"
            manifest = tomllib.loads((package / "Cargo.toml").read_text())
            assert manifest["package"]["version"] == version, name
            for dependency, options in manifest.get("dependencies", {}).items():
                if dependency in CRATES:
                    assert options["version"] == f"={version}", (name, dependency)
                    assert "path" not in options, (name, dependency)
            packages[name] = package
        assets = packages["wgt-language"] / "assets"
        assert (assets / "prelude.json").read_bytes() == (ROOT / "language/prelude.json").read_bytes()
        for source in (ROOT / "std").glob("*.wgt"):
            assert (assets / "std" / source.name).read_bytes() == source.read_bytes(), source.name

        consumer = work / "consumer"
        (consumer / "src").mkdir(parents=True)
        patches = "\n".join(f'{name} = {{ path = {json.dumps(str(path))} }}' for name, path in packages.items())
        (consumer / "Cargo.toml").write_text(f'''[workspace]
[package]
name = "weightlang-release-consumer"
version = "0.0.0"
edition = "2024"
[dependencies]
wgt = {{ package = "wgt-model", version = "={version}" }}
[patch.crates-io]
{patches}
''')
        (consumer / "model.wgt").write_text(MODEL)
        (consumer / "src/main.rs").write_text('''fn main() -> wgt::Result<()> {
    let inline = wgt::model!(r#"''' + MODEL + '''"#);
    let file = wgt::include_model!("model.wgt");
    let runtime = wgt::ModelDefinition::from_file(std::path::Path::new("model.wgt"))?;
    let endpoint = std::env::args_os().nth(1).expect("service endpoint");
    let client = wgt::Client::connect_to(std::path::Path::new(&endpoint))?;
    for definition in [&inline, &file, &runtime] {
        let model = client.initialize(definition, wgt::InstanceOptions::default())?;
        assert_eq!(model.backend(), "native-cpu");
        assert_eq!(model.forward(&[-3.0, 2.0])?, vec![0.0, 4.0]);
        model.close()?;
    }
    assert_eq!(client.stats()?["sessions"], 0);
    Ok(())
}
''')
        env = os.environ.copy()
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target/sdk-consumer"))
        rustc_version = subprocess.check_output(["rustc", "-vV"], env=env, text=True)
        host = next(line.removeprefix("host: ") for line in rustc_version.splitlines() if line.startswith("host: "))
        metadata = json.loads(subprocess.check_output(
            ["cargo", "metadata", "--offline", "--format-version=1", "--filter-platform", host],
            cwd=consumer, env=env,
        ))
        resolved = set()
        for package in metadata["packages"]:
            if package["name"] in packages:
                assert Path(package["manifest_path"]).parent.resolve() == packages[package["name"]]
                resolved.add(package["name"])
        assert resolved == set(CRATES), resolved
        subprocess.run(["cargo", "build", "--offline", "--locked"], cwd=consumer, env=env, check=True)
        daemon = subprocess.Popen([str(args.daemon.resolve()), "--test-directory", service_directory])
        try:
            endpoint = Path(service_directory) / "service.sock"
            deadline = time.monotonic() + 10
            while not endpoint.exists():
                if daemon.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("wgtime did not start for the packaged SDK consumer")
                time.sleep(0.02)
            subprocess.run(
                ["cargo", "run", "--offline", "--locked", "--", str(endpoint)],
                cwd=consumer, env=env, check=True, timeout=60,
            )
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
    print(f"Verified all four Rust SDK {version} archives and a real downstream service consumer.")


if __name__ == "__main__":
    main()
