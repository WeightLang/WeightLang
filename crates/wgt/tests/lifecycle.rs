//! End-to-end native lifecycle. Fixtures and all outputs stay in an owned disk-backed directory.
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const WGT: &str = env!("CARGO_BIN_EXE_wgt");
const WEIGHTC: &str = env!("CARGO_BIN_EXE_weightc");

fn scratch() -> tempfile::TempDir {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/lifecycle-tests");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn fixture(root: &Path, steps: u64, resume: bool, complete: bool) {
    for (name, text) in [
        (
            "model.wgt",
            include_str!("../../../examples/lifecycle/model.wgt"),
        ),
        (
            "data.jsonl",
            include_str!("../../../examples/lifecycle/data.jsonl"),
        ),
        (
            "wgt.mod",
            include_str!("../../../examples/lifecycle/wgt.mod"),
        ),
        (
            "tests/evaluation.wgt",
            include_str!("../../../examples/lifecycle/tests/evaluation.wgt"),
        ),
        (
            "gpu/model.wgt",
            include_str!("../../../examples/lifecycle/gpu/model.wgt"),
        ),
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    fs::create_dir_all(root.join("out")).unwrap();
    fs::write(
        root.join("run.jsonl"),
        json!({"steps": steps, "resume": resume, "complete": complete, "build": false}).to_string(),
    )
    .unwrap();
}
fn command(binary: &str, root: &Path) -> Command {
    let mut cmd = Command::new(binary);
    cmd.current_dir(root)
        .env("PATH", "/nonexistent")
        .env("WEIGHTLANG_CACHE", root.parent().unwrap().join("cli-cache"));
    cmd
}
fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn run(root: &Path) -> Value {
    let output = success(
        command(WGT, root)
            .args(["run", "model.wgt"])
            .output()
            .unwrap(),
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn read_json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, path: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for item in fs::read_dir(path).unwrap() {
            let path = item.unwrap().path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}
fn assert_checkpoint(root: &Path, steps: u64) {
    let checkpoint = read_json(root.join("out/checkpoint.json"));
    assert_eq!(checkpoint["format"], "weightlang.native-training.v1");
    assert_eq!(checkpoint["steps"], steps);
    assert_eq!(checkpoint["optimizer"], "adam");
    for name in ["first", "second"] {
        let moments = checkpoint[name].as_array().unwrap();
        assert!(!moments.is_empty());
        assert!(moments
            .iter()
            .flat_map(|v| v.as_array().unwrap())
            .any(|v| v.as_f64().unwrap() != 0.0));
    }
}
fn infer(root: &Path, model: &Path, expected: &Value) {
    let source = format!(
        "wgt 0.5\nfn main() {{ let net = model({}, \"adam\"); return predict(net, [0.25]); }}",
        serde_json::to_string(model.to_str().unwrap()).unwrap()
    );
    let source_dir = root.join("inference");
    fs::create_dir_all(&source_dir).unwrap();
    fs::write(source_dir.join("model.wgt"), source).unwrap();
    let before = snapshot(root);
    let output = success(
        command(WGT, root)
            .args(["run", "inference/model.wgt"])
            .output()
            .unwrap(),
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(&actual["result"], expected);
    assert_eq!(
        before,
        snapshot(root),
        "loading/inference must neither train nor create artifacts"
    );
}

#[test]
fn training_resume_evaluation_artifacts_and_negative_paths() {
    let temp = scratch();
    let full = temp.path().join("full");
    fixture(&full, 300, false, true);
    for source in ["model.wgt", "tests/evaluation.wgt", "gpu/model.wgt"] {
        success(
            command(WEIGHTC, &full)
                .args(["check", source])
                .output()
                .unwrap(),
        );
    }
    assert!(fs::read_dir(full.join("out")).unwrap().next().is_none());
    let report = run(&full);
    let metrics = &report["metrics"];
    assert!(
        metrics["final_loss"].as_f64().unwrap() < metrics["initial_loss"].as_f64().unwrap() * 0.01
    );
    assert!(metrics["held_out_mse"].as_f64().unwrap() < 1e-6);
    eprintln!("lifecycle training metrics: {metrics}");
    assert_checkpoint(&full, 300);

    // Separate CLI processes make this a real interruption, not two handles in one runtime.
    let first = temp.path().join("first");
    fixture(&first, 100, false, false);
    run(&first);
    assert_checkpoint(&first, 100);
    let resumed = temp.path().join("resumed");
    fixture(&resumed, 200, true, true);
    fs::copy(
        first.join("out/checkpoint.json"),
        resumed.join("out/resume.json"),
    )
    .unwrap();
    run(&resumed);
    assert_checkpoint(&resumed, 300);
    for name in ["model.json", "checkpoint.json"] {
        assert_eq!(
            fs::read(full.join("out").join(name)).unwrap(),
            fs::read(resumed.join("out").join(name)).unwrap(),
            "deterministic resume: {name}"
        );
    }
    let before = snapshot(&full);
    let output = success(command(WGT, &full).arg("test").output().unwrap());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.contains("test_held_out_loss") && text.contains("test_cpu_benchmark"),
        "{text}"
    );
    assert_eq!(before, snapshot(&full));
    infer(&full, &full.join("out/model.json"), &metrics["prediction"]);

    let before = snapshot(&full);
    let duplicate = command(WGT, &full)
        .args(["run", "model.wgt"])
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("exist"));
    assert_eq!(before, snapshot(&full));
    let corrupt = temp.path().join("corrupt");
    fixture(&corrupt, 200, true, true);
    let mut checkpoint = read_json(first.join("out/checkpoint.json"));
    checkpoint["format"] = json!("broken");
    fs::write(corrupt.join("out/resume.json"), checkpoint.to_string()).unwrap();
    let failed = command(WGT, &corrupt)
        .args(["run", "model.wgt"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("unsupported checkpoint format"));
    assert!(!corrupt.join("out/model.json").exists());

    let build = temp.path().join("build");
    fixture(&build, 300, false, true);
    fs::write(
        build.join("run.jsonl"),
        json!({"steps":300,"resume":false,"complete":true,"build":true}).to_string(),
    )
    .unwrap();
    let artifacts = temp.path().join("artifacts");
    let artifact = artifacts.join("LifecycleRegression");
    let output = success(
        command(WEIGHTC, &build)
            .args(["build", "model.wgt", "-o"])
            .arg(&artifact)
            .output()
            .unwrap(),
    );
    let manifest: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(manifest["training_steps"], 300);
    assert_eq!(
        fs::read(artifact.join("model.json")).unwrap(),
        fs::read(full.join("out/model.json")).unwrap()
    );
    assert_eq!(
        fs::read(artifact.join("checkpoint.json")).unwrap(),
        fs::read(full.join("out/checkpoint.json")).unwrap()
    );
    let before = snapshot(&artifact);
    let duplicate = command(WEIGHTC, &build)
        .args(["build", "model.wgt", "-o"])
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("output already exists"));
    assert_eq!(before, snapshot(&artifact));
    infer(&build, &artifact.join("model.json"), &metrics["prediction"]);

    // release-assets is the native packaging path; it requires a clean local Git source tree.
    let package = temp.path().join("package");
    fs::create_dir(&package).unwrap();
    for name in ["model.wgt", "data.jsonl", "run.jsonl", "wgt.mod"] {
        fs::copy(build.join(name), package.join(name)).unwrap();
    }
    for args in [
        vec!["init", "-q"],
        vec!["add", "."],
        vec![
            "-c",
            "user.name=Lifecycle",
            "-c",
            "user.email=lifecycle@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        success(
            Command::new("git")
                .current_dir(&package)
                .args(args)
                .output()
                .unwrap(),
        );
    }
    // Only git is exposed for packaging. Neither python nor python3 can be resolved.
    let tools = temp.path().join("tools");
    fs::create_dir(&tools).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("/usr/bin/git", tools.join("git")).unwrap();
    let release = temp.path().join("release");
    success(
        command(WGT, &package)
            .env("PATH", &tools)
            .args(["release-assets", "--version", "v0.5.0", "--artifacts"])
            .arg(&artifacts)
            .arg("-o")
            .arg(&release)
            .output()
            .unwrap(),
    );
    let index = read_json(release.join("wgt.index.json"));
    let files = index["artifacts"]["LifecycleRegression"]["files"]
        .as_object()
        .unwrap();
    for record in files.values() {
        let bytes = fs::read(release.join(record["asset"].as_str().unwrap())).unwrap();
        assert_eq!(
            weightlang_wgt::artifact::hash(&bytes),
            record["sha256"].as_str().unwrap()
        );
        assert_eq!(bytes.len() as u64, record["size"].as_u64().unwrap());
    }
    infer(
        &build,
        &release.join("LifecycleRegression.model.json"),
        &metrics["prediction"],
    );
    let before = snapshot(&release);
    let duplicate = command(WGT, &package)
        .env("PATH", &tools)
        .args(["release-assets", "--version", "v0.5.0", "--artifacts"])
        .arg(&artifacts)
        .arg("-o")
        .arg(&release)
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("output directory must be empty"));
    assert_eq!(before, snapshot(&release));
}

#[test]
fn aot_matches_interpreter_and_runs_without_toolchain_or_python() {
    let temp = scratch();
    let interpreted = temp.path().join("interpreted");
    let compiled = temp.path().join("compiled");
    fixture(&interpreted, 300, false, true);
    fixture(&compiled, 300, false, true);
    let expected = run(&interpreted);
    let executable = temp.path().join("lifecycle");
    // Compilation needs Cargo/rustc/linker; execution below has no PATH at all.
    success(
        Command::new(WEIGHTC)
            .current_dir(temp.path())
            .arg("compile")
            .arg(compiled.join("model.wgt"))
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap(),
    );
    assert!(
        fs::read_dir(compiled.join("out")).unwrap().next().is_none(),
        "compilation must not execute training"
    );
    fs::remove_file(compiled.join("model.wgt")).unwrap();
    let output = success(
        command(executable.to_str().unwrap(), &compiled)
            .arg("--root")
            .arg(&compiled)
            .output()
            .unwrap(),
    );
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, expected);
    for name in ["model.json", "checkpoint.json"] {
        assert_eq!(
            fs::read(interpreted.join("out").join(name)).unwrap(),
            fs::read(compiled.join("out").join(name)).unwrap()
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_benchmark_when_explicitly_required() {
    if std::env::var("WGT_REQUIRE_GPU").as_deref() != Ok("1") {
        eprintln!("GPU lifecycle skipped: set WGT_REQUIRE_GPU=1 to require hardware");
        return;
    }
    let temp = scratch();
    fixture(temp.path(), 300, false, true);
    run(temp.path());
    let output = success(
        command(WGT, temp.path())
            .args(["run", "gpu/model.wgt"])
            .output()
            .unwrap(),
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_ne!(report["result"]["device"], "cpu");
    eprintln!("GPU lifecycle benchmark: {}", report["result"]);
}

#[test]
fn package_and_export_native_artifacts() {
    let temp = scratch();
    fixture(temp.path(), 300, false, true);
    fs::write(
        temp.path().join("run.jsonl"),
        json!({"steps":300,"resume":false,"complete":true,"build":true}).to_string(),
    )
    .unwrap();
    success(
        command(WEIGHTC, temp.path())
            .args(["build", "model.wgt", "-o", "built"])
            .output()
            .unwrap(),
    );
    success(
        command(WGT, temp.path())
            .args(["package", "--entry", "built", "-o", "packaged"])
            .output()
            .unwrap(),
    );
    success(
        command(WGT, temp.path())
            .args(["export", "packaged", "-o", "exported.json"])
            .output()
            .unwrap(),
    );
    assert_eq!(
        read_json(temp.path().join("built/model.json")),
        read_json(temp.path().join("exported.json"))
    );
    let expected = read_json(temp.path().join("built/manifest.json"))["report"]["metrics"]
        ["prediction"]
        .clone();
    infer(temp.path(), &temp.path().join("exported.json"), &expected);
}
