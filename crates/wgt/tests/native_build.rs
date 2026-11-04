use std::{fs, process::Command};
use weightlang_runtime::{Model, Session};

fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_weightc"))
        .env("PATH", "/nonexistent")
        .args(args)
        .output()
        .unwrap()
}
fn temp() -> tempfile::TempDir {
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
#[test]
fn build_train_reload_and_protect_without_python() {
    let dir = temp();
    let output = dir.path().join("model");
    let example = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/native_training/build/model.wgt");
    let args = [
        "build",
        example.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ];
    let result = cli(&args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(report["training_steps"].as_u64().unwrap() > 0);
    assert!(
        report["report"]["metrics"]["final_batch_loss"]
            .as_f64()
            .unwrap()
            < 0.01
    );
    let bytes = fs::read(output.join("model.json")).unwrap();
    let model: Model = serde_json::from_slice(&bytes).unwrap();
    let mut session = Session::prepare(model, &output, 1048576).unwrap();
    assert!((session.run(&[0.25]).unwrap()[0] - 1.5).abs() < 0.15);
    assert!(
        weightlang_runtime::training::Trainer::restore(&output.join("checkpoint.json"), 1048576)
            .unwrap()
            .steps()
            > 0
    );
    assert!(!cli(&args).status.success());
    assert_eq!(bytes, fs::read(output.join("model.json")).unwrap());
}
#[test]
fn check_does_not_execute_and_failed_build_does_not_publish() {
    let dir = temp();
    let source = dir.path().join("source.wgt");
    let output = dir.path().join("out");
    fs::write(
        &source,
        "wgt 0.5\nfn main() { assert(false, \"failure\"); }",
    )
    .unwrap();
    assert!(cli(&["check", source.to_str().unwrap()]).status.success());
    assert!(!cli(&[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap()
    ])
    .status
    .success());
    assert!(!output.exists());
    fs::write(&source, "wgt 0.5\nfn main() { return 3; }").unwrap();
    assert!(!cli(&[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap()
    ])
    .status
    .success());
    assert!(!output.exists());
    fs::write(
        &source,
        "wgt 0.5\nfn main() { return mlp([1,1], \"identity\", 1, \"sgd\"); }",
    )
    .unwrap();
    assert!(!cli(&[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--max-operations",
        "1"
    ])
    .status
    .success());
    assert!(!output.exists());
    let result = cli(&[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ]);
    assert!(result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["state"], "initialized");
    assert_eq!(report["training_steps"], 0);
}
