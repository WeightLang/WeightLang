use std::{fs, path::Path, process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;

#[test]
fn continuous_let_native_executable_matches_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = include_str!("../../../examples/native_training/continuous_let/model.wgt");
    fs::write(&source, text).unwrap();
    let expected = script::run(text, dir.path(), 10000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let output = Command::new(executable)
        .env("PATH", "/nonexistent")
        .args(["--max-operations", "10000"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, serde_json::to_value(expected).unwrap());
}
