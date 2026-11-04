use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn postfix_mean_native_executable_matches_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model MeanEnergy
input x: f32[2]
output y: f32[1]
parameter scale: f32[1] = [1]
fn forward(x) {
    return mul(square(x).mean, scale);
}
fn loss(prediction, target) {
    return square(prediction - target).mean;
}
fn main() {
    let rows = [[1, 3], [2, 6]];
    assert(rows[0].mean == 2, "row mean");
    assert([1, 2, 3].mean == mean([1, 2, 3]), "function compatibility");
    let model = MeanEnergy("sgd");
    assert(model([1, 3]) == [5], "forward mean");
    let before = fit(model, [[1, 3]], [[2]], 0.01, "loss");
    let after = fit(model, [[1, 3]], [[2]], 0.01, "loss");
    assert(after < before, "mean gradient must train scale");
    return [rows[1].mean, before, after, model([1, 3])];
}
"#;
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
