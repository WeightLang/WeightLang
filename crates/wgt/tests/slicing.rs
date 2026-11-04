use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn slicing_aot_matches_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model SliceWindow
input x: f32[4]
output y: f32[2]
parameter scale: f32[2] = [1,1]
fn forward(x) { return (x[:])[1:3] * scale; }
fn loss(prediction, truth) {
    return square((prediction - truth)[:]).mean;
}
fn main() {
    let row = [10,20,30,40];
    let start = 1;
    assert(row[start:3] == [20,30], "runtime list endpoints");
    assert(row[:2] == [10,20], "prefix");
    assert(row[2:] == [30,40], "suffix");
    assert(row[4:] == [], "empty list slice");
    assert(row[:][1:3][0] == 20, "nested slicing");
    let net = SliceWindow("sgd");
    assert(net([1,2,3,4]) == [2,3], "initialized forward");
    fit(net, [[1,2,3,4]], [[0,0]], 0.01, "loss");
    assert(net([1,2,3,4])[0] < 2, "selected features train");
    return true;
}
"#;
    fs::write(&source, &text).unwrap();
    let expected = script::run(&text, dir.path(), 10000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(&source).unwrap();
    let output = Command::new(&executable).env("PATH", "/nonexistent")
        .args(["--max-operations", "10000"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, serde_json::to_value(expected).unwrap());

    // The same out-of-bounds error must survive generated Rust execution.
    let invalid = "wgt 0.5\nfn main() { return [1,2][:3]; }";
    let interpreted = script::run(invalid, dir.path(), 10000).unwrap_err();
    assert!(format!("{interpreted:#}").contains("slice endpoint out of bounds"));
    fs::write(&source, invalid).unwrap();
    let failing = dir.path().join("invalid");
    script_compiler::compile(&source, &failing).unwrap();
    let output = Command::new(failing).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("slice endpoint out of bounds"));
}
