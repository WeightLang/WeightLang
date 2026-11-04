use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn structured_sample_aot_matches_interpreter_including_budget_and_writes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model Policy
input x: f32[3]
parameter memory: f32[3] = [0,0,0]
output y: f32[3]
fn forward(x) { let next=memory+x; write(memory,next); return next; }
fn main() {
 let net=Policy("adam"); seed(91); let result=[];
 for i in range(0,32) {
  let {action:a,log_prob:lp}=sample from net([random(),random(),random()]);
  result=append(result,[a,lp]);
 }
 assert(net([0,0,0])==[0,0,0],"sampling committed write");
 return [result,random()];
}"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let expected = script::run(text, dir.path(), 100_000).unwrap();
    let output = Command::new(&executable).env("PATH", "/nonexistent")
        .arg("--root").arg(dir.path()).args(["--max-operations", "100000"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = serde_json::to_value(expected).unwrap();
    assert_eq!(actual["operations"], expected["operations"]);
    assert_eq!(actual["metrics"], expected["metrics"]);
    assert_eq!(actual["result"][1], expected["result"][1]);
    for (actual, expected) in actual["result"][0].as_array().unwrap().iter()
        .zip(expected["result"][0].as_array().unwrap()) {
        assert_eq!(actual[0], expected[0]);
        // AOT JSON parsing may round the last decimal bit differently from
        // the in-memory interpreter value. Actions, RNG and budget stay exact.
        assert!((actual[1].as_f64().unwrap() - expected[1].as_f64().unwrap()).abs() < 1e-14);
    }
}
