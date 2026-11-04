use std::{fs, path::Path, process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;

#[test]
fn file_syntax_aot_and_cross_process_checkpoint_resume() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
fn main() {
 let mode = load text from "mode.txt";
 if mode == "initialize" {
  let net = mlp([1,1],"identity",4,"adam");
  train net on [[1]] against [[3]] {rate:0.01;loss:mse;};
  save checkpoint net to "checkpoint.json";
  return net([1]);
 }
 assert true, missing();
 let net = load checkpoint from "checkpoint.json";
 let alias = net;
 let initial = net([1]);
 train net on [[1]] against [[3]] {rate:0.01;loss:mse;};
 reload checkpoint alias from "checkpoint.json";
 assert net([1]) == initial;
 train alias on [[1]] against [[3]] {rate:0.01;loss:mse;};
 save json net([1]) to "result.json" overwrite;
 return load json from "result.json";
}"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    fs::write(dir.path().join("mode.txt"), "initialize").unwrap();
    let output = Command::new(&executable)
        .env("PATH", "/nonexistent")
        .arg("--root")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::write(dir.path().join("mode.txt"), "resume").unwrap();
    let expected = script::run(text, dir.path(), 10_000_000).unwrap();
    let output = Command::new(&executable)
        .env("PATH", "/nonexistent")
        .arg("--root")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, serde_json::to_value(expected).unwrap());
    let policy = dir.path().join("policy.json");
    fs::write(
        &policy,
        r#"{"work":100000,"workspace_bytes":4096,"calls":2,"deadline_ms":10000}"#,
    )
    .unwrap();
    let restricted = Command::new(&executable)
        .env("PATH", "/nonexistent")
        .env("WGT_HOST_POLICY", &policy)
        .arg("--root")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!restricted.status.success());
    assert!(String::from_utf8_lossy(&restricted.stderr)
        .contains("host policy requires capability-based I/O"));
    let continuous = script::run(r#"wgt 0.5
fn main(){ let n=mlp([1,1],"identity",4,"adam"); train n on [[1]] against [[3]] {rate:0.01;loss:mse;}; train n on [[1]] against [[3]] {rate:0.01;loss:mse;}; return n([1]); }"#, dir.path(), 100_000).unwrap();
    assert_eq!(actual["result"], continuous.result);
}
