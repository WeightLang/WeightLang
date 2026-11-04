use std::{fs, path::Path, process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;

#[test]
fn typed_ppo_aot_preserves_training_and_probability_evaluation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = include_str!("../../../examples/native_training/ppo/model.wgt");
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let expected = script::run(text, dir.path(), 10_000_000).unwrap();
    let output = Command::new(&executable).env("PATH", "/nonexistent")
        .arg("--root").arg(dir.path()).args(["--max-operations", "10000000"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = serde_json::to_value(expected).unwrap();
    assert_eq!(actual["operations"], expected["operations"]);
    for key in ["initial_success_probability", "final_success_probability"] {
        assert!((actual["metrics"][key].as_f64().unwrap()-expected["metrics"][key].as_f64().unwrap()).abs()<1e-12);
    }
    assert!((actual["result"].as_f64().unwrap()-expected["result"].as_f64().unwrap()).abs()<1e-12);
    assert!(actual["result"].as_f64().unwrap()>0.85);
}
