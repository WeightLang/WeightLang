use std::{fs, path::Path, process::Command};
use weightlang_wgt::script_compiler;

// Compile once, then change execution-time grants. The generated binary must use
// the same accounting and restrictions as both native CLI execution paths.
#[test]
fn resources_match_cli_and_real_aot_with_runtime_grants() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("program");
    let policy = dir.path().join("policy.json");
    let text = r#"wgt 0.5
fn read_then_fail() {
    let read = compute_call("file_read.v1", ["data", 2]);
    assert read["ok"], "authorized read failed";
    assert false, "child failed after consuming bytes";
}
fn main() {
    compute_budget(10000, 4096, 10, 10000);
    let child = compute_scope("read_then_fail", [], [200,4096,3,1000,2,0,0,0,0]);
    assert !child["ok"], "child should fail";
    let read = compute_call("file_read.v1", ["data", 3]);
    assert read["ok"] && read["value"] == "abc", "parent grant was not restored";
    let denied = compute_call("file_read.v1", ["data", 1]);
    assert denied["error"] == "disk_read_exhausted", "failed child was refunded";
    let status = compute_host_status();
    return [status["disk_read_bytes"], status["disk_read_remaining"], compute_status()["calls_remaining"]];
}"#;
    fs::write(&source, text).unwrap();
    fs::write(dir.path().join("input.txt"), "abcdef").unwrap();
    let grants = serde_json::json!({"work":10000,"workspace_bytes":4096,"calls":10,"deadline_ms":10000,
        "disk_read_bytes":5,"files":{"data":dir.path().join("input.txt")}});
    fs::write(&policy, serde_json::to_vec(&grants).unwrap()).unwrap();
    // Compilation must not read policy or execute the main function.
    script_compiler::compile(&source, &executable).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .args(["run", source.to_str().unwrap()])
        .env("WGT_HOST_POLICY", &policy)
        .env("WGT_WEIGHTFS", "off")
        .output()
        .unwrap();
    assert!(
        interpreted.status.success(),
        "{}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let expected: serde_json::Value = serde_json::from_slice(&interpreted.stdout).unwrap();
    assert_eq!(expected["result"], serde_json::json!([5.0, 0.0, 6.0]));
    fs::remove_file(&source).unwrap();
    let run = || {
        Command::new(&executable)
            .env("PATH", "/nonexistent")
            .env("WGT_HOST_POLICY", &policy)
            .env("WGT_WEIGHTFS", "off")
            .output()
            .unwrap()
    };
    let compiled = run();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&compiled.stdout).unwrap();
    assert_eq!(actual["result"], expected["result"]);
    let mut changed = grants;
    changed["disk_read_bytes"] = 1.into();
    fs::write(&policy, serde_json::to_vec(&changed).unwrap()).unwrap();
    let denied = run();
    assert!(
        !denied.status.success(),
        "compiled binary reused old authority"
    );
}
