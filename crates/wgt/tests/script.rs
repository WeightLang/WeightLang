use std::path::Path;
use std::process::Command;

#[test]
fn native_script_cli_does_not_need_python_or_a_project_environment() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/native_training/supervised/model.wgt");
    for check in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
        command
            .current_dir(std::env::temp_dir())
            .env("PATH", "/nonexistent")
            .args(["script", example.to_str().unwrap()]);
        if check {
            command.arg("--check");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !check {
            let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(report["metrics"]["final_batch_loss"].as_f64().unwrap() < 0.01);
        }
    }
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .args(["script", example.to_str().unwrap(), "--max-operations", "1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("budget exceeded"));
}
