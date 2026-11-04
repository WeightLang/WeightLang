use std::{fs, process::{Child, Command}, time::{Duration, Instant}};
use wgtime_core::{Model, Session};

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn daemon() -> (Daemon, tempfile::TempDir, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    let socket_root = tempfile::Builder::new()
        .prefix("native-build-ipc-")
        .tempdir_in(&root)
        .unwrap();
    fs::set_permissions(socket_root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let binary = std::env::var_os("WGTIME_TEST_DAEMON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/debug/wgtime")
        });
    assert!(binary.is_file(), "build wgtime-service first or set WGTIME_TEST_DAEMON");
    let child = Command::new(binary)
        .arg("--test-directory")
        .arg(socket_root.path())
        .spawn()
        .unwrap();
    let mut daemon = Daemon(child);
    let endpoint = socket_root.path().join("service.sock");
    let begin = Instant::now();
    while !endpoint.exists() {
        assert!(begin.elapsed() < Duration::from_secs(10), "wgtime service startup timed out");
        assert!(daemon.0.try_wait().unwrap().is_none(), "wgtime service exited during startup");
        std::thread::sleep(Duration::from_millis(5));
    }
    (daemon, socket_root, endpoint)
}
fn cli(endpoint: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_weightc"))
        .env("WGTIME_ENDPOINT", endpoint)
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
    let (_daemon, _socket_root, endpoint) = daemon();
    let dir = temp();
    let output = dir.path().join("model");
    let source = dir.path().join("model.wgt");
    fs::write(
        &source,
        r#"wgt 0.5
model Regression
input x: f32[1]
output y: f32[1]
layer output: linear(out=1)
init seed=42
fn main() {
    let net = Regression("adam");
    let final_batch_loss = 0;
    for step in range(0, 300) {
        final_batch_loss = train net on [[-1], [0], [1]] against [[-1], [1], [3]] {rate:0.05;loss:mse;};
    }
    metric("final_batch_loss", final_batch_loss);
    return net;
}
"#,
    )
    .unwrap();
    let args = ["build", source.to_str().unwrap(), "-o", output.to_str().unwrap()];
    let result = cli(&endpoint, &args);
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
        wgtime_core::training::Trainer::restore(&output.join("checkpoint.json"), 1048576)
            .unwrap()
            .steps()
            > 0
    );
    assert!(!cli(&endpoint, &args).status.success());
    assert_eq!(bytes, fs::read(output.join("model.json")).unwrap());
}
#[test]
fn check_does_not_execute_and_failed_build_does_not_publish() {
    let dir = temp();
    let endpoint = dir.path().join("missing-service.sock");
    let source = dir.path().join("source.wgt");
    let output = dir.path().join("out");
    fs::write(
        &source,
        "wgt 0.5\nfn main() { assert false, \"failure\"; }",
    )
    .unwrap();
    let check = cli(&endpoint, &["check", source.to_str().unwrap()]);
    assert!(check.status.success(), "{}", String::from_utf8_lossy(&check.stderr));

    let (_daemon, _socket_root, endpoint) = daemon();
    let failed = cli(&endpoint, &[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap()
    ]);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("line 2: failure"), "{}", String::from_utf8_lossy(&failed.stderr));
    assert!(!output.exists());

    fs::write(&source, "wgt 0.5\nfn main() { return 3; }").unwrap();
    let failed = cli(&endpoint, &[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap()
    ]);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("model"), "{}", String::from_utf8_lossy(&failed.stderr));
    assert!(!output.exists());

    fs::write(
        &source,
        "wgt 0.5\nfn main() { return mlp([1,1], \"identity\", 1, \"sgd\"); }",
    )
    .unwrap();
    let failed = cli(&endpoint, &[
        "build",
        source.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--max-operations",
        "1"
    ]);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("operation"), "{}", String::from_utf8_lossy(&failed.stderr));
    assert!(!output.exists());

    let result = cli(&endpoint, &[
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
