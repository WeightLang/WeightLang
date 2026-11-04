#![cfg(target_os = "linux")]
use std::{
    fs,
    process::Command,
    time::{Duration, Instant},
};
fn execute(name: &str, code: &str, limits: serde_json::Value) -> std::process::Output {
    let root = std::path::PathBuf::from("/tmp/weightlang")
        .join(format!("isolation-{name}-{}", std::process::id()));
    fs::create_dir_all(root.parent().unwrap()).unwrap();
    fs::create_dir(&root).unwrap();
    let script = root.join("main.wgt");
    let policy = root.join("policy.json");
    fs::write(&script, format!("wgt 0.5\nfn main() {{ {code} }}")).unwrap();
    let mut p =
        serde_json::json!({"work":10000,"workspace_bytes":4096,"calls":8,"deadline_ms":10000});
    p.as_object_mut()
        .unwrap()
        .extend(limits.as_object().unwrap().clone());
    fs::write(&policy, serde_json::to_vec(&p).unwrap()).unwrap();
    let start = Instant::now();
    let output = Command::new(transport_test_executable())
        .args(["script", script.to_str().unwrap()])
        .env("WGT_HOST_POLICY", &policy)
        .env_remove("WGT_RESOURCE_WORKER")
        .output()
        .unwrap();
    assert!(start.elapsed() < Duration::from_secs(12));
    fs::remove_file(script).unwrap();
    fs::remove_file(policy).unwrap();
    fs::remove_dir(root).unwrap();
    output
}
#[test]
fn cpu_limit_terminates_native_work() {
    let out = execute(
        "cpu",
        r#"let model = mlp([1000,1000], "tanh", 42, "sgd"); let x=range(0,1000); while true { model(x); }"#,
        serde_json::json!({"cpu_seconds":1,"address_space_bytes":536870912}),
    );
    assert!(!out.status.success());
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(error.contains("signal: 9"), "{error}");
    assert!(!error.contains("operation budget exceeded"), "{error}");
}
#[test]
fn memory_limit_contains_allocation_failure() {
    let out = execute(
        "memory",
        r#"let model = mlp([1999,1999], "tanh", 42, "adam"); return 1;"#,
        serde_json::json!({"address_space_bytes":33554432}),
    );
    assert!(!out.status.success());
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(
        error.contains("memory allocation")
            || error.contains("failed to map")
            || error.contains("Cannot allocate memory"),
        "{error}"
    );
}
#[test]
fn successful_worker_reports_cpu_and_limits() {
    let out = execute(
        "success",
        "return compute_host_status();",
        serde_json::json!({"cpu_seconds":2,"address_space_bytes":536870912}),
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(value["result"]["cpu_time_us"].as_f64().unwrap() > 0.);
    assert_eq!(value["result"]["cpu_limit_seconds"], 2.);
    assert_eq!(value["result"]["address_space_limit_bytes"], 536870912.);
}
#[test]
fn cpu_only_limit_allows_small_program() {
    let out = execute("marker", "return 1;", serde_json::json!({"cpu_seconds":1}));
    assert!(out.status.success());
}

fn transport_test_executable() -> std::path::PathBuf {
    let path = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("examples").join(format!("core-transport-test{}",std::env::consts::EXE_SUFFIX));
    assert!(path.is_file(), "build the private process harness with cargo build -p wgtime-core --example core-transport-test before running this test");
    path
}
