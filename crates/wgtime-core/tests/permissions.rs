//! Run policy cases in child processes: environment changes must not race other tests.
use std::{path::Path, process::Command};
use wgtime_core::script::{self, compiled, Env, Flow, Runtime, Value};

fn main_body(runtime: &mut Runtime, _: &mut Env) -> anyhow::Result<Flow> {
    runtime.tick()?;
    let result = if std::env::var("WGT_PERMISSION_CASE").unwrap() == "io" {
        runtime.call(
            "jsonl",
            vec![Value::Text("missing-permission-test.jsonl".into())],
        )?
    } else {
        Value::Number(7.)
    };
    Ok(Flow::Return(result))
}

#[test]
fn interpreter_and_aot_share_host_permissions() {
    if let Ok(case) = std::env::var("WGT_PERMISSION_CASE") {
        let source = if case == "io" {
            "wgt 0.5\nfn main() { return jsonl(\"missing-permission-test.jsonl\"); }"
        } else {
            "wgt 0.5\nfn main() { return 7; }"
        };
        let interpreted = script::run(source, Path::new("."), 100);
        let compiled = compiled::run(
            vec![(
                "main",
                compiled::Function {
                    args: vec![],
                    body: main_body,
                    async_body: None,
                },
            )],
            Path::new("."),
            100,
        );
        for result in [interpreted, compiled] {
            match case.as_str() {
                "none" | "basic" => assert_eq!(result.unwrap().result, serde_json::json!(7.0)),
                "io" => assert!(format!("{:#}", result.err().unwrap())
                    .contains("host policy requires capability-based I/O")),
                "expired" => assert!(
                    format!("{:#}", result.err().unwrap()).contains("host deadline exceeded")
                ),
                _ => panic!("unknown case"),
            }
        }
        return;
    }
    let root = Path::new("/tmp/weightlang").join(format!("permissions-{}", std::process::id()));
    std::fs::create_dir_all(root.parent().unwrap()).unwrap();
    std::fs::create_dir(&root).unwrap();
    let policy = root.join("policy.json");
    for case in ["none", "basic", "io", "expired"] {
        std::fs::write(&policy, serde_json::to_vec(&serde_json::json!({"work":100,"workspace_bytes":4096,"calls":2,"deadline_ms":if case=="expired" {0} else {10000}})).unwrap()).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "interpreter_and_aot_share_host_permissions",
                "--nocapture",
            ])
            .env("WGT_PERMISSION_CASE", case)
            .env_remove("WGT_HOST_POLICY")
            .env("WGT_WEIGHTFS", "off");
        if case != "none" {
            child.env("WGT_HOST_POLICY", &policy);
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "case={case}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_file(policy).unwrap();
    std::fs::remove_dir(root).unwrap();
}
