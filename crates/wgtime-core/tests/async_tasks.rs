//! End-to-end contracts for the first native async language delivery.
use std::path::Path;
use wgtime_core::script;

#[test]
fn async_syntax_calls_ignore_local_bindings_like_synchronous_calls() {
    for (qualifier, invocation) in [("", "value(record)"), ("async ", "await value(record)")] {
        let source = format!(
            "wgt 0.5\n{qualifier}fn value(record){{let __record_field=0;return record.values[1];}}\nfn main(){{let record={{values:[1,2]}};return {invocation};}}"
        );
        assert_eq!(
            script::run(&source, Path::new("."), 100_000).unwrap().result,
            serde_json::json!(2.)
        );

        let source = source.replace("let __record_field=0;return record.values[1];", "let len=0;return len(record.values);");
        let error = script::run(&source, Path::new("."), 100_000).unwrap_err();
        assert!(format!("{error:#}").contains("local binding len is not callable"));
    }
}

#[test]
fn async_read_charges_exact_successful_bytes() {
    let root=std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/async-scheduler-sol/budget-test");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"),b"abc").unwrap();
    std::fs::write(root.join("b.txt"),b"abcde").unwrap();
    let source=|path:&str|format!("wgt 0.5\nasync fn read(){{return await load text from \"{path}\";}}\nfn main(){{return await read();}}");
    let a=script::run(&source("a.txt"),&root,100000).unwrap();
    let b=script::run(&source("b.txt"),&root,100000).unwrap();
    assert_eq!(b.operations-a.operations,2,"poll count must not change byte billing");
    std::fs::remove_file(root.join("a.txt")).unwrap();
    std::fs::remove_file(root.join("b.txt")).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn async_user_functions_are_cold_and_nested_await_executes() {
    let source = r#"wgt 0.5
async fn bad() { assert false, "cold body executed"; }
async fn increment(x) { return x + 1; }
async fn twice(x) { return await increment(await increment(x)); }
fn main() {
    let unused = bad();
    tasks { let task = spawn twice(40); return await task; }
}
"#;
    assert_eq!(
        script::run(source, Path::new("."), 100_000)
            .unwrap()
            .result
            .as_f64(),
        Some(42.0)
    );
}

#[test]
fn await_rejects_synchronous_values() {
    let error = script::run(
        "wgt 0.5\nfn main() { return await 1; }",
        Path::new("."),
        1000,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("future or task"));
}

#[test]
fn async_budget_does_not_reset_in_child_runtimes() {
    let source = "wgt 0.5\nasync fn spin() { while true { let x = 1; } }\nfn main() { return await spin(); }";
    let error = script::run(source, Path::new("."), 100).unwrap_err();
    assert!(format!("{error:#}").contains("budget"));
}

#[test]
fn async_resource_creation_through_sync_helper_is_rejected() {
    let source = "wgt 0.5\nfn helper() { return read_jsonl(\"missing.jsonl\"); }\nasync fn child() { return helper(); }\nfn main() { return await child(); }";
    let error = script::run(source, Path::new("."), 1000).unwrap_err();
    assert!(format!("{error:#}").contains("async"));
}

#[test]
fn async_sleep_overlaps_and_text_reads_execute_on_pool() {
    use std::time::{Duration, Instant};
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/async-scheduler-sol/io-test");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("input.txt"), "hello async").unwrap();
    let source = r#"wgt 0.5
async fn work() { await sleep(200); return await load text from "input.txt"; }
fn main() { tasks { let a=spawn work(); let b=spawn work(); return [await a,await b]; } }
"#;
    let start = Instant::now();
    let report = script::run(source, &root, 1_000_000).unwrap();
    assert!(
        start.elapsed() < Duration::from_millis(380),
        "two 200ms sleeps did not overlap"
    );
    assert_eq!(
        report.result,
        serde_json::json!(["hello async", "hello async"])
    );
    std::fs::remove_file(root.join("input.txt")).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn async_scope_rejects_active_compute_budget() {
    let source="wgt 0.5\nasync fn child(){return 1;}\nfn main(){compute_budget(100,1024,2,1000);return await child();}";
    let error = script::run(source, Path::new("."), 1000).unwrap_err();
    assert!(format!("{error:#}").contains("async"));
}

#[test]
fn task_failure_survives_scope_join() {
    let source="wgt 0.5\nasync fn bad(){assert false,\"child failure\";}\nfn main(){tasks {let task=spawn bad();}return 1;}";
    let error = script::run(source, Path::new("."), 1000).unwrap_err();
    assert!(format!("{error:#}").contains("child failure"));
}

#[test]
fn task_rng_is_reproducible_and_split_by_spawn_order() {
    let source="wgt 0.5\nasync fn draw(){await sleep(1);return random();}\nfn main(){seed(42);tasks{let a=spawn draw();let b=spawn draw();return [await a,await b];}}";
    let first = script::run(source, Path::new("."), 100000).unwrap().result;
    let second = script::run(source, Path::new("."), 100000).unwrap().result;
    assert_eq!(first, second);
    assert_ne!(first[0], first[1]);
}

#[test]
fn async_limits_scope_and_model_resources_are_explicit() {
    for (source, expected) in [
        ("wgt 0.5\nasync fn child(){return 1;}\nfn main(){return spawn child();}","tasks scope"),
        ("wgt 0.5\nasync fn child(){tasks{return 1;}}\nfn main(){return await child();}","unavailable"),
        ("wgt 0.5\nasync fn main(){return 1;}","synchronous"),
        ("wgt 0.5\nasync fn recurse(){return await recurse();}\nfn main(){return await recurse();}","recursion limit"),
        ("wgt 0.5\nmodel M\ninput x: f32[1]\noutput y: f32[1]\nlayer id: identity\ninit seed=1\nasync fn child(x){return 1;}\nfn main(){let net=M(\"sgd\");return await child([net]);}","plain data"),
    ] {
        let error=script::run(source,Path::new("."),100000).unwrap_err();
        assert!(format!("{error:#}").contains(expected),"{error:#}");
    }
}

#[test]
fn host_policy_is_rejected_before_async_body_or_io() {
    if std::env::var_os("WGT_ASYNC_POLICY_TEST").is_some() {
        let source="wgt 0.5\nasync fn child(){return await load text from \"missing\";}\nfn main(){return await child();}";
        let error = script::run(source, Path::new("."), 100000).unwrap_err();
        assert!(format!("{error:#}").contains("async"));
        return;
    }
    let root = std::path::PathBuf::from("/tmp/weightlang");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("async-policy-{}.json", std::process::id()));
    std::fs::write(
        &path,
        r#"{"work":100000,"workspace_bytes":4096,"calls":2,"deadline_ms":10000}"#,
    )
    .unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "host_policy_is_rejected_before_async_body_or_io"])
        .env("WGT_ASYNC_POLICY_TEST", "1")
        .env("WGT_HOST_POLICY", &path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn unspawned_future_is_not_a_serializable_result() {
    let source = "wgt 0.5\nasync fn child() { return 1; }\nfn main() { return child(); }";
    assert!(script::run(source, Path::new("."), 1000).is_err());
}
