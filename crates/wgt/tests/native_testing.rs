use std::{
    fs,
    process::{Command, Output},
};

fn project() -> tempfile::TempDir {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("wgt.toml"),
        "[project]\nmodule = \"local/native-tests\"\nedition = \"0.5\"\n",
    )
    .unwrap();
    fs::create_dir(project.path().join("tests")).unwrap();
    project
}
fn run(project: &tempfile::TempDir, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wgt"))
        .env("WGT_WEIGHTFS", "off")
        .current_dir(project.path())
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn discovery_filter_failure_isolation_location_and_exit_status() {
    let project = project();
    fs::write(project.path().join("tests/check.wgt"), "wgt 0.5\nfn test_a_fail() {\n assert(false, \"threshold failed\");\n}\nfn test_b_pass() { assert(true, \"ok\"); }\nfn helper() {}\n").unwrap();
    let output = run(&project, &["test"]);
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stdout.contains("test_b_pass"), "{stdout} {stderr}");
    assert!(stdout.contains("1 passed; 1 failed; 2 total"), "{stdout}");
    assert!(stderr.contains("check.wgt:3"), "{stderr}");
    assert!(stderr.contains("threshold failed"));
    let filtered = run(&project, &["test", "--filter", "test_b"]);
    assert!(
        filtered.status.success(),
        "{}",
        String::from_utf8_lossy(&filtered.stderr)
    );
    assert!(String::from_utf8_lossy(&filtered.stdout).contains("1 passed; 0 failed; 1 total"));
    assert!(!run(&project, &["test", "--filter", "missing"])
        .status
        .success());
}

#[test]
fn broken_files_and_json_fixtures_coexist_and_compiled_is_explicitly_rejected() {
    let project = project();
    fs::write(
        project.path().join("tests/broken.wgt"),
        "wgt 0.5\nfn test_broken( {",
    )
    .unwrap();
    fs::write(
        project.path().join("tests/ok.wgt"),
        "wgt 0.5\nfn test_ok() {} ",
    )
    .unwrap();
    fs::write(project.path().join("tests/fixture.json"), "{}").unwrap();
    let result = run(&project, &["test"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed; 2 failed; 3 total"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("test requires source"));
    let result = run(&project, &["test", "--compiled"]);
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("wgt test --compiled is not supported")
    );
}

#[test]
fn successful_native_json_fixture_and_wgt_share_one_summary() {
    let project = project();
    let model =
        serde_json::json!({"version":1,"input":1,"parameters":[],"layers":[{"op":"identity"}]});
    fs::write(project.path().join("model.json"), model.to_string()).unwrap();
    fs::write(
        project.path().join("tests/fixed.json"),
        r#"{"model":"model.json","input":[2],"expected":[2]}"#,
    )
    .unwrap();
    fs::write(
        project.path().join("tests/script.wgt"),
        "wgt 0.5\nfn test_ok() { assert(true, \"ok\"); }",
    )
    .unwrap();
    let result = run(&project, &["test"]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("2 passed; 0 failed; 2 total"));
}

#[test]
fn evaluation_and_benchmark_share_interpreted_and_aot_semantics() {
    use weightlang_runtime::script;
    let cache = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/native-test-task");
    fs::create_dir_all(&cache).unwrap();
    let temp = tempfile::tempdir_in(cache).unwrap();
    let source = r#"wgt 0.5
fn test_evaluation() {
    let reader = jsonl("flag.jsonl");
    assert(batch(reader,1)[0]["ok"], "evaluation flag failed");
    close(reader);
    let net = mlp([1,1], "identity", 42, "sgd");
    let before = net([1]);
    let timing = benchmark(net, [[[0],[1]],[[1],[3]],0.1,"mse"], 2, 4, "step", "train:cpu");
    assert(timing["repeats"] == 4, "repeat count");
    assert(net([1]) != before, "training benchmark must update weights");
    return mse(net([1]), [3]);
}
fn main() { return test_evaluation(); }
"#;
    fs::write(temp.path().join("model.wgt"), source).unwrap();
    fs::write(temp.path().join("flag.jsonl"), "{\"ok\":true}\n").unwrap();
    let interpreted = script::run(source, temp.path(), 10000).unwrap();
    let executable = temp
        .path()
        .join(format!("evaluation{}", std::env::consts::EXE_SUFFIX));
    weightlang_wgt::script_compiler::compile(&temp.path().join("model.wgt"), &executable).unwrap();
    let output = Command::new(&executable)
        .env("WGT_WEIGHTFS", "off")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let compiled: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        (compiled["result"].as_f64().unwrap() - interpreted.result.as_f64().unwrap()).abs() < 1e-12
    );
    assert_eq!(compiled["operations"], interpreted.operations);
    for key in ["device", "mode", "unit", "warmup", "repeats", "scope"] {
        assert_eq!(
            compiled["metrics"]["step"][key],
            interpreted.metrics["step"][key]
        );
    }
    fs::write(temp.path().join("flag.jsonl"), "{\"ok\":false}\n").unwrap();
    let error = script::run(source, temp.path(), 10000).unwrap_err();
    assert!(format!("{error:#}").contains("line 4"));
    let output = Command::new(&executable)
        .env("WGT_WEIGHTFS", "off")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("line 4") && error.contains("evaluation flag failed"),
        "{error}"
    );
}
