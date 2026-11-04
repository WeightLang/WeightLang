use std::{fs, path::Path, process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;

#[test]
fn compiled_and_interpreted_share_semantics_without_source_or_toolchain_at_run_time() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("test.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
fn factorial(n) { if n <= 1 { return 1; } return n * factorial(n - 1); }
fn early() { for i in range(0, 5) { while true { return 7; } } }
fn bump(model) { return fit(model, [[0], [1]], [[1], [3]], 0.05, "mse"); }
fn main() {
    let reader = jsonl("case.jsonl");
    let rows = batch(reader, 1);
    close(reader);
    let mode = rows[0]["mode"];
    if mode == 1 { return 1 / 0; }
    if mode == 2 { return missing; }
    if mode == 3 { return factorial(100); }
    if mode == 4 { let x = []; return x[0]; }
    if mode == 5 { return false || 2; }
    if mode == 6 { absent = random(); }
    if mode == 7 { return factorial(); }
    assert(true || unknown(), "short circuit");
    assert(!(false && unknown()), "short circuit");
    let n = 0;
    for i in range(0, 10) {
        if i == 2 { continue; }
        if i == 6 { break; }
        n = n + i;
    }
    while n < 20 { n = n + 1; }
    if true { let scoped = 3; }
    let scoped = scoped + 1;
    seed(17);
    let model = mlp([1,1], "identity", 42, "adam");
    for step in range(0, 100) { bump(model); }
    save_model(model, "model.json");
    save_checkpoint(model, "checkpoint.json");
    metric("summary", [n, scoped, early(), factorial(5), random(), "x" + "y"]);
    return model([0.25]);
}
"#;
    fs::write(&source, text).unwrap();
    let code = script::codegen::rust_source(text, ".").unwrap();
    assert!(!code.contains("script::run"));
    assert!(!code.contains("Expr::"));
    assert!(code.contains("while "));
    script_compiler::compile(&source, &executable).unwrap();
    assert!(
        !dir.path().join("model.json").exists(),
        "compiling must not execute training"
    );
    assert!(script_compiler::compile(&source, &executable).is_err());
    fs::remove_file(&source).unwrap();
    for mode in 0..8 {
        let interpreted = dir.path().join(format!("interpreted-{mode}"));
        let compiled = dir.path().join(format!("compiled-{mode}"));
        for root in [&interpreted, &compiled] {
            fs::create_dir(root).unwrap();
            fs::write(root.join("case.jsonl"), format!("{{\"mode\":{mode}}}\n")).unwrap();
        }
        let reference = script::run(text, &interpreted, 100000);
        let actual = Command::new(&executable)
            .env("PATH", "/nonexistent")
            .arg("--root")
            .arg(&compiled)
            .args(["--max-operations", "100000"])
            .output()
            .unwrap();
        if mode == 0 {
            assert!(
                actual.status.success(),
                "{}",
                String::from_utf8_lossy(&actual.stderr)
            );
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(serde_json::to_value(reference.unwrap()).unwrap(), actual);
            for name in ["model.json", "checkpoint.json"] {
                assert_eq!(
                    fs::read(interpreted.join(name)).unwrap(),
                    fs::read(compiled.join(name)).unwrap()
                );
            }
        } else {
            let error = format!("{:#}", reference.unwrap_err());
            assert!(!actual.status.success());
            let last = error.rsplit(": ").next().unwrap();
            assert!(
                String::from_utf8_lossy(&actual.stderr).contains(last),
                "mode {mode}: {} vs {error}",
                String::from_utf8_lossy(&actual.stderr)
            );
        }
    }
    let limited = Command::new(&executable)
        .args(["--max-operations", "1"])
        .output()
        .unwrap();
    assert!(!limited.status.success());
    assert!(String::from_utf8_lossy(&limited.stderr).contains("budget exceeded"));
}

#[test]
fn compiled_cli_runs_existing_reinforcement_script() {
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/native_training/reinforce.wgt");
    let reference = script::run_file(&source, 10_000_000).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .arg("script")
        .arg(&source)
        .arg("--compiled")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(serde_json::to_value(reference).unwrap(), actual);
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .arg("script")
        .arg(&source)
        .args(["--compiled", "--check"])
        .output()
        .unwrap();
    assert!(!result.status.success());
}

#[test]
fn unified_model_and_forward_match_in_native_executables() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    for (name, text) in [
        ("graph", "wgt 0.5\nmodel Net\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(out=1)\ninit seed=7\nfn main() { let m = Net(\"sgd\"); for i in range(0, 100) { fit(m, [[0], [1]], [[1], [3]], 0.05, \"mse\"); } return m([1]); }"),
        ("forward", include_str!("../../../examples/self_modifying/model.wgt")),
    ] {
        let source = dir.path().join(format!("{name}.wgt"));
        let executable = dir.path().join(name);
        fs::write(&source, text).unwrap();
        let expected = script::run(text, dir.path(), 100_000).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        let output = Command::new(&executable).env("PATH", "/nonexistent")
            .args(["--max-operations", "100000"]).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
    }
}

#[test]
fn compiled_ppo_uses_native_training_syntax() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("ppo");
    let text = include_str!("../../../examples/native_training/ppo/model.wgt");
    fs::write(&source, text).unwrap();
    let expected = script::run(text, dir.path(), 10_000_000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let output = Command::new(&executable).env("PATH", "/nonexistent").output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected=serde_json::to_value(expected).unwrap();
    // JSON parsing and optimized host f64 arithmetic may differ by one ULP.
    // Dispatch counts and the initial baseline must still match exactly.
    assert_eq!(actual["operations"],expected["operations"]);
    assert_eq!(actual["metrics"]["initial_success_probability"],expected["metrics"]["initial_success_probability"]);
    for (a,b) in [(&actual["result"],&expected["result"]),(&actual["metrics"]["final_success_probability"],&expected["metrics"]["final_success_probability"])] {
        assert!((a.as_f64().unwrap()-b.as_f64().unwrap()).abs()<1e-12,"{a} != {b}");
    }
}
