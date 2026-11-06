mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn postfix_mean_native_executable_matches_interpreter() {
    let service = support::Service::start().unwrap();
    let dir = support::scratch().unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model MeanEnergy
input x: f32[2]
output y: f32[1]
parameter scale: f32[1] = [1]
fn forward(x) {
    return mul(square(x).mean, scale);
}
fn loss(prediction, target) {
    return square(prediction - target).mean;
}
fn main() {
    let model = MeanEnergy("sgd");
    assert model([1, 3]) == [5], "forward mean";
    let before = train model on [[1, 3]] against [[2]] {rate:0.01;loss:loss;};
    let after = train model on [[1, 3]] against [[2]] {rate:0.01;loss:loss;};
    assert after < before, "mean gradient must train scale";
    return [before, after, model([1, 3])[0]];
}
"#;
    fs::write(&source, text).unwrap();
    let expected = script::run(text, dir.path(), 10000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let output = service
        .configure(
            Command::new(executable)
                .env("PATH", "/nonexistent")
                .args(["--max-operations", "10000"]),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for index in 0..3 {
        assert_eq!(
            actual["result"][index].as_f64().unwrap() as f32,
            expected.result[index].as_f64().unwrap() as f32
        );
    }
    assert_eq!(
        actual["metrics"],
        serde_json::to_value(expected.metrics).unwrap()
    );
}
