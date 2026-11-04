use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn continuous_let_native_executable_matches_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model ContinuousLet
input x: f32[2]
output y: f32[1]
parameter scale: f32[1] = [1]
fn forward(x) {
    let energy = square(x),
        reduced = energy.mean,
        output = mul(reduced, scale);
    return output;
}
fn loss(prediction, target) {
    let residual = prediction - target, squared = square(residual);
    return squared.mean;
}
fn main() {
    seed(7);
    let expected = [random(), random()];
    seed(7);
    let first = random(), second = random();
    assert [first, second] == expected, "left-to-right once";
    let rows = [[1, 3], [2, 6]],
        net = ContinuousLet("sgd"),
        initial = net(rows[0]);
    assert initial == [5], "forward numerical result";
    let before = fit(net, [rows[0]], [[2]], 0.01, "loss"),
        after = fit(net, [rows[0]], [[2]], 0.01, "loss");
    assert after < before, "loss must decrease";
    return [initial, before, after, net(rows[0])];
}
"#;
    fs::write(&source, text).unwrap();
    let expected = script::run(text, dir.path(), 10000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let output = Command::new(executable)
        .env("PATH", "/nonexistent")
        .args(["--max-operations", "10000"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, serde_json::to_value(expected).unwrap());
}
