use std::io::Write;
use std::process::{Command, Stdio};
const MODEL: &str = "wgt 0.5\nmodel Scale\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=[2]\nfn forward(x){return x*gain;}\n";
fn run(source: &str, args: &[&str]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn default_is_backend_independent_and_keeps_the_shared_plan() {
    let native = run(MODEL, &["compile", "-"]);
    assert!(
        native.status.success(),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );
    let mut native: serde_json::Value = serde_json::from_slice(&native.stdout).unwrap();
    assert_eq!(native["format"], "weightlang.wgt.v1");
    let torch = run(MODEL, &["compile", "-", "--target", "torch"]);
    assert!(torch.status.success());
    let torch: serde_json::Value = serde_json::from_slice(&torch.stdout).unwrap();
    native["format"] = serde_json::json!("weightlang.torch.v1");
    assert_eq!(native, torch);
}
#[test]
fn native_target_rejects_torch_dialect_without_initialization() {
    let source = include_str!("../../../python/examples/dynamic_recurrent/model.wgt");
    for args in [
        vec!["compile", "-"],
        vec!["compile", "-", "--target", "wgt"],
    ] {
        let output = run(source, &args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("torch"));
        assert!(output.stdout.is_empty());
    }
    assert!(
        run(source, &["compile", "-", "--target", "torch"])
            .status
            .success()
    );
}
#[test]
fn generic_target_preserves_shape_fingerprint_and_explicit_ports() {
    let source =
        "wgt 0.5\ndim B=2\nmodel Shapes\nfn forward(x:f32[B,1])->(f32[B,1],){return (x,);}\n";
    let output = run(
        source,
        &["compile", "-", "--target", "wgt", "--input-shape", "x=3,1"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        value["forward_interface"]["inputs"][0]["shape"],
        serde_json::json!([3, 1])
    );
    assert_eq!(value["template_fingerprint"].as_str().unwrap().len(), 64);
}
