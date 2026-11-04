use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn cli(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(root)
        .env("PATH", "/nonexistent")
        .args(args)
        .output()
        .unwrap()
}
fn ok(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn fail(output: Output, expected: &str) {
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(expected),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn fixture(root: &Path) {
    fs::write(root.join("model.wgt"), "wgt 0.5\nfn main() { let net = mlp([1,1], \"identity\", 42, \"adam\"); fit(net, [[0]], [[1]], 0.05, \"mse\"); return net; }").unwrap();
    ok(Command::new(env!("CARGO_BIN_EXE_weightc"))
        .current_dir(root)
        .env("PATH", "/nonexistent")
        .args(["build", "model.wgt", "-o", "built"])
        .output()
        .unwrap());
}
#[test]
fn native_directory_package_export_and_protection() {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let root = tempfile::tempdir_in(scratch).unwrap();
    let root = root.path();
    fixture(root);
    ok(cli(root, &["package", "--entry", "built", "-o", "bundle"]));
    ok(cli(root, &["export", "bundle", "-o", "inference.json"]));
    ok(cli(root, &["export", "inference.json", "-o", "copy.json"]));
    assert_eq!(
        fs::read(root.join("inference.json")).unwrap(),
        fs::read(root.join("copy.json")).unwrap()
    );
    let model: Value =
        serde_json::from_slice(&fs::read(root.join("inference.json")).unwrap()).unwrap();
    assert!(model.get("optimizer").is_none());
    assert_eq!(
        fs::read(root.join("built/checkpoint.json")).unwrap(),
        fs::read(root.join("bundle/checkpoint.json")).unwrap()
    );
    fail(
        cli(root, &["package", "--entry", "built", "-o", "bundle"]),
        "output already exists",
    );
    fail(
        cli(root, &["export", "built", "-o", "inference.json"]),
        "output already exists",
    );
    fail(
        cli(root, &["package", "--entry", "built", "-o", "bad.zip"]),
        "archive output is unsupported",
    );
    assert!(!root.join("bad.zip").exists());
    for args in [
        vec!["export", "model.wgt", "-o", "bad.json"],
        vec!["export", "built/checkpoint.json", "-o", "bad.json"],
    ] {
        assert!(!cli(root, &args).status.success());
        assert!(!root.join("bad.json").exists());
    }
    fail(
        cli(root, &["export", "built", "-o", "bad.json", "--force"]),
        "does not support",
    );
    let original = fs::read(root.join("built/model.json")).unwrap();
    fs::write(root.join("built/model.json"), b"{}").unwrap();
    assert!(
        !cli(root, &["package", "--entry", "built", "-o", "corrupt"])
            .status
            .success()
    );
    assert!(!root.join("corrupt").exists());
    fs::write(root.join("built/model.json"), original).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(root.join("built/manifest.json")).unwrap()).unwrap();
    manifest["model_sha256"] = json!("wrong");
    fs::write(root.join("built/manifest.json"), manifest.to_string()).unwrap();
    fail(
        cli(root, &["export", "built", "-o", "corrupt.json"]),
        "checksum mismatch",
    );
    assert!(!root.join("corrupt.json").exists());
}
#[cfg(unix)]
#[test]
fn rejects_symlinks_and_external_parameter_paths() {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let root = tempfile::tempdir_in(scratch).unwrap();
    let root = root.path();
    fixture(root);
    std::os::unix::fs::symlink(root.join("built/model.json"), root.join("link.json")).unwrap();
    fail(
        cli(root, &["export", "link.json", "-o", "out.json"]),
        "no symlinks",
    );
    let mut model: Value =
        serde_json::from_slice(&fs::read(root.join("built/model.json")).unwrap()).unwrap();
    model["parameters"][0]["kind"] = json!("file");
    model["parameters"][0]["path"] = json!("../../outside");
    model["parameters"][0]["offset"] = json!(0);
    model["parameters"][0]
        .as_object_mut()
        .unwrap()
        .remove("data");
    fs::write(root.join("external.json"), model.to_string()).unwrap();
    fail(
        cli(root, &["export", "external.json", "-o", "out.json"]),
        "external parameter files",
    );
    assert!(!root.join("out.json").exists());
}
