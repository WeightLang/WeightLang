//! Real Cargo consumers exercise proc-macro diagnostics and the renamed facade.
//! Run serially after the workspace is ready; generated packages share the
//! project's disk-backed compiler cache and never connect to wgtime.
use std::{fs, path::Path, process::Command};

fn check(root: &Path, source: &str) -> std::process::Output {
    fs::write(root.join("src/main.rs"), source).unwrap();
    Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/wgtime-target"))
        .env("CARGO_BUILD_JOBS", "2")
        .output()
        .unwrap()
}

#[test]
fn facade_alias_compiles_and_invalid_definitions_fail_during_compilation() {
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache = project.join(".cache/weightlang/macro-consumers");
    fs::create_dir_all(&cache).unwrap();
    let root = cache.join(format!("consumer-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    fs::create_dir(root.join("src")).unwrap();
    let facade = project.join("crates/wgt-model").canonicalize().unwrap();
    // JSON quoting is also valid TOML basic-string quoting for this path.
    let path = format!("{:?}", facade.to_str().unwrap());
    fs::write(root.join("Cargo.toml"), format!(
        "[package]\nname=\"wgt-macro-consumer\"\nversion=\"0.0.0\"\nedition=\"2021\"\n[workspace]\n[dependencies]\nrenamed_wgt={{package=\"wgt-model\",path={path}}}\n"
    )).unwrap();
    fs::write(root.join("model.wgt"), include_str!("fixtures/model.wgt")).unwrap();
    let valid = check(
        &root,
        r##"fn main() {
    let inline = renamed_wgt::model!(crate = renamed_wgt; r#"wgt 0.5
model Renamed
input x:f32[1]
output y:f32[1]
fn forward(x){return x;}
"#);
    let file = renamed_wgt::include_model!(crate = renamed_wgt; "model.wgt");
    assert_eq!(inline.name(), "Renamed");
    assert_eq!(file.name(), "MacroProbe");
}"##,
    );
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );

    let invalid = check(
        &root,
        r##"fn main() {
    let _ = renamed_wgt::model!(crate = renamed_wgt; r#"wgt 0.5
model Invalid
input x:f32[1]
output y:f32[1]
fn forward(x){return @;}
"#);
}"##,
    );
    assert!(!invalid.status.success());
    let error = String::from_utf8_lossy(&invalid.stderr);
    assert!(error.contains("invalid WGT model definition"), "{error}");
    assert!(error.contains("line "), "missing WGT location: {error}");

    // Changing only the included WGT file must invalidate Cargo's cached build.
    let file_consumer =
        "fn main(){let _=renamed_wgt::include_model!(crate = renamed_wgt; \"model.wgt\");}";
    assert!(check(&root, file_consumer).status.success());
    fs::write(root.join("model.wgt"), "wgt 0.5\nfn main(){return @;}\n").unwrap();
    let changed = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(project.join(".cache/weightlang/wgtime-target"))
        .env("CARGO_BUILD_JOBS", "2")
        .output()
        .unwrap();
    assert!(!changed.status.success());
    assert!(String::from_utf8_lossy(&changed.stderr).contains("invalid WGT model definition"));
    fs::remove_dir_all(root).unwrap();
}
