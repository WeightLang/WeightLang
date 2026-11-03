use serde_json::json;
use std::{fs, process::Command};
use weightlang_wgt::registry::latest_release;

#[test]
fn latest_is_numeric_stable_published_and_package_scoped() {
    let releases = json!([
        {"tag_name":"v99.0.0","draft":false,"prerelease":false},
        {"tag_name":"packages/gates/v0.9.0","draft":false,"prerelease":false},
        {"tag_name":"packages/gates/v0.10.0","draft":false,"prerelease":false},
        {"tag_name":"packages/gates/v1.0.0","draft":true,"prerelease":false},
        {"tag_name":"packages/gates/v2.0.0","draft":false,"prerelease":true},
        {"tag_name":"packages/gates/v3.0.0-rc.1","draft":false,"prerelease":false},
        {"tag_name":"packages/gates/v04.0.0","draft":false,"prerelease":false},
        {"tag_name":"packages/other/v8.0.0","draft":false,"prerelease":false}
    ]);
    let releases = releases.as_array().unwrap();
    assert_eq!(latest_release("github.com/example/repo/packages/gates", releases).unwrap(), "v0.10.0");
    assert_eq!(latest_release("github.com/example/repo", releases).unwrap(), "v99.0.0");
    assert!(latest_release("github.com/example/repo/missing", releases).is_err());
    assert!(latest_release("github.com/example/repo", &[]).is_err());
}

#[test]
fn implicit_latest_rejects_offline_and_locked_without_editing_project() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("demo");
    fs::create_dir(&project).unwrap();
    let run = |args: &[&str]| Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(&project).env("WEIGHTLANG_CACHE", temp.path().join("cache"))
        .args(args).output().unwrap();
    let init = run(&["init", "--module", "local/demo"]);
    assert!(init.status.success(), "{}", String::from_utf8_lossy(&init.stderr));
    let manifest = project.join("wgt.toml");
    let before = fs::read(&manifest).unwrap();
    let sum = fs::read(project.join("wgt.sum")).unwrap();
    for args in [
        vec!["--offline", "add", "github.com/example/repo"],
        vec!["--offline", "add", "github.com/example/repo", "--version", "latest"],
        vec!["--locked", "add", "github.com/example/repo"],
    ] {
        let output = run(&args);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("requires network") || error.contains("incompatible with --locked"), "{error}");
        assert_eq!(fs::read(&manifest).unwrap(), before);
        assert_eq!(fs::read(project.join("wgt.sum")).unwrap(), sum);
    }
}
