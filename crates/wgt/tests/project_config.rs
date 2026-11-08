use std::{fs, process::Command};
use weightlang_wgt::manifest::{self, Manifest, ProjectConfig};

const CONFIG: &str = r#"# preserved comment
[project]
module = "local/demo"
edition = "0.5"
[model]
source = "models/main.wgt"
output = "weights"
[runtime]
device = "cpu"
[runtime.placements]
"body.weight" = "cpu"
[dependencies]
"github.com/example/repo" = "v1.2.3"
"#;

#[test]
fn toml_roundtrip_and_validation() {
    assert_eq!(
        ProjectConfig::parse(CONFIG).unwrap().revisions.max_versions,
        5
    );
    for limit in [0, 1, 12] {
        let configured = format!("{CONFIG}\n[revisions]\nmax_versions = {limit}\n");
        assert_eq!(
            ProjectConfig::parse(&configured)
                .unwrap()
                .revisions
                .max_versions,
            limit
        );
    }
    for invalid in ["-1", "true", "1.5", "\"5\""] {
        let configured = format!("{CONFIG}\n[revisions]\nmax_versions = {invalid}\n");
        assert!(ProjectConfig::parse(&configured).is_err());
    }
    let manifest = Manifest::parse(CONFIG).unwrap();
    assert_eq!(manifest.dependencies["github.com/example/repo"], "v1.2.3");
    let text = manifest.edited("github.com/example/repo", Some("v1.2.4"));
    assert!(text.contains("# preserved comment"));
    let parsed = Manifest::parse(&text).unwrap();
    assert_eq!(parsed.dependencies["github.com/example/repo"], "v1.2.4");
    assert!(
        Manifest::parse(&parsed.edited("github.com/example/repo", None))
            .unwrap()
            .dependencies
            .is_empty()
    );
    assert!(ProjectConfig::parse(&CONFIG.replace("source =", "soruce =")).is_err());
    assert!(ProjectConfig::parse(&CONFIG.replace("models/main.wgt", "../main.wgt")).is_err());
    assert!(ProjectConfig::parse(&CONFIG.replace("v1.2.3", "latest")).is_err());
}

#[test]
fn sync_is_offline_idempotent_and_locked() {
    let temp = tempfile::tempdir().unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(temp.path())
            .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
            .args(args)
            .output()
            .unwrap()
    };
    assert!(run(&["new", "demo"]).status.success());
    let project = temp.path().join("demo");
    assert!(project.join("wgt.toml").is_file());
    let config = fs::read_to_string(project.join("wgt.toml")).unwrap();
    assert!(config.contains("[revisions]\nmax_versions = 5"));
    assert_eq!(
        ProjectConfig::parse(&config)
            .unwrap()
            .revisions
            .max_versions,
        5
    );
    assert!(!project.join("wgt.mod").exists());
    let checked = run(&["-C", "demo", "check"]);
    assert!(
        checked.status.success(),
        "fresh project must be a valid model: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    assert!(!project.join("weights").exists());
    for _ in 0..2 {
        let result = run(&["-C", "demo", "sync", "--no-env", "--offline", "--locked"]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fs::write(
        project.join("wgt.sum"),
        "github.com/example/old v1.0.0 commit abc\n",
    )
    .unwrap();
    assert!(
        !run(&["-C", "demo", "sync", "--no-env", "--offline", "--locked"])
            .status
            .success()
    );
    assert!(run(&["-C", "demo", "sync", "--no-env", "--offline"])
        .status
        .success());
    assert_eq!(fs::read_to_string(project.join("wgt.sum")).unwrap(), "");
    fs::remove_file(project.join("wgt.sum")).unwrap();
    assert!(!run(&["-C", "demo", "sync", "--no-env", "--locked"])
        .status
        .success());
    fs::write(project.join("wgt.mod"), "module local/other\nwgt 0.5\n").unwrap();
    assert!(manifest::manifest_path(&project).is_err());
    assert!(!run(&["-C", "demo", "sync", "--no-env"]).status.success());
}

#[test]
fn offline_sync_requires_locked_dependencies() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("wgt.toml"), CONFIG).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
        .args(["sync", "--no-env", "--offline"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!temp.path().join("wgt.sum").exists());
}
