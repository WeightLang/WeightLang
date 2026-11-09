use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

const SOURCE: &str = "wgt 0.5\nmodel M\nfn forward(x:f32[1])->f32[1]{return x;}\n";

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(root)
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", root.join("cache"));
    command
}

#[test]
fn project_format_and_alias_check_exclude_artifacts_and_ignore_old_toolchain_pin() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join(".wgt-version"), "99.99.99\n").unwrap();
    fs::write(root.join("model.wgt"), SOURCE).unwrap();
    for skipped in [
        "weights",
        "target",
        "cache",
        ".git",
        ".cache",
        "node_modules",
        "vendor",
        "dependencies",
    ] {
        fs::create_dir(root.join(skipped)).unwrap();
        fs::write(
            root.join(skipped).join("invalid.wgt"),
            "must remain untouched",
        )
        .unwrap();
    }
    let checked = command(root).args(["format", "--check"]).output().unwrap();
    assert_eq!(
        checked.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    assert_eq!(fs::read_to_string(root.join("model.wgt")).unwrap(), SOURCE);
    let formatted = command(root).arg("fmt").output().unwrap();
    assert!(
        formatted.status.success(),
        "{}",
        String::from_utf8_lossy(&formatted.stderr)
    );
    let output = fs::read_to_string(root.join("model.wgt")).unwrap();
    assert_eq!(output, wgt_language::format::format_source(SOURCE).unwrap());
    assert!(
        command(root)
            .args(["format", "--check"])
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(
        fs::read_to_string(root.join("weights/invalid.wgt")).unwrap(),
        "must remain untouched"
    );
}

#[test]
fn invalid_source_prevents_all_writes_and_explicit_paths_are_supported() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("a.wgt"), SOURCE).unwrap();
    fs::write(root.join("b.wgt"), "wgt 0.5\nfn broken(){return 1}\n").unwrap();
    let output = command(root)
        .args(["format", "a.wgt", "b.wgt"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(root.join("a.wgt")).unwrap(), SOURCE);
    let output = command(root).args(["format", "a.wgt"]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_ne!(fs::read_to_string(root.join("a.wgt")).unwrap(), SOURCE);
}

#[test]
fn stdin_writes_only_formatted_source_and_check_never_prints_it() {
    let directory = tempfile::tempdir().unwrap();
    for check in [false, true] {
        let mut command = command(directory.path());
        command
            .args(["format", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if check {
            command.arg("--check");
        }
        let mut child = command.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(SOURCE.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        if check {
            assert_eq!(output.status.code(), Some(1));
            assert!(output.stdout.is_empty());
        } else {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                wgt_language::format::format_source(SOURCE).unwrap()
            );
        }
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
}
