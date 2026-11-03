#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn derived_export_reports_removed_python_cli() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/derived\nwgt 0.5\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("child.wgt"),
        "wgt 0.5\nmodel child extends \"weights/base\" sha256=\"hash\":\n  remove layer old\n",
    )
    .unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(
        &compiler,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > args.txt\nexit 1\n",
    )
    .unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .env("WEIGHTC", compiler)
        .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
        .args(["export", "child.wgt", "--offline"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Python weightc CLI has been removed"));
    assert!(!temp.path().join("args.txt").exists());
}
