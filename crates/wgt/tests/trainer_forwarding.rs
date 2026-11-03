#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn train_does_not_forward_to_external_compiler() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/trainer\nwgt 0.5\n",
    )
    .unwrap();
    fs::write(temp.path().join("model.wgt"), "wgt 0.5\nmodel demo\n").unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(
        &compiler,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > args.txt\nexit 1\n",
    )
    .unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .env("WEIGHTC", &compiler)
        .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
        .args([
            "train",
            "model.wgt",
            "--trainer",
            "search=local.py:train",
            "--trainer",
            "other=other.py:run",
            "--offline",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Python weightc CLI has been removed"));
    assert!(!temp.path().join("args.txt").exists());
}
