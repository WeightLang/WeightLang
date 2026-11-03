#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn train_forwards_explicit_trainer_bindings() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/trainer\nwgt 0.3\n",
    )
    .unwrap();
    fs::write(temp.path().join("model.wgt"), "wgt 0.3\nmodel demo\n").unwrap();
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
    assert!(!output.status.success()); // fake compiler stops after capturing argv
    let args = fs::read_to_string(temp.path().join("args.txt")).unwrap();
    assert!(args.contains("--trainer\nsearch=local.py:train\n--trainer\nother=other.py:run\n"));
}
