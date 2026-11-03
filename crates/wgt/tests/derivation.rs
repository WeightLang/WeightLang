#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn discovers_derived_model_for_build() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/derived\nwgt 0.3\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("child.wgt"),
        "wgt 0.3\nmodel child extends \"weights/base\" sha256=\"hash\":\n  remove layer old\n",
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
        .args(["build", "child.wgt", "--offline"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let args = fs::read_to_string(temp.path().join("args.txt")).unwrap();
    assert!(args.starts_with("build\n"));
    assert!(args.contains("child.wgt\n"));
}
