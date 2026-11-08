#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn native_lifecycle_commands_do_not_launch_python_or_external_compiler() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/trainer\nwgt 0.5\n",
    )
    .unwrap();
    fs::write(temp.path().join("model.wgt"), "wgt 0.5\nmodel demo\n").unwrap();
    for name in ["compiler", "uv", "wgt-train", "wgt-infer"] {
        let executable = temp.path().join(name);
        fs::write(
            &executable,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > invoked.txt\nexit 1\n",
        )
        .unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let runtime = temp.path().join(".cache/weightlang/runtime/active.json");
    fs::create_dir_all(runtime.parent().unwrap()).unwrap();
    fs::write(&runtime, "invalid").unwrap();
    for command in ["train", "infer"] {
        let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(temp.path())
            .env_clear()
            .env("PATH", temp.path())
            .env("WEIGHTC", temp.path().join("compiler"))
            .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
            .args(["--offline", command])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("requires a Python starter project"),
            "{error}"
        );
    }
    assert!(!temp.path().join("invoked.txt").exists());
    assert_eq!(fs::read_to_string(runtime).unwrap(), "invalid");
}
