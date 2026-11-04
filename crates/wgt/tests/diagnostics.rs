#![cfg(unix)]
use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn diagnostic_commands_report_removed_python_cli() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("wgt.mod"), "module local/debug\nwgt 0.5\n").unwrap();
    fs::write(temp.path().join("model.wgt"), "wgt 0.5\nmodel demo\n").unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(&compiler, "#!/bin/sh\nprintf '%s\\n' \"$@\" > args.txt\nprintf '%s\\n' '{\"code\":\"E0301\"}' >&2\nexit 7\n").unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    for args in [
        vec!["check", "model.wgt", "--error-format", "json"],
        vec![
            "trace",
            "model.wgt",
            "--sample",
            "sample.json",
            "--watch",
            "memory",
            "--time",
            "0,7",
            "--weights",
            "weights/saved",
            "--json",
            "--error-format",
            "json",
        ],
        vec![
            "debug",
            "model.wgt",
            "--steps",
            "3",
            "--max-parameters",
            "10000",
            "--weights",
            "weights/saved",
            "--json",
            "--error-format",
            "json",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(temp.path())
            .env("WEIGHTC", &compiler)
            .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
            .args(&args)
            .arg("--offline")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("Python weightc CLI has been removed")
        );
        assert!(!temp.path().join("args.txt").exists());
    }
}
