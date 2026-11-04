#![cfg(unix)]
use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn bundle_commands_do_not_invoke_old_compiler() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("wgt.toml"), "[project]\nmodule='local/demo'\nedition='0.5'\n[model]\nsource='model.wgt'\n[package]\nname='demo'\n[train]\npython='train.py:train'\n").unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(&compiler, "#!/bin/sh\nprintf '%s\\n' \"$@\" > args.txt\n").unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    for command in [vec!["build"], vec!["package", "--reference", "model.wgt=weights/demo"], vec!["train", "--fresh"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_wgt")).current_dir(temp.path())
            .env("WEIGHTC", &compiler).args(&command).output().unwrap();
        assert!(!output.status.success());
        let expected = if command[0] == "package" { "native package does not support --force, --weights or --reference" } else { "Python weightc CLI has been removed" };
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
        assert!(!temp.path().join("args.txt").exists());
    }
}
