#![cfg(unix)]
use std::{fs, os::unix::fs::PermissionsExt, process::Command};

#[test]
fn build_packages_sources_and_train_is_explicit() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("wgt.toml"), "[project]\nmodule='local/demo'\nedition='0.3'\n[model]\nsource='model.wgt'\n[package]\nname='demo'\n[train]\npython='train.py:train'\n").unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(&compiler, "#!/bin/sh\nprintf '%s\\n' \"$@\" > args.txt\n").unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    for command in [vec!["build"], vec!["package", "--reference", "model.wgt=weights/demo"], vec!["train", "--fresh"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_wgt")).current_dir(temp.path())
            .env("WEIGHTC", &compiler).args(&command).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let args = fs::read_to_string(temp.path().join("args.txt")).unwrap();
        assert!(args.starts_with(&format!("{}\n", command[0])));
        assert!(!args.contains("--no-store"));
    }
}
