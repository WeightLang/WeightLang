use std::{fs, process::Command};

#[test]
fn native_script_check_needs_no_python() {
    let cache = tempfile::tempdir().unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/native_training/build.wgt");
    let output = Command::new(env!("CARGO_BIN_EXE_weightc"))
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", cache.path())
        .arg("check")
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("no execution"));
}

#[test]
fn legacy_commands_do_not_launch_override_or_managed_python() {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("wgt.toml"),
        "[project]\nmodule = \"local/demo\"\nedition = \"0.5\"\n",
    )
    .unwrap();
    // An invalid managed environment must not be inspected or repaired for old CLI commands.
    fs::create_dir_all(root.path().join(".cache/weightlang/runtime")).unwrap();
    fs::write(
        root.path().join(".cache/weightlang/runtime/active.json"),
        "invalid",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(root.path())
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", root.path().join("cache"))
        .env("WEIGHTC", "/must-not-be-executed")
        .args(["build"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        error.contains("Python weightc CLI has been removed"),
        "{error}"
    );
    assert!(!root.path().join("dist").exists());
    assert!(!root.path().join("weights").exists());
}

#[test]
fn repository_source_has_no_python_distribution_or_tests() {
    fn check(path: &std::path::Path) {
        if !path.exists() {
            return;
        }
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_symlink() {
                continue;
            }
            if path.is_dir() {
                if ![
                    ".cache",
                    "__pycache__",
                    "node_modules",
                    "weights",
                    ".venv",
                    "target",
                ]
                .contains(&entry.file_name().to_str().unwrap())
                {
                    check(&path);
                }
            } else {
                assert!(
                    !path.extension().is_some_and(|e| e == "py" || e == "pyi"),
                    "{}",
                    path.display()
                );
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for directory in [
        "src",
        "tests",
        "scripts",
        "examples",
        "benchmarks",
        "crates",
    ] {
        check(&root.join(directory));
    }
    for file in [
        "pyproject.toml",
        "MANIFEST.in",
        ".github/workflows/publish-pypi.yml",
        "crates/wgt/build.rs",
    ] {
        assert!(!root.join(file).exists(), "{file}");
    }
}

#[cfg(unix)]
#[test]
fn installer_updates_both_native_binaries_without_python() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let cargo = root.path().join("cargo");
    fs::write(
        &cargo,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$INSTALL_LOG\"\n",
    )
    .unwrap();
    fs::set_permissions(&cargo, fs::Permissions::from_mode(0o755)).unwrap();
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let log = root.path().join("install.log");
    let output = Command::new("/bin/sh")
        .arg(source.join("install.sh"))
        .current_dir(root.path())
        .env("PATH", root.path())
        .env("INSTALL_LOG", &log)
        .env("WEIGHTLANG_SOURCE", &source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let args = fs::read_to_string(log).unwrap();
    assert!(args.contains("install\n--locked\n--force\n--path\n"));
    assert!(args.contains("--bin\nweightc\n--bin\nwgt\n"));
}

#[cfg(unix)]
#[test]
fn release_packaging_requires_matching_pair_and_preserves_existing_output() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    for name in ["wgt", "weightc"] {
        let path = root.path().join(name);
        fs::write(&path, format!("#!/bin/sh\necho '{name} 8.9.0'\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/package_wgt.sh");
    let output = root.path().join("release");
    let run = || {
        Command::new("/bin/sh")
            .arg(&script)
            .arg("--binary")
            .arg(root.path().join("wgt"))
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    let packaged = run();
    assert!(
        packaged.status.success(),
        "{}",
        String::from_utf8_lossy(&packaged.stderr)
    );
    let sums = fs::read_to_string(output.join("SHA256SUMS")).unwrap();
    assert_eq!(sums.lines().count(), 2);
    for line in sums.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(
            fields[0],
            weightlang_wgt::artifact::hash(&fs::read(output.join(fields[1])).unwrap())
        );
    }
    assert!(!run().status.success());
    assert_eq!(fs::read_to_string(output.join("SHA256SUMS")).unwrap(), sums);
    let invalid = root.path().join("invalid");
    fs::write(
        root.path().join("weightc"),
        "#!/bin/sh\necho 'weightc 1.0.0'\n",
    )
    .unwrap();
    assert!(!Command::new("/bin/sh")
        .arg(&script)
        .arg("--binary")
        .arg(root.path().join("wgt"))
        .arg("--output")
        .arg(&invalid)
        .output()
        .unwrap()
        .status
        .success());
    assert!(!invalid.exists());
}
