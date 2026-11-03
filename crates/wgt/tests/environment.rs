use std::{fs, path::Path, process::Command};
use weightlang_wgt::{
    bootstrap,
    environment::{self, EnvironmentConfig},
    manifest::ProjectConfig,
};

#[test]
fn backend_and_platform_selection() {
    for os in ["linux", "windows"] {
        assert_eq!(
            environment::select_backend(os, "x86_64", "auto", false).unwrap(),
            "cpu"
        );
        assert_eq!(
            environment::select_backend(os, "x86_64", "auto", true).unwrap(),
            "cuda"
        );
        assert!(environment::select_backend(os, "x86_64", "cuda", false).is_err());
        assert_eq!(
            environment::select_backend(os, "x86_64", "cu128", true).unwrap(),
            "cuda"
        );
    }
    assert_eq!(
        environment::select_backend("macos", "aarch64", "auto", false).unwrap(),
        "mps"
    );
    assert_eq!(
        environment::select_backend("macos", "x86_64", "cpu", false).unwrap(),
        "cpu"
    );
    assert!(environment::select_backend("macos", "aarch64", "cuda", true).is_err());
    assert!(environment::select_backend("windows", "x86_64", "mps", false).is_err());
    for os in ["linux", "macos", "windows"] {
        for arch in ["x86_64", "aarch64"] {
            assert!(bootstrap::uv_asset(os, arch).is_ok());
        }
    }
    assert!(bootstrap::uv_asset("freebsd", "x86_64").is_err());
}

#[test]
fn invalid_environment_fails_before_downloads() {
    for python in ["system", "3", "3.9", "3.12\n"] {
        let config = EnvironmentConfig {
            python: python.into(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
    for backend in ["latest", "cu", "cpu:0", "rocm"] {
        let config = EnvironmentConfig {
            backend: backend.into(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
    let config = EnvironmentConfig {
        dependencies: vec!["torch\n--index-url http://invalid".into()],
        ..Default::default()
    };
    assert!(config.validate().is_err());
    assert!(ProjectConfig::parse(
        "[project]\nmodule='local/demo'\nedition='0.3'\n[environment]\npythno='3.12'\n"
    )
    .is_err());
}

#[test]
fn checksum_validation_and_bundled_compiler() {
    let temp = tempfile::tempdir().unwrap();
    let lock = serde_json::json!({"format":"weightlang.environment-lock","format_version":1,"targets":{
        "linux-x86_64-cpu": {"config_sha256":"x", "compiler_sha256":"y", "python":"3.12.1", "backend":"cpu",
        "torch_backend":"cpu", "uv_version":"uv 0.13.6", "requirements_sha256":"bad", "requirements":"torch==2.6", "verification":{}}
    }});
    fs::write(
        temp.path().join("wgt.lock"),
        serde_json::to_vec(&lock).unwrap(),
    )
    .unwrap();
    assert!(environment::read_lock(temp.path()).is_err());
    assert!(environment::COMPILER_FILES
        .iter()
        .any(|(name, _)| *name == "weightlang/__main__.py"));
    assert!(environment::COMPILER_FILES
        .iter()
        .any(|(name, _)| *name == "weightlang/std/nn.wgt"));
    assert!(environment::COMPILER_FILES
        .iter()
        .all(|(name, _)| !name.contains("__pycache__")));
}

#[test]
fn bootstrap_extracts_only_one_regular_executable() {
    fn archive(names: &[&str]) -> Vec<u8> {
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut tar = tar::Builder::new(encoder);
        for name in names {
            let bytes = b"executable";
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(&mut header, name, &bytes[..]).unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap()
    }
    assert_eq!(
        bootstrap::extract_uv(&archive(&["bin/uv", "bin/uvx"]), false, "uv").unwrap(),
        b"executable"
    );
    assert!(bootstrap::extract_uv(&archive(&["bin/uv", "other/uv"]), false, "uv").is_err());
    assert!(bootstrap::extract_uv(&archive(&["bin/uvx"]), false, "uv").is_err());
    let cursor = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    use std::io::Write;
    zip.start_file("uv.exe", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(b"windows-executable").unwrap();
    let bytes = zip.finish().unwrap().into_inner();
    assert_eq!(
        bootstrap::extract_uv(&bytes, true, "uv.exe").unwrap(),
        b"windows-executable"
    );
}

#[cfg(unix)]
fn fake_uv(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, r##"#!/bin/sh
printf '%s\n' "$*" >> "$WGT_TEST_LOG"
case "$*" in
  '--version') echo 'uv 0.13.6';;
  *'venv '*)
    for arg in "$@"; do dest="$arg"; done
    mkdir -p "$dest/bin"
    cat > "$dest/bin/python" <<'PY'
#!/bin/sh
if [ "$WGT_TEST_PROBE_FAIL" = 1 ]; then echo 'injected runtime failure' >&2; exit 1; fi
printf '%s\n' '{"python":"3.12.10","torch":"2.6.0+cpu","cuda":null,"device":"cpu","packages":{"torch":"2.6.0+cpu"},"tensor_test":"forward+backward passed","compiler":"0.2.0"}'
PY
    chmod +x "$dest/bin/python";;
  *'pip compile'*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --output-file ]; then shift; printf 'torch==2.6.0+cpu --hash=sha256:%064d\n' 0 > "$1"; fi
      shift
    done;;
  *'pip sync'*) if [ "$WGT_TEST_FAIL" = 1 ]; then echo 'injected install failure' >&2; exit 7; fi;;
  *'pip check'*) ;;
  *) echo 'unexpected fake uv call' >&2; exit 9;;
esac
"##).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[test]
fn sync_lock_reuse_repair_and_failed_install_preserve_previous_environment() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    fs::create_dir(&project).unwrap();
    let config = "[project]\nmodule='local/demo'\nedition='0.3'\n[environment]\npython='3.12'\nbackend='cpu'\n";
    fs::write(project.join("wgt.toml"), config).unwrap();
    fs::write(project.join("wgt.sum"), "").unwrap();
    let uv = temp.path().join("uv");
    fake_uv(&uv);
    let log = temp.path().join("calls.log");
    let run = |args: &[&str], fail: bool| {
        Command::new(env!("CARGO_BIN_EXE_wgt"))
            .arg("-C")
            .arg(&project)
            .args(args)
            .env("WGT_UV", &uv)
            .env("WGT_TEST_LOG", &log)
            .env("WGT_TEST_FAIL", if fail { "1" } else { "0" })
            .env("WEIGHTLANG_CACHE", temp.path().join("models"))
            .output()
            .unwrap()
    };
    let first = run(&["sync", "--offline"], false);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let state = project.join(".cache/weightlang/runtime/active.json");
    let before_state = fs::read(&state).unwrap();
    let before_lock = fs::read(project.join("wgt.lock")).unwrap();
    let before_calls = fs::read(&log).unwrap();
    assert!(run(&["sync", "--offline", "--locked"], false)
        .status
        .success());
    assert_eq!(fs::read(&log).unwrap(), before_calls); // verified runtime reuse needs no uv process
    assert_eq!(fs::read(&state).unwrap(), before_state);
    assert!(run(&["doctor"], false).status.success());
    assert!(!run(&["sync", "--upgrade-env", "--offline"], true)
        .status
        .success());
    assert_eq!(fs::read(&state).unwrap(), before_state);
    assert_eq!(fs::read(project.join("wgt.lock")).unwrap(), before_lock);
    let failed_probe = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .arg("-C")
        .arg(&project)
        .args(["sync", "--upgrade-env", "--offline"])
        .env("WGT_UV", &uv)
        .env("WGT_TEST_LOG", &log)
        .env("WGT_TEST_PROBE_FAIL", "1")
        .env("WEIGHTLANG_CACHE", temp.path().join("models"))
        .output()
        .unwrap();
    assert!(!failed_probe.status.success());
    assert_eq!(fs::read(&state).unwrap(), before_state);
    assert_eq!(fs::read(project.join("wgt.lock")).unwrap(), before_lock);
    let runtime = project.join(".cache/weightlang/runtime");
    assert_eq!(
        fs::read_dir(&runtime)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("env-"))
            .count(),
        1
    );
    // A missing Python is repaired using locked versions, without touching the lock.
    let active: serde_json::Value = serde_json::from_slice(&before_state).unwrap();
    fs::remove_file(
        runtime
            .join(active["directory"].as_str().unwrap())
            .join("bin/python"),
    )
    .unwrap();
    assert!(run(&["sync", "--offline", "--locked"], false)
        .status
        .success());
    assert_eq!(fs::read(project.join("wgt.lock")).unwrap(), before_lock);
    let calls = fs::read_to_string(&log).unwrap();
    assert!(calls.contains("--require-hashes"));
    assert!(calls.contains("--managed-python --python 3.12.10"));
    fs::write(project.join("wgt.toml"), config.replace("3.12", "3.13")).unwrap();
    let calls_before = fs::read(&log).unwrap();
    assert!(!run(&["sync", "--offline", "--locked"], false)
        .status
        .success());
    assert_eq!(fs::read(&log).unwrap(), calls_before);
}

#[test]
fn uv_versions_are_checked_before_use() {
    assert!(bootstrap::compatible_version("uv 0.13.6 (test)"));
    assert!(bootstrap::compatible_version("uv 1.0.0"));
    assert!(!bootstrap::compatible_version("uv 0.5.0"));
    assert!(!bootstrap::compatible_version("unexpected output"));
}
