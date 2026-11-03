use std::{fs, path::Path, process::Command};
use weightlang_wgt::toolchain::{self, CURRENT, PIN};

#[test]
fn exact_versions_and_checksum_entries() {
    for value in ["0.1.0", "12.34.56"] {
        toolchain::validate(value).unwrap();
    }
    for value in [
        "",
        "latest",
        "v0.1.0",
        "01.1.0",
        "../x",
        "0.1",
        "0.1.0\n",
        "0.1.0 extra",
    ] {
        assert!(toolchain::validate(value).is_err());
    }
    let sum = "a".repeat(64);
    assert_eq!(
        toolchain::checksum(&format!("{sum}  wgt-linux-x86_64\n"), "wgt-linux-x86_64").unwrap(),
        sum
    );
    assert!(toolchain::checksum(&format!("{sum} a\n{sum} a\n"), "a").is_err());
    assert!(toolchain::checksum("bad a", "a").is_err());
    for os in ["linux", "macos", "windows"] {
        for arch in ["x86_64", "aarch64"] {
            assert!(toolchain::asset(os, arch).is_ok());
        }
    }
    assert!(toolchain::asset("linux", "mips").is_err());
}
fn command(root: &Path, cache: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(root)
        .env("WEIGHTLANG_CACHE", cache)
        .env_remove("_WGT_DISPATCHED")
        .env_remove("_WGT_VERSION_PROBE");
    command
}
#[test]
fn project_pins_discovery_override_and_init_protection() {
    let tmp = scratch();
    let root = tmp.path();
    let cache = root.join("cache");
    assert!(command(root, &cache)
        .args(["new", "demo"])
        .status()
        .unwrap()
        .success());
    let demo = root.join("demo");
    assert_eq!(
        fs::read_to_string(demo.join(PIN)).unwrap(),
        format!("{CURRENT}\n")
    );
    let nested = demo.join("nested");
    fs::create_dir(&nested).unwrap();
    assert_eq!(toolchain::discover(&nested).unwrap().unwrap().1, CURRENT);
    assert!(command(&nested, &cache)
        .args(["version", "8.9.0"])
        .status()
        .unwrap()
        .success());
    assert_eq!(toolchain::discover(&nested).unwrap().unwrap().1, "8.9.0");
    for option in [
        format!("-C={}", nested.display()),
        format!("--directory={}", nested.display()),
    ] {
        let output = command(root, &cache)
            .args([option.as_str(), "version"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).starts_with("8.9.0"));
    }
    let missing = command(root, &cache)
        .arg("-C")
        .arg(&nested)
        .arg("--version")
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("requires wgt 8.9.0"));
    let offline = command(&nested, &cache)
        .args(["sync", "--offline"])
        .output()
        .unwrap();
    assert!(!offline.status.success());
    assert!(String::from_utf8_lossy(&offline.stderr).contains("--offline"));
    assert!(command(&nested, &cache)
        .args(["version", CURRENT])
        .status()
        .unwrap()
        .success());
    let protected = root.join("protected");
    fs::create_dir(&protected).unwrap();
    fs::write(protected.join(PIN), CURRENT).unwrap();
    assert!(!command(root, &cache)
        .args(["init", "protected"])
        .output()
        .unwrap()
        .status
        .success());
    assert!(!protected.join("wgt.toml").exists());
    fs::write(nested.join(PIN), "latest\n").unwrap();
    assert!(toolchain::discover(&nested).is_err());
    assert!(command(&nested, &cache)
        .args(["version", CURRENT])
        .status()
        .unwrap()
        .success());
}
#[cfg(unix)]
#[test]
fn local_install_dispatch_forwarding_and_tamper_detection() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = scratch();
    let root = tmp.path();
    let cache = root.join("cache");
    let source = root.join("trusted executable");
    fs::write(&source, "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'wgt 8.9.0'; exit 0; fi\nprintf '%s\\n' \"$@\" > forwarded\nexit 17\n").unwrap();
    fs::set_permissions(&source, fs::Permissions::from_mode(0o755)).unwrap();
    let missing_compiler = command(root, &cache)
        .args(["toolchain", "8.9.0", "--offline", "--path"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(!missing_compiler.status.success());
    assert!(!cache.join("toolchains/8.9.0").exists());
    fs::write(root.join("weightc"), "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'weightc 8.9.0'; exit 0; fi\nprintf '%s\\n' \"$@\" > compiler-forwarded\nexit 19\n").unwrap();
    fs::set_permissions(root.join("weightc"), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(command(root, &cache)
        .args(["toolchain", "8.9.0", "--offline", "--path"])
        .arg(&source)
        .status()
        .unwrap()
        .success());
    assert!(!command(root, &cache)
        .args(["toolchain", "8.9.1", "--path"])
        .arg(&source)
        .output()
        .unwrap()
        .status
        .success());
    fs::write(root.join(PIN), "8.9.0\n").unwrap();
    let result = command(root, &cache)
        .args(["future-command", "--future-flag", "a b"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(17));
    assert_eq!(
        fs::read_to_string(root.join("forwarded")).unwrap(),
        "future-command\n--future-flag\na b\n"
    );
    assert!(command(root, &cache)
        .arg("--version")
        .output()
        .unwrap()
        .status
        .success());
    let compiler = Command::new(env!("CARGO_BIN_EXE_weightc"))
        .current_dir(root)
        .env("WEIGHTLANG_CACHE", &cache)
        .args(["check", "future.wgt"])
        .output()
        .unwrap();
    assert_eq!(compiler.status.code(), Some(19));
    assert_eq!(
        fs::read_to_string(root.join("compiler-forwarded")).unwrap(),
        "check\nfuture.wgt\n"
    );
    let compiler_path = cache
        .join("toolchains/8.9.0")
        .join(format!(
            "{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
        .join("weightc");
    let compiler_bytes = fs::read(&compiler_path).unwrap();
    fs::write(&compiler_path, "tampered compiler").unwrap();
    let rejected = command(root, &cache).arg("--version").output().unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("SHA256"));
    fs::write(&compiler_path, compiler_bytes).unwrap();
    let path = cache
        .join("toolchains/8.9.0")
        .join(format!(
            "{}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
        .join("wgt");
    fs::write(path, "tampered").unwrap();
    let tampered = command(root, &cache).arg("--version").output().unwrap();
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stderr).contains("SHA256"));
}

fn scratch() -> tempfile::TempDir {
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

#[test]
fn release_download_is_exact_and_checked() {
    struct Remote {
        corrupt: bool,
    }
    impl weightlang_wgt::registry::Remote for Remote {
        fn resolve(&self, _: &str, _: &str) -> anyhow::Result<String> {
            unreachable!()
        }
        fn download(&self, url: &str, _: u64) -> anyhow::Result<Vec<u8>> {
            assert!(url.starts_with(
                "https://github.com/WeightLang/WeightLang/releases/download/wgt-v8.9.0/"
            ));
            if url.ends_with("/SHA256SUMS") {
                return Ok(format!(
                    "{}  wgt-linux-x86_64\n",
                    weightlang_wgt::artifact::hash(b"binary")
                )
                .into_bytes());
            }
            assert!(url.ends_with("/wgt-linux-x86_64"));
            Ok(if self.corrupt {
                b"broken".to_vec()
            } else {
                b"binary".to_vec()
            })
        }
    }
    assert_eq!(
        toolchain::download(&Remote { corrupt: false }, "8.9.0", "linux", "x86_64").unwrap(),
        b"binary"
    );
    assert!(toolchain::download(&Remote { corrupt: true }, "8.9.0", "linux", "x86_64").is_err());
    assert!(toolchain::download(&Remote { corrupt: false }, "../bad", "linux", "x86_64").is_err());
}

#[cfg(unix)]
#[test]
fn update_download_selection_and_failure_preservation() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = scratch();
    let root = tmp.path();
    let cache = root.join("cache");
    for version in ["8.9.0", "8.8.0", CURRENT] {
        let source = root.join(format!("wgt-{version}"));
        fs::write(&source, format!("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'wgt {version}'; exit 0; fi\nprintf '%s\\n' \"$@\" > forwarded\nexit 17\n")).unwrap();
        fs::set_permissions(&source, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(
            root.join("weightc"),
            format!("#!/bin/sh\necho 'weightc {version}'\n"),
        )
        .unwrap();
        fs::set_permissions(root.join("weightc"), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(command(root, &cache)
            .args(["toolchain", version, "--path"])
            .arg(source)
            .output()
            .unwrap()
            .status
            .success());
    }
    let run = |args: &[&str]| {
        let result = command(root, &cache).args(args).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        result
    };
    run(&["download", "8.9.0", "--offline"]);
    assert!(!root.join(PIN).exists());
    assert!(!cache.join("toolchains/default").exists());
    run(&["self", "update", "--version", "8.9.0", "--offline"]);
    assert_eq!(
        fs::read_to_string(cache.join("toolchains/default")).unwrap(),
        "8.9.0\n"
    );
    assert_eq!(
        String::from_utf8_lossy(&run(&["--version"]).stdout).trim(),
        "wgt 8.9.0"
    );
    run(&["download", "8.8.0", "--offline"]);
    assert_eq!(
        String::from_utf8_lossy(&run(&["--version"]).stdout).trim(),
        "wgt 8.9.0"
    );
    run(&["update", "--version", "8.8.0", "--offline"]);
    assert_eq!(
        String::from_utf8_lossy(&run(&["--version"]).stdout).trim(),
        "wgt 8.8.0"
    );
    for args in [
        vec!["update", "--offline"],
        vec!["self", "update", "--version", "latest", "--offline"],
        vec!["self", "update", "--locked"],
        vec!["update", "--version", "7.0.0", "--offline"],
        vec!["update", "--version", "8.9.0", "--locked"],
        vec!["download", "../oops"],
    ] {
        assert!(!command(root, &cache)
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
        assert_eq!(
            fs::read_to_string(cache.join("toolchains/default")).unwrap(),
            "8.8.0\n"
        );
    }
    fs::write(root.join(PIN), "8.9.0\n").unwrap();
    let nested = root.join("nested");
    fs::create_dir(&nested).unwrap();
    assert!(command(&nested, &cache)
        .args(["update", "--version", CURRENT, "--offline"])
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(
        fs::read_to_string(root.join(PIN)).unwrap(),
        format!("{CURRENT}\n")
    );
    assert!(!nested.join(PIN).exists());
    assert_eq!(
        fs::read_to_string(cache.join("toolchains/default")).unwrap(),
        "8.8.0\n"
    );
    fs::write(root.join(PIN), "8.9.0\n").unwrap();
    let result = command(root, &cache)
        .args(["update", "github.com/example/models", "--version", "1.0.0"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(17));
    assert!(fs::read_to_string(root.join("forwarded"))
        .unwrap()
        .contains("github.com/example/models"));
    let result = command(root, &cache)
        .args(["--registry", "WeightLang/registry", "update", "--all"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(17));
    assert!(fs::read_to_string(root.join("forwarded"))
        .unwrap()
        .contains("--all"));
}

#[test]
fn latest_native_release_filters_and_orders_versions() {
    struct Releases;
    impl weightlang_wgt::registry::Remote for Releases {
        fn resolve(&self, _: &str, _: &str) -> anyhow::Result<String> {
            unreachable!()
        }
        fn download(&self, url: &str, _: u64) -> anyhow::Result<Vec<u8>> {
            assert!(url.ends_with("releases?per_page=100&page=1"));
            Ok(serde_json::to_vec(&serde_json::json!([
                {"tag_name":"v99.0.0", "draft":false, "prerelease":false},
                {"tag_name":"wgt-v1.9.0", "draft":false, "prerelease":false},
                {"tag_name":"wgt-v3.0.0", "draft":true, "prerelease":false},
                {"tag_name":"wgt-v2.0.0", "draft":false, "prerelease":true},
                {"tag_name":"wgt-v1.10.0", "draft":false, "prerelease":false,
                 "assets":[{"name":"wgt-linux-x86_64"},{"name":"weightc-linux-x86_64"},{"name":"SHA256SUMS"}]},
                {"tag_name":"wgt-v9.0.0", "draft":false, "prerelease":false,
                 "assets":[{"name":"wgt-macos-aarch64"},{"name":"SHA256SUMS"}]},
                {"tag_name":"wgt-v9.1.0", "draft":false, "prerelease":false,
                 "assets":[{"name":"wgt-linux-x86_64"}]}
            ]))?)
        }
    }
    assert_eq!(
        toolchain::latest_for(&Releases, "linux", "x86_64").unwrap(),
        "1.10.0"
    );
}

#[test]
fn self_subcommands_bypass_missing_project_toolchain() {
    let tmp = scratch();
    let root = tmp.path();
    let cache = root.join("cache");
    fs::write(root.join(PIN), "9.9.9\n").unwrap();
    for args in [vec!["self", "--help"], vec!["self", "update", "--help"]] {
        let result = command(root, &cache).args(args).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("update"));
    }
    for args in [vec!["self"], vec!["self", "unknown"]] {
        let result = command(root, &cache).args(args).output().unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("wgt self"));
    }
    assert_eq!(fs::read_to_string(root.join(PIN)).unwrap(), "9.9.9\n");
    assert!(!cache.exists());
}
