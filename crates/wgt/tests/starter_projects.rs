use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn wgt(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(root)
        .env("WEIGHTLANG_CACHE", root.join("cache"))
        .args(args)
        .output()
        .unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn every_template_starts_offline_without_initializing_weights() {
    let temp = tempfile::tempdir().unwrap();
    let listed = wgt(temp.path(), &["templates"]);
    success(&listed);
    for template in ["native", "mlp", "tiny-lm"] {
        assert!(String::from_utf8_lossy(&listed.stdout).contains(template));
        success(&wgt(
            temp.path(),
            &["--offline", "new", template, "--template", template],
        ));
        success(&wgt(temp.path(), &["-C", template, "check"]));
        let root = temp.path().join(template);
        assert!(!root.join("weights").exists());
        assert!(!root.join(".venv").exists());
        assert!(!root.join("uv.lock").exists());
        if template == "native" {
            assert!(root.join("model.wgt").is_file());
            assert!(!root.join("pyproject.toml").exists());
        } else {
            assert!(
                !root.join("model.wgt").exists(),
                "Python starters must not carry an ignored WGT model"
            );
            assert!(root.join("workflow.py").is_file());
            assert_eq!(
                fs::read_to_string(root.join(".python-version")).unwrap(),
                "3.12\n"
            );
            let text = fs::read_to_string(root.join("pyproject.toml")).unwrap();
            assert!(!text.contains("PROJECT_"));
            assert!(text.contains("torch-backend = \"auto\""));
            assert!(text.contains("device = \"auto\""));
            assert!(!text.contains("pytorch-cpu"));
        }
    }
}

#[test]
fn init_current_directory_keeps_user_files_and_refuses_output_collisions() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("notes.txt"), "keep me").unwrap();
    fs::write(temp.path().join("workflow.py"), "keep this runner").unwrap();
    assert!(
        !wgt(temp.path(), &["init", "--template", "tiny-lm"])
            .status
            .success()
    );
    assert!(!temp.path().join("pyproject.toml").exists());
    assert!(!temp.path().join("wgt.sum").exists());
    fs::remove_file(temp.path().join("workflow.py")).unwrap();
    fs::create_dir(temp.path().join(".gitignore")).unwrap();
    assert!(
        !wgt(temp.path(), &["init", "--template", "tiny-lm"])
            .status
            .success()
    );
    assert!(!temp.path().join("pyproject.toml").exists());
    fs::remove_dir(temp.path().join(".gitignore")).unwrap();
    success(&wgt(
        temp.path(),
        &["init", "--template", "tiny-lm", "--python", "3.11"],
    ));
    assert_eq!(
        fs::read_to_string(temp.path().join("notes.txt")).unwrap(),
        "keep me"
    );
    assert_eq!(
        fs::read_to_string(temp.path().join(".python-version")).unwrap(),
        "3.11\n"
    );
    let before = fs::read(temp.path().join("pyproject.toml")).unwrap();
    assert!(
        !wgt(temp.path(), &["init", "--template", "mlp"])
            .status
            .success()
    );
    assert_eq!(
        fs::read(temp.path().join("pyproject.toml")).unwrap(),
        before
    );
}

#[test]
fn invalid_template_options_do_not_create_partial_projects() {
    let temp = tempfile::tempdir().unwrap();
    for args in [
        vec!["init", "bad", "--template", "tiny-lm", "--python", "2.7"],
        vec!["init", "bad", "--template", "native", "--python", "3.12"],
        vec!["init", "bad", "--template", "unknown"],
    ] {
        assert!(!wgt(temp.path(), &args).status.success());
        assert!(!temp.path().join("bad").exists());
    }
}

#[test]
fn guided_init_accepts_beginner_defaults() {
    let temp = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .args(["init", "demo", "--interactive"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n\n").unwrap();
    success(&child.wait_with_output().unwrap());
    let config = fs::read_to_string(temp.path().join("demo/pyproject.toml")).unwrap();
    assert!(config.contains("kind = \"tiny-lm\""));
}

#[cfg(unix)]
#[test]
fn workflow_options_are_forwarded_and_failed_uv_status_is_preserved() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    success(&wgt(
        temp.path(),
        &["init", "demo", "--template", "tiny-lm"],
    ));
    // Locked execution uses an already selected source; backend discovery is a
    // separate operation and must not intercept the child's exit status.
    let config_path = temp.path().join("demo/pyproject.toml");
    let mut config = fs::read_to_string(&config_path).unwrap();
    config.push_str("\n[tool.uv.sources]\ntorch = { index = \"pytorch-cpu\" }\n\n[[tool.uv.index]]\nname = \"pytorch-cpu\"\nurl = \"https://download.pytorch.org/whl/cpu\"\nexplicit = true\n");
    fs::write(config_path, config).unwrap();
    let executable = temp.path().join("uv");
    fs::write(
        &executable,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$WGT_TEST_UV_LOG\"\nexit 7\n",
    )
    .unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    let log = temp.path().join("uv-args");
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .env("PATH", temp.path())
        .env("WGT_TEST_UV_LOG", &log)
        .args([
            "-C",
            "demo",
            "train",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--offline",
            "--locked",
        ])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(7),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let forwarded = fs::read_to_string(&log).unwrap();
    assert!(forwarded.contains("--offline\n"), "{forwarded}");
    assert!(forwarded.contains("--locked\n"), "{forwarded}");
    assert!(
        forwarded.contains("train\n--steps\n2\n--device\ncpu\n"),
        "{forwarded}"
    );
    assert!(!temp.path().join("demo/weights").exists());
}

#[cfg(all(unix, not(target_os = "macos")))]
#[test]
fn uv_selects_the_install_source_before_sync_without_overwriting_user_requirements() {
    use std::os::unix::fs::PermissionsExt;
    for backend in ["cpu", "cu128"] {
        let temp = tempfile::tempdir().unwrap();
        success(&wgt(temp.path(), &["init", "demo", "--template", "mlp"]));
        let root = temp.path().join("demo");
        let path = root.join("pyproject.toml");
        let mut config = fs::read_to_string(&path).unwrap();
        config.push_str("\n[tool.uv.pip]\noutput-file = \"requirements.txt\"\n");
        fs::write(&path, &config).unwrap();
        fs::write(root.join("requirements.txt"), "keep-user-requirements\n").unwrap();
        let python = temp.path().join("selected-python");
        fs::write(&python, "project interpreter fixture").unwrap();
        let executable = temp.path().join("uv");
        fs::write(&executable, r#"#!/bin/sh
printf '%s\n' "$@" >> "$WGT_TEST_UV_LOG"
case " $* " in
  *" --no-sync "*) printf '%s\n' "$WGT_TEST_PYTHON" ;;
  *" pip compile "*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output-file" ]; then shift; output="$1"; fi
      shift
    done
    printf 'torch==2.15.0\n    # from https://download.pytorch.org/whl/%s\n' "$WGT_TEST_BACKEND" > "$output"
    ;;
esac
"#).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        let log = temp.path().join("uv-args");
        let run = |args: &[&str]| {
            Command::new(env!("CARGO_BIN_EXE_wgt"))
                .current_dir(&root)
                .env("PATH", temp.path())
                .env("WGT_TEST_UV_LOG", &log)
                .env("WGT_TEST_PYTHON", &python)
                .env("WGT_TEST_BACKEND", backend)
                .env_remove("UV_TORCH_BACKEND")
                .args(args)
                .output()
                .unwrap()
        };
        success(&run(&["sync"]));
        let selected = fs::read_to_string(&path).unwrap();
        assert!(selected.contains(&format!("https://download.pytorch.org/whl/{backend}")));
        assert!(selected.contains("wgt-pytorch"));
        let before = config.parse::<toml_edit::DocumentMut>().unwrap();
        let after = selected.parse::<toml_edit::DocumentMut>().unwrap();
        assert_eq!(before["project"].to_string(), after["project"].to_string());
        assert_eq!(
            fs::read_to_string(root.join("requirements.txt")).unwrap(),
            "keep-user-requirements\n"
        );
        let calls = fs::read_to_string(&log).unwrap();
        assert!(calls.find("--no-sync\n").unwrap() < calls.find("pip\ncompile\n").unwrap());
        assert!(calls.contains(&format!("--python\n{}\n", python.display())));
        let output = calls
            .lines()
            .skip_while(|line| *line != "--output-file")
            .nth(1)
            .unwrap();
        assert!(
            !Path::new(output).exists(),
            "temporary probe output was not removed"
        );
        assert!(calls.ends_with("sync\n"));
        fs::write(&log, "").unwrap();
        success(&run(&[
            "--offline",
            "--locked",
            "run",
            "--",
            "python",
            "--version",
        ]));
        let calls = fs::read_to_string(&log).unwrap();
        assert!(
            !calls.contains("compile"),
            "offline/locked execution must reuse the saved source"
        );
        assert!(calls.contains("--offline\n--locked\n--\npython\n--version\n"));
        assert_eq!(fs::read_to_string(&path).unwrap(), selected);
    }
}
