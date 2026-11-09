use std::fs;
use weightlang_wgt::{
    manifest::{self, Manifest, ProjectConfig},
    python_env,
};

const TEMPLATE: &str = include_str!("../templates/tiny-lm.toml");
const NATIVE: &str = "[project]\nmodule = \"local/native\"\nedition = \"0.5\"\n";
const ORDINARY_PYPROJECT: &str =
    "[project]\nname = \"ordinary-python\"\nversion = \"0.1.0\"\ndependencies = [\"requests\"]\n";
// Keep an explicit historical layout so changing the current starter cannot
// silently remove compatibility coverage for embedded WeightLang settings.
const LEGACY_PYPROJECT: &str = r#"# Existing Python project metadata must remain unchanged.
[project]
name = "python-example"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = ["torch>=2.6,<3", "transformers>=4.50,<5"]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.wgt.project]
module = "local/python-example"
edition = "0.5"

[tool.wgt.python]
runner = "workflow.py"

[tool.wgt.model]
kind = "tiny-lm"
hidden_size = 64
layers = 2
heads = 4
context_length = 128
dropout = 0.0

[tool.wgt.runtime]
device = "auto"

[tool.wgt.train]
data = "data.txt"
steps = 100
batch_size = 4
learning_rate = 0.003
weight_decay = 0.01
seed = 42
output = "weights/model"
log_every = 10

[tool.wgt.infer]
checkpoint = "weights/model"
prompt = "WeightLang"
max_new_tokens = 32 # Tokens here are UTF-8 bytes, not words.
temperature = 0.8
top_k = 40
seed = 42
"#;

fn temp_project() -> tempfile::TempDir {
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("wgt-python-project-")
        .tempdir_in(root)
        .unwrap()
}

fn starter() -> String {
    TEMPLATE
        .replace("PROJECT_NAME", "python-example")
        .replace("PROJECT_MODULE", "local/python-example")
}

#[test]
fn pyproject_reads_weightlang_settings_independently_of_python_metadata() {
    let config = ProjectConfig::parse(LEGACY_PYPROJECT).unwrap();
    assert_eq!(config.project.module, "local/python-example");
    assert_eq!(config.project.edition, "0.5");
    assert_eq!(config.model.kind.as_deref(), Some("tiny-lm"));
    assert_eq!(config.model.hidden_size, Some(64));
    assert_eq!(config.runtime.device.as_deref(), Some("auto"));
    assert_eq!(config.python.unwrap().runner.to_str(), Some("workflow.py"));
    assert!(config.dependencies.is_empty());

    let manifest = Manifest::parse(LEGACY_PYPROJECT).unwrap();
    assert_eq!(manifest.module, "local/python-example");
    assert!(manifest.dependencies.is_empty());
}

#[test]
fn editing_weightlang_dependencies_preserves_python_dependencies_uv_and_comments() {
    let original = LEGACY_PYPROJECT;
    let before = original.parse::<toml_edit::DocumentMut>().unwrap();
    let assert_preserved = |edited: &str| {
        let after = edited.parse::<toml_edit::DocumentMut>().unwrap();
        assert_eq!(before["project"].to_string(), after["project"].to_string());
        assert_eq!(
            before["tool"]["uv"].to_string(),
            after["tool"]["uv"].to_string()
        );
        for line in original
            .lines()
            .filter(|line| line.trim_start().starts_with('#'))
        {
            assert!(edited.contains(line), "lost comment: {line}");
        }
        assert!(edited.contains("# Tokens here are UTF-8 bytes, not words."));
    };

    let added = Manifest::parse(original)
        .unwrap()
        .edited("github.com/example/model", Some("v1.2.3"));
    assert_preserved(&added);
    let added = Manifest::parse(&added).unwrap();
    assert_eq!(added.dependencies["github.com/example/model"], "v1.2.3");

    let updated = added.edited("github.com/example/model", Some("v1.2.4"));
    assert_preserved(&updated);
    let updated = Manifest::parse(&updated).unwrap();
    assert_eq!(updated.dependencies["github.com/example/model"], "v1.2.4");

    let removed = updated.edited("github.com/example/model", None);
    assert_preserved(&removed);
    assert!(Manifest::parse(&removed).unwrap().dependencies.is_empty());
}

#[test]
fn discovery_finds_pyproject_from_nested_directory_and_ignores_ordinary_pyproject() {
    let temp = temp_project();
    fs::write(temp.path().join("pyproject.toml"), LEGACY_PYPROJECT).unwrap();
    let child = temp.path().join("examples").join("python-only");
    fs::create_dir_all(child.join("src")).unwrap();
    fs::write(child.join("pyproject.toml"), ORDINARY_PYPROJECT).unwrap();

    assert!(ProjectConfig::read(&child).unwrap().is_none());
    assert_eq!(
        manifest::project(&child.join("src")).unwrap(),
        temp.path().canonicalize().unwrap()
    );
    assert_eq!(
        manifest::manifest_path(temp.path()).unwrap(),
        temp.path().join("pyproject.toml")
    );
    assert!(
        ProjectConfig::read(temp.path())
            .unwrap()
            .unwrap()
            .python
            .is_some()
    );
}

#[test]
fn ordinary_pyproject_can_coexist_with_a_native_manifest() {
    let temp = temp_project();
    fs::write(temp.path().join("wgt.toml"), NATIVE).unwrap();
    fs::write(temp.path().join("pyproject.toml"), ORDINARY_PYPROJECT).unwrap();
    assert_eq!(
        manifest::manifest_path(temp.path()).unwrap(),
        temp.path().join("wgt.toml")
    );
    let config = ProjectConfig::read(temp.path()).unwrap().unwrap();
    assert_eq!(config.project.module, "local/native");
    assert!(config.python.is_none());
}

#[test]
fn standalone_python_configuration_keeps_host_files_independent() {
    let temp = temp_project();
    let config_path = temp.path().join("wgt.toml");
    let cargo = "[package]\nname = \"rust-host\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";
    fs::write(&config_path, starter()).unwrap();
    fs::write(temp.path().join("pyproject.toml"), ORDINARY_PYPROJECT).unwrap();
    fs::write(temp.path().join("Cargo.toml"), cargo).unwrap();
    let nested = temp.path().join("examples").join("nested");
    fs::create_dir_all(&nested).unwrap();
    assert_eq!(
        manifest::project(&nested).unwrap(),
        temp.path().canonicalize().unwrap()
    );
    assert_eq!(manifest::manifest_path(temp.path()).unwrap(), config_path);
    let config = ProjectConfig::read(temp.path()).unwrap().unwrap();
    assert_eq!(config.project.module, "local/python-example");
    assert_eq!(config.python.unwrap().runner.to_str(), Some("workflow.py"));

    let edited = Manifest::parse(&starter())
        .unwrap()
        .edited("github.com/example/model", Some("v1.2.3"));
    fs::write(&config_path, &edited).unwrap();
    let document = edited.parse::<toml_edit::DocumentMut>().unwrap();
    assert!(document.get("tool").is_none());
    assert_eq!(
        ProjectConfig::read(temp.path())
            .unwrap()
            .unwrap()
            .dependencies["github.com/example/model"],
        "v1.2.3"
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("pyproject.toml")).unwrap(),
        ORDINARY_PYPROJECT
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("Cargo.toml")).unwrap(),
        cargo
    );
}

#[test]
fn pyproject_cannot_compete_with_another_weightlang_manifest() {
    for (name, content) in [
        ("wgt.toml", NATIVE),
        ("wgt.mod", "module local/native\nwgt 0.5\n"),
    ] {
        let temp = temp_project();
        fs::write(temp.path().join("pyproject.toml"), LEGACY_PYPROJECT).unwrap();
        fs::write(temp.path().join(name), content).unwrap();
        assert!(
            manifest::manifest_path(temp.path()).is_err(),
            "accepted {name}"
        );
        assert!(
            manifest::project(temp.path()).is_err(),
            "discovered ambiguous {name}"
        );
        assert!(
            ProjectConfig::read(temp.path()).is_err(),
            "read ambiguous {name}"
        );
    }
}

#[test]
fn equal_standalone_and_embedded_settings_still_conflict() {
    let temp = temp_project();
    fs::write(temp.path().join("wgt.toml"), starter()).unwrap();
    fs::write(temp.path().join("pyproject.toml"), LEGACY_PYPROJECT).unwrap();
    let error = ProjectConfig::read(temp.path()).unwrap_err().to_string();
    assert!(error.contains("multiple WeightLang manifests"), "{error}");
}

#[test]
fn python_configuration_requires_a_safe_runner_path_in_both_layouts() {
    for original in [starter(), LEGACY_PYPROJECT.to_owned()] {
        for runner in [
            "../workflow.py",
            "/tmp/workflow.py",
            "scripts/../workflow.py",
        ] {
            let unsafe_config =
                original.replace("runner = \"workflow.py\"", &format!("runner = {runner:?}"));
            assert!(
                ProjectConfig::parse(&unsafe_config).is_err(),
                "accepted {runner}"
            );
        }
    }
}

#[test]
fn check_rejects_invalid_architecture_training_and_inference_settings() {
    let valid = starter();
    python_env::check(&ProjectConfig::parse(&valid).unwrap()).unwrap();
    for (description, from, to) in [
        ("zero attention heads", "heads = 4", "heads = 0"),
        (
            "indivisible hidden size",
            "hidden_size = 64",
            "hidden_size = 63",
        ),
        (
            "output outside weights",
            "output = \"weights/model\"",
            "output = \"elsewhere.pt\"",
        ),
        (
            "missing inference checkpoint",
            "checkpoint = \"weights/model\"\n",
            "",
        ),
        (
            "invalid runtime device",
            "device = \"auto\"",
            "device = \"not-a-device\"",
        ),
    ] {
        let invalid = valid.replace(from, to);
        assert_ne!(valid, invalid, "test fixture did not change: {description}");
        let config = ProjectConfig::parse(&invalid).unwrap();
        assert!(
            python_env::check(&config).is_err(),
            "accepted {description}"
        );
    }
}

#[test]
fn runtime_accepts_auto_accelerators_and_omitted_settings() {
    for device in ["auto", "cpu", "cuda", "cuda:1", "xpu", "xpu:1", "mps"] {
        let text = starter().replace("device = \"auto\"", &format!("device = {device:?}"));
        python_env::check(&ProjectConfig::parse(&text).unwrap()).unwrap();
    }
    for device in ["cuda:-1", "xpu:-1", "xpu:", "cuda:١", "mps:0"] {
        let text = starter().replace("device = \"auto\"", &format!("device = {device:?}"));
        assert!(python_env::check(&ProjectConfig::parse(&text).unwrap()).is_err());
    }
    let mut document = starter().parse::<toml_edit::DocumentMut>().unwrap();
    document.as_table_mut().remove("runtime");
    let config = ProjectConfig::parse(&document.to_string()).unwrap();
    assert!(config.runtime.device.is_none());
    python_env::check(&config).unwrap();
}

#[cfg(unix)]
mod host_dispatch {
    use super::*;
    use std::{os::unix::fs::PermissionsExt, path::PathBuf, process::Command};

    struct Project {
        _temp: tempfile::TempDir,
        root: PathBuf,
        nested: PathBuf,
        config: PathBuf,
        runner: PathBuf,
        bin: PathBuf,
        config_trace: PathBuf,
        args_trace: PathBuf,
    }

    impl Project {
        fn new(legacy: bool) -> Self {
            let temp = temp_project();
            let root = temp.path().canonicalize().unwrap();
            let config = root.join(if legacy { "pyproject.toml" } else { "wgt.toml" });
            let text = if legacy {
                LEGACY_PYPROJECT.to_owned()
            } else {
                fs::write(root.join("pyproject.toml"), ORDINARY_PYPROJECT).unwrap();
                starter()
            };
            fs::write(
                &config,
                text.replace("steps = 100", "steps = 0").replace(
                    "runner = \"workflow.py\"",
                    "runner = \"scripts/workflow.py\"",
                ),
            )
            .unwrap();
            let runner = root.join("scripts/workflow.py");
            fs::create_dir_all(runner.parent().unwrap()).unwrap();
            fs::write(
                &runner,
                "# The uv fixture records dispatch without executing Python.\n",
            )
            .unwrap();
            let nested = root.join("examples/nested");
            fs::create_dir_all(&nested).unwrap();
            let bin = root.join("bin");
            fs::create_dir_all(&bin).unwrap();
            let uv = bin.join("uv");
            fs::write(
                &uv,
                "#!/bin/sh\nprintf '%s\\n' \"$WGT_CONFIG_FILE\" > \"$WGT_TEST_CONFIG\"\nprintf '%s\\n' \"$@\" > \"$WGT_TEST_ARGS\"\nexit 27\n",
            )
            .unwrap();
            fs::set_permissions(&uv, fs::Permissions::from_mode(0o755)).unwrap();
            Self {
                config_trace: root.join("config-trace"),
                args_trace: root.join("args-trace"),
                _temp: temp,
                root,
                nested,
                config,
                runner,
                bin,
            }
        }

        fn command(&self) -> Command {
            let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
            command
                .current_dir(&self.nested)
                .env_clear()
                .env("PATH", &self.bin)
                .env("WEIGHTLANG_CACHE", self.root.join("cache"))
                // An inherited value must never override CLI project discovery.
                .env("WGT_CONFIG_FILE", self.root.join("unselected.toml"))
                .env("WGT_TEST_CONFIG", &self.config_trace)
                .env("WGT_TEST_ARGS", &self.args_trace)
                .args(["--offline", "--locked"]);
            command
        }

        fn assert_dispatch(&self, args: &[&str], expected: &[String]) {
            let output = self.command().args(args).output().unwrap();
            assert_eq!(
                output.status.code(),
                Some(27),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                fs::read_to_string(&self.config_trace).unwrap().trim_end(),
                self.config.to_str().unwrap()
            );
            let actual = fs::read_to_string(&self.args_trace).unwrap();
            let actual: Vec<_> = actual.lines().collect();
            let mut all = vec![
                "--directory".to_owned(),
                self.root.to_str().unwrap().to_owned(),
                "run".to_owned(),
                "--offline".to_owned(),
                "--locked".to_owned(),
                "--".to_owned(),
            ];
            all.extend_from_slice(expected);
            assert_eq!(actual, all);
        }
    }

    #[test]
    fn cli_passes_absolute_selected_config_and_valid_step_override_to_both_layouts() {
        for legacy in [false, true] {
            let project = Project::new(legacy);
            project.assert_dispatch(
                &["train", "--steps", "2", "--device", "cpu"],
                &[
                    "python".into(),
                    project.runner.to_str().unwrap().into(),
                    "train".into(),
                    "--steps".into(),
                    "2".into(),
                    "--device".into(),
                    "cpu".into(),
                ],
            );
        }
    }

    #[test]
    fn inference_leaves_checkpoint_format_and_effective_options_to_the_host() {
        let project = Project::new(false);
        assert!(!project.root.join("weights/model").exists());
        project.assert_dispatch(
            &["infer", "--prompt", "hello from a custom host"],
            &[
                "python".into(),
                project.runner.to_str().unwrap().into(),
                "infer".into(),
                "--prompt".into(),
                "hello from a custom host".into(),
            ],
        );
    }

    #[test]
    fn arbitrary_host_command_receives_selected_configuration() {
        let project = Project::new(false);
        project.assert_dispatch(
            &["run", "--", "python", "--version"],
            &["python".into(), "--version".into()],
        );
    }

    #[test]
    fn check_still_rejects_invalid_file_values_without_starting_uv() {
        for legacy in [false, true] {
            let project = Project::new(legacy);
            let output = project.command().arg("check").output().unwrap();
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("train.steps"), "{error}");
            assert!(!project.config_trace.exists());
            assert!(!project.args_trace.exists());
        }
    }
}
