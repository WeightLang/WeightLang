use std::fs;
use weightlang_wgt::{
    manifest::{self, Manifest, ProjectConfig},
    python_env,
};

const TEMPLATE: &str = include_str!("../templates/tiny-lm.toml");
const NATIVE: &str = "[project]\nmodule = \"local/native\"\nedition = \"0.5\"\n";
const ORDINARY_PYPROJECT: &str =
    "[project]\nname = \"ordinary-python\"\nversion = \"0.1.0\"\ndependencies = [\"requests\"]\n";

fn starter() -> String {
    TEMPLATE
        .replace("PROJECT_NAME", "python-example")
        .replace("PROJECT_MODULE", "local/python-example")
}

#[test]
fn pyproject_reads_weightlang_settings_independently_of_python_metadata() {
    let config = ProjectConfig::parse(&starter()).unwrap();
    assert_eq!(config.project.module, "local/python-example");
    assert_eq!(config.project.edition, "0.5");
    assert_eq!(config.model.kind.as_deref(), Some("tiny-lm"));
    assert_eq!(config.model.hidden_size, Some(64));
    assert_eq!(config.python.unwrap().runner.to_str(), Some("workflow.py"));
    assert!(config.dependencies.is_empty());

    let manifest = Manifest::parse(&starter()).unwrap();
    assert_eq!(manifest.module, "local/python-example");
    assert!(manifest.dependencies.is_empty());
}

#[test]
fn editing_weightlang_dependencies_preserves_python_dependencies_uv_and_comments() {
    let original = starter();
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

    let added = Manifest::parse(&original)
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
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("pyproject.toml"), starter()).unwrap();
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
    let temp = tempfile::tempdir().unwrap();
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
fn pyproject_cannot_compete_with_another_weightlang_manifest() {
    for (name, content) in [
        ("wgt.toml", NATIVE),
        ("wgt.mod", "module local/native\nwgt 0.5\n"),
    ] {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("pyproject.toml"), starter()).unwrap();
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
fn python_configuration_requires_pyproject_filename_and_safe_runner_path() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("wgt.toml"), starter()).unwrap();
    let error = ProjectConfig::read(temp.path()).unwrap_err().to_string();
    assert!(error.contains("pyproject.toml"), "{error}");

    for runner in [
        "../workflow.py",
        "/tmp/workflow.py",
        "scripts/../workflow.py",
    ] {
        let unsafe_config =
            starter().replace("runner = \"workflow.py\"", &format!("runner = {runner:?}"));
        assert!(
            ProjectConfig::parse(&unsafe_config).is_err(),
            "accepted {runner}"
        );
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
            "device = \"cpu\"",
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
