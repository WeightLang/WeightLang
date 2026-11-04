use anyhow::{ensure, Result};
use std::{collections::BTreeMap, fs, process::Command};
use weightlang_wgt::{
    manifest::{Lock, Manifest},
    registry::{self, Remote},
    source_package,
};
struct Memory {
    commit: String,
    files: BTreeMap<String, Vec<u8>>,
}
impl Remote for Memory {
    fn resolve(&self, _: &str, _: &str) -> Result<String> {
        Ok(self.commit.clone())
    }
    fn download(&self, url: &str, max: u64) -> Result<Vec<u8>> {
        let bytes = self
            .files
            .get(url.rsplit('/').next().unwrap())
            .ok_or_else(|| anyhow::anyhow!("unexpected download: {url}"))?
            .clone();
        ensure!(bytes.len() as u64 <= max, "size");
        Ok(bytes)
    }
}
fn git(project: &std::path::Path, args: &[&str]) {
    assert!(Command::new("git")
        .arg("-C")
        .arg(project)
        .args(args)
        .status()
        .unwrap()
        .success());
}
#[test]
fn source_release_fetch_optional_weights_lock_and_tamper() {
    let root = std::env::var_os("CLAUDE_JOB_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("weightlang"))
        .join("tmp");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let project = dir.path().join("project");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("wgt.toml"),
        "[project]\nmodule='github.com/test/source'\nedition='0.5'\n",
    )
    .unwrap();
    fs::write(
        project.join("model.wgt"),
        "wgt 0.5\nimport math from \"./math.wgt\"\nfn main() { return 1; }\n",
    )
    .unwrap();
    fs::write(
        project.join("math.wgt"),
        "wgt 0.5\noperators Math\nfn twice(x) { return x + x; }\n",
    )
    .unwrap();
    fs::write(project.join("credentials.json"), "secret").unwrap();
    fs::create_dir(project.join("weights")).unwrap();
    fs::write(project.join("weights/checkpoint.json"), "never distribute").unwrap();
    git(&project, &["init", "-q"]);
    git(&project, &["add", "."]);
    git(
        &project,
        &[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    let release = dir.path().join("release");
    let index = source_package::package(&project, None, "v0.5.0", &release).unwrap();
    assert_eq!(index["format_version"], 2);
    assert!(index["weights"].as_object().unwrap().is_empty());
    assert_eq!(index["sources"].as_object().unwrap().len(), 3);
    let mut remote = Memory {
        commit: index["commit"].as_str().unwrap().into(),
        files: fs::read_dir(&release)
            .unwrap()
            .map(|p| {
                let p = p.unwrap().path();
                (
                    p.file_name().unwrap().to_str().unwrap().into(),
                    fs::read(p).unwrap(),
                )
            })
            .collect(),
    };
    let manifest =
        Manifest::parse("module local/consumer\nwgt 0.5\nrequire github.com/test/source v0.5.0\n")
            .unwrap();
    let cache = dir.path().join("cache");
    let lock = registry::fetch(&manifest, &Lock::new(), &cache, false, &remote).unwrap();
    registry::verify(&manifest, &lock, &cache).unwrap();
    let consumer = "wgt 0.5\nimport math from \"github.com/test/source/Math\"\nmodel Net {\ninput x: f32[1]\noutput y: f32[1]\nfn forward(x) { return math.twice(x); }\n}\nfn main() { let net = Net(); return net([2]); }";
    let dependencies = BTreeMap::from([(
        "github.com/test/source".into(),
        registry::module_dir(&cache, "github.com/test/source", "v0.5.0")
            .to_string_lossy()
            .into_owned(),
    )]);
    wgtime_core::script::run_with_dependencies(consumer, dir.path(), 10000, dependencies)
        .unwrap();
    assert_eq!(
        lock,
        registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap()
    );
    assert!(registry::fetch_selected(
        &manifest,
        &lock,
        &cache,
        false,
        &remote,
        &["github.com/test/source=missing".into()]
    )
    .is_err());
    let installed = registry::module_dir(&cache, "github.com/test/source", "v0.5.0");
    fs::write(installed.join("model.wgt"), "tampered").unwrap();
    assert!(registry::verify(&manifest, &lock, &cache).is_err());
    registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap();
    let mut bad = index.clone();
    let value = bad["sources"]
        .as_object_mut()
        .unwrap()
        .remove("model.wgt")
        .unwrap();
    bad["sources"]["../escape.wgt"] = value;
    remote
        .files
        .insert("wgt.index.json".into(), serde_json::to_vec(&bad).unwrap());
    assert!(registry::fetch(&manifest, &Lock::new(), &cache, false, &remote).is_err());
    assert!(!cache.join("escape.wgt").exists());
}

#[test]
fn explicit_weights_are_separate_and_checkpoint_is_rejected() {
    use wgtime_core::weight_artifact::encode;
    let root = std::env::var_os("CLAUDE_JOB_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("weightlang"))
        .join("tmp");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let project = dir.path().join("project");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("wgt.mod"),
        "module github.com/test/weights\nwgt 0.5\n",
    )
    .unwrap();
    fs::write(
        project.join("model.wgt"),
        "wgt 0.5\nfn main() { return 1; }\n",
    )
    .unwrap();
    git(&project, &["init", "-q"]);
    git(&project, &["add", "."]);
    git(
        &project,
        &[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    let weights = dir.path().join("weights");
    fs::create_dir(&weights).unwrap();
    fs::write(
        weights.join("trained.safetensors"),
        encode(&[("layer.weight".into(), vec![1], vec![0.5])], 4096).unwrap(),
    )
    .unwrap();
    fs::write(
        weights.join("weights.json"),
        r#"{"trained":{"file":"trained.safetensors","model":"model.wgt"}}"#,
    )
    .unwrap();
    let release = dir.path().join("release");
    let index = source_package::package(&project, Some(&weights), "v0.5.0", &release).unwrap();
    assert_eq!(
        index["weights"]["trained"]["parameters"][0]["name"],
        "layer.weight"
    );
    let mut remote = Memory {
        commit: index["commit"].as_str().unwrap().into(),
        files: fs::read_dir(&release)
            .unwrap()
            .map(|p| {
                let p = p.unwrap().path();
                (
                    p.file_name().unwrap().to_str().unwrap().into(),
                    fs::read(p).unwrap(),
                )
            })
            .collect(),
    };
    let manifest =
        Manifest::parse("module local/consumer\nwgt 0.5\nrequire github.com/test/weights v0.5.0\n")
            .unwrap();
    let cache = dir.path().join("cache");
    let source_lock = registry::fetch(&manifest, &Lock::new(), &cache, false, &remote).unwrap();
    assert!(
        !registry::module_dir(&cache, "github.com/test/weights", "v0.5.0")
            .join("weights")
            .exists()
    );
    let selection = vec!["github.com/test/weights=trained".into()];
    let lock =
        registry::fetch_selected(&manifest, &source_lock, &cache, false, &remote, &selection)
            .unwrap();
    registry::verify(&manifest, &lock, &cache).unwrap();
    assert_eq!(
        lock,
        registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap()
    );
    remote
        .files
        .get_mut("weights-trained.safetensors")
        .unwrap()
        .push(1);
    assert!(registry::fetch_selected(&manifest, &lock, &cache, true, &remote, &selection).is_err());
    fs::write(
        weights.join("weights.json"),
        r#"{"trained":{"file":"checkpoint.json","model":"model.wgt"}}"#,
    )
    .unwrap();
    assert!(source_package::package(
        &project,
        Some(&weights),
        "v0.5.0",
        &dir.path().join("rejected-release")
    )
    .unwrap_err()
    .to_string()
    .contains("checkpoints are not weights"));
}
