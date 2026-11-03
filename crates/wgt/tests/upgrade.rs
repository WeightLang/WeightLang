use anyhow::{ensure, Result};
use serde_json::json;
use std::{fs, process::Command};
use weightlang_wgt::{
    artifact::hash,
    manifest::{Lock, Manifest},
    package_index::{Record, Selection},
    registry::Remote,
    upgrade::{self, Candidate},
};

const SOURCE: &[u8] = b"wgt 0.5\nmodel demo\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(in=1, out=1)\ninit seed=1\n";
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
struct Packages {
    broken: bool,
}
impl Packages {
    fn weights(&self) -> Vec<u8> {
        let mut header=serde_json::to_vec(&json!({"fc.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"fc.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[4,8]}})).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0; 8]);
        bytes
    }
    fn file(&self, name: &str) -> Vec<u8> {
        match name {
            "source.wgt" => SOURCE.to_vec(),
            "weights.safetensors" => self.weights(),
            "manifest.json" => serde_json::to_vec(&json!({"format":"weightlang.artifact","format_version":2,"model":"demo",
                "source":{"file":"source.wgt","sha256":hash(SOURCE)},
                "weights":{"file":"weights.safetensors","sha256":hash(&self.weights()),"tensors":{
                    "fc.weight":{"shape":[1,1],"dtype":"float32","origin":"random-init"},"fc.bias":{"shape":[1],"dtype":"float32","origin":"random-init"}}},
                "interface":{"input":{"name":"x","dtype":"f32","shape":[1]},"output":{"name":"y","dtype":"f32","shape":[1]}},
                "layers":[{"name":"fc","op":"linear","args":{"in":1,"out":1}}]})).unwrap(),
            _ => panic!("unexpected file"),
        }
    }
    fn index(&self, package: &str) -> Vec<u8> {
        let mut files = serde_json::Map::new();
        for name in ["source.wgt", "manifest.json", "weights.safetensors"] {
            let bytes = self.file(name);
            files.insert(
                name.into(),
                json!({"asset":format!("demo.{name}"),"sha256":hash(&bytes),"size":bytes.len()}),
            );
        }
        serde_json::to_vec(&json!({"format":"weightlang.package-index","format_version":1,"module":format!("github.com/example/{package}"),"version":"v0.2.0","tag":"v0.2.0","commit":COMMIT,"artifacts":{"demo":{"source_path":"main.wgt","files":files}}})).unwrap()
    }
}
impl Remote for Packages {
    fn resolve(&self, _: &str, _: &str) -> Result<String> {
        Ok(COMMIT.into())
    }
    fn download(&self, url: &str, _: u64) -> Result<Vec<u8>> {
        if url.contains("raw.githubusercontent.com") {
            return Ok(SOURCE.to_vec());
        }
        let package = if url.contains("/example/a/") {
            "a"
        } else {
            "b"
        };
        ensure!(
            !(self.broken && package == "b"),
            "simulated second-package network failure"
        );
        if url.ends_with("wgt.index.json") {
            return Ok(self.index(package));
        }
        Ok(self.file(
            url.rsplit('/')
                .next()
                .unwrap()
                .strip_prefix("demo.")
                .unwrap(),
        ))
    }
}
fn manifest() -> Manifest {
    Manifest::parse("# retain comment\nmodule local/demo\nwgt 0.5\nrequire github.com/example/a v0.1.0\nrequire github.com/example/b v0.1.0\n").unwrap()
}
#[test]
fn all_dependencies_are_pinned_and_failure_leaves_project_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("wgt.mod");
    let manifest = manifest();
    fs::write(&path, &manifest.text).unwrap();
    fs::write(temp.path().join("wgt.sum"), b"old lock\n").unwrap();
    let resolver = |_: &str, _: Option<(&str, &str)>| {
        Ok(Candidate {
            version: "v0.2.0".into(),
            selection: None,
        })
    };
    assert!(upgrade::plan(
        &manifest,
        &Lock::new(),
        temp.path(),
        &Packages { broken: true },
        resolver
    )
    .is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), manifest.text);
    assert_eq!(
        fs::read(temp.path().join("wgt.sum")).unwrap(),
        b"old lock\n"
    );
    let result = upgrade::plan(
        &manifest,
        &Lock::new(),
        temp.path(),
        &Packages { broken: false },
        resolver,
    )
    .unwrap();
    upgrade::save(&path, &result).unwrap();
    assert!(fs::read_to_string(&path)
        .unwrap()
        .starts_with("# retain comment"));
    assert!(Manifest::read(&path)
        .unwrap()
        .dependencies
        .values()
        .all(|v| v == "v0.2.0"));
    assert_eq!(result.changes.len(), 2);
}
#[test]
fn registered_dependencies_use_locked_registry_and_name() {
    let temp = tempfile::tempdir().unwrap();
    let manifest = manifest();
    let mut known = Lock::new();
    let module = "github.com/example/a".to_owned();
    for (key, value) in [
        ("registry", "custom/index"),
        ("registry-name", "example/model"),
    ] {
        known.insert((module.clone(), "v0.1.0".into(), key.into()), value.into());
    }
    let remote = Packages { broken: false };
    let result = upgrade::plan(&manifest, &known, temp.path(), &remote, |m, provenance| {
        let selection = if m == module {
            assert_eq!(provenance, Some(("custom/index", "example/model")));
            Some(Selection {
                registry: "custom/index".into(),
                commit: "c".repeat(40),
                record_sha256: "d".repeat(64),
                record: Record {
                    format_version: 1,
                    name: "example/model".into(),
                    module: module.clone(),
                    version: "v0.2.0".into(),
                    source_commit: COMMIT.into(),
                    index_sha256: hash(&remote.index("a")),
                    license: "MIT".into(),
                    edition: "0.5".into(),
                    yanked: false,
                },
            })
        } else {
            assert!(provenance.is_none());
            None
        };
        Ok(Candidate {
            version: "v0.2.0".into(),
            selection,
        })
    })
    .unwrap();
    assert_eq!(
        result.lock[&(module, "v0.2.0".into(), "registry".into())],
        "custom/index"
    );
    assert!(result.lock.keys().all(|(_, v, _)| v == "v0.2.0"));
}
#[test]
fn failed_manifest_write_restores_lock() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("wgt.mod");
    fs::create_dir(&path).unwrap();
    fs::write(temp.path().join("wgt.sum"), b"original").unwrap();
    let result = upgrade::Upgrade {
        manifest: manifest(),
        lock: Lock::new(),
        changes: vec![],
    };
    assert!(upgrade::save(&path, &result).is_err());
    assert_eq!(fs::read(temp.path().join("wgt.sum")).unwrap(), b"original");
}
#[test]
fn cli_rejects_conflicting_or_non_network_upgrades() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("demo");
    fs::create_dir(&project).unwrap();
    fs::write(project.join("wgt.mod"), manifest().text).unwrap();
    fs::write(project.join("wgt.sum"), b"").unwrap();
    for args in [
        vec!["update", "--all", "--offline"],
        vec!["update", "--all", "--locked"],
        vec!["update", "--all", "--version", "v1.0.0"],
        vec!["update", "--all", "github.com/example/a"],
    ] {
        assert!(!Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(&project)
            .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
        assert_eq!(
            fs::read_to_string(project.join("wgt.mod")).unwrap(),
            manifest().text
        );
        assert!(fs::read(project.join("wgt.sum")).unwrap().is_empty());
    }
}
