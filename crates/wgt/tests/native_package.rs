use anyhow::Result;
use std::{collections::BTreeMap, fs, path::Path, process::Command};
use weightlang_wgt::{
    manifest::{Lock, Manifest},
    package_artifact::Info,
    registry::{self, Remote},
};

struct MemoryRemote {
    commit: String,
    files: BTreeMap<String, Vec<u8>>,
}
impl Remote for MemoryRemote {
    fn resolve(&self, _: &str, _: &str) -> Result<String> {
        Ok(self.commit.clone())
    }
    fn download(&self, url: &str, max: u64) -> Result<Vec<u8>> {
        let bytes = self
            .files
            .get(url.rsplit('/').next().unwrap())
            .ok_or_else(|| anyhow::anyhow!("unexpected download {url}"))?
            .clone();
        anyhow::ensure!(bytes.len() as u64 <= max, "size limit");
        Ok(bytes)
    }
}
#[test]
fn native_release_roundtrip_verifies_source_weights_checkpoint_and_lock() {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let dir = tempfile::tempdir_in(scratch).unwrap();
    let project = dir.path().join("project");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("wgt.mod"),
        "module github.com/example/native\nwgt 0.5\n",
    )
    .unwrap();
    let text="wgt 0.5\nmodel Net\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(out=1)\ninit weights=xavier_uniform(7) bias=zeros()\nfn main() { let net = Net(\"sgd\"); train net on [[0]] against [[1]] {rate:0.1;loss:mse;}; return net; }";
    fs::create_dir(project.join("net")).unwrap();
    fs::write(project.join("net/model.wgt"), text).unwrap();
    for args in [
        vec!["init", "-q"],
        vec!["add", "."],
        vec![
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        assert!(Command::new("git")
            .current_dir(&project)
            .args(args)
            .status()
            .unwrap()
            .success());
    }
    let artifacts = dir.path().join("models");
    let model = artifacts.join("Net");
    let output = Command::new(env!("CARGO_BIN_EXE_weightc"))
        .arg("legacy-build")
        .arg(project.join("net/model.wgt"))
        .arg("-o")
        .arg(&model)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(Info::read(&model).unwrap().untrained.is_empty());
    let release = dir.path().join("release");
    let index = registry::package(&project, &artifacts, "v0.5.0", &release, false).unwrap();
    assert_eq!(index["artifacts"]["Net"]["source_path"], "net/model.wgt");
    let mut files: BTreeMap<String, Vec<u8>> = fs::read_dir(&release)
        .unwrap()
        .map(|p| {
            let p = p.unwrap().path();
            (
                p.file_name().unwrap().to_str().unwrap().into(),
                fs::read(p).unwrap(),
            )
        })
        .collect();
    files.insert("model.wgt".into(), text.as_bytes().to_vec());
    let remote = MemoryRemote {
        commit: index["commit"].as_str().unwrap().into(),
        files,
    };
    let manifest = Manifest::parse(
        "module local/consumer\nwgt 0.5\nrequire github.com/example/native v0.5.0\n",
    )
    .unwrap();
    let cache = dir.path().join("cache");
    let lock = registry::fetch(&manifest, &Lock::new(), &cache, false, &remote).unwrap();
    registry::verify(&manifest, &lock, &cache).unwrap();
    assert_eq!(
        lock,
        registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap()
    );
    let cached = registry::module_dir(&cache, "github.com/example/native", "v0.5.0").join("Net");
    let source = format!(
        "wgt 0.5\nfn main() {{ let m = model({:?}, \"sgd\"); return m([0]); }}",
        cached.join("model.json").to_str().unwrap()
    );
    let result = wgtime_core::script::run(&source, Path::new("."), 1000).unwrap();
    assert!((result.result[0].as_f64().unwrap() - 0.2).abs() < 1e-6);
    fs::write(cached.join("checkpoint.json"), "{}").unwrap();
    assert!(registry::verify(&manifest, &lock, &cache).is_err());
}
