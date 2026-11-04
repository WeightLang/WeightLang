use anyhow::Result;
use std::{collections::BTreeMap, fs, process::Command};
use weightlang_wgt::{
    manifest::{Lock, Manifest},
    registry::{self, Remote},
    script_compiler,
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
            .ok_or_else(|| anyhow::anyhow!("missing {url}"))?
            .clone();
        anyhow::ensure!(bytes.len() as u64 <= max, "size limit");
        Ok(bytes)
    }
}

#[test]
fn source_operator_release_lock_install_train_and_tamper_detection() {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let dir = tempfile::tempdir_in(scratch).unwrap();
    let project = dir.path().join("project");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("wgt.mod"),
        "module github.com/example/operators\nwgt 0.5\n",
    )
    .unwrap();
    let library = format!(
        "{}\nfn smooth(x) {{ return swish(x); }}\n",
        include_str!("../../../examples/native_training/custom_operators/ops.wgt")
    );
    fs::write(project.join("ops.wgt"), &library).unwrap();
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
    let release = dir.path().join("release");
    let index = registry::package(
        &project,
        &dir.path().join("absent-models"),
        "v0.5.0",
        &release,
        false,
    )
    .unwrap();
    assert_eq!(
        index["artifacts"]["Activations"]["format"],
        "weightlang.operators.v1"
    );
    assert!(!release.join("Activations.model.json").exists());
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
    files.insert("ops.wgt".into(), library.as_bytes().to_vec());
    let remote = MemoryRemote {
        commit: index["commit"].as_str().unwrap().into(),
        files,
    };
    let manifest = Manifest::parse(
        "module local/consumer\nwgt 0.5\nrequire github.com/example/operators v0.5.0\n",
    )
    .unwrap();
    let cache = dir.path().join("cache");
    let lock = registry::fetch(&manifest, &Lock::new(), &cache, false, &remote).unwrap();
    registry::verify(&manifest, &lock, &cache).unwrap();
    assert_eq!(
        lock,
        registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap()
    );
    let module = registry::module_dir(&cache, "github.com/example/operators", "v0.5.0");
    let consumer = dir.path().join("consumer");
    fs::create_dir(&consumer).unwrap();
    fs::write(
        consumer.join("wgt.mod"),
        "module local/consumer\nwgt 0.5\nrequire github.com/example/operators v0.5.0\n",
    )
    .unwrap();
    fs::write(
        consumer.join("wgt.sum"),
        weightlang_wgt::manifest::lock_text(&lock),
    )
    .unwrap();
    let runnable = include_str!("../../../examples/native_training/custom_operators/model.wgt")
        .replace("./ops.wgt", "github.com/example/operators/Activations")
        .replace("activation.swish", "activation.smooth");
    fs::write(consumer.join("model.wgt"), runnable).unwrap();
    let run_cli = || {
        Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(&consumer)
            .env("WEIGHTLANG_CACHE", &cache)
            .env("WGT_WEIGHTFS", "off")
            .args(["--offline", "--locked", "run", "model.wgt"])
            .output()
            .unwrap()
    };
    let cli = run_cli();
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let source=include_str!("../../../examples/native_training/custom_operators/model.wgt")
        .replace("./ops.wgt","github.com/example/operators/Activations")
        .replace("return net([1, 2]);",r#"
            save_checkpoint(net, "checkpoint.json"); save_model(net,"model.json");
            let restored=load_checkpoint("checkpoint.json"); let loaded=load_model("model.json", "none");
            assert(net([1,2])==loaded([1,2]),"portable model");
            assert(fit(net,[[1,2]],[[0,0]],0.03,"loss")==fit(restored,[[1,2]],[[0,0]],0.03,"loss"),"Adam restore");
            return net([1,2]);
        "#);
    let deps = BTreeMap::from([(
        "github.com/example/operators".into(),
        module.to_str().unwrap().into(),
    )]);
    weightlang_runtime::script::run_with_dependencies(&source, dir.path(), 100000, deps).unwrap();
    // The model is self-contained after lowering; no source library is needed to load it.
    fs::remove_dir_all(module.join("Activations")).unwrap();
    let loaded = "wgt 0.5\nfn main() { return predict(model(\"model.json\",\"none\"),[1,2]); }";
    weightlang_runtime::script::run(loaded, dir.path(), 10000).unwrap();
    assert!(registry::verify(&manifest, &lock, &cache).is_err());
    registry::fetch(&manifest, &lock, &cache, true, &remote).unwrap();
    fs::write(module.join("Activations/source.wgt"), "tampered").unwrap();
    assert!(registry::verify(&manifest, &lock, &cache).is_err());
    assert!(!run_cli().status.success());
}

#[test]
fn compiled_operator_model_runs_without_source_library_or_toolchain() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let text = include_str!("../../../examples/native_training/custom_operators/model.wgt");
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("program");
    fs::write(&source, text).unwrap();
    fs::write(
        dir.path().join("ops.wgt"),
        include_str!("../../../examples/native_training/custom_operators/ops.wgt"),
    )
    .unwrap();
    let expected = weightlang_runtime::script::run(text, dir.path(), 100000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    fs::remove_file(dir.path().join("ops.wgt")).unwrap();
    let output = Command::new(executable)
        .env("PATH", "/nonexistent")
        .env("WGT_WEIGHTFS", "off")
        .args(["--max-operations", "100000"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}
