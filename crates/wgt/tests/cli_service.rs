//! Real generated-client and artifact coverage against an isolated service.
mod support;
use std::{fs, process::Command};
use weightlang_wgt::script_compiler;

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON and local Cargo toolchain"]
fn explicit_entries_build_and_export_use_service() {
    let service = support::Service::start().unwrap();
    let dir = support::scratch().unwrap();
    let source = dir.path().join("model.wgt");
    fs::write(&source,"wgt 0.5\nfn main(){let net=mlp([1,1],\"identity\",7,\"sgd\");save model net to \"model.json\";return net([1]);}").unwrap();
    let executable = dir.path().join("client");
    script_compiler::compile(&source, &executable).unwrap();
    assert!(
        !dir.path().join("model.json").exists(),
        "compilation executed the entry"
    );
    fs::remove_file(&source).unwrap();
    let run = service
        .configure(Command::new(executable).env("PATH", "/nonexistent"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let cli = |args: &[&str]| {
        service
            .configure(
                Command::new(env!("CARGO_BIN_EXE_wgt"))
                    .current_dir(dir.path())
                    .args(args),
            )
            .output()
            .unwrap()
    };
    let export = cli(&["export", "model.json", "-o", "copy.json"]);
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    assert_eq!(
        fs::read(dir.path().join("model.json")).unwrap(),
        fs::read(dir.path().join("copy.json")).unwrap()
    );

    let named = "wgt 0.5\nfn main(){assert false,\"main must not run\";}\nfn test_ok(){assert true;}\nfn test_fail(){assert false,\"named failure\";}";
    for (entry, succeeds) in [("test_ok", true), ("test_fail", false)] {
        let binary = dir.path().join(entry);
        script_compiler::compile_text_entry(named, dir.path(), &binary, entry).unwrap();
        let result = service
            .configure(Command::new(binary).env("PATH", "/nonexistent"))
            .output()
            .unwrap();
        assert_eq!(
            result.status.success(),
            succeeds,
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!error.contains("main must not run"), "{error}");
        if !succeeds {
            assert!(error.contains("named failure"), "{error}");
        }
    }

    // This explicit compatibility build path remains implemented; package/export only consume its outputs.
    let build_source = dir.path().join("build-source");
    fs::create_dir(&build_source).unwrap();
    fs::write(build_source.join("model.wgt"),"wgt 0.5\nfn main(){let net=mlp([1,1],\"identity\",7,\"adam\");train net on [[1]] against [[2]] {rate:0.03;};return net;}").unwrap();
    let build = service
        .configure(
            Command::new(env!("CARGO_BIN_EXE_weightc"))
                .current_dir(dir.path())
                .args(["build", "build-source/model.wgt", "-o", "built"]),
        )
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let manifest: serde_json::Value = serde_json::from_slice(&build.stdout).unwrap();
    assert_eq!(manifest["training_steps"], 1);
    assert!(dir.path().join("built/checkpoint.json").is_file());
    let package = cli(&["package", "--entry", "built", "-o", "bundle"]);
    assert!(
        package.status.success(),
        "{}",
        String::from_utf8_lossy(&package.stderr)
    );
    let before = fs::read(dir.path().join("built/checkpoint.json")).unwrap();
    let repeated = service
        .configure(
            Command::new(env!("CARGO_BIN_EXE_weightc"))
                .current_dir(dir.path())
                .args(["build", "build-source/model.wgt", "-o", "built"]),
        )
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert_eq!(
        fs::read(dir.path().join("built/checkpoint.json")).unwrap(),
        before
    );

    // A local source package is resolved explicitly at compile time; runtime needs neither source.
    let dependency = dir.path().join("dependency");
    fs::create_dir(&dependency).unwrap();
    fs::write(
        dependency.join("model.wgt"),
        "wgt 0.5\noperators Shift\nfn shifted(x){return x+x;}\n",
    )
    .unwrap();
    fs::write(
        dependency.join("wgt.index.json"),
        r#"{"format_version":2,"sources":{"model.wgt":{}}}"#,
    )
    .unwrap();
    let source = "wgt 0.5\nimport shift from \"local/dependency/Shift\"\nmodel Probe\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return shift.shifted(x);}\nfn main(){let net=Probe(\"none\");return net([2]);}\n";
    let dependencies = std::collections::BTreeMap::from([(
        "local/dependency".into(),
        dependency.to_str().unwrap().to_owned(),
    )]);
    let executable = dir.path().join("imported-client");
    weightlang_wgt::script_compiler::compile_text_entry_with_dependencies(
        source,
        dir.path(),
        &executable,
        "main",
        &dependencies,
    )
    .unwrap();
    fs::remove_dir_all(&dependency).unwrap();
    let imported = Command::new(&executable)
        .env("PATH", "/nonexistent")
        .env("WGTIME_ENDPOINT", service.endpoint())
        .output()
        .unwrap();
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stderr)
    );
    let imported: serde_json::Value = serde_json::from_slice(&imported.stdout).unwrap();
    assert_eq!(imported["result"], serde_json::json!([4.]));
}
