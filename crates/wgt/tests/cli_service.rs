//! Focused real-service coverage for CLI execution without a linked model core.
use std::{fs, process::Command, time::{Duration, Instant}};

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON and local Cargo toolchain"]
fn ordinary_run_named_tests_and_export_use_service() {
    use std::os::unix::fs::PermissionsExt;
    struct Daemon(std::process::Child);
    impl Drop for Daemon { fn drop(&mut self) { let _=self.0.kill(); let _=self.0.wait(); } }
    let cache=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/cli-service-tests");
    fs::create_dir_all(&cache).unwrap();
    let dir=tempfile::tempdir_in(cache).unwrap();
    let socket_root=tempfile::Builder::new().prefix("cli-ipc-").tempdir_in("/tmp/weightlang").unwrap();
    fs::set_permissions(socket_root.path(),fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon=Daemon(Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(socket_root.path()).spawn().unwrap());
    let endpoint=socket_root.path().join("service.sock");
    let begin=Instant::now(); while !endpoint.exists() { assert!(begin.elapsed()<Duration::from_secs(10)); std::thread::sleep(Duration::from_millis(5)); }
    fs::write(dir.path().join("wgt.toml"),"[project]\nmodule=\"local/cli-service\"\nedition=\"0.5\"\n").unwrap();
    fs::write(dir.path().join("model.wgt"),"wgt 0.5\nfn main(){let net=mlp([1,1],\"identity\",7,\"sgd\");save model net to \"model.json\";return net([1]);}").unwrap();
    let cli=|args:&[&str]| Command::new(env!("CARGO_BIN_EXE_wgt")).current_dir(dir.path()).env("WGTIME_ENDPOINT",&endpoint).args(args).output().unwrap();
    let check=cli(&["run","model.wgt","--check"]);
    assert!(check.status.success(),"{}",String::from_utf8_lossy(&check.stderr));
    assert!(!dir.path().join("model.json").exists());
    let run=cli(&["run","model.wgt"]);
    assert!(run.status.success(),"{}",String::from_utf8_lossy(&run.stderr));
    assert!(dir.path().join("model.json").exists());
    let export=cli(&["export","model.json","-o","copy.json"]);
    assert!(export.status.success(),"{}",String::from_utf8_lossy(&export.stderr));
    assert_eq!(fs::read(dir.path().join("model.json")).unwrap(),fs::read(dir.path().join("copy.json")).unwrap());
    fs::create_dir(dir.path().join("tests")).unwrap();
    fs::write(dir.path().join("tests/check.wgt"),"wgt 0.5\nfn main(){assert false,\"main must not run\";}\nfn test_ok(){assert true;}\nfn test_fail(){assert false,\"named failure\";}").unwrap();
    let tests=cli(&["test"]);
    assert!(!tests.status.success());
    let stdout=String::from_utf8_lossy(&tests.stdout); let stderr=String::from_utf8_lossy(&tests.stderr);
    assert!(stdout.contains("1 passed; 1 failed; 2 total"),"{stdout} {stderr}");
    assert!(stderr.contains("named failure"),"{stderr}");
    assert!(!stderr.contains("main must not run"),"{stderr}");
    let filtered=cli(&["test","--filter","test_ok"]);
    assert!(filtered.status.success(),"{}",String::from_utf8_lossy(&filtered.stderr));

    // Legacy build still executes explicit main and saves a resumable artifact via IPC.
    fs::write(dir.path().join("build.wgt"),"wgt 0.5\nfn main(){let net=mlp([1,1],\"identity\",7,\"adam\");train net on [[1]] against [[2]] {rate:0.03;};return net;}").unwrap();
    let build=Command::new(env!("CARGO_BIN_EXE_weightc")).current_dir(dir.path()).env("WGTIME_ENDPOINT",&endpoint).args(["build","build.wgt","-o","built"]).output().unwrap();
    assert!(build.status.success(),"{}",String::from_utf8_lossy(&build.stderr));
    let manifest:serde_json::Value=serde_json::from_slice(&build.stdout).unwrap();
    assert_eq!(manifest["training_steps"],1);
    assert!(dir.path().join("built/checkpoint.json").is_file());
    let package=cli(&["package","--entry","built","-o","bundle"]);
    assert!(package.status.success(),"{}",String::from_utf8_lossy(&package.stderr));
    let repeated=Command::new(env!("CARGO_BIN_EXE_weightc")).current_dir(dir.path()).env("WGTIME_ENDPOINT",&endpoint).args(["build","build.wgt","-o","built"]).output().unwrap();
    assert!(!repeated.status.success());

    // A local source package is resolved explicitly at compile time; runtime needs neither source.
    let dependency=dir.path().join("dependency");fs::create_dir(&dependency).unwrap();
    fs::write(dependency.join("model.wgt"),"wgt 0.5\noperators Shift\nfn shifted(x){return x+x;}\n").unwrap();
    fs::write(dependency.join("wgt.index.json"),r#"{"format_version":2,"sources":{"model.wgt":{}}}"#).unwrap();
    let source="wgt 0.5\nimport shift from \"local/dependency/Shift\"\nmodel Probe\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return shift.shifted(x);}\nfn main(){let net=Probe(\"none\");return net([2]);}\n";
    let dependencies=std::collections::BTreeMap::from([("local/dependency".into(),dependency.to_str().unwrap().to_owned())]);
    let executable=dir.path().join("imported-client");
    weightlang_wgt::script_compiler::compile_text_entry_with_dependencies(source,dir.path(),&executable,"main",&dependencies).unwrap();
    fs::remove_dir_all(&dependency).unwrap();
    let imported=Command::new(&executable).env("PATH","/nonexistent").env("WGTIME_ENDPOINT",&endpoint).output().unwrap();
    assert!(imported.status.success(),"{}",String::from_utf8_lossy(&imported.stderr));
    let imported:serde_json::Value=serde_json::from_slice(&imported.stdout).unwrap();
    assert_eq!(imported["result"],serde_json::json!([4.]));
}
