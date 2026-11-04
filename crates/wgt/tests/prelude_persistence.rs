use std::{fs, path::Path, process::Command};
use weightlang_wgt::script_compiler;

#[test]
fn persistence_aliases_and_resume_match_aot_across_processes() {
    let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/prelude-tests");
    fs::create_dir_all(&cache).unwrap();
    let dir = tempfile::tempdir_in(cache).unwrap();
    use std::os::unix::fs::PermissionsExt;
    struct Daemon(std::process::Child, std::path::PathBuf);
    impl Drop for Daemon {fn drop(&mut self){let _=self.0.kill();let _=self.0.wait();let _=fs::remove_dir_all(&self.1);}}
    let socket_root=std::path::PathBuf::from(format!("/tmp/weightlang/prelude-ipc-{}",std::process::id()));
    fs::create_dir(&socket_root).unwrap();fs::set_permissions(&socket_root,fs::Permissions::from_mode(0o700)).unwrap();
    let daemon_path=std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt WGTIME_TEST_DAEMON required for default AOT");
    let _daemon=Daemon(Command::new(daemon_path).arg("--test-directory").arg(&socket_root).spawn().unwrap(),socket_root.clone());
    let endpoint=socket_root.join("service.sock");let begin=std::time::Instant::now();while !endpoint.exists(){assert!(begin.elapsed()<std::time::Duration::from_secs(10));std::thread::sleep(std::time::Duration::from_millis(5));}
    let template = r#"wgt 0.5
fn main() {
    let mode = MODE;
    if mode == 0 {
        let net = mlp([1,2,1],"tanh",7,"adam");
        train net on [[1]] against [[2]] { rate: 0.03; loss: mse; };
        save checkpoint net to "checkpoint.json";
        save model net to "model.json";
        let old = load checkpoint from "checkpoint.json";
        train old on [[1]] against [[2]] { rate: 0.03; loss: mse; };
        save checkpoint old to "expected.json";
        return old([1]);
    }
    if mode == 1 {
        let net = load checkpoint from "checkpoint.json";
        train net on [[1]] against [[2]] { rate: 0.03; loss: mse; };
        save checkpoint net to "actual.json";
        return net([1]);
    }
    if mode == 4 {
        let loaded = load model from "model.json" using "sgd";
        let alias = load model from "model.json" using "sgd";
        assert loaded([1]) == alias([1]), "model alias";
        return loaded([1]);
    }
    if mode == 2 { return load checkpoint from "model.json"; }
    return load model from "checkpoint.json" using "adam";
}
"#;
    let interpreted = dir.path().join("interpreted");
    let compiled = dir.path().join("compiled");
    fs::create_dir(&interpreted).unwrap();
    fs::create_dir(&compiled).unwrap();
    for mode in 0..5 {
        let source = dir.path().join(format!("mode-{mode}.wgt"));
        let executable = dir.path().join(format!("program-{mode}"));
        let text = template.replace("MODE", &mode.to_string());
        fs::write(&source, &text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(&source).unwrap();
        let reference = weightlang_runtime::script::run(&text, &interpreted, 100_000);
        let actual = Command::new(&executable).env("PATH", "/nonexistent").env("WGTIME_ENDPOINT", &endpoint).arg("--root").arg(&compiled).output().unwrap();
        if mode < 2 || mode == 4 {
            assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(serde_json::to_value(reference.unwrap()).unwrap(), actual);
        } else {
            let expected = "load format does not match model/checkpoint kind";
            assert!(format!("{:#}", reference.unwrap_err()).contains(expected));
            assert!(!actual.status.success());
            assert!(String::from_utf8_lossy(&actual.stderr).contains(expected));
        }
    }
    for root in [&interpreted, &compiled] {
        assert_eq!(fs::read(root.join("expected.json")).unwrap(), fs::read(root.join("actual.json")).unwrap());
    }
}
