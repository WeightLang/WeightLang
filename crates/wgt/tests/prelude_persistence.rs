use std::{fs, path::Path, process::Command};
use weightlang_wgt::script_compiler;

#[test]
fn persistence_aliases_and_resume_match_aot_across_processes() {
    let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/prelude-tests");
    fs::create_dir_all(&cache).unwrap();
    let dir = tempfile::tempdir_in(cache).unwrap();
    let source = dir.path().join("model.wgt");
    fs::write(&source, r#"wgt 0.5
fn main() {
    let rows = jsonl("mode.jsonl");
    let mode = batch(rows,1)[0];
    close(rows);
    if mode == 0 {
        let net = mlp([1,2,1],"tanh",7,"adam");
        fit(net,[[1]],[[2]],0.03,"mse");
        save_checkpoint(net,"checkpoint.json");
        save_model(net,"model.json");
        let old = restore("checkpoint.json");
        let loaded = load_model("model.json","sgd");
        assert(predict(loaded,[1]) == predict(model("model.json","sgd"),[1]),"model alias");
        fit(old,[[1]],[[2]],0.03,"mse");
        save(old,"expected.json");
        return predict(old,[1]);
    }
    if mode == 1 {
        let net = load_checkpoint("checkpoint.json");
        fit(net,[[1]],[[2]],0.03,"mse");
        save_checkpoint(net,"actual.json");
        return predict(net,[1]);
    }
    if mode == 2 { return load_checkpoint("model.json"); }
    return load_model("checkpoint.json","adam");
}
"#).unwrap();
    let executable = dir.path().join("program");
    script_compiler::compile(&source, &executable).unwrap();
    let interpreted = dir.path().join("interpreted");
    let compiled = dir.path().join("compiled");
    fs::create_dir(&interpreted).unwrap();
    fs::create_dir(&compiled).unwrap();
    for mode in 0..4 {
        for root in [&interpreted, &compiled] {
            fs::write(root.join("mode.jsonl"), format!("{mode}\n")).unwrap();
        }
        let reference = weightlang_runtime::script::run(&fs::read_to_string(&source).unwrap(), &interpreted, 100_000);
        let actual = Command::new(&executable).env("PATH", "/nonexistent").arg("--root").arg(&compiled).output().unwrap();
        if mode < 2 {
            assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(serde_json::to_value(reference.unwrap()).unwrap(), actual);
        } else {
            let expected = if mode == 2 { "use load_model" } else { "use load_checkpoint" };
            assert!(format!("{:#}", reference.unwrap_err()).contains(expected));
            assert!(!actual.status.success());
            assert!(String::from_utf8_lossy(&actual.stderr).contains(expected));
        }
    }
    for root in [&interpreted, &compiled] {
        assert_eq!(fs::read(root.join("expected.json")).unwrap(), fs::read(root.join("actual.json")).unwrap());
    }
}
