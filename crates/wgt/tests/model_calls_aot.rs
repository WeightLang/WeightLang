use std::{fs, path::Path, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn model_calls_match_interpreter_in_compiled_functions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
fn infer(net,x) { return net(x); }
fn main() {
 let rows=jsonl("case.jsonl");
 let mode=batch(rows,1)[0]["mode"];
 close(rows);
 let net=mlp([2,1],"identity",42,"adam");
 if mode==1 { return net([1]); }
 if mode==2 { let len=3; return len([1]); }
 if mode==3 { return net([1],[2]); }
 let other=mlp([2,1],"identity",7,"sgd");
 let len=net;
 let mean=0;
 let one_hot=0;
 assert([2,4].mean==3,"postfix mean bypasses local binding");
 assert(onehot[2] 1 == [0,1],"onehot bypasses local binding");
 assert(len([1,2])==predict(net,[1,2]),"builtin shadowing");
 assert(infer(other,[1,2])==predict(other,[1,2]),"parameter dispatch");
 return infer(net,[1,2]);
}"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    for mode in 0..4 {
        fs::write(dir.path().join("case.jsonl"), format!("{{\"mode\":{mode}}}\n")).unwrap();
        let reference = script::run(text, dir.path(), 100_000);
        let actual = Command::new(&executable).env("PATH", "/nonexistent")
            .arg("--root").arg(dir.path()).args(["--max-operations", "100000"])
            .output().unwrap();
        if mode == 0 {
            assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
            let value: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(serde_json::to_value(reference.unwrap()).unwrap(), value);
        } else {
            assert!(!actual.status.success());
            let error = format!("{:#}", reference.unwrap_err());
            let expected = error.rsplit(": ").next().unwrap();
            assert!(String::from_utf8_lossy(&actual.stderr).contains(expected), "{error}");
        }
    }
}
