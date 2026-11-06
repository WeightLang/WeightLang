mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn model_calls_match_core_in_compiled_functions() {
    let service = support::Service::start().unwrap();
    let dir = support::scratch().unwrap();
    let text = r#"wgt 0.5
fn infer(net,x) { return net(x); }
fn main() {
 let net=mlp([2,1],"identity",42,"adam");
 let other=mlp([2,1],"identity",7,"sgd");
 let len=net;
 assert len([1,2])==net([1,2]),"builtin shadowing";
 assert infer(other,[1,2])==other([1,2]),"parameter dispatch";
 return infer(net,[1,2]);
}"#;
    for (case, text, diagnostic) in [
        ("valid", text.to_owned(), None),
        (
            "binding",
            text.replace("return infer(net,[1,2]);", "let len=3; return len([1]);"),
            Some("not callable"),
        ),
        (
            "shape",
            text.replace("return infer(net,[1,2]);", "return net([1]);"),
            Some("input"),
        ),
        (
            "arity",
            text.replace("return infer(net,[1,2]);", "return net([1],[2]);"),
            Some("one input"),
        ),
    ] {
        let source = dir.path().join("model.wgt");
        let executable = dir.path().join(case);
        fs::write(&source, &text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        let reference = script::run(&text, dir.path(), 100_000);
        let actual = service
            .configure(Command::new(&executable).env("PATH", "/nonexistent"))
            .output()
            .unwrap();
        if let Some(diagnostic) = diagnostic {
            assert!(reference.is_err());
            assert!(!actual.status.success());
            assert!(
                String::from_utf8_lossy(&actual.stderr).contains(diagnostic),
                "{}",
                String::from_utf8_lossy(&actual.stderr)
            );
        } else {
            assert!(
                actual.status.success(),
                "{}",
                String::from_utf8_lossy(&actual.stderr)
            );
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            let expected = reference.unwrap();
            // Model results are f32 even though both JSON reports carry JSON numbers.
            assert_eq!(
                actual["result"][0].as_f64().unwrap() as f32,
                expected.result[0].as_f64().unwrap() as f32
            );
            assert_eq!(
                actual["metrics"],
                serde_json::to_value(expected.metrics).unwrap()
            );
        }
    }
}
