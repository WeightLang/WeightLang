mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn native_async_aot_matches_interpreter_without_source() {
    let directory = support::scratch().unwrap();
    let source = directory.path().join("model.wgt");
    let text = r#"wgt 0.5
async fn work(x){return x+1;}
async fn nested(x){return await work(await work(x));}
fn main(){tasks{
    let a=spawn nested(40);
    let b=spawn work(0);
    let result=await a;
    assert await b==1;
    let after=spawn work(result);
    return await after;
}}
"#;
    fs::write(&source, text).unwrap();
    let reference = script::run(text, directory.path(), 100_000).unwrap();
    assert_eq!(reference.result, serde_json::json!(43.));
    let executable = directory.path().join("compiled");
    script_compiler::compile(&source, &executable).unwrap();

    let error_text = r#"wgt 0.5
async fn work(x){return x+1;}
fn main(){tasks{
    let a=spawn work(40);
    let b=spawn work(0);
    assert await a==41;
    assert await b==1;
    assert false,"parent resumed after targets";
}}
"#;
    fs::write(&source, error_text).unwrap();
    let error = script::run(error_text, directory.path(), 100_000).unwrap_err();
    assert!(format!("{error:#}").contains("parent resumed after targets"));
    let error_executable = directory.path().join("parent-error");
    script_compiler::compile(&source, &error_executable).unwrap();
    fs::remove_file(source).unwrap();

    let service = support::Service::start().unwrap();
    let output = service
        .configure(Command::new(executable).env("PATH", "/nonexistent"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let compiled: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(compiled["result"], reference.result);

    let output = service
        .configure(Command::new(error_executable).env("PATH", "/nonexistent"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("parent resumed after targets"), "{error}");
}

#[test]
fn asynchronous_expressions_keep_the_common_value_contract() {
    let directory = support::scratch().unwrap();
    let source = directory.path().join("model.wgt");
    let cases = [
        (
            "values",
            "let __record_field=0;assert record.values[1]==2;assert true||missing();assert !(false&&missing());return -record.values[0]+record.values[1];",
            None,
        ),
        ("and_type", "return true&&2;", Some("expected bool")),
        ("or_type", "return false||2;", Some("expected bool")),
        (
            "index",
            "return record.values[2];",
            Some("index out of bounds"),
        ),
        (
            "assertion",
            "assert false,\"async assertion\";",
            Some("line 2: async assertion"),
        ),
    ];
    let mut compiled = Vec::new();
    for (name, body, expected_error) in cases {
        let text = format!(
            "wgt 0.5\nasync fn work(record){{{body}}}\nfn main(){{return await work({{values:[1,2]}});}}\n"
        );
        let reference = script::run(&text, directory.path(), 100_000);
        if expected_error.is_none() {
            assert_eq!(reference.unwrap().result, serde_json::json!(1.));
        } else {
            assert!(reference.is_err(), "invalid {name} executed in the core");
        }
        let executable = directory.path().join(name);
        fs::write(&source, text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        compiled.push((name, executable, expected_error));
    }
    fs::remove_file(source).unwrap();
    let service = support::Service::start().unwrap();
    for (name, executable, expected_error) in compiled {
        let output = service
            .configure(Command::new(executable).env("PATH", "/nonexistent"))
            .output()
            .unwrap();
        if let Some(expected_error) = expected_error {
            assert!(
                !output.status.success(),
                "invalid {name} executed in the client"
            );
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains(expected_error), "{name}: {error}");
        } else {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(report["result"], serde_json::json!(1.));
        }
    }
}
