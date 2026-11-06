mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn assertions_preserve_aot_evaluation_and_diagnostics() {
    let service = support::Service::start().unwrap();
    let dir = support::scratch().unwrap();
    for (case, text) in [
        "wgt 0.5\nfn main(){assert 2 > 1, missing(); let parenthesized = (1); assert parenthesized == 1; return 42;}",
        "wgt 0.5\nfn main(){assert false, \"explicit message\";}",
        "wgt 0.5\nfn main(){assert true,missing();}",
    ].into_iter().enumerate() {
        let source = dir.path().join("model.wgt");
        let executable = dir.path().join(format!("compiled-{case}"));
        fs::write(&source, text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        let expected = script::run(text, dir.path(), 10_000_000);
        let actual = service.configure(Command::new(executable).env("PATH", "/nonexistent")).output().unwrap();
        match expected {
            Ok(report) => {
                assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
                let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
                assert_eq!(actual["result"], report.result);
                assert_eq!(actual["metrics"], serde_json::to_value(report.metrics).unwrap());
            }
            Err(error) => {
                assert!(!actual.status.success());
                let error = format!("{error:#}");
                let actual = String::from_utf8_lossy(&actual.stderr);
                assert!(actual.contains("line 2") && actual.contains(error.rsplit(": ").next().unwrap()), "{actual}");
            }
        }
    }
}
