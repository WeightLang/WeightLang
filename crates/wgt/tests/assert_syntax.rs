use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn assertions_preserve_aot_evaluation_and_diagnostics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    for (case, text) in [
        "wgt 0.5\nfn main(){seed(42); assert random() >= 0, missing(); assert (1) == 1; return random();}",
        "wgt 0.5\nfn main(){assert false, \"explicit message\";}",
        "wgt 0.5\nfn main(){assert(true,missing());}",
    ].into_iter().enumerate() {
        let source = dir.path().join("model.wgt");
        let executable = dir.path().join(format!("compiled-{case}"));
        fs::write(&source, text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        let expected = script::run(text, dir.path(), 10_000_000);
        let actual = Command::new(executable).env("PATH", "/nonexistent").output().unwrap();
        match expected {
            Ok(report) => {
                assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
                assert_eq!(serde_json::from_slice::<serde_json::Value>(&actual.stdout).unwrap(), serde_json::to_value(report).unwrap());
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
