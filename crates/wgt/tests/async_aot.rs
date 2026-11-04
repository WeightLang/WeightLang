use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn native_async_aot_matches_interpreter_without_source() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/async-scheduler-sol/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let text = r#"wgt 0.5
async fn work(x) { await sleep(10); return x + 1; }
async fn nested(x) { return await work(await work(x)); }
async fn long() { await sleep(100); assert false, "long sibling failed"; }
fn main() { tasks { let a = spawn nested(40); let b = spawn work(0); let result = await a; assert await b == 1; let after = spawn work(result); return await after; } }
"#;
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    fs::write(&source, text).unwrap();
    let interpreted = script::run(text, dir.path(), 100000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(&source).unwrap();
    let output = Command::new(executable)
        .env("PATH", "/nonexistent")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let compiled: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(compiled["result"], interpreted.result);
    // Both engines must resume the parent after short target completion while
    // long is pending: the parent assertion must win over the later child error.
    let target_text=r#"wgt 0.5
async fn long(){await sleep(100);assert false,"long sibling failed";}
async fn short(){return 1;}
fn main(){tasks{let a=spawn long();let b=spawn short();let value=await b;assert false,"parent resumed after short";}}
"#;
    fs::write(&source,target_text).unwrap();
    let interpreted=script::run(target_text,dir.path(),100000).unwrap_err();
    assert!(format!("{interpreted:#}").contains("parent resumed after short"));
    let target_executable=dir.path().join("target-wait");
    script_compiler::compile(&source,&target_executable).unwrap();
    fs::remove_file(&source).unwrap();
    let output=Command::new(target_executable).env("PATH","/nonexistent").output().unwrap();
    assert!(!output.status.success());
    let error=String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("parent resumed after short"),"{error}");
}
