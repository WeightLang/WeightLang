use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn sample_from_aot_preserves_rng_state_and_failure_semantics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model Policy
input x: f32[3]
parameter memory: f32[3] = [0,0,0]
output y: f32[3]
fn forward(x) { let next=memory+x; write(memory,next); return next; }
fn draw(net,x) { return sample from net(x); }
fn main() {
 let rows=jsonl("case.jsonl"); let mode=batch(rows,1)[0]["mode"]; close(rows);
 let net=Policy("adam"); seed(91);
 if mode==1 { return sample from net([1]); }
 if mode==2 { let invalid=1; return sample from invalid([1,2,3]); }
 if mode==3 { return sample from net([1e300,0,0]); }
 let actions=[];
 for i in range(0,32) {
  let action=draw(net,[random(),random(),random()]);
  assert(action>=0 && action<3,"class range"); actions=append(actions,action);
 }
 assert(predict(net,[0,0,0])==[0,0,0],"sample committed write");
 let expected=net([1,2,3]); assert(net([0,0,0])==expected,"model call did not commit write");
 return [actions,random(),sample 0 {then{1/0}else{42}}];
}"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    for mode in 0..4 {
        fs::write(
            dir.path().join("case.jsonl"),
            format!("{{\"mode\":{mode}}}\n"),
        )
        .unwrap();
        let expected = script::run(text, dir.path(), 100_000);
        let output = Command::new(&executable)
            .env("PATH", "/nonexistent")
            .arg("--root")
            .arg(dir.path())
            .args(["--max-operations", "100000"])
            .output()
            .unwrap();
        if mode == 0 {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(actual, serde_json::to_value(expected.unwrap()).unwrap());
        } else {
            assert!(!output.status.success());
            let error = format!("{:#}", expected.unwrap_err());
            assert!(
                String::from_utf8_lossy(&output.stderr)
                    .contains(error.rsplit(": ").next().unwrap()),
                "{error}"
            );
        }
    }
}
