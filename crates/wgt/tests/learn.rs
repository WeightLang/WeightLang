use std::{fs,path::Path,process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;
#[test]
fn learn_aot_calls_static_learner_once_after_unlocking() {
 let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");fs::create_dir_all(&root).unwrap();
 let dir=tempfile::tempdir_in(root).unwrap();let source=dir.path().join("model.wgt");let executable=dir.path().join("compiled");
 let text=r#"wgt 0.5
model Policy
input x:f32[1]
output y:f32[1]
parameter w:f32[1]=[1]
fn forward(x){return w*x;}
fn loss(p,t){return sum(square(p-t));}
fn update(model,batch){train model on (batch.x,batch.y) rate 0.1 clip 1;return len(batch);}
fn main(){let model=Policy("sgd");let update=3;let report=learn model using update {for i in [1,2,3]{yield {x:[i],y:[0]};}};return [report,model([1])];}"#;
 fs::write(&source,text).unwrap();script_compiler::compile(&source,&executable).unwrap();fs::remove_file(source).unwrap();
 let expected=script::run(text,dir.path(),100000).unwrap();
 let output=Command::new(executable).env("PATH","/nonexistent").arg("--root").arg(dir.path()).args(["--max-operations","100000"]).output().unwrap();
 assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
 let actual:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(actual,serde_json::to_value(expected).unwrap());
}
