use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;
#[test]
fn record_collection_aot_preserves_control_flow_projection_and_budgets() {
 let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");fs::create_dir_all(&root).unwrap();
 let dir=tempfile::tempdir_in(root).unwrap();let source=dir.path().join("model.wgt");let executable=dir.path().join("compiled");
 let text=r#"wgt 0.5
fn early(){let b=collect {return 7;};return 0;}
fn main(){let batch=collect {for i in range(0,35){let inner=collect {yield {x:i};};if i==1 {continue;}yield {x:[i],y:[i*2],n:len(inner)};}};
 let sizes=[];let total=0;for part in batch size 16 {sizes=append(sizes,len(part));for x in part.x {total=total+x[0];}}
 return [sizes,total,batch[0].y,early()];}"#;
 fs::write(&source,text).unwrap();script_compiler::compile(&source,&executable).unwrap();fs::remove_file(source).unwrap();
 let expected=script::run(text,dir.path(),100000).unwrap();
 let output=Command::new(executable).env("PATH","/nonexistent").arg("--root").arg(dir.path()).args(["--max-operations","100000"]).output().unwrap();
 assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
 let actual:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(actual,serde_json::to_value(expected).unwrap());
}
