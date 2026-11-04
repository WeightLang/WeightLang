use std::{fs,path::Path,process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;
#[test]
fn native_agent_environment_aot_executes_default_learner_and_lifecycle() {
 let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");fs::create_dir_all(&root).unwrap();
 let dir=tempfile::tempdir_in(root).unwrap();let source=dir.path().join("model.wgt");let executable=dir.path().join("compiled");
 let text=r#"wgt 0.5
model Policy
input x:f32[1]
output y:f32[1]
parameter w:f32[1]=[1]
fn forward(x){return w*x;}
fn loss(p,t){return sum(square(p-t));}
agent Actor {policy;learner:update;}
environment Counter {observation:f32[1];action:f32[1];feedback:f32[1];reset:reset_env;step:step_env;}
fn reset_env(state,seed){return {state:state,observation:[state]};}
fn step_env(state,action){let next=state+action[0];return {state:next,observation:[next],feedback:[next],terminated:next>=3,truncated:false};}
fn update(actor,batch){let model=actor.policy;train model on batch.x against batch.y {rate:0.1;clip:1;loss:loss;};return len(batch);}
fn main(){let model=Policy("sgd");let actor=Actor(model);let env=Counter(0,7);let report=learn actor {for i in [1,2,3]{let t=env.step([1]);yield {x:[i],y:[0]};}};return [report,model([1]),env.reset()];}"#;
 fs::write(&source,text).unwrap();script_compiler::compile(&source,&executable).unwrap();fs::remove_file(source).unwrap();
 let expected=script::run(text,dir.path(),100000).unwrap();
 let output=Command::new(executable).env("PATH","/nonexistent").arg("--root").arg(dir.path()).args(["--max-operations","100000"]).output().unwrap();
 assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
 let actual:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(actual,serde_json::to_value(expected).unwrap());
}
