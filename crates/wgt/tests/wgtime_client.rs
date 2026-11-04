//! Default generated clients must run with no source/toolchain and no execution-core dependency.
#[test]
#[ignore="requires WGTIME_TEST_DAEMON; unified training real default AOT"]
fn default_client_unified_training() {
 let root=PathBuf::from("/tmp/weightlang").join(format!("unified-client-{}",std::process::id()));fs::create_dir_all(&root).unwrap();fs::set_permissions(&root,fs::Permissions::from_mode(0o700)).unwrap();
 let child=Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(&root).spawn().unwrap();let _daemon=Daemon{child,root:root.clone()};let endpoint=root.join("service.sock");let begin=Instant::now();while !endpoint.exists(){assert!(begin.elapsed()<Duration::from_secs(10));std::thread::sleep(Duration::from_millis(5));}
 let source=root.join("model.wgt");let binary=root.join("program");
 let text=r#"wgt 0.5
model Regression
input x:f32[1]
output y:f32[1]
parameter w:f32[1]=[0]
fn forward(x){return x*w;}
fn loss(prediction,target){return square(prediction-target).mean;}
fn scalar(net,x,y){return 17;}
fn nested(net,x,y){return {loss:17,trace:[1,2]};}
fn main(){
 let net=Regression("sgd");let model=net;assert model([1])==[0];
 let loss=train net on [[1],[2]] against [[2],[4]] {rate:0.1;loss:loss;clip:1;};
 assert loss==10;assert net([1])[0]>0.099;assert net([1])[0]<0.101;
 let value=train net on [[1]] against [[2]] using scalar;assert value==17;
 let record=train net on [[1]] against [[2]] using nested;assert record.trace==[1,2];
 let classifier=mlp([1,2],"identity",42,"sgd");let other=train classifier on [[1]] against [[2,0]] {rate:0.01;clip:1;};
 let ce=train classifier on [[1],[2]] against [0,1] {rate:0.01;loss:cross_entropy;};
 return [loss,value,other,ce];
}"#;
 fs::write(&source,text).unwrap();let interpreted=wgtime_core::script::run(text,&root,100000).unwrap();
 weightlang_wgt::script_compiler::compile(&source,&binary).unwrap();fs::remove_file(&source).unwrap();
 let out=Command::new(&binary).env_clear().env("PATH","/nonexistent").env("WGTIME_ENDPOINT",&endpoint).output().unwrap();assert!(out.status.success(),"{}",String::from_utf8_lossy(&out.stderr));let json:serde_json::Value=serde_json::from_slice(&out.stdout).unwrap();assert_eq!(json["result"][1],interpreted.result[1],"strategy f64 result remains exact");for index in [0,2,3] {assert_eq!(json["result"][index].as_f64().unwrap() as f32,interpreted.result[index].as_f64().unwrap() as f32,"model loss must retain exact f32 value across JSON transport");}
 let removed="wgt 0.5\nfn main(){return model(\"missing.json\",\"sgd\");}";
 fs::write(&source,removed).unwrap();let rejected=root.join("removed");weightlang_wgt::script_compiler::compile(&source,&rejected).unwrap();
 let out=Command::new(&rejected).env("WGTIME_ENDPOINT",&endpoint).output().unwrap();assert!(!out.status.success());assert!(String::from_utf8_lossy(&out.stderr).contains("model(...) loading was removed"));
}


#[test]
#[ignore="requires WGTIME_TEST_DAEMON; real default-client budget regression"]
fn default_client_operation_budgets() {
    let root=std::path::PathBuf::from("/tmp/weightlang").join(format!("client-budget-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&root,std::fs::Permissions::from_mode(0o700)).unwrap();
    let child=std::process::Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(&root).spawn().unwrap();
    let _daemon=Daemon{child,root:root.clone()};
    let endpoint=root.join("service.sock");let start=std::time::Instant::now();while !endpoint.exists(){assert!(start.elapsed()<std::time::Duration::from_secs(10));std::thread::sleep(std::time::Duration::from_millis(5));}
    let cases=[
        ("for", "fn main(){for x in [0,1,2]{} return 0;}",Some(11)),
        ("while", "fn main(){while false{} return 0;}",Some(5)),
        ("expr", "fn main(){return 1+2*3;}",Some(7)),
        ("short", "fn main(){return false && missing();}",Some(4)),
        ("async", "async fn worker(){for x in [0,1,2]{} return 0;} fn main(){tasks{let a=spawn worker();let b=spawn worker();let x=await a;let y=await b;}return 0;}",Some(40)),
        ("records", "fn main(){let rows=collect{for x in [0,1,2]{yield {x:x};}};for part in rows size 2{} return 0;}",Some(32)),
        ("flow", "fn helper(){tasks{for x in [0,1,2]{if x==0{continue;}if x==1{break;}}return 7;}} fn main(){let x=helper();assert x==7;tasks{}return x;}",None),
        ("async_helper", "fn helper(){for x in [0,1,2]{}return 0;}async fn worker(){return helper();}fn main(){tasks{let a=spawn worker();let b=spawn worker();let x=await a;let y=await b;}return 0;}",None),
    ];
    for (name,body,expected) in cases {
        let dir=root.join(name);std::fs::create_dir(&dir).unwrap();let source=dir.join("model.wgt");let binary=dir.join("program");
        std::fs::write(&source,format!("wgt 0.5\n{body}")).unwrap();
        weightlang_wgt::script_compiler::compile(&source,&binary).unwrap();std::fs::remove_file(&source).unwrap();
        let run=|budget:u64|std::process::Command::new(&binary).arg("--max-operations").arg(budget.to_string()).env_clear().env("PATH","/nonexistent").env("WGTIME_ENDPOINT",&endpoint).output().unwrap();
        let mut threshold=None;
        for budget in 0..200 {let out=run(budget);if out.status.success(){threshold=Some(budget);break;}let error=String::from_utf8_lossy(&out.stderr);assert!(error.contains("budget"),"{name} budget {budget}: {error}");}
        let threshold=threshold.expect("bounded example must complete");
        if let Some(expected)=expected{assert_eq!(threshold,expected,"{name}");}
        assert!(run(threshold).status.success(),"cleanup after exhausted tasks: {name}");
        assert!(!run(threshold-1).status.success(),"{name} bypassed budget");
        println!("client-budget {name} minimum={threshold}");
    }
}
use std::{fs,path::PathBuf,process::{Child,Command,Stdio},time::{Duration,Instant}};
use std::os::unix::fs::PermissionsExt;
struct Daemon{child:Child,root:PathBuf}
impl Drop for Daemon{fn drop(&mut self){let _=self.child.kill();let _=self.child.wait();let _=fs::remove_dir_all(&self.root);}}
#[test]
#[ignore="requires WGTIME_TEST_DAEMON pointing at a prebuilt service; explicit local end-to-end check"]
fn default_client_records_and_persistence_without_source(){
    let root=PathBuf::from("/tmp/weightlang").join(format!("client-regression-{}",std::process::id()));
    fs::create_dir_all(&root).unwrap();fs::set_permissions(&root,fs::Permissions::from_mode(0o700)).unwrap();
    let child=Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(&root).stdout(Stdio::null()).spawn().unwrap();
    let daemon=Daemon{child,root:root.clone()};let endpoint=root.join("service.sock");let start=Instant::now();while !endpoint.exists(){assert!(start.elapsed()<Duration::from_secs(10));std::thread::sleep(Duration::from_millis(5));}
    let source=root.join("model.wgt");let binary=root.join("program");
    fs::write(&source,"wgt 0.5\nfn main(){let rows=collect{for i in range(0,5){yield {input:[i],target:[i*2]};}};let count=0;for part in rows size 2{count=count+len(part);}assert count==5;assert rows[2].input==[2];let net=mlp([1,2,1],\"tanh\",42,\"adam\");train net on [[1],[2]] against [[2],[4]] {rate:0.01;loss:mse;};save checkpoint net to \"state.json\";let resumed=load checkpoint from \"state.json\";let a=train net on [[1],[2]] against [[2],[4]] {rate:0.01;loss:mse;};let b=train resumed on [[1],[2]] against [[2],[4]] {rate:0.01;loss:mse;};assert a==b;assert net([3])==resumed([3]);save checkpoint resumed to \"state.json\" overwrite;return rows.target;}").unwrap();
    weightlang_wgt::script_compiler::compile(&source,&binary).unwrap();fs::remove_file(&source).unwrap();
    let run=||Command::new(&binary).env_clear().env("PATH","/nonexistent").env("WGTIME_ENDPOINT",&endpoint).output().unwrap();
    let first=run();assert!(first.status.success(),"{}",String::from_utf8_lossy(&first.stderr));
    let result:serde_json::Value=serde_json::from_slice(&first.stdout).unwrap();assert_eq!(result["result"],serde_json::json!([[0.],[2.],[4.],[6.],[8.]]));
    let before=fs::read(root.join("state.json")).unwrap();let second=run();assert!(!second.status.success());assert_eq!(fs::read(root.join("state.json")).unwrap(),before);
    assert_eq!(fs::read_dir(&root).unwrap().count(),4,"temporary files leaked");
    drop(daemon);
}
