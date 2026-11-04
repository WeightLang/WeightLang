use std::path::Path;
use weightlang_runtime::script;
const MODEL: &str = "wgt 0.5\nmodel Regression\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1]=[0]\nfn forward(x){return x*w;}\nfn loss(prediction,target){return square(prediction-target).mean;}\n";
fn run(body:&str)->anyhow::Result<script::Report>{script::run(&format!("{MODEL}\n{body}"),Path::new("."),100000)}
#[test]
fn local_model_call_is_not_removed_loader() {
 run("fn main(){let model=Regression(\"none\");return model([1]);}").unwrap();
 let error=run("fn main(){return model(\"missing.json\",\"sgd\");}").unwrap_err();
 assert!(format!("{error:#}").contains("model(...) loading was removed"));
}
#[test]
fn unified_builtin_custom_and_clip(){
 let out=run("fn main(){let m=Regression(\"sgd\");let loss=train m on [[1],[2]] against [[2],[4]] {rate:0.1;loss:loss;clip:1;};assert loss==10;assert m([1])[0]>0.099;assert m([1])[0]<0.101;return loss;}").unwrap();assert_eq!(out.result,serde_json::json!(10.));
 let out=run("fn main(){let m=mlp([1,1],\"identity\",42,\"sgd\");let loss=train m on [[1]] against [[2]] {rate:0.01;clip:1;};return loss;}").unwrap();assert!(out.result.as_f64().unwrap().is_finite());
}
#[test]
fn clipped_updates_and_invalid_options_are_atomic() {
 use weightlang_runtime::training::{mlp,Optimizer,Trainer,Objective};
 let mut trainer=Trainer::new(mlp(&[1,1],"identity",42).unwrap(),Path::new("."),1024*1024,Optimizer::Adam).unwrap();
 let inputs=vec![vec![1.]];let targets=vec![vec![10.]];
 let before=trainer.checkpoint_bytes_bounded(1024*1024).unwrap();
 for (rate,clip) in [(0.,Some(1.)),(0.1,Some(0.)),(0.1,Some(f32::NAN))] {
  assert!(trainer.step_clipped(&inputs,Objective::Mse(&targets),rate,clip).is_err());
  assert_eq!(before,trainer.checkpoint_bytes_bounded(1024*1024).unwrap());
 }
 let mut sgd=Trainer::new(mlp(&[1,1],"identity",42).unwrap(),Path::new("."),1024*1024,Optimizer::Sgd).unwrap();
 let output=sgd.predict(&[1.]).unwrap()[0];
 sgd.step_clipped(&inputs,Objective::Mse(&targets),0.1,Some(1.)).unwrap();
 let change=(sgd.predict(&[1.]).unwrap()[0]-output).abs();
 assert!((change-0.1*2f32.sqrt()).abs()<1e-5,"global two-parameter gradient norm: {change}");
}
#[test]
fn train_options_and_old_forms_reject(){
 for source in ["fit(m,[],[],0.1,\"mse\")","train(m,[],[],0.1,1)","train m on ([],[]) rate 0.1 clip 1","train m on [] against [] {rate:1;rate:2;}","train m on [] against [] {loss:mse;}","train m on [] against [] {rate:1;unknown:2;}"] {assert!(script::parse(&format!("wgt 0.5\nfn main(){{{source};}}")).is_err(),"{source}");}
}
#[test]
fn options_once_in_source_order_and_strategy_return(){
 run("fn mark(value,expected){assert random()==expected;return value;}fn main(){seed(7);let a=random();let b=random();let c=random();let d=random();seed(7);let m=Regression(\"sgd\");let value=train mark(m,a) on mark([[1]],b) against mark([[2]],c) {rate:mark(0.1,d);loss:loss;};assert value==4;return value;}").unwrap();
 let out=run("fn noop(net,x,y){return 17;}fn main(){let m=Regression(\"none\");let metrics=train m on [[1]] against [[2]] using noop;assert m([1])==[0];return metrics;}").unwrap();assert_eq!(out.result,serde_json::json!(17.));
}
