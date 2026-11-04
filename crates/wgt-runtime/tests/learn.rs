use std::path::Path;
use weightlang_runtime::script;
const MODEL:&str="wgt 0.5\nmodel Policy\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1] = [1]\nfn forward(x){return w*x;}\nfn loss(p,t){return sum(square(p-t));}\n";
#[test]
fn learner_runs_once_after_full_collection_and_unlocks_before_training() {
 let source=format!(r#"{MODEL}
fn update(model,batch) {{assert len(batch)==3, "complete batch";train model on batch.x against batch.y {{rate:0.1;clip:1;loss:loss;}};return len(batch);}}
fn main() {{let model=Policy("sgd");let alias=model;
 let report=learn model using update {{for i in [1,2,3] {{yield {{x:[i],y:[0]}};}}}};
 assert alias([1])[0]<1, "learner updated same model";return report;}}"#);
 assert_eq!(script::run(&source,Path::new("."),100000).unwrap().result,serde_json::json!(3.));
}
#[test]
fn return_skips_learner_and_releases_identity_lock() {
 let source=format!(r#"{MODEL}
fn forbidden(model,batch) {{assert false, "learner must not run";}}
fn collect(model) {{learn model using forbidden {{return 7;}} return 0;}}
fn main() {{let model=Policy("sgd");assert collect(model)==7, "return propagation";train model on [[1]] against [[0]] {{rate:0.1;clip:1;loss:loss;}};return model([1])[0];}}"#);
 assert!(script::run(&source,Path::new("."),100000).unwrap().result.as_f64().unwrap()<1.);
}
#[test]
fn alias_mutation_is_rejected_before_update_but_other_model_is_allowed() {
 let bad=format!(r#"{MODEL}
fn update(m,b) {{return 0;}}
fn main(){{let m=Policy("sgd");let alias=m;learn m using update {{train alias on [[1]] against [[0]] {{rate:0.1;clip:1;loss:loss;}};}}}}"#);
 assert!(format!("{:#}",script::run(&bad,Path::new("."),100000).unwrap_err()).contains("mutation is forbidden"));
 let good=bad.replace("let alias=m;","let alias=Policy(\"sgd\");");
 script::run(&good,Path::new("."),100000).unwrap();
}
