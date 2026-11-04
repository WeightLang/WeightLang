use std::path::Path;
use weightlang_runtime::script;
const PPO: &str=include_str!("../../../examples/native_training/ppo/model.wgt");
#[test]
fn typed_ppo_nonuniform_scores_have_explicit_probabilities_and_log_likelihood() {
 let prefix=PPO.split("fn main()").next().unwrap();
 let source=format!(r#"{prefix}
fn main() {{let policy=Policy("adam");
 train policy on ([[1,0],[0,1]],[[0,1,-0.6931471805599453,1],[0,1,-0.6931471805599453,1]]) rate 0.03 clip 0.5;
 let scores=policy([1,0]);let p=scores.probabilities;
 let {{action,log_prob}}=sample from policy([1,0]);return [scores,p,action,log_prob];}}"#);
 let value=script::run(&source,Path::new("."),100000).unwrap().result;
 let z:Vec<_>=value[0].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect();
 assert_ne!(z[0],z[1]);let max=z[0].max(z[1]);let sum=(z[0]-max).exp()+(z[1]-max).exp();
 for i in 0..2 {assert!((value[1][i].as_f64().unwrap()-(z[i]-max).exp()/sum).abs()<1e-12);}
 let action=value[2].as_f64().unwrap() as usize;
 assert!((value[3].as_f64().unwrap()-(z[action]-max-sum.ln())).abs()<1e-12);
}
