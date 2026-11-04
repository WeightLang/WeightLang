use std::path::Path;
use wgtime_core::script;
const DECL:&str=r#"wgt 0.5
model Policy
input x:f32[1]
output y:logits[2]
parameter w:f32[2,1]=[0,0]
fn forward(x){return linear(x,w);}
fn loss(p,t){return mean(square(p-t));}
agent Actor {policy;learner:update;}
environment Counter {observation:f32[1];action:f32[1];feedback:f32[1];reset:reset_counter;step:step_counter;}
fn reset_counter(state,seed){return {state:state,observation:[state]};}
fn step_counter(state,action){let next=state+action[0];return {state:next,observation:[next],feedback:[next],terminated:next>=2,truncated:false};}
fn update(actor,batch){let policy=actor.policy;train policy on batch.x against batch.y {rate:0.1;clip:1;loss:loss;};return len(batch);}
"#;
#[test]
fn native_agent_default_learner_and_environment_instances_run() {
 let source=format!(r#"{DECL}
fn main(){{let policy=Policy("sgd");let actor=Actor(policy);let env=Counter(0,7);let other=Counter(10,8);
 assert env.observation==[0], "constructor reset";assert other.observation==[10], "instance isolation";
 let first=env.step([1]);assert !first.terminated && !first.truncated, "ongoing";let second=env.step([1]);assert second.terminated, "done";
 assert env.reset()==[0], "reset";let action=actor.act([1]);assert action>=0 && action<2, "act";
 let report=learn actor {{yield {{x:[1],y:[1,0]}};}};return [report,policy([1]),other.observation];}}"#);
 let result=script::run(&source,Path::new("."),100000).unwrap().result;
 assert_eq!(result[0],serde_json::json!(1.));assert_eq!(result[2],serde_json::json!([10.]));
 assert!(result[1][0].as_f64().unwrap()>0.);
}
#[test]
fn environment_step_after_done_requires_reset() {
 let source=format!("{DECL}fn main(){{let env=Counter(0,7);env.step([2]);return env.step([1]);}}");
 assert!(format!("{:#}",script::run(&source,Path::new("."),100000).unwrap_err()).contains("requires reset"));
}
#[test]
fn missing_default_learner_does_not_guess_algorithm() {
 let source=DECL.replace("agent Actor {policy;learner:update;}","agent Actor {policy;}");
 let source=format!("{source}fn main(){{let actor=Actor(Policy(\"sgd\"));learn actor {{yield {{x:[1],y:[1,0]}};}}}}");
 assert!(format!("{:#}",script::run(&source,Path::new("."),100000).unwrap_err()).contains("no default learner"));
}

#[test]
fn environment_rng_is_independent_and_reset_repeats_the_seed() {
 let source=DECL.replace("return {state:state,observation:[state]};","return {state:state,observation:[random()]};")
 .replace("let next=state+action[0];","let next=state+random();");
 let source=format!(r#"{source}
fn main(){{seed(91);let host=random();seed(91);let a=Counter(0,7);let b=Counter(0,7);
 assert random()==host, "host RNG changed";assert a.observation==b.observation, "same seed reset";
 let first=a.observation;let x=a.step([0]);let y=b.step([0]);assert x==y, "instance streams";
 assert a.reset()==first, "reset seed repeat";return true;}}"#);
 assert_eq!(script::run(&source,Path::new("."),100000).unwrap().result,true);
}

#[test]
fn declaration_names_schema_and_handler_arity_are_checked() {
 for declaration in [
  "agent Actor {critic;}",
  "agent Actor {policy;policy;}",
  "environment Counter {observation:f32[0];action:f32[1];feedback:f32[1];reset:reset_counter;step:step_counter;}",
 ] {
  let source=format!("wgt 0.5\n{declaration}\nfn main(){{return 0;}}");
  assert!(script::parse(&source).is_err(),"{declaration}");
 }
 let source=DECL.replace("fn reset_counter(state,seed)","fn reset_counter(state)");
 let source=format!("{source}fn main(){{return Counter(0,7);}}");
 assert!(script::run(&source,Path::new("."),100000).is_err());
}

#[test]
fn async_handlers_and_learners_are_rejected() {
 for (old,new) in [("fn update(actor,batch)","async fn update(actor,batch)"),("fn reset_counter(state,seed)","async fn reset_counter(state,seed)"),("fn step_counter(state,action)","async fn step_counter(state,action)")] {
  let source=DECL.replace(old,new);
  let source=format!("{source}fn main(){{let actor=Actor(Policy(\"sgd\"));let env=Counter(0,7);learn actor {{}}}}");
  assert!(format!("{:#}",script::run(&source,Path::new("."),100000).unwrap_err()).contains("synchronous"));
 }
}
