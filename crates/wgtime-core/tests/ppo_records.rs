use std::path::Path;
use wgtime_core::script;
const OLD: &str=include_str!("fixtures/ppo_legacy/model.wgt");
const NEW: &str = r#"wgt 0.5
model Policy
input observation: f32[2]
output logits: f32[2]
parameter weights: f32[2,2] = [0,0,0,0]
target trajectory: f32[4]
fn forward(observation) { return linear(observation,weights); }
fn loss(prediction,trajectory) { let lp=log_softmax(prediction), selected=sum(lp*slice(trajectory,0,2)), old=detach(slice(trajectory,2,1)), advantage=detach(slice(trajectory,3,1)), ratio=exp(selected-old), surrogate=minimum(ratio*advantage,clip(ratio,0.8,1.2)*advantage), entropy=sum(softmax(prediction)*lp); return -surrogate+[0.01]*entropy; }
fn environment(first,second) { if first==1 && second==1 { return 1; } return 0; }
fn main() { seed(7); let policy=Policy("adam"), critic=mlp([2,8,1],"tanh",19,"adam"); let initial=softmax(policy([1,0]))[1]*softmax(policy([0,1]))[1]; for iteration in range(0, 40) { let rollout=collect { for episode in range(0,24) { let x0=[1,0],x1=[0,1]; let {action:a0,log_prob:lp0}=sample from policy(x0); let {action:a1,log_prob:lp1}=sample from policy(x1); let v0=critic(x0)[0],v1=critic(x1)[0],reward=environment(a0,a1),advantages=gae([0,reward],[v0,v1],[v1,0],[false,true],[false,false],0.99,0.95); yield {observation:x0,target:concat(onehot[2] a0,[lp0,advantages[0]]),value_target:[advantages[0]+v0]}; yield {observation:x1,target:concat(onehot[2] a1,[lp1,advantages[1]]),value_target:[advantages[1]+v1]}; } }; for epoch in range(0,4) { for minibatch in rollout size 16 { train policy on minibatch.observation against minibatch.target {rate:0.03;clip:0.5;loss:loss;}; } } train critic on rollout.observation against rollout.value_target {rate:0.03;loss:mse;}; } let success=softmax(policy([1,0]))[1]*softmax(policy([0,1]))[1]; assert success>0.85; return success; }
"#;

#[test]
fn record_ppo_preserves_seed_rollout_and_every_training_update() {
 let old=OLD.replace("for iteration in range(0, 40) {", "let trace=[]; for iteration in range(0, 40) {")
  .replace("train policy on xs against ys {rate:0.03;clip:0.5;loss:loss;};", "train policy on xs against ys {rate:0.03;clip:0.5;loss:loss;}; trace=append(trace,concat(policy([1,0]),policy([0,1])));")
  .replace("train critic on observations against returns {rate:0.03;loss:mse;};", "trace=append(trace,[len(observations),len(targets),random()]); train critic on observations against returns {rate:0.03;loss:mse;};")
  .replace("return success;", "return [success,trace];");
 // Both programs use legacy f32 logits so the comparison isolates record
 // collection; typed logits intentionally use different sampling arithmetic.
 let new=NEW.replace("for iteration in range(0, 40) {", "let trace=[]; for iteration in range(0, 40) {")
  .replace("train policy on minibatch.observation against minibatch.target {rate:0.03;clip:0.5;loss:loss;};", "train policy on minibatch.observation against minibatch.target {rate:0.03;clip:0.5;loss:loss;}; trace=append(trace,concat(policy([1,0]),policy([0,1])));")
  .replace("train critic on rollout.observation against rollout.value_target {rate:0.03;loss:mse;};", "trace=append(trace,[len(rollout),len(rollout.target),random()]); train critic on rollout.observation against rollout.value_target {rate:0.03;loss:mse;};")
  .replace("return success;", "return [success,trace];");
 let old=script::run(&old,Path::new("."),10_000_000).unwrap();
 let new=script::run(&new,Path::new("."),10_000_000).unwrap();
 assert_eq!(old.result[1].as_array().unwrap().len(),40*13);
 assert_eq!(old.result,new.result,"all 480 updates, row counts and RNG tails must match");
}
