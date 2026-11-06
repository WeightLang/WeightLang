use std::path::Path;
use wgtime_core::script;
const OLD: &str=include_str!("fixtures/ppo_legacy/model.wgt");
const NEW: &str = r#"wgt 0.5
model Policy
input observation: f32[2]
output logits: logits[2]
parameter weights: f32[2,2] = [0,0,0,0]
target trajectory: f32[4]
fn forward(observation) { return linear(observation,weights); }
fn loss(prediction,trajectory) { let lp=log_softmax(prediction), selected=sum(lp*slice(trajectory,0,2)), old=detach(slice(trajectory,2,1)), advantage=detach(slice(trajectory,3,1)), ratio=exp(selected-old), surrogate=minimum(ratio*advantage,clip(ratio,0.8,1.2)*advantage), entropy=sum(softmax(prediction)*lp); return -surrogate+[0.01]*entropy; }
fn environment(first,second) { if first==1 && second==1 { return 1; } return 0; }
fn main() {
 seed(7); let policy=Policy("adam"), critic=mlp([2,8,1],"tanh",19,"adam");
 let rollout=collect {
  for episode in range(0,24) {
   let x0=[1,0], x1=[0,1];
   let {action:a0,log_prob:lp0}=sample from policy(x0);
   let {action:a1,log_prob:lp1}=sample from policy(x1);
   let v0=critic(x0)[0], v1=critic(x1)[0], reward=environment(a0,a1), advantages=gae([0,reward],[v0,v1],[v1,0],[false,true],[false,false],0.99,0.95);
   yield {observation:x0,target:concat(onehot[2] a0,[lp0,advantages[0]]),value_target:[advantages[0]+v0]};
   yield {observation:x1,target:concat(onehot[2] a1,[lp1,advantages[1]]),value_target:[advantages[1]+v1]};
  }
 };
 return [rollout.observation,rollout.target,rollout.value_target,random()];
}
"#;
#[test]
fn collected_samples_match_legacy_rows_and_old_policy_log_probabilities() {
 let old=OLD.replace("# 固定轨迹做多轮小批次更新；不重新计算旧 log_prob 或 advantage。", "return [observations,targets,returns,random()];");
 let old=script::run(&old,Path::new("."),100000).unwrap().result;
 let new=script::run(NEW,Path::new("."),100000).unwrap().result;
 assert_eq!(old,new);
 assert_eq!(old[0].as_array().unwrap().len(),48);
 // The first two target entries encode the action one-hot; entry 2 is
 // log_prob, entry 3 the unchanged GAE advantage. Equality covers all rows.
 assert_eq!(old[1].as_array().unwrap().len(),48);
}
