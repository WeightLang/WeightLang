use std::path::Path;
use weightlang_runtime::script;
const OLD: &str=include_str!("fixtures/ppo_legacy/model.wgt");
const NEW: &str=include_str!("../../../examples/native_training/ppo/model.wgt");

#[test]
fn record_ppo_preserves_seed_rollout_and_every_training_update() {
 let old=OLD.replace("for iteration in range(0, 40) {", "let trace=[]; for iteration in range(0, 40) {")
  .replace("train policy on (xs, ys) rate 0.03 clip 0.5;", "train policy on (xs, ys) rate 0.03 clip 0.5; trace=append(trace,concat(policy([1,0]),policy([0,1])));")
  .replace("fit(critic, observations, returns, 0.03, \"mse\");", "trace=append(trace,[len(observations),len(targets),random()]); fit(critic, observations, returns, 0.03, \"mse\");")
  .replace("return success;", "return [success,trace];");
 let new=NEW.replace("for iteration in range(0, 40) {", "let trace=[]; for iteration in range(0, 40) {")
  .replace("train policy on (minibatch.observation, minibatch.target) rate 0.03 clip 0.5;", "train policy on (minibatch.observation, minibatch.target) rate 0.03 clip 0.5; trace=append(trace,concat(policy([1,0]),policy([0,1])));")
  .replace("fit(critic, rollout.observation, rollout.value_target, 0.03, \"mse\");", "trace=append(trace,[len(rollout),len(rollout.target),random()]); fit(critic, rollout.observation, rollout.value_target, 0.03, \"mse\");")
  .replace("return success;", "return [success,trace];");
 let old=script::run(&old,Path::new("."),10_000_000).unwrap();
 let new=script::run(&new,Path::new("."),10_000_000).unwrap();
 assert_eq!(old.result[1].as_array().unwrap().len(),40*13);
 assert_eq!(old.result,new.result,"all 480 updates, row counts and RNG tails must match");
}
