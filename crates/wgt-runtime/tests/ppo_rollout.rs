use std::path::Path;
use weightlang_runtime::script;
const OLD: &str=include_str!("fixtures/ppo_legacy/model.wgt");
const NEW: &str=include_str!("../../../examples/native_training/ppo/model.wgt");
#[test]
fn collected_samples_match_legacy_rows_and_old_policy_log_probabilities() {
 let old=OLD.replace("# 固定轨迹做多轮小批次更新；不重新计算旧 log_prob 或 advantage。", "return [observations,targets,returns,random()];");
 let new=NEW.replace("# 48 行按顺序共同切成三批，每轮四个 epoch 共 12 次更新；尾批也会保留。", "return [rollout.observation,rollout.target,rollout.value_target,random()];");
 let old=script::run(&old,Path::new("."),100000).unwrap().result;
 let new=script::run(&new,Path::new("."),100000).unwrap().result;
 assert_eq!(old,new);
 assert_eq!(old[0].as_array().unwrap().len(),48);
 // The first two target entries encode the action one-hot; entry 2 is
 // log_prob, entry 3 the unchanged GAE advantage. Equality covers all rows.
 assert_eq!(old[1].as_array().unwrap().len(),48);
}
