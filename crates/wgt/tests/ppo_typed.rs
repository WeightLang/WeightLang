use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[test]
fn typed_ppo_aot_preserves_training_and_probability_evaluation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
model Policy
input observation: f32[2]
output logits: logits[2]
parameter weights: f32[2,2] = [0,0,0,0]
target trajectory: f32[4]
fn forward(observation) { return linear(observation, weights); }
fn loss(prediction, trajectory) {
    let log_probs = log_softmax(prediction);
    let selected = sum(log_probs * slice(trajectory, 0, 2));
    let old_log_prob = detach(slice(trajectory, 2, 1));
    let advantage = detach(slice(trajectory, 3, 1));
    let ratio = exp(selected - old_log_prob);
    let surrogate = minimum(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage);
    let negative_entropy = sum(softmax(prediction) * log_probs);
    return -surrogate + [0.01] * negative_entropy;
}
fn environment(first, second) {
    if first == 1 && second == 1 { return 1; }
    return 0;
}
fn main() {
    seed(7);
    let policy = Policy("adam");
    let critic = mlp([2, 8, 1], "tanh", 19, "adam");
    let initial = policy([1,0]).probabilities[1] * policy([0,1]).probabilities[1];
    for iteration in range(0, 40) {
        let rollout = collect {
            for episode in range(0, 24) {
                let x0 = [1,0];
                let x1 = [0,1];
                let {action: a0, log_prob: lp0} = sample from policy(x0);
                let {action: a1, log_prob: lp1} = sample from policy(x1);
                let v0 = critic(x0)[0];
                let v1 = critic(x1)[0];
                let reward = environment(a0, a1);
                let advantages = gae([0,reward], [v0,v1], [v1,0], [false,true], [false,false], 0.99, 0.95);
                yield {observation: x0, target: concat(onehot[2] a0, [lp0, advantages[0]]), value_target: [advantages[0]+v0]};
                yield {observation: x1, target: concat(onehot[2] a1, [lp1, advantages[1]]), value_target: [advantages[1]+v1]};
            }
        };
        for epoch in range(0, 4) {
            for minibatch in rollout size 16 {
                train policy on (minibatch.observation, minibatch.target) rate 0.03 clip 0.5;
            }
        }
        fit(critic, rollout.observation, rollout.value_target, 0.03, "mse");
    }
    let success = policy([1,0]).probabilities[1] * policy([0,1]).probabilities[1];
    metric("initial_success_probability", initial);
    metric("final_success_probability", success);
    assert(success > 0.85, "PPO did not learn delayed reward");
    return success;
}
"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let expected = script::run(text, dir.path(), 10_000_000).unwrap();
    let output = Command::new(&executable).env("PATH", "/nonexistent")
        .arg("--root").arg(dir.path()).args(["--max-operations", "10000000"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = serde_json::to_value(expected).unwrap();
    assert_eq!(actual["operations"], expected["operations"]);
    for key in ["initial_success_probability", "final_success_probability"] {
        assert!((actual["metrics"][key].as_f64().unwrap()-expected["metrics"][key].as_f64().unwrap()).abs()<1e-12);
    }
    assert!((actual["result"].as_f64().unwrap()-expected["result"].as_f64().unwrap()).abs()<1e-12);
    assert!(actual["result"].as_f64().unwrap()>0.85);
}
