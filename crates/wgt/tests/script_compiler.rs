use std::{fs, path::Path, process::Command};
use wgtime_core::script;
use weightlang_wgt::script_compiler;

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires real GPU and native compilation"]
fn compiled_gpu_and_wgsl() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    for (name, text) in [
        (
            "gpu",
            r#"wgt 0.5
model Regression
input x: f32[1]
output y: f32[1]
layer output: linear(out=1)
init seed=42
fn main() {
    let hardware = devices();
    assert(len(hardware) > 1, "a hardware GPU and a --features gpu build are required");
    let model = Regression("adam");
    training_device(model, hardware[1]["id"], 67108864, 4);
    let initial = 0;
    let final = 0;
    for step in range(0, 100) {
        final = fit(model, [[0], [1]], [[1], [3]], 0.05, "mse");
        if step == 0 { initial = final; }
    }
    assert(final < initial * 0.05, "GPU regression did not converge");
    metric("training", training_status(model));
    metric("initial_loss", initial);
    metric("final_loss", final);
    return model([1]);
}
"#,
        ),
        (
            "wgsl",
            r#"wgt 0.5
fn main() {
    let hardware = devices();
    assert(len(hardware) > 1, "a hardware GPU and a --features gpu build are required");
    let source = "@group(0) @binding(0) var<storage, read> x: array<f32>; @group(0) @binding(1) var<storage, read_write> y: array<f32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { if id.x < arrayLength(&x) && id.x < arrayLength(&y) { y[id.x] = x[id.x] * 2.0; } }";
    let kernel = wgsl_kernel(source, "main", hardware[1]["id"], 1048576);
    let result = wgsl_run(kernel, [1, 2, 3, 4], 4, [1, 1, 1]);
    assert(result == [2, 4, 6, 8], "WGSL result mismatch");
    return result;
}
"#,
        ),
    ] {
        let text = text.replace("return model([1]);", r#"
            save checkpoint model to "checkpoint.json";
            let restored = load checkpoint from "checkpoint.json";
            assert training_status(restored)["device"] == "cpu", "restore must select CPU";
            assert training_status(restored)["steps"] == 100, "optimizer step lost";
            assert restored([1]) == model([1]), "checkpoint weight mismatch";
            training_device(restored, hardware[1]["id"], 67108864, 4);
            let a = train model on [[0],[1]] against [[1],[3]] {rate:0.05;loss:mse;};
            let b = train restored on [[0],[1]] against [[1],[3]] {rate:0.05;loss:mse;};
            assert a == b && model([1]) == restored([1]), "Adam continuation mismatch";
            return restored([1]);
        "#);
        let case = dir.path().join(name);
        fs::create_dir(&case).unwrap();
        let source = case.join("model.wgt");
        let executable = case.join("program");
        fs::write(&source, &text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        assert!(
            !case.join("checkpoint.json").exists(),
            "compile executed training"
        );
        fs::remove_file(&source).unwrap();
        let expected = script::run(&text, &case, 100000).unwrap();
        if case.join("checkpoint.json").exists() {
            fs::remove_file(case.join("checkpoint.json")).unwrap();
        }
        let output = Command::new(&executable)
            .env("PATH", "/nonexistent")
            .args(["--max-operations", "100000"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
    }
}

#[test]
fn compiled_and_interpreted_share_semantics_without_source_or_toolchain_at_run_time() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("test.wgt");
    let executable = dir.path().join("compiled");
    let text = r#"wgt 0.5
fn factorial(n) { if n <= 1 { return 1; } return n * factorial(n - 1); }
fn early() { for i in range(0, 5) { while true { return 7; } } }
fn bump(model) { return train model on [[0], [1]] against [[1], [3]] {rate:0.05;loss:mse;}; }
fn main() {
    let reader = jsonl("case.jsonl");
    let rows = batch(reader, 1);
    close(reader);
    let mode = rows[0]["mode"];
    if mode == 1 { return 1 / 0; }
    if mode == 2 { return missing; }
    if mode == 3 { return factorial(100); }
    if mode == 4 { let x = []; return x[0]; }
    if mode == 5 { return false || 2; }
    if mode == 6 { absent = random(); }
    if mode == 7 { return factorial(); }
    assert true || unknown(), "short circuit";
    assert !(false && unknown()), "short circuit";
    let n = 0;
    for i in range(0, 10) {
        if i == 2 { continue; }
        if i == 6 { break; }
        n = n + i;
    }
    while n < 20 { n = n + 1; }
    if true { let scoped = 3; }
    let scoped = scoped + 1;
    seed(17);
    let model = mlp([1,1], "identity", 42, "adam");
    for step in range(0, 100) { bump(model); }
    save model model to "model.json";
    save checkpoint model to "checkpoint.json";
    metric("summary", [n, scoped, early(), factorial(5), random(), "x" + "y"]);
    return model([0.25]);
}
"#;
    fs::write(&source, text).unwrap();
    let code = script::codegen::rust_source(text, ".").unwrap();
    assert!(!code.contains("script::run"));
    assert!(!code.contains("Expr::"));
    assert!(code.contains("while "));
    script_compiler::compile(&source, &executable).unwrap();
    assert!(
        !dir.path().join("model.json").exists(),
        "compiling must not execute training"
    );
    assert!(script_compiler::compile(&source, &executable).is_err());
    fs::remove_file(&source).unwrap();
    for mode in 0..8 {
        let interpreted = dir.path().join(format!("interpreted-{mode}"));
        let compiled = dir.path().join(format!("compiled-{mode}"));
        for root in [&interpreted, &compiled] {
            fs::create_dir(root).unwrap();
            fs::write(root.join("case.jsonl"), format!("{{\"mode\":{mode}}}\n")).unwrap();
        }
        let reference = script::run(text, &interpreted, 100000);
        let actual = Command::new(&executable)
            .env("PATH", "/nonexistent")
            .arg("--root")
            .arg(&compiled)
            .args(["--max-operations", "100000"])
            .output()
            .unwrap();
        if mode == 0 {
            assert!(
                actual.status.success(),
                "{}",
                String::from_utf8_lossy(&actual.stderr)
            );
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            assert_eq!(serde_json::to_value(reference.unwrap()).unwrap(), actual);
            for name in ["model.json", "checkpoint.json"] {
                assert_eq!(
                    fs::read(interpreted.join(name)).unwrap(),
                    fs::read(compiled.join(name)).unwrap()
                );
            }
        } else {
            let error = format!("{:#}", reference.unwrap_err());
            assert!(!actual.status.success());
            let last = error.rsplit(": ").next().unwrap();
            assert!(
                String::from_utf8_lossy(&actual.stderr).contains(last),
                "mode {mode}: {} vs {error}",
                String::from_utf8_lossy(&actual.stderr)
            );
        }
    }
    let limited = Command::new(&executable)
        .args(["--max-operations", "1"])
        .output()
        .unwrap();
    assert!(!limited.status.success());
    assert!(String::from_utf8_lossy(&limited.stderr).contains("budget exceeded"));
}

#[test]
fn compiled_cli_runs_existing_reinforcement_script() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let text = r#"wgt 0.5
fn environment(action) { if action == 1 { return 1; } return 0; }
fn main() {
    seed(7);
    let policy = mlp([1, 2], "identity", 3, "adam");
    let initial = predict(policy, [1]);
    let average = 0;
    for iteration in range(0, 80) {
        let observations = [];
        let actions = [];
        let rewards = [];
        for episode in range(0, 32) {
            let action = sample(policy, [1]);
            observations = append(observations, [1]);
            actions = append(actions, action);
            rewards = append(rewards, environment(action));
        }
        reinforce(policy, observations, actions, rewards, 0.05);
        average = sum(rewards) / len(rewards);
    }
    let final = predict(policy, [1]);
    metric("mean_reward_last_batch", average);
    metric("initial_action_margin", initial[1] - initial[0]);
    metric("final_action_margin", final[1] - final[0]);
    assert(final[1] - final[0] > 3, "policy failed to favor rewarded action");
    return average;
}
"#;
    fs::write(&source, text).unwrap();
    let reference = script::run(text, dir.path(), 10_000_000).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .args(["script"])
        .arg(&source)
        .arg("--compiled")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(serde_json::to_value(reference).unwrap(), actual);
    let result = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .arg("script")
        .arg(&source)
        .args(["--compiled", "--check"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
}

#[test]
fn unified_model_and_forward_match_in_native_executables() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    for (name, text) in [
        ("graph", "wgt 0.5\nmodel Net\ninput x: f32[1]\noutput y: f32[1]\nlayer fc: linear(out=1)\ninit seed=7\nfn main() { let m = Net(\"sgd\"); for i in range(0, 100) { train m on [[0], [1]] against [[1], [3]] {rate:0.05;loss:mse;}; } return m([1]); }"),
        ("forward", r#"wgt 0.5
model AdaptiveMemory
input x: f32[2]
output y: f32[1]
parameter memory: f32[1,1] = [0]
frozen updater: f32[1,1] = [0.5]
frozen router: f32[1,1] = [1]
fn forward(x) {
    let observation = slice(x, 0, 1);
    let query = slice(x, 1, 1);
    let delta = linear(observation, updater);
    let old = read(memory);
    let next = add(old, delta);
    write(memory, next);
    let gate = linear(observation, router);
    if positive(gate) { return linear(query, memory); }
    return query;
}
fn main() {
    let model = AdaptiveMemory("none");
    assert(model([2, 3])[0] == 3, "first state update");
    return model([4, 3]);
}
"#),
        ("working-memory", r#"wgt 0.5
model WorkingMemory
input x: f32[2]
output y: f32[2]
parameter address_logits: f32[2] = [0,0]
frozen first: f32[2] = [1,0]
frozen second: f32[2] = [0,1]
fn forward(x) {
    let bank = memory_alloc(2,2);
    memory_write(bank, first, x);
    memory_write(bank, second, -x);
    let result = memory_read(bank, softmax(address_logits));
    memory_free(bank);
    return result;
}
fn main() {
    compute_budget(100000,65536,10,60000);
    let m = WorkingMemory("adam");
    let inputs = [[1,2],[-1,-2],[2,-1],[-2,1]];
    let first_loss = fit(m,inputs,inputs,0.04,"mse");
    for step in range(0,300) { fit(m,inputs,inputs,0.04,"mse"); }
    let last_loss = fit(m,inputs,inputs,0.04,"mse");
    assert(last_loss < first_loss * 0.01,"memory addressing did not learn");
    let expected = predict(m,[3,-4]);
    assert(mse(expected,[3,-4])<0.05,"held-out memory read failed");
    execution_start(m,[3,-4],1000,65536);
    let progress = execution_resume(m,2);
    while !progress["done"] { progress = execution_resume(m,2); }
    assert(progress["output"]==expected,"resumed execution differs");
    assert(progress["peak_memory_bytes"]==16,"unexpected workspace peak");
    metric("initial_loss",first_loss);
    metric("final_loss",last_loss);
    metric("held_out_mse",mse(expected,[3,-4]));
    metric("execution_work",progress["work"]);
    metric("peak_workspace_bytes",progress["peak_memory_bytes"]);
    return true;
}
"#),
    ] {
        let source = dir.path().join(format!("{name}.wgt"));
        let executable = dir.path().join(name);
        fs::write(&source, text).unwrap();
        let expected = script::run(text, dir.path(), 100_000).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        let output = Command::new(&executable).env("PATH", "/nonexistent")
            .args(["--max-operations", "100000"]).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(actual, serde_json::to_value(expected).unwrap());
    }
}

#[test]
fn compiled_ppo_uses_native_training_syntax() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let source = dir.path().join("model.wgt");
    let executable = dir.path().join("ppo");
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
fn environment(first, second) { if first == 1 && second == 1 { return 1; } return 0; }
fn main() {
    seed(7);
    let policy = Policy("adam");
    let critic = mlp([2, 8, 1], "tanh", 19, "adam");
    let initial = policy([1,0]).probabilities[1] * policy([0,1]).probabilities[1];
    for iteration in range(0, 40) {
        let rollout = collect {
            for episode in range(0, 24) {
                let x0 = [1,0]; let x1 = [0,1];
                let {action: a0, log_prob: lp0} = sample from policy(x0);
                let {action: a1, log_prob: lp1} = sample from policy(x1);
                let v0 = critic(x0)[0]; let v1 = critic(x1)[0];
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
    let expected = script::run(text, dir.path(), 10_000_000).unwrap();
    script_compiler::compile(&source, &executable).unwrap();
    fs::remove_file(source).unwrap();
    let output = Command::new(&executable)
        .env("PATH", "/nonexistent")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = serde_json::to_value(expected).unwrap();
    // JSON parsing and optimized host f64 arithmetic may differ by one ULP.
    // Dispatch counts and the initial baseline must still match exactly.
    assert_eq!(actual["operations"], expected["operations"]);
    assert_eq!(
        actual["metrics"]["initial_success_probability"],
        expected["metrics"]["initial_success_probability"]
    );
    for (a, b) in [
        (&actual["result"], &expected["result"]),
        (
            &actual["metrics"]["final_success_probability"],
            &expected["metrics"]["final_success_probability"],
        ),
    ] {
        assert!(
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-12,
            "{a} != {b}"
        );
    }
}
