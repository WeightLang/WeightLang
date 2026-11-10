mod support;

use std::{fs, process::Command};
use weightlang_wgt::script_compiler;
use wgtime_core::script;

#[test]
fn client_entry_keeps_helpers_shared_with_resident_strategies() {
    let directory = support::scratch().unwrap();
    let source = directory.path().join("model.wgt");
    let executable = directory.path().join("program");
    let text = r#"wgt 0.5
model Identity
input x:f32[1]
output y:f32[1]
fn forward(x){return x;}
fn shared(x){return x+1;}
fn condition(x){return x==3;}
fn samples(){return [[2]];}
fn resident_helper(x){if x>0{return shared(x);}return 0;}
fn strategy(model,inputs,targets){return resident_helper(inputs[0][0]);}
async fn next(x){return x+1;}
async fn nested(x){return await next(shared(x));}
fn main(){return [1,2,3][1:];}
fn chosen(){
    let model=Identity("none");
    let boxed={value:shared(2)};
    assert condition(boxed.value),"client helper was omitted";
    let asynchronous=0;
    if condition(boxed.value){tasks{let job=spawn nested(boxed.value);asynchronous=await job;}}
    let result=train model on samples() against [[0]] using strategy;
    return [boxed.value,result,asynchronous];
}
"#;
    fs::write(&source, text).unwrap();
    script_compiler::compile_text_entry(text, directory.path(), &executable, "chosen").unwrap();
    fs::remove_file(source).unwrap();
    let service = support::Service::start().unwrap();
    let output = service
        .configure(Command::new(executable).env("PATH", "/nonexistent"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["result"], serde_json::json!([3., 3., 5.]));
}

#[test]
fn compiled_control_flow_training_and_persistence_need_no_source_or_toolchain() {
    let directory = support::scratch().unwrap();
    let text = r#"wgt 0.5
fn factorial(n){if n<=1{return 1;}return n*factorial(n-1);}
fn early(){for i in range(0,5){while true{return 7;}}}
fn bump(model){return train model on [[0],[1]] against [[1],[3]] {rate:0.05;loss:mse;};}
fn main(){
 let mode=MODE;
 if mode==1{return 1/0;}
 if mode==2{return missing;}
 if mode==3{let empty=[];return empty[0];}
 if mode==4{return false||2;}
 if mode==5{absent=3;}
 if mode==6{return factorial();}
 if mode==7{return true&&2;}
 assert true||unknown(),"short circuit";
 assert !(false&&unknown()),"short circuit";
 let n=0;
 for i in range(0,10){if i==2{continue;}if i==6{break;}n=n+i;}
 while n<20{n=n+1;}
 if true{let scoped=3;}
 let scoped=scoped+1;
 let model=mlp([1,1],"identity",42,"adam");
 for step in range(0,100){bump(model);}
 save model model to "model.json";
 save checkpoint model to "checkpoint.json";
 metric("summary",[n,scoped,early(),factorial(5)]);
 return model([0.25]);
}
"#;
    let mut cases = Vec::new();
    for mode in 0..8 {
        let text = text.replace("MODE", &mode.to_string());
        let interpreted = directory.path().join(format!("interpreted-{mode}"));
        let compiled = directory.path().join(format!("compiled-{mode}"));
        fs::create_dir(&interpreted).unwrap();
        fs::create_dir(&compiled).unwrap();
        let source = compiled.join("model.wgt");
        let executable = compiled.join("program");
        fs::write(&source, &text).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        assert!(
            !compiled.join("model.json").exists(),
            "compilation executed training"
        );
        assert!(
            script_compiler::compile(&source, &executable).is_err(),
            "compiler replaced an executable"
        );
        fs::remove_file(source).unwrap();
        let reference = script::run(&text, &interpreted, 100000);
        cases.push((mode, interpreted, compiled, executable, reference));
    }
    let service = support::Service::start().unwrap();
    for (mode, interpreted, compiled, executable, reference) in cases {
        let actual = service
            .configure(
                Command::new(&executable)
                    .env("PATH", "/nonexistent")
                    .arg("--root")
                    .arg(&compiled)
                    .args(["--max-operations", "100000"]),
            )
            .output()
            .unwrap();
        if mode == 0 {
            assert!(
                actual.status.success(),
                "{}",
                String::from_utf8_lossy(&actual.stderr)
            );
            let actual: serde_json::Value = serde_json::from_slice(&actual.stdout).unwrap();
            let reference = reference.unwrap();
            assert_eq!(
                actual["result"][0].as_f64().unwrap() as f32,
                reference.result[0].as_f64().unwrap() as f32
            );
            assert_eq!(
                actual["metrics"],
                serde_json::to_value(reference.metrics).unwrap()
            );
            assert_eq!(
                actual["metrics"]["summary"],
                serde_json::json!([20., 4., 7., 120.])
            );
            for name in ["model.json", "checkpoint.json"] {
                assert_eq!(
                    fs::read(interpreted.join(name)).unwrap(),
                    fs::read(compiled.join(name)).unwrap()
                );
            }
            let limited = service
                .configure(Command::new(&executable).args(["--max-operations", "1"]))
                .output()
                .unwrap();
            assert!(!limited.status.success());
            assert!(String::from_utf8_lossy(&limited.stderr).contains("operation budget exceeded"));
        } else {
            assert!(
                reference.is_err(),
                "invalid mode {mode} executed in the core"
            );
            assert!(
                !actual.status.success(),
                "invalid mode {mode} executed in the client"
            );
            let error = String::from_utf8_lossy(&actual.stderr);
            let expected = [
                "",
                "nonfinite arithmetic",
                "undefined variable",
                "index out of bounds",
                "expected bool",
                "undefined variable",
                "wrong argument count",
                "expected bool",
            ][mode];
            assert!(error.contains(expected), "mode {mode}: {error}");
        }
    }
}

#[test]
fn reinforcement_retains_core_learning_threshold() {
    let directory = support::scratch().unwrap();
    let text = r#"wgt 0.5
fn environment(action) { if action == 1 { return 1; } return 0; }
fn main() {
    seed(7);
    let policy = mlp([1, 2], "identity", 3, "adam");
    let initial = policy([1]);
    let average = 0;
    for iteration in range(0, 80) {
        let observations = [];
        let actions = [];
        let rewards = [];
        for episode in range(0, 32) {
            let action = sample from policy([1]);
            observations = append(observations, [1]);
            actions = append(actions, action);
            rewards = append(rewards, environment(action));
        }
        reinforce(policy, observations, actions, rewards, 0.05);
        average = sum(rewards) / len(rewards);
    }
    let final = policy([1]);
    metric("mean_reward_last_batch", average);
    metric("initial_action_margin", initial[1] - initial[0]);
    metric("final_action_margin", final[1] - final[0]);
    assert final[1] - final[0] > 3, "policy failed to favor rewarded action";
    return average;
}
"#;
    let report = script::run(text, directory.path(), 10_000_000).unwrap();
    assert!(report.metrics["final_action_margin"].as_f64().unwrap() > 3.);
    assert_eq!(report.result, report.metrics["mean_reward_last_batch"]);
}

#[test]
fn unified_model_and_forward_match_in_native_executables() {
    let directory = support::scratch().unwrap();
    let mut cases = Vec::new();
    for (name, text) in [
        (
            "graph",
            "wgt 0.5\nmodel Net\ninput x:f32[1]\noutput y:f32[1]\nlayer fc:linear(out=1)\ninit weights=xavier_uniform(7) bias=zeros()\nfn main(){let m=Net(\"sgd\");for i in range(0,100){train m on [[0],[1]] against [[1],[3]] {rate:0.05;loss:mse;};}return m([1]);}",
        ),
        (
            "forward",
            r#"wgt 0.5
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
    assert model([2, 3])[0] == 3, "first state update";
    return model([4, 3]);
}
"#,
        ),
    ] {
        let root = directory.path().join(name);
        fs::create_dir(&root).unwrap();
        let source = root.join("model.wgt");
        let executable = root.join("program");
        fs::write(&source, text).unwrap();
        let expected = script::run(text, &root, 100_000).unwrap();
        script_compiler::compile(&source, &executable).unwrap();
        fs::remove_file(source).unwrap();
        cases.push((executable, expected));
    }
    let service = support::Service::start().unwrap();
    for (executable, expected) in cases {
        let output = service
            .configure(
                Command::new(executable)
                    .env("PATH", "/nonexistent")
                    .args(["--max-operations", "100000"]),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            actual["result"][0].as_f64().unwrap() as f32,
            expected.result[0].as_f64().unwrap() as f32
        );
        assert_eq!(
            actual["metrics"],
            serde_json::to_value(expected.metrics).unwrap()
        );
    }
}
