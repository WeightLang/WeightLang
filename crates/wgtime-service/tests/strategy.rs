use wgtime_core::adaptive::{parse, AdaptiveSession};
use wgtime_service::strategy::Strategy;
const MODEL: &str = "wgt 0.5\nmodel Regression\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1,1] = [0]\nfn forward(x) { return linear(x,w); }\nfn loss(prediction,target) { return square(prediction-target).mean; }\n";
const STEP: &str = "fn step(net,inputs,targets) { let g=differentiate(net,inputs,targets); let s=gradient_snapshot(g); return commit_gradients(g, [[s[\"parameters\"][0][0]-0.1*s[\"gradients\"][0][0]]], [1]); }";
const SAM: &str = r#"wgt 0.5
model Regression
input x: f32[1]
output y: f32[1]
parameter w: f32[1,1] = [0]
fn forward(x) { return linear(x, w); }
fn loss(prediction, target) { return square(prediction-target).mean; }
fn sam_step(net, inputs, targets, rate, rho, beta) {
    let first = differentiate(net, inputs, targets);
    let original = gradient_snapshot(first);
    let squared_norm = 0;
    for gradient in original["gradients"] {
        for g in gradient { squared_norm = squared_norm + g*g; }
    }
    let scale = rho / (sqrt(squared_norm) + 0.00000001);
    let perturbed = [];
    for p in range(0, len(original["parameters"])) {
        let values = [];
        for j in range(0, len(original["parameters"][p])) {
            values = append(values, original["parameters"][p][j] + scale*original["gradients"][p][j]);
        }
        perturbed = append(perturbed, values);
    }
    let second = differentiate_at(first, perturbed);
    let sharp = gradient_snapshot(second);
    let previous = [];
    if sharp["state"] != null {
        assert sharp["state"][0] == 1, "SAM momentum state version";
        previous = sharp["state"][1];
    }
    let candidates = [];
    let velocities = [];
    for p in range(0, len(sharp["parameters"])) {
        let values = [];
        let velocity = [];
        for j in range(0, len(sharp["parameters"][p])) {
            let old = 0;
            if sharp["state"] != null { old = previous[p][j]; }
            let v = beta*old + sharp["gradients"][p][j];
            values = append(values, sharp["parameters"][p][j] - rate*v);
            velocity = append(velocity, v);
        }
        candidates = append(candidates, values);
        velocities = append(velocities, velocity);
    }
    release_gradient(first);
    return commit_gradients(second, candidates, [1, velocities]);
}
fn main() {
    let net = Regression("none");
    let inputs = [[1], [2]];
    let targets = [[2], [4]];
    sam_step(net, inputs, targets, 0.1, 0.1, 0);
    let g = differentiate(net, inputs, targets);
    let s = gradient_snapshot(g);
    assert s["steps"] == 1, "SAM commits once";
    let error = s["parameters"][0][0] - 1.05;
    assert error*error < 0.000000000001, "SAM updates original base";
    release_gradient(g);
    save checkpoint net to "../../../weights/custom_sam/step1.json";
    let resumed = load checkpoint from "../../../weights/custom_sam/step1.json";
    for epoch in range(0, 5) {
        sam_step(net, inputs, targets, 0.1, 0.1, 0.9);
        sam_step(resumed, inputs, targets, 0.1, 0.1, 0.9);
    }
    assert net([3]) == resumed([3]), "SAM checkpoint resume";
    return net([3]);
}
"#;
fn session() -> AdaptiveSession {
    AdaptiveSession::prepare(parse(MODEL).unwrap(), 1024 * 1024).unwrap()
}
fn source(body: &str) -> String {
    format!("{MODEL}\n{STEP}\nfn strategy(net,inputs,targets){{{body}}}\nfn main(){{return null;}}")
}
fn execute(s: &Strategy, m: &mut AdaptiveSession) -> anyhow::Result<serde_json::Value> {
    s.execute(
        m,
        &[vec![1.], vec![2.]],
        &[vec![2.], vec![4.]],
        &mut 1_000_000,
        16 * 1024 * 1024,
    )
}
#[test]
fn bounded_actual_results_and_resource_rejection() {
 for (body,expected) in [("return null;",serde_json::Value::Null),("return 17;",serde_json::json!(17.)),("return {value:17,trace:[1,2]};",serde_json::json!({"value":17.,"trace":[1.,2.]}))] {
  assert_eq!(execute(&Strategy::compile(&source(body)).unwrap(),&mut session()).unwrap()["result"],expected);
 }
 let mut m=session();
 let strategy=Strategy::compile(&source("step(net,inputs,targets);return net;")).unwrap();
 assert!(execute(&strategy,&mut m).unwrap_err().to_string().contains("strategy return"));
 assert_eq!(m.parameter("w").unwrap(),[1.]);
 let deep=format!("return {}0{};","[".repeat(34),"]".repeat(34));
 assert!(execute(&Strategy::compile(&source(&deep)).unwrap(),&mut session()).unwrap_err().to_string().contains("nesting"));
}
#[test]
fn zero_multi_commit_and_error_preserve_explicit_commits() {
    let mut m = session();
    let s = Strategy::compile(&source("return inputs;")).unwrap();
    let metrics = execute(&s, &mut m).unwrap();
    assert_eq!(metrics["commits"], 0);
    assert_eq!(metrics.as_object().unwrap().len(), 4);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    let s = Strategy::compile(&source(
        "step(net,inputs,targets); step(net,inputs,targets);",
    ))
    .unwrap();
    assert_eq!(execute(&s, &mut m).unwrap()["commits"], 2);
    assert_eq!(m.parameter("w").unwrap(), [1.5]);
    let s = Strategy::compile(&source(
        "step(net,inputs,targets); assert false, \"after commit\";",
    ))
    .unwrap();
    assert!(execute(&s, &mut m)
        .unwrap_err()
        .to_string()
        .contains("after commit"));
    assert_eq!(m.parameter("w").unwrap(), [1.75]);
    let snap = m
        .differentiate_custom(&[vec![1.]], &[vec![2.]], &mut 100000, 1024 * 1024)
        .unwrap();
    assert_eq!(snap.state.steps, 3);
}
#[test]
fn rejects_capabilities_recursion_and_invalid_arity() {
    for body in [
        "load text from \"secret\";",
        "http_get(\"https://example.com\");",
        "differentiate(net,inputs);",
        "strategy(net,inputs,targets);",
        "save checkpoint net to \"x\";",
    ] {
        assert!(Strategy::compile(&source(body)).is_err(), "{body}");
    }
}
#[test]
fn stale_handles_and_budgets_do_not_commit() {
    let stale=Strategy::compile(&source("let g=differentiate(net,inputs,targets); step(net,inputs,targets); let s=gradient_snapshot(g); commit_gradients(g,s[\"parameters\"],null);")).unwrap();
    let mut m = session();
    assert!(execute(&stale, &mut m)
        .unwrap_err()
        .to_string()
        .contains("stale"));
    assert_eq!(m.parameter("w").unwrap(), [1.]);
    let normal = Strategy::compile(&source("step(net,inputs,targets);")).unwrap();
    let mut m = session();
    assert!(normal
        .execute(&mut m, &[vec![1.]], &[vec![2.]], &mut 1, 1024 * 1024)
        .is_err());
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert!(normal
        .execute(&mut m, &[vec![1.]], &[vec![2.]], &mut 100000, 64)
        .is_err());
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    let large = Strategy::compile(&source("let x=range(0,1000000000);")).unwrap();
    assert!(execute(&large, &mut m).is_err());
    let released = Strategy::compile(&source(
        "let g=differentiate(net,inputs,targets); release_gradient(g); gradient_snapshot(g);",
    ))
    .unwrap();
    assert!(execute(&released, &mut m)
        .unwrap_err()
        .to_string()
        .contains("released"));
    // A failed invocation's local identities and gradients never affect the next one.
    assert_eq!(execute(&normal, &mut m).unwrap()["commits"], 1);
}
#[test]
fn existing_sam_is_resident_and_matches_direct_wgt() {
    let original = SAM;
    let wrapped=format!("{}\nfn strategy(net,inputs,targets){{return sam_step(net,inputs,targets,0.1,0.1,0.9);}}", original.split("fn main()").next().unwrap());
    let strategy = Strategy::compile(&wrapped).unwrap();
    let mut m = session();
    // Execute the same source directly without its checkpoint filesystem side effects.
    let direct=format!("{}\nfn main(){{let net=Regression(\"none\");let inputs=[[1],[2]];let targets=[[2],[4]];for epoch in range(0,6){{sam_step(net,inputs,targets,0.1,0.1,0.9);}}return net([3]);}}",original.split("fn main()").next().unwrap());
    let report = wgtime_core::script::run(
        &direct,
        std::path::Path::new("/tmp/weightlang"),
        10_000_000,
    )
    .unwrap();
    for _ in 0..6 {
        let metrics = execute(&strategy, &mut m).unwrap();
        assert_eq!(metrics["commits"], 1);
        assert_eq!(metrics.as_object().unwrap().len(), 4);
    }
    let actual = m.evaluate(&[3.], 100000).unwrap();
    assert_eq!(report.result, serde_json::json!([actual[0] as f64]));
    let g = m
        .differentiate_custom(&[vec![1.]], &[vec![2.]], &mut 100000, 1024 * 1024)
        .unwrap();
    assert_eq!(g.state.steps, 6);
}
#[test]
fn independent_sessions_can_execute_shared_code_concurrently() {
    let strategy =
        std::sync::Arc::new(Strategy::compile(&source("step(net,inputs,targets);")).unwrap());
    let threads: Vec<_> = (0..4)
        .map(|i| {
            let strategy = strategy.clone();
            std::thread::spawn(move || {
                let mut m = session();
                for _ in 0..i {
                    execute(&strategy, &mut m).unwrap();
                }
                m.parameter("w").unwrap()[0]
            })
        })
        .collect();
    let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(values, [0., 1., 1.5, 1.75]);
}

#[test]
fn configured_strategy_checkpoint_resume() {
    let text = format!("{MODEL}\n{STEP}\nfn strategy(net,inputs,targets,config){{ let loss=step(net,inputs,targets); assert config[0] == 7; return loss; }}\nfn main(){{return null;}}");
    let s = Strategy::compile(&text).unwrap();
    let mut m = session();
    s.execute_with_parameters(
        &mut m,
        &[vec![1.]],
        &[vec![2.]],
        &serde_json::json!([7]),
        &mut 100000,
        1024 * 1024,
    )
    .unwrap();
    let directory = std::path::PathBuf::from(format!(
        "/tmp/weightlang/strategy-checkpoint-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("step.json");
    m.save_checkpoint(&path).unwrap();
    let saved = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut resumed = AdaptiveSession::restore_checkpoint(saved, 1024 * 1024).unwrap();
    for _ in 0..3 {
        let a = s
            .execute_with_parameters(
                &mut m,
                &[vec![1.]],
                &[vec![2.]],
                &serde_json::json!([7]),
                &mut 100000,
                1024 * 1024,
            )
            .unwrap();
        let b = s
            .execute_with_parameters(
                &mut resumed,
                &[vec![1.]],
                &[vec![2.]],
                &serde_json::json!([7]),
                &mut 100000,
                1024 * 1024,
            )
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(m.parameter("w").unwrap(), resumed.parameter("w").unwrap());
    }
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
#[ignore = "explicit release timing probe; not a performance gate"]
fn resident_sam_timing_probe() {
    let original = SAM;
    let iterations = 100;
    let direct=format!("{}\nfn strategy(net,inputs,targets){{sam_step(net,inputs,targets,0.01,0.1,0.9);}}\nfn main(){{let net=Regression(\"none\");let inputs=[[1],[2]];let targets=[[2],[4]];for i in range(0,{iterations}){{strategy(net,inputs,targets);}} return net([3]);}}",original.split("fn main()").next().unwrap());
    let strategy = Strategy::compile(&direct).unwrap();
    for round in 0..5 {
        let now = std::time::Instant::now();
        let report = wgtime_core::script::run(
            &direct,
            std::path::Path::new("/tmp/weightlang"),
            100_000_000,
        )
        .unwrap();
        let direct_ns = now.elapsed().as_nanos();
        let mut m = session();
        let now = std::time::Instant::now();
        for _ in 0..iterations {
            execute(&strategy, &mut m).unwrap();
        }
        let resident_ns = now.elapsed().as_nanos();
        assert_eq!(
            report.result,
            serde_json::json!([m.evaluate(&[3.], 100000).unwrap()[0] as f64])
        );
        println!(
            "round={round} commits={iterations} direct_ns={direct_ns} resident_ns={resident_ns}"
        );
    }
}
