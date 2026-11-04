use std::{collections::BTreeMap, path::Path};
use weightlang_runtime::{
    adaptive::{parse, AdaptiveSession},
    script,
    training::{Objective, Optimizer},
};

const PROGRAM: &str = "wgt 0.5
model Router
input x: f32[2]
output y: f32[2]
parameter p: f32[1] = [0.3]
parameter a: f32[2] = [0.4, -0.2]
parameter b: f32[2] = [-0.8, 0.9]
fn forward(x) {
    return gate(softmax(x), route(p, a, b));
}";

fn session(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 1024 * 1024).unwrap()
}

#[test]
fn gate_route_softmax_gradients_match_finite_differences() {
    let mut s = session(PROGRAM);
    let x = [0.2, -0.7];
    let seed = [0.7, -0.4];
    let mut pass = s.forward(&x, 100, 65536).unwrap();
    pass.backward(&seed, &BTreeMap::new()).unwrap();
    let score =
        |s: &mut AdaptiveSession, x: &[f32], bindings: &BTreeMap<String, Vec<f32>>| -> f32 {
            s.forward_with_parameters(x, bindings, 100, 65536)
                .unwrap()
                .output()
                .iter()
                .zip(seed)
                .map(|(y, g)| y * g)
                .sum()
        };
    let eps = 0.001;
    for name in ["p", "a", "b"] {
        let original = s.parameter(name).unwrap().to_vec();
        for i in 0..original.len() {
            let mut plus = original.clone();
            plus[i] += eps;
            let mut minus = original.clone();
            minus[i] -= eps;
            let numeric = (score(&mut s, &x, &BTreeMap::from([(name.into(), plus)]))
                - score(&mut s, &x, &BTreeMap::from([(name.into(), minus)])))
                / (2. * eps);
            assert!(
                (numeric - pass.parameter_gradient(name).unwrap()[i]).abs() < 0.0001,
                "{name}[{i}]"
            );
        }
    }
    for i in 0..2 {
        let mut plus = x;
        plus[i] += eps;
        let mut minus = x;
        minus[i] -= eps;
        let numeric = (score(&mut s, &plus, &BTreeMap::new())
            - score(&mut s, &minus, &BTreeMap::new()))
            / (2. * eps);
        assert!((numeric - pass.input_gradient().unwrap()[i]).abs() < 0.0001);
    }
    // Softmax remains normalized for large logits and is invariant to a common shift.
    let large = s.run(&[10000., 10000.], 100).unwrap().to_vec();
    assert_eq!(large, s.forward(&[0., 0.], 100, 65536).unwrap().output());
}

#[test]
fn probabilities_endpoints_shapes_and_aliases() {
    let mut s = session(PROGRAM);
    for (p, expected) in [(0., [-0.4, 0.45]), (1., [0.2, -0.1])] {
        s.replace_parameter("p", &[p]).unwrap();
        assert_eq!(s.run(&[0., 0.], 100).unwrap(), expected);
    }
    s.replace_parameter("p", &[1.1]).unwrap();
    assert!(s.run(&[0., 0.], 100).is_err());
    assert!(parse(&PROGRAM.replace("f32[1] = [0.3]", "f32[3] = [0.3,0.3,0.3]")).is_err());
    let mut s = session(&PROGRAM.replace("route(p, a, b)", "route(p, a, a)"));
    let mut pass = s.forward(&[0., 0.], 100, 65536).unwrap();
    pass.backward(&[1., 1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("p").unwrap(), [0.]);
    assert_eq!(pass.parameter_gradient("a").unwrap(), [0.5, 0.5]);
    let mut s = session(&PROGRAM.replace("f32[1] = [0.3]", "f32[2] = [0.2,0.8]"));
    let mut pass = s.forward(&[0., 0.], 100, 65536).unwrap();
    pass.backward(&[1., 1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("p").unwrap(), [0.6, -0.55]);
}

#[test]
fn sgd_classification_and_policy_updates_preserve_frozen_parameters() {
    let source = "wgt 0.5
model Classifier
input x: f32[1]
output y: f32[2]
parameter w: f32[2,1] = [0,0]
frozen b: f32[2] = [0.1,-0.1]
fn forward(x) { return linear(x,w,b); }
fn main() {
    let m=Classifier(\"sgd\");
    let before=fit(m,[[-1],[1]],[0,1],0.1,\"cross_entropy\");
    for i in range(0,100) { fit(m,[[-1],[1]],[0,1],0.1,\"cross_entropy\"); }
    let after=fit(m,[[-1],[1]],[0,1],0.1,\"cross_entropy\");
    assert(after<before*0.1,\"classification failed\");
    reinforce(m,[[-1],[1]],[0,1],[1,1],0.1);
    seed(42);
    let action=sample from m([1]);
    assert(action==0 || action==1,\"invalid sampled action\");
    return predict(m,[0]);
}";
    let report = script::run(source, Path::new("."), 100000).unwrap();
    assert_eq!(
        report.result,
        serde_json::json!([0.1f32 as f64, -0.1f32 as f64])
    );
}

#[test]
fn disabled_optimizer_bad_checkpoints_and_tape_exhaustion_are_rejected() {
    let mut s = session(PROGRAM);
    assert!(s
        .train_step(
            &[vec![0., 0.]],
            Objective::Mse(&[vec![0., 0.]]),
            0.1,
            &mut 1000,
            65536
        )
        .is_err());
    s.set_optimizer(Optimizer::Adam);
    assert!(s
        .train_step(
            &[vec![0., 0.]],
            Objective::Mse(&[vec![0., 0.]]),
            0.1,
            &mut 1000,
            1
        )
        .is_err());
    assert_eq!(s.revision(), 0);
    let bad = serde_json::json!({"format":"weightlang.forward-training.v1","program":s.program(),"optimizer":{"optimizer":"adam","steps":1,"first":[],"second":[]}});
    assert!(AdaptiveSession::restore_checkpoint(bad, 65536).is_err());
}

#[test]
fn declared_router_and_gate_learn_end_to_end() {
    let source = include_str!("../../../examples/native_training/soft_routing.wgt");
    let report = script::run(source, Path::new("."), 1_000_000).unwrap();
    assert!((report.result[0].as_f64().unwrap() - 1.).abs() < 0.1);
}

#[test]
fn failed_batch_rolls_back_weights_optimizer_and_forward_writes() {
    let source = PROGRAM.replace("return gate", "write(a, mul(a, a)); return gate");
    let mut s = session(&source);
    let mut control = session(&source);
    s.set_optimizer(Optimizer::Adam);
    control.set_optimizer(Optimizer::Adam);
    let before = serde_json::to_value(s.program()).unwrap();
    assert!(s
        .train_step(
            &[vec![0., 0.], vec![0., 0.]],
            Objective::Mse(&[vec![0., 0.], vec![f32::NAN, 0.]]),
            0.01,
            &mut 1000,
            65536
        )
        .is_err());
    assert_eq!(serde_json::to_value(s.program()).unwrap(), before);
    assert_eq!(s.revision(), 0);
    assert!(s
        .train_step(
            &[vec![0., 0.]],
            Objective::Mse(&[vec![0., 0.]]),
            0.01,
            &mut 2,
            65536
        )
        .is_err());
    assert_eq!(serde_json::to_value(s.program()).unwrap(), before);
    for model in [&mut s, &mut control] {
        model
            .train_step(
                &[vec![0., 0.]],
                Objective::Mse(&[vec![0., 0.]]),
                0.01,
                &mut 1000,
                65536,
            )
            .unwrap();
    }
    assert_eq!(
        serde_json::to_value(s.program()).unwrap(),
        serde_json::to_value(control.program()).unwrap()
    );
}

#[test]
fn checkpoints_resume_exactly_and_exports_load_without_training() {
    let root = std::env::temp_dir()
        .join("weightlang")
        .join(format!("routing-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = format!(
        "{PROGRAM}
fn main() {{
    let m=Router(\"adam\");
    fit(m,[[0,0]],[[0,0]],0.01,\"mse\");
    save(m,\"checkpoint.json\");
    save_model(m,\"model.json\");
    let restored=restore(\"checkpoint.json\");
    let exported=model(\"model.json\",\"none\");
    assert(predict(m,[0,0])==predict(exported,[0,0]),\"export mismatch\");
    let a=fit(m,[[0,0]],[[0,0]],0.01,\"mse\");
    let b=fit(restored,[[0,0]],[[0,0]],0.01,\"mse\");
    assert(a==b,\"loss mismatch\");
    assert(predict(m,[0,0])==predict(restored,[0,0]),\"Adam resume mismatch\");
    return true;
}}"
    );
    let result = script::run(&source, &root, 10000);
    let refusal = script::run(&source, &root, 10000);
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(result.unwrap().result, true);
    assert!(refusal.is_err());
}
