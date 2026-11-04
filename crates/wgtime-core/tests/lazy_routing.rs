use std::{collections::BTreeMap, path::Path};
use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    script,
};

const PREFIX: &str = "wgt 0.5
model Lazy
input x: f32[1]
output y: f32[1]
parameter p: f32[1] = [1]
frozen large: f32[1,1] = [1e30]
";

fn session(body: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(
        parse(&format!("{PREFIX}\nfn forward(x) {{ {body} }}")).unwrap(),
        65536,
    )
    .unwrap()
}

#[test]
fn endpoints_skip_unselected_expressions_but_fractional_weights_do_not() {
    let mut s = session("return route(p,x,linear(x,large));");
    assert_eq!(s.run(&[1e20], 100).unwrap(), [1e20]);
    // Lazy control flow survives IR serialization; old eager artifacts are not rewritten.
    let encoded = serde_json::to_vec(s.program()).unwrap();
    let mut loaded =
        AdaptiveSession::prepare(serde_json::from_slice(&encoded).unwrap(), 65536).unwrap();
    assert_eq!(loaded.run(&[1e20], 100).unwrap(), [1e20]);
    s.replace_parameter("p", &[0.999]).unwrap();
    assert!(s.run(&[1e20], 100).is_err());
    s.replace_parameter("p", &[0.]).unwrap();
    assert!(s.run(&[1e20], 100).is_err());
    let mut s = session("return route(p,linear(x,large),x);");
    s.replace_parameter("p", &[0.]).unwrap();
    assert_eq!(s.run(&[1e20], 100).unwrap(), [1e20]);
    s.replace_parameter("p", &[1.1]).unwrap();
    assert!(s.run(&[1e20], 100).is_err());
}

#[test]
fn zero_gate_skips_input_and_nested_routes_skip_entire_subtrees() {
    let mut s = session("return gate(p,linear(x,large));");
    s.replace_parameter("p", &[0.]).unwrap();
    assert_eq!(s.run(&[1e20], 100).unwrap(), [0.]);
    assert!(s.forward(&[1e20], 100, 65536).is_err());
    let mut s = session("return route(p,x,route(p,linear(x,large),linear(x,large)));");
    assert_eq!(s.run(&[1e20], 100).unwrap(), [1e20]);
    // Existing eager statements keep their meaning; route does not move computations.
    let mut s = session("let other=linear(x,large); return route(p,x,other);");
    assert!(s.run(&[1e20], 100).is_err());
}

#[test]
fn vector_endpoints_skip_whole_branches_but_mixed_masks_compute_both() {
    let source = "wgt 0.5
model Vector
input x: f32[2]
output y: f32[2]
parameter p: f32[2] = [1,1]
fn forward(x) { return route(p,x,mul(x,x)); }";
    let mut s = AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap();
    assert_eq!(s.run(&[1e20, 1e20], 100).unwrap(), [1e20, 1e20]);
    s.replace_parameter("p", &[1., 0.]).unwrap();
    assert!(s.run(&[1e20, 1e20], 100).is_err());
}

#[test]
fn gradients_at_endpoints_still_include_the_other_branch() {
    let mut s = session("return route(p,x,mul(x,x));");
    for p in [0., 1.] {
        s.replace_parameter("p", &[p]).unwrap();
        let mut pass = s.forward(&[2.], 100, 65536).unwrap();
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.parameter_gradient("p").unwrap(), [-2.]);
        assert_eq!(
            pass.input_gradient().unwrap(),
            [if p == 1. { 1. } else { 4. }]
        );
    }
    let mut s = session("return gate(p,mul(x,x));");
    s.replace_parameter("p", &[0.]).unwrap();
    let mut pass = s.forward(&[2.], 100, 65536).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("p").unwrap(), [4.]);
}

#[test]
fn skipped_work_uses_no_instruction_budget_and_can_change_between_calls() {
    let mut s = session("return route(p,x,tanh(tanh(tanh(tanh(x)))));");
    let selected = s.run(&[0.5], 100).unwrap().to_vec();
    let short = s.steps();
    let _ = s.forward(&[0.5], 100, 65536).unwrap();
    assert!(s.steps() > short);
    assert_eq!(s.run(&[0.5], short).unwrap(), selected);
    s.replace_parameter("p", &[0.]).unwrap();
    assert!(s.run(&[0.5], short).is_err());
    let y = s.run(&[0.5], 100).unwrap()[0];
    assert_eq!(y, 0.5f32.tanh().tanh().tanh().tanh());
    s.replace_parameter("p", &[1.]).unwrap();
    assert_eq!(s.run(&[0.5], 100).unwrap(), [0.5]);
}

#[test]
fn lazy_sample_does_not_commit_writes_and_example_executes() {
    let source = format!(
        "{PREFIX}
fn forward(x) {{ let selected=route(p,x,linear(x,large)); write(p,x); return selected; }}
fn main() {{
    let model=Lazy(\"none\");
    assert(sample from model([1e20])==0,\"invalid action\");
    return model([1e20]);
}}"
    );
    let report = script::run(&source, Path::new("."), 1000).unwrap();
    assert_eq!(report.result, serde_json::json!([1e20f32 as f64]));
    let training = "wgt 0.5
model LazyTrain
input x: f32[1]
output y: f32[1]
parameter p: f32[1] = [0.5]
fn forward(x) { return route(p,x,-x); }
fn loss(prediction,target) { return square(prediction-target).mean; }
fn main() { let model=LazyTrain(\"sgd\"); train model on [[1]] against [[1]] {rate:0.1;loss:loss;}; assert model([1]) != [0], \"training runs\"; return true; }
";
    script::run(training, Path::new("."), 1000).unwrap();
}
