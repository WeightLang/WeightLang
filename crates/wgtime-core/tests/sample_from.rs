use std::path::Path;
use wgtime_core::{script, training::random};

const MODEL: &str =
    "wgt 0.5\nmodel Policy\ninput x: f32[3]\noutput y: f32[3]\nfn forward(x) { return x; }\n";

// Preserve the removed builtin's numerical order as an independent regression oracle.
fn legacy_sample(logits: &[f32], rng: &mut u64) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probabilities: Vec<f64> = logits.iter().map(|n| ((*n - max) as f64).exp()).collect();
    let mut draw = random(rng) * probabilities.iter().sum::<f64>();
    let mut action = logits.len() - 1;
    for (i, p) in probabilities.iter().enumerate() {
        draw -= p;
        if draw <= 0. {
            action = i;
            break;
        }
    }
    action
}

#[test]
fn categorical_sampling_preserves_legacy_results_and_rng() {
    for logits in [[0., 0., 0.], [1000., 999., -1000.], [-1e30, 1e30, 0.]] {
        for seed in [1, 7, 91] {
            let mut rng = seed;
            let expected: Vec<_> = (0..64)
                .map(|_| legacy_sample(&logits, &mut rng) as f64)
                .collect();
            let next = random(&mut rng);
            let input = format!("[{},{},{}]", logits[0], logits[1], logits[2]);
            let text = format!("{MODEL} fn main() {{ let net=Policy(\"adam\"); seed({seed}); let actions=[]; for i in range(0,64) {{ actions=append(actions, sample from net({input})); }} return [actions,random()]; }}");
            let result = script::run(&text, Path::new("."), 100_000).unwrap().result;
            assert_eq!(result, serde_json::json!([expected, next]));
        }
    }
}

#[test]
fn input_is_evaluated_once_and_model_parameter_is_supported() {
    let text = format!(
        r#"{MODEL}
fn draw(net,x) {{ return sample from net(x); }}
fn main() {{
 let net=Policy("adam"); seed(23);
 let x=[random(),random(),random()]; let expected=draw(net,x); let tail=random();
 seed(23); let actual=sample from net([random(),random(),random()]);
 assert actual==expected, "single input evaluation"; assert random()==tail, "RNG advancement";
 let sample=0; seed(23); return sample from net(x);
}}"#
    );
    let result = script::run(&text, Path::new("."), 10_000).unwrap().result;
    assert!((0.0..3.0).contains(&result.as_f64().unwrap()));
}

#[test]
fn sampling_discards_forward_writes_and_keeps_optimizer_state() {
    let text = r#"wgt 0.5
model Stateful
input x: f32[3]
parameter memory: f32[3] = [0,0,0]
output y: f32[3]
fn forward(x) { let next=memory+x; write(memory,next); return next; }
fn loss(p,t) { return mean(square(p-t)); }
fn main() {
 # Independent constructors give the control and sampled model equal initial weights and Adam state.
 let sampled=Stateful("adam"); let control=Stateful("adam");
 for i in range(0,4) { let action=sample from sampled([1,2,3]); assert action>=0 && action<3, "class range"; }
 assert sampled([0,0,0])==control([0,0,0]), "sample committed write";
 assert train sampled on [[1,2,3]] against [[0,0,0]] { loss: loss; rate: 0.01; clip: 1; }==train control on [[1,2,3]] against [[0,0,0]] { loss: loss; rate: 0.01; clip: 1; }, "optimizer changed";
 assert sampled([0,0,0])==control([0,0,0]), "training states differ";
 let written=sampled([1,2,3]);
 assert sampled([0,0,0])==written, "ordinary call must commit write";
 return true;
}"#;
    assert_eq!(
        script::run(text, Path::new("."), 100_000).unwrap().result,
        true
    );
}

#[test]
fn sampling_rejects_old_calls_invalid_models_and_inputs() {
    for (expr, message) in [
        ("sample(net,[1,2,3])", "use sample from model(input)"),
        ("sample from net()", "exactly one input"),
        ("sample from net([1],[2])", "exactly one input"),
        ("sample from (net)([1,2,3])", "model handle variable"),
        ("sample from net([1,2])", "input"),
        ("sample from net([1e300,0,0])", "f32"),
        ("sample from net([1/0,0,0])", "nonfinite"),
        ("sample from absent([1,2,3])", "undefined variable"),
    ] {
        let text = format!("{MODEL} fn main() {{ let net=Policy(\"adam\"); return {expr}; }}");
        let error = script::run(&text, Path::new("."), 10_000).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{expr}: {error:#}");
    }
    let overflow = MODEL.replace("return x;", "return exp(x);");
    let overflow = format!(
        "{overflow} fn main(){{let net=Policy(\"adam\");return sample from net([1000,0,0]);}}"
    );
    assert!(script::run(&overflow, Path::new("."), 1000).is_err());
    let text = "wgt 0.5\nfn main(){let net=1;return sample from net([1]);}";
    assert!(script::run(text, Path::new("."), 1000).is_err());
    let sequential="wgt 0.5\nmodel Identity\ninput x: f32[3]\noutput y: f32[3]\nlayer y: identity\ninit seed=1\nfn main(){let net=Identity(\"adam\");seed(7);return sample from net([0,0,0]);}";
    let expected = legacy_sample(&[0., 0., 0.], &mut 7) as f64;
    assert_eq!(
        script::run(sequential, Path::new("."), 1000)
            .unwrap()
            .result
            .as_f64(),
        Some(expected)
    );
    assert!(script::run(sequential, Path::new("."), 5).is_err());
    let tensor = MODEL.replace("return x;", "return sample from x(x);");
    assert!(format!("{:#}", script::parse(&tensor).unwrap_err()).contains("discrete host"));
}

#[test]
fn probability_blocks_keep_lazy_branches_endpoints_and_rng() {
    let text = r#"wgt 0.5
fn main(){seed(7);let next=random();seed(7);
 assert sample 0 {then{1/0}else{2}}==2, "zero";
 assert sample (1) {then{3}else{1/0}}==3, "one";
 assert random()==next, "endpoint RNG";
 return sample 0.5 {then{true}else{false}};
}"#;
    assert!(script::run(text, Path::new("."), 1000)
        .unwrap()
        .result
        .is_boolean());
    for p in ["-0.1", "1.1", "1/0"] {
        let text = format!("wgt 0.5\nfn main(){{return sample {p} {{then{{1}}else{{0}}}};}}");
        assert!(script::run(&text, Path::new("."), 1000).is_err());
    }
}
