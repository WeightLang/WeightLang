use serde_json::json;
use wgtime_core::adaptive::{parse, AdaptiveSession};

const MODEL: &str = "wgt 0.5\nmodel Regression\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1,1] = [0]\nfrozen b: f32[1] = [0]\nfn forward(x) { return linear(x,w,b); }\nfn loss(prediction,target) { return square(prediction-target).mean; }\n";
fn session() -> AdaptiveSession {
    AdaptiveSession::prepare(parse(MODEL).unwrap(), 1024 * 1024).unwrap()
}
fn gradient(
    m: &mut AdaptiveSession,
) -> wgtime_core::adaptive::optimization::GradientSnapshot {
    m.differentiate_custom(
        &[vec![1.], vec![2.]],
        &[vec![2.], vec![4.]],
        &mut 100000,
        1024 * 1024,
    )
    .unwrap()
}

#[test]
fn temporary_sam_gradients_keep_original_base_and_commit_once() {
    let mut m = session();
    let first = gradient(&mut m);
    // g=-10, rho=.1 => perturb w to -.1; second g=5*(-.1-2)=-10.5.
    let second = m
        .differentiate_at_custom(&first, &[vec![-0.1]], &mut 100000, 1024 * 1024)
        .unwrap();
    assert_eq!(second.parameters, [vec![0.]]);
    assert!((second.gradients[0][0] + 10.5).abs() < 1e-5);
    assert_eq!(second.state.steps, 0);
    assert_eq!(m.revision(), 0);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert_eq!(m.parameter("b").unwrap(), [0.]);
    assert!(m
        .differentiate_at_custom(&first, &[vec![-0.1]], &mut 1, 1024 * 1024)
        .is_err());
    assert_eq!(m.revision(), 0);
    m.commit_custom(&second, vec![vec![1.05]], json!([1, [-10.5]]))
        .unwrap();
    assert_eq!(gradient(&mut m).state.steps, 1);
    assert!(m
        .differentiate_at_custom(&first, &[vec![-0.1]], &mut 100000, 1024 * 1024)
        .is_err());
    assert_eq!(m.parameter("w").unwrap(), [1.05]);
}

#[test]
fn temporary_zero_gradient_and_repeated_parameter_uses() {
    let mut m = session();
    m.replace_parameter("w", &[2.]).unwrap();
    let first = gradient(&mut m);
    assert_eq!(first.gradients, [vec![0.]]);
    let second = m
        .differentiate_at_custom(&first, &first.parameters, &mut 100000, 1024 * 1024)
        .unwrap();
    assert_eq!(second.gradients, [vec![0.]]);
    let text = MODEL.replace("linear(x,w,b)", "add(linear(x,w,b),linear(x,w,b))");
    let mut shared = AdaptiveSession::prepare(parse(&text).unwrap(), 1024 * 1024).unwrap();
    let first = gradient(&mut shared);
    assert_eq!(first.names, ["w"]);
    assert_eq!(first.gradients, [vec![-20.]]);
    let second = shared
        .differentiate_at_custom(&first, &[vec![-0.1]], &mut 100000, 1024 * 1024)
        .unwrap();
    assert!((second.gradients[0][0] + 22.).abs() < 1e-5);
}

#[test]
fn temporary_pass_preserves_forward_state_and_failed_bindings() {
    let text = MODEL.replace(
        "return linear(x,w,b);",
        "write(w,add(w,[1])); return linear(x,w,b);",
    );
    let mut m = AdaptiveSession::prepare(parse(&text).unwrap(), 1024 * 1024).unwrap();
    let first = gradient(&mut m);
    assert_eq!(first.gradients, [vec![-5.]]);
    let second = m
        .differentiate_at_custom(&first, &[vec![-0.1]], &mut 100000, 1024 * 1024)
        .unwrap();
    assert!((second.gradients[0][0] + 5.5).abs() < 1e-5);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert!(m
        .differentiate_at_custom(&first, &[vec![f32::MAX]], &mut 100000, 1024 * 1024)
        .is_err());
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert_eq!(m.revision(), 0);
}

#[test]
fn host_temporary_gradient_reuses_batch_without_consuming_rng() {
    let source = format!(
        "{MODEL}\nfn main() {{
        let net=Regression(\"none\"); seed(9); let expected=random(); seed(9);
        let g=differentiate(net,[[1],[2]],[[2],[4]]);
        let second=differentiate_at(g,[[-0.1]]);
        let s=gradient_snapshot(second);
        assert s[\"parameters\"]==[[0]], \"original commit base\";
        assert random()==expected, \"differentiation must not consume host RNG\";
        release_gradient(g); commit_gradients(second,[[1.05]],[1]);
        return net([1]);
    }}"
    );
    let report =
        wgtime_core::script::run(&source, std::path::Path::new("/tmp/weightlang"), 100000)
            .unwrap();
    assert!((report.result[0].as_f64().unwrap() - 1.05).abs() < 0.000001);
}

#[test]
fn custom_candidates_match_hand_calculation_and_fail_atomically() {
    let mut m = session();
    let g = gradient(&mut m);
    assert_eq!(g.loss, 10.);
    assert_eq!(g.names, ["w"]);
    assert_eq!(g.gradients, [vec![-10.]]);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert!(m
        .commit_custom(&g, vec![vec![f32::INFINITY]], json!([1, [-10.]]))
        .is_err());
    assert!(m
        .commit_custom(&g, vec![vec![1., 2.]], json!([1, [-10.]]))
        .is_err());
    assert!(m
        .commit_custom(&g, vec![vec![1.]], json!("x".repeat(1024 * 1024)))
        .is_err());
    assert_eq!(m.revision(), 0);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    assert_eq!(
        m.commit_custom(&g, vec![vec![1.]], json!([1, [-10.]]))
            .unwrap(),
        10.
    );
    assert!(m.commit_custom(&g, vec![vec![1.]], json!(null)).is_err());
    let next = gradient(&mut m);
    assert_eq!(next.gradients, [vec![-5.]]);
    assert_eq!(next.state.value, json!([1, [-10.]]));
    // momentum: velocity = 0.9 * -10 + -5 = -14; w = 1 - 0.1 * -14.
    m.commit_custom(&next, vec![vec![2.4]], json!([2, [-14.]]))
        .unwrap();
    assert_eq!(m.parameter("b").unwrap(), [0.]);
}

#[test]
fn custom_loss_gradient_matches_finite_difference_and_depth_is_bounded() {
    let mut m = session();
    m.replace_parameter("w", &[0.3]).unwrap();
    let analytic = gradient(&mut m).gradients[0][0];
    let eps = 0.001;
    m.replace_parameter("w", &[0.3 + eps]).unwrap();
    let plus = gradient(&mut m).loss;
    m.replace_parameter("w", &[0.3 - eps]).unwrap();
    let minus = gradient(&mut m).loss;
    assert!((analytic - (plus - minus) / (2. * eps)).abs() < 0.001);
    let g = gradient(&mut m);
    let mut deep = json!(null);
    for _ in 0..33 {
        deep = json!([deep]);
    }
    let revision = m.revision();
    assert!(m.commit_custom(&g, vec![vec![1.]], deep).is_err());
    assert_eq!(m.revision(), revision);
}

#[test]
fn budgets_and_external_parameter_changes_reject_without_committing() {
    let mut m = session();
    assert!(m
        .differentiate_custom(&[vec![1.]], &[vec![2.]], &mut 10000, 1)
        .is_err());
    let mut remaining = 1;
    assert!(m
        .differentiate_custom(&[vec![1.]], &[vec![2.]], &mut remaining, 1024 * 1024)
        .is_err());
    assert_eq!(remaining, 0);
    assert_eq!(m.parameter("w").unwrap(), [0.]);
    let g = gradient(&mut m);
    m.replace_parameter("w", &[0.5]).unwrap();
    assert!(m.commit_custom(&g, vec![vec![1.]], json!([])).is_err());
    assert_eq!(m.parameter("w").unwrap(), [0.5]);
}

#[test]
fn host_gradient_handles_are_bounded_released_and_reject_resource_state() {
    let run = |body: &str| {
        wgtime_core::script::run(
            &format!("{MODEL}\nfn main() {{ let net = Regression(\"none\"); {body} }}"),
            std::path::Path::new("/tmp/weightlang"),
            1000000,
        )
    };
    let good = "for i in range(0,40) { let g=differentiate(net,[[1]],[[2]]); release_gradient(g); } return 1;";
    assert_eq!(run(good).unwrap().result, json!(1.0));
    assert!(
        run("for i in range(0,17) { let g=differentiate(net,[[1]],[[2]]); }")
            .unwrap_err()
            .chain()
            .any(|e| e.to_string().contains("gradient handle limit"))
    );
    assert!(run(
        "let g=differentiate(net,[[1]],[[2]]); release_gradient(g); gradient_snapshot(g);"
    )
    .is_err());
    assert!(run("let g=differentiate(net,[[1]],[[2]]); commit_gradients(g,[[1]],[net]);").is_err());
    assert!(run("let g=differentiate(net,[[1]],[[2]]); commit_gradients(g,[[1]],null); commit_gradients(g,[[1]],null);").is_err());
}

#[test]
fn custom_checkpoint_is_strict_and_resumes_state() {
    let mut m = session();
    let g = gradient(&mut m);
    m.commit_custom(&g, vec![vec![1.]], json!([1, [-10.]]))
        .unwrap();
    let directory = std::path::PathBuf::from("/tmp/weightlang")
        .join(format!("custom-optimization-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("checkpoint.json");
    m.save_checkpoint(&path).unwrap();
    let record: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(record["format"], "weightlang.forward-training.v2");
    let mut restored = AdaptiveSession::restore_checkpoint(record.clone(), 1024 * 1024).unwrap();
    let a = gradient(&mut m);
    let b = gradient(&mut restored);
    assert_eq!(a.parameters, b.parameters);
    assert_eq!(a.gradients, b.gradients);
    assert_eq!(a.state.value, b.state.value);
    for model in [&mut m, &mut restored] {
        let g = gradient(model);
        model
            .commit_custom(&g, vec![vec![2.4]], json!([2, [-14.]]))
            .unwrap();
    }
    assert_eq!(
        gradient(&mut m).state.value,
        gradient(&mut restored).state.value
    );
    let mut invalid = record.clone();
    invalid["surprise"] = json!(true);
    assert!(AdaptiveSession::restore_checkpoint(invalid, 1024 * 1024).is_err());
    let mut invalid = record;
    invalid["custom_optimizer"]["unexpected"] = json!(true);
    assert!(AdaptiveSession::restore_checkpoint(invalid, 1024 * 1024).is_err());
    assert!(m.save_checkpoint(&path).is_err());
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
