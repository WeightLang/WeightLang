use std::path::Path;
fn run(body: &str) -> anyhow::Result<serde_json::Value> {
    Ok(weightlang_runtime::script::run(
        &format!("wgt 0.5\nfn main() {{ {body} }}"),
        Path::new("."),
        10000,
    )?
    .result)
}
#[test]
fn stable_math_and_domains() {
    assert_eq!(run("return rsqrt(4);").unwrap(), 0.5);
    assert_eq!(run("return hypot(3,4);").unwrap(), 5.0);
    assert!(run("return hypot(1e308,1e308);")
        .unwrap()
        .as_f64()
        .unwrap()
        .is_finite());
    // Separate multiply and add round away the residual; FMA must preserve it.
    assert!(
        run("return fma(1.0000000000000002,0.9999999999999998,-1);")
            .unwrap()
            .as_f64()
            .unwrap()
            < 0.0
    );
    for expr in ["expm1(1e-16)", "log1p(1e-16)"] {
        let v = run(&format!("return {expr};")).unwrap().as_f64().unwrap();
        assert!((v / 1e-16 - 1.0).abs() < 1e-14);
    }
    for expr in [
        "rsqrt(0)",
        "rsqrt(-1)",
        "log1p(-1)",
        "expm1(1000)",
        "fma(1,2)",
        "hypot([],1)",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    assert_eq!(run("return sigmoid(-1000);").unwrap(), 0.0);
    assert_eq!(run("return softplus(1000);").unwrap(), 1000.0);
    assert_eq!(run("return silu(0);").unwrap(), 0.0);
}
#[test]
fn library_examples_execute() {
    for source in [
        include_str!("../../../std/math/scalar.wgt"),
        include_str!("../../../std/activation/stable.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
}

#[test]
fn vector_algorithms_and_extremes() {
    for source in [
        include_str!("../../../std/linalg/vector.wgt"),
        include_str!("../../../std/reduction/mean.wgt"),
        include_str!("../../../std/probability/softmax.wgt"),
        include_str!("../../../std/normalization/l2.wgt"),
        include_str!("../../../std/loss/basic.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
    assert_eq!(run("return mean([1e308,1e308]);").unwrap(), 1e308);
    assert_eq!(
        run("return softmax([1000,1000]);").unwrap(),
        serde_json::json!([0.5, 0.5])
    );
    assert_eq!(
        run("return log_softmax([1000,-1000])[1];").unwrap(),
        -2000.0
    );
    let v = run("return l2_normalize([1e308,1e308],1e-6);").unwrap();
    assert!((v[0].as_f64().unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    assert!(
        run("return l2_normalize([1e-308],10)[0];")
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.0
    );
    for expr in [
        "mean([])",
        "norm([])",
        "dot([1],[1,2])",
        "mse([1],[])",
        "softmax([[]])",
        "l2_normalize([1],0)",
        "cross_entropy([1,2],2)",
        "cross_entropy([1,2],0.5)",
        "log_softmax([1e308,-1e308])",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    assert!(weightlang_runtime::script::run(
        "wgt 0.5\nfn main() { return norm(range(0,100)); }",
        Path::new("."),
        50
    )
    .is_err());
}

#[test]
fn statistics_ordering_and_transforms() {
    for source in [
        include_str!("../../../std/statistics/descriptive.wgt"),
        include_str!("../../../std/ordering/ranking.wgt"),
        include_str!("../../../std/linalg/transforms.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
    for (expr, expected) in [
        ("variance([1,2,3,4],0)", 1.25),
        ("variance([1,2,3,4],1)", 5.0 / 3.0),
        ("stddev([-1e308,1e308],0)", 1e308),
        ("variance([2,2,2,2,2,2,2,2,2,2],0)", 0.0),
        ("median([-1e308,1e308])", 0.0),
        ("median([1e308,1e308])", 1e308),
        ("quantile([3,1,2],0)", 1.0),
        ("quantile([3,1,2],1)", 3.0),
        ("cosine_similarity([1e308,1e308],[-1e308,-1e308])", -1.0),
        ("cosine_similarity([1e-308,1e-308],[1e-308,1e-308])", 1.0),
    ] {
        let actual = run(&format!("return {expr};")).unwrap().as_f64().unwrap();
        assert!(
            (actual - expected).abs() <= 1e-14 * expected.abs().max(1.0),
            "{expr}: {actual}"
        );
    }
    assert_eq!(
        run("return argsort([2,1,2,1]);").unwrap(),
        serde_json::json!([1.0, 3.0, 0.0, 2.0])
    );
    assert_eq!(
        run("return clip([1,2],1,1);").unwrap(),
        serde_json::json!([1.0, 1.0])
    );
    assert_eq!(
        run("return median([5e-324,5e-324]);")
            .unwrap()
            .as_f64()
            .unwrap(),
        f64::from_bits(1)
    );
    for expr in [
        "min([])",
        "sort([])",
        "variance([1],1)",
        "stddev([1],-1)",
        "variance([1,2],0.5)",
        "quantile([1],1.1)",
        "quantile([1],-0.1)",
        "clip([1],2,1)",
        "cosine_similarity([0],[1])",
        "cosine_similarity([1],[1,2])",
        "variance([-1e308,1e308],0)",
        "cumsum([1e308,1e308])",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    assert!(weightlang_runtime::script::run(
        "wgt 0.5\nfn main() { return sort(range(0,20)); }",
        Path::new("."),
        50
    )
    .is_err());
}

#[test]
fn activations_robust_losses_and_signals() {
    for source in [
        include_str!("../../../std/activation/piecewise.wgt"),
        include_str!("../../../std/loss/robust.wgt"),
        include_str!("../../../std/signal/smoothing.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
    for (expr, expected) in [
        ("elu(-1000,2)", -2.0),
        ("hard_swish(1e308)", 1e308),
        ("hard_swish(-1e308)", 0.0),
        ("hard_sigmoid(-3)", 0.0),
        ("hard_sigmoid(3)", 1.0),
        ("bce_logits([1000,-1000],[0,1])", 1000.0),
        ("bce_logits([1e308,-1e308],[0.5,0.5])", 5e307),
        ("bce_logits([0],[0.5])", std::f64::consts::LN_2),
        ("huber([1],[0],1)", 0.5),
        ("huber([2],[0],1)", 1.5),
        ("log_cosh([1000],[0])", 1000.0 - std::f64::consts::LN_2),
        ("log_cosh([1e-8],[0])", 5e-17),
    ] {
        let actual = run(&format!("return {expr};")).unwrap().as_f64().unwrap();
        assert!(
            (actual - expected).abs() <= 1e-14 * expected.abs().max(1e-20),
            "{expr}: {actual}"
        );
    }
    assert_eq!(
        run("return ema([1e308,1],1);").unwrap(),
        serde_json::json!([1e308, 1.0])
    );
    assert_eq!(
        run("return ema([-1e308,1e308],0.5);").unwrap(),
        serde_json::json!([-1e308, 0.0])
    );
    assert_eq!(
        run("return moving_mean([1e308,1e308],2);").unwrap(),
        serde_json::json!([1e308])
    );
    for expr in [
        "relu([])",
        "elu(1,-1)",
        "leaky_relu(-1,-1)",
        "bce_logits([1],[2])",
        "bce_logits([1],[])",
        "huber([1],[0],0)",
        "log_cosh([1],[1,2])",
        "diff([1])",
        "ema([1],0)",
        "ema([1],1.1)",
        "moving_mean([1],0)",
        "moving_mean([1],2)",
        "moving_mean([1],0.5)",
        "diff([-1e308,1e308])",
        "log_cosh([-1e308],[1e308])",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    assert!(weightlang_runtime::script::run(
        "wgt 0.5\nfn main() { return moving_mean(range(0,20),10); }",
        Path::new("."),
        50
    )
    .is_err());
}

#[test]
fn vector_construction_and_indexing() {
    for source in [
        include_str!("../../../std/data/construction.wgt"),
        include_str!("../../../std/data/indexing.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
    for expr in [
        "zeros(0)",
        "ones(0)",
        "full(0,2)",
        "concat([],[])",
        "vector_slice([1],1,1)",
        "gather([],[])",
        "compress([],[])",
        "reverse([])",
    ] {
        assert_eq!(
            run(&format!("return {expr};")).unwrap(),
            serde_json::json!([]),
            "{expr}"
        );
    }
    assert_eq!(
        run("return linspace(-1e308,1e308,3);").unwrap(),
        serde_json::json!([-1e308, 0.0, 1e308])
    );
    assert_eq!(
        run("return linspace(1e308,1e308,3);").unwrap(),
        serde_json::json!([1e308, 1e308, 1e308])
    );
    assert_eq!(
        run("return linspace(2,3,1);").unwrap(),
        serde_json::json!([2.0])
    );
    assert_eq!(
        run("return one_hot(0,1);").unwrap(),
        serde_json::json!([1.0])
    );
    assert_eq!(
        run("return gather([2,4],[1,1]);").unwrap(),
        serde_json::json!([4.0, 4.0])
    );
    for expr in [
        "zeros(-1)",
        "ones(1.5)",
        "zeros(1000001)",
        "full(1,true)",
        "linspace(0,1,0)",
        "linspace(0,1,1.5)",
        "one_hot(0,0)",
        "one_hot(2,2)",
        "vector_slice([1],1,0)",
        "vector_slice([1],0,2)",
        "gather([1],[1])",
        "gather([1],[-1])",
        "gather([1],[0.5])",
        "gather([1],[true])",
        "compress([1],[1])",
        "compress([1],[])",
        "concat([[]],[])",
        "reverse([true])",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    for body in [
        "return zeros(100);",
        "return one_hot(0,100);",
        "return gather(range(0,100),[]);",
    ] {
        assert!(weightlang_runtime::script::run(
            &format!("wgt 0.5\nfn main() {{ {body} }}"),
            Path::new("."),
            50
        )
        .is_err());
    }
}

#[test]
fn probability_analysis_and_temperature() {
    for source in [
        include_str!("../../../std/probability/distributions.wgt"),
        include_str!("../../../std/probability/temperature.wgt"),
    ] {
        weightlang_runtime::script::run(source, Path::new("."), 10000).unwrap();
    }
    for (expr, expected) in [
        ("entropy([0.5,0.5])", std::f64::consts::LN_2),
        ("kl_divergence([1,0],[0.5,0.5])", std::f64::consts::LN_2),
        ("js_divergence([1,0],[0,1])", std::f64::consts::LN_2),
        (
            "cross_entropy_probs([0.5,0.5],[0.5,0.5])",
            std::f64::consts::LN_2,
        ),
        ("js_divergence([0.25,0.75],[0.25,0.75])", 0.0),
    ] {
        let actual = run(&format!("return {expr};")).unwrap().as_f64().unwrap();
        assert!((actual - expected).abs() < 1e-14, "{expr}: {actual}");
    }
    assert_eq!(
        run("return normalize_probabilities([1e308,1e308]);").unwrap(),
        serde_json::json!([0.5, 0.5])
    );
    assert_eq!(
        run("return normalize_probabilities([5e-324,5e-324]);").unwrap(),
        serde_json::json!([0.5, 0.5])
    );
    assert_eq!(
        run("return temperature_softmax([0,1],5e-324);").unwrap(),
        serde_json::json!([0.0, 1.0])
    );
    let v = run("return temperature_softmax([-1e308,1e308],1e308);").unwrap();
    assert!((v[0].as_f64().unwrap() - 1.0 / (1.0 + 2.0_f64.exp())).abs() < 1e-14);
    let a = run("return js_divergence([0.1,0.9],[0.8,0.2]);")
        .unwrap()
        .as_f64()
        .unwrap();
    let b = run("return js_divergence([0.8,0.2],[0.1,0.9]);")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((a - b).abs() < 1e-14);
    assert!(run("return js_divergence([1,5e-324],[1,0]);").is_ok());
    assert!(run("return entropy([0.5,0.5000000001]);").is_ok());
    for expr in [
        "entropy([])",
        "entropy([2,2])",
        "entropy([-0.1,1.1])",
        "normalize_probabilities([0,0])",
        "normalize_probabilities([-1,2])",
        "kl_divergence([1,0],[0,1])",
        "cross_entropy_probs([1,0],[0,1])",
        "js_divergence([1],[0.5,0.5])",
        "temperature_softmax([1],0)",
        "temperature_softmax([1],-1)",
    ] {
        assert!(run(&format!("return {expr};")).is_err(), "{expr}");
    }
    assert!(weightlang_runtime::script::run(
        "wgt 0.5\nfn main() { return normalize_probabilities(range(0,100)); }",
        Path::new("."),
        50
    )
    .is_err());
}
