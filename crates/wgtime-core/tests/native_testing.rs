use std::{collections::BTreeMap, path::Path};
use wgtime_core::script::{self, testing::Suite};

#[test]
fn tests_are_sorted_and_run_independently_without_main() {
    let suite = Suite::parse("wgt 0.5\nfn test_z() { assert false, \"broken\"; }\nfn test_a() { return random(); }\nfn test_bad(x) {}\nfn helper() {}\nfn main() { assert false, \"must not run\"; }").unwrap();
    assert_eq!(suite.names(), &["test_a", "test_bad", "test_z"]);
    let run = |name| suite.run(name, Path::new("."), 1000, BTreeMap::new());
    let first = run("test_a").unwrap();
    assert!(format!("{:#}", run("test_z").unwrap_err()).contains("line 2"));
    assert!(run("test_bad").is_err());
    let next = run("test_a").unwrap();
    assert_eq!(first.result, next.result);
    assert_eq!(first.operations, next.operations);
    assert!(run("helper").is_err());
}

#[test]
fn test_only_program_checks_but_cannot_run_or_compile_implicitly() {
    let source = "wgt 0.5\nfn test_only() { assert true, \"ok\"; }";
    assert!(script::parse(source).is_ok());
    assert!(script::run(source, Path::new("."), 100).is_err());
    assert!(script::codegen::rust_source(source, ".").is_err());
}

#[test]
fn cpu_forward_and_training_benchmarks_have_same_statistics_contract() {
    let source = r#"wgt 0.5
fn main() {
    let net = mlp([1,1], "identity", 42, "sgd");
    benchmark(net, [0], 0, 3, "forward");
    benchmark(net, [[[-1],[0],[1]], [[-1],[1],[3]], 0.1, "mse"], 2, 3, "training", "train:cpu");
}"#;
    let report = script::run(source, Path::new("."), 10000).unwrap();
    for name in ["forward", "training"] {
        let result = &report.metrics[name];
        assert_eq!(result["device"], "cpu");
        assert_eq!(result["repeats"], 3);
        assert!(result["p50"].as_f64().unwrap() <= result["p95"].as_f64().unwrap());
    }
}

#[test]
fn unavailable_gpu_never_returns_cpu_measurements() {
    let error = script::run(r#"wgt 0.5
fn main() { let net=mlp([1,1], "identity", 42, "sgd"); benchmark(net,[0],0,1,"gpu","vulkan:999999"); }
"#, Path::new("."), 1000).unwrap_err();
    assert!(format!("{error:#}").contains("GPU"));
}

#[test]
fn model_constructor_is_not_discovered_as_a_test() {
    let suite = Suite::parse("wgt 0.5\nmodel test_model\ninput x: f32[1]\noutput y: f32[1]\nlayer same: identity\ninit seed=1\nfn test_real() {}").unwrap();
    assert_eq!(suite.names(), &["test_real"]);
}

#[test]
fn operation_budget_failure_does_not_starve_the_next_test() {
    let suite = Suite::parse(
        "wgt 0.5\nfn test_loop() { while true {} }\nfn test_ok() { assert true, \"ok\"; }",
    )
    .unwrap();
    let error = suite
        .run("test_loop", Path::new("."), 50, BTreeMap::new())
        .unwrap_err();
    assert!(format!("{error:#}").contains("operation budget exceeded"));
    assert!(suite
        .run("test_ok", Path::new("."), 50, BTreeMap::new())
        .is_ok());
}
