use std::{collections::BTreeMap, path::Path};
use wgtime_core::{
    adaptive::{parse, AdaptiveSession},
    script,
};

const SOURCE: &str = "wgt 0.5
model Custom
input x: f32[2]
output y: f32[2]
parameter gain: f32[2] = [2, 3]
fn twice(a) { let x = a + a; return x; }
fn custom(a,b) { let x = twice(a); return x * b; }
fn forward(x) { let a = custom(x,gain); return custom(a,gain); }
fn loss(prediction,target) { return error(prediction,target); }
fn error(a,b) { let x = a-b; return square(x).mean; }
";

#[test]
fn tensor_helpers_are_hygienic_differentiable_and_portable() {
    let mut model = AdaptiveSession::prepare(parse(SOURCE).unwrap(), 65536).unwrap();
    let mut pass = model.forward(&[1., 2.], 1000, 65536).unwrap();
    assert_eq!(pass.output(), [16., 72.]);
    pass.backward(&[1., 1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("gain").unwrap(), [16., 48.]);
    let ir = serde_json::to_string(model.program()).unwrap();
    let mut reloaded = AdaptiveSession::prepare(serde_json::from_str(&ir).unwrap(), 65536).unwrap();
    assert_eq!(reloaded.run(&[1., 2.], 1000).unwrap(), [16., 72.]);
    assert!(reloaded.run(&[1., 2.], 1).is_err());
    let source = format!("{SOURCE}\nfn main() {{ let m=Custom(\"adam\"); let a=train m on [[1,2]] against [[0,0]] {{rate:0.01;loss:loss;}}; let b=train m on [[1,2]] against [[0,0]] {{rate:0.01;loss:loss;}}; assert b<a, \"gradient update\"; return true; }}");
    assert_eq!(
        script::run(&source, Path::new("."), 10000).unwrap().result,
        true
    );
}

#[test]
fn helper_validation_rejects_capture_recursion_effects_and_shape_errors() {
    let nested = format!("{}x{}", "twice(".repeat(40), ")".repeat(40));
    assert!(parse(&SOURCE.replace("custom(a,gain)", &nested)).is_err());
    for body in [
        "return gain;",
        "return twice(a);",
        "return random();",
        "let a=a; return a;",
        "write(gain,a); return a;",
        "if true { return a; } return a;",
        "return a + [1];",
        "return twice();",
    ] {
        let source = SOURCE.replace("let x = a + a; return x;", body);
        assert!(parse(&source).is_err(), "{body}");
    }
    for source in [
        "wgt 0.5\noperators Bad\nfn main() { return [1]; }",
        "wgt 0.5\noperators Bad\nfn a(x) { return b(x); }\nfn b(x) { return a(x); }",
        "wgt 0.5\noperators Bad\nfn a(x) { return read(w); }",
    ] {
        assert!(script::operators::library(source).is_err());
    }
}

#[test]
fn local_import_example_trains_and_paths_are_confined() {
    let root = std::env::temp_dir().join(format!("weightlang-custom-operators-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let operators = "wgt 0.5\noperators Activations\nfn swish(x) { return x * sigmoid(x); }\n";
    let source = "wgt 0.5\nimport activation from \"./ops.wgt\"\nmodel CustomOperators\ninput x: f32[2]\noutput y: f32[2]\nparameter gain: f32[2] = [1,1]\nfn forward(x) { return activation.swish(x*gain); }\nfn loss(prediction,target) { return square(prediction-target).mean; }\nfn main() { let net=CustomOperators(\"adam\"); let before=train net on [[-1,2],[1,-2]] against [[0,0],[0,0]] {rate:0.03;loss:loss;}; for step in range(0,40) { train net on [[-1,2],[1,-2]] against [[0,0],[0,0]] {rate:0.03;loss:loss;}; } let after=train net on [[-1,2],[1,-2]] against [[0,0],[0,0]] {rate:0.03;loss:loss;}; assert after<before; return net([1,2]); }";
    std::fs::write(root.join("ops.wgt"), operators).unwrap();
    script::run(source, &root, 100000).unwrap();
    assert!(script::run(
        &source.replace("./ops.wgt", "./../custom_operators/ops.wgt"),
        &root,
        100000
    )
    .is_err());
    assert!(
        script::parse(source).is_err(),
        "context-free parser must not silently ignore imports"
    );
    assert!(script::codegen::rust_source(source, root.to_str().unwrap())
        .unwrap()
        .contains("__adaptive"));
    let twice_imported = source
        .replace(
            "model CustomOperators",
            "import second from \"./ops.wgt\"\nmodel CustomOperators",
        )
        .replace("activation.swish(x * gain)", "second.swish(x * gain)");
    assert_eq!(
        script::run(source, &root, 100000).unwrap().result,
        script::run(&twice_imported, &root, 100000).unwrap().result
    );
    let test = source.replace("fn main()", "fn test_training()");
    let suite = script::testing::Suite::parse_with_imports(&test, &root, &BTreeMap::new()).unwrap();
    suite
        .run("test_training", &root, 100000, BTreeMap::new())
        .unwrap();
}

#[test]
fn arguments_are_evaluated_once_and_inference_laziness_is_preserved() {
    let source="wgt 0.5\nmodel Once\ninput x: f32[3]\noutput y: f32[3]\nfn twice(a) { return a+a; }\nfn forward(x) { return twice(resource_state()); }";
    let program = parse(source).unwrap();
    let json = serde_json::to_string(&program).unwrap();
    assert_eq!(json.matches("resource_state").count(), 1);
    let source="wgt 0.5\nmodel Lazy\ninput x: f32[1]\noutput y: f32[1]\nfrozen p: prob = 0\nfn explode(a) { return square(a); }\nfn forward(x) { return gate(p,explode(x)); }";
    let mut model = AdaptiveSession::prepare(parse(source).unwrap(), 65536).unwrap();
    assert_eq!(model.run(&[1e30], 1000).unwrap(), [0.]);
    assert!(model.forward(&[1e30], 1000, 65536).is_err());
}
