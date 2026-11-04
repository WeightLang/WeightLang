use std::{collections::BTreeMap, path::Path};
use wgtime_core::script;

fn evaluate(
    body: &str,
    input: &str,
    width: usize,
    output: usize,
) -> anyhow::Result<serde_json::Value> {
    let source = format!("wgt 0.5\nimport act from \"std/activation\"\nimport losses from \"std/loss\"\nmodel Probe\ninput x: f32[{width}]\noutput y: f32[{output}]\nfn forward(x) {{ return {body}; }}\nfn main() {{ let net=Probe(\"sgd\"); return net({input}); }}");
    Ok(script::run(&source, Path::new("."), 100000)?.result)
}

fn close(actual: &serde_json::Value, expected: &[f64]) {
    let actual = actual.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        let a = a.as_f64().unwrap();
        assert!((a - e).abs() < 2e-5 * e.abs().max(1.0), "{a} != {e}");
    }
}

#[test]
fn wgt_activation_math_and_boundaries() {
    close(
        &evaluate("act.swish(x)", "[-1000,0,1000]", 3, 3).unwrap(),
        &[0., 0., 1000.],
    );
    close(
        &evaluate("act.swiglu(x,[2,3,4])", "[-1,0,2]", 3, 3).unwrap(),
        &[-2. / (1. + 1f64.exp()), 0., 8. / (1. + (-2f64).exp())],
    );
    close(
        &evaluate("act.squared_relu(x)", "[-3,0,2]", 3, 3).unwrap(),
        &[0., 0., 4.],
    );
    assert!(
        evaluate("act.swiglu(x,[1])", "[1,2,3]", 3, 3).is_err(),
        "no scalar broadcast"
    );
    assert!(
        evaluate("act.squared_relu(x)", "[1e30]", 1, 1).is_err(),
        "overflow must not silently clip"
    );
}

#[test]
fn wgt_loss_reductions_and_stable_logits() {
    close(
        &evaluate("losses.squared_error(x,[1,1,1])", "[2,3,4]", 3, 1).unwrap(),
        &[14. / 3.],
    );
    close(
        &evaluate("losses.half_squared_error(x,[1,1,1])", "[2,3,4]", 3, 1).unwrap(),
        &[7. / 3.],
    );
    close(
        &evaluate("losses.categorical_logits(x,[0,1])", "[1000,-1000]", 2, 1).unwrap(),
        &[2000.],
    );
    close(
        &evaluate(
            "losses.categorical_logits(x,[0.25,0.75])",
            "[1000,1000]",
            2,
            1,
        )
        .unwrap(),
        &[2f64.ln()],
    );
    assert!(evaluate("losses.squared_error(x,[0])", "[1,2]", 2, 1).is_err());
}

#[test]
fn bundled_imports_are_exact_closed_and_do_not_use_working_directory() {
    for path in [
        "std/../activation/operators",
        "std/activation/operators.wgt",
        "std/activation/missing",
        "std/nn",
    ] {
        let source = format!("wgt 0.5\nimport ops from \"{path}\"\nfn main() {{ return true; }}");
        let error = script::run(&source, Path::new("/does/not/exist"), 1000).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported native standard operator module"),
            "{error:#}"
        );
    }
    for path in ["std/activation", "std/loss"] {
        let source = script::operators::standard_source(path).unwrap();
        script::operators::library(source).unwrap();
    }
    for path in ["std/activation/operators", "std/loss/operators"] {
        let text=format!("wgt 0.5\nimport ops from \"{path}\"\nfn main(){{return null;}}");
        assert!(script::parse_with_imports(&text,Path::new("."),&BTreeMap::new()).unwrap_err().to_string().contains("internal /operators import was removed"));
    }
    let source = "wgt 0.5\nimport ops from \"std/activation\"\nimport ops from \"std/loss\"\nfn main() { return true; }";
    assert!(script::parse_with_imports(source, Path::new("."), &BTreeMap::new()).is_err());
    let ordinary = "wgt 0.5\nimport act from \"std/activation\"\nfn main(){return act.swish([1]);}";
    assert!(script::run(ordinary, Path::new("/does/not/exist"), 10000).is_err());
}

#[test]
fn imported_activation_and_loss_train_and_codegen_embeds_the_computation() {
    let source = r#"wgt 0.5
import activation from "std/activation"
import loss_ops from "std/loss"
model Probe
input x:f32[1]
output y:f32[1]
parameter gain:f32[1]=[1]
fn forward(x){return activation.squared_relu(x*gain);}
fn loss(prediction,target){return loss_ops.squared_error(prediction,target);}
fn main(){let net=Probe("sgd");let before=train net on [[2]] against [[0]] {rate:0.01;loss:loss;};let after=train net on [[2]] against [[0]] {rate:0.01;loss:loss;};assert after<before;return net([2]);}
"#;
    // Bundled imports do not depend on the source tree being present at runtime.
    script::run(source, Path::new("/does/not/exist"), 100000).unwrap();
    let rust = script::codegen::rust_source(source, "/does/not/exist").unwrap();
    assert!(!rust.contains("include_str!"));
    assert!(!rust.contains("standard_source("));
    assert!(rust.contains("__adaptive"));
}
