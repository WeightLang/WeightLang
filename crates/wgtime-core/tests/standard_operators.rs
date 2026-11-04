use wgt_language::model::ModelDefinition;
use wgtime_core::{
    adaptive::{self, AdaptiveSession},
    training::Optimizer,
};

const MEMORY: usize = 8 * 1024 * 1024;
const WORK: u64 = 100_000;

fn session(source: &str, compiled: bool) -> anyhow::Result<AdaptiveSession> {
    // Resolve the complete bundled computation closure before core execution.
    // Neither definition parsing nor execution has a source-directory argument.
    let definition = ModelDefinition::parse(source)?;
    assert!(!definition.source().lines().any(|line| line.trim_start().starts_with("import ")));
    let mut model = AdaptiveSession::prepare(adaptive::parse(definition.source())?, MEMORY)?;
    if compiled {
        model.enable_compiled()?;
        assert_eq!(model.execution_backend(), "native-cpu");
    }
    Ok(model)
}

fn evaluate(
    body: &str,
    input: &[f32],
    width: usize,
    output: usize,
    compiled: bool,
) -> anyhow::Result<Vec<f32>> {
    let source = format!("wgt 0.5\nimport act from \"std/activation\"\nimport losses from \"std/loss\"\nmodel Probe\ninput x: f32[{width}]\noutput y: f32[{output}]\nfn forward(x) {{ return {body}; }}\n");
    let mut model = session(&source, compiled)?;
    let result = model.run(input, WORK)?.to_vec();
    assert_eq!(model.compiled_forward_counts(), if compiled { (1, 0) } else { (0, 0) });
    Ok(result)
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, e) in actual.iter().zip(expected) {
        let a = f64::from(a);
        assert!((a - e).abs() < 2e-5 * e.abs().max(1.0), "{a} != {e}");
    }
}

#[test]
fn wgt_activation_math_and_boundaries() {
    for compiled in [false, true] {
        close(
            &evaluate("act.swish(x)", &[-1000., 0., 1000.], 3, 3, compiled).unwrap(),
            &[0., 0., 1000.],
        );
        close(
            &evaluate("act.swiglu(x,[2,3,4])", &[-1., 0., 2.], 3, 3, compiled).unwrap(),
            &[-2. / (1. + 1f64.exp()), 0., 8. / (1. + (-2f64).exp())],
        );
        close(
            &evaluate("act.squared_relu(x)", &[-3., 0., 2.], 3, 3, compiled).unwrap(),
            &[0., 0., 4.],
        );
        assert!(
            evaluate("act.swiglu(x,[1])", &[1., 2., 3.], 3, 3, compiled).is_err(),
            "no scalar broadcast"
        );
        assert!(
            evaluate("act.squared_relu(x)", &[1e30], 1, 1, compiled).is_err(),
            "overflow must not silently clip"
        );
    }
}

#[test]
fn wgt_loss_reductions_and_stable_logits() {
    for compiled in [false, true] {
        close(
            &evaluate("losses.squared_error(x,[1,1,1])", &[2., 3., 4.], 3, 1, compiled).unwrap(),
            &[14. / 3.],
        );
        close(
            &evaluate("losses.half_squared_error(x,[1,1,1])", &[2., 3., 4.], 3, 1, compiled).unwrap(),
            &[7. / 3.],
        );
        close(
            &evaluate("losses.categorical_logits(x,[0,1])", &[1000., -1000.], 2, 1, compiled).unwrap(),
            &[2000.],
        );
        close(
            &evaluate("losses.categorical_logits(x,[0.25,0.75])", &[1000., 1000.], 2, 1, compiled).unwrap(),
            &[2f64.ln()],
        );
        assert!(evaluate("losses.squared_error(x,[0])", &[1., 2.], 2, 1, compiled).is_err());
    }
}

fn imported_model(path: &str, body: &str) -> String {
    format!("wgt 0.5\nimport ops from \"{path}\"\nmodel Probe\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){{return {body};}}\n")
}

#[test]
fn bundled_imports_are_exact_closed_and_do_not_use_working_directory() {
    for path in [
        "std/../activation/operators",
        "std/activation/operators.wgt",
        "std/activation/missing",
        "std/nn",
    ] {
        let error = ModelDefinition::parse(&imported_model(path, "x")).unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("unknown model standard module: {path}")),
            "{error:#}"
        );
    }
    for (path, body, expected) in [
        ("std/activation", "ops.squared_relu(x)", 4.),
        ("std/loss", "ops.squared_error(x,[0])", 4.),
    ] {
        let definition = ModelDefinition::parse(&imported_model(path, body)).unwrap();
        let closed = definition.source().to_owned();
        assert!(!closed.contains("import "));
        // Reparse only the embedded closure, without the original module paths.
        // This replaces host codegen-string assertions with actual native runs.
        for compiled in [false, true] {
            let mut model = session(&closed, compiled).unwrap();
            close(&model.run(&[2.], WORK).unwrap(), &[expected]);
            assert_eq!(model.compiled_forward_counts(), if compiled { (1, 0) } else { (0, 0) });
        }
    }
    for path in ["std/activation/operators", "std/loss/operators"] {
        let error = ModelDefinition::parse(&imported_model(path, "x")).unwrap_err();
        assert!(format!("{error:#}").contains(&format!("unknown model standard module: {path}")), "{error:#}");
    }
    let duplicate = "wgt 0.5\nimport ops from \"std/activation\"\nimport ops from \"std/loss\"\nmodel Probe\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return x;}";
    assert!(ModelDefinition::parse(duplicate).is_err());
    let ordinary = "wgt 0.5\nimport act from \"std/activation\"\nfn main(){return act.swish([1]);}";
    assert!(ModelDefinition::parse(ordinary).is_err());
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
"#;
    let definition = ModelDefinition::parse(source).unwrap();
    let closed = definition.source();
    assert!(!closed.contains("import "));
    assert!(!closed.contains("include_str!"));
    assert!(!closed.contains("standard_source("));
    for compiled in [false, true] {
        let mut model = session(closed, compiled).unwrap();
        model.set_optimizer(Optimizer::Sgd).unwrap();
        let inputs = [vec![2.]];
        let targets = [vec![0.]];
        let mut remaining = WORK;
        let before = model.train_custom(&inputs, &targets, 0.01, &mut remaining, MEMORY).unwrap();
        // y=4*g^2, loss=16*g^4, dloss/dg=64*g^3; SGD gives 1 -> 0.36.
        close(&[before], &[16.]);
        close(model.parameter("gain").unwrap(), &[0.36]);
        let mut remaining = WORK;
        let after = model.train_custom(&inputs, &targets, 0.01, &mut remaining, MEMORY).unwrap();
        assert!(after < before);
        close(&[after], &[16. * 0.36f64.powi(4)]);
        let expected_gain = 0.36 - 0.01 * 64. * 0.36f64.powi(3);
        close(model.parameter("gain").unwrap(), &[expected_gain]);
        close(&model.run(&[2.], WORK).unwrap(), &[4. * expected_gain.powi(2)]);
        assert_eq!(model.compiled_forward_counts(), if compiled { (3, 2) } else { (0, 0) });
    }
}
