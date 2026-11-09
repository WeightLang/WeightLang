//! Numeric operations needed by explicit local updates remain ordinary tensor operations.
use std::collections::BTreeMap;
use wgt_language::{
    adaptive::{Instruction, Program},
    model::{ModelDefinition, ModelPlan},
};
use wgtime_core::adaptive::AdaptiveSession;

fn program(source: &str) -> Program {
    let ModelPlan::Forward(program) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("forward model expected")
    };
    program
}

fn session(source: &str, compiled: bool) -> AdaptiveSession {
    let mut session = AdaptiveSession::prepare(program(source), 1024 * 1024).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    session
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs() <= 1e-6 * (1. + b.abs()),
            "{actual:?} != {expected:?}"
        );
    }
}

const DIFFERENTIABLE: &str = r#"wgt 0.5
model Numeric
input x:f32[2]
output y:f32[2]
parameter denominator:f32[2]=[2,4]
parameter gain:f32[1]=[-1.5]
# Runtime scalar scaling is explicit; division still requires equal shapes.
fn normalize(values, divisor, rate) {
    return scale(sqrt(values / divisor), rate);
}
fn forward(x) { return normalize(x,denominator,gain); }
"#;

#[test]
fn source_and_serialized_ir_preserve_new_contracts_and_old_literal_scale() {
    let plan = program(DIFFERENTIABLE);
    assert_eq!(plan.version, 3);
    for name in ["div", "sqrt", "scale_by"] {
        assert!(plan.instructions.iter().any(|i| i.contract().name == name));
    }
    let encoded = serde_json::to_value(&plan).unwrap();
    let decoded: Program = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    let literal = program(&DIFFERENTIABLE.replace(
        "sqrt(values / divisor), rate",
        "sqrt(div(values,divisor)), 2",
    ));
    assert!(
        literal
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Scale { factor: 2., .. }))
    );
    assert!(
        !literal
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::ScaleBy { .. }))
    );
    let mut old = plan.clone();
    old.version = 2;
    assert!(old.validate_tensor_contracts().is_err());
}

#[test]
fn division_sqrt_and_runtime_scale_have_the_same_analytic_vjp_in_both_backends() {
    for compiled in [false, true] {
        let mut session = session(DIFFERENTIABLE, compiled);
        let mut pass = session.forward(&[8., 36.], 1000, 65536).unwrap();
        close(pass.output(), &[-3., -4.5]);
        pass.backward(&[1., 2.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[-0.1875, -0.125]);
        close(
            pass.parameter_gradient("denominator").unwrap(),
            &[0.75, 1.125],
        );
        close(pass.parameter_gradient("gain").unwrap(), &[8.]);
        // Independent central differences include the runtime scalar's gradient.
        for name in ["denominator", "gain"] {
            for i in 0..session.parameter(name).unwrap().len() {
                let mut plus = session.parameter(name).unwrap().to_vec();
                let mut minus = plus.clone();
                plus[i] += 0.001;
                minus[i] -= 0.001;
                let score = |s: &mut AdaptiveSession, value: Vec<f32>| {
                    let p = s
                        .forward_with_parameters(
                            &[8., 36.],
                            &BTreeMap::from([(name.into(), value)]),
                            1000,
                            65536,
                        )
                        .unwrap();
                    p.output()[0] + 2. * p.output()[1]
                };
                let numeric = (score(&mut session, plus) - score(&mut session, minus)) / 0.002;
                assert!((numeric - pass.parameter_gradient(name).unwrap()[i]).abs() < 0.001);
            }
        }
        if compiled {
            assert!(session.compiled_forward_counts().1 > 0);
        }
    }
}

#[test]
fn multiaxis_values_scale_by_runtime_reductions_and_shared_scalar_aliases() {
    let source = r#"wgt 0.5
model Matrix
input x:f32[4]
output y:f32[4]
fn forward(x) {
    let matrix=x as [2,2];
    let root=sqrt(matrix);
    return scale(root,sum(x)) as [4];
}
"#;
    for compiled in [false, true] {
        let mut s = session(source, compiled);
        let mut pass = s.forward(&[1., 4., 9., 16.], 1000, 65536).unwrap();
        close(pass.output(), &[30., 60., 90., 120.]);
        pass.backward(&[1.; 4], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[25., 17.5, 15., 13.75]);
        let alias = "wgt 0.5\nmodel Alias\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return scale(x,x)+div(x,x);}\n";
        let mut s = session(alias, compiled);
        let mut pass = s.forward(&[3.], 100, 65536).unwrap();
        close(pass.output(), &[10.]);
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[6.]);
    }
}

#[test]
fn hard_comparison_is_exact_on_finite_extremes_and_rejects_automatic_differentiation() {
    let source = r#"wgt 0.5
model Compare
input x:f32[2]
output y:f32[2]
parameter threshold:f32[2]=[-3.402823466e38,3.402823466e38]
fn compare(a,b){return ge(a,b);}
fn forward(x){return compare(x,threshold);}
"#;
    for compiled in [false, true] {
        let mut s = session(source, compiled);
        assert_eq!(s.run(&[f32::MAX, -f32::MAX], 100).unwrap(), &[1., 0.]);
        assert_eq!(s.evaluate(&[-f32::MAX, f32::MAX], 100).unwrap(), [1., 1.]);
        let error = s.forward(&[1., 2.], 100, 65536).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("ge does not support automatic differentiation")
        );
        assert!(s.run(&[f32::NAN, 0.], 100).is_err());
        assert!(s.run(&[f32::INFINITY, 0.], 100).is_err());
    }
}

#[test]
fn division_and_sqrt_reject_nonfinite_values_and_zero_derivative_paths_skip_singularities() {
    for compiled in [false, true] {
        let div = "wgt 0.5\nmodel Division\ninput x:f32[2]\noutput y:f32[2]\nfn forward(x){return div([1,1],x);}\n";
        let mut s = session(div, compiled);
        for x in [[0., 1.], [-0., 1.], [f32::from_bits(1), 1.]] {
            assert!(s.evaluate(&x, 100).is_err());
        }
        let sqrt = "wgt 0.5\nmodel Root\ninput x:f32[2]\noutput y:f32[2]\nfn forward(x){return sqrt(x);}\n";
        let mut s = session(sqrt, compiled);
        assert!(s.evaluate(&[-1., 4.], 100).is_err());
        let mut pass = s.forward(&[0., 4.], 100, 65536).unwrap();
        pass.backward(&[0., 1.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[0., 0.25]);
        assert!(
            pass.backward(&[1., 0.], &BTreeMap::new())
                .unwrap_err()
                .to_string()
                .contains("nonfinite")
        );
    }
}

#[test]
fn division_vjp_and_scalar_reduction_avoid_spurious_f32_intermediate_overflow() {
    let div = "wgt 0.5\nmodel Tiny\ninput x:f32[1]\noutput y:f32[1]\nparameter divisor:f32[1]=[1e-30]\nfn forward(x){return div(x,divisor);}\n";
    let scale = "wgt 0.5\nmodel Cancellation\ninput x:f32[2]\noutput y:f32[2]\nparameter scalar:f32[1]=[0]\nfn forward(x){return scale(x,scalar);}\n";
    for compiled in [false, true] {
        let mut s = session(div, compiled);
        let mut pass = s.forward(&[1e-30], 100, 65536).unwrap();
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[1e30]);
        close(pass.parameter_gradient("divisor").unwrap(), &[-1e30]);
        let mut s = session(scale, compiled);
        let mut pass = s.forward(&[1e30, -1e30], 100, 65536).unwrap();
        pass.backward(&[1e30, 1e30], &BTreeMap::new()).unwrap();
        assert_eq!(pass.parameter_gradient("scalar").unwrap(), &[0.]);
        assert_eq!(pass.input_gradient().unwrap(), &[0., 0.]);
    }
}

#[test]
fn new_operations_reject_shape_aliases_and_never_introduce_implicit_broadcast() {
    for expression in [
        "ge(x as [2,2],x)",
        "div(x as [2,2],x)",
        "div(x,[2])",
        "scale(x,[2,3])",
        "scale(x,[2] as [1,1])",
    ] {
        let source = format!(
            "wgt 0.5\nmodel Invalid\ninput x:f32[4]\noutput y:f32[4]\nfn forward(x){{return ({expression}) as [4];}}\n"
        );
        assert!(ModelDefinition::parse(&source).is_err(), "{expression}");
    }
}
