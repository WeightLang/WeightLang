use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::AdaptiveSession;

const SOURCE: &str = include_str!("../../../examples/native_training/explicit_hidden/model.wgt");
const MEMORY: usize = 1024 * 1024;
fn session(source: &str) -> AdaptiveSession {
    let definition = ModelDefinition::parse(source).unwrap();
    let ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
        panic!()
    };
    AdaptiveSession::prepare(program, MEMORY).unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < tolerance, "{a} != {b}");
    }
}

#[test]
fn packed_forward_vjp_handles_both_inputs_and_zero_hidden_seed() {
    let mut model = session(SOURCE);
    // Packed [x | hidden] and [prediction | next_hidden], in schema order.
    let input = [2., -4., 3., 5.];
    let mut pass = model.forward(&input, 100_000, MEMORY).unwrap();
    close(pass.output(), &[4., 3., 3.5, -1.5], 1e-6);
    pass.backward(&[1., 2., 0., 0.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &[0.5, 1., 1., 2.], 1e-6);
    close(pass.parameter_gradient("gain").unwrap(), &[2., -8.], 1e-6);

    let seed = [0.7, -0.3, 0.4, 0.9];
    let mut pass = model.forward(&input, 100_000, MEMORY).unwrap();
    pass.backward(&seed, &BTreeMap::new()).unwrap();
    let dot = |output: &[f32]| output.iter().zip(seed).map(|(a, b)| a * b).sum::<f32>();
    let epsilon = 0.001;
    for i in 0..input.len() {
        let mut plus = input;
        let mut minus = input;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        let high = dot(model.run(&plus, 100_000).unwrap());
        let low = dot(model.run(&minus, 100_000).unwrap());
        assert!((pass.input_gradient().unwrap()[i] - (high - low) / (2. * epsilon)).abs() < 0.001);
    }
    for i in 0..2 {
        let mut plus = [0.5, 0.5];
        let mut minus = plus;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        model.replace_parameter("gain", &plus).unwrap();
        let high = dot(model.run(&input, 100_000).unwrap());
        model.replace_parameter("gain", &minus).unwrap();
        let low = dot(model.run(&input, 100_000).unwrap());
        model.replace_parameter("gain", &[0.5, 0.5]).unwrap();
        assert!(
            (pass.parameter_gradient("gain").unwrap()[i] - (high - low) / (2. * epsilon)).abs()
                < 0.001
        );
    }
}

#[test]
fn repeated_output_accumulates_gradients_and_matrix_inputs_restore_axes() {
    let source = "wgt 0.5\nmodel Repeat\nfn forward(x: f32[2,1], h: f32[1,2]) -> (f32[2,2], f32[2,2]) {\nlet y = x @ h;\nreturn (y, y);\n}\n";
    let mut model = session(source);
    let mut pass = model.forward(&[2., 3., 4., 5.], 100_000, MEMORY).unwrap();
    close(pass.output(), &[8., 10., 12., 15., 8., 10., 12., 15.], 1e-6);
    pass.backward(&[1., 1., 1., 1., 2., 2., 2., 2.], &BTreeMap::new())
        .unwrap();
    close(pass.input_gradient().unwrap(), &[27., 27., 15., 15.], 1e-6);
}

#[test]
fn branch_returns_share_one_ordered_output_contract() {
    let source = "wgt 0.5\nmodel Branch\nfn forward(x: f32[1], h: f32[1]) -> (f32[1], f32[1]) {\nif x.mean > 0 { return (x,h); } else { return (h,x); }\n}\n";
    let mut model = session(source);
    for (input, expected, gradient) in [
        ([2., 3.], [2., 3.], [1., 0.]),
        ([-2., 3.], [3., -2.], [0., 1.]),
    ] {
        let mut pass = model.forward(&input, 100_000, MEMORY).unwrap();
        close(pass.output(), &expected, 1e-6);
        pass.backward(&[1., 0.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &gradient, 1e-6);
    }
}

#[test]
fn singleton_tuple_returns_one_logical_output_with_input_gradient() {
    let source = "wgt 0.5\nmodel Singleton\nfn forward(x: f32[2], h: f32[2]) -> (f32[2],) {\nreturn (x*h,);\n}\n";
    let mut model = session(source);
    let mut pass = model.forward(&[2., 3., 4., 5.], 100_000, MEMORY).unwrap();
    close(pass.output(), &[8., 15.], 1e-6);
    pass.backward(&[1., 2.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &[4., 10., 2., 6.], 1e-6);
}
