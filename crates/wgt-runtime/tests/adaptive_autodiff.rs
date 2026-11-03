use std::collections::BTreeMap;
use weightlang_runtime::adaptive::{parse, AdaptiveSession};

const EXAMPLE: &str = include_str!("../../../examples/self_modifying/model.wgt");
fn model(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 100_000).unwrap()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-4, "{actual:?} != {expected:?}");
    }
}

#[test]
fn differentiates_through_read_write_and_new_weight_use_without_committing() {
    let mut m = model(EXAMPLE);
    let mut pass = m.forward(&[2., 3.], 100, 4096).unwrap();
    close(pass.output(), &[3.]);
    close(pass.state("memory").unwrap(), &[1.]);
    close(m.parameter("memory").unwrap(), &[0.]);
    assert_eq!(m.revision(), 0);
    assert!(pass.input_gradient().is_err());
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &[1.5, 1.]);
    close(pass.parameter_gradient("updater").unwrap(), &[6.]);
    close(pass.parameter_gradient("memory").unwrap(), &[3.]);
    close(pass.parameter_gradient("router").unwrap(), &[0.]);
    // The final-state adjoint and output adjoint share the same updated weight.
    pass.backward(&[1.], &BTreeMap::from([("memory".into(), vec![2.])]))
        .unwrap();
    close(pass.input_gradient().unwrap(), &[2.5, 1.]);
    close(pass.parameter_gradient("updater").unwrap(), &[10.]);
    close(pass.parameter_gradient("memory").unwrap(), &[5.]);
    // A later mutable inference call cannot change an existing forward snapshot.
    m.run(&[4., 3.], 100).unwrap();
    pass.backward(&[2.], &BTreeMap::new()).unwrap();
    close(pass.parameter_gradient("updater").unwrap(), &[12.]);
    assert_eq!(m.revision(), 1);
}

#[test]
fn branches_only_differentiate_the_executed_path() {
    let mut m = model(EXAMPLE);
    for x in [-2., 0.] {
        let mut pass = m.forward(&[x, 3.], 100, 4096).unwrap();
        pass.backward(&[1.], &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &[0., 1.]);
        for name in ["memory", "updater", "router"] {
            close(pass.parameter_gradient(name).unwrap(), &[0.]);
        }
    }
}

#[test]
fn repeated_weight_writes_preserve_all_versions_through_loop() {
    let source = "wgt 0.5
model Test
input x: f32[1]
parameter memory: f32[1,1] = [1]
frozen policy: f32[1,1] = [1.1]
frozen counter: f32[1] = [3]
frozen decrement: f32[1] = [-1]
output y: f32[1]
fn forward(x) {
    let count = read(counter);
    let step = read(decrement);
    while true {
    let v = read(memory);
    v = linear(v,policy);
    write(memory, v);
    count = add(count,step);
    if positive(count) { } else { break; }
    }
    return v;
}";
    let mut m = model(source);
    let mut pass = m.forward(&[0.], 100, 4096).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    close(pass.output(), &[1.331]);
    close(pass.parameter_gradient("policy").unwrap(), &[3.63]);
    close(pass.parameter_gradient("memory").unwrap(), &[1.331]);
    close(pass.parameter_gradient("counter").unwrap(), &[0.]);
    close(pass.input_gradient().unwrap(), &[0.]);
    close(m.parameter("memory").unwrap(), &[1.]);
}

#[test]
fn overwritten_initial_parameter_is_disconnected_and_alias_gradients_sum() {
    let mut m = model(
        "wgt 0.5
model Test
input x: f32[1]
parameter memory: f32[1,1] = [99]
output y: f32[1]
fn forward(x) {
    write(memory, x);
    let y = linear(x,memory);
    y = add(y,y);
    return y;
}",
    );
    let mut pass = m.forward(&[3.], 20, 4096).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    close(pass.output(), &[18.]);
    close(pass.input_gradient().unwrap(), &[12.]);
    close(pass.parameter_gradient("memory").unwrap(), &[0.]);
}

#[test]
fn all_numeric_operators_match_central_differences() {
    let source = "wgt 0.5
model Test
input x: f32[2]
frozen w: f32[2,2] = [0.2,-0.1,0.3,0.4]
frozen b: f32[2] = [0.1,0.2]
output y: f32[1]
fn forward(x) {
    let y = linear(x,w,b);
    y = tanh(y);
    let z = sigmoid(y);
    y = mul(y,z);
    y = relu(y);
    y = add(y,z);
    let result = slice(y, 1, 1);
    return result;
}";
    let mut m = model(source);
    let input = [0.5, 0.7];
    let mut pass = m.forward(&input, 100, 4096).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    let eps = 0.001;
    for i in 0..2 {
        let mut x = input;
        x[i] += eps;
        let plus = m.forward(&x, 100, 4096).unwrap().output()[0];
        x[i] -= 2. * eps;
        let minus = m.forward(&x, 100, 4096).unwrap().output()[0];
        close(
            &[pass.input_gradient().unwrap()[i]],
            &[(plus - minus) / (2. * eps)],
        );
    }
    for name in ["w", "b"] {
        let original = m.parameter(name).unwrap().to_vec();
        for i in 0..original.len() {
            let mut v = original.clone();
            v[i] += eps;
            let plus = m
                .forward_with_parameters(
                    &input,
                    &BTreeMap::from([(name.into(), v.clone())]),
                    100,
                    4096,
                )
                .unwrap()
                .output()[0];
            v[i] -= 2. * eps;
            let minus = m
                .forward_with_parameters(&input, &BTreeMap::from([(name.into(), v)]), 100, 4096)
                .unwrap()
                .output()[0];
            close(
                &[pass.parameter_gradient(name).unwrap()[i]],
                &[(plus - minus) / (2. * eps)],
            );
        }
        close(m.parameter(name).unwrap(), &original);
    }
}

#[test]
fn tape_budget_errors_and_bindings_never_modify_session() {
    let mut m = model(EXAMPLE);
    let bytes = m.forward(&[2., 3.], 100, 4096).unwrap().tape_bytes();
    assert!(m.forward(&[2., 3.], 100, bytes).is_ok());
    assert!(m
        .forward(&[2., 3.], 100, bytes - 1)
        .err()
        .expect("expected tape budget error")
        .to_string()
        .contains("budget"));
    assert!(m.forward(&[2., 3.], 7, 4096).is_err());
    let bindings = BTreeMap::from([("updater".into(), vec![0.75])]);
    let pass = m
        .forward_with_parameters(&[2., 3.], &bindings, 100, 4096)
        .unwrap();
    close(pass.output(), &[4.5]);
    close(m.parameter("updater").unwrap(), &[0.5]);
    assert!(m
        .forward_with_parameters(&[2., 3.], &bindings, 100, 1)
        .is_err());
    close(m.parameter("updater").unwrap(), &[0.5]);
    close(m.parameter("memory").unwrap(), &[0.]);
    assert_eq!(m.revision(), 0);
}

#[test]
fn invalid_or_nonfinite_vjp_is_not_exposed_as_a_valid_gradient() {
    let mut m = model(EXAMPLE);
    let mut pass = m.forward(&[2., 3.], 100, 4096).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert!(pass.backward(&[f32::NAN], &BTreeMap::new()).is_err());
    assert!(pass.input_gradient().is_err());
    assert!(pass
        .backward(&[1.], &BTreeMap::from([("missing".into(), vec![1.])]))
        .is_err());
    assert!(pass
        .backward(&[1.], &BTreeMap::from([("memory".into(), vec![1., 2.])]))
        .is_err());
    let mut m=model("wgt 0.5
model Test
input x: f32[1]
frozen w: f32[1,1] = [1e-30]
output y: f32[1]
fn forward(x) {
    let y = linear(x,w);
    return y;
}");
    let mut pass = m.forward(&[1e30], 10, 4096).unwrap();
    assert!(pass.backward(&[1e30], &BTreeMap::new()).is_err());
    assert!(pass.parameter_gradient("w").is_err());
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("w").unwrap(), &[1e30]);
}

#[test]
fn relu_uses_zero_derivative_at_zero_and_negative_inputs() {
    let mut m = model("wgt 0.5
model Test
input x: f32[3]
output y: f32[3]
fn forward(x) {
    let y = relu(x);
    return y;
}");
    let mut pass = m.forward(&[-1., 0., 2.], 10, 4096).unwrap();
    pass.backward(&[1., 1., 1.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &[0., 0., 1.]);
}
