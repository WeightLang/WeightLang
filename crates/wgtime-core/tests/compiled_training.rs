//! Native forwards must retain the same value-version tape as the reference executor.
use std::collections::BTreeMap;
use wgtime_core::adaptive::{AdaptiveSession, parse};

fn instance(source: &str, compiled: bool) -> AdaptiveSession {
    let mut session = AdaptiveSession::prepare(parse(source).unwrap(), 1024 * 1024).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    session
}

#[test]
fn compiled_forward_tape_matches_reference_and_uses_current_parameters() {
    // x is a two-feature vector; the scalar reduction makes each feature's
    // contribution visible to both the input and parameter adjoints.
    let source = "wgt 0.5\nmodel Product\ninput x:f32[2]\noutput y:f32[1]\nparameter w:f32[2]=[2,3]\nfn forward(x){let y=x*w;return sum(y);}\n";
    let mut reference = instance(source, false);
    let mut native = instance(source, true);
    for weights in [[2., 3.], [4., -1.]] {
        reference.replace_parameter("w", &weights).unwrap();
        native.replace_parameter("w", &weights).unwrap();
        assert_eq!(
            reference.evaluate(&[3., 5.], 100).unwrap(),
            native.evaluate(&[3., 5.], 100).unwrap()
        );
        let mut expected = reference.forward(&[3., 5.], 100, 65536).unwrap();
        let mut actual = native.forward(&[3., 5.], 100, 65536).unwrap();
        assert_eq!(expected.output(), actual.output());
        expected.backward(&[1.], &BTreeMap::new()).unwrap();
        actual.backward(&[1.], &BTreeMap::new()).unwrap();
        assert_eq!(
            expected.input_gradient().unwrap(),
            actual.input_gradient().unwrap()
        );
        assert_eq!(actual.parameter_gradient("w").unwrap(), &[3., 5.]);
        assert_eq!(native.execution_backend(), "native-cpu");
    }
    assert!(native.forward(&[3., 5.], 1, 65536).is_err());
    assert_eq!(native.parameter("w").unwrap(), &[4., -1.]);
}

#[test]
fn compiled_gate_is_lazy_for_prediction_and_complete_for_training() {
    let source = "wgt 0.5\nmodel Gated\ninput x:f32[1]\noutput y:f32[1]\nfrozen p:prob=0\nfn explode(a){return square(a);}\nfn forward(x){return gate(p,explode(x));}\n";
    let mut native = instance(source, true);
    assert_eq!(native.evaluate(&[f32::MAX], 100).unwrap(), vec![0.]);
    assert!(native.forward(&[f32::MAX], 100, 65536).is_err());
    let mut pass = native.forward(&[2.], 100, 65536).unwrap();
    pass.backward(&[1.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[0.]);
    assert_eq!(pass.parameter_gradient("p").unwrap(), &[4.]);
}

#[test]
fn compiled_custom_loss_training_matches_reference_update() {
    let source = "wgt 0.5\nmodel Scale\ninput x:f32[1]\noutput y:f32[1]\nparameter w:f32[1]=[2]\nfn forward(x){return x*w;}\nfn loss(prediction,target){return square(prediction-target).mean;}\n";
    let mut reference = instance(source, false);
    let mut native = instance(source, true);
    for session in [&mut reference, &mut native] {
        session
            .set_optimizer(wgtime_core::training::Optimizer::Sgd)
            .unwrap();
        let loss = session
            .train_custom(&[vec![1.]], &[vec![0.]], 0.1, &mut 1000, 65536)
            .unwrap();
        assert!((loss - 4.).abs() < 1e-6);
        assert!((session.parameter("w").unwrap()[0] - 1.6).abs() < 1e-6);
    }
    assert_eq!(
        reference.parameter("w").unwrap(),
        native.parameter("w").unwrap()
    );
}

#[test]
fn compiled_initialization_rejects_unreplaced_stateful_operations() {
    let source = "wgt 0.5\nmodel Stateful\ninput x:f32[1]\noutput y:f32[1]\nparameter w:f32[1]=[0]\nfn forward(x){write(w,x);return x;}\n";
    let mut session = instance(source, false);
    assert!(session.enable_compiled().is_err());
    assert!(!session.is_compiled());
    assert_eq!(session.parameter("w").unwrap(), &[0.]);
}
