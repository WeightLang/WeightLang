use wgtime_core::adaptive::{AdaptiveSession, parse};

const MODEL: &str = include_str!("fixtures/cnn_classifier/model.wgt");

fn session(native: bool) -> AdaptiveSession {
    let mut session = AdaptiveSession::prepare(parse(MODEL).unwrap(), 4 * 1024 * 1024).unwrap();
    if native {
        session.enable_compiled().unwrap();
    }
    session
}

fn mse(values: &[f32], target: &[f32]) -> f32 {
    values.iter().zip(target).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / values.len() as f32
}

#[test]
fn cnn_classifier_interpreter_and_compiled_forward_match() {
    let input: Vec<f32> = (0..64).map(|i| ((i * 11 % 23) as f32 - 11.0) / 13.0).collect();
    let mut interpreted = session(false);
    let mut compiled = session(true);

    let expected = interpreted.evaluate(&input, 2_000_000).unwrap();
    let actual = compiled.evaluate(&input, 2_000_000).unwrap();
    assert_eq!(expected.len(), 3);
    assert!(expected.iter().all(|v| v.is_finite()));
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 1e-6, "output[{index}]: {a} != {b}");
    }
    assert!(compiled.compiled_forward_counts().0 > 0);
}

#[test]
fn cnn_classifier_conv_and_classifier_weights_receive_gradients() {
    let input: Vec<f32> = (0..64).map(|i| ((i * 5 % 19) as f32 - 9.0) / 10.0).collect();
    let mut model = session(false);
    let output = model.evaluate(&input, 2_000_000).unwrap();
    let target = vec![0.2, -0.1, 0.3];
    let seed: Vec<_> = output.iter().zip(&target).map(|(a, b)| 2.0 * (a - b) / 3.0).collect();

    let mut pass = model.forward(&input, 2_000_000, 4 * 1024 * 1024).unwrap();
    pass.backward(&seed, &Default::default()).unwrap();
    for name in ["conv1_kernel", "conv2_kernel", "classifier_weight"] {
        let gradient = pass.parameter_gradient(name).unwrap();
        assert!(gradient.iter().any(|g| g.abs() > 1e-9), "{name} received no gradient");
        assert!(gradient.iter().all(|g| g.is_finite()), "{name} has nonfinite gradient");
    }
}

#[test]
fn cnn_classifier_can_train_with_native_cpu_forward() {
    let input: Vec<f32> = (0..64).map(|i| ((i * 3 % 17) as f32 - 8.0) / 10.0).collect();
    let target = vec![0.1, -0.2, 0.25];
    let mut model = session(true);
    model.set_optimizer(wgtime_core::training::Optimizer::Sgd).unwrap();
    let before = mse(&model.evaluate(&input, 2_000_000).unwrap(), &target);
    for _ in 0..80 {
        model.train_custom(&[input.clone()], &[target.clone()], 0.03, &mut 2_000_000, 4 * 1024 * 1024).unwrap();
    }
    let after = mse(&model.evaluate(&input, 2_000_000).unwrap(), &target);
    assert!(after < before, "CNN training did not reduce MSE: {before} -> {after}");
}
