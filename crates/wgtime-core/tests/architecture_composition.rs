//! Small architecture compositions, not full ResNet or stateful RNN coverage.
//! Scalar f64 oracles are independent of runtime tensor operations and autograd.
use std::collections::BTreeMap;
use wgtime_core::{
    adaptive::{AdaptiveSession, parse},
    training::Optimizer,
};

const MEMORY: usize = 2 * 1024 * 1024;
const WORK: u64 = 100_000;
const STEPS: usize = 400;
type Weights = BTreeMap<String, Vec<f64>>;
type Reference = fn(&[f64], &Weights) -> Vec<f64>;

struct Case {
    source: &'static str,
    names: &'static [&'static str],
    probe: &'static [f32],
    reference: Reference,
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
}
fn residual_case() -> Case {
    let inputs = vec![vec![0.7, -0.3], vec![-0.4, 0.8], vec![0.2, 0.5]];
    let targets = inputs
        .iter()
        .map(|x| x.iter().map(|v| v * 0.5).collect())
        .collect();
    Case {
        source: include_str!("fixtures/residual_mlp/model.wgt"),
        names: &[
            "up0", "hidden0", "down0", "bias0", "up1", "hidden1", "down1", "bias1",
        ],
        probe: &[0.7, -0.3],
        reference: residual_reference,
        inputs,
        targets,
    }
}
fn recurrent_case() -> Case {
    let inputs = vec![
        vec![0.7, -0.3, 0.2, 0.6, -0.4, 0.5],
        vec![-0.4, 0.8, 0.5, -0.2, 0.3, -0.6],
        vec![0.2, 0.5, -0.6, 0.4, 0.7, 0.1],
    ];
    let targets = inputs
        .iter()
        .map(|x| vec![0.3 * x[2] + 0.1 * x[4], 0.3 * x[3] + 0.1 * x[5]])
        .collect();
    Case {
        source: include_str!("fixtures/recurrent_cell/model.wgt"),
        names: &["input_weight", "recurrent_weight", "bias"],
        probe: &[0.7, -0.3, 0.2, 0.6, -0.4, 0.5],
        reference: recurrent_reference,
        inputs,
        targets,
    }
}
fn autoencoder_case() -> Case {
    // A rank-two plane fits the two-dimensional bottleneck; target equals input.
    let inputs = vec![
        vec![0.6, 0.0, 0.3],
        vec![-0.6, 0.0, -0.3],
        vec![0.0, 0.6, -0.3],
        vec![0.0, -0.6, 0.3],
    ];
    Case {
        source: include_str!("fixtures/autoencoder/model.wgt"),
        names: &["encoder", "encoder_bias", "decoder", "decoder_bias"],
        probe: &[0.6, -0.2, 0.4],
        reference: autoencoder_reference,
        targets: inputs.clone(),
        inputs,
    }
}
fn session(case: &Case, compiled: bool) -> AdaptiveSession {
    let mut model = AdaptiveSession::prepare(parse(case.source).unwrap(), MEMORY).unwrap();
    if compiled {
        model.enable_compiled().unwrap();
    }
    model.set_optimizer(Optimizer::Sgd).unwrap();
    assert_eq!(model.is_compiled(), compiled);
    assert_eq!(
        model.execution_backend(),
        if compiled {
            "native-cpu"
        } else {
            "reference-interpreter"
        }
    );
    model
}
fn weights(model: &AdaptiveSession, names: &[&str]) -> Weights {
    names
        .iter()
        .map(|&name| {
            (
                name.to_owned(),
                model
                    .parameter(name)
                    .unwrap()
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect(),
            )
        })
        .collect()
}
// Row-major affine map, written as scalar loops without runtime helpers.
fn affine(x: &[f64], matrix: &[f64], bias: &[f64]) -> Vec<f64> {
    let mut y = bias.to_vec();
    for j in 0..y.len() {
        for i in 0..x.len() {
            y[j] += x[i] * matrix[i * bias.len() + j];
        }
    }
    y
}
fn residual_reference(x: &[f64], p: &Weights) -> Vec<f64> {
    let mut state = x.to_vec();
    for block in 0..2 {
        let h: Vec<_> = affine(
            &state,
            &p[&format!("up{block}")],
            &p[&format!("hidden{block}")],
        )
        .into_iter()
        .map(f64::tanh)
        .collect();
        let correction = affine(&h, &p[&format!("down{block}")], &p[&format!("bias{block}")]);
        for j in 0..2 {
            state[j] += correction[j];
        }
    }
    state
}
fn recurrent_reference(x: &[f64], p: &Weights) -> Vec<f64> {
    let mut hidden = x[4..6].to_vec();
    for observation in x[..4].chunks_exact(2) {
        let observed = affine(observation, &p["input_weight"], &p["bias"]);
        let carried = affine(&hidden, &p["recurrent_weight"], &[0.0, 0.0]);
        hidden = observed
            .iter()
            .zip(carried)
            .map(|(a, b)| (a + b).tanh())
            .collect();
    }
    hidden
}
fn autoencoder_reference(x: &[f64], p: &Weights) -> Vec<f64> {
    let h: Vec<_> = affine(x, &p["encoder"], &p["encoder_bias"])
        .into_iter()
        .map(f64::tanh)
        .collect();
    affine(&h, &p["decoder"], &p["decoder_bias"])
}
fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(a) - b).abs() <= tolerance,
            "element {i}: {a} vs {b}, tolerance {tolerance}"
        );
    }
}
fn dot(values: Vec<f64>, seed: &[f32]) -> f64 {
    values
        .iter()
        .zip(seed)
        .map(|(v, &s)| v * f64::from(s))
        .sum()
}
fn check_forward_and_gradients(case: Case) {
    let x: Vec<f64> = case.probe.iter().map(|&v| f64::from(v)).collect();
    for compiled in [false, true] {
        let mut model = session(&case, compiled);
        let p = weights(&model, case.names);
        let expected = (case.reference)(&x, &p);
        close(&model.evaluate(case.probe, WORK).unwrap(), &expected, 2e-6);
        let seed: Vec<f32> = (0..expected.len())
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.4 })
            .collect();
        let mut pass = model.forward(case.probe, WORK, MEMORY).unwrap();
        pass.backward(&seed, &BTreeMap::new()).unwrap();
        close(pass.output(), &expected, 2e-6);
        // Central differences use the independent f64 oracle, including both
        // uses of shared RNN weights and the caller-provided initial hidden.
        let epsilon = 1e-5;
        for &name in case.names {
            for i in 0..p[name].len() {
                let mut plus = p.clone();
                plus.get_mut(name).unwrap()[i] += epsilon;
                let mut minus = p.clone();
                minus.get_mut(name).unwrap()[i] -= epsilon;
                let numerical = (dot((case.reference)(&x, &plus), &seed)
                    - dot((case.reference)(&x, &minus), &seed))
                    / (2.0 * epsilon);
                let analytical = pass.parameter_gradient(name).unwrap()[i];
                assert!(
                    (f64::from(analytical) - numerical).abs() < 2e-5,
                    "{name}[{i}]: {analytical} vs {numerical}"
                );
            }
        }
        for i in 0..x.len() {
            let mut plus = x.clone();
            plus[i] += epsilon;
            let mut minus = x.clone();
            minus[i] -= epsilon;
            let numerical = (dot((case.reference)(&plus, &p), &seed)
                - dot((case.reference)(&minus, &p), &seed))
                / (2.0 * epsilon);
            assert!(
                (f64::from(pass.input_gradient().unwrap()[i]) - numerical).abs() < 2e-5,
                "input[{i}]: oracle {numerical}"
            );
        }
        assert_eq!(
            model.compiled_forward_counts(),
            if compiled { (2, 1) } else { (0, 0) }
        );
        if case.probe.len() == 6 {
            assert_eq!(
                model.program().parameters.len(),
                3,
                "RNN must own just one parameter set"
            );
            assert!(
                pass.input_gradient().unwrap()[4..6]
                    .iter()
                    .all(|g| g.abs() > 1e-4),
                "both initial hidden channels must affect output"
            );
            let mut reset = case.probe.to_vec();
            reset[4..6].fill(0.0);
            let altered = model.evaluate(&reset, WORK).unwrap();
            assert!(
                altered
                    .iter()
                    .zip(&expected)
                    .any(|(&a, &b)| (f64::from(a) - b).abs() > 1e-3)
            );
            // Interleaving a different call cannot persist a hidden state.
            close(&model.evaluate(case.probe, WORK).unwrap(), &expected, 2e-6);
        }
    }
}
fn mse(model: &mut AdaptiveSession, case: &Case) -> f32 {
    case.inputs
        .iter()
        .zip(&case.targets)
        .map(|(x, target)| {
            model
                .evaluate(x, WORK)
                .unwrap()
                .iter()
                .zip(target)
                .map(|(y, t)| (y - t).powi(2))
                .sum::<f32>()
                / target.len() as f32
        })
        .sum::<f32>()
        / case.inputs.len() as f32
}
fn check_training_and_roundtrip(case: Case) {
    let mut outcomes = Vec::new();
    for compiled in [false, true] {
        let mut model = session(&case, compiled);
        let before = mse(&mut model, &case);
        assert!(before > 1e-5, "nontrivial initial training error required");
        let counts = model.compiled_forward_counts();
        for _ in 0..STEPS {
            let mut remaining = WORK;
            let loss = model
                .train_custom(&case.inputs, &case.targets, 0.15, &mut remaining, MEMORY)
                .unwrap();
            assert!(loss.is_finite());
        }
        let after_counts = model.compiled_forward_counts();
        let tapes = (STEPS * case.inputs.len()) as u64;
        assert_eq!(
            (after_counts.0 - counts.0, after_counts.1 - counts.1),
            if compiled { (tapes, tapes) } else { (0, 0) }
        );
        let after = mse(&mut model, &case);
        assert!(after < before * 0.5, "training MSE {before} -> {after}");
        let bytes = model.weights_bytes(MEMORY).unwrap();
        let mut restored = session(&case, compiled);
        restored.bind_weights(&bytes, "sgd").unwrap();
        for &name in case.names {
            assert_eq!(
                restored.parameter(name).unwrap(),
                model.parameter(name).unwrap()
            );
        }
        for input in case
            .inputs
            .iter()
            .map(Vec::as_slice)
            .chain(std::iter::once(case.probe))
        {
            let output = model.evaluate(input, WORK).unwrap();
            assert_eq!(restored.evaluate(input, WORK).unwrap(), output);
        }
        assert_eq!(
            restored.compiled_forward_counts(),
            if compiled {
                ((case.inputs.len() + 1) as u64, 0)
            } else {
                (0, 0)
            }
        );
        // Reloaded SGD must continue from identical weights, not initial values.
        for instance in [&mut model, &mut restored] {
            let mut remaining = WORK;
            instance
                .train_custom(&case.inputs, &case.targets, 0.05, &mut remaining, MEMORY)
                .unwrap();
        }
        for &name in case.names {
            assert_eq!(
                restored.parameter(name).unwrap(),
                model.parameter(name).unwrap()
            );
        }
        outcomes.push(weights(&model, case.names));
        eprintln!("compiled={compiled}, architecture MSE {before} -> {after}");
    }
    for &name in case.names {
        for (a, b) in outcomes[0][name].iter().zip(&outcomes[1][name]) {
            assert!(
                (a - b).abs() < 2e-5,
                "backend training disagreement for {name}: {a} vs {b}"
            );
        }
    }
}
#[test]
fn residual_mlp_reference_and_finite_differences() {
    check_forward_and_gradients(residual_case());
}
#[test]
fn recurrent_cell_shared_weights_and_initial_hidden_finite_differences() {
    check_forward_and_gradients(recurrent_case());
}
#[test]
fn autoencoder_reference_and_finite_differences() {
    check_forward_and_gradients(autoencoder_case());
}
#[test]
fn residual_mlp_training_and_weights_roundtrip() {
    check_training_and_roundtrip(residual_case());
}
#[test]
fn recurrent_cell_training_and_weights_roundtrip() {
    check_training_and_roundtrip(recurrent_case());
}
#[test]
fn autoencoder_training_and_weights_roundtrip() {
    check_training_and_roundtrip(autoencoder_case());
}
