//! Learn a numerical algorithm, not a stack of independently weighted layers.
//!
//! The shared cell is Richardson relaxation for h - observation = 0, initialized
//! from a caller-provided estimate. The teacher requests three half-residual
//! corrections, expressed independently as initial + 7/8 * residual. With gain
//! bounded above by 1/2, neither one nor two iterations can represent the teacher
//! for a nonzero residual. Training must enable the solver and select depth 3.
//! Integer/bool hard primals use STE surrogate gradients, not true derivatives
//! of rounding/comparisons. This exercises the service's core execution engine;
//! it does not claim to test the IPC facade or a released language capability.

use wgtime_core::{
    adaptive::{self, AdaptiveSession},
    training::Optimizer,
};

const SOURCE: &str = include_str!("fixtures/differentiable_program/model.wgt");
const MEMORY: usize = 8 * 1024 * 1024;
const WORK: u64 = 100_000;
const TRAIN_STEPS: usize = 96;
const INPUTS: [[f32; 2]; 4] = [[-2.0, 0.0], [2.0, 0.0], [-0.5, 0.5], [1.0, 2.0]];
const HELD_OUT: [[f32; 2]; 4] = [[3.0, -1.0], [-1.25, 2.5], [0.125, -0.25], [4.0, 4.0]];

fn session(compiled: bool) -> AdaptiveSession {
    let mut model = AdaptiveSession::prepare(adaptive::parse(SOURCE).unwrap(), MEMORY).unwrap();
    if compiled {
        // Failure must fail the test, never silently select the interpreter.
        model.enable_compiled().unwrap();
    }
    model.set_optimizer(Optimizer::Sgd).unwrap();
    assert_backend(&model, compiled);
    model
}

fn assert_backend(model: &AdaptiveSession, compiled: bool) {
    assert_eq!(model.is_compiled(), compiled);
    assert_eq!(
        model.execution_backend(),
        if compiled { "native-cpu" } else { "reference-interpreter" }
    );
}

fn scalar(model: &AdaptiveSession, name: &str) -> f32 {
    model.parameter(name).unwrap()[0]
}

fn near(actual: f32, expected: f32) {
    let tolerance = 2e-6 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

// Independent closed form, with no recurrence or WGT execution in the oracle.
fn teacher([observation, initial]: [f32; 2]) -> f32 {
    initial + 0.875 * (observation - initial)
}

fn reference(input: [f32; 2], gain: f32, depth_shadow: f32, enabled_shadow: f32) -> f32 {
    let [observation, initial] = input;
    if enabled_shadow < 0.5 {
        initial
    } else {
        initial + (1.0 - (1.0 - gain).powi(depth_shadow.round() as i32))
            * (observation - initial)
    }
}

fn half_mse(model: &mut AdaptiveSession, inputs: &[[f32; 2]]) -> f32 {
    inputs.iter().map(|&input| {
        let prediction = model.run(&input, WORK).unwrap()[0];
        0.5 * (prediction - teacher(input)).powi(2)
    }).sum::<f32>() / inputs.len() as f32
}

#[test]
fn hard_forward_matches_closed_form_including_shadow_thresholds() {
    for compiled in [false, true] {
        let mut model = session(compiled);
        let mut calls = 0;
        for gain in [0.1, 0.25, 0.5] {
            for depth in [1.0, 1.49, 1.5, 2.49, 2.5, 3.0] {
                for enabled in [0.0, 0.49, 0.5, 1.0] {
                    model.replace_parameter("gain", &[gain]).unwrap();
                    model.replace_parameter("depth", &[depth]).unwrap();
                    model.replace_parameter("enabled", &[enabled]).unwrap();
                    for input in INPUTS.into_iter().chain(HELD_OUT) {
                        near(model.run(&input, WORK).unwrap()[0], reference(input, gain, depth, enabled));
                        calls += 1;
                    }
                }
            }
        }
        // Counts must account for every native call; no uncounted interpreter fallback.
        assert_eq!(model.compiled_forward_counts(), if compiled { (calls, 0) } else { (0, 0) });
    }
}

#[test]
fn training_learns_gain_switch_and_required_depth_and_reloads_shadows() {
    let inputs: Vec<Vec<f32>> = INPUTS.iter().map(|input| input.to_vec()).collect();
    let targets: Vec<Vec<f32>> = INPUTS.iter().map(|&input| vec![teacher(input)]).collect();
    let mut learned = Vec::new();

    for compiled in [false, true] {
        let mut model = session(compiled);
        assert_eq!(scalar(&model, "gain"), 0.25);
        assert_eq!(scalar(&model, "depth"), 1.0);
        assert_eq!(scalar(&model, "enabled"), 0.0);
        let before = half_mse(&mut model, &INPUTS);
        let counts_before_training = model.compiled_forward_counts();
        let mut first_loss = None;
        for _ in 0..TRAIN_STEPS {
            let mut remaining = WORK;
            let loss = model.train_custom(&inputs, &targets, 0.2, &mut remaining, MEMORY).unwrap();
            assert!(loss.is_finite());
            first_loss.get_or_insert(loss);
        }
        near(first_loss.unwrap(), before);
        let counts_after_training = model.compiled_forward_counts();
        let expected_tapes = (TRAIN_STEPS * INPUTS.len()) as u64;
        assert_eq!(
            (counts_after_training.0 - counts_before_training.0,
             counts_after_training.1 - counts_before_training.1),
            if compiled { (expected_tapes, expected_tapes) } else { (0, 0) }
        );

        let after = half_mse(&mut model, &INPUTS);
        let held_out = half_mse(&mut model, &HELD_OUT);
        let gain = scalar(&model, "gain");
        let depth = scalar(&model, "depth");
        let enabled = scalar(&model, "enabled");
        eprintln!("compiled={compiled}: half_mse {before} -> {after}, held_out={held_out}, gain={gain}, depth_shadow={depth}, enabled_shadow={enabled}");
        assert!(after < before * 1e-4 && after < 1e-7, "loss {before} -> {after}");
        assert!(held_out < 1e-7, "held-out half MSE {held_out}");
        near(gain, 0.5);
        assert_eq!(depth.round(), 3.0, "depth shadow {depth}");
        assert!(enabled >= 0.5, "enabled shadow {enabled}");
        assert_backend(&model, compiled);

        // Ablations give each learned control an observable algorithmic role.
        model.replace_parameter("enabled", &[0.0]).unwrap();
        near(half_mse(&mut model, &INPUTS), before);
        model.replace_parameter("enabled", &[enabled]).unwrap();
        model.replace_parameter("depth", &[2.0]).unwrap();
        assert!(half_mse(&mut model, &INPUTS) > 0.01);
        model.replace_parameter("depth", &[depth]).unwrap();
        model.replace_parameter("gain", &[0.25]).unwrap();
        assert!(half_mse(&mut model, &INPUTS) > 0.05);
        model.replace_parameter("gain", &[gain]).unwrap();

        // Save/reload real weight-artifact bytes without leaving disposable files.
        // Exact shadow equality matters: saving only hard int/bool primals would
        // change subsequent surrogate optimization after reload.
        let bytes = model.weights_bytes(MEMORY).unwrap();
        let mut restored = session(compiled);
        restored.bind_weights(&bytes, "sgd").unwrap();
        for name in ["gain", "depth", "enabled"] {
            assert_eq!(restored.parameter(name).unwrap(), model.parameter(name).unwrap());
        }
        for input in HELD_OUT {
            near(restored.run(&input, WORK).unwrap()[0], teacher(input));
        }
        assert_backend(&restored, compiled);
        assert_eq!(restored.compiled_forward_counts(), if compiled { (HELD_OUT.len() as u64, 0) } else { (0, 0) });

        // The engine checkpoint additionally preserves optimizer state. Restore,
        // then make a nonzero update on both instances to check continuation.
        let checkpoint = serde_json::from_slice(&model.checkpoint_bytes_bounded(MEMORY).unwrap()).unwrap();
        let mut resumed = AdaptiveSession::restore_checkpoint(checkpoint, MEMORY).unwrap();
        if compiled {
            resumed.enable_compiled().unwrap();
        }
        let adjusted_targets: Vec<Vec<f32>> = targets.iter().map(|row| vec![row[0] * 0.9]).collect();
        for instance in [&mut model, &mut resumed] {
            let mut remaining = WORK;
            instance.train_custom(&inputs, &adjusted_targets, 0.02, &mut remaining, MEMORY).unwrap();
        }
        for name in ["gain", "depth", "enabled"] {
            assert_eq!(resumed.parameter(name).unwrap(), model.parameter(name).unwrap());
        }
        assert!(scalar(&resumed, "gain") < gain, "continuation must update the gain");
        assert_backend(&resumed, compiled);
        assert_eq!(resumed.compiled_forward_counts().1, if compiled { INPUTS.len() as u64 } else { 0 });
        learned.push([gain, depth, enabled]);
    }
    for (reference, native) in learned[0].iter().zip(&learned[1]) {
        near(*native, *reference);
    }
}
