use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::{adaptive::AdaptiveSession, training::Optimizer};

const SOURCE: &str = include_str!("../../../examples/native_training/shared_recurrent/model.wgt");
const BUDGET: usize = 8 * 1024 * 1024;
const WORK: u64 = 1_000_000;
const INPUT: [f32; 6] = [0.7, -0.4, -0.2, 0.6, 0.3, -0.5];
const TARGET: [f32; 2] = [0.15, -0.25];
const WEIGHT_IH: [f32; 12] = [
    0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3, 0.6, 0.2, -0.4, 0.3,
];
const WEIGHT_HH: [f32; 12] = [
    -0.3, 0.2, 0.1, -0.4, 0.5, 0.3, -0.2, 0.6, 0.4, -0.5, 0.2, 0.1,
];
const BIAS_IH: [f32; 6] = [0.11, -0.07, 0.03, 0.13, -0.17, 0.19];
const BIAS_HH: [f32; 6] = [-0.05, 0.09, -0.12, 0.08, 0.23, -0.21];

fn session(compiled: bool) -> AdaptiveSession {
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
        panic!("shared recurrence must lower to a differentiable forward plan")
    };
    assert_eq!(
        program.parameters.len(),
        4,
        "steps must share parameter storage"
    );
    let mut session = AdaptiveSession::prepare(program, BUDGET).unwrap();
    for (name, values) in [
        ("weight_ih", WEIGHT_IH.as_slice()),
        ("weight_hh", WEIGHT_HH.as_slice()),
        ("bias_ih", BIAS_IH.as_slice()),
        ("bias_hh", BIAS_HH.as_slice()),
    ] {
        session.replace_parameter(name, values).unwrap();
    }
    session.set_optimizer(Optimizer::Sgd).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    session
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-6, "element {i}: {a} != {b}");
    }
}

fn mse(output: &[f32]) -> f32 {
    output
        .iter()
        .zip(TARGET)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / 2.
}

#[test]
fn shared_gru_interpreted_and_compiled_match_and_train() {
    let mut interpreted = session(false);
    let mut compiled = session(true);
    let initial = interpreted.run(&INPUT, WORK).unwrap().to_vec();
    // Independently evaluated reset-after-projection formula, with nonzero b_hn.
    // h1=[0.38738998,-0.40882076], h2=[0.26216899,0.01957080].
    close(&initial, &[0.262169, 0.019570801]);
    close(compiled.run(&INPUT, WORK).unwrap(), &initial);
    // Explicit hidden is part of the input; inference does not retain it.
    let alternate = [-0.6, 0.8, 0.5, -0.3, -0.4, 0.2];
    let other = interpreted.run(&alternate, WORK).unwrap().to_vec();
    close(compiled.run(&alternate, WORK).unwrap(), &other);
    assert!(
        initial
            .iter()
            .zip(&other)
            .any(|(a, b)| (a - b).abs() > 1e-3)
    );
    close(interpreted.run(&INPUT, WORK).unwrap(), &initial);

    for step in 0..40 {
        let mut losses = Vec::new();
        for model in [&mut interpreted, &mut compiled] {
            let before = mse(model.run(&INPUT, WORK).unwrap());
            let mut remaining = WORK;
            let loss = model
                .train_custom(
                    &[INPUT.to_vec()],
                    &[TARGET.to_vec()],
                    0.05,
                    &mut remaining,
                    BUDGET,
                )
                .unwrap();
            assert!(
                (loss - before).abs() < 2e-6,
                "step {step}: custom loss mismatch"
            );
            losses.push(loss);
        }
        close(&losses[..1], &losses[1..]);
        close(
            compiled.run(&INPUT, WORK).unwrap(),
            interpreted.run(&INPUT, WORK).unwrap(),
        );
        for (left, right) in interpreted
            .program()
            .parameters
            .iter()
            .zip(&compiled.program().parameters)
        {
            assert_eq!(left.name, right.name);
            close(&left.data, &right.data);
        }
    }
    let trained = interpreted.run(&INPUT, WORK).unwrap().to_vec();
    assert!(
        mse(&trained) < mse(&initial) * 0.5,
        "training must materially reduce loss"
    );
    for model in [&mut interpreted, &mut compiled] {
        let weights = model.weights_bytes(BUDGET).unwrap();
        let mut restored = session(false);
        restored.bind_weights(&weights, "sgd").unwrap();
        close(restored.run(&INPUT, WORK).unwrap(), &trained);
    }
}
