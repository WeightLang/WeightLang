use std::path::Path;
use weightlang_runtime::{
    training::{mlp, Objective, Optimizer, Trainer},
    Layer, Model, Parameter, Value,
};

fn trainer(optimizer: Optimizer) -> Trainer {
    Trainer::new(
        mlp(&[1, 4, 1], "tanh", 42).unwrap(),
        Path::new("."),
        1_000_000,
        optimizer,
    )
    .unwrap()
}
#[test]
fn analytic_gradients_match_finite_differences() {
    for activation in ["relu", "tanh", "sigmoid", "identity"] {
        let mut t = Trainer::new(
            mlp(&[2, 3, 1], activation, 5).unwrap(),
            Path::new("."),
            1_000_000,
            Optimizer::Sgd,
        )
        .unwrap();
        // Keep finite differences away from ReLU's nondifferentiable zero.
        if activation == "relu" {
            t.session
                .write_parameters(&[weightlang_runtime::ParameterWrite {
                    name: "layer0.bias",
                    start: 0,
                    values: &[2., -2., 2.],
                }])
                .unwrap();
        }
        let inputs = vec![vec![0.3, -0.2], vec![0.1, 0.7]];
        let targets = vec![vec![0.5], vec![-0.1]];
        let (_, grad) = t.gradients(&inputs, Objective::Mse(&targets)).unwrap();
        let names: Vec<_> = t
            .session
            .parameter_info()
            .iter()
            .map(|p| (p.name.to_string(), p.shape.iter().product::<usize>()))
            .collect();
        for (i, (name, n)) in names.iter().enumerate() {
            for j in 0..*n {
                let mut original = [0.];
                t.session.read_parameter(name, j, &mut original).unwrap();
                let mut losses = Vec::new();
                for delta in [-0.001, 0.001] {
                    let v = [original[0] + delta];
                    t.session
                        .write_parameters(&[weightlang_runtime::ParameterWrite {
                            name,
                            start: j,
                            values: &v,
                        }])
                        .unwrap();
                    losses.push(t.gradients(&inputs, Objective::Mse(&targets)).unwrap().0);
                }
                t.session
                    .write_parameters(&[weightlang_runtime::ParameterWrite {
                        name,
                        start: j,
                        values: &original,
                    }])
                    .unwrap();
                let numerical = (losses[1] - losses[0]) / 0.002;
                assert!(
                    (numerical - grad[i][j]).abs() < 0.002,
                    "{activation} {name} {j}: {numerical} != {}",
                    grad[i][j]
                );
            }
        }
    }
}
#[test]
fn shared_parameters_accumulate_and_frozen_weights_stay_fixed() {
    let model = Model {
        output_type: Default::default(),
        version: 1,
        input: 1,
        parameters: vec![
            Parameter {
                name: "w".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::Resident { data: vec![2.] },
            },
            Parameter {
                name: "alias".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::View {
                    source: "w".into(),
                    start: 0,
                },
            },
        ],
        layers: vec![
            Layer::Linear {
                weight: "w".into(),
                bias: None,
            },
            Layer::Linear {
                weight: "alias".into(),
                bias: None,
            },
        ],
    };
    let mut t = Trainer::new(model, Path::new("."), 10000, Optimizer::Sgd).unwrap();
    let x = vec![vec![1.]];
    let y = vec![vec![0.]];
    let (_, g) = t.gradients(&x, Objective::Mse(&y)).unwrap();
    assert_eq!(g[0][0], 32.);
    t.step(&x, Objective::Mse(&y), 0.01).unwrap();
    assert!((t.predict(&[1.]).unwrap()[0] - 1.68 * 1.68).abs() < 1e-5);
    let mut model = mlp(&[1, 1], "identity", 1).unwrap();
    model.parameters[0].frozen = true;
    let mut t = Trainer::new(model, Path::new("."), 10000, Optimizer::Adam).unwrap();
    let mut before = [0.];
    t.session
        .read_parameter("layer0.weight", 0, &mut before)
        .unwrap();
    t.step(&x, Objective::Mse(&y), 0.1).unwrap();
    let mut after = [0.];
    t.session
        .read_parameter("layer0.weight", 0, &mut after)
        .unwrap();
    assert_eq!(before, after);
}
#[test]
fn sgd_and_adam_train_and_invalid_batches_do_not_mutate() {
    for optimizer in [Optimizer::Sgd, Optimizer::Adam] {
        let mut t = trainer(optimizer);
        let x = vec![vec![-1.], vec![1.]];
        let y = vec![vec![-1.], vec![3.]];
        let initial = t.gradients(&x, Objective::Mse(&y)).unwrap().0;
        for _ in 0..200 {
            t.step(&x, Objective::Mse(&y), 0.03).unwrap();
        }
        assert!(t.gradients(&x, Objective::Mse(&y)).unwrap().0 < initial * 0.01);
        let before = t.predict(&[0.4]).unwrap();
        assert!(t
            .step(&x, Objective::Mse(&[vec![0.], vec![f32::NAN]]), 0.1)
            .is_err());
        assert_eq!(before, t.predict(&[0.4]).unwrap());
    }
}
#[test]
fn policy_gradient_moves_probability_in_reward_direction() {
    let mut t = Trainer::new(
        mlp(&[1, 2], "identity", 4).unwrap(),
        Path::new("."),
        10000,
        Optimizer::Sgd,
    )
    .unwrap();
    let before = t.predict(&[1.]).unwrap();
    t.step(
        &[vec![1.]],
        Objective::Policy {
            actions: &[1],
            advantages: &[1.],
        },
        0.1,
    )
    .unwrap();
    let after = t.predict(&[1.]).unwrap();
    assert!(after[1] - after[0] > before[1] - before[0]);
    assert!(t
        .step(
            &[vec![1.]],
            Objective::Policy {
                actions: &[2],
                advantages: &[1.]
            },
            0.1
        )
        .is_err());
    // A common large offset must not cancel log(2) in cross-entropy.
    t.session
        .write_parameters(&[
            weightlang_runtime::ParameterWrite {
                name: "layer0.weight",
                start: 0,
                values: &[1e20, 1e20],
            },
            weightlang_runtime::ParameterWrite {
                name: "layer0.bias",
                start: 0,
                values: &[0., 0.],
            },
        ])
        .unwrap();
    let (loss, _) = t
        .gradients(
            &[vec![1.]],
            Objective::Policy {
                actions: &[1],
                advantages: &[1.],
            },
        )
        .unwrap();
    assert!((loss - 2f32.ln()).abs() < 1e-6);
}
#[test]
fn checkpoint_resume_matches_uninterrupted_adam_and_never_overwrites() {
    let dir = std::env::temp_dir().join("weightlang");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("native-training-{}.json", std::process::id()));
    let mut t = trainer(Optimizer::Adam);
    let x = vec![vec![1.]];
    let y = vec![vec![3.]];
    for _ in 0..5 {
        t.step(&x, Objective::Mse(&y), 0.01).unwrap();
    }
    t.save(&path).unwrap();
    assert!(t.save(&path).is_err());
    let mut restored = Trainer::restore(&path, 1_000_000).unwrap();
    for _ in 0..5 {
        t.step(&x, Objective::Mse(&y), 0.01).unwrap();
        restored.step(&x, Objective::Mse(&y), 0.01).unwrap();
    }
    assert_eq!(
        t.predict(&[0.7]).unwrap(),
        restored.predict(&[0.7]).unwrap()
    );
    std::fs::remove_file(path).unwrap();
}
