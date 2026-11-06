use std::path::Path;
use wgtime_core::training::{Objective, ParameterGradient, Trainer, UpdateStrategy, mlp};

#[derive(Clone, Debug, Default, PartialEq)]
struct Momentum {
    steps: u64,
    velocity: Vec<Vec<f32>>,
}

impl UpdateStrategy for Momentum {
    fn propose(
        &self,
        parameters: &[ParameterGradient],
        step: u64,
    ) -> anyhow::Result<(Vec<Vec<f32>>, Self)> {
        let mut next = self.clone();
        if next.velocity.is_empty() {
            next.velocity = parameters
                .iter()
                .map(|p| vec![0.0; p.values.len()])
                .collect();
        }
        let mut candidates = Vec::with_capacity(parameters.len());
        for (index, parameter) in parameters.iter().enumerate() {
            let mut values = parameter.values.clone();
            if !parameter.frozen {
                for (i, value) in values.iter_mut().enumerate() {
                    next.velocity[index][i] =
                        0.9 * next.velocity[index][i] + parameter.gradients[i];
                    *value -= 0.05 * next.velocity[index][i];
                }
            }
            candidates.push(values);
        }
        next.steps = step;
        Ok((candidates, next))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct InvalidStrategy {
    steps: u64,
}

impl UpdateStrategy for InvalidStrategy {
    fn propose(
        &self,
        parameters: &[ParameterGradient],
        step: u64,
    ) -> anyhow::Result<(Vec<Vec<f32>>, Self)> {
        let mut candidates: Vec<_> = parameters.iter().map(|p| p.values.clone()).collect();
        candidates[0][0] = f32::NAN;
        Ok((candidates, Self { steps: step }))
    }
}

#[test]
fn custom_strategy_updates_weights_and_owns_its_state() {
    let model = mlp(&[1, 1], "identity", 42).unwrap();
    let mut trainer = Trainer::inference(model, Path::new("."), 1024 * 1024).unwrap();
    let before = trainer.predict(&[1.0]).unwrap();
    let inputs = vec![vec![1.0]];
    let targets = vec![vec![2.0]];
    let mut strategy = Momentum::default();

    let loss = trainer
        .step_with_strategy(&inputs, Objective::Mse(&targets), &mut strategy)
        .unwrap();
    let after = trainer.predict(&[1.0]).unwrap();

    assert!(loss.is_finite());
    assert_ne!(before, after);
    assert_eq!(strategy.steps, 1);
    assert_eq!(trainer.steps(), 1);
}

#[test]
fn invalid_candidate_does_not_commit_model_or_strategy_state() {
    let model = mlp(&[1, 1], "identity", 42).unwrap();
    let mut trainer = Trainer::inference(model, Path::new("."), 1024 * 1024).unwrap();
    let before = trainer.predict(&[1.0]).unwrap();
    let mut strategy = InvalidStrategy::default();

    assert!(
        trainer
            .step_with_strategy(&[vec![1.0]], Objective::Mse(&[vec![2.0]]), &mut strategy,)
            .is_err()
    );

    assert_eq!(trainer.predict(&[1.0]).unwrap(), before);
    assert_eq!(trainer.steps(), 0);
    assert_eq!(strategy.steps, 0);
}

#[test]
fn parameter_identities_match_gradients_inspection_and_snapshots() {
    use wgtime_core::{Layer, Model, Parameter, Value};

    #[derive(Clone)]
    struct UpdateWeight;
    impl UpdateStrategy for UpdateWeight {
        fn propose(
            &self,
            parameters: &[ParameterGradient],
            step: u64,
        ) -> anyhow::Result<(Vec<Vec<f32>>, Self)> {
            assert_eq!(step, 1);
            assert_eq!(
                parameters
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                ["z_weight", "a_bias"]
            );
            assert_eq!(parameters[0].shape, [1, 1]);
            assert_eq!(parameters[1].shape, [1]);
            assert_eq!(parameters[0].gradients, [30.]);
            assert_eq!(parameters[1].gradients, [10.]);
            let candidates = parameters
                .iter()
                .map(|p| {
                    if p.name == "z_weight" {
                        vec![4.]
                    } else {
                        p.values.clone()
                    }
                })
                .collect();
            Ok((candidates, self.clone()))
        }
    }

    let model = Model {
        output_type: Default::default(),
        version: 1,
        input: 1,
        parameters: vec![
            Parameter {
                name: "z_weight".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::Resident { data: vec![2.] },
            },
            Parameter {
                name: "a_bias".into(),
                shape: vec![1],
                frozen: false,
                value: Value::Resident { data: vec![-1.] },
            },
        ],
        layers: vec![Layer::Linear {
            weight: "z_weight".into(),
            bias: Some("a_bias".into()),
        }],
    };
    let mut trainer = Trainer::inference(model, Path::new("."), 1024 * 1024).unwrap();
    let names = trainer
        .session
        .parameter_info()
        .iter()
        .map(|p| p.name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["z_weight", "a_bias"]);

    trainer
        .step_with_strategy(&[vec![3.]], Objective::Mse(&[vec![0.]]), &mut UpdateWeight)
        .unwrap();
    assert_eq!(trainer.predict(&[3.]).unwrap(), [11.]);
    let snapshot = trainer.snapshot().unwrap();
    assert_eq!(
        snapshot
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["z_weight", "a_bias"]
    );
    let mut restored = Trainer::inference(snapshot, Path::new("."), 1024 * 1024).unwrap();
    assert_eq!(restored.predict(&[3.]).unwrap(), [11.]);
}

#[test]
fn custom_updates_preserve_shared_parameter_storage() {
    use wgtime_core::{Layer, Model, Parameter, Value};

    let model = Model {
        output_type: Default::default(),
        version: 1,
        input: 1,
        parameters: vec![
            Parameter {
                name: "weight".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::Resident { data: vec![2.] },
            },
            Parameter {
                name: "shared".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::View {
                    source: "weight".into(),
                    start: 0,
                },
            },
        ],
        layers: vec![Layer::Linear {
            weight: "shared".into(),
            bias: None,
        }],
    };
    let mut trainer = Trainer::inference(model, Path::new("."), 1024 * 1024).unwrap();
    let mut strategy = Momentum::default();
    for expected in [1.8_f32, 1.44] {
        trainer
            .step_with_strategy(&[vec![1.]], Objective::Mse(&[vec![0.]]), &mut strategy)
            .unwrap();
        assert!((trainer.predict(&[1.]).unwrap()[0] - expected).abs() < 1e-6);
    }
    let snapshot = trainer.snapshot().unwrap();
    assert!(matches!(&snapshot.parameters[1].value,
        Value::View { source, start: 0 } if source == "weight"));
    let mut restored = Trainer::inference(snapshot, Path::new("."), 1024 * 1024).unwrap();
    assert!((restored.predict(&[1.]).unwrap()[0] - 1.44).abs() < 1e-6);
}
