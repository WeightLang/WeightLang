use std::path::Path;
use wgtime_core::training::{mlp, Objective, ParameterGradient, Trainer, UpdateStrategy};

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
            next.velocity = parameters.iter().map(|p| vec![0.0; p.values.len()]).collect();
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

    assert!(trainer
        .step_with_strategy(
            &[vec![1.0]],
            Objective::Mse(&[vec![2.0]]),
            &mut strategy,
        )
        .is_err());

    assert_eq!(trainer.predict(&[1.0]).unwrap(), before);
    assert_eq!(trainer.steps(), 0);
    assert_eq!(strategy.steps, 0);
}
