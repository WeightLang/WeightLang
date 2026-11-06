use std::path::Path;
use wgtime_core::{Model, Parameter, Value, Layer, categorical::VectorType};
use wgtime_core::training::{Objective, ParameterGradient, Trainer, UpdateStrategy};

#[derive(Clone, Default)]
struct Strategy { calls: usize }
impl UpdateStrategy for Strategy {
    fn propose(&self, parameters: &[ParameterGradient], step: u64) -> anyhow::Result<(Vec<Vec<f32>>, Self)> {
        assert_eq!(parameters[0].name, "projection");
        assert_eq!(parameters[1].name, "bias");
        assert_eq!(step, 1);
        let values = parameters.iter().map(|p| {
            if p.frozen { p.values.clone() } else {
                p.values.iter().zip(&p.gradients).map(|(v, g)| v - 0.1 * g).collect()
            }
        }).collect();
        Ok((values, Self { calls: self.calls + 1 }))
    }
}
fn model() -> Model {
    Model {
        output_type: VectorType::F32, version: 1, input: 2,
        parameters: vec![
            Parameter { name: "projection".into(), shape: vec![1, 2], frozen: false, value: Value::Resident { data: vec![0.4, -0.2] } },
            Parameter { name: "bias".into(), shape: vec![1], frozen: true, value: Value::Resident { data: vec![0.1] } },
        ],
        layers: vec![Layer::Linear { weight: "projection".into(), bias: Some("bias".into()) }],
    }
}

#[test]
fn custom_strategy_gets_names_in_parameter_order_and_commits_once() {
    let mut trainer = Trainer::inference(model(), Path::new("."), 1 << 20).unwrap();
    let mut strategy = Strategy::default();
    trainer.step_with_strategy(&[vec![1., 2.]], Objective::Mse(&[vec![1.]]), &mut strategy).unwrap();
    assert_eq!(strategy.calls, 1);
    assert_eq!(trainer.steps(), 1);
}

#[test]
fn custom_strategy_updates_shared_view_without_detaching_storage() {
    let mut definition = model();
    definition.parameters.push(Parameter {
        name: "shared_projection".into(), shape: vec![1, 2], frozen: false,
        value: Value::View { source: "projection".into(), start: 0 },
    });
    definition.layers = vec![Layer::Linear { weight: "shared_projection".into(), bias: Some("bias".into()) }];
    let mut trainer = Trainer::inference(definition, Path::new("."), 1 << 20).unwrap();
    let before = trainer.predict(&[1., 2.]).unwrap();
    let mut strategy = Strategy::default();
    trainer.step_with_strategy(&[vec![1., 2.]], Objective::Mse(&[vec![1.]]), &mut strategy).unwrap();
    assert_eq!(strategy.calls, 1);
    assert_eq!(trainer.steps(), 1);
    assert_ne!(trainer.predict(&[1., 2.]).unwrap(), before);
    let saved = trainer.snapshot().unwrap();
    assert!(matches!(&saved.parameters[2].value,
        Value::View { source, start: 0 } if source == "projection"));
}
