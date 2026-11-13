mod support;

use wgt::{InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

const SOURCE: &str = include_str!("../../../examples/native_training/shared_recurrent/model.wgt");
const INPUT: [f32; 6] = [0.7, -0.4, -0.2, 0.6, 0.3, -0.5];
const TARGET: [f32; 2] = [0.15, -0.25];

#[test]
fn shared_recurrent_definition_has_one_shared_parameter_set() {
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let interface = definition.interface().unwrap();
    assert_eq!(interface.input, 6);
    assert_eq!(interface.output, 2);
    let parameters = definition.parameters();
    assert_eq!(parameters.len(), 4);
    assert_eq!(
        parameters
            .iter()
            .map(|p| (p.name.as_str(), p.shape.as_slice()))
            .collect::<Vec<_>>(),
        [
            ("weight_ih", &[6, 2][..]),
            ("weight_hh", &[6, 2][..]),
            ("bias_ih", &[6][..]),
            ("bias_hh", &[6][..]),
        ]
    );
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
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn shared_recurrent_native_service_trains_and_roundtrips_weights() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let model = client
        .initialize(
            &definition,
            InstanceOptions {
                optimizer: Some(Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(model.backend(), "native-cpu");
    // Bind every gate and bias deterministically, including the reset-gated b_hn.
    // The same row-major (r,z,n) values are used by core and Python acceptance.
    model
        .update_parameters(&[
            (
                "weight_ih".into(),
                vec![6, 2],
                vec![
                    0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3, 0.6, 0.2, -0.4, 0.3,
                ],
            ),
            (
                "weight_hh".into(),
                vec![6, 2],
                vec![
                    -0.3, 0.2, 0.1, -0.4, 0.5, 0.3, -0.2, 0.6, 0.4, -0.5, 0.2, 0.1,
                ],
            ),
            (
                "bias_ih".into(),
                vec![6],
                vec![0.11, -0.07, 0.03, 0.13, -0.17, 0.19],
            ),
            (
                "bias_hh".into(),
                vec![6],
                vec![-0.05, 0.09, -0.12, 0.08, 0.23, -0.21],
            ),
        ])
        .unwrap();
    // External candidate updates intentionally clear the previous optimizer.
    // Rebind these complete values with an explicit SGD selection before training.
    let initial_weights = service.path().join("initial.safetensors");
    model.save_weights(&initial_weights).unwrap();
    model
        .load_weights(&initial_weights, Some(Optimizer::Sgd))
        .unwrap();
    let original_parameters = model.parameters().unwrap();
    let initial = model.forward(&INPUT).unwrap();
    for (actual, expected) in initial.iter().zip([0.262169, 0.019570801]) {
        assert!(
            (actual - expected).abs() < 2e-6,
            "GRU output {actual} != {expected}"
        );
    }
    assert_eq!(initial.len(), 2);
    assert_eq!(
        initial,
        model.forward(&INPUT).unwrap(),
        "hidden must remain explicit"
    );
    for _ in 0..40 {
        model
            .train(
                &[INPUT.to_vec()],
                Targets::Values(&[TARGET.to_vec()]),
                TrainOptions {
                    rate: 0.05,
                    loss: Loss::Custom,
                    clip: None,
                },
            )
            .unwrap();
    }
    let trained = model.forward(&INPUT).unwrap();
    assert!(
        mse(&trained) < mse(&initial) * 0.5,
        "GRU training did not reduce MSE"
    );
    let trained_parameters = model.parameters().unwrap();
    assert_ne!(trained_parameters, original_parameters);
    let weights = service.path().join("shared_recurrent.safetensors");
    model.save_weights(&weights).unwrap();
    let restored = client
        .initialize(
            &definition,
            InstanceOptions {
                optimizer: Some(Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(restored.backend(), "native-cpu");
    restored
        .load_weights(&weights, Some(Optimizer::Sgd))
        .unwrap();
    assert_eq!(trained_parameters, restored.parameters().unwrap());
    assert_eq!(trained, restored.forward(&INPUT).unwrap());
    let alternate = [-0.6, 0.8, 0.5, -0.3, -0.4, 0.2];
    assert_eq!(
        model.forward(&alternate).unwrap(),
        restored.forward(&alternate).unwrap()
    );
    restored.close().unwrap();
    model.close().unwrap();
    let stats = client.stats().unwrap();
    assert_eq!(stats["sessions"], 0);
    assert_eq!(stats["reserved"], 0);
}
