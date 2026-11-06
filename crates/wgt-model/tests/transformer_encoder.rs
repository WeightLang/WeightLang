mod support;
use std::fs;
use wgt::{InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

const SOURCE: &str = include_str!("fixtures/transformer_encoder/model.wgt");
const INPUT: [f32; 8] = [0.2, -0.1, 0.4, 0.3, -0.3, 0.5, 0.1, -0.2];

#[test]
fn transformer_encoder_definition_and_geometry_are_valid() {
    let embedded = wgt::include_model!("tests/fixtures/transformer_encoder/model.wgt");
    let parsed = ModelDefinition::parse(SOURCE).unwrap();
    let interface = parsed.interface().unwrap();
    assert_eq!(interface.input, 8);
    assert_eq!(interface.output, 8);
    assert_eq!(embedded.parameters().len(), 10);
    assert_eq!(embedded.parameters()[0].shape, [4, 4]);
    assert_eq!(embedded.parameters()[6].shape, [4, 8]);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn transformer_encoder_native_service_trains_and_roundtrips_weights() {
    let scratch = support::Service::start().unwrap();
    let client = scratch.client().unwrap();
    let definition = wgt::include_model!("tests/fixtures/transformer_encoder/model.wgt");
    let path = scratch.path().join("model.wgt");
    fs::write(&path, SOURCE).unwrap();
    let runtime = ModelDefinition::from_file(&path).unwrap();
    fs::remove_file(path).unwrap();

    let model = client.initialize(&definition, InstanceOptions {
        optimizer: Some(Optimizer::Sgd), ..Default::default()
    }).unwrap();
    assert_eq!(model.backend(), "native-cpu");

    let target = vec![0.1, -0.2, 0.2, 0.1, -0.1, 0.3, 0.0, -0.1];
    let mse = |output: &[f32]| output.iter().zip(&target)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>() / target.len() as f32;
    let before = mse(&model.forward(&INPUT).unwrap());
    for _ in 0..100 {
        model.train(&[INPUT.to_vec()], Targets::Values(&[target.clone()]),
            TrainOptions { rate: 0.02, loss: Loss::Custom, clip: None }).unwrap();
    }
    let trained = model.forward(&INPUT).unwrap();
    let after = mse(&trained);
    assert!(after < before, "encoder training did not reduce MSE: {before} -> {after}");

    let weights = scratch.path().join("transformer.safetensors");
    model.save_weights(&weights).unwrap();
    let restored = client.initialize(&runtime, InstanceOptions {
        optimizer: Some(Optimizer::Sgd), ..Default::default()
    }).unwrap();
    restored.load_weights(&weights, Some(Optimizer::Sgd)).unwrap();
    assert_eq!(trained, restored.forward(&INPUT).unwrap());

    restored.close().unwrap();
    model.close().unwrap();
    let stats = client.stats().unwrap();
    assert_eq!(stats["sessions"], 0);
    assert_eq!(stats["reserved"], 0);
}
