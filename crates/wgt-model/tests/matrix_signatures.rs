mod support;
use wgt::{InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn matrix_function_contracts_execute_and_train_through_the_service() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(include_str!(
        "../../../examples/native_training/matrix_signatures/model.wgt"
    ))
    .unwrap();
    let model = client
        .initialize(
            &definition,
            InstanceOptions {
                optimizer: Some(Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    let input = vec![1., 2., 3., 4., 5., 6.];
    assert_eq!(model.forward(&input).unwrap(), [1.5, 1.5, 3.75, 3.75]);
    model
        .train(
            &[input.clone()],
            Targets::Values(&[vec![0.; 4]]),
            TrainOptions {
                rate: 0.001,
                loss: Loss::Custom,
                clip: None,
            },
        )
        .unwrap();
    assert!((model.forward(&input).unwrap()[0] - 1.46475).abs() < 1e-6);
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}
