mod support;
use wgt::{
    InstanceOptions, Loss, ModelDefinition, Optimizer, OutputSlice, StateLink, TrainOptions,
};
const SOURCE: &str = "wgt 0.5\nmodel Recurrent\ninput x:f32[2]\noutput y:f32[1]\nparameter gain:f32[1]=[0.5]\nfn forward(x){return x[0:1]+x[1:2]*gain;}\n";
#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn service_recurrent_backpropagates_final_loss_through_explicit_hidden() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let model = client
        .initialize(
            &ModelDefinition::parse(SOURCE).unwrap(),
            InstanceOptions {
                optimizer: Some(Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    let inputs = vec![vec![1., 0.4], vec![2., 99.], vec![3., -99.]];
    let targets = vec![None, None, Some(vec![0.])];
    let loss = model
        .train_recurrent(
            &inputs,
            &targets,
            StateLink {
                input_offset: 1,
                output_offset: 0,
                width: 1,
            },
            OutputSlice {
                offset: 0,
                width: 1,
            },
            TrainOptions {
                rate: 0.001,
                loss: Loss::Mse,
                clip: None,
            },
        )
        .unwrap();
    assert!((loss - 18.49).abs() < 1e-5);
    let values = model.parameters().unwrap();
    assert!((values[0].2[0] - 0.47162).abs() < 1e-6);
    let before = model.parameters().unwrap();
    assert!(
        model
            .train_recurrent(
                &inputs,
                &[None, None, None],
                StateLink {
                    input_offset: 1,
                    output_offset: 0,
                    width: 1
                },
                OutputSlice {
                    offset: 0,
                    width: 1
                },
                TrainOptions {
                    rate: 0.001,
                    loss: Loss::Mse,
                    clip: None
                }
            )
            .is_err()
    );
    assert_eq!(model.parameters().unwrap(), before);
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}
