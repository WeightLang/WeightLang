mod support;
use std::collections::BTreeMap;
use wgt::{Dimensions, InstanceOptions, Loss, ModelTemplate, Optimizer, Targets, TrainOptions};

const SOURCE: &str = include_str!("../../../examples/native_training/input_shapes/model.wgt");

#[test]
fn shape_bound_definitions_keep_parameter_geometry_and_validate_ports() {
    let template = ModelTemplate::parse(SOURCE).unwrap();
    for rows in [1, 7] {
        let definition = template
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![rows, 3])]),
                &Dimensions::new(),
            )
            .unwrap();
        assert_eq!(definition.interface().unwrap().input, rows * 3);
        assert_eq!(definition.interface().unwrap().output, rows * 2);
        assert_eq!(definition.parameters()[0].shape, [3, 2]);
        assert!(definition.compile_plan_with_modules().is_ok());
    }
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn shape_bound_instances_execute_train_and_remain_independent() {
    let scratch = support::Service::start().unwrap();
    let client = scratch.client().unwrap();
    let template = ModelTemplate::parse(SOURCE).unwrap();
    let mut models = Vec::new();
    for rows in [2, 7] {
        let definition = template
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![rows, 3])]),
                &Dimensions::new(),
            )
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
        assert_eq!(model.backend(), "native-cpu");
        assert_eq!(
            model.forward(&vec![1.; rows * 3]).unwrap(),
            vec![0.75; rows * 2]
        );
        assert!(model.forward(&vec![1.; rows * 3 + 1]).is_err());
        models.push(model);
    }
    // Training one specialization must not silently change another instance.
    models[0]
        .train(
            &[vec![1.; 6]],
            Targets::Values(&[vec![0.; 4]]),
            TrainOptions {
                rate: 0.01,
                loss: Loss::Mse,
                clip: None,
            },
        )
        .unwrap();
    assert!(models[0].forward(&[1.; 6]).unwrap()[0] < 0.75);
    assert_eq!(models[1].forward(&[1.; 21]).unwrap(), vec![0.75; 14]);
    for model in models {
        model.close().unwrap();
    }
    assert_eq!(client.stats().unwrap()["sessions"], 0);
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}
