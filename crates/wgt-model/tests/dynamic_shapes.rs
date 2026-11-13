mod support;
use std::collections::BTreeMap;
use wgt::{Dimensions, InstanceOptions, Loss, ModelTemplate, Optimizer, Targets, TrainOptions};

const SOURCE: &str = include_str!("../../../examples/native_training/input_shapes/model.wgt");
fn shapes(rows: usize, features: usize) -> BTreeMap<String, Vec<usize>> {
    BTreeMap::from([("x".into(), vec![rows, features])])
}
fn close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter().zip(right).all(|(a, b)| (a - b).abs() < 2e-6),
        "{left:?} != {right:?}"
    );
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn one_service_instance_changes_shape_without_resetting_adam_or_parameters() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let template = ModelTemplate::parse(SOURCE).unwrap();
    let options = InstanceOptions {
        optimizer: Some(Optimizer::Adam),
        ..Default::default()
    };
    let mut dynamic = client
        .initialize_dynamic(&template, &shapes(2, 3), &Dimensions::new(), options)
        .unwrap();
    let baseline = client
        .initialize(
            &template
                .bind_input_shapes(&shapes(2, 3), &Dimensions::new())
                .unwrap(),
            options,
        )
        .unwrap();
    for rows in [2, 7, 3, 2, 7, 1, 2] {
        let input = vec![1.; rows * 3];
        let actual = dynamic.forward(&input, &shapes(rows, 3)).unwrap();
        let expected = baseline.forward(&[1.; 6]).unwrap();
        for row in actual.chunks_exact(2) {
            close(row, &expected[..2]);
        }
        // Mean MSE has the same parameter gradient for repeated identical rows.
        dynamic
            .train(
                &[input],
                &shapes(rows, 3),
                Targets::Values(&[vec![0.; rows * 2]]),
                TrainOptions {
                    rate: 0.01,
                    loss: Loss::Mse,
                    clip: None,
                },
            )
            .unwrap();
        baseline
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
        let actual = dynamic.model().parameters().unwrap();
        let expected = baseline.parameters().unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!((&a.0, &a.1), (&b.0, &b.1));
            close(&a.2, &b.2);
        }
        assert_eq!(
            client.stats().unwrap()["sessions"],
            2,
            "shape changes must not create service instances"
        );
    }
    let before = dynamic.model().parameters().unwrap();
    let width = dynamic.model().interface().unwrap().input;
    assert!(
        dynamic.specialize(&shapes(2, 4)).is_err(),
        "parameter shapes cannot change"
    );
    assert!(dynamic.forward(&[1.], &shapes(2, 3)).is_err());
    assert_eq!(dynamic.model().parameters().unwrap(), before);
    assert_eq!(dynamic.model().interface().unwrap().input, width);
    dynamic.close().unwrap();
    baseline.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn rejected_geometry_admission_keeps_old_native_model_usable() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let template = ModelTemplate::parse(SOURCE).unwrap();
    let mut model = client
        .initialize_dynamic(
            &template,
            &shapes(2, 3),
            &Dimensions::new(),
            InstanceOptions {
                memory_bytes: 16 * 1024,
                optimizer: Some(Optimizer::Sgd),
            },
        )
        .unwrap();
    let parameters = model.model().parameters().unwrap();
    assert!(model.specialize(&shapes(5000, 3)).is_err());
    assert_eq!(model.model().parameters().unwrap(), parameters);
    assert_eq!(
        model.forward(&[1.; 6], &shapes(2, 3)).unwrap(),
        vec![0.75; 4]
    );
    assert_eq!(client.stats().unwrap()["sessions"], 1);
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn tuple_ports_and_dynamic_geometry_share_one_native_model() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let template = ModelTemplate::parse(include_str!(
        "../../../examples/native_training/dynamic_hidden/model.wgt"
    ))
    .unwrap();
    let shapes = |rows| {
        BTreeMap::from([
            ("x".into(), vec![rows, 1]),
            ("hidden".into(), vec![rows, 1]),
        ])
    };
    let mut model = client
        .initialize_dynamic(
            &template,
            &shapes(2),
            &Dimensions::new(),
            InstanceOptions::default(),
        )
        .unwrap();
    let parameters = model.model().parameters().unwrap();
    for rows in [2, 7, 3, 2] {
        let output = model
            .forward_ports(&[vec![1.; rows], vec![0.2; rows]], &shapes(rows))
            .unwrap();
        close(&output[0], &vec![0.7; rows]);
        close(&output[1], &vec![1.1; rows]);
        assert_eq!(model.model().parameters().unwrap(), parameters);
        assert_eq!(client.stats().unwrap()["sessions"], 1);
    }
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}
