//! Real daemon coverage for three small compositions, with explicit CPU backend.
//! Run only with a prebuilt WGTIME_TEST_DAEMON; no runtime build fallback.
mod support;
use std::fs;
use wgt::{InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

struct Case {
    name: &'static str,
    definition: ModelDefinition,
    source: &'static str,
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
}
fn cases() -> [Case; 3] {
    let residual = vec![vec![0.7, -0.3], vec![-0.4, 0.8], vec![0.2, 0.5]];
    let recurrent = vec![
        vec![0.7, -0.3, 0.2, 0.6, -0.4, 0.5],
        vec![-0.4, 0.8, 0.5, -0.2, 0.3, -0.6],
        vec![0.2, 0.5, -0.6, 0.4, 0.7, 0.1],
    ];
    let autoencoder = vec![
        vec![0.6, 0.0, 0.3],
        vec![-0.6, 0.0, -0.3],
        vec![0.0, 0.6, -0.3],
        vec![0.0, -0.6, 0.3],
    ];
    [
        Case {
            name: "residual_mlp",
            definition: wgt::include_model!("../wgtime-core/tests/fixtures/residual_mlp/model.wgt"),
            source: include_str!("../../wgtime-core/tests/fixtures/residual_mlp/model.wgt"),
            targets: residual
                .iter()
                .map(|x| x.iter().map(|v| v * 0.5).collect())
                .collect(),
            inputs: residual,
        },
        Case {
            name: "recurrent_cell",
            definition: wgt::include_model!(
                "../wgtime-core/tests/fixtures/recurrent_cell/model.wgt"
            ),
            source: include_str!("../../wgtime-core/tests/fixtures/recurrent_cell/model.wgt"),
            targets: recurrent
                .iter()
                .map(|x| vec![0.3 * x[2] + 0.1 * x[4], 0.3 * x[3] + 0.1 * x[5]])
                .collect(),
            inputs: recurrent,
        },
        Case {
            name: "autoencoder",
            definition: wgt::include_model!("../wgtime-core/tests/fixtures/autoencoder/model.wgt"),
            source: include_str!("../../wgtime-core/tests/fixtures/autoencoder/model.wgt"),
            targets: autoencoder.clone(),
            inputs: autoencoder,
        },
    ]
}
fn mse(predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    assert_eq!(predictions.len(), targets.len());
    predictions
        .iter()
        .zip(targets)
        .map(|(y, t)| {
            assert_eq!(y.len(), t.len());
            y.iter().zip(t).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / t.len() as f32
        })
        .sum::<f32>()
        / targets.len() as f32
}
#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn compositions_native_service_train_roundtrip_and_release() {
    let scratch = support::Service::start().unwrap();
    let client = scratch.client().unwrap();
    for case in cases() {
        let source_path = scratch.path().join("model.wgt");
        fs::write(&source_path, case.source).unwrap();
        let runtime = ModelDefinition::from_file(&source_path).unwrap();
        fs::remove_file(source_path).unwrap();
        let options = || InstanceOptions {
            optimizer: Some(Optimizer::Sgd),
            ..Default::default()
        };
        let model = client.initialize(&case.definition, options()).unwrap();
        assert_eq!(model.backend(), "native-cpu");
        let initial: Vec<_> = case
            .inputs
            .iter()
            .map(|x| model.forward(x).unwrap())
            .collect();
        // A second independently initialized definition must use the same initial values.
        let restored = client.initialize(&runtime, options()).unwrap();
        assert_eq!(restored.backend(), "native-cpu");
        for (x, y) in case.inputs.iter().zip(&initial) {
            assert_eq!(&restored.forward(x).unwrap(), y);
        }
        let before = mse(&initial, &case.targets);
        assert!(
            before > 1e-5,
            "{} must start with nontrivial error",
            case.name
        );
        for _ in 0..400 {
            model
                .train(
                    &case.inputs,
                    Targets::Values(&case.targets),
                    TrainOptions {
                        rate: 0.15,
                        loss: Loss::Custom,
                        clip: None,
                    },
                )
                .unwrap();
        }
        let trained: Vec<_> = case
            .inputs
            .iter()
            .map(|x| model.forward(x).unwrap())
            .collect();
        let after = mse(&trained, &case.targets);
        assert!(
            after < before * 0.5,
            "{} MSE {before} -> {after}",
            case.name
        );
        let path = scratch.path().join(format!("{}.safetensors", case.name));
        model.save_weights(&path).unwrap();
        restored.load_weights(&path, Some(Optimizer::Sgd)).unwrap();
        for (x, y) in case.inputs.iter().zip(&trained) {
            assert_eq!(&restored.forward(x).unwrap(), y);
        }
        for net in [&model, &restored] {
            net.train(
                &case.inputs,
                Targets::Values(&case.targets),
                TrainOptions {
                    rate: 0.05,
                    loss: Loss::Custom,
                    clip: None,
                },
            )
            .unwrap();
        }
        for x in &case.inputs {
            assert_eq!(model.forward(x).unwrap(), restored.forward(x).unwrap());
        }
        let stats = client.stats().unwrap();
        let executions = stats["execution"].as_array().unwrap();
        assert_eq!(executions.len(), 2);
        for execution in executions {
            assert_eq!(execution["backend"], "native-cpu");
            assert!(execution["native_tape_forwards"].as_u64().unwrap() > 0);
        }
        restored.close().unwrap();
        model.close().unwrap();
        let released = client.stats().unwrap();
        assert_eq!(released["sessions"], 0);
        assert_eq!(released["reserved"], 0);
        eprintln!(
            "{} real daemon: MSE {before} -> {after}; native CPU, weights reload, continued SGD and release",
            case.name
        );
    }
}
