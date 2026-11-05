//! Real-service tests for the existing channel-first CNN fixtures.
//! Each rank uses the same input, target and convergence threshold as the core
//! convolution training test. Requires a prebuilt daemon; never builds one.
use std::{
    fs,
    process::{Child, Command},
    time::{Duration, Instant},
};
use wgt::{Client, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn check_service(definition: ModelDefinition, source: &str, input_len: usize, output_len: usize) {
    use std::os::unix::fs::PermissionsExt;

    assert_eq!(definition.interface().input, input_len);
    assert_eq!(definition.interface().output, output_len);
    let root = std::env::var_os("CLAUDE_JOB_DIR")
        .map(std::path::PathBuf::from)
        .map(|p| p.join("tmp"))
        .unwrap_or_else(|| "/tmp/weightlang".into());
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("convolution-service-")
        .tempdir_in(&root)
        .unwrap();
    fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon = Daemon(
        Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required"))
            .arg("--test-directory")
            .arg(scratch.path())
            .spawn()
            .unwrap(),
    );
    let endpoint = scratch.path().join("service.sock");
    let started = Instant::now();
    while !endpoint.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "daemon startup timed out"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let client = Client::connect_to(&endpoint).unwrap();
    let source_path = scratch.path().join("model.wgt");
    fs::write(&source_path, source).unwrap();
    let runtime = ModelDefinition::from_file(&source_path).unwrap();
    fs::remove_file(source_path).unwrap();
    assert_eq!(runtime.interface().input, input_len);
    assert_eq!(runtime.interface().output, output_len);

    let options = || InstanceOptions {
        optimizer: Some(Optimizer::Sgd),
        ..Default::default()
    };
    let model = client.initialize(&definition, options()).unwrap();
    let restored = client.initialize(&runtime, options()).unwrap();
    assert_eq!(model.backend(), "native-cpu");
    assert_eq!(restored.backend(), "native-cpu");
    let input = vec![0.4; input_len];
    let targets = vec![vec![0.15; output_len]];
    let inputs = vec![input.clone()];
    let loss = |y: &[f32]| {
        assert_eq!(y.len(), output_len);
        assert!(y.iter().all(|v| v.is_finite()));
        y.iter().map(|v| (v - 0.15).powi(2)).sum::<f32>() / output_len as f32
    };
    let initial = model.forward(&input).unwrap();
    assert_eq!(restored.forward(&input).unwrap(), initial);
    let before = loss(&initial);
    assert!(before > 1e-5, "nontrivial initial error required: {before}");
    for _ in 0..100 {
        model
            .train(
                &inputs,
                Targets::Values(&targets),
                TrainOptions {
                    rate: 0.3,
                    loss: Loss::Custom,
                    clip: None,
                },
            )
            .unwrap();
    }
    let trained = model.forward(&input).unwrap();
    let after = loss(&trained);
    assert!(
        after < before * 0.1,
        "CNN {input_len}->{output_len} MSE {before} -> {after}"
    );
    let weights = scratch.path().join("weights.safetensors");
    model.save_weights(&weights).unwrap();
    restored
        .load_weights(&weights, Some(Optimizer::Sgd))
        .unwrap();
    assert_eq!(restored.forward(&input).unwrap(), trained);

    // Check the complete reloaded forward on a different spatial pattern too.
    let probe: Vec<_> = (0..input_len)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) / 10.0)
        .collect();
    assert_eq!(
        model.forward(&probe).unwrap(),
        restored.forward(&probe).unwrap()
    );
    for instance in [&model, &restored] {
        instance
            .train(
                &inputs,
                Targets::Values(&targets),
                TrainOptions {
                    rate: 0.1,
                    loss: Loss::Custom,
                    clip: None,
                },
            )
            .unwrap();
    }
    assert_eq!(
        model.forward(&input).unwrap(),
        restored.forward(&input).unwrap()
    );
    assert_eq!(
        model.forward(&probe).unwrap(),
        restored.forward(&probe).unwrap()
    );
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
        "CNN {input_len}->{output_len}: real native service MSE {before} -> {after}; weights reload, continued SGD and release"
    );
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn cnn_1d_native_service_training_and_weights_roundtrip() {
    check_service(
        wgt::include_model!("../wgtime-core/tests/fixtures/cnn_1d/model.wgt"),
        include_str!("../../wgtime-core/tests/fixtures/cnn_1d/model.wgt"),
        10,
        12,
    );
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn cnn_2d_native_service_training_and_weights_roundtrip() {
    check_service(
        wgt::include_model!("../wgtime-core/tests/fixtures/cnn_2d/model.wgt"),
        include_str!("../../wgtime-core/tests/fixtures/cnn_2d/model.wgt"),
        30,
        12,
    );
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn cnn_3d_native_service_training_and_weights_roundtrip() {
    check_service(
        wgt::include_model!("../wgtime-core/tests/fixtures/cnn_3d/model.wgt"),
        include_str!("../../wgtime-core/tests/fixtures/cnn_3d/model.wgt"),
        48,
        16,
    );
}
