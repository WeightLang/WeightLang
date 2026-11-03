#![cfg(feature = "gpu")]
use std::path::Path;
use weightlang_runtime::{
    execution::devices,
    training::{mlp, GpuTraining, Objective, Optimizer, Trainer, TrainingBackend, TrainingError},
    Layer, Model, Parameter, ParameterWrite, Value,
};
const BUDGET: usize = 64 * 1024 * 1024;

fn cpu(model: Model) -> Trainer {
    Trainer::new(model, Path::new("."), BUDGET, Optimizer::Sgd).unwrap()
}
fn attach(t: &mut Trainer, id: &str) {
    let backend = GpuTraining::new(id, 128, BUDGET).expect("listed GPU must initialize");
    eprintln!(
        "GPU TRAINING: {} / {}",
        backend.device_info().id,
        backend.device_info().name
    );
    t.set_backend(TrainingBackend::Gpu(backend)).unwrap();
}
fn exported_model(t: &mut Trainer) -> Model {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join("weightlang").join(format!(
        "gpu-export-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("model.json");
    t.export(&path).unwrap();
    let model = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    model
}
fn snapshot(t: &mut Trainer) -> Vec<u8> {
    serde_json::to_vec(&exported_model(t)).unwrap()
}
fn close(a: f32, b: f32, atol: f32, rtol: f32) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
        "{a} != {b}, tolerance {atol} + {rtol}*abs(reference)"
    );
}
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            close(a, b, 2e-5, 2e-4);
        }
    }
}

#[test]
fn unavailable_device_and_invalid_limits_are_typed() {
    let error = GpuTraining::new("nonexistent:0", 8, BUDGET).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<TrainingError>(),
        Some(TrainingError::Unavailable(_))
    ));
    assert!(matches!(
        GpuTraining::new("nonexistent:0", 0, BUDGET)
            .unwrap_err()
            .downcast_ref::<TrainingError>(),
        Some(TrainingError::InvalidInput(_))
    ));
}

/// Explicit opt-in: ordinary cargo test reports this as ignored, never as GPU proof.
/// WGT_REQUIRE_GPU=1 turns absent hardware into failure for acceptance runs.
#[test]
#[ignore = "requires real GPU; run --ignored --nocapture, WGT_REQUIRE_GPU=1 to require hardware"]
fn real_gpu_mse_gradients_finite_difference_sgd_and_atomicity() {
    let candidates = devices();
    let requested = std::env::var("WGT_TRAINING_GPU").ok();
    let adapter = candidates
        .iter()
        .find(|d| {
            requested
                .as_ref()
                .map_or(d.name.contains("4060"), |id| id == &d.id)
        })
        .or_else(|| {
            if requested.is_none() {
                candidates.iter().find(|d| d.id != "cpu")
            } else {
                None
            }
        });
    let Some(adapter) = adapter else {
        assert!(
            std::env::var_os("WGT_REQUIRE_GPU").is_none(),
            "required hardware GPU absent"
        );
        eprintln!("SKIP: no requested hardware GPU available; no GPU computation tested");
        return;
    };
    let id = &adapter.id;
    for activation in ["relu", "tanh", "sigmoid", "identity"] {
        let mut reference = cpu(mlp(&[2, 3, 2], activation, 5).unwrap());
        if activation == "relu" {
            reference
                .session
                .write_parameters(&[ParameterWrite {
                    name: "layer0.bias",
                    start: 0,
                    values: &[2., -2., 2.],
                }])
                .unwrap();
        }
        let mut gpu = cpu(exported_model(&mut reference));
        attach(&mut gpu, id);
        let inputs = vec![vec![0.3, -0.2], vec![0.1, 0.7], vec![-0.4, 0.2]];
        let targets = vec![vec![0.5, 0.2], vec![-0.1, 0.3], vec![0.2, -0.6]];
        let before = snapshot(&mut gpu);
        let (loss, gradients) = gpu.gradients(&inputs, Objective::Mse(&targets)).unwrap();
        let (expected, expected_gradients) = reference
            .gradients(&inputs, Objective::Mse(&targets))
            .unwrap();
        close(loss, expected, 2e-5, 2e-4);
        compare(&gradients, &expected_gradients);
        assert_eq!(snapshot(&mut gpu), before);
        assert_eq!(gpu.steps(), 0);
        // Central differences use GPU loss, so the check is independent of CPU backprop.
        // h=1e-3 balances f32 subtraction roundoff and O(h^2) truncation. ReLU is
        // kept away from its kink. Near-zero derivatives require an absolute floor.
        let params: Vec<_> = gpu
            .session
            .parameter_info()
            .iter()
            .map(|p| (p.name.to_string(), p.shape.iter().product::<usize>()))
            .collect();
        let mut max_difference = 0f32;
        for (i, (name, n)) in params.iter().enumerate() {
            for (j, &gradient) in gradients[i].iter().enumerate().take(*n) {
                let mut original = [0.];
                gpu.session.read_parameter(name, j, &mut original).unwrap();
                let mut losses = [0.; 2];
                for (slot, delta) in [-0.001, 0.001].into_iter().enumerate() {
                    gpu.session
                        .write_parameters(&[ParameterWrite {
                            name,
                            start: j,
                            values: &[original[0] + delta],
                        }])
                        .unwrap();
                    losses[slot] = gpu.gradients(&inputs, Objective::Mse(&targets)).unwrap().0;
                }
                gpu.session
                    .write_parameters(&[ParameterWrite {
                        name,
                        start: j,
                        values: &original,
                    }])
                    .unwrap();
                let numerical = (losses[1] - losses[0]) / 0.002;
                max_difference = max_difference.max((numerical - gradient).abs());
                close(numerical, gradient, 2e-3, 2e-3);
            }
        }
        eprintln!("{activation}: gradient CPU/GPU matched; GPU finite-difference maximum absolute error {max_difference:e}");
        for _ in 0..20 {
            let a = gpu.step(&inputs, Objective::Mse(&targets), 0.02).unwrap();
            let b = reference
                .step(&inputs, Objective::Mse(&targets), 0.02)
                .unwrap();
            close(a, b, 3e-5, 3e-4);
        }
        assert_eq!(gpu.steps(), reference.steps());
        let a = exported_model(&mut gpu);
        let b = exported_model(&mut reference);
        for (a, b) in a.parameters.iter().zip(&b.parameters) {
            let (Value::Resident { data: a }, Value::Resident { data: b }) = (&a.value, &b.value)
            else {
                panic!("resident model")
            };
            for (&a, &b) in a.iter().zip(b) {
                close(a, b, 3e-5, 3e-4);
            }
        }
        let before = snapshot(&mut gpu);
        let steps = gpu.steps();
        let revision = gpu.session.revision();
        for bad in [f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(gpu
                .step(&[vec![bad, 0.]], Objective::Mse(&[vec![0., 0.]]), 0.1)
                .is_err());
            assert_eq!(snapshot(&mut gpu), before);
            assert_eq!(gpu.steps(), steps);
            assert_eq!(gpu.session.revision(), revision);
        }
        assert!(gpu
            .step(
                &inputs,
                Objective::Policy {
                    actions: &[0, 0, 0],
                    advantages: &[1., 1., 1.]
                },
                0.01
            )
            .is_err());
        assert_eq!(snapshot(&mut gpu), before);
        checkpoint(&mut gpu, id, &inputs, &targets);
    }
    shared(id);
    overflow_and_limits(id);
    wide_batch(id);
}

fn checkpoint(gpu: &mut Trainer, id: &str, x: &[Vec<f32>], y: &[Vec<f32>]) {
    let dir = std::env::temp_dir().join("weightlang").join(format!(
        "gpu-checkpoint-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("checkpoint.json");
    gpu.save(&path).unwrap();
    assert!(gpu.save(&path).is_err());
    let mut restored = Trainer::restore(&path, BUDGET).unwrap();
    assert!(matches!(restored.backend(), TrainingBackend::Cpu));
    assert_eq!(snapshot(gpu), snapshot(&mut restored));
    assert_eq!(gpu.steps(), restored.steps());
    attach(&mut restored, id);
    gpu.step(x, Objective::Mse(y), 0.02).unwrap();
    restored.step(x, Objective::Mse(y), 0.02).unwrap();
    assert_eq!(snapshot(gpu), snapshot(&mut restored));
    std::fs::remove_dir_all(dir).unwrap();
}

fn shared(id: &str) {
    // Two repeated uses, an offset view, an add expression with duplicate sources,
    // and a frozen dependency chain exercise owner folding independently of MLP.
    let model = Model {
        version: 1,
        input: 1,
        parameters: vec![
            Parameter {
                name: "owner".into(),
                shape: vec![2],
                frozen: false,
                value: Value::Resident {
                    data: vec![9., 0.5],
                },
            },
            Parameter {
                name: "view".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::View {
                    source: "owner".into(),
                    start: 1,
                },
            },
            Parameter {
                name: "sum".into(),
                shape: vec![1, 1],
                frozen: false,
                value: Value::Add {
                    base: "view".into(),
                    delta: "view".into(),
                },
            },
            Parameter {
                name: "fixed".into(),
                shape: vec![1],
                frozen: true,
                value: Value::Resident { data: vec![0.2] },
            },
        ],
        layers: vec![
            Layer::Linear {
                weight: "sum".into(),
                bias: Some("fixed".into()),
            },
            Layer::Linear {
                weight: "sum".into(),
                bias: None,
            },
        ],
    };
    let mut reference = cpu(model);
    let mut gpu = cpu(exported_model(&mut reference));
    attach(&mut gpu, id);
    let x = vec![vec![0.2], vec![-0.3]];
    let y = vec![vec![0.4], vec![0.1]];
    compare(
        &gpu.gradients(&x, Objective::Mse(&y)).unwrap().1,
        &reference.gradients(&x, Objective::Mse(&y)).unwrap().1,
    );
    for _ in 0..5 {
        gpu.step(&x, Objective::Mse(&y), 0.01).unwrap();
        reference.step(&x, Objective::Mse(&y), 0.01).unwrap();
    }
    close(
        gpu.predict(&[0.3]).unwrap()[0],
        reference.predict(&[0.3]).unwrap()[0],
        2e-5,
        2e-4,
    );
    let mut fixed = [0.];
    gpu.session.read_parameter("fixed", 0, &mut fixed).unwrap();
    assert_eq!(fixed, [0.2]);
    let mut frozen = exported_model(&mut reference);
    frozen.parameters[2].frozen = true;
    let mut gpu = cpu(frozen);
    attach(&mut gpu, id);
    let before = snapshot(&mut gpu);
    gpu.step(&x, Objective::Mse(&y), 0.01).unwrap();
    assert_eq!(snapshot(&mut gpu), before);
}
fn overflow_and_limits(id: &str) {
    let mut t = cpu(mlp(&[1, 1], "identity", 1).unwrap());
    attach(&mut t, id);
    let before = snapshot(&mut t);
    // First owner has zero gradient, later bias candidate overflows. No partial commit.
    let e = t
        .step(&[vec![0.]], Objective::Mse(&[vec![-2.]]), f32::MAX)
        .unwrap_err();
    assert!(
        matches!(
            e.downcast_ref::<TrainingError>(),
            Some(TrainingError::NonFinite(_))
        ),
        "{e:#}"
    );
    assert_eq!(snapshot(&mut t), before);
    assert_eq!(t.steps(), 0);
    assert_eq!(t.session.revision(), 0);
    t.step(&[vec![0.]], Objective::Mse(&[vec![-2.]]), 0.01)
        .unwrap();
    let before = snapshot(&mut t);
    assert!(t
        .gradients(&vec![vec![0.]; 129], Objective::Mse(&vec![vec![0.]; 129]))
        .is_err());
    t.set_backend(TrainingBackend::Gpu(GpuTraining::new(id, 1, 1).unwrap()))
        .unwrap();
    let e = t
        .step(&[vec![0.]], Objective::Mse(&[vec![0.]]), 0.1)
        .unwrap_err();
    assert!(matches!(
        e.downcast_ref::<TrainingError>(),
        Some(TrainingError::OutOfBudget)
    ));
    assert_eq!(snapshot(&mut t), before);
    let mut adam = Trainer::new(
        mlp(&[1, 1], "identity", 1).unwrap(),
        Path::new("."),
        BUDGET,
        Optimizer::Adam,
    )
    .unwrap();
    assert!(adam
        .set_backend(TrainingBackend::Gpu(
            GpuTraining::new(id, 1, BUDGET).unwrap()
        ))
        .is_err());
    assert!(matches!(adam.backend(), TrainingBackend::Cpu));
}
fn wide_batch(id: &str) {
    // Nonmultiples of 64 and multiple workgroups exercise flattening and tail lanes.
    let mut reference = cpu(mlp(&[65, 67, 3], "sigmoid", 7).unwrap());
    let mut gpu = cpu(exported_model(&mut reference));
    attach(&mut gpu, id);
    let x: Vec<_> = (0..17)
        .map(|b| (0..65).map(|i| ((i + b) % 11) as f32 / 20. - 0.2).collect())
        .collect();
    let y = vec![vec![0.2, 0.3, -0.1]; 17];
    let (a, ga) = gpu.gradients(&x, Objective::Mse(&y)).unwrap();
    let (b, gb) = reference.gradients(&x, Objective::Mse(&y)).unwrap();
    close(a, b, 2e-5, 2e-4);
    compare(&ga, &gb);
    eprintln!("wide batch [17,65] -> [17,67] -> [17,3]: CPU/GPU gradients matched");
}
