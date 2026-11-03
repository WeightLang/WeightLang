#![cfg(feature = "gpu")]
use std::path::Path;
use weightlang_runtime::{
    execution::devices,
    training::{mlp, GpuTraining, Objective, Optimizer, Trainer, TrainingBackend, TrainingError},
    Layer, Model, Parameter, ParameterWrite, Value,
};
const BUDGET: usize = 64 * 1024 * 1024;
// Hardware acceptance shares one adapter. Serialize discovery/device lifetimes:
// concurrent Vulkan initialization on this runner has produced a native SIGSEGV.
static GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let _guard = GPU_TEST.lock().expect("GPU acceptance lock poisoned");
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
    let _guard = GPU_TEST.lock().expect("GPU acceptance lock poisoned");
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

fn hardware() -> Option<String> {
    let candidates = devices();
    let requested = std::env::var("WGT_TRAINING_GPU").ok();
    let found = candidates.iter().find(|d| {
        requested
            .as_ref()
            .map_or(d.name.contains("4060"), |id| id == &d.id)
    });
    if found.is_none() {
        assert!(
            std::env::var_os("WGT_REQUIRE_GPU").is_none(),
            "required GPU absent"
        );
        eprintln!("SKIP: requested GPU absent");
    }
    found.map(|d| d.id.clone())
}
fn adam(model: Model) -> Trainer {
    Trainer::new(model, Path::new("."), BUDGET, Optimizer::Adam).unwrap()
}
fn record(t: &mut Trainer) -> serde_json::Value {
    let dir = std::env::temp_dir().join("weightlang").join(format!(
        "t4-record-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.json");
    t.save(&path).unwrap();
    let value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    value
}
fn restore_record(value: &serde_json::Value) -> Trainer {
    let dir = std::env::temp_dir().join("weightlang").join(format!(
        "t4-restore-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.json");
    std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    let t = Trainer::restore(&path, BUDGET).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    t
}
fn compare_state(a: &mut Trainer, b: &mut Trainer) {
    let a = record(a);
    let b = record(b);
    assert_eq!(a["steps"], b["steps"]);
    assert_eq!(a["optimizer"], b["optimizer"]);
    for key in ["first", "second"] {
        let x: Vec<Vec<f32>> = serde_json::from_value(a[key].clone()).unwrap();
        let y: Vec<Vec<f32>> = serde_json::from_value(b[key].clone()).unwrap();
        compare(&x, &y);
    }
    let x: Model = serde_json::from_value(a["model"].clone()).unwrap();
    let y: Model = serde_json::from_value(b["model"].clone()).unwrap();
    for (x, y) in x.parameters.iter().zip(&y.parameters) {
        match (&x.value, &y.value) {
            (Value::Resident { data: x }, Value::Resident { data: y }) => {
                for (&x, &y) in x.iter().zip(y) {
                    close(x, y, 3e-5, 3e-4);
                }
            }
            _ => assert_eq!(
                serde_json::to_value(x).unwrap(),
                serde_json::to_value(y).unwrap()
            ),
        }
    }
}
fn policy<'a>(actions: &'a [usize], advantages: &'a [f32]) -> Objective<'a> {
    Objective::Policy {
        actions,
        advantages,
    }
}

#[test]
#[ignore = "requires RTX GPU; WGT_REQUIRE_GPU=1 requires hardware"]
fn real_gpu_policy_adam_multistep_gradients_and_cross_backend_checkpoint() {
    let _guard = GPU_TEST.lock().expect("GPU acceptance lock poisoned");
    let Some(id) = hardware() else { return };
    let x = [vec![0.3, -0.2], vec![0.1, 0.7], vec![-0.4, 0.2]];
    let actions = [0, 2, 1];
    // CE and reward-baseline REINFORCE, including negative and zero advantages.
    for advantages in [[1., 1., 1.], [2. - 0.5, -0.7 - 0.5, 0.5 - 0.5]] {
        for optimizer in [Optimizer::Sgd, Optimizer::Adam] {
            let model = mlp(&[2, 5, 3], "tanh", 41).unwrap();
            let mut reference = Trainer::new(
                mlp(&[2, 5, 3], "tanh", 41).unwrap(),
                Path::new("."),
                BUDGET,
                optimizer,
            )
            .unwrap();
            let mut gpu = Trainer::new(model, Path::new("."), BUDGET, optimizer).unwrap();
            attach(&mut gpu, &id);
            let before = record(&mut gpu);
            let (loss, gradients) = gpu.gradients(&x, policy(&actions, &advantages)).unwrap();
            let (expected, expected_gradients) = reference
                .gradients(&x, policy(&actions, &advantages))
                .unwrap();
            close(loss, expected, 2e-5, 2e-4);
            compare(&gradients, &expected_gradients);
            assert_eq!(before, record(&mut gpu));
            let params: Vec<_> = gpu
                .session
                .parameter_info()
                .iter()
                .map(|p| p.name.to_string())
                .collect();
            for (i, name) in params.iter().enumerate() {
                for (j, &gradient) in gradients[i].iter().enumerate() {
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
                        losses[slot] = gpu.gradients(&x, policy(&actions, &advantages)).unwrap().0;
                    }
                    gpu.session
                        .write_parameters(&[ParameterWrite {
                            name,
                            start: j,
                            values: &original,
                        }])
                        .unwrap();
                    close((losses[1] - losses[0]) / 0.002, gradient, 2e-3, 2e-3);
                }
            }
            for _ in 0..20 {
                let a = gpu.step(&x, policy(&actions, &advantages), 0.01).unwrap();
                let b = reference
                    .step(&x, policy(&actions, &advantages), 0.01)
                    .unwrap();
                close(a, b, 3e-5, 3e-4);
                compare_state(&mut gpu, &mut reference);
            }
            // Both directions preserve all moments; also compare same-GPU resume.
            let mut cpu_from_gpu = restore_record(&record(&mut gpu));
            let mut gpu_from_cpu = restore_record(&record(&mut reference));
            let mut gpu_from_gpu = restore_record(&record(&mut gpu));
            attach(&mut gpu_from_cpu, &id);
            attach(&mut gpu_from_gpu, &id);
            for _ in 0..5 {
                for t in [
                    &mut gpu,
                    &mut reference,
                    &mut cpu_from_gpu,
                    &mut gpu_from_cpu,
                    &mut gpu_from_gpu,
                ] {
                    t.step(&x, policy(&actions, &advantages), 0.007).unwrap();
                }
                compare_state(&mut gpu, &mut cpu_from_gpu);
                compare_state(&mut reference, &mut gpu_from_cpu);
                assert_eq!(record(&mut gpu), record(&mut gpu_from_gpu));
            }
            // Exercise the actual weightfs Trainer adapter (not a FUSE mount).
            use weightlang_runtime::weightfs::Weights;
            for t in [&mut gpu, &mut reference] {
                let previous = record(t);
                t.replace("layer0.bias", &[0.1, -0.1, 0.2, 0., -0.2])
                    .unwrap();
                let edited = record(t);
                assert_eq!(previous["first"], edited["first"]);
                assert_eq!(previous["second"], edited["second"]);
                t.step(&x, policy(&actions, &advantages), 0.01).unwrap();
            }
            compare_state(&mut gpu, &mut reference);
            eprintln!("T4 {optimizer:?} advantages={advantages:?}: finite differences, 20 steps, bidirectional checkpoint, weightfs passed");
        }
    }
}

#[test]
#[ignore = "requires RTX GPU; WGT_REQUIRE_GPU=1 requires hardware"]
fn real_gpu_stable_policy_analytic_gradient_and_adam_mse() {
    let _guard = GPU_TEST.lock().expect("GPU acceptance lock poisoned");
    let Some(id) = hardware() else { return };
    let mut model = mlp(&[1, 3], "identity", 1).unwrap();
    model.parameters[0].value = Value::Resident { data: vec![0.; 3] };
    model.parameters[1].value = Value::Resident {
        data: vec![1000., 1001., -1000.],
    };
    let mut gpu = adam(model);
    let mut reference = adam(exported_model(&mut gpu));
    attach(&mut gpu, &id);
    for advantages in [[1.], [-2.], [0.]] {
        let (loss, g) = gpu
            .gradients(&[vec![0.]], policy(&[2], &advantages))
            .unwrap();
        let (loss_cpu, g_cpu) = reference
            .gradients(&[vec![0.]], policy(&[2], &advantages))
            .unwrap();
        close(
            loss,
            advantages[0] * (2001. + (1. + (-1f32).exp()).ln()),
            3e-4,
            2e-5,
        );
        close(loss, loss_cpu, 3e-4, 2e-5);
        compare(&g, &g_cpu);
        let p0 = 1. / (1. + 1f32.exp());
        compare(
            &[g[1].clone()],
            &[vec![
                advantages[0] * p0,
                advantages[0] * (1. - p0),
                -advantages[0],
            ]],
        );
    }
    let mut gpu = adam(mlp(&[2, 3, 2], "sigmoid", 5).unwrap());
    let mut reference = adam(exported_model(&mut gpu));
    attach(&mut gpu, &id);
    let x = [vec![0.2, -0.3], vec![0.4, 0.1]];
    let y = [vec![0.3, -0.1], vec![0.1, 0.2]];
    // Independent analytical Adam recurrence uses the observed gradient, including
    // both bias corrections and epsilon placement, across changing batches/steps.
    let mut first: Vec<Vec<f32>> = gpu
        .session
        .parameter_info()
        .iter()
        .map(|p| vec![0.; p.shape.iter().product()])
        .collect();
    let mut second = first.clone();
    for step in 1..=20 {
        let (_, g) = gpu.gradients(&x, Objective::Mse(&y)).unwrap();
        let model = exported_model(&mut gpu);
        gpu.step(&x, Objective::Mse(&y), 0.02).unwrap();
        reference.step(&x, Objective::Mse(&y), 0.02).unwrap();
        for (i, p) in model.parameters.iter().enumerate() {
            let Value::Resident { data } = &p.value else {
                panic!("resident")
            };
            let mut actual = vec![0.; data.len()];
            gpu.session.read_parameter(&p.name, 0, &mut actual).unwrap();
            for j in 0..data.len() {
                first[i][j] = 0.9 * first[i][j] + 0.1 * g[i][j];
                second[i][j] = 0.999 * second[i][j] + 0.001 * g[i][j] * g[i][j];
                let direction = (first[i][j] / (1. - 0.9f32.powi(step)))
                    / ((second[i][j] / (1. - 0.999f32.powi(step))).sqrt() + 1e-8);
                close(actual[j], data[j] - 0.02 * direction, 2e-5, 2e-4);
            }
        }
        compare_state(&mut gpu, &mut reference);
    }
}

#[test]
#[ignore = "requires RTX GPU; WGT_REQUIRE_GPU=1 requires hardware"]
fn real_gpu_adam_shared_frozen_and_transaction_rollback() {
    let _guard = GPU_TEST.lock().expect("GPU acceptance lock poisoned");
    let Some(id) = hardware() else { return };
    let model = Model {
        version: 1,
        input: 2,
        parameters: vec![
            Parameter {
                name: "owner".into(),
                shape: vec![5],
                frozen: false,
                value: Value::Resident {
                    data: vec![9., 0.2, -0.1, 0.4, 0.3],
                },
            },
            Parameter {
                name: "view".into(),
                shape: vec![2, 2],
                frozen: false,
                value: Value::View {
                    source: "owner".into(),
                    start: 1,
                },
            },
            Parameter {
                name: "sum".into(),
                shape: vec![2, 2],
                frozen: false,
                value: Value::Add {
                    base: "view".into(),
                    delta: "view".into(),
                },
            },
            Parameter {
                name: "fixed".into(),
                shape: vec![2],
                frozen: true,
                value: Value::Resident {
                    data: vec![0.2, -0.1],
                },
            },
        ],
        layers: vec![
            Layer::Linear {
                weight: "sum".into(),
                bias: Some("fixed".into()),
            },
            Layer::Linear {
                weight: "sum".into(),
                bias: Some("fixed".into()),
            },
        ],
    };
    let mut reference = adam(model);
    let mut gpu = adam(exported_model(&mut reference));
    attach(&mut gpu, &id);
    let x = [vec![0.2, -0.3], vec![-0.1, 0.4]];
    for advantages in [[1., 1.], [1.5, -0.5]] {
        let (_, g) = gpu.gradients(&x, policy(&[0, 1], &advantages)).unwrap();
        compare(
            &g,
            &reference
                .gradients(&x, policy(&[0, 1], &advantages))
                .unwrap()
                .1,
        );
        // Owner finite differences cover repeated views and duplicate add sources.
        for (j, &gradient) in g[0].iter().enumerate() {
            let mut value = [0.];
            gpu.session.read_parameter("owner", j, &mut value).unwrap();
            let mut loss = [0.; 2];
            for (slot, delta) in [-0.001, 0.001].into_iter().enumerate() {
                gpu.session
                    .write_parameters(&[ParameterWrite {
                        name: "owner",
                        start: j,
                        values: &[value[0] + delta],
                    }])
                    .unwrap();
                loss[slot] = gpu.gradients(&x, policy(&[0, 1], &advantages)).unwrap().0;
            }
            gpu.session
                .write_parameters(&[ParameterWrite {
                    name: "owner",
                    start: j,
                    values: &value,
                }])
                .unwrap();
            close((loss[1] - loss[0]) / 0.002, gradient, 2e-3, 2e-3);
        }
        for _ in 0..5 {
            gpu.step(&x, policy(&[0, 1], &advantages), 0.01).unwrap();
            reference
                .step(&x, policy(&[0, 1], &advantages), 0.01)
                .unwrap();
            compare_state(&mut gpu, &mut reference);
        }
    }
    let state = record(&mut gpu);
    assert_eq!(state["first"][3], serde_json::json!([0., 0.]));
    assert_eq!(state["second"][3], serde_json::json!([0., 0.]));
    let mut frozen = exported_model(&mut gpu);
    frozen.parameters[2].frozen = true;
    let mut gpu = adam(frozen);
    attach(&mut gpu, &id);
    let before = record(&mut gpu);
    gpu.step(&x, policy(&[0, 1], &[1., -1.]), 0.01).unwrap();
    let after = record(&mut gpu);
    for key in ["model", "first", "second"] {
        assert_eq!(before[key], after[key]);
    }

    let mut gpu = adam(mlp(&[1, 2], "identity", 1).unwrap());
    attach(&mut gpu, &id);
    gpu.step(&[vec![0.]], policy(&[0], &[1.]), 0.01).unwrap();
    let before = record(&mut gpu);
    let revision = gpu.session.revision();
    // Finite gradient, overflowing second moment on the later bias owner.
    let error = gpu
        .step(&[vec![0.]], policy(&[0], &[1e23]), 0.01)
        .unwrap_err();
    assert!(
        matches!(
            error.downcast_ref::<TrainingError>(),
            Some(TrainingError::NonFinite(_))
        ),
        "{error:#}"
    );
    assert_eq!(before, record(&mut gpu));
    assert_eq!(revision, gpu.session.revision());
    for (actions, advantages) in [
        (&[2][..], &[1.][..]),
        (&[][..], &[1.][..]),
        (&[0][..], &[f32::NAN][..]),
    ] {
        assert!(gpu
            .step(&[vec![0.]], policy(actions, advantages), 0.01)
            .is_err());
        assert_eq!(before, record(&mut gpu));
        assert_eq!(revision, gpu.session.revision());
    }
    use weightlang_runtime::weightfs::Weights;
    gpu.replace("layer0.bias", &[f32::MAX, f32::MAX]).unwrap();
    let before = record(&mut gpu);
    let revision = gpu.session.revision();
    assert!(gpu
        .step(&[vec![0.]], policy(&[0], &[1.]), f32::MAX)
        .is_err());
    assert_eq!(before, record(&mut gpu));
    assert_eq!(revision, gpu.session.revision());
    // Recovery keeps pre-failure history and matches the CPU checkpoint consumer.
    let mut reference = restore_record(&before);
    gpu.replace("layer0.bias", &[0., 0.]).unwrap();
    reference.replace("layer0.bias", &[0., 0.]).unwrap();
    gpu.step(&[vec![0.]], policy(&[0], &[1.]), 0.01).unwrap();
    reference
        .step(&[vec![0.]], policy(&[0], &[1.]), 0.01)
        .unwrap();
    compare_state(&mut gpu, &mut reference);
}
