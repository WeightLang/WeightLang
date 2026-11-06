use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};
use wgtime_core::{accelerator::*, training::mlp, Model};
struct MockNpu {
    gate: Arc<(Mutex<bool>, Condvar)>,
    active: AtomicUsize,
    peak: AtomicUsize,
    started: AtomicUsize,
}
impl AcceleratorBackend for MockNpu {
    fn device(&self) -> &str {
        "npu:test"
    }
    fn kind(&self) -> DeviceKind {
        DeviceKind::Npu
    }
    fn run(&self, _: Model, input: Vec<f32>, memory: usize) -> anyhow::Result<Vec<f32>> {
        assert_eq!(memory, 4096);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.started.fetch_add(1, Ordering::SeqCst);
        let (lock, cv) = &*self.gate;
        let mut released = lock.lock().unwrap();
        while !*released {
            released = cv.wait(released).unwrap();
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(vec![input.iter().sum()])
    }
}
fn model() -> Model {
    mlp(&[2, 1], "tanh", 42).unwrap()
}
fn grant(parallel: usize, queued: usize) -> DeviceGrant {
    DeviceGrant {
        device: "npu:test".into(),
        memory_bytes: 4096 * parallel,
        max_parallel: parallel,
        max_queued: queued,
        submissions: 3,
        compute_units: 15,
    }
}
fn backend() -> Arc<MockNpu> {
    Arc::new(MockNpu {
        gate: Arc::new((Mutex::new(false), Condvar::new())),
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
        started: AtomicUsize::new(0),
    })
}
fn release(backend: &MockNpu) {
    *backend.gate.0.lock().unwrap() = true;
    backend.gate.1.notify_all();
}
#[test]
fn npu_provider_queue_concurrency_and_quotas() {
    let backend = backend();
    let pool = DevicePool::new(grant(2, 1), backend.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let jobs = (0..3)
        .map(|_| pool.submit(model(), vec![2., 4.], deadline).unwrap())
        .collect::<Vec<_>>();
    assert!(pool
        .submit(model(), vec![2., 4.], deadline)
        .unwrap_err()
        .to_string()
        .contains("queue full"));
    let start = Instant::now();
    while backend.active.load(Ordering::SeqCst) != 2 {
        assert!(start.elapsed() < Duration::from_secs(3));
        std::thread::yield_now();
    }
    assert_eq!(pool.status().unwrap()["queued"], 1);
    assert!(pool.wait(jobs[0], Duration::from_millis(1)).is_err());
    release(&backend);
    for id in jobs {
        let result = pool.wait(id, Duration::from_secs(2)).unwrap();
        assert!(result.ok);
        assert_eq!(result.output, vec![6.]);
    }
    assert_eq!(backend.peak.load(Ordering::SeqCst), 2);
    let state = pool.status().unwrap();
    assert_eq!(state["compute_units"], 15);
    assert_eq!(state["submissions_remaining"], 0);
    assert!(pool.submit(model(), vec![2., 4.], deadline).is_err());
}
#[test]
fn queued_expiry_does_not_execute_or_refund() {
    let backend = backend();
    let pool = DevicePool::new(grant(1, 1), backend.clone()).unwrap();
    let first = pool
        .submit(
            model(),
            vec![1., 2.],
            Instant::now() + Duration::from_secs(5),
        )
        .unwrap();
    let second = pool
        .submit(
            model(),
            vec![1., 2.],
            Instant::now() + Duration::from_millis(20),
        )
        .unwrap();
    std::thread::sleep(Duration::from_millis(40));
    release(&backend);
    assert!(pool.wait(first, Duration::from_secs(2)).unwrap().ok);
    let result = pool.wait(second, Duration::from_secs(2)).unwrap();
    assert!(!result.ok);
    assert!(result.error.contains("before execution"));
    assert_eq!(backend.started.load(Ordering::SeqCst), 1);
    assert_eq!(pool.status().unwrap()["compute_units"], 10);
}
#[test]
fn rejects_unknown_gpu_and_npu_grant_mismatch() {
    assert!(NativeGpu::new("npu:missing").is_err());
    let mut g = grant(1, 0);
    g.device = "npu:other".into();
    assert!(DevicePool::new(g, backend()).is_err());
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires real accelerator hardware; run explicitly"]
fn hardware_gpu_inference_and_memory_quota() {
    let device = wgtime_core::execution::devices()
        .into_iter()
        .find(|d| d.id != "cpu")
        .expect("hardware GPU required");
    let mut trainer = wgtime_core::training::Trainer::new(
        mlp(&[2, 4, 1], "tanh", 42).unwrap(),
        std::path::Path::new("."),
        1 << 20,
        wgtime_core::training::Optimizer::Sgd,
    )
    .unwrap();
    let expected = trainer.predict(&[1., 2.]).unwrap();
    let grant = DeviceGrant {
        device: device.id.clone(),
        memory_bytes: 1 << 20,
        max_parallel: 1,
        max_queued: 1,
        submissions: 2,
        compute_units: 1000,
    };
    let pool = DevicePool::new(grant, Arc::new(NativeGpu::new(&device.id).unwrap())).unwrap();
    let job = pool
        .submit(
            trainer.snapshot().unwrap(),
            vec![1., 2.],
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
    let result = pool.wait(job, Duration::from_secs(30)).unwrap();
    assert!(result.ok, "{}", result.error);
    for (actual, expected) in result.output.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-4 + expected.abs() * 1e-3);
    }
    let tiny = DeviceGrant {
        device: device.id.clone(),
        memory_bytes: trainer.session.report().managed_bytes,
        max_parallel: 1,
        max_queued: 0,
        submissions: 1,
        compute_units: 1000,
    };
    let pool = DevicePool::new(tiny, Arc::new(NativeGpu::new(&device.id).unwrap())).unwrap();
    let job = pool
        .submit(
            trainer.snapshot().unwrap(),
            vec![1., 2.],
            Instant::now() + Duration::from_secs(30),
        )
        .unwrap();
    let rejected = pool.wait(job, Duration::from_secs(30)).unwrap();
    assert!(!rejected.ok);
    assert!(
        rejected.error.contains("combined host/GPU budget exceeded"),
        "{}",
        rejected.error
    );
}

#[test]
fn registered_npu_wgt_flow() {
    if std::env::var_os("WGT_TEST_NPU_WORKER").is_some() {
        let backend = backend();
        release(&backend);
        register_backend(backend).unwrap();
        let source = r#"wgt 0.5
fn main() {
    compute_budget(100,4096,4,10000);
    let model=mlp([2,1],"tanh",42,"sgd");
    let job=device_submit(model,[2,4],"accelerator");
    assert job["ok"],"submit";
    let remaining=compute_status();
    assert remaining["work_remaining"]==95,"device work must debit script budget";
    let result=device_wait("accelerator",job["job"],1000);
    assert result["ok"] && result["output"]==[6],"registered provider result";
    return device_status("accelerator");
}"#;
        let report =
            wgtime_core::script::run(source, std::path::Path::new("."), 10000).unwrap();
        assert_eq!(report.result["kind"], "Npu");
        assert_eq!(report.result["compute_units"], 5.);
        return;
    }
    let dir = std::path::PathBuf::from("/tmp/weightlang")
        .join(format!("npu-contract-{}", std::process::id()));
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    std::fs::create_dir(&dir).unwrap();
    let policy = dir.join("policy.json");
    std::fs::write(&policy,serde_json::to_vec(&serde_json::json!({"work":100,"workspace_bytes":4096,"calls":4,"deadline_ms":10000,"devices":{"accelerator":grant(1,1)}})).unwrap()).unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "registered_npu_wgt_flow", "--nocapture"])
        .env("WGT_HOST_POLICY", &policy)
        .env("WGT_TEST_NPU_WORKER", "1")
        .output()
        .unwrap();
    std::fs::remove_file(policy).unwrap();
    std::fs::remove_dir(dir).unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn compute_quota_rejection_and_backend_failure_are_accounted() {
    struct WrongShape;
    impl AcceleratorBackend for WrongShape {
        fn device(&self) -> &str {
            "npu:shape-test"
        }
        fn kind(&self) -> DeviceKind {
            DeviceKind::Npu
        }
        fn run(&self, _: Model, _: Vec<f32>, _: usize) -> anyhow::Result<Vec<f32>> {
            Ok(vec![1., 2.])
        }
    }
    let g = DeviceGrant {
        device: "npu:shape-test".into(),
        memory_bytes: 4096,
        max_parallel: 1,
        max_queued: 0,
        submissions: 5,
        compute_units: 5,
    };
    let pool = DevicePool::new(g, Arc::new(WrongShape)).unwrap();
    let job = pool
        .submit(
            model(),
            vec![1., 2.],
            Instant::now() + Duration::from_secs(3),
        )
        .unwrap();
    let result = pool.wait(job, Duration::from_secs(3)).unwrap();
    assert!(!result.ok);
    assert!(result.error.contains("shape mismatch"));
    let state = pool.status().unwrap();
    assert_eq!(state["compute_units"], 5);
    assert_eq!(state["submissions_remaining"], 4);
    assert!(pool
        .submit(
            model(),
            vec![1., 2.],
            Instant::now() + Duration::from_secs(3)
        )
        .unwrap_err()
        .to_string()
        .contains("compute quota exhausted"));
    assert_eq!(pool.status().unwrap()["submitted"], 1);
}
