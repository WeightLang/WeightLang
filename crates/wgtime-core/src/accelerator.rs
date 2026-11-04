//! Host-owned bounded device pools. No implicit CPU fallback or background training.
use crate::{execution::ExecutionSession, Layer, Model};
use anyhow::{ensure, Context, Result};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Condvar, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceGrant {
    pub device: String,
    pub memory_bytes: usize,
    pub max_parallel: usize,
    pub max_queued: usize,
    pub submissions: u64,
    pub compute_units: u64,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub enum DeviceKind {
    Gpu,
    Npu,
}
/// Trusted native host extension point, including vendor NPU runtimes.
/// Implementations must obey the per-job managed memory budget and synchronize results.
pub trait AcceleratorBackend: Send + Sync + 'static {
    fn device(&self) -> &str;
    fn kind(&self) -> DeviceKind;
    fn run(&self, model: Model, input: Vec<f32>, memory_bytes: usize) -> Result<Vec<f32>>;
}
pub struct NativeGpu {
    device: String,
}
impl NativeGpu {
    pub fn new(device: &str) -> Result<Self> {
        ensure!(
            crate::execution::devices()
                .iter()
                .any(|d| d.id == device && d.id != "cpu"),
            "GPU device unavailable or backend not compiled: {device}"
        );
        Ok(Self {
            device: device.into(),
        })
    }
}
impl AcceleratorBackend for NativeGpu {
    fn device(&self) -> &str {
        &self.device
    }
    fn kind(&self) -> DeviceKind {
        DeviceKind::Gpu
    }
    fn run(&self, model: Model, input: Vec<f32>, memory_bytes: usize) -> Result<Vec<f32>> {
        let mut session = ExecutionSession::prepare_exact(
            model,
            std::path::Path::new("."),
            memory_bytes,
            &self.device,
        )?;
        Ok(session.run(&input)?.to_vec())
    }
}
struct Job {
    id: u64,
    model: Model,
    input: Vec<f32>,
    deadline: Instant,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct JobResult {
    pub ok: bool,
    pub output: Vec<f32>,
    pub error: String,
    pub execution_wall_us: u64,
}
struct State {
    queue: VecDeque<Job>,
    results: BTreeMap<u64, Option<JobResult>>,
    active: usize,
    submitted: u64,
    units: u64,
    closed: bool,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}
pub struct DevicePool {
    grant: DeviceGrant,
    kind: DeviceKind,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}
impl DevicePool {
    pub fn new(grant: DeviceGrant, backend: Arc<dyn AcceleratorBackend>) -> Result<Self> {
        ensure!(
            grant.device == backend.device(),
            "device grant/backend mismatch"
        );
        ensure!(
            (1..=32).contains(&grant.max_parallel)
                && grant.max_queued <= 1024
                && grant.submissions <= 10000,
            "invalid device queue limits"
        );
        ensure!(
            grant.memory_bytes >= grant.max_parallel,
            "insufficient device memory grant"
        );
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                results: BTreeMap::new(),
                active: 0,
                submitted: 0,
                units: 0,
                closed: false,
            }),
            changed: Condvar::new(),
        });
        let mut pool = Self {
            kind: backend.kind(),
            grant,
            shared,
            workers: vec![],
        };
        for _ in 0..pool.grant.max_parallel {
            let shared = pool.shared.clone();
            let backend = backend.clone();
            let memory = pool.grant.memory_bytes / pool.grant.max_parallel;
            pool.workers.push(
                std::thread::Builder::new()
                    .name("wgt-device".into())
                    .spawn(move || loop {
                        let job = {
                            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = state.queue.pop_front() {
                                    state.active += 1;
                                    break job;
                                }
                                if state.closed {
                                    return;
                                }
                                state = shared
                                    .changed
                                    .wait(state)
                                    .unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        let start = Instant::now();
                        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || -> Result<Vec<f32>> {
                                ensure!(
                                    Instant::now() < job.deadline,
                                    "device job deadline exceeded before execution"
                                );
                                let expected_width = model_cost(&job.model)?.1;
                                let result = backend.run(job.model, job.input, memory)?;
                                ensure!(
                                    result.len() == expected_width,
                                    "device output shape mismatch"
                                );
                                ensure!(
                                    Instant::now() < job.deadline,
                                    "device job deadline exceeded"
                                );
                                ensure!(
                                    result.iter().all(|v| v.is_finite()),
                                    "nonfinite device output"
                                );
                                Ok(result)
                            },
                        ));
                        let (output, error) = match outcome {
                            Ok(Ok(v)) => (v, String::new()),
                            Ok(Err(e)) => (vec![], format!("{e:#}")),
                            Err(_) => (vec![], "device backend panicked".into()),
                        };
                        let result = JobResult {
                            ok: error.is_empty(),
                            output,
                            error,
                            execution_wall_us: start.elapsed().as_micros().min(u64::MAX as u128)
                                as u64,
                        };
                        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        state.active -= 1;
                        state.results.insert(job.id, Some(result));
                        shared.changed.notify_all();
                    })?,
            );
        }
        Ok(pool)
    }
    pub fn submit(&self, model: Model, input: Vec<f32>, deadline: Instant) -> Result<u64> {
        ensure!(
            input.len() == model.input && input.iter().all(|v| v.is_finite()),
            "invalid device input"
        );
        // A queued snapshot must be self-contained; a backend must not read model-controlled files.
        ensure!(
            model.parameters.iter().all(|p| !matches!(
                p.value,
                crate::Value::File { .. }
                    | crate::Value::ResidentFile { .. }
                    | crate::Value::PackedFile { .. }
            )),
            "device job requires resident parameters"
        );
        let units = compute_units(&model)?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("device queue poisoned"))?;
        ensure!(
            !state.closed && Instant::now() < deadline,
            "device queue closed or deadline exceeded"
        );
        ensure!(
            state.active + state.queue.len() < self.grant.max_parallel + self.grant.max_queued,
            "device queue full"
        );
        ensure!(
            state.submitted < self.grant.submissions,
            "device submission quota exhausted"
        );
        ensure!(
            units <= self.grant.compute_units.saturating_sub(state.units),
            "device compute quota exhausted"
        );
        let id = state.submitted;
        state.submitted += 1;
        state.units += units;
        state.results.insert(id, None);
        state.queue.push_back(Job {
            id,
            model,
            input,
            deadline,
        });
        self.shared.changed.notify_one();
        Ok(id)
    }
    /// Timeout stops waiting, not a running hardware dispatch. The job remains charged and collectible.
    pub fn wait(&self, id: u64, timeout: Duration) -> Result<JobResult> {
        let start = Instant::now();
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("device queue poisoned"))?;
        loop {
            if let Some(result) = state.results.get(&id).context("unknown device job")? {
                return Ok(result.clone());
            }
            let remaining = timeout.saturating_sub(start.elapsed());
            ensure!(!remaining.is_zero(), "device wait timed out");
            state = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| anyhow::anyhow!("device queue poisoned"))?
                .0;
        }
    }
    pub fn status(&self) -> Result<serde_json::Value> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("device queue poisoned"))?;
        Ok(
            serde_json::json!({"device":self.grant.device,"kind":self.kind,"active":state.active,"queued":(state.active+state.queue.len()).saturating_sub(self.grant.max_parallel),"ready":state.queue.len().min(self.grant.max_parallel-state.active),"max_parallel":self.grant.max_parallel,"max_queued":self.grant.max_queued,"managed_memory_bytes":self.grant.memory_bytes,"per_job_memory_bytes":self.grant.memory_bytes/self.grant.max_parallel,"submitted":state.submitted,"submissions_remaining":self.grant.submissions-state.submitted,"compute_units":state.units,"compute_remaining":self.grant.compute_units-state.units}),
        )
    }
}
impl Drop for DevicePool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.closed = true;
            while let Some(job) = state.queue.pop_front() {
                state.results.insert(
                    job.id,
                    Some(JobResult {
                        ok: false,
                        output: vec![],
                        error: "device pool closed".into(),
                        execution_wall_us: 0,
                    }),
                );
            }
            self.shared.changed.notify_all();
        }
        for thread in self.workers.drain(..) {
            let _ = thread.join();
        }
    }
}
/// Deterministic scalar-work tariff: linear 2*in*out (+bias), activation one unit per element.
pub fn compute_units(model: &Model) -> Result<u64> {
    Ok(model_cost(model)?.0)
}
fn model_cost(model: &Model) -> Result<(u64, usize)> {
    let mut width = model.input as u64;
    let mut work = 0u64;
    for layer in &model.layers {
        let units = match layer {
            Layer::Linear { weight, bias } => {
                let p = model
                    .parameters
                    .iter()
                    .find(|p| &p.name == weight)
                    .context("missing weight")?;
                ensure!(
                    p.shape.len() == 2 && p.shape[1] as u64 == width,
                    "invalid linear shape"
                );
                let n = p.shape[0] as u64;
                let cost = width
                    .checked_mul(n)
                    .and_then(|v| v.checked_mul(2))
                    .and_then(|v| v.checked_add(if bias.is_some() { n } else { 0 }))
                    .context("compute tariff overflow")?;
                width = n;
                cost
            }
            _ => width,
        };
        work = work.checked_add(units).context("compute tariff overflow")?;
    }
    Ok((work.max(1), width as usize))
}

static BACKENDS: std::sync::OnceLock<Mutex<BTreeMap<String, Arc<dyn AcceleratorBackend>>>> =
    std::sync::OnceLock::new();
/// Register a trusted vendor backend before creating WGT runtimes. No script-side registration.
pub fn register_backend(backend: Arc<dyn AcceleratorBackend>) -> Result<()> {
    ensure!(
        !backend.device().is_empty() && backend.device() != "cpu",
        "invalid accelerator backend ID"
    );
    let mut registry = BACKENDS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow::anyhow!("backend registry poisoned"))?;
    ensure!(
        !registry.contains_key(backend.device()),
        "backend already registered"
    );
    registry.insert(backend.device().into(), backend);
    Ok(())
}
pub(crate) fn backend(device: &str) -> Result<Arc<dyn AcceleratorBackend>> {
    if let Some(registry) = BACKENDS.get() {
        if let Some(backend) = registry
            .lock()
            .map_err(|_| anyhow::anyhow!("backend registry poisoned"))?
            .get(device)
        {
            return Ok(backend.clone());
        }
    }
    if device.starts_with("npu:") {
        anyhow::bail!("NPU backend not registered: {device}");
    }
    Ok(Arc::new(NativeGpu::new(device)?))
}
