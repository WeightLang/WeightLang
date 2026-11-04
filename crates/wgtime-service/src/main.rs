use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use wgtime_core::training::{mlp, Objective, Optimizer, Trainer};
use wgtime::{receive, send, Reply, Request, PROTOCOL};

mod metrics;
#[cfg(feature="allocation-metrics")]
#[global_allocator]
static ALLOCATOR: metrics::Counting=metrics::Counting;
const MAX_CONNECTIONS: usize = 64;
const MAX_SESSION_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONNECTION_BYTES: usize = 512 * 1024 * 1024;
struct Shared {
    next: AtomicU64,
    connections: AtomicUsize,
}
struct Guard {
    directory: PathBuf,
    _lock: File,
}
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.directory.join("service.sock"));
    }
}
fn listen(directory: &Path) -> Result<(UnixListener, Guard)> {
    let uid = unsafe { libc::geteuid() };
    let metadata =
        fs::symlink_metadata(directory).context("wgtime directory must already exist")?;
    ensure!(
        metadata.is_dir() && metadata.uid() == uid && metadata.mode() & 0o077 == 0,
        "wgtime directory must be owned by service UID with mode 0700"
    );
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(directory.join("service.lock"))?;
    let metadata = lock.metadata()?;
    ensure!(
        metadata.is_file()
            && metadata.uid() == uid
            && metadata.nlink() == 1
            && metadata.mode() & 0o077 == 0,
        "unsafe service lock"
    );
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "wgtime already running or lock unavailable"
    );
    let endpoint = directory.join("service.sock");
    if let Ok(metadata) = fs::symlink_metadata(&endpoint) {
        ensure!(
            metadata.file_type().is_socket() && metadata.uid() == uid,
            "refusing unknown endpoint"
        );
        ensure!(
            UnixStream::connect(&endpoint).is_err(),
            "live endpoint exists without matching lock"
        );
        fs::remove_file(&endpoint)?;
    }
    let listener = UnixListener::bind(&endpoint)?;
    fs::set_permissions(&endpoint, fs::Permissions::from_mode(0o600))?;
    Ok((
        listener,
        Guard {
            directory: directory.into(),
            _lock: lock,
        },
    ))
}
struct Connection {
    sessions: HashMap<u64, (Trainer, usize)>,
    reserved: usize,
    ready: bool,
    tensors: HashMap<u64, wgtime::buffer::Mapping>,
    checkpoints: HashMap<u64, Vec<u8>>,
    adaptive: HashMap<u64, (wgtime_core::adaptive::AdaptiveSession, usize)>,
    strategies: HashMap<u64, wgtime_service::strategy::Strategy>,
}
impl Connection {
    fn execute(&mut self, request: Request, shared: &Shared) -> Result<serde_json::Value> {
        if let Request::Hello { protocol } = request {
            ensure!(
                !self.ready && protocol == PROTOCOL,
                "invalid protocol handshake"
            );
            self.ready = true;
            return Ok(serde_json::json!({"protocol": PROTOCOL, "pid": std::process::id()}));
        }
        ensure!(self.ready, "handshake required");
        match request {
            Request::Create {
                widths,
                activation,
                seed,
                optimizer,
                memory,
            } => {
                ensure!(self.sessions.len() + self.adaptive.len() < 64, "session limit exceeded");
                ensure!(
                    memory > 0
                        && memory <= MAX_SESSION_BYTES
                        && self
                            .reserved
                            .checked_add(memory)
                            .is_some_and(|n| n <= MAX_CONNECTION_BYTES),
                    "session memory budget exceeded"
                );
                ensure!(
                    (2..=64).contains(&widths.len())
                        && widths.iter().all(|n| (1..=16384).contains(n)),
                    "invalid widths"
                );
                let count = widths
                    .windows(2)
                    .try_fold(0usize, |n, w| {
                        n.checked_add(w[0].checked_mul(w[1])?.checked_add(w[1])?)
                    })
                    .context("parameter size overflow")?;
                ensure!(
                    count <= 4_000_000 && count.checked_mul(16).is_some_and(|n| n <= memory),
                    "parameter budget exceeded"
                );
                let optimizer = match optimizer.as_str() {
                    "sgd" => Optimizer::Sgd,
                    "adam" => Optimizer::Adam,
                    _ => bail!("unsupported optimizer"),
                };
                let trainer = Trainer::new(
                    mlp(&widths, &activation, seed)?,
                    Path::new("."),
                    memory,
                    optimizer,
                )?;
                // A random process prefix makes handles from a previous daemon epoch invalid.
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                ensure!(handle != u64::MAX, "handle namespace exhausted");
                self.sessions.insert(handle, (trainer, memory));
                self.reserved += memory;
                Ok(serde_json::json!({"handle": handle}))
            }
            Request::InitializeDefinition { source, memory, optimizer } => {
                ensure!(source.len() <= 1024 * 1024, "definition exceeds 1 MiB");
                ensure!(self.sessions.len()+self.adaptive.len()<64 && memory>0 && memory<=MAX_SESSION_BYTES && self.reserved.checked_add(memory).is_some_and(|n| n<=MAX_CONNECTION_BYTES), "definition admission budget exceeded");
                ensure!(optimizer.as_deref().is_none_or(|s| matches!(s,"sgd"|"adam")), "unsupported optimizer");
                // Re-parse untrusted source in the service; macro validation is not authority.
                let definition = wgt_language::model::ModelDefinition::parse(&source)?;
                // Reject an impossible parameter budget before materializing any values.
                // Core preparation still checks activations, workspace and optimizer state.
                let parameter_bytes = definition.parameters().iter().try_fold(0usize, |total, parameter| {
                    let count = parameter.shape.iter().try_fold(1usize, |n, axis| n.checked_mul(*axis)).context("parameter shape overflow")?;
                    total.checked_add(count.checked_mul(4).context("parameter byte overflow")?).context("parameter byte overflow")
                })?;
                ensure!(parameter_bytes <= memory, "definition parameter memory budget exceeded");
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                match definition.instantiate_plan()? {
                    wgt_language::model::ModelPlan::Graph(graph) => {
                        let model = wgtime_core::source::Graph(graph).instantiate()?;
                        let trainer = match optimizer.as_deref() {
                            None => Trainer::inference(model, Path::new("."), memory)?,
                            Some(name) => Trainer::new(model, Path::new("."), memory, if name=="adam" {Optimizer::Adam} else {Optimizer::Sgd})?,
                        };
                        self.sessions.insert(handle,(trainer,memory));
                    }
                    wgt_language::model::ModelPlan::Forward(program) => {
                        let mut session = wgtime_core::adaptive::AdaptiveSession::prepare(program, memory)?;
                        if let Some(name)=optimizer { session.set_optimizer(if name=="adam" {Optimizer::Adam} else {Optimizer::Sgd})?; }
                        self.adaptive.insert(handle,(session,memory));
                    }
                }
                self.reserved += memory;
                Ok(serde_json::json!({"handle":handle}))
            }
            Request::Predict { handle, input } => {
                if let Some((session, _)) = self.adaptive.get_mut(&handle) {
                    return Ok(
                        serde_json::json!({"output": session.evaluate(&input, 1_000_000)?, "revision": session.revision()}),
                    );
                }
                let trainer = &mut self
                    .sessions
                    .get_mut(&handle)
                    .context("unknown session or foreign owner")?
                    .0;
                Ok(
                    serde_json::json!({"output": trainer.predict(&input)?, "steps": trainer.steps()}),
                )
            }
            Request::CreateTrainingProgram { program, memory, optimizer } => {
                ensure!(self.adaptive.len()+self.sessions.len()<64 && memory>0 && memory<=MAX_SESSION_BYTES && self.reserved.checked_add(memory).is_some_and(|n| n<=MAX_CONNECTION_BYTES), "program admission budget exceeded");
                ensure!(matches!(optimizer.as_str(), "none"|"sgd"|"adam"), "unknown optimizer");
                let mut session = wgtime_core::adaptive::AdaptiveSession::prepare(serde_json::from_value(program)?, memory)?;
                match optimizer.as_str() { "sgd"=>session.set_optimizer(Optimizer::Sgd)?, "adam"=>session.set_optimizer(Optimizer::Adam)?, _=>{} }
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.adaptive.insert(handle,(session,memory));
                self.reserved += memory;
                Ok(serde_json::json!({"handle":handle}))
            }
            Request::TrainStep { handle, inputs, targets, rate, loss, clip } => {
                ensure!(!inputs.is_empty() && inputs.len() <= 65536, "invalid batch size");
                ensure!(rate.is_finite() && rate > 0., "learning rate must be positive and finite");
                ensure!(clip.is_none_or(|v| v.is_finite() && v > 0.), "gradient clip norm must be positive and finite");
                ensure!(matches!(loss.as_str(), "mse" | "cross_entropy" | "loss"), "unknown training loss");
                let matrix: Vec<Vec<f32>> = if loss == "cross_entropy" { Vec::new() } else { serde_json::from_value(targets.clone())? };
                let actions: Vec<usize> = if loss == "cross_entropy" {
                    let classes = if let Some((session,_)) = self.adaptive.get(&handle) {session.output_len()}
                        else {self.sessions.get(&handle).context("unknown session or foreign owner")?.0.session.report().output};
                    targets.as_array().context("class targets must be a list")?.iter().map(|value| {
                        if let Some(index)=value.as_u64() {
                            let index=usize::try_from(index).context("class index out of range")?;
                            ensure!(index<classes,"class index out of range");Ok(index)
                        } else {
                            let index=value.as_f64().context("class index must be numeric")?;
                            ensure!(index.is_finite() && index>=0. && index.fract()==0. && index<(classes as f64),"class index must be an integer within output classes");
                            Ok(index as usize)
                        }
                    }).collect::<Result<Vec<_>>>()?
                } else { Vec::new() };
                let advantages = vec![1.; inputs.len()];
                let objective = || if loss == "cross_entropy" { Objective::Policy { actions: &actions, advantages: &advantages } } else { Objective::Mse(&matrix) };
                let value = if let Some((session, _)) = self.adaptive.get_mut(&handle) {
                    let mut remaining = 10_000_000;
                    if loss == "loss" {
                        if let Some(limit) = clip { session.train_clipped(&inputs, &matrix, rate, limit, &mut remaining, 64*1024*1024)? }
                        else { session.train_custom(&inputs, &matrix, rate, &mut remaining, 64*1024*1024)? }
                    } else { session.train_builtin_clipped(&inputs, objective(), rate, clip, &mut remaining, 64*1024*1024)? }
                } else {
                    ensure!(loss != "loss", "custom loss requires a forward model with fn loss");
                    self.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0.step_clipped(&inputs, objective(), rate, clip)?
                };
                Ok(serde_json::json!({"loss":value}))
            }
            Request::Drop { handle } => {
                let bytes = if let Some((_, bytes)) = self.sessions.remove(&handle) {
                    bytes
                } else {
                    self.adaptive
                        .remove(&handle)
                        .context("unknown session or foreign owner")?
                        .1
                };
                self.reserved -= bytes;
                Ok(serde_json::json!({"released": true}))
            }
            Request::PersistFd { .. } | Request::LoadFd { .. } | Request::InspectArtifactFd { .. } | Request::ExportWeightsFd { .. } | Request::BindWeightsFd { .. } => {
                bail!("file operation requires descriptor")
            }
            Request::CreateGraph {
                graph,
                optimizer,
                memory,
            } => {
                ensure!(
                    self.sessions.len() + self.adaptive.len() < 64
                        && memory > 0
                        && memory <= MAX_SESSION_BYTES
                        && self
                            .reserved
                            .checked_add(memory)
                            .is_some_and(|n| n <= MAX_CONNECTION_BYTES),
                    "graph admission budget exceeded"
                );
                let graph: wgtime_core::source::Graph = serde_json::from_value(graph)?;
                let optimizer = match optimizer.as_str() {
                    "adam" => Optimizer::Adam,
                    "sgd" => Optimizer::Sgd,
                    _ => bail!("unsupported optimizer"),
                };
                let trainer =
                    Trainer::new(graph.instantiate()?, Path::new("."), memory, optimizer)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.sessions.insert(handle, (trainer, memory));
                self.reserved += memory;
                Ok(serde_json::json!({"handle":handle}))
            }
            Request::RegisterStrategy { source, entry } => {
                ensure!(self.strategies.len() < 64, "strategy limit exceeded");
                let strategy = wgtime_service::strategy::Strategy::compile_entry(&source, &entry)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.strategies.insert(handle, strategy);
                Ok(serde_json::json!({"strategy": handle}))
            }
            Request::ExecuteStrategy {
                handle,
                strategy,
                inputs,
                targets,
                operations,
                memory,
            } => {
                ensure!(
                    operations > 0
                        && operations <= 100_000_000
                        && memory > 0
                        && memory <= MAX_SESSION_BYTES,
                    "strategy execution budget exceeded"
                );
                let strategy = self
                    .strategies
                    .get(&strategy)
                    .context("unknown strategy or foreign owner")?;
                let (session, _) = self
                    .adaptive
                    .get_mut(&handle)
                    .context("unknown adaptive session or foreign owner")?;
                let mut remaining = operations;
                strategy.execute(session, &inputs, &targets, &mut remaining, memory)
            }
            Request::Checkpoint { handle } => {
                ensure!(self.checkpoints.len() < 4, "checkpoint slot limit exceeded");
                let trainer = &mut self
                    .sessions
                    .get_mut(&handle)
                    .context("unknown session or foreign owner")?
                    .0;
                let bytes = trainer.checkpoint_bytes_bounded(16 * 1024 * 1024)?;
                let checkpoint = shared.next.fetch_add(1, Ordering::Relaxed);
                self.checkpoints.insert(checkpoint, bytes);
                Ok(serde_json::json!({"checkpoint": checkpoint}))
            }
            Request::Restore { checkpoint, memory } => {
                ensure!(
                    self.sessions.len() + self.adaptive.len() < 64
                        && memory > 0
                        && memory <= MAX_SESSION_BYTES
                        && self
                            .reserved
                            .checked_add(memory)
                            .is_some_and(|n| n <= MAX_CONNECTION_BYTES),
                    "restore budget exceeded"
                );
                let bytes = self
                    .checkpoints
                    .get(&checkpoint)
                    .context("unknown checkpoint or foreign owner")?;
                let trainer = Trainer::restore_value(serde_json::from_slice(bytes)?, memory)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.sessions.insert(handle, (trainer, memory));
                self.reserved += memory;
                Ok(serde_json::json!({"handle": handle}))
            }
            Request::RegisterTensor { .. } => {
                bail!("descriptor registration requires data channel")
            }
            Request::PredictBinary { .. } => bail!("binary request requires binary response"),
            Request::PredictTensor {
                handle,
                tensor,
                start,
                count,
            } => {
                let mapping = self
                    .tensors
                    .get(&tensor)
                    .context("unknown tensor or foreign owner")?;
                let end = start.checked_add(count).context("tensor range overflow")?;
                let input = mapping
                    .values()
                    .get(start..end)
                    .context("tensor range out of bounds")?;
                let trainer = &mut self
                    .sessions
                    .get_mut(&handle)
                    .context("unknown session or foreign owner")?
                    .0;
                Ok(serde_json::json!({"output": trainer.predict(input)?, "steps": trainer.steps()}))
            }
            Request::ReleaseTensor { tensor } => {
                self.tensors
                    .remove(&tensor)
                    .context("unknown tensor or foreign owner")?;
                Ok(serde_json::json!({"released": true}))
            }
            Request::Stats => Ok(
                serde_json::json!({"sessions": self.sessions.len() + self.adaptive.len(), "reserved": self.reserved, "connections": shared.connections.load(Ordering::Relaxed),"allocator":metrics::snapshot()}),
            ),
            Request::Hello { .. } => unreachable!(),
        }
    }
}
fn connection(mut stream: UnixStream, shared: Arc<Shared>) {
    let result = (|| -> Result<()> {
        ensure!(
            wgtime::peer_uid(&stream)? == unsafe { libc::geteuid() },
            "unauthorized client UID"
        );
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        let mut state = Connection {
            sessions: HashMap::new(),
            reserved: 0,
            ready: false,
            tensors: HashMap::new(),
            checkpoints: HashMap::new(),
            adaptive: HashMap::new(),
            strategies: HashMap::new(),
        };
        let mut response_bytes=Vec::new();
        #[cfg(feature = "latency-probe")]
        let mut probe_samples = Vec::with_capacity(8192);
        loop {
            let request = match wgtime::receive_request(&mut stream) {
                Ok(request) => request,
                Err(_) => break,
            };
            if matches!(request, Request::PersistFd { .. } | Request::LoadFd { .. } | Request::InspectArtifactFd { .. } | Request::ExportWeightsFd { .. } | Request::BindWeightsFd { .. }) {
                use std::io::{Read, Seek, Write};
                let mut file = wgtime::buffer::receive_file(&stream)?;
                let result = (|| -> Result<serde_json::Value> {
                    ensure!(
                        state.ready && file.metadata()?.is_file(),
                        "invalid persistence descriptor"
                    );
                    match request {
                        Request::ExportWeightsFd { handle } => {
                            ensure!(file.metadata()?.len() == 0, "destination must be empty staging file");
                            let bytes = if let Some((session, _)) = state.adaptive.get(&handle) {
                                session.weights_bytes(64 * 1024 * 1024)?
                            } else {
                                state.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0.weights_bytes(64 * 1024 * 1024)?
                            };
                            file.write_all(&bytes)?;
                            file.sync_all()?;
                            Ok(serde_json::json!({"bytes":bytes.len()}))
                        }
                        Request::BindWeightsFd { handle, optimizer } => {
                            ensure!(file.metadata()?.len() <= 64 * 1024 * 1024, "weights byte limit exceeded");
                            file.rewind()?;
                            let mut bytes = Vec::new();
                            file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
                            ensure!(bytes.len() <= 64 * 1024 * 1024, "weights byte limit exceeded");
                            if let Some((session, _)) = state.adaptive.get_mut(&handle) {
                                session.bind_weights(&bytes, &optimizer)?;
                            } else {
                                state.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0.bind_weights(&bytes, &optimizer)?;
                            }
                            Ok(serde_json::json!({}))
                        }
                        Request::PersistFd { handle, checkpoint } => {
                            ensure!(
                                file.metadata()?.len() == 0,
                                "destination must be empty staging file"
                            );
                            let bytes = if let Some((session, _)) = state.adaptive.get(&handle) {
                                if checkpoint {
                                    session.checkpoint_bytes_bounded(64 * 1024 * 1024)?
                                } else {
                                    serde_json::to_vec(session.program())?
                                }
                            } else {
                                let trainer = &mut state
                                    .sessions
                                    .get_mut(&handle)
                                    .context("unknown session or foreign owner")?
                                    .0;
                                if checkpoint {
                                    trainer.checkpoint_bytes_bounded(64 * 1024 * 1024)?
                                } else {
                                    serde_json::to_vec(&trainer.snapshot()?)?
                                }
                            };
                            ensure!(
                                bytes.len() <= 64 * 1024 * 1024,
                                "persistence byte limit exceeded"
                            );
                            file.write_all(&bytes)?;
                            file.sync_all()?;
                            Ok(serde_json::json!({"bytes":bytes.len()}))
                        }
                        Request::InspectArtifactFd { checkpoint, memory } => {
                            ensure!(memory > 0 && memory <= MAX_SESSION_BYTES, "inspection memory budget exceeded");
                            ensure!(file.metadata()?.len() <= 64 * 1024 * 1024, "artifact byte limit exceeded");
                            file.rewind()?;
                            let mut bytes = Vec::new();
                            file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
                            ensure!(bytes.len() <= 64 * 1024 * 1024, "artifact byte limit exceeded");
                            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                            let trainer = if checkpoint {
                                Trainer::restore_value(value, memory)?
                            } else {
                                Trainer::new(serde_json::from_value(value)?, Path::new("."), memory, Optimizer::Sgd)?
                            };
                            let report = trainer.session.report();
                            Ok(serde_json::json!({"input":report.input,"output":report.output,"steps":trainer.steps()}))
                        }
                        Request::LoadFd {
                            checkpoint,
                            optimizer,
                            memory,
                        } => {
                            ensure!(
                                state.sessions.len() + state.adaptive.len() < 64
                                    && memory > 0
                                    && memory <= MAX_SESSION_BYTES
                                    && state
                                        .reserved
                                        .checked_add(memory)
                                        .is_some_and(|n| n <= MAX_CONNECTION_BYTES),
                                "load admission budget exceeded"
                            );
                            ensure!(
                                file.metadata()?.len() <= 64 * 1024 * 1024,
                                "checkpoint byte limit exceeded"
                            );
                            file.rewind()?;
                            let mut bytes = Vec::new();
                            file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
                            ensure!(
                                bytes.len() <= 64 * 1024 * 1024,
                                "checkpoint byte limit exceeded"
                            );
                            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                            let artifact_checkpoint = match value.get("format") {
                                None => false,
                                Some(tag) if matches!(tag.as_str(), Some(
                                    "weightlang.forward-training.v1"
                                        | "weightlang.forward-training.v2"
                                        | "weightlang.native-training.v1"
                                )) => true,
                                Some(_) => bail!("unsupported checkpoint format"),
                            };
                            ensure!(
                                checkpoint == artifact_checkpoint,
                                "load format does not match model/checkpoint kind"
                            );
                            let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                            if value
                                .get("format")
                                .and_then(|n| n.as_str())
                                .is_some_and(|n| n.starts_with("weightlang.forward"))
                                || (!checkpoint && value.get("instructions").is_some())
                            {
                                let mut session = if checkpoint {
                                    wgtime_core::adaptive::AdaptiveSession::restore_checkpoint(value,memory)?
                                } else {
                                    wgtime_core::adaptive::AdaptiveSession::prepare(
                                        serde_json::from_value(value)?,
                                        memory,
                                    )?
                                };
                                if !checkpoint {
                                    match optimizer.as_str() {
                                        "none" => {}
                                        "adam" => session.set_optimizer(Optimizer::Adam)?,
                                        "sgd" => session.set_optimizer(Optimizer::Sgd)?,
                                        _ => bail!("unsupported optimizer"),
                                    }
                                }
                                state.adaptive.insert(handle, (session, memory));
                            } else {
                                let trainer = if checkpoint {
                                    Trainer::restore_value(value, memory)?
                                } else {
                                    let optimizer = match optimizer.as_str() {
                                        "adam" => Optimizer::Adam,
                                        "sgd" => Optimizer::Sgd,
                                        _ => bail!("unsupported optimizer"),
                                    };
                                    Trainer::new(
                                        serde_json::from_value(value)?,
                                        Path::new("."),
                                        memory,
                                        optimizer,
                                    )?
                                };
                                state.sessions.insert(handle, (trainer, memory));
                            }
                            state.reserved += memory;
                            Ok(serde_json::json!({"handle":handle}))
                        }
                        _ => unreachable!(),
                    }
                })();
                send(&mut stream, &Reply::result(result))?;
                continue;
            }
            if let Request::PredictBinary {
                handle,
                tensor,
                start,
                count,
            } = request
            {
                use std::io::Write;
                #[cfg(feature = "latency-probe")]
                let probe_begin = std::time::Instant::now();
                #[cfg(feature = "latency-probe")]
                let echo = count & (1usize << 63) != 0;
                #[cfg(feature = "latency-probe")]
                let count = count & !(1usize << 63);
                #[cfg(feature = "latency-probe")]
                let mut marks = [0u64; 3];
                let result = (|| -> Result<()> {
                    ensure!(state.ready, "handshake required");
                    let mapping = state
                        .tensors
                        .get(&tensor)
                        .context("unknown tensor or foreign owner")?;
                    let end = start.checked_add(count).context("tensor range overflow")?;
                    let input = mapping
                        .values()
                        .get(start..end)
                        .context("tensor range out of bounds")?;
                    let trainer = &mut state
                        .sessions
                        .get_mut(&handle)
                        .context("unknown session or foreign owner")?
                        .0;
                    #[cfg(feature = "latency-probe")]
                    ensure!(input.len() == trainer.session.report().input, "input length mismatch");
                    #[cfg(feature = "latency-probe")]
                    { marks[0] = probe_begin.elapsed().as_nanos() as u64; }
                    #[cfg(feature = "latency-probe")]
                    let output = if echo { &[][..] } else {
                        trainer.session.run(input).map_err(anyhow::Error::msg)?
                    };
                    #[cfg(not(feature = "latency-probe"))]
                    let output = trainer.session.run(input).map_err(anyhow::Error::msg)?;
                    #[cfg(feature = "latency-probe")]
                    { marks[1] = probe_begin.elapsed().as_nanos() as u64; }
                    ensure!(output.iter().all(|n| n.is_finite()), "nonfinite prediction");
                    // Encode while borrowing the session output. The connection executes
                    // serially, so no update can mutate it before this copy completes.
                    let size = output
                        .len()
                        .checked_mul(4)
                        .context("output size overflow")?;
                    ensure!(size <= wgtime::MAX_FRAME, "output frame exceeds limit");
                    response_bytes.clear();
                    response_bytes.extend_from_slice(&0u32.to_le_bytes());
                    response_bytes.extend_from_slice(&(size as u32).to_le_bytes());
                    // f32 storage is initialized and borrowed through this synchronous copy.
                    response_bytes.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), size)
                    });
                    Ok(())
                })();
                match result {
                    Ok(()) => {
                        #[cfg(feature = "latency-probe")]
                        { marks[2] = probe_begin.elapsed().as_nanos() as u64; }
                        stream.write_all(&response_bytes)?;
                        #[cfg(feature = "latency-probe")]
                        if probe_samples.len() < 8192 {
                            probe_samples.push((echo, marks, probe_begin.elapsed().as_nanos() as u64));
                        }
                    },
                    Err(error) => {
                        let error = format!("{error:#}");
                        stream.write_all(&1u32.to_le_bytes())?;
                        stream.write_all(&(error.len() as u32).to_le_bytes())?;
                        stream.write_all(error.as_bytes())?;
                    }
                }
                continue;
            }
            #[cfg(feature = "latency-probe")]
            if matches!(request, Request::Stats) {
                let mut stats = state.execute(Request::Stats, &shared)?;
                stats["latency_samples"] = serde_json::to_value(std::mem::take(&mut probe_samples))?;
                stats["sample_format"] = serde_json::json!("[echo,[validation_end,compute_end,encode_end],write_end] ns since dispatch; excludes receive/parse; write is not client receipt");
                send(&mut stream, &Reply::result(Ok(stats)))?;
                probe_samples = Vec::with_capacity(8192);
                continue;
            }
            let result = if let Request::RegisterTensor {
                count,
                shape,
                dtype,
            } = request
            {
                // Consume the descriptor even when metadata is rejected, preserving framing.
                let file = wgtime::buffer::receive_file(&stream)?;
                (|| -> Result<serde_json::Value> {
                    ensure!(
                        state.ready && state.tensors.len() < 64,
                        "tensor admission limit exceeded"
                    );
                    ensure!(
                        dtype == "f32"
                            && !shape.is_empty()
                            && shape.len() <= 8
                            && shape.iter().all(|n| *n > 0),
                        "invalid tensor dtype or shape"
                    );
                    let elements = shape
                        .iter()
                        .try_fold(1usize, |n, width| n.checked_mul(*width))
                        .context("shape overflow")?;
                    ensure!(elements == count, "shape extent mismatch");
                    let mapping = wgtime::buffer::Mapping::open(file, count)?;
                    let tensor = shared.next.fetch_add(1, Ordering::Relaxed);
                    state.tensors.insert(tensor, mapping);
                    Ok(serde_json::json!({"tensor": tensor}))
                })()
            } else {
                state.execute(request, &shared)
            };
            send(&mut stream, &Reply::result(result))?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("wgtime connection: {error:#}");
    }
    shared.connections.fetch_sub(1, Ordering::Relaxed);
}
#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn listener_guard_removes_socket_before_unlocking_and_allows_restart() -> Result<()> {
        let base = std::env::var_os("CLAUDE_JOB_DIR")
            .map(|path| PathBuf::from(path).join("tmp"))
            .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
        fs::create_dir_all(&base)?;
        let directory = base.join(format!("wgtime-guard-{}", std::process::id()));
        fs::create_dir(&directory)?;
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
        let (listener, guard) = listen(&directory)?;
        let lock = File::open(directory.join("service.lock"))?;
        assert_eq!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            -1
        );
        assert!(listen(&directory).is_err());
        assert!(directory.join("service.sock").exists());
        drop(listener);
        drop(guard);
        assert!(!directory.join("service.sock").exists());
        assert_eq!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_UN) }, 0);
        let (listener, guard) = listen(&directory)?;
        assert!(directory.join("service.sock").exists());
        drop(listener);
        drop(guard);
        assert!(!directory.join("service.sock").exists());
        Ok(())
    }

    fn shared() -> Arc<Shared> {
        Arc::new(Shared {
            next: AtomicU64::new(1),
            connections: AtomicUsize::new(0),
        })
    }

    fn state() -> Connection {
        Connection {
            sessions: HashMap::new(),
            reserved: 0,
            ready: true,
            tensors: HashMap::new(),
            checkpoints: HashMap::new(),
            adaptive: HashMap::new(),
            strategies: HashMap::new(),
        }
    }

    fn program(memory: usize) -> Request {
        Request::CreateTrainingProgram {
            program: serde_json::json!({
                "version": 1, "input": "x",
                "registers": [{"name": "x", "len": 1, "shape": [1]}],
                "parameters": [], "instructions": [{"op": "return", "value": "x"}]
            }),
            memory,
            optimizer: "none".into(),
        }
    }

    fn create() -> Request {
        Request::Create {
            widths: vec![1, 1],
            activation: "tanh".into(),
            seed: 1,
            optimizer: "sgd".into(),
            memory: 1024 * 1024,
        }
    }

    #[test]
    fn adaptive_drop_returns_reservation_and_rejects_stale_or_foreign_handles() -> Result<()> {
        let shared = shared();
        let mut owner = state();
        let handle = owner.execute(program(MAX_SESSION_BYTES), &shared)?["handle"]
            .as_u64()
            .unwrap();
        assert!(state().execute(Request::Drop { handle }, &shared).is_err());
        assert_eq!(owner.reserved, MAX_SESSION_BYTES);
        owner.execute(Request::Drop { handle }, &shared)?;
        assert_eq!(owner.reserved, 0);
        assert!(owner.execute(Request::Drop { handle }, &shared).is_err());
        assert!(owner
            .execute(
                Request::Predict {
                    handle,
                    input: vec![1.]
                },
                &shared
            )
            .is_err());
        for _ in 0..3 {
            let handle = owner.execute(program(MAX_SESSION_BYTES), &shared)?["handle"]
                .as_u64()
                .unwrap();
            owner.execute(Request::Drop { handle }, &shared)?;
        }
        assert_eq!(owner.execute(Request::Stats, &shared)?["sessions"], 0);
        Ok(())
    }

    #[test]
    fn mixed_models_share_admission_limit_and_failures_preserve_reservation() -> Result<()> {
        let shared = shared();
        let mut owner = state();
        let trainer = owner.execute(create(), &shared)?["handle"]
            .as_u64()
            .unwrap();
        let checkpoint = owner.execute(Request::Checkpoint { handle: trainer }, &shared)?
            ["checkpoint"]
            .as_u64()
            .unwrap();
        let mut adaptive = 0;
        for _ in 0..63 {
            adaptive = owner.execute(program(1024), &shared)?["handle"]
                .as_u64()
                .unwrap();
        }
        let reserved = owner.reserved;
        assert!(owner.execute(create(), &shared).is_err());
        assert!(owner
            .execute(
                Request::Restore {
                    checkpoint,
                    memory: 1024 * 1024
                },
                &shared
            )
            .is_err());
        assert_eq!(owner.reserved, reserved);
        assert_eq!(owner.execute(Request::Stats, &shared)?["sessions"], 64);
        owner.execute(Request::Drop { handle: adaptive }, &shared)?;
        owner.execute(
            Request::Restore {
                checkpoint,
                memory: 1024 * 1024,
            },
            &shared,
        )?;
        let reserved = owner.reserved;
        assert!(owner
            .execute(Request::Drop { handle: u64::MAX }, &shared)
            .is_err());
        assert_eq!(owner.reserved, reserved);
        Ok(())
    }

    #[test]
    fn tensor_release_rejects_foreign_and_stale_handles() -> Result<()> {
        let shared = shared();
        let mut owner = state();
        let mut tensor = wgtime::buffer::WritableTensor::new(1)?;
        tensor.write(&[1.])?;
        let tensor = tensor.commit()?;
        let (sender, receiver) = UnixStream::pair()?;
        tensor.send(&sender)?;
        owner.tensors.insert(
            7,
            wgtime::buffer::Mapping::open(wgtime::buffer::receive_file(&receiver)?, 1)?,
        );
        assert!(state()
            .execute(Request::ReleaseTensor { tensor: 7 }, &shared)
            .is_err());
        assert_eq!(owner.tensors[&7].values(), &[1.]);
        owner.execute(Request::ReleaseTensor { tensor: 7 }, &shared)?;
        assert!(owner.tensors.is_empty());
        assert!(owner
            .execute(Request::ReleaseTensor { tensor: 7 }, &shared)
            .is_err());
        Ok(())
    }

    #[test]
    fn fd_loaded_program_drop_returns_reservation() -> Result<()> {
        use std::io::Write;
        use std::os::fd::FromRawFd;
        let shared = shared();
        shared.connections.store(1, Ordering::Relaxed);
        let (mut client, server) = UnixStream::pair()?;
        let worker_shared = shared.clone();
        let worker = std::thread::spawn(move || connection(server, worker_shared));
        send(&mut client, &Request::Hello { protocol: PROTOCOL })?;
        let _: Reply = receive(&mut client)?;
        let fd =
            unsafe { libc::memfd_create(c"wgtime-lifecycle-test".as_ptr(), libc::MFD_CLOEXEC) };
        ensure!(fd >= 0, "test memfd creation failed");
        let mut file = unsafe { File::from_raw_fd(fd) };
        let Request::CreateTrainingProgram { program, .. } = program(MAX_SESSION_BYTES) else {
            unreachable!()
        };
        file.write_all(&serde_json::to_vec(&program)?)?;
        send(
            &mut client,
            &Request::LoadFd {
                checkpoint: false,
                optimizer: "none".into(),
                memory: MAX_SESSION_BYTES,
            },
        )?;
        wgtime::buffer::send_fd(&client, file.as_raw_fd())?;
        let reply: Reply = receive(&mut client)?;
        assert!(reply.error.is_none(), "{reply:?}");
        let handle = reply.value.unwrap()["handle"].as_u64().unwrap();
        send(&mut client, &Request::Stats)?;
        let reply: Reply = receive(&mut client)?;
        assert_eq!(reply.value.unwrap()["reserved"], MAX_SESSION_BYTES);
        send(&mut client, &Request::Drop { handle })?;
        let reply: Reply = receive(&mut client)?;
        assert!(reply.error.is_none(), "{reply:?}");
        send(&mut client, &Request::Stats)?;
        let reply: Reply = receive(&mut client)?;
        assert_eq!(reply.value.unwrap()["reserved"], 0);
        drop(client);
        worker.join().unwrap();
        assert_eq!(shared.connections.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn eof_and_malformed_frames_return_connection_slot() -> Result<()> {
        for malformed in [false, true] {
            let shared = shared();
            shared.connections.store(1, Ordering::Relaxed);
            let (mut client, server) = UnixStream::pair()?;
            let worker_shared = shared.clone();
            let worker = std::thread::spawn(move || connection(server, worker_shared));
            send(&mut client, &Request::Hello { protocol: PROTOCOL })?;
            let _: Reply = receive(&mut client)?;
            send(&mut client, &program(MAX_SESSION_BYTES))?;
            let reply: Reply = receive(&mut client)?;
            assert!(reply.error.is_none(), "{reply:?}");
            if malformed {
                use std::io::Write;
                client.write_all(&1u32.to_le_bytes())?;
                client.write_all(b"!")?;
            }
            drop(client);
            worker.join().unwrap();
            assert_eq!(shared.connections.load(Ordering::Relaxed), 0);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let directory = match args.next() {
        None => PathBuf::from("/run/wgtime"),
        Some(arg) if arg == "--test-directory" => {
            PathBuf::from(args.next().context("missing test directory")?)
        }
        _ => bail!("usage: wgtime [--test-directory DIRECTORY]"),
    };
    ensure!(args.next().is_none(), "unexpected argument");
    let (listener, _guard) = listen(&directory)?;
    let mut random = [0; 8];
    use std::io::Read;
    File::open("/dev/urandom")?.read_exact(&mut random)?;
    let shared = Arc::new(Shared {
        next: AtomicU64::new(u64::from_ne_bytes(random) & 0x7fff_ffff_ffff_0000),
        connections: AtomicUsize::new(0),
    });
    for stream in listener.incoming() {
        let stream = stream?;
        if shared.connections.fetch_add(1, Ordering::Relaxed) >= MAX_CONNECTIONS {
            shared.connections.fetch_sub(1, Ordering::Relaxed);
            continue;
        }
        let worker_shared = shared.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("wgtime-client".into())
            .spawn(move || connection(stream, worker_shared))
        {
            shared.connections.fetch_sub(1, Ordering::Relaxed);
            eprintln!("cannot start connection worker: {error}");
        }
    }
    Ok(())
}
