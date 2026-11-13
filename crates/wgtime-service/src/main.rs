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
use wgtime::{send, Reply, Request, PROTOCOL};
#[cfg(test)]
use wgtime::receive;

mod admission;
mod metrics;
mod model_lifecycle;
mod persistence;
#[cfg(test)]
mod weights_tests;
mod sessions;
use sessions::{Session, Sessions, MAX_SESSION_BYTES};
#[cfg(feature="allocation-metrics")]
#[global_allocator]
static ALLOCATOR: metrics::Counting=metrics::Counting;
const MAX_CONNECTIONS: usize = 64;
struct Shared {
    next: AtomicU64,
    connections: AtomicUsize,
}
impl Shared {
    fn next_handle(&self) -> Result<u64> {
        let mut current = self.next.load(Ordering::Relaxed);
        loop {
            let next = current.checked_add(1).context("handle namespace exhausted")?;
            match self.next.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(handle) => return Ok(handle),
                Err(actual) => current = actual,
            }
        }
    }
}
// The accept loop owns admission; this guard owns exactly one admitted slot.
// It also runs when a worker unwinds, after its connection-local state is dropped.
struct ConnectionSlot<'a>(&'a AtomicUsize);
impl Drop for ConnectionSlot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
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
    sessions: Sessions,
    ready: bool,
    tensors: HashMap<u64, wgtime::buffer::Mapping>,
    checkpoints: HashMap<u64, Vec<u8>>,
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
                let admission = self.sessions.admit(memory)?;
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
                let trainer = Trainer::new_self_contained(
                    mlp(&widths, &activation, seed)?,
                    memory,
                    optimizer,
                )?;
                let handle = shared.next_handle()?;
                admission.insert(handle, Session::Trainer(trainer));
                Ok(serde_json::json!({"handle": handle}))
            }
            Request::InitializeDefinition { source, memory, optimizer } => {
                self.initialize_definition(source, memory, optimizer, None, shared)
            }
            Request::RebindDefinition { handle, source } => {
                let definition = wgt_language::model::ModelDefinition::parse(&source)?;
                let wgt_language::model::ModelPlan::Forward(program) = definition.compile_plan_with_modules()?.0 else {
                    bail!("geometry rebinding requires a differentiable forward plan");
                };
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("geometry rebinding requires an owned forward model");
                };
                session.rebind_geometry(program)?;
                Ok(serde_json::json!({"backend":session.execution_backend(),"revision":session.revision(),"forward_interface":definition.forward_interface()}))
            }
            Request::CallOperation { handle, name, inputs } => {
                let Some(Session::Operations(session)) = self.sessions.get_mut(&handle) else {
                    bail!("unknown operation model or foreign owner");
                };
                let output = session.call(&name, &inputs, 1_000_000)?;
                Ok(serde_json::json!({"output":output,"revision":session.revision()}))
            }
            Request::OperationSchemas { handle } => {
                let Some(Session::Operations(session)) = self.sessions.get(&handle) else {
                    bail!("unknown operation model or foreign owner");
                };
                Ok(serde_json::to_value(session.schemas())?)
            }
            Request::StateValues { handle } => {
                let Some(Session::Operations(session)) = self.sessions.get(&handle) else {
                    bail!("unknown operation model or foreign owner");
                };
                Ok(serde_json::to_value(session.state_values())?)
            }
            Request::ParameterValues { handle } => match self.sessions.get(&handle) {
                Some(Session::Operations(session)) => Ok(serde_json::to_value(session.parameter_values())?),
                Some(Session::Adaptive(session)) => Ok(serde_json::to_value(session.program().parameters.iter().map(|p| (p.name.clone(), p.shape.clone(), p.data.clone())).collect::<Vec<_>>())?),
                _ => bail!("unknown model or foreign owner"),
            },
            Request::SetOptimizer { handle, optimizer } => {
                let optimizer=match optimizer.as_str() {
                    "sgd" => Optimizer::Sgd,
                    "adam" => Optimizer::Adam,
                    _ => bail!("unsupported optimizer"),
                };
                let Some(Session::Adaptive(session))=self.sessions.get_mut(&handle) else {
                    bail!("optimizer configuration requires an owned forward model");
                };
                session.set_optimizer(optimizer)?;
                Ok(serde_json::json!({"revision":session.revision()}))
            }
            Request::UpdateParameters { handle, candidates } => {
                let revision = match self.sessions.get_mut(&handle) {
                    Some(Session::Operations(session)) => {
                        session.update_parameters(&candidates)?;
                        session.revision()
                    }
                    Some(Session::Adaptive(session)) => {
                        session.update_candidates(&candidates)?;
                        session.revision()
                    }
                    _ => bail!("unknown model or foreign owner"),
                };
                Ok(serde_json::json!({"revision": revision}))
            }
            Request::TrainRecurrent { handle, inputs, targets, input_offset, output_offset,
                state_width, supervised_offset, supervised_width, rate, clip } => {
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("recurrent differentiation requires an owned forward program");
                };
                let loss=session.train_recurrent(&inputs,&targets,
                    wgtime_core::adaptive::recurrent::StateLink {input_offset,output_offset,width:state_width},
                    wgtime_core::adaptive::recurrent::OutputSlice {offset:supervised_offset,width:supervised_width},
                    rate,clip,&mut 1_000_000,64*1024*1024)?;
                Ok(serde_json::json!({"loss":loss,"revision":session.revision()}))
            }
            Request::TrainStepWithPolicy { handle, inputs, targets, rate, loss, clip, straight_through } => {
                ensure!(!matches!(self.sessions.get(&handle), Some(Session::Operations(_))), "typed operations do not support differentiation or gradients across state");
                let Some(Session::Adaptive(session)) = self.sessions.get(&handle) else {
                    bail!("unknown model or foreign owner");
                };
                ensure!(straight_through || !session.program().parameters.iter().any(|p| matches!(p.learn,Some(wgt_language::adaptive::LearnDomain::Bool | wgt_language::adaptive::LearnDomain::Int{..}))), "ContinuousOnly rejects models containing bool/int parameters; explicitly select StraightThrough");
                self.execute(Request::TrainStep {handle,inputs,targets,rate,loss,clip},shared)
            }
            Request::Predict { handle, input } => match self.sessions.get_mut(&handle) {
                Some(Session::Adaptive(session)) => {
                    let output = session.run(&input, 1_000_000)?.to_vec();
                    Ok(serde_json::json!({"output": output, "revision": session.revision()}))
                }
                Some(Session::Trainer(trainer)) => Ok(serde_json::json!({"output": trainer.predict(&input)?, "steps": trainer.steps()})),
                _ => bail!("unknown session or foreign owner"),
            },
            Request::FeatureEffects { handle, input, seed, dictionary, instruction, occurrence, max_steps, max_bytes } => {
                ensure!(max_steps > 0 && max_steps <= 1_000_000 && max_bytes <= 262144,
                    "IPC feature experiment budget exceeded");
                let dictionary = serde_json::from_value::<wgtime_core::adaptive::research::FeatureDictionary>(dictionary)?;
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("feature effects require an owned forward program");
                };
                Ok(serde_json::to_value(session.feature_effects(&input, &seed, &dictionary,
                    instruction, occurrence, max_steps, max_bytes)?)?)
            },
            Request::AttributeActivations { handle, input, seed, max_steps, max_bytes } => {
                ensure!(max_steps > 0 && max_steps <= 1_000_000 && max_bytes <= 262144,
                    "IPC attribution budget exceeded");
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("attribution requires an owned forward program");
                };
                let report = session.attribute(&input, &seed, max_steps, max_bytes)?;
                ensure!(report.events.len() <= 1024, "IPC attribution event limit exceeded");
                Ok(serde_json::to_value(report)?)
            },
            Request::DiscoverCircuit { handle, input, seed, max_nodes, max_steps, max_bytes, epsilon } => {
                ensure!(max_steps > 0 && max_steps <= 1_000_000 && max_bytes <= 262144 && max_nodes <= 32,
                    "IPC circuit budget exceeded");
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("circuit discovery requires an owned forward program");
                };
                Ok(serde_json::to_value(session.circuit(&input, &seed, max_nodes, max_steps, max_bytes, epsilon)?)?)
            },
            Request::InspectActivations { handle, input, replacements, interventions, max_events, max_bytes, max_steps } => {
                ensure!(max_events <= 1024 && max_bytes <= 262144, "IPC research capture limit exceeded");
                ensure!(max_steps > 0 && max_steps <= 1_000_000, "invalid research step budget");
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("activation inspection requires an owned forward program");
                };
                let mut research = wgtime_core::adaptive::research::Research::new(max_events, max_bytes);
                research.replacements = replacements;
                research.interventions = interventions.into_iter().map(|patch|
                    wgtime_core::adaptive::research::Intervention {
                        instruction: patch.instruction, occurrence: patch.occurrence, value: patch.value,
                    }).collect();
                let (output, research) = session.inspect(&input, max_steps, research)?;
                Ok(serde_json::json!({"output": output, "events": research.events}))
            },
            Request::CreateTrainingProgram { program, memory, optimizer } => {
                let admission = self.sessions.admit(memory).context("program admission budget exceeded")?;
                ensure!(matches!(optimizer.as_str(), "none"|"sgd"|"adam"), "unknown optimizer");
                let mut session = wgtime_core::adaptive::AdaptiveSession::prepare(serde_json::from_value(program)?, memory)?;
                match optimizer.as_str() { "sgd"=>session.set_optimizer(Optimizer::Sgd)?, "adam"=>session.set_optimizer(Optimizer::Adam)?, _=>{} }
                let handle = shared.next_handle()?;
                admission.insert(handle, Session::Adaptive(session));
                Ok(serde_json::json!({"handle":handle}))
            }
            Request::TrainStep { handle, inputs, targets, rate, loss, clip } => {
                ensure!(!matches!(self.sessions.get(&handle), Some(Session::Operations(_))), "typed operations do not support differentiation or gradients across state");
                ensure!(!inputs.is_empty() && inputs.len() <= 65536, "invalid batch size");
                ensure!(rate.is_finite() && rate > 0., "learning rate must be positive and finite");
                ensure!(clip.is_none_or(|v| v.is_finite() && v > 0.), "gradient clip norm must be positive and finite");
                ensure!(matches!(loss.as_str(), "mse" | "cross_entropy" | "loss"), "unknown training loss");
                let matrix: Vec<Vec<f32>> = if loss == "cross_entropy" { Vec::new() } else { serde_json::from_value(targets.clone())? };
                let actions: Vec<usize> = if loss == "cross_entropy" {
                    let classes = match self.sessions.get(&handle) {
                        Some(Session::Adaptive(session)) => session.output_len(),
                        Some(Session::Trainer(trainer)) => trainer.session.report().output,
                        _ => bail!("unknown session or foreign owner"),
                    };
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
                ensure!(loss != "loss" || matches!(self.sessions.get(&handle), Some(Session::Adaptive(_))), "custom loss requires a forward model with fn loss");
                let value = match self.sessions.get_mut(&handle) {
                    Some(Session::Adaptive(session)) => {
                        let mut remaining = 10_000_000;
                        if loss == "loss" {
                            if let Some(limit) = clip { session.train_clipped(&inputs, &matrix, rate, limit, &mut remaining, 64*1024*1024)? }
                            else { session.train_custom(&inputs, &matrix, rate, &mut remaining, 64*1024*1024)? }
                        } else { session.train_builtin_clipped(&inputs, objective(), rate, clip, &mut remaining, 64*1024*1024)? }
                    }
                    Some(Session::Trainer(trainer)) => {
                        trainer.step_clipped(&inputs, objective(), rate, clip)?
                    }
                    _ => bail!("unknown session or foreign owner"),
                };
                Ok(serde_json::json!({"loss":value}))
            }
            Request::Drop { handle } => {
                self.sessions.remove(&handle)?;
                Ok(serde_json::json!({"released": true}))
            }
            Request::PersistFd { .. } | Request::LoadFd { .. } | Request::InspectArtifactFd { .. } | Request::ExportWeightsFd { .. } | Request::BindWeightsFd { .. } | Request::InitializeWeightsFd { .. } => {
                bail!("file operation requires descriptor")
            }
            Request::CreateGraph {
                graph,
                optimizer,
                memory,
            } => {
                let admission = self.sessions.admit(memory).context("graph admission budget exceeded")?;
                let graph: wgtime_core::source::Graph = serde_json::from_value(graph)?;
                let optimizer = match optimizer.as_str() {
                    "adam" => Optimizer::Adam,
                    "sgd" => Optimizer::Sgd,
                    _ => bail!("unsupported optimizer"),
                };
                let trainer =
                    Trainer::new_self_contained(graph.instantiate_bounded(memory)?, memory, optimizer)?;
                let handle = shared.next_handle()?;
                admission.insert(handle, Session::Trainer(trainer));
                Ok(serde_json::json!({"handle":handle}))
            }
            Request::RegisterStrategy { source, entry } => {
                ensure!(self.strategies.len() < 64, "strategy limit exceeded");
                let strategy = wgtime_service::strategy::Strategy::compile_entry(&source, &entry)?;
                let handle = shared.next_handle()?;
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
                let Some(Session::Adaptive(session)) = self.sessions.get_mut(&handle) else {
                    bail!("unknown adaptive session or foreign owner");
                };
                let mut remaining = operations;
                strategy.execute(session, &inputs, &targets, &mut remaining, memory)
            }
            Request::Checkpoint { handle } => {
                ensure!(self.checkpoints.len() < 4, "checkpoint slot limit exceeded");
                let Some(Session::Trainer(trainer)) = self.sessions.get_mut(&handle) else {
                    bail!("unknown session or foreign owner");
                };
                let bytes = trainer.checkpoint_bytes_bounded(16 * 1024 * 1024)?;
                let checkpoint = shared.next_handle()?;
                self.checkpoints.insert(checkpoint, bytes);
                Ok(serde_json::json!({"checkpoint": checkpoint}))
            }
            Request::Restore { checkpoint, memory } => {
                let admission = self.sessions.admit(memory).context("restore budget exceeded")?;
                let bytes = self
                    .checkpoints
                    .get(&checkpoint)
                    .context("unknown checkpoint or foreign owner")?;
                let trainer = Trainer::restore_self_contained_value(serde_json::from_slice(bytes)?, memory)?;
                let handle = shared.next_handle()?;
                admission.insert(handle, Session::Trainer(trainer));
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
                let Some(Session::Trainer(trainer)) = self.sessions.get_mut(&handle) else {
                    bail!("unknown session or foreign owner");
                };
                Ok(serde_json::json!({"output": trainer.predict(input)?, "steps": trainer.steps()}))
            }
            Request::ReleaseTensor { tensor } => {
                self.tensors
                    .remove(&tensor)
                    .context("unknown tensor or foreign owner")?;
                Ok(serde_json::json!({"released": true}))
            }
            Request::Stats => Ok(
                serde_json::json!({"sessions": self.sessions.len(), "reserved": self.sessions.reserved(), "connections": shared.connections.load(Ordering::Relaxed),"allocator":metrics::snapshot(),"execution":self.sessions.iter().filter_map(|(handle, session)| {
                    let Session::Adaptive(session) = session else { return None; };
                    let (forward,tape)=session.compiled_forward_counts();
                    Some(serde_json::json!({"handle":handle,"backend":session.execution_backend(),"native_forwards":forward,"native_tape_forwards":tape}))
                }).collect::<Vec<_>>()}),
            ),
            Request::Hello { .. } => unreachable!(),
        }
    }
}
fn connection(mut stream: UnixStream, shared: Arc<Shared>) {
    let _slot = ConnectionSlot(&shared.connections);
    let result = (|| -> Result<()> {
        ensure!(
            wgtime::peer_uid(&stream)? == unsafe { libc::geteuid() },
            "unauthorized client UID"
        );
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        let mut state = Connection {
            sessions: Sessions::default(),
            ready: false,
            tensors: HashMap::new(),
            checkpoints: HashMap::new(),
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
            if request.uses_persistence_descriptor() {
                let file = wgtime::buffer::receive_file(&stream)?;
                let result = state.execute_persistence(request, file, &shared);
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
                    let Some(Session::Trainer(trainer)) = state.sessions.get_mut(&handle) else {
                        bail!("unknown session or foreign owner");
                    };
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
                        wgtime::send_binary_error(&mut stream, &error)?;
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
                    let tensor = shared.next_handle()?;
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
}
#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    mod hardening;

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
            sessions: Sessions::default(),
            ready: true,
            tensors: HashMap::new(),
            checkpoints: HashMap::new(),
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

    fn operation() -> Request {
        Request::InitializeDefinition {
            source: "wgt 0.5\nmodel Cell\nstate activity:f32[1]=[0];\nfn observe(x:f32[1])->f32[1]{return activity+x;}\n".into(),
            memory: 1024 * 1024,
            optimizer: None,
        }
    }

    #[test]
    fn adaptive_predict_commits_writes_and_rolls_back_failed_output() -> Result<()> {
        let source = r#"wgt 0.5
model AdaptiveMemory
input x: f32[2]
output y: f32[1]
parameter memory: f32[1,1] = [0]
frozen updater: f32[1,1] = [0.5]
fn forward(x) {
    let observation = slice(x, 0, 1);
    let query = slice(x, 1, 1);
    let next = add(read(memory), linear(observation, updater));
    write(memory, next);
    return linear(query, memory);
}
"#;
        let shared = shared();
        let mut owner = state();
        let program = wgt_language::adaptive::parse(source)?;
        let handle = owner.execute(
            Request::CreateTrainingProgram {
                program: serde_json::to_value(program)?,
                memory: 1024 * 1024,
                optimizer: "none".into(),
            },
            &shared,
        )?["handle"]
            .as_u64()
            .unwrap();
        for (input, output, revision) in [([2., 3.], 3., 1), ([4., 3.], 9., 2)] {
            assert_eq!(
                owner.execute(Request::Predict { handle, input: input.into() }, &shared)?,
                serde_json::json!({"output": [output], "revision": revision})
            );
        }
        let parameters = owner.execute(Request::ParameterValues { handle }, &shared)?;
        let error = owner.execute(
            Request::Predict { handle, input: vec![4., f32::MAX] },
            &shared,
        ).unwrap_err();
        assert!(error.to_string().contains("nonfinite activation"), "{error:#}");
        assert_eq!(owner.execute(Request::ParameterValues { handle }, &shared)?, parameters);
        assert_eq!(
            owner.execute(Request::Predict { handle, input: vec![0., 3.] }, &shared)?,
            serde_json::json!({"output": [9.], "revision": 3})
        );

        let stats = owner.execute(Request::Stats, &shared)?;
        let initialized = owner.execute(
            Request::InitializeDefinition {
                source: source.into(),
                memory: 1024 * 1024,
                optimizer: None,
            },
            &shared,
        )?;
        assert_eq!(initialized["backend"], "native-cpu");
        let compiled_handle = initialized["handle"].as_u64().unwrap();
        for (input, output, revision) in [([2., 3.], 3., 1), ([4., 3.], 9., 2)] {
            assert_eq!(
                owner.execute(Request::Predict { handle: compiled_handle, input: input.into() }, &shared)?,
                serde_json::json!({"output": [output], "revision": revision})
            );
        }
        let parameters = owner.execute(Request::ParameterValues { handle: compiled_handle }, &shared)?;
        let error = owner.execute(
            Request::Predict { handle: compiled_handle, input: vec![4., f32::MAX] },
            &shared,
        ).unwrap_err();
        assert!(error.to_string().contains("nonfinite"), "{error:#}");
        assert_eq!(owner.execute(Request::ParameterValues { handle: compiled_handle }, &shared)?, parameters);
        assert_eq!(
            owner.execute(Request::Predict { handle: compiled_handle, input: vec![0., 3.] }, &shared)?,
            serde_json::json!({"output": [9.], "revision": 3})
        );
        owner.execute(Request::Drop { handle: compiled_handle }, &shared)?;
        let after = owner.execute(Request::Stats, &shared)?;
        assert_eq!(after["sessions"], stats["sessions"]);
        assert_eq!(after["reserved"], stats["reserved"]);
        Ok(())
    }

    #[test]
    fn adaptive_drop_returns_reservation_and_rejects_stale_or_foreign_handles() -> Result<()> {
        let shared = shared();
        let mut owner = state();
        let handle = owner.execute(program(MAX_SESSION_BYTES), &shared)?["handle"]
            .as_u64()
            .unwrap();
        assert!(state().execute(Request::Drop { handle }, &shared).is_err());
        assert_eq!(owner.execute(Request::Stats, &shared)?["reserved"], MAX_SESSION_BYTES);
        owner.execute(Request::Drop { handle }, &shared)?;
        assert_eq!(owner.execute(Request::Stats, &shared)?["reserved"], 0);
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
        let named = owner.execute(operation(), &shared)?["handle"].as_u64().unwrap();
        let mut adaptive = 0;
        for _ in 0..62 {
            adaptive = owner.execute(program(1024), &shared)?["handle"]
                .as_u64()
                .unwrap();
        }
        let reserved = owner.execute(Request::Stats, &shared)?["reserved"].as_u64().unwrap();
        assert!(owner.execute(create(), &shared).is_err());
        assert!(owner.execute(operation(), &shared).is_err());
        assert!(owner
            .execute(
                Request::Restore {
                    checkpoint,
                    memory: 1024 * 1024
                },
                &shared
            )
            .is_err());
        assert_eq!(owner.execute(Request::Stats, &shared)?["reserved"], reserved);
        assert_eq!(owner.execute(Request::Stats, &shared)?["sessions"], 64);
        owner.execute(Request::Drop { handle: adaptive }, &shared)?;
        owner.execute(
            Request::Restore {
                checkpoint,
                memory: 1024 * 1024,
            },
            &shared,
        )?;
        let reserved = owner.execute(Request::Stats, &shared)?["reserved"].as_u64().unwrap();
        assert!(owner
            .execute(Request::Drop { handle: u64::MAX }, &shared)
            .is_err());
        assert_eq!(owner.execute(Request::Stats, &shared)?["reserved"], reserved);
        owner.execute(Request::Drop { handle: named }, &shared)?;
        assert_eq!(owner.execute(Request::Stats, &shared)?["sessions"], 63);
        assert_eq!(owner.execute(Request::Stats, &shared)?["reserved"], reserved - 1024 * 1024);
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
