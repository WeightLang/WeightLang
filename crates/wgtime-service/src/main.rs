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
use weightlang_runtime::training::{mlp, Objective, Optimizer, Trainer};
use wgtime::{receive, send, Reply, Request, PROTOCOL};

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
    adaptive: HashMap<u64, weightlang_runtime::adaptive::AdaptiveSession>,
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
                ensure!(self.sessions.len() < 64, "session limit exceeded");
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
            Request::Predict { handle, input } => {
                if let Some(session) = self.adaptive.get_mut(&handle) {
                    return Ok(serde_json::json!({"output": session.evaluate(&input, 1_000_000)?, "revision": session.revision()}));
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
            Request::Train {
                handle,
                inputs,
                targets,
                rate,
            } => {
                ensure!(
                    !inputs.is_empty() && inputs.len() <= 65536,
                    "invalid batch size"
                );
                let trainer = &mut self
                    .sessions
                    .get_mut(&handle)
                    .context("unknown session or foreign owner")?
                    .0;
                let loss = trainer.step(&inputs, Objective::Mse(&targets), rate)?;
                Ok(serde_json::json!({"loss": loss, "steps": trainer.steps()}))
            }
            Request::Drop { handle } => {
                let (_, bytes) = self
                    .sessions
                    .remove(&handle)
                    .context("unknown session or foreign owner")?;
                self.reserved -= bytes;
                Ok(serde_json::json!({"released": true}))
            }
            Request::CreateProgram { program, memory } => {
                ensure!(self.adaptive.len() + self.sessions.len() < 64 && memory > 0 && memory <= MAX_SESSION_BYTES && self.reserved.checked_add(memory).is_some_and(|n| n <= MAX_CONNECTION_BYTES), "program admission budget exceeded");
                let program = serde_json::from_value(program)?;
                let session = weightlang_runtime::adaptive::AdaptiveSession::prepare(program, memory)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.adaptive.insert(handle, session);
                self.reserved += memory;
                Ok(serde_json::json!({"handle": handle}))
            }
            Request::RegisterStrategy { source, entry } => {
                ensure!(self.strategies.len() < 64, "strategy limit exceeded");
                let strategy = wgtime_service::strategy::Strategy::compile_entry(&source, &entry)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.strategies.insert(handle, strategy);
                Ok(serde_json::json!({"strategy": handle}))
            }
            Request::ExecuteStrategy { handle, strategy, inputs, targets, operations, memory } => {
                ensure!(operations > 0 && operations <= 100_000_000 && memory > 0 && memory <= MAX_SESSION_BYTES, "strategy execution budget exceeded");
                let strategy = self.strategies.get(&strategy).context("unknown strategy or foreign owner")?;
                let session = self.adaptive.get_mut(&handle).context("unknown adaptive session or foreign owner")?;
                let mut remaining = operations;
                strategy.execute(session, &inputs, &targets, &mut remaining, memory)
            }
            Request::Checkpoint { handle } => {
                ensure!(self.checkpoints.len() < 4, "checkpoint slot limit exceeded");
                let trainer = &mut self.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0;
                let bytes = trainer.checkpoint_bytes_bounded(16 * 1024 * 1024)?;
                let checkpoint = shared.next.fetch_add(1, Ordering::Relaxed);
                self.checkpoints.insert(checkpoint, bytes);
                Ok(serde_json::json!({"checkpoint": checkpoint}))
            }
            Request::Restore { checkpoint, memory } => {
                ensure!(self.sessions.len() < 64 && memory > 0 && memory <= MAX_SESSION_BYTES && self.reserved.checked_add(memory).is_some_and(|n| n <= MAX_CONNECTION_BYTES), "restore budget exceeded");
                let bytes = self.checkpoints.get(&checkpoint).context("unknown checkpoint or foreign owner")?;
                let trainer = Trainer::restore_value(serde_json::from_slice(bytes)?, memory)?;
                let handle = shared.next.fetch_add(1, Ordering::Relaxed);
                self.sessions.insert(handle, (trainer, memory));
                self.reserved += memory;
                Ok(serde_json::json!({"handle": handle}))
            }
            Request::RegisterTensor { .. } => bail!("descriptor registration requires data channel"),
            Request::PredictBinary { .. } => bail!("binary request requires binary response"),
            Request::PredictTensor { handle, tensor, start, count } => {
                let mapping = self.tensors.get(&tensor).context("unknown tensor or foreign owner")?;
                let end = start.checked_add(count).context("tensor range overflow")?;
                let input = mapping.values().get(start..end).context("tensor range out of bounds")?;
                let trainer = &mut self.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0;
                Ok(serde_json::json!({"output": trainer.predict(input)?, "steps": trainer.steps()}))
            }
            Request::ReleaseTensor { tensor } => {
                self.tensors.remove(&tensor).context("unknown tensor or foreign owner")?;
                Ok(serde_json::json!({"released": true}))
            }
            Request::Stats => Ok(
                serde_json::json!({"sessions": self.sessions.len(), "reserved": self.reserved, "connections": shared.connections.load(Ordering::Relaxed)}),
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
        loop {
            let request = match receive(&mut stream) {
                Ok(request) => request,
                Err(_) => break,
            };
            if let Request::PredictBinary { handle, tensor, start, count } = request {
                use std::io::Write;
                let result = (|| -> Result<Vec<f32>> {
                    ensure!(state.ready, "handshake required");
                    let mapping = state.tensors.get(&tensor).context("unknown tensor or foreign owner")?;
                    let end = start.checked_add(count).context("tensor range overflow")?;
                    let input = mapping.values().get(start..end).context("tensor range out of bounds")?;
                    let trainer = &mut state.sessions.get_mut(&handle).context("unknown session or foreign owner")?.0;
                    Ok(trainer.session.run(input).map_err(anyhow::Error::msg)?.to_vec())
                })();
                match result {
                    Ok(output) => {
                        ensure!(output.iter().all(|n| n.is_finite()), "nonfinite prediction");
                        let size = output.len().checked_mul(4).context("output size overflow")?;
                        ensure!(size <= wgtime::MAX_FRAME, "output frame exceeds limit");
                        let mut bytes = Vec::with_capacity(8 + size);
                        bytes.extend_from_slice(&0u32.to_le_bytes());
                        bytes.extend_from_slice(&(size as u32).to_le_bytes());
                        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), size) });
                        stream.write_all(&bytes)?;
                    }
                    Err(error) => {
                        let error = format!("{error:#}");
                        stream.write_all(&1u32.to_le_bytes())?;
                        stream.write_all(&(error.len() as u32).to_le_bytes())?;
                        stream.write_all(error.as_bytes())?;
                    }
                }
                continue;
            }
            let result = if let Request::RegisterTensor { count, shape, dtype } = request {
                // Consume the descriptor even when metadata is rejected, preserving framing.
                let file = wgtime::buffer::receive_file(&stream)?;
                (|| -> Result<serde_json::Value> {
                    ensure!(state.ready && state.tensors.len() < 64, "tensor admission limit exceeded");
                    ensure!(dtype == "f32" && !shape.is_empty() && shape.len() <= 8 && shape.iter().all(|n| *n > 0), "invalid tensor dtype or shape");
                    let elements = shape.iter().try_fold(1usize, |n, width| n.checked_mul(*width)).context("shape overflow")?;
                    ensure!(elements == count, "shape extent mismatch");
                    let mapping = wgtime::buffer::Mapping::open(file, count)?;
                    let tensor = shared.next.fetch_add(1, Ordering::Relaxed);
                    state.tensors.insert(tensor, mapping);
                    Ok(serde_json::json!({"tensor": tensor}))
                })()
            } else { state.execute(request, &shared) };
            send(&mut stream, &Reply::result(result))?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("wgtime connection: {error:#}");
    }
    shared.connections.fetch_sub(1, Ordering::Relaxed);
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
