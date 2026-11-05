//! Local IPC only. This crate must never depend on an execution backend.
use anyhow::{ensure, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_ENDPOINT: &str = "/run/wgtime/service.sock";
pub const MAX_FRAME: usize = 16 * 1024 * 1024;
pub const PROTOCOL: u32 = 1;
#[cfg(target_os = "linux")]
pub mod buffer;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Request {
    Hello {
        protocol: u32,
    },
    Create {
        widths: Vec<usize>,
        activation: String,
        seed: u64,
        optimizer: String,
        memory: usize,
    },
    InitializeDefinition { source: String, memory: usize, optimizer: Option<String> },
    CallOperation { handle: u64, name: String, inputs: Vec<Vec<f32>> },
    OperationSchemas { handle: u64 },
    ParameterValues { handle: u64 },
    StateValues { handle: u64 },
    UpdateParameters { handle: u64, candidates: Vec<(String, Vec<usize>, Vec<f32>)> },
    TrainStepWithPolicy { handle: u64, inputs: Vec<Vec<f32>>, targets: serde_json::Value, rate: f32, loss: String, clip: Option<f32>, straight_through: bool },
    Predict {
        handle: u64,
        input: Vec<f32>,
    },
    CreateTrainingProgram { program: serde_json::Value, memory: usize, optimizer: String },
    TrainStep {
        handle: u64,
        inputs: Vec<Vec<f32>>,
        targets: serde_json::Value,
        rate: f32,
        loss: String,
        clip: Option<f32>,
    },
    Drop {
        handle: u64,
    },
    RegisterTensor {
        count: usize,
        shape: Vec<usize>,
        dtype: String,
    },
    PredictTensor {
        handle: u64,
        tensor: u64,
        start: usize,
        count: usize,
    },
    PredictBinary {
        handle: u64,
        tensor: u64,
        start: usize,
        count: usize,
    },
    ReleaseTensor {
        tensor: u64,
    },
    PersistFd {
        handle: u64,
        checkpoint: bool,
    },
    ExportWeightsFd { handle: u64 },
    BindWeightsFd { handle: u64, optimizer: String },
    InspectArtifactFd {
        checkpoint: bool,
        memory: usize,
    },
    LoadFd {
        checkpoint: bool,
        optimizer: String,
        memory: usize,
    },
    CreateGraph {
        graph: serde_json::Value,
        optimizer: String,
        memory: usize,
    },
    RegisterStrategy {
        source: String,
        entry: String,
    },
    ExecuteStrategy {
        handle: u64,
        strategy: u64,
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        operations: u64,
        memory: usize,
    },
    Checkpoint {
        handle: u64,
    },
    Restore {
        checkpoint: u64,
        memory: usize,
    },
    Stats,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub value: Option<serde_json::Value>,
    pub error: Option<String>,
}
impl Reply {
    pub fn result(value: Result<serde_json::Value>) -> Self {
        match value {
            Ok(value) => Self {
                value: Some(value),
                error: None,
            },
            Err(error) => Self {
                value: None,
                error: Some(format!("{error:#}")),
            },
        }
    }
}
pub fn send<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<usize> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "IPC frame exceeds limit");
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(bytes.len() + 4)
}
pub fn receive<T: DeserializeOwned>(stream: &mut UnixStream) -> Result<T> {
    let mut size = [0; 4];
    stream.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as usize;
    ensure!(size <= MAX_FRAME, "IPC frame exceeds limit");
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
/// Fixed-width hot inference frame; all descriptors are still validated by the service.
pub fn receive_request(stream:&mut UnixStream)->Result<Request>{
    let mut header=[0u8;4];stream.read_exact(&mut header)?;
    let size=u32::from_le_bytes(header);
    if size==u32::MAX {
        let mut bytes=[0u8;32];stream.read_exact(&mut bytes)?;
        let field=|i:usize|u64::from_le_bytes(bytes[i..i+8].try_into().unwrap());
        return Ok(Request::PredictBinary{handle:field(0),tensor:field(8),start:usize::try_from(field(16))?,count:usize::try_from(field(24))?});
    }
    ensure!(size as usize<=MAX_FRAME,"IPC frame exceeds limit");
    let mut bytes=vec![0;size as usize];stream.read_exact(&mut bytes)?;Ok(serde_json::from_slice(&bytes)?)
}
#[cfg(target_os = "linux")]
pub fn peer_uid(stream: &UnixStream) -> Result<u32> {
    use std::os::fd::AsRawFd;
    let mut credentials: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let status = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut size,
        )
    };
    ensure!(
        status == 0,
        "cannot authenticate IPC peer: {}",
        std::io::Error::last_os_error()
    );
    ensure!(
        size as usize == std::mem::size_of::<libc::ucred>(),
        "invalid peer credentials"
    );
    Ok(credentials.uid)
}

pub struct Client {
    stream: UnixStream,
    pub transmitted_bytes: u64,
    endpoint: std::path::PathBuf,
    expected_uid: u32,
}
impl Client {
    #[cfg(target_os = "linux")]
    pub fn connect(endpoint: &Path, expected_uid: u32) -> Result<Self> {
        let stream = UnixStream::connect(endpoint)
            .context("wgtime unavailable; embedded execution is forbidden")?;
        ensure!(
            peer_uid(&stream)? == expected_uid,
            "wgtime service identity mismatch"
        );
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        let mut client = Self {
            stream,
            transmitted_bytes: 0,
            endpoint: endpoint.to_owned(),
            expected_uid,
        };
        client.call(Request::Hello { protocol: PROTOCOL })?;
        Ok(client)
    }
    #[cfg(target_os = "linux")]
    pub fn reconnect(&self) -> Result<Self> {
        Self::connect(&self.endpoint, self.expected_uid)
    }
    #[cfg(target_os = "linux")]
    pub fn register_tensor(
        &mut self,
        tensor: &buffer::SealedTensor,
        shape: Vec<usize>,
    ) -> Result<u64> {
        let request = Request::RegisterTensor {
            count: tensor.count,
            shape,
            dtype: "f32".into(),
        };
        self.transmitted_bytes += send(&mut self.stream, &request)? as u64;
        tensor.send(&self.stream)?;
        let reply: Reply = receive(&mut self.stream)?;
        ensure!(
            reply.error.is_none(),
            "wgtime: {}",
            reply.error.unwrap_or_default()
        );
        reply.value.context("missing buffer response")?["tensor"]
            .as_u64()
            .context("missing tensor handle")
    }
    pub fn call_with_file(
        &mut self,
        request: Request,
        file: &std::fs::File,
    ) -> Result<serde_json::Value> {
        use std::os::fd::AsRawFd;
        self.transmitted_bytes += send(&mut self.stream, &request)? as u64;
        buffer::send_fd(&self.stream, file.as_raw_fd())?;
        let reply: Reply = receive(&mut self.stream)?;
        ensure!(
            reply.error.is_none(),
            "wgtime: {}",
            reply.error.unwrap_or_default()
        );
        reply.value.context("missing fd response")
    }
    pub fn predict_into(
        &mut self,
        handle: u64,
        tensor: u64,
        start: usize,
        count: usize,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let mut frame=[0u8;36];frame[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        for (index,value) in [handle,tensor,start as u64,count as u64].iter().enumerate(){frame[4+index*8..12+index*8].copy_from_slice(&value.to_le_bytes());}
        self.stream.write_all(&frame)?;self.transmitted_bytes+=frame.len() as u64;
        let mut header = [0u8; 8];
        self.stream.read_exact(&mut header)?;
        let status = u32::from_le_bytes(header[..4].try_into().unwrap());
        let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        ensure!(size <= MAX_FRAME, "invalid binary response length");
        if status != 0 {
            let mut bytes = vec![0; size];
            self.stream.read_exact(&mut bytes)?;
            anyhow::bail!("wgtime: {}", String::from_utf8_lossy(&bytes));
        }
        ensure!(size % 4 == 0, "invalid f32 payload");
        output.resize(size / 4, 0.);
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), size) };
        self.stream.read_exact(bytes)?;
        ensure!(output.iter().all(|n| n.is_finite()), "nonfinite response");
        Ok(())
    }
    pub fn call(&mut self, request: Request) -> Result<serde_json::Value> {
        self.transmitted_bytes += send(&mut self.stream, &request)? as u64;
        let reply: Reply = receive(&mut self.stream)?;
        ensure!(
            reply.error.is_none(),
            "wgtime: {}",
            reply.error.unwrap_or_default()
        );
        reply.value.context("wgtime sent an invalid reply")
    }
}
