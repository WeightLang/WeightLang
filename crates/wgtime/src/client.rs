use crate::{receive, send, Reply, Request, MAX_FRAME};
#[cfg(target_os = "linux")]
use crate::{buffer, peer_uid, PROTOCOL};
use anyhow::{bail, ensure, Context, Result};
use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const DEFAULT_TIMEOUT: Option<Duration> = Some(Duration::from_secs(30));

pub struct Client {
    stream: UnixStream,
    pub transmitted_bytes: u64,
    endpoint: PathBuf,
    expected_uid: u32,
    poisoned: bool,
    read_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
}

// Arm before any I/O. An incomplete exchange, including one abandoned by
// unwinding, closes the socket rather than allowing a late reply to be reused.
struct PendingExchange<'a>(&'a mut Client);
impl Drop for PendingExchange<'_> {
    fn drop(&mut self) {
        if self.0.poisoned {
            let _ = self.0.stream.shutdown(Shutdown::Both);
        }
    }
}

impl Client {
    #[cfg(target_os = "linux")]
    pub fn connect(endpoint: &Path, expected_uid: u32) -> Result<Self> {
        Self::connect_with_timeouts(endpoint, expected_uid, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT)
    }

    /// Configure socket read/write timeouts explicitly. `None` disables that
    /// timeout; zero durations are rejected. These are socket timeouts, not a
    /// total request deadline, and they do not cancel work in the service.
    #[cfg(target_os = "linux")]
    pub fn connect_with_timeouts(
        endpoint: &Path,
        expected_uid: u32,
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
    ) -> Result<Self> {
        validate_timeouts(read_timeout, write_timeout)?;
        let stream = UnixStream::connect(endpoint)
            .context("wgtime unavailable; embedded execution is forbidden")?;
        ensure!(peer_uid(&stream)? == expected_uid, "wgtime service identity mismatch");
        stream.set_read_timeout(read_timeout)?;
        stream.set_write_timeout(write_timeout)?;
        let mut client = Self {
            stream,
            transmitted_bytes: 0,
            endpoint: endpoint.to_owned(),
            expected_uid,
            poisoned: false,
            read_timeout,
            write_timeout,
        };
        client.call(Request::Hello { protocol: PROTOCOL })?;
        Ok(client)
    }

    /// Open a new connection, preserving timeout settings. The new connection
    /// cannot reuse handles owned by the old one. Never replay a failed mutation
    /// automatically: a lost response does not prove the mutation was rolled back.
    #[cfg(target_os = "linux")]
    pub fn reconnect(&self) -> Result<Self> {
        Self::connect_with_timeouts(
            &self.endpoint, self.expected_uid, self.read_timeout, self.write_timeout,
        )
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Change timeouts before starting a long-running request.
    pub fn set_timeouts(
        &mut self,
        read_timeout: Option<Duration>,
        write_timeout: Option<Duration>,
    ) -> Result<()> {
        self.ensure_usable()?;
        validate_timeouts(read_timeout, write_timeout)?;
        self.exchange(|stream, _| {
            stream.set_read_timeout(read_timeout)?;
            stream.set_write_timeout(write_timeout)?;
            Ok(())
        })?;
        self.read_timeout = read_timeout;
        self.write_timeout = write_timeout;
        Ok(())
    }

    fn ensure_usable(&self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "wgtime connection is poisoned; reconnect and recreate connection-owned handles"
        );
        Ok(())
    }

    fn exchange<T>(
        &mut self,
        action: impl FnOnce(&mut UnixStream, &mut u64) -> Result<T>,
    ) -> Result<T> {
        self.ensure_usable()?;
        self.poisoned = true;
        let pending = PendingExchange(self);
        let client = &mut *pending.0;
        let result = action(&mut client.stream, &mut client.transmitted_bytes);
        client.poisoned = result.is_err();
        result
    }

    #[cfg(target_os = "linux")]
    pub fn register_tensor(&mut self, tensor: &buffer::SealedTensor, shape: Vec<usize>) -> Result<u64> {
        let reply = self.exchange(|stream, transmitted| {
            let request = Request::RegisterTensor { count: tensor.count, shape, dtype: "f32".into() };
            *transmitted = transmitted.saturating_add(send(stream, &request)? as u64);
            tensor.send(stream)?;
            receive_reply(stream)
        })?;
        let value = reply_result(reply)?;
        if let Some(handle) = value["tensor"].as_u64() {
            return Ok(handle);
        }
        self.poisoned = true;
        let _ = self.stream.shutdown(Shutdown::Both);
        bail!("wgtime sent an invalid tensor handle")
    }

    #[cfg(target_os = "linux")]
    pub fn call_with_file(&mut self, request: Request, file: &std::fs::File) -> Result<serde_json::Value> {
        use std::os::fd::AsRawFd;
        self.ensure_usable()?;
        ensure!(request.uses_persistence_descriptor(), "request does not accept a persistence descriptor");
        let reply = self.exchange(|stream, transmitted| {
            *transmitted = transmitted.saturating_add(send(stream, &request)? as u64);
            buffer::send_fd(stream, file.as_raw_fd())?;
            receive_reply(stream)
        })?;
        reply_result(reply)
    }

    /// On failure output is cleared, so a partial read cannot expose stale or
    /// partially overwritten predictions. Successful calls reuse its allocation.
    pub fn predict_into(
        &mut self,
        handle: u64,
        tensor: u64,
        start: usize,
        count: usize,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let result = self.exchange(|stream, transmitted| {
            let mut frame = [0u8; 36];
            frame[..4].copy_from_slice(&u32::MAX.to_le_bytes());
            let fields = [handle, tensor, u64::try_from(start)?, u64::try_from(count)?];
            for (index, value) in fields.iter().enumerate() {
                frame[4 + index * 8..12 + index * 8].copy_from_slice(&value.to_le_bytes());
            }
            stream.write_all(&frame)?;
            *transmitted = transmitted.saturating_add(frame.len() as u64);
            let mut header = [0u8; 8];
            stream.read_exact(&mut header)?;
            let status = u32::from_le_bytes(header[..4].try_into().unwrap());
            let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
            ensure!(status <= 1, "invalid binary response status");
            ensure!(size <= MAX_FRAME, "invalid binary response length");
            if status == 1 {
                let mut bytes = vec![0; size];
                stream.read_exact(&mut bytes)?;
                return Ok(Some(String::from_utf8(bytes).context("invalid binary error encoding")?));
            }
            ensure!(size % 4 == 0, "invalid f32 payload");
            output.resize(size / 4, 0.);
            // SAFETY: resize initialized size/4 f32 elements. All f32 bit patterns
            // are valid; the mutable byte borrow is confined to this read. On any
            // failure the caller-visible vector is cleared below.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), size)
            };
            stream.read_exact(bytes)?;
            ensure!(output.iter().all(|n| n.is_finite()), "nonfinite response");
            Ok(None)
        });
        match result {
            Ok(None) => Ok(()),
            Ok(Some(error)) => {
                output.clear();
                bail!("wgtime: {error}")
            }
            Err(error) => {
                output.clear();
                Err(error)
            }
        }
    }

    pub fn call(&mut self, request: Request) -> Result<serde_json::Value> {
        self.ensure_usable()?;
        ensure!(
            !request.uses_persistence_descriptor() && !matches!(
                &request, Request::RegisterTensor { .. } | Request::PredictBinary { .. }
            ),
            "request requires its descriptor or binary client method"
        );
        let reply = self.exchange(|stream, transmitted| {
            *transmitted = transmitted.saturating_add(send(stream, &request)? as u64);
            receive_reply(stream)
        })?;
        reply_result(reply)
    }
}

fn validate_timeouts(read: Option<Duration>, write: Option<Duration>) -> Result<()> {
    ensure!(read.is_none_or(|value| !value.is_zero()) && write.is_none_or(|value| !value.is_zero()),
        "socket timeouts must be nonzero or None");
    Ok(())
}

fn receive_reply(stream: &mut UnixStream) -> Result<Reply> {
    let reply: Reply = receive(stream)?;
    ensure!(reply.value.is_some() != reply.error.is_some(), "wgtime sent an invalid reply");
    Ok(reply)
}

// Application errors are decoded only after a complete, valid frame has disarmed
// PendingExchange. A rejected model is not itself a transport failure.
fn reply_result(reply: Reply) -> Result<serde_json::Value> {
    match (reply.value, reply.error) {
        (Some(value), None) => Ok(value),
        (None, Some(error)) => bail!("wgtime: {error}"),
        _ => bail!("wgtime sent an invalid reply"),
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
