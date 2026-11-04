//! Host grants are loaded at execution time, never from model artifacts.
use super::*;
use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

use crate::script::permissions::Policy;
pub(super) struct Host {
    pub policy: Policy,
    pub pools: BTreeMap<String, crate::accelerator::DevicePool>,
    files: BTreeMap<String, File>,
    pub disk_read: u64,
    pub disk_written: u64,
    outputs: BTreeMap<String, (File, std::ffi::CString)>,
    pub network_read: u64,
    pub requests: u64,
    pub deadline: Instant,
}
impl Host {
    pub fn new(policy: Policy) -> Result<Self> {
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(policy.deadline_ms))
            .context("host deadline overflow")?;
        crate::isolation::Limits {
            cpu_seconds: policy.cpu_seconds,
            address_space_bytes: policy.address_space_bytes,
        }
        .verify()?;
        ensure!(policy.calls <= 10000, "host calls exceeds 10000");
        ensure!(
            policy.workspace_bytes <= 1_000_000_000
                && policy.work <= 1_000_000_000
                && policy.deadline_ms <= 1_000_000_000,
            "host budget exceeds supported range"
        );
        let mut pools = BTreeMap::new();
        let mut device_ids = std::collections::BTreeSet::new();
        for (alias, value) in &policy.devices {
            let grant: crate::accelerator::DeviceGrant =
                serde_json::from_value(value.clone()).context("invalid device grant")?;
            ensure!(
                device_ids.insert(grant.device.clone()),
                "duplicate physical device grant"
            );
            pools.insert(
                alias.clone(),
                crate::accelerator::DevicePool::new(
                    grant.clone(),
                    crate::accelerator::backend(&grant.device)?,
                )?,
            );
        }
        let mut files = BTreeMap::new();
        for (alias, path) in &policy.files {
            ensure!(path.is_absolute(), "host file paths must be absolute");
            // Pin the authorized file, rather than resolving a model-controlled path on each call.
            ensure!(
                std::fs::metadata(path)?.is_file(),
                "host grant requires regular file"
            );
            let file = File::open(path)?;
            ensure!(
                file.metadata()?.is_file(),
                "host grant requires regular file"
            );
            files.insert(alias.clone(), file);
        }
        for endpoint in policy.endpoints.values() {
            let url = reqwest::Url::parse(endpoint)?;
            ensure!(
                ["http", "https"].contains(&url.scheme())
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.fragment().is_none(),
                "invalid host endpoint"
            );
        }
        let mut outputs = BTreeMap::new();
        for (alias, path) in &policy.outputs {
            ensure!(path.is_absolute(), "output paths must be absolute");
            let name = path.file_name().context("output requires filename")?;
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::ffi::OsStrExt;
                let directory = File::open(path.parent().context("output requires parent")?)?;
                ensure!(
                    directory.metadata()?.is_dir(),
                    "output parent must be directory"
                );
                outputs.insert(
                    alias.clone(),
                    (directory, std::ffi::CString::new(name.as_bytes())?),
                );
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = name;
                bail!("disk write capabilities currently require Linux");
            }
        }
        Ok(Self {
            pools,
            outputs,
            disk_written: 0,

            policy,
            files,
            disk_read: 0,
            network_read: 0,
            requests: 0,
            deadline,
        })
    }
    pub fn remaining_ms(&self) -> u64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64
    }
    pub fn execute(
        &mut self,
        capability: &str,
        input: &[Value],
        workspace: usize,
        timeout: u64,
    ) -> Result<serde_json::Value> {
        ensure!(input.len() == 2, "invalid_input");
        let alias = input[0]
            .text()
            .map_err(|_| anyhow::anyhow!("invalid_input"))?;
        if capability == "file_write.v1" {
            let text = input[1]
                .text()
                .map_err(|_| anyhow::anyhow!("invalid_input"))?;
            ensure!(self.remaining_ms().min(timeout) > 0, "deadline_exceeded");
            ensure!(text.len() <= workspace, "workspace_exceeded");
            ensure!(
                text.len() as u64
                    <= self
                        .policy
                        .disk_write_bytes
                        .saturating_sub(self.disk_written),
                "disk_write_exhausted"
            );
            ensure!(
                text.len() as u64
                    <= self
                        .policy
                        .disk_storage_bytes
                        .saturating_sub(self.disk_written),
                "disk_storage_exhausted"
            );
            let (directory, name) = self.outputs.get(alias).context("permission_denied")?;
            #[cfg(target_os = "linux")]
            {
                use std::{
                    io::Write,
                    os::fd::{AsRawFd, FromRawFd},
                };
                // SAFETY: valid pinned directory fd and NUL-terminated basename. O_EXCL
                // rejects existing files/symlinks; no script-controlled path traversal.
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_WRONLY
                            | libc::O_CREAT
                            | libc::O_EXCL
                            | libc::O_CLOEXEC
                            | libc::O_NOFOLLOW,
                        0o600,
                    )
                };
                if fd < 0 {
                    let error = std::io::Error::last_os_error();
                    bail!(if error.kind() == std::io::ErrorKind::AlreadyExists {
                        "already_exists"
                    } else {
                        "io_error"
                    });
                }
                let mut file = unsafe { File::from_raw_fd(fd) };
                let mut rest = text.as_bytes();
                while !rest.is_empty() {
                    let n = file
                        .write(rest)
                        .map_err(|_| anyhow::anyhow!("write_partial"))?;
                    ensure!(n > 0, "write_partial");
                    self.disk_written += n as u64;
                    rest = &rest[n..];
                }
                file.sync_all()
                    .map_err(|_| anyhow::anyhow!("write_partial"))?;
                return Ok(serde_json::json!({"bytes_written":text.len(),"created":true}));
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (directory, name);
                bail!("unsupported_platform");
            }
        }
        let limit = input[1]
            .integer()
            .map_err(|_| anyhow::anyhow!("invalid_input"))?;
        ensure!(limit > 0 && limit <= workspace, "workspace_exceeded");
        let request_deadline = self.deadline.min(Instant::now() + Duration::from_millis(timeout));
        ensure!(request_deadline.saturating_duration_since(Instant::now()).as_millis() > 0, "deadline_exceeded");
        match capability {
            "file_read.v1" => {
                let file = self.files.get_mut(alias).context("permission_denied")?;
                ensure!(
                    limit as u64 <= self.policy.disk_read_bytes.saturating_sub(self.disk_read),
                    "disk_read_exhausted"
                );
                file.seek(SeekFrom::Start(0))
                    .map_err(|_| anyhow::anyhow!("io_error"))?;
                let bytes = read_bounded(file, limit, &mut self.disk_read)?;
                let text = String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("invalid_utf8"))?;
                Ok(serde_json::json!(text))
            }
            "http_get.v1" => {
                let url = self
                    .policy
                    .endpoints
                    .get(alias)
                    .context("permission_denied")?;
                ensure!(
                    self.requests < self.policy.network_requests,
                    "network_requests_exhausted"
                );
                ensure!(
                    limit as u64
                        <= self
                            .policy
                            .network_body_bytes
                            .saturating_sub(self.network_read),
                    "network_body_exhausted"
                );
                let client = reqwest::blocking::Client::builder()
                    .no_proxy()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| anyhow::anyhow!("network_error"))?;
                // Client setup is part of the allowance. Never restart the timeout
                // from the duration measured before initialization.
                let timeout = request_deadline.saturating_duration_since(Instant::now()).as_millis() as u64;
                ensure!(timeout > 0, "deadline_exceeded");
                self.requests += 1;
                let mut response = client
                    .get(url)
                    .timeout(Duration::from_millis(timeout))
                    .send()
                    .map_err(|_| anyhow::anyhow!("network_error"))?;
                let status = response.status().as_u16();
                let bytes = read_bounded(&mut response, limit, &mut self.network_read)?;
                let text = String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("invalid_utf8"))?;
                Ok(serde_json::json!({"status":status,"body":text}))
            }
            _ => bail!("unknown_capability"),
        }
    }
}
// Return a bounded prefix; do not read an extra byte just to probe EOF.
// Account successful chunks even if a later read fails or UTF-8 decoding fails.
fn read_bounded(reader: &mut impl Read, limit: usize, used: &mut u64) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut chunk = [0u8; 8192];
    while result.len() < limit {
        let count = (limit - result.len()).min(chunk.len());
        let n = reader
            .read(&mut chunk[..count])
            .map_err(|_| anyhow::anyhow!("io_error"))?;
        if n == 0 {
            break;
        }
        *used += n as u64;
        result.extend_from_slice(&chunk[..n]);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_reads_are_charged_on_failure() {
        struct Broken(bool);
        impl Read for Broken {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("broken"));
                }
                self.0 = true;
                out[..2].copy_from_slice(b"ab");
                Ok(2)
            }
        }
        let mut used = 0;
        assert!(read_bounded(&mut Broken(false), 4, &mut used).is_err());
        assert_eq!(used, 2);
    }
}
