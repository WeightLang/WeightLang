//! Linux process limits for explicit script execution. Never mutate an embedding host.
use anyhow::{ensure, Context, Result};
use std::{
    io::Read,
    process::Command,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Default, serde::Deserialize)]
pub(crate) struct Limits {
    #[serde(default)]
    pub cpu_seconds: Option<u64>,
    #[serde(default)]
    pub address_space_bytes: Option<u64>,
}
impl Limits {
    pub fn enabled(self) -> bool {
        self.cpu_seconds.is_some() || self.address_space_bytes.is_some()
    }
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.cpu_seconds.is_none_or(|n| n > 0 && n <= 1_000_000_000),
            "cpu_seconds must be in 1..1000000000"
        );
        ensure!(
            self.address_space_bytes
                .is_none_or(|n| n > 0 && n <= i64::MAX as u64),
            "address_space_bytes must be positive and fit i64"
        );
        Ok(())
    }
    pub fn verify(self) -> Result<()> {
        self.validate()?;
        if !self.enabled() {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            for (resource, bound) in [
                (libc::RLIMIT_CPU, self.cpu_seconds),
                (libc::RLIMIT_AS, self.address_space_bytes),
            ] {
                if let Some(bound) = bound {
                    let mut limit = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    // SAFETY: valid writable rlimit pointer and supported resource.
                    ensure!(
                        unsafe { libc::getrlimit(resource, &mut limit) } == 0,
                        "getrlimit failed"
                    );
                    ensure!(limit.rlim_cur <= bound && limit.rlim_max <= bound, "hard resource limits require isolated CLI execution or an already limited worker");
                }
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("hard process limits currently require Linux")
    }
}

/// Invoke from CLI execution branches only. Returns child status to the supervisor.
/// Checking and compilation must not invoke this function.
pub fn supervise_current() -> Result<Option<u8>> {
    let Some(path) = std::env::var_os("WGT_HOST_POLICY") else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(1_048_577)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 1_048_576, "host policy exceeds 1 MiB");
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let limits: Limits = serde_json::from_value(value.clone())?;
    limits.validate()?;
    if !limits.enabled() {
        return Ok(None);
    }
    if std::env::var_os("WGT_RESOURCE_WORKER").as_deref() == Some(std::ffi::OsStr::new("1")) {
        limits.verify()?;
        return Ok(None);
    }
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!("hard process limits currently require Linux");
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args(std::env::args_os().skip(1))
            .env("WGT_RESOURCE_WORKER", "1");
        // Only async-signal-safe syscalls in the post-fork hook; no allocation or locks.
        unsafe {
            command.pre_exec(move || {
                for (resource, bound) in [
                    (libc::RLIMIT_CPU, limits.cpu_seconds),
                    (libc::RLIMIT_AS, limits.address_space_bytes),
                    (libc::RLIMIT_CORE, Some(0)),
                ] {
                    if let Some(bound) = bound {
                        let limit = libc::rlimit {
                            rlim_cur: bound,
                            rlim_max: bound,
                        };
                        if libc::setrlimit(resource, &limit) != 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                    }
                }
                Ok(())
            });
        }
        let deadline = value["deadline_ms"]
            .as_u64()
            .context("deadline_ms required")?;
        let mut child = command.spawn().context("spawn isolated WGT worker")?;
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    eprintln!("isolated WGT worker failed: {status}");
                }
                return Ok(Some(if status.success() { 0 } else { 1 }));
            }
            if start.elapsed().as_millis() >= deadline as u128 {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("isolated WGT worker deadline exceeded");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// Whole-process user + system CPU time, including native computation and threads.
pub(crate) fn cpu_micros() -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the object on success.
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "getrusage failed"
        );
        let usage = unsafe { usage.assume_init() };
        Ok(
            (usage.ru_utime.tv_sec as u64 + usage.ru_stime.tv_sec as u64) * 1_000_000
                + usage.ru_utime.tv_usec as u64
                + usage.ru_stime.tv_usec as u64,
        )
    }
    #[cfg(not(target_os = "linux"))]
    anyhow::bail!("CPU accounting currently requires Linux")
}
