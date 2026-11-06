use anyhow::{Context, Result, ensure};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command},
    time::{Duration, Instant},
};

pub fn scratch() -> Result<tempfile::TempDir> {
    let root = std::env::var_os("CLAUDE_JOB_DIR")
        .map(|path| PathBuf::from(path).join("tmp"))
        .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
    fs::create_dir_all(&root)?;
    Ok(tempfile::Builder::new()
        .prefix("wgt-model-")
        .tempdir_in(root)?)
}

pub struct Service {
    child: Child,
    directory: tempfile::TempDir,
}

impl Service {
    pub fn start() -> Result<Self> {
        let directory = scratch()?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))?;
        let child = Command::new(
            std::env::var_os("WGTIME_TEST_DAEMON").context("prebuilt daemon required")?,
        )
        .arg("--test-directory")
        .arg(directory.path())
        .spawn()?;
        let mut service = Self { child, directory };
        let deadline = Instant::now() + Duration::from_secs(10);
        let timeout = Some(Duration::from_millis(100));
        loop {
            if let Some(status) = service.child.try_wait()? {
                anyhow::bail!("wgtime exited before the protocol handshake: {status}");
            }
            match wgtime::Client::connect_with_timeouts(
                &service.endpoint(),
                unsafe { libc::geteuid() },
                timeout,
                timeout,
            ) {
                Ok(_) => return Ok(service),
                Err(error) => ensure!(
                    Instant::now() < deadline,
                    "wgtime handshake timed out: {error:#}"
                ),
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    pub fn path(&self) -> &Path {
        self.directory.path()
    }

    pub fn endpoint(&self) -> PathBuf {
        self.path().join("service.sock")
    }

    pub fn client(&self) -> Result<wgt::Client> {
        wgt::Client::connect_to(&self.endpoint())
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
