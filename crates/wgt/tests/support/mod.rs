use anyhow::{Context, Result, ensure};
use std::{
    fs,
    io::Read,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

pub fn scratch() -> Result<tempfile::TempDir> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/cli-tests");
    fs::create_dir_all(&root)?;
    Ok(tempfile::tempdir_in(root)?)
}

pub struct Service {
    child: Child,
    directory: tempfile::TempDir,
}

impl Service {
    pub fn start() -> Result<Self> {
        let root = std::env::temp_dir().join("weightlang");
        fs::create_dir_all(&root)?;
        let directory = tempfile::Builder::new()
            .prefix("cli-service-")
            .tempdir_in(root)?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))?;
        let daemon = match std::env::var_os("WGTIME_TEST_DAEMON") {
            Some(path) => PathBuf::from(path),
            None => std::env::current_exe()?
                .parent()
                .and_then(Path::parent)
                .context("Cargo target directory")?
                .join(format!("wgtime{}", std::env::consts::EXE_SUFFIX)),
        };
        let child = Command::new(&daemon)
            .arg("--test-directory")
            .arg(directory.path())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "start {}; build wgtime-service or set WGTIME_TEST_DAEMON",
                    daemon.display()
                )
            })?;
        let mut service = Self { child, directory };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = service.child.try_wait()? {
                let mut error = String::new();
                if let Some(mut stderr) = service.child.stderr.take() {
                    stderr.read_to_string(&mut error)?;
                }
                anyhow::bail!("wgtime exited before handshake ({status}): {error}");
            }
            let timeout = Some(Duration::from_millis(100));
            match wgtime::Client::connect_with_timeouts(
                &service.endpoint(),
                fs::metadata(service.directory.path())?.uid(),
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

    pub fn endpoint(&self) -> PathBuf {
        self.directory.path().join("service.sock")
    }

    pub fn client(&self) -> Result<wgtime::Client> {
        wgtime::Client::connect(&self.endpoint(), fs::metadata(self.directory.path())?.uid())
    }

    pub fn configure<'a>(&self, command: &'a mut Command) -> &'a mut Command {
        command
            .env("WGTIME_ENDPOINT", self.endpoint())
            .env("WGT_WEIGHTFS", "off")
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
