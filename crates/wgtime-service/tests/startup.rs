#![cfg(target_os = "linux")]

use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use wgtime::{Client, Request};

// Only children and paths owned by this test are cleaned up, even on assertion failure.
struct Race {
    children: Vec<Child>,
    root: PathBuf,
}
impl Drop for Race {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Race {
    fn new() -> Result<Self> {
        let base = std::env::var_os("CLAUDE_JOB_DIR")
            .map(|path| PathBuf::from(path).join("tmp"))
            .unwrap_or_else(|| PathBuf::from("/tmp/weightlang"));
        fs::create_dir_all(&base)?;
        let root = base.join(format!(
            "wgtime-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        Ok(Self {
            children: Vec::new(),
            root,
        })
    }

    fn spawn(&mut self, gated: bool) -> Result<()> {
        // Each shell reports that it reached the barrier, then exec replaces it
        // with an independent daemon process. Paths are argv, never shell code.
        let mut command = if gated {
            let mut command = Command::new("/bin/sh");
            command.args([
                "-c",
                "printf R; read -r gate; exec \"$@\"",
                "wgtime-start-gate",
            ]);
            command.arg(env!("CARGO_BIN_EXE_wgtime"));
            command
        } else {
            Command::new(env!("CARGO_BIN_EXE_wgtime"))
        };
        let child = command
            .arg("--test-directory")
            .arg(&self.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        self.children.push(child);
        Ok(())
    }

    fn client(&self) -> Result<Client> {
        // Client::connect sends Hello and validates the reply and peer UID.
        Client::connect(&self.root.join("service.sock"), unsafe { libc::geteuid() })
    }

    fn await_single_winner(&mut self, contenders: usize) -> Result<usize> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let mut alive = Vec::new();
            for (index, child) in self.children.iter_mut().take(contenders).enumerate() {
                if let Some(status) = child.try_wait()? {
                    anyhow::ensure!(!status.success(), "contender exited successfully");
                } else {
                    alive.push(index);
                }
            }
            anyhow::ensure!(!alive.is_empty(), "all contenders exited");
            if alive.len() == 1 && self.client().is_ok() {
                return Ok(alive[0]);
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "startup race timed out: {alive:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn inode(path: &Path) -> Result<(u64, u64)> {
    let metadata = fs::symlink_metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

fn assert_locked(lock: &File) {
    assert_eq!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        -1,
        "winner lost its singleton lock"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::EWOULDBLOCK)
    );
}

#[test]
fn simultaneous_cold_start_has_one_winner_and_survives_losers_and_restart() -> Result<()> {
    const CONTENDERS: usize = 8;
    let mut race = Race::new()?;
    assert!(!race.root.join("service.sock").exists());
    assert!(!race.root.join("service.lock").exists());
    for _ in 0..CONTENDERS {
        race.spawn(true)?;
    }
    // Wait for all independent processes before releasing any of them.
    for child in &mut race.children {
        let mut ready = [0];
        child
            .stdout
            .as_mut()
            .context("missing ready pipe")?
            .read_exact(&mut ready)?;
        assert_eq!(ready, [b'R']);
    }
    for child in &mut race.children {
        child
            .stdin
            .take()
            .context("missing gate pipe")?
            .write_all(b"go\n")?;
    }
    let winner = race.await_single_winner(CONTENDERS)?;
    for (index, child) in race.children.iter_mut().enumerate() {
        if index != winner {
            let mut error = String::new();
            child.stderr.take().unwrap().read_to_string(&mut error)?;
            assert!(
                error.contains("wgtime already running or lock unavailable"),
                "{error}"
            );
        }
    }
    let socket = race.root.join("service.sock");
    let lock_path = race.root.join("service.lock");
    let socket_inode = inode(&socket)?;
    let lock_inode = inode(&lock_path)?;
    let lock = OpenOptions::new().read(true).write(true).open(&lock_path)?;
    assert_locked(&lock);
    let mut client = race.client()?;
    assert_eq!(client.call(Request::Stats)?["sessions"], 0);

    // A late loser must leave both the winning endpoint and its locked inode intact.
    race.spawn(false)?;
    let late = race.children.len() - 1;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = race.children[late].try_wait()? {
            assert!(!status.success());
            break;
        }
        anyhow::ensure!(Instant::now() < deadline, "late contender did not exit");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(inode(&socket)?, socket_inode);
    assert_eq!(inode(&lock_path)?, lock_inode);
    assert_locked(&lock);
    assert_eq!(client.call(Request::Stats)?["sessions"], 0);
    assert_eq!(race.client()?.call(Request::Stats)?["sessions"], 0);
    drop(client);

    // SIGKILL deliberately leaves a stale socket. Only this test's winner is
    // killed; a fresh process must recover it after the kernel releases flock.
    race.children[winner].kill()?;
    race.children[winner].wait()?;
    assert!(socket.exists());
    assert_eq!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_UN) }, 0);
    race.spawn(false)?;
    let restarted = race.children.len() - 1;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        anyhow::ensure!(
            race.children[restarted].try_wait()?.is_none(),
            "restart exited"
        );
        if let Ok(mut client) = race.client() {
            assert_eq!(client.call(Request::Stats)?["sessions"], 0);
            break;
        }
        anyhow::ensure!(Instant::now() < deadline, "restart handshake timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(inode(&lock_path)?, lock_inode);
    assert_locked(&lock);
    Ok(())
}
