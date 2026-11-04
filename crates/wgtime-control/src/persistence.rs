//! Client-authorized file descriptors; weights are never read into the control process.
use crate::{ModelLease, Runtime, Value};
use anyhow::{ensure, Context, Result};
use std::{
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use wgtime::Request;
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
impl Runtime {
    pub fn save_file(
        &mut self,
        kind: &str,
        value: Value,
        path: Value,
        overwrite: bool,
    ) -> Result<()> {
        self.tick()?;
        let path = self.root.join(path.text()?);
        let Value::Model(model) = value else {
            anyhow::bail!("expected model lease")
        };
        let old = match fs::symlink_metadata(&path) {
            Ok(m) => {
                ensure!(overwrite, "target already exists; use overwrite");
                ensure!(
                    m.is_file() && !m.permissions().readonly(),
                    "overwrite requires writable regular file"
                );
                Some(m)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let (temporary, file) = loop {
            let candidate = parent.join(format!(
                ".wgtime-save-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => break (Temporary(candidate), file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        };
        self.client.call_with_file(
            Request::PersistFd {
                handle: model.handle,
                checkpoint: kind == "checkpoint",
            },
            &file,
        )?;
        if let Some(old) = old {
            file.set_permissions(old.permissions())?;
        }
        file.sync_all()?;
        drop(file);
        if overwrite {
            if let Ok(metadata) = fs::symlink_metadata(&path) {
                ensure!(metadata.is_file(), "overwrite requires regular file");
            }
            fs::rename(&temporary.0, &path)?;
        } else {
            fs::hard_link(&temporary.0, &path)?;
        }
        Ok(())
    }
    pub fn load_file(
        &mut self,
        kind: &str,
        path: Value,
        optimizer: Option<Value>,
    ) -> Result<Value> {
        self.tick()?;
        let file = OpenOptions::new()
            .read(true)
            .open(self.root.join(path.text()?))?;
        ensure!(file.metadata()?.is_file(), "load requires regular file");
        let optimizer = optimizer
            .as_ref()
            .map(Value::text)
            .transpose()?
            .unwrap_or("none")
            .to_owned();
        let reply = self.client.call_with_file(
            Request::LoadFd {
                checkpoint: kind == "checkpoint",
                optimizer,
                memory: 256 * 1024 * 1024,
            },
            &file,
        )?;
        Ok(Value::Model(ModelLease {
            handle: reply["handle"].as_u64().context("missing loaded handle")?,
        }))
    }
}
