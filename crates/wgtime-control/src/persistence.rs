//! Client-authorized file descriptors; weights are never read into the control process.
use crate::{ModelLease, Runtime, Value};
use anyhow::{ensure, Context, Result};
use std::{fs::{self, OpenOptions}, path::Path};
use wgtime::{publication::{self, Mode}, Request};

/// Fixed output names shared by compiled build clients and native artifact tooling.
pub const BUILD_MODEL_FILE: &str = "model.json";
pub const BUILD_CHECKPOINT_FILE: &str = "checkpoint.json";

fn artifact_client() -> Result<wgtime::Client> {
    let endpoint = std::env::var("WGTIME_ENDPOINT").unwrap_or_else(|_| wgtime::DEFAULT_ENDPOINT.into());
    wgtime::Client::connect(Path::new(&endpoint), unsafe { libc::geteuid() })
}

/// Validate an artifact in the service without retaining a model lease.
pub fn inspect_artifact(path: &Path, checkpoint: bool) -> Result<serde_json::Value> {
    let file = OpenOptions::new().read(true).open(path)?;
    ensure!(file.metadata()?.is_file(), "artifact requires regular file");
    artifact_client()?.call_with_file(Request::InspectArtifactFd { checkpoint, memory: 256 * 1024 * 1024 }, &file)
}

/// Normalize an inference artifact through the service, with no-clobber publication.
pub fn export_model(input: &Path, output: &Path) -> Result<()> {
    let mut client = artifact_client()?;
    let input = OpenOptions::new().read(true).open(input)?;
    ensure!(input.metadata()?.is_file(), "artifact requires regular file");
    let reply = client.call_with_file(Request::LoadFd { checkpoint: false, optimizer: "sgd".into(), memory: 256 * 1024 * 1024 }, &input)?;
    let handle = reply["handle"].as_u64().context("missing loaded handle")?;
    publication::write(output, Mode::CreateNew, |file| {
        client.call_with_file(Request::PersistFd { handle, checkpoint: false }, file)?;
        Ok(())
    })
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
        if !overwrite {
            match fs::symlink_metadata(&path) {
                Ok(_) => anyhow::bail!("target already exists; use overwrite"),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(error.into()),
            }
        }
        let mode = if overwrite { Mode::Replace } else { Mode::CreateNew };
        publication::write(&path, mode, |file| {
            self.client.call_with_file(
                if kind == "weights" { Request::ExportWeightsFd { handle: model.handle } } else {
                    Request::PersistFd { handle: model.handle, checkpoint: kind == "checkpoint" }
                },
                file,
            )?;
            Ok(())
        })
    }
    pub fn reload_file(&mut self, kind: &str, target: Value, path: Value, optimizer: Option<Value>) -> Result<()> {
        self.tick()?;
        ensure!(kind == "weights", "client reload currently requires weights");
        let Value::Model(model) = target else { anyhow::bail!("reload weights requires model lease") };
        let optimizer = optimizer.as_ref().context("reload weights requires using optimizer")?.text()?.to_owned();
        let file = OpenOptions::new().read(true).open(self.root.join(path.text()?))?;
        ensure!(file.metadata()?.is_file(), "load requires regular file");
        self.client.call_with_file(Request::BindWeightsFd { handle: model.handle, optimizer }, &file)?;
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
