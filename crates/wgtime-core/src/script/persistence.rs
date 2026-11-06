//! Runtime implementation shared by interpreted and compiled file syntax.
use super::{Runtime, Value};
use crate::file_io;
use anyhow::{bail, ensure, Context, Result};

const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

enum LoadedModel {
    Adaptive(crate::adaptive::AdaptiveSession),
    Sequential(crate::training::Trainer),
}

impl Runtime {
    fn authorize_file_syntax(&self) -> Result<()> {
        ensure!(
            self.permissions.allows_legacy_io(),
            "host policy requires capability-based I/O"
        );
        self.permissions.check_deadline()
    }

    fn charge_file_bytes(&mut self, bytes: usize) -> Result<()> {
        ensure!(
            bytes <= MAX_FILE_BYTES,
            "file exceeds 64 MiB syntax IO limit"
        );
        ensure!(
            bytes as u64 <= self.remaining,
            "operation budget exhausted by file IO bytes"
        );
        self.remaining -= bytes as u64;
        Ok(())
    }

    /// Entry point for generated Rust; values and path have already been evaluated once.
    pub fn save_file(
        &mut self,
        format: &str,
        value: Value,
        path: Value,
        overwrite: bool,
    ) -> Result<()> {
        self.authorize_file_syntax()?;
        let path = self.root.join(path.text()?);
        let limit = MAX_FILE_BYTES.min(self.remaining as usize);
        let bytes = match format {
            "text" => {
                let text = value.text()?;
                ensure!(
                    text.len() <= MAX_FILE_BYTES && text.len() as u64 <= self.remaining,
                    "text exceeds IO byte budget"
                );
                text.as_bytes().to_vec()
            }
            "json" => {
                validate_json(&value)?;
                file_io::encode_json(&value.json()?, limit)?
            }
            "model" => match value {
                Value::Adaptive(index) => {
                    file_io::encode_json(self.adaptive[index].lock()?.program(), limit)?
                }
                Value::Model(index) => file_io::encode_json(&self.models[index].lock()?.snapshot()?, limit)?,
                _ => bail!("save model requires a model handle"),
            },
            "weights" => match value {
                Value::Adaptive(index) => self.adaptive[index].lock()?.weights_bytes(limit)?,
                Value::Model(index) => self.models[index].lock()?.weights_bytes(limit)?,
                _ => bail!("save weights requires a model handle"),
            },
            "checkpoint" => match value {
                Value::Adaptive(index) => self.adaptive[index].lock()?.checkpoint_bytes_bounded(limit)?,
                Value::Model(index) => self.models[index].lock()?.checkpoint_bytes_bounded(limit)?,
                _ => bail!("save checkpoint requires a model handle"),
            },
            _ => bail!("unsupported save format"),
        };
        self.charge_file_bytes(bytes.len())?;
        self.permissions.check_deadline()?;
        file_io::publish(&path, &bytes, overwrite)?;
        Ok(())
    }

    fn install_loaded(&mut self, candidate: LoadedModel) -> Result<Value> {
        match candidate {
            LoadedModel::Adaptive(session) => {
                ensure!(self.adaptive.len() < 64, "model handle limit exceeded");
                let index = self.adaptive.len();
                self.adaptive.push(crate::weightfs::Mounted::auto(session)?);
                Ok(Value::Adaptive(index))
            }
            LoadedModel::Sequential(trainer) => {
                ensure!(self.models.len() < 64, "model handle limit exceeded");
                let index = self.models.len();
                self.models.push(crate::weightfs::Mounted::auto(trainer)?);
                Ok(Value::Model(index))
            }
        }
    }

    pub fn reload_file(
        &mut self,
        format: &str,
        target: Value,
        path: Value,
        optimizer: Option<Value>,
    ) -> Result<()> {
        self.authorize_file_syntax()?;
        ensure!(
            matches!(target, Value::Adaptive(_) | Value::Model(_)),
            "reload requires an existing model handle"
        );
        let path = self.root.join(path.text()?);
        let bytes =
            file_io::read_bounded(&path, MAX_FILE_BYTES.min(self.remaining as usize))?;
        self.charge_file_bytes(bytes.len())?;
        if format == "weights" {
            self.permissions.check_deadline()?;
            let optimizer = optimizer.as_ref().context("reload weights requires using optimizer")?.text()?;
            return match target {
                Value::Adaptive(index) => self.adaptive[index].lock()?.bind_weights(&bytes, optimizer),
                Value::Model(index) => self.models[index].lock()?.bind_weights(&bytes, optimizer),
                _ => unreachable!(),
            };
        }
        let candidate = prepare_model(format, &bytes, &path, optimizer.as_ref())?;
        self.permissions.check_deadline()?;
        match (target, candidate) {
            (Value::Adaptive(index), LoadedModel::Adaptive(candidate)) => {
                self.adaptive[index].lock()?.reload_candidate(candidate)
            }
            (Value::Model(index), LoadedModel::Sequential(candidate)) => {
                self.models[index].lock()?.reload_candidate(candidate)
            }
            _ => bail!("reload requires identical model kind"),
        }
    }

    pub(super) fn load_file(&mut self, format: &str, args: &[Value]) -> Result<Value> {
        self.authorize_file_syntax()?;
        let path = self.root.join(args[0].text()?);
        let bytes =
            file_io::read_bounded(&path, MAX_FILE_BYTES.min(self.remaining as usize))?;
        self.charge_file_bytes(bytes.len())?;
        match format {
            "text" => Ok(Value::Text(String::from_utf8(bytes)?)),
            "json" => Value::from_json(serde_json::from_slice(&bytes)?),
            "model" | "checkpoint" => {
                self.install_loaded(prepare_model(format, &bytes, &path, args.get(1))?)
            }
            _ => bail!("unsupported load format"),
        }
    }
}

fn prepare_model(
    format: &str,
    bytes: &[u8],
    path: &std::path::Path,
    optimizer: Option<&Value>,
) -> Result<LoadedModel> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let checkpoint = format == "checkpoint";
    let tag = value.get("format").and_then(|v| v.as_str());
    ensure!(
        if checkpoint {
            matches!(
                tag,
                Some(
                    "weightlang.forward-training.v1"
                        | "weightlang.forward-training.v2"
                        | "weightlang.native-training.v1"
                )
            )
        } else {
            value.get("format").is_none()
        },
        "load format does not match model/checkpoint kind"
    );
    let adaptive = value.get("instructions").is_some()
        || matches!(
            tag,
            Some("weightlang.forward-training.v1" | "weightlang.forward-training.v2")
        );
    if adaptive {
        let mut session = if checkpoint {
            crate::adaptive::AdaptiveSession::restore_checkpoint(value, 64 * 1024 * 1024)?
        } else {
            crate::adaptive::AdaptiveSession::prepare(
                serde_json::from_value(value)?,
                64 * 1024 * 1024,
            )?
        };
        if !checkpoint {
            match optimizer
                .context("load model requires using optimizer")?
                .text()?
            {
                "none" => {}
                "sgd" => session.set_optimizer(crate::training::Optimizer::Sgd)?,
                "adam" => session.set_optimizer(crate::training::Optimizer::Adam)?,
                _ => bail!("forward optimizer must be none, sgd or adam"),
            }
        }
        Ok(LoadedModel::Adaptive(session))
    } else {
        let trainer = if checkpoint {
            crate::training::Trainer::restore_value(value, 256 * 1024 * 1024)?
        } else {
            let optimizer = match optimizer
                .context("load model requires using optimizer")?
                .text()?
            {
                "sgd" => crate::training::Optimizer::Sgd,
                "adam" => crate::training::Optimizer::Adam,
                _ => bail!("optimizer must be sgd or adam"),
            };
            crate::training::Trainer::new(
                serde_json::from_value(value)?,
                path.parent().unwrap_or(std::path::Path::new(".")),
                256 * 1024 * 1024,
                optimizer,
            )?
        };
        Ok(LoadedModel::Sequential(trainer))
    }
}

fn validate_json(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) => ensure!(n.is_finite(), "JSON requires finite numbers"),
        Value::List(values) | Value::CategoricalVector(_, values) => {
            for value in values {
                validate_json(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_json(value)?;
            }
        }
        Value::Text(_) | Value::Bool(_) | Value::Null => {}
        _ => bail!("resource handles cannot be saved as JSON"),
    }
    Ok(())
}
