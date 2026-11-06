//! Explicit CPU/GPU vector training. The forward tape and optimizer are explicit training state;
//! inference sessions retain their allocation-free contract.
use crate::{accumulate, Layer, Model, Op, Parameter, Session, Storage, Value};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
mod backend;
mod cpu;
pub use backend::{TrainingBackend, TrainingError};
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GpuTraining;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Optimizer {
    Sgd,
    Adam,
}

/// Immutable parameter/gradient view passed to a user-defined update strategy.
#[derive(Clone, Debug)]
pub struct ParameterGradient {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub gradients: Vec<f32>,
    pub frozen: bool,
}

/// Algorithm-independent CPU update hook. The strategy owns its state (momentum,
/// moments, population, schedules, etc.) and returns absolute candidate values.
/// The trainer validates every candidate before committing any parameter.
pub trait UpdateStrategy: Clone {
    /// Return candidates and a staged next strategy state. The trainer only
    /// installs that state after all parameter candidates pass validation.
    fn propose(
        &self,
        parameters: &[ParameterGradient],
        step: u64,
    ) -> Result<(Vec<Vec<f32>>, Self)>;
}

pub enum Objective<'a> {
    Mse(&'a [Vec<f32>]),
    /// Minimize mean(-advantage * log pi(action)); CE uses advantages of one.
    Policy {
        actions: &'a [usize],
        advantages: &'a [f32],
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    format: String,
    model: Model,
    optimizer: Optimizer,
    steps: u64,
    first: Vec<Vec<f32>>,
    second: Vec<Vec<f32>>,
}

pub struct Trainer {
    backend: TrainingBackend,
    pub session: Session,
    optimizer: Option<Optimizer>,
    steps: u64,
    first: Vec<Vec<f32>>,
    second: Vec<Vec<f32>>,
}

impl Trainer {
    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn new(model: Model, root: &Path, budget: usize, optimizer: Optimizer) -> Result<Self> {
        let session = Session::prepare(model, root, budget)?;
        // An optimizer cannot update file-backed values. Frozen files are allowed.
        ensure!(
            session
                .parameters
                .iter()
                .all(|p| p.frozen || !matches!(p.storage, Storage::File { .. })),
            "trainable file parameters must first be materialized"
        );
        let zeros = || session.parameters.iter().map(|p| vec![0.; p.len]).collect();
        Ok(Self {
            backend: TrainingBackend::Cpu,
            first: zeros(),
            second: zeros(),
            session,
            optimizer: Some(optimizer),
            steps: 0,
        })
    }

    /// Prepare an untrusted, self-contained model without opening parameter paths.
    /// File descriptors may carry an artifact, but its contents are not authority
    /// to open additional files in the service process.
    pub fn new_self_contained(model: Model, budget: usize, optimizer: Optimizer) -> Result<Self> {
        Self::ensure_self_contained(&model)?;
        Self::new(model, Path::new("."), budget, optimizer)
    }

    fn ensure_self_contained(model: &Model) -> Result<()> {
        for parameter in &model.parameters {
            // An allow-list makes future storage variants require an explicit
            // review before they are accepted at the service boundary.
            ensure!(
                matches!(
                    &parameter.value,
                    Value::Resident { .. } | Value::Packed { .. }
                        | Value::View { .. } | Value::Add { .. }
                ),
                "self-contained model required: external parameter files are forbidden"
            );
        }
        Ok(())
    }

    /// Prepare inference without creating optimizer moments or permitting updates.
    pub fn inference(model: Model, root: &Path, budget: usize) -> Result<Self> {
        Ok(Self { backend: TrainingBackend::Cpu, session: Session::prepare(model,root,budget)?, optimizer: None, steps: 0, first: Vec::new(), second: Vec::new() })
    }

    pub fn predict(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.iter().all(|v| v.is_finite()), "nonfinite input");
        let out = self.session.run(x).map_err(anyhow::Error::msg)?.to_vec();
        ensure!(out.iter().all(|v| v.is_finite()), "nonfinite prediction");
        Ok(out)
    }

    /// Compute batch-mean gradients without changing weights or optimizer state.
    pub fn gradients(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        self.backend.gradients(&mut self.session, inputs, objective)
    }

    /// Select an explicit training backend, also exposed through WGT training_device.
    /// Changing backend preserves weights, steps and checkpoint state.
    pub fn set_backend(&mut self, backend: TrainingBackend) -> Result<()> {
        self.backend = backend;
        Ok(())
    }

    pub fn backend(&self) -> &TrainingBackend {
        &self.backend
    }

    pub fn step(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
        rate: f32,
    ) -> Result<f32> {
        self.step_clipped(inputs, objective, rate, None)
    }

    /// Run one CPU training step with a caller-defined parameter update algorithm.
    ///
    /// The caller owns strategy state and may implement Momentum, AdamW, SAM,
    /// evolutionary search, or another rule. This path deliberately does not
    /// select or mutate the built-in SGD/Adam state. Candidate validation is
    /// completed before any model parameter or revision is changed.
    pub fn step_with_strategy<S: UpdateStrategy>(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
        strategy: &mut S,
    ) -> Result<f32> {
        ensure!(
            matches!(&self.backend, TrainingBackend::Cpu),
            "user-defined update strategies currently require the CPU backend"
        );
        ensure!(
            self.optimizer.is_none(),
            "select optimizer none when using a user-defined update strategy"
        );
        self.session.ensure_mutable()?;
        ensure!(self.session.parameters.iter().all(|p| matches!(&p.storage, Storage::Resident(_))),
            "custom strategy requires resident parameter storage");
        let next_step = self.steps.checked_add(1).context("strategy step overflow")?;
        let next_revision = self.session.revision.checked_add(1).context("revision overflow")?;
        let (loss, gradients) = self.gradients(inputs, objective)?;
        let names = self.session.parameter_names();
        let parameters = self.session.parameters.iter().enumerate().map(|(i, p)| {
            let values = match &p.storage {
                Storage::Resident(values) => values.to_vec(),
                _ => Vec::new(),
            };
            ParameterGradient {
                name: names[i].to_owned(),
                shape: p.shape.clone(),
                values,
                gradients: gradients[i].clone(),
                frozen: p.frozen,
            }
        }).collect::<Vec<_>>();
        let (candidates, next_strategy) = strategy.propose(&parameters, next_step)?;
        ensure!(candidates.len() == parameters.len(), "strategy candidate count mismatch");
        for (parameter, candidate) in parameters.iter().zip(&candidates) {
            ensure!(
                candidate.len() == parameter.values.len(),
                "strategy candidate shape mismatch for {}",
                parameter.name
            );
            ensure!(
                candidate.iter().all(|value| value.is_finite()),
                "strategy produced nonfinite values for {}",
                parameter.name
            );
            if parameter.frozen {
                ensure!(
                    candidate == &parameter.values,
                    "strategy attempted to update frozen parameter {}",
                    parameter.name
                );
            }
        }
        for (parameter, candidate) in self.session.parameters.iter_mut().zip(candidates) {
            if !parameter.frozen {
                if let Storage::Resident(values) = &mut parameter.storage {
                    values.copy_from_slice(&candidate);
                }
            }
        }
        self.steps = next_step;
        self.session.revision = next_revision;
        *strategy = next_strategy;
        Ok(loss)
    }

    pub fn step_clipped(&mut self, inputs: &[Vec<f32>], objective: Objective<'_>, rate: f32, clip: Option<f32>) -> Result<f32> {
        let optimizer = self.optimizer.context("training requires an explicitly selected optimizer")?;
        ensure!(clip.is_none_or(|v| v.is_finite() && v > 0.), "gradient clip norm must be positive and finite");
        self.session.ensure_mutable()?;
        ensure!(
            rate.is_finite() && rate > 0.,
            "learning rate must be positive and finite"
        );
        let step = self
            .steps
            .checked_add(1)
            .context("optimizer step overflow")?;
        let revision = self
            .session
            .revision
            .checked_add(1)
            .context("revision overflow")?;
        #[cfg(feature = "gpu")]
        if let TrainingBackend::Gpu(gpu) = &mut self.backend {
            ensure!(clip.is_none(), "gradient clipping is currently CPU-only");
            let rule = match optimizer {
                Optimizer::Sgd => gpu::UpdateRule::Sgd,
                Optimizer::Adam => gpu::UpdateRule::Adam {
                    first: &self.first,
                    second: &self.second,
                    step,
                },
            };
            let (loss, updates) = gpu.step(&mut self.session, inputs, objective, rate, rule)?;
            // No fallible operation remains after staging every owner. The CPU copy
            // is the committed state, including after device loss or external edits.
            for candidate in updates {
                self.session.parameters[candidate.index].storage =
                    Storage::Resident(candidate.values.into_boxed_slice());
                if let Some((first, second)) = candidate.moments {
                    self.first[candidate.index] = first;
                    self.second[candidate.index] = second;
                }
            }
            self.steps = step;
            self.session.revision = revision;
            return Ok(loss);
        }
        let (loss, mut gradients) = self.gradients(inputs, objective)?;
        if let Some(limit) = clip {
            let norm = gradients.iter().enumerate().filter(|(i, _)| !self.session.parameters[*i].frozen)
                .flat_map(|(_, values)| values).map(|g| f64::from(*g).powi(2)).sum::<f64>().sqrt();
            if norm > f64::from(limit) {
                let scale = f64::from(limit) / norm;
                for g in gradients.iter_mut().flatten() { *g = (f64::from(*g) * scale) as f32; }
            }
        }
        let (mut first, mut second) = (self.first.clone(), self.second.clone());
        let mut updates = Vec::new();
        for (i, parameter) in self.session.parameters.iter().enumerate() {
            if parameter.frozen {
                continue;
            }
            if let Storage::Resident(values) = &parameter.storage {
                let mut updated = values.to_vec();
                for j in 0..values.len() {
                    let g = gradients[i][j];
                    let direction = match optimizer {
                        Optimizer::Sgd => g,
                        Optimizer::Adam => {
                            first[i][j] = 0.9 * first[i][j] + 0.1 * g;
                            second[i][j] = 0.999 * second[i][j] + 0.001 * g * g;
                            let m = first[i][j] / (1. - 0.9f32.powf(step as f32));
                            let v = second[i][j] / (1. - 0.999f32.powf(step as f32));
                            m / (v.sqrt() + 1e-8)
                        }
                    };
                    updated[j] -= rate * direction;
                }
                ensure!(
                    updated.iter().all(|v| v.is_finite()),
                    "nonfinite parameter update"
                );
                updates.push((i, updated));
            }
        }
        ensure!(
            first.iter().chain(&second).flatten().all(|v| v.is_finite()),
            "nonfinite optimizer state"
        );
        for (i, values) in updates {
            self.session.parameters[i].storage = Storage::Resident(values.into_boxed_slice());
        }
        self.first = first;
        self.second = second;
        self.steps = step;
        self.session.revision = revision;
        Ok(loss)
    }

    pub fn weights_bytes(&mut self, limit: usize) -> Result<Vec<u8>> {
        let model = self.snapshot()?;
        let parameters = model.parameters.into_iter().filter_map(|p| match p.value {
            Value::Resident { data } => Some((p.name, p.shape, data)),
            Value::View { .. } | Value::Add { .. } => None,
            _ => unreachable!("snapshot materializes file storage"),
        }).collect::<Vec<_>>();
        crate::weight_artifact::encode(&parameters, limit)
    }

    pub fn bind_weights(&mut self, bytes: &[u8], optimizer: &str) -> Result<()> {
        self.session.ensure_mutable()?;
        let optimizer = match optimizer {
            "none" => None,
            "sgd" => Some(Optimizer::Sgd),
            "adam" => Some(Optimizer::Adam),
            _ => anyhow::bail!("optimizer must be none, sgd or adam"),
        };
        let mut model = self.snapshot()?;
        let schema = model.parameters.iter().filter(|p| matches!(p.value, Value::Resident { .. }))
            .map(|p| (p.name.clone(), p.shape.clone())).collect::<Vec<_>>();
        let values = crate::weight_artifact::decode(&schema, bytes, 64 * 1024 * 1024)?;
        let mut values = values.into_iter();
        for parameter in &mut model.parameters {
            if let Value::Resident { data } = &mut parameter.value { *data = values.next().unwrap(); }
        }
        let candidate = match optimizer {
            Some(optimizer) => Self::new(model, Path::new("."), self.session.report.managed_bytes, optimizer)?,
            None => Self::inference(model, Path::new("."), self.session.report.managed_bytes)?,
        };
        self.reload_candidate(candidate)
    }

    pub fn snapshot(&mut self) -> Result<Model> {
        let names: Vec<_> = self.session.parameter_names().into_iter().map(str::to_owned).collect();
        let mut parameters = Vec::new();
        for i in 0..names.len() {
            let p = &self.session.parameters[i];
            let value = match &p.storage {
                Storage::View { source, start } => Value::View {
                    source: names[*source].clone(),
                    start: *start,
                },
                Storage::Add { base, delta } => Value::Add {
                    base: names[*base].clone(),
                    delta: names[*delta].clone(),
                },
                _ => {
                    let mut data = vec![0.; p.len];
                    accumulate(&mut self.session.parameters, i, 0, &mut data)
                        .map_err(anyhow::Error::msg)?;
                    ensure!(
                        data.iter().all(|v| v.is_finite()),
                        "nonfinite snapshot parameter"
                    );
                    Value::Resident { data }
                }
            };
            let p = &self.session.parameters[i];
            parameters.push(Parameter {
                name: names[i].clone(),
                shape: p.shape.clone(),
                frozen: p.frozen,
                value,
            });
        }
        let layers = self
            .session
            .ops
            .iter()
            .map(|op| match *op {
                Op::Linear { weight, bias, .. } => Layer::Linear {
                    weight: names[weight].clone(),
                    bias: bias.map(|i| names[i].clone()),
                },
                Op::Relu => Layer::Relu,
                Op::Tanh => Layer::Tanh,
                Op::Sigmoid => Layer::Sigmoid,
                Op::Identity => Layer::Identity,
            })
            .collect();
        Ok(Model {
            output_type: self.session.output_type,
            version: if self.session.output_type.is_plain() { 1 } else { 2 },
            input: self.session.report.input,
            parameters,
            layers,
        })
    }

    pub(crate) fn reload_candidate(&mut self, mut candidate: Self) -> Result<()> {
        self.session.ensure_mutable()?;
        let structure = |model: Model| -> Result<serde_json::Value> {
            let mut value = serde_json::to_value(model)?;
            for parameter in value["parameters"].as_array_mut().context("missing parameters")? {
                if parameter.get("kind").and_then(|v| v.as_str()) == Some("resident") {
                    parameter.as_object_mut().context("invalid parameter")?.remove("data");
                }
            }
            Ok(value)
        };
        ensure!(structure(self.snapshot()?)? == structure(candidate.snapshot()?)?, "reload requires identical model structure");
        candidate.session.revision = self.session.revision.checked_add(1).context("revision overflow")?;
        *self = candidate;
        Ok(())
    }

    /// Create a new checkpoint; never overwrite an existing file.
    pub fn save(&mut self, path: &Path) -> Result<()> {
        write_new(path, &self.checkpoint_bytes()?)
    }

    pub(crate) fn checkpoint_bytes(&mut self) -> Result<Vec<u8>> {
        self.checkpoint_bytes_bounded(usize::MAX)
    }

    /// Encode a bounded checkpoint for a trusted service-side persistence owner.
    pub fn checkpoint_bytes_bounded(&mut self, limit: usize) -> Result<Vec<u8>> {
        let record = Checkpoint {
            format: "weightlang.native-training.v1".into(),
            model: self.snapshot()?,
            optimizer: self.optimizer.context("inference instance has no training checkpoint; save weights instead")?,
            steps: self.steps,
            first: self.first.clone(),
            second: self.second.clone(),
        };
        crate::script::file_io::encode_json(&record, limit)
    }

    /// Export native inference IR, without optimizer state or training callbacks.
    pub fn export(&mut self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec(&self.snapshot()?)?;
        write_new(path, &bytes)
    }

    pub fn restore(path: &Path, budget: usize) -> Result<Self> {
        Self::restore_value(serde_json::from_reader(std::fs::File::open(path)?)?, budget)
    }

    /// Restore a trusted local checkpoint; parameter paths retain their legacy semantics.
    pub fn restore_value(value: serde_json::Value, budget: usize) -> Result<Self> {
        Self::restore_value_with_policy(value, budget, false)
    }

    /// Restore an untrusted checkpoint without opening embedded parameter paths.
    pub fn restore_self_contained_value(value: serde_json::Value, budget: usize) -> Result<Self> {
        Self::restore_value_with_policy(value, budget, true)
    }

    fn restore_value_with_policy(
        value: serde_json::Value,
        budget: usize,
        self_contained: bool,
    ) -> Result<Self> {
        let c: Checkpoint = serde_json::from_value(value)?;
        if self_contained {
            Self::ensure_self_contained(&c.model)?;
        }
        ensure!(
            c.format == "weightlang.native-training.v1",
            "unsupported checkpoint format"
        );
        let mut trainer = Self::new(c.model, Path::new("."), budget, c.optimizer)?;
        for state in [&c.first, &c.second] {
            ensure!(
                state.len() == trainer.session.parameters.len(),
                "optimizer state count mismatch"
            );
            for (values, p) in state.iter().zip(&trainer.session.parameters) {
                ensure!(
                    values.len() == p.len && values.iter().all(|v| v.is_finite()),
                    "invalid optimizer state"
                );
            }
        }
        ensure!(
            c.second.iter().flatten().all(|v| *v >= 0.),
            "negative second moment"
        );
        trainer.steps = c.steps;
        trainer.first = c.first;
        trainer.second = c.second;
        Ok(trainer)
    }
}

pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temporary = parent.join(format!(".wgt-save-{}-{nonce}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result: Result<()> = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        // Link publication is atomic and refuses an existing destination, including symlinks.
        std::fs::hard_link(&temporary, path)?;
        Ok(())
    })();
    drop(file);
    let cleanup = std::fs::remove_file(&temporary);
    result?;
    cleanup?;
    Ok(())
}

/// Deterministic local PRNG; does not touch process-global random state.
pub fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
}

pub fn mlp(widths: &[usize], activation: &str, seed: u64) -> Result<Model> {
    ensure!(
        widths.len() >= 2 && widths.len() <= 64 && widths.iter().all(|n| *n > 0 && *n <= 16384),
        "invalid MLP widths"
    );
    let mut total = 0usize;
    for pair in widths.windows(2) {
        total = total
            .checked_add(pair[0] * pair[1] + pair[1])
            .context("model size overflow")?;
    }
    ensure!(
        total <= 4_000_000,
        "script MLP exceeds 4 million parameters"
    );
    ensure!(
        matches!(activation, "relu" | "tanh" | "sigmoid" | "identity"),
        "unknown activation"
    );
    let (mut parameters, mut layers, mut rng) = (Vec::new(), Vec::new(), seed);
    for (index, pair) in widths.windows(2).enumerate() {
        let (input, output) = (pair[0], pair[1]);
        let weight = format!("layer{index}.weight");
        let bias = format!("layer{index}.bias");
        let scale = (6. / (input + output) as f64).sqrt();
        parameters.push(Parameter {
            name: weight.clone(),
            shape: vec![output, input],
            frozen: false,
            value: Value::Resident {
                data: (0..input * output)
                    .map(|_| ((random(&mut rng) * 2. - 1.) * scale) as f32)
                    .collect(),
            },
        });
        parameters.push(Parameter {
            name: bias.clone(),
            shape: vec![output],
            frozen: false,
            value: Value::Resident {
                data: vec![0.; output],
            },
        });
        layers.push(Layer::Linear {
            weight,
            bias: Some(bias),
        });
        if index + 2 < widths.len() {
            layers.push(match activation {
                "relu" => Layer::Relu,
                "tanh" => Layer::Tanh,
                "sigmoid" => Layer::Sigmoid,
                "identity" => Layer::Identity,
                _ => bail!("unknown activation"),
            });
        }
    }
    Ok(Model {
        output_type: crate::categorical::VectorType::F32,
        version: 1,
        input: widths[0],
        parameters,
        layers,
    })
}

#[cfg(test)]
mod security_tests;
