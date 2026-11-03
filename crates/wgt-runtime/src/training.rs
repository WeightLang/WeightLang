//! CPU vector training. The forward tape and optimizer are explicit training state;
//! inference sessions retain their allocation-free contract.
use crate::{accumulate, Layer, Model, Op, Parameter, Session, Storage, Value};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Optimizer {
    Sgd,
    Adam,
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
    pub session: Session,
    optimizer: Optimizer,
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
            first: zeros(),
            second: zeros(),
            session,
            optimizer,
            steps: 0,
        })
    }

    pub fn predict(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.iter().all(|v| v.is_finite()), "nonfinite input");
        let out = self.session.run(x).map_err(anyhow::Error::msg)?.to_vec();
        ensure!(out.iter().all(|v| v.is_finite()), "nonfinite prediction");
        Ok(out)
    }

    fn tape(&mut self, x: &[f32]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            x.len() == self.session.report.input && x.iter().all(|v| v.is_finite()),
            "invalid training input"
        );
        let mut tape = vec![x.to_vec()];
        for op in &self.session.ops {
            let x = tape.last().context("empty tape")?;
            let y = match *op {
                Op::Linear {
                    weight,
                    bias,
                    input,
                    output,
                } => {
                    let mut w = vec![0.; input * output];
                    accumulate(&mut self.session.parameters, weight, 0, &mut w)
                        .map_err(anyhow::Error::msg)?;
                    let mut y = vec![0.; output];
                    if let Some(bias) = bias {
                        accumulate(&mut self.session.parameters, bias, 0, &mut y)
                            .map_err(anyhow::Error::msg)?;
                    }
                    for (r, value) in y.iter_mut().enumerate() {
                        for c in 0..input {
                            *value += w[r * input + c] * x[c];
                        }
                    }
                    y
                }
                Op::Relu => x.iter().map(|v| v.max(0.)).collect(),
                Op::Tanh => x.iter().map(|v| v.tanh()).collect(),
                Op::Sigmoid => x.iter().map(|v| 1. / (1. + (-v).exp())).collect(),
                Op::Identity => x.clone(),
            };
            ensure!(
                y.iter().all(|v| v.is_finite()),
                "nonfinite training activation"
            );
            tape.push(y);
        }
        Ok(tape)
    }

    /// Compute batch-mean gradients without changing weights or optimizer state.
    pub fn gradients(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        ensure!(!inputs.is_empty(), "training batch must be nonempty");
        match &objective {
            Objective::Mse(y) => ensure!(y.len() == inputs.len(), "target batch length mismatch"),
            Objective::Policy {
                actions,
                advantages,
            } => ensure!(
                actions.len() == inputs.len() && advantages.len() == inputs.len(),
                "policy batch length mismatch"
            ),
        }
        let mut gradients: Vec<Vec<f32>> = self
            .session
            .parameters
            .iter()
            .map(|p| vec![0.; p.len])
            .collect();
        let mut loss = 0.;
        for (sample, x) in inputs.iter().enumerate() {
            let tape = self.tape(x)?;
            let y = tape.last().context("empty tape")?;
            let mut g = vec![0.; y.len()];
            match &objective {
                Objective::Mse(targets) => {
                    let target = &targets[sample];
                    ensure!(
                        target.len() == y.len() && target.iter().all(|v| v.is_finite()),
                        "invalid target"
                    );
                    for i in 0..y.len() {
                        let diff = y[i] - target[i];
                        loss += diff * diff / y.len() as f32;
                        g[i] = 2. * diff / (y.len() * inputs.len()) as f32;
                    }
                }
                Objective::Policy {
                    actions,
                    advantages,
                } => {
                    let (action, advantage) = (actions[sample], advantages[sample]);
                    ensure!(
                        action < y.len() && advantage.is_finite(),
                        "invalid action or advantage"
                    );
                    let max = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let total: f32 = y.iter().map(|v| (v - max).exp()).sum();
                    loss += advantage * (total.ln() - (y[action] - max));
                    for i in 0..y.len() {
                        g[i] = advantage
                            * ((y[i] - max).exp() / total - if i == action { 1. } else { 0. })
                            / inputs.len() as f32;
                    }
                }
            }
            for (index, op) in self.session.ops.iter().enumerate().rev() {
                let (x, y) = (&tape[index], &tape[index + 1]);
                match *op {
                    Op::Linear {
                        weight,
                        bias,
                        input,
                        output,
                    } => {
                        let mut w = vec![0.; input * output];
                        accumulate(&mut self.session.parameters, weight, 0, &mut w)
                            .map_err(anyhow::Error::msg)?;
                        let mut previous = vec![0.; input];
                        for r in 0..output {
                            for c in 0..input {
                                gradients[weight][r * input + c] += g[r] * x[c];
                                previous[c] += g[r] * w[r * input + c];
                            }
                            if let Some(bias) = bias {
                                gradients[bias][r] += g[r];
                            }
                        }
                        g = previous;
                    }
                    Op::Relu => {
                        for (v, x) in g.iter_mut().zip(x) {
                            if *x <= 0. {
                                *v = 0.;
                            }
                        }
                    }
                    Op::Tanh => {
                        for (v, y) in g.iter_mut().zip(y) {
                            *v *= 1. - y * y;
                        }
                    }
                    Op::Sigmoid => {
                        for (v, y) in g.iter_mut().zip(y) {
                            *v *= y * (1. - y);
                        }
                    }
                    Op::Identity => {}
                }
            }
        }
        // Fold expression gradients into storage owners, preserving parameter sharing.
        for index in (0..self.session.parameters.len()).rev() {
            let parameter = &self.session.parameters[index];
            if parameter.frozen {
                gradients[index].fill(0.);
                continue;
            }
            let (earlier, current) = gradients.split_at_mut(index);
            let current = &current[0];
            match parameter.storage {
                Storage::View { source, start } => {
                    for (j, gradient) in current.iter().enumerate() {
                        earlier[source][start + j] += gradient;
                    }
                }
                Storage::Add { base, delta } => {
                    for (j, gradient) in current.iter().enumerate() {
                        earlier[base][j] += gradient;
                        earlier[delta][j] += gradient;
                    }
                }
                _ => {}
            }
        }
        loss /= inputs.len() as f32;
        ensure!(
            loss.is_finite() && gradients.iter().flatten().all(|v| v.is_finite()),
            "nonfinite loss or gradient"
        );
        Ok((loss, gradients))
    }

    pub fn step(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
        rate: f32,
    ) -> Result<f32> {
        ensure!(
            rate.is_finite() && rate > 0.,
            "learning rate must be positive and finite"
        );
        let (loss, gradients) = self.gradients(inputs, objective)?;
        let step = self
            .steps
            .checked_add(1)
            .context("optimizer step overflow")?;
        let revision = self
            .session
            .revision
            .checked_add(1)
            .context("revision overflow")?;
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
                    let direction = match self.optimizer {
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

    fn snapshot(&mut self) -> Result<Model> {
        let mut names = vec![String::new(); self.session.parameters.len()];
        for (name, &index) in &self.session.names {
            names[index] = name.clone();
        }
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
            version: 1,
            input: self.session.report.input,
            parameters,
            layers,
        })
    }

    /// Create a new checkpoint; never overwrite an existing file.
    pub fn save(&mut self, path: &Path) -> Result<()> {
        let record = Checkpoint {
            format: "weightlang.native-training.v1".into(),
            model: self.snapshot()?,
            optimizer: self.optimizer,
            steps: self.steps,
            first: self.first.clone(),
            second: self.second.clone(),
        };
        let bytes = serde_json::to_vec(&record)?;
        write_new(path, &bytes)
    }

    /// Export native inference IR, without optimizer state or training callbacks.
    pub fn export(&mut self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec(&self.snapshot()?)?;
        write_new(path, &bytes)
    }

    pub fn restore(path: &Path, budget: usize) -> Result<Self> {
        let c: Checkpoint = serde_json::from_reader(std::fs::File::open(path)?)?;
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

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
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
        version: 1,
        input: widths[0],
        parameters,
        layers,
    })
}
