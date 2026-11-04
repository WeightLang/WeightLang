//! Optimize initial parameters through a complete executed forward program.
//! Forward writes are differentiable local state, never batch-side effects.
use super::AdaptiveSession;
use crate::training::{Objective, Optimizer};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct State {
    optimizer: Optimizer,
    steps: u64,
    first: Vec<Vec<f32>>,
    second: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    format: String,
    program: super::Program,
    optimizer: Option<State>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CustomCheckpoint {
    format: String,
    program: super::Program,
    custom_optimizer: super::optimization::CustomState,
}

enum LossMode {
    Builtin,
    Custom { clip: Option<f32> },
}

impl AdaptiveSession {
    pub fn set_optimizer(&mut self, optimizer: Optimizer) -> Result<()> {
        self.ensure_mutable()?;
        self.custom_optimizer = None;
        let zeros = || {
            self.program
                .parameters
                .iter()
                .map(|p| vec![0.; p.data.len()])
                .collect()
        };
        self.optimizer = Some(State {
            optimizer,
            steps: 0,
            first: zeros(),
            second: zeros(),
        });
        Ok(())
    }

    /// All samples start from identical parameters; commit only a finite full batch.
    /// remaining charges forward instructions across the entire batch.
    pub fn train_step(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
        rate: f32,
        remaining: &mut u64,
        tape_budget: usize,
    ) -> Result<f32> {
        self.train_step_impl(
            inputs,
            objective,
            rate,
            remaining,
            tape_budget,
            LossMode::Builtin,
        )
    }

    pub fn train_custom(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        rate: f32,
        remaining: &mut u64,
        tape_budget: usize,
    ) -> Result<f32> {
        self.train_step_impl(
            inputs,
            Objective::Mse(targets),
            rate,
            remaining,
            tape_budget,
            LossMode::Custom { clip: None },
        )
    }

    /// Clip the global batch-mean gradient before committing one optimizer step.
    pub fn train_clipped(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        rate: f32,
        clip: f32,
        remaining: &mut u64,
        tape_budget: usize,
    ) -> Result<f32> {
        ensure!(
            clip.is_finite() && clip > 0.,
            "gradient clip norm must be positive and finite"
        );
        self.train_step_impl(
            inputs,
            Objective::Mse(targets),
            rate,
            remaining,
            tape_budget,
            LossMode::Custom { clip: Some(clip) },
        )
    }

    fn train_step_impl(
        &mut self,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
        rate: f32,
        remaining: &mut u64,
        tape_budget: usize,
        mode: LossMode,
    ) -> Result<f32> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        let mut loss_session = if matches!(mode, LossMode::Custom { .. }) {
            let program = self
                .program
                .loss
                .as_ref()
                .context("fit loss requires fn loss(prediction,target)")?;
            let session = Self::prepare((**program).clone(), tape_budget)?;
            ensure!(
                session.input_len() > self.output_len() && session.output_len() == 1,
                "custom loss shape mismatch"
            );
            Some(session)
        } else {
            None
        };
        ensure!(
            rate.is_finite() && rate > 0.,
            "learning rate must be positive and finite"
        );
        let mut state = self
            .optimizer
            .clone()
            .context("forward training requires optimizer sgd or adam")?;
        ensure!(!inputs.is_empty(), "empty training batch");
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
            .program
            .parameters
            .iter()
            .map(|p| vec![0.; p.data.len()])
            .collect();
        let mut loss = 0.;
        for (sample, x) in inputs.iter().enumerate() {
            ensure!(*remaining > 0, "operation budget exceeded");
            let result = self.forward(x, *remaining, tape_budget);
            *remaining = remaining.saturating_sub(self.steps());
            let mut pass = result?;
            let y = pass.output();
            let mut g = vec![0.; y.len()];
            match &objective {
                Objective::Mse(targets) => {
                    let target = &targets[sample];
                    ensure!(
                        target.len()
                            == loss_session
                                .as_ref()
                                .map_or(y.len(), |s| s.input_len() - y.len())
                            && target.iter().all(|v| v.is_finite()),
                        "invalid target"
                    );
                    if let Some(session) = &mut loss_session {
                        let input: Vec<f32> = y.iter().chain(target).copied().collect();
                        let result = session.forward(
                            &input,
                            *remaining,
                            tape_budget.saturating_sub(pass.tape_bytes()),
                        );
                        *remaining = remaining.saturating_sub(session.steps());
                        let mut loss_pass = result?;
                        loss += loss_pass.output()[0] / inputs.len() as f32;
                        loss_pass.backward(&[1. / inputs.len() as f32], &BTreeMap::new())?;
                        g.copy_from_slice(&loss_pass.input_gradient()?[..y.len()]);
                    } else {
                        for i in 0..y.len() {
                            let diff = y[i] - target[i];
                            loss += diff * diff / y.len() as f32 / inputs.len() as f32;
                            g[i] = 2. * diff / y.len() as f32 / inputs.len() as f32;
                        }
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
                    loss += advantage * (total.ln() - (y[action] - max)) / inputs.len() as f32;
                    for i in 0..y.len() {
                        g[i] = advantage
                            * ((y[i] - max).exp() / total - if i == action { 1. } else { 0. })
                            / inputs.len() as f32;
                    }
                }
            }
            pass.backward(&g, &BTreeMap::new())?;
            for (p, gradient) in self.program.parameters.iter().zip(&mut gradients) {
                if !p.frozen {
                    for (dst, src) in gradient.iter_mut().zip(pass.parameter_gradient(&p.name)?) {
                        *dst += src;
                    }
                }
            }
        }
        ensure!(
            loss.is_finite() && gradients.iter().flatten().all(|v| v.is_finite()),
            "nonfinite loss or gradient"
        );
        if let LossMode::Custom { clip: Some(limit) } = mode {
            let norm = gradients
                .iter()
                .flatten()
                .map(|g| f64::from(*g).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > f64::from(limit) {
                let scale = f64::from(limit) / norm;
                for g in gradients.iter_mut().flatten() {
                    *g = (f64::from(*g) * scale) as f32;
                }
            }
        }
        state.steps = state
            .steps
            .checked_add(1)
            .context("optimizer step overflow")?;
        let revision = self.revision.checked_add(1).context("revision overflow")?;
        let mut updates = Vec::new();
        for (i, p) in self.program.parameters.iter().enumerate() {
            let mut values = p.data.clone();
            if !p.frozen {
                for (j, value) in values.iter_mut().enumerate() {
                    let g = gradients[i][j];
                    let direction = match state.optimizer {
                        Optimizer::Sgd => g,
                        Optimizer::Adam => {
                            state.first[i][j] = 0.9 * state.first[i][j] + 0.1 * g;
                            state.second[i][j] = 0.999 * state.second[i][j] + 0.001 * g * g;
                            let m = state.first[i][j] / (1. - 0.9f32.powf(state.steps as f32));
                            let v = state.second[i][j] / (1. - 0.999f32.powf(state.steps as f32));
                            m / (v.sqrt() + 1e-8)
                        }
                    };
                    *value -= rate * direction;
                }
            }
            updates.push(values);
        }
        ensure!(
            updates
                .iter()
                .chain(&state.first)
                .chain(&state.second)
                .flatten()
                .all(|v| v.is_finite()),
            "nonfinite parameter or optimizer update"
        );
        for (p, values) in self.program.parameters.iter_mut().zip(updates) {
            p.data = values;
        }
        self.optimizer = Some(state);
        self.revision = revision;
        Ok(loss)
    }

    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        crate::training::write_new(path, &self.checkpoint_bytes()?)
    }

    pub(crate) fn checkpoint_bytes(&self) -> Result<Vec<u8>> {
        self.ensure_idle()?;
        let value = if let Some(state) = &self.custom_optimizer {
            super::optimization::validate_state(&state.value)?;
            serde_json::json!({"format":"weightlang.forward-training.v2", "program":self.program, "custom_optimizer":state})
        } else {
            serde_json::json!({"format":"weightlang.forward-training.v1", "program":self.program, "optimizer":self.optimizer})
        };
        Ok(serde_json::to_vec(&value)?)
    }

    pub fn checkpoint_bytes_bounded(&self, limit: usize) -> Result<Vec<u8>> {
        self.ensure_idle()?;
        let value = if let Some(state) = &self.custom_optimizer {
            super::optimization::validate_state(&state.value)?;
            serde_json::json!({"format":"weightlang.forward-training.v2", "program":self.program, "custom_optimizer":state})
        } else {
            serde_json::json!({"format":"weightlang.forward-training.v1", "program":self.program, "optimizer":self.optimizer})
        };
        crate::script::file_io::encode_json(&value, limit)
    }

    pub fn restore_checkpoint(value: serde_json::Value, budget: usize) -> Result<Self> {
        if value.get("format").and_then(|v| v.as_str()) == Some("weightlang.forward-training.v2") {
            let checkpoint: CustomCheckpoint = serde_json::from_value(value)?;
            super::optimization::validate_state(&checkpoint.custom_optimizer.value)?;
            let mut session = Self::prepare(checkpoint.program, budget)?;
            session.custom_optimizer = Some(checkpoint.custom_optimizer);
            return Ok(session);
        }
        let checkpoint: Checkpoint = serde_json::from_value(value)?;
        ensure!(
            checkpoint.format == "weightlang.forward-training.v1",
            "unsupported forward checkpoint"
        );
        let mut session = Self::prepare(checkpoint.program, budget)?;
        if let Some(state) = &checkpoint.optimizer {
            for moments in [&state.first, &state.second] {
                ensure!(
                    moments.len() == session.program.parameters.len(),
                    "optimizer state count mismatch"
                );
                for (m, p) in moments.iter().zip(&session.program.parameters) {
                    ensure!(
                        m.len() == p.data.len() && m.iter().all(|v| v.is_finite()),
                        "invalid optimizer state"
                    );
                }
            }
            ensure!(
                state.second.iter().flatten().all(|v| *v >= 0.),
                "negative second moment"
            );
        }
        session.optimizer = checkpoint.optimizer;
        Ok(session)
    }
}
