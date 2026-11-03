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

impl AdaptiveSession {
    pub fn set_optimizer(&mut self, optimizer: Optimizer) {
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
        self.train_step_impl(inputs,objective,rate,remaining,tape_budget,false)
    }

    pub fn train_custom(&mut self, inputs:&[Vec<f32>], targets:&[Vec<f32>], rate:f32, remaining:&mut u64, tape_budget:usize) -> Result<f32> {
        self.train_step_impl(inputs,Objective::Mse(targets),rate,remaining,tape_budget,true)
    }

    fn train_step_impl(&mut self, inputs:&[Vec<f32>], objective:Objective<'_>, rate:f32, remaining:&mut u64, tape_budget:usize, custom:bool) -> Result<f32> {
        let mut loss_session=if custom {
            let program=self.program.loss.as_ref().context("fit loss requires fn loss(prediction,target)")?;
            let session=Self::prepare((**program).clone(),tape_budget)?;
            ensure!(session.input_len()==2*self.output_len() && session.output_len()==1,"custom loss shape mismatch");
            Some(session)
        } else {None};
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
                        target.len() == y.len() && target.iter().all(|v| v.is_finite()),
                        "invalid target"
                    );
                    if let Some(session)=&mut loss_session {
                        let input:Vec<f32>=y.iter().chain(target).copied().collect();
                        let result=session.forward(&input,*remaining,tape_budget.saturating_sub(pass.tape_bytes()));
                        *remaining=remaining.saturating_sub(session.steps());
                        let mut loss_pass=result?;
                        loss += loss_pass.output()[0]/inputs.len() as f32;
                        loss_pass.backward(&[1./inputs.len() as f32],&BTreeMap::new())?;
                        g.copy_from_slice(&loss_pass.input_gradient()?[..y.len()]);
                    } else { for i in 0..y.len() {
                        let diff = y[i] - target[i];
                        loss += diff * diff / y.len() as f32 / inputs.len() as f32;
                        g[i] = 2. * diff / y.len() as f32 / inputs.len() as f32;
                    } }
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
        let value = serde_json::json!({"format":"weightlang.forward-training.v1", "program":self.program, "optimizer":self.optimizer});
        crate::training::write_new(path, &serde_json::to_vec(&value)?)
    }

    pub fn restore_checkpoint(value: serde_json::Value, budget: usize) -> Result<Self> {
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
