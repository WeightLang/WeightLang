//! CPU differentiation and explicit transactional host-defined updates.
use super::AdaptiveSession;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAX_STATE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomState {
    pub steps: u64,
    pub value: serde_json::Value,
}

#[derive(Debug)]
pub struct GradientSnapshot {
    pub revision: u64,
    pub loss: f32,
    pub names: Vec<String>,
    pub shapes: Vec<Vec<usize>>,
    pub parameters: Vec<Vec<f32>>,
    pub gradients: Vec<Vec<f32>>,
    pub state: CustomState,
    pub bytes: usize,
}

pub fn validate_state(value: &serde_json::Value) -> Result<()> {
    fn visit(value: &serde_json::Value, depth: usize, bytes: &mut usize) -> Result<()> {
        ensure!(depth <= 32, "optimizer state nesting exceeds 32");
        *bytes = bytes
            .checked_add(16)
            .context("optimizer state size overflow")?;
        match value {
            serde_json::Value::Number(n) => ensure!(
                n.as_f64().is_some_and(f64::is_finite),
                "nonfinite optimizer state"
            ),
            serde_json::Value::String(s) => {
                *bytes = bytes
                    .checked_add(s.len())
                    .context("optimizer state size overflow")?
            }
            serde_json::Value::Array(values) => {
                for v in values {
                    visit(v, depth + 1, bytes)?;
                }
            }
            serde_json::Value::Object(values) => {
                for (k, v) in values {
                    *bytes = bytes
                        .checked_add(k.len())
                        .context("optimizer state size overflow")?;
                    visit(v, depth + 1, bytes)?;
                }
            }
            _ => {}
        }
        ensure!(
            *bytes <= MAX_STATE_BYTES,
            "optimizer state exceeds byte budget"
        );
        Ok(())
    }
    visit(value, 0, &mut 0)
}

impl AdaptiveSession {
    /// Differentiate a complete batch without committing forward writes.
    pub fn differentiate_custom(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        remaining: &mut u64,
        budget: usize,
    ) -> Result<GradientSnapshot> {
        ensure!(
            self.optimizer.is_none(),
            "custom optimization requires optimizer none"
        );
        ensure!(
            !inputs.is_empty() && inputs.len() == targets.len(),
            "invalid training batch"
        );
        let loss_program = self
            .program
            .loss
            .as_ref()
            .context("differentiate requires fn loss")?;
        let mut loss_session = Self::prepare((**loss_program).clone(), budget)?;
        let indices: Vec<_> = self
            .program
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.frozen)
            .map(|(i, _)| i)
            .collect();
        let count = indices.iter().try_fold(0usize, |n, &i| {
            n.checked_add(self.program.parameters[i].data.len())
                .context("gradient size overflow")
        })?;
        let state = self.custom_optimizer.clone().unwrap_or(CustomState {
            steps: 0,
            value: serde_json::Value::Null,
        });
        let state_bytes = serde_json::to_vec(&state)?.len();
        let metadata = indices.iter().try_fold(0usize, |n, &i| {
            let p = &self.program.parameters[i];
            n.checked_add(p.name.len() + p.shape.len() * std::mem::size_of::<usize>() + 128)
                .context("gradient metadata overflow")
        })?;
        let bytes = count
            .checked_mul(12)
            .and_then(|n| n.checked_add(state_bytes))
            .and_then(|n| n.checked_add(metadata))
            .context("gradient size overflow")?;
        ensure!(bytes <= budget, "gradient snapshot exceeds byte budget");
        let mut gradients: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| vec![0.; self.program.parameters[i].data.len()])
            .collect();
        let mut loss = 0.;
        for (input, target) in inputs.iter().zip(targets) {
            ensure!(*remaining > 0, "operation budget exceeded");
            let result = self.forward(input, *remaining, budget - bytes);
            *remaining = remaining.saturating_sub(self.steps());
            let mut pass = result?;
            let loss_input: Vec<_> = pass.output().iter().chain(target).copied().collect();
            let result = loss_session.forward(
                &loss_input,
                *remaining,
                budget.saturating_sub(bytes + pass.tape_bytes()),
            );
            *remaining = remaining.saturating_sub(loss_session.steps());
            let mut loss_pass = result?;
            // Charge adjoint traversal and batch reduction before exposing gradients.
            let work = (pass.tape_bytes() / 4 + loss_pass.tape_bytes() / 4 + count) as u64;
            let available = *remaining;
            *remaining = remaining.saturating_sub(work);
            ensure!(
                available >= work,
                "operation budget exceeded during backward"
            );
            loss += loss_pass.output()[0] / inputs.len() as f32;
            loss_pass.backward(&[1. / inputs.len() as f32], &BTreeMap::new())?;
            pass.backward(
                &loss_pass.input_gradient()?[..pass.output().len()],
                &BTreeMap::new(),
            )?;
            for (&index, gradient) in indices.iter().zip(&mut gradients) {
                for (dst, src) in gradient
                    .iter_mut()
                    .zip(pass.parameter_gradient(&self.program.parameters[index].name)?)
                {
                    *dst += src;
                }
            }
        }
        ensure!(
            loss.is_finite() && gradients.iter().flatten().all(|v| v.is_finite()),
            "nonfinite loss or gradient"
        );
        Ok(GradientSnapshot {
            revision: self.revision,
            loss,
            names: indices
                .iter()
                .map(|&i| self.program.parameters[i].name.clone())
                .collect(),
            shapes: indices
                .iter()
                .map(|&i| self.program.parameters[i].shape.clone())
                .collect(),
            parameters: indices
                .iter()
                .map(|&i| self.program.parameters[i].data.clone())
                .collect(),
            gradients,
            state,
            bytes,
        })
    }

    /// Validate everything before changing any persistent model or optimizer state.
    pub fn commit_custom(
        &mut self,
        snapshot: &GradientSnapshot,
        values: Vec<Vec<f32>>,
        state: serde_json::Value,
    ) -> Result<f32> {
        ensure!(
            self.optimizer.is_none(),
            "custom optimization requires optimizer none"
        );
        ensure!(
            snapshot.revision == self.revision,
            "stale gradient snapshot"
        );
        validate_state(&state)?;
        let indices: Vec<_> = self
            .program
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.frozen)
            .map(|(i, _)| i)
            .collect();
        ensure!(
            values.len() == indices.len() && snapshot.names.len() == indices.len(),
            "candidate parameter count mismatch"
        );
        for ((&index, name), value) in indices.iter().zip(&snapshot.names).zip(&values) {
            let p = &self.program.parameters[index];
            ensure!(
                &p.name == name && p.data.len() == value.len(),
                "candidate parameter layout mismatch"
            );
            ensure!(
                value.iter().all(|v| v.is_finite()),
                "nonfinite candidate parameter"
            );
        }
        let steps = self
            .custom_optimizer
            .as_ref()
            .map_or(0, |s| s.steps)
            .checked_add(1)
            .context("optimizer step overflow")?;
        let revision = self.revision.checked_add(1).context("revision overflow")?;
        for (index, value) in indices.into_iter().zip(values) {
            self.program.parameters[index].data = value;
        }
        self.custom_optimizer = Some(CustomState {
            steps,
            value: state,
        });
        self.revision = revision;
        Ok(snapshot.loss)
    }
}
