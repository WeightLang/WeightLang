//! Explicit host-scheduled recurrent tapes; no persistent hidden state or Python.
use super::{AdaptiveSession, ForwardPass, Instruction};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Connect one output slice to the next call's input slice. All values are f32;
/// callers select these ranges from the shared model's logical port schema.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateLink {
    pub input_offset: usize,
    pub output_offset: usize,
    pub width: usize,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSlice {
    pub offset: usize,
    pub width: usize,
}

#[derive(Debug, Serialize)]
pub struct SequenceReport {
    pub loss: f32,
    pub outputs: Vec<Vec<f32>>,
    /// Supplied hidden slots after step zero are overwritten, hence zero gradients.
    pub input_gradients: Vec<Vec<f32>>,
    pub parameter_gradients: BTreeMap<String, Vec<f32>>,
}

impl AdaptiveSession {
    fn recurrent_budget(&self, requested: usize) -> Result<usize> {
        let optimizer_bytes = self.optimizer_storage_bytes()?;
        let available = self
            .memory_limit
            .checked_sub(self.managed_bytes)
            .and_then(|n| n.checked_sub(optimizer_bytes))
            .context("recurrent session memory budget exceeded")?;
        Ok(requested.min(available))
    }

    /// Mean squared error over only the selected output features and supervised
    /// timesteps. A None target leaves that step unsupervised. Backpropagates
    /// through the explicit hidden links, accumulating shared parameter VJPs.
    /// This creates no optimizer updates and commits no forward writes.
    pub fn recurrent_gradients(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Option<Vec<f32>>],
        link: StateLink,
        supervised: OutputSlice,
        remaining: &mut u64,
        tape_budget: usize,
    ) -> Result<SequenceReport> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        ensure!(
            !self
                .program
                .instructions
                .iter()
                .any(|op| matches!(op, Instruction::Write { .. })),
            "recurrent differentiation requires pure forward and explicit hidden inputs"
        );
        ensure!(
            !inputs.is_empty() && inputs.len() <= 1024 && targets.len() == inputs.len(),
            "recurrent sequence requires 1..1024 steps and matching targets"
        );
        ensure!(
            link.width > 0
                && link
                    .input_offset
                    .checked_add(link.width)
                    .is_some_and(|n| n <= self.input_len())
                && link
                    .output_offset
                    .checked_add(link.width)
                    .is_some_and(|n| n <= self.output_len()),
            "invalid recurrent state link"
        );
        ensure!(
            supervised.width > 0
                && supervised
                    .offset
                    .checked_add(supervised.width)
                    .is_some_and(|n| n <= self.output_len()),
            "invalid supervised output slice"
        );
        ensure!(
            inputs
                .iter()
                .all(|row| row.len() == self.input_len() && row.iter().all(|x| x.is_finite())),
            "recurrent inputs must match the finite input contract"
        );
        let count = targets.iter().filter(|target| target.is_some()).count();
        ensure!(
            count > 0
                && targets
                    .iter()
                    .flatten()
                    .all(|row| row.len() == supervised.width && row.iter().all(|x| x.is_finite())),
            "recurrent targets must be finite and match the supervised slice"
        );
        ensure!(
            !self.program.instructions.iter().any(|op| matches!(
                op,
                Instruction::MemoryAlloc { .. }
                    | Instruction::MemoryRead { .. }
                    | Instruction::MemoryWrite { .. }
                    | Instruction::MemoryFree { .. }
            )),
            "recurrent sequences require explicit tensor state; workspace memory operations are unsupported"
        );
        let tape_budget = self.recurrent_budget(tape_budget)?;
        let parameter_elements = self.program.parameters.iter().try_fold(0usize, |n, p| {
            n.checked_add(p.data.len())
                .context("recurrent parameter size overflow")
        })?;
        let retained = inputs
            .len()
            .checked_mul(
                self.input_len()
                    .checked_add(self.output_len())
                    .context("recurrent size overflow")?,
            )
            .and_then(|n| n.checked_add(parameter_elements))
            .and_then(|n| n.checked_add(self.input_len()))
            .and_then(|n| n.checked_add(self.output_len()))
            .and_then(|n| n.checked_add(link.width))
            .and_then(|n| n.checked_mul(4))
            .context("recurrent storage overflow")?;
        ensure!(
            retained < tape_budget,
            "recurrent tape memory budget exceeded"
        );
        let mut used = retained;
        let mut passes: Vec<ForwardPass> = Vec::with_capacity(inputs.len());
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        for (step, input) in inputs.iter().enumerate() {
            ensure!(*remaining > 0, "recurrent operation budget exceeded");
            let mut input = input.clone();
            if step > 0 {
                input[link.input_offset..link.input_offset + link.width].copy_from_slice(
                    &outputs[step - 1][link.output_offset..link.output_offset + link.width],
                );
            }
            let pass = self.forward(&input, *remaining, tape_budget - used);
            *remaining = remaining.saturating_sub(self.steps());
            let pass = pass?;
            used = used
                .checked_add(pass.tape_bytes())
                .context("recurrent tape size overflow")?;
            ensure!(used <= tape_budget, "recurrent tape memory budget exceeded");
            outputs.push(pass.output().to_vec());
            passes.push(pass);
        }
        let mut parameter_gradients: BTreeMap<String, Vec<f32>> = self
            .program
            .parameters
            .iter()
            .map(|p| (p.name.clone(), vec![0.; p.data.len()]))
            .collect();
        let mut input_gradients = vec![Vec::new(); inputs.len()];
        let mut carry = vec![0.; link.width];
        let denominator = (count * supervised.width) as f32;
        let mut loss = 0f32;
        for step in (0..inputs.len()).rev() {
            let pass = &mut passes[step];
            let mut seed = vec![0.; self.output_len()];
            if let Some(target) = &targets[step] {
                for (i, &target) in target.iter().enumerate() {
                    let delta = outputs[step][supervised.offset + i] - target;
                    loss += delta * delta / denominator;
                    seed[supervised.offset + i] = 2. * delta / denominator;
                }
            }
            for (seed, carry) in seed[link.output_offset..link.output_offset + link.width]
                .iter_mut()
                .zip(&carry)
            {
                *seed += carry;
            }
            pass.backward(&seed, &BTreeMap::new())?;
            let mut gradient = pass.input_gradient()?.to_vec();
            carry.copy_from_slice(&gradient[link.input_offset..link.input_offset + link.width]);
            if step > 0 {
                gradient[link.input_offset..link.input_offset + link.width].fill(0.);
            }
            input_gradients[step] = gradient;
            for (name, total) in &mut parameter_gradients {
                for (total, value) in total.iter_mut().zip(pass.parameter_gradient(name)?) {
                    *total += value;
                }
            }
        }
        ensure!(
            loss.is_finite()
                && parameter_gradients
                    .values()
                    .flatten()
                    .chain(input_gradients.iter().flatten())
                    .all(|v| v.is_finite()),
            "nonfinite recurrent loss or gradient"
        );
        Ok(SequenceReport {
            loss,
            outputs,
            input_gradients,
            parameter_gradients,
        })
    }

    /// One atomic shared-parameter SGD/Adam step for the full recurrent sequence.
    /// Failed forward/backward/update leaves parameters and optimizer unchanged.
    pub fn train_recurrent(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Option<Vec<f32>>],
        link: StateLink,
        supervised: OutputSlice,
        rate: f32,
        clip: Option<f32>,
        remaining: &mut u64,
        tape_budget: usize,
    ) -> Result<f32> {
        ensure!(
            rate.is_finite() && rate > 0.,
            "learning rate must be positive and finite"
        );
        let clipping = crate::training::optimizer::GradientClip::new(clip)?;
        ensure!(
            self.optimizer.is_some(),
            "forward training requires optimizer sgd or adam"
        );
        // Reserve cloned optimizer moments and atomic replacement parameters in
        // addition to the report's aggregated gradient; transfer that gradient.
        let staging = self
            .program
            .parameters
            .iter()
            .try_fold(0usize, |total, p| {
                total
                    .checked_add(p.data.len())
                    .context("recurrent staging overflow")
            })?
            .checked_mul(12)
            .context("recurrent staging overflow")?;
        let tape_budget = self
            .recurrent_budget(tape_budget)?
            .checked_sub(staging)
            .context("recurrent optimizer staging memory budget exceeded")?;
        let mut report =
            self.recurrent_gradients(inputs, targets, link, supervised, remaining, tape_budget)?;
        let gradients = self
            .program
            .parameters
            .iter()
            .map(|p| {
                report
                    .parameter_gradients
                    .remove(&p.name)
                    .context("missing recurrent parameter gradient")
            })
            .collect::<Result<Vec<_>>>()?;
        self.apply_gradient_batch(gradients, rate, clipping)?;
        Ok(report.loss)
    }
}
