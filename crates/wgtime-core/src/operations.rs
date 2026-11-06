//! One service-owned storage set shared by every native operation.
use crate::compiled_forward::CompiledForward;
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};
use wgt_language::{
    adaptive::{Instruction, Parameter},
    operations::{OperationSchema, OperationsPlan},
};

struct Entry {
    schema: OperationSchema,
    code: CompiledForward,
    bindings: Vec<usize>,
    writes: BTreeMap<usize, usize>,
}
pub struct OperationSession {
    storage: Vec<Parameter>,
    parameter_count: usize,
    names: BTreeMap<String, usize>,
    entries: BTreeMap<String, Entry>,
    revision: u64,
    native_calls: u64,
    memory: usize,
}
impl OperationSession {
    pub fn prepare(plan: OperationsPlan, memory: usize) -> Result<Self> {
        let parameter_count = plan.parameters.len();
        let mut storage = plan.parameters;
        storage.extend(plan.states);
        let mut names = BTreeMap::new();
        let mut storage_bytes = 0usize;
        for (index, parameter) in storage.iter().enumerate() {
            ensure!(
                names.insert(parameter.name.clone(), index).is_none(),
                "duplicate operation storage identity: {}",
                parameter.name
            );
            storage_bytes = parameter
                .data
                .len()
                .checked_mul(4)
                .and_then(|bytes| storage_bytes.checked_add(bytes))
                .context("operation storage overflow")?;
        }
        // Storage, candidate writes and persistence staging are covered by admission.
        let mut remaining = memory
            .checked_sub(
                storage_bytes
                    .checked_mul(4)
                    .context("operation storage overflow")?,
            )
            .context("operation storage budget exceeded")?;
        let mut entries = BTreeMap::new();
        for (schema, mut program) in plan.entries {
            for parameter in &mut program.parameters {
                let &slot = names
                    .get(&parameter.name)
                    .context("operation initializer schema mismatch")?;
                parameter.data.clone_from(&storage[slot].data);
            }
            let code = CompiledForward::compile_transaction(&program, remaining)?;
            remaining = remaining
                .checked_sub(code.workspace_bytes())
                .context("operation workspace budget exceeded")?;
            let bindings = code
                .parameter_names()
                .iter()
                .map(|name| {
                    names
                        .get(name)
                        .copied()
                        .context("operation storage schema mismatch")
                })
                .collect::<Result<Vec<_>>>()?;
            let writes = program
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(i, op)| {
                    if let Instruction::Write { parameter, .. } = op {
                        Some(
                            names
                                .get(parameter)
                                .copied()
                                .context("operation write schema mismatch")
                                .map(|slot| (i, slot)),
                        )
                    } else {
                        None
                    }
                })
                .collect::<Result<_>>()?;
            entries.insert(
                schema.name.clone(),
                Entry {
                    schema,
                    code,
                    bindings,
                    writes,
                },
            );
            // Program initializer copies are released here; only native code/schema survive.
        }
        Ok(Self {
            storage,
            parameter_count,
            names,
            entries,
            revision: 0,
            native_calls: 0,
            memory,
        })
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn native_calls(&self) -> u64 {
        self.native_calls
    }
    pub fn schemas(&self) -> Vec<OperationSchema> {
        self.entries.values().map(|e| e.schema.clone()).collect()
    }
    pub fn call(
        &mut self,
        name: &str,
        inputs: &[Vec<f32>],
        budget: u64,
    ) -> Result<Option<Vec<f32>>> {
        let entry = self
            .entries
            .get_mut(name)
            .context("unknown model operation")?;
        ensure!(
            inputs.len() == entry.schema.inputs.len(),
            "operation argument count mismatch"
        );
        for (values, schema) in inputs.iter().zip(&entry.schema.inputs) {
            ensure!(
                values.len() == schema.width && values.iter().all(|v| v.is_finite()),
                "operation argument shape/finite mismatch: {}",
                schema.name
            );
        }
        let input = if inputs.is_empty() {
            vec![0.]
        } else {
            inputs.iter().flatten().copied().collect()
        };
        let values = entry
            .bindings
            .iter()
            .map(|&slot| self.storage[slot].data.as_slice())
            .collect::<Vec<_>>();
        let mut updates = BTreeMap::new();
        let writes = &entry.writes;
        let output =
            entry
                .code
                .run_with_hook(&input, &values, budget, false, &mut |index, values| {
                    if let Some(&slot) = writes.get(&index) {
                        updates.insert(slot, values.to_vec());
                    }
                    Ok(())
                })?;
        for (&slot, data) in &updates {
            validate(&self.storage[slot], data, true)?;
        }
        let next = if updates.is_empty() {
            self.revision
        } else {
            self.revision.checked_add(1).context("revision overflow")?
        };
        for (slot, data) in updates {
            self.storage[slot].data = data;
        }
        self.revision = next;
        self.native_calls = self.native_calls.saturating_add(1);
        Ok(entry.schema.output.map(|_| output))
    }
    pub fn parameter_values(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.storage[..self.parameter_count]
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.data.clone()))
            .collect()
    }
    pub fn state_values(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.storage[self.parameter_count..]
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.data.clone()))
            .collect()
    }
    /// Complete named candidate batch; no gradients, projection, or partial commit.
    pub fn update_parameters(
        &mut self,
        candidates: &[(String, Vec<usize>, Vec<f32>)],
    ) -> Result<()> {
        ensure!(!candidates.is_empty(), "empty parameter candidate batch");
        let mut seen = BTreeSet::new();
        let mut updates = Vec::with_capacity(candidates.len());
        for (name, shape, data) in candidates {
            let &slot = self
                .names
                .get(name)
                .filter(|&&slot| slot < self.parameter_count)
                .context("unknown candidate parameter")?;
            ensure!(seen.insert(slot), "duplicate candidate parameter");
            let p = &self.storage[slot];
            ensure!(
                !p.frozen && !p.probability,
                "candidate cannot update frozen or probability parameter"
            );
            ensure!(&p.shape == shape, "candidate parameter shape mismatch");
            validate(p, data, true)?;
            updates.push((slot, data));
        }
        let next = self.revision.checked_add(1).context("revision overflow")?;
        for (slot, data) in updates {
            self.storage[slot].data.clone_from(data);
        }
        self.revision = next;
        Ok(())
    }
    pub fn weights_bytes(&self, limit: usize) -> Result<Vec<u8>> {
        crate::weight_artifact::encode(&self.parameter_values(), limit.min(self.memory))
    }
    pub fn bind_weights(&mut self, bytes: &[u8], optimizer: &str) -> Result<()> {
        ensure!(
            optimizer == "none",
            "typed operations do not support optimizers or differentiation"
        );
        let parameters = &mut self.storage[..self.parameter_count];
        let schema = parameters
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone()))
            .collect::<Vec<_>>();
        let decoded = crate::weight_artifact::decode(&schema, bytes, self.memory)?;
        for (p, data) in parameters.iter().zip(&decoded) {
            validate(p, data, false)?;
            ensure!(
                !p.frozen || p.data == *data,
                "cannot replace frozen parameter"
            );
        }
        let next = self.revision.checked_add(1).context("revision overflow")?;
        for (p, data) in parameters.iter_mut().zip(decoded) {
            p.data = data;
        }
        self.revision = next;
        Ok(())
    }
}
fn validate(parameter: &Parameter, data: &[f32], strict_discrete: bool) -> Result<()> {
    ensure!(
        !parameter.frozen || parameter.data == data,
        "cannot update frozen parameter"
    );
    ensure!(
        data.len() == parameter.data.len() && data.iter().all(|v| v.is_finite()),
        "candidate storage shape/finite mismatch"
    );
    if let Some(domain) = &parameter.learn {
        for v in data {
            domain.validate(*v)?;
            if strict_discrete {
                match domain {
                    wgt_language::adaptive::LearnDomain::Bool => {
                        ensure!(*v == 0. || *v == 1., "bool candidate must be 0 or 1")
                    }
                    wgt_language::adaptive::LearnDomain::Int { .. } => {
                        ensure!(v.fract() == 0., "int candidate must be integral")
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
