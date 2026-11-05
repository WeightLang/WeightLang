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
    writes: BTreeMap<usize, String>,
}
pub struct OperationSession {
    parameters: Vec<Parameter>,
    states: Vec<Parameter>,
    entries: BTreeMap<String, Entry>,
    revision: u64,
    native_calls: u64,
    memory: usize,
}
impl OperationSession {
    pub fn prepare(plan: OperationsPlan, memory: usize) -> Result<Self> {
        let storage: usize = plan
            .parameters
            .iter()
            .chain(&plan.states)
            .map(|p| p.data.len() * 4)
            .sum();
        // Storage, candidate writes and persistence staging are covered by admission.
        let mut remaining = memory
            .checked_sub(
                storage
                    .checked_mul(4)
                    .context("operation storage overflow")?,
            )
            .context("operation storage budget exceeded")?;
        let mut entries = BTreeMap::new();
        for (schema, mut program) in plan.entries {
            for parameter in &mut program.parameters {
                parameter.data = plan
                    .parameters
                    .iter()
                    .chain(&plan.states)
                    .find(|p| p.name == parameter.name)
                    .context("operation initializer schema mismatch")?
                    .data
                    .clone();
            }
            let code = CompiledForward::compile_transaction(&program, remaining)?;
            remaining = remaining
                .checked_sub(code.workspace_bytes())
                .context("operation workspace budget exceeded")?;
            let writes = program
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(i, op)| {
                    if let Instruction::Write { parameter, .. } = op {
                        Some((i, parameter.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            entries.insert(
                schema.name.clone(),
                Entry {
                    schema,
                    code,
                    writes,
                },
            );
            // Program initializer copies are released here; only native code/schema survive.
        }
        Ok(Self {
            parameters: plan.parameters,
            states: plan.states,
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
        let all = self
            .parameters
            .iter()
            .chain(&self.states)
            .map(|p| (p.name.as_str(), p))
            .collect::<BTreeMap<_, _>>();
        let values = entry
            .code
            .parameter_names()
            .iter()
            .map(|n| {
                all.get(n.as_str())
                    .map(|p| p.data.as_slice())
                    .context("operation storage schema mismatch")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut updates = BTreeMap::new();
        let writes = &entry.writes;
        let output =
            entry
                .code
                .run_with_hook(&input, &values, budget, false, &mut |index, values| {
                    if let Some(name) = writes.get(&index) {
                        updates.insert(name.clone(), values.to_vec());
                    }
                    Ok(())
                })?;
        for (name, data) in &updates {
            validate(all[name.as_str()], data, true)?;
        }
        let next = if updates.is_empty() {
            self.revision
        } else {
            self.revision.checked_add(1).context("revision overflow")?
        };
        for p in self.parameters.iter_mut().chain(&mut self.states) {
            if let Some(data) = updates.remove(&p.name) {
                p.data = data;
            }
        }
        self.revision = next;
        self.native_calls = self.native_calls.saturating_add(1);
        Ok(entry.schema.output.map(|_| output))
    }
    pub fn parameter_values(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.parameters
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.data.clone()))
            .collect()
    }
    pub fn state_values(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.states
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
        for (name, shape, data) in candidates {
            ensure!(seen.insert(name), "duplicate candidate parameter");
            let p = self
                .parameters
                .iter()
                .find(|p| &p.name == name)
                .context("unknown candidate parameter")?;
            ensure!(
                !p.frozen && !p.probability,
                "candidate cannot update frozen or probability parameter"
            );
            ensure!(&p.shape == shape, "candidate parameter shape mismatch");
            validate(p, data, true)?;
        }
        let next = self.revision.checked_add(1).context("revision overflow")?;
        for (name, _, data) in candidates {
            self.parameters
                .iter_mut()
                .find(|p| &p.name == name)
                .unwrap()
                .data
                .clone_from(data);
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
        let schema = self
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone()))
            .collect::<Vec<_>>();
        let decoded = crate::weight_artifact::decode(&schema, bytes, self.memory)?;
        for (p, data) in self.parameters.iter().zip(&decoded) {
            validate(p, data, false)?;
            ensure!(
                !p.frozen || p.data == *data,
                "cannot replace frozen parameter"
            );
        }
        let next = self.revision.checked_add(1).context("revision overflow")?;
        for (p, data) in self.parameters.iter_mut().zip(decoded) {
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
