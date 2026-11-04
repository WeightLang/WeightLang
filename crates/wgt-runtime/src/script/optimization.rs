//! Bounded gradient resources for ordinary WGT host programs; never differentiates host code.
use super::{number, vector, Runtime, Value};
use crate::adaptive::optimization::GradientSnapshot;
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap};

const MEMORY: usize = 64 * 1024 * 1024;
const MAX_HANDLES: usize = 16;

#[derive(Default)]
pub(super) struct Gradients {
    next: u64,
    bytes: usize,
    entries: HashMap<u64, (usize, GradientSnapshot)>,
}
impl Runtime {
    pub(super) fn optimization_builtin(&mut self, name: &str, args: &[Value]) -> Result<Value> {
        if name == "differentiate" {
            let Value::Adaptive(model) = args[0] else {
                bail!("differentiate requires a CPU forward model")
            };
            ensure!(
                self.gradients.entries.len() < MAX_HANDLES,
                "live gradient handle limit exceeded; release or commit gradients"
            );
            let next = self
                .gradients
                .next
                .checked_add(1)
                .context("gradient identity overflow")?;
            let snapshot = self.adaptive[model].lock()?.differentiate_custom(
                &args[1].matrix()?,
                &args[2].matrix()?,
                &mut self.remaining,
                MEMORY - self.gradients.bytes,
            )?;
            self.gradients.bytes += snapshot.bytes;
            self.gradients.entries.insert(next, (model, snapshot));
            self.gradients.next = next;
            return Ok(Value::Gradient(next));
        }
        let Value::Gradient(id) = args[0] else {
            bail!("expected gradient handle")
        };
        let (model, snapshot) = self
            .gradients
            .entries
            .get(&id)
            .context("gradient handle released or committed")?;
        if name == "gradient_snapshot" {
            let mut fields = BTreeMap::new();
            fields.insert("loss".into(), number(snapshot.loss as f64)?);
            fields.insert(
                "parameters".into(),
                Value::List(snapshot.parameters.iter().cloned().map(vector).collect()),
            );
            fields.insert(
                "gradients".into(),
                Value::List(snapshot.gradients.iter().cloned().map(vector).collect()),
            );
            fields.insert(
                "names".into(),
                Value::List(snapshot.names.iter().cloned().map(Value::Text).collect()),
            );
            fields.insert(
                "shapes".into(),
                Value::List(
                    snapshot
                        .shapes
                        .iter()
                        .map(|s| Value::List(s.iter().map(|&n| Value::Number(n as f64)).collect()))
                        .collect(),
                ),
            );
            fields.insert(
                "state".into(),
                Value::from_json(snapshot.state.value.clone())?,
            );
            fields.insert("steps".into(), number(snapshot.state.steps as f64)?);
            return Ok(Value::Object(fields));
        }
        let result = if name == "commit_gradients" {
            // JSON conversion rejects every resource handle, including nested ones.
            let state = args[2].json()?;
            let loss =
                self.adaptive[*model]
                    .lock()?
                    .commit_custom(snapshot, args[1].matrix()?, state)?;
            number(loss as f64)?
        } else {
            ensure!(name == "release_gradient", "unknown gradient operation");
            Value::Null
        };
        let (_, snapshot) = self
            .gradients
            .entries
            .remove(&id)
            .expect("validated handle");
        self.gradients.bytes -= snapshot.bytes;
        Ok(result)
    }
}
