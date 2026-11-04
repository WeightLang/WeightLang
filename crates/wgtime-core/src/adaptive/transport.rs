//! Explicit JSON commands for native forward tapes. Losses and optimizers live
//! outside the interpreter; the transport never executes caller code.
use super::{AdaptiveSession, ForwardPass};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Parameters,
    Forward {
        input: Vec<f32>,
        #[serde(default)]
        parameters: BTreeMap<String, Vec<f32>>,
        tape_budget_bytes: Option<usize>,
    },
    Backward {
        tape: u64,
        output_gradient: Vec<f32>,
        #[serde(default)]
        state_gradients: BTreeMap<String, Vec<f32>>,
        #[serde(default)]
        retain: bool,
    },
    Release {
        tape: u64,
    },
}

#[derive(Default)]
pub struct Commands {
    tapes: BTreeMap<u64, ForwardPass>,
    next: u64,
}

impl Commands {
    /// Total tensor budget includes session storage and all live tapes, even when
    /// callers interleave several forward/backward graphs in the same process.
    pub fn execute(
        &mut self,
        session: &mut AdaptiveSession,
        line: &str,
        max_steps: u64,
        total_budget: usize,
    ) -> Result<serde_json::Value> {
        match serde_json::from_str::<Command>(line)? {
            Command::Parameters => {
                Ok(serde_json::json!({"parameters":session.program().parameters}))
            }
            Command::Forward {
                input,
                parameters,
                tape_budget_bytes,
            } => {
                ensure!(
                    self.tapes.len() < 64,
                    "at most 64 live forward tapes; release unused tapes"
                );
                let used =
                    self.tapes
                        .values()
                        .try_fold(session.managed_bytes(), |used, pass| {
                            used.checked_add(pass.tape_bytes())
                                .context("tape budget overflow")
                        })?;
                let remaining = total_budget
                    .checked_sub(used)
                    .context("total tape budget exceeded")?;
                let budget = tape_budget_bytes.unwrap_or(remaining);
                ensure!(
                    budget <= remaining,
                    "requested tape budget exceeds available total budget"
                );
                let next = self.next.checked_add(1).context("tape identity overflow")?;
                let pass =
                    session.forward_with_parameters(&input, &parameters, max_steps, budget)?;
                let state: BTreeMap<_, _> = pass
                    .parameter_names()
                    .map(|name| Ok((name, pass.state(name)?)))
                    .collect::<Result<_>>()?;
                let report = serde_json::json!({"tape":self.next,"output":pass.output(),"state":state,"tape_bytes":pass.tape_bytes(),"steps":pass.steps()});
                self.tapes.insert(self.next, pass);
                self.next = next;
                Ok(report)
            }
            Command::Backward {
                tape,
                output_gradient,
                state_gradients,
                retain,
            } => {
                let result = (|| {
                    let pass = self
                        .tapes
                        .get_mut(&tape)
                        .context("unknown or released tape")?;
                    pass.backward(&output_gradient, &state_gradients)?;
                    let gradients: BTreeMap<_, _> = pass
                        .parameter_names()
                        .map(|name| Ok((name, pass.parameter_gradient(name)?)))
                        .collect::<Result<_>>()?;
                    Ok(
                        serde_json::json!({"input_gradient":pass.input_gradient()?,"parameter_gradients":gradients}),
                    )
                })();
                if !retain {
                    self.tapes.remove(&tape);
                }
                result
            }
            Command::Release { tape } => {
                ensure!(
                    self.tapes.remove(&tape).is_some(),
                    "unknown or released tape"
                );
                Ok(serde_json::json!({"released":tape}))
            }
        }
    }
}
