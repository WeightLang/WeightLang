//! CPU reference: preserve sample order, reductions and activation arithmetic.
use super::Objective;
use crate::{accumulate, Op, Session, Storage};
use anyhow::{ensure, Context, Result};

fn tape(session: &mut Session, x: &[f32]) -> Result<Vec<Vec<f32>>> {
    ensure!(
        x.len() == session.report.input && x.iter().all(|v| v.is_finite()),
        "invalid training input"
    );
    let mut tape = vec![x.to_vec()];
    for op in &session.ops {
        let x = tape.last().context("empty tape")?;
        let y = match *op {
            Op::Linear {
                weight,
                bias,
                input,
                output,
            } => {
                let mut w = vec![0.; input * output];
                accumulate(&mut session.parameters, weight, 0, &mut w)
                    .map_err(anyhow::Error::msg)?;
                let mut y = vec![0.; output];
                if let Some(bias) = bias {
                    accumulate(&mut session.parameters, bias, 0, &mut y)
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
    session: &mut Session,
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
    let mut gradients: Vec<Vec<f32>> = session.parameters.iter().map(|p| vec![0.; p.len]).collect();
    let mut loss = 0.;
    for (sample, x) in inputs.iter().enumerate() {
        let tape = tape(session, x)?;
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
        for (index, op) in session.ops.iter().enumerate().rev() {
            let (x, y) = (&tape[index], &tape[index + 1]);
            match *op {
                Op::Linear {
                    weight,
                    bias,
                    input,
                    output,
                } => {
                    let mut w = vec![0.; input * output];
                    accumulate(&mut session.parameters, weight, 0, &mut w)
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
    for index in (0..session.parameters.len()).rev() {
        let parameter = &session.parameters[index];
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
