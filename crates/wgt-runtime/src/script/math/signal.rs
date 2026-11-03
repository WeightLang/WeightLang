use super::super::Value;
use super::vector::{list, values};
use anyhow::{ensure, Result};

pub(in crate::script) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let x = values(&args[0])?;
    match name {
        "diff" => {
            ensure!(x.len() >= 2, "diff requires at least two elements");
            list(x.windows(2).map(|w| w[1] - w[0]))
        }
        "ema" => {
            let alpha = args[1].number()?;
            ensure!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0,1]");
            let mut state = x[0];
            list(x.iter().enumerate().map(|(i, &v)| {
                if i > 0 {
                    state = if alpha == 1.0 {
                        v
                    } else if state.is_sign_negative() == v.is_sign_negative() {
                        state + alpha * (v - state)
                    } else {
                        (1.0 - alpha) * state + alpha * v
                    };
                }
                state
            }))
        }
        "moving_mean" => {
            let w = args[1].integer()?;
            ensure!(w > 0 && w <= x.len(), "window must be in 1..=N");
            // Direct per-window averaging avoids drift from rolling add/subtract updates.
            list(
                x.windows(w)
                    .map(|window| window.iter().map(|v| v / w as f64).sum()),
            )
        }
        _ => unreachable!("validated signal operation"),
    }
}
