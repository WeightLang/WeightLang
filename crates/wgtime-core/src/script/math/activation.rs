use super::super::{number, Value};
use anyhow::{ensure, Result};

pub(in crate::script) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let x = args[0].number()?;
    ensure!(x.is_finite(), "activation requires finite x");
    let alpha = if args.len() == 2 {
        let a = args[1].number()?;
        ensure!(
            a.is_finite() && a >= 0.0,
            "alpha must be finite and nonnegative"
        );
        a
    } else {
        0.0
    };
    number(match name {
        "relu" => x.max(0.0),
        "leaky_relu" => {
            if x >= 0.0 {
                x
            } else {
                alpha * x
            }
        }
        "elu" => {
            if x >= 0.0 {
                x
            } else {
                alpha * x.exp_m1()
            }
        }
        "hard_sigmoid" => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        "hard_swish" => {
            if x <= -3.0 {
                0.0
            } else if x >= 3.0 {
                x
            } else {
                x * (x / 6.0 + 0.5)
            }
        }
        _ => unreachable!("validated activation"),
    })
}
