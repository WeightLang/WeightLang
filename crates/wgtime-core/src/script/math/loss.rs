use super::super::{number, Value};
use super::vector::values;
use anyhow::{ensure, Result};

pub(in crate::script) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let x = values(&args[0])?;
    let y = values(&args[1])?;
    ensure!(x.len() == y.len(), "vector lengths must match");
    let delta = if name == "huber" {
        let d = args[2].number()?;
        ensure!(
            d.is_finite() && d > 0.0,
            "delta must be finite and positive"
        );
        d
    } else {
        0.0
    };
    let n = x.len() as f64;
    let mut result = 0.0;
    for (&x, &y) in x.iter().zip(&y) {
        let loss = match name {
            "bce_logits" => {
                ensure!((0.0..=1.0).contains(&y), "BCE targets must be in [0,1]");
                // Branch avoids both exp overflow and infinity-times-zero at hard labels.
                (if x >= 0.0 { (1.0 - y) * x } else { -y * x }) + (-x.abs()).exp().ln_1p()
            }
            "huber" => {
                let d = (x - y).abs();
                if d <= delta {
                    (0.5 * d) * d
                } else {
                    delta * (d - 0.5 * delta)
                }
            }
            "log_cosh" => {
                let d = (x - y).abs();
                if d <= 1.0 {
                    // cosh(d)-1 = 2*sinh(d/2)^2 preserves tiny residual losses.
                    let s = (0.5 * d).sinh();
                    (2.0 * s * s).ln_1p()
                } else {
                    d + (-2.0 * d).exp().ln_1p() - std::f64::consts::LN_2
                }
            }
            _ => unreachable!("validated loss"),
        };
        ensure!(loss.is_finite(), "nonfinite per-element loss");
        result += loss / n;
    }
    number(result)
}
