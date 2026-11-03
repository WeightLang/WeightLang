//! Single-axis f64 algorithms; no tensor conversion or implicit batch axis.
use super::super::{number, Value};
use anyhow::{ensure, Result};

pub(super) fn values(value: &Value) -> Result<Vec<f64>> {
    let list = value.list()?;
    ensure!(!list.is_empty(), "expected a nonempty numeric vector");
    list.iter()
        .map(|v| {
            let x = v.number()?;
            ensure!(x.is_finite(), "nonfinite vector element");
            Ok(x)
        })
        .collect()
}
pub(super) fn list(xs: impl Iterator<Item = f64>) -> Result<Value> {
    Ok(Value::List(xs.map(number).collect::<Result<_>>()?))
}
pub(in crate::script) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let x = values(&args[0])?;
    let n = x.len() as f64;
    match name {
        "dot" | "mse" | "mae" => {
            let y = values(&args[1])?;
            ensure!(x.len() == y.len(), "vector lengths must match");
            let mut result = 0.0;
            for (&a, &b) in x.iter().zip(&y) {
                result = match name {
                    "dot" => a.mul_add(b, result),
                    "mse" => {
                        let d = (a - b) / n.sqrt();
                        d.mul_add(d, result)
                    }
                    _ => result + (a - b).abs() / n,
                };
            }
            number(result)
        }
        "mean" => number(x.iter().map(|v| v / n).sum()),
        "norm" => number(x.iter().fold(0.0_f64, |s, v| s.hypot(*v))),
        "l2_normalize" => {
            let eps = args[1].number()?;
            ensure!(
                eps.is_finite() && eps > 0.0,
                "epsilon must be finite and positive"
            );
            // Scale before squaring; even norms larger than f64::MAX can normalize.
            let scale = x.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
            if scale == 0.0 {
                return list(x.into_iter());
            }
            let norm = x.iter().map(|v| (v / scale).powi(2)).sum::<f64>().sqrt();
            if eps / scale > norm {
                list(x.iter().map(|v| v / eps))
            } else {
                list(x.iter().map(|v| (v / scale) / norm))
            }
        }
        "softmax" | "log_softmax" | "logsumexp" | "cross_entropy" => {
            let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total = x.iter().map(|v| (v - max).exp()).sum::<f64>();
            let log_total = total.ln();
            match name {
                "softmax" => list(x.iter().map(|v| (v - max).exp() / total)),
                "log_softmax" => list(x.iter().map(|v| (v - max) - log_total)),
                "logsumexp" => number(max + log_total),
                _ => {
                    let target = args[1].integer()?;
                    ensure!(target < x.len(), "target class out of bounds");
                    number((max - x[target]) + log_total)
                }
            }
        }
        _ => unreachable!("validated vector builtin"),
    }
}
