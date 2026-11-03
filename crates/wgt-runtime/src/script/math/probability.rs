use super::super::{number, Value};
use super::vector::{list, values};
use anyhow::{ensure, Result};

fn distribution(value: &Value) -> Result<Vec<f64>> {
    let mut p = values(value)?;
    ensure!(
        p.iter().all(|v| (0.0..=1.0).contains(v)),
        "probabilities must be in [0,1]"
    );
    let sum = p.iter().sum::<f64>();
    ensure!(
        (sum - 1.0).abs() <= 1e-9,
        "probabilities must sum to 1 (absolute tolerance 1e-9)"
    );
    for v in &mut p {
        *v /= sum;
    }
    Ok(p)
}

pub(in crate::script) fn call(name: &str, a: &[Value]) -> Result<Value> {
    if name == "normalize_probabilities" {
        let x = values(&a[0])?;
        ensure!(x.iter().all(|v| *v >= 0.0), "weights must be nonnegative");
        let scale = x.iter().copied().fold(0.0_f64, f64::max);
        ensure!(scale > 0.0, "weights must have positive total mass");
        let total = x.iter().map(|v| v / scale).sum::<f64>();
        return list(x.iter().map(|v| (v / scale) / total));
    }
    if name == "temperature_softmax" {
        let x = values(&a[0])?;
        let t = a[1].number()?;
        ensure!(
            t.is_finite() && t > 0.0,
            "temperature must be finite and positive"
        );
        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = x
            .iter()
            .map(|v| {
                let d = v - max;
                // Opposite large signs may overflow subtraction even when division rescales it.
                let scaled = if d.is_finite() {
                    d / t
                } else {
                    v / t - max / t
                };
                scaled.exp()
            })
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        return list(weights.into_iter().map(|v| v / total));
    }
    let p = distribution(&a[0])?;
    if name == "entropy" {
        return number(p.iter().filter(|v| **v > 0.0).map(|v| -v * v.ln()).sum());
    }
    let q = distribution(&a[1])?;
    ensure!(p.len() == q.len(), "distribution lengths must match");
    let mut result = 0.0;
    for (&p, &q) in p.iter().zip(&q) {
        if name == "js_divergence" {
            let max = p.max(q);
            if max == 0.0 {
                continue;
            }
            // Compute log((p+q)/2) without underflowing the mixture at subnormal inputs.
            let log_m = max.ln() + (p.min(q) / max).ln_1p() - std::f64::consts::LN_2;
            if p > 0.0 {
                result += 0.5 * (p * (p.ln() - log_m));
            }
            if q > 0.0 {
                result += 0.5 * (q * (q.ln() - log_m));
            }
        } else if p > 0.0 {
            ensure!(
                q > 0.0,
                "positive target mass with zero reference probability gives infinite divergence"
            );
            result += match name {
                "kl_divergence" => p * (p.ln() - q.ln()),
                "cross_entropy_probs" => -p * q.ln(),
                _ => unreachable!("validated probability function"),
            };
        }
    }
    // Exact divergences are nonnegative; remove small negative roundoff.
    number(result.max(0.0))
}
