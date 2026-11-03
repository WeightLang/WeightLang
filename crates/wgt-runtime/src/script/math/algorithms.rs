//! Statistics, ordering and vector transforms on a single finite f64 axis.
use super::super::{number, Value};
use super::vector::{list, values};
use anyhow::{ensure, Result};

pub(in crate::script) fn work(name: &str, len: usize) -> u64 {
    let n = len as u64;
    if matches!(name, "sort" | "argsort" | "median" | "quantile") {
        let levels = if n <= 1 {
            0
        } else {
            u64::BITS - (n - 1).leading_zeros()
        };
        n.saturating_mul(1 + u64::from(levels))
    } else {
        n
    }
}

fn scale(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |s, v| s.max(v.abs()))
}

fn deviation(x: &[f64], ddof: usize) -> Result<f64> {
    ensure!(ddof < x.len(), "ddof must be smaller than vector length");
    let s = scale(x);
    if s == 0.0 {
        return Ok(0.0);
    }
    // Normalize before centering/squaring: stddev can remain finite even when variance overflows.
    let anchor = x[0] / s;
    let mean = x
        .iter()
        .map(|v| (v / s - anchor) / (x.len() as f64))
        .sum::<f64>();
    let squared = x
        .iter()
        .map(|v| {
            let d = (v / s - anchor) - mean;
            d * d
        })
        .sum::<f64>();
    Ok(s * (squared / (x.len() - ddof) as f64).sqrt())
}

pub(in crate::script) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let mut x = values(&args[0])?;
    match name {
        "min" | "max" | "argmin" | "argmax" => {
            let mut best = 0;
            for i in 1..x.len() {
                if if matches!(name, "min" | "argmin") {
                    x[i] < x[best]
                } else {
                    x[i] > x[best]
                } {
                    best = i;
                }
            }
            number(if name.starts_with("arg") {
                best as f64
            } else {
                x[best]
            })
        }
        "variance" | "stddev" => {
            let s = deviation(&x, args[1].integer()?)?;
            number(if name == "variance" { s * s } else { s })
        }
        "sort" | "median" | "quantile" => {
            let q = if name == "quantile" {
                args[1].number()?
            } else {
                0.5
            };
            ensure!((0.0..=1.0).contains(&q), "quantile requires 0 <= q <= 1");
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if name == "sort" {
                return list(x.into_iter());
            }
            let pos = q * (x.len() - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let t = pos - lo as f64;
            // Interpolate without adding two large endpoints or subtracting opposite extremes.
            number(if lo == hi {
                x[lo]
            } else if x[lo].is_sign_negative() == x[hi].is_sign_negative() {
                x[lo] + t * (x[hi] - x[lo])
            } else {
                (1.0 - t) * x[lo] + t * x[hi]
            })
        }
        "argsort" => {
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
            list(indices.into_iter().map(|i| i as f64))
        }
        "cumsum" => {
            let mut sum = 0.0;
            list(x.into_iter().map(|v| {
                sum += v;
                sum
            }))
        }
        "clip" => {
            let lo = args[1].number()?;
            let hi = args[2].number()?;
            ensure!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "clip requires finite lower <= upper"
            );
            list(x.into_iter().map(|v| v.clamp(lo, hi)))
        }
        "cosine_similarity" => {
            let y = values(&args[1])?;
            ensure!(x.len() == y.len(), "vector lengths must match");
            let (sx, sy) = (scale(&x), scale(&y));
            ensure!(
                sx > 0.0 && sy > 0.0,
                "cosine similarity requires nonzero vectors"
            );
            let (mut dot, mut xx, mut yy) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (&a, &b) in x.iter().zip(&y) {
                let (a, b) = (a / sx, b / sy);
                dot = a.mul_add(b, dot);
                xx = a.mul_add(a, xx);
                yy = b.mul_add(b, yy);
            }
            number((dot / xx.sqrt() / yy.sqrt()).clamp(-1.0, 1.0))
        }
        _ => unreachable!("validated algorithm builtin"),
    }
}
