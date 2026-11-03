//! Bounded numeric vector construction and indexing. Empty vectors are supported.
use super::super::{number, Value};
use anyhow::{ensure, Result};

fn finite(v: &Value) -> Result<f64> {
    let x = v.number()?;
    ensure!(x.is_finite(), "expected finite number");
    Ok(x)
}
fn numeric(v: &Value) -> Result<&[Value]> {
    let xs = v.list()?;
    for x in xs {
        finite(x)?;
    }
    Ok(xs)
}

pub(in crate::script) fn call(name: &str, a: &[Value], remaining: &mut u64) -> Result<Value> {
    // Compute an output upper bound and charge all input/output elements before allocating.
    let n = match name {
        "zeros" | "ones" | "full" => a[0].integer()?,
        "linspace" => a[2].integer()?,
        "one_hot" => a[1].integer()?,
        "concat" => a[0]
            .list()?
            .len()
            .checked_add(a[1].list()?.len())
            .ok_or_else(|| anyhow::anyhow!("size overflow"))?,
        "vector_slice" => {
            let (start, end) = (a[1].integer()?, a[2].integer()?);
            ensure!(
                start <= end && end <= a[0].list()?.len(),
                "slice requires 0 <= start <= end <= N"
            );
            end - start
        }
        "gather" => a[1].list()?.len(),
        "compress" | "reverse" => a[0].list()?.len(),
        _ => unreachable!("validated data builtin"),
    };
    ensure!(n <= 1_000_000, "vector output limit exceeded");
    let work = a.iter().fold(n as u64, |sum, v| {
        sum.saturating_add(match v {
            Value::List(xs) => xs.len() as u64,
            _ => 0,
        })
    });
    ensure!(*remaining >= work, "operation budget exhausted");
    *remaining -= work;
    match name {
        "zeros" | "ones" | "full" => {
            let value = match name {
                "zeros" => 0.0,
                "ones" => 1.0,
                _ => finite(&a[1])?,
            };
            Ok(Value::List(vec![Value::Number(value); n]))
        }
        "linspace" => {
            let (start, end) = (finite(&a[0])?, finite(&a[1])?);
            ensure!(n >= 1, "linspace requires count >= 1");
            let xs = (0..n)
                .map(|i| {
                    if i == 0 {
                        return number(start);
                    }
                    if i == n - 1 {
                        return number(end);
                    }
                    let t = i as f64 / (n - 1) as f64;
                    number(if start.is_sign_negative() == end.is_sign_negative() {
                        start + t * (end - start)
                    } else {
                        (1.0 - t) * start + t * end
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Value::List(xs))
        }
        "one_hot" => {
            let index = a[0].integer()?;
            ensure!(index < n, "one_hot index must be less than classes");
            let mut out = vec![Value::Number(0.0); n];
            out[index] = Value::Number(1.0);
            Ok(Value::List(out))
        }
        "concat" => {
            let (x, y) = (numeric(&a[0])?, numeric(&a[1])?);
            Ok(Value::List(x.iter().chain(y).cloned().collect()))
        }
        "vector_slice" => {
            let x = numeric(&a[0])?;
            Ok(Value::List(x[a[1].integer()?..a[2].integer()?].to_vec()))
        }
        "gather" => {
            let x = numeric(&a[0])?;
            let out = a[1]
                .list()?
                .iter()
                .map(|index| {
                    let i = index.integer()?;
                    ensure!(i < x.len(), "gather index out of bounds");
                    Ok(x[i].clone())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Value::List(out))
        }
        "compress" => {
            let x = numeric(&a[0])?;
            let mask = a[1].list()?;
            ensure!(x.len() == mask.len(), "mask length must match vector");
            let mut out = Vec::new();
            for (v, m) in x.iter().zip(mask) {
                if m.boolean()? {
                    out.push(v.clone());
                }
            }
            Ok(Value::List(out))
        }
        "reverse" => Ok(Value::List(numeric(&a[0])?.iter().rev().cloned().collect())),
        _ => unreachable!("validated data builtin"),
    }
}
