//! Native scalar math shared by interpreted and compiled WGT.
use super::{number, Value};
use anyhow::{ensure, Result};

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(super) fn call(name: &str, args: &[Value]) -> Result<Value> {
    let x = args[0].number()?;
    number(match name {
        "rsqrt" => {
            ensure!(x > 0.0, "rsqrt requires x > 0");
            1.0 / x.sqrt()
        }
        "fma" => x.mul_add(args[1].number()?, args[2].number()?),
        "hypot" => x.hypot(args[1].number()?),
        "expm1" => x.exp_m1(),
        "log1p" => {
            ensure!(x > -1.0, "log1p requires x > -1");
            x.ln_1p()
        }
        "sigmoid" => sigmoid(x),
        "softplus" => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        "silu" => x * sigmoid(x),
        _ => unreachable!("validated math builtin"),
    })
}

pub(super) mod activation;
pub(super) mod algorithms;
pub(super) mod data;
pub(super) mod loss;
pub(super) mod probability;
pub(super) mod signal;
pub(super) mod vector;
