use super::Value;
use anyhow::{ensure, Result};

pub(super) fn encode(index: &Value, classes: &Value, remaining: &mut u64) -> Result<Value> {
    let (index, classes) = (index.integer()?, classes.integer()?);
    ensure!((1..=1_000_000).contains(&classes), "onehot classes must be 1..1000000");
    ensure!(index < classes, "onehot index must be less than classes");
    ensure!(*remaining >= classes as u64, "operation budget exhausted");
    *remaining -= classes as u64;
    let mut values = vec![Value::Number(0.); classes];
    values[index] = Value::Number(1.);
    Ok(Value::List(values))
}
