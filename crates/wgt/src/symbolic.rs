//! Exhaustively verify bounded integer shape contracts with the ordinary checker.
use crate::contracts::{infer_named, Contracts};
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const LIMIT: i64 = 2147483647;
fn evaluate(expr: &Value, env: &BTreeMap<String, i64>, depth: usize) -> Result<i64> {
    ensure!(depth <= 16, "dimension expression exceeds depth 16");
    let value = if let Some(n) = expr.as_i64() {
        n
    } else if let Some(name) = expr.as_str() {
        *env.get(name).context("unknown dimension")?
    } else {
        let nodes = expr.as_array().context("invalid dimension expression")?;
        ensure!(nodes.len() == 3, "invalid dimension expression");
        let a = evaluate(&nodes[1], env, depth + 1)?;
        let b = evaluate(&nodes[2], env, depth + 1)?;
        match nodes[0].as_str() {
            Some("+") => a + b,
            Some("-") => a - b,
            Some("*") => a * b,
            Some("//") if b > 0 => a.div_euclid(b),
            Some("%") if b > 0 => a.rem_euclid(b),
            _ => bail!("invalid integer dimension operation or divisor"),
        }
    };
    ensure!(
        (-LIMIT..=LIMIT).contains(&value),
        "dimension arithmetic overflow"
    );
    Ok(value)
}
fn satisfies(constraints: &[Value], env: &BTreeMap<String, i64>) -> Result<bool> {
    let mut result = true;
    for c in constraints {
        let nodes = c.as_array().context("invalid dimension constraint")?;
        ensure!(nodes.len() == 3, "invalid dimension constraint");
        let a = evaluate(&nodes[1], env, 0)?;
        let b = evaluate(&nodes[2], env, 0)?;
        result &= match nodes[0].as_str() {
            Some("==") => a == b,
            Some("!=") => a != b,
            Some("<") => a < b,
            Some("<=") => a <= b,
            Some(">") => a > b,
            Some(">=") => a >= b,
            _ => bail!("invalid dimension comparison"),
        };
    }
    Ok(result)
}
fn specialize(interface: &Value, env: &BTreeMap<String, i64>) -> Result<Value> {
    let mut result = interface.clone();
    for key in ["inputs", "outputs"] {
        for port in result[key].as_array_mut().context("missing named ports")? {
            let shape = port["shape"].as_array().context("missing shape")?;
            let expressions = port["shape_expr"]
                .as_array()
                .context("missing symbolic shape")?;
            ensure!(
                shape.len() == expressions.len(),
                "shape expression rank mismatch"
            );
            let resolved = expressions
                .iter()
                .map(|e| {
                    let n = evaluate(e, env, 0)?;
                    ensure!(n >= 0, "symbolic port dimensions must remain nonnegative");
                    Ok(Value::from(n))
                })
                .collect::<Result<Vec<_>>>()?;
            port["shape"] = Value::Array(resolved);
        }
    }
    for (key, ports) in [("input", "inputs"), ("output", "outputs")] {
        let mut first = result[ports][0].clone();
        first
            .as_object_mut()
            .context("missing primary port")?
            .remove("from");
        result[key] = first;
    }
    Ok(result)
}
fn strip_records(layers: &mut Value) -> Result<usize> {
    let mut count = 0;
    for layer in layers.as_array_mut().context("invalid layers")? {
        ensure!(
            !matches!(
                layer["op"].as_str(),
                Some("rearrange" | "reduce" | "flatten")
            ),
            "operator does not support bounded dimensions"
        );
        let obj = layer.as_object_mut().context("invalid layer")?;
        obj.remove("input");
        obj.remove("output");
        obj.remove("results");
        count += 1;
        if let Some(children) = obj.get_mut("layers") {
            count += strip_records(children)?;
        }
    }
    Ok(count)
}
pub fn validate(layers: &Value, interface: &Value) -> Result<()> {
    let dimensions = interface["dimensions"]
        .as_object()
        .context("missing dimensions")?;
    ensure!(
        (1..=8).contains(&dimensions.len()),
        "declare between 1 and 8 dimensions"
    );
    let constraints = interface["constraints"]
        .as_array()
        .context("missing constraints")?;
    ensure!(constraints.len() <= 64, "too many constraints");
    let mut domains = Vec::new();
    let mut combinations = 1u64;
    for (name, bounds) in dimensions {
        ensure!(crate::manifest::identifier(name), "invalid dimension name");
        ensure!(
            bounds.as_object().is_some_and(|o| o.len() == 2),
            "invalid dimension bounds"
        );
        let low = bounds["min"].as_i64().context("invalid min")?;
        let high = bounds["max"].as_i64().context("invalid max")?;
        ensure!(
            0 <= low && low <= high && high <= LIMIT,
            "invalid dimension bounds"
        );
        combinations = combinations
            .checked_mul((high - low + 1) as u64)
            .context("domain size overflow")?;
        ensure!(
            combinations <= 4096,
            "dimension domain exceeds 4096 combinations"
        );
        domains.push((name.clone(), low, high));
    }
    domains.sort_by(|a, b| a.0.cmp(&b.0));
    let mut bound = BTreeSet::new();
    for p in interface["inputs"].as_array().context("missing inputs")? {
        for e in p["shape_expr"]
            .as_array()
            .context("missing symbolic shape")?
        {
            if let Some(s) = e.as_str() {
                bound.insert(s.to_owned());
            }
        }
    }
    ensure!(
        bound == dimensions.keys().cloned().collect(),
        "each dimension must appear directly on an input axis"
    );
    let mut unrecorded = layers.clone();
    let count = strip_records(&mut unrecorded)?;
    ensure!(
        combinations * count as u64 <= 200000,
        "symbolic validation exceeds node budget"
    );
    fn emissions(layers: &Value,prefix: &str,contracts: &Contracts) -> Result<BTreeMap<String,(String,Vec<u64>)>> {
        let mut result = BTreeMap::new();
        for layer in layers.as_array().context("invalid layers")? {
            let name = layer["name"].as_str().context("invalid name")?;
            let path = if prefix.is_empty() { name.to_owned() } else { format!("{prefix}.{name}") };
            if layer["op"] == "scan_state" {
                let typ = &contracts.get(&path).context("missing scan contract")?.1;
                result.insert(path.clone(),(typ.dtype.clone(),typ.shape[1..].to_vec()));
            }
            if let Some(children) = layer.get("layers") { result.extend(emissions(children,&path,contracts)?); }
        }
        Ok(result)
    }
    let mut baseline = None;
    let mut emission_baseline = None;
    for index in 0..combinations {
        let mut quotient = index;
        let mut env = BTreeMap::new();
        for (name, low, high) in domains.iter().rev() {
            let width = (high - low + 1) as u64;
            env.insert(name.clone(), low + (quotient % width) as i64);
            quotient /= width;
        }
        if !satisfies(constraints, &env)? {
            continue;
        }
        let resolved = specialize(interface, &env)?;
        let mut tensors = BTreeMap::new();
        let mut contracts = Contracts::new();
        infer_named(&unrecorded, &resolved, &mut tensors, &mut contracts)?;
        let emitted = emissions(&unrecorded,"",&contracts)?;
        if let Some(ref expected) = emission_baseline { ensure!(&emitted == expected,"scan_state emitted sample shape/dtype must be invariant"); }
        else { emission_baseline = Some(emitted); }
        if let Some(ref expected) = baseline {
            ensure!(&tensors == expected, "dimension-dependent parameter layout");
        } else {
            for key in ["input", "output", "inputs", "outputs"] {
                ensure!(
                    interface[key] == resolved[key],
                    "symbolic representative mismatch"
                );
            }
            baseline = Some(tensors);
        }
    }
    ensure!(
        baseline.is_some(),
        "dimension constraints have no legal assignment"
    );
    Ok(())
}

pub fn label(port: &Value) -> Result<String> {
    fn expression(value: &Value) -> String {
        if let Some(items) = value.as_array().filter(|v| v.len() == 3) {
            format!(
                "({} {} {})",
                expression(&items[1]),
                items[0].as_str().unwrap_or("?"),
                expression(&items[2])
            )
        } else if let Some(name) = value.as_str() {
            name.to_owned()
        } else {
            value.to_string()
        }
    }
    let exprs = port["shape_expr"]
        .as_array()
        .context("missing symbolic shape")?;
    let dims = exprs
        .iter()
        .enumerate()
        .map(|(index, e)| {
            let value = expression(e);
            if let Some(axis) = port["axes"][index].as_str() {
                format!("{axis}={value}")
            } else {
                value
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "{}[{dims}]",
        port["dtype"].as_str().context("missing dtype")?
    ))
}
