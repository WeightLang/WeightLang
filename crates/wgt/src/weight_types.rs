//! Validate finite domains and disjoint parameter slices before loading payloads.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn validate(interface: &Value, shapes: &BTreeMap<String, Vec<u64>>, aliases: &[(String, String)]) -> Result<()> {
    let types = interface["weight_types"].as_object().context("missing weight_types")?;
    for (name, values) in types {
        ensure!(crate::manifest::identifier(name), "invalid weighttype name");
        let values = values.as_array().context("weighttype must be a list")?;
        ensure!(!values.is_empty(), "empty weighttype");
        let mut seen = Vec::new();
        for value in values {
            let number = value.as_f64().context("weighttype must contain numbers")? as f32;
            ensure!(number.is_finite() && !seen.contains(&number), "nonfinite or duplicate f32 weighttype value");
            seen.push(number);
        }
    }
    let assignments = interface["weight_assignments"].as_array().context("missing weight_assignments")?;
    let mut rules: BTreeMap<&str, Vec<Option<(usize,u64,u64)>>> = BTreeMap::new();
    for spec in assignments {
        let fields = spec.as_object().context("invalid weight assignment")?;
        ensure!((fields.len() == 2 || fields.len() == 5) && fields.keys().all(|k| ["target","type","axis","start","stop"].contains(&k.as_str())), "invalid weight fields");
        let target = spec["target"].as_str().context("invalid weight target")?;
        let shape = shapes.get(target).context("unknown weight parameter")?;
        ensure!(types.contains_key(spec["type"].as_str().context("invalid weight type")?), "unknown weighttype");
        if let Some(adapters) = interface["adapters"].as_object() {
            for adapter in adapters.values() {
                let owner = target.rsplit_once('.').map(|(p,_)| p).unwrap_or("");
                ensure!(adapter["target"].as_str() != Some(owner), "weight types on adapters unsupported");
            }
        }
        let slice = if fields.len() == 5 {
            let axis = spec["axis"].as_u64().context("invalid weight axis")? as usize;
            let start = spec["start"].as_u64().context("invalid weight start")?;
            let stop = spec["stop"].as_u64().context("invalid weight stop")?;
            ensure!(axis < shape.len() && start < stop && stop <= shape[axis], "invalid weight slice");
            Some((axis,start,stop))
        } else { None };
        let canonical = aliases.iter().find(|(a,_)| a == target).map(|(_,b)| b.as_str()).unwrap_or(target);
        let previous = rules.entry(canonical).or_default();
        for other in previous.iter() {
            match (other,slice) {
                (Some((a,s,e)),Some((b,t,f))) => ensure!(*a == b && (*e <= t || f <= *s), "overlapping weight slices or mixed axes"),
                _ => anyhow::bail!("overlapping weight assignments"),
            }
        }
        previous.push(slice);
    }
    Ok(())
}
