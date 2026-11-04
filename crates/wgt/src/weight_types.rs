//! Validate finite domains and disjoint parameter slices before loading payloads.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn validate(
    interface: &Value,
    shapes: &BTreeMap<String, Vec<u64>>,
    aliases: &[(String, String)],
) -> Result<()> {
    let types = interface["weight_types"]
        .as_object()
        .context("missing weight_types")?;
    for (name, values) in types {
        ensure!(crate::manifest::identifier(name), "invalid wtype name");
        let values = values.as_array().context("wtype must be a list")?;
        ensure!(!values.is_empty(), "empty wtype");
        let mut seen = Vec::new();
        for value in values {
            let number = value.as_f64().context("wtype must contain numbers")? as f32;
            ensure!(
                number.is_finite() && !seen.contains(&number),
                "nonfinite or duplicate f32 wtype value"
            );
            seen.push(number);
        }
    }
    let assignments = interface["weight_assignments"]
        .as_array()
        .context("missing weight_assignments")?;
    type Slice = Option<(usize, u64, u64)>;
    let mut rules: BTreeMap<&str, Vec<Slice>> = BTreeMap::new();
    for spec in assignments {
        let fields = spec.as_object().context("invalid weight assignment")?;
        ensure!(
            (fields.len() == 2 || fields.len() == 5)
                && fields
                    .keys()
                    .all(|k| ["target", "type", "axis", "start", "stop"].contains(&k.as_str())),
            "invalid weight fields"
        );
        let target = spec["target"].as_str().context("invalid weight target")?;
        let shape = shapes.get(target).context("unknown weight parameter")?;
        ensure!(
            types.contains_key(spec["type"].as_str().context("invalid weight type")?),
            "unknown wtype"
        );
        if let Some(adapters) = interface["adapters"].as_object() {
            for adapter in adapters.values() {
                let owner = target.rsplit_once('.').map(|(p, _)| p).unwrap_or("");
                ensure!(
                    adapter["target"].as_str() != Some(owner),
                    "weight types on adapters unsupported"
                );
            }
        }
        let slice = if fields.len() == 5 {
            let axis = spec["axis"].as_u64().context("invalid weight axis")? as usize;
            let start = spec["start"].as_u64().context("invalid weight start")?;
            let stop = spec["stop"].as_u64().context("invalid weight stop")?;
            ensure!(
                axis < shape.len() && start < stop && stop <= shape[axis],
                "invalid weight slice"
            );
            Some((axis, start, stop))
        } else {
            None
        };
        let canonical = aliases
            .iter()
            .find(|(a, _)| a == target)
            .map(|(_, b)| b.as_str())
            .unwrap_or(target);
        let previous = rules.entry(canonical).or_default();
        for other in previous.iter() {
            match (other, slice) {
                (Some((a, s, e)), Some((b, t, f))) => ensure!(
                    *a == b && (*e <= t || f <= *s),
                    "overlapping weight slices or mixed axes"
                ),
                _ => anyhow::bail!("overlapping weight assignments"),
            }
        }
        previous.push(slice);
    }
    Ok(())
}

/// Match construction order: implicit embedding ties, then explicit share bindings.
pub fn parameter_aliases(
    layers: &Value,
    shapes: &BTreeMap<String, Vec<u64>>,
    explicit: &[(String, String)],
) -> Result<Vec<(String, String)>> {
    fn visit(layers: &Value, prefix: &str, owners: &mut BTreeMap<String, String>) -> Result<()> {
        let mut embeddings = Vec::new();
        for layer in layers.as_array().context("invalid layers")? {
            let name = layer["name"].as_str().context("invalid layer name")?;
            let path = if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            };
            if let Some(children) = layer.get("layers") {
                visit(children, &path, owners)?;
            }
            if layer["op"] == "embedding" {
                embeddings.push(format!("{path}.weight"));
            }
            let tie = layer["args"]["tie_embedding"].as_u64().unwrap_or(0);
            if tie > 0 {
                let owner = embeddings
                    .get((tie - 1) as usize)
                    .context("invalid embedding tie")?;
                let owner = owners
                    .get(owner)
                    .context("missing embedding parameter")?
                    .clone();
                owners.insert(format!("{path}.weight"), owner);
            }
        }
        Ok(())
    }
    let mut owners: BTreeMap<String, String> = shapes
        .keys()
        .map(|key| (key.clone(), key.clone()))
        .collect();
    visit(layers, "", &mut owners)?;
    let before = owners.clone();
    for (destination, source) in explicit {
        owners.insert(
            destination.clone(),
            before
                .get(source)
                .context("missing shared parameter")?
                .clone(),
        );
    }
    Ok(owners
        .into_iter()
        .filter(|(path, owner)| path != owner)
        .collect())
}
