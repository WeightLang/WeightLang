use crate::{artifact::dimension, contracts::TensorType, manifest::identifier};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn side(text: &str) -> Result<Vec<Vec<String>>> {
    let mut remaining = text.trim();
    let mut groups = Vec::new();
    let mut seen = BTreeSet::new();
    while !remaining.is_empty() {
        let (names, rest) = if let Some(inner) = remaining.strip_prefix('(') {
            let end = inner.find(')').context("unclosed axis group")?;
            let rest = &inner[end + 1..];
            ensure!(
                rest.is_empty() || rest.starts_with(char::is_whitespace),
                "axis groups need whitespace separation"
            );
            (inner[..end].split_whitespace().collect::<Vec<_>>(), rest)
        } else {
            let end = remaining
                .find(char::is_whitespace)
                .unwrap_or(remaining.len());
            (vec![&remaining[..end]], &remaining[end..])
        };
        ensure!(!names.is_empty(), "empty axis group");
        let mut group = Vec::new();
        for name in names {
            ensure!(identifier(name), "invalid pattern axis {name:?}");
            ensure!(
                seen.insert(name.to_owned()),
                "duplicate pattern axis {name}"
            );
            group.push(name.to_owned());
        }
        groups.push(group);
        remaining = rest.trim_start();
    }
    ensure!(
        !groups.is_empty(),
        "pattern must retain a nonempty sample shape"
    );
    Ok(groups)
}

fn product(mut values: impl Iterator<Item = u64>) -> Result<u64> {
    values
        .try_fold(1u64, |n, size| n.checked_mul(size))
        .context("transform shape size overflow")
}

/// Validate the artifact independently of the Python compiler. Pattern axes
/// describe sample dimensions; the optional runtime batch axis is not included.
pub(crate) fn infer(input: &TensorType, op: &str, args: &Value) -> Result<TensorType> {
    let pattern = args["pattern"]
        .as_str()
        .context("missing transform pattern")?;
    let (left, right) = pattern.split_once("->").context("pattern needs one ->")?;
    ensure!(!right.contains("->"), "pattern needs exactly one ->");
    let left = side(left)?;
    let right = side(right)?;
    ensure!(
        left.len() == input.shape.len(),
        "pattern input rank mismatch"
    );
    for (group, name) in left.iter().zip(&input.axes) {
        if group.len() == 1 {
            ensure!(
                name.as_ref().is_none_or(|name| name == &group[0]),
                "pattern axis conflicts with named input axis"
            );
        }
    }
    let inputs: BTreeSet<_> = left.iter().flatten().cloned().collect();
    let outputs: BTreeSet<_> = right.iter().flatten().cloned().collect();
    ensure!(
        outputs.is_subset(&inputs),
        "pattern introduces an unknown output axis"
    );
    if op == "rearrange" {
        ensure!(inputs == outputs, "rearrange must preserve every axis");
    } else {
        ensure!(
            matches!(args["reduction"].as_str(), Some("mean" | "sum")),
            "reduce requires mean or sum"
        );
        ensure!(input.dtype == "f32", "reduce requires f32");
    }
    let sizes = args["sizes"]
        .as_object()
        .context("transform sizes must be an object")?;
    let mut resolved = BTreeMap::new();
    for (name, value) in sizes {
        ensure!(inputs.contains(name), "unused axis size {name}");
        resolved.insert(name.clone(), dimension(value)?);
    }
    for (group, &size) in left.iter().zip(&input.shape) {
        let unknown: Vec<_> = group
            .iter()
            .filter(|name| !resolved.contains_key(*name))
            .collect();
        ensure!(unknown.len() <= 1, "ambiguous split: supply axis sizes");
        let known = product(group.iter().filter_map(|name| resolved.get(name).copied()))?;
        if let Some(name) = unknown.first() {
            ensure!(
                size % known == 0,
                "axis sizes do not divide input dimension"
            );
            let inferred = size / known;
            ensure!(inferred > 0, "axis size must be positive");
            resolved.insert((*name).clone(), inferred);
        } else {
            ensure!(known == size, "axis sizes disagree with input dimension");
        }
    }
    // Check total element arithmetic even when every individual dimension fits.
    product(input.shape.iter().copied())?;
    let shape = right
        .iter()
        .map(|group| product(group.iter().map(|name| resolved[name])))
        .collect::<Result<Vec<_>>>()?;
    let axes = right
        .iter()
        .map(|group| (group.len() == 1).then(|| group[0].clone()))
        .collect();
    Ok(TensorType {
        dtype: input.dtype.clone(),
        shape,
        axes,
    })
}
