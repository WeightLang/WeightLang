//! Standard safetensors weights, independent of model graphs and optimizer state.
use anyhow::{ensure, Context, Result};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSchema {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Inspect a bounded, finite F32 artifact without running or constructing a model.
pub fn inspect(bytes: &[u8], limit: usize) -> Result<Vec<TensorSchema>> {
    ensure!(bytes.len() <= limit, "weights byte limit exceeded");
    let tensors = SafeTensors::deserialize(bytes)?;
    let mut schema = Vec::new();
    for (name, tensor) in tensors.tensors() {
        ensure!(tensor.dtype() == Dtype::F32, "weights dtype must be F32: {name}");
        ensure!(tensor.data().chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()), "nonfinite weights: {name}");
        schema.push(TensorSchema { name, shape: tensor.shape().to_vec(), dtype: "F32".into() });
    }
    schema.sort_by(|a,b| a.name.cmp(&b.name));
    Ok(schema)
}

pub fn encode(parameters: &[(String, Vec<usize>, Vec<f32>)], limit: usize) -> Result<Vec<u8>> {
    let mut data = BTreeMap::new();
    let mut total = 0usize;
    for (name, shape, values) in parameters {
        ensure!(values.iter().all(|n| n.is_finite()), "nonfinite weights: {name}");
        let count = shape.iter().try_fold(1usize, |a,b| a.checked_mul(*b)).context("weights shape overflow")?;
        ensure!(count == values.len(), "weights shape mismatch: {name}");
        total = total.checked_add(values.len().checked_mul(4).context("weights size overflow")?).context("weights size overflow")?;
        ensure!(total <= limit, "weights byte limit exceeded");
        ensure!(data.insert(name.as_str(), values.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>()).is_none(), "duplicate weight name: {name}");
    }
    let views = parameters.iter().map(|(name, shape, _)| {
        Ok((name.as_str(), TensorView::new(Dtype::F32, shape.clone(), &data[name.as_str()])?))
    }).collect::<Result<Vec<_>>>()?;
    let bytes = safetensors::serialize(views, None)?;
    ensure!(bytes.len() <= limit, "weights byte limit exceeded");
    Ok(bytes)
}

/// Exact keys and dimensions; no renaming, transposition, or dtype conversion.
pub fn decode(schema: &[(String, Vec<usize>)], bytes: &[u8], limit: usize) -> Result<Vec<Vec<f32>>> {
    let actual = inspect(bytes, limit)?;
    ensure!(actual.len() == schema.len(), "weights parameter key set mismatch");
    let tensors = SafeTensors::deserialize(bytes)?;
    let mut seen = std::collections::BTreeSet::new();
    schema.iter().map(|(name, shape)| {
        ensure!(seen.insert(name), "duplicate parameter name: {name}");
        let tensor = tensors.tensor(name).with_context(|| format!("missing weight: {name}"))?;
        ensure!(tensor.shape() == shape, "weights shape mismatch: {name}");
        Ok(tensor.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }).collect()
}
