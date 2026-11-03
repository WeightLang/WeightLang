use crate::artifact::{dimension, string};
use crate::manifest::identifier;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorType {
    pub dtype: String,
    pub shape: Vec<u64>,
    #[serde(skip_serializing_if = "unnamed")]
    pub axes: Vec<Option<String>>,
}
fn unnamed(axes: &[Option<String>]) -> bool {
    axes.iter().all(Option::is_none)
}
pub type Contracts = BTreeMap<String, (TensorType, TensorType)>;
impl TensorType {
    pub fn parse(v: &Value) -> Result<Self> {
        let dtype = string(v, "dtype")?.to_owned();
        ensure!(
            dtype == "f32" || dtype == "i64",
            "unsupported interface dtype"
        );
        let shape = v["shape"]
            .as_array()
            .context("missing shape")?
            .iter()
            .map(dimension)
            .collect::<Result<Vec<_>>>()?;
        ensure!(!shape.is_empty(), "sample shape cannot be empty");
        let axes = if let Some(value) = v.get("axes") {
            let names = value.as_array().context("axes must be an array")?;
            ensure!(names.len() == shape.len(), "axes rank mismatch");
            let mut seen = std::collections::BTreeSet::new();
            names
                .iter()
                .map(|name| {
                    if name.is_null() {
                        return Ok(None);
                    }
                    let name = name.as_str().context("axis must be a name or null")?;
                    ensure!(
                        identifier(name) && seen.insert(name),
                        "invalid or duplicate axis name"
                    );
                    Ok(Some(name.to_owned()))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; shape.len()]
        };
        Ok(Self { dtype, shape, axes })
    }
    pub fn compatible(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.shape == other.shape
            && self.axes.len() == self.shape.len()
            && other.axes.len() == other.shape.len()
            && self
                .axes
                .iter()
                .zip(&other.axes)
                .all(|(a, b)| a.is_none() || b.is_none() || a == b)
    }
    fn refine(&mut self, other: &Self) -> Result<()> {
        ensure!(self.compatible(other), "shape/dtype/axis mismatch");
        for (a, b) in self.axes.iter_mut().zip(&other.axes) {
            if a.is_none() {
                *a = b.clone();
            }
        }
        let names: Vec<_> = self.axes.iter().flatten().collect();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        ensure!(unique.len() == names.len(), "duplicate inferred axis name");
        Ok(())
    }
    pub fn label(&self) -> String {
        format!(
            "{}[{}]",
            self.dtype,
            self.shape
                .iter()
                .zip(&self.axes)
                .map(|(size, name)| match name {
                    Some(name) => format!("{name}={size}"),
                    None => size.to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        )
    }
    pub fn width(&self) -> u64 {
        *self.shape.last().unwrap()
    }
}
pub fn infer(
    layers: &Value,
    prefix: &str,
    typ: TensorType,
    tensors: &mut BTreeMap<String, Vec<u64>>,
    contracts: &mut Contracts,
) -> Result<TensorType> {
    let output = infer_sequence(layers, prefix, typ, tensors, contracts)?;
    validate_records(layers, prefix, contracts)?;
    Ok(output)
}
fn validate_records(layers: &Value, prefix: &str, contracts: &Contracts) -> Result<()> {
    for layer in layers.as_array().context("layers must be an array")? {
        let name = string(layer, "name")?;
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        let (input, output) = contracts.get(&path).context("missing layer contract")?;
        for (key, inferred) in [("input", input), ("output", output)] {
            if let Some(recorded) = layer.get(key) {
                ensure!(
                    recorded == &serde_json::to_value(inferred)?,
                    "recorded {key} contract mismatch at {path}"
                );
            }
        }
        if matches!(layer["op"].as_str(), Some("seq" | "residual")) {
            validate_records(&layer["layers"], &path, contracts)?;
        }
    }
    Ok(())
}
fn infer_sequence(
    layers: &Value,
    prefix: &str,
    mut typ: TensorType,
    tensors: &mut BTreeMap<String, Vec<u64>>,
    contracts: &mut Contracts,
) -> Result<TensorType> {
    let mut names = std::collections::BTreeSet::new();
    for l in layers.as_array().context("layers must be an array")? {
        let name = string(l, "name")?;
        ensure!(
            identifier(name) && names.insert(name),
            "invalid or duplicate layer name"
        );
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        let before = typ.clone();
        let op = string(l, "op")?;
        let a = &l["args"];
        let arg = |key: &str| dimension(&a[key]);
        if op == "residual" {
            typ = infer_sequence(&l["layers"], &path, typ, tensors, contracts)?;
            let mut entry = before.clone();
            entry
                .refine(&typ)
                .context("residual must preserve shape, dtype and axes")?;
            if entry != before {
                typ = infer_sequence(&l["layers"], &path, entry.clone(), tensors, contracts)?;
                ensure!(
                    typ.compatible(&entry),
                    "residual body conflicts with refined input axes"
                );
            }
        } else if op == "seq" {
            typ = infer_sequence(&l["layers"], &path, typ, tensors, contracts)?;
        } else if op == "embedding" {
            ensure!(
                typ.dtype == "i64" && typ.shape.len() == 1,
                "embedding expects i64[sequence]"
            );
            let dim = arg("dim")?;
            tensors.insert(format!("{path}.weight"), vec![arg("vocab")?, dim]);
            typ.dtype = "f32".into();
            typ.shape.push(dim);
            typ.axes.push(None);
        } else if matches!(op, "rearrange" | "reduce") {
            typ = crate::transforms::infer(&typ, op, a)
                .with_context(|| format!("invalid {op} at {path}"))?;
        } else {
            ensure!(typ.dtype == "f32", "{op} requires f32");
            let width = typ.width();
            match op {
                "linear" => {
                    ensure!(arg("in")? == width, "linear input width mismatch");
                    let n = arg("out")?;
                    tensors.insert(format!("{path}.weight"), vec![n, width]);
                    tensors.insert(format!("{path}.bias"), vec![n]);
                    *typ.shape.last_mut().unwrap() = n;
                }
                "layernorm" => {
                    ensure!(arg("dim")? == width, "layernorm width mismatch");
                    tensors.insert(format!("{path}.weight"), vec![width]);
                    tensors.insert(format!("{path}.bias"), vec![width]);
                }
                "position_embedding" | "self_attention" => {
                    ensure!(
                        typ.shape.len() == 2 && width == arg("dim")?,
                        "{op} requires [sequence,dim]"
                    );
                    if op == "position_embedding" {
                        let max_len = arg("max_len")?;
                        ensure!(typ.shape[0] <= max_len, "sequence exceeds max_len");
                        tensors.insert(format!("{path}.weight"), vec![max_len, width]);
                    } else {
                        let heads = arg("heads")?;
                        ensure!(
                            width.is_multiple_of(heads)
                                && matches!(a["causal"].as_u64(), Some(0 | 1)),
                            "invalid heads or causal mask"
                        );
                        let triple = width.checked_mul(3).context("attention size overflow")?;
                        tensors.insert(format!("{path}.qkv.weight"), vec![triple, width]);
                        tensors.insert(format!("{path}.qkv.bias"), vec![triple]);
                        tensors.insert(format!("{path}.proj.weight"), vec![width, width]);
                        tensors.insert(format!("{path}.proj.bias"), vec![width]);
                    }
                }
                "meanpool" => {
                    ensure!(typ.shape.len() == 2, "meanpool expects [sequence,dim]");
                    typ.shape = vec![width];
                    typ.axes = vec![typ.axes.last().cloned().unwrap()];
                }
                "flatten" => {
                    typ.shape = vec![typ
                        .shape
                        .iter()
                        .try_fold(1u64, |a, b| a.checked_mul(*b))
                        .context("shape size overflow")?];
                    typ.axes = vec![None];
                }
                "conv2d" | "maxpool2d" | "globalavgpool2d" => {
                    ensure!(typ.shape.len() == 3, "{op} expects [channels,height,width]");
                    let channels = typ.shape[0];
                    if op == "globalavgpool2d" {
                        typ.shape = vec![channels];
                        typ.axes = vec![typ.axes[0].clone()];
                    } else {
                        let kernel = arg("kernel")?;
                        let stride = arg("stride")?;
                        let padding = if op == "conv2d" {
                            a["padding"].as_u64().context("invalid padding")?
                        } else {
                            0
                        };
                        for v in &mut typ.shape[1..] {
                            *v = v
                                .checked_add(padding.checked_mul(2).context("padding overflow")?)
                                .and_then(|n| n.checked_sub(kernel))
                                .context("kernel produces empty spatial dimension")?
                                / stride
                                + 1;
                        }
                        if op == "conv2d" {
                            ensure!(arg("in")? == channels, "conv2d channels mismatch");
                            let n = arg("out")?;
                            tensors.insert(
                                format!("{path}.weight"),
                                vec![n, channels, kernel, kernel],
                            );
                            tensors.insert(format!("{path}.bias"), vec![n]);
                            typ.shape[0] = n;
                        }
                    }
                }
                "dropout" => {
                    let p = a["p"].as_f64().context("dropout p")?;
                    ensure!((0.0..1.0).contains(&p), "invalid dropout p");
                }
                "relu" | "tanh" | "sigmoid" | "gelu" | "silu" | "softmax" | "identity" => {}
                _ => anyhow::bail!("unsupported op {op:?}"),
            }
        }
        contracts.insert(path, (before, typ.clone()));
    }
    Ok(typ)
}
