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
        Ok(Self { dtype, shape })
    }
    pub fn label(&self) -> String {
        format!(
            "{}[{}]",
            self.dtype,
            self.shape
                .iter()
                .map(u64::to_string)
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
        if op == "seq" || op == "residual" {
            typ = infer(&l["layers"], &path, typ, tensors, contracts)?;
            ensure!(
                op != "residual" || typ == before,
                "residual shape/dtype mismatch"
            );
        } else if op == "embedding" {
            ensure!(
                typ.dtype == "i64" && typ.shape.len() == 1,
                "embedding expects i64[sequence]"
            );
            let dim = arg("dim")?;
            tensors.insert(format!("{path}.weight"), vec![arg("vocab")?, dim]);
            typ.dtype = "f32".into();
            typ.shape.push(dim);
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
                }
                "flatten" => {
                    typ.shape = vec![typ
                        .shape
                        .iter()
                        .try_fold(1u64, |a, b| a.checked_mul(*b))
                        .context("shape size overflow")?];
                }
                "conv2d" | "maxpool2d" | "globalavgpool2d" => {
                    ensure!(typ.shape.len() == 3, "{op} expects [channels,height,width]");
                    let channels = typ.shape[0];
                    if op == "globalavgpool2d" {
                        typ.shape = vec![channels];
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
        if l.get("input").is_some() {
            ensure!(
                TensorType::parse(&l["input"])? == before,
                "recorded input contract mismatch at {path}"
            );
        }
        if l.get("output").is_some() {
            ensure!(
                TensorType::parse(&l["output"])? == typ,
                "recorded output contract mismatch at {path}"
            );
        }
        contracts.insert(path, (before, typ.clone()));
    }
    Ok(typ)
}
