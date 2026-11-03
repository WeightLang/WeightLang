use crate::manifest::{identifier, relative_file};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub const FILES: [&str; 3] = ["manifest.json", "weights.safetensors", "source.wgt"];
pub fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .with_context(|| format!("missing string {key}"))
}
pub fn dimension(v: &Value) -> Result<u64> {
    let n = v.as_u64().context("dimension must be an integer")?;
    ensure!(n > 0, "dimension must be positive");
    Ok(n)
}
pub fn expected(
    layers: &Value,
    prefix: &str,
    width: u64,
    out: &mut BTreeMap<String, Vec<u64>>,
) -> Result<u64> {
    let mut w = width;
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
        match string(l, "op")? {
            "linear" => {
                ensure!(
                    dimension(&l["args"]["in"])? == w,
                    "linear input width mismatch at {path}"
                );
                let n = dimension(&l["args"]["out"])?;
                out.insert(format!("{path}.weight"), vec![n, w]);
                out.insert(format!("{path}.bias"), vec![n]);
                w = n;
            }
            "layernorm" => {
                ensure!(
                    dimension(&l["args"]["dim"])? == w,
                    "layernorm width mismatch"
                );
                out.insert(format!("{path}.weight"), vec![w]);
                out.insert(format!("{path}.bias"), vec![w]);
            }
            "seq" | "residual" => {
                let n = expected(&l["layers"], &path, w, out)?;
                ensure!(l["op"] != "residual" || n == w, "residual width mismatch");
                w = n;
            }
            "dropout" => {
                let p = l["args"]["p"].as_f64().context("missing dropout p")?;
                ensure!((0.0..1.0).contains(&p), "invalid dropout p");
            }
            "relu" | "tanh" | "sigmoid" | "gelu" | "silu" | "softmax" | "identity" => {}
            op => anyhow::bail!("unsupported op {op:?}"),
        }
    }
    Ok(w)
}
pub struct Artifact {
    pub root: PathBuf,
    pub manifest: Value,
    pub shapes: BTreeMap<String, Vec<u64>>,
}
impl Artifact {
    pub fn read(path: &Path) -> Result<Self> {
        let root = if path.is_file() {
            path.parent().context("no artifact parent")?
        } else {
            path
        };
        let m: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        ensure!(
            m["format"] == "weightlang.artifact"
                && matches!(m["format_version"].as_u64(), Some(1 | 2)),
            "unsupported artifact format"
        );
        ensure!(identifier(string(&m, "model")?), "invalid model name");
        for (kind, file) in [("weights", "weights.safetensors"), ("source", "source.wgt")] {
            let name = string(&m[kind], "file")?;
            relative_file(name)?;
            ensure!(name == file, "expected {file}");
            ensure!(
                hash_file(&root.join(name))? == string(&m[kind], "sha256")?,
                "{kind} sha256 mismatch"
            );
        }
        let mut shapes = BTreeMap::new();
        for side in ["input", "output"] {
            ensure!(
                m["interface"][side]["dtype"] == "f32",
                "expected f32 interface"
            );
            ensure!(
                m["interface"][side]["shape"]
                    .as_array()
                    .is_some_and(|a| a.len() == 1),
                "expected 1-D interface"
            );
        }
        let width = expected(
            &m["layers"],
            "",
            dimension(&m["interface"]["input"]["shape"][0])?,
            &mut shapes,
        )?;
        ensure!(
            width == dimension(&m["interface"]["output"]["shape"][0])?,
            "output width mismatch"
        );
        let tensors = m["weights"]["tensors"]
            .as_object()
            .context("missing tensor metadata")?;
        ensure!(shapes.len() == tensors.len(), "tensor table mismatch");
        let mut file = fs::File::open(root.join("weights.safetensors"))?;
        let size = file.metadata()?.len();
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        ensure!(
            length <= 16 * 1024 * 1024 && length <= size.saturating_sub(8),
            "invalid safetensors header length"
        );
        let mut header = vec![0u8; length as usize];
        file.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header)?;
        let header = header.as_object().context("invalid safetensors header")?;
        ensure!(
            header
                .keys()
                .filter(|k| k.as_str() != "__metadata__")
                .count()
                == shapes.len(),
            "safetensors tensor table mismatch"
        );
        let mut ranges = Vec::new();
        for (name, shape) in &shapes {
            let t = header.get(name).context("missing tensor")?;
            ensure!(
                t["dtype"] == "F32" && t["shape"] == serde_json::to_value(shape)?,
                "tensor shape/dtype mismatch: {name}"
            );
            let metadata = tensors.get(name).context("missing tensor provenance")?;
            ensure!(
                matches!(
                    metadata["origin"].as_str(),
                    Some("trained" | "random-init" | "imported" | "imported+trained")
                ),
                "unknown origin"
            );
            let offsets = t["data_offsets"]
                .as_array()
                .context("missing tensor offsets")?;
            ensure!(offsets.len() == 2, "invalid offsets");
            let start = offsets[0].as_u64().context("invalid start offset")?;
            let end = offsets[1].as_u64().context("invalid end offset")?;
            let bytes = shape
                .iter()
                .try_fold(4u64, |a, b| a.checked_mul(*b))
                .context("tensor size overflow")?;
            ensure!(
                end.checked_sub(start) == Some(bytes) && end <= size - 8 - length,
                "invalid tensor range"
            );
            ranges.push((start, end));
        }
        ranges.sort();
        let mut end = 0;
        for (start, next) in ranges {
            ensure!(start == end, "safetensors gap or overlap");
            end = next;
        }
        ensure!(end == size - 8 - length, "trailing safetensors data");
        file.seek(SeekFrom::Start(8 + length))?;
        Ok(Self {
            root: root.to_owned(),
            manifest: m,
            shapes,
        })
    }
    pub fn untrained(&self) -> Vec<String> {
        self.manifest["weights"]["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, t)| {
                t["origin"] == "random-init"
                    || (t["origin"] == "imported" && t["upstream_origin"] == "random-init")
            })
            .map(|(n, _)| n.clone())
            .collect()
    }
}
