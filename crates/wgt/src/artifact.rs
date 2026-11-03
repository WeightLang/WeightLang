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
pub struct Artifact {
    pub root: PathBuf,
    pub manifest: Value,
    pub shapes: BTreeMap<String, Vec<u64>>,
    pub contracts: crate::contracts::Contracts,
}
impl Artifact {
    pub fn read(path: &Path) -> Result<Self> {
        let root = if path.is_file() {
            path.parent().context("no artifact parent")?
        } else {
            path
        };
        let manifest_path = root.join("manifest.json");
        let bytes = fs::read(&manifest_path).with_context(|| {
            format!(
                "cannot read artifact manifest {}; artifact paths are relative to the current directory (or -C). Check the path or build the model first",
                manifest_path.display()
            )
        })?;
        let m: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid artifact manifest {}", manifest_path.display()))?;
        ensure!(
            m["format"] == "weightlang.artifact"
                && matches!(m["format_version"].as_u64(), Some(1..=8)),
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
        if m["format_version"] == 5 || (m["format_version"].as_u64().unwrap() >= 6 && m["interface"].get("dimensions").is_some()) {
            crate::symbolic::validate(&m["layers"], &m["interface"])?;
        } else {
            ensure!(m["interface"].get("dimensions").is_none() && m["interface"].get("constraints").is_none(), "symbolic metadata requires v5");
            for key in ["inputs", "outputs"] {
                if let Some(ports) = m["interface"][key].as_array() {
                    ensure!(ports.iter().all(|p| p.get("shape_expr").is_none()), "symbolic metadata requires v5");
                }
            }
        }
        let mut shapes = BTreeMap::new();
        let input = crate::contracts::TensorType::parse(&m["interface"]["input"])?;
        let output = crate::contracts::TensorType::parse(&m["interface"]["output"])?;
        if m["format_version"].as_u64().unwrap() < 4 {
            ensure!(!input.shape.is_empty() && !output.shape.is_empty() && input.shape.iter().chain(output.shape.iter()).all(|n| *n > 0), "scalar/empty ports require v4 or later");
        }
        if m["format_version"].as_u64().unwrap() < 3 {
            ensure!(
                input.dtype == "f32"
                    && output.dtype == "f32"
                    && input.shape.len() == 1
                    && output.shape.len() == 1,
                "v1/v2 artifacts require 1-D f32 interfaces"
            );
        }
        let mut contracts = crate::contracts::Contracts::new();
        if m["format_version"].as_u64().unwrap() >= 4 {
            crate::contracts::infer_named(&m["layers"], &m["interface"], &mut shapes, &mut contracts)?;
        } else {
            let inferred = crate::contracts::infer(&m["layers"], "", input, &mut shapes, &mut contracts)?;
            ensure!(inferred.compatible(&output), "output shape/dtype/axis mismatch");
        }
        if m["format_version"].as_u64().unwrap() >= 6 {
            crate::parameter_identity::adapters(&m["interface"], &m["layers"], &mut shapes)?;
        } else {
            ensure!(m["interface"].get("adapters").is_none(), "adapters require v6");
        }
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
        let mut aliases = if m["format_version"].as_u64().unwrap() >= 6 {
            crate::parameter_identity::validate(&m["interface"], &shapes)?
        } else {
            ensure!(m["interface"].get("parameter_ids").is_none() && m["interface"].get("parameter_shares").is_none(), "parameter identities require v6");
            Vec::new()
        };
        if m["format_version"] == 7 || (m["format_version"] == 8 && m["interface"].get("weight_types").is_some()) {
            aliases = crate::weight_types::parameter_aliases(&m["layers"], &shapes, &aliases)?;
            crate::weight_types::validate(&m["interface"], &shapes, &aliases)?;
        } else {
            ensure!(m["interface"].get("weight_types").is_none() && m["interface"].get("weight_assignments").is_none(), "weight types require v7");
        }
        if m["format_version"] == 8 {
            let precision_aliases = crate::weight_types::parameter_aliases(&m["layers"], &shapes, &aliases)?;
            crate::precision::validate(&m["interface"], &m["layers"], &precision_aliases)?;
        } else {
            ensure!(m["interface"].get("precision").is_none(), "precision requires v8");
        }
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
                    Some("trained" | "random-init" | "imported" | "imported+trained" | "edited")
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
        for (destination,owner) in aliases {
            let d = header[&destination]["data_offsets"][0].as_u64().context("invalid shared offset")?;
            let o = header[&owner]["data_offsets"][0].as_u64().context("invalid owner offset")?;
            let end = header[&owner]["data_offsets"][1].as_u64().context("invalid owner end")?;
            let mut checked = 0;
            let mut left = [0u8;8192];
            let mut right = [0u8;8192];
            while checked < end-o {
                let size = (end-o-checked).min(8192) as usize;
                file.seek(SeekFrom::Start(8+length+d+checked))?;
                file.read_exact(&mut left[..size])?;
                file.seek(SeekFrom::Start(8+length+o+checked))?;
                file.read_exact(&mut right[..size])?;
                ensure!(left[..size] == right[..size], "conflicting shared weights");
                checked += size as u64;
            }
        }
        file.seek(SeekFrom::Start(8 + length))?;
        Ok(Self {
            root: root.to_owned(),
            manifest: m,
            shapes,
            contracts,
        })
    }
    pub fn untrained(&self) -> Vec<String> {
        self.manifest["weights"]["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, t)| {
                matches!(t["origin"].as_str(), Some("random-init" | "edited"))
                    || (t["origin"] == "imported"
                        && matches!(
                            t["upstream_origin"].as_str(),
                            Some("random-init" | "edited")
                        ))
            })
            .map(|(n, _)| n.clone())
            .collect()
    }
}
