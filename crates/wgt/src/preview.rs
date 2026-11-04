// Bounded parameter previews. Rendering and tensor axes are architecture-independent.
use crate::artifact::Artifact;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct NamedView {
    pub label: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Serialize)]
pub struct Overview {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}
#[derive(Serialize)]
pub struct TensorData {
    pub name: String,
    pub owner: String,
    pub shape: Vec<u64>,
    pub elements: usize,
    pub origin: String,
    pub values: Option<Vec<f32>>,
    pub baseline_values: Option<Vec<f32>>,
    pub overview: Overview,
    pub baseline_overview: Option<Overview>,
    pub delta_overview: Option<Overview>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub baseline_min: Option<f64>,
    pub baseline_max: Option<f64>,
    pub changed_elements: Option<u64>,
    pub max_abs_delta: Option<f64>,
    pub views: Vec<NamedView>,
}
fn signature(layers: &Value) -> Value {
    Value::Array(
        layers
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                json!({
                    "name":l["name"],"op":l["op"],"args":l.get("args").unwrap_or(&json!({})),
                    "layers":l.get("layers").map(signature)
                })
            })
            .collect(),
    )
}
struct Reader {
    file: BufReader<File>,
    header: Value,
    start: u64,
}
impl Reader {
    fn open(artifact: &Artifact) -> Result<Self> {
        let mut file = BufReader::new(File::open(artifact.root.join("weights.safetensors"))?);
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)?;
        let length = u64::from_le_bytes(bytes);
        ensure!(length <= 16 * 1024 * 1024, "oversize safetensors header");
        let mut header = vec![0; usize::try_from(length)?];
        file.read_exact(&mut header)?;
        Ok(Self {
            file,
            header: serde_json::from_slice(&header)?,
            start: 8 + length,
        })
    }
    fn tensor(&mut self, name: &str) -> Result<()> {
        let offset = self.header[name]["data_offsets"][0]
            .as_u64()
            .context("missing tensor offset")?;
        self.file.seek(SeekFrom::Start(self.start + offset))?;
        Ok(())
    }
    fn scalar(&mut self) -> Result<f32> {
        let mut bytes = [0; 4];
        self.file.read_exact(&mut bytes)?;
        let value = f32::from_le_bytes(bytes);
        ensure!(value.is_finite(), "non-finite parameter in visualization");
        Ok(value)
    }
}
pub fn read(artifact: &Artifact, baseline: Option<&Artifact>) -> Result<Vec<TensorData>> {
    if let Some(base) = baseline {
        ensure!(
            artifact.shapes == base.shapes
                && signature(&artifact.manifest["layers"]) == signature(&base.manifest["layers"]),
            "incompatible comparison architecture"
        );
        for side in ["input", "output"] {
            ensure!(
                artifact.manifest["interface"][side]["shape"]
                    == base.manifest["interface"][side]["shape"]
                    && artifact.manifest["interface"][side]["dtype"]
                        == base.manifest["interface"][side]["dtype"],
                "incompatible comparison interface"
            );
        }
    }
    let mut reader = Reader::open(artifact)?;
    let mut previous = baseline.map(Reader::open).transpose()?;
    // Limit embedded exact scalars globally, independently of tensor count.
    let mut budget = 1_000_000usize;
    let grid = ((1_000_000usize / artifact.shapes.len().max(1)) as f64)
        .sqrt()
        .floor()
        .clamp(1., 64.) as usize;
    let mut result = Vec::new();
    for (name, shape) in &artifact.shapes {
        reader.tensor(name)?;
        if let Some(r) = &mut previous {
            r.tensor(name)?;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(usize::try_from(*d).ok()?))
            .context("tensor count overflow")?;
        let rows = if shape.len() < 2 {
            1
        } else {
            usize::try_from(shape[0])?
        };
        let cols = count / rows;
        let rs = rows.div_ceil(grid);
        let cs = cols.div_ceil(grid);
        let nr = rows.div_ceil(rs);
        let nc = cols.div_ceil(cs);
        let exact = count <= 65_536 && count <= budget;
        if exact {
            budget -= count;
        }
        let mut values = exact.then(|| Vec::with_capacity(count));
        let mut old_values = (exact && baseline.is_some()).then(|| Vec::with_capacity(count));
        let mut sums = vec![0.; nr * nc];
        let mut old_sums = baseline.map(|_| vec![0.; nr * nc]);
        let mut counts = vec![0usize; nr * nc];
        let (mut min, mut max, mut total) = (f64::INFINITY, f64::NEG_INFINITY, 0.);
        let (mut old_min, mut old_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut changed, mut max_delta) = (0u64, 0f64);
        for index in 0..count {
            let value = reader.scalar()?;
            let v = f64::from(value);
            let tile = (index / cols / rs) * nc + index % cols / cs;
            sums[tile] += v;
            counts[tile] += 1;
            total += v;
            min = min.min(v);
            max = max.max(v);
            if let Some(values) = &mut values {
                values.push(value);
            }
            if let (Some(r), Some(old_sums)) = (&mut previous, &mut old_sums) {
                let old = r.scalar()?;
                let b = f64::from(old);
                old_sums[tile] += b;
                old_min = old_min.min(b);
                old_max = old_max.max(b);
                if value != old {
                    changed += 1;
                }
                max_delta = max_delta.max((v - b).abs());
                if let Some(values) = &mut old_values {
                    values.push(old);
                }
            }
        }
        for i in 0..sums.len() {
            sums[i] /= counts[i] as f64;
            if let Some(old_sums) = &mut old_sums {
                old_sums[i] /= counts[i] as f64;
            }
        }
        let deltas = old_sums
            .as_ref()
            .map(|old| sums.iter().zip(old).map(|(a, b)| a - b).collect());
        let metadata = &artifact.manifest["weights"]["tensors"][name];
        let views: Vec<NamedView> = metadata
            .get("views")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_default();
        for view in &views {
            ensure!(
                shape.len() == 2 && view.start < view.end && view.end <= rows,
                "invalid named tensor view"
            );
        }
        result.push(TensorData {
            name: name.clone(),
            owner: String::new(),
            shape: shape.clone(),
            elements: count,
            origin: metadata["origin"].as_str().unwrap().into(),
            values,
            baseline_values: old_values,
            overview: Overview {
                rows: nr,
                cols: nc,
                values: sums,
            },
            baseline_overview: old_sums.map(|values| Overview {
                rows: nr,
                cols: nc,
                values,
            }),
            delta_overview: deltas.map(|values| Overview {
                rows: nr,
                cols: nc,
                values,
            }),
            min,
            max,
            mean: total / count as f64,
            baseline_min: baseline.map(|_| old_min),
            baseline_max: baseline.map(|_| old_max),
            changed_elements: baseline.map(|_| changed),
            max_abs_delta: baseline.map(|_| max_delta),
            views,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture(values: &[f32]) -> (tempfile::TempDir, Artifact) {
        std::fs::create_dir_all("/tmp/weightlang").unwrap();
        let directory = tempfile::tempdir_in("/tmp/weightlang").unwrap();
        let header = serde_json::to_vec(&json!({
            "w": {"dtype": "F32", "shape": [1, values.len()],
                  "data_offsets": [0, values.len() * 4]}
        }))
        .unwrap();
        let mut file = File::create(directory.path().join("weights.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let artifact = Artifact {
            root: directory.path().to_path_buf(),
            manifest: json!({"layers": [], "weights": {"tensors": {
                "w": {"origin": "test", "views": [{"label": "row", "start": 0, "end": 1}]}
            }}}),
            shapes: [("w".into(), vec![1, values.len() as u64])].into(),
            contracts: Default::default(),
        };
        (directory, artifact)
    }

    #[test]
    fn previews_preserve_tiled_statistics_with_and_without_baseline() {
        // 65 columns create 33 tiles, with only one element in the final tile.
        let values: Vec<_> = (0..65).map(|i| i as f32).collect();
        let baseline: Vec<_> = values.iter().map(|v| v - 2.).collect();
        let (_current_dir, current) = fixture(&values);
        let (_baseline_dir, baseline) = fixture(&baseline);
        let plain = read(&current, None).unwrap().remove(0);
        let compared = read(&current, Some(&baseline)).unwrap().remove(0);
        assert_eq!(plain.values.as_ref().unwrap(), &values);
        assert_eq!(plain.overview.rows, 1);
        assert_eq!(plain.overview.cols, 33);
        assert_eq!(plain.overview.values[0], 0.5);
        assert_eq!(plain.overview.values[32], 64.);
        assert_eq!((plain.min, plain.max, plain.mean), (0., 64., 32.));
        assert_eq!(plain.views[0].label, "row");
        assert!(plain.baseline_values.is_none());
        assert!(plain.baseline_overview.is_none());
        assert!(plain.delta_overview.is_none());
        assert!(plain.baseline_min.is_none());
        assert!(plain.baseline_max.is_none());
        assert!(plain.changed_elements.is_none());
        assert!(plain.max_abs_delta.is_none());
        assert_eq!(compared.values, plain.values);
        assert_eq!(compared.overview.values, plain.overview.values);
        assert_eq!(compared.baseline_values.unwrap()[64], 62.);
        let old = compared.baseline_overview.unwrap();
        assert_eq!((old.rows, old.cols), (1, 33));
        assert_eq!((old.values[0], old.values[32]), (-1.5, 62.));
        assert_eq!(compared.delta_overview.unwrap().values, vec![2.; 33]);
        assert_eq!(compared.baseline_min, Some(-2.));
        assert_eq!(compared.baseline_max, Some(62.));
        assert_eq!(compared.changed_elements, Some(65));
        assert_eq!(compared.max_abs_delta, Some(2.));
    }
}
