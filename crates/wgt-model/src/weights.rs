//! Immutable F32 snapshots and explicit, complete parameter binding plans.
//!
//! Host applications own file access and transformations. No operation here
//! creates a service instance, changes a source snapshot or executes forward.
use crate::{ModelDefinition, ParameterSchema};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};
use wgt_language::weight_artifact::{self, TensorSchema};

pub const FORMAT: &str = "weightlang.weights.v1";
pub const MAX_BYTES: usize = 64 * 1024 * 1024;

/// Owned, row-major F32 values. Construction and transformations reject nonfinite values.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        ensure!(
            !shape.is_empty() && shape.len() <= 8 && shape.iter().all(|&d| d > 0 && d <= 16384),
            "invalid weight shape"
        );
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .context("weight shape overflow")?;
        ensure!(
            count <= MAX_BYTES / 4 && count == data.len(),
            "weight element count mismatch or budget exceeded"
        );
        ensure!(data.iter().all(|x| x.is_finite()), "nonfinite weight value");
        Ok(Self { shape, data })
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn values(&self) -> &[f32] {
        &self.data
    }
    pub fn scaled(&self, scale: f32) -> Result<Self> {
        ensure!(scale.is_finite(), "nonfinite weight scale");
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|x| x * scale).collect(),
        )
    }
    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure!(
            self.shape == other.shape,
            "weight addition requires identical shapes"
        );
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
    /// `(1 - alpha) * self + alpha * other`; alpha is explicitly bounded to [0, 1].
    pub fn lerp(&self, other: &Self, alpha: f32) -> Result<Self> {
        ensure!(
            alpha.is_finite() && (0.0..=1.0).contains(&alpha),
            "blend alpha must be in [0,1]"
        );
        ensure!(
            self.shape == other.shape,
            "weight blend requires identical shapes"
        );
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
                .collect(),
        )
    }
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        Self::new(shape, self.data.clone())
    }
    /// Select a nonempty half-open region on an explicit axis.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self> {
        ensure!(
            axis < self.shape.len() && start < end && end <= self.shape[axis],
            "invalid weight slice"
        );
        let inner: usize = self.shape[axis + 1..].iter().product();
        let outer: usize = self.shape[..axis].iter().product();
        let mut data = Vec::with_capacity(outer * (end - start) * inner);
        for block in 0..outer {
            let offset = block * self.shape[axis] * inner;
            data.extend_from_slice(&self.data[offset + start * inner..offset + end * inner]);
        }
        let mut shape = self.shape.clone();
        shape[axis] = end - start;
        Self::new(shape, data)
    }
    pub fn concat(axis: usize, tensors: &[&Self]) -> Result<Self> {
        let first = tensors
            .first()
            .context("concat requires at least one tensor")?;
        ensure!(axis < first.shape.len(), "concat axis out of range");
        let mut shape = first.shape.clone();
        shape[axis] = 0;
        for tensor in tensors {
            ensure!(
                tensor.shape.len() == shape.len()
                    && tensor
                        .shape
                        .iter()
                        .enumerate()
                        .all(|(i, d)| i == axis || *d == shape[i]),
                "concat shape mismatch"
            );
            shape[axis] = shape[axis]
                .checked_add(tensor.shape[axis])
                .context("concat shape overflow")?;
        }
        ensure!(shape[axis] <= 16384, "concat axis exceeds shape bound");
        let total = tensors
            .iter()
            .try_fold(0usize, |n, t| n.checked_add(t.data.len()))
            .context("concat size overflow")?;
        ensure!(total <= MAX_BYTES / 4, "concat exceeds weight budget");
        let inner: usize = first.shape[axis + 1..].iter().product();
        let outer: usize = first.shape[..axis].iter().product();
        let mut data = Vec::with_capacity(total);
        for block in 0..outer {
            for tensor in tensors {
                let width = tensor.shape[axis] * inner;
                data.extend_from_slice(&tensor.data[block * width..(block + 1) * width]);
            }
        }
        Self::new(shape, data)
    }
    pub fn transpose(&self) -> Result<Self> {
        ensure!(self.shape.len() == 2, "transpose requires a rank-2 weight");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; self.data.len()];
        for row in 0..rows {
            for col in 0..cols {
                data[col * rows + row] = self.data[row * cols + col];
            }
        }
        Self::new(vec![cols, rows], data)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub revision: String,
    pub payload: String,
    pub tensors: Vec<TensorSchema>,
    pub provenance: serde_json::Value,
}

/// Immutable snapshot; exposed tensor references cannot mutate its values.
#[derive(Clone, Debug)]
pub struct WeightSet {
    tensors: BTreeMap<String, Tensor>,
    manifest: Manifest,
    bytes: Arc<Vec<u8>>,
}

impl WeightSet {
    pub fn new(tensors: BTreeMap<String, Tensor>) -> Result<Self> {
        Self::with_provenance(tensors, serde_json::json!({}))
    }
    fn with_provenance(
        tensors: BTreeMap<String, Tensor>,
        provenance: serde_json::Value,
    ) -> Result<Self> {
        ensure!(
            tensors.keys().all(|name| !name.is_empty()),
            "empty weight name"
        );
        let parameters = tensors
            .iter()
            .map(|(name, t)| (name.clone(), t.shape.clone(), t.data.clone()))
            .collect::<Vec<_>>();
        let bytes = weight_artifact::encode(&parameters, MAX_BYTES)?;
        let manifest = Manifest {
            format: FORMAT.into(),
            revision: digest(&bytes),
            payload: "weights.safetensors".into(),
            tensors: weight_artifact::inspect(&bytes, MAX_BYTES)?,
            provenance,
        };
        Ok(Self {
            tensors,
            manifest,
            bytes: Arc::new(bytes),
        })
    }
    pub fn revision(&self) -> &str {
        &self.manifest.revision
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("unknown weight: {name}"))
    }
    pub fn reference(&self, name: &str) -> Result<WeightRef<'_>> {
        Ok(WeightRef {
            revision: self.revision(),
            name: self
                .tensors
                .get_key_value(name)
                .context("unknown weight reference")?
                .0,
            tensor: self.tensor(name)?,
        })
    }
    pub fn bindings(&self) -> BTreeMap<String, Vec<f32>> {
        self.tensors
            .iter()
            .map(|(name, t)| (name.clone(), t.data.clone()))
            .collect()
    }
    pub fn validate_for(&self, definition: &ModelDefinition) -> Result<()> {
        ensure!(
            self.tensors.len() == definition.parameters().len(),
            "weights parameter key set mismatch"
        );
        for schema in definition.parameters() {
            validate_tensor(schema, self.tensor(&schema.name)?)?;
        }
        Ok(())
    }
    /// Open a manifest-backed snapshot, optionally requiring its exact content revision.
    pub fn open(directory: impl AsRef<Path>, expected_revision: Option<&str>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest_bytes = read_bounded(&directory.join("manifest.json"), 1024 * 1024)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest.format == FORMAT && manifest.payload == "weights.safetensors",
            "unsupported weight manifest"
        );
        ensure!(
            manifest.provenance.is_object(),
            "weight provenance must be an object"
        );
        let bytes = read_bounded(&directory.join(&manifest.payload), MAX_BYTES)?;
        ensure!(
            manifest.revision == digest(&bytes),
            "weight revision mismatch"
        );
        ensure!(
            expected_revision.is_none_or(|expected| expected == manifest.revision),
            "pinned weight revision mismatch"
        );
        let actual = weight_artifact::inspect(&bytes, MAX_BYTES)?;
        let mut declared = manifest.tensors.clone();
        declared.sort_by(|a, b| a.name.cmp(&b.name));
        ensure!(actual == declared, "weight manifest tensor table mismatch");
        let schema = actual
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect::<Vec<_>>();
        let values = weight_artifact::decode(&schema, &bytes, MAX_BYTES)?;
        let tensors = schema
            .into_iter()
            .zip(values)
            .map(|((name, shape), data)| Ok((name, Tensor::new(shape, data)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            tensors,
            manifest,
            bytes: Arc::new(bytes),
        })
    }
    /// Publish a new directory. The manifest is published last; existing paths are never replaced.
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        let parent = directory
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        fs::create_dir(directory).with_context(|| {
            format!(
                "snapshot already exists or cannot be created: {}",
                directory.display()
            )
        })?;
        let result = (|| {
            wgtime::publication::write(
                &directory.join("weights.safetensors"),
                wgtime::publication::Mode::CreateNew,
                |mut file| Ok(file.write_all(self.bytes.as_ref())?),
            )?;
            let manifest = serde_json::to_vec_pretty(&self.manifest)?;
            ensure!(
                manifest.len() <= 1024 * 1024,
                "weight manifest exceeds limit"
            );
            wgtime::publication::write(
                &directory.join("manifest.json"),
                wgtime::publication::Mode::CreateNew,
                |mut file| Ok(file.write_all(&manifest)?),
            )?;
            fs::File::open(directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(directory);
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WeightRef<'a> {
    revision: &'a str,
    name: &'a str,
    tensor: &'a Tensor,
}
impl WeightRef<'_> {
    pub fn revision(&self) -> &str {
        self.revision
    }
    pub fn name(&self) -> &str {
        self.name
    }
    pub fn tensor(&self) -> &Tensor {
        self.tensor
    }
}

type Initializer = Box<dyn FnOnce(&[usize]) -> Result<Tensor>>;
enum Value {
    Tensor(Tensor),
    Initialize(Initializer),
}

#[derive(Clone, Debug, Serialize)]
pub struct PlanPreview {
    pub targets: Vec<serde_json::Value>,
    pub missing: Vec<String>,
    pub complete: bool,
}

/// Each target has one value. A base explicitly supplies the whole matching schema;
/// replacements are named exceptions and may occur at most once per target.
pub struct Plan {
    definition: ModelDefinition,
    schema: Vec<ParameterSchema>,
    values: BTreeMap<String, Value>,
    sources: BTreeMap<String, serde_json::Value>,
    replaced: BTreeSet<String>,
}

impl Plan {
    pub fn new(definition: &ModelDefinition) -> Self {
        Self {
            definition: definition.clone(),
            schema: definition.parameters().to_vec(),
            values: BTreeMap::new(),
            sources: BTreeMap::new(),
            replaced: BTreeSet::new(),
        }
    }
    pub fn from_base(definition: &ModelDefinition, base: &WeightSet) -> Result<Self> {
        base.validate_for(definition)?;
        let mut plan = Self::new(definition);
        for name in base.names() {
            plan.copy(name, base.reference(name)?)?;
        }
        Ok(plan)
    }
    fn schema(&self, name: &str) -> Result<&ParameterSchema> {
        self.schema
            .iter()
            .find(|s| s.name == name)
            .with_context(|| format!("unknown target parameter: {name}"))
    }
    fn vacant(&self, name: &str) -> Result<()> {
        self.schema(name)?;
        ensure!(
            !self.values.contains_key(name),
            "duplicate parameter binding: {name}"
        );
        Ok(())
    }
    pub fn set(&mut self, name: &str, tensor: Tensor) -> Result<&mut Self> {
        self.vacant(name)?;
        validate_tensor(self.schema(name)?, &tensor)?;
        self.values.insert(name.into(), Value::Tensor(tensor));
        self.sources
            .insert(name.into(), serde_json::json!({"kind":"value"}));
        Ok(self)
    }
    pub fn copy(&mut self, target: &str, source: WeightRef<'_>) -> Result<&mut Self> {
        self.set(target, source.tensor.clone())?;
        self.sources.insert(
            target.into(),
            serde_json::json!({"kind":"snapshot","revision":source.revision,"name":source.name}),
        );
        Ok(self)
    }
    /// Bind an explicitly computed tensor and record the fixed snapshots it used.
    pub fn set_with_sources(
        &mut self,
        name: &str,
        tensor: Tensor,
        sources: &[WeightRef<'_>],
    ) -> Result<&mut Self> {
        self.set(name, tensor)?;
        self.record_sources(name, sources);
        Ok(self)
    }
    pub fn replace(&mut self, name: &str, tensor: Tensor) -> Result<&mut Self> {
        validate_tensor(self.schema(name)?, &tensor)?;
        ensure!(
            self.values.contains_key(name),
            "replace requires an existing binding: {name}"
        );
        ensure!(
            !self.replaced.contains(name),
            "duplicate replacement: {name}"
        );
        self.replaced.insert(name.into());
        self.values.insert(name.into(), Value::Tensor(tensor));
        let base = self.sources.remove(name).unwrap();
        self.sources.insert(
            name.into(),
            serde_json::json!({"kind":"replacement","base":base}),
        );
        Ok(self)
    }
    pub fn replace_with_sources(
        &mut self,
        name: &str,
        tensor: Tensor,
        sources: &[WeightRef<'_>],
    ) -> Result<&mut Self> {
        self.replace(name, tensor)?;
        self.record_sources(name, sources);
        Ok(self)
    }
    fn record_sources(&mut self, name: &str, sources: &[WeightRef<'_>]) {
        self.sources.get_mut(name).unwrap()["inputs"] = serde_json::Value::Array(
            sources
                .iter()
                .map(|source| serde_json::json!({"revision":source.revision,"name":source.name}))
                .collect(),
        );
    }
    /// Register a host initializer. Preview never calls it; build invokes it once.
    pub fn initialize(
        &mut self,
        name: &str,
        initializer: impl FnOnce(&[usize]) -> Result<Tensor> + 'static,
    ) -> Result<&mut Self> {
        self.vacant(name)?;
        self.values
            .insert(name.into(), Value::Initialize(Box::new(initializer)));
        self.sources
            .insert(name.into(), serde_json::json!({"kind":"initializer"}));
        Ok(self)
    }
    /// Explicitly select this parameter's WGT recipe; other recipes are never evaluated.
    pub fn initialize_declared(&mut self, name: &str) -> Result<&mut Self> {
        self.vacant(name)?;
        let recipe = self.definition.parameter_initializer(name)?;
        ensure!(
            !matches!(recipe, wgt_language::initializer::Recipe::Unbound),
            "parameter {name} has no declared initializer"
        );
        let provenance = serde_json::to_value(&recipe)?;
        let definition = self.definition.clone();
        let key = name.to_owned();
        self.initialize(name, move |shape| {
            Tensor::new(shape.to_vec(), definition.initialize_parameter(&key)?)
        })?;
        self.sources.insert(
            name.into(),
            serde_json::json!({"kind":"declared_initializer","recipe":provenance}),
        );
        Ok(self)
    }
    pub fn preview(&self) -> PlanPreview {
        let missing = self
            .schema
            .iter()
            .filter(|s| !self.values.contains_key(&s.name))
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();
        PlanPreview {
            complete: missing.is_empty(), missing,
            targets: self.schema.iter().map(|s| serde_json::json!({"name":s.name,"shape":s.shape,"dtype":"F32","source":self.sources.get(&s.name)})).collect(),
        }
    }
    /// Validate the complete plan before invoking any initializer, then validate
    /// every resulting value before returning an independently owned snapshot.
    pub fn build(mut self) -> Result<WeightSet> {
        let preview = self.preview();
        ensure!(
            preview.complete,
            "unbound parameters: {}",
            preview.missing.join(", ")
        );
        let mut tensors = BTreeMap::new();
        for schema in &self.schema {
            let tensor = match self.values.remove(&schema.name).unwrap() {
                Value::Tensor(tensor) => tensor,
                Value::Initialize(initialize) => initialize(&schema.shape)
                    .with_context(|| format!("initializer failed: {}", schema.name))?,
            };
            validate_tensor(schema, &tensor)?;
            tensors.insert(schema.name.clone(), tensor);
        }
        WeightSet::with_provenance(tensors, serde_json::json!({"bindings":self.sources}))
    }
}

fn validate_tensor(schema: &ParameterSchema, tensor: &Tensor) -> Result<()> {
    ensure!(
        tensor.shape == schema.shape,
        "weight shape mismatch: {}",
        schema.name
    );
    schema.validate_values(&tensor.data)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= limit as u64,
        "weight file exceeds byte limit"
    );
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= limit, "weight file exceeds byte limit");
    Ok(bytes)
}
