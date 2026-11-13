//! Shape-driven construction of fixed model definitions, without initialization.
use crate::{
    dimensions::{self, Dimensions},
    model::ModelDefinition,
    script::parse::{self, ShapeExtent},
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// A root-source snapshot with typed forward inputs. Binding logical shapes
/// constructs an ordinary fixed-size definition; it never executes a model or
/// initializes parameters. Imported libraries retain their own dimension scope
/// and are frozen only after the first successful binding.
#[derive(Clone, Debug)]
pub struct ModelTemplate {
    source: String,
    path: Option<PathBuf>,
    dimensions: BTreeMap<String, Option<usize>>,
    inputs: Vec<(String, Vec<ShapeExtent>)>,
    frozen: RefCell<Option<FrozenSources>>,
}

#[derive(Clone, Debug)]
struct FrozenSources {
    imports: BTreeMap<PathBuf, String>,
    standards: BTreeMap<String, String>,
}

impl ModelTemplate {
    /// Extract the input contract. Full model/body validation occurs on binding.
    /// Inline templates follow ModelDefinition's inline import discovery rules.
    pub fn parse(source: &str) -> Result<Self> {
        ensure!(source.len() <= 1024 * 1024, "model template exceeds 1 MiB");
        let (dimensions, stripped) = dimensions::declarations(source)?;
        let unit = crate::source::split_unspecialized(&stripped)?;
        ensure!(
            unit.models.is_empty(),
            "model templates do not yet support submodel declarations"
        );
        let declarations = unit
            .declarations
            .lines()
            .map(|line| line.split('#').next().unwrap_or("").trim())
            .collect::<Vec<_>>();
        ensure!(
            !declarations
                .iter()
                .any(|line| line.starts_with("target torch")
                    || line.starts_with("state ")
                    || line.starts_with("input ")
                    || line.starts_with("output ")
                    || line.starts_with("layer ")),
            "model templates require a typed single forward, without root ports, state, layers or a torch-module target"
        );
        let models = declarations
            .iter()
            .filter_map(|line| line.strip_prefix("model "))
            .collect::<Vec<_>>();
        ensure!(
            models.len() == 1 && crate::script::operators::identifier(models[0].trim()),
            "model templates require one root model"
        );
        let inputs = parse::template_forward_inputs(&unit.functions)?;
        for extent in inputs.iter().flat_map(|(_, shape)| shape) {
            if let ShapeExtent::Symbol(name) = extent {
                ensure!(
                    dimensions.contains_key(name),
                    "undeclared input dimension: {name}"
                );
            }
        }
        Ok(Self {
            source: source.to_owned(),
            path: None,
            dimensions,
            inputs,
            frozen: RefCell::new(None),
        })
    }

    /// Read a bounded regular file and retain its root text and import location.
    /// Libraries are resolved by the ordinary frontend at binding time.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path
            .as_ref()
            .canonicalize()
            .context("model template path")?;
        let mut template = Self::parse(&crate::model_modules::read(&path)?)?;
        template.path = Some(path);
        Ok(template)
    }

    /// Infer declared dimensions from a logical input shape, then run the shared
    /// concrete frontend. Shape values override defaults, but must agree with
    /// explicit bindings and with every occurrence of the same symbol.
    /// Dimensions absent from the input require a default or explicit binding.
    pub fn bind_input_shapes(
        &self,
        input_shapes: &BTreeMap<String, Vec<usize>>,
        explicit: &Dimensions,
    ) -> Result<ModelDefinition> {
        ensure!(
            input_shapes.len() == self.inputs.len()
                && self
                    .inputs
                    .iter()
                    .all(|(name, _)| input_shapes.contains_key(name)),
            "input shapes must contain exactly the forward input ports: {}",
            self.inputs
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        for (name, &value) in explicit {
            ensure!(
                self.dimensions.contains_key(name),
                "unknown dimension binding: {name}"
            );
            ensure!(
                (1..=16384).contains(&value),
                "dimension must be in 1..=16384: {name}"
            );
        }
        let mut bindings = explicit.clone();
        let mut total = 0usize;
        for (name, contract) in &self.inputs {
            let actual = &input_shapes[name];
            ensure!(
                actual.len() == contract.len(),
                "input {name} rank mismatch: expected {}, got {}",
                contract.len(),
                actual.len()
            );
            let elements =
                crate::tensor::validate_shape(actual, crate::tensor::ShapeLimits::REGISTER)?;
            total = total
                .checked_add(elements)
                .context("forward input size overflow")?;
            ensure!(total <= 16384, "forward input exceeds 16384 values");
            for (axis, (extent, &value)) in contract.iter().zip(actual).enumerate() {
                match extent {
                    ShapeExtent::Fixed(expected) => ensure!(
                        *expected == value,
                        "input {name} axis {axis} mismatch: expected {expected}, got {value}"
                    ),
                    ShapeExtent::Symbol(symbol) => {
                        if let Some(previous) = bindings.get(symbol) {
                            ensure!(
                                *previous == value,
                                "dimension {symbol} conflict at input {name} axis {axis}: bound {previous}, got {value}"
                            );
                        } else {
                            bindings.insert(symbol.clone(), value);
                        }
                    }
                }
            }
        }
        // Specialize only the captured root; the existing resolver independently
        // specializes each imported compilation unit and enforces closure limits.
        let specialized = dimensions::specialize(&self.source, &bindings)?;
        let cached = self.frozen.borrow();
        let candidate;
        let frozen = if let Some(frozen) = cached.as_ref() {
            frozen
        } else {
            let (_, dependencies) =
                crate::model_modules::resolve(&specialized, self.path.as_deref())?;
            let mut imports = BTreeMap::new();
            for path in dependencies {
                // The root is already captured, including when its physical file
                // changes or disappears. Never replace it with a disk read.
                if self.path.as_ref() != Some(&path) {
                    imports.insert(path.clone(), crate::model_modules::read(&path)?);
                }
            }
            let bytes = imports
                .values()
                .try_fold(self.source.len(), |bytes, source| {
                    bytes
                        .checked_add(source.len())
                        .context("model module size overflow")
                })?;
            ensure!(
                bytes <= 1024 * 1024,
                "model dependency closure exceeds 1 MiB"
            );
            let mut standards = BTreeMap::new();
            capture_standards(&specialized, &mut standards)?;
            for source in imports.values() {
                capture_standards(source, &mut standards)?;
            }
            candidate = FrozenSources { imports, standards };
            &candidate
        };
        // Return only a definition resolved from the exact captured closure.
        // Discovery, bounded reads, resolution and validation must all succeed
        // before publishing the snapshot; a failed bind cannot poison it.
        let (resolved, dependencies) = crate::model_modules::resolve_with_overlays(
            &specialized,
            self.path.as_deref(),
            &frozen.imports,
        )?;
        let resolved_imports = dependencies
            .iter()
            .filter(|path| self.path.as_ref() != Some(*path))
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            resolved_imports == frozen.imports.keys().collect(),
            "model dependency closure changed during template capture"
        );
        let definition = ModelDefinition::parse_resolved(&resolved)?;
        ensure!(
            definition.states().is_empty()
                && definition
                    .operations()
                    .iter()
                    .all(|operation| operation.name == "forward"),
            "model templates support only one forward entrypoint, not additional typed operations or persistent state"
        );
        let snapshot = cached.is_none().then(|| frozen.clone());
        drop(cached);
        if let Some(snapshot) = snapshot {
            *self.frozen.borrow_mut() = Some(snapshot);
        }
        Ok(definition)
    }

    /// SHA-256 identity of the captured root and frozen import closure, independent
    /// of shape bindings. Available only after a successful bind. Standard modules
    /// contribute their actual embedded source, recursively, rather than just names.
    pub fn source_fingerprint(&self) -> Result<String> {
        let frozen = self.frozen.borrow();
        let frozen = frozen
            .as_ref()
            .context("template source fingerprint requires a successful binding")?;
        let mut hash = Sha256::new();
        fn field(hash: &mut Sha256, bytes: &[u8]) {
            hash.update((bytes.len() as u64).to_be_bytes());
            hash.update(bytes);
        }
        field(&mut hash, b"weightlang.model-template.sources.v1");
        field(&mut hash, self.source.as_bytes());
        field(&mut hash, &(frozen.imports.len() as u64).to_be_bytes());
        for (path, source) in &frozen.imports {
            field(&mut hash, path.as_os_str().as_encoded_bytes());
            field(&mut hash, source.as_bytes());
        }
        field(&mut hash, &(frozen.standards.len() as u64).to_be_bytes());
        for (path, source) in &frozen.standards {
            field(&mut hash, path.as_bytes());
            field(&mut hash, source.as_bytes());
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

fn capture_standards(source: &str, standards: &mut BTreeMap<String, String>) -> Result<()> {
    for (_, path) in crate::script::operators::imports(source)? {
        if path.starts_with("std/") && !standards.contains_key(&path) {
            let source = crate::script::operators::standard_source(&path)
                .with_context(|| format!("unknown model standard module: {path}"))?;
            standards.insert(path, source.to_owned());
            capture_standards(source, standards)?;
        }
    }
    Ok(())
}
