//! Native f32 inference with checked parameter expressions and reusable CPU storage.
//! `Session::run` allocates no heap memory. Preparation and transport may allocate.
pub mod accelerator;
pub mod adaptive;
pub mod categorical;
pub mod collection;
pub mod execution;
pub mod isolation;
mod parameter_ops;
mod quantized;
pub mod script;
pub mod source;
pub mod training;
pub mod tasks;
pub mod openresponses;
pub mod weightfs;
pub use parameter_ops::{ParameterInfo, ParameterWrite};
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub mod wgsl;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(default, skip_serializing_if = "categorical::VectorType::is_plain")]
    pub output_type: categorical::VectorType,
    pub version: u32,
    pub input: usize,
    pub parameters: Vec<Parameter>,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub frozen: bool,
    #[serde(flatten)]
    pub value: Value,
}

/// References resolve only to earlier declarations, so cycles cannot be constructed.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Value {
    Resident {
        data: Vec<f32>,
    },
    Packed {
        storage: String,
        group: usize,
        data: Vec<u8>,
        scales: Vec<f32>,
    },
    PackedFile {
        storage: String,
        group: usize,
        path: PathBuf,
        offset: u64,
        scales_offset: u64,
    },
    File {
        path: PathBuf,
        offset: u64,
    },
    ResidentFile {
        path: PathBuf,
        offset: u64,
    },
    View {
        source: String,
        start: usize,
    },
    Add {
        base: String,
        delta: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Layer {
    Linear {
        weight: String,
        bias: Option<String>,
    },
    Relu,
    Tanh,
    Sigmoid,
    Identity,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub device: String,
    pub resident_bytes: usize,
    pub activation_bytes: usize,
    pub workspace_bytes: usize,
    pub managed_bytes: usize,
    pub input: usize,
    pub output: usize,
    pub streamed_parameters: usize,
}

#[derive(Debug)]
enum Storage {
    Resident(Box<[f32]>),
    Packed(quantized::Packed),
    File { handle: File, offset: u64 },
    View { source: usize, start: usize },
    Add { base: usize, delta: usize },
}
#[derive(Debug)]
struct Tensor {
    shape: Vec<usize>,
    len: usize,
    frozen: bool,
    storage: Storage,
}
#[derive(Debug)]
enum Op {
    Linear {
        weight: usize,
        bias: Option<usize>,
        input: usize,
        output: usize,
    },
    Relu,
    Tanh,
    Sigmoid,
    Identity,
}

/// Owns all execution buffers and file handles; Drop releases them deterministically.
#[derive(Debug)]
pub struct Session {
    output_type: categorical::VectorType,
    names: HashMap<String, usize>,
    revision: u64,
    collection: collection::CollectionState,
    parameters: Vec<Tensor>,
    ops: Vec<Op>,
    left: Vec<f32>,
    right: Vec<f32>,
    workspace: Vec<f32>,
    report: Report,
}

fn elements(shape: &[usize]) -> Result<usize> {
    ensure!(
        !shape.is_empty() && !shape.contains(&0),
        "parameter shapes must be nonempty and positive"
    );
    shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d).context("shape overflow"))
}
fn bytes(n: usize) -> Result<usize> {
    n.checked_mul(4).context("byte size overflow")
}

fn zero_buffer(len: usize) -> Result<Vec<f32>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .context("cannot reserve native execution buffer")?;
    buffer.resize(len, 0.);
    Ok(buffer)
}

impl Session {
    pub fn begin_collection(&self) -> Result<collection::CollectionToken> { self.collection.begin() }
    pub(crate) fn ensure_mutable(&self) -> Result<()> { self.collection.ensure_mutable() }
    /// Validate all static contracts and the managed tensor budget before allocating arenas.
    /// File offsets are byte offsets into immutable little-endian f32 files.
    /// The budget excludes IR metadata, JSON parsing, OS page cache and a 4 KiB I/O stack buffer.
    pub fn prepare(model: Model, root: &Path, budget: usize) -> Result<Self> {
        ensure!(matches!(model.version, 1 | 2), "unsupported native IR version");
        ensure!(model.version >= 2 || model.output_type == categorical::VectorType::F32,
            "categorical output metadata requires native IR version 2");
        ensure!(
            model.input > 0 && !model.layers.is_empty(),
            "input and layers must be nonempty"
        );
        let mut names = HashMap::new();
        let mut parameters: Vec<Tensor> = Vec::new();
        let mut resident = 0usize;
        let mut streamed = 0;
        let mut depths: Vec<usize> = Vec::new();
        for p in model.parameters {
            ensure!(
                !p.name.is_empty() && !names.contains_key(&p.name),
                "duplicate/empty parameter identity: {}",
                p.name
            );
            let len = elements(&p.shape)?;
            let byte_len = bytes(len)?;
            let lookup = |name: &str| -> Result<usize> {
                names
                    .get(name)
                    .copied()
                    .with_context(|| format!("unknown/forward parameter reference: {name}"))
            };
            let (storage, depth) = match p.value {
                Value::Resident { data } => {
                    ensure!(
                        data.len() == len && data.iter().all(|x| x.is_finite()),
                        "invalid resident parameter: {}",
                        p.name
                    );
                    resident = resident
                        .checked_add(byte_len)
                        .context("resident size overflow")?;
                    (Storage::Resident(data.into_boxed_slice()), 0usize)
                }
                Value::Packed {
                    storage,
                    group,
                    data,
                    scales,
                } => {
                    let packed = quantized::Packed::new(&storage, group, &p.shape, data, scales)?;
                    resident = resident
                        .checked_add(packed.bytes())
                        .context("packed budget overflow")?;
                    (Storage::Packed(packed), 0)
                }
                Value::PackedFile {
                    storage,
                    group,
                    path,
                    offset,
                    scales_offset,
                } => {
                    let (_, _, n, count) = quantized::Packed::layout(&storage, group, &p.shape)?;
                    let scale_bytes = bytes(count)?;
                    let packed_bytes = n
                        .checked_add(scale_bytes)
                        .context("packed budget overflow")?;
                    resident = resident
                        .checked_add(packed_bytes)
                        .context("packed budget overflow")?;
                    ensure!(resident <= budget, "packed storage exceeds budget");
                    let mut file = File::open(root.join(path))?;
                    let length = file.metadata()?.len();
                    ensure!(
                        offset.checked_add(n as u64).is_some_and(|v| v <= length)
                            && scales_offset
                                .checked_add(scale_bytes as u64)
                                .is_some_and(|v| v <= length),
                        "truncated packed file"
                    );
                    let mut data = vec![0; n];
                    file.seek(SeekFrom::Start(offset))?;
                    file.read_exact(&mut data)?;
                    file.seek(SeekFrom::Start(scales_offset))?;
                    let mut scales = Vec::with_capacity(count);
                    for _ in 0..count {
                        let mut raw = [0; 4];
                        file.read_exact(&mut raw)?;
                        scales.push(f32::from_le_bytes(raw));
                    }
                    let packed = quantized::Packed::new(&storage, group, &p.shape, data, scales)?;
                    (Storage::Packed(packed), 0)
                }
                Value::ResidentFile { path, offset } => {
                    resident = resident
                        .checked_add(byte_len)
                        .context("resident size overflow")?;
                    ensure!(resident <= budget, "resident storage exceeds budget");
                    let mut handle = File::open(root.join(path))?;
                    ensure!(
                        offset % 4 == 0
                            && offset.checked_add(byte_len as u64).is_some_and(
                                |v| v <= handle.metadata().map(|m| m.len()).unwrap_or(0)
                            ),
                        "invalid resident file range"
                    );
                    let mut data = zero_buffer(len)?;
                    read_add(&mut handle, offset, &mut data).map_err(anyhow::Error::msg)?;
                    (Storage::Resident(data.into_boxed_slice()), 0)
                }
                Value::File { path, offset } => {
                    ensure!(offset % 4 == 0, "file offset must be f32-aligned");
                    let handle =
                        File::open(root.join(path)).context("opening streamed parameter")?;
                    let end = offset
                        .checked_add(u64::try_from(byte_len)?)
                        .context("file offset overflow")?;
                    ensure!(
                        handle.metadata()?.len() >= end,
                        "truncated parameter file: {}",
                        p.name
                    );
                    streamed += 1;
                    (Storage::File { handle, offset }, 0)
                }
                Value::View { source, start } => {
                    let source = lookup(&source)?;
                    ensure!(
                        start
                            .checked_add(len)
                            .is_some_and(|end| end <= parameters[source].len),
                        "parameter view out of bounds"
                    );
                    (Storage::View { source, start }, depths[source] + 1)
                }
                Value::Add { base, delta } => {
                    let (base, delta) = (lookup(&base)?, lookup(&delta)?);
                    ensure!(
                        parameters[base].shape == p.shape && parameters[delta].shape == p.shape,
                        "base/delta shape mismatch"
                    );
                    (
                        Storage::Add { base, delta },
                        depths[base].max(depths[delta]) + 1,
                    )
                }
            };
            ensure!(depth <= 32, "parameter expression depth exceeds 32");
            depths.push(depth);
            names.insert(p.name, parameters.len());
            parameters.push(Tensor {
                shape: p.shape,
                len,
                frozen: p.frozen || matches!(storage, Storage::Packed(_)),
                storage,
            });
        }
        // Freeze dependencies as well: a frozen view/expression cannot be changed
        // indirectly through its storage owner. This conservatively freezes full owners.
        for index in (0..parameters.len()).rev() {
            if parameters[index].frozen {
                match parameters[index].storage {
                    Storage::View { source, .. } => parameters[source].frozen = true,
                    Storage::Add { base, delta } => {
                        parameters[base].frozen = true;
                        parameters[delta].frozen = true;
                    }
                    _ => {}
                }
            }
        }
        let mut width = model.input;
        let mut maximum = width;
        let mut workspace = 0;
        let mut ops = Vec::new();
        for layer in model.layers {
            let op = match layer {
                Layer::Linear { weight, bias } => {
                    let weight = *names.get(&weight).context("unknown linear weight")?;
                    let shape = &parameters[weight].shape;
                    ensure!(
                        shape.len() == 2 && shape[1] == width,
                        "linear weight must be [out, input]"
                    );
                    let output = shape[0];
                    let bias = bias
                        .map(|name| names.get(&name).copied().context("unknown bias"))
                        .transpose()?;
                    if let Some(bias) = bias {
                        ensure!(
                            parameters[bias].shape == [output],
                            "bias must have shape [out]"
                        );
                    }
                    let dense = if matches!(parameters[weight].storage, Storage::Packed(_)) {
                        0
                    } else {
                        parameters[weight].len
                    };
                    workspace = workspace.max(
                        dense
                            .checked_add(if bias.is_some() { output } else { 0 })
                            .context("workspace overflow")?,
                    );
                    let op = Op::Linear {
                        weight,
                        bias,
                        input: width,
                        output,
                    };
                    width = output;
                    maximum = maximum.max(width);
                    op
                }
                Layer::Relu => Op::Relu,
                Layer::Tanh => Op::Tanh,
                Layer::Sigmoid => Op::Sigmoid,
                Layer::Identity => Op::Identity,
            };
            ops.push(op);
        }
        let activation_bytes = bytes(maximum)?
            .checked_mul(2)
            .context("activation overflow")?;
        let workspace_bytes = bytes(workspace)?;
        let managed_bytes = resident
            .checked_add(activation_bytes)
            .and_then(|n| n.checked_add(workspace_bytes))
            .context("budget overflow")?;
        ensure!(
            managed_bytes <= budget,
            "managed memory budget exceeded: need {managed_bytes}, have {budget}"
        );
        let report = Report {
            device: "cpu".into(),
            resident_bytes: resident,
            activation_bytes,
            workspace_bytes,
            managed_bytes,
            input: model.input,
            output: width,
            streamed_parameters: streamed,
        };
        Ok(Self {
            output_type: model.output_type,
            names,
            revision: 0,
            collection: collection::CollectionState::default(),
            parameters,
            ops,
            left: zero_buffer(maximum)?,
            right: zero_buffer(maximum)?,
            workspace: zero_buffer(workspace)?,
            report,
        })
    }

    pub fn output_type(&self) -> categorical::VectorType { self.output_type }

    pub fn report(&self) -> &Report {
        &self.report
    }

    /// One unbatched vector. Output borrows the session and is valid until its next mutable use.
    /// Successful calls allocate no heap memory, including file-backed parameter evaluation.
    pub fn run(&mut self, input: &[f32]) -> std::result::Result<&[f32], &'static str> {
        if input.len() != self.report.input {
            return Err("input length mismatch");
        }
        self.left[..input.len()].copy_from_slice(input);
        let mut width = input.len();
        for op in &self.ops {
            match *op {
                Op::Linear {
                    weight,
                    bias,
                    input,
                    output,
                } => {
                    if matches!(self.parameters[weight].storage, Storage::Packed(_)) {
                        self.workspace[..if bias.is_some() { output } else { 0 }].fill(0.);
                        if let Some(bias) = bias {
                            accumulate(
                                &mut self.parameters,
                                bias,
                                0,
                                &mut self.workspace[..output],
                            )?;
                        }
                        if let Storage::Packed(packed) = &self.parameters[weight].storage {
                            for row in 0..output {
                                self.right[row] = packed.dot(row, &self.left[..input])
                                    + if bias.is_some() {
                                        self.workspace[row]
                                    } else {
                                        0.
                                    };
                            }
                        }
                        std::mem::swap(&mut self.left, &mut self.right);
                        width = output;
                        continue;
                    }
                    // Resident matrices do not need copying. Keep expression/file fallback below.
                    if matches!(self.parameters[weight].storage, Storage::Resident(_)) {
                        self.workspace[..if bias.is_some() { output } else { 0 }].fill(0.);
                        if let Some(bias) = bias {
                            accumulate(
                                &mut self.parameters,
                                bias,
                                0,
                                &mut self.workspace[..output],
                            )?;
                        }
                        if let Storage::Resident(data) = &self.parameters[weight].storage {
                            for row in 0..output {
                                self.right[row] = quantized::dot(
                                    &data[row * input..(row + 1) * input],
                                    &self.left[..input],
                                ) + if bias.is_some() {
                                    self.workspace[row]
                                } else {
                                    0.
                                };
                            }
                        }
                        std::mem::swap(&mut self.left, &mut self.right);
                        width = output;
                        continue;
                    }
                    let n = self.parameters[weight].len;
                    self.workspace[..n].fill(0.);
                    accumulate(&mut self.parameters, weight, 0, &mut self.workspace[..n])?;
                    if let Some(bias) = bias {
                        self.workspace[n..n + output].fill(0.);
                        accumulate(
                            &mut self.parameters,
                            bias,
                            0,
                            &mut self.workspace[n..n + output],
                        )?;
                    }
                    for row in 0..output {
                        let mut value = if bias.is_some() {
                            self.workspace[n + row]
                        } else {
                            0.
                        };
                        for col in 0..input {
                            value += self.left[col] * self.workspace[row * input + col];
                        }
                        self.right[row] = value;
                    }
                    std::mem::swap(&mut self.left, &mut self.right);
                    width = output;
                }
                Op::Relu => {
                    for x in &mut self.left[..width] {
                        if *x < 0. {
                            *x = 0.;
                        }
                    }
                }
                Op::Tanh => {
                    for x in &mut self.left[..width] {
                        *x = x.tanh();
                    }
                }
                Op::Sigmoid => {
                    for x in &mut self.left[..width] {
                        *x = 1. / (1. + (-*x).exp());
                    }
                }
                Op::Identity => {}
            }
        }
        if !self.output_type.is_plain() {
            let output = &self.left[..width];
            if output.is_empty() || output.iter().any(|x| !x.is_finite()) {
                return Err("categorical output must be a nonempty finite vector");
            }
            if self.output_type == categorical::VectorType::Probabilities {
                if output.iter().any(|x| !(0. ..=1.).contains(x)) {
                    return Err("probabilities must each be in [0,1]");
                }
                let total: f64 = output.iter().map(|x| *x as f64).sum();
                if (total - 1.).abs() > categorical::PROBABILITY_SUM_TOLERANCE {
                    return Err("probabilities must sum to 1 within absolute tolerance 1e-5");
                }
            }
        }
        Ok(&self.left[..width])
    }
}

// A parameter expression adds into the caller's workspace. Views retain identity;
// additions never mutate their sources or allocate a full permanent effective weight.
fn accumulate(
    parameters: &mut [Tensor],
    index: usize,
    start: usize,
    out: &mut [f32],
) -> std::result::Result<(), &'static str> {
    match &mut parameters[index].storage {
        Storage::Resident(data) => {
            for (dst, src) in out.iter_mut().zip(data[start..].iter()) {
                *dst += src;
            }
        }
        Storage::Packed(packed) => {
            for (i, value) in out.iter_mut().enumerate() {
                *value += packed.value(start + i);
            }
        }
        Storage::File { handle, offset } => {
            read_add(handle, *offset + start as u64 * 4, out)?;
        }
        Storage::View {
            source,
            start: offset,
        } => {
            let (source, start) = (*source, start + *offset);
            accumulate(parameters, source, start, out)?;
        }
        Storage::Add { base, delta } => {
            let (base, delta) = (*base, *delta);
            accumulate(parameters, base, start, out)?;
            accumulate(parameters, delta, start, out)?;
        }
    }
    Ok(())
}

// Keep the fixed I/O buffer out of recursive expression frames.
#[inline(never)]
fn read_add(
    handle: &mut File,
    offset: u64,
    out: &mut [f32],
) -> std::result::Result<(), &'static str> {
    handle
        .seek(SeekFrom::Start(offset))
        .map_err(|_| "parameter seek failed")?;
    let mut buffer = [0u8; 4096];
    for chunk in out.chunks_mut(1024) {
        handle
            .read_exact(&mut buffer[..chunk.len() * 4])
            .map_err(|_| "parameter read failed")?;
        for (dst, raw) in chunk.iter_mut().zip(buffer.as_chunks::<4>().0.iter()) {
            let value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            if !value.is_finite() {
                return Err("nonfinite file parameter");
            }
            *dst += value;
        }
    }
    Ok(())
}
