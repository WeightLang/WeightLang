use serde::{Serialize, Deserialize};
mod source;
mod conv;
mod contracts;
pub use contracts::{OperatorContract, OPERATOR_CONTRACTS};
pub use conv::{ConvGeometry, Pool2dGeometry};
pub use source::{parse, parse_with_functions, parse_with_metadata};
pub(crate) use source::{validate_definition, lower_operation, analyze_definition};
pub(crate) type InferredTypes = std::collections::BTreeMap<
    String, std::collections::BTreeMap<String, crate::tensor::TensorType>,
>;

/// Source-level instance identity retained separately from the flattened IR.
/// Repeated calls to one path share the same mapped parameter storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleSchema {
    pub path: String,
    pub model: String,
    pub input: usize,
    pub output: usize,
    pub parameters: std::collections::BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    #[serde(default, skip_serializing_if = "crate::categorical::VectorType::is_plain")]
    pub output_type: crate::categorical::VectorType,
    pub version: u32,
    pub input: String,
    pub registers: Vec<Register>,
    pub parameters: Vec<Parameter>,
    pub instructions: Vec<Instruction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<Box<Program>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Register {
    pub name: String,
    pub len: usize,
    /// Explicit row-major geometry; never reconstructed from the flat length.
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    #[serde(default)]
    pub frozen: bool,
    /// Trainable probabilities store logits; frozen probabilities store exact values.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub probability: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learn: Option<LearnDomain>,
}

/// Projected parameter-space optimization. Discrete reads use identity STE;
/// storage, optimizer moments and snapshots retain the continuous shadow value.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LearnDomain {
    Float,
    Bounded { low: f32, high: f32 },
    Positive { low: f32 },
    Int { low: f32, high: f32 },
    Bool,
}
impl LearnDomain {
    pub fn validate(&self, value: f32) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(value.is_finite(), "learn value must be finite");
        match *self {
            Self::Float => {},
            Self::Bounded { low, high } | Self::Int { low, high } => {
                ensure!(low.is_finite() && high.is_finite() && low < high, "invalid learn interval");
                if matches!(self, Self::Int { .. }) {
                    ensure!(low.fract() == 0. && high.fract() == 0. && low.abs() <= 16777216. && high.abs() <= 16777216., "learn int bounds must be exact f32 integers within +/-16777216");
                }
                ensure!((low..=high).contains(&value), "learn value outside declared interval");
            },
            Self::Positive { low } => {
                ensure!(low.is_finite() && low >= 0. && low < f32::MAX, "invalid learn positive bound");
                ensure!(value > low, "learn value must exceed strict lower bound");
            },
            Self::Bool => ensure!((0. ..=1.).contains(&value), "learn bool shadow must be in [0,1]"),
        }
        Ok(())
    }
    pub fn project(&self, value: f32) -> f32 {
        match *self {
            Self::Float => value,
            Self::Bounded { low, high } | Self::Int { low, high } => value.clamp(low, high),
            Self::Positive { low } => value.max(if low == 0. { f32::from_bits(1) } else { f32::from_bits(low.to_bits() + 1) }),
            Self::Bool => value.clamp(0., 1.),
        }
    }
    pub fn read(&self, value: f32) -> f32 {
        match self { Self::Int { .. } => value.round(), Self::Bool => if value >= 0.5 { 1. } else { 0. }, _ => value }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instruction {
    Conv { out: String, input: String, kernel: String, bias: String, stride: Vec<usize>, padding: Vec<usize>, dilation: Vec<usize>, groups: usize },
    MaxPool2d { out: String, input: String, kernel: Vec<usize>, stride: Vec<usize> },
    AvgPool2d { out: String, input: String, kernel: Vec<usize>, stride: Vec<usize> },
    Reshape { out: String, input: String },
    Transpose { out: String, input: String },
    /// General axis permutation. axes[out_axis] = input_axis.
    Permute { out: String, input: String, axes: Vec<usize> },
    Matmul { out: String, left: String, right: String },
    /// Batched rank-3 matrix product with equal batch extents.
    BatchMatmul { out: String, left: String, right: String },
    LayerNorm { out: String, input: String, gamma: String, beta: String, axis: usize, epsilon: f32 },
    SoftmaxAxis { out: String, input: String, axis: usize },
    Scale { out: String, input: String, factor: f32 },
    /// Runtime scalar scaling is explicit; other elementwise operators never broadcast.
    ScaleBy { out: String, input: String, scalar: String },
    /// Finite elementwise comparison; tracing this operation for AD is rejected.
    Ge { out: String, left: String, right: String },
    Div { out: String, left: String, right: String },
    Sqrt { out: String, input: String },
    Constant {
        out: String,
        data: Vec<f32>,
    },
    Detach {
        out: String,
        input: String,
    },
    Exp {
        out: String,
        input: String,
    },
    Log {
        out: String,
        input: String,
    },
    LogSoftmax {
        out: String,
        input: String,
    },
    Minimum {
        out: String,
        left: String,
        right: String,
    },
    Clip {
        out: String,
        input: String,
        low: f32,
        high: f32,
    },
    Sum {
        out: String,
        input: String,
    },
    MemoryAlloc {
        out: String,
        slots: usize,
        width: usize,
    },
    MemoryFree {
        memory: String,
    },
    MemoryRead {
        out: String,
        memory: String,
        address: String,
    },
    MemoryWrite {
        memory: String,
        address: String,
        value: String,
    },
    ResourceState {
        out: String,
    },
    Read {
        out: String,
        parameter: String,
    },
    Write {
        parameter: String,
        value: String,
    },
    Linear {
        out: String,
        input: String,
        weight: String,
        bias: Option<String>,
    },
    Add {
        out: String,
        left: String,
        right: String,
    },
    Mul {
        out: String,
        left: String,
        right: String,
    },
    Tanh {
        out: String,
        input: String,
    },
    Sigmoid {
        out: String,
        input: String,
    },
    Relu {
        out: String,
        input: String,
    },
    Softmax {
        out: String,
        input: String,
    },
    ValidateProbabilities {
        out: String,
        input: String,
    },
    Neg {
        out: String,
        input: String,
    },
    Mean {
        out: String,
        input: String,
    },
    Gate {
        out: String,
        probability: String,
        input: String,
    },
    Route {
        out: String,
        probability: String,
        yes: String,
        no: String,
    },
    /// Skip an unevaluated expression only when no differentiation tape is requested.
    InferenceSkip {
        probability: String,
        when_one: bool,
        target: String,
    },
    Slice {
        out: String,
        input: String,
        start: usize,
    },
    Label {
        name: String,
    },
    Branch {
        condition: String,
        positive: String,
        otherwise: String,
    },
    Jump {
        target: String,
    },
    Return {
        value: String,
    },
}

/// Register roles and the oldest IR version that can represent an instruction.
/// Memory handles are separate from tensor inputs and outputs.
pub struct InstructionMetadata<'a> {
    pub minimum_version: u32,
    pub output: Option<&'a str>,
    pub inputs: [Option<&'a str>; 3],
    pub memory: Option<&'a str>,
}

impl Instruction {
    pub fn metadata(&self) -> InstructionMetadata<'_> {
        use Instruction::*;
        let (output, inputs, memory) = match self {
            Conv { out, input, kernel, bias, .. }
            | LayerNorm { out, input, gamma: kernel, beta: bias, .. } =>
                (Some(out), [Some(input), Some(kernel), Some(bias)], None),
            MaxPool2d { out, input, .. }
            | AvgPool2d { out, input, .. }
            | Reshape { out, input }
            | Transpose { out, input }
            | Permute { out, input, .. }
            | SoftmaxAxis { out, input, .. }
            | Scale { out, input, .. }
            | Linear { out, input, .. }
            | Tanh { out, input }
            | Sigmoid { out, input }
            | Relu { out, input }
            | Softmax { out, input }
            | Detach { out, input }
            | Exp { out, input }
            | Log { out, input }
            | Sqrt { out, input }
            | LogSoftmax { out, input }
            | ValidateProbabilities { out, input }
            | Clip { out, input, .. }
            | Sum { out, input }
            | Neg { out, input }
            | Mean { out, input }
            | Slice { out, input, .. } =>
                (Some(out), [Some(input), None, None], None),
            Matmul { out, left, right }
            | BatchMatmul { out, left, right }
            | Add { out, left, right }
            | Mul { out, left, right }
            | Ge { out, left, right }
            | Div { out, left, right }
            | Minimum { out, left, right } =>
                (Some(out), [Some(left), Some(right), None], None),
            ScaleBy { out, input, scalar } =>
                (Some(out), [Some(input), Some(scalar), None], None),
            Constant { out, .. } | Read { out, .. } | ResourceState { out } =>
                (Some(out), [None; 3], None),
            Write { value, .. } | Return { value } =>
                (None, [Some(value), None, None], None),
            Gate { out, probability, input } =>
                (Some(out), [Some(probability), Some(input), None], None),
            Route { out, probability, yes, no } =>
                (Some(out), [Some(probability), Some(yes), Some(no)], None),
            InferenceSkip { probability, .. } =>
                (None, [Some(probability), None, None], None),
            Branch { condition, .. } =>
                (None, [Some(condition), None, None], None),
            MemoryAlloc { out, .. } | MemoryFree { memory: out } =>
                (None, [None; 3], Some(out)),
            MemoryRead { out, memory, address } =>
                (Some(out), [Some(address), None, None], Some(memory)),
            MemoryWrite { memory, address, value } =>
                (None, [Some(address), Some(value), None], Some(memory)),
            Label { .. } | Jump { .. } => (None, [None; 3], None),
        };
        InstructionMetadata {
            minimum_version: self.contract().min_ir_version,
            output: output.map(String::as_str),
            inputs: inputs.map(|input| input.map(String::as_str)),
            memory: memory.map(String::as_str),
        }
    }
}
