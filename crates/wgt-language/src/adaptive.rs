use serde::{Serialize, Deserialize};
mod source;
pub use source::{parse, parse_with_functions};
pub(crate) use source::validate_definition;
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instruction {
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
