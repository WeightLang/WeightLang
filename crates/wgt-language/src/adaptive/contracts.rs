//! Shared metadata validation for source lowering and untrusted adaptive IR.
//! No parameter storage is initialized by this pass.
use super::{Instruction, Parameter, Program, Register};
use crate::tensor::{self, DType, ShapeLimits, TensorType};
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct OperatorContract {
    pub name: &'static str,
    pub min_ir_version: u32,
    pub shape: &'static str,
    pub layout: &'static str,
    pub gradient: &'static str,
    pub effect: &'static str,
}
include!("contract_catalog.rs");

impl Register {
    pub fn validate_tensor_shape(&self, version: u32) -> Result<()> {
        let elements = tensor::validate_shape(&self.shape, ShapeLimits::REGISTER)
            .with_context(|| format!("register {}", self.name))?;
        ensure!(elements == self.len, "register shape/storage mismatch: {}", self.name);
        ensure!(version >= 3 || self.shape == [self.len],
            "multiaxis registers require adaptive IR version 3");
        Ok(())
    }
    pub fn tensor_type(&self) -> Result<TensorType> {
        let ty = TensorType::unnamed(DType::F32, self.shape.clone(), ShapeLimits::REGISTER)?;
        ensure!(ty.elements() == self.len, "register shape/storage mismatch: {}", self.name);
        Ok(ty)
    }
}
impl Parameter {
    /// Metadata only: a static definition deliberately has no materialized data.
    pub fn tensor_type(&self) -> Result<TensorType> {
        Ok(TensorType::unnamed(DType::F32, self.shape.clone(), ShapeLimits::PARAMETER)?)
    }
}

impl Program {
    pub fn validate_tensor_contracts(&self) -> Result<()> {
        ensure!(matches!(self.version, 1 | 2 | 3), "unsupported adaptive IR version");
        let mut registers = HashMap::new();
        for register in &self.registers {
            ensure!(!register.name.is_empty() && registers.insert(register.name.as_str(), register).is_none(),
                "duplicate/empty register: {}", register.name);
            register.validate_tensor_shape(self.version)?;
        }
        let mut parameters = std::collections::HashSet::new();
        for parameter in &self.parameters {
            ensure!(!parameter.name.is_empty() && parameters.insert(parameter.name.as_str()),
                "duplicate/empty parameter: {}", parameter.name);
            tensor::validate_shape(&parameter.shape, ShapeLimits::PARAMETER)
                .with_context(|| format!("parameter {}", parameter.name))?;
        }
        ensure!(registers.contains_key(self.input.as_str()), "unknown input register");
        let reg = |name: &str| -> Result<&Register> {
            registers.get(name).copied().with_context(|| format!("unknown tensor register: {name}"))
        };
        let ty = |name: &str| reg(name)?.tensor_type();
        let same = |a: &str, b: &str| -> Result<()> {
            ensure!(reg(a)?.shape == reg(b)?.shape, "tensor shape mismatch: {a} vs {b}");
            Ok(())
        };
        for (index, instruction) in self.instructions.iter().enumerate() {
            let contract = instruction.contract();
            ensure!(self.version >= contract.min_ir_version,
                "{} requires adaptive IR version {}", contract.name, contract.min_ir_version);
            let validate = || -> Result<()> {
                use Instruction as I;
                match instruction {
                    I::Reshape { out, input } => ensure!(reg(out)?.len == reg(input)?.len, "reshape element count mismatch"),
                    I::Transpose { out, input } => ensure!(reg(out)?.shape == ty(input)?.transpose(ShapeLimits::REGISTER)?.shape(), "transpose shape mismatch"),
                    I::Permute { out, input, axes } => ensure!(reg(out)?.shape == ty(input)?.permute(axes, ShapeLimits::REGISTER)?.shape(), "permute output shape mismatch"),
                    I::Matmul { out, left, right } => ensure!(reg(out)?.shape == ty(left)?.matmul(&ty(right)?, ShapeLimits::REGISTER)?.shape(), "matmul shape mismatch"),
                    I::BatchMatmul { out, left, right } => ensure!(reg(out)?.shape == ty(left)?.batch_matmul(&ty(right)?, ShapeLimits::REGISTER)?.shape(), "batch_matmul shape mismatch"),
                    I::LayerNorm { out, input, gamma, beta, axis, epsilon } => {
                        same(out, input)?;
                        ty(input)?.validate_normalization(&ty(gamma)?, &ty(beta)?, *axis)?;
                        ensure!(epsilon.is_finite() && *epsilon > 0., "LayerNorm epsilon must be positive and finite");
                    }
                    I::SoftmaxAxis { out, input, axis } => {
                        same(out, input)?;
                        ensure!(*axis < reg(input)?.shape.len(), "tensor axis out of bounds");
                    }
                    I::Scale { out, input, factor } => {
                        same(out, input)?; ensure!(factor.is_finite(), "nonfinite scale factor");
                    }
                    I::Constant { out, data } => ensure!(data.len() == reg(out)?.len && data.iter().all(|v| v.is_finite()), "constant shape/value mismatch"),
                    I::Add { out, left, right } | I::Mul { out, left, right } | I::Minimum { out, left, right } => {
                        same(out, left)?; same(left, right)?;
                    }
                    I::Clip { out, input, low, high } => {
                        same(out, input)?; ensure!(low.is_finite() && high.is_finite() && low <= high, "invalid clip interval");
                    }
                    I::Detach { out, input } | I::Exp { out, input } | I::Log { out, input }
                    | I::Neg { out, input } | I::Tanh { out, input } | I::Sigmoid { out, input }
                    | I::Relu { out, input } => same(out, input)?,
                    I::Softmax { out, input } | I::LogSoftmax { out, input } | I::ValidateProbabilities { out, input } => {
                        same(out, input)?; ensure!(reg(input)?.shape.len() == 1, "vector normalization requires rank 1");
                    }
                    I::Mean { out, input } | I::Sum { out, input } => {
                        reg(input)?; ensure!(reg(out)?.shape == [1], "reduction output must be f32[1]");
                    }
                    I::Gate { out, probability, input } => {
                        same(out, input)?;
                        ensure!(reg(probability)?.shape == [1] || reg(probability)?.shape == reg(input)?.shape, "gate shape mismatch");
                    }
                    I::Route { out, probability, yes, no } => {
                        same(out, yes)?; same(yes, no)?;
                        ensure!(reg(probability)?.shape == [1] || reg(probability)?.shape == reg(yes)?.shape, "route shape mismatch");
                    }
                    I::Slice { out, input, start } => {
                        let input_len = reg(input)?.len;
                        ensure!(reg(out)?.shape.len() == 1 && start.checked_add(reg(out)?.len).is_some_and(|end| end <= input_len), "slice shape/range mismatch");
                    }
                    I::Conv { out, input, kernel, bias, stride, padding, dilation, groups } => {
                        let geometry = super::ConvGeometry::new(&reg(input)?.shape, &reg(kernel)?.shape, &reg(bias)?.shape, stride, padding, dilation, *groups)?;
                        ensure!(reg(out)?.shape == geometry.shape(), "conv output shape mismatch");
                    }
                    I::MaxPool2d { out, input, kernel, stride } | I::AvgPool2d { out, input, kernel, stride } => {
                        let geometry = super::Pool2dGeometry::new(&reg(input)?.shape, kernel, stride)?;
                        ensure!(reg(out)?.shape == geometry.shape(), "pool output shape mismatch");
                    }
                    // Parameter ownership, mutable state, memory handles and
                    // control-flow validity remain the runtime's responsibility.
                    I::Read { out, .. } | I::ResourceState { out } => { reg(out)?; }
                    I::Write { value, .. } | I::Return { value } => { reg(value)?; }
                    I::Linear { out, input, .. } => { reg(out)?; reg(input)?; }
                    I::MemoryAlloc { out, .. } => { reg(out)?; }
                    I::MemoryFree { memory } => { reg(memory)?; }
                    I::MemoryRead { out, memory, address } => { reg(out)?; reg(memory)?; reg(address)?; }
                    I::MemoryWrite { memory, address, value } => { reg(memory)?; reg(address)?; reg(value)?; }
                    I::InferenceSkip { probability, .. } => { reg(probability)?; }
                    I::Branch { condition, .. } => { reg(condition)?; }
                    I::Label { .. } | I::Jump { .. } => {}
                }
                Ok(())
            };
            validate().with_context(|| format!("tensor contract at instruction {index} ({})", contract.name))?;
        }
        if let Some(loss) = &self.loss {
            ensure!(loss.loss.is_none(), "nested loss programs are unsupported");
            loss.validate_tensor_contracts().context("loss tensor contracts")?;
        }
        Ok(())
    }
}
