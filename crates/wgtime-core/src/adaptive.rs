//! Bounded, stateful forward programs. Writes are visible immediately inside a
//! call, and committed only when that call returns successfully.
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

pub use wgt_language::adaptive::{parse, parse_with_functions};
mod autodiff;
pub use autodiff::ForwardPass;
mod training;
mod reload;
pub mod optimization;
pub mod transport;
mod workspace;
mod tensor;
mod unary;
use unary::Unary;
use tensor::TensorView;
pub use workspace::ExecutionProgress;

pub use wgt_language::adaptive::{Program, Register, Parameter, Instruction};
#[derive(Debug, Clone, Copy)]
enum Op {
    Conv(usize, usize, usize, usize, wgt_language::adaptive::ConvGeometry),
    MaxPool2d(usize, usize, wgt_language::adaptive::Pool2dGeometry),
    AvgPool2d(usize, usize, wgt_language::adaptive::Pool2dGeometry),
    Reshape(usize, usize),
    Transpose(usize, usize, usize, usize),
    Permute(usize, usize, [usize; 8], [usize; 8], usize),
    Matmul(usize, usize, usize, usize, usize, usize),
    BatchMatmul(usize, usize, usize, usize, usize, usize, usize),
    LayerNorm(usize, usize, usize, usize, tensor::AxisGroups, f32),
    SoftmaxAxis(usize, usize, tensor::AxisGroups),
    Scale(usize, usize, f32),

    Constant(usize, usize),
    Minimum(usize, usize, usize),
    Clip(usize, usize, f32, f32),
    Sum(usize, usize),
    MemoryAlloc(usize, usize, usize),
    MemoryFree(usize),
    MemoryRead(usize, usize, usize),
    MemoryWrite(usize, usize, usize),
    ResourceState(usize),
    Read(usize, usize),
    Write(usize, usize),
    Linear(usize, usize, usize, Option<usize>),
    Binary(usize, usize, usize, bool),
    Unary(usize, usize, Unary),
    Mean(usize, usize),
    Blend(usize, usize, usize, Option<usize>),
    InferenceSkip(usize, bool, usize),
    Slice(usize, usize, usize),
    Label,
    Branch(usize, usize, usize),
    Jump(usize),
    Return(usize),
}

/// CPU-only experimental runtime; owns persistent weights and per-call registers.
/// Budget covers f32 storage (including rollback and scratch), not program metadata.
pub struct AdaptiveSession {
    program: Program,
    ops: Vec<Op>,
    registers: Vec<Vec<f32>>,
    /// Non-owning strided views over a canonical register; materialized lazily.
    views: Vec<Option<TensorView>>,
    initialized: Vec<bool>,
    backup: Vec<Vec<f32>>,
    scratch: Vec<f32>,
    input: usize,
    output_len: usize,
    managed_bytes: usize,
    revision: u64,
    collection: crate::collection::CollectionState,
    steps: u64,
    optimizer: Option<training::State>,
    custom_optimizer: Option<optimization::CustomState>,
    workspace: workspace::Workspace,
    memory_limit: usize,
    pc: usize,
    wrote: bool,
    pending: Option<Box<AdaptiveSession>>,
    execution_limit: u64,
    compiled: Option<crate::compiled_forward::CompiledForward>,
    compiled_forward_calls: u64,
    compiled_tape_calls: u64,
}

fn names<'a>(items: impl Iterator<Item = &'a str>) -> Result<HashMap<String, usize>> {
    let mut result = HashMap::new();
    for (index, name) in items.enumerate() {
        ensure!(
            !name.is_empty() && result.insert(name.into(), index).is_none(),
            "duplicate/empty identity: {name}"
        );
    }
    Ok(result)
}

impl AdaptiveSession {
    pub fn begin_collection(&self) -> Result<crate::collection::CollectionToken> {
        self.ensure_idle()?;
        self.collection.begin()
    }
    pub(crate) fn ensure_mutable(&self) -> Result<()> { self.collection.ensure_mutable() }
    pub fn prepare(program: Program, budget_bytes: usize) -> Result<Self> {
        ensure!(matches!(program.version, 1 | 2 | 3), "unsupported adaptive IR version");
        ensure!(program.version >= 2 || (program.output_type.is_plain()
            && !program.instructions.iter().any(|i| i.metadata().minimum_version == 2)),
            "categorical output metadata and validation require adaptive IR version 2");
        ensure!(program.instructions.iter().all(|i| i.metadata().minimum_version <= program.version),
            "tensor operations require adaptive IR version 3");
        ensure!(!program.instructions.is_empty(), "empty adaptive program");
        if let Some(loss) = &program.loss {
            ensure!(
                loss.loss.is_none() && loss.parameters.is_empty(),
                "loss must be a parameter-free tensor program"
            );
        }
        let regs = names(program.registers.iter().map(|r| r.name.as_str()))?;
        let params = names(program.parameters.iter().map(|p| p.name.as_str()))?;
        let input = *regs.get(&program.input).context("unknown input register")?;
        let mut labels = HashMap::new();
        for (i, op) in program.instructions.iter().enumerate() {
            if let Instruction::Label { name } = op {
                ensure!(
                    !name.is_empty() && labels.insert(name.as_str(), i).is_none(),
                    "duplicate/empty label"
                );
            }
        }
        let reg = |s: &str| {
            regs.get(s)
                .copied()
                .with_context(|| format!("unknown register: {s}"))
        };
        let dest = |s: &str| -> Result<usize> {
            let r = reg(s)?;
            ensure!(r != input, "input register is read-only");
            Ok(r)
        };
        let param = |s: &str| {
            params
                .get(s)
                .copied()
                .with_context(|| format!("unknown parameter: {s}"))
        };
        let label = |s: &str| {
            labels
                .get(s)
                .copied()
                .with_context(|| format!("unknown label: {s}"))
        };
        let len = |r: usize| program.registers[r].len;
        let memories = workspace::validate(&program, &regs)?;
        let mut floats = 0usize;
        for p in &program.parameters {
            if let Some(domain) = &p.learn {
                ensure!(p.shape.len()==1 && !p.probability && !p.frozen, "learn parameter must be a trainable vector");
                ensure!(p.data.len() == p.shape[0], "learn shadow shape mismatch");
                for &value in &p.data {domain.validate(value).with_context(|| format!("learn {}", p.name))?;}
            }
            if p.probability {
                ensure!(p.shape == [1], "prob parameter must be scalar");
                ensure!(
                    !p.frozen || p.data.iter().all(|v| (0. ..=1.).contains(v)),
                    "frozen prob must be in [0,1]"
                );
            }
            let size = crate::elements(&p.shape)?;
            ensure!(
                size == p.data.len() && p.data.iter().all(|v| v.is_finite()),
                "invalid parameter: {}",
                p.name
            );
            floats = floats
                .checked_add(size.checked_mul(2).context("parameter size overflow")?)
                .context("budget overflow")?;
        }
        for instruction in &program.instructions {
            if let Instruction::Constant { data, .. } = instruction {
                floats = floats
                    .checked_add(data.len())
                    .context("constant size overflow")?;
            }
        }
        let mut maximum = 0;
        for r in &program.registers {
            ensure!(r.len > 0, "register length must be positive");
            ensure!(crate::elements(&r.shape)? == r.len && r.shape.len() <= 8 && r.len <= 16_777_216 && r.shape.iter().all(|d| *d <= 16384), "register shape/storage mismatch");
            // Older IR carries explicit geometry too. Its operations retain their
            // historical flat semantics; only v3 admits multidimensional registers.
            ensure!(program.version >= 3 || r.shape == [r.len], "multiaxis registers require adaptive IR version 3");

            floats = floats
                .checked_add(r.len)
                .context("register size overflow")?;
            maximum = maximum.max(r.len);
        }
        let managed_bytes = crate::bytes(
            floats
                .checked_add(maximum)
                .context("scratch size overflow")?,
        )?;
        ensure!(
            managed_bytes <= budget_bytes,
            "adaptive managed memory budget exceeded: need {managed_bytes}, have {budget_bytes}"
        );
        let mut ops = Vec::new();
        let mut output_len = None;
        for (pc, instruction) in program.instructions.iter().enumerate() {
            let op = (|| -> Result<Op> {
                Ok(match instruction {
                    Instruction::Conv { out,input,kernel,bias,stride,padding,dilation,groups } => {
                        let (r,x,k,b)=(dest(out)?,reg(input)?,reg(kernel)?,reg(bias)?);
                        let g=wgt_language::adaptive::ConvGeometry::new(&program.registers[x].shape,&program.registers[k].shape,&program.registers[b].shape,stride,padding,dilation,*groups)?;
                        ensure!(program.registers[r].shape==g.shape(),"conv output shape mismatch");
                        Op::Conv(r,x,k,b,g)
                    },
                    Instruction::MaxPool2d { out, input, kernel, stride } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        let g = wgt_language::adaptive::Pool2dGeometry::new(&program.registers[x].shape, kernel, stride)?;
                        ensure!(program.registers[r].shape == g.shape(), "max_pool2d output shape mismatch");
                        Op::MaxPool2d(r, x, g)
                    },
                    Instruction::AvgPool2d { out, input, kernel, stride } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        let g = wgt_language::adaptive::Pool2dGeometry::new(&program.registers[x].shape, kernel, stride)?;
                        ensure!(program.registers[r].shape == g.shape(), "avg_pool2d output shape mismatch");
                        Op::AvgPool2d(r, x, g)
                    },
                    Instruction::Reshape { out, input } => {

                        let (r,x) = (dest(out)?,reg(input)?);
                        ensure!(len(r) == len(x), "reshape element count mismatch");
                        Op::Reshape(r,x)
                    },
                    Instruction::Transpose { out, input } => {
                        let (r,x) = (dest(out)?,reg(input)?);
                        let shape = &program.registers[x].shape;
                        ensure!(shape.len() == 2 && program.registers[r].shape == [shape[1],shape[0]], "transpose shape mismatch");
                        Op::Transpose(r,x,shape[0],shape[1])
                    },
                    Instruction::Permute { out, input, axes } => {
                        let (r,x) = (dest(out)?,reg(input)?);
                        let shape = &program.registers[x].shape;
                        ensure!(axes.len() == shape.len() && (1..=8).contains(&axes.len()),
                            "permute rank mismatch");
                        let mut seen = [false; 8];
                        let mut order = [0usize; 8];
                        for (out_axis, &input_axis) in axes.iter().enumerate() {
                            ensure!(input_axis < shape.len() && !seen[input_axis],
                                "permute axes must form a bijection");
                            seen[input_axis] = true;
                            order[out_axis] = input_axis;
                        }
                        let output_shape = axes.iter().map(|&axis| shape[axis]).collect::<Vec<_>>();
                        ensure!(program.registers[r].shape == output_shape, "permute output shape mismatch");
                        let mut input_shape = [1usize; 8];
                        input_shape[..shape.len()].copy_from_slice(shape);
                        Op::Permute(r,x,input_shape,order,axes.len())
                    },
                    Instruction::Matmul { out, left, right } => {
                        let (r,a,b) = (dest(out)?,reg(left)?,reg(right)?);
                        let sa = &program.registers[a].shape; let sb = &program.registers[b].shape;
                        ensure!(sa.len() == 2 && sb.len() == 2 && sa[1] == sb[0] && program.registers[r].shape == [sa[0],sb[1]], "matmul shape mismatch");
                        Op::Matmul(r,a,b,sa[0],sa[1],sb[1])
                    },
                    Instruction::BatchMatmul { out, left, right } => {
                        let (r,a,b) = (dest(out)?,reg(left)?,reg(right)?);
                        let sa = &program.registers[a].shape;
                        let sb = &program.registers[b].shape;
                        ensure!(sa.len() == 3 && sb.len() == 3
                            && sa[0] == sb[0] && sa[2] == sb[1]
                            && program.registers[r].shape == [sa[0],sa[1],sb[2]],
                            "batch_matmul shape mismatch");
                        Op::BatchMatmul(r,a,b,sa[0],sa[1],sa[2],sb[2])
                    },
                    Instruction::LayerNorm { out, input, gamma, beta, axis, epsilon } => {
                        let (r,x,g,b) = (dest(out)?,reg(input)?,reg(gamma)?,reg(beta)?);
                        let groups = tensor::AxisGroups::new(&program.registers[x].shape, *axis)?;
                        ensure!(program.registers[g].shape == [groups.width]
                            && program.registers[b].shape == [groups.width]
                            && program.registers[r].shape == program.registers[x].shape,
                            "LayerNorm tensor geometry mismatch");
                        ensure!(epsilon.is_finite() && *epsilon > 0., "LayerNorm epsilon must be positive and finite");
                        Op::LayerNorm(r,x,g,b,groups,*epsilon)
                    },
                    Instruction::SoftmaxAxis { out, input, axis } => {
                        let (r,x) = (dest(out)?,reg(input)?);
                        ensure!(program.registers[r].shape == program.registers[x].shape, "axis softmax shape mismatch");
                        Op::SoftmaxAxis(r,x,tensor::AxisGroups::new(&program.registers[x].shape,*axis)?)
                    },
                    Instruction::Scale { out, input, factor } => {
                        let (r,x) = (dest(out)?,reg(input)?);
                        ensure!(factor.is_finite() && program.registers[r].shape == program.registers[x].shape, "scale shape/factor mismatch");
                        Op::Scale(r,x,*factor)
                    },
                    Instruction::MemoryAlloc { out, slots, width } => {
                        Op::MemoryAlloc(dest(out)?, *slots, *width)
                    }
                    Instruction::MemoryFree { memory } => Op::MemoryFree(reg(memory)?),
                    Instruction::MemoryRead {
                        out,
                        memory,
                        address,
                    } => {
                        let (r, m, a) = (dest(out)?, reg(memory)?, reg(address)?);
                        let (slots, width) = memories[&m];
                        ensure!(
                            len(r) == width && len(a) == slots,
                            "memory_read shape mismatch"
                        );
                        Op::MemoryRead(r, m, a)
                    }
                    Instruction::MemoryWrite {
                        memory,
                        address,
                        value,
                    } => {
                        let (m, a, v) = (reg(memory)?, reg(address)?, reg(value)?);
                        let (slots, width) = memories[&m];
                        ensure!(
                            len(a) == slots && len(v) == width,
                            "memory_write shape mismatch"
                        );
                        Op::MemoryWrite(m, a, v)
                    }
                    Instruction::ResourceState { out } => {
                        let r = dest(out)?;
                        ensure!(len(r) == 3, "resource_state requires f32[3]");
                        Op::ResourceState(r)
                    }
                    Instruction::Read { out, parameter } => {
                        let (r, p) = (dest(out)?, param(parameter)?);
                        ensure!(
                            len(r) == program.parameters[p].data.len(),
                            "read length mismatch"
                        );
                        Op::Read(r, p)
                    }
                    Instruction::Write { parameter, value } => {
                        let (p, r) = (param(parameter)?, reg(value)?);
                        ensure!(
                            !program.parameters[p].frozen,
                            "cannot write frozen parameter: {parameter}"
                        );
                        ensure!(
                            !program.parameters[p].probability && program.parameters[p].learn.is_none(),
                            "write to prob/learn is unsupported; use optimizer updates"
                        );
                        ensure!(
                            len(r) == program.parameters[p].data.len(),
                            "write length mismatch"
                        );
                        Op::Write(p, r)
                    }
                    Instruction::Linear {
                        out,
                        input,
                        weight,
                        bias,
                    } => {
                        let (r, x, w) = (dest(out)?, reg(input)?, param(weight)?);
                        ensure!(program.parameters[w].learn.is_none(), "linear cannot consume raw learn shadow");
                        ensure!(
                            program.parameters[w].shape == [len(r), len(x)],
                            "linear weight must be [output, input]"
                        );
                        let b = bias.as_deref().map(param).transpose()?;
                        if let Some(b) = b {
                            ensure!(program.parameters[b].learn.is_none(), "linear bias cannot consume raw learn shadow");
                            ensure!(
                                program.parameters[b].shape == [len(r)],
                                "bias shape mismatch"
                            );
                        }
                        Op::Linear(r, x, w, b)
                    }
                    Instruction::Add { out, left, right }
                    | Instruction::Mul { out, left, right } => {
                        let (r, a, b) = (dest(out)?, reg(left)?, reg(right)?);
                        ensure!(
                            len(r) == len(a) && len(r) == len(b),
                            "elementwise length mismatch"
                        );
                        Op::Binary(r, a, b, matches!(instruction, Instruction::Mul { .. }))
                    }
                    Instruction::Constant { out, data } => {
                        let r = dest(out)?;
                        ensure!(
                            data.len() == len(r) && data.iter().all(|v| v.is_finite()),
                            "invalid tensor constant"
                        );
                        Op::Constant(r, ops.len())
                    }
                    Instruction::Minimum { out, left, right } => {
                        let (r, a, b) = (dest(out)?, reg(left)?, reg(right)?);
                        ensure!(
                            len(r) == len(a) && len(a) == len(b),
                            "minimum shape mismatch"
                        );
                        Op::Minimum(r, a, b)
                    }
                    Instruction::Clip {
                        out,
                        input,
                        low,
                        high,
                    } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(
                            len(r) == len(x) && low.is_finite() && high.is_finite() && low <= high,
                            "invalid clip shape or bounds"
                        );
                        Op::Clip(r, x, *low, *high)
                    }
                    Instruction::Sum { out, input } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(len(r) == 1, "sum output must be f32[1]");
                        Op::Sum(r, x)
                    }
                    Instruction::Detach { out, input }
                    | Instruction::Exp { out, input }
                    | Instruction::Log { out, input }
                    | Instruction::LogSoftmax { out, input }
                    | Instruction::Tanh { out, input }
                    | Instruction::Sigmoid { out, input }
                    | Instruction::Softmax { out, input }
                    | Instruction::ValidateProbabilities { out, input }
                    | Instruction::Neg { out, input }
                    | Instruction::Relu { out, input } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(len(r) == len(x), "unary length mismatch");
                        let kind = match instruction {
                            Instruction::Tanh { .. } => Unary::Tanh,
                            Instruction::Sigmoid { .. } => Unary::Sigmoid,
                            Instruction::Relu { .. } => Unary::Relu,
                            Instruction::Softmax { .. } => Unary::Softmax,
                            Instruction::Neg { .. } => Unary::Neg,
                            Instruction::Detach { .. } => Unary::Detach,
                            Instruction::Exp { .. } => Unary::Exp,
                            Instruction::Log { .. } => Unary::Log,
                            Instruction::LogSoftmax { .. } => Unary::LogSoftmax,
                            Instruction::ValidateProbabilities { .. } => Unary::ValidateProbabilities,
                            _ => unreachable!(),
                        };
                        Op::Unary(r, x, kind)
                    }
                    Instruction::Mean { out, input } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(len(r) == 1, "mean output must be f32[1]");
                        Op::Mean(r, x)
                    }
                    Instruction::Gate {
                        out,
                        probability,
                        input,
                    } => {
                        let (r, p, x) = (dest(out)?, reg(probability)?, reg(input)?);
                        ensure!(
                            len(r) == len(x) && (len(p) == 1 || len(p) == len(r)),
                            "gate shape mismatch"
                        );
                        Op::Blend(r, p, x, None)
                    }
                    Instruction::Route {
                        out,
                        probability,
                        yes,
                        no,
                    } => {
                        let (r, p, a, b) = (dest(out)?, reg(probability)?, reg(yes)?, reg(no)?);
                        ensure!(
                            len(r) == len(a)
                                && len(r) == len(b)
                                && (len(p) == 1 || len(p) == len(r)),
                            "route shape mismatch"
                        );
                        Op::Blend(r, p, a, Some(b))
                    }
                    Instruction::InferenceSkip {
                        probability,
                        when_one,
                        target,
                    } => Op::InferenceSkip(reg(probability)?, *when_one, label(target)?),
                    Instruction::Slice { out, input, start } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(
                            start.checked_add(len(r)).is_some_and(|end| end <= len(x)),
                            "slice out of bounds"
                        );
                        Op::Slice(r, x, *start)
                    }
                    Instruction::Label { .. } => Op::Label,
                    Instruction::Branch {
                        condition,
                        positive,
                        otherwise,
                    } => {
                        let c = reg(condition)?;
                        ensure!(
                            len(c) == 1,
                            "branch requires one scalar in a length-one register"
                        );
                        Op::Branch(c, label(positive)?, label(otherwise)?)
                    }
                    Instruction::Jump { target } => Op::Jump(label(target)?),
                    Instruction::Return { value } => {
                        let r = reg(value)?;
                        ensure!(
                            output_len.is_none_or(|n| n == len(r)),
                            "all returns must have equal length"
                        );
                        output_len = Some(len(r));
                        Op::Return(r)
                    }
                })
            })()
            .with_context(|| format!("instruction {pc}"))?;
            if program.version >= 3 {
                let shape = |r: usize| &program.registers[r].shape;
                match op {
                    Op::Read(r,_) => ensure!(*shape(r) == [program.registers[r].len], "parameter reads require a vector; reshape explicitly"),
                    Op::Write(_,r) => ensure!(*shape(r) == [program.registers[r].len], "parameter writes require a vector; reshape explicitly"),
                    Op::Binary(r,a,b,_) | Op::Minimum(r,a,b) => ensure!(shape(r) == shape(a) && shape(a) == shape(b), "elementwise geometry mismatch"),
                    Op::Unary(r,x,k) => {
                        ensure!(shape(r) == shape(x), "unary geometry mismatch");
                        ensure!(!matches!(k, Unary::Softmax | Unary::LogSoftmax | Unary::ValidateProbabilities) || shape(x).len() == 1, "vector softmax/validation requires rank 1");
                    },
                    Op::Clip(r,x,_,_) => ensure!(shape(r) == shape(x), "clip geometry mismatch"),
                    Op::Linear(r,x,_,_) => ensure!(shape(r).len() == 1 && shape(x).len() == 1, "linear requires vectors"),
                    Op::Slice(r,x,_) => ensure!(shape(r).len() == 1 && shape(x).len() == 1, "slice requires vectors"),
                    Op::Blend(r,p,a,b) => ensure!(shape(r) == shape(a) && b.is_none_or(|b| shape(b) == shape(a)) && (shape(p) == &[1] || shape(p) == shape(a)), "blend geometry mismatch"),
                    Op::Return(r) => ensure!(shape(r).len() == 1, "model output requires explicit vector reshape"),
                    _ => {},
                }
            }
            ops.push(op);
        }
        let output_len = output_len.context("program requires a return")?;
        let view_registers = ops.iter().filter_map(|op| match op {
            Op::Reshape(register, _) | Op::Transpose(register, _, _, _) | Op::Permute(register, ..) => Some(*register),
            _ => None,
        }).collect::<std::collections::HashSet<_>>();
        let registers = program
            .registers
            .iter()
            .enumerate()
            .map(|(index, register)| {
                if view_registers.contains(&index) { Ok(Vec::new()) } else { crate::zero_buffer(register.len) }
            })
            .collect::<Result<_>>()?;
        let backup = program
            .parameters
            .iter()
            .map(|p| crate::zero_buffer(p.data.len()))
            .collect::<Result<_>>()?;
        Ok(Self {
            workspace: workspace::Workspace::new(program.registers.len()),
            memory_limit: budget_bytes,
            pc: 0,
            wrote: false,
            pending: None,
            execution_limit: 0,
            compiled: None,
            compiled_forward_calls: 0,
            compiled_tape_calls: 0,
            initialized: vec![false; program.registers.len()],
            views: vec![None; program.registers.len()],
            program,
            ops,
            registers,
            backup,
            scratch: crate::zero_buffer(maximum)?,
            input,
            output_len,
            managed_bytes,
            revision: 0,
            collection: crate::collection::CollectionState::default(),
            steps: 0,
            optimizer: None,
            custom_optimizer: None,
        })
    }

    /// Require native forward execution for this instance, including training tapes.
    /// Compilation and workspace admission complete before changing execution mode.
    pub fn enable_compiled(&mut self) -> Result<()> {
        self.ensure_idle()?;
        if self.compiled.is_some() { return Ok(()); }
        let remaining = self.memory_limit.checked_sub(self.managed_bytes)
            .context("compiled workspace exceeds session memory budget")?;
        let compiled = crate::compiled_forward::CompiledForward::compile(&self.program, remaining)?;
        if let Some(loss) = &self.program.loss {
            // Validate the complete executable definition before publishing an instance.
            crate::compiled_forward::CompiledForward::compile(loss, self.memory_limit)?;
        }
        self.managed_bytes = self.managed_bytes.checked_add(compiled.workspace_bytes())
            .context("compiled memory size overflow")?;
        self.compiled = Some(compiled);
        Ok(())
    }

    pub fn is_compiled(&self) -> bool { self.compiled.is_some() }
    pub fn execution_backend(&self) -> &'static str {
        if self.is_compiled() { "native-cpu" } else { "reference-interpreter" }
    }
    /// Successful native calls and the subset that recorded a gradient tape.
    pub fn compiled_forward_counts(&self) -> (u64, u64) {
        (self.compiled_forward_calls, self.compiled_tape_calls)
    }

    pub fn managed_bytes(&self) -> usize {
        self.managed_bytes
    }
    pub fn input_len(&self) -> usize {
        self.registers[self.input].len()
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    /// Includes the current persistent weights and executable instructions. Serialize
    /// explicitly to retain state; running never writes files or trains a controller.
    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn parameter(&self, name: &str) -> Result<&[f32]> {
        Ok(&self
            .program
            .parameters
            .iter()
            .find(|p| p.name == name)
            .context("unknown parameter")?
            .data)
    }

    /// Replace one live parameter between executions; existing tapes remain snapshots.
    pub fn replace_parameter(&mut self, name: &str, values: &[f32]) -> Result<()> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        let p = self
            .program
            .parameters
            .iter_mut()
            .find(|p| p.name == name)
            .context("unknown parameter")?;
        ensure!(!p.frozen, "frozen parameter");
        ensure!(values.len() == p.data.len(), "parameter length mismatch");
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite parameter");
        if let Some(domain) = &p.learn { for &value in values {domain.validate(value)?;} }
        let revision = self.revision.checked_add(1).context("revision overflow")?;
        p.data.copy_from_slice(values);
        self.revision = revision;
        Ok(())
    }

    fn ready_view(&self, index: usize) -> Result<()> {
        ensure!(
            self.initialized[index],
            "read of uninitialized register: {}",
            self.program.registers[index].name
        );
        Ok(())
    }
    fn read_value(&self, register: usize, logical_index: usize) -> f32 {
        if let Some(view) = self.views[register] {
            self.registers[view.base][view.physical_index(logical_index)]
        } else {
            self.registers[register][logical_index]
        }
    }
    fn materialize_view(&mut self, register: usize) {
        let Some(view) = self.views[register] else { return; };
        let len = self.program.registers[register].len;
        self.registers[register].resize(len, 0.);
        for index in 0..len {
            let value = self.registers[view.base][view.physical_index(index)];
            self.registers[register][index] = value;
        }
        self.views[register] = None;
    }
    fn ready(&mut self, index: usize) -> Result<()> {
        self.ready_view(index)?;
        self.materialize_view(index);
        Ok(())
    }
    fn detach_views_of(&mut self, base: usize) {
        for register in 0..self.views.len() {
            if self.views[register].is_some_and(|view| view.base == base) {
                self.materialize_view(register);
            }
        }
    }
    fn make_view(&mut self, op: Op, output: usize) -> Result<TensorView> {
        let input = match op {
            Op::Reshape(_, input) | Op::Transpose(_, input, ..) | Op::Permute(_, input, ..) => input,
            _ => bail!("instruction is not a tensor view"),
        };
        self.ready_view(input)?;
        if matches!(op, Op::Reshape(..)) && self.views[input].is_some_and(|view| !view.is_contiguous()) {
            self.materialize_view(input);
        }
        let view = self.views[input]
            .unwrap_or_else(|| TensorView::contiguous(input, &self.program.registers[input].shape));
        Ok(match op {
            Op::Reshape(..) => view.reshaped(&self.program.registers[output].shape),
            Op::Transpose(..) => view.permuted(&[1, 0]),
            Op::Permute(_, _, _, axes, rank) => view.permuted(&axes[..rank]),
            _ => unreachable!(),
        })
    }

    /// One forward transaction; state persists between successful calls. Invalid
    /// arithmetic, undefined registers, fallthrough and exhausted budgets roll back
    /// all weight writes. max_steps counts instruction work, including jumps and
    /// the N*D surcharge for allocating/accessing task-local memory.
    pub fn run(&mut self, input: &[f32], max_steps: u64) -> Result<&[f32]> {
        if self.program.instructions.iter().any(|i| matches!(i, Instruction::Write { .. })) {
            self.ensure_mutable()?;
        }
        self.begin(input, max_steps)?;
        let result = self.execute(max_steps, None).and_then(|(r, wrote)| {
            if wrote {
                self.revision = self.revision.checked_add(1).context("revision overflow")?;
            }
            Ok(r)
        });
        match result {
            Ok(r) => {
                self.workspace.clear();
                Ok(&self.registers[r])
            }
            Err(error) => {
                self.restore();
                Err(error)
            }
        }
    }

    /// Read-only inference: execute lazily without recording a gradient tape or
    /// committing forward writes. Used for action sampling, not differentiation.
    pub fn evaluate(&mut self, input: &[f32], max_steps: u64) -> Result<Vec<f32>> {
        self.begin(input, max_steps)?;
        let result = self
            .execute(max_steps, None)
            .map(|(r, _)| self.registers[r].clone());
        self.restore();
        result
    }

    /// Evaluate a contiguous row-major batch shaped [batch_size, input_len].
    /// Samples share the same parameter snapshot and are executed independently in
    /// input order. max_steps is a total budget across the whole batch.
    pub fn evaluate_batch(
        &mut self,
        inputs: &[f32],
        batch_size: usize,
        max_steps: u64,
    ) -> Result<Vec<f32>> {
        ensure!(batch_size > 0, "batch_size must be positive");
        let input_width = self.input_len();
        let expected = batch_size.checked_mul(input_width)
            .context("batch input size overflow")?;
        ensure!(inputs.len() == expected && inputs.iter().all(|v| v.is_finite()),
            "invalid batch input: expected [batch_size, input_len]");
        ensure!(max_steps > 0, "max_steps must be positive");
        let output_capacity = batch_size.checked_mul(self.output_len())
            .context("batch output size overflow")?;
        let mut outputs = Vec::with_capacity(output_capacity);
        let mut remaining = max_steps;
        for sample in inputs.chunks_exact(input_width) {
            let output = self.evaluate(sample, remaining)?;
            let used = self.steps;
            ensure!(used <= remaining, "batch execution budget accounting error");
            remaining -= used;
            outputs.extend_from_slice(&output);
        }
        Ok(outputs)
    }

    fn begin(&mut self, input: &[f32], max_steps: u64) -> Result<()> {
        self.ensure_idle()?;
        self.steps = 0;
        self.pc = 0;
        self.wrote = false;
        self.workspace.clear();
        ensure!(
            input.len() == self.input_len() && input.iter().all(|v| v.is_finite()),
            "invalid adaptive input"
        );
        ensure!(max_steps > 0, "max_steps must be positive");
        self.initialized.fill(false);
        self.views.fill(None);
        self.initialized[self.input] = true;
        self.registers[self.input].copy_from_slice(input);
        for (p, b) in self.program.parameters.iter().zip(&mut self.backup) {
            b.copy_from_slice(&p.data);
        }
        Ok(())
    }

    fn restore(&mut self) {
        self.workspace.clear();
        for (p, b) in self.program.parameters.iter_mut().zip(&self.backup) {
            p.data.copy_from_slice(b);
        }
    }

    fn execute(
        &mut self,
        max_steps: u64,
        tape: Option<&mut autodiff::Tape>,
    ) -> Result<(usize, bool)> {
        if let Some(compiled) = &mut self.compiled {
            let parameters: Vec<&[f32]> = self.program.parameters.iter()
                .map(|parameter| parameter.data.as_slice()).collect();
            let ops = &self.ops;
            let mut returned = None;
            let mut tape = tape;
            let traced = tape.is_some();
            let result = compiled.run_with_hook(&self.registers[self.input], &parameters, max_steps, traced, &mut |index, value| {
                let op = ops[index];
                if let Op::Return(register) = op {
                    returned = Some(register);
                } else if let Some(tape) = tape.as_deref_mut() {
                    tape.record(op, value)?;
                }
                Ok(())
            });
            self.steps = compiled.last_steps();
            let output = result?;
            let register = returned.context("compiled forward did not return")?;
            self.registers[register].resize(output.len(), 0.);
            self.registers[register].copy_from_slice(&output);
            self.compiled_forward_calls = self.compiled_forward_calls.saturating_add(1);
            if traced { self.compiled_tape_calls = self.compiled_tape_calls.saturating_add(1); }
            return Ok((register, false));
        }
        self.execute_slice(max_steps, tape, u64::MAX)?
            .context("unexpected execution suspension")
    }

    fn execute_slice(
        &mut self,
        max_steps: u64,
        mut tape: Option<&mut autodiff::Tape>,
        quantum: u64,
    ) -> Result<Option<(usize, bool)>> {
        let mut count = 0;
        loop {
            if count == quantum {
                return Ok(None);
            }
            count += 1;
            let pc = self.pc;
            ensure!(
                self.steps < max_steps,
                "adaptive instruction budget exceeded"
            );
            let op = *self
                .ops
                .get(pc)
                .context("adaptive program fell through without return")?;
            let cost = match op {
                Op::Conv(_,_,_,_,g) => g.work()? as u64,
                Op::MaxPool2d(_,_,g) | Op::AvgPool2d(_,_,g) => g.work()? as u64,
                Op::Reshape(r,_) | Op::Permute(r,..) => (self.program.registers[r].len as u64).saturating_add(1),
                Op::Transpose(_,_,m,n) => m.checked_mul(n).and_then(|v|v.checked_add(1)).context("transpose work overflow")? as u64,
                Op::Matmul(_,_,_,m,k,n) => m.checked_mul(k).and_then(|v|v.checked_mul(n)).and_then(|v|v.checked_add(1)).context("matmul work overflow")? as u64,
                Op::BatchMatmul(_,_,_,batch,m,k,n) => batch.checked_mul(m).and_then(|v|v.checked_mul(k)).and_then(|v|v.checked_mul(n)).and_then(|v|v.checked_add(1)).context("batch_matmul work overflow")? as u64,
                Op::LayerNorm(_,_,_,_,groups,_) => groups.len().checked_mul(3).and_then(|v|v.checked_add(1)).context("LayerNorm work overflow")? as u64,
                _ => self.workspace.cost(op)?,
            };

            ensure!(
                cost <= max_steps - self.steps,
                "adaptive instruction budget exceeded"
            );
            self.steps += cost;
            let mut output = None;
            match op {
                Op::MemoryAlloc(m, slots, width) => {
                    self.workspace.allocate(
                        m,
                        slots,
                        width,
                        self.memory_limit - self.managed_bytes,
                    )?;
                    if let Some(tape) = tape.as_deref_mut() {
                        tape.memory_allocate(m, self.workspace.data(m)?)?;
                    }
                }
                Op::MemoryFree(m) => {
                    self.workspace.free(m)?;
                }
                Op::MemoryRead(r, m, a) => {
                    self.ready(a)?;
                    self.workspace.read(
                        m,
                        &self.registers[a],
                        &mut self.scratch[..self.program.registers[r].len],
                    )?;
                    output = Some(r);
                }
                Op::MemoryWrite(m, a, v) => {
                    self.ready(a)?;
                    self.ready(v)?;
                    self.workspace
                        .write(m, &self.registers[a], &self.registers[v])?;
                    if let Some(tape) = tape.as_deref_mut() {
                        tape.memory_write(m, a, v, self.workspace.data(m)?)?;
                    }
                }
                Op::ResourceState(r) => {
                    self.scratch[..3].copy_from_slice(&[
                        (max_steps - self.steps) as f32,
                        (self.memory_limit - self.managed_bytes - self.workspace.bytes) as f32,
                        self.workspace.bytes as f32,
                    ]);
                    output = Some(r);
                }
                Op::Read(r, p) => {
                    let parameter = &self.program.parameters[p];
                    let data = &parameter.data;
                    self.scratch[..data.len()].copy_from_slice(data);
                    if let Some(domain) = &parameter.learn {
                        for (target,value) in self.scratch.iter_mut().zip(data) {*target=domain.read(*value);}
                    }
                    output = Some(r);
                }
                Op::Write(p, r) => {
                    self.ready(r)?;
                    if let Some(tape) = tape.as_deref_mut() {
                        tape.write(p, r)?;
                    }
                    self.program.parameters[p]
                        .data
                        .copy_from_slice(&self.registers[r]);
                    self.wrote = true;
                }
                Op::Conv(r,x,k,b,g) => {
                    self.ready(x)?;self.ready(k)?;self.ready(b)?;
                    for i in 0..self.program.registers[r].len {
                        let mut sum=f64::from(self.registers[b][i/g.output_volume()]);
                        g.pairs(i,|xi,ki|sum+=f64::from(self.registers[x][xi])*f64::from(self.registers[k][ki]));
                        self.scratch[i]=sum as f32;
                    }
                    output=Some(r);
                },
                Op::MaxPool2d(r, x, g) => {
                    self.ready(x)?;
                    for i in 0..g.output_len() {
                        let mut maximum = f32::NEG_INFINITY;
                        g.visit(i, |index| maximum = maximum.max(self.registers[x][index]));
                        self.scratch[i] = maximum;
                    }
                    output = Some(r);
                },
                Op::AvgPool2d(r, x, g) => {
                    self.ready(x)?;
                    let divisor = g.window_len() as f64;
                    for i in 0..g.output_len() {
                        let mut sum = 0f64;
                        g.visit(i, |index| sum += f64::from(self.registers[x][index]));
                        self.scratch[i] = (sum / divisor) as f32;
                    }
                    output = Some(r);
                },

                Op::Reshape(r,x) => {
                    self.ready_view(x)?;
                    output = Some(r);
                },
                Op::Transpose(r,x,_,_) => {
                    self.ready_view(x)?;
                    output = Some(r);
                },
                Op::Permute(r,x,_,_,_) => {
                    self.ready_view(x)?;
                    output = Some(r);
                },
                Op::Matmul(r,a,b,m,k,n) => {
                    self.ready_view(a)?; self.ready_view(b)?;
                    for i in 0..m { for j in 0..n {
                        let mut sum = 0f64;
                        for q in 0..k {
                            sum += f64::from(self.read_value(a,i*k+q)) * f64::from(self.read_value(b,q*n+j));
                        }
                        self.scratch[i*n+j] = sum as f32;
                    } }
                    output = Some(r);
                },
                Op::BatchMatmul(r,a,b,batch,m,k,n) => {
                    self.ready_view(a)?; self.ready_view(b)?;
                    for batch_index in 0..batch {
                        let a_base = batch_index * m * k;
                        let b_base = batch_index * k * n;
                        let out_base = batch_index * m * n;
                        for i in 0..m { for j in 0..n {
                            let mut sum = 0f64;
                            for q in 0..k {
                                sum += f64::from(self.read_value(a,a_base+i*k+q))
                                    * f64::from(self.read_value(b,b_base+q*n+j));
                            }
                            self.scratch[out_base+i*n+j] = sum as f32;
                        } }
                    }
                    output = Some(r);
                },
                Op::LayerNorm(r,x,gamma,beta,groups,epsilon) => {
                    self.ready(x)?; self.ready(gamma)?; self.ready(beta)?;
                    let len = self.program.registers[r].len;
                    groups.layer_norm(&self.registers[x], &self.registers[gamma], &self.registers[beta],
                        epsilon, &mut self.scratch[..len])?;
                    output = Some(r);
                },
                Op::SoftmaxAxis(r,x,groups) => {
                    self.ready(x)?;
                    groups.softmax(&self.registers[x], &mut self.scratch[..self.program.registers[r].len])?;
                    output = Some(r);
                },
                Op::Scale(r,x,factor) => {
                    self.ready(x)?;
                    for (i,v) in self.registers[x].iter().enumerate() { self.scratch[i] = v * factor; }
                    output = Some(r);
                },
                Op::Linear(r, x, w, b) => {
                    self.ready(x)?;
                    let x = &self.registers[x];
                    for row in 0..self.program.registers[r].len {
                        let mut v = b.map_or(0., |b| self.program.parameters[b].data[row]);
                        for (col, value) in x.iter().enumerate() {
                            v += value * self.program.parameters[w].data[row * x.len() + col];
                        }
                        self.scratch[row] = v;
                    }
                    output = Some(r);
                }
                Op::Binary(r, a, b, mul) => {
                    self.ready(a)?;
                    self.ready(b)?;
                    for i in 0..self.program.registers[r].len {
                        self.scratch[i] = if mul {
                            self.registers[a][i] * self.registers[b][i]
                        } else {
                            self.registers[a][i] + self.registers[b][i]
                        };
                    }
                    output = Some(r);
                }
                Op::Constant(r, i) => {
                    let Instruction::Constant { data, .. } = &self.program.instructions[i] else {
                        unreachable!()
                    };
                    self.scratch[..data.len()].copy_from_slice(data);
                    output = Some(r);
                }
                Op::Minimum(r, a, b) => {
                    self.ready(a)?;
                    self.ready(b)?;
                    for i in 0..self.program.registers[r].len {
                        self.scratch[i] = self.registers[a][i].min(self.registers[b][i]);
                    }
                    output = Some(r);
                }
                Op::Clip(r, x, low, high) => {
                    self.ready(x)?;
                    for (i, v) in self.registers[x].iter().enumerate() {
                        self.scratch[i] = v.clamp(low, high);
                    }
                    output = Some(r);
                }
                Op::Sum(r, x) => {
                    self.ready(x)?;
                    self.scratch[0] =
                        self.registers[x].iter().map(|v| f64::from(*v)).sum::<f64>() as f32;
                    output = Some(r);
                }
                Op::Unary(r, x, kind) => {
                    self.ready(x)?;
                    kind.forward(&self.registers[x], &mut self.scratch[..self.program.registers[r].len])?;
                    output = Some(r);
                }
                Op::Mean(r, x) => {
                    self.ready(x)?;
                    self.scratch[0] = (self.registers[x].iter().map(|v| f64::from(*v)).sum::<f64>()
                        / self.registers[x].len() as f64)
                        as f32;
                    output = Some(r);
                }
                Op::Blend(r, p, a, b) => {
                    self.ready(p)?;
                    ensure!(
                        self.registers[p].iter().all(|v| (0. ..=1.).contains(v)),
                        "gate/route probability must be in [0,1]"
                    );
                    let all_zero = tape.is_none() && self.registers[p].iter().all(|v| *v == 0.);
                    let all_one = tape.is_none() && self.registers[p].iter().all(|v| *v == 1.);
                    if all_zero || all_one {
                        let selected = if all_one { Some(a) } else { b };
                        let n = self.program.registers[r].len;
                        if let Some(selected) = selected {
                            self.ready(selected)?;
                            self.scratch[..n].copy_from_slice(&self.registers[selected]);
                        } else {
                            self.scratch[..n].fill(0.);
                        }
                    } else {
                        self.ready(a)?;
                        if let Some(b) = b {
                            self.ready(b)?;
                        }
                        for i in 0..self.program.registers[r].len {
                            let p =
                                self.registers[p][if self.registers[p].len() == 1 { 0 } else { i }];
                            self.scratch[i] = p * self.registers[a][i]
                                + (1. - p) * b.map_or(0., |b| self.registers[b][i]);
                        }
                    }
                    output = Some(r);
                }
                Op::InferenceSkip(p, when_one, target) => {
                    self.ready(p)?;
                    ensure!(
                        self.registers[p].iter().all(|v| (0. ..=1.).contains(v)),
                        "gate/route probability must be in [0,1]"
                    );
                    let endpoint = if when_one { 1. } else { 0. };
                    if tape.is_none() && self.registers[p].iter().all(|v| *v == endpoint) {
                        self.pc = target;
                        continue;
                    }
                }
                Op::Slice(r, x, start) => {
                    self.ready(x)?;
                    let n = self.program.registers[r].len;
                    self.scratch[..n].copy_from_slice(&self.registers[x][start..start + n]);
                    output = Some(r);
                }
                Op::Label => {}
                Op::Branch(c, yes, no) => {
                    self.ready(c)?;
                    self.pc = if self.registers[c][0] > 0. { yes } else { no };
                    continue;
                }
                Op::Jump(target) => {
                    self.pc = target;
                    continue;
                }
                Op::Return(r) => {
                    self.ready(r)?;
                    if self.program.output_type == crate::categorical::VectorType::Probabilities {
                        let values: Vec<f64> = self.registers[r].iter().map(|x| *x as f64).collect();
                        crate::categorical::validate_probabilities(&values)?;
                    }
                    return Ok(Some((r, self.wrote)));
                }
            }
            if let Some(r) = output {
                if matches!(op, Op::Reshape(..) | Op::Transpose(..) | Op::Permute(..)) {
                    self.detach_views_of(r);
                    let view = self.make_view(op, r)?;
                    if view.base == r {
                        let n = self.program.registers[r].len;
                        for index in 0..n {
                            self.scratch[index] = self.registers[r][view.physical_index(index)];
                        }
                        self.registers[r].copy_from_slice(&self.scratch[..n]);
                        self.views[r] = None;
                    } else {
                        self.views[r] = Some(view);
                    }
                    self.initialized[r] = true;
                    if let Some(tape) = tape.as_deref_mut() {
                        let values = (0..self.program.registers[r].len)
                            .map(|index| self.read_value(r, index))
                            .collect::<Vec<_>>();
                        tape.record(op, &values)?;
                    }
                } else {
                    let n = self.program.registers[r].len;
                    ensure!(
                        self.scratch[..n].iter().all(|v| v.is_finite()),
                        "nonfinite activation at instruction {pc}"
                    );
                    if let Some(tape) = tape.as_deref_mut() {
                        tape.record(op, &self.scratch[..n])?;
                    }
                    self.detach_views_of(r);
                    self.registers[r].resize(n, 0.);
                    self.registers[r].copy_from_slice(&self.scratch[..n]);
                    self.views[r] = None;
                    self.initialized[r] = true;
                }
            }
            self.pc += 1;
        }
    }
}
