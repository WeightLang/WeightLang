//! Bounded, stateful forward programs. Writes are visible immediately inside a
//! call, and committed only when that call returns successfully.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod source;
pub use source::parse;
mod autodiff;
pub use autodiff::ForwardPass;
pub mod transport;
mod training;
pub mod optimization;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
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
    Constant { out:String, data:Vec<f32> },
    Detach { out:String, input:String },
    Exp { out:String, input:String },
    Log { out:String, input:String },
    LogSoftmax { out:String, input:String },
    Minimum { out:String, left:String, right:String },
    Clip { out:String, input:String, low:f32, high:f32 },
    Sum { out:String, input:String },
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
    Softmax { out: String, input: String },
    Neg { out: String, input: String },
    Mean { out: String, input: String },
    Gate { out: String, probability: String, input: String },
    Route { out: String, probability: String, yes: String, no: String },
    /// Skip an unevaluated expression only when no differentiation tape is requested.
    InferenceSkip { probability: String, when_one: bool, target: String },
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

#[derive(Debug, Clone, Copy)]
enum Op {
    Constant(usize,usize),
    Minimum(usize,usize,usize),
    Clip(usize,usize,f32,f32),
    Sum(usize,usize),
    Read(usize, usize),
    Write(usize, usize),
    Linear(usize, usize, usize, Option<usize>),
    Binary(usize, usize, usize, bool),
    Unary(usize, usize, u8),
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
    initialized: Vec<bool>,
    backup: Vec<Vec<f32>>,
    scratch: Vec<f32>,
    input: usize,
    output_len: usize,
    managed_bytes: usize,
    revision: u64,
    steps: u64,
    optimizer: Option<training::State>,
    custom_optimizer: Option<optimization::CustomState>,
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
    pub fn prepare(program: Program, budget_bytes: usize) -> Result<Self> {
        ensure!(program.version == 1, "unsupported adaptive IR version");
        ensure!(!program.instructions.is_empty(), "empty adaptive program");
        if let Some(loss) = &program.loss {
            ensure!(loss.loss.is_none() && loss.parameters.is_empty(), "loss must be a parameter-free tensor program");
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
        let mut floats = 0usize;
        for p in &program.parameters {
            if p.probability {
                ensure!(p.shape == [1], "prob parameter must be scalar");
                ensure!(!p.frozen || p.data.iter().all(|v| (0. ..=1.).contains(v)), "frozen prob must be in [0,1]");
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
            if let Instruction::Constant{data,..}=instruction {
                floats=floats.checked_add(data.len()).context("constant size overflow")?;
            }
        }
        let mut maximum = 0;
        for r in &program.registers {
            ensure!(r.len > 0, "register length must be positive");
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
                        ensure!(!program.parameters[p].probability, "write to prob is unsupported; use optimizer updates");
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
                        ensure!(
                            program.parameters[w].shape == [len(r), len(x)],
                            "linear weight must be [output, input]"
                        );
                        let b = bias.as_deref().map(param).transpose()?;
                        if let Some(b) = b {
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
                    Instruction::Constant{out,data} => {
                        let r=dest(out)?;
                        ensure!(data.len()==len(r) && data.iter().all(|v|v.is_finite()),"invalid tensor constant");
                        Op::Constant(r,ops.len())
                    }
                    Instruction::Minimum{out,left,right} => {
                        let (r,a,b)=(dest(out)?,reg(left)?,reg(right)?);
                        ensure!(len(r)==len(a) && len(a)==len(b),"minimum shape mismatch");
                        Op::Minimum(r,a,b)
                    }
                    Instruction::Clip{out,input,low,high} => {
                        let (r,x)=(dest(out)?,reg(input)?);
                        ensure!(len(r)==len(x) && low.is_finite() && high.is_finite() && low<=high,"invalid clip shape or bounds");
                        Op::Clip(r,x,*low,*high)
                    }
                    Instruction::Sum{out,input} => {
                        let (r,x)=(dest(out)?,reg(input)?);ensure!(len(r)==1,"sum output must be f32[1]");Op::Sum(r,x)
                    }
                    Instruction::Detach { out, input }
                    | Instruction::Exp { out, input }
                    | Instruction::Log { out, input }
                    | Instruction::LogSoftmax { out, input }
                    | Instruction::Tanh { out, input }
                    | Instruction::Sigmoid { out, input }
                    | Instruction::Softmax { out, input }
                    | Instruction::Neg { out, input }
                    | Instruction::Relu { out, input } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(len(r) == len(x), "unary length mismatch");
                        let kind = match instruction {
                            Instruction::Tanh { .. } => 0,
                            Instruction::Sigmoid { .. } => 1,
                            Instruction::Softmax { .. } => 3,
                            Instruction::Neg { .. } => 4,
                            Instruction::Detach { .. } => 5,
                            Instruction::Exp { .. } => 6,
                            Instruction::Log { .. } => 7,
                            Instruction::LogSoftmax { .. } => 8,
                            _ => 2,
                        };
                        Op::Unary(r, x, kind)
                    }
                    Instruction::Mean { out, input } => {
                        let (r,x)=(dest(out)?,reg(input)?);
                        ensure!(len(r)==1,"mean output must be f32[1]");
                        Op::Mean(r,x)
                    }
                    Instruction::Gate { out, probability, input } => {
                        let (r, p, x) = (dest(out)?, reg(probability)?, reg(input)?);
                        ensure!(len(r) == len(x) && (len(p) == 1 || len(p) == len(r)), "gate shape mismatch");
                        Op::Blend(r, p, x, None)
                    }
                    Instruction::Route { out, probability, yes, no } => {
                        let (r, p, a, b) = (dest(out)?, reg(probability)?, reg(yes)?, reg(no)?);
                        ensure!(len(r) == len(a) && len(r) == len(b) && (len(p) == 1 || len(p) == len(r)), "route shape mismatch");
                        Op::Blend(r, p, a, Some(b))
                    }
                    Instruction::InferenceSkip { probability, when_one, target } => {
                        Op::InferenceSkip(reg(probability)?, *when_one, label(target)?)
                    }
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
            ops.push(op);
        }
        let output_len = output_len.context("program requires a return")?;
        let registers = program
            .registers
            .iter()
            .map(|r| crate::zero_buffer(r.len))
            .collect::<Result<_>>()?;
        let backup = program
            .parameters
            .iter()
            .map(|p| crate::zero_buffer(p.data.len()))
            .collect::<Result<_>>()?;
        Ok(Self {
            initialized: vec![false; program.registers.len()],
            program,
            ops,
            registers,
            backup,
            scratch: crate::zero_buffer(maximum)?,
            input,
            output_len,
            managed_bytes,
            revision: 0,
            steps: 0,
            optimizer: None,
            custom_optimizer: None,
        })
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
        let p = self.program.parameters.iter_mut().find(|p| p.name == name)
            .context("unknown parameter")?;
        ensure!(!p.frozen, "frozen parameter");
        ensure!(values.len() == p.data.len(), "parameter length mismatch");
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite parameter");
        let revision = self.revision.checked_add(1).context("revision overflow")?;
        p.data.copy_from_slice(values);
        self.revision = revision;
        Ok(())
    }

    fn ready(&self, index: usize) -> Result<()> {
        ensure!(
            self.initialized[index],
            "read of uninitialized register: {}",
            self.program.registers[index].name
        );
        Ok(())
    }

    /// One forward transaction; state persists between successful calls. Invalid
    /// arithmetic, undefined registers, fallthrough and exhausted budgets roll back
    /// all weight writes. max_steps counts every instruction, including jumps.
    pub fn run(&mut self, input: &[f32], max_steps: u64) -> Result<&[f32]> {
        self.begin(input, max_steps)?;
        let result = self.execute(max_steps, None).and_then(|(r, wrote)| {
            if wrote {
                self.revision = self.revision.checked_add(1).context("revision overflow")?;
            }
            Ok(r)
        });
        match result {
            Ok(r) => Ok(&self.registers[r]),
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
        let result = self.execute(max_steps, None).map(|(r, _)| self.registers[r].clone());
        self.restore();
        result
    }

    fn begin(&mut self, input: &[f32], max_steps: u64) -> Result<()> {
        self.steps = 0;
        ensure!(
            input.len() == self.input_len() && input.iter().all(|v| v.is_finite()),
            "invalid adaptive input"
        );
        ensure!(max_steps > 0, "max_steps must be positive");
        self.initialized.fill(false);
        self.initialized[self.input] = true;
        self.registers[self.input].copy_from_slice(input);
        for (p, b) in self.program.parameters.iter().zip(&mut self.backup) {
            b.copy_from_slice(&p.data);
        }
        Ok(())
    }

    fn restore(&mut self) {
        for (p, b) in self.program.parameters.iter_mut().zip(&self.backup) {
            p.data.copy_from_slice(b);
        }
    }

    fn execute(
        &mut self,
        max_steps: u64,
        mut tape: Option<&mut autodiff::Tape>,
    ) -> Result<(usize, bool)> {
        let mut pc = 0;
        let mut wrote = false;
        loop {
            ensure!(
                self.steps < max_steps,
                "adaptive instruction budget exceeded"
            );
            let op = self
                .ops
                .get(pc)
                .context("adaptive program fell through without return")?;
            self.steps += 1;
            let mut output = None;
            match *op {
                Op::Read(r, p) => {
                    let data = &self.program.parameters[p].data;
                    self.scratch[..data.len()].copy_from_slice(data);
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
                    wrote = true;
                }
                Op::Linear(r, x, w, b) => {
                    self.ready(x)?;
                    let x = &self.registers[x];
                    for row in 0..self.registers[r].len() {
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
                    for i in 0..self.registers[r].len() {
                        self.scratch[i] = if mul {
                            self.registers[a][i] * self.registers[b][i]
                        } else {
                            self.registers[a][i] + self.registers[b][i]
                        };
                    }
                    output = Some(r);
                }
                Op::Constant(r,i) => {
                    let Instruction::Constant{data,..}=&self.program.instructions[i] else {unreachable!()};
                    self.scratch[..data.len()].copy_from_slice(data);output=Some(r);
                }
                Op::Minimum(r,a,b) => {
                    self.ready(a)?;self.ready(b)?;
                    for i in 0..self.registers[r].len() {self.scratch[i]=self.registers[a][i].min(self.registers[b][i]);}output=Some(r);
                }
                Op::Clip(r,x,low,high) => {
                    self.ready(x)?;for (i,v) in self.registers[x].iter().enumerate() {self.scratch[i]=v.clamp(low,high);}output=Some(r);
                }
                Op::Sum(r,x) => {
                    self.ready(x)?;self.scratch[0]=self.registers[x].iter().map(|v|f64::from(*v)).sum::<f64>() as f32;output=Some(r);
                }
                Op::Unary(r, x, kind) => {
                    self.ready(x)?;
                    let max = if kind == 3 || kind == 8 { self.registers[x].iter().copied().fold(f32::NEG_INFINITY, f32::max) } else { 0. };
                    let total: f64 = if kind == 3 || kind == 8 { self.registers[x].iter().map(|v| (f64::from(*v) - f64::from(max)).exp()).sum() } else { 1. };
                    for (i, v) in self.registers[x].iter().enumerate() {
                        self.scratch[i] = match kind {
                            0 => v.tanh(),
                            1 => 1. / (1. + (-v).exp()),
                            3 => ((f64::from(*v) - f64::from(max)).exp() / total) as f32,
                            4 => -v,
                            5 => *v,
                            6 => v.exp(),
                            7 => v.ln(),
                            8 => (f64::from(*v)-f64::from(max)-total.ln()) as f32,
                            _ => v.max(0.),
                        };
                    }
                    output = Some(r);
                }
                Op::Mean(r,x) => {
                    self.ready(x)?;
                    self.scratch[0]=(self.registers[x].iter().map(|v| f64::from(*v)).sum::<f64>() / self.registers[x].len() as f64) as f32;
                    output=Some(r);
                }
                Op::Blend(r, p, a, b) => {
                    self.ready(p)?;
                    ensure!(self.registers[p].iter().all(|v| (0. ..=1.).contains(v)), "gate/route probability must be in [0,1]");
                    let all_zero = tape.is_none() && self.registers[p].iter().all(|v| *v == 0.);
                    let all_one = tape.is_none() && self.registers[p].iter().all(|v| *v == 1.);
                    if all_zero || all_one {
                        let selected = if all_one { Some(a) } else { b };
                        let n = self.registers[r].len();
                        if let Some(selected) = selected {
                            self.ready(selected)?;
                            self.scratch[..n].copy_from_slice(&self.registers[selected]);
                        } else {
                            self.scratch[..n].fill(0.);
                        }
                    } else {
                        self.ready(a)?;
                        if let Some(b) = b { self.ready(b)?; }
                        for i in 0..self.registers[r].len() {
                            let p = self.registers[p][if self.registers[p].len() == 1 { 0 } else { i }];
                            self.scratch[i] = p * self.registers[a][i] + (1. - p) * b.map_or(0., |b| self.registers[b][i]);
                        }
                    }
                    output = Some(r);
                }
                Op::InferenceSkip(p, when_one, target) => {
                    self.ready(p)?;
                    ensure!(self.registers[p].iter().all(|v| (0. ..=1.).contains(v)), "gate/route probability must be in [0,1]");
                    let endpoint = if when_one { 1. } else { 0. };
                    if tape.is_none() && self.registers[p].iter().all(|v| *v == endpoint) {
                        pc = target;
                        continue;
                    }
                }
                Op::Slice(r, x, start) => {
                    self.ready(x)?;
                    let n = self.registers[r].len();
                    self.scratch[..n].copy_from_slice(&self.registers[x][start..start + n]);
                    output = Some(r);
                }
                Op::Label => {}
                Op::Branch(c, yes, no) => {
                    self.ready(c)?;
                    pc = if self.registers[c][0] > 0. { yes } else { no };
                    continue;
                }
                Op::Jump(target) => {
                    pc = target;
                    continue;
                }
                Op::Return(r) => {
                    self.ready(r)?;
                    return Ok((r, wrote));
                }
            }
            if let Some(r) = output {
                let n = self.registers[r].len();
                ensure!(
                    self.scratch[..n].iter().all(|v| v.is_finite()),
                    "nonfinite activation at instruction {pc}"
                );
                if let Some(tape) = tape.as_deref_mut() {
                    tape.record(*op, &self.scratch[..n])?;
                }
                self.registers[r].copy_from_slice(&self.scratch[..n]);
                self.initialized[r] = true;
            }
            pc += 1;
        }
    }
}
