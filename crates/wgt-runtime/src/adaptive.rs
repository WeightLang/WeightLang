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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub version: u32,
    pub input: String,
    pub registers: Vec<Register>,
    pub parameters: Vec<Parameter>,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Register {
    pub name: String,
    pub len: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    #[serde(default)]
    pub frozen: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instruction {
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
    Read(usize, usize),
    Write(usize, usize),
    Linear(usize, usize, usize, Option<usize>),
    Binary(usize, usize, usize, bool),
    Unary(usize, usize, u8),
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
                    Instruction::Tanh { out, input }
                    | Instruction::Sigmoid { out, input }
                    | Instruction::Relu { out, input } => {
                        let (r, x) = (dest(out)?, reg(input)?);
                        ensure!(len(r) == len(x), "unary length mismatch");
                        let kind = match instruction {
                            Instruction::Tanh { .. } => 0,
                            Instruction::Sigmoid { .. } => 1,
                            _ => 2,
                        };
                        Op::Unary(r, x, kind)
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

    fn begin(&mut self, input: &[f32], max_steps: u64) -> Result<()> {
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
        self.steps = 0;
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
                Op::Unary(r, x, kind) => {
                    self.ready(x)?;
                    for (i, v) in self.registers[x].iter().enumerate() {
                        self.scratch[i] = match kind {
                            0 => v.tanh(),
                            1 => 1. / (1. + (-v).exp()),
                            _ => v.max(0.),
                        };
                    }
                    output = Some(r);
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
