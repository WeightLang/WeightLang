//! Task-local differentiable memory and resumable inference transactions.
use super::{AdaptiveSession, Instruction as I, Op, Program};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::HashMap;

pub(crate) fn validate(
    program: &Program,
    regs: &HashMap<String, usize>,
) -> Result<HashMap<usize, (usize, usize)>> {
    let mut memories = HashMap::new();
    for instruction in &program.instructions {
        if let I::MemoryAlloc { out, slots, width } = instruction {
            let r = *regs.get(out).context("unknown memory register")?;
            ensure!(
                out != &program.input && program.registers[r].len == 1,
                "invalid memory handle register"
            );
            ensure!(
                *slots > 0 && *slots <= 16384 && *width > 0 && *width <= 16384,
                "memory dimensions must be 1..16384"
            );
            if let Some(old) = memories.insert(r, (*slots, *width)) {
                ensure!(old == (*slots, *width), "memory shape changes");
            }
        }
    }
    for instruction in &program.instructions {
        let metadata = instruction.metadata();
        if let Some(memory) = metadata.memory {
            ensure!(
                regs.get(memory).is_some_and(|r| memories.contains_key(r)),
                "unknown memory handle"
            );
        }
        for name in metadata.inputs.into_iter().flatten().chain(metadata.output) {
            ensure!(
                !regs.get(name).is_some_and(|r| memories.contains_key(r)),
                "memory handle cannot be used as a tensor: {name}"
            );
        }
    }
    Ok(memories)
}

struct Bank {
    slots: usize,
    width: usize,
    data: Vec<f32>,
}
pub(super) struct Workspace {
    banks: Vec<Option<Bank>>,
    pub bytes: usize,
    peak: usize,
}
impl Workspace {
    pub fn new(registers: usize) -> Self {
        Self {
            banks: (0..registers).map(|_| None).collect(),
            bytes: 0,
            peak: 0,
        }
    }
    pub fn clear(&mut self) {
        for bank in &mut self.banks {
            *bank = None;
        }
        self.bytes = 0;
        self.peak = 0;
    }
    fn bank(&self, m: usize) -> Result<&Bank> {
        self.banks[m]
            .as_ref()
            .context("memory is not allocated or has been freed")
    }
    pub fn data(&self, m: usize) -> Result<&[f32]> {
        Ok(&self.bank(m)?.data)
    }
    pub fn cost(&self, op: Op) -> Result<u64> {
        let n = match op {
            Op::MemoryAlloc(_, n, d) => n.checked_mul(d).context("memory size overflow")?,
            Op::MemoryRead(_, m, _) | Op::MemoryWrite(m, _, _) => self.bank(m)?.data.len(),
            _ => 0,
        };
        Ok(1 + n as u64)
    }
    pub fn allocate(&mut self, m: usize, slots: usize, width: usize, limit: usize) -> Result<()> {
        ensure!(
            self.banks[m].is_none(),
            "memory already allocated; free it before reallocating"
        );
        let n = slots.checked_mul(width).context("memory size overflow")?;
        let bytes = crate::bytes(n)?
            .checked_add(self.bytes)
            .context("memory size overflow")?;
        ensure!(bytes <= limit, "adaptive workspace memory budget exceeded");
        let data = crate::zero_buffer(n)?;
        self.banks[m] = Some(Bank { slots, width, data });
        self.bytes = bytes;
        self.peak = self.peak.max(bytes);
        Ok(())
    }
    pub fn free(&mut self, m: usize) -> Result<()> {
        let bank = self.banks[m]
            .take()
            .context("memory is not allocated or has been freed")?;
        self.bytes -= bank.data.len() * 4;
        Ok(())
    }
    fn address(bank: &Bank, a: &[f32]) -> Result<()> {
        ensure!(
            a.len() == bank.slots && a.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v)),
            "memory address must have one finite [0,1] weight per slot"
        );
        Ok(())
    }
    pub fn read(&self, m: usize, a: &[f32], out: &mut [f32]) -> Result<()> {
        let bank = self.bank(m)?;
        Self::address(bank, a)?;
        ensure!(out.len() == bank.width, "memory read width mismatch");
        for (j, dst) in out.iter_mut().enumerate() {
            *dst = a
                .iter()
                .enumerate()
                .map(|(i, w)| f64::from(*w) * f64::from(bank.data[i * bank.width + j]))
                .sum::<f64>() as f32;
        }
        Ok(())
    }
    pub fn write(&mut self, m: usize, a: &[f32], v: &[f32]) -> Result<()> {
        Self::address(self.bank(m)?, a)?;
        let bank = self.banks[m].as_mut().unwrap();
        ensure!(
            v.len() == bank.width && v.iter().all(|v| v.is_finite()),
            "invalid memory write value"
        );
        for (i, w) in a.iter().enumerate() {
            for (j, value) in v.iter().enumerate() {
                let dst = &mut bank.data[i * bank.width + j];
                *dst = ((1. - f64::from(*w)) * f64::from(*dst) + f64::from(*w) * f64::from(*value))
                    as f32;
            }
        }
        ensure!(
            bank.data.iter().all(|v| v.is_finite()),
            "nonfinite memory write"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ExecutionProgress {
    pub done: bool,
    pub output: Option<Vec<f32>>,
    /// Cumulative forward work, not wall time or hardware instructions.
    pub work: u64,
    pub memory_bytes: usize,
    pub peak_memory_bytes: usize,
}
impl AdaptiveSession {
    pub fn execution_active(&self) -> bool {
        self.pending.is_some()
    }
    pub(super) fn ensure_idle(&self) -> Result<()> {
        ensure!(
            !self.execution_active(),
            "execution active; resume or cancel first"
        );
        Ok(())
    }
    /// Prepare a separate bounded transaction; the public model stays unchanged.
    /// The additional session has its own memory_limit, including rollback storage.
    pub fn start_execution(
        &mut self,
        input: &[f32],
        max_work: u64,
        memory_bytes: usize,
    ) -> Result<()> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        ensure!(
            !self.is_compiled(),
            "resumable execution is unavailable for compiled models; use run or evaluate"
        );
        ensure!(
            memory_bytes <= self.memory_limit,
            "execution memory exceeds session limit"
        );
        let mut pending = Self::prepare(self.program.clone(), memory_bytes)?;
        pending.begin(input, max_work)?;
        pending.execution_limit = max_work;
        self.steps = 0;
        self.pending = Some(Box::new(pending));
        Ok(())
    }
    /// Execute at most `instructions` VM instructions and `available_work` work.
    /// Exhausting the lifetime/work allowance aborts; reaching a quantum suspends.
    pub fn resume_execution(
        &mut self,
        instructions: u64,
        available_work: u64,
    ) -> Result<ExecutionProgress> {
        self.ensure_mutable()?;
        ensure!(instructions > 0, "execution quantum must be positive");
        let mut pending = self.pending.take().context("no active execution")?;
        let limit = pending
            .execution_limit
            .min(pending.steps.saturating_add(available_work));
        let result = pending.execute_slice(limit, None, instructions);
        self.steps = pending.steps;
        let progress = match result? {
            Some((r, wrote)) => {
                let revision = if wrote {
                    self.revision.checked_add(1).context("revision overflow")?
                } else {
                    self.revision
                };
                let output = pending.registers[r].clone();
                if wrote {
                    for (p, new) in self
                        .program
                        .parameters
                        .iter_mut()
                        .zip(&pending.program.parameters)
                    {
                        p.data.copy_from_slice(&new.data);
                    }
                }
                self.revision = revision;
                ExecutionProgress {
                    done: true,
                    output: Some(output),
                    work: pending.steps,
                    memory_bytes: 0,
                    peak_memory_bytes: pending.workspace.peak,
                }
            }
            None => {
                let progress = ExecutionProgress {
                    done: false,
                    output: None,
                    work: pending.steps,
                    memory_bytes: pending.workspace.bytes,
                    peak_memory_bytes: pending.workspace.peak,
                };
                self.pending = Some(pending);
                progress
            }
        };
        Ok(progress)
    }
    /// Cancellation never commits working weights or refunds completed work.
    pub fn cancel_execution(&mut self) -> bool {
        self.pending.take().is_some()
    }
    pub fn execution_memory_limit(&self) -> Result<usize> {
        Ok(self
            .pending
            .as_ref()
            .context("no active execution")?
            .memory_limit)
    }
}
