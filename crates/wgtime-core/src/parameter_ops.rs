//! Explicit storage access; update algorithms remain ordinary caller computations.
use crate::{accumulate, Session, Storage};
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ParameterInfo<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub writable: bool,
}

/// A contiguous flat f32 replacement. `start` is an element offset, not bytes.
#[derive(Debug)]
pub struct ParameterWrite<'a> {
    pub name: &'a str,
    pub start: usize,
    pub values: &'a [f32],
}

impl Session {
    pub fn parameter_info(&self) -> Vec<ParameterInfo<'_>> {
        let mut names: Vec<_> = self.names.iter().collect();
        names.sort_by_key(|(_, index)| **index);
        names
            .into_iter()
            .map(|(name, &index)| ParameterInfo {
                name,
                shape: &self.parameters[index].shape,
                writable: self.write_target(name, 0, 0).is_ok(),
            })
            .collect()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Materialize any parameter expression into caller-owned storage, without allocation.
    /// On I/O failure `out` may be partially filled; session parameters are unchanged.
    pub fn read_parameter(&mut self, name: &str, start: usize, out: &mut [f32]) -> Result<()> {
        let &index = self.names.get(name).context("unknown parameter")?;
        ensure!(
            start
                .checked_add(out.len())
                .is_some_and(|end| end <= self.parameters[index].len),
            "parameter read out of bounds"
        );
        out.fill(0.);
        accumulate(&mut self.parameters, index, start, out).map_err(anyhow::Error::msg)?;
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "nonfinite effective parameter"
        );
        Ok(())
    }

    fn write_target(&self, name: &str, start: usize, len: usize) -> Result<(usize, usize)> {
        let &mut_index = self.names.get(name).context("unknown parameter")?;
        let mut index = mut_index;
        ensure!(
            start
                .checked_add(len)
                .is_some_and(|end| end <= self.parameters[index].len),
            "parameter write out of bounds"
        );
        let mut offset = start;
        loop {
            let tensor = &self.parameters[index];
            ensure!(!tensor.frozen, "frozen parameter");
            match tensor.storage {
                Storage::Resident(_) => return Ok((index, offset)),
                Storage::View { source, start } => {
                    index = source;
                    offset += start;
                }
                _ => anyhow::bail!(
                    "parameter is read-only: write a resident owner, not file/add storage"
                ),
            }
        }
    }

    /// Validate every write before changing any storage. Overlapping aliases are rejected.
    /// No heap allocation on success. Every nonempty batch increments revision once.
    pub fn write_parameters(&mut self, writes: &[ParameterWrite<'_>]) -> Result<()> {
        self.ensure_mutable()?;
        if writes.is_empty() {
            return Ok(());
        }
        let next = self
            .revision
            .checked_add(1)
            .context("parameter revision overflow")?;
        for (i, write) in writes.iter().enumerate() {
            let (owner, start) = self.write_target(write.name, write.start, write.values.len())?;
            ensure!(
                write.values.iter().all(|v| v.is_finite()),
                "nonfinite parameter write"
            );
            for earlier in &writes[..i] {
                let (other, offset) =
                    self.write_target(earlier.name, earlier.start, earlier.values.len())?;
                ensure!(
                    owner != other
                        || write.values.is_empty()
                        || earlier.values.is_empty()
                        || start >= offset + earlier.values.len()
                        || offset >= start + write.values.len(),
                    "overlapping parameter writes"
                );
            }
        }
        for write in writes {
            let (owner, start) = self.write_target(write.name, write.start, write.values.len())?;
            if let Storage::Resident(data) = &mut self.parameters[owner].storage {
                data[start..start + write.values.len()].copy_from_slice(write.values);
            }
        }
        self.revision = next;
        Ok(())
    }
}
