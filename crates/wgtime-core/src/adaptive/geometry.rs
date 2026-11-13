//! Atomic replacement of one concrete, pure forward geometry.
use super::{AdaptiveSession, Instruction, Program};
use anyhow::{Context, Result, ensure};

fn pure(program: &Program) -> Result<()> {
    ensure!(
        !program
            .instructions
            .iter()
            .any(|op| matches!(op, Instruction::Write { .. })),
        "geometry rebind requires a pure source forward program; persistent parameter/state writes are unsupported"
    );
    if let Some(loss) = &program.loss {
        ensure!(
            loss.parameters.is_empty() && loss.loss.is_none(),
            "geometry rebind requires a parameter-free source forward loss"
        );
        pure(loss)?;
    }
    Ok(())
}

// Match prepare's managed f32 accounting before copying live parameter data.
// Metadata and caller-owned IR are outside the session's storage budget.
fn managed(program: &Program) -> Result<usize> {
    let mut floats = 0usize;
    let mut add = |count: usize| -> Result<()> {
        floats = floats
            .checked_add(count)
            .context("geometry memory size overflow")?;
        Ok(())
    };
    for parameter in &program.parameters {
        add(crate::elements(&parameter.shape)?
            .checked_mul(2)
            .context("geometry parameter size overflow")?)?;
    }
    for register in &program.registers {
        add(register.len)?;
    }
    add(program.registers.iter().map(|r| r.len).max().unwrap_or(0))?;
    for instruction in &program.instructions {
        if let Instruction::Constant { data, .. } = instruction {
            add(data.len())?;
        }
    }
    crate::bytes(floats)
}

impl AdaptiveSession {
    /// Rebind a concrete shared-frontend Forward plan without initializing weights.
    /// Parameter schema and declaration order must match, excluding data. Existing
    /// values and their canonical Vec allocations survive, as do optimizer moments,
    /// collection identity, revisions and counters. Only one plan remains active.
    /// Admission includes both live and candidate managed storage (and compiled
    /// workspace); an error leaves the live session completely untouched.
    pub fn rebind_geometry(&mut self, mut program: Program) -> Result<()> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        pure(&self.program)?;
        pure(&program)?;
        ensure!(
            self.program.parameters.len() == program.parameters.len(),
            "geometry rebind requires identical parameter storage schema"
        );
        for (old, new) in self.program.parameters.iter().zip(&program.parameters) {
            // Recipe/domain types intentionally do not implement PartialEq. Compare
            // only this small schema, never serialize the live weight payload.
            ensure!(
                old.name == new.name
                    && old.shape == new.shape
                    && old.frozen == new.frozen
                    && old.probability == new.probability
                    && serde_json::to_value(&old.learn)? == serde_json::to_value(&new.learn)?
                    && serde_json::to_value(&old.initializer)?
                        == serde_json::to_value(&new.initializer)?,
                "geometry rebind parameter storage schema mismatch: expected {}",
                old.name
            );
        }
        let optimizer_bytes = self.optimizer_storage_bytes()?;
        let available = self
            .memory_limit
            .checked_sub(self.managed_bytes)
            .and_then(|n| n.checked_sub(optimizer_bytes))
            .context("geometry rebind transient memory budget exceeded")?;
        ensure!(
            managed(&program)? <= available,
            "geometry rebind transient memory budget exceeded"
        );
        for (old, new) in self.program.parameters.iter().zip(&mut program.parameters) {
            new.data = old.data.clone();
        }
        let mut candidate = Self::prepare(program, available)?;
        if self.is_compiled() {
            // The shared selector may use a truthful interpreter fallback for IR
            // operations without a compiled kernel; admission remains transient.
            candidate.enable_compiled()?;
        }
        // Validate the optional loss too. It is prepared transiently during training;
        // here its validation workspace must also fit alongside both forward plans.
        if let Some(loss) = &candidate.program.loss {
            let remaining = available
                .checked_sub(candidate.managed_bytes)
                .context("geometry loss transient memory budget exceeded")?;
            ensure!(
                managed(loss)? <= remaining,
                "geometry loss transient memory budget exceeded"
            );
            let mut loss = Self::prepare((**loss).clone(), remaining)?;
            if self.is_compiled() {
                loss.enable_compiled()?;
            }
        }

        // No fallible work beyond this point. Leave session identity, optimizer and
        // accounting policy in place; replace only the executable geometry/storage.
        for (old, new) in self
            .program
            .parameters
            .iter_mut()
            .zip(&mut candidate.program.parameters)
        {
            std::mem::swap(&mut old.data, &mut new.data);
        }
        self.program = candidate.program;
        self.ops = candidate.ops;
        self.registers = candidate.registers;
        self.views = candidate.views;
        self.initialized = candidate.initialized;
        self.backup = candidate.backup;
        self.scratch = candidate.scratch;
        self.input = candidate.input;
        self.output_len = candidate.output_len;
        self.managed_bytes = candidate.managed_bytes;
        self.observation_sites = candidate.observation_sites;
        self.workspace = candidate.workspace;
        self.compiled = candidate.compiled;
        self.pc = 0;
        self.wrote = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgt_language::model::{ModelDefinition, ModelPlan};

    #[test]
    fn rebind_keeps_private_optimizer_history_and_session_policy() {
        let definition = |rows| {
            ModelDefinition::parse(&format!(
                "wgt 0.5\nmodel Geometry\nparameter w: f32[1,1] = full(0.8)\n\
                 fn forward(x: f32[{rows},1]) -> f32[{rows},1] {{\n\
                 return x @ (w as [1,1]);\n}}\n"
            ))
            .unwrap()
        };
        let ModelPlan::Forward(program) = definition(2).instantiate_plan().unwrap() else {
            panic!()
        };
        let mut session = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
        session
            .set_optimizer(crate::training::Optimizer::Adam)
            .unwrap();
        session
            .train_step(
                &[vec![1., 2.]],
                crate::training::Objective::Mse(&[vec![0., 0.]]),
                0.01,
                &mut 100_000,
                1024 * 1024,
            )
            .unwrap();
        let optimizer = serde_json::to_value(&session.optimizer).unwrap();
        session.execution_limit = 12345;
        session.compiled_forward_calls = 17;
        session.compiled_tape_calls = 9;
        let before = (session.steps, session.revision, session.memory_limit);
        let pointer = session.program.parameters[0].data.as_ptr();
        let (ModelPlan::Forward(program), _) = definition(7).compile_plan_with_modules().unwrap()
        else {
            panic!()
        };
        session.rebind_geometry(program).unwrap();
        assert_eq!(serde_json::to_value(&session.optimizer).unwrap(), optimizer);
        assert_eq!(
            (session.steps, session.revision, session.memory_limit),
            before
        );
        assert_eq!(session.execution_limit, 12345);
        assert_eq!(session.compiled_forward_counts(), (17, 9));
        assert_eq!(session.program.parameters[0].data.as_ptr(), pointer);
    }
}
