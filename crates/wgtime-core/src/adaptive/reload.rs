use super::AdaptiveSession;
use anyhow::{ensure, Context, Result};

impl AdaptiveSession {
    pub fn weights_bytes(&self, limit: usize) -> Result<Vec<u8>> {
        crate::weight_artifact::encode(&self.program.parameters.iter().map(|p| (p.name.clone(), p.shape.clone(), p.data.clone())).collect::<Vec<_>>(), limit)
    }

    pub fn bind_weights(&mut self, bytes: &[u8], optimizer: &str) -> Result<()> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        let schema = self.program.parameters.iter().map(|p| (p.name.clone(), p.shape.clone())).collect::<Vec<_>>();
        let values = crate::weight_artifact::decode(&schema, bytes, 64 * 1024 * 1024)?;
        let mut program = self.program.clone();
        for (parameter, data) in program.parameters.iter_mut().zip(values) { parameter.data = data; }
        let mut candidate = Self::prepare(program, self.memory_limit)?;
        match optimizer {
            "none" => {},
            "sgd" => candidate.set_optimizer(crate::training::Optimizer::Sgd)?,
            "adam" => candidate.set_optimizer(crate::training::Optimizer::Adam)?,
            _ => anyhow::bail!("forward optimizer must be none, sgd or adam"),
        }
        self.reload_candidate(candidate)
    }

    /// Replace validated state without changing the enclosing Mounted identity.
    pub(crate) fn reload_candidate(&mut self, mut candidate: Self) -> Result<()> {
        self.ensure_mutable()?;
        self.ensure_idle()?;
        let structure = |session: &Self| -> Result<serde_json::Value> {
            let mut value = serde_json::to_value(session.program())?;
            for parameter in value["parameters"]
                .as_array_mut()
                .context("missing parameters")?
            {
                parameter
                    .as_object_mut()
                    .context("invalid parameter")?
                    .remove("data");
            }
            Ok(value)
        };
        ensure!(
            structure(self)? == structure(&candidate)?,
            "reload requires identical model structure"
        );
        if self.is_compiled() { candidate.enable_compiled()?; }
        candidate.compiled_forward_calls = self.compiled_forward_calls;
        candidate.compiled_tape_calls = self.compiled_tape_calls;
        candidate.revision = self.revision.checked_add(1).context("revision overflow")?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session(data: f32) -> AdaptiveSession {
        AdaptiveSession::prepare(
            crate::adaptive::Program {
                output_type: crate::categorical::VectorType::F32,
                version: 1,
                input: "x".into(),
                registers: vec![crate::adaptive::Register {
                    name: "x".into(),
                    len: 1,
                    shape: vec![1],
                }],
                parameters: vec![crate::adaptive::Parameter {
                    name: "w".into(),
                    shape: vec![1],
                    data: vec![data],
                    frozen: false,
                    probability: false,
                    learn: None,
                }],
                instructions: vec![crate::adaptive::Instruction::Return { value: "x".into() }],
                loss: None,
            },
            1024 * 1024,
        )
        .unwrap()
    }
    #[test]
    fn collecting_live_identity_rejects_reload_before_swap() {
        let mut current=session(1.);
        let token=current.begin_collection().unwrap();
        assert!(current.reload_candidate(session(2.)).is_err());
        assert_eq!(current.parameter("w").unwrap(),[1.]);
        assert_eq!(current.revision(),0);
        drop(token);
        current.reload_candidate(session(2.)).unwrap();
        let token=current.begin_collection().unwrap();
        assert!(current.reload_candidate(session(3.)).is_err());
        drop(token);
    }
    #[test]
    fn rejected_structure_preserves_state_and_revision() {
        let mut current = session(1.);
        let mut candidate = session(2.);
        candidate.program.parameters[0].name = "other".into();
        let before = serde_json::to_value(current.program()).unwrap();
        assert!(current.reload_candidate(candidate).is_err());
        assert_eq!(serde_json::to_value(current.program()).unwrap(), before);
        assert_eq!(current.revision(), 0);
        current.reload_candidate(session(2.)).unwrap();
        assert_eq!(current.revision(), 1);
        assert_eq!(current.program().parameters[0].data, vec![2.]);
        current.reload_candidate(session(1.)).unwrap();
        assert_eq!(current.revision(), 2);
    }
}
