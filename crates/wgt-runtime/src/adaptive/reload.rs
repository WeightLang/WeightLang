use super::AdaptiveSession;
use anyhow::{ensure, Context, Result};

impl AdaptiveSession {
    /// Replace validated state without changing the enclosing Mounted identity.
    pub(crate) fn reload_candidate(&mut self, mut candidate: Self) -> Result<()> {
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
                }],
                parameters: vec![crate::adaptive::Parameter {
                    name: "w".into(),
                    shape: vec![1],
                    data: vec![data],
                    frozen: false,
                    probability: false,
                }],
                instructions: vec![crate::adaptive::Instruction::Return { value: "x".into() }],
                loss: None,
            },
            1024 * 1024,
        )
        .unwrap()
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
