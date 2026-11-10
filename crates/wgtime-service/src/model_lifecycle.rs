//! The same admission and preparation boundary serves explicit initialization
//! and complete weight restoration. Restoration never executes parameter recipes.
use super::{Connection, Session, Shared, admission};
use anyhow::{Context, Result, ensure};
use wgtime_core::training::Optimizer;

impl Connection {
    pub(super) fn initialize_definition(
        &mut self,
        source: String,
        memory: usize,
        optimizer: Option<String>,
        weights: Option<&[u8]>,
        shared: &Shared,
    ) -> Result<serde_json::Value> {
        ensure!(source.len() <= 1024 * 1024, "definition exceeds 1 MiB");
        let admission = self
            .sessions
            .admit(memory)
            .context("definition admission budget exceeded")?;
        ensure!(
            optimizer
                .as_deref()
                .is_none_or(|s| matches!(s, "sgd" | "adam")),
            "unsupported optimizer"
        );
        let definition = wgt_language::model::ModelDefinition::parse(&source)?;
        let required = admission::storage_bytes(
            definition
                .parameters()
                .iter()
                .map(|p| p.shape.as_slice())
                .chain(definition.states().iter().map(|s| s.shape.as_slice())),
            if definition.operations().is_empty() {
                1
            } else {
                4
            },
        )?;
        ensure!(
            required <= memory,
            "definition storage memory budget exceeded"
        );
        let plan = if let Some(bytes) = weights {
            let schema = definition
                .parameters()
                .iter()
                .map(|p| (p.name.clone(), p.shape.clone()))
                .collect::<Vec<_>>();
            let values = wgt_language::weight_artifact::decode(&schema, bytes, 64 * 1024 * 1024)?;
            let bindings = schema
                .into_iter()
                .zip(values)
                .map(|((name, _), values)| (name, values))
                .collect();
            definition.instantiate_plan_with_bindings(&bindings)?
        } else {
            definition.instantiate_plan()?
        };
        let handle = shared.next_handle()?;
        let program = match plan {
            wgt_language::model::ModelPlan::Graph(graph) => {
                let model = wgtime_core::source::Graph(graph).instantiate_bounded(memory)?;
                wgtime_core::compiled_forward::CompiledForward::model_program(&model)?
            }
            wgt_language::model::ModelPlan::Forward(program) => program,
            wgt_language::model::ModelPlan::Operations(plan) => {
                ensure!(
                    optimizer.is_none(),
                    "typed operations do not support optimizers or cross-state differentiation"
                );
                let session = wgtime_core::operations::OperationSession::prepare(plan, memory)?;
                admission.insert(handle, Session::Operations(session));
                return Ok(serde_json::json!({"handle":handle,"backend":"native-operations"}));
            }
        };
        let mut session = wgtime_core::adaptive::AdaptiveSession::prepare(program, memory)?;
        session.enable_compiled()?;
        if let Some(name) = optimizer {
            session.set_optimizer(if name == "adam" {
                Optimizer::Adam
            } else {
                Optimizer::Sgd
            })?;
        }
        let backend = session.execution_backend();
        admission.insert(handle, Session::Adaptive(session));
        Ok(serde_json::json!({"handle":handle,"backend":backend}))
    }
}
