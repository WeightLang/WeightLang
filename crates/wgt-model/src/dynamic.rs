//! One service instance with shape-specific execution and stable parameter storage.
use crate::{
    Client, Dimensions, InstanceOptions, Model, ModelTemplate, Result, Targets, TrainOptions,
};
use anyhow::ensure;
use std::collections::BTreeMap;

type InputShapes = BTreeMap<String, Vec<usize>>;

/// Shape binding belongs to the shared frontend; the service retains one handle
/// and one optimizer. No secondary service instances or weight synchronization.
pub struct DynamicModel {
    template: ModelTemplate,
    dimensions: Dimensions,
    shapes: InputShapes,
    model: Model,
}

impl Client {
    /// Initialize once from a logical input shape. Later shapes cannot change
    /// parameters or their initializer/domain contracts.
    pub fn initialize_dynamic(
        &self,
        template: &ModelTemplate,
        input_shapes: &InputShapes,
        dimensions: &Dimensions,
        options: InstanceOptions,
    ) -> Result<DynamicModel> {
        let definition = template.bind_input_shapes(input_shapes, dimensions)?;
        ensure!(
            definition.interface().is_some() && definition.operations().is_empty(),
            "dynamic models require a differentiable forward definition"
        );
        let (plan, _) = definition.compile_plan_with_modules()?;
        let wgt_language::model::ModelPlan::Forward(program) = plan else {
            anyhow::bail!("dynamic models require a differentiable forward definition");
        };
        ensure!(
            !program.instructions.iter().any(|instruction| matches!(
                instruction,
                wgt_language::adaptive::Instruction::Write { .. }
            )),
            "dynamic models do not support persistent parameter writes"
        );
        let model = self.initialize(&definition, options)?;
        Ok(DynamicModel {
            template: template.clone(),
            dimensions: dimensions.clone(),
            shapes: input_shapes.clone(),
            model,
        })
    }
}

impl DynamicModel {
    /// Current concrete definition and the same live parameter/optimizer owner.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Atomic geometry preparation and admission. Failed rebinding leaves the
    /// previous geometry and live optimizer intact. One active plan is retained.
    pub fn specialize(&mut self, input_shapes: &InputShapes) -> Result<()> {
        if self.shapes == *input_shapes {
            return Ok(());
        }
        let definition = self
            .template
            .bind_input_shapes(input_shapes, &self.dimensions)?;
        self.model.rebind_geometry(&definition)?;
        self.shapes.clone_from(input_shapes);
        Ok(())
    }

    /// Values use the existing flat row-major f32 ABI; shape axes are explicit.
    pub fn forward(&mut self, input: &[f32], input_shapes: &InputShapes) -> Result<Vec<f32>> {
        let definition = self
            .template
            .bind_input_shapes(input_shapes, &self.dimensions)?;
        let width = definition
            .interface()
            .ok_or_else(|| anyhow::anyhow!("dynamic forward requires differentiable forward"))?
            .input;
        ensure!(
            input.len() == width && input.iter().all(|v| v.is_finite()),
            "input must be finite and match its logical shapes"
        );
        self.specialize(input_shapes)?;
        self.model.forward(input)
    }

    /// Logical multi-input counterpart. Ordering follows the source signature;
    /// output buffers follow its tuple order, with no implicit hidden mutation.
    pub fn forward_ports(
        &mut self,
        inputs: &[Vec<f32>],
        input_shapes: &InputShapes,
    ) -> Result<Vec<Vec<f32>>> {
        let definition = self
            .template
            .bind_input_shapes(input_shapes, &self.dimensions)?;
        let ports = definition
            .forward_interface()
            .ok_or_else(|| anyhow::anyhow!("model has no tuple forward interface"))?;
        ensure!(
            inputs.len() == ports.inputs.len()
                && inputs
                    .iter()
                    .zip(&ports.inputs)
                    .all(|(value, port)| value.len() == port.width
                        && value.iter().all(|v| v.is_finite())),
            "forward input ports do not match logical shapes"
        );
        self.specialize(input_shapes)?;
        self.model.forward_ports(inputs)
    }

    /// A batch shares one logical shape. Optimizer moments survive shape changes;
    /// changing sample cardinality does not reset SGD/Adam or trainable parameters.
    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        input_shapes: &InputShapes,
        targets: Targets<'_>,
        options: TrainOptions,
    ) -> Result<f32> {
        let definition = self
            .template
            .bind_input_shapes(input_shapes, &self.dimensions)?;
        let interface = definition
            .interface()
            .ok_or_else(|| anyhow::anyhow!("dynamic training requires forward"))?;
        ensure!(
            !inputs.is_empty()
                && inputs
                    .iter()
                    .all(|row| row.len() == interface.input && row.iter().all(|v| v.is_finite())),
            "training inputs must be finite and match their logical shape"
        );
        ensure!(
            options.rate.is_finite() && options.rate > 0.,
            "learning rate must be positive and finite"
        );
        ensure!(
            options.clip.is_none_or(|v| v.is_finite() && v > 0.),
            "gradient clip norm must be positive and finite"
        );
        match &targets {
            Targets::Values(rows) => {
                let width = if matches!(options.loss, crate::Loss::Custom) {
                    interface.target.unwrap_or(interface.output)
                } else {
                    interface.output
                };
                ensure!(
                    rows.len() == inputs.len()
                        && rows
                            .iter()
                            .all(|row| row.len() == width && row.iter().all(|v| v.is_finite())),
                    "targets must be finite and match the shaped output contract"
                );
                ensure!(
                    !matches!(options.loss, crate::Loss::CrossEntropy),
                    "loss and target kind do not match"
                );
            }
            Targets::Classes(rows) => {
                ensure!(
                    matches!(options.loss, crate::Loss::CrossEntropy)
                        && rows.len() == inputs.len()
                        && rows.iter().all(|index| *index < interface.output),
                    "class targets do not match shaped output contract"
                );
            }
        }
        self.specialize(input_shapes)?;
        self.model.train(inputs, targets, options)
    }

    pub fn close(self) -> Result<()> {
        self.model.close()
    }
}
