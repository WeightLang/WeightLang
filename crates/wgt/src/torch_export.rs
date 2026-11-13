//! Model-only bridge to PyTorch. The native frontend remains the sole WGT parser.
use anyhow::{Context, Result};
use serde::Serialize;
use std::{io::Write, path::Path};
use wgt_language::{
    adaptive::{ModuleSchema, Parameter, Program},
    model::{ModelDefinition, ModelInterface, ModelPlan, ParameterSchema},
    operations::OperationSchema,
    source::Graph,
};

pub const FORMAT: &str = "weightlang.torch.v1";

/// Versioned architecture and declared initializers, without trained weights or
/// optimizer/service state. The nested graph/program uses the native frontend IR.
#[derive(Serialize)]
pub struct Architecture {
    pub format: &'static str,
    pub name: String,
    pub interface: Option<ModelInterface>,
    pub parameters: Vec<ParameterSchema>,
    pub source: String,
    pub modules: Vec<ModuleSchema>,
    /// Stable symbolic-source identity for per-shape compilation; not a weight digest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_interface: Option<wgt_language::model::ForwardInterface>,
    #[serde(flatten)]
    pub plan: Plan,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Plan {
    TorchModule {
        torch_module: wgt_language::torch_model::TorchModel,
    },
    Sequential {
        graph: Graph,
        graph_parameters: Vec<Parameter>,
    },
    Forward {
        program: Program,
    },
    Operations {
        operations: Operations,
    },
}

/// Typed entrypoints share one initialized storage set. Entry programs retain
/// their native parameter schemas, whose data arrays are intentionally empty.
#[derive(Serialize)]
pub struct Operations {
    pub parameters: Vec<Parameter>,
    pub states: Vec<Parameter>,
    pub entries: Vec<Operation>,
}

#[derive(Serialize)]
pub struct Operation {
    pub schema: OperationSchema,
    pub program: Program,
}

/// Resolve pure model imports and lower to native model IR for the torch SDK.
/// This copies declared initializers; it never executes forward or host code.
pub fn compile(path: &Path) -> Result<Architecture> {
    compile_with_dimensions(path, &Default::default())
}

pub fn compile_with_dimensions(path: &Path, dimensions: &wgt_language::dimensions::Dimensions) -> Result<Architecture> {
    let source = wgt_language::source::read_model_source(path)?;
    wgt_language::backend::preflight_source(&source, wgt_language::backend::Backend::Torch)?;
    if wgt_language::torch_model::is_torch_model(&source) {
        return compile_source_with_dimensions(&source, dimensions);
    }
    let definition = ModelDefinition::from_file_with_dimensions(path, dimensions)
        .with_context(|| format!("invalid model definition {}", path.display()))?;
    from_definition(definition)
}

/// Compile an in-memory definition; relative imports require the file entrypoint.
pub fn compile_source(source: &str) -> Result<Architecture> {
    compile_source_with_dimensions(source, &Default::default())
}

pub fn compile_source_with_dimensions(source: &str, dimensions: &wgt_language::dimensions::Dimensions) -> Result<Architecture> {
    wgt_language::backend::preflight_source(source, wgt_language::backend::Backend::Torch)?;
    if wgt_language::torch_model::is_torch_model(source) {
        anyhow::ensure!(dimensions.is_empty(), "torch module dimensions are explicit constructor arguments");
        let module = wgt_language::torch_model::parse(source)?;
        return Ok(Architecture {
            format: FORMAT, name: module.models[0].clone(), interface: None,
            parameters: Vec::new(), source: source.to_owned(), modules: Vec::new(), template_fingerprint: None, forward_interface: None,
            plan: Plan::TorchModule { torch_module: module },
        });
    }
    from_definition(ModelDefinition::parse_with_dimensions(source, dimensions).context("invalid model definition")?)
}

fn from_definition(definition: ModelDefinition) -> Result<Architecture> {
    let interface = definition.interface().cloned();
    let (native_plan, modules) = definition.compile_plan_with_modules()?;
    let plan = match native_plan {
        ModelPlan::Graph(graph) => Plan::Sequential {
            graph_parameters: graph.parameter_definitions()?,
            graph,
        },
        ModelPlan::Forward(program) => {
            program.validate_tensor_contracts()?;
            Plan::Forward { program }
        }
        ModelPlan::Operations(plan) => Plan::Operations {
            operations: Operations {
                parameters: plan.parameters,
                states: plan.states,
                entries: plan
                    .entries
                    .into_iter()
                    .map(|(schema, program)| {
                        program
                            .validate_tensor_contracts()
                            .with_context(|| format!("invalid operation {}", schema.name))?;
                        Ok(Operation { schema, program })
                    })
                    .collect::<Result<_>>()?,
            },
        },
    };
    Ok(Architecture {
        format: FORMAT,
        name: definition.name().to_owned(),
        interface,
        parameters: definition.parameters().to_vec(),
        source: definition.source().to_owned(),
        modules,
        template_fingerprint: None,
        forward_interface: definition.forward_interface().cloned(),
        plan,
    })
}

/// Write only a new architecture file, publishing it after complete serialization.
pub fn write(architecture: &Architecture, output: &Path) -> Result<()> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut stage = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut stage, architecture)?;
    writeln!(stage)?;
    stage.persist_noclobber(output).with_context(|| {
        format!(
            "cannot create {} (existing files are never overwritten)",
            output.display()
        )
    })?;
    Ok(())
}
