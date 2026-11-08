//! Model-only bridge to PyTorch. The native frontend remains the sole WGT parser.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{io::Write, path::Path};
use wgt_language::{
    adaptive::{Instruction, LearnDomain, Program},
    model::{ModelDefinition, ModelInterface, ModelPlan, ParameterSchema},
    source::Graph,
};

pub const FORMAT: &str = "weightlang.torch.v1";

/// Versioned architecture and declared initializers, without trained weights or
/// optimizer/service state. The nested graph/program uses the native frontend IR.
#[derive(Serialize)]
pub struct Architecture {
    pub format: &'static str,
    pub name: String,
    pub interface: ModelInterface,
    pub parameters: Vec<ParameterSchema>,
    pub source: String,
    #[serde(flatten)]
    pub plan: Plan,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Plan {
    Sequential { graph: Graph },
    Forward { program: Program },
}

/// Resolve pure model imports and lower to a graph supported by the torch SDK.
/// This copies declared initializers; it never executes forward or host code.
pub fn compile(path: &Path) -> Result<Architecture> {
    let definition = ModelDefinition::from_file(path)
        .with_context(|| format!("invalid model definition {}", path.display()))?;
    from_definition(definition)
}

/// Compile an in-memory definition; relative imports require the file entrypoint.
pub fn compile_source(source: &str) -> Result<Architecture> {
    from_definition(ModelDefinition::parse(source).context("invalid model definition")?)
}

fn from_definition(definition: ModelDefinition) -> Result<Architecture> {
    let interface = definition.interface().context(
        "PyTorch export requires a single forward model; typed operations and persistent state are not supported",
    )?.clone();
    let plan = match definition.instantiate_plan()? {
        ModelPlan::Graph(graph) => Plan::Sequential { graph },
        ModelPlan::Forward(program) => {
            validate_program(&program)?;
            Plan::Forward { program }
        }
        ModelPlan::Operations(_) => {
            anyhow::bail!("PyTorch export does not support typed operations or persistent state")
        }
    };
    Ok(Architecture {
        format: FORMAT,
        name: definition.name().to_owned(),
        interface,
        parameters: definition.parameters().to_vec(),
        source: definition.source().to_owned(),
        plan,
    })
}

fn validate_program(program: &Program) -> Result<()> {
    program.validate_tensor_contracts()?;
    for parameter in &program.parameters {
        ensure!(
            matches!(&parameter.learn, None | Some(LearnDomain::Float)),
            "PyTorch export does not support constrained learn parameter '{}'; use an ordinary float parameter with a Python optimizer",
            parameter.name
        );
    }
    for instruction in &program.instructions {
        use Instruction::*;
        ensure!(
            !matches!(
                instruction,
                Write { .. }
                    | MemoryAlloc { .. }
                    | MemoryFree { .. }
                    | MemoryRead { .. }
                    | MemoryWrite { .. }
                    | ResourceState { .. }
                    | InferenceSkip { .. }
                    | Branch { .. }
                    | Jump { .. }
                    | Label { .. }
            ),
            "PyTorch export does not support '{}' instructions; this target requires a pure forward graph without dynamic control flow or mutable state",
            instruction.contract().name
        );
    }
    if let Some(loss) = &program.loss {
        validate_program(loss).context("PyTorch loss export")?;
    }
    Ok(())
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
