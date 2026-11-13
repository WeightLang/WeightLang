//! Model-only definitions. Parsing validates plans without materializing parameter storage.
use crate::{adaptive, categorical::VectorType, script, source};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_PARAMETER_ELEMENTS: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInterface {
    pub input: usize,
    pub output: usize,
    pub output_type: VectorType,
    pub target: Option<usize>,
}
/// One named logical f32 input within the packed native vector ABI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub width: usize,
}
/// One positional logical f32 output; tuple order is its stable identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardOutput {
    pub shape: Vec<usize>,
    pub offset: usize,
    pub width: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardInterface {
    pub inputs: Vec<ForwardInput>,
    pub outputs: Vec<ForwardOutput>,
    pub input: usize,
    pub output: usize,
    pub tuple: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub shape: Vec<usize>,
    pub frozen: bool,
    #[serde(default)]
    pub probability: bool,
    pub learn: Option<adaptive::LearnDomain>,
    /// Eligible for atomic external candidate updates; independent of AD support.
    /// Older schemas default to false rather than accidentally evolving frozen/prob storage.
    #[serde(default)]
    pub candidate_updatable: bool,
}

/// Source and validated schemas; no initialized parameters, optimizer or service connection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelDefinition {
    source: String,
    name: String,
    interface: Option<ModelInterface>,
    forward_interface: Option<ForwardInterface>,
    parameters: Vec<ParameterSchema>,
    operations: Vec<crate::operations::OperationSchema>,
    states: Vec<crate::operations::StateSchema>,
}
impl TryFrom<String> for ModelDefinition {
    type Error = anyhow::Error;
    fn try_from(source: String) -> Result<Self> { Self::parse(&source) }
}
impl From<ModelDefinition> for String {
    fn from(value: ModelDefinition) -> Self { value.source }
}

pub enum ModelPlan {
    Graph(source::Graph),
    Forward(adaptive::Program),
    Operations(crate::operations::OperationsPlan),
}
impl ModelDefinition {
    /// Bind named dimensions before validating a reusable model definition.
    pub fn parse_with_dimensions(text: &str, dimensions: &crate::dimensions::Dimensions) -> Result<Self> {
        crate::backend::preflight_source(text, crate::backend::Backend::Rust)?;
        Self::parse(&crate::dimensions::specialize(text, dimensions)?)
    }
    /// File imports remain relative to the original file; dimensions are bound
    /// only in that file, not injected into separately imported libraries.
    pub fn from_file_with_dimensions(path: impl AsRef<Path>, dimensions: &crate::dimensions::Dimensions) -> Result<Self> {
        let path = path.as_ref();
        let text = crate::model_modules::read(path)?;
        crate::backend::preflight_source(&text, crate::backend::Backend::Rust)?;
        let text = crate::dimensions::specialize(&text, dimensions)?;
        let (source, _) = crate::model_modules::resolve(&text, Some(path))?;
        Self::parse_resolved(&source)
    }
    pub fn parse(text: &str) -> Result<Self> {
        crate::backend::preflight_source(text, crate::backend::Backend::Rust)?;
        let (source, _) = crate::model_modules::resolve(text, None)?;
        Self::parse_resolved(&source)
    }
    pub(crate) fn parse_resolved(text: &str) -> Result<Self> {
        Self::parse_resolved_impl(text, false)
    }
    // Admission is private and used exclusively for output of lower_forward;
    // stored/serialized definitions always re-enter the ordinary source guards.
    fn parse_resolved_impl(text: &str, packed_forward: bool) -> Result<Self> {
        crate::backend::preflight_source(text, crate::backend::Backend::Rust)?;
        ensure!(text.len() <= MAX_SOURCE_BYTES, "model definition exceeds 1 MiB source limit");
        let unit = source::split(text)?;
        let functions = if packed_forward {
            script::parse::parse_packed_forward_functions(&unit.functions)?
        } else { script::parse::parse_functions(&unit.functions)? };
        if functions.get("forward").is_some_and(|f| f.forward_signature.is_some()) {
            let (lowered, ports) = crate::model_syntax::lower_forward(text)?;
            let mut definition = Self::parse_resolved_impl(&lowered, true)?;
            // Serialization reparses this resolved source to recover the exact
            // logical ports; generated packed code is never the source of truth.
            definition.source = text.into();
            definition.forward_interface = Some(ports);
            return Ok(definition);
        }
        if functions.values().any(|f| f.signature.is_some()) || unit.declarations.lines().any(|l| l.trim().starts_with("state ")) || (!functions.contains_key("forward") && functions.values().any(|f| f.args.is_empty())) {
            let names: Vec<_> = unit.declarations.lines().filter_map(|line| line.split('#').next().unwrap_or("").trim().strip_prefix("model ")).collect();
            ensure!(names.len() == 1 && script::operators::identifier(names[0].trim()), "model definition requires one valid model name");
            let plan = crate::operations::lower(text, false)?;
            return Ok(Self { source: text.into(), name: names[0].trim().into(), interface: None, forward_interface: None, parameters: plan.parameter_schemas(), states: plan.state_schemas(), operations: plan.entries.into_iter().map(|(schema,_)|schema).collect() });
        }
        for (name, function) in &functions {
            ensure!(name != "main" && !name.starts_with("__"), "model definitions cannot contain application entrypoints or agent/environment declarations: {name}");
            ensure!(!function.is_async, "model definitions cannot contain async functions");
            // Validate even unused helpers: a model cannot hide host execution in dead code.
            if name != "forward" && name != "loss" {
                script::operators::validate(name, &functions, &mut Vec::new())?;
            }
        }
        let names: Vec<_> = unit.declarations.lines().filter_map(|line| line.split('#').next().unwrap_or("").trim().strip_prefix("model ")).collect();
        ensure!(names.len() == 1 && script::operators::identifier(names[0].trim()), "model definition requires one valid model name");
        let name = names[0].trim().to_owned();
        let (interface, parameters) = if functions.contains_key("forward") {
            let plan = adaptive::validate_definition(text, &functions)?;
            let input = plan.registers.iter().find(|r| r.name == plan.input).context("missing input register")?.len;
            let output = unit.declarations.lines().find_map(|line| line.trim().strip_prefix("output ")).context("missing output declaration")?;
            let (_, output, output_type) = source::typed_vector_port(output)?;
            let target = unit.declarations.lines().find_map(|line| line.trim().strip_prefix("target "))
                .map(source::vector_port).transpose()?.map(|(_,width)| width)
                .or_else(|| plan.loss.as_ref().map(|_| output));
            let mut seen = std::collections::HashSet::new();
            let mut total = 0usize;
            let parameters = plan.parameters.into_iter().map(|p| {
                ensure!(script::operators::identifier(&p.name) && p.name != plan.input && seen.insert(p.name.clone()), "duplicate or invalid parameter name: {}", p.name);
                let count = validate_shape(&p.shape)?;
                total = total.checked_add(count).context("parameter size overflow")?;
                ensure!(total <= MAX_PARAMETER_ELEMENTS, "model parameter element limit exceeded");
                Ok(ParameterSchema { name:p.name, shape:p.shape, frozen:p.frozen, probability:p.probability, learn:p.learn, candidate_updatable:!p.frozen && !p.probability })
            }).collect::<Result<Vec<_>>>()?;
            (ModelInterface {input,output,output_type,target}, parameters)
        } else {
            ensure!(functions.is_empty(), "sequential graph definitions cannot contain application functions or loss without forward");
            let (_, graph) = source::graph(&unit.declarations)?;
            let mut parameters = Vec::new();
            let mut total = 0usize;
            for layer in &graph.layers {
                if layer.op == "linear" {
                    for (name,shape) in std::iter::once((format!("{}.weight",layer.name),vec![layer.output,layer.input]))
                        .chain(layer.bias.then(||(format!("{}.bias",layer.name),vec![layer.output]))) {
                        total=total.checked_add(validate_shape(&shape)?).context("parameter size overflow")?;
                        ensure!(total <= MAX_PARAMETER_ELEMENTS, "model parameter element limit exceeded");
                        parameters.push(ParameterSchema{name,shape,frozen:false,probability:false,learn:None,candidate_updatable:true});
                    }
                }
            }
            let output=graph.layers.last().context("model requires layers")?.output;
            (ModelInterface{input:graph.input,output,output_type:graph.output_type,target:None},parameters)
        };
        Ok(Self {source:text.into(),name,interface:Some(interface),forward_interface:None,parameters,operations:Vec::new(),states:Vec::new()})
    }
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_dependencies(path).map(|(definition, _)| definition)
    }
    /// Resolve and embed the full pure-computation dependency closure. Returned
    /// paths are build dependencies, never runtime lookup requirements.
    pub fn from_file_with_dependencies(path: impl AsRef<Path>) -> Result<(Self, Vec<std::path::PathBuf>)> {
        let path = path.as_ref();
        let source = crate::model_modules::read(path)?;
        crate::backend::preflight_source(&source, crate::backend::Backend::Rust)?;
        let (source, dependencies) = crate::model_modules::resolve(&source, Some(path))?;
        let definition = Self::parse_resolved(&source).with_context(|| path.display().to_string())?;
        Ok((definition, dependencies))
    }
    pub fn name(&self) -> &str { &self.name }
    /// Legacy single-forward interface. For typed models use operations().
    pub fn interface(&self) -> Option<&ModelInterface> { self.interface.as_ref() }
    pub fn forward_interface(&self) -> Option<&ForwardInterface> { self.forward_interface.as_ref() }
    pub fn legacy_interface(&self) -> Option<&ModelInterface> { self.interface.as_ref() }
    pub fn operations(&self) -> &[crate::operations::OperationSchema] { &self.operations }
    pub fn states(&self) -> &[crate::operations::StateSchema] { &self.states }
    pub fn parameters(&self) -> &[ParameterSchema] { &self.parameters }
    pub fn source(&self) -> &str { &self.source }
    /// Explicit service-side initialization boundary; callers must enforce their memory budget.
    pub fn instantiate_plan(&self) -> Result<ModelPlan> {
        self.instantiate_plan_with_modules().map(|(plan, _)| plan)
    }
    /// Instantiate the same native plan and retain source-level submodule identities.
    /// This performs no forward evaluation or service interaction.
    pub fn instantiate_plan_with_modules(&self) -> Result<(ModelPlan, Vec<adaptive::ModuleSchema>)> {
        let (mut plan, modules) = self.compile_plan_with_modules()?;
        materialize_plan(&mut plan, None)?;
        Ok((plan, modules))
    }
    /// Lower architecture and initializer recipes without generating weights.
    pub fn compile_plan_with_modules(&self) -> Result<(ModelPlan, Vec<adaptive::ModuleSchema>)> {
        let lowered;
        let source = if self.forward_interface.is_some() {
            lowered = crate::model_syntax::lower_forward(&self.source)?.0;
            lowered.as_str()
        } else { self.source.as_str() };
        let unit = source::split(source)?;
        let functions = if self.forward_interface.is_some() {
            script::parse::parse_packed_forward_functions(&unit.functions)?
        } else { script::parse::parse_functions(&unit.functions)? };
        if !self.operations.is_empty() {
            Ok((ModelPlan::Operations(crate::operations::lower(&self.source, false)?), Vec::new()))
        } else if functions.contains_key("forward") {
            let (program, modules) = if self.forward_interface.is_some() {
                adaptive::compile_packed_forward_with_metadata(source)?
            } else { adaptive::compile_with_metadata(source)? };
            Ok((ModelPlan::Forward(program), modules))
        } else {
            let mut graph = source::graph(&self.source)?.1;
            graph.parameters = Some(graph.parameter_definitions()?);
            Ok((ModelPlan::Graph(graph), Vec::new()))
        }
    }
    /// Complete external parameter bindings bypass every parameter initializer.
    /// State recipes remain independent and are explicitly initialized once.
    pub fn instantiate_plan_with_bindings(&self, bindings: &BTreeMap<String, Vec<f32>>) -> Result<ModelPlan> {
        ensure!(bindings.len() == self.parameters.len(), "weight binding parameter key set mismatch");
        for schema in &self.parameters {
            schema.validate_values(bindings.get(&schema.name).with_context(|| format!("missing weight binding: {}", schema.name))?)?;
        }
        let (mut plan, _) = self.compile_plan_with_modules()?;
        materialize_plan(&mut plan, Some(bindings))?;
        Ok(plan)
    }
    /// Retrieve one declared recipe; unrelated parameter initializers never run.
    pub fn parameter_initializer(&self, name: &str) -> Result<crate::initializer::Recipe> {
        let (plan, _) = self.compile_plan_with_modules()?;
        let parameters = match plan {
            ModelPlan::Graph(graph) => graph.parameter_definitions()?,
            ModelPlan::Forward(program) => program.parameters,
            ModelPlan::Operations(plan) => plan.parameters,
        };
        parameters.into_iter().find(|p| p.name == name)
            .with_context(|| format!("unknown declared parameter: {name}"))?
            .initializer.context("parameter has no initializer recipe")
    }
    pub fn initialize_parameter(&self, name: &str) -> Result<Vec<f32>> {
        let schema = self.parameters.iter().find(|p| p.name == name).context("unknown declared parameter")?;
        let values = self.parameter_initializer(name)?.materialize(&schema.shape)?;
        schema.validate_values(&values)?;
        Ok(values)
    }
}

impl ParameterSchema {
    /// Validate storage-domain values. Discrete learn shadows remain continuous;
    /// no rounding, projection or implicit logit/probability conversion occurs.
    pub fn validate_values(&self, values: &[f32]) -> Result<()> {
        ensure!(validate_shape(&self.shape)? == values.len(), "weight binding shape mismatch: {}", self.name);
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite weight binding: {}", self.name);
        for value in values {
            if let Some(domain) = &self.learn { domain.validate(*value).with_context(|| format!("parameter {}", self.name))?; }
            if self.probability && self.frozen { ensure!((0. ..=1.).contains(value), "frozen probability binding outside [0,1]: {}", self.name); }
        }
        Ok(())
    }
}

fn materialize_plan(plan: &mut ModelPlan, bindings: Option<&BTreeMap<String, Vec<f32>>>) -> Result<()> {
    fn fill(parameter: &mut adaptive::Parameter, bindings: Option<&BTreeMap<String, Vec<f32>>>) -> Result<()> {
        parameter.data = if let Some(bindings) = bindings {
            bindings.get(&parameter.name).with_context(|| format!("missing weight binding: {}", parameter.name))?.clone()
        } else {
            parameter.initializer.as_ref().context("missing initializer recipe")?
                .materialize(&parameter.shape).with_context(|| format!("parameter {}", parameter.name))?
        };
        ParameterSchema { name: parameter.name.clone(), shape: parameter.shape.clone(), frozen: parameter.frozen, probability: parameter.probability, learn: parameter.learn.clone(), candidate_updatable: !parameter.frozen && !parameter.probability }.validate_values(&parameter.data)
    }
    match plan {
        ModelPlan::Graph(graph) => {
            let mut parameters = graph.parameter_definitions()?;
            for p in &mut parameters { fill(p, bindings)?; }
            graph.parameters = Some(parameters);
        },
        ModelPlan::Forward(program) => { for p in &mut program.parameters { fill(p, bindings)?; } },
        ModelPlan::Operations(plan) => {
            for p in &mut plan.parameters { fill(p, bindings)?; }
            for p in &mut plan.states { fill(p, None)?; }
        },
    }
    Ok(())
}

pub(crate) fn validate_shape(shape: &[usize]) -> Result<usize> {
    crate::tensor::validate_shape(shape, crate::tensor::ShapeLimits::PARAMETER)
        .context("invalid parameter shape")
}

#[cfg(test)]
mod tests {
    #[test]
    fn static_lowering_never_materializes_parameter_data() {
        let source="wgt 0.5\nmodel Scale\ninput x: f32[1]\noutput y: f32[1]\nparameter gain: f32[1] = [2]\nfn forward(x) { return x * gain; }\n";
        let unit=crate::source::split(source).unwrap();
        let functions=crate::script::parse::parse_functions(&unit.functions).unwrap();
        let validated=crate::adaptive::validate_definition(source,&functions).unwrap();
        assert!(!validated.parameters.is_empty());
        assert!(validated.parameters.iter().all(|parameter|parameter.data.is_empty()));
        let initialized=crate::adaptive::parse(source).unwrap();
        assert_eq!(initialized.parameters[0].data,vec![2.]);
    }
}
