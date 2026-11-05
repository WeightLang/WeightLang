//! Model-only definitions. Parsing validates plans without materializing parameter storage.
use crate::{adaptive, categorical::VectorType, script, source};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_PARAMETER_ELEMENTS: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInterface {
    pub input: usize,
    pub output: usize,
    pub output_type: VectorType,
    pub target: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub shape: Vec<usize>,
    pub frozen: bool,
    pub learn: Option<adaptive::LearnDomain>,
}

/// Source and validated schemas; no initialized parameters, optimizer or service connection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelDefinition {
    source: String,
    name: String,
    interface: Option<ModelInterface>,
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
    pub fn parse(text: &str) -> Result<Self> {
        let (source, _) = crate::model_modules::resolve(text, None)?;
        Self::parse_resolved(&source)
    }
    fn parse_resolved(text: &str) -> Result<Self> {
        ensure!(text.len() <= MAX_SOURCE_BYTES, "model definition exceeds 1 MiB source limit");
        let unit = source::split(text)?;
        let functions = script::parse::parse_functions(&unit.functions)?;
        if functions.values().any(|f| f.signature.is_some()) || unit.declarations.lines().any(|l| l.trim().starts_with("state ")) || (!functions.contains_key("forward") && functions.values().any(|f| f.args.is_empty())) {
            let names: Vec<_> = unit.declarations.lines().filter_map(|line| line.split('#').next().unwrap_or("").trim().strip_prefix("model ")).collect();
            ensure!(names.len() == 1 && script::operators::identifier(names[0].trim()), "model definition requires one valid model name");
            let plan = crate::operations::lower(text, false)?;
            return Ok(Self { source: text.into(), name: names[0].trim().into(), interface: None, parameters: plan.parameter_schemas(), states: plan.state_schemas(), operations: plan.entries.into_iter().map(|(schema,_)|schema).collect() });
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
                Ok(ParameterSchema { name:p.name, shape:p.shape, frozen:p.frozen, learn:p.learn })
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
                        parameters.push(ParameterSchema{name,shape,frozen:false,learn:None});
                    }
                }
            }
            let output=graph.layers.last().context("model requires layers")?.output;
            (ModelInterface{input:graph.input,output,output_type:graph.output_type,target:None},parameters)
        };
        Ok(Self {source:text.into(),name,interface:Some(interface),parameters,operations:Vec::new(),states:Vec::new()})
    }
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_dependencies(path).map(|(definition, _)| definition)
    }
    /// Resolve and embed the full pure-computation dependency closure. Returned
    /// paths are build dependencies, never runtime lookup requirements.
    pub fn from_file_with_dependencies(path: impl AsRef<Path>) -> Result<(Self, Vec<std::path::PathBuf>)> {
        let path = path.as_ref();
        let source = crate::model_modules::read(path)?;
        let (source, dependencies) = crate::model_modules::resolve(&source, Some(path))?;
        let definition = Self::parse_resolved(&source).with_context(|| path.display().to_string())?;
        Ok((definition, dependencies))
    }
    pub fn name(&self) -> &str { &self.name }
    /// Legacy single-forward interface. For typed models use operations().
    pub fn interface(&self) -> Option<&ModelInterface> { self.interface.as_ref() }
    pub fn legacy_interface(&self) -> Option<&ModelInterface> { self.interface.as_ref() }
    pub fn operations(&self) -> &[crate::operations::OperationSchema] { &self.operations }
    pub fn states(&self) -> &[crate::operations::StateSchema] { &self.states }
    pub fn parameters(&self) -> &[ParameterSchema] { &self.parameters }
    pub fn source(&self) -> &str { &self.source }
    /// Explicit service-side initialization boundary; callers must enforce their memory budget.
    pub fn instantiate_plan(&self) -> Result<ModelPlan> {
        let unit = source::split(&self.source)?;
        let functions = script::parse::parse_functions(&unit.functions)?;
        if !self.operations.is_empty() {
            Ok(ModelPlan::Operations(crate::operations::lower(&self.source, true)?))
        } else if functions.contains_key("forward") {
            Ok(ModelPlan::Forward(adaptive::parse(&self.source)?))
        } else {
            Ok(ModelPlan::Graph(source::graph(&self.source)?.1))
        }
    }
}

pub(crate) fn validate_shape(shape: &[usize]) -> Result<usize> {
    ensure!(!shape.is_empty() && shape.len() <= 8, "parameter shape requires 1..8 axes");
    let count=shape.iter().try_fold(1usize,|count,width| {
        ensure!(*width > 0, "parameter axes must be positive");
        count.checked_mul(*width).context("parameter shape overflow")
    })?;
    ensure!(count <= MAX_PARAMETER_ELEMENTS, "parameter element limit exceeded");
    Ok(count)
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

// Stream the flat initializer during validation; never allocate a weight vector in parse/macros.
pub(crate) fn validate_initializer(text: &str, count: usize, materialize: bool) -> Result<Vec<f32>> {
    use serde::de::{DeserializeSeed, SeqAccess, Visitor};
    struct Initializer { count: usize, materialize: bool }
    impl<'de> Visitor<'de> for Initializer {
        type Value=Vec<f32>;
        fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result { f.write_str("a flat finite f32 initializer") }
        fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->std::result::Result<Self::Value,A::Error> {
            let mut values=Vec::new();
            let mut actual=0usize;
            while let Some(value)=seq.next_element::<f32>()? {
                if !value.is_finite() || actual >= self.count { return Err(serde::de::Error::custom("invalid initializer value or length")); }
                actual+=1;
                if self.materialize {values.push(value);}
            }
            if actual!=self.count {return Err(serde::de::Error::custom("initializer length does not match shape"));}
            Ok(values)
        }
    }
    impl<'de> DeserializeSeed<'de> for Initializer {
        type Value=Vec<f32>;
        fn deserialize<D:serde::Deserializer<'de>>(self,deserializer:D)->std::result::Result<Self::Value,D::Error> {deserializer.deserialize_seq(self)}
    }
    let mut decoder=serde_json::Deserializer::from_str(text);
    let values=Initializer{count,materialize}.deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(values)
}
