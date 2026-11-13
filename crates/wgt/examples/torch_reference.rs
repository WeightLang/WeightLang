//! Native differential oracle for the Python SDK tests; not a service or SDK backend.
//! Read one JSON request from stdin and write one JSON result to stdout.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, io};
use wgt_language::{
    adaptive::Program,
    model::{ModelDefinition, ModelPlan},
};
use wgtime_core::{
    adaptive::AdaptiveSession,
    operations::OperationSession,
    training::{Objective, Optimizer, Trainer},
};

type Values = BTreeMap<String, Vec<f32>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: String,
    #[serde(default)]
    state_link: Option<wgtime_core::adaptive::recurrent::StateLink>,
    #[serde(default)]
    supervised: Option<wgtime_core::adaptive::recurrent::OutputSlice>,
    #[serde(default)]
    sequence_targets: Option<Vec<Option<Vec<f32>>>>,
    #[serde(default)]
    optimizer: Option<String>,
    #[serde(default)]
    dimensions: wgt_language::dimensions::Dimensions,
    #[serde(default)]
    input_shapes: BTreeMap<String, Vec<usize>>,
    #[serde(default)]
    dictionary: Option<wgtime_core::adaptive::research::FeatureDictionary>,
    #[serde(default)]
    instruction: Option<usize>,
    #[serde(default)]
    occurrence: usize,
    #[serde(default)]
    input: Vec<f32>,
    #[serde(default)]
    inputs: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    gradient: Option<Vec<f32>>,
    #[serde(default)]
    target: Option<Vec<f32>>,
    #[serde(default)]
    targets: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_rate")]
    rate: f32,
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default)]
    bindings: Values,
    #[serde(default)]
    calls: Vec<Call>,
    #[serde(default = "default_work")]
    max_work: u64,
    #[serde(default = "default_memory")]
    memory_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    name: String,
    #[serde(default)]
    inputs: Vec<Vec<f32>>,
    #[serde(default)]
    max_work: Option<u64>,
}

fn default_mode() -> String {
    "forward".into()
}
fn default_work() -> u64 {
    1_000_000
}
fn default_memory() -> usize {
    64 * 1024 * 1024
}
fn default_rate() -> f32 {
    0.01
}
fn default_steps() -> usize {
    1
}

fn parameters(program: &Program) -> Values {
    program
        .parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.data.clone()))
        .collect()
}

fn storage(values: Vec<(String, Vec<usize>, Vec<f32>)>) -> Values {
    values
        .into_iter()
        .map(|(name, _, data)| (name, data))
        .collect()
}

fn adaptive(program: Program, request: &Request) -> Result<Value> {
    let mut session = AdaptiveSession::prepare(program, request.memory_bytes)?;
    if request.mode == "features" {
        let seed = request.gradient.as_deref().context("feature mode requires gradient seed")?;
        let dictionary = request.dictionary.as_ref().context("feature mode requires dictionary")?;
        let instruction = request.instruction.context("feature mode requires instruction")?;
        return Ok(serde_json::to_value(session.feature_effects(&request.input, seed, dictionary,
            instruction, request.occurrence, request.max_work, request.memory_bytes)?)?);
    }
    if request.mode == "research" {
        ensure!(request.bindings.is_empty(), "research oracle uses declared parameter values");
        let seed = request.gradient.as_deref().context("research mode requires gradient seed")?;
        let attribution = session.attribute(&request.input, seed, request.max_work, request.memory_bytes)?;
        let circuit = session.circuit(&request.input, seed, 32, request.max_work, request.memory_bytes, 1e-6)?;
        return Ok(json!({"attribution": attribution, "circuit": circuit}));
    }
    if matches!(request.mode.as_str(), "recurrent" | "train_recurrent") {
        for (name, values) in &request.bindings { session.replace_parameter(name, values)?; }
        let link = request.state_link.context("recurrent mode requires state_link")?;
        let supervised = request.supervised.context("recurrent mode requires supervised output slice")?;
        let inputs = request.inputs.as_deref().context("recurrent mode requires inputs")?;
        let targets = request.sequence_targets.as_deref().context("recurrent mode requires sequence_targets")?;
        if request.mode == "train_recurrent" {
            session.set_optimizer(match request.optimizer.as_deref().unwrap_or("sgd") {
                "sgd" => Optimizer::Sgd, "adam" => Optimizer::Adam,
                _ => anyhow::bail!("recurrent optimizer must be sgd or adam"),
            })?;
            ensure!(request.steps > 0, "recurrent steps must be positive");
            let mut loss=0.;
            for _ in 0..request.steps {
                loss=session.train_recurrent(inputs,targets,link,supervised,request.rate,None,
                    &mut { request.max_work },request.memory_bytes)?;
            }
            return Ok(json!({"loss":loss,"parameters":parameters(session.program()),"revision":session.revision()}));
        }
        return Ok(serde_json::to_value(session.recurrent_gradients(inputs,targets,link,supervised,
            &mut { request.max_work },request.memory_bytes)?)?);
    }
    if request.mode == "train" {
        for (name, values) in &request.bindings {
            session.replace_parameter(name, values)?;
        }
        session.set_optimizer(Optimizer::Sgd)?;
        let inputs = request
            .inputs
            .clone()
            .unwrap_or_else(|| vec![request.input.clone()]);
        let targets = request
            .targets
            .clone()
            .or_else(|| request.target.clone().map(|target| vec![target]))
            .context("train mode requires targets")?;
        ensure!(request.steps > 0, "train steps must be positive");
        let mut losses = Vec::new();
        for _ in 0..request.steps {
            let mut remaining = request.max_work;
            losses.push(session.train_custom(
                &inputs,
                &targets,
                request.rate,
                &mut remaining,
                request.memory_bytes,
            )?);
        }
        return Ok(
            json!({"kind":"forward","loss":losses.last(),"losses":losses,"parameters":parameters(session.program()),"revision":session.revision()}),
        );
    }
    if request.mode == "forward" {
        let mut pass = session.forward_with_parameters(
            &request.input,
            &request.bindings,
            request.max_work,
            request.memory_bytes,
        )?;
        let mut result = json!({"kind":"forward","output":pass.output(),"parameters":parameters(session.program()),"steps":pass.steps()});
        let gradient = if let Some(target) = &request.target {
            let loss_program = session
                .program()
                .loss
                .as_deref()
                .context("target requires declared fn loss in the reference forward mode")?
                .clone();
            let mut loss = AdaptiveSession::prepare(loss_program, request.memory_bytes)?;
            let joined = pass
                .output()
                .iter()
                .chain(target)
                .copied()
                .collect::<Vec<_>>();
            let mut loss_pass = loss.forward(&joined, request.max_work, request.memory_bytes)?;
            loss_pass.backward(&[1.], &BTreeMap::new())?;
            result["loss"] = json!(loss_pass.output()[0]);
            result["target_gradient"] = json!(&loss_pass.input_gradient()?[pass.output().len()..]);
            loss_pass.input_gradient()?[..pass.output().len()].to_vec()
        } else {
            request
                .gradient
                .clone()
                .unwrap_or_else(|| vec![1.; pass.output().len()])
        };
        pass.backward(&gradient, &BTreeMap::new())?;
        let mut gradients = Values::new();
        let mut staged = Values::new();
        for name in pass.parameter_names() {
            gradients.insert(name.to_owned(), pass.parameter_gradient(name)?.to_vec());
            staged.insert(name.to_owned(), pass.state(name)?.to_vec());
        }
        result["gradients"] = json!(gradients);
        result["input_gradient"] = json!(pass.input_gradient()?);
        result["staged_parameters"] = json!(staged);
        return Ok(result);
    }
    ensure!(
        matches!(request.mode.as_str(), "run" | "evaluate"),
        "unknown reference mode"
    );
    for (name, values) in &request.bindings {
        session.replace_parameter(name, values)?;
    }
    let inputs = request
        .inputs
        .clone()
        .unwrap_or_else(|| vec![request.input.clone()]);
    let mut results = Vec::new();
    for input in inputs {
        let output = if request.mode == "run" {
            session.run(&input, request.max_work).map(<[f32]>::to_vec)
        } else {
            session.evaluate(&input, request.max_work)
        };
        let mut result = json!({"parameters":parameters(session.program()),"revision":session.revision(),"steps":session.steps()});
        match output {
            Ok(value) => result["output"] = json!(value),
            Err(error) => result["error"] = json!(format!("{error:#}")),
        }
        results.push(result);
    }
    Ok(
        json!({"kind":"forward","results":results,"parameters":parameters(session.program()),"revision":session.revision()}),
    )
}

fn operations(plan: wgt_language::operations::OperationsPlan, request: &Request) -> Result<Value> {
    ensure!(
        request.bindings.is_empty(),
        "operations reference uses explicit update operations rather than parameter bindings"
    );
    let mut session = OperationSession::prepare(plan, request.memory_bytes)?;
    let mut results = Vec::new();
    for call in &request.calls {
        let output = session.call(
            &call.name,
            &call.inputs,
            call.max_work.unwrap_or(request.max_work),
        );
        let mut result = json!({"parameters":storage(session.parameter_values()),"states":storage(session.state_values()),"revision":session.revision()});
        match output {
            Ok(value) => result["output"] = json!(value),
            Err(error) => result["error"] = json!(format!("{error:#}")),
        }
        results.push(result);
    }
    Ok(
        json!({"kind":"operations","results":results,"parameters":storage(session.parameter_values()),"states":storage(session.state_values()),"revision":session.revision()}),
    )
}

fn sequential(graph: wgt_language::source::Graph, request: &Request) -> Result<Value> {
    let mut model = wgtime_core::source::Graph(graph).instantiate_bounded(request.memory_bytes)?;
    for (name, values) in &request.bindings {
        let parameter = model
            .parameters
            .iter_mut()
            .find(|parameter| &parameter.name == name)
            .context("unknown graph binding")?;
        ensure!(
            values.len() == parameter.shape.iter().product::<usize>()
                && values.iter().all(|value| value.is_finite()),
            "invalid graph binding"
        );
        parameter.value = wgtime_core::Value::Resident {
            data: values.clone(),
        };
    }
    let initial = model
        .parameters
        .iter()
        .map(|parameter| {
            let wgtime_core::Value::Resident { data } = &parameter.value else {
                unreachable!()
            };
            (parameter.name.clone(), data.clone())
        })
        .collect::<Values>();
    let names = model
        .parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect::<Vec<_>>();
    let mut trainer = Trainer::inference(model, std::path::Path::new("."), request.memory_bytes)?;
    let output = trainer.predict(&request.input)?;
    let mut result = json!({"kind":"sequential","output":output,"parameters":initial});
    if let Some(target) = &request.target {
        let (loss, gradients) =
            trainer.gradients(&[request.input.clone()], Objective::Mse(&[target.clone()]))?;
        result["loss"] = json!(loss);
        result["gradients"] = json!(names.into_iter().zip(gradients).collect::<Values>());
    }
    Ok(result)
}

fn main() -> Result<()> {
    let request: Request = serde_json::from_reader(io::stdin().lock())?;
    let definition = if request.input_shapes.is_empty() {
        ModelDefinition::parse_with_dimensions(&request.source, &request.dimensions)?
    } else {
        wgt_language::model_template::ModelTemplate::parse(&request.source)?
            .bind_input_shapes(&request.input_shapes, &request.dimensions)?
    };
    let result = match definition.instantiate_plan()? {
        ModelPlan::Graph(graph) => sequential(graph, &request),
        ModelPlan::Forward(program) => adaptive(program, &request),
        ModelPlan::Operations(plan) => operations(plan, &request),
    }?;
    serde_json::to_writer(io::stdout().lock(), &result)?;
    Ok(())
}
