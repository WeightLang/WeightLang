//! Typed model operations lowered to the native tensor instruction set.
use crate::{
    adaptive::{self, Instruction, Program},
    model::{ParameterSchema, validate_shape},
    script::{
        self,
        parse::{Expr, Statement, Stmt},
    },
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod composition;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgumentSchema {
    pub name: String,
    pub width: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Effects {
    pub reads_state: Vec<String>,
    pub writes_state: Vec<String>,
    pub writes_parameters: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationSchema {
    pub name: String,
    pub inputs: Vec<ArgumentSchema>,
    pub output: Option<usize>,
    pub effects: Effects,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSchema {
    pub name: String,
    pub shape: Vec<usize>,
}
pub struct OperationsPlan {
    pub parameters: Vec<adaptive::Parameter>,
    pub states: Vec<adaptive::Parameter>,
    pub entries: Vec<(OperationSchema, Program)>,
}

pub(crate) fn lower(text: &str, materialize: bool) -> Result<OperationsPlan> {
    lower_with_types(text, materialize, false).map(|(plan, _)| plan)
}

pub(crate) fn analyze(text: &str) -> Result<(OperationsPlan, adaptive::InferredTypes)> {
    lower_with_types(text, false, true)
}

fn lower_with_types(text: &str, materialize: bool, collect_types: bool) -> Result<(OperationsPlan, adaptive::InferredTypes)> {
    let flat = composition::flatten(text)?;
    let functions = flat.functions;
    let initialization_helpers = crate::initializer::helpers(&flat.declarations, &functions)?;
    ensure!(
        !text.contains("wgt_operation_input") && !text.contains("wgt_operation_output"),
        "reserved operation compiler name"
    );
    ensure!(
        !functions.contains_key("loss"),
        "typed operations do not support loss or cross-state differentiation; use legacy forward training"
    );
    let mut declarations = String::new();
    let mut state_names = BTreeSet::new();
    for line in flat.declarations.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(body) = line.strip_prefix("state ") {
            let body = body.trim_end_matches(';').trim();
            let (name, rest) = body
                .split_once(':')
                .context("state requires name: f32[N] = initializer")?;
            let name = name.trim();
            ensure!(
                script::operators::identifier(name) && state_names.insert(name.to_owned()),
                "duplicate or invalid state name"
            );
            let ty = rest.split_once('=').map_or(rest, |(ty, _)| ty);
            crate::source::vector_port(&format!("{name}: {}", ty.trim()))?;
            declarations.push_str(&format!("parameter {body}\n"));
        } else {
            ensure!(
                !line.starts_with("input ")
                    && !line.starts_with("output ")
                    && !line.starts_with("target "),
                "typed operations declare ports in function signatures, not input/output/target declarations"
            );
            declarations.push_str(line);
            declarations.push('\n');
        }
    }
    let mut entries = Vec::new();
    let mut inferred = adaptive::InferredTypes::new();
    let mut storage = None;
    for (name, function) in functions.iter().collect::<BTreeMap<_, _>>() {
        if initialization_helpers.contains(name) && function.signature.is_none() {
            script::operators::validate(name, &functions, &mut Vec::new())?;
            continue;
        }
        ensure!(
            !function.is_async && name != "main" && !name.starts_with("__"),
            "model operation cannot be an application/async/internal entrypoint"
        );
        let void_signature = script::parse::OperationSignature {
            inputs: Vec::new(),
            output: None,
        };
        let signature = if let Some(signature) = &function.signature {
            signature
        } else if function.args.is_empty() {
            &void_signature
        } else {
            script::operators::validate(name, &functions, &mut Vec::new())?;
            continue;
        };
        ensure!(
            name != "loss",
            "loss is reserved for legacy forward differentiation"
        );
        let input_width = signature
            .inputs
            .iter()
            .try_fold(0usize, |n, w| n.checked_add(*w))
            .context("operation input size overflow")?;
        ensure!(
            input_width <= 16384,
            "operation aggregate input exceeds 16384 values"
        );
        let mut body = Vec::new();
        let mut start = 0;
        for (arg, width) in function.args.iter().zip(&signature.inputs) {
            ensure!(
                !state_names.contains(arg),
                "operation input shadows state: {arg}"
            );
            body.push(Statement {
                line: 1,
                kind: Stmt::Let(
                    arg.clone(),
                    Expr::Call(
                        "slice".into(),
                        vec![
                            Expr::Name("wgt_operation_input".into()),
                            Expr::Number(start as f64),
                            Expr::Number(*width as f64),
                        ],
                    ),
                ),
            });
            start += width;
        }
        let mut operation_body = function.body.clone();
        fn normalize(body: &mut [Statement], void: bool) -> Result<()> {
            for statement in body {
                match &mut statement.kind {
                    Stmt::Return(value) if void => {
                        ensure!(
                            matches!(value, Expr::Null),
                            "void operation cannot return a tensor"
                        );
                        *value = Expr::List(vec![Expr::Number(0.)]);
                    }
                    Stmt::If(_, yes, no) => {
                        normalize(yes, void)?;
                        normalize(no, void)?;
                    }
                    Stmt::While(_, body) => normalize(body, void)?,
                    Stmt::Eval(Expr::Call(op, _)) if op == "write" => {
                        anyhow::bail!("typed operations require explicit <- updates")
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        normalize(&mut operation_body, signature.output.is_none())?;
        body.extend(operation_body);
        if signature.output.is_none() {
            body.push(Statement {
                line: 1,
                kind: Stmt::Return(Expr::List(vec![Expr::Number(0.)])),
            });
        }
        let mut helpers = functions.clone();
        let mut forward = function.clone();
        forward.args = vec!["wgt_operation_input".into()];
        forward.signature = None;
        forward.body = body;
        helpers.insert("forward".into(), forward);
        let source = format!(
            "{declarations}\ninput wgt_operation_input: f32[{}]\noutput wgt_operation_output: f32[{}]\n",
            input_width.max(1),
            signature.output.unwrap_or(1)
        );
        let mut plan = if collect_types {
            let (plan, mut types) = adaptive::analyze_definition(&source, &helpers)
                .with_context(|| format!("operation {name}"))?;
            if let Some(public_name) = flat.entries.get(name) {
                if let Some(types) = types.remove("forward") { inferred.insert(public_name.clone(), types); }
            }
            plan
        } else {
            adaptive::lower_operation(&source, &helpers, materialize && storage.is_none())
                .with_context(|| format!("operation {name}"))?
        };
        let mut seen = BTreeSet::new();
        let mut total = 0usize;
        for parameter in &plan.parameters {
            ensure!(
                script::operators::identifier(&parameter.name) && seen.insert(&parameter.name),
                "duplicate parameter/state name: {}",
                parameter.name
            );
            ensure!(
                !function.args.contains(&parameter.name),
                "operation input shadows storage: {}",
                parameter.name
            );
            total = total
                .checked_add(validate_shape(&parameter.shape)?)
                .context("storage size overflow")?;
        }
        ensure!(
            total <= 16 * 1024 * 1024,
            "model storage element limit exceeded"
        );
        let mut effects = Effects::default();
        for instruction in &plan.instructions {
            match instruction {
                Instruction::Read { parameter, .. } if state_names.contains(parameter) => {
                    effects.reads_state.push(parameter.clone())
                }
                Instruction::Write { parameter, value } => {
                    let p = plan
                        .parameters
                        .iter()
                        .find(|p| &p.name == parameter)
                        .context("unknown update target")?;
                    let r = plan
                        .registers
                        .iter()
                        .find(|r| &r.name == value)
                        .context("unknown update value")?;
                    ensure!(
                        !p.frozen && p.shape == r.shape,
                        "frozen target or update shape mismatch: {parameter}"
                    );
                    ensure!(
                        !p.probability,
                        "transactional probability updates are unsupported"
                    );
                    if state_names.contains(parameter) {
                        effects.writes_state.push(parameter.clone());
                    } else {
                        effects.writes_parameters.push(parameter.clone());
                    }
                }
                _ => {}
            }
        }
        for names in [
            &mut effects.reads_state,
            &mut effects.writes_state,
            &mut effects.writes_parameters,
        ] {
            names.sort();
            names.dedup();
        }
        ensure!(
            !flat.readonly.contains(name)
                || effects.writes_parameters.is_empty() && effects.writes_state.is_empty(),
            "forward must not update state or parameters; use a named update operation"
        );
        if storage.is_none() {
            storage = Some(plan.parameters.clone());
        }
        for parameter in &mut plan.parameters {
            parameter.data = Vec::new();
        }
        if let Some(public_name) = flat.entries.get(name) { entries.push((
            OperationSchema {
                name: public_name.clone(),
                inputs: function
                    .args
                    .iter()
                    .zip(&signature.inputs)
                    .map(|(name, width)| ArgumentSchema {
                        name: name.clone(),
                        width: *width,
                    })
                    .collect(),
                output: signature.output,
                effects,
            },
            plan,
        )); }
    }
    ensure!(
        !entries.is_empty(),
        "model requires at least one typed operation"
    );
    let (states, parameters) = storage
        .unwrap()
        .into_iter()
        .partition(|p| state_names.contains(&p.name));
    Ok((OperationsPlan {
        parameters,
        states,
        entries,
    }, inferred))
}
impl OperationsPlan {
    pub fn parameter_schemas(&self) -> Vec<ParameterSchema> {
        self.parameters
            .iter()
            .map(|p| ParameterSchema {
                name: p.name.clone(),
                shape: p.shape.clone(),
                frozen: p.frozen,
                probability: p.probability,
                learn: p.learn.clone(),
                candidate_updatable: !p.frozen && !p.probability,
            })
            .collect()
    }
    pub fn state_schemas(&self) -> Vec<StateSchema> {
        self.states
            .iter()
            .map(|p| StateSchema {
                name: p.name.clone(),
                shape: p.shape.clone(),
            })
            .collect()
    }
}
