//! Explicit WGT host scripts, interpreted entirely in Rust. No Python bridge.
mod benchmark;
pub mod async_runtime;
mod async_codegen;
pub mod codegen;
pub mod compiled;
mod dataset;
pub(crate) mod file_io;
mod persistence;
mod math;
mod one_hot;
pub mod operators;
pub(crate) mod parse;
pub mod prelude;
mod resources;
mod openresponses;
mod modules;
mod permissions;
pub mod testing;
mod training_device;
mod optimization;
pub mod agents;
mod strategies;
pub mod record_schema;
pub mod records;
use crate::training::{mlp, random, Objective, Optimizer, Trainer};
use anyhow::{bail, ensure, Context, Result};
pub fn parse(source: &str) -> Result<Program> {
    parse_linked(source, Program::new())
}
pub fn parse_with_imports(
    source: &str,
    root: &Path,
    dependencies: &BTreeMap<String, String>,
) -> Result<Program> {
    let (source,modules)=modules::link(source,root)?;
    let (source, functions) = operators::link(&source, root, dependencies)?;
    let mut program=parse_linked(&source, functions)?;
    for (name,function) in modules {ensure!(program.insert(name,function).is_none(),"duplicate module function");}
    Ok(program)
}
fn parse_linked(source: &str, imported: Program) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let mut program = parse::parse_functions(&unit.functions)?;
    for (name, function) in &imported {
        ensure!(
            program.insert(name.clone(), function.clone()).is_none(),
            "duplicate imported operator: {name}"
        );
    }
    let model_name = unit
        .declarations
        .lines()
        .find_map(|l| {
            l.split('#')
                .next()
                .unwrap_or("")
                .trim()
                .strip_prefix("model ")
        })
        .map(str::trim);
    if let Some(name) = model_name {
        ensure!(
            !name.starts_with("__")
                && !name.is_empty()
                && name
                    .bytes()
                    .enumerate()
                    .all(|(i, c)| c.is_ascii_alphabetic()
                        || c == b'_'
                        || (i > 0 && c.is_ascii_digit())),
            "invalid model name"
        );
        ensure!(
            !program.contains_key(name) && !prelude::is_builtin(name) && prelude::migration(name).is_none(),
            "model name conflicts with function: {name}"
        );
        let (builtin, plan) = if program.contains_key("forward") {
            let plan = crate::adaptive::parse_with_functions(source, &program)?;
            program.remove("forward");
            program.remove("loss");
            ("__adaptive", serde_json::to_string(&plan)?)
        } else {
            (
                "__graph",
                serde_json::to_string(&crate::source::graph(&unit.declarations)?.1)?,
            )
        };
        program.insert(
            name.into(),
            parse::Function {
                is_async: false,
                args: vec!["optimizer".into()],
                body: vec![Statement {
                    line: 1,
                    kind: Stmt::Return(Expr::Call(
                        builtin.into(),
                        vec![Expr::Text(plan), Expr::Name("optimizer".into())],
                    )),
                }],
            },
        );
    } else {
        ensure!(
            unit.declarations
                .lines()
                .all(|l| l.trim().is_empty() || l.trim().starts_with('#') || l.trim() == "wgt 0.5"),
            "model declarations require model NAME"
        );
        ensure!(
            !program.contains_key("forward"),
            "forward requires model declarations"
        );
    }
    if let Some(main) = program.get("main") {
        ensure!(main.args.is_empty(), "main takes no arguments");
    } else {
        ensure!(
            model_name.is_some() || program.keys().any(|name| name.starts_with("test_")),
            "program requires fn main(), test_* functions or model declarations"
        );
    }
    for name in imported.keys() {
        program.remove(name);
    }
    Ok(program)
}
pub use parse::{Expr, Program, Statement, Stmt};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
    Bool(bool),
    Null,
    List(Vec<Value>),
    CategoricalVector(crate::categorical::VectorType, Vec<Value>),
    Object(BTreeMap<String, Value>),
    RecordBatch(records::RecordBatch),
    Model(usize),
    Adaptive(usize),
    Reader(usize),
    Kernel(usize),
    Gradient(u64),
    Future(usize),
    Task(usize),
    Agent(usize),
    Environment(usize),
    ResponseSession(usize),
}
impl Value {
    pub fn number(&self) -> Result<f64> {
        if let Self::Number(n) = self {
            Ok(*n)
        } else {
            bail!("expected number")
        }
    }
    pub fn integer(&self) -> Result<usize> {
        let n = self.number()?;
        ensure!(
            (0. ..=1_000_000_000.).contains(&n) && n.fract() == 0.,
            "expected integer in 0..1000000000"
        );
        Ok(n as usize)
    }
    fn float(&self) -> Result<f32> {
        let n = self.number()? as f32;
        ensure!(n.is_finite(), "number exceeds f32 range");
        Ok(n)
    }
    pub fn boolean(&self) -> Result<bool> {
        if let Self::Bool(b) = self {
            Ok(*b)
        } else {
            bail!("condition must be bool (no implicit tensor/number truth conversion)")
        }
    }
    pub fn text(&self) -> Result<&str> {
        if let Self::Text(s) = self {
            Ok(s)
        } else {
            bail!("expected string")
        }
    }
    pub fn list(&self) -> Result<&[Value]> {
        if let Self::List(v) | Self::CategoricalVector(_, v) = self {
            Ok(v)
        } else {
            bail!("expected list")
        }
    }
    fn vector(&self) -> Result<Vec<f32>> {
        self.list()?.iter().map(Self::float).collect()
    }
    fn matrix(&self) -> Result<Vec<Vec<f32>>> {
        self.list()?.iter().map(Self::vector).collect()
    }
    fn model(&self) -> Result<usize> {
        if let Self::Model(i) = self {
            Ok(*i)
        } else {
            bail!("expected model handle")
        }
    }
    fn reader(&self) -> Result<usize> {
        if let Self::Reader(i) = self {
            Ok(*i)
        } else {
            bail!("expected reader handle")
        }
    }
    fn json(&self) -> Result<serde_json::Value> {
        Ok(match self {
            Self::Number(n) => serde_json::json!(n),
            Self::Text(s) => serde_json::json!(s),
            Self::Bool(b) => serde_json::json!(b),
            Self::Null => serde_json::Value::Null,
            Self::List(v) | Self::CategoricalVector(_, v) => {
                serde_json::Value::Array(v.iter().map(Self::json).collect::<Result<_>>()?)
            }
            Self::RecordBatch(batch) => serde_json::Value::Array(batch.rows.iter().map(Self::json).collect::<Result<_>>()?),
            Self::Object(v) => serde_json::Value::Object(
                v.iter()
                    .map(|(k, v)| Ok((k.clone(), v.json()?)))
                    .collect::<Result<_>>()?,
            ),
            _ => bail!("resource handles cannot be returned as JSON"),
        })
    }
    fn from_json(v: serde_json::Value) -> Result<Self> {
        Ok(match v {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(b) => Self::Bool(b),
            serde_json::Value::String(s) => Self::Text(s),
            serde_json::Value::Number(n) => number(n.as_f64().context("invalid JSON number")?)?,
            serde_json::Value::Array(v) => {
                Self::List(v.into_iter().map(Self::from_json).collect::<Result<_>>()?)
            }
            serde_json::Value::Object(v) => Self::Object(
                v.into_iter()
                    .map(|(k, v)| Ok((k, Self::from_json(v)?)))
                    .collect::<Result<_>>()?,
            ),
        })
    }
}
fn number(n: f64) -> Result<Value> {
    ensure!(n.is_finite(), "nonfinite arithmetic result");
    Ok(Value::Number(n))
}
fn vector(v: Vec<f32>) -> Value {
    Value::List(v.into_iter().map(|n| Value::Number(n as f64)).collect())
}
fn typed_vector(v: Vec<f32>, kind: crate::categorical::VectorType) -> Result<Value> {
    let values: Vec<f64> = v.iter().map(|x| *x as f64).collect();
    if kind == crate::categorical::VectorType::Probabilities {
        crate::categorical::validate_probabilities(&values)?;
    } else {
        crate::categorical::validate_finite(&values)?;
    }
    if kind == crate::categorical::VectorType::F32 { return Ok(vector(v)); }
    Ok(Value::CategoricalVector(kind, values.into_iter().map(Value::Number).collect()))
}

/// Validate borrowed strategy results before cloning/serializing across IPC.
pub fn validate_strategy_result(value: &Value) -> Result<()> {
    fn visit(value: &Value, depth: usize, bytes: &mut usize) -> Result<()> {
        ensure!(depth <= 32, "strategy return nesting exceeds 32");
        *bytes = bytes.checked_add(64).context("strategy return size overflow")?;
        match value {
            Value::Number(n) => ensure!(n.is_finite(), "nonfinite strategy return"),
            Value::Null | Value::Bool(_) => {},
            Value::Text(text) => *bytes = bytes.checked_add(text.len()).context("strategy return size overflow")?,
            Value::List(values) => for value in values { visit(value, depth + 1, bytes)?; },
            Value::Object(values) => for (key, value) in values {
                *bytes = bytes.checked_add(key.len()).context("strategy return size overflow")?;
                visit(value, depth + 1, bytes)?;
            },
            _ => bail!("strategy return must be bounded plain data; resource handles cannot cross the model boundary"),
        }
        ensure!(*bytes <= 1024*1024, "strategy return exceeds 1 MiB");
        Ok(())
    }
    visit(value, 0, &mut 0)
}
pub type Env = HashMap<String, Value>;
pub enum Flow {
    Next,
    Return(Value),
    Break,
    Continue,
}

#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub result: serde_json::Value,
    pub metrics: BTreeMap<String, serde_json::Value>,
    pub operations: u64,
}

pub fn run_file(path: &Path, max_operations: u64) -> Result<Report> {
    let source = std::fs::read_to_string(path)?;
    run(
        &source,
        path.parent().unwrap_or(Path::new(".")),
        max_operations,
    )
    .with_context(|| path.display().to_string())
}
pub fn run(source: &str, root: &Path, max_operations: u64) -> Result<Report> {
    run_with_dependencies(source, root, max_operations, BTreeMap::new())
}
pub fn run_with_dependencies(
    source: &str,
    root: &Path,
    max_operations: u64,
    dependencies: BTreeMap<String, String>,
) -> Result<Report> {
    let (runtime, value) = execute(source, root, max_operations, dependencies)?;
    Ok(Report {
        result: value.json()?,
        metrics: runtime.metrics,
        operations: max_operations - runtime.remaining,
    })
}

/// Execute an explicit build entrypoint and retain its returned model for export.
pub fn build(source: &str, root: &Path, max_operations: u64) -> Result<(Trainer, Report)> {
    let (mut runtime, value) = execute(source, root, max_operations, BTreeMap::new())?;
    let index = value
        .model()
        .context("build main() must return a model handle")?;
    let trainer = runtime.models.swap_remove(index).into_inner()?;
    let report = Report {
        result: serde_json::Value::Null,
        metrics: runtime.metrics,
        operations: max_operations - runtime.remaining,
    };
    Ok((trainer, report))
}

fn execute(
    source: &str,
    root: &Path,
    max_operations: u64,
    dependencies: BTreeMap<String, String>,
) -> Result<(Runtime, Value)> {
    let program = parse_with_imports(source, root, &dependencies)?;
    ensure!(
        program.contains_key("main"),
        "execution requires fn main(); model construction is explicit"
    );
    execute_program(program, root, max_operations, dependencies, "main")
}

fn execute_program(
    program: Program,
    root: &Path,
    max_operations: u64,
    dependencies: BTreeMap<String, String>,
    entry: &str,
) -> Result<(Runtime, Value)> {
    let (permissions, resources) = resources::Resources::from_env()?;
    let mut runtime = Runtime {
        permissions,
        resources,
        program,
        compiled: HashMap::new(),
        root: root.into(),
        dependencies,
        remaining: max_operations,
        gpu_benchmark_restricted: false,
        depth: 0,
        models: Vec::new(),
        adaptive: Vec::new(),
        readers: Vec::new(),
        gradients: optimization::Gradients::default(),
        #[cfg(feature = "gpu")]
        kernels: Vec::new(),
        rng: 0,
        metrics: BTreeMap::new(),
        agents: Vec::new(), environments: Vec::new(),
        async_state: async_runtime::AsyncState::default(),
        harness: openresponses::Harness::default(),
    };
    let result = runtime.call(entry, Vec::new())?;
    Ok((runtime, result))
}
pub struct Runtime {
    permissions: permissions::Permissions,
    resources: resources::Resources,
    dependencies: BTreeMap<String, String>,
    program: Program,
    compiled: HashMap<String, compiled::Function>,
    root: PathBuf,
    remaining: u64,
    // Direct benchmark dispatch cannot honor a capability-scoped compute budget.
    gpu_benchmark_restricted: bool,
    depth: usize,
    models: Vec<crate::weightfs::Mounted<Trainer>>,
    adaptive: Vec<crate::weightfs::Mounted<crate::adaptive::AdaptiveSession>>,
    readers: Vec<Option<dataset::Reader>>,
    gradients: optimization::Gradients,
    #[cfg(feature = "gpu")]
    kernels: Vec<crate::wgsl::Kernel>,
    rng: u64,
    metrics: BTreeMap<String, serde_json::Value>,
    agents: Vec<agents::Agent>,
    environments: Vec<agents::Environment>,
    async_state: async_runtime::AsyncState,
    harness: openresponses::Harness,
}
impl Runtime {
    pub fn tick(&mut self) -> Result<()> {
        self.permissions.check_deadline()?;
        self.resources.check_deadline()?;
        if let Some(context) = &self.async_state.context { return Ok(context.charge(1)?); }
        if let Some(scope) = self.async_state.scopes.last() { return Ok(scope.charge(1)?); }
        self.remaining = self
            .remaining
            .checked_sub(1)
            .context("script operation budget exceeded")?;
        Ok(())
    }
    fn block(&mut self, body: &[Statement], env: &mut Env) -> Result<Flow> {
        for statement in body {
            self.tick()?;
            let flow = self
                .statement(&statement.kind, env)
                .with_context(|| format!("line {}", statement.line))?;
            if !matches!(flow, Flow::Next) {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }
    fn statement(&mut self, statement: &Stmt, env: &mut Env) -> Result<Flow> {
        match statement {
            Stmt::Tasks(body) => {
                self.enter_tasks()?;
                let result = self.block(body, env);
                let cleanup = self.leave_tasks(result.is_ok());
                let flow = result?;
                cleanup?;
                return Ok(flow);
            }
            Stmt::Let(name, expr) => {
                let value = self.eval(expr, env)?;
                env.insert(name.clone(), value);
            }
            Stmt::LetSample(bindings, name, input) => {
                self.tick()?; // Match the scalar sampling expression's evaluation cost.
                self.tick()?; // Model lookup precedes input effects.
                let model = env.get(name).cloned()
                    .with_context(|| format!("undefined variable: {name}"))?;
                let input = self.eval(input, env)?;
                let (action, log_prob) = self.sample_model_result(model, input)?;
                for (field, binding) in bindings {
                    env.insert(binding.clone(), if field == "action" { action.clone() } else { log_prob.clone() });
                }
            }
            Stmt::Learn(name, target, learner, body) => {
                let target = self.eval(target, env)?;
                let learner = self.resolve_learner(&target, if learner.is_empty() {None}else{Some(learner)})?;
                let token = self.begin_collection(&target)?;
                let previous = env.insert("__collect".into(), Value::RecordBatch(Default::default()));
                let result = self.block(body, env);
                let batch = env.remove("__collect").expect("active collector");
                if let Some(previous) = previous { env.insert("__collect".into(), previous); }
                drop(token);
                let flow = result?;
                if !matches!(flow, Flow::Next) { return Ok(flow); }
                let report = self.call(&learner, vec![target, batch])?;
                if let Some(name) = name { env.insert(name.clone(), report); }
            }
            Stmt::Collect(name, body) => {
                let previous = env.insert("__collect".into(), Value::RecordBatch(records::RecordBatch::default()));
                let result = self.block(body, env);
                let batch = env.remove("__collect").expect("active collector");
                if let Some(previous) = previous { env.insert("__collect".into(), previous); }
                let flow = result?;
                if !matches!(flow, Flow::Next) { return Ok(flow); }
                env.insert(name.clone(), batch);
            }
            Stmt::Yield(fields) => {
                ensure!(env.contains_key("__collect"), "yield requires a lexical collect block");
                let fields = fields.iter().map(|(name,value)| Ok((name.clone(), self.eval(value, env)?)))
                    .collect::<Result<Vec<_>>>()?;
                self.yield_record(env, fields)?;
            }
            Stmt::ForBatches(name, values, size, body) => {
                let values = self.eval(values, env)?;
                let size = self.eval(size, env)?.integer()?;
                let Value::RecordBatch(batch) = values else { bail!("size iteration requires a record batch") };
                for range in record_schema::chunks(batch.rows.len(), size).map_err(anyhow::Error::msg)? {
                    self.tick()?;
                    let value = batch.chunk(range, &mut self.remaining)?;
                    env.insert(name.clone(), Value::RecordBatch(value));
                    match self.block(body, env)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
            }
            Stmt::Set(name, expr) => {
                ensure!(
                    env.contains_key(name),
                    "assignment to undeclared variable: {name}"
                );
                let value = self.eval(expr, env)?;
                env.insert(name.clone(), value);
            }
            Stmt::Eval(expr) => {
                self.eval(expr, env)?;
            }
            Stmt::Assert(condition, message) => {
                if !self.eval(condition, env)?.boolean()? {
                    let message = match message { Some(e) => self.eval(e, env)?, None => Value::Text("assertion failed".into()) };
                    bail!("{}", message.text()?);
                }
            }
            Stmt::Reload(format, target, path, optimizer) => {
                let target = self.eval(target, env)?;
                let path = self.eval(path, env)?;
                let optimizer = optimizer.as_ref().map(|e| self.eval(e, env)).transpose()?;
                self.reload_file(format, target, path, optimizer)?;
            }
            Stmt::Save(format, value, path, overwrite) => {
                let value = self.eval(value, env)?;
                let path = self.eval(path, env)?;
                self.save_file(format, value, path, *overwrite)?;
            }
            Stmt::Return(expr) => return Ok(Flow::Return(self.eval(expr, env)?)),
            Stmt::Break => return Ok(Flow::Break),
            Stmt::Continue => return Ok(Flow::Continue),
            Stmt::If(condition, yes, no) => {
                let b = self.eval(condition, env)?.boolean()?;
                return self.block(if b { yes } else { no }, env);
            }
            Stmt::While(condition, body) => {
                while self.eval(condition, env)?.boolean()? {
                    self.tick()?;
                    match self.block(body, env)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
            }
            Stmt::For(name, values, body) => {
                let values = self.eval(values, env)?;
                for value in values.list()? {
                    self.tick()?;
                    env.insert(name.clone(), value.clone());
                    match self.block(body, env)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
            }
        }
        Ok(Flow::Next)
    }
    fn eval(&mut self, expr: &Expr, env: &mut Env) -> Result<Value> {
        self.tick()?;
        match expr {
            Expr::Mix(..) => bail!("mix is only available in tensor forward/loss functions"),
            Expr::Sample(p, a, b) => {
                let p = self.eval(p, env)?;
                if self.draw_probability(p)? {
                    self.eval(a, env)
                } else {
                    self.eval(b, env)
                }
            }
            Expr::SampleFrom(name, input) => {
                self.tick()?; // Model lookup has the same budget cost as the former argument.
                let model = env.get(name).cloned()
                    .with_context(|| format!("undefined variable: {name}"))?;
                let input = self.eval(input, env)?;
                self.sample_model(model, input)
            }
            Expr::Number(n) => number(*n),
            Expr::Text(s) => Ok(Value::Text(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Null => Ok(Value::Null),
            Expr::Name(name) => env
                .get(name)
                .cloned()
                .with_context(|| format!("undefined variable: {name}")),
            Expr::List(values) => Ok(Value::List(
                values
                    .iter()
                    .map(|v| self.eval(v, env))
                    .collect::<Result<_>>()?,
            )),
            Expr::BuiltinCall(name, args) if name == "__await" || name == "__spawn" => {
                let value = self.eval(&args[0], env)?;
                if name == "__await" { self.await_value(value) } else { self.spawn_future(value) }
            }
            Expr::StaticCall(name, args) => {
                let args = args.iter().map(|v| self.eval(v, env)).collect::<Result<_>>()?;
                let result = self.call(name, args)?;
                validate_strategy_result(&result)?;
                Ok(result)
            }
            Expr::BuiltinCall(name, args) => {
                let args = args.iter().map(|v| self.eval(v, env)).collect::<Result<_>>()?;
                self.call(name, args)
            }
            Expr::Call(name, args) => {
                let args = args
                    .iter()
                    .map(|v| self.eval(v, env))
                    .collect::<Result<_>>()?;
                self.call_in_env(name, args, env)
            }
            Expr::Index(object, index) => {
                let object = self.eval(object, env)?;
                let index = self.eval(index, env)?;
                compiled::index(object, index)
            }
            Expr::Slice(object, start, end) => {
                let object = self.eval(object, env)?;
                let start = start.as_ref().map(|e| self.eval(e, env)).transpose()?;
                let end = end.as_ref().map(|e| self.eval(e, env)).transpose()?;
                compiled::slice(object, start, end)
            }
            Expr::Unary(op, value) => {
                let value = self.eval(value, env)?;
                compiled::unary(op, value)
            }
            Expr::Binary(op, left, right) => {
                let left = self.eval(left, env)?;
                if op == "&&" && !left.boolean()? {
                    return Ok(Value::Bool(false));
                }
                if op == "||" && left.boolean()? {
                    return Ok(Value::Bool(true));
                }
                let right = self.eval(right, env)?;
                compiled::binary(op, left, right)
            }
        }
    }
    /// Categorical inference deliberately uses evaluate, which does not commit forward writes.
    /// Keep the original f32 subtraction, f64 exponentiation and single RNG draw unchanged.
    pub fn sample_model(&mut self, model: Value, input: Value) -> Result<Value> {
        let (action, _) = self.sample_model_result(model, input)?;
        Ok(action)
    }

    /// Both fields describe one evaluation and one random draw. Sampling never
    /// commits forward writes. The legacy f32 action path retains its arithmetic.
    pub fn sample_model_result(&mut self, model: Value, input: Value) -> Result<(Value, Value)> {
        self.tick()?;
        let (logits, kind) = if let Value::Adaptive(index) = model {
            let mut session = self.adaptive[index].lock()?;
            ensure!(!session.execution_active(), "execution active; resume or cancel first");
            let result = session.evaluate(&input.vector()?, self.remaining);
            self.remaining = self.remaining.saturating_sub(session.steps());
            (result?, session.program().output_type)
        } else {
            let mut session = self.models[model.model()?].lock()?;
            let kind = session.session.output_type();
            (session.predict(&input.vector()?)?, kind)
        };
        if kind != crate::categorical::VectorType::F32 {
            let values: Vec<f64> = logits.iter().map(|x| *x as f64).collect();
            let probabilities = crate::categorical::probabilities(kind, &values)?;
            let action = crate::categorical::sample_probabilities(&probabilities, random(&mut self.rng))?;
            let log_prob = if kind == crate::categorical::VectorType::Probabilities {
                values[action].ln() - values.iter().sum::<f64>().ln()
            } else {
                Self::sample_log_probability(&logits, action)
            };
            return Ok((number(action as f64)?, number(log_prob)?));
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities: Vec<f64> = logits.iter().map(|n| ((*n - max) as f64).exp()).collect();
        let mut draw = random(&mut self.rng) * probabilities.iter().sum::<f64>();
        let mut action = logits.len() - 1;
        for (i, p) in probabilities.iter().enumerate() {
            draw -= p;
            if draw <= 0. {
                action = i;
                break;
            }
        }
        Ok((number(action as f64)?, number(Self::sample_log_probability(&logits, action))?))
    }

    fn sample_log_probability(logits: &[f32], action: usize) -> f64 {
        let max = logits.iter().map(|x| *x as f64).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (*x as f64 - max).exp()).sum();
        logits[action] as f64 - max - sum.ln()
    }

    pub fn validate_learner(&self, learner: &str) -> Result<()> {
        if let Some(function) = self.program.get(learner) {
            ensure!(!function.is_async && function.args.len() == 2, "learner must be a synchronous user function with two parameters");
        } else if let Some(function) = self.compiled.get(learner) {
            ensure!(function.async_body.is_none() && function.args.len() == 2, "learner must be a synchronous user function with two parameters");
        } else { bail!("unknown learner function: {learner}"); }
        Ok(())
    }

    pub fn begin_collection(&self, target: &Value) -> Result<crate::collection::CollectionToken> {
        match target {
            Value::Agent(i) => self.begin_collection(&self.agents[*i].policy),
            Value::Adaptive(index) => self.adaptive[*index].lock()?.begin_collection(),
            Value::Model(index) => self.models[*index].lock()?.session.begin_collection(),
            _ => bail!("learn requires a model or agent"),
        }
    }

    pub fn record_chunk(&mut self, batch: &records::RecordBatch, range: std::ops::Range<usize>) -> Result<records::RecordBatch> {
        batch.chunk(range, &mut self.remaining)
    }

    pub fn yield_record(&mut self, env: &mut Env, fields: Vec<(String, Value)>) -> Result<()> {
        let Some(Value::RecordBatch(batch)) = env.get_mut("__collect") else {
            bail!("yield requires a lexical collect block")
        };
        batch.push(fields, &mut self.remaining)
    }

    pub fn record_field(&mut self, value: Value, field: &str) -> Result<Value> {
        match value {
            Value::Environment(i) => match field {"observation"=>Ok(self.environments[i].observation.clone()),_=>bail!("environment field must be observation")},
            Value::Agent(i) => match field {"policy"=>Ok(self.agents[i].policy.clone()),"critic"=>self.agents[i].critic.clone().context("agent has no critic"),_=>bail!("agent field must be policy or critic")},
            Value::RecordBatch(batch) => batch.project(field, &mut self.remaining),
            Value::Object(fields) => fields.get(field).cloned().with_context(|| format!("missing record field: {field}")),
            _ => bail!("field projection requires a record or record batch"),
        }
    }

    pub fn draw_probability(&mut self, probability: Value) -> Result<bool> {
        let p = probability.number()?;
        ensure!(
            p.is_finite() && (0. ..=1.).contains(&p),
            "sample probability must be in [0,1]"
        );
        Ok(if p == 0. {
            false
        } else if p == 1. {
            true
        } else {
            random(&mut self.rng) < p
        })
    }

    /// Resolve source calls in the current function's environment before named functions.
    /// Model handles are the only callable values; functions retain their existing scope.
    pub fn call_in_env(&mut self, name: &str, args: Vec<Value>, env: &Env) -> Result<Value> {
        if let Some((base,method))=name.split_once('.') {
            if let Some(instance)=env.get(base) { return self.instance_method(instance.clone(),method,args); }
        }
        let Some(value) = env.get(name) else {
            return self.call(name, args);
        };
        (|| {
            ensure!(
                matches!(value, Value::Model(_) | Value::Adaptive(_)),
                "local binding {name} is not callable; expected a model handle"
            );
            ensure!(args.len() == 1, "model call expects 1 input, got {}", args.len());
            self.tick()?;
            self.builtin("__predict", &[value.clone(), args[0].clone()])
        })()
        .with_context(|| format!("{name}()"))
    }

    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        if name == "compute_budget" {
            self.gpu_benchmark_restricted = true;
        }
        self.tick()?;
        if let Some(function) = self.compiled.get(name).cloned() {
            if function.async_body.is_some() {
                ensure!(args.len() == function.args.len(), "{name}: argument count mismatch");
                return self.cold_call(name, args);
            }
            ensure!(
                args.len() == function.args.len(),
                "{name}: argument count mismatch"
            );
            ensure!(self.depth < 64, "function recursion limit exceeded");
            self.depth += 1;
            let mut env = function
                .args
                .into_iter()
                .map(String::from)
                .zip(args)
                .collect();
            let result = (function.body)(self, &mut env);
            self.depth -= 1;
            return match result? {
                Flow::Return(v) => Ok(v),
                Flow::Next => Ok(Value::Null),
                _ => bail!("loop control escaped function"),
            };
        }
        if let Some(function) = self.program.get(name).cloned() {
            if function.is_async {
                ensure!(args.len() == function.args.len(), "{name}: argument count mismatch");
                return self.cold_call(name, args);
            }
            ensure!(
                args.len() == function.args.len(),
                "{name}: argument count mismatch"
            );
            ensure!(self.depth < 64, "function recursion limit exceeded");
            self.depth += 1;
            let mut env = function.args.into_iter().zip(args).collect();
            let result = self.block(&function.body, &mut env);
            self.depth -= 1;
            return match result? {
                Flow::Return(v) => Ok(v),
                Flow::Next => Ok(Value::Null),
                _ => bail!("loop control escaped function"),
            };
        }
        ensure!(name != "model", "model(...) loading was removed; use load model from path using optimizer");
        self.builtin(name, &args)
            .with_context(|| format!("{name}()"))
    }
    fn builtin(&mut self, name: &str, a: &[Value]) -> Result<Value> {
        if matches!(name,"__agent"|"__environment"|"__record") { return self.agent_builtin(name,a); }
        if name.starts_with("responses_") {
            let expected = match name { "responses_scope" => 3, "responses_restrict" | "responses_submit" | "responses_next" => 2, "responses_session" | "responses_cancel" => 1, "responses_usage" => 0, _ => usize::MAX };
            ensure!(a.len() == expected, "invalid responses call or argument count");
            return self.responses_builtin(name, a);
        }
        if self.async_state.context.is_some() {
            ensure!(["sqrt", "exp", "log", "min", "max", "len", "seed", "random", "__record_field"].contains(&name), "{name}: builtin unavailable in first native async stage; resource creation and synchronous I/O are forbidden");
        }
        if name == "__record_field" {
            return self.record_field(a[0].clone(), a[1].text()?);
        }
        if let Some(format) = name.strip_prefix("__load_") {
            return self.load_file(format, a);
        }
        if name == "__probabilities" {
            let [Value::CategoricalVector(kind, values)] = a else {
                bail!(".probabilities requires logits[N] or probabilities[N]; use softmax(x) for an ordinary f32 vector")
            };
            let work = values.len() as u64;
            ensure!(self.remaining >= work, "operation budget exhausted");
            self.remaining -= work;
            let input = values.iter().map(Value::number).collect::<Result<Vec<_>>>()?;
            let output = crate::categorical::probabilities(*kind, &input)?;
            return Ok(Value::CategoricalVector(crate::categorical::VectorType::Probabilities,
                output.into_iter().map(Value::Number).collect()));
        }
        let entry = prelude::builtin(name).with_context(|| format!("unknown function: {name}"))?;
        let arity = entry.arity.as_ref().expect("builtin arity in prelude");
        ensure!(
            (arity.min..=arity.max).contains(&a.len()),
            "{}: expected {}..={} arguments, got {}",
            entry.signature, arity.min, arity.max, a.len()
        );
        if matches!(name, "differentiate" | "differentiate_at" | "gradient_snapshot" | "commit_gradients" | "release_gradient") {
            return self.optimization_builtin(name, a);
        }
        if name == "benchmark" {
            return self.benchmark(a);
        }
        if matches!(
            name,
            "devices" | "training_device" | "training_status" | "wgsl_kernel" | "wgsl_run"
        ) {
            return self.training_device_builtin(name, a);
        }
        if matches!(name, "__train_step" | "reinforce") {
            if let Some(Value::Model(index)) = a.first() {
                if !self.models[*index].lock()?.backend().is_cpu() {
                    self.resources.allow_direct_gpu_training()?;
                }
            }
        }
        if matches!(
            name,
            "execution_start" | "execution_resume" | "execution_cancel"
        ) {
            let count = match name {
                "execution_start" => 4,
                "execution_resume" => 2,
                _ => 1,
            };
            ensure!(a.len() == count, "{name}: argument count mismatch");
            let Value::Adaptive(index) = a[0] else {
                bail!("execution requires a forward model")
            };
            let mut session = self.adaptive[index].lock()?;
            if name == "execution_cancel" {
                return Ok(Value::Bool(session.cancel_execution()));
            }
            let (available, memory) = self.resources.execution_allowance(self.remaining)?;
            if name == "execution_start" {
                let work = (a[2].integer()? as u64).min(available);
                let bytes = a[3].integer()?.min(memory);
                self.resources.execution_start()?;
                session.start_execution(&a[1].vector()?, work, bytes)?;
                return Ok(Value::Null);
            }
            ensure!(
                session.execution_memory_limit()? <= memory,
                "execution exceeds current scope memory limit"
            );
            let quantum = a[1].integer()? as u64;
            ensure!(
                (1..=1000000).contains(&quantum),
                "execution quantum must be 1..1000000"
            );
            let before = session.steps();
            let result = session.resume_execution(quantum, available);
            self.resources
                .charge_execution(session.steps() - before, &mut self.remaining)?;
            return Value::from_json(serde_json::to_value(result?)?);
        }
        if name == "device_submit" {
            ensure!(
                a.len() == 3,
                "device_submit expects model, input, device alias"
            );
            let model = self.models[a[0].model()?].lock()?.snapshot()?;
            return self.resources.device_submit(
                model,
                a[1].vector()?,
                a[2].text()?,
                &mut self.remaining,
            );
        }
        if name == "device_wait" {
            ensure!(
                a.len() == 3,
                "device_wait expects device alias, job, timeout_ms"
            );
            return self.resources.device_wait(
                a[0].text()?,
                a[1].integer()? as u64,
                a[2].integer()? as u64,
            );
        }
        if name == "device_status" {
            ensure!(a.len() == 1, "device_status expects device alias");
            return self.resources.device_status(a[0].text()?);
        }
        if name == "compute_register" {
            ensure!(
                a.len() == 3,
                "compute_register expects name, function, limits"
            );
            let function = a[1].text()?;
            let arity = self
                .program
                .get(function)
                .map(|f| f.args.len())
                .or_else(|| self.compiled.get(function).map(|f| f.args.len()));
            ensure!(
                arity == Some(1) && !prelude::is_builtin(function),
                "capability requires one-argument WGT function"
            );
            return self
                .resources
                .register(a[0].text()?, function, a[2].list()?);
        }
        if name == "compute_call" && a.len() == 2 {
            if let Some((function, limits)) = self.resources.custom.get(a[0].text()?).cloned() {
                a[1].list()?;
                let trace_len = self.resources.trace_len();
                let result = self.builtin(
                    "compute_scope",
                    &[
                        Value::Text(function),
                        Value::List(vec![a[1].clone()]),
                        Value::List(limits),
                    ],
                );
                self.resources
                    .label_scope(a[0].text()?, trace_len, a[1].list()?.len());
                return result;
            }
        }
        if name == "compute_scope" {
            ensure!(
                a.len() == 3,
                "compute_scope expects function, arguments, limits"
            );
            let function = a[0].text()?;
            ensure!(
                !prelude::is_builtin(function)
                    && (self.program.contains_key(function)
                        || self.compiled.contains_key(function)),
                "scope requires a WGT function"
            );
            let arguments = a[1].list()?.to_vec();
            let frame = match self
                .resources
                .enter_scope(a[2].list()?, &mut self.remaining)
            {
                Ok(frame) => frame,
                Err(error) => {
                    return Value::from_json(
                        serde_json::json!({"ok":false,"error":error.to_string(),"value":null,"work":0}),
                    )
                }
            };
            let result = self.call(function, arguments).and_then(|v| v.json());
            let work = self
                .resources
                .leave_scope(frame, &mut self.remaining, result.is_ok())?;
            return Value::from_json(match result {
                Ok(value) => {
                    serde_json::json!({"ok":true,"error":"","value":value,"work":work.number()?})
                }
                Err(error) => {
                    serde_json::json!({"ok":false,"error":format!("{error:#}"),"value":null,"work":work.number()?})
                }
            });
        }
        if name.starts_with("compute_") {
            return self.resources.call(name, a, &mut self.remaining);
        }
        ensure!(
            self.permissions.allows_legacy_io()
                || ![
                    "dependency",
                    "load_model",
                    "save_model",
                    "jsonl",
                    "hf_dataset",
                    "batch",
                    "save_checkpoint",
                    "load_checkpoint"
                ]
                .contains(&name),
            "host policy requires capability-based I/O"
        );
        match name {
            "entropy"
            | "normalize_probabilities"
            | "kl_divergence"
            | "js_divergence"
            | "cross_entropy_probs"
            | "temperature_softmax" => {
                let n = a[0].list()?.len() as u64;
                let work = a.iter().try_fold(n, |sum, v| -> Result<u64> {
                    Ok(sum.saturating_add(match v {
                        Value::List(xs) => xs.len() as u64,
                        _ => 0,
                    }))
                })?;
                ensure!(self.remaining >= work, "operation budget exhausted");
                self.remaining -= work;
                math::probability::call(name, a)
            }
            "one_hot" => one_hot::encode(&a[0], &a[1], &mut self.remaining),
            "zeros" | "ones" | "full" | "linspace" | "concat" | "vector_slice" | "gather"
            | "compress" | "reverse" => math::data::call(name, a, &mut self.remaining),
            "relu" | "leaky_relu" | "elu" | "hard_sigmoid" | "hard_swish" => {
                math::activation::call(name, a)
            }
            "bce_logits" | "huber" | "log_cosh" | "diff" | "ema" | "moving_mean" => {
                let n = a[0].list()?.len();
                let work = if name == "moving_mean" {
                    let window = a[1].integer()?;
                    ensure!(window > 0 && window <= n, "window must be in 1..=N");
                    ((n - window + 1) as u64).saturating_mul(window as u64)
                } else {
                    n as u64
                };
                ensure!(self.remaining >= work, "operation budget exhausted");
                self.remaining -= work;
                if matches!(name, "diff" | "ema" | "moving_mean") {
                    math::signal::call(name, a)
                } else {
                    math::loss::call(name, a)
                }
            }
            "min" | "max" | "argmin" | "argmax" | "variance" | "stddev" | "median" | "quantile"
            | "sort" | "argsort" | "cumsum" | "cosine_similarity" | "clip" => {
                let work = math::algorithms::work(name, a[0].list()?.len());
                ensure!(self.remaining >= work, "operation budget exhausted");
                self.remaining -= work;
                math::algorithms::call(name, a)
            }
            "dot" | "norm" | "mean" | "softmax" | "log_softmax" | "logsumexp" | "l2_normalize"
            | "mse" | "mae" | "cross_entropy" => {
                // Charge linear work before allocating or traversing input vectors.
                let work = a[0].list()?.len() as u64;
                ensure!(self.remaining >= work, "operation budget exhausted");
                self.remaining -= work;
                math::vector::call(name, a)
            }
            "rsqrt" | "fma" | "hypot" | "expm1" | "log1p" | "sigmoid" | "softplus" | "silu" => {
                math::call(name, a)
            }
            "dependency" => Ok(Value::Text(
                self.dependencies
                    .get(a[0].text()?)
                    .cloned()
                    .context("dependency is not declared or resolved; run with wgt")?,
            )),
            "len" => number(match &a[0] { Value::RecordBatch(batch) => batch.rows.len(), value => value.list()?.len() } as f64),
            "range" => {
                let (start, end) = (a[0].integer()?, a[1].integer()?);
                ensure!(
                    end >= start && end - start <= 1_000_000,
                    "invalid or oversized range"
                );
                Ok(Value::List(
                    (start..end).map(|n| Value::Number(n as f64)).collect(),
                ))
            }
            "append" => {
                let mut list = a[0].list()?.to_vec();
                ensure!(list.len() < 1_000_000, "list limit exceeded");
                list.push(a[1].clone());
                Ok(Value::List(list))
            }
            "column" => {
                let key = a[1].text()?;
                Ok(Value::List(
                    a[0].list()?
                        .iter()
                        .map(|v| {
                            let Value::Object(row) = v else {
                                bail!("column requires records");
                            };
                            row.get(key).cloned().context("missing column")
                        })
                        .collect::<Result<_>>()?,
                ))
            }
            "sum" => number(
                a[0].list()?
                    .iter()
                    .map(Value::number)
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .sum(),
            ),
            "sqrt" => number(a[0].number()?.sqrt()),
            "log" => number(a[0].number()?.ln()),
            "exp" => number(a[0].number()?.exp()),
            "metric" => {
                self.metrics.insert(a[0].text()?.into(), a[1].json()?);
                Ok(Value::Null)
            }
            "seed" => {
                self.rng = a[0].integer()? as u64;
                Ok(Value::Null)
            }
            "random" => number(random(&mut self.rng)),
            "mlp" => {
                ensure!(self.models.len() < 64, "model handle limit exceeded");
                let widths = a[0]
                    .list()?
                    .iter()
                    .map(Value::integer)
                    .collect::<Result<Vec<_>>>()?;
                let optimizer = match a[3].text()? {
                    "sgd" => Optimizer::Sgd,
                    "adam" => Optimizer::Adam,
                    _ => bail!("optimizer must be sgd or adam"),
                };
                let model = mlp(&widths, a[1].text()?, a[2].integer()? as u64)?;
                let trainer = Trainer::new(model, &self.root, 256 * 1024 * 1024, optimizer)?;
                let index = self.models.len();
                self.models.push(crate::weightfs::Mounted::auto(trainer)?);
                Ok(Value::Model(index))
            }
            "__graph" => {
                ensure!(self.models.len() < 64, "model handle limit exceeded");
                let graph: crate::source::Graph = serde_json::from_str(a[0].text()?)?;
                let optimizer = match a[1].text()? {
                    "sgd" => Optimizer::Sgd,
                    "adam" => Optimizer::Adam,
                    _ => bail!("optimizer must be sgd or adam"),
                };
                let trainer = Trainer::new(
                    graph.instantiate()?,
                    &self.root,
                    256 * 1024 * 1024,
                    optimizer,
                )?;
                let index = self.models.len();
                self.models.push(crate::weightfs::Mounted::auto(trainer)?);
                Ok(Value::Model(index))
            }
            "__adaptive" => {
                ensure!(self.adaptive.len() < 64, "model handle limit exceeded");
                let program = serde_json::from_str(a[0].text()?)?;
                let mut session =
                    crate::adaptive::AdaptiveSession::prepare(program, 64 * 1024 * 1024)?;
                match a[1].text()? {
                    "none" => {}
                    "sgd" => session.set_optimizer(Optimizer::Sgd)?,
                    "adam" => session.set_optimizer(Optimizer::Adam)?,
                    _ => bail!("forward optimizer must be none, sgd or adam"),
                }
                let index = self.adaptive.len();
                self.adaptive.push(crate::weightfs::Mounted::auto(session)?);
                Ok(Value::Adaptive(index))
            }
            "__predict" => {
                let input = a[1].vector()?;
                if let Value::Adaptive(index) = a[0] {
                    let mut session = self.adaptive[index].lock()?;
                    ensure!(
                        !session.execution_active(),
                        "execution active; resume or cancel first"
                    );
                    ensure!(input.len() == session.input_len(),
                        "input shape mismatch: expected f32[{}], got f32[{}]",
                        session.input_len(), input.len());
                    let result = session
                        .run(&input, self.remaining)
                        .map(<[f32]>::to_vec);
                    self.remaining = self.remaining.saturating_sub(session.steps());
                    return typed_vector(result?, session.program().output_type);
                }
                let model = a[0].model()?;
                let mut session = self.models[model].lock()?;
                let expected = session.session.report().input;
                ensure!(input.len() == expected,
                    "input shape mismatch: expected f32[{expected}], got f32[{}]", input.len());
                let kind = session.session.output_type();
                typed_vector(session.predict(&input)?, kind)
            }
            "__train_step" => {
                let options = a[3].json()?;
                let options = options.as_object().context("expected training options")?;
                let rate = options.get("rate").and_then(serde_json::Value::as_f64).context("training requires rate")?;
                let loss = options.get("loss").and_then(serde_json::Value::as_str).unwrap_or("mse");
                let clip = options.get("clip").map(|v| v.as_f64().context("clip must be numeric")).transpose()?;
                ensure!(rate.is_finite() && rate > 0. && (rate as f32).is_finite(), "learning rate must be positive and finite");
                ensure!(clip.is_none_or(|v| v.is_finite() && v > 0. && (v as f32).is_finite()), "gradient clip norm must be positive and finite");
                let lowered = vec![a[0].clone(), a[1].clone(), a[2].clone(), number(rate)?, Value::Text(loss.into())];
                if let Some(limit) = clip {
                    if let Value::Adaptive(index) = a[0] {
                        if loss == "loss" { return number(self.adaptive[index].lock()?.train_clipped(&a[1].matrix()?, &a[2].matrix()?, rate as f32, limit as f32, &mut self.remaining, 64*1024*1024)? as f64); }
                    }
                }
                let a = lowered.as_slice();
                let clip = clip.map(|v| v as f32);
                if let Value::Adaptive(index) = a[0] {
                    let x = a[1].matrix()?;
                    let rate = a[3].float()?;
                    let loss = match a[4].text()? {
                        "loss" => self.adaptive[index].lock()?.train_custom(
                            &x,
                            &a[2].matrix()?,
                            rate,
                            &mut self.remaining,
                            64 * 1024 * 1024,
                        )?,
                        "mse" => self.adaptive[index].lock()?.train_builtin_clipped(
                            &x,
                            Objective::Mse(&a[2].matrix()?),
                            rate,
                            clip,
                            &mut self.remaining,
                            64 * 1024 * 1024,
                        )?,
                        "cross_entropy" => {
                            let actions = a[2]
                                .list()?
                                .iter()
                                .map(Value::integer)
                                .collect::<Result<Vec<_>>>()?;
                            self.adaptive[index].lock()?.train_builtin_clipped(
                                &x,
                                Objective::Policy {
                                    actions: &actions,
                                    advantages: &vec![1.; x.len()],
                                },
                                rate,
                                clip,
                                &mut self.remaining,
                                64 * 1024 * 1024,
                            )?
                        }
                        _ => bail!("forward loss must be mse, cross_entropy or loss"),
                    };
                    return number(loss as f64);
                }
                let model = a[0].model()?;
                let x = a[1].matrix()?;
                let rate = a[3].float()?;
                let loss = match a[4].text()? {
                    "mse" => self.models[model].lock()?.step_clipped(
                        &x,
                        Objective::Mse(&a[2].matrix()?),
                        rate,
                        clip,
                    )?,
                    "cross_entropy" => {
                        let actions = a[2]
                            .list()?
                            .iter()
                            .map(Value::integer)
                            .collect::<Result<Vec<_>>>()?;
                        self.models[model].lock()?.step_clipped(
                            &x,
                            Objective::Policy {
                                actions: &actions,
                                advantages: &vec![1.; x.len()],
                            },
                            rate,
                            clip,
                        )?
                    }
                    _ => bail!("loss must be mse or cross_entropy"),
                };
                number(loss as f64)
            }
            "reinforce" => {
                let x = a[1].matrix()?;
                let actions = a[2]
                    .list()?
                    .iter()
                    .map(Value::integer)
                    .collect::<Result<Vec<_>>>()?;
                let advantages = a[3].vector()?;
                if let Value::Adaptive(index) = a[0] {
                    return number(self.adaptive[index].lock()?.train_step(
                        &x,
                        Objective::Policy {
                            actions: &actions,
                            advantages: &advantages,
                        },
                        a[4].float()?,
                        &mut self.remaining,
                        64 * 1024 * 1024,
                    )? as f64);
                }
                let model = a[0].model()?;
                number(self.models[model].lock()?.step(
                    &x,
                    Objective::Policy {
                        actions: &actions,
                        advantages: &advantages,
                    },
                    a[4].float()?,
                )? as f64)
            }
            "gae" => {
                let (rewards, values, next) = (a[0].vector()?, a[1].vector()?, a[2].vector()?);
                let terminal = a[3]
                    .list()?
                    .iter()
                    .map(Value::boolean)
                    .collect::<Result<Vec<_>>>()?;
                let truncated = a[4]
                    .list()?
                    .iter()
                    .map(Value::boolean)
                    .collect::<Result<Vec<_>>>()?;
                let (gamma, lambda) = (a[5].float()?, a[6].float()?);
                ensure!(
                    (0. ..=1.).contains(&gamma) && (0. ..=1.).contains(&lambda),
                    "gamma/lambda must be in [0,1]"
                );
                let n = rewards.len();
                ensure!(
                    [values.len(), next.len(), terminal.len(), truncated.len()]
                        .iter()
                        .all(|len| *len == n),
                    "GAE lengths must match"
                );
                let mut advantages = vec![0.; n];
                let mut tail = 0.;
                for i in (0..n).rev() {
                    let delta =
                        rewards[i] + if terminal[i] { 0. } else { gamma * next[i] } - values[i];
                    tail = delta
                        + if terminal[i] || truncated[i] {
                            0.
                        } else {
                            gamma * lambda * tail
                        };
                    ensure!(tail.is_finite(), "nonfinite GAE");
                    advantages[i] = tail;
                }
                Ok(vector(advantages))
            }
            "jsonl" | "hf_dataset" => {
                ensure!(self.readers.len() < 1024, "reader handle limit exceeded");
                let reader = if name == "jsonl" {
                    dataset::Reader::jsonl(&self.root.join(a[0].text()?))?
                } else {
                    dataset::Reader::huggingface(a[0].text()?, a[1].text()?, a[2].text()?)?
                };
                let index = self.readers.len();
                self.readers.push(Some(reader));
                Ok(Value::Reader(index))
            }
            "batch" => {
                let n = a[1].integer()?;
                ensure!(n > 0 && n <= 65536, "batch size must be in 1..65536");
                let reader = self.readers[a[0].reader()?]
                    .as_mut()
                    .context("reader is closed")?;
                Ok(Value::List(reader.batch(n)?))
            }
            "close" => {
                self.readers[a[0].reader()?].take();
                Ok(Value::Null)
            }
            _ => bail!("unknown builtin"),
        }
    }
}
