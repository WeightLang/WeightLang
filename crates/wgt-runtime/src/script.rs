//! Explicit WGT host scripts, interpreted entirely in Rust. No Python bridge.
pub mod codegen;
pub mod compiled;
mod dataset;
mod one_hot;
pub(crate) mod parse;
use crate::training::{mlp, random, Objective, Optimizer, Trainer};
use anyhow::{bail, ensure, Context, Result};
pub fn parse(source: &str) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let mut program = parse::parse_functions(&unit.functions)?;
    let model_name = unit.declarations.lines().find_map(|l| l.split('#').next().unwrap_or("").trim().strip_prefix("model ")).map(str::trim);
    if let Some(name) = model_name {
        ensure!(!name.starts_with("__") && !name.is_empty() && name.bytes().enumerate().all(|(i,c)| c.is_ascii_alphabetic() || c==b'_' || (i>0 && c.is_ascii_digit())), "invalid model name");
        ensure!(!program.contains_key(name) && !BUILTINS.contains(&name), "model name conflicts with function: {name}");
        let (builtin, plan) = if program.remove("forward").is_some() {
            program.remove("loss");
            ("__adaptive", serde_json::to_string(&crate::adaptive::parse(source)?)?)
        } else {
            ("__graph", serde_json::to_string(&crate::source::graph(&unit.declarations)?.1)?)
        };
        program.insert(name.into(), parse::Function { args:vec!["optimizer".into()], body:vec![Statement { line:1, kind:Stmt::Return(Expr::Call(builtin.into(),vec![Expr::Text(plan),Expr::Name("optimizer".into())])) }] });
    } else {
        ensure!(unit.declarations.lines().all(|l| l.trim().is_empty() || l.trim().starts_with('#') || l.trim()=="wgt 0.5"), "model declarations require model NAME");
        ensure!(!program.contains_key("forward"), "forward requires model declarations");
    }
    if let Some(main) = program.get("main") { ensure!(main.args.is_empty(), "main takes no arguments"); }
    else { ensure!(model_name.is_some(), "program requires fn main() or model declarations"); }
    Ok(program)
}
use parse::{Expr, Program, Statement, Stmt};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

const BUILTINS: &[&str] = &[
    "one_hot",
    "__graph",
    "__adaptive",
    "dependency",
    "len",
    "range",
    "append",
    "column",
    "sum",
    "sqrt",
    "log",
    "exp",
    "assert",
    "metric",
    "seed",
    "random",
    "mlp",
    "model",
    "save_model",
    "predict",
    "fit",
    "reinforce",
    "sample",
    "gae",
    "jsonl",
    "hf_dataset",
    "batch",
    "close",
    "save",
    "restore",
    "benchmark",
];
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
    Bool(bool),
    Null,
    List(Vec<Value>),
    Object(BTreeMap<String, Value>),
    Model(usize),
    Adaptive(usize),
    Reader(usize),
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
        if let Self::List(v) = self {
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
            Self::List(v) => {
                serde_json::Value::Array(v.iter().map(Self::json).collect::<Result<_>>()?)
            }
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
    let program = parse(source)?;
    ensure!(program.contains_key("main"), "execution requires fn main(); model construction is explicit");
    let mut runtime = Runtime {
        program,
        compiled: HashMap::new(),
        root: root.into(),
        dependencies,
        remaining: max_operations,
        depth: 0,
        models: Vec::new(),
        adaptive: Vec::new(),
        readers: Vec::new(),
        rng: 0,
        metrics: BTreeMap::new(),
    };
    let result = runtime.call("main", Vec::new())?;
    Ok((runtime, result))
}
pub struct Runtime {
    dependencies: BTreeMap<String, String>,
    program: Program,
    compiled: HashMap<String, compiled::Function>,
    root: PathBuf,
    remaining: u64,
    depth: usize,
    models: Vec<crate::weightfs::Mounted<Trainer>>,
    adaptive: Vec<crate::weightfs::Mounted<crate::adaptive::AdaptiveSession>>,
    readers: Vec<Option<dataset::Reader>>,
    rng: u64,
    metrics: BTreeMap<String, serde_json::Value>,
}
impl Runtime {
    pub fn tick(&mut self) -> Result<()> {
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
            Stmt::Let(name, expr) => {
                let value = self.eval(expr, env)?;
                env.insert(name.clone(), value);
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
            Expr::Sample(p,a,b) => {
                let p=self.eval(p,env)?;
                if self.draw_probability(p)? { self.eval(a,env) } else { self.eval(b,env) }
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
            Expr::Call(name, args) => {
                let args = args
                    .iter()
                    .map(|v| self.eval(v, env))
                    .collect::<Result<_>>()?;
                self.call(name, args)
            }
            Expr::Index(object, index) => {
                let object = self.eval(object, env)?;
                let index = self.eval(index, env)?;
                compiled::index(object, index)
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
    pub fn draw_probability(&mut self, probability: Value) -> Result<bool> {
        let p=probability.number()?;
        ensure!(p.is_finite() && (0. ..=1.).contains(&p),"sample probability must be in [0,1]");
        Ok(if p==0. {false} else if p==1. {true} else {random(&mut self.rng)<p})
    }

    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        self.tick()?;
        if let Some(function) = self.compiled.get(name).cloned() {
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
        self.builtin(name, &args)
            .with_context(|| format!("{name}()"))
    }
    fn builtin(&mut self, name: &str, a: &[Value]) -> Result<Value> {
        let count = match name {
            "random" => 0,
            "dependency" | "len" | "sum" | "sqrt" | "log" | "exp" | "seed" | "jsonl" | "close"
            | "restore" => 1,
            "__graph" | "__adaptive" | "model" | "save_model" | "range" | "append" | "column" | "assert" | "metric"
            | "predict" | "sample" | "batch" | "save" | "one_hot" => 2,
            "hf_dataset" => 3,
            "mlp" => 4,
            "fit" | "reinforce" | "benchmark" => 5,
            "gae" => 7,
            _ => bail!("unknown function: {name}"),
        };
        ensure!(
            a.len() == count,
            "expected {count} arguments, got {}",
            a.len()
        );
        match name {
            "one_hot" => one_hot::encode(&a[0], &a[1], &mut self.remaining),
            "dependency" => Ok(Value::Text(
                self.dependencies
                    .get(a[0].text()?)
                    .cloned()
                    .context("dependency is not declared or resolved; run with wgt")?,
            )),
            "len" => number(a[0].list()?.len() as f64),
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
            "assert" => {
                ensure!(a[0].boolean()?, "{}", a[1].text()?);
                Ok(Value::Null)
            }
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
                let optimizer = match a[1].text()? { "sgd" => Optimizer::Sgd, "adam" => Optimizer::Adam, _ => bail!("optimizer must be sgd or adam") };
                let trainer = Trainer::new(graph.instantiate()?, &self.root, 256*1024*1024, optimizer)?;
                let index = self.models.len(); self.models.push(crate::weightfs::Mounted::auto(trainer)?); Ok(Value::Model(index))
            }
            "__adaptive" => {
                ensure!(self.adaptive.len() < 64, "model handle limit exceeded");
                let program = serde_json::from_str(a[0].text()?)?;
                let mut session = crate::adaptive::AdaptiveSession::prepare(program, 64*1024*1024)?;
                match a[1].text()? {
                    "none" => {},
                    "sgd" => session.set_optimizer(Optimizer::Sgd),
                    "adam" => session.set_optimizer(Optimizer::Adam),
                    _ => bail!("forward optimizer must be none, sgd or adam"),
                }
                let index = self.adaptive.len(); self.adaptive.push(crate::weightfs::Mounted::auto(session)?); Ok(Value::Adaptive(index))
            }
            "model" => {
                ensure!(self.models.len() < 64, "model handle limit exceeded");
                let path = self.root.join(a[0].text()?);
                let value: serde_json::Value = serde_json::from_reader(File::open(&path)?)?;
                if value.get("instructions").is_some() {
                    ensure!(self.adaptive.len() < 64, "model handle limit exceeded");
                    let mut session = crate::adaptive::AdaptiveSession::prepare(serde_json::from_value(value)?,64*1024*1024)?;
                    match a[1].text()? {
                        "none" => {},
                        "sgd" => session.set_optimizer(Optimizer::Sgd),
                        "adam" => session.set_optimizer(Optimizer::Adam),
                        _ => bail!("forward optimizer must be none, sgd or adam"),
                    }
                    let index=self.adaptive.len(); self.adaptive.push(crate::weightfs::Mounted::auto(session)?);
                    return Ok(Value::Adaptive(index));
                }
                let model = serde_json::from_value(value)?;
                let optimizer = match a[1].text()? {
                    "sgd" => Optimizer::Sgd,
                    "adam" => Optimizer::Adam,
                    _ => bail!("optimizer must be sgd or adam"),
                };
                let trainer = Trainer::new(
                    model,
                    path.parent().unwrap_or(Path::new(".")),
                    256 * 1024 * 1024,
                    optimizer,
                )?;
                let index = self.models.len();
                self.models.push(crate::weightfs::Mounted::auto(trainer)?);
                Ok(Value::Model(index))
            }
            "save_model" => {
                if let Value::Adaptive(index) = a[0] {
                    crate::training::write_new(&self.root.join(a[1].text()?), &serde_json::to_vec(self.adaptive[index].lock()?.program())?)?;
                    return Ok(Value::Null);
                }
                self.models[a[0].model()?].lock()?.export(&self.root.join(a[1].text()?))?;
                Ok(Value::Null)
            }
            "predict" => {
                if let Value::Adaptive(index) = a[0] {
                    let mut session=self.adaptive[index].lock()?;
                    let result=session.run(&a[1].vector()?,self.remaining).map(<[f32]>::to_vec);
                    self.remaining=self.remaining.saturating_sub(session.steps());
                    return Ok(vector(result?));
                }
                let model = a[0].model()?;
                Ok(vector(self.models[model].lock()?.predict(&a[1].vector()?)?))
            }
            "fit" => {
                if let Value::Adaptive(index) = a[0] {
                    let x=a[1].matrix()?;
                    let rate=a[3].float()?;
                    let loss=match a[4].text()? {
                        "loss" => self.adaptive[index].lock()?.train_custom(&x,&a[2].matrix()?,rate,&mut self.remaining,64*1024*1024)?,
                        "mse" => self.adaptive[index].lock()?.train_step(&x,Objective::Mse(&a[2].matrix()?),rate,&mut self.remaining,64*1024*1024)?,
                        "cross_entropy" => {
                            let actions=a[2].list()?.iter().map(Value::integer).collect::<Result<Vec<_>>>()?;
                            self.adaptive[index].lock()?.train_step(&x,Objective::Policy{actions:&actions,advantages:&vec![1.;x.len()]},rate,&mut self.remaining,64*1024*1024)?
                        }
                        _ => bail!("forward loss must be mse, cross_entropy or loss"),
                    };
                    return number(loss as f64);
                }
                let model = a[0].model()?;
                let x = a[1].matrix()?;
                let rate = a[3].float()?;
                let loss = match a[4].text()? {
                    "mse" => self.models[model].lock()?.step(&x, Objective::Mse(&a[2].matrix()?), rate)?,
                    "cross_entropy" => {
                        let actions = a[2]
                            .list()?
                            .iter()
                            .map(Value::integer)
                            .collect::<Result<Vec<_>>>()?;
                        self.models[model].lock()?.step(
                            &x,
                            Objective::Policy {
                                actions: &actions,
                                advantages: &vec![1.; x.len()],
                            },
                            rate,
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
                if let Value::Adaptive(index)=a[0] {
                    return number(self.adaptive[index].lock()?.train_step(&x,Objective::Policy{actions:&actions,advantages:&advantages},a[4].float()?,&mut self.remaining,64*1024*1024)? as f64);
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
            "sample" => {
                let logits = if let Value::Adaptive(index)=a[0] {
                    let mut session=self.adaptive[index].lock()?;
                    let result=session.evaluate(&a[1].vector()?,self.remaining);
                    self.remaining=self.remaining.saturating_sub(session.steps());
                    result?
                } else { self.models[a[0].model()?].lock()?.predict(&a[1].vector()?)? };
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probabilities: Vec<f64> =
                    logits.iter().map(|n| ((*n - max) as f64).exp()).collect();
                let mut draw = random(&mut self.rng) * probabilities.iter().sum::<f64>();
                let mut action = logits.len() - 1;
                for (i, p) in probabilities.iter().enumerate() {
                    draw -= p;
                    if draw <= 0. {
                        action = i;
                        break;
                    }
                }
                number(action as f64)
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
            "save" => {
                let path = self.root.join(a[1].text()?);
                if let Value::Adaptive(index)=a[0] {
                    self.adaptive[index].lock()?.save_checkpoint(&path)?;
                    return Ok(Value::Null);
                }
                self.models[a[0].model()?].lock()?.save(&path)?;
                Ok(Value::Null)
            }
            "restore" => {
                let path=self.root.join(a[0].text()?);
                let value:serde_json::Value=serde_json::from_reader(File::open(&path)?)?;
                if value.get("format").and_then(|v| v.as_str())==Some("weightlang.forward-training.v1") {
                    ensure!(self.adaptive.len()<64,"model handle limit exceeded");
                    let session=crate::adaptive::AdaptiveSession::restore_checkpoint(value,64*1024*1024)?;
                    let index=self.adaptive.len(); self.adaptive.push(crate::weightfs::Mounted::auto(session)?);
                    return Ok(Value::Adaptive(index));
                }
                ensure!(self.models.len() < 64, "model handle limit exceeded");
                let trainer = Trainer::restore(&self.root.join(a[0].text()?), 256 * 1024 * 1024)?;
                let index = self.models.len();
                self.models.push(crate::weightfs::Mounted::auto(trainer)?);
                Ok(Value::Model(index))
            }
            "benchmark" => {
                let model = a[0].model()?;
                let x = a[1].vector()?;
                let (warmup, repeats) = (a[2].integer()?, a[3].integer()?);
                let label = a[4].text()?;
                ensure!(
                    warmup <= 10000 && repeats > 0 && repeats <= 100000,
                    "invalid benchmark iterations"
                );
                self.models[model].lock()?.predict(&x)?; // validate outside timing
                for _ in 0..warmup {
                    self.tick()?;
                    std::hint::black_box(
                        self.models[model].lock()?
                            .session
                            .run(&x)
                            .map_err(anyhow::Error::msg)?,
                    );
                }
                let mut times = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    self.tick()?;
                    let start = Instant::now();
                    std::hint::black_box(
                        self.models[model].lock()?
                            .session
                            .run(&x)
                            .map_err(anyhow::Error::msg)?,
                    );
                    times.push(start.elapsed().as_secs_f64() * 1e6);
                }
                times.sort_by(f64::total_cmp);
                let report = serde_json::json!({"device":"cpu","unit":"us","warmup":warmup,"repeats":repeats,
                    "mean":times.iter().sum::<f64>()/repeats as f64,"p50":times[(repeats-1)/2],"p95":times[(repeats-1)*95/100],
                    "scope":"single-vector forward; excludes script dispatch, data loading and training"});
                self.metrics.insert(label.into(), report.clone());
                Value::from_json(report)
            }
            _ => bail!("unknown builtin"),
        }
    }
}
