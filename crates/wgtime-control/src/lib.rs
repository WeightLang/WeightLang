//! Client control only. Model state and execution live behind wgtime IPC.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::path::Path;
use wgtime::{Client, Request};

pub mod async_runtime;
pub mod harness;
pub mod openresponses;
pub mod persistence;
pub mod record_schema;
pub mod records;
pub mod tasks;

#[derive(Clone, Debug)]
pub struct ModelLease {
    handle: u64,
}
#[derive(Clone, Debug)]
pub enum Value {
    Data(Json),
    RecordBatch(records::RecordBatch),
    Model(ModelLease),
    Harness(harness::HarnessHandle),
    Future(usize),
    Task(usize),
}
impl Value {
    pub fn data(&self) -> Result<&Json> {
        match self {
            Self::Data(value) => Ok(value),
            _ => bail!("resource handle cannot be converted to JSON"),
        }
    }
    pub fn number(&self) -> Result<f64> {
        self.data()?.as_f64().context("expected number")
    }
    pub fn boolean(&self) -> Result<bool> {
        self.data()?.as_bool().context("expected bool")
    }
    pub fn text(&self) -> Result<&str> {
        self.data()?.as_str().context("expected string")
    }
    pub fn integer(&self) -> Result<usize> {
        let n = self.number()?;
        ensure!(
            n >= 0. && n <= 1e9 && n.fract() == 0.,
            "expected nonnegative integer"
        );
        Ok(n as usize)
    }
}
pub type Env = HashMap<String, Value>;
pub struct Runtime {
    client: Client,
    remaining: u64,
    pub metrics: serde_json::Map<String, Json>,
    functions: HashMap<&'static str, Function>,
    harness: harness::Harness,
    pub(crate) async_state: async_runtime::AsyncState,
    strategies: HashMap<(String, String), u64>,
    root: std::path::PathBuf,
}
#[derive(Clone)]
pub struct Function {
    pub args: &'static [&'static str],
    pub body: fn(&mut Runtime, &mut Env) -> Result<Value>,
    pub async_body:
        Option<fn(std::rc::Rc<std::cell::RefCell<Runtime>>, Env) -> async_runtime::FutureValue>,
}
impl Runtime {
    pub fn tick(&mut self) -> Result<()> {
        if let Some(context) = &self.async_state.context {
            context.charge(1)?;
            return Ok(());
        }
        if let Some(scope) = self.async_state.scopes.last() {
            scope.charge(1)?;
            return Ok(());
        }
        ensure!(self.remaining > 0, "operation budget exceeded");
        self.remaining -= 1;
        Ok(())
    }
    pub(crate) fn async_child(&self) -> Result<Self> {
        let mut state = async_runtime::AsyncState::default();
        state.pool = self.async_state.pool.clone();
        Ok(Self {
            client: self.client.reconnect()?,
            remaining: 0,
            metrics: Default::default(),
            functions: self.functions.clone(),
            harness: self.harness.child(),
            async_state: state,
            strategies: Default::default(),
            root: self.root.clone(),
        })
    }
    pub fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        self.tick()?;
        if let Some(function) = self.functions.get(name).cloned() {
            if function.async_body.is_some() {
                return self.cold_call(name, args);
            }
            ensure!(
                function.args.len() == args.len(),
                "wrong argument count for {name}"
            );
            let mut env = function
                .args
                .iter()
                .zip(args)
                .map(|(n, v)| ((*n).into(), v))
                .collect();
            return (function.body)(self, &mut env);
        }
        ensure!(name != "model", "model(...) loading was removed; use load model from path using optimizer");
        if name == "responses_next" {
            ensure!(
                self.async_state.context.is_some(),
                "responses_next requires async context"
            );
            return self.cold_internal("__async_responses", args);
        }
        if let Some(value) = self
            .harness
            .call(name, &args, Path::new("."), &mut self.remaining)?
        {
            return Ok(value);
        }
        match name {
            "__record" => {
                ensure!(args.len() % 2 == 0, "invalid record fields");
                let mut record = serde_json::Map::new();
                for fields in args.chunks(2) {
                    ensure!(
                        record
                            .insert(fields[0].text()?.into(), fields[1].data()?.clone())
                            .is_none(),
                        "duplicate record field"
                    );
                }
                Ok(Value::Data(Json::Object(record)))
            }
            "__record_field" => {
                ensure!(args.len() == 2, "invalid field access");
                records::field(&args[0],args[1].text()?)
            }
            "__strategy" => {
                ensure!(args.len() == 5, "strategy invocation contract");
                let Value::Model(model) = &args[2] else {
                    bail!("expected model lease")
                };
                let key = (args[0].text()?.to_owned(), args[1].text()?.to_owned());
                let strategy = if let Some(strategy) = self.strategies.get(&key) {
                    *strategy
                } else {
                    let registered = self.client.call(Request::RegisterStrategy {
                        source: key.0.clone(),
                        entry: key.1.clone(),
                    })?;
                    let strategy = registered["strategy"]
                        .as_u64()
                        .context("missing strategy")?;
                    self.strategies.insert(key, strategy);
                    strategy
                };
                let reply = self.client.call(Request::ExecuteStrategy {
                    handle: model.handle,
                    strategy,
                    inputs: serde_json::from_value(args[3].data()?.clone())?,
                    targets: serde_json::from_value(args[4].data()?.clone())?,
                    operations: 1_000_000,
                    memory: 256 * 1024 * 1024,
                })?;
                Ok(Value::Data(reply.get("result").context("strategy response missing result")?.clone()))
            }
            "__graph" => {
                ensure!(args.len() == 2, "graph constructor requires optimizer");
                let reply = self.client.call(Request::CreateGraph {
                    graph: serde_json::from_str(args[0].text()?)?,
                    optimizer: args[1].text()?.into(),
                    memory: 256 * 1024 * 1024,
                })?;
                Ok(Value::Model(ModelLease {
                    handle: reply["handle"].as_u64().context("missing handle")?,
                }))
            }
            "__adaptive" => {
                ensure!(
                    args.len() == 2,
                    "adaptive constructor requires optimizer"
                );
                let reply = self.client.call(Request::CreateTrainingProgram {
                    program: serde_json::from_str(args[0].text()?)?,
                    memory: 256 * 1024 * 1024,
                    optimizer: args[1].text()?.into(),
                })?;
                let handle = reply["handle"].as_u64().context("missing handle")?;
                Ok(Value::Model(ModelLease { handle }))
            }
            "mlp" => {
                ensure!(args.len() == 4, "mlp requires four arguments");
                let reply = self.client.call(Request::Create {
                    widths: args[0]
                        .data()?
                        .as_array()
                        .context("widths must be a list")?
                        .iter()
                        .map(|n| Value::Data(n.clone()).integer())
                        .collect::<Result<Vec<_>>>()?,
                    activation: args[1].text()?.into(),
                    seed: args[2].integer()? as u64,
                    optimizer: args[3].text()?.into(),
                    memory: 256 * 1024 * 1024,
                })?;
                Ok(Value::Model(ModelLease {
                    handle: reply["handle"].as_u64().context("missing handle")?,
                }))
            }
            "__train_step" => {
                ensure!(args.len() == 4, "training requires four lowered arguments");
                let options = args[3].data()?.as_object().context("expected training options")?;
                let rate = options.get("rate").and_then(Json::as_f64).context("training requires rate")? as f32;
                let loss = options.get("loss").and_then(Json::as_str).unwrap_or("mse").to_owned();
                let clip = options.get("clip").map(|v| v.as_f64().context("clip must be numeric").map(|v| v as f32)).transpose()?;
                let Value::Model(model) = &args[0] else {
                    bail!("expected model lease")
                };
                let reply = self.client.call(Request::TrainStep {
                    handle: model.handle,
                    inputs: serde_json::from_value(args[1].data()?.clone())?,
                    targets: args[2].data()?.clone(),
                    rate, loss, clip,
                })?;
                Ok(Value::Data(reply["loss"].clone()))
            }
            "metric" => {
                ensure!(args.len() == 2, "metric requires two args");
                self.metrics
                    .insert(args[0].text()?.into(), args[1].data()?.clone());
                Ok(Value::Data(Json::Null))
            }
            "len" => {
                ensure!(args.len() == 1, "len requires one arg");
                let len=match &args[0]{Value::RecordBatch(batch)=>batch.rows_len(),value=>value.data()?.as_array().context("expected list")?.len()};
                Ok(Value::Data(json!(len)))
            }
            "range" => {
                ensure!(args.len() == 2, "range requires two args");
                let start = args[0].integer()?;
                let end = args[1].integer()?;
                ensure!(end >= start && end - start <= 1_000_000, "range limit");
                Ok(Value::Data(json!((start..end).collect::<Vec<_>>())))
            }
            _ => bail!("unsupported client builtin: {name}"),
        }
    }
    pub fn call_in_env(&mut self, name: &str, args: Vec<Value>, env: &Env) -> Result<Value> {
        if let Some(Value::Model(model)) = env.get(name) {
            ensure!(args.len() == 1, "model requires one input");
            self.tick()?;
            let reply = self.client.call(Request::Predict {
                handle: model.handle,
                input: serde_json::from_value(args[0].data()?.clone())?,
            })?;
            return Ok(Value::Data(reply["output"].clone()));
        }
        self.call(name, args)
    }
}
pub fn number(n: f64) -> Result<Value> {
    ensure!(n.is_finite(), "nonfinite arithmetic");
    Ok(Value::Data(json!(n)))
}
fn equal_numbers(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    if a.is_f64() && b.is_f64() {
        return a.as_f64() == b.as_f64();
    }
    let integer = |n: &serde_json::Number| -> Option<i128> {
        n.as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
    };
    if !a.is_f64() && !b.is_f64() {
        return integer(a) == integer(b);
    }
    let (whole, float) = if a.is_f64() {
        (integer(b), a.as_f64())
    } else {
        (integer(a), b.as_f64())
    };
    let Some(whole) = whole else { return false };
    let Some(float) = float else { return false };
    float.is_finite()
        && float.fract() == 0.
        && float >= i64::MIN as f64
        && float < 18446744073709551616.0
        && float as i128 == whole
}
fn equal_json(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Number(a), Json::Number(b)) => equal_numbers(a, b),
        (Json::Array(a), Json::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal_json(a, b))
        }
        (Json::Object(a), Json::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| equal_json(a, b)))
        }
        _ => a == b,
    }
}
pub fn binary(op: &str, a: Value, b: Value) -> Result<Value> {
    Ok(match op {
        "+" => number(a.number()? + b.number()?)?,
        "-" => number(a.number()? - b.number()?)?,
        "*" => number(a.number()? * b.number()?)?,
        "/" => number(a.number()? / b.number()?)?,
        "<" => Value::Data(json!(a.number()? < b.number()?)),
        "<=" => Value::Data(json!(a.number()? <= b.number()?)),
        ">" => Value::Data(json!(a.number()? > b.number()?)),
        ">=" => Value::Data(json!(a.number()? >= b.number()?)),
        "==" => Value::Data(json!(equal_json(a.data()?, b.data()?))),
        "!=" => Value::Data(json!(!equal_json(a.data()?, b.data()?))),
        _ => bail!("unsupported operator {op}"),
    })
}
pub fn entry(functions: Vec<(&'static str, Function)>) -> Result<()> {
    entry_at(functions, ".")
}
pub fn entry_at(functions: Vec<(&'static str, Function)>, root: &str) -> Result<()> {
    entry_named(functions, root, "main")
}
pub fn entry_named(functions: Vec<(&'static str, Function)>, root: &str, entry: &str) -> Result<()> {
    let endpoint =
        std::env::var("WGTIME_ENDPOINT").unwrap_or_else(|_| wgtime::DEFAULT_ENDPOINT.into());
    let client = Client::connect(Path::new(&endpoint), unsafe { libc::geteuid() })?;
    let mut runtime = Runtime {
        client,
        remaining: 10_000_000,
        metrics: Default::default(),
        functions: functions.into_iter().collect(),
        harness: Default::default(),
        async_state: Default::default(),
        strategies: Default::default(),
        root: root.into(),
    };
    let mut args = std::env::args().skip(1);
    let mut build_output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => { runtime.root = args.next().context("missing root")?.into(); }
            "--build-output" => {
                let path = std::path::PathBuf::from(args.next().context("missing build output")?);
                ensure!(path.is_absolute(), "build output must be absolute");
                build_output = Some(path);
            }
            "--max-operations" => {
                runtime.remaining = args.next().context("missing operations")?.parse()?;
            }
            _ => bail!("unknown client argument {arg}"),
        }
    }
    let result = runtime.call(entry, vec![])?;
    if let Some(output) = build_output {
        ensure!(matches!(result, Value::Model(_)), "build entry must return a model");
        let model = output.join("model.json");
        let checkpoint = output.join("checkpoint.json");
        runtime.save_file("model", result.clone(), Value::Data(json!(model.to_str().context("UTF-8 build path required")?)), false)?;
        runtime.save_file("checkpoint", result, Value::Data(json!(checkpoint.to_str().context("UTF-8 build path required")?)), false)?;
        let build = persistence::inspect_artifact(&checkpoint, true)?;
        println!("{}", json!({"result":null,"metrics":runtime.metrics,"build":build}));
    } else {
        println!("{}", json!({"result":result.data()?,"metrics":runtime.metrics}));
    }
    Ok(())
}
