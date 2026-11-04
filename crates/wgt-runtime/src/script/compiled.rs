//! Shared execution support for generated Rust; no AST evaluation.
use super::*;

pub fn index(object: Value, index: Value) -> Result<Value> {
    match object {
        Value::List(v) | Value::CategoricalVector(_, v) => v
            .get(index.integer()?)
            .cloned()
            .context("list index out of bounds"),
        Value::RecordBatch(batch) => batch.rows.get(index.integer()?).cloned().context("record batch index out of bounds"),
        Value::Object(v) => v
            .get(index.text()?)
            .cloned()
            .context("missing object field"),
        _ => bail!("index requires list or object"),
    }
}
/// Shared half-open bounds for ordinary lists and tensor lowering. Never clamp.
pub(crate) fn slice_bounds(len: usize, start: Option<f64>, end: Option<f64>) -> Result<(usize, usize)> {
    let endpoint = |n: f64| -> Result<usize> {
        ensure!(n.is_finite() && n >= 0. && n.fract() == 0., "slice endpoints must be nonnegative integers; negative indices are unsupported");
        ensure!(n <= len as f64, "slice endpoint out of bounds (length {len})");
        Ok(n as usize)
    };
    let start = start.map(endpoint).transpose()?.unwrap_or(0);
    let end = end.map(endpoint).transpose()?.unwrap_or(len);
    ensure!(start <= end, "slice start must not exceed end");
    Ok((start, end))
}

pub fn slice(object: Value, start: Option<Value>, end: Option<Value>) -> Result<Value> {
    let values = object.list()?;
    let (start, end) = slice_bounds(values.len(),
        start.as_ref().map(Value::number).transpose()?,
        end.as_ref().map(Value::number).transpose()?)?;
    Ok(Value::List(values[start..end].to_vec()))
}

pub fn unary(op: &str, value: Value) -> Result<Value> {
    if op == "!" {
        Ok(Value::Bool(!value.boolean()?))
    } else {
        number(-value.number()?)
    }
}
pub fn binary(op: &str, left: Value, right: Value) -> Result<Value> {
    match op {
        "&&" | "||" => Ok(Value::Bool(right.boolean()?)),
        "==" => Ok(Value::Bool(left == right)),
        "!=" => Ok(Value::Bool(left != right)),
        "+" => match (&left, &right) {
            (Value::Text(a), Value::Text(b)) => Ok(Value::Text(format!("{a}{b}"))),
            _ => number(left.number()? + right.number()?),
        },
        "-" => number(left.number()? - right.number()?),
        "*" => number(left.number()? * right.number()?),
        "/" => number(left.number()? / right.number()?),
        "%" => number(left.number()? % right.number()?),
        "<" => Ok(Value::Bool(left.number()? < right.number()?)),
        "<=" => Ok(Value::Bool(left.number()? <= right.number()?)),
        ">" => Ok(Value::Bool(left.number()? > right.number()?)),
        ">=" => Ok(Value::Bool(left.number()? >= right.number()?)),
        _ => bail!("unknown operator"),
    }
}

#[derive(Clone)]
pub struct Function {
    pub args: Vec<&'static str>,
    pub body: fn(&mut Runtime, &mut Env) -> Result<Flow>,
    pub async_body: Option<fn(std::rc::Rc<std::cell::RefCell<Runtime>>,Env)->super::async_runtime::FutureValue>,
}

pub fn run(functions: Vec<(&'static str, Function)>, root: &Path, budget: u64) -> Result<Report> {
    run_with_dependencies(functions, root, budget, BTreeMap::new())
}
pub fn run_with_dependencies(
    functions: Vec<(&'static str, Function)>,
    root: &Path,
    budget: u64,
    dependencies: BTreeMap<String, String>,
) -> Result<Report> {
    let mut runtime = Runtime {
        permissions: permissions::Permissions::from_env()?,
        program: HashMap::new(),
        compiled: functions.into_iter().map(|(n, f)| (n.into(), f)).collect(),
        root: root.into(),
        dependencies,
        remaining: budget,
        gpu_benchmark_restricted: false,
        depth: 0,
        models: Vec::new(),
        adaptive: Vec::new(),
        readers: Vec::new(),
        gradients: super::optimization::Gradients::default(),
        rng: 0,
        metrics: BTreeMap::new(),
        async_state:super::async_runtime::AsyncState::default(),
    };
    let result = runtime.call("main", Vec::new())?.json()?;
    Ok(Report {
        result,
        metrics: runtime.metrics,
        operations: budget - runtime.remaining,
    })
}

/// Generated executables resolve data against the original root unless relocated explicitly.
pub fn entry(functions: Vec<(&'static str, Function)>, root: &str) -> Result<()> {
    entry_with_dependencies(functions, root, Vec::new())
}
pub fn entry_with_dependencies(
    functions: Vec<(&'static str, Function)>,
    root: &str,
    dependencies: Vec<(&str, &str)>,
) -> Result<()> {
    let mut root = PathBuf::from(root);
    let mut budget = 10_000_000;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => root = PathBuf::from(args.next().context("--root needs a directory")?),
            "--max-operations" => {
                budget = args
                    .next()
                    .context("--max-operations needs a value")?
                    .parse()?
            }
            _ => bail!("unknown argument: {arg}; accepts --root DIRECTORY --max-operations N"),
        }
    }
    let dependencies = dependencies
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
    let report = run_with_dependencies(functions, &root, budget, dependencies)
        .with_context(|| root.display().to_string())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
