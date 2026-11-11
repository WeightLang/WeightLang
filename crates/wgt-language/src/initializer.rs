//! Checked, serializable initialization expressions. Parsing never draws random
//! values or allocates generated tensors; hosts explicitly materialize recipes.
use crate::script::parse::{self, Expr, Program, Stmt};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Recipe {
    Unbound,
    Scalar { value: f64 },
    Literal { shape: Vec<usize>, values: Vec<f32> },
    Full { shape: Vec<usize>, value: f32 },
    Eye { shape: Vec<usize> },
    Uniform { shape: Vec<usize>, seed: u64, low: f64, high: f64 },
    Unary { function: String, input: Box<Recipe> },
    Binary { operator: String, left: Box<Recipe>, right: Box<Recipe> },
    Reshape { shape: Vec<usize>, input: Box<Recipe> },
    Transpose { input: Box<Recipe> },
    Concat { left: Box<Recipe>, right: Box<Recipe> },
    Slice { input: Box<Recipe>, start: usize, end: usize },
}

fn elements(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() { return Ok(1); }
    crate::model::validate_shape(shape)
}
fn finite(value: f64) -> Result<f64> {
    ensure!(value.is_finite(), "nonfinite initializer value");
    Ok(value)
}
fn f32_value(value: f64) -> Result<f32> {
    finite(value)?;
    let value = value as f32;
    ensure!(value.is_finite(), "initializer exceeds f32 range");
    Ok(value)
}
fn unary(function: &str, value: f64) -> Result<f64> {
    finite(match function {
        "neg" => -value,
        "sqrt" => value.sqrt(),
        "exp" => value.exp(),
        "log" => value.ln(),
        "tanh" => value.tanh(),
        "sigmoid" => 1. / (1. + (-value).exp()),
        "relu" => value.max(0.),
        "square" => value * value,
        _ => bail!("unsupported initializer unary function: {function}"),
    })
}
fn binary(operator: &str, left: f64, right: f64) -> Result<f64> {
    finite(match operator {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        "/" => { ensure!(right != 0., "initializer division by zero"); left / right },
        _ => bail!("unsupported initializer binary operator: {operator}"),
    })
}

impl Recipe {
    /// Sequential-model initialization treats each explicitly seeded generator
    /// as a stream across declarations. Derive the next state without drawing.
    pub(crate) fn sequence_streams(&mut self, offsets: &mut BTreeMap<u64, u64>) -> Result<()> {
        fn visit(recipe: &mut Recipe, offsets: &BTreeMap<u64, u64>, used: &mut BTreeMap<u64, u64>) -> Result<()> {
            match recipe {
                Recipe::Uniform { seed, shape, .. } => {
                    let original = *seed;
                    let count = elements(shape)? as u64;
                    *seed = original.wrapping_add(offsets.get(&original).copied().unwrap_or(0).wrapping_mul(0x9e3779b97f4a7c15));
                    // Reusing a pure let-bound random tensor shares its values;
                    // repeated recipe references consume its stream only once.
                    used.entry(original).and_modify(|n| *n = (*n).max(count)).or_insert(count);
                },
                Recipe::Unary { input, .. } | Recipe::Reshape { input, .. } | Recipe::Transpose { input } | Recipe::Slice { input, .. } => visit(input, offsets, used)?,
                Recipe::Binary { left, right, .. } | Recipe::Concat { left, right } => { visit(left, offsets, used)?; visit(right, offsets, used)?; },
                _ => {},
            }
            Ok(())
        }
        let mut used = BTreeMap::new();
        visit(self, offsets, &mut used)?;
        for (seed, count) in used { let value = offsets.entry(seed).or_default(); *value = value.wrapping_add(count); }
        Ok(())
    }
    pub fn shape(&self) -> Result<Vec<usize>> {
        self.checked_shape(0, &mut 1_000_000)
    }
    fn checked_shape(&self, depth: usize, remaining: &mut usize) -> Result<Vec<usize>> {
        ensure!(depth <= 64 && *remaining > 0, "initializer expression is too complex");
        *remaining -= 1;
        let shape = match self {
            Self::Unbound => bail!("parameter requires an explicit weight binding or initializer"),
            Self::Scalar { value } => { finite(*value)?; Vec::new() },
            Self::Literal { shape, values } => {
                ensure!(values.len() <= *remaining, "initializer literal expansion is too large");
                *remaining -= values.len();
                ensure!(elements(shape)? == values.len() && values.iter().all(|v| v.is_finite()), "invalid literal initializer");
                shape.clone()
            },
            Self::Full { shape, value } => { ensure!(value.is_finite(), "nonfinite fill initializer"); shape.clone() },
            Self::Eye { shape } => { ensure!(shape.len() == 2, "eye initializer requires a matrix"); shape.clone() },
            Self::Uniform { shape, low, high, .. } => {
                f32_value(*low)?; f32_value(*high)?;
                ensure!(low < high, "uniform initializer requires low < high");
                shape.clone()
            },
            Self::Unary { function, input } => {
                ensure!(["neg", "sqrt", "exp", "log", "tanh", "sigmoid", "relu", "square"].contains(&function.as_str()), "unsupported initializer unary function");
                input.checked_shape(depth + 1, remaining)?
            },
            Self::Binary { operator, left, right } => {
                let a = left.checked_shape(depth + 1, remaining)?;
                let b = right.checked_shape(depth + 1, remaining)?;
                if operator == "@" {
                    ensure!(a.len() == 2 && b.len() == 2 && a[1] == b[0], "initializer matmul shape mismatch");
                    vec![a[0], b[1]]
                } else {
                    ensure!(["+", "-", "*", "/"].contains(&operator.as_str()), "unsupported initializer binary operator");
                    ensure!(a == b || a.is_empty() || b.is_empty(), "initializer elementwise shape mismatch");
                    if a.is_empty() { b } else { a }
                }
            },
            Self::Reshape { shape, input } => {
                ensure!(elements(shape)? == elements(&input.checked_shape(depth + 1, remaining)?)?, "initializer reshape element mismatch");
                shape.clone()
            },
            Self::Transpose { input } => {
                let shape = input.checked_shape(depth + 1, remaining)?;
                ensure!(shape.len() == 2, "initializer transpose requires a matrix");
                vec![shape[1], shape[0]]
            },
            Self::Concat { left, right } => {
                let a = left.checked_shape(depth + 1, remaining)?;
                let b = right.checked_shape(depth + 1, remaining)?;
                ensure!(a.len() == 1 && b.len() == 1, "initializer concat requires vectors");
                vec![a[0].checked_add(b[0]).context("initializer size overflow")?]
            },
            Self::Slice { input, start, end } => {
                let shape = input.checked_shape(depth + 1, remaining)?;
                ensure!(shape.len() == 1 && start < end && *end <= shape[0], "invalid initializer vector slice");
                vec![end - start]
            },
        };
        elements(&shape)?;
        Ok(shape)
    }
    /// Validate before generating any tensor. The caller supplies the storage
    /// shape; no implicit resize, dtype cast or missing-value fallback is used.
    pub fn materialize(&self, expected: &[usize]) -> Result<Vec<f32>> {
        ensure!(self.shape()? == expected, "initializer shape does not match parameter shape");
        self.evaluate()
    }
    fn evaluate(&self) -> Result<Vec<f32>> {
        let output = match self {
            Self::Unbound => bail!("parameter requires an explicit weight binding or initializer"),
            Self::Scalar { value } => vec![f32_value(*value)?],
            Self::Literal { values, .. } => values.clone(),
            Self::Full { shape, value } => vec![*value; elements(shape)?],
            Self::Eye { shape } => {
                let mut values = vec![0.; elements(shape)?];
                for i in 0..shape[0].min(shape[1]) { values[i * shape[1] + i] = 1.; }
                values
            },
            Self::Uniform { shape, seed, low, high } => {
                let mut state = *seed;
                (0..elements(shape)?).map(|_| f32_value(low + (high - low) * crate::source::initialization_random(&mut state))).collect::<Result<Vec<_>>>()?
            },
            Self::Unary { function, input } => input.evaluate()?.into_iter().map(|v| f32_value(unary(function, v as f64)?)).collect::<Result<Vec<_>>>()?,
            Self::Binary { operator, left, right } => {
                let a = left.evaluate()?;
                let b = right.evaluate()?;
                let sa = left.shape()?;
                let sb = right.shape()?;
                if operator == "@" {
                    let mut values = vec![0.; sa[0] * sb[1]];
                    for i in 0..sa[0] { for j in 0..sb[1] {
                        let sum: f64 = (0..sa[1]).map(|k| a[i * sa[1] + k] as f64 * b[k * sb[1] + j] as f64).sum();
                        values[i * sb[1] + j] = f32_value(sum)?;
                    }}
                    values
                } else {
                    (0..a.len().max(b.len())).map(|i| f32_value(binary(operator, a[if sa.is_empty() { 0 } else { i }] as f64, b[if sb.is_empty() { 0 } else { i }] as f64)?)).collect::<Result<Vec<_>>>()?
                }
            },
            Self::Reshape { input, .. } => input.evaluate()?,
            Self::Transpose { input } => {
                let shape = input.shape()?;
                let values = input.evaluate()?;
                (0..shape[1]).flat_map(|j| (0..shape[0]).map(move |i| (i, j))).map(|(i,j)| values[i * shape[1] + j]).collect()
            },
            Self::Concat { left, right } => { let mut values = left.evaluate()?; values.extend(right.evaluate()?); values },
            Self::Slice { input, start, end } => input.evaluate()?[*start..*end].to_vec(),
        };
        ensure!(output.iter().all(|v| v.is_finite()), "nonfinite initialized weights");
        Ok(output)
    }
}

/// Initialization-only generators share the ordinary WGT call syntax.
pub fn intrinsic(name: &str) -> bool {
    matches!(name, "zeros" | "ones" | "full" | "eye" | "uniform" | "xavier_uniform")
}

/// Identify pure helpers used by storage recipes, so a zero-argument initializer
/// is not mistaken for a mutable zero-argument model operation.
pub(crate) fn helpers(declarations: &str, functions: &Program) -> Result<HashSet<String>> {
    fn calls(expr: &Expr, functions: &Program, names: &mut HashSet<String>) {
        match expr {
            Expr::Call(name, args) | Expr::BuiltinCall(name, args) => {
                if functions.contains_key(name) { names.insert(name.clone()); }
                for arg in args { calls(arg, functions, names); }
            },
            Expr::List(items) | Expr::Tuple(items) => { for item in items { calls(item, functions, names); } },
            Expr::Binary(_, left, right) | Expr::Index(left, right) => { calls(left, functions, names); calls(right, functions, names); },
            Expr::Unary(_, value) => calls(value, functions, names),
            Expr::Slice(value, start, end) => { calls(value, functions, names); for bound in start.iter().chain(end.iter()) { calls(bound, functions, names); } },
            _ => {},
        }
    }
    let mut names = HashSet::new();
    for line in declarations.lines() {
        let text = line.split('#').next().unwrap_or("").trim().trim_end_matches(';');
        let keyword = text.split_whitespace().next().unwrap_or("");
        if !["parameter", "frozen", "learn", "state"].contains(&keyword) { continue; }
        let Some(index) = text.rfind('=').filter(|i| *i == 0 || text.as_bytes()[i - 1] != b'.') else { continue; };
        let rhs = text[index + 1..].trim();
        if rhs == "external" { continue; }
        let parsed = parse::parse_functions(&format!("wgt 0.5\nfn initializer_value() {{ return {rhs}; }}"))?;
        if let Stmt::Return(expr) = &parsed["initializer_value"].body[0].kind { calls(expr, functions, &mut names); }
    }
    let mut checked = HashSet::new();
    loop {
        let pending: Vec<_> = names.difference(&checked).cloned().collect();
        if pending.is_empty() { break; }
        for name in pending {
            checked.insert(name.clone());
            for statement in &functions[&name].body {
                if let Stmt::Let(_, expr) | Stmt::Return(expr) = &statement.kind { calls(expr, functions, &mut names); }
            }
        }
    }
    Ok(names)
}

pub(crate) fn probability(text: Option<&str>, frozen: bool, functions: &Program) -> Result<Recipe> {
    let Some(text) = text.filter(|value| value.trim() != "external") else { return Ok(Recipe::Unbound); };
    if let Ok(value) = text.trim().parse::<f64>() {
        ensure!(value.is_finite() && (0. ..=1.).contains(&value), "prob initializer must be in [0,1]");
        let value = if frozen { value } else {
            ensure!(value > 0. && value < 1., "trainable prob initializer must be strictly between 0 and 1; use frozen for exact endpoints");
            value.ln() - (-value).ln_1p()
        };
        return Ok(Recipe::Full { shape: vec![1], value: value as f32 });
    }
    let value = parse(Some(text), &[1], functions)?;
    if frozen { return Ok(value); }
    // A probability-producing recipe is explicitly lowered into storage logits.
    // The log domains reject either endpoint and values outside the interval.
    Ok(Recipe::Binary { operator: "-".into(),
        left: Box::new(Recipe::Unary { function: "log".into(), input: Box::new(value.clone()) }),
        right: Box::new(Recipe::Unary { function: "log".into(), input: Box::new(Recipe::Binary { operator: "-".into(), left: Box::new(Recipe::Scalar { value: 1. }), right: Box::new(value) }) }),
    })
}

/// Parse through the existing function parser and inline its pure helper AST.
pub fn parse(text: Option<&str>, shape: &[usize], functions: &Program) -> Result<Recipe> {
    let Some(text) = text else { return Ok(Recipe::Unbound); };
    let text = text.trim().trim_end_matches(';').trim();
    if text == "external" { return Ok(Recipe::Unbound); }
    let parsed = parse::parse_functions(&format!("wgt 0.5\nfn initializer_value() {{ return {text}; }}"))?;
    ensure!(parsed.len() == 1, "initializer must be one expression");
    let function = &parsed["initializer_value"];
    ensure!(function.body.len() == 1, "initializer must be one expression");
    let Stmt::Return(expr) = &function.body[0].kind else { bail!("initializer requires an expression"); };
    let mut compiler = Compiler { shape, functions, stack: Vec::new(), nodes: 100_000 };
    let mut recipe = compiler.expression(expr, &HashMap::new())?;
    let actual = recipe.shape()?;
    // Existing array declarations are flat payloads with an explicit storage
    // shape. Other expressions must carry their own exact tensor geometry.
    if actual != shape && matches!(expr, Expr::List(_)) && elements(&actual)? == elements(shape)? {
        recipe = Recipe::Reshape { shape: shape.to_vec(), input: Box::new(recipe) };
    }
    ensure!(recipe.shape()? == shape, "initializer shape mismatch: expected {shape:?}, got {:?}", recipe.shape()?);
    Ok(recipe)
}

struct Compiler<'a> {
    shape: &'a [usize],
    functions: &'a Program,
    stack: Vec<String>,
    nodes: usize,
}
impl Compiler<'_> {
    fn expression(&mut self, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<Recipe> {
        ensure!(self.nodes > 0, "initializer expansion is too large");
        self.nodes -= 1;
        let recipe = match expr {
            Expr::Number(value) => Recipe::Scalar { value: finite(*value)? },
            Expr::Bool(value) => Recipe::Scalar { value: if *value { 1. } else { 0. } },
            Expr::Name(name) => env.get(name).with_context(|| format!("initializer cannot capture parameter or host value: {name}"))?.clone(),
            Expr::List(values) => {
                let values = values.iter().map(|v| self.number(v, env).and_then(f32_value)).collect::<Result<Vec<_>>>()?;
                ensure!(!values.is_empty(), "initializer literal cannot be empty");
                Recipe::Literal { shape: vec![values.len()], values }
            },
            Expr::Unary(op, value) if op == "-" => self.unary("neg", value, env)?,
            Expr::Binary(op, left, right) if ["+", "-", "*", "/", "@"].contains(&op.as_str()) => {
                let left = self.expression(left, env)?;
                let right = self.expression(right, env)?;
                if let (Recipe::Scalar { value: a }, Recipe::Scalar { value: b }) = (&left, &right) {
                    Recipe::Scalar { value: binary(op, *a, *b)? }
                } else { Recipe::Binary { operator: op.clone(), left: Box::new(left), right: Box::new(right) } }
            },
            Expr::Binary(op, value, power) if op == "**" && matches!(**power, Expr::Number(2.)) => self.unary("square", value, env)?,
            Expr::Call(name, args) | Expr::BuiltinCall(name, args) => self.call(name, args, env)?,
            Expr::Slice(value, start, end) => {
                let input = self.expression(value, env)?;
                let shape = input.shape()?;
                ensure!(shape.len() == 1, "initializer slice requires a vector");
                Recipe::Slice { input: Box::new(input), start: start.as_ref().map(|v| self.index(v, env)).transpose()?.unwrap_or(0), end: end.as_ref().map(|v| self.index(v, env)).transpose()?.unwrap_or(shape[0]) }
            },
            _ => bail!("initializer must be a pure, bounded tensor expression"),
        };
        recipe.shape()?;
        Ok(recipe)
    }
    fn number(&mut self, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<f64> {
        match self.expression(expr, env)? { Recipe::Scalar { value } => Ok(value), _ => bail!("initializer argument must be a constant scalar") }
    }
    fn index(&mut self, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<usize> {
        let value = self.number(expr, env)?;
        ensure!(value >= 0. && value.fract() == 0. && value <= 16_777_216., "initializer index must be a bounded nonnegative integer");
        Ok(value as usize)
    }
    fn seed(&mut self, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<u64> {
        if let Expr::Text(value) = expr { return value.parse().context("initializer seed must be an unsigned 64-bit integer"); }
        let value = self.number(expr, env)?;
        ensure!(value >= 0. && value.fract() == 0. && value <= 9_007_199_254_740_991., "seed must be an exact integer; quote u64 seeds above 2^53-1");
        Ok(value as u64)
    }
    fn dimensions(&mut self, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<Vec<usize>> {
        let Expr::List(values) = expr else { bail!("initializer shape requires an explicit dimension list"); };
        let shape = values.iter().map(|v| self.index(v, env)).collect::<Result<Vec<_>>>()?;
        crate::model::validate_shape(&shape)?;
        Ok(shape)
    }
    fn unary(&mut self, function: &str, expr: &Expr, env: &HashMap<String, Recipe>) -> Result<Recipe> {
        let input = self.expression(expr, env)?;
        if let Recipe::Scalar { value } = input { Ok(Recipe::Scalar { value: unary(function, value)? }) }
        else { Ok(Recipe::Unary { function: function.into(), input: Box::new(input) }) }
    }
    fn call(&mut self, name: &str, args: &[Expr], env: &HashMap<String, Recipe>) -> Result<Recipe> {
        if intrinsic(name) {
            let (shape, args) = if args.first().is_some_and(|v| matches!(v, Expr::List(_))) {
                (self.dimensions(&args[0], env)?, &args[1..])
            } else { (self.shape.to_vec(), args) };
            return match (name, args) {
                ("zeros", []) => Ok(Recipe::Full { shape, value: 0. }),
                ("ones", []) => Ok(Recipe::Full { shape, value: 1. }),
                ("full", [value]) => Ok(Recipe::Full { shape, value: f32_value(self.number(value, env)?)? }),
                ("eye", []) => Ok(Recipe::Eye { shape }),
                ("uniform", [seed, low, high]) => Ok(Recipe::Uniform { shape, seed: self.seed(seed, env)?, low: self.number(low, env)?, high: self.number(high, env)? }),
                ("xavier_uniform", [seed]) => {
                    ensure!(shape.len() == 2, "xavier_uniform requires a matrix; use explicit uniform for other shapes");
                    let limit = (6. / (shape[0] + shape[1]) as f64).sqrt();
                    Ok(Recipe::Uniform { shape, seed: self.seed(seed, env)?, low: -limit, high: limit })
                },
                _ => bail!("invalid initializer arguments for {name}"),
            };
        }
        if ["sqrt", "exp", "log", "tanh", "sigmoid", "relu", "square"].contains(&name) {
            ensure!(args.len() == 1, "initializer {name} expects one argument");
            return self.unary(name, &args[0], env);
        }
        match (name, args) {
            ("reshape", [input, shape]) => return Ok(Recipe::Reshape { shape: self.dimensions(shape, env)?, input: Box::new(self.expression(input, env)?) }),
            ("transpose", [input]) => return Ok(Recipe::Transpose { input: Box::new(self.expression(input, env)?) }),
            ("concat", [left, right]) => return Ok(Recipe::Concat { left: Box::new(self.expression(left, env)?), right: Box::new(self.expression(right, env)?) }),
            ("scale", [input, scale]) => return Ok(Recipe::Binary { operator: "*".into(), left: Box::new(self.expression(input, env)?), right: Box::new(Recipe::Scalar { value: self.number(scale, env)? }) }),
            ("matmul", [left, right]) => return Ok(Recipe::Binary { operator: "@".into(), left: Box::new(self.expression(left, env)?), right: Box::new(self.expression(right, env)?) }),
            _ => {},
        }
        ensure!(self.stack.len() < 32 && !self.stack.iter().any(|v| v == name), "recursive/deep initializer function: {name}");
        let function = self.functions.get(name).with_context(|| format!("unknown initialization function: {name}"))?.clone();
        ensure!(!function.is_async && function.signature.is_none() && function.tensor_signature.is_none() && function.args.len() == args.len(), "initializer function must be a pure helper with matching arguments");
        let mut locals = HashMap::new();
        for (name, expr) in function.args.iter().zip(args) { locals.insert(name.clone(), self.expression(expr, env)?); }
        self.stack.push(name.into());
        for (index, statement) in function.body.iter().enumerate() {
            match &statement.kind {
                Stmt::Let(name, expr) => {
                    ensure!(!locals.contains_key(name), "initializer local binding must be unique");
                    let value = self.expression(expr, &locals)?;
                    locals.insert(name.clone(), value);
                },
                Stmt::Return(expr) if index + 1 == function.body.len() => {
                    let value = self.expression(expr, &locals)?;
                    self.stack.pop();
                    return Ok(value);
                },
                _ => bail!("initializer helper supports immutable let and a final return"),
            }
        }
        bail!("initializer helper requires a final return")
    }
}
