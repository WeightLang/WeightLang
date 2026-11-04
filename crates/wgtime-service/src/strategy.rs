//! Registration-time lowering of a capability-free WGT subset to resident instructions.
//! No source evaluation, native libraries, paths, or runtime host calls during execution.
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use weightlang_runtime::{
    adaptive::{optimization::GradientSnapshot, AdaptiveSession},
    script::{self, Expr, Statement, Stmt, Value},
};

#[derive(Clone, Debug)]
enum Op {
    Constant(Value),
    Load(String),
    Store(String, bool),
    Drop,
    List(usize),
    Unary(String),
    Binary(String),
    Index,
    ReadPath(String, Vec<PathIndex>),
    Call(String, usize),
    Jump(usize),
    Unless(usize),
    Return,
    Assert,
}
#[derive(Clone, Debug)]
enum PathIndex {
    Literal(Value),
    Variable(String),
}
fn read_path(expr: &Expr, indices: &mut Vec<PathIndex>) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.clone()),
        Expr::Index(base, index) => {
            let name = read_path(base, indices)?;
            indices.push(match &**index {
                Expr::Name(n) => PathIndex::Variable(n.clone()),
                Expr::Text(s) => PathIndex::Literal(Value::Text(s.clone())),
                Expr::Number(n) => PathIndex::Literal(Value::Number(*n)),
                _ => return None,
            });
            Some(name)
        }
        _ => None,
    }
}
#[derive(Debug)]
struct Function {
    args: Vec<String>,
    code: Vec<Op>,
}
#[derive(Debug)]
pub struct Strategy {
    entry: String,
    functions: HashMap<String, Function>,
}
const PURE: &[(&str, usize)] = &[
    ("len", 1),
    ("range", 2),
    ("append", 2),
    ("sqrt", 1),
    ("abs", 1),
    ("min", 2),
    ("max", 2),
    ("differentiate", 3),
    ("differentiate_at", 2),
    ("gradient_snapshot", 1),
    ("commit_gradients", 3),
    ("release_gradient", 1),
];

impl Strategy {
    pub fn compile(source: &str) -> Result<Self> {
        Self::compile_entry(source, "strategy")
    }
    pub fn compile_entry(source: &str, entry: &str) -> Result<Self> {
        ensure!(
            source.len() <= 256 * 1024,
            "strategy source exceeds registration budget"
        );
        let program = script::parse(source)?;
        let entry_fn = program.get(entry).context("missing strategy entry")?;
        ensure!(
            matches!(entry_fn.args.len(), 3 | 4),
            "strategy entry requires three arguments or four with parameters"
        );
        let mut result = Self {
            entry: entry.into(),
            functions: HashMap::new(),
        };
        fn visit(
            name: &str,
            program: &script::Program,
            result: &mut Strategy,
            active: &mut HashSet<String>,
        ) -> Result<()> {
            ensure!(
                !active.contains(name),
                "recursive strategy calls are unsupported"
            );
            if result.functions.contains_key(name) {
                return Ok(());
            }
            ensure!(
                active.len() < 32 && result.functions.len() < 64,
                "strategy call graph budget exceeded"
            );
            let f = program.get(name).context("unknown strategy function")?;
            ensure!(!f.is_async, "async strategy functions are unsupported");
            active.insert(name.into());
            let mut compiler = Compiler {
                code: vec![],
                loops: vec![],
                serial: 0,
            };
            compiler.block(&f.body)?;
            compiler.code.push(Op::Constant(Value::Null));
            compiler.code.push(Op::Return);
            for op in &compiler.code {
                if let Op::Call(target, n) = op {
                    if target == "__record" { ensure!(n % 2 == 0, "invalid record fields"); }
                    else if let Some((_, arity)) = PURE.iter().find(|(s, _)| *s == target) {
                        ensure!(n == arity, "invalid strategy builtin arity: {target}");
                    } else {
                        let called = program
                            .get(target)
                            .with_context(|| format!("forbidden strategy call: {target}"))?;
                        ensure!(called.args.len() == *n, "invalid helper arity");
                        visit(target, program, result, active)?;
                    }
                }
            }
            active.remove(name);
            result.functions.insert(
                name.into(),
                Function {
                    args: f.args.clone(),
                    code: compiler.code,
                },
            );
            Ok(())
        }
        visit(entry, &program, &mut result, &mut HashSet::new())?;
        Ok(result)
    }
    pub fn execute(
        &self,
        session: &mut AdaptiveSession,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        remaining: &mut u64,
        memory: usize,
    ) -> Result<serde_json::Value> {
        self.execute_with_parameters(
            session,
            inputs,
            targets,
            &serde_json::Value::Null,
            remaining,
            memory,
        )
    }
    pub fn execute_with_parameters(
        &self,
        session: &mut AdaptiveSession,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        parameters: &serde_json::Value,
        remaining: &mut u64,
        memory: usize,
    ) -> Result<serde_json::Value> {
        let mut vm = Vm {
            session,
            remaining,
            memory,
            allocated: 0,
            gradients: HashMap::new(),
            gradient_bytes: 0,
            next: 0,
            commits: 0,
            loss: None,
        };
        let mut args = vec![Value::Adaptive(0), vm.batch(inputs)?, vm.batch(targets)?];
        if self.functions[&self.entry].args.len() == 4 {
            weightlang_runtime::adaptive::optimization::validate_state(parameters)?;
            vm.reserve(json_size(parameters)?)?;
            args.push(from_json(parameters));
        }
        let result = vm.call(self, &self.entry, args)?;
        weightlang_runtime::script::validate_strategy_result(&result)?;
        let result = json(&result)?;
        Ok(serde_json::json!({"commits":vm.commits,"loss":vm.loss,"remaining":*vm.remaining,"result":result}))
    }
}
struct Compiler {
    code: Vec<Op>,
    loops: Vec<(Vec<usize>, Vec<usize>)>,
    serial: usize,
}
impl Compiler {
    fn jump(&mut self, conditional: bool) -> usize {
        let i = self.code.len();
        self.code.push(if conditional {
            Op::Unless(0)
        } else {
            Op::Jump(0)
        });
        i
    }
    fn patch(&mut self, i: usize, target: usize) {
        match &mut self.code[i] {
            Op::Jump(n) | Op::Unless(n) => *n = target,
            _ => unreachable!(),
        }
    }
    fn expr(&mut self, e: &Expr) -> Result<()> {
        match e {
            Expr::Number(n) => self.code.push(Op::Constant(Value::Number(*n))),
            Expr::Bool(b) => self.code.push(Op::Constant(Value::Bool(*b))),
            Expr::Null => self.code.push(Op::Constant(Value::Null)),
            Expr::Text(s) => self.code.push(Op::Constant(Value::Text(s.clone()))),
            Expr::Name(n) => self.code.push(Op::Load(n.clone())),
            Expr::List(v) => {
                for x in v {
                    self.expr(x)?;
                }
                self.code.push(Op::List(v.len()));
            }
            Expr::Unary(op, x) => {
                self.expr(x)?;
                self.code.push(Op::Unary(op.clone()));
            }
            Expr::Binary(op, a, b) if op == "&&" || op == "||" => {
                self.expr(a)?;
                let branch = self.jump(true);
                if op == "||" {
                    self.code.push(Op::Constant(Value::Bool(true)));
                } else {
                    self.expr(b)?;
                }
                let end = self.jump(false);
                self.patch(branch, self.code.len());
                if op == "||" {
                    self.expr(b)?;
                } else {
                    self.code.push(Op::Constant(Value::Bool(false)));
                }
                self.patch(end, self.code.len());
            }
            Expr::Binary(op, a, b) => {
                self.expr(a)?;
                self.expr(b)?;
                self.code.push(Op::Binary(op.clone()));
            }
            Expr::Index(a, b) => {
                let mut path = vec![];
                if let Some(name) = read_path(e, &mut path) {
                    self.code.push(Op::ReadPath(name, path));
                } else {
                    self.expr(a)?;
                    self.expr(b)?;
                    self.code.push(Op::Index);
                }
            }
            Expr::Call(n, args) | Expr::BuiltinCall(n, args) | Expr::StaticCall(n, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.code.push(Op::Call(n.clone(), args.len()));
            }
            _ => bail!("unsupported expression in resident strategy"),
        }
        Ok(())
    }
    fn block(&mut self, body: &[Statement]) -> Result<()> {
        for s in body {
            match &s.kind {
                Stmt::Let(n, e) | Stmt::Set(n, e) => {
                    self.expr(e)?;
                    self.code
                        .push(Op::Store(n.clone(), matches!(s.kind, Stmt::Set(..))));
                }
                Stmt::Eval(e) => {
                    self.expr(e)?;
                    self.code.push(Op::Drop);
                }
                Stmt::Return(e) => {
                    self.expr(e)?;
                    self.code.push(Op::Return);
                }
                Stmt::Assert(e, msg) => {
                    self.expr(e)?;
                    let fail = self.jump(true);
                    let end = self.jump(false);
                    self.patch(fail, self.code.len());
                    if let Some(m) = msg {
                        self.expr(m)?;
                    } else {
                        self.code
                            .push(Op::Constant(Value::Text("assertion failed".into())));
                    }
                    self.code.push(Op::Assert);
                    self.patch(end, self.code.len());
                }
                Stmt::If(e, a, b) => {
                    self.expr(e)?;
                    let no = self.jump(true);
                    self.block(a)?;
                    let end = self.jump(false);
                    self.patch(no, self.code.len());
                    self.block(b)?;
                    self.patch(end, self.code.len());
                }
                Stmt::While(e, b) => {
                    let start = self.code.len();
                    self.expr(e)?;
                    let end = self.jump(true);
                    self.loops.push((vec![], vec![]));
                    self.block(b)?;
                    self.code.push(Op::Jump(start));
                    let finish = self.code.len();
                    self.patch(end, finish);
                    self.end_loop(finish, start);
                }
                Stmt::For(n, e, b) => {
                    let serial = self.serial;
                    self.serial += 1;
                    let values = format!("$values{serial}");
                    let index = format!("$index{serial}");
                    self.expr(e)?;
                    self.code.push(Op::Store(values.clone(), false));
                    self.code.push(Op::Constant(Value::Number(0.)));
                    self.code.push(Op::Store(index.clone(), false));
                    let start = self.code.len();
                    self.code.push(Op::Load(index.clone()));
                    self.code.push(Op::Load(values.clone()));
                    self.code.push(Op::Call("len".into(), 1));
                    self.code.push(Op::Binary("<".into()));
                    let end = self.jump(true);
                    self.code.push(Op::Load(values));
                    self.code.push(Op::Load(index.clone()));
                    self.code.push(Op::Index);
                    self.code.push(Op::Store(n.clone(), false));
                    self.loops.push((vec![], vec![]));
                    self.block(b)?;
                    let next = self.code.len();
                    self.code.push(Op::Load(index.clone()));
                    self.code.push(Op::Constant(Value::Number(1.)));
                    self.code.push(Op::Binary("+".into()));
                    self.code.push(Op::Store(index, true));
                    self.code.push(Op::Jump(start));
                    let finish = self.code.len();
                    self.patch(end, finish);
                    self.end_loop(finish, next);
                }
                Stmt::Break | Stmt::Continue => {
                    ensure!(!self.loops.is_empty(), "loop control outside strategy loop");
                    let j = self.jump(false);
                    let l = self.loops.last_mut().unwrap();
                    if matches!(s.kind, Stmt::Break) {
                        l.0.push(j)
                    } else {
                        l.1.push(j)
                    }
                }
                _ => bail!(
                    "unsupported statement in resident strategy at line {}",
                    s.line
                ),
            }
        }
        Ok(())
    }
    fn end_loop(&mut self, end: usize, next: usize) {
        let (b, c) = self.loops.pop().unwrap();
        for i in b {
            self.patch(i, end)
        }
        for i in c {
            self.patch(i, next)
        }
    }
}
struct Vm<'a> {
    session: &'a mut AdaptiveSession,
    remaining: &'a mut u64,
    memory: usize,
    allocated: usize,
    gradients: HashMap<u64, GradientSnapshot>,
    gradient_bytes: usize,
    next: u64,
    commits: u64,
    loss: Option<f32>,
}
fn size(v: &Value) -> Result<usize> {
    value_size(v, 0)
}
fn value_size(v: &Value, depth: usize) -> Result<usize> {
    ensure!(depth <= 64, "strategy value nesting exceeds 64");
    Ok(std::mem::size_of::<Value>()
        .max(64)
        .checked_add(match v {
            Value::Text(s) => s.len(),
            Value::List(xs) => xs.iter().try_fold(0usize, |a, x| {
                a.checked_add(value_size(x, depth + 1)?)
                    .context("value size overflow")
            })?,
            Value::Object(xs) => xs.iter().try_fold(0usize, |a, (k, x)| {
                a.checked_add(k.len())
                    .and_then(|a| a.checked_add(value_size(x, depth + 1).ok()?))
                    .context("value size overflow")
            })?,
            _ => 0,
        })
        .context("value size overflow")?)
}
fn json_size(v: &serde_json::Value) -> Result<usize> {
    let mut n = 128usize;
    match v {
        serde_json::Value::Array(xs) => {
            for x in xs {
                n = n
                    .checked_add(json_size(x)?)
                    .context("configuration size overflow")?;
            }
        }
        serde_json::Value::Object(xs) => {
            for (k, x) in xs {
                n = n
                    .checked_add(k.len())
                    .and_then(|n| n.checked_add(json_size(x).ok()?))
                    .context("configuration size overflow")?;
            }
        }
        serde_json::Value::String(s) => {
            n = n
                .checked_add(s.len())
                .context("configuration size overflow")?
        }
        _ => {}
    }
    Ok(n)
}
fn from_json(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => Value::Number(n.as_f64().unwrap()),
        serde_json::Value::String(s) => Value::Text(s.clone()),
        serde_json::Value::Array(v) => Value::List(v.iter().map(from_json).collect()),
        serde_json::Value::Object(v) => {
            Value::Object(v.iter().map(|(k, v)| (k.clone(), from_json(v))).collect())
        }
    }
}
fn json(v: &Value) -> Result<serde_json::Value> {
    Ok(match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::json!(b),
        Value::Number(n) => {
            ensure!(n.is_finite(), "nonfinite state");
            serde_json::json!(n)
        }
        Value::Text(s) => serde_json::json!(s),
        Value::List(xs) => serde_json::Value::Array(xs.iter().map(json).collect::<Result<_>>()?),
        Value::Object(xs) => serde_json::Value::Object(
            xs.iter()
                .map(|(k, v)| Ok((k.clone(), json(v)?)))
                .collect::<Result<_>>()?,
        ),
        _ => bail!("resource handles cannot enter optimizer state"),
    })
}
fn matrix(v: &Value) -> Result<Vec<Vec<f32>>> {
    v.list()?
        .iter()
        .map(|r| {
            r.list()?
                .iter()
                .map(|x| {
                    let n = x.number()? as f32;
                    ensure!(n.is_finite(), "nonfinite f32 parameter");
                    Ok(n)
                })
                .collect()
        })
        .collect()
}
fn number(n: f64) -> Result<Value> {
    ensure!(n.is_finite(), "nonfinite arithmetic result");
    Ok(Value::Number(n))
}
impl Vm<'_> {
    fn tick(&mut self, n: usize) -> Result<()> {
        let old = *self.remaining;
        *self.remaining = self.remaining.saturating_sub(n as u64);
        ensure!(old >= n as u64, "strategy operation budget exceeded");
        Ok(())
    }
    fn reserve(&mut self, n: usize) -> Result<()> {
        let next = self
            .allocated
            .checked_add(n)
            .context("strategy allocation overflow")?;
        ensure!(
            next.checked_add(self.gradient_bytes)
                .is_some_and(|n| n <= self.memory),
            "strategy memory budget exceeded"
        );
        self.tick(n / 32 + 1)?;
        self.allocated = next;
        Ok(())
    }
    fn clone_value(&mut self, v: &Value) -> Result<Value> {
        self.reserve(size(v)?)?;
        Ok(v.clone())
    }
    fn batch(&mut self, v: &[Vec<f32>]) -> Result<Value> {
        let n = v
            .iter()
            .try_fold(128usize, |n, row| {
                n.checked_add(128)
                    .and_then(|n| n.checked_add(row.len().checked_mul(128)?))
            })
            .context("batch size overflow")?;
        self.reserve(n)?;
        ensure!(v.iter().flatten().all(|n| n.is_finite()), "nonfinite batch");
        Ok(Value::List(
            v.iter()
                .map(|r| Value::List(r.iter().map(|n| Value::Number(*n as f64)).collect()))
                .collect(),
        ))
    }
    fn call(&mut self, strategy: &Strategy, name: &str, args: Vec<Value>) -> Result<Value> {
        if let Some(f) = strategy.functions.get(name) {
            self.reserve(f.args.iter().map(|n| n.len() + 64).sum())?;
            let mut env: HashMap<_, _> = f.args.iter().cloned().zip(args).collect();
            let mut stack: Vec<Value> = vec![];
            let mut pc = 0;
            while pc < f.code.len() {
                self.tick(1)?;
                let op = &f.code[pc];
                pc += 1;
                match op {
                    Op::Constant(v) => stack.push(self.clone_value(v)?),
                    Op::Load(n) => {
                        let v = env
                            .get(n)
                            .with_context(|| format!("undefined strategy variable: {n}"))?;
                        stack.push(self.clone_value(v)?);
                    }
                    Op::Store(n, declared) => {
                        ensure!(
                            !*declared || env.contains_key(n),
                            "assignment to undeclared variable"
                        );
                        env.insert(n.clone(), stack.pop().context("invalid strategy stack")?);
                    }
                    Op::Drop => {
                        stack.pop().context("invalid strategy stack")?;
                    }
                    Op::List(n) => {
                        self.reserve(128)?;
                        let xs = stack.split_off(stack.len() - n);
                        stack.push(Value::List(xs));
                    }
                    Op::Unary(op) => {
                        let v = stack.pop().unwrap();
                        stack.push(script::compiled::unary(op, v)?);
                    }
                    Op::Binary(op) => {
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        self.reserve(
                            size(&a)?
                                .checked_add(size(&b)?)
                                .context("binary size overflow")?,
                        )?;
                        let v = script::compiled::binary(op, a, b)?;
                        stack.push(v);
                    }
                    Op::Index => {
                        let i = stack.pop().unwrap();
                        let a = stack.pop().unwrap();
                        let v = match &a {
                            Value::List(xs) => {
                                xs.get(i.integer()?).context("list index out of bounds")?
                            }
                            Value::Object(xs) => {
                                xs.get(i.text()?).context("missing object field")?
                            }
                            _ => bail!("index requires list or object"),
                        };
                        stack.push(self.clone_value(v)?);
                    }
                    Op::ReadPath(name, path) => {
                        self.tick(path.len())?;
                        let mut value = env.get(name).context("undefined strategy variable")?;
                        for index in path {
                            let key = match index {
                                PathIndex::Literal(v) => v,
                                PathIndex::Variable(n) => {
                                    env.get(n).context("undefined strategy index")?
                                }
                            };
                            value = match value {
                                Value::List(xs) => {
                                    xs.get(key.integer()?).context("list index out of bounds")?
                                }
                                Value::Object(xs) => {
                                    xs.get(key.text()?).context("missing object field")?
                                }
                                _ => bail!("index requires list or object"),
                            };
                        }
                        stack.push(self.clone_value(value)?);
                    }
                    Op::Call(n, count) => {
                        let args = stack.split_off(stack.len() - count);
                        stack.push(self.call(strategy, n, args)?);
                    }
                    Op::Jump(n) => pc = *n,
                    Op::Unless(n) => {
                        if !stack.pop().unwrap().boolean()? {
                            pc = *n;
                        }
                    }
                    Op::Return => return Ok(stack.pop().unwrap()),
                    Op::Assert => bail!("{}", stack.pop().unwrap().text()?),
                }
            }
            return Ok(Value::Null);
        }
        match name {
            "__record" => {
                let mut record = std::collections::BTreeMap::new();
                for pair in args.chunks(2) {
                    let key = pair[0].text()?.to_owned();
                    self.reserve(key.len().checked_add(size(&pair[1])?).context("record size overflow")?)?;
                    ensure!(record.insert(key, pair[1].clone()).is_none(), "duplicate record field");
                }
                Ok(Value::Object(record))
            }
            "len" => number(args[0].list()?.len() as f64),
            "range" => {
                let a = args[0].integer()?;
                let b = args[1].integer()?;
                let n = b.saturating_sub(a);
                self.reserve(
                    n.checked_mul(128)
                        .and_then(|n| n.checked_add(128))
                        .context("range size overflow")?,
                )?;
                Ok(Value::List(
                    (a..b).map(|n| Value::Number(n as f64)).collect(),
                ))
            }
            "append" => {
                let mut args = args.into_iter();
                let Value::List(mut xs) = args.next().unwrap() else {
                    bail!("append requires list")
                };
                self.reserve(128)?;
                xs.push(args.next().unwrap());
                Ok(Value::List(xs))
            }
            "sqrt" => number(args[0].number()?.sqrt()),
            "abs" => number(args[0].number()?.abs()),
            "min" => number(args[0].number()?.min(args[1].number()?)),
            "max" => number(args[0].number()?.max(args[1].number()?)),
            "differentiate" | "differentiate_at" => {
                ensure!(
                    self.gradients.len() < 16,
                    "live gradient handle limit exceeded"
                );
                let budget = self
                    .memory
                    .saturating_sub(self.allocated + self.gradient_bytes);
                let snapshot = if name == "differentiate" {
                    ensure!(
                        matches!(args[0], Value::Adaptive(0)),
                        "expected current session"
                    );
                    self.reserve(size(&args[1])? + size(&args[2])?)?;
                    self.session.differentiate_custom(
                        &matrix(&args[1])?,
                        &matrix(&args[2])?,
                        self.remaining,
                        self.memory
                            .saturating_sub(self.allocated + self.gradient_bytes),
                    )?
                } else {
                    let Value::Gradient(id) = args[0] else {
                        bail!("expected gradient handle")
                    };
                    self.reserve(size(&args[1])?)?;
                    let base = self
                        .gradients
                        .get(&id)
                        .context("gradient handle released or committed")?;
                    self.session.differentiate_at_custom(
                        base,
                        &matrix(&args[1])?,
                        self.remaining,
                        budget.min(
                            self.memory
                                .saturating_sub(self.allocated + self.gradient_bytes),
                        ),
                    )?
                };
                self.gradient_bytes = self
                    .gradient_bytes
                    .checked_add(snapshot.bytes)
                    .context("gradient size overflow")?;
                self.next = self
                    .next
                    .checked_add(1)
                    .context("gradient identity overflow")?;
                self.gradients.insert(self.next, snapshot);
                Ok(Value::Gradient(self.next))
            }
            "gradient_snapshot" => {
                let Value::Gradient(id) = args[0] else {
                    bail!("expected gradient handle")
                };
                let s = self
                    .gradients
                    .get(&id)
                    .context("gradient handle released or committed")?;
                // Reserve for f64 Value lists and state before constructing any copies.
                let count = s
                    .parameters
                    .iter()
                    .chain(&s.gradients)
                    .try_fold(0usize, |a, x| {
                        a.checked_add(x.len()).context("snapshot size overflow")
                    })?;
                let bytes = count
                    .checked_mul(32)
                    .and_then(|n| n.checked_add(s.bytes.checked_mul(8)?))
                    .context("snapshot size overflow")?;
                self.reserve(bytes)?;
                let s = &self.gradients[&id];
                let rows = |xs: &[Vec<f32>]| {
                    Value::List(
                        xs.iter()
                            .map(|r| {
                                Value::List(r.iter().map(|n| Value::Number(*n as f64)).collect())
                            })
                            .collect(),
                    )
                };
                let mut fields = BTreeMap::new();
                fields.insert("loss".into(), Value::Number(s.loss as f64));
                fields.insert("parameters".into(), rows(&s.parameters));
                fields.insert("gradients".into(), rows(&s.gradients));
                fields.insert("state".into(), from_json(&s.state.value));
                fields.insert("steps".into(), Value::Number(s.state.steps as f64));
                fields.insert(
                    "names".into(),
                    Value::List(s.names.iter().cloned().map(Value::Text).collect()),
                );
                fields.insert(
                    "shapes".into(),
                    Value::List(
                        s.shapes
                            .iter()
                            .map(|r| {
                                Value::List(r.iter().map(|n| Value::Number(*n as f64)).collect())
                            })
                            .collect(),
                    ),
                );
                Ok(Value::Object(fields))
            }
            "commit_gradients" | "release_gradient" => {
                let Value::Gradient(id) = args[0] else {
                    bail!("expected gradient handle")
                };
                if name == "commit_gradients" {
                    self.reserve(size(&args[1])? + size(&args[2])?)?;
                    let s = self
                        .gradients
                        .get(&id)
                        .context("gradient handle released or committed")?;
                    let loss = self
                        .session
                        .commit_custom(s, matrix(&args[1])?, json(&args[2])?)?;
                    self.commits += 1;
                    self.loss = Some(loss);
                }
                let s = self
                    .gradients
                    .remove(&id)
                    .context("gradient handle released or committed")?;
                self.gradient_bytes -= s.bytes;
                if name == "commit_gradients" {
                    number(self.loss.unwrap() as f64)
                } else {
                    Ok(Value::Null)
                }
            }
            _ => bail!("forbidden resident strategy operation"),
        }
    }
}
