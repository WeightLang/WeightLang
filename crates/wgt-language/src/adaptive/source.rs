//! Lower ordinary WGT 0.5 forward functions to bounded transactional tensor IR.
use super::{Instruction as I, Parameter, Program, Register};
use crate::script::parse::{Expr, Statement, Stmt};
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;

pub fn parse(source: &str) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    parse_with_functions(source, &functions)
}
pub fn parse_with_functions(
    source: &str,
    functions: &crate::script::parse::Program,
) -> Result<Program> {
    parse_definition(source, functions, true)
}
pub(crate) fn lower_operation(source: &str, functions: &crate::script::parse::Program, materialize: bool) -> Result<Program> {
    parse_definition(source, functions, materialize)
}
pub(crate) fn validate_definition(
    source: &str,
    functions: &crate::script::parse::Program,
) -> Result<Program> {
    parse_definition(source, functions, false)
}
fn parse_definition(
    source: &str,
    functions: &crate::script::parse::Program,
    materialize: bool,
) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let forward = functions
        .get("forward")
        .context("stateful model requires fn forward(input)")?;
    let mut program = Program {
        output_type: crate::categorical::VectorType::F32,
        version: 1,
        input: String::new(),
        registers: Vec::new(),
        parameters: Vec::new(),
        instructions: Vec::new(),
        loss: None,
    };
    let (mut model, mut output) = (false, None);
    let mut target = None;
    let mut instances = std::collections::BTreeMap::new();
    for text in unit.declarations.lines() {
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() || text == "wgt 0.5" {
            continue;
        }
        let (kw, body) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        match kw {
            "model" => {
                ensure!(!model && !body.trim().is_empty(), "duplicate/empty model");
                model = true;
            }
            "input" => {
                ensure!(
                    program.input.is_empty(),
                    "only one native forward input is supported"
                );
                let (name, len) = crate::source::vector_port(body)?;
                program.input = name.clone();
                program.registers.push(Register {
                    name,
                    len,
                    shape: vec![len],
                });
            }
            "target" => {
                ensure!(target.is_none(), "duplicate loss target declaration");
                target = Some(crate::source::vector_port(body)?);
            }
            "output" => {
                ensure!(output.is_none(), "duplicate output");
                let (_, n, kind) = crate::source::typed_vector_port(body)?;
                output = Some(n);
                program.output_type = kind;
            }
            "learn" => {
                let (name, declaration) = body
                    .split_once(':')
                    .context("learn requires name: type = initializer;")?;
                let declaration = declaration
                    .trim()
                    .strip_suffix(';')
                    .context("learn declaration requires trailing semicolon")?;
                let (domain, initial) = declaration
                    .rsplit_once('=')
                    .context("learn requires explicit initializer")?;
                let domain = domain.trim();
                let (domain, count) = if let Some(start) = domain.find('[') {
                    let end = domain[start..]
                        .find(']')
                        .context("unterminated learn array shape")?
                        + start;
                    ensure!(
                        ["float", "bool", "int"].contains(&domain[..start].trim()),
                        "learn array shape must follow its type, before bounds"
                    );
                    let count: usize = domain[start + 1..end]
                        .parse()
                        .context("learn array requires fixed width")?;
                    ensure!(
                        (1..=16384).contains(&count),
                        "learn array width must be 1..16384"
                    );
                    (format!("{}{}", &domain[..start], &domain[end + 1..]), count)
                } else {
                    (domain.to_owned(), 1)
                };
                let initial_values = if initial.trim().starts_with('[') {
                    let value = initial
                        .trim()
                        .strip_prefix('[')
                        .unwrap()
                        .strip_suffix(']')
                        .context("invalid learn array initializer")?;
                    let values = value.split(',').map(str::trim).collect::<Vec<_>>();
                    ensure!(
                        values.len() == count,
                        "learn array initializer length mismatch"
                    );
                    values
                } else {
                    vec![initial.trim(); count]
                };
                let domain = domain.as_str();
                let mut data = Vec::new();
                let mut learned_domain = None;
                for initial in initial_values {
                    use super::LearnDomain as D;
                    let (domain, value) = if domain == "bool" {
                        (
                            D::Bool,
                            match initial.trim() {
                                "true" => 1.,
                                "false" => 0.,
                                _ => bail!("learn bool initializer must be true or false"),
                            },
                        )
                    } else {
                        let value: f32 = initial
                            .trim()
                            .parse()
                            .context("learn initializer must be a finite scalar")?;
                        let kind = if domain == "float" {
                            D::Float
                        } else if let Some(low) = domain.strip_prefix("float >") {
                            D::Positive {
                                low: low
                                    .trim()
                                    .parse()
                                    .context("invalid learn strict lower bound")?,
                            }
                        } else {
                            let (kind, bounds) = domain
                                .split_once(" in ")
                                .context("unsupported learn type; choice is not supported")?;
                            let (low, high) = bounds
                                .split_once("..=")
                                .context("learn bounds require inclusive low..=high")?;
                            let low64: f64 =
                                low.trim().parse().context("invalid learn lower bound")?;
                            let high64: f64 =
                                high.trim().parse().context("invalid learn upper bound")?;
                            if kind.trim() == "int" {
                                ensure!(
                                    low64.fract() == 0.
                                        && high64.fract() == 0.
                                        && low64.abs() <= 16777216.
                                        && high64.abs() <= 16777216.,
                                    "learn int bounds must be exact integers within +/-16777216"
                                );
                            }
                            let low = low64 as f32;
                            let high = high64 as f32;
                            match kind.trim() {
                                "float" => D::Bounded { low, high },
                                "int" => D::Int { low, high },
                                _ => bail!("unsupported learn type; choice is not supported"),
                            }
                        };
                        if matches!(kind, D::Int { .. }) {
                            let exact: f64 = initial
                                .trim()
                                .parse()
                                .context("invalid learn int initializer")?;
                            ensure!(
                                exact.is_finite()
                                    && exact.fract() == 0.
                                    && f64::from(value) == exact,
                                "learn int initializer must be an exact f32 integer"
                            );
                        }
                        (kind, value)
                    };
                    domain
                        .validate(value)
                        .with_context(|| format!("learn {}", name.trim()))?;
                    data.push(value);
                    learned_domain = Some(domain);
                }
                program.parameters.push(Parameter {
                    name: name.trim().into(),
                    shape: vec![count],
                    data: if materialize { data } else { vec![] },
                    frozen: false,
                    probability: false,
                    learn: learned_domain,
                });
            }
            "parameter" | "frozen" => {
                let (name, body) = body
                    .split_once(':')
                    .context("parameter requires name: f32[shape] = values")?;
                let (shape, values) = body
                    .split_once('=')
                    .context("parameter requires explicit values")?;
                let probability = shape.trim() == "prob";
                let (shape, data) = if probability {
                    let p: f64 = values
                        .trim()
                        .parse()
                        .context("prob initializer must be a scalar probability")?;
                    ensure!(
                        p.is_finite() && (0. ..=1.).contains(&p),
                        "prob initializer must be in [0,1]"
                    );
                    let value = if kw == "frozen" {
                        p
                    } else {
                        ensure!(
                            p > 0. && p < 1.,
                            "trainable prob initializer must be strictly between 0 and 1; use frozen for exact endpoints"
                        );
                        p.ln() - (-p).ln_1p()
                    };
                    (
                        vec![1],
                        if materialize {
                            vec![value as f32]
                        } else {
                            Vec::new()
                        },
                    )
                } else {
                    let shape = shape
                        .trim()
                        .strip_prefix("f32")
                        .context("parameter type must be f32 or prob")?;
                    let shape: Vec<usize> = serde_json::from_str(shape)?;
                    let count = crate::model::validate_shape(&shape)?;
                    let data =
                        crate::model::validate_initializer(values.trim(), count, materialize)?;
                    (shape, data)
                };
                program.parameters.push(Parameter {
                    name: name.trim().into(),
                    shape,
                    data,
                    frozen: kw == "frozen",
                    probability,
                    learn: None,
                });
            }
            _ => {
                let declaration = text
                    .strip_suffix(';')
                    .context("instance declaration requires semicolon")?;
                let (name, kind) = declaration
                    .split_once(':')
                    .context("unsupported model declaration")?;
                let name = name.trim();
                let kind = kind.trim();
                let (kind, count) = if let Some((kind, n)) = kind.split_once('[') {
                    (
                        kind.trim(),
                        n.strip_suffix(']')
                            .context("invalid instance array")?
                            .parse::<usize>()?,
                    )
                } else {
                    (kind, 1)
                };
                ensure!(
                    crate::script::operators::identifier(name) && (1..=256).contains(&count),
                    "invalid instance name or count"
                );
                let source = unit
                    .models
                    .iter()
                    .find(|(n, _)| n == kind)
                    .context("unknown submodel definition")?
                    .1
                    .clone();
                ensure!(
                    instances.insert(name.to_owned(), (source, count)).is_none(),
                    "duplicate instance declaration"
                );
            }
        }
    }
    ensure!(
        model && !program.input.is_empty() && output.is_some(),
        "forward model requires model, input and output declarations"
    );
    ensure!(
        forward.args == [program.input.clone()],
        "forward argument must match the declared input name"
    );
    let sizes = program
        .registers
        .iter()
        .map(|r| (r.name.clone(), r.len))
        .collect();
    // Every definition is validated, including unused definitions; no service state
    // or parameter values are created during this schema-only pass.
    for (_, source) in &unit.models {
        let child = crate::source::split(source)?;
        ensure!(
            child.models.is_empty(),
            "nested submodel definitions are unsupported"
        );
        let mut child_functions = crate::script::parse::parse_functions(&child.functions)?;
        for (name, f) in functions {
            if name != "forward" && name != "loss" {
                child_functions
                    .entry(name.clone())
                    .or_insert_with(|| f.clone());
            }
        }
        parse_definition(source, &child_functions, false).context("invalid submodel definition")?;
    }
    let mut modules = HashMap::new();
    for (name, (source, count)) in instances {
        let child_unit = crate::source::split(&source)?;
        ensure!(
            child_unit.models.is_empty(),
            "nested submodel definitions are unsupported"
        );
        let mut child_functions = crate::script::parse::parse_functions(&child_unit.functions)?;
        for (name, f) in functions {
            if name != "forward" && name != "loss" {
                child_functions
                    .entry(name.clone())
                    .or_insert_with(|| f.clone());
            }
        }
        let plan = parse_definition(&source, &child_functions, materialize)?;
        ensure!(plan.loss.is_none(), "submodel loss belongs to root model");
        for i in 0..count {
            let key = format!("{name}[{i}]");
            let mut mapping = HashMap::new();
            for p in &plan.parameters {
                let mut p = p.clone();
                let original = p.name.clone();
                p.name = format!("{name}_{i}_{}", p.name);
                ensure!(
                    !program.parameters.iter().any(|q| q.name == p.name),
                    "submodel parameter namespace collision"
                );
                mapping.insert(original, p.name.clone());
                program.parameters.push(p);
            }
            let input = plan
                .registers
                .iter()
                .find(|r| r.name == plan.input)
                .context("missing submodel input")?
                .len;
            let output = child_unit
                .declarations
                .lines()
                .find_map(|s| s.trim().strip_prefix("output "))
                .context("missing submodel output")?;
            let output = crate::source::typed_vector_port(output)?.1;
            modules.insert(
                key,
                ModuleInstance {
                    functions: child_functions.clone(),
                    names: mapping,
                    input,
                    output,
                },
            );
        }
    }
    let mut lower = Lower {
        modules,
        program,
        sizes,
        types: HashMap::new(),
        memories: HashMap::new(),
        next: 0,
        loops: Vec::new(),
        output: output.unwrap(),
        functions: functions.clone(),
        calls: 0,
        call_depth: 0,
    };
    lower.block(&forward.body)?;

    ensure!(
        lower
            .program
            .instructions
            .iter()
            .any(|i| matches!(i, I::Return { .. })),
        "forward requires a return"
    );
    ensure!(
        target.is_none() || functions.contains_key("loss"),
        "target requires fn loss"
    );
    if let Some(loss) = functions.get("loss") {
        ensure!(
            loss.args.len() == 2,
            "loss requires prediction and target arguments"
        );
        let n = lower.output;
        let m = if let Some((name, width)) = &target {
            ensure!(
                *name == loss.args[1],
                "target name must match second loss argument"
            );
            *width
        } else {
            n
        };
        let input = "__loss_input".to_string();
        let program = Program {
            output_type: crate::categorical::VectorType::F32,
            version: 1,
            input: input.clone(),
            registers: vec![Register {
                name: input.clone(),
                len: n + m,
                shape: vec![n + m],
            }],
            parameters: vec![],
            instructions: vec![],
            loss: None,
        };
        let mut loss_lower = Lower {
            program,
            sizes: HashMap::from([(input.clone(), n + m)]),
            types: HashMap::from([(loss.args[0].clone(), lower.program.output_type)]),
            memories: HashMap::new(),
            next: 0,
            loops: vec![],
            output: 1,
            functions: functions.clone(),
            modules: HashMap::new(),
            calls: 0,
            call_depth: 0,
        };
        for (i, name) in loss.args.iter().enumerate() {
            loss_lower.register(name, if i == 0 { n } else { m })?;
            loss_lower.program.instructions.push(I::Slice {
                out: name.clone(),
                input: input.clone(),
                start: i * n,
            });
        }
        loss_lower.block(&loss.body).context("fn loss")?;
        ensure!(
            loss_lower
                .program
                .instructions
                .iter()
                .any(|i| matches!(i, I::Return { .. })),
            "loss requires a scalar return"
        );
        if !lower.program.output_type.is_plain() {
            loss_lower.program.version = loss_lower.program.version.max(2);
        }
        lower.program.loss = Some(Box::new(loss_lower.program));
    }
    if !lower.program.output_type.is_plain() {
        lower.program.version = lower.program.version.max(2);
    }
    Ok(lower.program)
}
// Endpoints are compile-time scalar arithmetic, never tensor values or host calls.
fn slice_endpoint(expr: &Expr) -> Result<f64> {
    let value = match expr {
        Expr::Number(n) => *n,
        Expr::Unary(op, x) if op == "-" => -slice_endpoint(x)?,
        Expr::Binary(op, a, b) => {
            let a = slice_endpoint(a)?;
            let b = slice_endpoint(b)?;
            match op.as_str() {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                "%" => a % b,
                _ => bail!("tensor slice endpoints must be compile-time numeric expressions"),
            }
        }
        _ => bail!("tensor slice endpoints must be compile-time numeric expressions"),
    };
    ensure!(value.is_finite(), "tensor slice endpoint must be finite");
    Ok(value)
}

fn scalar_literal(expr: &Expr) -> Result<f32> {
    let n = match expr {
        Expr::Number(n) => *n,
        Expr::Unary(op, x) if op == "-" => {
            let Expr::Number(n) = **x else {
                bail!("expected signed numeric literal")
            };
            -n
        }
        _ => bail!("expected signed numeric literal"),
    };
    ensure!((n as f32).is_finite(), "nonfinite numeric literal");
    Ok(n as f32)
}
#[derive(Clone)]
struct ModuleInstance {
    functions: crate::script::parse::Program,
    names: HashMap<String, String>,
    input: usize,
    output: usize,
}
struct Lower {
    program: Program,
    sizes: HashMap<String, usize>,
    types: HashMap<String, crate::categorical::VectorType>,
    memories: HashMap<String, (usize, usize)>,
    next: usize,
    loops: Vec<(String, String)>,
    output: usize,
    functions: crate::script::parse::Program,
    modules: HashMap<String, ModuleInstance>,
    calls: usize,
    call_depth: usize,
}
impl Lower {
    fn operator(&mut self, out: &str, name: &str, args: &[Expr]) -> Result<()> {
        ensure!(self.call_depth < 32, "operator expansion depth exceeded");
        self.call_depth += 1;
        let result = self.expand_operator(out, name, args);
        self.call_depth -= 1;
        result
    }
    fn expand_operator(&mut self, out: &str, name: &str, args: &[Expr]) -> Result<()> {
        use crate::script::operators::{rewrite, validate};
        self.calls += 1;
        ensure!(self.calls <= 1024, "operator expansion limit exceeded");
        validate(name, &self.functions, &mut Vec::new())?;
        let f = self.functions[name].clone();
        ensure!(
            args.len() == f.args.len(),
            "operator arity mismatch: {name}"
        );
        let mut names = HashMap::new();
        // Evaluate each argument exactly once, in caller order, before binding locals.
        for (arg, value) in f.args.iter().zip(args) {
            let input = self.value(value)?;
            names.insert(arg.clone(), input);
        }
        for s in f.body {
            match s.kind {
                Stmt::Let(n, e) => {
                    let e = rewrite(&e, &names, &HashMap::new());
                    let local = self.fresh();
                    self.assign(&local, &e)
                        .with_context(|| format!("operator {name}, line {}", s.line))?;
                    names.insert(n, local);
                }
                Stmt::Return(e) => {
                    return self
                        .assign(out, &rewrite(&e, &names, &HashMap::new()))
                        .with_context(|| format!("operator {name}, line {}", s.line));
                }
                _ => unreachable!(),
            }
        }
        bail!("operator requires return")
    }
    fn fresh(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("__wgt_{n}")
    }
    fn register(&mut self, name: &str, len: usize) -> Result<()> {
        ensure!(name != self.program.input, "cannot overwrite model input");
        if let Some(old) = self.sizes.get(name) {
            ensure!(*old == len, "register shape changes: {name}");
        } else {
            self.sizes.insert(name.into(), len);
            self.program.registers.push(Register {
                name: name.into(),
                len,
                shape: vec![len],
            });
        }
        Ok(())
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .program
            .registers
            .iter()
            .find(|r| r.name == name)
            .context("undefined tensor shape")?
            .shape
            .clone())
    }
    fn register_shape(&mut self, name: &str, shape: Vec<usize>) -> Result<()> {
        let len = crate::model::validate_shape(&shape)?;
        if shape.len() > 1 {
            self.program.version = 3;
        }
        if self.sizes.contains_key(name) {
            ensure!(
                self.shape(name)? == shape,
                "register geometry changes: {name}"
            );
        } else {
            self.register(name, len)?;
            self.program.registers.last_mut().unwrap().shape = shape;
        }
        Ok(())
    }
    fn size(&self, name: &str) -> Result<usize> {
        self.sizes
            .get(name)
            .copied()
            .with_context(|| format!("undefined tensor: {name}"))
    }
    fn parameter(&self, expr: &Expr) -> Result<&Parameter> {
        let Expr::Name(name) = expr else {
            bail!("expected parameter name")
        };
        self.program
            .parameters
            .iter()
            .find(|p| &p.name == name)
            .with_context(|| format!("unknown parameter: {name}"))
    }
    fn value(&mut self, expr: &Expr) -> Result<String> {
        if let Expr::Name(name) = expr {
            ensure!(
                !self.memories.contains_key(name),
                "memory handle is not a tensor"
            );
        }
        if let Expr::Name(name) = expr {
            if self.sizes.contains_key(name) {
                return Ok(name.clone());
            }
        }
        let out = self.fresh();
        self.assign(&out, expr)?;
        Ok(out)
    }
    fn assign(&mut self, out: &str, expr: &Expr) -> Result<()> {
        use crate::categorical::VectorType;
        if let Expr::BuiltinCall(name, args) = expr {
            if name == "__probabilities" {
                let [value] = args.as_slice() else {
                    bail!(".probabilities expects one value")
                };
                let input = self.value(value)?;
                let kind = self.types.get(&input).copied().unwrap_or_default();
                ensure!(
                    self.shape(&input)?.len() == 1,
                    ".probabilities requires a rank-1 f32, logits or probabilities tensor"
                );
                self.register(out, self.size(&input)?)?;
                self.program
                    .instructions
                    .push(if kind != VectorType::Probabilities {
                        I::Softmax {
                            out: out.into(),
                            input,
                        }
                    } else {
                        I::ValidateProbabilities {
                            out: out.into(),
                            input,
                        }
                    });
                self.types.insert(out.into(), VectorType::Probabilities);
                return Ok(());
            }
            return self.assign(out, &Expr::Call(name.clone(), args.clone()));
        }
        ensure!(
            self.program.instructions.len() < 100_000 && self.program.registers.len() < 100_000,
            "tensor program expansion limit exceeded"
        );
        if let Expr::Call(name, args) = expr {
            if self.functions.contains_key(name) && !crate::script::operators::intrinsic(name) {
                return self.operator(out, name, args);
            }
        }
        if let Expr::Call(op, args) = expr {
            if op == "memory_alloc" {
                let [Expr::Number(slots), Expr::Number(width)] = args.as_slice() else {
                    bail!("memory_alloc requires two constant dimensions")
                };
                ensure!(
                    *slots >= 1.
                        && *slots <= 16384.
                        && slots.fract() == 0.
                        && *width >= 1.
                        && *width <= 16384.
                        && width.fract() == 0.,
                    "memory dimensions must be integers in 1..16384"
                );
                let shape = (*slots as usize, *width as usize);
                ensure!(
                    !self.sizes.contains_key(out) || self.memories.get(out) == Some(&shape),
                    "memory handle conflicts with tensor or shape"
                );
                ensure!(
                    !self.program.parameters.iter().any(|p| p.name == out),
                    "memory handle conflicts with parameter"
                );
                self.register(out, 1)?;
                self.memories.insert(out.into(), shape);
                self.program.instructions.push(I::MemoryAlloc {
                    out: out.into(),
                    slots: shape.0,
                    width: shape.1,
                });
                return Ok(());
            }
        }
        ensure!(
            !self.memories.contains_key(out),
            "cannot overwrite memory handle with tensor"
        );
        if let Expr::Name(name) = expr {
            ensure!(
                !self.memories.contains_key(name),
                "memory handle cannot be copied"
            );
        }
        let invocation = match expr {
            Expr::Invoke(target, args) => Some((&**target, args)),
            Expr::Call(name, args) if self.modules.contains_key(&format!("{name}[0]")) => {
                Some((expr, args))
            }
            _ => None,
        };
        if let Some((target, args)) = invocation {
            let (name, index) = match target {
                Expr::Index(x, i) => {
                    let Expr::Name(n) = &**x else {
                        bail!("instance index requires a declared array")
                    };
                    let i = self.static_index(i)?;
                    (n.clone(), i)
                }
                Expr::Call(n, _) | Expr::Name(n) => (n.clone(), 0),
                _ => bail!("model call requires a declared instance"),
            };
            let module = self
                .modules
                .get(&format!("{name}[{index}]"))
                .context("unknown instance or array index out of bounds")?
                .clone();
            let forward = module
                .functions
                .get("forward")
                .context("submodel requires forward")?
                .clone();
            ensure!(
                args.len() == 1 && forward.args.len() == 1,
                "submodel requires one input"
            );
            let input = self.value(&args[0])?;
            ensure!(
                self.shape(&input)? == vec![module.input],
                "submodel input contract mismatch"
            );
            let mut names = module.names.clone();
            names.insert(forward.args[0].clone(), input);
            let previous = std::mem::replace(&mut self.functions, module.functions);
            let result = (|| {
                for statement in forward.body {
                    match statement.kind {
                        Stmt::Let(n, e) => {
                            let e = crate::script::operators::rewrite(&e, &names, &HashMap::new());
                            let local = self.fresh();
                            self.assign(&local, &e)?;
                            names.insert(n, local);
                        }
                        Stmt::Return(e) => {
                            return self.assign(
                                out,
                                &crate::script::operators::rewrite(&e, &names, &HashMap::new()),
                            );
                        }
                        _ => bail!("submodel forward supports immutable let and final return"),
                    }
                }
                bail!("submodel requires return")
            })();
            self.functions = previous;
            result?;
            ensure!(
                self.shape(out)? == vec![module.output],
                "submodel output contract mismatch"
            );
            return Ok(());
        }
        if let Expr::Number(n) = expr {
            let value = *n as f32;
            ensure!(value.is_finite(), "scalar literal must be finite f32");
            self.register(out, 1)?;
            self.program.instructions.push(I::Constant {
                out: out.into(),
                data: vec![value],
            });
            return Ok(());
        }
        if let Expr::Index(x, index) = expr {
            let index = self.static_index(index)?;
            let x = self.value(x)?;
            ensure!(index < self.size(&x)?, "tensor index out of bounds");
            self.register(out, 1)?;
            self.program.instructions.push(I::Slice {
                out: out.into(),
                input: x,
                start: index,
            });
            return Ok(());
        }
        if let Expr::Binary(op, a, b) = expr {
            if op == "**" {
                ensure!(
                    matches!(**b, Expr::Number(2.)),
                    "tensor exponent supports only **2"
                );
                return self.assign(out, &Expr::Call("square".into(), vec![(**a).clone()]));
            }
            if op == "*" {
                if let Ok(factor) = scalar_literal(a) {
                    return self.assign(
                        out,
                        &Expr::Call(
                            "scale".into(),
                            vec![(**b).clone(), Expr::Number(f64::from(factor))],
                        ),
                    );
                }
                if let Ok(factor) = scalar_literal(b) {
                    return self.assign(
                        out,
                        &Expr::Call(
                            "scale".into(),
                            vec![(**a).clone(), Expr::Number(f64::from(factor))],
                        ),
                    );
                }
            }
        }
        if let Expr::Blend(arms) = expr {
            // Complementary weights share one evaluated scalar gate. A bare name
            // is side-effect free; general expressions retain separate arm evaluation.
            if let [(Expr::Name(p), yes), (Expr::Binary(op, one, other), no)] = arms.as_slice() {
                if op == "-"
                    && matches!(&**one, Expr::Number(1.))
                    && matches!(&**other,Expr::Name(n) if n==p)
                {
                    return self.assign(
                        out,
                        &Expr::Call(
                            "route".into(),
                            vec![Expr::Name(p.clone()), yes.clone(), no.clone()],
                        ),
                    );
                }
            }
            let mut sum = None;
            for (p, x) in arms {
                let term = self.fresh();
                self.assign(
                    &term,
                    &Expr::Call("gate".into(), vec![p.clone(), x.clone()]),
                )?;
                sum = Some(if let Some(previous) = sum {
                    let next = self.fresh();
                    self.assign(
                        &next,
                        &Expr::Call("add".into(), vec![Expr::Name(previous), Expr::Name(term)]),
                    )?;
                    next
                } else {
                    term
                });
            }
            return self.assign(out, &Expr::Name(sum.context("empty blend")?));
        }
        if let Expr::Conditional(condition, yes, no) = expr {
            if let Some(p) = self.learn_condition(condition)? {
                return self.assign(
                    out,
                    &Expr::Call("route".into(), vec![p, (**yes).clone(), (**no).clone()]),
                );
            }
            let (a, b, end) = (self.fresh(), self.fresh(), self.fresh());
            self.condition(condition, a.clone(), b.clone())?;
            self.program.instructions.push(I::Label { name: a });
            self.assign(out, yes)?;
            self.program.instructions.push(I::Jump {
                target: end.clone(),
            });
            self.program.instructions.push(I::Label { name: b });
            self.assign(out, no)?;
            self.program.instructions.push(I::Label { name: end });
            return Ok(());
        }
        if let Expr::Repeat {
            depth,
            index,
            carrying,
            initial,
            body,
        } = expr
        {
            ensure!(
                index.as_ref() != Some(carrying),
                "repeat index and carrying names must differ"
            );
            let (maximum, learned) = if let Some(p) = self.learn_parameter(depth) {
                let Some(super::LearnDomain::Int { low, high }) = p.learn else {
                    bail!("repeat requires bounded learn int or constant count")
                };
                ensure!(
                    low >= 0. && high <= 256.,
                    "repeat depth range must be within 0..=256"
                );
                (high as usize, true)
            } else {
                let n = slice_endpoint(depth)?;
                ensure!(
                    n >= 0. && n <= 256. && n.fract() == 0.,
                    "repeat count must be an integer in 0..=256"
                );
                (n as usize, false)
            };
            let mut state = self.value(initial)?;
            let carrying_shape = self.shape(&state)?;
            for i in 0..maximum {
                let mut names = HashMap::new();
                names.insert(carrying.clone(), state.clone());
                if let Some(index) = index {
                    let binding = self.fresh();
                    self.assign(&binding, &Expr::Number(i as f64))?;
                    names.insert(index.clone(), binding);
                }
                let candidate = crate::script::operators::rewrite(body, &names, &HashMap::new());
                let next = self.fresh();
                if learned {
                    let gate = self.threshold_gate(depth, (i + 1) as f32)?;
                    self.assign(
                        &next,
                        &Expr::Call("route".into(), vec![gate, candidate, Expr::Name(state)]),
                    )?;
                } else {
                    self.assign(&next, &candidate)?;
                }
                ensure!(
                    self.shape(&next)? == carrying_shape,
                    "repeat body must preserve carrying shape"
                );
                state = next;
            }
            return self.assign(out, &Expr::Name(state));
        }
        if let Expr::Mix(p, a, b) = expr {
            return self.assign(
                out,
                &Expr::Call(
                    "route".into(),
                    vec![(**p).clone(), (**a).clone(), (**b).clone()],
                ),
            );
        }
        // Bounded shared-depth unrolling: hard 0/1 forward gate for iteration
        // index, identity surrogate dgate/ddepth=1 during training. All bounded
        // candidate steps execute on tape; inference route can short circuit.
        if let Expr::Call(op, args) = expr {
            if op == "depth_gate" {
                let [depth, index] = args.as_slice() else {
                    bail!("depth_gate requires learn int and literal iteration index")
                };
                let parameter = self.parameter(depth)?;
                let Some(super::LearnDomain::Int { low, high }) = &parameter.learn else {
                    bail!("depth_gate requires learn int")
                };
                let index = scalar_literal(index)?;
                ensure!(
                    *low >= 0.
                        && *high <= 256.
                        && index.fract() == 0.
                        && index >= 1.
                        && index <= *high,
                    "depth_gate requires nonnegative depth bounded by 256 and index in 1..=maximum"
                );
                let hard = Expr::Call(
                    "detach".into(),
                    vec![Expr::Call(
                        "clip".into(),
                        vec![
                            Expr::Binary(
                                "-".into(),
                                Box::new(depth.clone()),
                                Box::new(Expr::List(vec![Expr::Number(f64::from(index - 1.))])),
                            ),
                            Expr::Number(0.),
                            Expr::Number(1.),
                        ],
                    )],
                );
                let surrogate = Expr::Binary(
                    "-".into(),
                    Box::new(depth.clone()),
                    Box::new(Expr::Call("detach".into(), vec![depth.clone()])),
                );
                return self.assign(
                    out,
                    &Expr::Binary("+".into(), Box::new(hard), Box::new(surrogate)),
                );
            }
        }
        // Public reads expose probabilities; optimizer/checkpoint storage remains logits.
        let read_parameter = match expr {
            Expr::Name(n) if !self.sizes.contains_key(n) => Some(expr),
            Expr::Call(op, args) if op == "read" && args.len() == 1 => Some(&args[0]),
            _ => None,
        };
        if let Some(e) = read_parameter {
            let p = self.parameter(e)?;
            if p.probability && !p.frozen {
                let name = p.name.clone();
                let raw = self.fresh();
                self.register(&raw, 1)?;
                self.register(out, 1)?;
                self.program.instructions.push(I::Read {
                    out: raw.clone(),
                    parameter: name,
                });
                self.program.instructions.push(I::Sigmoid {
                    out: out.into(),
                    input: raw,
                });
                return Ok(());
            }
        }
        if let Expr::Call(op, args) = expr {
            let tensor = match (op.as_str(), args.as_slice()) {
                ("reshape", [x, Expr::List(dims)]) => {
                    let x = self.value(x)?;
                    let shape = dims
                        .iter()
                        .map(|e| {
                            let n = scalar_literal(e)?;
                            ensure!(
                                n >= 1. && n <= 16384. && n.fract() == 0.,
                                "reshape dimensions must be integers in 1..16384"
                            );
                            Ok(n as usize)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    ensure!(
                        crate::model::validate_shape(&shape)? == self.size(&x)?,
                        "reshape element count mismatch"
                    );
                    Some((
                        I::Reshape {
                            out: out.into(),
                            input: x,
                        },
                        shape,
                    ))
                }
                (
                    "conv",
                    [
                        x,
                        k,
                        b,
                        Expr::List(stride),
                        Expr::List(padding),
                        Expr::List(dilation),
                        groups,
                    ],
                ) => {
                    let integers = |items: &[Expr]| -> Result<Vec<usize>> {
                        items
                            .iter()
                            .map(|e| {
                                let n = scalar_literal(e)?;
                                ensure!(
                                    n >= 0. && n <= 16384. && n.fract() == 0.,
                                    "conv options require integer literals in 0..16384"
                                );
                                Ok(n as usize)
                            })
                            .collect()
                    };
                    let stride = integers(stride)?;
                    let padding = integers(padding)?;
                    let dilation = integers(dilation)?;
                    let g = scalar_literal(groups)?;
                    ensure!(
                        g >= 1. && g <= 16384. && g.fract() == 0.,
                        "conv groups must be a positive integer literal"
                    );
                    let x = self.value(x)?;
                    let k = self.value(k)?;
                    let b = self.value(b)?;
                    let geometry = super::ConvGeometry::new(
                        &self.shape(&x)?,
                        &self.shape(&k)?,
                        &self.shape(&b)?,
                        &stride,
                        &padding,
                        &dilation,
                        g as usize,
                    )?;
                    Some((
                        I::Conv {
                            out: out.into(),
                            input: x,
                            kernel: k,
                            bias: b,
                            stride,
                            padding,
                            dilation,
                            groups: g as usize,
                        },
                        geometry.shape(),
                    ))
                }
                ("transpose", [x]) => {
                    let x = self.value(x)?;
                    let shape = self.shape(&x)?;
                    ensure!(shape.len() == 2, "transpose requires rank 2");
                    Some((
                        I::Transpose {
                            out: out.into(),
                            input: x,
                        },
                        vec![shape[1], shape[0]],
                    ))
                }
                ("matmul", [a, b]) => {
                    let a = self.value(a)?;
                    let b = self.value(b)?;
                    let sa = self.shape(&a)?;
                    let sb = self.shape(&b)?;
                    ensure!(
                        sa.len() == 2 && sb.len() == 2 && sa[1] == sb[0],
                        "matmul requires [M,K] and [K,N]"
                    );
                    Some((
                        I::Matmul {
                            out: out.into(),
                            left: a,
                            right: b,
                        },
                        vec![sa[0], sb[1]],
                    ))
                }
                ("softmax", [x, axis]) => {
                    let x = self.value(x)?;
                    let shape = self.shape(&x)?;
                    let axis = scalar_literal(axis)?;
                    ensure!(
                        axis >= 0. && axis.fract() == 0. && axis < shape.len() as f32,
                        "softmax axis must be a nonnegative in-range integer"
                    );
                    Some((
                        I::SoftmaxAxis {
                            out: out.into(),
                            input: x,
                            axis: axis as usize,
                        },
                        shape,
                    ))
                }
                ("scale", [x, factor]) => {
                    let factor = scalar_literal(factor)?;
                    let x = self.value(x)?;
                    let shape = self.shape(&x)?;
                    Some((
                        I::Scale {
                            out: out.into(),
                            input: x,
                            factor,
                        },
                        shape,
                    ))
                }
                _ => None,
            };
            if let Some((instruction, shape)) = tensor {
                self.register_shape(out, shape)?;
                self.program.version = 3;
                self.types.insert(out.into(), VectorType::F32);
                self.program.instructions.push(instruction);
                return Ok(());
            }
        }
        let (instruction, len) = match expr {
            Expr::List(items) => {
                let data = items
                    .iter()
                    .map(scalar_literal)
                    .collect::<Result<Vec<_>>>()?;
                ensure!(
                    !data.is_empty() && data.len() <= 16384,
                    "tensor literal width must be 1..16384"
                );
                let len = data.len();
                (
                    I::Constant {
                        out: out.into(),
                        data,
                    },
                    len,
                )
            }
            Expr::Slice(x, start, end) => {
                let x = self.value(x)?;
                let (start, end) = crate::script::compiled::slice_bounds(
                    self.size(&x)?,
                    start.as_ref().map(|e| slice_endpoint(e)).transpose()?,
                    end.as_ref().map(|e| slice_endpoint(e)).transpose()?,
                )?;
                ensure!(
                    start < end,
                    "empty tensor slices are unsupported (tensor shapes must be nonzero)"
                );
                (
                    I::Slice {
                        out: out.into(),
                        input: x,
                        start,
                    },
                    end - start,
                )
            }
            Expr::Sample(..) | Expr::SampleFrom(..) => bail!(
                "sample expressions are discrete host expressions; use mix for differentiable forward/loss"
            ),
            Expr::Unary(op, x) if op == "-" => {
                let x = self.value(x)?;
                let len = self.size(&x)?;
                (
                    I::Neg {
                        out: out.into(),
                        input: x,
                    },
                    len,
                )
            }
            Expr::Name(name) if self.sizes.contains_key(name) => (
                if self.shape(name)?.len() > 1 {
                    I::Reshape {
                        out: out.into(),
                        input: name.clone(),
                    }
                } else {
                    I::Slice {
                        out: out.into(),
                        input: name.clone(),
                        start: 0,
                    }
                },
                self.size(name)?,
            ),
            Expr::Name(_) => {
                let p = self.parameter(expr)?;
                (
                    I::Read {
                        out: out.into(),
                        parameter: p.name.clone(),
                    },
                    p.shape.iter().product(),
                )
            }
            Expr::Binary(op, a, b) if op == "@" => {
                let a = self.value(a)?;
                let b = self.value(b)?;
                let sa = self.shape(&a)?;
                let sb = self.shape(&b)?;
                ensure!(
                    sa.len() == 2 && sb.len() == 2 && sa[1] == sb[0],
                    "matrix product requires [M,K] @ [K,N]"
                );
                let output_shape = vec![sa[0], sb[1]];
                let len = output_shape.iter().product();
                (
                    I::Matmul {
                        out: out.into(),
                        left: a,
                        right: b,
                    },
                    len,
                )
            }
            Expr::Binary(op, a, b) if op == "+" || op == "*" || op == "-" => {
                let a = self.value(a)?;
                let b = self.value(b)?;
                let len = self.size(&a)?;
                ensure!(
                    self.shape(&a)? == self.shape(&b)?,
                    "tensor arithmetic shape mismatch"
                );
                let b = if op == "-" {
                    let neg = self.fresh();
                    self.register_shape(&neg, self.shape(&a)?)?;
                    self.program.instructions.push(I::Neg {
                        out: neg.clone(),
                        input: b,
                    });
                    neg
                } else {
                    b
                };
                (
                    if op == "*" {
                        I::Mul {
                            out: out.into(),
                            left: a,
                            right: b,
                        }
                    } else {
                        I::Add {
                            out: out.into(),
                            left: a,
                            right: b,
                        }
                    },
                    len,
                )
            }
            Expr::Call(op, args) => match (op.as_str(), args.as_slice()) {
                ("detach", [x]) | ("exp", [x]) | ("log", [x]) | ("log_softmax", [x]) => {
                    let x = self.value(x)?;
                    let len = self.size(&x)?;
                    (
                        match op.as_str() {
                            "detach" => I::Detach {
                                out: out.into(),
                                input: x,
                            },
                            "exp" => I::Exp {
                                out: out.into(),
                                input: x,
                            },
                            "log" => I::Log {
                                out: out.into(),
                                input: x,
                            },
                            _ => I::LogSoftmax {
                                out: out.into(),
                                input: x,
                            },
                        },
                        len,
                    )
                }
                ("minimum", [a, b]) => {
                    let a = self.value(a)?;
                    let b = self.value(b)?;
                    let len = self.size(&a)?;
                    ensure!(self.shape(&a)? == self.shape(&b)?, "minimum shape mismatch");
                    (
                        I::Minimum {
                            out: out.into(),
                            left: a,
                            right: b,
                        },
                        len,
                    )
                }
                ("clip", [x, low, high]) => {
                    let low = scalar_literal(low)?;
                    let high = scalar_literal(high)?;
                    ensure!(low <= high, "invalid clip bounds");
                    let x = self.value(x)?;
                    let len = self.size(&x)?;
                    (
                        I::Clip {
                            out: out.into(),
                            input: x,
                            low,
                            high,
                        },
                        len,
                    )
                }
                ("sum", [x]) => {
                    let x = self.value(x)?;
                    (
                        I::Sum {
                            out: out.into(),
                            input: x,
                        },
                        1,
                    )
                }
                ("resource_state", []) => (I::ResourceState { out: out.into() }, 3),
                ("memory_read", [Expr::Name(memory), address]) => {
                    let (slots, width) =
                        *self.memories.get(memory).context("unknown memory handle")?;
                    let address = self.value(address)?;
                    ensure!(
                        self.size(&address)? == slots,
                        "memory address shape mismatch"
                    );
                    (
                        I::MemoryRead {
                            out: out.into(),
                            memory: memory.clone(),
                            address,
                        },
                        width,
                    )
                }
                ("read", [p]) => {
                    let p = self.parameter(p)?;
                    (
                        I::Read {
                            out: out.into(),
                            parameter: p.name.clone(),
                        },
                        p.shape.iter().product(),
                    )
                }
                ("linear", [x, w]) | ("linear", [x, w, _]) => {
                    let x = self.value(x)?;
                    let w = self.parameter(w)?;
                    ensure!(
                        !w.probability && w.learn.is_none(),
                        "linear weight cannot be prob/learn; read architecture values explicitly"
                    );
                    ensure!(
                        w.shape.len() == 2 && w.shape[1] == self.size(&x)?,
                        "linear weight/input shape mismatch"
                    );
                    let (weight, len) = (w.name.clone(), w.shape[0]);
                    let bias = if args.len() == 3 {
                        let b = self.parameter(&args[2])?;
                        ensure!(
                            !b.probability && b.learn.is_none(),
                            "linear bias cannot be prob/learn"
                        );
                        Some(b.name.clone())
                    } else {
                        None
                    };
                    (
                        I::Linear {
                            out: out.into(),
                            input: x,
                            weight,
                            bias,
                        },
                        len,
                    )
                }
                ("add", [a, b]) | ("mul", [a, b]) => {
                    let a = self.value(a)?;
                    let b = self.value(b)?;
                    let len = self.size(&a)?;
                    ensure!(
                        self.shape(&a)? == self.shape(&b)?,
                        "tensor arithmetic shape mismatch"
                    );
                    (
                        if op == "add" {
                            I::Add {
                                out: out.into(),
                                left: a,
                                right: b,
                            }
                        } else {
                            I::Mul {
                                out: out.into(),
                                left: a,
                                right: b,
                            }
                        },
                        len,
                    )
                }
                ("mean", [x]) | ("square", [x]) => {
                    let x = self.value(x)?;
                    let len = self.size(&x)?;
                    if op == "mean" {
                        (
                            I::Mean {
                                out: out.into(),
                                input: x,
                            },
                            1,
                        )
                    } else {
                        (
                            I::Mul {
                                out: out.into(),
                                left: x.clone(),
                                right: x,
                            },
                            len,
                        )
                    }
                }
                ("gate", [p, x]) | ("route", [p, x, _]) => {
                    let p = self.value(p)?;
                    let after_a = self.fresh();
                    self.program.instructions.push(I::InferenceSkip {
                        probability: p.clone(),
                        when_one: false,
                        target: after_a.clone(),
                    });
                    let x = self.value(x)?;
                    let len = self.size(&x)?;
                    self.program.instructions.push(I::Label { name: after_a });
                    ensure!(
                        self.shape(&p)? == vec![1] || self.shape(&p)? == self.shape(&x)?,
                        "gate/route probability shape mismatch"
                    );
                    let instruction = if op == "gate" {
                        I::Gate {
                            out: out.into(),
                            probability: p,
                            input: x,
                        }
                    } else {
                        let after_b = self.fresh();
                        self.program.instructions.push(I::InferenceSkip {
                            probability: p.clone(),
                            when_one: true,
                            target: after_b.clone(),
                        });
                        let b = self.value(&args[2])?;
                        self.program.instructions.push(I::Label { name: after_b });
                        ensure!(
                            self.shape(&b)? == self.shape(&x)?,
                            "route branch shape mismatch"
                        );
                        I::Route {
                            out: out.into(),
                            probability: p,
                            yes: x,
                            no: b,
                        }
                    };
                    (instruction, len)
                }
                ("relu", [x]) | ("tanh", [x]) | ("sigmoid", [x]) | ("softmax", [x]) => {
                    let x = self.value(x)?;
                    let len = self.size(&x)?;
                    (
                        match op.as_str() {
                            "relu" => I::Relu {
                                out: out.into(),
                                input: x,
                            },
                            "tanh" => I::Tanh {
                                out: out.into(),
                                input: x,
                            },
                            "softmax" => I::Softmax {
                                out: out.into(),
                                input: x,
                            },
                            _ => I::Sigmoid {
                                out: out.into(),
                                input: x,
                            },
                        },
                        len,
                    )
                }
                ("slice", [x, Expr::Number(start), Expr::Number(len)]) => {
                    ensure!(
                        *start >= 0. && start.fract() == 0. && *len > 0. && len.fract() == 0.,
                        "slice needs nonnegative start and positive length"
                    );
                    let x = self.value(x)?;
                    ensure!(
                        *start + *len <= self.size(&x)? as f64,
                        "slice out of bounds"
                    );
                    (
                        I::Slice {
                            out: out.into(),
                            input: x,
                            start: *start as usize,
                        },
                        *len as usize,
                    )
                }
                _ => bail!("unsupported tensor expression or arity: {op}"),
            },
            _ => bail!("forward expressions must produce explicitly shaped tensors"),
        };
        let kind = match &instruction {
            I::Slice {
                input, start: 0, ..
            } if self.size(input)? == len => self.types.get(input).copied().unwrap_or_default(),
            I::Detach { input, .. } => self.types.get(input).copied().unwrap_or_default(),
            I::Softmax { .. } => VectorType::Probabilities,
            _ => VectorType::F32,
        };
        let shape = match &instruction {
            I::Read { .. } => vec![len],
            I::Reshape { input, .. } => self.shape(input)?,
            I::Slice {
                input, start: 0, ..
            } if matches!(expr, Expr::Name(_)) => self.shape(input)?,
            I::Neg { input, .. }
            | I::Detach { input, .. }
            | I::Exp { input, .. }
            | I::Log { input, .. }
            | I::Tanh { input, .. }
            | I::Sigmoid { input, .. }
            | I::Relu { input, .. }
            | I::Clip { input, .. } => self.shape(input)?,
            I::Add { left, .. } | I::Mul { left, .. } | I::Minimum { left, .. } => {
                self.shape(left)?
            }
            I::Matmul { left, right, .. } => {
                let left = self.shape(left)?;
                let right = self.shape(right)?;
                ensure!(
                    left.len() == 2 && right.len() == 2 && left[1] == right[0],
                    "matrix product requires [M,K] @ [K,N]"
                );
                vec![left[0], right[1]]
            }
            I::Gate { input, .. } | I::Route { yes: input, .. } => self.shape(input)?,
            I::Softmax { input, .. } | I::LogSoftmax { input, .. } => {
                ensure!(
                    self.shape(input)?.len() == 1,
                    "rank > 1 softmax requires explicit axis"
                );
                vec![len]
            }
            I::Linear { input, .. } => {
                ensure!(
                    self.shape(input)?.len() == 1,
                    "linear requires vector input; use matmul for matrices"
                );
                vec![len]
            }
            I::Slice { input, .. } => {
                ensure!(self.shape(input)?.len() == 1, "slice requires vector input");
                vec![len]
            }
            _ => vec![len],
        };
        self.register_shape(out, shape)?;

        self.types.insert(out.into(), kind);
        self.program.instructions.push(instruction);
        Ok(())
    }
    fn static_index(&self, expr: &Expr) -> Result<usize> {
        let n = if let Expr::Name(name) = expr {
            self.program
                .instructions
                .iter()
                .rev()
                .find_map(|i| match i {
                    I::Constant { out, data } if out == name && data.len() == 1 => {
                        Some(f64::from(data[0]))
                    }
                    _ => None,
                })
                .context("index must be a compile-time constant or repeat index")?
        } else {
            slice_endpoint(expr)?
        };
        ensure!(
            n >= 0. && n.fract() == 0. && n <= 16384.,
            "index must be a bounded nonnegative integer"
        );
        Ok(n as usize)
    }
    fn learn_parameter(&self, expr: &Expr) -> Option<&Parameter> {
        let name = match expr {
            Expr::Name(n) => n,
            Expr::Index(x, _) => match &**x {
                Expr::Name(n) => n,
                _ => return None,
            },
            _ => return None,
        };
        self.program
            .parameters
            .iter()
            .find(|p| p.name == *name && p.learn.is_some())
    }
    fn threshold_gate(&mut self, depth: &Expr, threshold: f32) -> Result<Expr> {
        let value = self.value(depth)?;
        let d = Expr::Name(value);
        let difference = Expr::Binary(
            "-".into(),
            Box::new(d.clone()),
            Box::new(Expr::Number(f64::from(threshold))),
        );
        let hard = Expr::Call(
            "detach".into(),
            vec![Expr::Call(
                "clip".into(),
                vec![
                    Expr::Binary("+".into(), Box::new(difference), Box::new(Expr::Number(1.))),
                    Expr::Number(0.),
                    Expr::Number(1.),
                ],
            )],
        );
        Ok(Expr::Binary(
            "+".into(),
            Box::new(hard),
            Box::new(Expr::Binary(
                "-".into(),
                Box::new(d.clone()),
                Box::new(Expr::Call("detach".into(), vec![d])),
            )),
        ))
    }
    fn learn_condition(&mut self, condition: &Expr) -> Result<Option<Expr>> {
        if self
            .learn_parameter(condition)
            .is_some_and(|p| matches!(p.learn, Some(super::LearnDomain::Bool)))
        {
            return Ok(Some(condition.clone()));
        }
        if let Expr::Binary(op, d, t) = condition {
            if op == ">="
                && self
                    .learn_parameter(d)
                    .is_some_and(|p| matches!(p.learn, Some(super::LearnDomain::Int { .. })))
            {
                let threshold = scalar_literal(t)?;
                ensure!(
                    threshold.fract() == 0.,
                    "learn int threshold must be an integer literal"
                );
                return Ok(Some(self.threshold_gate(d, threshold)?));
            }
        }
        Ok(None)
    }
    fn condition(&mut self, expr: &Expr, yes: String, no: String) -> Result<()> {
        if let Expr::Bool(value) = expr {
            self.program.instructions.push(I::Jump {
                target: if *value { yes } else { no },
            });
            return Ok(());
        }
        if let Expr::Binary(op, a, b) = expr {
            if op == ">" {
                let value = Expr::Binary("-".into(), a.clone(), b.clone());
                return self.condition(&Expr::Call("positive".into(), vec![value]), yes, no);
            }
            if op == ">=" {
                let value = Expr::Binary("-".into(), b.clone(), a.clone());
                return self.condition(&Expr::Call("positive".into(), vec![value]), no, yes);
            }
            if op == "<" {
                return self.condition(&Expr::Binary(">".into(), b.clone(), a.clone()), yes, no);
            }
            if op == "<=" {
                return self.condition(&Expr::Binary(">=".into(), b.clone(), a.clone()), yes, no);
            }
            if op == "==" || op == "!=" {
                let next = self.fresh();
                let (equal, unequal) = if op == "==" { (yes, no) } else { (no, yes) };
                self.condition(
                    &Expr::Binary(">".into(), a.clone(), b.clone()),
                    unequal.clone(),
                    next.clone(),
                )?;
                self.program.instructions.push(I::Label { name: next });
                return self.condition(
                    &Expr::Binary("<".into(), a.clone(), b.clone()),
                    unequal,
                    equal,
                );
            }
        }
        let Expr::Call(op, args) = expr else {
            bail!(
                "tensor condition requires bool literal, learn bool, scalar comparison or positive(tensor)"
            )
        };
        ensure!(
            op == "positive" && args.len() == 1,
            "tensor condition must be positive(tensor)"
        );
        let condition = self.value(&args[0])?;
        ensure!(self.size(&condition)? == 1, "positive requires f32[1]");
        self.program.instructions.push(I::Branch {
            condition,
            positive: yes,
            otherwise: no,
        });
        Ok(())
    }
    fn block(&mut self, body: &[Statement]) -> Result<()> {
        for s in body {
            (|| -> Result<()> {
                match &s.kind {
                    Stmt::Let(n, e) | Stmt::Set(n, e) => {
                        ensure!(!n.starts_with("__wgt_"), "reserved compiler name");
                        if matches!(&s.kind, Stmt::Set(..)) {
                            ensure!(
                                self.sizes.contains_key(n),
                                "assignment to undeclared tensor"
                            );
                        }
                        self.assign(n, e)?;
                    }
                    Stmt::Eval(Expr::BuiltinCall(op, args)) if op == "__transaction_write" && args.len() == 2 => {
                        let parameter = self.parameter(&args[0])?.name.clone();
                        let value = self.value(&args[1])?;
                        self.program.instructions.push(I::Write { parameter, value });
                    }
                    Stmt::Eval(Expr::Call(op, args)) if op == "write" && args.len() == 2 => {
                        let parameter = self.parameter(&args[0])?.name.clone();
                        let value = self.value(&args[1])?;
                        self.program
                            .instructions
                            .push(I::Write { parameter, value });
                    }
                    Stmt::Eval(Expr::Call(op, args))
                        if op == "memory_free" || op == "memory_write" =>
                    {
                        let Some(Expr::Name(memory)) = args.first() else {
                            bail!("expected memory handle")
                        };
                        let (slots, width) =
                            *self.memories.get(memory).context("unknown memory handle")?;
                        if op == "memory_free" {
                            ensure!(args.len() == 1, "memory_free expects one argument");
                            self.program.instructions.push(I::MemoryFree {
                                memory: memory.clone(),
                            });
                        } else {
                            ensure!(
                                args.len() == 3,
                                "memory_write expects memory, address and value"
                            );
                            let address = self.value(&args[1])?;
                            let value = self.value(&args[2])?;
                            ensure!(
                                self.size(&address)? == slots && self.size(&value)? == width,
                                "memory_write shape mismatch"
                            );
                            self.program.instructions.push(I::MemoryWrite {
                                memory: memory.clone(),
                                address,
                                value,
                            });
                        }
                    }
                    Stmt::Return(e) => {
                        let value = self.value(e)?;
                        ensure!(
                            self.shape(&value)? == vec![self.output],
                            "forward return/output shape mismatch"
                        );
                        self.program.instructions.push(I::Return { value });
                    }
                    Stmt::If(e, yes, no) => {
                        let (a, b, end) = (self.fresh(), self.fresh(), self.fresh());
                        self.condition(e, a.clone(), b.clone())?;
                        self.program.instructions.push(I::Label { name: a });
                        self.block(yes)?;
                        self.program.instructions.push(I::Jump {
                            target: end.clone(),
                        });
                        self.program.instructions.push(I::Label { name: b });
                        self.block(no)?;
                        self.program.instructions.push(I::Label { name: end });
                    }
                    Stmt::While(e, body) => {
                        let (start, go, end) = (self.fresh(), self.fresh(), self.fresh());
                        self.program.instructions.push(I::Label {
                            name: start.clone(),
                        });
                        self.condition(e, go.clone(), end.clone())?;
                        self.program.instructions.push(I::Label { name: go });
                        self.loops.push((start.clone(), end.clone()));
                        self.block(body)?;
                        self.loops.pop();
                        self.program.instructions.push(I::Jump { target: start });
                        self.program.instructions.push(I::Label { name: end });
                    }
                    Stmt::Break | Stmt::Continue => {
                        let (start, end) =
                            self.loops.last().context("loop control outside loop")?;
                        self.program.instructions.push(I::Jump {
                            target: if matches!(&s.kind, Stmt::Break) {
                                end
                            } else {
                                start
                            }
                            .clone(),
                        });
                    }
                    _ => bail!("unsupported forward statement"),
                }
                Ok(())
            })()
            .with_context(|| format!("line {}", s.line))?;
        }
        Ok(())
    }
}
