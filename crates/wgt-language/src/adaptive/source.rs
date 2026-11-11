//! Lower ordinary WGT 0.5 forward functions to bounded transactional tensor IR.
use super::{Instruction as I, ModuleSchema, Parameter, Program, Register};
use crate::script::parse::{Expr, Statement, Stmt};
use crate::tensor::{DType, ShapeLimits, TensorType};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};

pub fn parse(source: &str) -> Result<Program> {
    let normalized = crate::model_syntax::normalize(source)?;
    let source = normalized.as_str();
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    parse_with_functions(source, &functions)
}
/// Lower with the same parser while retaining source-level module identities.
/// The metadata does not alter instructions, parameter ordering or initialization.
pub fn parse_with_metadata(source: &str) -> Result<(Program, Vec<ModuleSchema>)> {
    let normalized = crate::model_syntax::normalize(source)?;
    let source = normalized.as_str();
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    parse_definition_with_metadata(source, &functions, true)
}
pub(crate) fn compile_with_metadata(source: &str) -> Result<(Program, Vec<ModuleSchema>)> {
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    parse_definition_with_metadata(source, &functions, false)
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
    parse_definition_with_metadata(source, functions, materialize).map(|(program, _)| program)
}
fn parse_definition_with_metadata(
    source: &str,
    functions: &crate::script::parse::Program,
    materialize: bool,
) -> Result<(Program, Vec<ModuleSchema>)> {
    parse_definition_with_details(source, functions, materialize, false)
        .map(|(program, modules, _)| (program, modules))
}

/// Compile-time types for editor requests; never initialize parameter storage.
pub(crate) fn analyze_definition(
    source: &str,
    functions: &crate::script::parse::Program,
) -> Result<(Program, super::InferredTypes)> {
    parse_definition_with_details(source, functions, false, true)
        .map(|(program, _, types)| (program, types))
}

fn parse_definition_with_details(
    source: &str,
    functions: &crate::script::parse::Program,
    materialize: bool,
    collect_types: bool,
) -> Result<(Program, Vec<ModuleSchema>, super::InferredTypes)> {
    let unit = crate::source::split(source)?;
    let forward = functions
        .get("forward")
        .context("stateful model requires fn forward(input)")?;
    let mut program = Program {
        observations: Vec::new(),
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
                let (name, declaration) = body.split_once(':').context("learn requires name: type")?;
                let declaration = declaration.trim().strip_suffix(';').context("learn declaration requires trailing semicolon")?;
                let assignment = declaration.rfind('=').filter(|i| *i == 0 || declaration.as_bytes()[i - 1] != b'.');
                let (domain, initial) = assignment.map_or((declaration, None), |i| (&declaration[..i], Some(declaration[i + 1..].trim())));
                let domain = domain.trim();
                let (domain, count) = if let Some(start) = domain.find('[') {
                    let end = domain[start..].find(']').context("unterminated learn array shape")? + start;
                    ensure!(["float", "bool", "int"].contains(&domain[..start].trim()), "learn array shape must follow its type, before bounds");
                    let count: usize = domain[start + 1..end].parse().context("learn array requires fixed width")?;
                    ensure!((1..=16384).contains(&count), "learn array width must be 1..16384");
                    (format!("{}{}", &domain[..start], &domain[end + 1..]), count)
                } else { (domain.to_owned(), 1) };
                let domain = parse_learn_domain(domain.trim())?;
                let shape = vec![count];
                let initializer = if let Some(initial) = initial.filter(|v| *v != "external") {
                    if matches!(domain, super::LearnDomain::Bool) { ensure!(initial.parse::<f64>().is_err(), "learn bool initializer must be true or false"); }
                    let initial = match initial { "true" => "1".to_owned(), "false" => "0".to_owned(), _ => initial.to_owned() };
                    let scalar = initial.parse::<f64>().ok();
                    if let Some(value) = scalar {
                        ensure!(value.is_finite() && (value as f32).is_finite(), "learn initializer must be finite");
                        if matches!(domain, super::LearnDomain::Int { .. }) { ensure!(value.fract() == 0. && (value as f32) as f64 == value, "learn int initializer must be an exact f32 integer"); }
                        domain.validate(value as f32)?;
                        crate::initializer::Recipe::Full { shape: shape.clone(), value: value as f32 }
                    } else { crate::initializer::parse(Some(&initial), &shape, functions)? }
                } else { crate::initializer::Recipe::Unbound };
                // Explicit numeric literals are checked without running a recipe.
                if let crate::initializer::Recipe::Literal { values, .. } = &initializer {
                    for value in values { domain.validate(*value)?; }
                }
                let data = if materialize {
                    let values = initializer.materialize(&shape).with_context(|| format!("learn {}", name.trim()))?;
                    for value in &values { domain.validate(*value)?; }
                    values
                } else { Vec::new() };
                program.parameters.push(Parameter { name: name.trim().into(), shape, data, initializer: Some(initializer), frozen: false, probability: false, learn: Some(domain) });
            }
            "parameter" | "frozen" => {
                let (name, body) = body
                    .split_once(':')
                    .context("parameter requires name: f32[shape] = values")?;
                let (shape, values) = body.trim_end_matches(';').split_once('=')
                    .map_or((body.trim().trim_end_matches(';'), None), |(shape, values)| (shape, Some(values)));
                let probability = shape.trim() == "prob";
                let (shape, initializer) = if probability {
                    (vec![1], crate::initializer::probability(values, kw == "frozen", functions)?)
                } else {
                    let shape = shape
                        .trim()
                        .strip_prefix("f32")
                        .context("parameter type must be f32 or prob")?;
                    let shape: Vec<usize> = serde_json::from_str(shape)?;
                    crate::model::validate_shape(&shape)?;
                    let initializer = crate::initializer::parse(values, &shape, functions)?;
                    (shape, initializer)
                };
                let data = if materialize { initializer.materialize(&shape).with_context(|| format!("parameter {}", name.trim()))? } else { Vec::new() };
                program.parameters.push(Parameter {
                    name: name.trim().into(),
                    shape,
                    data,
                    initializer: Some(initializer),
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
                    instances.insert(name.to_owned(), (kind.to_owned(), source, count)).is_none(),
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
    let mut module_schemas = Vec::new();
    for (name, (model, source, count)) in instances {
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
            module_schemas.push(ModuleSchema {
                path: key.clone(),
                model: model.clone(),
                input,
                output,
                parameters: mapping.iter().map(|(local, storage)| (local.clone(), storage.clone())).collect(),
            });
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
        axes: HashMap::new(),
        memories: HashMap::new(),
        next: 0,
        loops: Vec::new(),
        output: output.unwrap(),
        return_shape: forward.tensor_signature.as_ref().map(|signature| signature.output.clone()),
        functions: functions.clone(),
        calls: 0,
        call_depth: 0,
        tuples: HashMap::new(),
        constants: HashMap::new(),
        unavailable: HashSet::new(),
        tuple_bindings: HashSet::new(),
        source_line: 0,
        scope: vec!["forward".into()],
        operation_stack: Vec::new(),
        operation_returns: Vec::new(),
    };
    lower.validate_shape_helpers()?;
    let forward_body = lower.matrix_arguments(forward)?;
    lower.block(&forward_body)?;
    let mut inferred = super::InferredTypes::new();
    if collect_types { inferred.insert("forward".into(), lower.inferred_types()?); }

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
            observations: Vec::new(),
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
            axes: HashMap::new(),
            memories: HashMap::new(),
            next: 0,
            loops: vec![],
            output: 1,
            return_shape: loss.tensor_signature.as_ref().map(|signature| signature.output.clone()),
            functions: functions.clone(),
            modules: HashMap::new(),
            calls: 0,
            call_depth: 0,
            tuples: HashMap::new(),
            constants: HashMap::new(),
            unavailable: HashSet::new(),
            tuple_bindings: HashSet::new(),
            source_line: 0,
            scope: vec!["loss".into()],
            operation_stack: Vec::new(),
            operation_returns: Vec::new(),
        };
        for (i, name) in loss.args.iter().enumerate() {
            loss_lower.register(name, if i == 0 { n } else { m })?;
            loss_lower.program.instructions.push(I::Slice {
                out: name.clone(),
                input: input.clone(),
                start: i * n,
            });
        }
        let loss_body = loss_lower.matrix_arguments(loss)?;
        loss_lower.block(&loss_body).context("fn loss")?;
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
        if collect_types { inferred.insert("loss".into(), loss_lower.inferred_types()?); }
        lower.program.loss = Some(Box::new(loss_lower.program));
    }
    if !lower.program.output_type.is_plain() {
        lower.program.version = lower.program.version.max(2);
    }
    lower.program.validate_tensor_contracts()?;
    Ok((lower.program, module_schemas, inferred))
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
fn tensor_axis(tensor: &TensorType, expr: &Expr) -> Result<usize> {
    if let Expr::Text(name) = expr {
        return Ok(tensor.axis(name)?);
    }
    let value = scalar_literal(expr)?;
    ensure!(value >= 0. && value.fract() == 0. && value < tensor.rank() as f32,
        "tensor axis must be a nonnegative in-range integer or declared axis name");
    Ok(value as usize)
}
#[derive(Clone)]
struct ModuleInstance {
    functions: crate::script::parse::Program,
    names: HashMap<String, String>,
    input: usize,
    output: usize,
}
struct Lower {
    source_line: usize,
    scope: Vec<String>,
    program: Program,
    sizes: HashMap<String, usize>,
    types: HashMap<String, crate::categorical::VectorType>,
    /// Compile-time axis labels; empty labels are anonymous dimensions.
    axes: HashMap<String, Vec<Option<String>>>,
    memories: HashMap<String, (usize, usize)>,
    next: usize,
    loops: Vec<(String, String)>,
    output: usize,
    return_shape: Option<Vec<usize>>,
    functions: crate::script::parse::Program,
    modules: HashMap<String, ModuleInstance>,
    calls: usize,
    call_depth: usize,
    tuples: HashMap<String, Vec<String>>,
    constants: HashMap<String, f32>,
    unavailable: HashSet<String>,
    // Ordinary tensors retain their runtime uninitialized-register contract.
    // Definite-assignment rejection applies only to new tuple destructuring.
    tuple_bindings: HashSet<String>,
    operation_stack: Vec<String>,
    operation_returns: Vec<(String, String, Option<usize>)>,
}
#[derive(Clone)]
struct BindingFacts {
    constants: HashMap<String, f32>,
    tuples: HashMap<String, Vec<String>>,
    available: HashSet<String>,
}
impl Lower {
    fn inferred_types(&self) -> Result<std::collections::BTreeMap<String, TensorType>> {
        self.program.registers.iter()
            .filter(|register| !register.name.starts_with("__wgt_"))
            .map(|register| {
                let axes = self.axes.get(&register.name).cloned()
                    .unwrap_or_else(|| vec![None; register.shape.len()]);
                Ok((register.name.clone(), TensorType::new(DType::F32, register.shape.clone(), axes, ShapeLimits::REGISTER)?))
            })
            .collect()
    }
    fn facts(&self) -> BindingFacts {
        BindingFacts {
            constants: self.constants.clone(),
            tuples: self.tuples.clone(),
            available: self.sizes.keys().filter(|n| !self.unavailable.contains(*n)).cloned().collect(),
        }
    }
    fn restore_facts(&mut self, facts: &BindingFacts) {
        self.constants = facts.constants.clone();
        self.tuples = facts.tuples.clone();
        self.unavailable = self.sizes.keys().filter(|n| !facts.available.contains(*n)).cloned().collect();
    }
    fn merge_facts(&mut self, before: &BindingFacts, yes: &BindingFacts, no: &BindingFacts) {
        self.constants = yes.constants.iter()
            .filter(|(name, value)| no.constants.get(*name) == Some(*value))
            .map(|(name, value)| (name.clone(), *value)).collect();
        // Tuple fields have no runtime phi node. Only an existing immutable tuple
        // is available after a branch; branch-local tuples must stay local.
        self.tuples = before.tuples.clone();
        self.unavailable = self.sizes.keys()
            .filter(|n| !yes.available.contains(*n) || !no.available.contains(*n))
            .cloned().collect();
    }
    fn validate_shape_helpers(&self) -> Result<()> {
        let mut checker = Self {
            source_line: 0,
            scope: Vec::new(),
            program: Program {
                observations: Vec::new(),
                output_type: crate::categorical::VectorType::F32,
                version: 1,
                input: "wgt_shape_check".into(),
                registers: Vec::new(),
                parameters: Vec::new(),
                instructions: Vec::new(),
                loss: None,
            },
            sizes: HashMap::new(),
            types: HashMap::new(),
            axes: HashMap::new(),
            memories: HashMap::new(),
            next: 0,
            loops: Vec::new(),
            output: 1,
            return_shape: None,
            functions: self.functions.clone(),
            modules: HashMap::new(),
            calls: 0,
            call_depth: 0,
            tuples: HashMap::new(),
            constants: HashMap::new(),
            unavailable: HashSet::new(),
            tuple_bindings: HashSet::new(),
            operation_stack: Vec::new(),
            operation_returns: Vec::new(),
        };
        for (name, function) in &self.functions {
            if matches!(name.as_str(), "forward" | "loss") { continue; }
            let Some(signature) = &function.tensor_signature else { continue; };
            let mut args = Vec::new();
            for shape in &signature.inputs {
                let register = checker.fresh();
                checker.register_tensor(&register,
                    TensorType::unnamed(DType::F32, shape.clone(), ShapeLimits::REGISTER)?)?;
                args.push(Expr::Name(register));
            }
            let output = checker.fresh();
            checker.operator(&output, name, &args)
                .with_context(|| format!("function shape contract {name}"))?;
            // Only register metadata was constructed. No values, model weights,
            // initialization recipes or forward computations have executed.
            checker.program.registers.clear();
            checker.program.instructions.clear();
            checker.program.observations.clear();
            checker.sizes.clear();
            checker.types.clear();
            checker.axes.clear();
            checker.tuples.clear();
            checker.constants.clear();
            checker.unavailable.clear();
            checker.tuple_bindings.clear();
            checker.calls = 0;
        }
        Ok(())
    }
    fn matrix_arguments(&mut self, function: &crate::script::parse::Function) -> Result<Vec<Statement>> {
        let Some(signature) = &function.tensor_signature else { return Ok(function.body.clone()); };
        let mut names = HashMap::new();
        for (arg, shape) in function.args.iter().zip(&signature.inputs) {
            ensure!(self.size(arg)? == shape.iter().product::<usize>(), "matrix entrypoint input shape mismatch: {arg}");
            let local = self.fresh();
            self.assign(&local, &Expr::Call("reshape".into(), vec![Expr::Name(arg.clone()),
                Expr::List(shape.iter().map(|dimension| Expr::Number(*dimension as f64)).collect())]))?;
            names.insert(arg.clone(), local);
        }
        let mut body = function.body.clone();
        fn rewrite(body: &mut [Statement], names: &HashMap<String, String>) {
            for statement in body {
                match &mut statement.kind {
                    Stmt::Let(_, value) | Stmt::Set(_, value) | Stmt::Return(value) | Stmt::Eval(value) => {
                        *value = crate::script::operators::rewrite(value, names, &HashMap::new());
                    }
                    Stmt::LetTuple(_, value) => *value = crate::script::operators::rewrite(value, names, &HashMap::new()),
                    Stmt::If(value, yes, no) => {
                        *value = crate::script::operators::rewrite(value, names, &HashMap::new());
                        rewrite(yes, names); rewrite(no, names);
                    }
                    Stmt::While(value, body) => {
                        *value = crate::script::operators::rewrite(value, names, &HashMap::new());
                        rewrite(body, names);
                    }
                    _ => {}
                }
            }
        }
        rewrite(&mut body, &names);
        Ok(body)
    }
    fn operator(&mut self, out: &str, name: &str, args: &[Expr]) -> Result<()> {
        if self.functions[name].signature.is_some() {
            return self.operation(out, name, args, false);
        }
        ensure!(self.call_depth < 32, "operator expansion depth exceeded");
        self.call_depth += 1;
        let line = self.source_line;
        let result = self.expand_operator(out, name, args);
        self.source_line = line;
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
        for (index, (arg, value)) in f.args.iter().zip(args).enumerate() {
            let input = self.value(value)?;
            if let Some(signature) = &f.tensor_signature {
                ensure!(self.shape(&input)? == signature.inputs[index],
                    "function {name} argument {arg} shape mismatch: expected {:?}, got {:?}", signature.inputs[index], self.shape(&input)?);
            }
            names.insert(arg.clone(), input);
        }
        self.scope.push(name.into());
        let result = (|| {
            for s in f.body {
                self.source_line = s.line;
                match s.kind {
                    Stmt::Let(n, e) => {
                        let e = rewrite(&e, &names, &HashMap::new());
                        let local = self.fresh();
                        self.assign(&local, &e)
                            .with_context(|| format!("operator {name}, line {}", s.line))?;
                        names.insert(n, local);
                    }
                    Stmt::LetTuple(bindings, e) => {
                        let e = rewrite(&e, &names, &HashMap::new());
                        let value = self.value(&e)?;
                        let fields = self.tuples.get(&value).context("tuple binding requires a tuple result")?.clone();
                        ensure!(bindings.len() == fields.len(), "tuple binding arity mismatch");
                        for (name, value) in bindings.into_iter().zip(fields) { names.insert(name, value); }
                    }
                    Stmt::Return(e) => {
                        self.assign(out, &rewrite(&e, &names, &HashMap::new()))
                            .with_context(|| format!("operator {name}, line {}", s.line))?;
                        if let Some(signature) = &f.tensor_signature {
                            ensure!(self.shape(out)? == signature.output,
                                "function {name} return shape mismatch: expected {:?}, got {:?}", signature.output, self.shape(out)?);
                        }
                        return Ok(());
                    }
                    _ => unreachable!(),
                }
            }
            bail!("operator requires return")
        })();
        self.scope.pop();
        result
    }
    /// Inline a typed call into its caller's transaction. Return jumps are local,
    /// so an early return never exits the parent operation or commits partial state.
    fn operation(&mut self, out: &str, name: &str, args: &[Expr], discard: bool) -> Result<()> {
        use crate::script::operators::rewrite;
        ensure!(self.program.input == "wgt_operation_input", "typed calls require an operation model");
        ensure!(self.call_depth < 32 && !self.operation_stack.iter().any(|n| n == name), "recursive/deep operation call: {name}");
        self.calls += 1;
        ensure!(self.calls <= 1024, "operation expansion limit exceeded");
        let function = self.functions[name].clone();
        let signature = function.signature.as_ref().context("missing operation signature")?;
        ensure!(args.len() == function.args.len(), "operation arity mismatch: {name}");
        ensure!(discard || signature.output.is_some(), "void operation cannot be used as a tensor: {name}");
        fn returns(body: &[Statement]) -> bool {
            match body.last().map(|s| &s.kind) {
                Some(Stmt::Return(_)) => true,
                Some(Stmt::If(_, yes, no)) => returns(yes) && returns(no),
                _ => false,
            }
        }
        ensure!(signature.output.is_none() || returns(&function.body), "operation {name} must return on every path");
        let mut names = HashMap::new();
        for ((arg, width), expr) in function.args.iter().zip(&signature.inputs).zip(args) {
            let value = self.value(expr)?;
            ensure!(self.shape(&value)? == vec![*width], "operation input shape mismatch: {name}.{arg}");
            // Copy arguments: a callee's local assignment must not mutate a caller register.
            let local = self.fresh();
            self.assign(&local, &Expr::Name(value))?;
            names.insert(arg.clone(), local);
        }
        fn locals(lower: &mut Lower, body: &[Statement], names: &mut HashMap<String, String>) {
            for statement in body { match &statement.kind {
                Stmt::Let(name, _) => { names.entry(name.clone()).or_insert_with(|| lower.fresh()); }
                Stmt::LetTuple(bindings, _) => { for name in bindings { names.entry(name.clone()).or_insert_with(|| lower.fresh()); } }
                Stmt::If(_, yes, no) => { locals(lower, yes, names); locals(lower, no, names); }
                Stmt::While(_, body) => locals(lower, body, names),
                _ => {}
            }}
        }
        locals(self, &function.body, &mut names);
        fn rename(body: &mut [Statement], names: &HashMap<String, String>) {
            for statement in body { match &mut statement.kind {
                Stmt::Let(name, e) | Stmt::Set(name, e) => { *name = names.get(name).unwrap_or(name).clone(); *e = rewrite(e, names, &HashMap::new()); }
                Stmt::LetTuple(bindings, e) => { for name in bindings { *name = names.get(name).unwrap_or(name).clone(); } *e = rewrite(e, names, &HashMap::new()); }
                Stmt::Return(e) | Stmt::Eval(e) => *e = rewrite(e, names, &HashMap::new()),
                Stmt::If(e, yes, no) => { *e = rewrite(e, names, &HashMap::new()); rename(yes, names); rename(no, names); }
                Stmt::While(e, body) => { *e = rewrite(e, names, &HashMap::new()); rename(body, names); }
                _ => {}
            }}
        }
        let mut body = function.body;
        rename(&mut body, &names);
        let end = self.fresh();
        self.operation_returns.push((out.into(), end.clone(), signature.output));
        self.operation_stack.push(name.into());
        self.call_depth += 1;
        let loops = std::mem::take(&mut self.loops);
        let line = self.source_line;
        self.scope.push(name.into());
        let result = self.block(&body);
        self.scope.pop();
        self.source_line = line;
        self.loops = loops;
        self.call_depth -= 1;
        self.operation_stack.pop();
        self.operation_returns.pop();
        result.with_context(|| format!("operation call {name}"))?;
        if signature.output.is_none() { self.assign(out, &Expr::List(vec![Expr::Number(0.)]))?; }
        self.program.instructions.push(I::Label { name: end });
        Ok(())
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
            self.axes.insert(name.into(), vec![None]);
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
    fn tensor_type(&self, name: &str) -> Result<TensorType> {
        let shape = self.shape(name)?;
        let axes = self.axes.get(name).cloned().unwrap_or_else(|| vec![None; shape.len()]);
        Ok(TensorType::new(DType::F32, shape, axes, ShapeLimits::REGISTER)?)
    }
    fn gated_type(&self, probability: &str, value: TensorType) -> Result<TensorType> {
        let probability = self.tensor_type(probability)?;
        if probability.shape() == [1] {
            Ok(value)
        } else {
            value.elementwise(&probability, ShapeLimits::REGISTER)
                .context("gate/route probability shape or axis mismatch")
        }
    }
    fn register_tensor(&mut self, name: &str, tensor: TensorType) -> Result<()> {
        let tensor = if self.sizes.contains_key(name) {
            self.tensor_type(name)?.elementwise(&tensor, ShapeLimits::REGISTER)
                .with_context(|| format!("register tensor type changes: {name}"))?
        } else {
            tensor
        };
        self.register_shape(name, tensor.shape().to_vec())?;
        self.axes.insert(name.into(), tensor.axes().to_vec());
        Ok(())
    }
    fn register_shape(&mut self, name: &str, shape: Vec<usize>) -> Result<()> {
        let len = crate::tensor::validate_shape(&shape, crate::tensor::ShapeLimits::REGISTER)
            .with_context(|| format!("tensor register {name}"))?;
        if shape.len() > 1 {
            self.program.version = 3;
        }
        if self.sizes.contains_key(name) {
            ensure!(
                self.shape(name)? == shape,
                "register geometry changes: {name}"
            );
            self.axes.entry(name.into()).or_insert_with(|| vec![None; shape.len()]);
        } else {
            self.register(name, len)?;
            self.program.registers.last_mut().unwrap().shape = shape.clone();
            self.axes.insert(name.into(), vec![None; shape.len()]);
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
            ensure!(!self.tuple_bindings.contains(name) || !self.unavailable.contains(name), "tensor is not defined on every control-flow path: {name}");
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
        ensure!(!self.tuples.contains_key(out), "tuple bindings are immutable: {out}");
        let constant = match expr {
            Expr::Name(name) => self.constants.get(name).copied(),
            Expr::List(items) if items.len() == 1 => scalar_literal(&items[0]).ok(),
            _ => scalar_literal(expr).ok(),
        };
        self.constants.remove(out);
        self.assign_inner(out, expr)?;
        if let Some(value) = constant { self.constants.insert(out.into(), value); }
        self.unavailable.remove(out);
        Ok(())
    }
    fn assign_inner(&mut self, out: &str, expr: &Expr) -> Result<()> {
        if let Expr::Call(name, args) = expr {
            if name == "observe" && !self.functions.contains_key(name) {
                let [Expr::Text(label), value] = args.as_slice() else {
                    bail!("observe requires a literal name and a tensor");
                };
                ensure!(!label.is_empty() && label.len() <= 256, "observation name must be 1..256 bytes");
                let input = self.value(value)?;
                self.assign(out, &Expr::Name(input))?;
                self.program.observations.push(super::Observation {
                    line: self.source_line,
                    scope: self.scope.join("/"),
                    axes: self.axes.get(out).cloned().unwrap_or_default(),
                    name: label.clone(),
                    instruction: self.program.instructions.len() - 1,
                    register: out.into(),
                });
                return Ok(());
            }
        }

        use crate::categorical::VectorType;
        if let Expr::Name(name) = expr {
            ensure!(!self.tuple_bindings.contains(name) || !self.unavailable.contains(name), "tensor is not defined on every control-flow path: {name}");
        }
        if let Expr::Tuple(items) = expr {
            ensure!(!self.sizes.contains_key(out), "tensor binding cannot change to a tuple");
            let mut fields = Vec::with_capacity(items.len());
            for item in items {
                let value = self.value(item)?;
                ensure!(!self.tuples.contains_key(&value), "nested tensor tuples are unsupported");
                self.shape(&value)?;
                // Capture the value now, including when the source is a mutable local.
                let field = self.fresh();
                self.assign(&field, &Expr::Name(value))?;
                fields.push(field);
            }
            self.tuples.insert(out.into(), fields);
            return Ok(());
        }
        if let Expr::Name(name) = expr {
            if let Some(fields) = self.tuples.get(name).cloned() {
                ensure!(!self.sizes.contains_key(out), "tensor binding cannot change to a tuple");
                self.tuples.insert(out.into(), fields);
                return Ok(());
            }
        }
        ensure!(!self.tuples.contains_key(out), "tuple binding cannot change to a tensor");
        if let Expr::BuiltinCall(name, args) = expr {
            if name == "__operation_dispatch" {
                let [Expr::Text(base), index, Expr::Text(op), Expr::Number(count), rest @ ..] = args.as_slice() else { bail!("invalid operation dispatch"); };
                let index = self.static_index(index)?;
                ensure!(index < *count as usize, "submodel index out of bounds");
                let name = format!("{base}_{index}_{op}");
                ensure!(self.functions.get(&name).is_some_and(|f| f.signature.is_some()), "unknown submodel operation: {name}");
                return self.operation(out, &name, rest, false);
            }
            if name == "__tensor_annot" {
                let [Expr::List(dimensions), value] = args.as_slice() else {
                    bail!("invalid internal tensor annotation");
                };
                let input = self.value(value)?;
                let mut expected_shape = Vec::with_capacity(dimensions.len());
                let mut axes = Vec::with_capacity(dimensions.len());
                for dimension in dimensions {
                    let Expr::List(pair) = dimension else { bail!("invalid tensor dimension annotation"); };
                    let [Expr::Text(axis), dimension] = pair.as_slice() else { bail!("invalid tensor dimension annotation"); };
                    let expected = scalar_literal(dimension)?;
                    ensure!(expected >= 1. && expected <= 16384. && expected.fract() == 0.,
                        "tensor annotation dimensions must be integers in 1..16384");
                    expected_shape.push(expected as usize);
                    let axis = if axis.is_empty() { None } else { Some(axis.clone()) };
                    axes.push(axis);
                }
                let tensor = self.tensor_type(&input)?
                    .reshape(expected_shape, axes, ShapeLimits::REGISTER)
                    .context("tensor annotation")?;
                self.register_tensor(out, tensor)?;
                if input != out {
                    self.program.instructions.push(I::Reshape {
                        out: out.into(),
                        input: input.clone(),
                    });
                }
                self.program.version = 3;
                self.types.insert(out.into(), self.types.get(&input).copied().unwrap_or_default());
                return Ok(());
            }
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
                ensure!(out != self.program.input, "cannot overwrite model input");
                self.register_tensor(out, self.tensor_type(&input)?)?;
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
            let line = self.source_line;
            self.scope.push(format!("{name}[{index}]"));
            let result = (|| {
                for statement in forward.body {
                    self.source_line = statement.line;
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
            self.scope.pop();
            self.source_line = line;
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
            let before = self.facts();
            self.program.instructions.push(I::Label { name: a });
            self.assign(out, yes)?;
            ensure!(!self.tuples.contains_key(out), "conditional expressions cannot return tuples");
            let yes_facts = self.facts();
            self.program.instructions.push(I::Jump {
                target: end.clone(),
            });
            self.restore_facts(&before);
            self.program.instructions.push(I::Label { name: b });
            self.assign(out, no)?;
            ensure!(!self.tuples.contains_key(out), "conditional expressions cannot return tuples");
            let no_facts = self.facts();
            self.merge_facts(&before, &yes_facts, &no_facts);
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
        let matrix_arguments;
        let tensor_call = match expr {
            Expr::Call(op, args) => Some((op.as_str(), args.as_slice())),
            Expr::Binary(op, left, right) if op == "@" => {
                matrix_arguments = [(**left).clone(), (**right).clone()];
                Some(("matmul", matrix_arguments.as_slice()))
            }
            Expr::Binary(op, left, right) if op == "/" => {
                matrix_arguments = [(**left).clone(), (**right).clone()];
                Some(("div", matrix_arguments.as_slice()))
            }
            _ => None,
        };
        if let Some((op, args)) = tensor_call {
            let tensor = match (op, args) {
                ("reshape", [x, Expr::List(dims)]) => {
                    let x = self.value(x)?;
                    ensure!(x != out, "in-place reshape is unsupported; bind the view to a new name");
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
                    let tensor = self.tensor_type(&x)?.reshape(
                        shape.clone(), vec![None; shape.len()], ShapeLimits::REGISTER,
                    )?;
                    Some((
                        I::Reshape {
                            out: out.into(),
                            input: x,
                        },
                        tensor,
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
                        TensorType::unnamed(DType::F32, geometry.shape(), ShapeLimits::REGISTER)?,
                    ))
                }
                ("max_pool2d", [x, Expr::List(kernel), Expr::List(stride)])
                | ("avg_pool2d", [x, Expr::List(kernel), Expr::List(stride)]) => {
                    let dimensions = |items: &[Expr]| -> Result<Vec<usize>> {
                        items.iter().map(|item| {
                            let n = scalar_literal(item)?;
                            ensure!(n >= 1. && n <= 16384. && n.fract() == 0.,
                                "pool2d dimensions must be integer literals in 1..=16384");
                            Ok(n as usize)
                        }).collect()
                    };
                    let kernel = dimensions(kernel)?;
                    let stride = dimensions(stride)?;
                    let x = self.value(x)?;
                    let geometry = super::Pool2dGeometry::new(&self.shape(&x)?, &kernel, &stride)?;
                    let instruction = if op == "max_pool2d" {
                        I::MaxPool2d { out: out.into(), input: x, kernel, stride }
                    } else {
                        I::AvgPool2d { out: out.into(), input: x, kernel, stride }
                    };
                    Some((instruction, TensorType::unnamed(DType::F32, geometry.shape(), ShapeLimits::REGISTER)?))
                }
                ("transpose", [x]) => {
                    let x = self.value(x)?;
                    ensure!(x != out, "in-place transpose is unsupported; bind the view to a new name");
                    let tensor = self.tensor_type(&x)?.transpose(ShapeLimits::REGISTER)?;
                    Some((
                        I::Transpose {
                            out: out.into(),
                            input: x,
                        },
                        tensor,
                    ))
                }
                ("permute", [x, Expr::List(order)]) => {
                    let x = self.value(x)?;
                    ensure!(x != out, "in-place permute is unsupported; bind the view to a new name");
                    let input = self.tensor_type(&x)?;
                    let mut axes = Vec::with_capacity(order.len());
                    for axis in order {
                        let index = match axis {
                            Expr::Number(_) | Expr::Unary(_, _) => tensor_axis(&input, axis)?,
                            Expr::Name(name) => input.axis(name)?,
                            _ => bail!("permute axes must be integer indices or declared axis names"),
                        };
                        axes.push(index);
                    }
                    let tensor = input.permute(&axes, ShapeLimits::REGISTER)
                        .context("permute axes must form a bijection")?;
                    Some((
                        I::Permute { out: out.into(), input: x, axes },
                        tensor,
                    ))
                }
                ("batch_matmul", [a, b]) => {
                    let a = self.value(a)?;
                    let b = self.value(b)?;
                    let tensor = self.tensor_type(&a)?
                        .batch_matmul(&self.tensor_type(&b)?, ShapeLimits::REGISTER)
                        .context("batch_matmul")?;
                    Some((
                        I::BatchMatmul { out: out.into(), left: a, right: b },
                        tensor,
                    ))
                }
                ("matmul", [a, b]) => {
                    let a = self.value(a)?;
                    let b = self.value(b)?;
                    let tensor = self.tensor_type(&a)?
                        .matmul(&self.tensor_type(&b)?, ShapeLimits::REGISTER)
                        .context("matrix product axis mismatch")?;
                    Some((
                        I::Matmul {
                            out: out.into(),
                            left: a,
                            right: b,
                        },
                        tensor,
                    ))
                }
                ("layer_norm", [x, gamma, beta, axis, epsilon]) => {
                    let x = self.value(x)?;
                    let gamma = self.value(gamma)?;
                    let beta = self.value(beta)?;
                    let tensor = self.tensor_type(&x)?;
                    let axis = tensor_axis(&tensor, axis).context("LayerNorm")?;
                    tensor.validate_normalization(&self.tensor_type(&gamma)?, &self.tensor_type(&beta)?, axis)?;
                    let epsilon = scalar_literal(epsilon)?;
                    ensure!(epsilon > 0., "LayerNorm epsilon must be positive");
                    Some((
                        I::LayerNorm { out: out.into(), input: x, gamma, beta, axis, epsilon },
                        tensor,
                    ))
                }
                ("softmax", [x, axis]) => {
                    let x = self.value(x)?;
                    let tensor = self.tensor_type(&x)?;
                    let axis = tensor_axis(&tensor, axis).context("softmax")?;
                    Some((
                        I::SoftmaxAxis {
                            out: out.into(),
                            input: x,
                            axis,
                        },
                        tensor,
                    ))
                }
                ("scale", [x, factor]) => {
                    let x = self.value(x)?;
                    let tensor = self.tensor_type(&x)?;
                    let instruction = if let Ok(factor) = scalar_literal(factor) {
                        I::Scale { out: out.into(), input: x, factor }
                    } else {
                        let scalar = self.value(factor)?;
                        ensure!(self.shape(&scalar)? == [1], "runtime scale factor must have shape [1]");
                        I::ScaleBy { out: out.into(), input: x, scalar }
                    };
                    Some((instruction, tensor))
                }
                ("ge", [left, right]) | ("div", [left, right]) => {
                    let left = self.value(left)?;
                    let right = self.value(right)?;
                    let tensor = self.tensor_type(&left)?.elementwise(&self.tensor_type(&right)?, ShapeLimits::REGISTER)?;
                    let instruction = if op == "ge" {
                        I::Ge { out: out.into(), left, right }
                    } else {
                        I::Div { out: out.into(), left, right }
                    };
                    Some((instruction, tensor))
                }
                ("sqrt", [input]) => {
                    let input = self.value(input)?;
                    let tensor = self.tensor_type(&input)?;
                    Some((I::Sqrt { out: out.into(), input }, tensor))
                }
                _ => None,
            };
            if let Some((instruction, tensor)) = tensor {
                self.register_tensor(out, tensor)?;
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
            Expr::Binary(op, a, b) if op == "+" || op == "*" || op == "-" => {
                let a = self.value(a)?;
                let b = self.value(b)?;
                let len = self.size(&a)?;
                let b = if op == "-" {
                    let neg = self.fresh();
                    self.register_tensor(&neg, self.tensor_type(&b)?)?;
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
        let tensor = match &instruction {
            I::Reshape { input, .. } => self.tensor_type(input)?,
            I::Slice {
                input, start: 0, ..
            } if matches!(expr, Expr::Name(_)) => self.tensor_type(input)?,
            I::Neg { input, .. }
            | I::Detach { input, .. }
            | I::Exp { input, .. }
            | I::Log { input, .. }
            | I::Tanh { input, .. }
            | I::Sigmoid { input, .. }
            | I::Relu { input, .. }
            | I::Clip { input, .. } => self.tensor_type(input)?,
            I::Add { left, right, .. } | I::Mul { left, right, .. } | I::Minimum { left, right, .. } => {
                self.tensor_type(left)?.elementwise(&self.tensor_type(right)?, ShapeLimits::REGISTER)?
            }
            I::Gate { probability, input, .. } => self.gated_type(probability, self.tensor_type(input)?)?,
            I::Route { probability, yes, no, .. } => {
                let value = self.tensor_type(yes)?.elementwise(&self.tensor_type(no)?, ShapeLimits::REGISTER)?;
                self.gated_type(probability, value)?
            }
            I::Softmax { input, .. } | I::LogSoftmax { input, .. } => {
                let tensor = self.tensor_type(input)?;
                ensure!(tensor.rank() == 1, "rank > 1 softmax requires explicit axis");
                tensor
            }
            I::Linear { input, .. } => {
                ensure!(
                    self.shape(input)?.len() == 1,
                    "linear requires vector input; use matmul for matrices"
                );
                TensorType::unnamed(DType::F32, vec![len], ShapeLimits::REGISTER)?
            }
            I::Slice { input, .. } => {
                ensure!(self.shape(input)?.len() == 1, "slice requires vector input");
                TensorType::unnamed(DType::F32, vec![len], ShapeLimits::REGISTER)?
            }
            _ => TensorType::unnamed(DType::F32, vec![len], ShapeLimits::REGISTER)?,
        };
        self.register_tensor(out, tensor)?;

        self.types.insert(out.into(), kind);
        self.program.instructions.push(instruction);
        Ok(())
    }
    fn static_index(&self, expr: &Expr) -> Result<usize> {
        let n = if let Expr::Name(name) = expr {
            f64::from(*self.constants.get(name)
                .context("index must be a compile-time constant or repeat index")?)
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
            self.source_line = s.line;
            (|| -> Result<()> {
                match &s.kind {
                    Stmt::Let(n, e) | Stmt::Set(n, e) => {
                        ensure!(!n.starts_with("__wgt_") || !self.operation_stack.is_empty(), "reserved compiler name");
                        ensure!(!self.tuples.contains_key(n), "tuple bindings are immutable: {n}");
                        if matches!(&s.kind, Stmt::Set(..)) {
                            ensure!(
                                self.sizes.contains_key(n),
                                "assignment to undeclared tensor"
                            );
                        }
                        self.assign(n, e)?;
                    }
                    Stmt::LetTuple(bindings, e) => {
                        let value = self.value(e)?;
                        let fields = self.tuples.get(&value).context("tuple binding requires a tuple result")?.clone();
                        ensure!(bindings.len() == fields.len(), "tuple binding arity mismatch");
                        self.tuple_bindings.extend(bindings.iter().cloned());
                        for (name, value) in bindings.iter().zip(fields) { self.assign(name, &Expr::Name(value))?; }
                    }
                    Stmt::Eval(Expr::BuiltinCall(op, args)) if op == "__transaction_write" && args.len() == 2 => {
                        let target = self.parameter(&args[0])?;
                        let (parameter, expected) = (target.name.clone(), target.shape.clone());
                        ensure!(!target.frozen, "frozen update target: {parameter}");
                        ensure!(!target.probability, "transactional probability updates are unsupported");
                        let value = self.value(&args[1])?;
                        let actual = self.shape(&value)?;
                        ensure!(expected == actual, "update shape mismatch for {parameter}: expected {expected:?}, got {actual:?}");
                        self.program.instructions.push(I::Write { parameter, value });
                    }
                    Stmt::Eval(Expr::BuiltinCall(op, args)) if op == "__operation_dispatch" => {
                        let [Expr::Text(base), index, Expr::Text(op), Expr::Number(count), rest @ ..] = args.as_slice() else { bail!("invalid operation dispatch"); };
                        let index = self.static_index(index)?;
                        ensure!(index < *count as usize, "submodel index out of bounds");
                        let name = format!("{base}_{index}_{op}");
                        ensure!(self.functions.get(&name).is_some_and(|f| f.signature.is_some()), "unknown submodel operation");
                        let out = self.fresh();
                        self.operation(&out, &name, rest, true)?;
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
                        if let Some((out, end, width)) = self.operation_returns.last().cloned() {
                            if let Some(width) = width {
                                self.assign(&out, e)?;
                                ensure!(self.shape(&out)? == vec![width], "operation return shape mismatch");
                            } else {
                                ensure!(matches!(e, Expr::Null), "void operation cannot return a tensor");
                                self.assign(&out, &Expr::List(vec![Expr::Number(0.)]))?;
                            }
                            self.program.instructions.push(I::Jump { target: end });
                            return Ok(());
                        }
                        let mut value = self.value(e)?;
                        let expected = self.return_shape.clone().unwrap_or_else(|| vec![self.output]);
                        ensure!(self.shape(&value)? == expected,
                            "forward return/output shape mismatch: expected {:?}, got {:?}", expected, self.shape(&value)?);
                        if expected.len() > 1 {
                            let flat = self.fresh();
                            self.assign(&flat, &Expr::Call("reshape".into(), vec![Expr::Name(value),
                                Expr::List(vec![Expr::Number(self.output as f64)])]))?;
                            value = flat;
                        }
                        self.program.instructions.push(I::Return { value });
                    }
                    Stmt::Eval(Expr::Call(name, args)) if self.functions.get(name).is_some_and(|f| f.signature.is_some()) => {
                        let out = self.fresh();
                        self.operation(&out, name, args, true)?;
                    }
                    Stmt::If(e, yes, no) => {
                        let (a, b, end) = (self.fresh(), self.fresh(), self.fresh());
                        self.condition(e, a.clone(), b.clone())?;
                        let before = self.facts();
                        self.program.instructions.push(I::Label { name: a });
                        self.block(yes)?;
                        let yes_facts = self.facts();
                        self.program.instructions.push(I::Jump {
                            target: end.clone(),
                        });
                        self.restore_facts(&before);
                        self.program.instructions.push(I::Label { name: b });
                        self.block(no)?;
                        let no_facts = self.facts();
                        self.merge_facts(&before, &yes_facts, &no_facts);
                        self.program.instructions.push(I::Label { name: end });
                    }
                    Stmt::While(e, body) => {
                        fn assigned(body: &[Statement], names: &mut HashSet<String>) {
                            for statement in body {
                                match &statement.kind {
                                    Stmt::Let(name, _) | Stmt::Set(name, _) => { names.insert(name.clone()); }
                                    Stmt::LetTuple(bindings, _) => names.extend(bindings.iter().cloned()),
                                    Stmt::If(_, yes, no) => { assigned(yes, names); assigned(no, names); }
                                    Stmt::While(_, body) => assigned(body, names),
                                    _ => {}
                                }
                            }
                        }
                        let before = self.facts();
                        let mut changed = HashSet::new();
                        assigned(body, &mut changed);
                        for name in &changed { self.constants.remove(name); }
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
                        self.restore_facts(&before);
                        for name in changed { self.constants.remove(&name); }
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

fn parse_learn_domain(domain: &str) -> Result<super::LearnDomain> {
    use super::LearnDomain as D;
    let result = if domain == "float" { D::Float }
    else if domain == "bool" { D::Bool }
    else if let Some(low) = domain.strip_prefix("float >") {
        let low: f32 = low.trim().parse().context("invalid learn strict lower bound")?;
        ensure!(low.is_finite() && low >= 0. && low < f32::MAX, "invalid learn positive bound");
        D::Positive { low }
    } else {
        let (kind, bounds) = domain.split_once(" in ").context("unsupported learn type")?;
        let (low, high) = bounds.split_once("..=").context("learn bounds require inclusive low..=high")?;
        let low64: f64 = low.trim().parse().context("invalid learn lower bound")?;
        let high64: f64 = high.trim().parse().context("invalid learn upper bound")?;
        if kind.trim() == "int" {
            ensure!(low64.fract() == 0. && high64.fract() == 0. && low64.abs() <= 16777216. && high64.abs() <= 16777216., "learn int bounds must be exact f32 integers within +/-16777216");
        }
        let (low, high) = (low64 as f32, high64 as f32);
        ensure!(low.is_finite() && high.is_finite() && low < high, "invalid learn interval");
        match kind.trim() { "float" => D::Bounded { low, high }, "int" => D::Int { low, high }, _ => bail!("unsupported learn type") }
    };
    Ok(result)
}
