//! Static ownership and call linking for stateful model instances. No runtime VM.
use crate::{script::{operators, parse::{self, Expr, Program, Statement, Stmt}}, source};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) struct Flat {
    pub declarations: String,
    pub functions: Program,
    pub entries: BTreeMap<String, String>,
    pub readonly: BTreeSet<String>,
}

#[derive(Clone)]
struct Definition {
    source: String,
    environment: usize,
}

#[derive(Clone)]
struct Environment {
    definitions: BTreeMap<String, Definition>,
    pure_calls: HashMap<String, String>,
}

pub(super) fn flatten(text: &str) -> Result<Flat> {
    validate_declarations(text, &BTreeSet::new(), &HashMap::new(), &Program::new(), 0, &mut 0)?;
    let mut flat = Flat { declarations: String::new(), functions: Program::new(), entries: BTreeMap::new(), readonly: BTreeSet::new() };
    let mut count = 0;
    scope(text, "", &BTreeMap::new(), &HashMap::new(), &mut Vec::new(), &mut Vec::new(), &mut count, &mut flat)?;
    ensure!(flat.declarations.len() <= 1024 * 1024, "expanded model storage exceeds 1 MiB");
    Ok(flat)
}

fn scope(text: &str, prefix: &str, inherited: &BTreeMap<String, Definition>, inherited_calls: &HashMap<String, String>, environments: &mut Vec<Environment>, stack: &mut Vec<(usize, String)>, count: &mut usize, flat: &mut Flat) -> Result<HashMap<String, String>> {
    ensure!(stack.len() < 32, "submodel nesting exceeds 32");
    *count += 1;
    ensure!(*count <= 256, "model instance count exceeds 256");
    let unit = source::split(text)?;
    let environment = environments.len();
    let mut definitions = inherited.clone();
    for (name, body) in &unit.models { definitions.insert(name.clone(), Definition { source: body.clone(), environment }); }
    let mut functions = parse::parse_functions(&unit.functions)?;
    let initialization_helpers = crate::initializer::helpers(&unit.declarations, &functions)?;
    ensure!(!functions.contains_key("loss"), "stateful model composition does not support loss or cross-state AD");
    let mut names = HashMap::new();
    let mut calls = inherited_calls.clone();
    for (name, function) in &functions {
        let operation = function.signature.is_some() || (function.args.is_empty() && !initialization_helpers.contains(name));
        calls.insert(name.clone(), if prefix.is_empty() && operation { format!("wgt_operation_{name}") } else { format!("{prefix}{name}") });
    }
    let mut instances = BTreeMap::new();
    let mut input = None;
    let mut output = None;
    for line in unit.declarations.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let (keyword, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match keyword {
            "wgt" | "model" => { if prefix.is_empty() { flat.declarations.push_str(line); flat.declarations.push('\n'); } }
            "state" | "parameter" | "frozen" | "learn" => {
                let (name, ty) = rest.split_once(':').context("storage requires name: type")?;
                let name = name.trim();
                ensure!(operators::identifier(name), "invalid storage name: {name}");
                let target = format!("{prefix}{name}");
                ensure!(names.insert(name.to_owned(), target.clone()).is_none(), "duplicate storage: {name}");
                flat.declarations.push_str(&crate::model_modules::rewrite_storage_initializers(&format!("{keyword} {target}:{ty}"), &calls)?);
            }
            "input" | "output" => {
                ensure!(!prefix.is_empty(), "typed operations declare ports in function signatures, not input/output/target declarations");
                let port = source::vector_port(rest)?;
                let slot = if keyword == "input" { &mut input } else { &mut output };
                ensure!(slot.replace(port).is_none(), "duplicate submodel port");
            }
            "target" => anyhow::bail!("submodel loss belongs to a differentiable root model"),
            _ => {
                let (name, kind) = line.trim_end_matches(';').split_once(':').context("unsupported model declaration")?;
                ensure!(line.ends_with(';'), "instance declaration requires semicolon");
                let name = name.trim();
                let (kind, width) = if let Some((kind, width)) = kind.trim().split_once('[') {
                    (kind.trim(), width.strip_suffix(']').context("invalid instance array")?.trim().parse::<usize>()?)
                } else { (kind.trim(), 1) };
                ensure!(operators::identifier(name) && (1..=256).contains(&width), "invalid instance name or count");
                ensure!(!names.contains_key(name) && instances.insert(name.to_owned(), (kind.to_owned(), width)).is_none(), "duplicate instance name");
            }
        }
    }
    if input.is_some() || output.is_some() {
        let (input_name, width) = input.context("submodel requires input and output")?;
        let (_, out) = output.context("submodel requires input and output")?;
        let forward = functions.get_mut("forward").context("submodel requires forward")?;
        ensure!(forward.signature.is_none() && forward.args == [input_name], "invalid legacy submodel forward signature");
        forward.signature = Some(parse::OperationSignature { inputs: vec![width], output: Some(out) });
    }
    for (name, function) in &mut functions {
        ensure!(operators::identifier(name) && name != "main" && !function.is_async, "model cannot contain host/async entrypoints");
        if function.args.is_empty() && function.signature.is_none() && !initialization_helpers.contains(name) {
            function.signature = Some(parse::OperationSignature { inputs: Vec::new(), output: None });
        }
        let target = if prefix.is_empty() && function.signature.is_some() { format!("wgt_operation_{name}") } else { format!("{prefix}{name}") };
        if name == "forward" { flat.readonly.insert(target.clone()); }
        if prefix.is_empty() && function.signature.is_some() { flat.entries.insert(target.clone(), name.clone()); }
        calls.insert(name.clone(), target);
    }
    // Children can reuse parent pure helpers, but cannot implicitly call or capture
    // parent operations/storage. Their mutable ownership stays explicit.
    let mut pure_calls = inherited_calls.clone();
    pure_calls.extend(functions.iter().filter(|(_, f)| f.signature.is_none()).map(|(name, _)| (name.clone(), calls[name].clone())));
    // Bind definitions where they are declared, not where an instance happens
    // to be constructed. Nested declarations can shadow ancestor helpers/types.
    environments.push(Environment { definitions: definitions.clone(), pure_calls });
    let mut arrays = HashMap::new();
    for (name, (kind, width)) in instances {
        let child = definitions.get(&kind).with_context(|| format!("unknown submodel definition: {kind}"))?.clone();
        let identity = (child.environment, kind.clone());
        ensure!(!stack.contains(&identity), "recursive model instance: {kind}");
        stack.push(identity);
        let declaration_scope = environments[child.environment].clone();
        for i in 0..width {
            let child_prefix = format!("{prefix}{name}_{i}_");
            let child_calls = scope(&child.source, &child_prefix, &declaration_scope.definitions, &declaration_scope.pure_calls, environments, stack, count, flat)?;
            for (op, target) in child_calls {
                if i == 0 { calls.insert(format!("{name}.{op}"), target.clone()); }
                calls.insert(format!("{name}[{i}].{op}"), target.clone());
                if op == "forward" {
                    if i == 0 { calls.insert(name.clone(), target.clone()); }
                    calls.insert(format!("{name}[{i}]"), target);
                }
            }
        }
        stack.pop();
        arrays.insert(name, (format!("{prefix}"), width));
    }
    let exported = functions.iter().filter(|(_,f)| f.signature.is_some()).map(|(name,_)| (name.clone(), calls[name].clone())).collect();
    for (name, mut function) in functions {
        let mut visible = if function.signature.is_some() { names.clone() } else { HashMap::new() };
        for arg in &function.args {
            ensure!(function.signature.is_none() || !names.contains_key(arg), "operation input shadows storage: {arg}");
            visible.remove(arg);
        }
        rewrite_body(&mut function.body, &visible, &calls, &arrays)?;
        ensure!(flat.functions.insert(calls[&name].clone(), function).is_none(), "submodel function namespace collision");
    }
    Ok(exported)
}

// Inspect the declaration tree, not the instance tree. In particular, an unused
// array neither consumes the instance budget nor creates parameter/state data.
fn validate_declarations(text: &str, inherited_definitions: &BTreeSet<String>, inherited_calls: &HashMap<String, String>, inherited_functions: &Program, depth: usize, next: &mut usize) -> Result<()> {
    ensure!(depth < 32, "submodel declaration nesting exceeds 32");
    let unit = source::split(text)?;
    let mut definitions = inherited_definitions.clone();
    definitions.extend(unit.models.iter().map(|(name, _)| name.clone()));
    let mut functions = parse::parse_functions(&unit.functions)?;
    let initialization_helpers = crate::initializer::helpers(&unit.declarations, &functions)?;
    let mut storage = HashMap::new();
    let mut instance_names = BTreeSet::new();
    let mut declarations = String::new();
    let (mut model, mut input, mut output) = (false, None, None);
    for line in unit.declarations.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let (keyword, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match keyword {
            "wgt" => declarations.push_str("wgt 0.5\n"),
            "model" => {
                ensure!(!model && operators::identifier(rest.trim()), "model definition requires one valid model name");
                model = true;
                declarations.push_str(line);
                declarations.push('\n');
            }
            "state" | "parameter" | "frozen" | "learn" => {
                let (name, ty) = rest.split_once(':').context("storage requires name: type")?;
                let name = name.trim();
                ensure!(operators::identifier(name) && !instance_names.contains(name) && storage.insert(name.to_owned(), name.to_owned()).is_none(), "duplicate or invalid storage: {name}");
                if keyword == "state" {
                    let ty = ty.split_once('=').map_or(ty, |(ty, _)| ty).trim_end_matches(';');
                    source::vector_port(&format!("{name}: {}", ty.trim()))?;
                    declarations.push_str(&format!("parameter {}\n", rest.trim_end_matches(';').trim()));
                } else {
                    declarations.push_str(line);
                    declarations.push('\n');
                }
            }
            "input" | "output" => {
                ensure!(depth > 0, "typed operations declare ports in function signatures, not input/output/target declarations");
                let slot = if keyword == "input" { &mut input } else { &mut output };
                ensure!(slot.replace(source::vector_port(rest)?).is_none(), "duplicate submodel port");
            }
            "target" => anyhow::bail!("submodel loss belongs to a differentiable root model"),
            _ => {
                let (name, kind) = line.strip_suffix(';').context("instance declaration requires semicolon")?.split_once(':').context("unsupported model declaration")?;
                let name = name.trim();
                let (kind, width) = if let Some((kind, width)) = kind.trim().split_once('[') {
                    (kind.trim(), width.strip_suffix(']').context("invalid instance array")?.trim().parse::<usize>()?)
                } else { (kind.trim(), 1) };
                ensure!(operators::identifier(name) && (1..=256).contains(&width), "invalid instance name or count");
                ensure!(!storage.contains_key(name) && instance_names.insert(name.to_owned()), "duplicate instance name");
                ensure!(definitions.contains(kind), "unknown submodel definition: {kind}");
            }
        }
    }
    ensure!(model, "model definition requires one valid model name");
    if input.is_some() || output.is_some() {
        let (name, width) = input.context("submodel requires input and output")?;
        let (_, out) = output.context("submodel requires input and output")?;
        let forward = functions.get_mut("forward").context("submodel requires forward")?;
        ensure!(forward.signature.is_none() && forward.args == [name], "invalid legacy submodel forward signature");
        forward.signature = Some(parse::OperationSignature { inputs: vec![width], output: Some(out) });
    }
    // Reuse the regular declaration/initializer validator without executing a
    // declared function or materializing any declared parameter tensor.
    let mut argument = "wgt_validation_input".to_owned();
    while storage.contains_key(&argument) { argument.push('_'); }
    declarations.push_str(&format!("input {argument}: f32[1]\noutput wgt_validation_output: f32[1]\n"));
    let mut identity = inherited_functions.clone();
    for (original, target) in inherited_calls {
        if let Some(function) = inherited_functions.get(target) { identity.insert(original.clone(), function.clone()); }
    }
    identity.extend(functions.clone());
    identity.extend(parse::parse_functions(&format!("wgt 0.5\nfn forward({argument}) {{ return {argument}; }}"))?);
    let plan = crate::adaptive::validate_definition(&declarations, &identity)?;
    let total = plan.parameters.iter().try_fold(0usize, |total, parameter| {
        total.checked_add(crate::model::validate_shape(&parameter.shape)?).context("storage size overflow")
    })?;
    ensure!(total <= 16 * 1024 * 1024, "model storage element limit exceeded");

    let id = *next;
    *next += 1;
    let mut calls = inherited_calls.clone();
    for (name, function) in &mut functions {
        ensure!(operators::identifier(name) && name != "main" && name != "loss" && !function.is_async, "model cannot contain host/async/loss entrypoints");
        if function.signature.is_none() && function.args.is_empty() && !initialization_helpers.contains(name) {
            function.signature = Some(parse::OperationSignature { inputs: Vec::new(), output: None });
        }
        if let Some(signature) = &function.signature {
            ensure!(signature.inputs.iter().try_fold(0usize, |n, width| n.checked_add(*width)).is_some_and(|n| n <= 16384), "operation aggregate input exceeds 16384 values");
            ensure!(function.args.iter().all(|name| !storage.contains_key(name)), "operation input shadows storage");
        }
        calls.insert(name.clone(), format!("wgt_validation_{id}_{name}"));
    }
    let mut visible_functions = inherited_functions.clone();
    let mut pure_calls = inherited_calls.clone();
    let mut pure = Vec::new();
    for (name, mut function) in functions {
        let names = if function.signature.is_some() { storage.clone() } else { HashMap::new() };
        rewrite_body(&mut function.body, &names, &calls, &HashMap::new())?;
        validate_model_body(&function.body)?;
        if function.signature.is_none() {
            pure_calls.insert(name.clone(), calls[&name].clone());
            pure.push(calls[&name].clone());
        }
        visible_functions.insert(calls[&name].clone(), function);
    }
    for name in pure { operators::validate(&name, &visible_functions, &mut Vec::new())?; }
    for (name, child) in &unit.models {
        validate_declarations(child, &definitions, &pure_calls, &visible_functions, depth + 1, next).with_context(|| format!("invalid submodel definition: {name}"))?;
    }
    Ok(())
}

fn validate_model_body(body: &[Statement]) -> Result<()> {
    fn expression(value: &Expr) -> Result<()> {
        match value {
            Expr::Call(name, args) | Expr::BuiltinCall(name, args) => {
                ensure!(operators::intrinsic(name) || crate::initializer::intrinsic(name) || crate::script::prelude::builtin(name).is_none() || ["__probabilities", "__record_field", "__transaction_write", "__operation_dispatch"].contains(&name.as_str()), "model functions cannot call host functions: {name}");
                for arg in args { expression(arg)?; }
            }
            Expr::List(items) | Expr::Tuple(items) => { for item in items { expression(item)?; } }
            Expr::Invoke(target, args) => {
                expression(target)?;
                for arg in args { expression(arg)?; }
            }
            Expr::Index(a, b) | Expr::Binary(_, a, b) => { expression(a)?; expression(b)?; }
            Expr::Slice(value, start, end) => {
                expression(value)?;
                for bound in start.iter().chain(end.iter()) { expression(bound)?; }
            }
            Expr::Unary(_, value) => expression(value)?,
            Expr::Mix(a, b, c) | Expr::Conditional(a, b, c) => { expression(a)?; expression(b)?; expression(c)?; }
            Expr::Repeat { depth, initial, body, .. } => { expression(depth)?; expression(initial)?; expression(body)?; }
            Expr::Blend(arms) => { for (a, b) in arms { expression(a)?; expression(b)?; } }
            Expr::Sample(..) | Expr::SampleFrom(..) | Expr::StaticCall(..) => anyhow::bail!("host sampling/training expressions are not model computation"),
            Expr::Name(_) | Expr::Number(_) | Expr::Text(_) | Expr::Bool(_) | Expr::Null => {}
        }
        Ok(())
    }
    for statement in body {
        match &statement.kind {
            Stmt::Let(_, value) | Stmt::LetTuple(_, value) | Stmt::Set(_, value) | Stmt::Eval(value) | Stmt::Return(value) => expression(value)?,
            Stmt::If(condition, yes, no) => { expression(condition)?; validate_model_body(yes)?; validate_model_body(no)?; }
            Stmt::While(condition, body) => { expression(condition)?; validate_model_body(body)?; }
            Stmt::Break | Stmt::Continue => {}
            _ => anyhow::bail!("model functions cannot contain host execution statements"),
        }
    }
    Ok(())
}

fn rewrite_body(body: &mut [Statement], names: &HashMap<String, String>, calls: &HashMap<String, String>, arrays: &HashMap<String, (String, usize)>) -> Result<()> {
    for statement in body {
        match &mut statement.kind {
            Stmt::Let(name, expr) | Stmt::Set(name, expr) => {
                ensure!(!names.contains_key(name), "local binding shadows model storage: {name}");
                *expr = expression(expr, names, calls, arrays)?;
            }
            Stmt::LetTuple(bindings, expr) => {
                ensure!(bindings.iter().all(|name| !names.contains_key(name)), "tuple binding shadows storage");
                *expr = expression(expr, names, calls, arrays)?;
            }
            Stmt::Eval(expr) | Stmt::Return(expr) => *expr = expression(expr, names, calls, arrays)?,
            Stmt::If(expr, yes, no) => { *expr = expression(expr, names, calls, arrays)?; rewrite_body(yes,names,calls,arrays)?; rewrite_body(no,names,calls,arrays)?; }
            Stmt::While(expr, body) => { *expr = expression(expr,names,calls,arrays)?; rewrite_body(body,names,calls,arrays)?; }
            Stmt::Break | Stmt::Continue => {}
            _ => anyhow::bail!("model functions cannot contain host execution statements"),
        }
    }
    Ok(())
}

fn expression(expr: &Expr, names: &HashMap<String, String>, calls: &HashMap<String, String>, arrays: &HashMap<String, (String, usize)>) -> Result<Expr> {
    // Static array dispatch is preserved until tensor lowering, where an indexed
    // repeat's index is known. This never becomes a dynamic call instruction.
    let rewritten = operators::rewrite(expr, names, calls);
    fn visit(e: Expr, arrays: &HashMap<String, (String, usize)>) -> Result<Expr> {
        Ok(match e {
            Expr::Invoke(target, args) => {
                let args = args.into_iter().map(|x| visit(x,arrays)).collect::<Result<Vec<_>>>()?;
                let (target, op) = match *target {
                    Expr::BuiltinCall(ref name, ref fields) if name == "__record_field" && fields.len() == 2 => {
                        let Expr::Text(op) = &fields[1] else { anyhow::bail!("operation name must be static"); };
                        (fields[0].clone(), op.clone())
                    }
                    target => (target, "forward".to_owned()),
                };
                if let Expr::Index(base, index) = &target {
                    if let Expr::Name(name) = &**base {
                        if let Some((prefix, width)) = arrays.get(name) {
                            let mut values = vec![Expr::Text(format!("{prefix}{name}")), (**index).clone(), Expr::Text(op), Expr::Number(*width as f64)];
                            values.extend(args);
                            return Ok(Expr::BuiltinCall("__operation_dispatch".into(), values));
                        }
                    }
                }
                Expr::Invoke(Box::new(visit(target,arrays)?), args)
            }
            Expr::Call(name,args) => Expr::Call(name,args.into_iter().map(|e|visit(e,arrays)).collect::<Result<_>>()?),
            Expr::BuiltinCall(name,args) => Expr::BuiltinCall(name,args.into_iter().map(|e|visit(e,arrays)).collect::<Result<_>>()?),
            Expr::List(items) => Expr::List(items.into_iter().map(|e|visit(e,arrays)).collect::<Result<_>>()?),
            Expr::Tuple(items) => Expr::Tuple(items.into_iter().map(|e|visit(e,arrays)).collect::<Result<_>>()?),
            Expr::Index(value, index) => Expr::Index(Box::new(visit(*value, arrays)?), Box::new(visit(*index, arrays)?)),
            Expr::Slice(value, start, end) => Expr::Slice(
                Box::new(visit(*value, arrays)?),
                start.map(|e| visit(*e, arrays).map(Box::new)).transpose()?,
                end.map(|e| visit(*e, arrays).map(Box::new)).transpose()?,
            ),
            Expr::Binary(op,a,b) => Expr::Binary(op,Box::new(visit(*a,arrays)?),Box::new(visit(*b,arrays)?)),
            Expr::Unary(op,x) => Expr::Unary(op,Box::new(visit(*x,arrays)?)),
            Expr::Conditional(c,a,b) => Expr::Conditional(Box::new(visit(*c,arrays)?),Box::new(visit(*a,arrays)?),Box::new(visit(*b,arrays)?)),
            Expr::Repeat{depth,index,carrying,initial,body} => Expr::Repeat{depth,index,carrying,initial:Box::new(visit(*initial,arrays)?),body:Box::new(visit(*body,arrays)?)},
            Expr::Mix(p,a,b) => Expr::Mix(Box::new(visit(*p,arrays)?),Box::new(visit(*a,arrays)?),Box::new(visit(*b,arrays)?)),
            Expr::Blend(arms) => Expr::Blend(arms.into_iter().map(|(a,b)|Ok((visit(a,arrays)?,visit(b,arrays)?))).collect::<Result<_>>()?),
            other => other,
        })
    }
    visit(rewritten, arrays)
}
