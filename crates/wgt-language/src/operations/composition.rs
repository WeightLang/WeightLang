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

pub(super) fn flatten(text: &str) -> Result<Flat> {
    let mut flat = Flat { declarations: String::new(), functions: Program::new(), entries: BTreeMap::new(), readonly: BTreeSet::new() };
    let mut count = 0;
    scope(text, "", &BTreeMap::new(), &HashMap::new(), &mut Vec::new(), &mut count, &mut flat)?;
    ensure!(flat.declarations.len() <= 1024 * 1024, "expanded model storage exceeds 1 MiB");
    Ok(flat)
}

fn scope(text: &str, prefix: &str, inherited: &BTreeMap<String, String>, inherited_calls: &HashMap<String, String>, stack: &mut Vec<String>, count: &mut usize, flat: &mut Flat) -> Result<HashMap<String, String>> {
    ensure!(stack.len() < 32, "submodel nesting exceeds 32");
    *count += 1;
    ensure!(*count <= 256, "model instance count exceeds 256");
    let unit = source::split(text)?;
    let mut definitions = inherited.clone();
    for (name, body) in &unit.models { definitions.insert(name.clone(), body.clone()); }
    let mut functions = parse::parse_functions(&unit.functions)?;
    ensure!(!functions.contains_key("loss"), "stateful model composition does not support loss or cross-state AD");
    let mut names = HashMap::new();
    let mut calls = inherited_calls.clone();
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
                flat.declarations.push_str(&format!("{keyword} {target}:{ty}\n"));
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
        if function.args.is_empty() && function.signature.is_none() {
            function.signature = Some(parse::OperationSignature { inputs: Vec::new(), output: None });
        }
        let target = if prefix.is_empty() && function.signature.is_some() { format!("wgt_operation_{name}") } else { format!("{prefix}{name}") };
        if name == "forward" { flat.readonly.insert(target.clone()); }
        if prefix.is_empty() && function.signature.is_some() { flat.entries.insert(target.clone(), name.clone()); }
        calls.insert(name.clone(), target);
    }
    // Children can reuse parent pure helpers, but cannot implicitly call or capture
    // parent operations/storage. Their mutable ownership stays explicit.
    let pure_calls = functions.iter().filter(|(_, f)| f.signature.is_none()).map(|(name, _)| (name.clone(), calls[name].clone())).chain(inherited_calls.iter().map(|(k,v)| (k.clone(),v.clone()))).collect();
    let mut arrays = HashMap::new();
    for (name, (kind, width)) in instances {
        let child = definitions.get(&kind).with_context(|| format!("unknown submodel definition: {kind}"))?;
        ensure!(!stack.contains(&kind), "recursive model instance: {kind}");
        stack.push(kind.clone());
        for i in 0..width {
            let child_prefix = format!("{prefix}{name}_{i}_");
            let child_calls = scope(child, &child_prefix, &definitions, &pure_calls, stack, count, flat)?;
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
