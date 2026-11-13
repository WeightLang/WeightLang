//! Signature-based differentiable definitions reuse the existing tensor frontend.
//! A declared loss opts into differentiation; stateful operations remain explicit.
use crate::script::parse::{self, Expr, Stmt};
use anyhow::{Context, Result, ensure};

pub(crate) fn normalize(source: &str) -> Result<String> {
    let unit = crate::source::split(source)?;
    let mut functions = parse::parse_functions(&unit.functions)?;
    // Keep the explicit logical contract through import resolution and storage.
    // ModelDefinition lowers it temporarily for validation/compilation instead.
    if functions.get("forward").is_some_and(|f| f.forward_signature.is_some()) {
        return Ok(source.to_owned());
    }

    if let Some(contract) = functions
        .get("forward")
        .and_then(|f| f.tensor_signature.clone())
    {
        ensure!(
            contract.inputs.len() == 1,
            "matrix forward requires one tensor input"
        );
        let input: usize = contract.inputs[0].iter().product();
        let output: usize = contract.output.iter().product();
        ensure!(
            input <= 16384 && output <= 16384,
            "matrix entrypoint ports are limited to 16384 elements"
        );
        no_updates(&functions["forward"].body)?;
        let mut target = output;
        if let Some(loss) = functions.get_mut("loss") {
            ensure!(
                loss.args.len() == 2,
                "loss requires prediction and target arguments"
            );
            if let Some(signature) = &loss.tensor_signature {
                ensure!(
                    signature.inputs[0] == contract.output && signature.output == [1],
                    "matrix loss signature must match forward output and return f32[1]"
                );
                target = signature.inputs[1].iter().product();
            } else if let Some(signature) = loss.signature.take() {
                ensure!(
                    signature.inputs.len() == 2
                        && signature.inputs[0] == output
                        && signature.output == Some(1),
                    "loss signature must match forward output and return f32[1]"
                );
                target = signature.inputs[1];
            }
            ensure!(target <= 16384, "matrix loss target exceeds 16384 elements");
        }
        let has_ports = unit
            .declarations
            .lines()
            .any(|line| line.trim().starts_with("input "));
        if has_ports {
            return Ok(source.to_owned());
        }
        for line in unit.declarations.lines().map(str::trim) {
            ensure!(
                !["output ", "target ", "state ", "layer "]
                    .iter()
                    .any(|prefix| line.starts_with(prefix)),
                "matrix forward signatures cannot mix root ports, persistent state or sequential layers"
            );
        }
        let mut result = unit.declarations;
        result.push_str(&format!(
            "\ninput {}: f32[{input}]\noutput wgt_signature_output: f32[{output}]\n",
            functions["forward"].args[0]
        ));
        if let Some(loss) = functions.get("loss") {
            result.push_str(&format!("target {}: f32[{target}]\n", loss.args[1]));
        }
        for (name, body) in &unit.models {
            result.push_str(&format!(
                "\nmodel {name} {{\n{}\n}}\n",
                body.lines().skip(2).collect::<Vec<_>>().join("\n")
            ));
        }
        for (name, function) in functions
            .iter()
            .collect::<std::collections::BTreeMap<_, _>>()
        {
            crate::model_modules::emit_function(&mut result, name, function)?;
        }
        return Ok(result);
    }
    let Some(signature) = functions.get("forward").and_then(|f| f.signature.clone()) else {
        return Ok(source.to_owned());
    };
    if !functions.contains_key("loss") {
        return Ok(source.to_owned());
    }
    ensure!(
        signature.inputs.len() == 1,
        "differentiable forward currently requires one tensor input"
    );
    let output = signature
        .output
        .context("differentiable forward requires a tensor return type")?;
    for line in unit.declarations.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        ensure!(
            !["input ", "output ", "target ", "state ", "layer "]
                .iter()
                .any(|prefix| line.starts_with(prefix)),
            "signature-based forward/loss cannot mix root ports, persistent state or sequential layers; pass state explicitly"
        );
    }
    for (name, function) in &functions {
        ensure!(
            matches!(name.as_str(), "forward" | "loss") || function.signature.is_none(),
            "differentiable model helpers must be pure untyped functions: {name}"
        );
    }
    no_updates(&functions["forward"].body)?;
    let loss = &functions["loss"];
    ensure!(
        loss.args.len() == 2,
        "loss requires prediction and target arguments"
    );
    let target = if let Some(loss_type) = &loss.signature {
        ensure!(
            loss_type.inputs.len() == 2
                && loss_type.inputs[0] == output
                && loss_type.output == Some(1),
            "loss signature requires prediction matching forward output and f32[1] return"
        );
        loss_type.inputs[1]
    } else {
        output
    };
    let mut result = unit.declarations;
    result.push_str(&format!("\ninput {}: f32[{}]\noutput wgt_signature_output: f32[{output}]\ntarget {}: f32[{target}]\n",
        functions["forward"].args[0], signature.inputs[0], functions["loss"].args[1]));
    for (name, body) in &unit.models {
        result.push_str(&format!(
            "model {name} {{\n{}\n}}\n",
            body.lines().skip(2).collect::<Vec<_>>().join("\n")
        ));
    }
    functions.get_mut("forward").unwrap().signature = None;
    functions.get_mut("loss").unwrap().signature = None;
    for (name, function) in functions
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>()
    {
        crate::model_modules::emit_function(&mut result, &name, &function)?;
    }
    ensure!(
        result.len() <= 1024 * 1024,
        "normalized model exceeds 1 MiB"
    );
    Ok(result)
}

fn no_updates(statements: &[parse::Statement]) -> Result<()> {
    for statement in statements {
        match &statement.kind {
            Stmt::Eval(Expr::Call(name, _)) | Stmt::Eval(Expr::BuiltinCall(name, _)) => {
                ensure!(
                    name != "write" && name != "__transaction_write",
                    "typed forward must be pure; return next state explicitly"
                );
            }
            Stmt::If(_, yes, no) => {
                no_updates(yes)?;
                no_updates(no)?;
            }
            Stmt::While(_, body) => no_updates(body)?,
            _ => {}
        }
    }
    Ok(())
}

/// Lower an explicit tuple Forward without running initialization or losing the
/// original source signature. The IR continues to have exactly one vector port.
pub(crate) fn lower_forward(source: &str) -> Result<(String, crate::model::ForwardInterface)> {
    use crate::model::{ForwardInput, ForwardInterface, ForwardOutput};
    use parse::{Statement, Function};
    parse::validate_forward_compiler_names(source)?;
    let unit = crate::source::split(source)?;
    let mut functions = parse::parse_functions(&unit.functions)?;
    ensure!(!functions.contains_key("loss"),
        "tuple forward does not yet support fn loss; use a Python host loss or native output gradient seeds");
    let forward = functions.get("forward").context("missing pure forward")?;
    let signature = forward.forward_signature.as_ref().context("missing tuple forward signature")?;
    ensure!(functions.iter().all(|(name, f)| name == "forward" || (f.signature.is_none() && f.forward_signature.is_none())),
        "pure forward cannot contain typed operations; use pure tensor helpers");
    for line in unit.declarations.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        ensure!(!["input ", "output ", "target ", "state ", "layer "].iter().any(|p| line.starts_with(p)),
            "tuple forward cannot mix root ports, persistent state or sequential layers");
        if let Some((_, storage)) = line.split_once(' ') {
            if let Some((name, _)) = storage.split_once(':') {
                ensure!(!forward.args.iter().any(|arg| arg == name.trim()), "forward input shadows storage: {}", name.trim());
            }
        }
    }
    fn width(shape: &[usize], total: &mut usize) -> Result<usize> {
        let width = crate::tensor::validate_shape(shape, crate::tensor::ShapeLimits::REGISTER)?;
        *total = total.checked_add(width).context("forward packed width overflow")?;
        ensure!(*total <= 16384, "forward packed ports exceed 16384 f32 elements");
        Ok(width)
    }
    let mut interface = ForwardInterface { inputs: Vec::new(), outputs: Vec::new(), input: 0, output: 0, tuple: true };
    for (name, shape) in forward.args.iter().zip(&signature.inputs) {
        let offset = interface.input;
        let width = width(shape, &mut interface.input)?;
        interface.inputs.push(ForwardInput { name: name.clone(), shape: shape.clone(), offset, width });
    }
    for shape in &signature.outputs {
        let offset = interface.output;
        let width = width(shape, &mut interface.output)?;
        interface.outputs.push(ForwardOutput { shape: shape.clone(), offset, width });
    }
    fn returns(body: &[Statement]) -> bool {
        body.iter().any(|s| match &s.kind {
            Stmt::Return(_) => true,
            Stmt::If(_, yes, no) => returns(yes) && returns(no),
            _ => false,
        })
    }
    ensure!(returns(&forward.body), "pure forward must return its tensor tuple on every path");
    let mut counter = 0usize;
    fn rewrite_returns(body: &mut Vec<Statement>, outputs: &[ForwardOutput], counter: &mut usize) -> Result<()> {
        let mut result = Vec::new();
        for mut statement in std::mem::take(body) {
            match &mut statement.kind {
                Stmt::Return(expr) => {
                    let id = *counter;
                    *counter += 1;
                    let names: Vec<_> = (0..outputs.len()).map(|i| format!("wgt_multi_return_{id}_{i}")).collect();
                    result.push(Statement { line: statement.line, kind: Stmt::LetTuple(names.clone(), expr.clone()) });
                    let mut packed = None;
                    for (name, port) in names.iter().zip(outputs) {
                        let checked = format!("{name}_checked");
                        let shape = Expr::List(port.shape.iter().map(|n| Expr::Number(*n as f64)).collect());
                        result.push(Statement { line: statement.line, kind: Stmt::Let(checked.clone(),
                            Expr::Call("wgt_forward_exact".into(), vec![Expr::Name(name.clone()), shape])) });
                        let flat = format!("{name}_flat");
                        result.push(Statement { line: statement.line, kind: Stmt::Let(flat.clone(),
                            Expr::Call("reshape".into(), vec![Expr::Name(checked), Expr::List(vec![Expr::Number(port.width as f64)])])) });
                        packed = Some(match packed {
                            None => flat,
                            Some(previous) => {
                                let joined = format!("{name}_joined");
                                result.push(Statement { line: statement.line, kind: Stmt::Let(joined.clone(),
                                    Expr::Call("concat".into(), vec![Expr::Name(previous), Expr::Name(flat), Expr::Number(0.)])) });
                                joined
                            }
                        });
                    }
                    statement.kind = Stmt::Return(Expr::Name(packed.context("empty forward tuple")?));
                }
                Stmt::If(_, yes, no) => { rewrite_returns(yes, outputs, counter)?; rewrite_returns(no, outputs, counter)?; }
                Stmt::While(_, body) => rewrite_returns(body, outputs, counter)?,
                _ => {}
            }
            result.push(statement);
        }
        *body = result;
        Ok(())
    }
    let mut forward: Function = forward.clone();
    rewrite_returns(&mut forward.body, &interface.outputs, &mut counter)?;
    let mut body = Vec::new();
    for port in &interface.inputs {
        let slice = format!("wgt_multi_arg_{}", body.len());
        body.push(Statement { line: 1, kind: Stmt::Let(slice.clone(), Expr::Call("slice".into(), vec![
            Expr::Name("wgt_multi_input".into()), Expr::Number(port.offset as f64), Expr::Number(port.width as f64)])) });
        body.push(Statement { line: 1, kind: Stmt::Let(port.name.clone(), Expr::Call("reshape".into(), vec![
            Expr::Name(slice), Expr::List(port.shape.iter().map(|n| Expr::Number(*n as f64)).collect())])) });
    }
    body.extend(forward.body);
    forward.args = vec!["wgt_multi_input".into()];
    forward.signature = None;
    forward.tensor_signature = None;
    forward.forward_signature = None;
    forward.body = body;
    functions.insert("forward".into(), forward);
    let mut lowered = unit.declarations;
    lowered.push_str(&format!("\ninput wgt_multi_input: f32[{}]\noutput wgt_multi_output: f32[{}]\n", interface.input, interface.output));
    for (name, model) in &unit.models {
        lowered.push_str(&format!("\nmodel {name} {{\n{}\n}}\n", model.lines().skip(2).collect::<Vec<_>>().join("\n")));
    }
    for (name, function) in functions.iter().collect::<std::collections::BTreeMap<_,_>>() {
        crate::model_modules::emit_function(&mut lowered, name, function)?;
    }
    ensure!(lowered.len() <= 1024 * 1024, "packed forward source exceeds 1 MiB");
    // Persistent writes can be hidden behind helper calls. Validate their actual
    // lowered effects instead of relying only on surface assignment syntax.
    let unit = crate::source::split(&lowered)?;
    let functions = parse::parse_packed_forward_functions(&unit.functions)?;
    let plan = crate::adaptive::validate_definition(&lowered, &functions)?;
    ensure!(!plan.instructions.iter().any(|i| matches!(i, crate::adaptive::Instruction::Write { .. })),
        "pure tuple forward cannot write persistent parameters/state; return next state explicitly");
    Ok((lowered, interface))
}
