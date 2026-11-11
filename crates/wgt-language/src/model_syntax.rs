//! Signature-based differentiable definitions reuse the existing tensor frontend.
//! A declared loss opts into differentiation; stateful operations remain explicit.
use crate::script::parse::{self, Expr, Stmt};
use anyhow::{Context, Result, ensure};

pub(crate) fn normalize(source: &str) -> Result<String> {
    let unit = crate::source::split(source)?;
    let mut functions = parse::parse_functions(&unit.functions)?;
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
