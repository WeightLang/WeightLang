//! Signature-based differentiable definitions reuse the existing tensor frontend.
//! A declared loss opts into differentiation; stateful operations remain explicit.
use crate::script::parse::{self, Expr, Stmt};
use anyhow::{Context, Result, ensure};

pub(crate) fn normalize(source: &str) -> Result<String> {
    let unit = crate::source::split(source)?;
    let mut functions = parse::parse_functions(&unit.functions)?;
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
