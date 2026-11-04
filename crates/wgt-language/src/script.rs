pub mod parse;
pub mod operators;
mod modules;
pub mod prelude;
mod strategies;
pub mod agents;
pub mod compiled;
use anyhow::{ensure, Result};
use std::{collections::BTreeMap,path::Path};
pub use parse::{Expr, Program, Statement, Stmt};
pub fn parse(source: &str) -> Result<Program> {
    parse_linked(source, Program::new())
}
pub fn parse_with_imports(
    source: &str,
    root: &Path,
    dependencies: &BTreeMap<String, String>,
) -> Result<Program> {
    let (source,modules)=modules::link(source,root)?;
    let (source, functions) = operators::link(&source, root, dependencies)?;
    let mut program=parse_linked(&source, functions)?;
    for (name,function) in modules {ensure!(program.insert(name,function).is_none(),"duplicate module function");}
    Ok(program)
}
fn parse_linked(source: &str, imported: Program) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let mut program = parse::parse_functions(&unit.functions)?;
    for (name, function) in &imported {
        ensure!(
            program.insert(name.clone(), function.clone()).is_none(),
            "duplicate imported operator: {name}"
        );
    }
    let model_name = unit
        .declarations
        .lines()
        .find_map(|l| {
            l.split('#')
                .next()
                .unwrap_or("")
                .trim()
                .strip_prefix("model ")
        })
        .map(str::trim);
    if let Some(name) = model_name {
        ensure!(
            !name.starts_with("__")
                && !name.is_empty()
                && name
                    .bytes()
                    .enumerate()
                    .all(|(i, c)| c.is_ascii_alphabetic()
                        || c == b'_'
                        || (i > 0 && c.is_ascii_digit())),
            "invalid model name"
        );
        ensure!(
            !program.contains_key(name) && !prelude::is_builtin(name) && prelude::migration(name).is_none(),
            "model name conflicts with function: {name}"
        );
        let (builtin, plan) = if program.contains_key("forward") {
            let plan = crate::adaptive::parse_with_functions(source, &program)?;
            program.remove("forward");
            program.remove("loss");
            ("__adaptive", serde_json::to_string(&plan)?)
        } else {
            (
                "__graph",
                serde_json::to_string(&crate::source::graph(&unit.declarations)?.1)?,
            )
        };
        program.insert(
            name.into(),
            parse::Function {
                is_async: false,
                args: vec!["optimizer".into()],
                body: vec![Statement {
                    line: 1,
                    kind: Stmt::Return(Expr::Call(
                        builtin.into(),
                        vec![Expr::Text(plan), Expr::Name("optimizer".into())],
                    )),
                }],
            },
        );
    } else {
        ensure!(
            unit.declarations
                .lines()
                .all(|l| l.trim().is_empty() || l.trim().starts_with('#') || l.trim() == "wgt 0.5"),
            "model declarations require model NAME"
        );
        ensure!(
            !program.contains_key("forward"),
            "forward requires model declarations"
        );
    }
    if let Some(main) = program.get("main") {
        ensure!(main.args.is_empty(), "main takes no arguments");
    } else {
        ensure!(
            model_name.is_some() || program.keys().any(|name| name.starts_with("test_")),
            "program requires fn main(), test_* functions or model declarations"
        );
    }
    for name in imported.keys() {
        program.remove(name);
    }
    Ok(program)
}
