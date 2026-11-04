//! Ordinary script modules, distinct from pure tensor operator libraries.
use super::parse::{self, Expr, Program, Statement, Stmt};
use anyhow::{ensure, Result};
use std::{collections::HashMap, path::Path};

pub(super) fn load(path: &str, root: &Path) -> Result<Option<Program>> {
    let source = if let Some(relative) = path.strip_prefix("./") {
        ensure!(
            Path::new(relative)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "module path must stay under source directory"
        );
        let root = root.canonicalize()?;
        let file = root.join(relative).canonicalize()?;
        ensure!(file.starts_with(root), "module escapes source directory");
        ensure!(
            std::fs::metadata(&file)?.len() <= 1024 * 1024,
            "module exceeds 1 MiB"
        );
        std::fs::read_to_string(file)?
    } else {
        anyhow::bail!("unsupported script module import: {path}");
    };
    let unit = crate::source::split(&source)?;
    let declarations = unit
        .declarations
        .lines()
        .map(|l| l.split('#').next().unwrap().trim())
        .filter(|l| !l.is_empty() && *l != "wgt 0.5")
        .collect::<Vec<_>>();
    if !declarations.iter().any(|l| l.starts_with("module ")) {
        return Ok(None);
    }
    ensure!(
        declarations.len() == 1,
        "script module permits only module NAME and functions"
    );
    ensure!(
        super::operators::identifier(declarations[0].strip_prefix("module ").unwrap()),
        "invalid module name"
    );
    let functions = parse::parse_functions(&unit.functions)?;
    for name in functions.keys() {
        ensure!(
            super::operators::identifier(name) && !super::prelude::is_builtin(name),
            "reserved module function name: {name}"
        );
    }
    ensure!(
        !functions.contains_key("main")
            && !functions.contains_key("forward")
            && !functions.contains_key("loss"),
        "module cannot declare entrypoints"
    );
    Ok(Some(functions))
}
pub(super) fn link(source: &str, root: &Path) -> Result<(String, Program)> {
    let mut linked = Program::new();
    let mut aliases = Vec::new();
    for (alias, path) in super::operators::imports(source)? {
        if let Some(functions) = load(&path, root)? {
            aliases.push(alias.clone());
            let calls = functions
                .keys()
                .map(|n| (n.clone(), format!("{alias}.{n}")))
                .collect::<HashMap<_, _>>();
            for (name, mut function) in functions {
                for statement in &mut function.body {
                    rewrite_stmt(statement, &calls);
                }
                linked.insert(calls[&name].clone(), function);
            }
        }
    }
    let unit = crate::source::split(source)?;
    let stripped = source
        .lines()
        .zip(unit.declarations.split('\n'))
        .map(|(line, declaration)| {
            if aliases.iter().any(|alias| {
                declaration
                    .trim()
                    .starts_with(&format!("import {alias} from "))
            }) {
                ""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok((stripped, linked))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_module_keeps_async_functions_and_hygienic_helpers() {
        let root =
            std::env::temp_dir().join(format!("weightlang-module-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("helper.wgt");
        std::fs::write(
            &path,
            "wgt 0.5\nmodule Helpers\nfn local(x){return x+1;}\nasync fn work(x){return local(x);}",
        )
        .unwrap();
        let (_, program) = link(
            "wgt 0.5\nimport dev from \"./helper.wgt\"\nfn main(){return 0;}",
            &root,
        )
        .unwrap();
        assert!(program["dev.work"].is_async);
        let Stmt::Return(Expr::Call(name, _)) = &program["dev.work"].body[0].kind else {
            panic!("missing rewritten call")
        };
        assert_eq!(name, "dev.local");
        std::fs::write(
            &path,
            "wgt 0.5\nmodule Helpers\nimport cycle from \"./helper.wgt\"\nfn helper(){return 1;}",
        )
        .unwrap();
        assert!(link(
            "wgt 0.5\nimport dev from \"./helper.wgt\"\nfn main(){return 0;}",
            &root
        )
        .is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}

fn rewrite_expr(expr: &mut Expr, calls: &HashMap<String, String>) {
    match expr {
        Expr::Call(name, args) | Expr::StaticCall(name, args) => {
            if let Some(mapped) = calls.get(name) {
                *name = mapped.clone();
            }
            for arg in args {
                rewrite_expr(arg, calls);
            }
        }
        Expr::BuiltinCall(_, args) | Expr::List(args) => {
            for arg in args {
                rewrite_expr(arg, calls);
            }
        }
        Expr::Index(a, b) | Expr::Binary(_, a, b) => {
            rewrite_expr(a, calls);
            rewrite_expr(b, calls);
        }
        Expr::Slice(a, b, c) => {
            rewrite_expr(a, calls);
            for e in b.iter_mut().chain(c.iter_mut()) {
                rewrite_expr(e, calls);
            }
        }
        Expr::Unary(_, a) | Expr::SampleFrom(_, a) => rewrite_expr(a, calls),
        Expr::Sample(a, b, c) | Expr::Mix(a, b, c) => {
            rewrite_expr(a, calls);
            rewrite_expr(b, calls);
            rewrite_expr(c, calls);
        }
        _ => {}
    }
}
fn rewrite_stmt(statement: &mut Statement, calls: &HashMap<String, String>) {
    match &mut statement.kind {
        Stmt::Let(_, e)
        | Stmt::Set(_, e)
        | Stmt::Eval(e)
        | Stmt::Return(e)
        | Stmt::LetSample(_, _, e) => rewrite_expr(e, calls),
        Stmt::Assert(e, message) => {
            rewrite_expr(e, calls);
            if let Some(e) = message {
                rewrite_expr(e, calls);
            }
        }
        Stmt::If(e, a, b) => {
            rewrite_expr(e, calls);
            for s in a.iter_mut().chain(b) {
                rewrite_stmt(s, calls);
            }
        }
        Stmt::While(e, body) | Stmt::For(_, e, body) | Stmt::Learn(_, e, _, body) => {
            rewrite_expr(e, calls);
            for s in body {
                rewrite_stmt(s, calls);
            }
        }
        Stmt::Tasks(body) | Stmt::Collect(_, body) => {
            for s in body {
                rewrite_stmt(s, calls);
            }
        }
        Stmt::ForBatches(_, a, b, body) => {
            rewrite_expr(a, calls);
            rewrite_expr(b, calls);
            for s in body {
                rewrite_stmt(s, calls);
            }
        }
        Stmt::Yield(fields) => {
            for (_, e) in fields {
                rewrite_expr(e, calls);
            }
        }
        Stmt::Save(_, a, b, _) => {
            rewrite_expr(a, calls);
            rewrite_expr(b, calls);
        }
        Stmt::Reload(_, a, b, c) => {
            rewrite_expr(a, calls);
            rewrite_expr(b, calls);
            if let Some(e) = c {
                rewrite_expr(e, calls);
            }
        }
        _ => {}
    }
}
