//! Validate syntax-bound training strategies after collecting all user declarations.
use super::parse::{Expr, Program, Statement, Stmt};
use anyhow::{ensure, Context, Result};

pub(super) fn validate(program: &Program) -> Result<()> {
    fn expression(expr: &Expr, program: &Program) -> Result<()> {
        match expr {
            Expr::StaticCall(name, args) => {
                ensure!(
                    !super::prelude::is_builtin(name),
                    "train strategy must be a declared user function: {name}"
                );
                let function = program
                    .get(name)
                    .with_context(|| format!("unknown train strategy: {name}"))?;
                ensure!(
                    function.args.len() == 3,
                    "train strategy {name} must have exactly three parameters"
                );
                for arg in args {
                    expression(arg, program)?;
                }
            }
            Expr::Call(_, args) | Expr::BuiltinCall(_, args) | Expr::List(args) => {
                for arg in args {
                    expression(arg, program)?;
                }
            }
            Expr::Index(a, b) | Expr::Binary(_, a, b) => {
                expression(a, program)?;
                expression(b, program)?;
            }
            Expr::Slice(x, a, b) => {
                expression(x, program)?;
                for e in a.iter().chain(b.iter()) {
                    expression(e, program)?;
                }
            }
            Expr::Unary(_, x) | Expr::SampleFrom(_, x) => expression(x, program)?,
            Expr::Sample(p, a, b) | Expr::Mix(p, a, b) => {
                for e in [p, a, b] {
                    expression(e, program)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn block(body: &[Statement], program: &Program) -> Result<()> {
        for statement in body {
            match &statement.kind {
                Stmt::Let(_, e) | Stmt::LetSample(_,_,e)
                | Stmt::Set(_, e)
                | Stmt::Eval(e)
                | Stmt::Return(e) => expression(e, program)?,
                Stmt::If(e, a, b) => {
                    expression(e, program)?;
                    block(a, program)?;
                    block(b, program)?;
                }
                Stmt::While(e, b) | Stmt::For(_, e, b) => {
                    expression(e, program)?;
                    block(b, program)?;
                }
                Stmt::Assert(e,message) => { expression(e,program)?; if let Some(m)=message { expression(m,program)?; } }
                Stmt::Collect(_,body) => block(body,program)?,
                Stmt::Yield(fields) => { for (_,e) in fields { expression(e,program)?; } }
                Stmt::ForBatches(_,a,b,body) => {expression(a,program)?;expression(b,program)?;block(body,program)?;}
                Stmt::Save(_,a,b,_) => { expression(a,program)?; expression(b,program)?; }
                Stmt::Reload(_,a,b,optimizer) => { expression(a,program)?; expression(b,program)?; if let Some(e)=optimizer {expression(e,program)?;} }
                Stmt::Break | Stmt::Continue => {}
            }
        }
        Ok(())
    }
    for function in program.values() {
        block(&function.body, program)?;
    }
    Ok(())
}
