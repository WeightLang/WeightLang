//! Lower script functions, expressions and control flow to Rust, retaining dynamic value semantics.
use super::parse::{Expr, Statement, Stmt};
use anyhow::Result;

fn expr(e: &Expr) -> String {
    let value = match e {
        Expr::Number(n) => format!("Value::Number({n:?})"),
        Expr::Text(s) => format!("Value::Text({s:?}.into())"),
        Expr::Bool(b) => format!("Value::Bool({b})"),
        Expr::Null => "Value::Null".into(),
        Expr::Name(n) => format!(
            "env.get({n:?}).cloned().context({:?})?",
            format!("undefined variable: {n}")
        ),
        Expr::List(v) => format!(
            "Value::List(vec![{}])",
            v.iter().map(expr).collect::<Vec<_>>().join(",")
        ),
        Expr::Call(n, args) => format!(
            "{{ let args = vec![{}]; rt.call({n:?}, args)? }}",
            args.iter().map(expr).collect::<Vec<_>>().join(",")
        ),
        Expr::Index(a, b) => format!(
            "{{ let a = {}; let b = {}; compiled::index(a,b)? }}",
            expr(a),
            expr(b)
        ),
        Expr::Unary(op, v) => format!("compiled::unary({op:?}, {})?", expr(v)),
        Expr::Binary(op, a, b) => {
            let a = expr(a);
            let b = expr(b);
            match op.as_str() {
                "&&" => format!("{{ let a = {a}; if !a.boolean()? {{ Value::Bool(false) }} else {{ let b = {b}; compiled::binary(\"&&\", a,b)? }} }}"),
                "||" => format!("{{ let a = {a}; if a.boolean()? {{ Value::Bool(true) }} else {{ let b = {b}; compiled::binary(\"||\", a,b)? }} }}"),
                _ => format!("{{ let a = {a}; let b = {b}; compiled::binary({op:?},a,b)? }}"),
            }
        }
    };
    format!("{{ rt.tick()?; {value} }}")
}
fn block(body: &[Statement]) -> String {
    let mut code = String::from("(|| -> Result<Flow> {\n");
    for statement in body {
        let inner = match &statement.kind {
            Stmt::Let(n,e) => format!("let value = {}; env.insert({n:?}.into(),value);",expr(e)),
            Stmt::Set(n,e) => format!("ensure!(env.contains_key({n:?}), {:?}); let value = {}; env.insert({n:?}.into(),value);", format!("assignment to undeclared variable: {n}"), expr(e)),
            Stmt::Eval(e) => format!("let _ = {};",expr(e)),
            Stmt::Return(e) => format!("return Ok(Flow::Return({}));",expr(e)),
            Stmt::Break => "return Ok(Flow::Break);".into(),
            Stmt::Continue => "return Ok(Flow::Continue);".into(),
            Stmt::If(e,yes,no) => format!("return if {}.boolean()? {{ {} }} else {{ {} }};",expr(e),block(yes),block(no)),
            Stmt::While(e,body) => format!("while {}.boolean()? {{ rt.tick()?; match {}? {{ Flow::Break => break, Flow::Return(v) => return Ok(Flow::Return(v)), _ => {{}} }} }}",expr(e),block(body)),
            Stmt::For(n,e,body) => format!("let values = {}; for value in values.list()? {{ rt.tick()?; env.insert({n:?}.into(),value.clone()); match {}? {{ Flow::Break => break, Flow::Return(v) => return Ok(Flow::Return(v)), _ => {{}} }} }}",expr(e),block(body)),
        };
        code.push_str(&format!("rt.tick()?;\nlet flow = (|| -> Result<Flow> {{ {inner}\n#[allow(unreachable_code)] Ok(Flow::Next) }})().context(\"line {}\")?;\nif !matches!(flow, Flow::Next) {{ return Ok(flow); }}\n",statement.line));
    }
    code.push_str("Ok(Flow::Next) })()");
    code
}
/// Generated source contains executable Rust control flow, never embedded WGT or an AST.
pub fn rust_source(source: &str, root: &str) -> Result<String> {
    rust_source_with_dependencies(source, root, &std::collections::BTreeMap::new())
}
pub fn rust_source_with_dependencies(
    source: &str,
    root: &str,
    dependencies: &std::collections::BTreeMap<String, String>,
) -> Result<String> {
    let program = super::parse(source)?;
    anyhow::ensure!(program.contains_key("main"), "compilation requires fn main()");
    let mut functions: Vec<_> = program.iter().collect();
    functions.sort_by_key(|(name, _)| *name);
    let mut code = String::from("#![allow(unused_imports,unused_variables,unused_mut)]\nuse anyhow::{Result,Context,ensure};\nuse weightlang_runtime::script::{Runtime,Env,Flow,Value,compiled};\n");
    let mut entries = Vec::new();
    for (i, (name, f)) in functions.iter().enumerate() {
        code.push_str(&format!(
            "fn f{i}(rt: &mut Runtime, env: &mut Env) -> Result<Flow> {{ {} }}\n",
            block(&f.body)
        ));
        entries.push(format!(
            "({name:?}, compiled::Function {{ args: vec![{}], body: f{i} }})",
            f.args
                .iter()
                .map(|a| format!("{a:?}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    code.push_str(&format!(
        "fn main() -> Result<()> {{ compiled::entry_with_dependencies(vec![{}], {root:?}, vec![{}]) }}\n",
        entries.join(","), dependencies.iter().map(|(k,v)| format!("({k:?},{v:?})")).collect::<Vec<_>>().join(",")
    ));
    Ok(code)
}
