//! Compile native client futures; no embedded AST or model executor.
use crate::client_expression::Expressions;
use anyhow::{Result, bail};
use wgt_language::script::{Statement, Stmt};
fn block(body: &[Statement], expressions: &mut Expressions) -> Result<String> {
    let mut code = String::new();
    for statement in body {
        code.push_str("rt.borrow_mut().tick()?;");
        code.push_str(&match &statement.kind {
        Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expressions.expression(e)?),
        Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"undefined variable\");let value={};env.insert({n:?}.into(),value);",expressions.expression(e)?),
        Stmt::Eval(e)=>format!("let _={};",expressions.expression(e)?),Stmt::Return(e)=>format!("return Ok({});",expressions.expression(e)?),
        Stmt::Assert(e,message)=>expressions.assertion(statement.line,e,message.as_ref())?,
        Stmt::If(e,a,b)=>format!("if {}.boolean()?{{{}}}else{{{}}}",expressions.expression(e)?,block(a, expressions)?,block(b, expressions)?),
        Stmt::While(e,body)=>format!("while {}.boolean()?{{{}}}",expressions.expression(e)?,block(body, expressions)?),
        Stmt::For(n,e,body)=>format!("let values={};for item in values.data()?.as_array().context(\"expected list\")?{{rt.borrow_mut().tick()?;env.insert({n:?}.into(),Value::Data(item.clone()));{}}}",expressions.expression(e)?,block(body, expressions)?),
        Stmt::Break=>"break;".into(),Stmt::Continue=>"continue;".into(),
        _=>bail!("statement line {} not migrated to async client",statement.line)
    });
    }
    Ok(code)
}
pub fn function(index: usize, body: &[Statement], expressions: &mut Expressions) -> Result<String> {
    Ok(format!(
        "fn af{index}(rt:std::rc::Rc<std::cell::RefCell<Runtime>>,mut env:Env)->wgtime_control::async_runtime::FutureValue{{Box::pin(async move{{let result:Result<Value>=async{{{}Ok(Value::Data(serde_json::Value::Null))}}.await;wgtime_control::async_runtime::plain_result(result.map_err(wgtime_control::async_runtime::task_error)?)}})}}\n",
        block(body, expressions)?
    ))
}
