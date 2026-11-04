//! Compile native client futures; no embedded AST or model executor.
use anyhow::{bail, Result};
use wgt_language::script::{Expr, Statement, Stmt};
fn expression(e: &Expr) -> Result<String> {
    let value=match e {
        Expr::Number(n)=>format!("number({n:?})?"),Expr::Text(s)=>format!("Value::Data(json!({s:?}))"),Expr::Bool(b)=>format!("Value::Data(json!({b}))"),Expr::Null=>"Value::Data(serde_json::Value::Null)".into(),
        Expr::Name(n)=>format!("env.get({n:?}).cloned().context(\"undefined async variable\")?"),
        Expr::List(values)=>format!("{{let values:Vec<Value>=vec![{}];Value::Data(serde_json::Value::Array(values.iter().map(|v|v.data().cloned()).collect::<Result<Vec<_>>>()?))}}",values.iter().map(expression).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::BuiltinCall(name,args) if name=="__await"=>format!("{{let value={};wgtime_control::async_runtime::await_local(rt.clone(),value).await?}}",expression(&args[0])?),
        Expr::BuiltinCall(name,_) if name=="__spawn"=>bail!("spawn inside async unavailable"),
        Expr::Call(name,args)|Expr::BuiltinCall(name,args)=>format!("{{let args=vec![{}];let value=rt.borrow_mut().call_in_env({name:?},args,&env)?;value}}",args.iter().map(expression).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::Binary(op,a,b)=>{
            let(a,b)=(expression(a)?,expression(b)?);
            match op.as_str(){"&&"=>format!("{{if {a}.boolean()?{{Value::Data(json!({b}.boolean()?))}}else{{Value::Data(json!(false))}}}}"),"||"=>format!("{{if {a}.boolean()?{{Value::Data(json!(true))}}else{{Value::Data(json!({b}.boolean()?))}}}}"),_=>format!("{{let a={a};let b={b};binary({op:?},a,b)?}}")}
        }
        Expr::Index(a,b)=>format!("{{let a={};let b={};let key=b.data()?;let item=if let Some(name)=key.as_str(){{a.data()?.get(name)}}else{{a.data()?.get(b.integer()?)}}.context(\"index missing\")?;Value::Data(item.clone())}}",expression(a)?,expression(b)?),
        Expr::Unary(op,a)=>match op.as_str(){"!"=>format!("Value::Data(json!(!{}.boolean()?))",expression(a)?),"-"=>format!("number(-{}.number()?)?",expression(a)?),_=>bail!("unsupported unary")},
        _=>bail!("expression not migrated to client async: {e:?}")
    };
    Ok(format!("{{rt.borrow_mut().tick()?;{value}}}"))
}
fn block(body: &[Statement]) -> Result<String> {
    let mut code = String::new();
    for statement in body {
        code.push_str("rt.borrow_mut().tick()?;");
        code.push_str(&match &statement.kind {
        Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expression(e)?),
        Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"undefined variable\");let value={};env.insert({n:?}.into(),value);",expression(e)?),
        Stmt::Eval(e)=>format!("let _={};",expression(e)?),Stmt::Return(e)=>format!("return Ok({});",expression(e)?),
        Stmt::Assert(e,message)=>format!("if !{}.boolean()?{{let message={};anyhow::bail!(\"{{}}\",message.text()?);}}",expression(e)?,message.as_ref().map(expression).transpose()?.unwrap_or_else(||"Value::Data(json!(\"assertion failed\"))".into())),
        Stmt::If(e,a,b)=>format!("if {}.boolean()?{{{}}}else{{{}}}",expression(e)?,block(a)?,block(b)?),
        Stmt::While(e,body)=>format!("while {}.boolean()?{{{}}}",expression(e)?,block(body)?),
        Stmt::For(n,e,body)=>format!("let values={};for item in values.data()?.as_array().context(\"expected list\")?{{rt.borrow_mut().tick()?;env.insert({n:?}.into(),Value::Data(item.clone()));{}}}",expression(e)?,block(body)?),
        Stmt::Break=>"break;".into(),Stmt::Continue=>"continue;".into(),
        _=>bail!("statement line {} not migrated to async client",statement.line)
    });
    }
    Ok(code)
}
pub fn function(index: usize, body: &[Statement]) -> Result<String> {
    Ok(format!("fn af{index}(rt:std::rc::Rc<std::cell::RefCell<Runtime>>,mut env:Env)->wgtime_control::async_runtime::FutureValue{{Box::pin(async move{{let result:Result<Value>=async{{{}Ok(Value::Data(serde_json::Value::Null))}}.await;wgtime_control::async_runtime::plain_result(result.map_err(wgtime_control::async_runtime::task_error)?)}})}}\n",block(body)?))
}
