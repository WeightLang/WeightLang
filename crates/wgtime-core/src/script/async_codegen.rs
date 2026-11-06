//! AOT emits native Rust futures, never WGT/AST replay.
use super::parse::{Expr, Statement, Stmt};
use anyhow::{bail, Result};
fn expr(e: &Expr) -> Result<String> {
    let value = match e {
        Expr::Number(n)=>format!("Value::Number({n:?})"),
        Expr::Text(s)=>format!("Value::Text({s:?}.into())"),
        Expr::Bool(b)=>format!("Value::Bool({b})"),Expr::Null=>"Value::Null".into(),
        Expr::Name(n)=>format!("env.get({n:?}).cloned().context(\"undefined async variable\")?"),
        Expr::List(items)=>format!("Value::List(vec![{}])",items.iter().map(expr).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::BuiltinCall(n,args) if n=="__await"=>format!("{{ let future={}; async_runtime::await_local(rt.clone(),future).await? }}",expr(&args[0])?),
        Expr::BuiltinCall(n,_) if n=="__spawn"=>bail!("spawn inside async fn is unavailable in first native async stage"),
        Expr::BuiltinCall(n,args) if n=="__load_text"=>format!("{{ let path={}; let value=rt.borrow_mut().cold_call(\"__async_read\",vec![path])?; value }}",expr(&args[0])?),
        Expr::Call(n,args) if n=="sleep"=>{
            anyhow::ensure!(args.len()==1,"sleep expects one millisecond argument");
            format!("{{ let ms={}; let value=rt.borrow_mut().cold_call(\"__async_sleep\",vec![ms])?; value }}",expr(&args[0])?)
        }
        Expr::BuiltinCall(n,args)=>format!("{{ let args=vec![{}]; let value=rt.borrow_mut().call({n:?},args)?; value }}",args.iter().map(expr).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::Call(n,args)=>format!("{{ let args=vec![{}]; let value=rt.borrow_mut().call_in_env({n:?},args,&env)?; value }}",args.iter().map(expr).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::Unary(op,x)=>format!("compiled::unary({op:?},{})?",expr(x)?),
        Expr::Binary(op,a,b)=>{
            let (a,b)=(expr(a)?,expr(b)?);
            match op.as_str(){
                "&&"=>format!("{{ let a={a}; if !a.boolean()? {{ Value::Bool(false) }} else {{ compiled::binary(\"&&\",a,{b})? }} }}"),
                "||"=>format!("{{ let a={a}; if a.boolean()? {{ Value::Bool(true) }} else {{ compiled::binary(\"||\",a,{b})? }} }}"),
                _=>format!("{{let a={a};let b={b};compiled::binary({op:?},a,b)?}}"),
            }
        }
        Expr::Index(a,b)=>format!("{{let a={};let b={};compiled::index(a,b)?}}",expr(a)?,expr(b)?),
        _=>bail!("expression unavailable in first native async AOT stage"),
    };
    Ok(format!("{{ rt.borrow_mut().tick()?; {value} }}"))
}
fn block(body: &[Statement]) -> Result<String> {
    let mut code = String::new();
    for statement in body {
        code.push_str("rt.borrow_mut().tick()?;\n");
        let inner=match &statement.kind {
            Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expr(e)?),
            Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"assignment to undeclared variable\");let value={};env.insert({n:?}.into(),value);",expr(e)?),
            Stmt::Return(e)=>format!("return Ok({});",expr(e)?),
            Stmt::Eval(e)=>format!("let _={};",expr(e)?),
            Stmt::Assert(e,message)=>format!("if !{}.boolean()? {{let message={};anyhow::bail!(\"{{}}\",message.text()?);}}",expr(e)?,message.as_ref().map(expr).transpose()?.unwrap_or_else(||"Value::Text(\"assertion failed\".into())".into())),
            Stmt::If(e,yes,no)=>format!("if {}.boolean()? {{{}}} else {{{}}}",expr(e)?,block(yes)?,block(no)?),
            Stmt::While(e,body)=>format!("while {}.boolean()? {{{}}}",expr(e)?,block(body)?),
            Stmt::For(n,e,body)=>format!("let values={};for value in values.list()? {{env.insert({n:?}.into(),value.clone());{}}}",expr(e)?,block(body)?),
            Stmt::Break=>"break;".into(),Stmt::Continue=>"continue;".into(),
            _=>bail!("line {}: statement unavailable in first native async AOT stage",statement.line),
        };
        code.push_str(&inner);
        code.push('\n');
    }
    Ok(code)
}
pub(super) fn function(index: usize, body: &[Statement]) -> Result<String> {
    Ok(format!("fn af{index}(rt:std::rc::Rc<std::cell::RefCell<Runtime>>,mut env:Env)->async_runtime::FutureValue {{ Box::pin(async move {{ let result:Result<Value> = async {{ {} Ok(Value::Null) }}.await; let value=result.map_err(async_runtime::task_error)?;async_runtime::plain_result(value) }}) }}\n",block(body)?))
}
