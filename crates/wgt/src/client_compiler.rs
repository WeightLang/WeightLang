//! First explicit client-only compilation slice. Unsupported constructs fail before Cargo.
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;
use weightlang_runtime::script::{Expr, Statement, Stmt};
fn expression(e: &Expr) -> Result<String> {
    Ok(match e {
        Expr::Number(n)=>format!("number({n:?})?"),
        Expr::Text(s)=>format!("Value::Data(json!({s:?}))"),
        Expr::Bool(b)=>format!("Value::Data(json!({b}))"),
        Expr::Null=>"Value::Data(serde_json::Value::Null)".into(),
        Expr::Name(n)=>format!("env.get({n:?}).cloned().context(\"undefined variable\")?"),
        Expr::List(values)=>format!("Value::Data(json!([{}]))",values.iter().map(|e|Ok(format!("{}.data()?",expression(e)?))).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::Call(name,args)=>{
            ensure!(!name.starts_with("__"),"client-only compiler does not yet support model declaration constructors");
            format!("rt.call_in_env({name:?},vec![{}],env)?",args.iter().map(expression).collect::<Result<Vec<_>>>()?.join(","))
        }
        Expr::Binary(op,a,b)=>format!("binary({op:?},{},{})?",expression(a)?,expression(b)?),
        Expr::Index(a,b)=>format!("Value::Data({}.data()?.get({}.integer()?).context(\"index out of bounds\")?.clone())",expression(a)?,expression(b)?),
        _=>bail!("construct not yet migrated to client-only compiler: {e:?}"),
    })
}
fn block(statements:&[Statement])->Result<String>{
    let mut code=String::new();
    for statement in statements {
        code.push_str("rt.tick()?;");
        code.push_str(&match &statement.kind {
            Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expression(e)?),
            Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"undefined variable\");let value={};env.insert({n:?}.into(),value);",expression(e)?),
            Stmt::Eval(e)=>format!("let _={};",expression(e)?),
            Stmt::Return(e)=>format!("return Ok({});",expression(e)?),
            Stmt::Assert(e,_)=>format!("ensure!({}.boolean()?,\"assertion failed\");",expression(e)?),
            Stmt::If(e,a,b)=>format!("if {}.boolean()?{{{}}}else{{{}}}",expression(e)?,block(a)?,block(b)?),
            Stmt::For(n,e,body)=>format!("let items={}.data()?.as_array().context(\"expected list\")?.clone();for value in items{{env.insert({n:?}.into(),Value::Data(value));{}}}",expression(e)?,block(body)?),
            Stmt::While(e,body)=>format!("while {}.boolean()?{{rt.tick()?;{}}}",expression(e)?,block(body)?),
            _=>bail!("statement at line {} not yet migrated to client-only compiler",statement.line),
        });
    }
    Ok(code)
}
pub fn source(source:&str,root:&Path)->Result<String>{
    let program=weightlang_runtime::script::parse_with_imports(source,root,&Default::default())?;
    ensure!(program.contains_key("main"),"client compilation requires main");
    let mut entries=Vec::new();
    let mut code="#![allow(unused_imports,unused_variables,unreachable_code)]\nuse anyhow::{Result,Context,ensure};use serde_json::json;use wgtime_control::{Runtime,Env,Value,Function,number,binary};\n".to_owned();
    let mut functions:Vec<_>=program.iter().collect();functions.sort_by_key(|(n,_)|*n);
    for (i,(name,function)) in functions.iter().enumerate(){
        ensure!(!function.is_async,"async not yet migrated to client-only compiler");
        code.push_str(&format!("fn f{i}(rt:&mut Runtime,env:&mut Env)->Result<Value>{{{}Ok(Value::Data(serde_json::Value::Null))}}\n",block(&function.body)?));
        entries.push(format!("({name:?},Function{{args:&{:?},body:f{i}}})",function.args));
    }
    code.push_str(&format!("fn main()->Result<()>{{wgtime_control::entry(vec![{}])}}",entries.join(",")));
    Ok(code)
}
