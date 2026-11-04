//! First explicit client-only compilation slice. Unsupported constructs fail before Cargo.
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;
use weightlang_runtime::script::{Expr, Statement, Stmt};
fn expression(e: &Expr) -> Result<String> {
    let value = match e {
        Expr::Number(n)=>format!("number({n:?})?"),
        Expr::Text(s)=>format!("Value::Data(json!({s:?}))"),
        Expr::Bool(b)=>format!("Value::Data(json!({b}))"),
        Expr::Null=>"Value::Data(serde_json::Value::Null)".into(),
        Expr::Name(n)=>format!("env.get({n:?}).cloned().context(\"undefined variable\")?"),
        Expr::List(values)=>format!("{{let values=vec![{}];Value::Data(serde_json::Value::Array(values.iter().map(|v:&Value|v.data().cloned()).collect::<Result<Vec<_>>>()?))}}",values.iter().map(expression).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::BuiltinCall(name,args) if name=="__load_model"||name=="__load_checkpoint" => format!("{{let path={};let optimizer={};rt.load_file({:?},path,optimizer)?}}",expression(&args[0])?,if args.len()>1{format!("Some({})",expression(&args[1])?)}else{"None".into()},if name=="__load_checkpoint"{"checkpoint"}else{"model"}),
        Expr::BuiltinCall(name,args) if name=="__await"||name=="__spawn" => format!("{{let value={};rt.{}(value)?}}",expression(&args[0])?,if name=="__await"{"await_value"}else{"spawn_future"}),
        Expr::BuiltinCall(name,args)=>format!("{{let args=vec![{}];rt.call({name:?},args)?}}",args.iter().map(expression).collect::<Result<Vec<_>>>()?.join(",")),
        Expr::StaticCall(name,args)=> {
            format!("{{let args=vec![Value::Data(json!(__STRATEGY_SOURCE)),Value::Data(json!({name:?})),{}];rt.call(\"__strategy\",args)?}}",args.iter().map(expression).collect::<Result<Vec<_>>>()?.join(","))
        }
        Expr::Call(name,args)=>{
            ensure!(!name.starts_with("__") || matches!(name.as_str(),"__graph"|"__adaptive"),"unsupported internal constructor");
            format!("{{let args=vec![{}];rt.call_in_env({name:?},args,env)?}}",args.iter().map(expression).collect::<Result<Vec<_>>>()?.join(","))
        }
        Expr::Binary(op,a,b)=>{let a=expression(a)?;let b=expression(b)?;match op.as_str(){"&&"=>format!("{{if {a}.boolean()?{{{b}}}else{{Value::Data(json!(false))}}}}"),"||"=>format!("{{if {a}.boolean()?{{Value::Data(json!(true))}}else{{{b}}}}}"),_=>format!("{{let a={a};let b={b};binary({op:?},a,b)?}}")}},
        Expr::Unary(op,e)=>match op.as_str(){"!"=>format!("Value::Data(json!(!{}.boolean()?))",expression(e)?),"-"=>format!("number(-{}.number()?)?",expression(e)?),_=>bail!("unsupported unary")},
        Expr::Index(a,b)=>format!("{{let a={};let b={};wgtime_control::records::index(a,b)?}}",expression(a)?,expression(b)?),
        _=>bail!("construct not yet migrated to client-only compiler: {e:?}"),
    };
    Ok(format!("{{rt.tick()?;{value}}}"))
}
fn block(statements: &[Statement]) -> Result<String> {
    let mut code = String::new();
    for statement in statements {
        code.push_str("rt.tick()?;");
        code.push_str(&match &statement.kind {
            Stmt::Save(kind,value,path,overwrite)=>format!("let value={};let path={};rt.save_file({kind:?},value,path,{overwrite})?;",expression(value)?,expression(path)?),
            Stmt::Tasks(body)=>{
                format!("rt.enter_tasks()?;let result=(||->Result<Flow>{{{}Ok(Flow::Next)}})();let cleanup=rt.leave_tasks(result.is_ok());let flow=result?;cleanup?;if !matches!(flow,Flow::Next){{return Ok(flow);}}",block(body)?)
            }
            Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expression(e)?),
            Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"undefined variable\");let value={};env.insert({n:?}.into(),value);",expression(e)?),
            Stmt::Collect(name,body)=>format!("let previous=env.insert(\"__collect\".into(),Value::RecordBatch(Default::default()));let result=(||->Result<Flow>{{{}Ok(Flow::Next)}})();let batch=env.remove(\"__collect\").unwrap();if let Some(old)=previous{{env.insert(\"__collect\".into(),old);}}let flow=result?;if !matches!(flow,Flow::Next){{return Ok(flow);}}env.insert({name:?}.into(),batch);",block(body)?),
            Stmt::Yield(fields)=>format!("let fields=vec![{}];rt.yield_record(env,fields)?;",fields.iter().map(|(name,e)|Ok(format!("({name:?}.into(),{})",expression(e)?))).collect::<Result<Vec<_>>>()?.join(",")),
            Stmt::ForBatches(name,batch,size,body)=>format!("let batch={};let size={}.integer()?;let Value::RecordBatch(batch)=batch else{{anyhow::bail!(\"size iteration requires records\")}};for range in wgtime_control::record_schema::chunks(batch.rows_len(),size).map_err(anyhow::Error::msg)?{{rt.tick()?;let chunk=rt.record_chunk(&batch,range)?;env.insert({name:?}.into(),chunk);match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expression(batch)?,expression(size)?,block(body)?),
            Stmt::Eval(e)=>format!("let _={};",expression(e)?),
            Stmt::Return(e)=>format!("return Ok(Flow::Return({}));",expression(e)?),
            Stmt::Break=>"return Ok(Flow::Break);".into(),
            Stmt::Continue=>"return Ok(Flow::Continue);".into(),
            Stmt::Assert(e,message)=>format!("if !{}.boolean()?{{anyhow::bail!(\"{{}}\",{}.text()?);}}",expression(e)?,message.as_ref().map(expression).transpose()?.unwrap_or_else(||"Value::Data(json!(\"assertion failed\"))".into())),
            Stmt::If(e,a,b)=>format!("if {}.boolean()?{{{}}}else{{{}}}",expression(e)?,block(a)?,block(b)?),
            Stmt::For(n,e,body)=>format!("let items={}.data()?.as_array().context(\"expected list\")?.clone();for value in items{{rt.tick()?;env.insert({n:?}.into(),Value::Data(value));match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expression(e)?,block(body)?),
            Stmt::While(e,body)=>format!("while {}.boolean()?{{rt.tick()?;match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expression(e)?,block(body)?),
            _=>bail!("statement at line {} not yet migrated to client-only compiler",statement.line),
        });
    }
    Ok(code)
}
pub fn source(source: &str, root: &Path) -> Result<String> {
    let program =
        weightlang_runtime::script::parse_with_imports(source, root, &Default::default())?;
    ensure!(
        program.contains_key("main"),
        "client compilation requires main"
    );
    let mut entries = Vec::new();
    let mut code="#![allow(unused_imports,unused_variables,unreachable_code)]\nuse anyhow::{Result,Context,ensure};use serde_json::json;use wgtime_control::{Runtime,Env,Value,Function,number,binary};\n".to_owned();
    code.push_str("enum Flow{Next,Return(Value),Break,Continue}\n");
    code.push_str(&format!("const __STRATEGY_SOURCE:&str={source:?};\n"));
    let mut resident = std::collections::HashSet::new();
    fn resident_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::StaticCall(n, _) => {
                out.insert(n.clone());
            }
            Expr::Call(_, args) | Expr::List(args) => {
                for e in args {
                    resident_expr(e, out)
                }
            }
            Expr::Binary(_, a, b) | Expr::Index(a, b) => {
                resident_expr(a, out);
                resident_expr(b, out);
            }
            _ => {}
        }
    }
    fn resident_block(body: &[Statement], out: &mut std::collections::HashSet<String>) {
        for s in body {
            match &s.kind {
                Stmt::Eval(e) | Stmt::Return(e) | Stmt::Let(_, e) | Stmt::Set(_, e) => {
                    resident_expr(e, out)
                }
                Stmt::For(_, _, b) | Stmt::While(_, b) => resident_block(b, out),
                Stmt::If(_, a, b) => {
                    resident_block(a, out);
                    resident_block(b, out);
                }
                _ => {}
            }
        }
    }
    for f in program.values() {
        resident_block(&f.body, &mut resident);
    }
    // The service owns strategy execution; omit its entire reachable user-call graph.
    loop {
        let mut added = resident.clone();
        for name in &resident {
            if let Some(f) = program.get(name) {
                fn calls(e: &Expr, out: &mut std::collections::HashSet<String>) {
                    match e {
                        Expr::Call(n, args) => {
                            out.insert(n.clone());
                            for e in args {
                                calls(e, out)
                            }
                        }
                        Expr::List(args) => {
                            for e in args {
                                calls(e, out)
                            }
                        }
                        Expr::Index(a, b) | Expr::Binary(_, a, b) => {
                            calls(a, out);
                            calls(b, out)
                        }
                        _ => {}
                    }
                }
                for s in &f.body {
                    match &s.kind {
                        Stmt::Eval(e) | Stmt::Return(e) | Stmt::Let(_, e) | Stmt::Set(_, e) => {
                            calls(e, &mut added)
                        }
                        _ => {}
                    }
                }
            }
        }
        if added == resident {
            break;
        }
        resident = added;
    }
    let mut functions: Vec<_> = program
        .iter()
        .filter(|(n, _)| !resident.contains(*n))
        .collect();
    functions.sort_by_key(|(n, _)| *n);
    for (i, (name, function)) in functions.iter().enumerate() {
        let async_body = if function.is_async {
            code.push_str(&crate::client_async_codegen::function(i, &function.body)?);
            code.push_str(&format!("fn f{i}(_: &mut Runtime,_:&mut Env)->Result<Value>{{anyhow::bail!(\"async requires await\")}}\n"));
            format!("Some(af{i})")
        } else {
            code.push_str(&format!("fn f{i}(rt:&mut Runtime,env:&mut Env)->Result<Value>{{match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Return(v)=>Ok(v),Flow::Next=>Ok(Value::Data(serde_json::Value::Null)),_=>anyhow::bail!(\"loop control outside loop\")}}}}\n",block(&function.body)?));
            "None".into()
        };
        entries.push(format!(
            "({name:?},Function{{args:&{:?},body:f{i},async_body:{async_body}}})",
            function.args
        ));
    }
    code.push_str(&format!(
        "fn main()->Result<()>{{wgtime_control::entry_at(vec![{}],{:?})}}",
        entries.join(","),
        root.to_str().context("root must be UTF8")?
    ));
    Ok(code)
}
