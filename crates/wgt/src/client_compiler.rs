//! First explicit client-only compilation slice. Unsupported constructs fail before Cargo.
use crate::client_expression::{Expressions, Mode};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;
use std::path::Path;
use wgt_language::script::{Statement, Stmt};
fn block(statements: &[Statement], expressions: &mut Expressions) -> Result<String> {
    let mut code = String::new();
    for statement in statements {
        code.push_str("rt.tick()?;");
        code.push_str(&match &statement.kind {
            Stmt::Reload(kind,target,path,optimizer)=>format!("let target={};let path={};let optimizer={};rt.reload_file({kind:?},target,path,optimizer)?;",expressions.expression(target)?,expressions.expression(path)?,optimizer.as_ref().map(|e| expressions.expression(e).map(|s|format!("Some({s})"))).transpose()?.unwrap_or_else(||"None".into())),
            Stmt::Save(kind,value,path,overwrite)=>format!("let value={};let path={};rt.save_file({kind:?},value,path,{overwrite})?;",expressions.expression(value)?,expressions.expression(path)?),
            Stmt::Tasks(body)=>{
                format!("rt.enter_tasks()?;let result=(||->Result<Flow>{{{}Ok(Flow::Next)}})();let cleanup=rt.leave_tasks(result.is_ok());let flow=result?;cleanup?;if !matches!(flow,Flow::Next){{return Ok(flow);}}",block(body, expressions)?)
            }
            Stmt::Let(n,e)=>format!("let value={};env.insert({n:?}.into(),value);",expressions.expression(e)?),
            Stmt::Set(n,e)=>format!("ensure!(env.contains_key({n:?}),\"undefined variable\");let value={};env.insert({n:?}.into(),value);",expressions.expression(e)?),
            Stmt::Collect(name,body)=>format!("let previous=env.insert(\"__collect\".into(),Value::RecordBatch(Default::default()));let result=(||->Result<Flow>{{{}Ok(Flow::Next)}})();let batch=env.remove(\"__collect\").unwrap();if let Some(old)=previous{{env.insert(\"__collect\".into(),old);}}let flow=result?;if !matches!(flow,Flow::Next){{return Ok(flow);}}env.insert({name:?}.into(),batch);",block(body, expressions)?),
            Stmt::Yield(fields)=>format!("let fields=vec![{}];rt.yield_record(env,fields)?;",fields.iter().map(|(name,e)|Ok(format!("({name:?}.into(),{})",expressions.expression(e)?))).collect::<Result<Vec<_>>>()?.join(",")),
            Stmt::ForBatches(name,batch,size,body)=>format!("let batch={};let size={}.integer()?;let Value::RecordBatch(batch)=batch else{{anyhow::bail!(\"size iteration requires records\")}};for range in wgtime_control::record_schema::chunks(batch.rows_len(),size).map_err(anyhow::Error::msg)?{{rt.tick()?;let chunk=rt.record_chunk(&batch,range)?;env.insert({name:?}.into(),chunk);match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expressions.expression(batch)?,expressions.expression(size)?,block(body, expressions)?),
            Stmt::Eval(e)=>format!("let _={};",expressions.expression(e)?),
            Stmt::Return(e)=>format!("return Ok(Flow::Return({}));",expressions.expression(e)?),
            Stmt::Break=>"return Ok(Flow::Break);".into(),
            Stmt::Continue=>"return Ok(Flow::Continue);".into(),
            Stmt::Assert(e,message)=>expressions.assertion(statement.line,e,message.as_ref())?,
            Stmt::If(e,a,b)=>format!("if {}.boolean()?{{{}}}else{{{}}}",expressions.expression(e)?,block(a, expressions)?,block(b, expressions)?),
            Stmt::For(n,e,body)=>format!("let items={}.data()?.as_array().context(\"expected list\")?.clone();for value in items{{rt.tick()?;env.insert({n:?}.into(),Value::Data(value));match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expressions.expression(e)?,block(body, expressions)?),
            Stmt::While(e,body)=>format!("while {}.boolean()?{{rt.tick()?;match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Next|Flow::Continue=>{{}},Flow::Break=>break,Flow::Return(v)=>return Ok(Flow::Return(v))}}}}",expressions.expression(e)?,block(body, expressions)?),
            _=>bail!("statement at line {} not yet migrated to client-only compiler",statement.line),
        });
    }
    Ok(code)
}
pub fn source(source: &str, root: &Path) -> Result<String> {
    source_entry(source, root, "main")
}
pub fn source_entry(source: &str, root: &Path, entry: &str) -> Result<String> {
    source_entry_with_dependencies(source, root, entry, &Default::default())
}
pub fn source_entry_with_dependencies(
    source: &str,
    root: &Path,
    entry: &str,
    dependencies: &std::collections::BTreeMap<String, String>,
) -> Result<String> {
    let program = wgt_language::script::parse_with_imports(source, root, dependencies)?;
    ensure!(
        program.contains_key(entry),
        "client compilation requires entry {entry}"
    );
    let mut entries = Vec::new();
    let mut code="#![allow(unused_imports,unused_variables,unreachable_code)]\nuse anyhow::{Result,Context,ensure};use serde_json::json;use wgtime_control::{Runtime,Env,Value,Function,number,binary};\n".to_owned();
    code.push_str("enum Flow{Next,Return(Value),Break,Continue}\n");
    code.push_str(&format!("const __STRATEGY_SOURCE:&str={source:?};\n"));
    let mut pending = BTreeSet::from([entry.to_owned()]);
    let mut emitted = BTreeSet::new();
    let mut uses_strategy = false;
    while let Some(name) = pending.pop_first() {
        if !emitted.insert(name.clone()) {
            continue;
        }
        let function = &program[&name];
        let i = entries.len();
        let mut expressions = Expressions::new(if function.is_async {
            Mode::Async
        } else {
            Mode::Sync
        });
        let async_body = if function.is_async {
            code.push_str(&crate::client_async_codegen::function(
                i,
                &function.body,
                &mut expressions,
            )?);
            code.push_str(&format!("fn f{i}(_: &mut Runtime,_:&mut Env)->Result<Value>{{anyhow::bail!(\"async requires await\")}}\n"));
            format!("Some(af{i})")
        } else {
            code.push_str(&format!("fn f{i}(rt:&mut Runtime,env:&mut Env)->Result<Value>{{match (||->Result<Flow>{{{}Ok(Flow::Next)}})()?{{Flow::Return(v)=>Ok(v),Flow::Next=>Ok(Value::Data(serde_json::Value::Null)),_=>anyhow::bail!(\"loop control outside loop\")}}}}\n",block(&function.body, &mut expressions)?));
            "None".into()
        };
        uses_strategy |= expressions.uses_strategy();
        for called in expressions.calls() {
            if program.contains_key(&called) && !emitted.contains(&called) {
                pending.insert(called);
            }
        }
        entries.push(format!(
            "({name:?},Function{{args:&{:?},body:f{i},async_body:{async_body}}})",
            function.args
        ));
    }
    ensure!(
        !uses_strategy || wgt_language::script::operators::imports(source)?.is_empty(),
        "'train ... using' requires import-free source; resident strategies do not support imports"
    );
    code.push_str(&format!(
        "fn main()->Result<()>{{wgtime_control::entry_named(vec![{}],{:?},{entry:?})}}",
        entries.join(","),
        root.to_str().context("root must be UTF8")?
    ));
    Ok(code)
}
