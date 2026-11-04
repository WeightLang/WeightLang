//! Client control only. Model state and execution live behind wgtime IPC.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::path::Path;
use wgtime::{Client, Request};

#[derive(Clone, Debug)]
pub struct ModelLease { handle: u64 }
#[derive(Clone, Debug)]
pub enum Value {
    Data(Json),
    Model(ModelLease),
}
impl Value {
    pub fn data(&self) -> Result<&Json> { match self { Self::Data(value) => Ok(value), _ => bail!("resource handle cannot be converted to JSON") } }
    pub fn number(&self) -> Result<f64> { self.data()?.as_f64().context("expected number") }
    pub fn boolean(&self) -> Result<bool> { self.data()?.as_bool().context("expected bool") }
    pub fn text(&self) -> Result<&str> { self.data()?.as_str().context("expected string") }
    pub fn integer(&self) -> Result<usize> { let n=self.number()?; ensure!(n>=0. && n<=1e9 && n.fract()==0.,"expected nonnegative integer"); Ok(n as usize) }
}
pub type Env = HashMap<String, Value>;
pub struct Runtime { client: Client, remaining: u64, pub metrics: serde_json::Map<String,Json>, functions: HashMap<&'static str, Function> }
#[derive(Clone)]
pub struct Function { pub args: &'static [&'static str], pub body: fn(&mut Runtime,&mut Env)->Result<Value> }
impl Runtime {
    pub fn tick(&mut self) -> Result<()> { ensure!(self.remaining>0,"operation budget exceeded"); self.remaining-=1; Ok(()) }
    pub fn call(&mut self,name:&str,args:Vec<Value>)->Result<Value> {
        self.tick()?;
        if let Some(function)=self.functions.get(name).cloned() {
            ensure!(function.args.len()==args.len(),"wrong argument count for {name}");
            let mut env=function.args.iter().zip(args).map(|(n,v)|((*n).into(),v)).collect();
            return (function.body)(self,&mut env);
        }
        match name {
            "mlp" => {
                ensure!(args.len()==4,"mlp requires four arguments");
                let reply=self.client.call(Request::Create { widths:serde_json::from_value(args[0].data()?.clone())?, activation:args[1].text()?.into(), seed:args[2].integer()? as u64, optimizer:args[3].text()?.into(), memory:256*1024*1024 })?;
                Ok(Value::Model(ModelLease {handle:reply["handle"].as_u64().context("missing handle")?}))
            }
            "fit" => {
                ensure!(args.len()==5 && args[4].text()?=="mse","client first slice supports fit MSE only");
                let Value::Model(model)=&args[0] else {bail!("expected model lease")};
                let reply=self.client.call(Request::Train {handle:model.handle,inputs:serde_json::from_value(args[1].data()?.clone())?,targets:serde_json::from_value(args[2].data()?.clone())?,rate:args[3].number()? as f32})?;
                Ok(Value::Data(reply["loss"].clone()))
            }
            "metric" => {ensure!(args.len()==2,"metric requires two args");self.metrics.insert(args[0].text()?.into(),args[1].data()?.clone());Ok(Value::Data(Json::Null))}
            "len" => {ensure!(args.len()==1,"len requires one arg"); Ok(Value::Data(json!(args[0].data()?.as_array().context("expected list")?.len())))}
            "range" => {ensure!(args.len()==2,"range requires two args");let start=args[0].integer()?;let end=args[1].integer()?;ensure!(end>=start&&end-start<=1_000_000,"range limit");Ok(Value::Data(json!((start..end).collect::<Vec<_>>())))}
            _ => bail!("unsupported client builtin: {name}"),
        }
    }
    pub fn call_in_env(&mut self,name:&str,args:Vec<Value>,env:&Env)->Result<Value> {
        if let Some(Value::Model(model))=env.get(name) {ensure!(args.len()==1,"model requires one input");let reply=self.client.call(Request::Predict {handle:model.handle,input:serde_json::from_value(args[0].data()?.clone())?})?;return Ok(Value::Data(reply["output"].clone()));}
        self.call(name,args)
    }
}
pub fn number(n:f64)->Result<Value>{ensure!(n.is_finite(),"nonfinite arithmetic");Ok(Value::Data(json!(n)))}
pub fn binary(op:&str,a:Value,b:Value)->Result<Value>{
    Ok(match op {
        "+"=>number(a.number()?+b.number()?)?,"-"=>number(a.number()?-b.number()?)?,"*"=>number(a.number()?*b.number()?)?,"/"=>number(a.number()?/b.number()?)?,
        "<"=>Value::Data(json!(a.number()?<b.number()?)),"<="=>Value::Data(json!(a.number()?<=b.number()?)),">"=>Value::Data(json!(a.number()?>b.number()?)),">="=>Value::Data(json!(a.number()?>=b.number()?)),"=="=>Value::Data(json!(a.data()?==b.data()?)),"!="=>Value::Data(json!(a.data()?!=b.data()?)),_=>bail!("unsupported operator {op}")})
}
pub fn entry(functions:Vec<(&'static str,Function)>)->Result<()> {
    let endpoint=std::env::var("WGTIME_ENDPOINT").unwrap_or_else(|_|wgtime::DEFAULT_ENDPOINT.into());
    let client=Client::connect(Path::new(&endpoint),unsafe{libc::geteuid()})?;
    let mut runtime=Runtime {client,remaining:10_000_000,metrics:Default::default(),functions:functions.into_iter().collect()};
    let result=runtime.call("main",vec![])?;
    println!("{}",json!({"result":result.data()?,"metrics":runtime.metrics}));
    Ok(())
}
