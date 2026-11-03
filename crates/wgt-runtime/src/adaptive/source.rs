//! Lower ordinary WGT 0.5 forward functions to bounded transactional tensor IR.
use super::{Instruction as I, Parameter, Program, Register};
use crate::script::parse::{Expr, Statement, Stmt};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

pub fn parse(source: &str) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    let forward = functions.get("forward").context("stateful model requires fn forward(input)")?;
    let mut program = Program { version:1, input:String::new(), registers:Vec::new(), parameters:Vec::new(), instructions:Vec::new() };
    let (mut model, mut output) = (false, None);
    for (line, text) in unit.declarations.lines().enumerate() {
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() || text == "wgt 0.5" { continue; }
        let (kw, body) = text.split_once(char::is_whitespace).context("expected declaration")?;
        match kw {
            "model" => { ensure!(!model && !body.trim().is_empty(), "duplicate/empty model"); model = true; },
            "input" => {
                ensure!(program.input.is_empty(), "only one native forward input is supported");
                let (name, len) = crate::source::vector_port(body)?;
                program.input = name.clone();
                program.registers.push(Register { name, len });
            },
            "output" => { ensure!(output.is_none(), "duplicate output"); output = Some(crate::source::vector_port(body)?.1); },
            "parameter" | "frozen" => {
                let (name, body) = body.split_once(':').context("parameter requires name: f32[shape] = values")?;
                let (shape, values) = body.split_once('=').context("parameter requires explicit values")?;
                let shape = shape.trim().strip_prefix("f32").context("parameter type must be f32")?;
                program.parameters.push(Parameter { name:name.trim().into(), shape:serde_json::from_str(shape)?, data:serde_json::from_str(values.trim())?, frozen:kw == "frozen" });
            },
            _ => bail!("line {}: unsupported declaration in forward model: {kw}", line+1)
        }
    }
    ensure!(model && !program.input.is_empty() && output.is_some(), "forward model requires model, input and output declarations");
    ensure!(forward.args == [program.input.clone()], "forward argument must match the declared input name");
    let sizes = program.registers.iter().map(|r| (r.name.clone(),r.len)).collect();
    let mut lower = Lower { program, sizes, next:0, loops:Vec::new(), output:output.unwrap() };
    lower.block(&forward.body)?;
    ensure!(lower.program.instructions.iter().any(|i| matches!(i,I::Return{..})), "forward requires a return");
    Ok(lower.program)
}
struct Lower {
    program: Program,
    sizes: HashMap<String,usize>,
    next: usize,
    loops: Vec<(String,String)>,
    output: usize,
}
impl Lower {
    fn fresh(&mut self) -> String { let n = self.next; self.next += 1; format!("__wgt_{n}") }
    fn register(&mut self, name: &str, len: usize) -> Result<()> {
        ensure!(name != self.program.input, "cannot overwrite model input");
        if let Some(old) = self.sizes.get(name) { ensure!(*old == len, "register shape changes: {name}"); }
        else { self.sizes.insert(name.into(),len); self.program.registers.push(Register { name:name.into(),len }); }
        Ok(())
    }
    fn size(&self, name:&str) -> Result<usize> { self.sizes.get(name).copied().with_context(|| format!("undefined tensor: {name}")) }
    fn parameter(&self, expr:&Expr) -> Result<&Parameter> {
        let Expr::Name(name) = expr else { bail!("expected parameter name") };
        self.program.parameters.iter().find(|p| &p.name==name).with_context(||format!("unknown parameter: {name}"))
    }
    fn value(&mut self, expr:&Expr) -> Result<String> {
        if let Expr::Name(name) = expr { if self.sizes.contains_key(name) { return Ok(name.clone()); } }
        let out = self.fresh(); self.assign(&out,expr)?; Ok(out)
    }
    fn assign(&mut self, out:&str, expr:&Expr) -> Result<()> {
        let (instruction,len) = match expr {
            Expr::Name(name) if self.sizes.contains_key(name) => {
                (I::Slice { out:out.into(), input:name.clone(), start:0 }, self.size(name)?)
            },
            Expr::Name(_) => {
                let p = self.parameter(expr)?;
                (I::Read { out:out.into(), parameter:p.name.clone() },p.shape.iter().product())
            },
            Expr::Binary(op,a,b) if op=="+" || op=="*" => {
                let a = self.value(a)?; let b = self.value(b)?; let len=self.size(&a)?;
                ensure!(len==self.size(&b)?, "tensor arithmetic shape mismatch");
                (if op=="+" {I::Add{out:out.into(),left:a,right:b}} else {I::Mul{out:out.into(),left:a,right:b}},len)
            },
            Expr::Call(op,args) => match (op.as_str(),args.as_slice()) {
                ("read",[p]) => {let p=self.parameter(p)?; (I::Read{out:out.into(),parameter:p.name.clone()},p.shape.iter().product())},
                ("linear",[x,w]) | ("linear",[x,w,_]) => {
                    let x=self.value(x)?; let w=self.parameter(w)?;
                    ensure!(w.shape.len()==2 && w.shape[1]==self.size(&x)?, "linear weight/input shape mismatch");
                    let (weight,len)=(w.name.clone(),w.shape[0]);
                    let bias=if args.len()==3 {Some(self.parameter(&args[2])?.name.clone())}else{None};
                    (I::Linear{out:out.into(),input:x,weight,bias},len)
                },
                ("add",[a,b]) | ("mul",[a,b]) => {
                    let a=self.value(a)?; let b=self.value(b)?; let len=self.size(&a)?;
                    ensure!(len==self.size(&b)?, "tensor arithmetic shape mismatch");
                    (if op=="add" {I::Add{out:out.into(),left:a,right:b}}else{I::Mul{out:out.into(),left:a,right:b}},len)
                },
                ("relu",[x]) | ("tanh",[x]) | ("sigmoid",[x]) => {
                    let x=self.value(x)?; let len=self.size(&x)?;
                    (match op.as_str() {"relu"=>I::Relu{out:out.into(),input:x},"tanh"=>I::Tanh{out:out.into(),input:x},_=>I::Sigmoid{out:out.into(),input:x}},len)
                },
                ("slice",[x,Expr::Number(start),Expr::Number(len)]) => {
                    ensure!(*start>=0. && start.fract()==0. && *len>0. && len.fract()==0., "slice needs nonnegative start and positive length");
                    let x=self.value(x)?;
                    ensure!(*start + *len <= self.size(&x)? as f64, "slice out of bounds");
                    (I::Slice{out:out.into(),input:x,start:*start as usize},*len as usize)
                },
                _ => bail!("unsupported tensor expression or arity: {op}")
            },
            _ => bail!("forward expressions must produce explicitly shaped tensors")
        };
        self.register(out,len)?;
        self.program.instructions.push(instruction); Ok(())
    }
    fn condition(&mut self, expr:&Expr, yes:String, no:String) -> Result<()> {
        if let Expr::Bool(value)=expr { self.program.instructions.push(I::Jump{target:if *value {yes}else{no}}); return Ok(()); }
        let Expr::Call(op,args)=expr else {bail!("tensor condition must be positive(tensor)")};
        ensure!(op=="positive" && args.len()==1, "tensor condition must be positive(tensor)");
        let condition=self.value(&args[0])?;
        ensure!(self.size(&condition)?==1, "positive requires f32[1]");
        self.program.instructions.push(I::Branch{condition,positive:yes,otherwise:no}); Ok(())
    }
    fn block(&mut self, body:&[Statement]) -> Result<()> {
        for s in body {
            (|| -> Result<()> {
                match &s.kind {
                    Stmt::Let(n,e) | Stmt::Set(n,e) => {
                        ensure!(!n.starts_with("__wgt_"), "reserved compiler name");
                        if matches!(&s.kind,Stmt::Set(..)) {ensure!(self.sizes.contains_key(n), "assignment to undeclared tensor");}
                        self.assign(n,e)?;
                    },
                    Stmt::Eval(Expr::Call(op,args)) if op=="write" && args.len()==2 => {
                        let parameter=self.parameter(&args[0])?.name.clone();
                        let value=self.value(&args[1])?;
                        self.program.instructions.push(I::Write{parameter,value});
                    },
                    Stmt::Return(e) => {let value=self.value(e)?; ensure!(self.size(&value)?==self.output,"forward return/output shape mismatch");self.program.instructions.push(I::Return{value});},
                    Stmt::If(e,yes,no) => {
                        let (a,b,end)=(self.fresh(),self.fresh(),self.fresh());
                        self.condition(e,a.clone(),b.clone())?;
                        self.program.instructions.push(I::Label{name:a});self.block(yes)?;
                        self.program.instructions.push(I::Jump{target:end.clone()});
                        self.program.instructions.push(I::Label{name:b});self.block(no)?;
                        self.program.instructions.push(I::Label{name:end});
                    },
                    Stmt::While(e,body) => {
                        let (start,go,end)=(self.fresh(),self.fresh(),self.fresh());
                        self.program.instructions.push(I::Label{name:start.clone()});
                        self.condition(e,go.clone(),end.clone())?;
                        self.program.instructions.push(I::Label{name:go});
                        self.loops.push((start.clone(),end.clone()));self.block(body)?;self.loops.pop();
                        self.program.instructions.push(I::Jump{target:start});self.program.instructions.push(I::Label{name:end});
                    },
                    Stmt::Break | Stmt::Continue => {
                        let (start,end)=self.loops.last().context("loop control outside loop")?;
                        self.program.instructions.push(I::Jump{target:if matches!(&s.kind,Stmt::Break){end}else{start}.clone()});
                    },
                    _ => bail!("unsupported forward statement")
                }
                Ok(())
            })().with_context(||format!("line {}",s.line))?;
        }
        Ok(())
    }
}
