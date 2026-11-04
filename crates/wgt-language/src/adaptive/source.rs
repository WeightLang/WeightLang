//! Lower ordinary WGT 0.5 forward functions to bounded transactional tensor IR.
use super::{Instruction as I, Parameter, Program, Register};
use crate::script::parse::{Expr, Statement, Stmt};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

pub fn parse(source: &str) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let functions = crate::script::parse::parse_functions(&unit.functions)?;
    parse_with_functions(source, &functions)
}
pub fn parse_with_functions(
    source: &str,
    functions: &crate::script::parse::Program,
) -> Result<Program> {
    parse_definition(source, functions, true)
}
pub(crate) fn validate_definition(source: &str, functions: &crate::script::parse::Program) -> Result<Program> {
    parse_definition(source, functions, false)
}
fn parse_definition(source: &str, functions: &crate::script::parse::Program, materialize: bool) -> Result<Program> {
    let unit = crate::source::split(source)?;
    let forward = functions
        .get("forward")
        .context("stateful model requires fn forward(input)")?;
    let mut program = Program {
        output_type: crate::categorical::VectorType::F32,
        version: 1,
        input: String::new(),
        registers: Vec::new(),
        parameters: Vec::new(),
        instructions: Vec::new(),
        loss: None,
    };
    let (mut model, mut output) = (false, None);
    let mut target = None;
    for (line, text) in unit.declarations.lines().enumerate() {
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() || text == "wgt 0.5" {
            continue;
        }
        let (kw, body) = text
            .split_once(char::is_whitespace)
            .context("expected declaration")?;
        match kw {
            "model" => {
                ensure!(!model && !body.trim().is_empty(), "duplicate/empty model");
                model = true;
            }
            "input" => {
                ensure!(
                    program.input.is_empty(),
                    "only one native forward input is supported"
                );
                let (name, len) = crate::source::vector_port(body)?;
                program.input = name.clone();
                program.registers.push(Register { name, len, shape: vec![len] });
            }
            "target" => {
                ensure!(target.is_none(), "duplicate loss target declaration");
                target = Some(crate::source::vector_port(body)?);
            }
            "output" => {
                ensure!(output.is_none(), "duplicate output");
                let (_, n, kind) = crate::source::typed_vector_port(body)?;
                output = Some(n);
                program.output_type = kind;
            }
            "learn" => {
                let (name, declaration) = body.split_once(':').context("learn requires name: type = initializer;")?;
                let declaration = declaration.trim().strip_suffix(';').context("learn declaration requires trailing semicolon")?;
                let (domain, initial) = declaration.rsplit_once('=').context("learn requires explicit initializer")?;
                let domain = domain.trim();
                use super::LearnDomain as D;
                let (domain, value) = if domain == "bool" {
                    (D::Bool, match initial.trim() { "true" => 1., "false" => 0., _ => bail!("learn bool initializer must be true or false") })
                } else {
                    let value: f32 = initial.trim().parse().context("learn initializer must be a finite scalar")?;
                    let kind = if domain == "float" { D::Float }
                    else if let Some(low) = domain.strip_prefix("float >") {
                        D::Positive { low: low.trim().parse().context("invalid learn strict lower bound")? }
                    } else {
                        let (kind, bounds) = domain.split_once(" in ").context("unsupported learn type; choice is not supported")?;
                        let (low, high) = bounds.split_once("..=").context("learn bounds require inclusive low..=high")?;
                        let low64: f64 = low.trim().parse().context("invalid learn lower bound")?;
                        let high64: f64 = high.trim().parse().context("invalid learn upper bound")?;
                        if kind.trim() == "int" {
                            ensure!(low64.fract() == 0. && high64.fract() == 0. && low64.abs() <= 16777216. && high64.abs() <= 16777216., "learn int bounds must be exact integers within +/-16777216");
                        }
                        let low = low64 as f32;
                        let high = high64 as f32;
                        match kind.trim() { "float" => D::Bounded { low, high }, "int" => D::Int { low, high }, _ => bail!("unsupported learn type; choice is not supported") }
                    };
                    if matches!(kind, D::Int { .. }) {
                        let exact: f64 = initial.trim().parse().context("invalid learn int initializer")?;
                        ensure!(exact.is_finite() && exact.fract() == 0. && f64::from(value) == exact, "learn int initializer must be an exact f32 integer");
                    }
                    (kind, value)
                };
                domain.validate(value).with_context(|| format!("learn {}", name.trim()))?;
                program.parameters.push(Parameter { name: name.trim().into(), shape: vec![1], data: if materialize { vec![value] } else { vec![] }, frozen: false, probability: false, learn: Some(domain) });
            }
            "parameter" | "frozen" => {
                let (name, body) = body
                    .split_once(':')
                    .context("parameter requires name: f32[shape] = values")?;
                let (shape, values) = body
                    .split_once('=')
                    .context("parameter requires explicit values")?;
                let probability = shape.trim() == "prob";
                let (shape, data) = if probability {
                    let p: f64 = values
                        .trim()
                        .parse()
                        .context("prob initializer must be a scalar probability")?;
                    ensure!(
                        p.is_finite() && (0. ..=1.).contains(&p),
                        "prob initializer must be in [0,1]"
                    );
                    let value = if kw == "frozen" {
                        p
                    } else {
                        ensure!(p>0. && p<1.,"trainable prob initializer must be strictly between 0 and 1; use frozen for exact endpoints");
                        p.ln() - (-p).ln_1p()
                    };
                    (vec![1], if materialize { vec![value as f32] } else { Vec::new() })
                } else {
                    let shape = shape
                        .trim()
                        .strip_prefix("f32")
                        .context("parameter type must be f32 or prob")?;
                    let shape: Vec<usize> = serde_json::from_str(shape)?;
                    let count = crate::model::validate_shape(&shape)?;
                    let data = crate::model::validate_initializer(values.trim(), count, materialize)?;
                    (shape, data)
                };
                program.parameters.push(Parameter {
                    name: name.trim().into(),
                    shape,
                    data,
                    frozen: kw == "frozen",
                    probability,
                    learn: None,
                });
            }
            _ => bail!(
                "line {}: unsupported declaration in forward model: {kw}",
                line + 1
            ),
        }
    }
    ensure!(
        model && !program.input.is_empty() && output.is_some(),
        "forward model requires model, input and output declarations"
    );
    ensure!(
        forward.args == [program.input.clone()],
        "forward argument must match the declared input name"
    );
    let sizes = program
        .registers
        .iter()
        .map(|r| (r.name.clone(), r.len))
        .collect();
    let mut lower = Lower {
        program,
        sizes,
        types: HashMap::new(),
        memories: HashMap::new(),
        next: 0,
        loops: Vec::new(),
        output: output.unwrap(),
        functions,
        calls: 0,
        call_depth: 0,
    };
    lower.block(&forward.body)?;

    ensure!(
        lower
            .program
            .instructions
            .iter()
            .any(|i| matches!(i, I::Return { .. })),
        "forward requires a return"
    );
    ensure!(
        target.is_none() || functions.contains_key("loss"),
        "target requires fn loss"
    );
    if let Some(loss) = functions.get("loss") {
        ensure!(
            loss.args.len() == 2,
            "loss requires prediction and target arguments"
        );
        let n = lower.output;
        let m = if let Some((name, width)) = &target {
            ensure!(
                *name == loss.args[1],
                "target name must match second loss argument"
            );
            *width
        } else {
            n
        };
        let input = "__loss_input".to_string();
        let program = Program {
            output_type: crate::categorical::VectorType::F32,
            version: 1,
            input: input.clone(),
            registers: vec![Register {
                name: input.clone(),
                len: n + m,
                shape: vec![n + m],
            }],
            parameters: vec![],
            instructions: vec![],
            loss: None,
        };
        let mut loss_lower = Lower {
            program,
            sizes: HashMap::from([(input.clone(), n + m)]),
            types: HashMap::from([(loss.args[0].clone(), lower.program.output_type)]),
            memories: HashMap::new(),
            next: 0,
            loops: vec![],
            output: 1,
            functions,
            calls: 0,
            call_depth: 0,
        };
        for (i, name) in loss.args.iter().enumerate() {
            loss_lower.register(name, if i == 0 { n } else { m })?;
            loss_lower.program.instructions.push(I::Slice {
                out: name.clone(),
                input: input.clone(),
                start: i * n,
            });
        }
        loss_lower.block(&loss.body).context("fn loss")?;
        ensure!(
            loss_lower
                .program
                .instructions
                .iter()
                .any(|i| matches!(i, I::Return { .. })),
            "loss requires a scalar return"
        );
        if !lower.program.output_type.is_plain() {
            loss_lower.program.version = loss_lower.program.version.max(2);
        }
        lower.program.loss = Some(Box::new(loss_lower.program));
    }
    if !lower.program.output_type.is_plain() {
        lower.program.version = lower.program.version.max(2);
    }
    Ok(lower.program)
}
// Endpoints are compile-time scalar arithmetic, never tensor values or host calls.
fn slice_endpoint(expr: &Expr) -> Result<f64> {
    let value = match expr {
        Expr::Number(n) => *n,
        Expr::Unary(op, x) if op == "-" => -slice_endpoint(x)?,
        Expr::Binary(op, a, b) => {
            let a = slice_endpoint(a)?;
            let b = slice_endpoint(b)?;
            match op.as_str() {
                "+" => a + b, "-" => a - b, "*" => a * b,
                "/" => a / b, "%" => a % b,
                _ => bail!("tensor slice endpoints must be compile-time numeric expressions"),
            }
        }
        _ => bail!("tensor slice endpoints must be compile-time numeric expressions"),
    };
    ensure!(value.is_finite(), "tensor slice endpoint must be finite");
    Ok(value)
}

fn scalar_literal(expr: &Expr) -> Result<f32> {
    let n = match expr {
        Expr::Number(n) => *n,
        Expr::Unary(op, x) if op == "-" => {
            let Expr::Number(n) = **x else {
                bail!("expected signed numeric literal")
            };
            -n
        }
        _ => bail!("expected signed numeric literal"),
    };
    ensure!((n as f32).is_finite(), "nonfinite numeric literal");
    Ok(n as f32)
}
struct Lower<'a> {
    program: Program,
    sizes: HashMap<String, usize>,
    types: HashMap<String, crate::categorical::VectorType>,
    memories: HashMap<String, (usize, usize)>,
    next: usize,
    loops: Vec<(String, String)>,
    output: usize,
    functions: &'a crate::script::parse::Program,
    calls: usize,
    call_depth: usize,
}
impl Lower<'_> {
    fn operator(&mut self, out: &str, name: &str, args: &[Expr]) -> Result<()> {
        ensure!(self.call_depth < 32, "operator expansion depth exceeded");
        self.call_depth += 1;
        let result = self.expand_operator(out, name, args);
        self.call_depth -= 1;
        result
    }
    fn expand_operator(&mut self, out: &str, name: &str, args: &[Expr]) -> Result<()> {
        use crate::script::operators::{rewrite, validate};
        self.calls += 1;
        ensure!(self.calls <= 1024, "operator expansion limit exceeded");
        validate(name, self.functions, &mut Vec::new())?;
        let f = self.functions[name].clone();
        ensure!(
            args.len() == f.args.len(),
            "operator arity mismatch: {name}"
        );
        let mut names = HashMap::new();
        // Evaluate each argument exactly once, in caller order, before binding locals.
        for (arg, value) in f.args.iter().zip(args) {
            let input = self.value(value)?;
            names.insert(arg.clone(), input);
        }
        for s in f.body {
            match s.kind {
                Stmt::Let(n, e) => {
                    let e = rewrite(&e, &names, &HashMap::new());
                    let local = self.fresh();
                    self.assign(&local, &e)
                        .with_context(|| format!("operator {name}, line {}", s.line))?;
                    names.insert(n, local);
                }
                Stmt::Return(e) => {
                    return self
                        .assign(out, &rewrite(&e, &names, &HashMap::new()))
                        .with_context(|| format!("operator {name}, line {}", s.line))
                }
                _ => unreachable!(),
            }
        }
        bail!("operator requires return")
    }
    fn fresh(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("__wgt_{n}")
    }
    fn register(&mut self, name: &str, len: usize) -> Result<()> {
        ensure!(name != self.program.input, "cannot overwrite model input");
        if let Some(old) = self.sizes.get(name) {
            ensure!(*old == len, "register shape changes: {name}");
        } else {
            self.sizes.insert(name.into(), len);
            self.program.registers.push(Register {
                name: name.into(),
                len,
                shape: vec![len],
            });
        }
        Ok(())
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.program.registers.iter().find(|r| r.name == name).context("undefined tensor shape")?.shape.clone())
    }
    fn register_shape(&mut self, name: &str, shape: Vec<usize>) -> Result<()> {
        let len = crate::model::validate_shape(&shape)?;
        if shape.len() > 1 { self.program.version = 3; }
        if self.sizes.contains_key(name) {
            ensure!(self.shape(name)? == shape, "register geometry changes: {name}");
        } else {
            self.register(name, len)?;
            self.program.registers.last_mut().unwrap().shape = shape;
        }
        Ok(())
    }
    fn size(&self, name: &str) -> Result<usize> {
        self.sizes
            .get(name)
            .copied()
            .with_context(|| format!("undefined tensor: {name}"))
    }
    fn parameter(&self, expr: &Expr) -> Result<&Parameter> {
        let Expr::Name(name) = expr else {
            bail!("expected parameter name")
        };
        self.program
            .parameters
            .iter()
            .find(|p| &p.name == name)
            .with_context(|| format!("unknown parameter: {name}"))
    }
    fn value(&mut self, expr: &Expr) -> Result<String> {
        if let Expr::Name(name) = expr {
            ensure!(
                !self.memories.contains_key(name),
                "memory handle is not a tensor"
            );
        }
        if let Expr::Name(name) = expr {
            if self.sizes.contains_key(name) {
                return Ok(name.clone());
            }
        }
        let out = self.fresh();
        self.assign(&out, expr)?;
        Ok(out)
    }
    fn assign(&mut self, out: &str, expr: &Expr) -> Result<()> {
        use crate::categorical::VectorType;
        if let Expr::BuiltinCall(name, args) = expr {
            if name == "__probabilities" {
                let [value] = args.as_slice() else { bail!(".probabilities expects one value") };
                let input = self.value(value)?;
                let kind = self.types.get(&input).copied().unwrap_or_default();
                ensure!(kind != VectorType::F32, ".probabilities requires a logits[N] or probabilities[N] tensor");
                self.register(out, self.size(&input)?)?;
                self.program.instructions.push(if kind == VectorType::Logits {
                    I::Softmax { out: out.into(), input }
                } else { I::ValidateProbabilities { out: out.into(), input } });
                self.types.insert(out.into(), VectorType::Probabilities);
                return Ok(());
            }
            return self.assign(out, &Expr::Call(name.clone(), args.clone()));
        }
        ensure!(
            self.program.instructions.len() < 100_000 && self.program.registers.len() < 100_000,
            "tensor program expansion limit exceeded"
        );
        if let Expr::Call(name, args) = expr {
            if self.functions.contains_key(name) && !crate::script::operators::intrinsic(name) {
                return self.operator(out, name, args);
            }
        }
        if let Expr::Call(op, args) = expr {
            if op == "memory_alloc" {
                let [Expr::Number(slots), Expr::Number(width)] = args.as_slice() else {
                    bail!("memory_alloc requires two constant dimensions")
                };
                ensure!(
                    *slots >= 1.
                        && *slots <= 16384.
                        && slots.fract() == 0.
                        && *width >= 1.
                        && *width <= 16384.
                        && width.fract() == 0.,
                    "memory dimensions must be integers in 1..16384"
                );
                let shape = (*slots as usize, *width as usize);
                ensure!(
                    !self.sizes.contains_key(out) || self.memories.get(out) == Some(&shape),
                    "memory handle conflicts with tensor or shape"
                );
                ensure!(
                    !self.program.parameters.iter().any(|p| p.name == out),
                    "memory handle conflicts with parameter"
                );
                self.register(out, 1)?;
                self.memories.insert(out.into(), shape);
                self.program.instructions.push(I::MemoryAlloc {
                    out: out.into(),
                    slots: shape.0,
                    width: shape.1,
                });
                return Ok(());
            }
        }
        ensure!(
            !self.memories.contains_key(out),
            "cannot overwrite memory handle with tensor"
        );
        if let Expr::Name(name) = expr {
            ensure!(
                !self.memories.contains_key(name),
                "memory handle cannot be copied"
            );
        }
        if let Expr::Mix(p, a, b) = expr {
            return self.assign(
                out,
                &Expr::Call(
                    "route".into(),
                    vec![(**p).clone(), (**a).clone(), (**b).clone()],
                ),
            );
        }
        // Bounded shared-depth unrolling: hard 0/1 forward gate for iteration
        // index, identity surrogate dgate/ddepth=1 during training. All bounded
        // candidate steps execute on tape; inference route can short circuit.
        if let Expr::Call(op, args) = expr {
            if op == "depth_gate" {
                let [depth, index] = args.as_slice() else { bail!("depth_gate requires learn int and literal iteration index") };
                let parameter = self.parameter(depth)?;
                let Some(super::LearnDomain::Int { low, high }) = &parameter.learn else { bail!("depth_gate requires learn int") };
                let index = scalar_literal(index)?;
                ensure!(*low >= 0. && *high <= 256. && index.fract() == 0. && index >= 1. && index <= *high, "depth_gate requires nonnegative depth bounded by 256 and index in 1..=maximum");
                let hard = Expr::Call("detach".into(), vec![Expr::Call("clip".into(), vec![Expr::Binary("-".into(), Box::new(depth.clone()), Box::new(Expr::List(vec![Expr::Number(f64::from(index - 1.))]))), Expr::Number(0.), Expr::Number(1.)])]);
                let surrogate = Expr::Binary("-".into(), Box::new(depth.clone()), Box::new(Expr::Call("detach".into(), vec![depth.clone()])));
                return self.assign(out, &Expr::Binary("+".into(), Box::new(hard), Box::new(surrogate)));
            }
        }
        // Public reads expose probabilities; optimizer/checkpoint storage remains logits.
        let read_parameter = match expr {
            Expr::Name(n) if !self.sizes.contains_key(n) => Some(expr),
            Expr::Call(op, args) if op == "read" && args.len() == 1 => Some(&args[0]),
            _ => None,
        };
        if let Some(e) = read_parameter {
            let p = self.parameter(e)?;
            if p.probability && !p.frozen {
                let name = p.name.clone();
                let raw = self.fresh();
                self.register(&raw, 1)?;
                self.register(out, 1)?;
                self.program.instructions.push(I::Read {
                    out: raw.clone(),
                    parameter: name,
                });
                self.program.instructions.push(I::Sigmoid {
                    out: out.into(),
                    input: raw,
                });
                return Ok(());
            }
        }
        if let Expr::Call(op, args) = expr {
            let tensor = match (op.as_str(), args.as_slice()) {
                ("reshape", [x, Expr::List(dims)]) => {
                    let x = self.value(x)?;
                    let shape = dims.iter().map(|e| {
                        let n = scalar_literal(e)?;
                        ensure!(n >= 1. && n <= 16384. && n.fract() == 0., "reshape dimensions must be integers in 1..16384");
                        Ok(n as usize)
                    }).collect::<Result<Vec<_>>>()?;
                    ensure!(crate::model::validate_shape(&shape)? == self.size(&x)?, "reshape element count mismatch");
                    Some((I::Reshape { out: out.into(), input: x }, shape))
                },
                ("transpose", [x]) => {
                    let x = self.value(x)?; let shape = self.shape(&x)?;
                    ensure!(shape.len() == 2, "transpose requires rank 2");
                    Some((I::Transpose { out: out.into(), input: x }, vec![shape[1], shape[0]]))
                },
                ("matmul", [a,b]) => {
                    let a = self.value(a)?; let b = self.value(b)?;
                    let sa = self.shape(&a)?; let sb = self.shape(&b)?;
                    ensure!(sa.len() == 2 && sb.len() == 2 && sa[1] == sb[0], "matmul requires [M,K] and [K,N]");
                    Some((I::Matmul { out: out.into(), left:a, right:b }, vec![sa[0], sb[1]]))
                },
                ("softmax", [x,axis]) => {
                    let x = self.value(x)?; let shape = self.shape(&x)?;
                    let axis = scalar_literal(axis)?;
                    ensure!(axis >= 0. && axis.fract() == 0. && axis < shape.len() as f32, "softmax axis must be a nonnegative in-range integer");
                    Some((I::SoftmaxAxis { out: out.into(), input:x, axis:axis as usize }, shape))
                },
                ("scale", [x,factor]) => {
                    let factor = scalar_literal(factor)?;
                    let x = self.value(x)?; let shape = self.shape(&x)?;
                    Some((I::Scale { out: out.into(), input:x, factor }, shape))
                },
                _ => None,
            };
            if let Some((instruction, shape)) = tensor {
                self.register_shape(out, shape)?;
                self.program.version = 3;
                self.types.insert(out.into(), VectorType::F32);
                self.program.instructions.push(instruction);
                return Ok(());
            }
        }
        let (instruction,len) = match expr {
            Expr::List(items) => {
                let data=items.iter().map(scalar_literal).collect::<Result<Vec<_>>>()?;
                ensure!(!data.is_empty() && data.len()<=16384,"tensor literal width must be 1..16384");
                let len=data.len(); (I::Constant{out:out.into(),data},len)
            },
            Expr::Slice(x, start, end) => {
                let x = self.value(x)?;
                let (start, end) = crate::script::compiled::slice_bounds(self.size(&x)?,
                    start.as_ref().map(|e| slice_endpoint(e)).transpose()?,
                    end.as_ref().map(|e| slice_endpoint(e)).transpose()?)?;
                ensure!(start < end, "empty tensor slices are unsupported (tensor shapes must be nonzero)");
                (I::Slice { out: out.into(), input: x, start }, end - start)
            },
            Expr::Sample(..) | Expr::SampleFrom(..) => bail!("sample expressions are discrete host expressions; use mix for differentiable forward/loss"),
            Expr::Unary(op,x) if op=="-" => {
                let x=self.value(x)?; let len=self.size(&x)?;
                (I::Neg{out:out.into(),input:x},len)
            },
            Expr::Name(name) if self.sizes.contains_key(name) => {
                (if self.shape(name)?.len() > 1 { I::Reshape { out:out.into(), input:name.clone() } } else { I::Slice { out:out.into(), input:name.clone(), start:0 } }, self.size(name)?)
            },
            Expr::Name(_) => {
                let p = self.parameter(expr)?;
                (I::Read { out:out.into(), parameter:p.name.clone() },p.shape.iter().product())
            },
            Expr::Binary(op,a,b) if op=="+" || op=="*" || op=="-" => {
                let a = self.value(a)?; let b = self.value(b)?; let len=self.size(&a)?;
                ensure!(self.shape(&a)? == self.shape(&b)?, "tensor arithmetic shape mismatch");
                let b=if op=="-" {
                    let neg=self.fresh(); self.register_shape(&neg,self.shape(&a)?)?;
                    self.program.instructions.push(I::Neg{out:neg.clone(),input:b}); neg
                } else { b };
                (if op=="*" {I::Mul{out:out.into(),left:a,right:b}} else {I::Add{out:out.into(),left:a,right:b}},len)
            },
            Expr::Call(op,args) => match (op.as_str(),args.as_slice()) {
                ("detach",[x]) | ("exp",[x]) | ("log",[x]) | ("log_softmax",[x]) => {
                    let x=self.value(x)?; let len=self.size(&x)?;
                    (match op.as_str() {"detach"=>I::Detach{out:out.into(),input:x},"exp"=>I::Exp{out:out.into(),input:x},"log"=>I::Log{out:out.into(),input:x},_=>I::LogSoftmax{out:out.into(),input:x}},len)
                },
                ("minimum",[a,b]) => {
                    let a=self.value(a)?; let b=self.value(b)?; let len=self.size(&a)?;
                    ensure!(self.shape(&a)? == self.shape(&b)?,"minimum shape mismatch");
                    (I::Minimum{out:out.into(),left:a,right:b},len)
                },
                ("clip",[x,low,high]) => {
                    let low=scalar_literal(low)?;let high=scalar_literal(high)?;
                    ensure!(low<=high,"invalid clip bounds");
                    let x=self.value(x)?;let len=self.size(&x)?;
                    (I::Clip{out:out.into(),input:x,low,high},len)
                },
                ("sum",[x]) => {
                    let x=self.value(x)?;
                    (I::Sum{out:out.into(),input:x},1)
                },
                ("resource_state",[]) => (I::ResourceState{out:out.into()},3),
                ("memory_read",[Expr::Name(memory),address]) => {
                    let (slots,width)=*self.memories.get(memory).context("unknown memory handle")?;
                    let address=self.value(address)?;
                    ensure!(self.size(&address)?==slots,"memory address shape mismatch");
                    (I::MemoryRead{out:out.into(),memory:memory.clone(),address},width)
                },
                ("read",[p]) => {let p=self.parameter(p)?; (I::Read{out:out.into(),parameter:p.name.clone()},p.shape.iter().product())},
                ("linear",[x,w]) | ("linear",[x,w,_]) => {
                    let x=self.value(x)?; let w=self.parameter(w)?;
                    ensure!(!w.probability && w.learn.is_none(),"linear weight cannot be prob/learn; read architecture values explicitly");
                    ensure!(w.shape.len()==2 && w.shape[1]==self.size(&x)?, "linear weight/input shape mismatch");
                    let (weight,len)=(w.name.clone(),w.shape[0]);
                    let bias=if args.len()==3 {let b=self.parameter(&args[2])?; ensure!(!b.probability && b.learn.is_none(),"linear bias cannot be prob/learn");Some(b.name.clone())}else{None};
                    (I::Linear{out:out.into(),input:x,weight,bias},len)
                },
                ("add",[a,b]) | ("mul",[a,b]) => {
                    let a=self.value(a)?; let b=self.value(b)?; let len=self.size(&a)?;
                    ensure!(self.shape(&a)? == self.shape(&b)?, "tensor arithmetic shape mismatch");
                    (if op=="add" {I::Add{out:out.into(),left:a,right:b}}else{I::Mul{out:out.into(),left:a,right:b}},len)
                },
                ("mean",[x]) | ("square",[x]) => {
                    let x=self.value(x)?; let len=self.size(&x)?;
                    if op=="mean" {(I::Mean{out:out.into(),input:x},1)}
                    else {(I::Mul{out:out.into(),left:x.clone(),right:x},len)}
                },
                ("gate",[p,x]) | ("route",[p,x,_]) => {
                    let p=self.value(p)?;
                    let after_a=self.fresh();
                    self.program.instructions.push(I::InferenceSkip{probability:p.clone(),when_one:false,target:after_a.clone()});
                    let x=self.value(x)?; let len=self.size(&x)?;
                    self.program.instructions.push(I::Label{name:after_a});
                    ensure!(self.size(&p)?==1 || self.size(&p)?==len, "gate/route probability shape mismatch");
                    let instruction=if op=="gate" { I::Gate{out:out.into(),probability:p,input:x} } else {
                        let after_b=self.fresh();
                        self.program.instructions.push(I::InferenceSkip{probability:p.clone(),when_one:true,target:after_b.clone()});
                        let b=self.value(&args[2])?;
                        self.program.instructions.push(I::Label{name:after_b});
                        ensure!(self.size(&b)?==len,"route branch shape mismatch");
                        I::Route{out:out.into(),probability:p,yes:x,no:b}
                    };
                    (instruction,len)
                },
                ("relu",[x]) | ("tanh",[x]) | ("sigmoid",[x]) | ("softmax",[x]) => {
                    let x=self.value(x)?; let len=self.size(&x)?;
                    (match op.as_str() {"relu"=>I::Relu{out:out.into(),input:x},"tanh"=>I::Tanh{out:out.into(),input:x},"softmax"=>I::Softmax{out:out.into(),input:x},_=>I::Sigmoid{out:out.into(),input:x}},len)
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
        let kind = match &instruction {
            I::Slice { input, start: 0, .. } if self.size(input)? == len => self.types.get(input).copied().unwrap_or_default(),
            I::Detach { input, .. } => self.types.get(input).copied().unwrap_or_default(),
            I::Softmax { .. } => VectorType::Probabilities,
            _ => VectorType::F32,
        };
        let shape = match &instruction {
            I::Read { .. } => vec![len],
            I::Reshape { input, .. } => self.shape(input)?,
            I::Slice { input, start: 0, .. } if matches!(expr, Expr::Name(_)) => self.shape(input)?,
            I::Neg { input, .. } | I::Detach { input, .. } | I::Exp { input, .. } | I::Log { input, .. }
            | I::Tanh { input, .. } | I::Sigmoid { input, .. } | I::Relu { input, .. } | I::Clip { input, .. } => self.shape(input)?,
            I::Add { left, .. } | I::Mul { left, .. } | I::Minimum { left, .. } => self.shape(left)?,
            I::Gate { input, .. } | I::Route { yes: input, .. } => self.shape(input)?,
            I::Softmax { input, .. } | I::LogSoftmax { input, .. } => {
                ensure!(self.shape(input)?.len() == 1, "rank > 1 softmax requires explicit axis");
                vec![len]
            },
            I::Linear { input, .. } => {
                ensure!(self.shape(input)?.len() == 1, "linear requires vector input; use matmul for matrices");
                vec![len]
            },
            I::Slice { input, .. } => {
                ensure!(self.shape(input)?.len() == 1, "slice requires vector input");
                vec![len]
            },
            _ => vec![len],
        };
        self.register_shape(out, shape)?;

        self.types.insert(out.into(), kind);
        self.program.instructions.push(instruction);
        Ok(())
    }
    fn condition(&mut self, expr: &Expr, yes: String, no: String) -> Result<()> {
        if let Expr::Bool(value) = expr {
            self.program.instructions.push(I::Jump {
                target: if *value { yes } else { no },
            });
            return Ok(());
        }
        let Expr::Call(op, args) = expr else {
            bail!("tensor condition must be positive(tensor)")
        };
        ensure!(
            op == "positive" && args.len() == 1,
            "tensor condition must be positive(tensor)"
        );
        let condition = self.value(&args[0])?;
        ensure!(self.size(&condition)? == 1, "positive requires f32[1]");
        self.program.instructions.push(I::Branch {
            condition,
            positive: yes,
            otherwise: no,
        });
        Ok(())
    }
    fn block(&mut self, body: &[Statement]) -> Result<()> {
        for s in body {
            (|| -> Result<()> {
                match &s.kind {
                    Stmt::Let(n, e) | Stmt::Set(n, e) => {
                        ensure!(!n.starts_with("__wgt_"), "reserved compiler name");
                        if matches!(&s.kind, Stmt::Set(..)) {
                            ensure!(
                                self.sizes.contains_key(n),
                                "assignment to undeclared tensor"
                            );
                        }
                        self.assign(n, e)?;
                    }
                    Stmt::Eval(Expr::Call(op, args)) if op == "write" && args.len() == 2 => {
                        let parameter = self.parameter(&args[0])?.name.clone();
                        let value = self.value(&args[1])?;
                        self.program
                            .instructions
                            .push(I::Write { parameter, value });
                    }
                    Stmt::Eval(Expr::Call(op, args))
                        if op == "memory_free" || op == "memory_write" =>
                    {
                        let Some(Expr::Name(memory)) = args.first() else {
                            bail!("expected memory handle")
                        };
                        let (slots, width) =
                            *self.memories.get(memory).context("unknown memory handle")?;
                        if op == "memory_free" {
                            ensure!(args.len() == 1, "memory_free expects one argument");
                            self.program.instructions.push(I::MemoryFree {
                                memory: memory.clone(),
                            });
                        } else {
                            ensure!(
                                args.len() == 3,
                                "memory_write expects memory, address and value"
                            );
                            let address = self.value(&args[1])?;
                            let value = self.value(&args[2])?;
                            ensure!(
                                self.size(&address)? == slots && self.size(&value)? == width,
                                "memory_write shape mismatch"
                            );
                            self.program.instructions.push(I::MemoryWrite {
                                memory: memory.clone(),
                                address,
                                value,
                            });
                        }
                    }
                    Stmt::Return(e) => {
                        let value = self.value(e)?;
                        ensure!(
                            self.shape(&value)? == vec![self.output],
                            "forward return/output shape mismatch"
                        );
                        self.program.instructions.push(I::Return { value });
                    }
                    Stmt::If(e, yes, no) => {
                        let (a, b, end) = (self.fresh(), self.fresh(), self.fresh());
                        self.condition(e, a.clone(), b.clone())?;
                        self.program.instructions.push(I::Label { name: a });
                        self.block(yes)?;
                        self.program.instructions.push(I::Jump {
                            target: end.clone(),
                        });
                        self.program.instructions.push(I::Label { name: b });
                        self.block(no)?;
                        self.program.instructions.push(I::Label { name: end });
                    }
                    Stmt::While(e, body) => {
                        let (start, go, end) = (self.fresh(), self.fresh(), self.fresh());
                        self.program.instructions.push(I::Label {
                            name: start.clone(),
                        });
                        self.condition(e, go.clone(), end.clone())?;
                        self.program.instructions.push(I::Label { name: go });
                        self.loops.push((start.clone(), end.clone()));
                        self.block(body)?;
                        self.loops.pop();
                        self.program.instructions.push(I::Jump { target: start });
                        self.program.instructions.push(I::Label { name: end });
                    }
                    Stmt::Break | Stmt::Continue => {
                        let (start, end) =
                            self.loops.last().context("loop control outside loop")?;
                        self.program.instructions.push(I::Jump {
                            target: if matches!(&s.kind, Stmt::Break) {
                                end
                            } else {
                                start
                            }
                            .clone(),
                        });
                    }
                    _ => bail!("unsupported forward statement"),
                }
                Ok(())
            })()
            .with_context(|| format!("line {}", s.line))?;
        }
        Ok(())
    }
}
