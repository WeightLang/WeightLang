//! Service-side model compilation. Generated C is compiled once, then called through one ABI.
//! Parameters remain service-owned and are supplied afresh on every invocation.
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};
use wgt_language::adaptive::{Instruction as I, Parameter, Program, Register};

type Hook = unsafe extern "C" fn(*mut std::ffi::c_void, usize, *const f32, usize) -> i32;
type Entry = unsafe extern "C" fn(
    *const f32,
    *const *const f32,
    *mut f32,
    *mut f32,
    u64,
    usize,
    *mut u64,
    i32,
    Option<Hook>,
    *mut std::ffi::c_void,
) -> i32;
struct Code {
    entry: Entry,
    _library: libloading::Library,
    _directory: tempfile::TempDir,
}
static CACHE: OnceLock<Mutex<HashMap<String, Weak<Code>>>> = OnceLock::new();

pub struct CompiledForward {
    code: Arc<Code>,
    names: Vec<String>,
    lengths: Vec<usize>,
    input: usize,
    output: usize,
    workspace: Vec<f32>,
    candidates: Vec<Option<usize>>,
    memory_available: usize,
    completed: bool,
    last_steps: u64,
}
impl CompiledForward {
    pub fn compile(program: &Program, memory: usize) -> Result<Self> {
        for instruction in &program.instructions {
            if let I::Write { parameter, .. } = instruction {
                let parameter = program.parameters.iter().find(|p| p.name == *parameter)
                    .context("unknown compiled write parameter")?;
                ensure!(!parameter.frozen && !parameter.probability && parameter.learn.is_none(),
                    "ordinary forward cannot write frozen, probability or learn parameters");
            }
        }
        Self::compile_transaction(program, memory)
    }
    pub fn compile_transaction(program: &Program, memory: usize) -> Result<Self> {
        let generated = Generator::new(program)?.generate()?;
        let bytes = generated
            .workspace
            .checked_mul(4)
            .context("compiled workspace overflow")?;
        ensure!(
            bytes <= memory,
            "compiled forward workspace budget exceeded"
        );
        let code = load_code(&generated.source)?;
        Ok(Self {
            code,
            names: program.parameters.iter().map(|p| p.name.clone()).collect(),
            lengths: program.parameters.iter().map(|p| p.data.len()).collect(),
            input: generated.input,
            output: generated.output,
            workspace: vec![0.; generated.workspace],
            candidates: generated.candidates,
            memory_available: memory - bytes,
            completed: false,
            last_steps: 0,
        })
    }
    /// Sequential resident models use exactly the same native code generator.
    pub fn compile_model(model: &crate::Model, memory: usize) -> Result<Self> {
        Self::compile(&Self::model_program(model)?, memory)
    }
    pub fn model_program(model: &crate::Model) -> Result<Program> {
        let mut program = Program {
            observations: Vec::new(),
            version: if model.output_type.is_plain() { 1 } else { 2 },
            output_type: model.output_type,
            input: "input".into(),
            registers: vec![Register {
                name: "input".into(),
                len: model.input,
                shape: vec![model.input],
            }],
            parameters: Vec::new(),
            instructions: Vec::new(),
            loss: None,
        };
        for p in &model.parameters {
            let crate::Value::Resident { data } = &p.value else {
                bail!("compiled graph requires resident parameters")
            };
            program.parameters.push(Parameter {
                name: p.name.clone(),
                shape: p.shape.clone(),
                data: data.clone(),
                initializer: None,
                frozen: p.frozen,
                probability: false,
                learn: None,
            });
        }
        let mut previous = "input".to_owned();
        let mut width = model.input;
        for (index, layer) in model.layers.iter().enumerate() {
            let out = format!("layer_{index}");
            let instruction = match layer {
                crate::Layer::Linear { weight, bias } => {
                    let parameter = program
                        .parameters
                        .iter()
                        .find(|p| p.name == *weight)
                        .context("unknown linear weight")?;
                    ensure!(
                        parameter.shape.len() == 2 && parameter.shape[1] == width,
                        "compiled graph linear shape mismatch"
                    );
                    width = parameter.shape[0];
                    I::Linear {
                        out: out.clone(),
                        input: previous,
                        weight: weight.clone(),
                        bias: bias.clone(),
                    }
                }
                crate::Layer::Tanh => I::Tanh {
                    out: out.clone(),
                    input: previous,
                },
                crate::Layer::Sigmoid => I::Sigmoid {
                    out: out.clone(),
                    input: previous,
                },
                crate::Layer::Relu => I::Relu {
                    out: out.clone(),
                    input: previous,
                },
                crate::Layer::Identity => I::Slice {
                    out: out.clone(),
                    input: previous,
                    start: 0,
                },
            };
            program.registers.push(Register {
                name: out.clone(),
                len: width,
                shape: vec![width],
            });
            program.instructions.push(instruction);
            previous = out;
        }
        program.instructions.push(I::Return { value: previous });
        Ok(program)
    }
    pub fn last_steps(&self) -> u64 {
        self.last_steps
    }
    pub fn parameter_names(&self) -> &[String] {
        &self.names
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.len() * 4
    }
    /// Final private storage after a successful native call containing writes.
    /// The caller owns committing these values; unsuccessful calls never publish candidates.
    pub fn parameter_candidate(&self, index: usize) -> Result<&[f32]> {
        ensure!(self.completed, "compiled forward has no successful parameter candidate");
        let offset = self.candidates.get(index).copied().flatten()
            .context("compiled parameter has no candidate storage")?;
        Ok(&self.workspace[offset..offset + self.lengths[index]])
    }
    pub fn run(&mut self, input: &[f32], parameters: &[&[f32]]) -> Result<Vec<f32>> {
        self.invoke(input, parameters, 1_000_000, self.memory_available, false, &mut |_, _| Ok(()))
    }
    pub fn run_budget(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        budget: u64,
    ) -> Result<Vec<f32>> {
        self.invoke(input, parameters, budget, self.memory_available, false, &mut |_, _| Ok(()))
    }
    pub fn run_with_hook(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        budget: u64,
        traced: bool,
        hook: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.invoke(input, parameters, budget, self.memory_available, traced, hook)
    }
    /// Execute with the host's currently available working-memory allowance.
    /// Fixed model and compiled workspace reservations have already been deducted.
    pub fn run_with_resources(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        budget: u64,
        memory_available: usize,
        traced: bool,
        hook: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.invoke(input, parameters, budget, memory_available, traced, hook)
    }
    pub fn run_traced(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        hook: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.run_traced_budget(input, parameters, 1_000_000, hook)
    }
    pub fn run_traced_budget(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        budget: u64,
        hook: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.invoke(input, parameters, budget, self.memory_available, true, hook)
    }
    fn invoke(
        &mut self,
        input: &[f32],
        parameters: &[&[f32]],
        budget: u64,
        memory_available: usize,
        traced: bool,
        hook: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.completed = false;
        self.last_steps = 0;
        ensure!(
            input.len() == self.input && input.iter().all(|v| v.is_finite()),
            "invalid compiled forward input"
        );
        ensure!(
            parameters.len() == self.lengths.len()
                && parameters
                    .iter()
                    .zip(&self.lengths)
                    .all(|(v, n)| v.len() == *n && v.iter().all(|v| v.is_finite())),
            "compiled parameter schema mismatch"
        );
        struct Context<'a> {
            hook: &'a mut dyn FnMut(usize, &[f32]) -> Result<()>,
            error: Option<anyhow::Error>,
        }
        unsafe extern "C" fn callback(
            context: *mut std::ffi::c_void,
            index: usize,
            data: *const f32,
            len: usize,
        ) -> i32 {
            // Only the matching native call supplies this live, exclusive context pointer.
            let context = unsafe { &mut *(context as *mut Context<'_>) };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Generated code passes a validated register range for the duration of this call.
                (context.hook)(index, unsafe { std::slice::from_raw_parts(data, len) })
            }));
            match result {
                Ok(Ok(())) => 0,
                Ok(Err(error)) => {
                    context.error = Some(error);
                    1
                }
                Err(_) => {
                    context.error = Some(anyhow::anyhow!("compiled forward trace hook panicked"));
                    1
                }
            }
        }
        let pointers: Vec<_> = parameters.iter().map(|p| p.as_ptr()).collect();
        let mut output = vec![0.; self.output];
        let mut context = Context { hook, error: None };
        self.last_steps = 0;
        let status = unsafe {
            (self.code.entry)(
                input.as_ptr(),
                pointers.as_ptr(),
                self.workspace.as_mut_ptr(),
                output.as_mut_ptr(),
                budget,
                memory_available,
                &mut self.last_steps,
                i32::from(traced),
                Some(callback),
                (&mut context as *mut Context<'_>).cast(),
            )
        };
        if let Some(error) = context.error {
            return Err(error);
        }
        ensure!(
            status == 0,
            "compiled forward failed: {}",
            match status {
                1 => "nonfinite value",
                2 => "instruction budget exceeded",
                3 => "uninitialized register",
                4 => "invalid probability",
                5 => "trace hook failed",
                7 => "working memory budget exceeded",
                8 => "working memory allocation failed",
                9 => "memory is not allocated or has been freed",
                10 => "memory already allocated; free it before reallocating",
                11 => "memory address must have one finite [0,1] weight per slot",
                12 => "ge does not support automatic differentiation; use non-AD evaluation",
                _ => "invalid control flow",
            }
        );
        self.completed = true;
        Ok(output)
    }
}

fn load_code(source: &str) -> Result<Arc<Code>> {
    // Source contains only generated tokens, checked indices and finite numeric literals.
    // Full source is the collision-free process-local key; never load user-provided libraries.
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow::anyhow!("compiled cache lock poisoned"))?;
    cache.retain(|_, code| code.strong_count() > 0);
    if let Some(code) = cache.get(source).and_then(Weak::upgrade) {
        return Ok(code);
    }
    ensure!(
        cfg!(target_os = "linux"),
        "compiled forward currently requires Linux ELF shared libraries"
    );
    let root = std::env::var_os("WGTIME_COMPILE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".cache/weightlang/compiled-forward"));
    fs::create_dir_all(&root)?;
    let directory = tempfile::Builder::new()
        .prefix("native-")
        .tempdir_in(root)?;
    let source_path = directory.path().join("model.c");
    let library_path = directory.path().join("model.so");
    fs::write(&source_path, source)?;
    let log_path = directory.path().join("compiler.log");
    let log = fs::File::create(&log_path)?;
    let mut child = Command::new("cc")
        .args([
            "-std=c11",
            "-shared",
            "-fPIC",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-Werror=implicit-function-declaration",
        ])
        .arg(&source_path)
        .arg("-o")
        .arg(&library_path)
        .arg("-lm")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .spawn()
        .context("compiled forward requires a local C compiler (cc)")?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("compiled forward C compilation timed out after 30 seconds")
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        use std::io::Read;
        let mut diagnostic = String::new();
        fs::File::open(log_path)?
            .take(4096)
            .read_to_string(&mut diagnostic)?;
        bail!("compiled forward C compilation failed: {diagnostic}");
    }
    let library = unsafe { libloading::Library::new(&library_path) }?;
    let entry = unsafe { *library.get::<Entry>(b"wgt_forward_v2\0")? };
    let code = Arc::new(Code {
        entry,
        _library: library,
        _directory: directory,
    });
    cache.insert(source.into(), Arc::downgrade(&code));
    Ok(code)
}

struct Generated {
    source: String,
    input: usize,
    output: usize,
    workspace: usize,
    candidates: Vec<Option<usize>>,
}
struct Generator<'a> {
    program: &'a Program,
    registers: HashMap<&'a str, (usize, usize, usize)>,
    parameters: HashMap<&'a str, (usize, usize)>,
    memories: HashMap<usize, (usize, usize)>,
    workspace: usize,
    max: usize,
}
impl<'a> Generator<'a> {
    fn new(program: &'a Program) -> Result<Self> {
        ensure!(
            program.instructions.len() <= 16_384 && program.registers.len() <= 16_384,
            "compiled model instruction/register limit exceeded"
        );
        program.validate_tensor_contracts()?;
        let mut registers = HashMap::new();
        let mut workspace = 0usize;
        let mut max = 0;
        for (i, r) in program.registers.iter().enumerate() {
            registers.insert(r.name.as_str(), (workspace, r.len, i));
            workspace = workspace.checked_add(r.len).context("workspace overflow")?;
            max = max.max(r.len);
        }
        let mut parameters = HashMap::new();
        for (i, p) in program.parameters.iter().enumerate() {
            let len = p.shape.iter().product::<usize>();
            ensure!(
                len == p.data.len(),
                "compiled parameter data/shape mismatch"
            );
            ensure!(p.data.iter().all(|v| v.is_finite()), "nonfinite parameter");
            parameters.insert(p.name.as_str(), (i, len));
        }
        let memory_registers = registers.iter()
            .map(|(name, (_, _, index))| ((*name).to_owned(), *index)).collect();
        let memories = crate::adaptive::workspace::validate(program, &memory_registers)?;
        Ok(Self {
            program,
            registers,
            parameters,
            memories,
            workspace,
            max,
        })
    }
    fn reg(&self, name: &str) -> Result<(usize, usize, usize)> {
        self.registers
            .get(name)
            .copied()
            .context("unknown compiled register")
    }
    fn par(&self, name: &str) -> Result<(usize, usize)> {
        self.parameters
            .get(name)
            .copied()
            .context("unknown compiled parameter")
    }
    fn memory(&self, name: &str) -> Result<(usize, usize, usize)> {
        let (_, _, index) = self.reg(name)?;
        let &(slots, width) = self.memories.get(&index).context("unknown compiled memory handle")?;
        Ok((index, slots, width))
    }
    fn cost(&self, instruction: &I) -> Result<usize> {
        let shape = |name: &str| -> Result<&[usize]> {
            Ok(&self.program.registers[self.reg(name)?.2].shape)
        };
        let extra = match instruction {
            I::Conv { input, kernel, bias, stride, padding, dilation, groups, .. } => {
                return wgt_language::adaptive::ConvGeometry::new(shape(input)?, shape(kernel)?, shape(bias)?, stride, padding, dilation, *groups)?.work();
            }
            I::MaxPool2d { input, kernel, stride, .. } | I::AvgPool2d { input, kernel, stride, .. } => {
                return wgt_language::adaptive::Pool2dGeometry::new(shape(input)?, kernel, stride)?.work();
            }
            I::Reshape { out, .. } | I::Transpose { out, .. } | I::Permute { out, .. } => self.reg(out)?.1,
            I::Matmul { left, right, .. } => {
                let a = shape(left)?;
                let b = shape(right)?;
                a[0].checked_mul(a[1]).and_then(|n| n.checked_mul(b[1]))
                    .context("matmul work overflow")?
            }
            I::BatchMatmul { left, right, .. } => {
                let a = shape(left)?;
                let b = shape(right)?;
                a[0].checked_mul(a[1]).and_then(|n| n.checked_mul(a[2]))
                    .and_then(|n| n.checked_mul(b[2]))
                    .context("batch_matmul work overflow")?
            }
            I::LayerNorm { out, .. } => self.reg(out)?.1.checked_mul(3).context("LayerNorm work overflow")?,
            I::MemoryAlloc { slots, width, .. } => slots.checked_mul(*width).context("memory size overflow")?,
            I::MemoryRead { memory, .. } | I::MemoryWrite { memory, .. } => {
                let (_, slots, width) = self.memory(memory)?;
                slots.checked_mul(width).context("memory size overflow")?
            }
            _ => 0,
        };
        extra.checked_add(1).context("compiled instruction work overflow")
    }
    fn literal(value: f32) -> Result<String> {
        ensure!(value.is_finite(), "nonfinite compiled constant");
        Ok(format!("{:.9e}f", value))
    }
    fn generate(&self) -> Result<Generated> {
        let (input_offset, input, input_id) = self.reg(&self.program.input)?;
        let mut labels = HashMap::new();
        for (index, instruction) in self.program.instructions.iter().enumerate() {
            if let I::Label { name } = instruction {
                ensure!(
                    labels.insert(name.as_str(), index).is_none(),
                    "duplicate compiled label"
                );
            }
        }
        let target = |name: &str| -> Result<usize> {
            labels.get(name).copied().context("unknown compiled label")
        };
        let mut code = String::from(
            "#include <math.h>\n#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\ntypedef int (*trace_fn)(void*,size_t,const float*,size_t);\nstatic int execute(const float *input,const float *const *p,float *r,float *output,uint64_t budget,size_t memory_available,uint64_t *used,int traced,trace_fn trace,void *context,float **memory){\nsize_t memory_used=0;\n",
        );
        code += &format!(
            "unsigned char ready[{}]={{0}};float *t=r+{};memcpy(r+{},input,{}*sizeof(float));ready[{}]=1;\n",
            self.registers.len().max(1),
            self.workspace,
            input_offset,
            input,
            input_id
        );
        let mut transaction_bytes = 0usize;
        let mut candidates = vec![None; self.parameters.len()];
        if self.program.instructions.iter().any(|i| matches!(i, I::Write { .. })) {
            // Candidate storage is native workspace, never the caller's immutable parameter slices.
            let offset = self.workspace + self.max;
            code += &format!("const float *original[{}];float *candidate[{}];", self.parameters.len().max(1), self.parameters.len().max(1));
            for (index, parameter) in self.program.parameters.iter().enumerate() {
                let len = parameter.data.len();
                candidates[index] = Some(offset + transaction_bytes);
                code += &format!("original[{index}]=p[{index}];candidate[{index}]=r+{};memcpy(candidate[{index}],original[{index}],{len}*sizeof(float));", offset + transaction_bytes);
                transaction_bytes = transaction_bytes.checked_add(len).context("transaction workspace overflow")?;
            }
            code += "p=(const float *const *)candidate;";
        }
        let mut output_width = None;
        for (index, instruction) in self.program.instructions.iter().enumerate() {
            code += &format!("L{index}:;if(*used>=budget)return 2;{{\n");
            // Memory access validates liveness before charging work, as the reference does.
            if let I::MemoryRead { memory, .. } | I::MemoryWrite { memory, .. } = instruction {
                let (id, _, _) = self.memory(memory)?;
                code += &format!("if(!memory[{id}])return 9;");
            }
            let cost = self.cost(instruction)?;
            code += &format!("if(budget-*used<{cost})return 2;*used+={cost};");
            let metadata = instruction.metadata();
            let mut reads = metadata.inputs;
            let mut body = String::new();
            match instruction {
                I::Conv { input:x,kernel,bias,stride,padding,dilation,groups,.. } => {
                    let (a,_,xi)=self.reg(x)?;let (k,_,ki)=self.reg(kernel)?;let (b,_,bi)=self.reg(bias)?;
                    let g=wgt_language::adaptive::ConvGeometry::new(&self.program.registers[xi].shape,&self.program.registers[ki].shape,&self.program.registers[bi].shape,stride,padding,dilation,*groups)?;
                    let ov=g.output_volume();let iv=g.input_volume();let kv=g.kernel_volume();let cg=g.channels/g.groups;let og=g.outputs/g.groups;let count=g.outputs*ov;
                    body+=&format!("for(size_t z=0;z<{count};z++){{size_t oc=z/{ov},pos=z%{ov},coords[3]={{0}};");
                    for d in (0..g.rank).rev(){body+=&format!("coords[{d}]=pos%{};pos/={};",g.output[d],g.output[d]);}
                    body+=&format!("double sum=r[{b}+oc];for(size_t c=0;c<{cg};c++)for(size_t q=0;q<{kv};q++){{size_t rem=q,kc[3]={{0}},ix=0;int valid=1;");
                    for d in (0..g.rank).rev(){body+=&format!("kc[{d}]=rem%{};rem/={};",g.kernel[d],g.kernel[d]);}
                    for d in 0..g.rank {body+=&format!("{{long p=(long)(coords[{d}]*{}+kc[{d}]*{})-{};if(p<0||p>={})valid=0;else ix=ix*{}+(size_t)p;}}",g.stride[d],g.dilation[d],g.padding[d],g.input[d],g.input[d]);}
                    body+=&format!("if(valid)sum+=(double)r[{a}+(oc/{og}*{cg}+c)*{iv}+ix]*(double)r[{k}+(oc*{cg}+c)*{kv}+q];}}t[z]=(float)sum;}}");
                },
                I::MaxPool2d { out:o,input:x,kernel,stride }
                | I::AvgPool2d { out:o,input:x,kernel,stride } => {
                    let (a,_,xi) = self.reg(x)?;
                    let g = wgt_language::adaptive::Pool2dGeometry::new(
                        &self.program.registers[xi].shape, kernel, stride
                    )?;
                    let (_,out_len,_) = self.reg(o)?;
                    let spatial = g.output_h * g.output_w;
                    let window = g.window_len();
                    let is_max = matches!(instruction, I::MaxPool2d { .. });
                    if is_max {
                        body += &format!("for(size_t z=0;z<{out_len};z++){{size_t c=z/{spatial},p=z%{spatial},oy=p/{},ox=p%{};float v=-INFINITY;for(size_t ky=0;ky<{};ky++)for(size_t kx=0;kx<{};kx++){{size_t ix=(c*{}+oy*{}+ky)*{}+ox*{}+kx;v=fmaxf(v,r[{a}+ix]);}}t[z]=v;}}",
                            g.output_w,g.output_w,g.kernel_h,g.kernel_w,g.input_h,g.stride_h,g.input_w,g.stride_w);
                    } else {
                        body += &format!("for(size_t z=0;z<{out_len};z++){{size_t c=z/{spatial},p=z%{spatial},oy=p/{},ox=p%{};double v=0.;for(size_t ky=0;ky<{};ky++)for(size_t kx=0;kx<{};kx++){{size_t ix=(c*{}+oy*{}+ky)*{}+ox*{}+kx;v+=(double)r[{a}+ix];}}t[z]=(float)(v/{window}.0);}}",
                            g.output_w,g.output_w,g.kernel_h,g.kernel_w,g.input_h,g.stride_h,g.input_w,g.stride_w);
                    }
                },
                I::Reshape { input:x,.. } => {
                    let (a,n,_) = self.reg(x)?;
                    body += &format!("memcpy(t,r+{a},{n}*sizeof(float));");
                },
                I::Transpose { input:x,.. } => {
                    let (a,_,_) = self.reg(x)?;
                    let shape = &self.program.registers[self.reg(x)?.2].shape;
                    let (m,n) = (shape[0],shape[1]);
                    body += &format!("for(size_t i=0;i<{m};i++)for(size_t j=0;j<{n};j++)t[j*{m}+i]=r[{a}+i*{n}+j];");
                },
                I::Permute { out:o,input:x,axes } => {
                    let (a,_,xi) = self.reg(x)?;
                    let input_shape = &self.program.registers[xi].shape;
                    let rank = input_shape.len();
                    let (_,out_len,out_id) = self.reg(o)?;
                    let output_shape = &self.program.registers[out_id].shape;
                    let mut body_code = format!("for(size_t z=0;z<{out_len};z++){{size_t rem=z,oc[{rank}]={{0}},ic[{rank}]={{0}};");
                    for axis in (0..rank).rev() {
                        body_code += &format!("oc[{axis}]=rem%{};rem/={};", output_shape[axis], output_shape[axis]);
                    }
                    for axis in 0..rank {
                        body_code += &format!("ic[{}]=oc[{axis}];", axes[axis]);
                    }
                    body_code += "size_t src=0;";
                    for axis in 0..rank {
                        body_code += &format!("src=src*{}+ic[{axis}];", input_shape[axis]);
                    }
                    body_code += &format!("t[z]=r[{a}+src];}}");
                    body += &body_code;
                },
                I::Matmul { left,right,.. } => {
                    let (a,_,ai) = self.reg(left)?; let (b,_,bi) = self.reg(right)?;
                    let sa = &self.program.registers[ai].shape; let sb = &self.program.registers[bi].shape;
                    let (m,k,n)=(sa[0],sa[1],sb[1]);
                    body += &format!("for(size_t i=0;i<{m};i++)for(size_t j=0;j<{n};j++){{double sum=0.;for(size_t q=0;q<{k};q++)sum+=(double)r[{a}+i*{k}+q]*(double)r[{b}+q*{n}+j];t[i*{n}+j]=(float)sum;}}");
                },
                I::BatchMatmul { left,right,.. } => {
                    let (a,_,ai) = self.reg(left)?;
                    let (b,_,bi) = self.reg(right)?;
                    let sa = &self.program.registers[ai].shape;
                    let sb = &self.program.registers[bi].shape;
                    let (batch,m,k,n)=(sa[0],sa[1],sa[2],sb[2]);
                    body += &format!("for(size_t z=0;z<{batch};z++)for(size_t i=0;i<{m};i++)for(size_t j=0;j<{n};j++){{double sum=0.;for(size_t q=0;q<{k};q++)sum+=(double)r[{a}+z*{m}*{k}+i*{k}+q]*(double)r[{b}+z*{k}*{n}+q*{n}+j];t[z*{m}*{n}+i*{n}+j]=(float)sum;}}");
                },
                I::LayerNorm { input:x,gamma,beta,axis,epsilon,.. } => {
                    let (a,_,ai) = self.reg(x)?;
                    let (g,_,_) = self.reg(gamma)?;
                    let (b,_,_) = self.reg(beta)?;
                    let shape = &self.program.registers[ai].shape;
                    let (outer,width,inner) = (
                        shape[..*axis].iter().product::<usize>(),
                        shape[*axis],
                        shape[axis+1..].iter().product::<usize>(),
                    );
                    body += &format!("for(size_t o=0;o<{outer};o++)for(size_t i=0;i<{inner};i++){{double mean=0.;for(size_t a=0;a<{width};a++){{size_t idx=(o*{width}+a)*{inner}+i;mean+=(double)r[{a}+idx];}}mean/={width};double variance=0.;for(size_t a=0;a<{width};a++){{size_t idx=(o*{width}+a)*{inner}+i;double delta=(double)r[{a}+idx]-mean;variance+=delta*delta;}}variance/={width};double inv=1.0/sqrt(variance+{epsilon:.17});for(size_t a=0;a<{width};a++){{size_t idx=(o*{width}+a)*{inner}+i;t[idx]=(float)(((double)r[{a}+idx]-mean)*inv*(double)r[{g}+a]+(double)r[{b}+a]);}}}}");
                },
                I::Scale { input:x,factor,.. } => {
                    let (a,n,_) = self.reg(x)?;
                    body += &format!("for(size_t i=0;i<{n};i++)t[i]=r[{a}+i]*{};",Self::literal(*factor)?);
                },
                I::ScaleBy { input:x,scalar,.. } => {
                    let (a,n,_) = self.reg(x)?;
                    let (s,_,_) = self.reg(scalar)?;
                    body += &format!("for(size_t i=0;i<{n};i++)t[i]=r[{a}+i]*r[{s}];");
                },
                I::SoftmaxAxis { input:x,axis,.. } => {
                    let (a,_,id)=self.reg(x)?;
                    let shape=&self.program.registers[id].shape;
                    let outer:usize=shape[..*axis].iter().product();
                    let width=shape[*axis]; let inner:usize=shape[*axis+1..].iter().product();
                    body += &format!("for(size_t o=0;o<{outer};o++)for(size_t j=0;j<{inner};j++){{float maximum=-INFINITY;for(size_t q=0;q<{width};q++)maximum=fmaxf(maximum,r[{a}+(o*{width}+q)*{inner}+j]);double sum=0.;for(size_t q=0;q<{width};q++)sum+=exp((double)r[{a}+(o*{width}+q)*{inner}+j]-(double)maximum);for(size_t q=0;q<{width};q++){{size_t z=(o*{width}+q)*{inner}+j;t[z]=(float)(exp((double)r[{a}+z]-(double)maximum)/sum);}}}}");
                },
                I::Constant { data, .. } => {
                    for (i, v) in data.iter().enumerate() {
                        body += &format!("t[{i}]={};", Self::literal(*v)?);
                    }
                }
                I::Read { out: o, parameter } => {
                    let (p, n) = self.par(parameter)?;
                    ensure!(self.reg(o)?.1 == n, "compiled read shape mismatch");
                    body += &format!("memcpy(t,p[{p}],{n}*sizeof(float));");
                    match &self.program.parameters[p].learn {
                        Some(wgt_language::adaptive::LearnDomain::Int { .. }) => body += &format!("for(size_t j=0;j<{n};j++)t[j]=roundf(t[j]);"),
                        Some(wgt_language::adaptive::LearnDomain::Bool) => body += &format!("for(size_t j=0;j<{n};j++)t[j]=t[j]>=0.5f?1.0f:0.0f;"),
                        _ => {},
                    }
                }
                I::Linear {
                    out: o,
                    input: x,
                    weight,
                    bias,
                } => {
                    let (a, n, _) = self.reg(x)?;
                    let m = self.reg(o)?.1;
                    let (w, len) = self.par(weight)?;
                    ensure!(
                        n.checked_mul(m) == Some(len),
                        "compiled linear shape mismatch"
                    );
                    let b = if let Some(b) = bias {
                        let (b, len) = self.par(b)?;
                        ensure!(len == m, "compiled bias shape mismatch");
                        format!("p[{b}][i]")
                    } else {
                        "0.0f".into()
                    };
                    body += &format!(
                        "for(size_t i=0;i<{m};i++){{float v={b};for(size_t j=0;j<{n};j++)v+=r[{a}+j]*p[{w}][i*{n}+j];t[i]=v;}}"
                    );
                }
                I::Add {
                    left,
                    right,
                    ..
                }
                | I::Mul {
                    left,
                    right,
                    ..
                }
                | I::Minimum {
                    left,
                    right,
                    ..
                }
                | I::Div {
                    left,
                    right,
                    ..
                }
                | I::Ge {
                    left,
                    right,
                    ..
                } => {
                    let (a, n, _) = self.reg(left)?;
                    let (b, _, _) = self.reg(right)?;
                    let value = match instruction {
                        I::Add { .. } => format!("r[{a}+i]+r[{b}+i]"),
                        I::Mul { .. } => format!("r[{a}+i]*r[{b}+i]"),
                        I::Div { .. } => format!("r[{a}+i]/r[{b}+i]"),
                        I::Ge { .. } => format!("r[{a}+i]>=r[{b}+i]?1.0f:0.0f"),
                        _ => format!("fminf(r[{a}+i],r[{b}+i])"),
                    };
                    if matches!(instruction, I::Ge { .. }) {
                        body += "if(traced)return 12;";
                    }
                    body += &format!("for(size_t i=0;i<{n};i++)t[i]={value};");
                }
                I::Dropout {..} | I::MaskedSoftmax {..} | I::Gelu {..} | I::TopkMask {..} => anyhow::bail!("complex tensor execution requires the reference-interpreter backend"),
                I::Tanh { input: x, .. }
                | I::Sigmoid { input: x, .. }
                | I::Relu { input: x, .. }
                | I::Neg { input: x, .. }
                | I::Detach { input: x, .. }
                | I::Exp { input: x, .. }
                | I::Log { input: x, .. }
                | I::Sqrt { input: x, .. }
                | I::ValidateProbabilities { input: x, .. } => {
                    let (a, n, _) = self.reg(x)?;
                    let value = match instruction {
                        I::Tanh { .. } => format!("tanhf(r[{a}+i])"),
                        I::Sigmoid { .. } => format!("1.0f/(1.0f+expf(-r[{a}+i]))"),
                        I::Relu { .. } => format!("fmaxf(r[{a}+i],0.0f)"),
                        I::Neg { .. } => format!("-r[{a}+i]"),
                        I::Exp { .. } => format!("expf(r[{a}+i])"),
                        I::Log { .. } => format!("logf(r[{a}+i])"),
                        I::Sqrt { .. } => format!("sqrtf(r[{a}+i])"),
                        _ => format!("r[{a}+i]"),
                    };
                    if matches!(instruction, I::ValidateProbabilities { .. }) {
                        body += &format!(
                            "double sum=0.;for(size_t i=0;i<{n};i++){{if(r[{a}+i]<0.f||r[{a}+i]>1.f)return 4;sum+=r[{a}+i];}}if(fabs(sum-1.)>1e-5)return 4;"
                        );
                    }
                    body += &format!("for(size_t i=0;i<{n};i++)t[i]={value};");
                }
                I::Softmax { input: x, .. } | I::LogSoftmax { input: x, .. } => {
                    let (a, n, _) = self.reg(x)?;
                    body += &format!(
                        "float maximum=-INFINITY;for(size_t i=0;i<{n};i++)maximum=fmaxf(maximum,r[{a}+i]);double sum=0.;for(size_t i=0;i<{n};i++)sum+=exp((double)r[{a}+i]-(double)maximum);"
                    );
                    let value = if matches!(instruction, I::Softmax { .. }) {
                        format!("exp((double)r[{a}+i]-maximum)/sum")
                    } else {
                        format!("(double)r[{a}+i]-maximum-log(sum)")
                    };
                    body += &format!("for(size_t i=0;i<{n};i++)t[i]=(float)({value});");
                }
                I::Sum { input: x, .. } | I::Mean { input: x, .. } => {
                    let (a, n, _) = self.reg(x)?;
                    body += &format!(
                        "double sum=0.;for(size_t i=0;i<{n};i++)sum+=r[{a}+i];t[0]=(float)(sum/{});",
                        if matches!(instruction, I::Mean { .. }) {
                            n
                        } else {
                            1
                        }
                    );
                }
                I::Clip {
                    input: x,
                    low: min,
                    high: max,
                    ..
                } => {
                    let (a, n, _) = self.reg(x)?;
                    body += &format!(
                        "for(size_t i=0;i<{n};i++)t[i]=fminf(fmaxf(r[{a}+i],{}),{});",
                        Self::literal(*min)?,
                        Self::literal(*max)?
                    );
                }
                I::Embedding { .. } | I::Concat { .. } | I::Gather { .. } | I::DynamicSlice { .. } =>
                    bail!("runtime indexing requires the reference-interpreter backend"),
                I::Slice {
                    out: o,
                    input: x,
                    start,
                } => {
                    let (a, _, _) = self.reg(x)?;
                    let m = self.reg(o)?.1;
                    body += &format!("memcpy(t,r+{},{}*sizeof(float));", a + start, m);
                }
                I::Label { .. } => {}
                I::Jump { target: name } => body += &format!("goto L{};", target(name)?),
                I::Branch {
                    condition,
                    positive,
                    otherwise,
                } => {
                    let (a, n, _) = self.reg(condition)?;
                    ensure!(n == 1, "compiled branch condition must be scalar");
                    body += &format!(
                        "if(r[{a}]>0.f)goto L{};else goto L{};",
                        target(positive)?,
                        target(otherwise)?
                    );
                }
                I::Return { value } => {
                    let (a, n, _) = self.reg(value)?;
                    ensure!(
                        output_width.is_none_or(|old| old == n),
                        "compiled return shape mismatch"
                    );
                    output_width = Some(n);
                    if self.program.output_type == crate::categorical::VectorType::Probabilities {
                        body += &format!(
                            "double sum=0.;for(size_t i=0;i<{n};i++){{if(r[{a}+i]<0.f||r[{a}+i]>1.f)return 4;sum+=r[{a}+i];}}if(fabs(sum-1.)>1e-5)return 4;"
                        );
                    }
                    body += &format!(
                        "if(trace&&trace(context,{index},r+{a},{n}))return 5;memcpy(output,r+{a},{n}*sizeof(float));return 0;"
                    );
                }
                I::Gate {
                    probability,
                    input: x,
                    ..
                }
                | I::Route {
                    probability,
                    yes: x,
                    ..
                } => {
                    // Endpoint inference only reads the selected branch; traced calls read both.
                    reads = [Some(probability.as_str()), None, None];
                    let (a, n, aid) = self.reg(x)?;
                    let (p, pn, _) = self.reg(probability)?;
                    let other = if let I::Route { no, .. } = instruction {
                        let (b, _, bid) = self.reg(no)?;
                        Some((b, bid))
                    } else {
                        None
                    };
                    body += &format!(
                        "int zero=1,one=1;for(size_t i=0;i<{pn};i++){{float v=r[{p}+i];if(v<0.f||v>1.f)return 4;zero&=v==0.f;one&=v==1.f;}}"
                    );
                    let bvalue = other.map_or("0.0f".into(), |(b, _)| format!("r[{b}+i]"));
                    let bready = other.map_or(String::new(), |(_, id)| {
                        format!("if(!ready[{id}])return 3;")
                    });
                    body += &format!(
                        "if(!traced&&zero){{{bready}for(size_t i=0;i<{n};i++)t[i]={bvalue};}}else if(!traced&&one){{if(!ready[{aid}])return 3;memcpy(t,r+{a},{n}*sizeof(float));}}else{{if(!ready[{aid}])return 3;{bready}for(size_t i=0;i<{n};i++){{float v=r[{p}+{}];t[i]=v*r[{a}+i]+(1.f-v)*({bvalue});}}}}",
                        if pn == 1 { "0" } else { "i" }
                    );
                }
                I::InferenceSkip {
                    probability,
                    when_one,
                    target: name,
                } => {
                    let (p, n, _) = self.reg(probability)?;
                    body += &format!(
                        "if(!traced){{int skip=1;for(size_t i=0;i<{n};i++){{float v=r[{p}+i];if(v<0.f||v>1.f)return 4;skip&=v=={}.f;}}if(skip)goto L{};}}",
                        if *when_one { 1 } else { 0 },
                        target(name)?
                    );
                }

                I::Write { parameter, value } => {
                    let (p, n) = self.par(parameter)?;
                    let (a, m, _) = self.reg(value)?;
                    ensure!(n == m && !self.program.parameters[p].frozen, "invalid transactional write");
                    body += &format!("for(size_t i=0;i<{n};i++)if(!isfinite(r[{a}+i]))return 1;memcpy(candidate[{p}],r+{a},{n}*sizeof(float));if(trace&&trace(context,{index},r+{a},{n}))return 5;");
                }
                I::MemoryAlloc { out, .. } => {
                    let (id, slots, width) = self.memory(out)?;
                    let len = slots.checked_mul(width).context("memory size overflow")?;
                    let bytes = len.checked_mul(4).context("memory size overflow")?;
                    body += &format!("if(memory[{id}])return 10;if({bytes}>memory_available-memory_used)return 7;memory[{id}]=calloc({len},sizeof(float));if(!memory[{id}])return 8;memory_used+={bytes};if(trace&&trace(context,{index},memory[{id}],{len}))return 5;");
                }
                I::MemoryFree { memory } => {
                    let (id, slots, width) = self.memory(memory)?;
                    let bytes = slots.checked_mul(width).and_then(|n|n.checked_mul(4)).context("memory size overflow")?;
                    // A non-null dummy pointer keeps the zero-length Rust callback slice valid.
                    body += &format!("if(!memory[{id}])return 9;free(memory[{id}]);memory[{id}]=NULL;memory_used-={bytes};if(trace&&trace(context,{index},r,0))return 5;");
                }
                I::MemoryRead { out, memory, address } => {
                    let (id, slots, width) = self.memory(memory)?;
                    let (a, n, _) = self.reg(address)?;
                    ensure!(n == slots && self.reg(out)?.1 == width, "compiled memory read shape mismatch");
                    body += &format!("for(size_t i=0;i<{slots};i++)if(!isfinite(r[{a}+i])||r[{a}+i]<0.f||r[{a}+i]>1.f)return 11;for(size_t j=0;j<{width};j++){{double sum=0.;for(size_t i=0;i<{slots};i++)sum+=(double)r[{a}+i]*(double)memory[{id}][i*{width}+j];t[j]=(float)sum;}}");
                }
                I::MemoryWrite { memory, address, value } => {
                    let (id, slots, width) = self.memory(memory)?;
                    let (a, n, _) = self.reg(address)?;
                    let (v, m, _) = self.reg(value)?;
                    ensure!(n == slots && m == width, "compiled memory write shape mismatch");
                    let len = slots.checked_mul(width).context("memory size overflow")?;
                    body += &format!("for(size_t i=0;i<{slots};i++)if(!isfinite(r[{a}+i])||r[{a}+i]<0.f||r[{a}+i]>1.f)return 11;for(size_t i=0;i<{slots};i++)for(size_t j=0;j<{width};j++){{double a=r[{a}+i];size_t z=i*{width}+j;memory[{id}][z]=(float)((1.-a)*(double)memory[{id}][z]+a*(double)r[{v}+j]);if(!isfinite(memory[{id}][z]))return 1;}}if(trace&&trace(context,{index},memory[{id}],{len}))return 5;");
                }
                I::ResourceState { out } => {
                    ensure!(self.reg(out)?.1 == 3, "resource state output must have three elements");
                    body += "t[0]=(float)(budget-*used);t[1]=(float)(memory_available-memory_used);t[2]=(float)memory_used;";
                }
            }
            for name in reads.into_iter().flatten() {
                let (_, _, id) = self.reg(name)?;
                code += &format!("if(!ready[{id}])return 3;");
            }
            code += &body;
            if let Some(out) = metadata.output {
                let (a, n, id) = self.reg(out)?;
                code += &format!(
                    "for(size_t i=0;i<{n};i++)if(!isfinite(t[i]))return 1;memcpy(r+{a},t,{n}*sizeof(float));ready[{id}]=1;if(trace&&trace(context,{index},r+{a},{n}))return 5;"
                );
            }
            code += "}\n";
        }
        code += "return 6;}\n";
        // All exits, including budget exhaustion and rejected trace callbacks, free every bank.
        // The execution function cannot bypass this cleanup boundary with an early return.
        code += &format!("int wgt_forward_v2(const float *input,const float *const *p,float *r,float *output,uint64_t budget,size_t memory_available,uint64_t *used,int traced,trace_fn trace,void *context){{float *memory[{}]={{0}};int status=execute(input,p,r,output,budget,memory_available,used,traced,trace,context,memory);for(size_t i=0;i<{};i++)free(memory[i]);return status;}}\n", self.registers.len().max(1), self.registers.len());
        ensure!(
            code.len() <= 8 * 1024 * 1024,
            "compiled source byte limit exceeded"
        );
        Ok(Generated {
            source: code,
            input,
            output: output_width.context("compiled forward requires return")?,
            workspace: self
                .workspace
                .checked_add(self.max)
                .and_then(|n| n.checked_add(transaction_bytes))
                .context("workspace overflow")?,
            candidates,
        })
    }
}
