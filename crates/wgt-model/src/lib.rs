//! Rust owns application control; WGT defines model architecture and tensor computation.
//! Definitions are pure data. Only explicit initialization connects them to service state.
//! Clients and models use `Rc<RefCell<_>>` and are deliberately neither Send nor Sync.
use anyhow::{ensure, Context, Result};
use std::{cell::RefCell, fs::{self, OpenOptions}, path::Path, rc::Rc};
use wgtime::Request;
pub use wgt_language::model::{ModelDefinition, ModelInterface, ParameterSchema};
pub use wgt_model_macros::{model, include_model};

#[derive(Clone, Copy, Debug)]
pub enum Optimizer { Sgd, Adam }
impl Optimizer { fn name(self) -> &'static str { match self { Self::Sgd=>"sgd", Self::Adam=>"adam" } } }
#[derive(Clone, Copy, Debug)]
pub struct InstanceOptions { pub memory_bytes: usize, pub optimizer: Option<Optimizer> }
impl Default for InstanceOptions { fn default()->Self { Self { memory_bytes: 64*1024*1024, optimizer: None } } }
#[derive(Clone, Copy, Debug)]
pub enum Loss { Mse, CrossEntropy, Custom }
#[derive(Clone, Copy, Debug)]
pub struct TrainOptions { pub rate: f32, pub loss: Loss, pub clip: Option<f32> }
pub enum Targets<'a> { Values(&'a [Vec<f32>]), Classes(&'a [usize]) }

#[derive(Clone)]
pub struct Client { connection: Rc<RefCell<wgtime::Client>> }
impl Client {
    pub fn connect() -> Result<Self> {
        let endpoint=std::env::var_os("WGTIME_ENDPOINT").unwrap_or_else(||wgtime::DEFAULT_ENDPOINT.into());
        Self::connect_to(Path::new(&endpoint))
    }
    pub fn connect_to(endpoint: &Path) -> Result<Self> {
        Ok(Self { connection: Rc::new(RefCell::new(wgtime::Client::connect(endpoint, unsafe {libc::geteuid()})?)) })
    }
    /// Initialize parameters explicitly; parsing a definition never calls this method.
    pub fn initialize(&self, definition: &ModelDefinition, options: InstanceOptions) -> Result<Model> {
        let reply=self.connection.borrow_mut().call(Request::InitializeDefinition {source:definition.source().into(),memory:options.memory_bytes,optimizer:options.optimizer.map(|o|o.name().into())})?;
        Ok(Model { connection:self.connection.clone(), handle:Some(reply["handle"].as_u64().context("missing model handle")?), interface:definition.interface().clone(), backend:reply["backend"].as_str().context("missing execution backend")?.into() })
    }
    pub fn stats(&self) -> Result<serde_json::Value> { self.connection.borrow_mut().call(Request::Stats) }
}

pub struct Model { connection: Rc<RefCell<wgtime::Client>>, handle: Option<u64>, interface: ModelInterface, backend: String }
impl Model {
    fn handle(&self)->Result<u64> { self.handle.context("model is closed") }
    pub fn interface(&self)->&ModelInterface { &self.interface }
    /// Execution backend reported by the initialized service instance.
    pub fn backend(&self)->&str { &self.backend }
    /// One finite f32 input vector; multi-input models are not supported in this slice.
    pub fn forward(&self, input:&[f32])->Result<Vec<f32>> {
        ensure!(input.len()==self.interface.input && input.iter().all(|n|n.is_finite()), "input must be a finite vector of width {}",self.interface.input);
        let reply=self.connection.borrow_mut().call(Request::Predict {handle:self.handle()?,input:input.into()})?;
        let output:Vec<f32>=serde_json::from_value(reply.get("output").context("missing output")?.clone())?;
        ensure!(output.len()==self.interface.output && output.iter().all(|n|n.is_finite()),"invalid service output");
        Ok(output)
    }
    /// CPU SGD/Adam only, selected explicitly at initialization or weight binding.
    pub fn train(&self, inputs:&[Vec<f32>], targets:Targets<'_>, options:TrainOptions)->Result<f32> {
        ensure!(!inputs.is_empty() && inputs.iter().all(|row| row.len()==self.interface.input && row.iter().all(|n|n.is_finite())), "training inputs must be a nonempty finite matrix of model input width");
        ensure!(options.rate.is_finite() && options.rate>0., "learning rate must be positive and finite");
        ensure!(options.clip.is_none_or(|n|n.is_finite() && n>0.), "gradient clip norm must be positive and finite");
        match &targets {
            Targets::Values(rows) => ensure!(rows.len()==inputs.len() && rows.iter().all(|row|row.iter().all(|n|n.is_finite())), "targets must be finite and match batch size"),
            Targets::Classes(rows) => ensure!(rows.len()==inputs.len() && rows.iter().all(|n|*n<self.interface.output), "class targets must match batch size and output range"),
        }
        let targets=match (options.loss,targets) {
            (Loss::CrossEntropy,Targets::Classes(values))=>serde_json::to_value(values)?,
            (Loss::Mse|Loss::Custom,Targets::Values(values))=>serde_json::to_value(values)?,
            _=>anyhow::bail!("loss and target kind do not match"),
        };
        let reply=self.connection.borrow_mut().call(Request::TrainStep {handle:self.handle()?,inputs:inputs.into(),targets,rate:options.rate,loss:match options.loss {Loss::Mse=>"mse",Loss::CrossEntropy=>"cross_entropy",Loss::Custom=>"loss"}.into(),clip:options.clip})?;
        let loss=reply["loss"].as_f64().context("missing loss")? as f32;
        ensure!(loss.is_finite(),"nonfinite service loss");Ok(loss)
    }
    /// Bind weights to an existing initialized architecture; never guesses a graph.
    /// Success resets optimizer state; failure preserves the existing instance.
    pub fn load_weights(&self,path:&Path,optimizer:Option<Optimizer>)->Result<()> {
        let file=OpenOptions::new().read(true).open(path)?;
        ensure!(file.metadata()?.is_file(),"weights require regular file");
        self.connection.borrow_mut().call_with_file(Request::BindWeightsFd {handle:self.handle()?,optimizer:optimizer.map_or("none",Optimizer::name).into()},&file)?;
        Ok(())
    }
    /// Atomically publish a new standard safetensors file; never overwrite a target.
    pub fn save_weights(&self,path:&Path)->Result<()> {
        use std::sync::atomic::{AtomicU64,Ordering};
        static NEXT:AtomicU64=AtomicU64::new(0);
        let parent=path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary { fn drop(&mut self) { let _=fs::remove_file(&self.0); } }
        let (temporary,file)=loop {
            let candidate=parent.join(format!(".wgt-weights-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            match OpenOptions::new().write(true).create_new(true).open(&candidate) {
                Ok(file)=>break(Temporary(candidate),file),
                Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>continue,
                Err(e)=>return Err(e.into()),
            }
        };
        self.connection.borrow_mut().call_with_file(Request::ExportWeightsFd {handle:self.handle()?},&file)?;
        file.sync_all()?;fs::hard_link(&temporary.0,path)?;Ok(())
    }
    pub fn close(mut self)->Result<()> {
        self.connection.borrow_mut().call(Request::Drop {handle:self.handle()?})?;
        self.handle=None;Ok(())
    }
}
impl Drop for Model {
    fn drop(&mut self) { if let Some(handle)=self.handle.take() { let _=self.connection.borrow_mut().call(Request::Drop {handle}); } }
}
