//! Rust owns application control; WGT defines model architecture and tensor computation.
//! Definitions are pure data. Only explicit initialization connects them to service state.
//! Clients and models use `Rc<RefCell<_>>` and are deliberately neither Send nor Sync.
use anyhow::{ensure, Context};
pub use anyhow::Result;
use std::{cell::RefCell, fs::OpenOptions, path::Path, rc::Rc};
use wgtime::Request;
pub use wgt_language::model::{ModelDefinition, ModelInterface, ParameterSchema};
pub use wgt_model_macros::{model, include_model};
pub use wgt_language::operations::{ArgumentSchema, Effects, OperationSchema, StateSchema};
pub mod evolution;
pub mod weights;
/// ContinuousOnly conservatively rejects every model containing bool/int parameters.
#[derive(Clone, Copy, Debug)]
pub enum DifferentiationPolicy { ContinuousOnly, StraightThrough }
pub type ParameterCandidate = (String, Vec<usize>, Vec<f32>);

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
        self.model_from_reply(definition, reply)
    }
    /// Construct an instance from complete safetensors values, without running parameter initializers.
    /// Runtime state uses its declared initialization; weights never restore optimizer/state implicitly.
    pub fn load_weights(&self, definition: &ModelDefinition, path: &Path, options: InstanceOptions) -> Result<Model> {
        self.load_weights_pinned(definition, path, None, options)
    }
    /// Load a manifest-backed immutable snapshot and pin the bytes checked by the service.
    pub fn load_snapshot(&self, definition: &ModelDefinition, directory: &Path, revision: Option<&str>, options: InstanceOptions) -> Result<Model> {
        let snapshot = weights::WeightSet::open(directory, revision)?;
        snapshot.validate_for(definition)?;
        self.load_weights_pinned(definition, &directory.join("weights.safetensors"), Some(snapshot.revision().into()), options)
    }
    fn load_weights_pinned(&self, definition: &ModelDefinition, path: &Path, revision: Option<String>, options: InstanceOptions) -> Result<Model> {
        let file = OpenOptions::new().read(true).open(path)?;
        ensure!(file.metadata()?.is_file(), "weights require regular file");
        let reply = self.connection.borrow_mut().call_with_file(Request::InitializeWeightsFd {
            source: definition.source().into(), memory: options.memory_bytes,
            optimizer: options.optimizer.map(|o| o.name().into()), revision,
        }, &file)?;
        self.model_from_reply(definition, reply)
    }
    fn model_from_reply(&self, definition: &ModelDefinition, reply: serde_json::Value) -> Result<Model> {
        Ok(Model { connection:self.connection.clone(), handle:Some(reply["handle"].as_u64().context("missing model handle")?), definition:definition.clone(), interface:definition.interface().cloned(), operations:definition.operations().to_vec(), backend:reply["backend"].as_str().context("missing execution backend")?.into() })
    }
    pub fn stats(&self) -> Result<serde_json::Value> { self.connection.borrow_mut().call(Request::Stats) }
}

pub struct Model { connection: Rc<RefCell<wgtime::Client>>, handle: Option<u64>, definition: ModelDefinition, interface: Option<ModelInterface>, operations: Vec<OperationSchema>, backend: String }
impl Model {
    fn handle(&self)->Result<u64> { self.handle.context("model is closed") }
    pub fn definition(&self)->&ModelDefinition { &self.definition }
    pub fn interface(&self)->Option<&ModelInterface> { self.interface.as_ref() }
    pub fn operations(&self)->&[OperationSchema] { &self.operations }
    pub fn call(&self,name:&str,inputs:&[Vec<f32>])->Result<Option<Vec<f32>>> {
        if name == "forward" && self.operations.is_empty() && self.interface.is_some() {
            ensure!(inputs.len() == 1, "forward requires one input vector");
            return self.forward(&inputs[0]).map(Some);
        }
        let schema=self.operations.iter().find(|s|s.name==name).context("unknown operation")?;
        ensure!(inputs.len()==schema.inputs.len() && inputs.iter().zip(&schema.inputs).all(|(v,s)|v.len()==s.width && v.iter().all(|x|x.is_finite())),"operation arguments do not match schema");
        let reply=self.connection.borrow_mut().call(Request::CallOperation{handle:self.handle()?,name:name.into(),inputs:inputs.into()})?;
        let output:Option<Vec<f32>>=serde_json::from_value(reply.get("output").context("missing operation output")?.clone())?;
        ensure!(match (&output,schema.output) {(None,None)=>true,(Some(v),Some(n))=>v.len()==n && v.iter().all(|x|x.is_finite()),_=>false},"invalid service operation output");
        Ok(output)
    }
    pub fn parameters(&self)->Result<Vec<ParameterCandidate>> { Ok(serde_json::from_value(self.connection.borrow_mut().call(Request::ParameterValues{handle:self.handle()?})?)?) }
    /// Capture one complete parameter snapshot; this excludes runtime and optimizer state.
    pub fn snapshot(&self) -> Result<weights::WeightSet> {
        weights::WeightSet::new(self.parameters()?.into_iter().map(|(name,shape,data)| {
            Ok((name,weights::Tensor::new(shape,data)?))
        }).collect::<Result<_>>()?)
    }
    pub fn state(&self)->Result<Vec<ParameterCandidate>> { Ok(serde_json::from_value(self.connection.borrow_mut().call(Request::StateValues{handle:self.handle()?})?)?) }
    /// Atomically replace a validated candidate batch; clears legacy optimizer state.
    pub fn update_parameters(&self,candidates:&[ParameterCandidate])->Result<()> {
        self.connection.borrow_mut().call(Request::UpdateParameters{handle:self.handle()?,candidates:candidates.into()})?; Ok(())
    }
    pub fn train_with_policy(&self,inputs:&[Vec<f32>],targets:Targets<'_>,options:TrainOptions,policy:DifferentiationPolicy)->Result<f32> {
        self.train_internal(inputs,targets,options,Some(policy))
    }
    /// Execution backend reported by the initialized service instance.
    pub fn backend(&self)->&str { &self.backend }
    /// Execute one finite f32 input vector and commit explicit forward writes on success.
    /// Failed execution preserves parameters. Typed multi-input models use `call`.
    pub fn forward(&self, input:&[f32])->Result<Vec<f32>> {
        let interface=self.interface.as_ref().context("typed model requires an explicit named operation call")?;
        ensure!(input.len()==interface.input && input.iter().all(|n|n.is_finite()), "input must be a finite vector of width {}",interface.input);
        let reply=self.connection.borrow_mut().call(Request::Predict {handle:self.handle()?,input:input.into()})?;
        let output:Vec<f32>=serde_json::from_value(reply.get("output").context("missing output")?.clone())?;
        ensure!(output.len()==interface.output && output.iter().all(|n|n.is_finite()),"invalid service output");
        Ok(output)
    }
    /// CPU SGD/Adam only, selected explicitly at initialization or weight binding.
    pub fn train(&self, inputs:&[Vec<f32>], targets:Targets<'_>, options:TrainOptions)->Result<f32> {
        self.train_internal(inputs,targets,options,None)
    }
    fn train_internal(&self, inputs:&[Vec<f32>], targets:Targets<'_>, options:TrainOptions, policy:Option<DifferentiationPolicy>)->Result<f32> {
        let interface=self.interface.as_ref().context("typed operations do not support differentiation or gradients across state")?;
        ensure!(!inputs.is_empty() && inputs.iter().all(|row| row.len()==interface.input && row.iter().all(|n|n.is_finite())), "training inputs must be a nonempty finite matrix of model input width");
        ensure!(options.rate.is_finite() && options.rate>0., "learning rate must be positive and finite");
        ensure!(options.clip.is_none_or(|n|n.is_finite() && n>0.), "gradient clip norm must be positive and finite");
        match &targets {
            Targets::Values(rows) => ensure!(rows.len()==inputs.len() && rows.iter().all(|row|row.iter().all(|n|n.is_finite())), "targets must be finite and match batch size"),
            Targets::Classes(rows) => ensure!(rows.len()==inputs.len() && rows.iter().all(|n|*n<interface.output), "class targets must match batch size and output range"),
        }
        let targets=match (options.loss,targets) {
            (Loss::CrossEntropy,Targets::Classes(values))=>serde_json::to_value(values)?,
            (Loss::Mse|Loss::Custom,Targets::Values(values))=>serde_json::to_value(values)?,
            _=>anyhow::bail!("loss and target kind do not match"),
        };
        let loss=match options.loss {Loss::Mse=>"mse",Loss::CrossEntropy=>"cross_entropy",Loss::Custom=>"loss"}.into();
        let request=if let Some(policy)=policy {Request::TrainStepWithPolicy{handle:self.handle()?,inputs:inputs.into(),targets,rate:options.rate,loss,clip:options.clip,straight_through:matches!(policy,DifferentiationPolicy::StraightThrough)}} else {Request::TrainStep{handle:self.handle()?,inputs:inputs.into(),targets,rate:options.rate,loss,clip:options.clip}};
        let reply=self.connection.borrow_mut().call(request)?;
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
        wgtime::publication::write(path, wgtime::publication::Mode::CreateNew, |file| {
            self.connection.borrow_mut().call_with_file(Request::ExportWeightsFd {handle:self.handle()?},file)?;
            Ok(())
        })
    }
    pub fn close(mut self)->Result<()> {
        self.connection.borrow_mut().call(Request::Drop {handle:self.handle()?})?;
        self.handle=None;Ok(())
    }
}
impl Drop for Model {
    fn drop(&mut self) { if let Some(handle)=self.handle.take() { let _=self.connection.borrow_mut().call(Request::Drop {handle}); } }
}
