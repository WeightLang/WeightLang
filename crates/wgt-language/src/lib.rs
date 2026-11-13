//! Static WGT syntax, lowering and artifact schemas; no model execution.
pub mod backend;
pub mod script;
pub mod source;
pub mod adaptive;
pub mod tensor;
pub mod categorical;
pub mod weight_artifact;
pub mod model;
pub mod operations;
pub mod initializer;
pub mod dimensions;
pub mod model_template;
mod model_syntax;
mod model_modules;

pub mod ide;
pub mod format;

/// PyTorch-target dynamic model syntax, checked without executing Python.
pub mod torch_model;
