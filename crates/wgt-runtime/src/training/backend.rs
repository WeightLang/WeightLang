use super::{cpu, Objective};
use crate::Session;
use anyhow::Result;
use std::fmt;

/// Explicit training backend. CPU remains the default.
#[derive(Debug, Default)]
pub enum TrainingBackend {
    #[default]
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(super::GpuTraining),
}

/// GPU failures are downcastable from the existing Trainer anyhow::Result API.
#[derive(Debug)]
#[non_exhaustive]
pub enum TrainingError {
    Unavailable(String),
    Unsupported(&'static str),
    InvalidInput(&'static str),
    OutOfBudget,
    NonFinite(&'static str),
    Device(String),
}
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(s) => write!(f, "training device unavailable: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported training operation: {s}"),
            Self::InvalidInput(s) => write!(f, "invalid training input: {s}"),
            Self::OutOfBudget => write!(f, "training memory or device limit exceeded"),
            Self::NonFinite(s) => write!(f, "nonfinite training {s}"),
            Self::Device(s) => write!(f, "training device failure: {s}"),
        }
    }
}
impl std::error::Error for TrainingError {}

impl TrainingBackend {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn device_id(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => &gpu.device_info().id,
        }
    }
    pub(super) fn gradients(
        &mut self,
        session: &mut Session,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        match self {
            Self::Cpu => cpu::gradients(session, inputs, objective),
            #[cfg(feature = "gpu")]
            Self::Gpu(gpu) => gpu.gradients(session, inputs, objective),
        }
    }
}
