//! CPU optimizer policy; callers own storage selection and transactional commits.
use super::Optimizer;
use anyhow::{Result, ensure};

pub(crate) struct GradientClip(Option<f32>);

impl GradientClip {
    pub(crate) fn new(limit: Option<f32>) -> Result<Self> {
        ensure!(
            limit.is_none_or(|value| value.is_finite() && value > 0.),
            "gradient clip norm must be positive and finite"
        );
        Ok(Self(limit))
    }

    /// Expression derivatives must already be folded into their writable owners.
    pub(crate) fn apply(&self, gradients: &mut [Vec<f32>], writable: impl Fn(usize) -> bool) {
        let Some(limit) = self.0 else { return };
        let norm = gradients.iter().enumerate()
            .filter(|(index, _)| writable(*index))
            .flat_map(|(_, values)| values)
            .map(|gradient| f64::from(*gradient).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm > f64::from(limit) {
            let scale = f64::from(limit) / norm;
            for (index, values) in gradients.iter_mut().enumerate() {
                if writable(index) {
                    for gradient in values {
                        *gradient = (f64::from(*gradient) * scale) as f32;
                    }
                }
            }
        }
    }
}

impl Optimizer {
    /// Update staged moments for one gradient at the one-based optimizer step.
    pub(crate) fn direction(self, gradient: f32, first: &mut f32, second: &mut f32, step: u64) -> f32 {
        match self {
            Self::Sgd => gradient,
            Self::Adam => {
                *first = 0.9 * *first + 0.1 * gradient;
                *second = 0.999 * *second + 0.001 * gradient * gradient;
                let m = *first / (1. - 0.9f32.powf(step as f32));
                let v = *second / (1. - 0.999f32.powf(step as f32));
                m / (v.sqrt() + 1e-8)
            }
        }
    }
}
