//! Explicit mutually exclusive category-vector contracts.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// Element storage remains f32 in models; this describes the vector's meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorType {
    #[default]
    F32,
    Logits,
    Probabilities,
}
impl VectorType {
    pub fn is_plain(&self) -> bool { *self == Self::F32 }
}

/// Absolute tolerance for the sum of a declared probability vector.
/// Validation never rescales the caller's values.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-5;

pub fn validate_finite(values: &[f64]) -> Result<()> {
    ensure!(!values.is_empty(), "category vector must not be empty");
    ensure!(values.iter().all(|x| x.is_finite()), "category vector must contain only finite values");
    Ok(())
}

pub fn validate_probabilities(values: &[f64]) -> Result<()> {
    validate_finite(values)?;
    ensure!(values.iter().all(|x| (0. ..=1.).contains(x)), "probabilities must each be in [0,1]");
    let sum: f64 = values.iter().sum();
    ensure!((sum - 1.).abs() <= PROBABILITY_SUM_TOLERANCE,
        "probabilities must sum to 1 within absolute tolerance {PROBABILITY_SUM_TOLERANCE}; got {sum}");
    Ok(())
}

/// Stable softmax. Subtract in f64, including when the source storage is f32.
pub fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    validate_finite(values)?;
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut result: Vec<f64> = values.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = result.iter().sum();
    for x in &mut result { *x /= sum; }
    Ok(result)
}

/// Explicit conversion never guesses whether an ordinary vector is categorical.
pub fn probabilities(kind: VectorType, values: &[f64]) -> Result<Vec<f64>> {
    match kind {
        VectorType::F32 | VectorType::Logits => softmax(values),
        VectorType::Probabilities => {
            validate_probabilities(values)?;
            Ok(values.to_vec())
        }
    }
}

/// Draw from actual stored mass without rewriting a probability vector.
/// A half-open threshold and explicit zero skip prevent zero-mass categories
/// from winning at RNG zero or at cumulative boundaries.
pub fn sample_probabilities(values: &[f64], unit_draw: f64) -> Result<usize> {
    validate_probabilities(values)?;
    ensure!(unit_draw.is_finite() && (0. ..1.).contains(&unit_draw), "RNG draw must be in [0,1)");
    let total: f64 = values.iter().sum();
    let mut remaining = unit_draw * total;
    let mut last_positive = None;
    for (index, mass) in values.iter().copied().enumerate() {
        if mass == 0. { continue; }
        last_positive = Some(index);
        if remaining < mass { return Ok(index); }
        remaining -= mass;
    }
    // Rounding can leave a tiny residual at the upper endpoint.
    Ok(last_positive.expect("validated positive total"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_extremes_and_translation() {
        assert_eq!(softmax(&[f64::MAX, -f64::MAX]).unwrap(), vec![1., 0.]);
        let a = softmax(&[-1000., 0., 1000.]).unwrap();
        let b = softmax(&[0., 1000., 2000.]).unwrap();
        assert_eq!(a, b);
        assert_eq!(softmax(&[1e30, 1e30]).unwrap(), vec![0.5, 0.5]);
    }
    #[test]
    fn strict_probability_contract_and_identity() {
        let p = [0.1, 0.2, 0.700001];
        assert_eq!(probabilities(VectorType::Probabilities, &p).unwrap(), p);
        for p in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![-0.1, 1.1], vec![0.2, 0.2], vec![1.000001]] {
            assert!(validate_probabilities(&p).is_err(), "{p:?}");
        }
        assert_eq!(probabilities(VectorType::F32, &[0.2, 0.8]).unwrap(), softmax(&[0.2, 0.8]).unwrap());
        for kind in [VectorType::F32, VectorType::Logits, VectorType::Probabilities] {
            for values in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![f64::NEG_INFINITY]] {
                assert!(probabilities(kind, &values).is_err());
            }
        }
        assert!(softmax(&[]).is_err());
        assert!(softmax(&[f64::NEG_INFINITY]).is_err());
    }
    #[test]
    fn sampling_skips_zero_mass_and_scales_actual_total() {
        for total in [0.999995, 1., 1.000005] {
            for p in [vec![0., total / 2., 0., total / 2., 0.], vec![0., total / 3., 2. * total / 3., 0.]] {
                for draw in [0., 0.25, 0.5, 0.75, 1. - f64::EPSILON] {
                    let index = sample_probabilities(&p, draw).unwrap();
                    assert!(p[index] > 0.);
                }
            }
            assert_eq!(sample_probabilities(&[0., total / 2., 0., total / 2., 0.], 0.5).unwrap(), 3);
        }
        assert!(sample_probabilities(&[1.], 1.).is_err());
    }
    #[test]
    fn maximum_width_softmax_satisfies_probability_contract() {
        for values in [vec![0.; 16384], (0..16384).map(|i| i as f64 / 17. - 500.).collect()] {
            let p = softmax(&values).unwrap();
            validate_probabilities(&p).unwrap();
            let stored: Vec<f64> = p.iter().map(|x| (*x as f32) as f64).collect();
            validate_probabilities(&stored).unwrap();
        }
    }
    #[test]
    fn softmax_vjp_matches_finite_difference() {
        let x = [-3., 0.25, 4.];
        let upstream = [0.3, -0.8, 1.2];
        let p = softmax(&x).unwrap();
        let dot: f64 = p.iter().zip(upstream).map(|(p, g)| p * g).sum();
        for i in 0..x.len() {
            let mut plus = x; plus[i] += 1e-5;
            let mut minus = x; minus[i] -= 1e-5;
            let objective = |x: &[f64]| softmax(x).unwrap().iter().zip(upstream).map(|(p, g)| p * g).sum::<f64>();
            let numerical = (objective(&plus) - objective(&minus)) / 2e-5;
            let analytic = p[i] * (upstream[i] - dot);
            assert!((numerical - analytic).abs() < 1e-9);
        }
    }
}
