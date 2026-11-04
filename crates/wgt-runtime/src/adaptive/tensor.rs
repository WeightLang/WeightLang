//! Row-major tensor geometry shared by native tensor instructions.
//! Storage is flat; geometry is never inferred from equal element counts.
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct AxisGroups {
    pub outer: usize,
    pub width: usize,
    pub inner: usize,
}
impl AxisGroups {
    pub fn new(shape: &[usize], axis: usize) -> Result<Self> {
        ensure!(!shape.is_empty() && shape.len() <= 8, "tensor rank must be 1..8");
        ensure!(axis < shape.len(), "tensor axis out of bounds");
        let product = |dims: &[usize]| dims.iter().try_fold(1usize, |n, &d| {
            ensure!(d > 0 && d <= 16384, "tensor dimensions must be 1..16384");
            n.checked_mul(d).context("tensor shape overflow")
        });
        let outer = product(&shape[..axis])?;
        let width = product(&shape[axis..=axis])?;
        let inner = product(&shape[axis + 1..])?;
        outer.checked_mul(width).and_then(|n| n.checked_mul(inner)).context("tensor shape overflow")?;
        Ok(Self { outer, width, inner })
    }
    pub fn len(self) -> usize { self.outer * self.width * self.inner }
    fn index(self, outer: usize, axis: usize, inner: usize) -> usize {
        (outer * self.width + axis) * self.inner + inner
    }
    pub fn rms(self, input: &[f32], epsilon: f32, output: &mut [f32]) -> Result<()> {
        ensure!(input.len() == self.len() && output.len() == self.len(), "rms tensor storage/shape mismatch");
        ensure!(epsilon.is_finite() && epsilon > 0., "normalization epsilon must be positive and finite");
        for outer in 0..self.outer {
            for inner in 0..self.inner {
                let mean_square = (0..self.width).map(|a| {
                    let v = f64::from(input[self.index(outer, a, inner)]);
                    v * v
                }).sum::<f64>() / self.width as f64;
                let inverse = (mean_square + f64::from(epsilon)).sqrt().recip();
                for a in 0..self.width {
                    let i = self.index(outer, a, inner);
                    output[i] = (f64::from(input[i]) * inverse) as f32;
                }
            }
        }
        Ok(())
    }
    pub fn rms_vjp(self, input: &[f32], gradient: &[f32], epsilon: f32, result: &mut [f32]) -> Result<()> {
        ensure!(input.len() == self.len() && gradient.len() == self.len() && result.len() == self.len(), "rms gradient storage/shape mismatch");
        ensure!(epsilon.is_finite() && epsilon > 0., "normalization epsilon must be positive and finite");
        for outer in 0..self.outer {
            for inner in 0..self.inner {
                let mut square = 0.;
                let mut dot = 0.;
                for a in 0..self.width {
                    let i = self.index(outer, a, inner);
                    let x = f64::from(input[i]);
                    square += x * x;
                    dot += x * f64::from(gradient[i]);
                }
                let inverse = (square / self.width as f64 + f64::from(epsilon)).sqrt().recip();
                for a in 0..self.width {
                    let i = self.index(outer, a, inner);
                    result[i] += (f64::from(gradient[i]) * inverse
                        - f64::from(input[i]) * dot * inverse.powi(3) / self.width as f64) as f32;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strided_axis_and_vjp() {
        let groups = AxisGroups::new(&[2, 3, 2], 1).unwrap();
        let x = [1., 2., 3., 5., -2., 7., 4., -3., 2., 6., 8., 1.];
        let g = [0.2, -0.3, 0.5, 0.8, -0.2, 0.1, 0.7, -0.4, 0.9, 0.3, 0.2, -0.8];
        let mut y = [0.; 12];
        groups.rms(&x, 0.01, &mut y).unwrap();
        let scale = ((1. + 9. + 4.) / 3. + 0.01f32).sqrt();
        assert!((y[0] - 1. / scale).abs() < 1e-6);
        assert!((y[4] + 2. / scale).abs() < 1e-6);
        let mut analytical = [0.; 12];
        groups.rms_vjp(&x, &g, 0.01, &mut analytical).unwrap();
        for i in 0..12 {
            let mut plus = x;
            let mut minus = x;
            plus[i] += 0.002;
            minus[i] -= 0.002;
            let mut yp = [0.; 12];
            let mut ym = [0.; 12];
            groups.rms(&plus, 0.01, &mut yp).unwrap();
            groups.rms(&minus, 0.01, &mut ym).unwrap();
            let numerical: f32 = yp.iter().zip(ym).zip(g).map(|((&p, m), g)| (p - m) * g / 0.004).sum();
            assert!((numerical - analytical[i]).abs() < 8e-5, "axis derivative {i}: {numerical} vs {}", analytical[i]);
        }
    }
    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(AxisGroups::new(&[2, 0], 1).is_err());
        assert!(AxisGroups::new(&[2, 3], 2).is_err());
        assert!(AxisGroups::new(&[], 0).is_err());
        let groups = AxisGroups::new(&[2, 3], 1).unwrap();
        assert!(groups.rms(&[1.; 6], 0., &mut [0.; 6]).is_err());
        assert!(groups.rms(&[1.; 3], 1e-6, &mut [0.; 3]).is_err());
    }
}
