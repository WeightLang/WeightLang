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
    pub fn softmax(self, input: &[f32], output: &mut [f32]) -> Result<()> {
        ensure!(input.len() == self.len() && output.len() == self.len(), "softmax storage/shape mismatch");
        for o in 0..self.outer { for i in 0..self.inner {
            let max = (0..self.width).map(|a| input[self.index(o,a,i)]).fold(f32::NEG_INFINITY,f32::max);
            let sum: f64 = (0..self.width).map(|a| (f64::from(input[self.index(o,a,i)])-f64::from(max)).exp()).sum();
            for a in 0..self.width {
                let j = self.index(o,a,i);
                output[j] = ((f64::from(input[j])-f64::from(max)).exp()/sum) as f32;
            }
        } }
        Ok(())
    }
    pub fn softmax_vjp(self, value: &[f32], gradient: &[f32], result: &mut [f32]) {
        for o in 0..self.outer { for i in 0..self.inner {
            let dot: f64 = (0..self.width).map(|a| { let j=self.index(o,a,i); f64::from(value[j])*f64::from(gradient[j]) }).sum();
            for a in 0..self.width {
                let j=self.index(o,a,i);
                result[j] += (f64::from(value[j])*(f64::from(gradient[j])-dot)) as f32;
            }
        } }
    }
    pub fn layer_norm(self, input: &[f32], gamma: &[f32], beta: &[f32], epsilon: f32, output: &mut [f32]) -> Result<()> {
        ensure!(input.len() == self.len() && output.len() == self.len(), "layer_norm tensor storage/shape mismatch");
        ensure!(gamma.len() == self.width && beta.len() == self.width, "layer_norm scale/bias width mismatch");
        ensure!(epsilon.is_finite() && epsilon > 0., "normalization epsilon must be positive and finite");
        for outer in 0..self.outer {
            for inner in 0..self.inner {
                let mean = (0..self.width).map(|axis| f64::from(input[self.index(outer, axis, inner)])).sum::<f64>() / self.width as f64;
                let variance = (0..self.width).map(|axis| {
                    let delta = f64::from(input[self.index(outer, axis, inner)]) - mean;
                    delta * delta
                }).sum::<f64>() / self.width as f64;
                let inverse = (variance + f64::from(epsilon)).sqrt().recip();
                for axis in 0..self.width {
                    let index = self.index(outer, axis, inner);
                    let normalized = (f64::from(input[index]) - mean) * inverse;
                    output[index] = (normalized * f64::from(gamma[axis]) + f64::from(beta[axis])) as f32;
                }
            }
        }
        Ok(())
    }
    pub fn layer_norm_vjp(self, input: &[f32], gamma: &[f32], gradient: &[f32], epsilon: f32,
        mut accumulate: impl FnMut(usize, usize, f32, f32, f32)) -> Result<()> {
        ensure!(input.len() == self.len() && gradient.len() == self.len(),
            "layer_norm gradient tensor storage/shape mismatch");
        ensure!(gamma.len() == self.width,
            "layer_norm gradient scale/bias width mismatch");
        ensure!(epsilon.is_finite() && epsilon > 0., "normalization epsilon must be positive and finite");
        for outer in 0..self.outer {
            for inner in 0..self.inner {
                let mean = (0..self.width).map(|axis| f64::from(input[self.index(outer, axis, inner)])).sum::<f64>() / self.width as f64;
                let variance = (0..self.width).map(|axis| {
                    let delta = f64::from(input[self.index(outer, axis, inner)]) - mean;
                    delta * delta
                }).sum::<f64>() / self.width as f64;
                let inverse = (variance + f64::from(epsilon)).sqrt().recip();
                let mut sum_gradient = 0f64;
                let mut sum_gradient_normalized = 0f64;
                for axis in 0..self.width {
                    let index = self.index(outer, axis, inner);
                    let normalized = (f64::from(input[index]) - mean) * inverse;
                    let scaled_gradient = f64::from(gradient[index]) * f64::from(gamma[axis]);
                    sum_gradient += scaled_gradient;
                    sum_gradient_normalized += scaled_gradient * normalized;
                }
                for axis in 0..self.width {
                    let index = self.index(outer, axis, inner);
                    let normalized = (f64::from(input[index]) - mean) * inverse;
                    let scaled_gradient = f64::from(gradient[index]) * f64::from(gamma[axis]);
                    let dx = inverse * (scaled_gradient
                        - sum_gradient / self.width as f64
                        - normalized * sum_gradient_normalized / self.width as f64);
                    accumulate(index, axis, dx as f32,
                        (f64::from(gradient[index]) * normalized) as f32, gradient[index]);
                }
            }
        }
        Ok(())
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
    #[test]
    fn layer_norm_vjp_matches_finite_differences() {
        let groups = AxisGroups::new(&[2, 3, 2], 1).unwrap();
        let input = [0.2, -0.1, 0.7, 0.4, -0.3, 0.9, 0.1, 0.5, -0.8, 0.6, 0.3, -0.2];
        let gamma = [0.8, 1.1, -0.4];
        let beta = [0.1, -0.2, 0.3];
        let gradient = [0.2, -0.3, 0.5, 0.8, -0.2, 0.1, 0.7, -0.4, 0.9, 0.3, 0.2, -0.8];
        let epsilon = 1e-4;
        let mut output = [0.; 12];
        groups.layer_norm(&input, &gamma, &beta, epsilon, &mut output).unwrap();
        let mut dx = [0.; 12];
        let mut dgamma = [0.; 3];
        let mut dbeta = [0.; 3];
        groups.layer_norm_vjp(&input, &gamma, &gradient, epsilon, |index, axis, x, g, b| {
            dx[index] += x;
            dgamma[axis] += g;
            dbeta[axis] += b;
        }).unwrap();
        let loss = |x: &[f32; 12], g: &[f32; 3], b: &[f32; 3]| {
            let mut y = [0.; 12];
            groups.layer_norm(x, g, b, epsilon, &mut y).unwrap();
            y.iter().zip(gradient).map(|(y, dy)| y * dy).sum::<f32>()
        };
        let step = 0.001;
        for i in 0..12 {
            let mut plus = input;
            let mut minus = input;
            plus[i] += step;
            minus[i] -= step;
            let numerical = (loss(&plus, &gamma, &beta) - loss(&minus, &gamma, &beta)) / (2. * step);
            assert!((numerical - dx[i]).abs() < 2e-3, "input {i}: {numerical} vs {}", dx[i]);
        }
        for i in 0..3 {
            let mut plus = gamma;
            let mut minus = gamma;
            plus[i] += step;
            minus[i] -= step;
            let numerical = (loss(&input, &plus, &beta) - loss(&input, &minus, &beta)) / (2. * step);
            assert!((numerical - dgamma[i]).abs() < 2e-3, "gamma {i}: {numerical} vs {}", dgamma[i]);
            let mut plus = beta;
            let mut minus = beta;
            plus[i] += step;
            minus[i] -= step;
            let numerical = (loss(&input, &gamma, &plus) - loss(&input, &gamma, &minus)) / (2. * step);
            assert!((numerical - dbeta[i]).abs() < 2e-3, "beta {i}: {numerical} vs {}", dbeta[i]);
        }
    }

}
