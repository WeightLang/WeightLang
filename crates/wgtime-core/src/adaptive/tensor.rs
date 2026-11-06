//! Row-major tensor geometry shared by native tensor instructions.
//! Storage is flat; geometry is never inferred from equal element counts.
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy)]
pub(super) struct TensorView {
    pub base: usize,
    offset: usize,
    shape: [usize; 8],
    strides: [usize; 8],
    rank: usize,
}

impl TensorView {
    pub fn contiguous(base: usize, dimensions: &[usize]) -> Self {
        let mut shape = [1; 8];
        shape[..dimensions.len()].copy_from_slice(dimensions);
        Self {
            base,
            offset: 0,
            shape,
            strides: Self::contiguous_strides(dimensions),
            rank: dimensions.len(),
        }
    }

    fn contiguous_strides(shape: &[usize]) -> [usize; 8] {
        let mut strides = [1; 8];
        let mut stride = 1;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride *= shape[axis];
        }
        strides
    }

    pub fn is_contiguous(self) -> bool {
        wgt_language::tensor::is_contiguous_layout(&self.shape[..self.rank], &self.strides[..self.rank])
    }

    pub fn reshaped(self, shape: &[usize]) -> Self {
        Self { offset: self.offset, ..Self::contiguous(self.base, shape) }
    }

    pub fn permuted(mut self, axes: &[usize]) -> Self {
        let previous = self;
        for (axis, &source) in axes.iter().enumerate() {
            self.shape[axis] = previous.shape[source];
            self.strides[axis] = previous.strides[source];
        }
        self
    }

    pub fn validate(self, storage_elements: usize) -> Result<Self> {
        ensure!((1..=wgt_language::tensor::MAX_RANK).contains(&self.rank), "invalid tensor view rank");
        wgt_language::tensor::validate_layout(&self.shape[..self.rank], &self.strides[..self.rank],
            self.offset, storage_elements, wgt_language::tensor::ShapeLimits::REGISTER)?;
        Ok(self)
    }

    pub fn physical_index(self, mut logical_index: usize) -> usize {
        let mut offset = self.offset;
        for axis in (0..self.rank).rev() {
            offset += (logical_index % self.shape[axis]) * self.strides[axis];
            logical_index /= self.shape[axis];
        }
        offset
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AxisGroups {
    pub outer: usize,
    pub width: usize,
    pub inner: usize,
}
impl AxisGroups {
    pub fn new(shape: &[usize], axis: usize) -> Result<Self> {
        wgt_language::tensor::validate_shape(shape, wgt_language::tensor::ShapeLimits::REGISTER)
            .context("normalization tensor shape")?;
        ensure!(axis < shape.len(), "tensor axis out of bounds");
        // All extents are positive and the full product is checked above.
        let outer = shape[..axis].iter().product();
        let width = shape[axis];
        let inner = shape[axis + 1..].iter().product();
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

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_geometry_is_rejected() {
        assert!(AxisGroups::new(&[2, 0], 1).is_err());
        assert!(AxisGroups::new(&[2, 3], 2).is_err());
        assert!(AxisGroups::new(&[], 0).is_err());
        let groups = AxisGroups::new(&[2, 3], 1).unwrap();
        assert!(groups.layer_norm(&[1.; 6], &[1.; 3], &[0.; 3], 0., &mut [0.; 6]).is_err());
        assert!(groups.layer_norm(&[1.; 3], &[1.; 3], &[0.; 3], 1e-6, &mut [0.; 3]).is_err());
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
