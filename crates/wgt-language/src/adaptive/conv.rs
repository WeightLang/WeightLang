//! Checked channel-first, unbatched spatial cross-correlation geometry.
use anyhow::{Context, Result, ensure};
#[derive(Clone, Copy, Debug)]
pub struct ConvGeometry {
    pub rank: usize,
    pub input: [usize; 3],
    pub kernel: [usize; 3],
    pub output: [usize; 3],
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub channels: usize,
    pub outputs: usize,
    pub groups: usize,
}
impl ConvGeometry {
    pub fn new(
        x: &[usize],
        k: &[usize],
        bias: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Result<Self> {
        ensure!(
            (2..=4).contains(&x.len()) && k.len() == x.len() + 1,
            "conv requires 1..3 spatial axes and channel-first kernel"
        );
        crate::model::validate_shape(x)?;
        crate::model::validate_shape(k)?;
        let rank = x.len() - 1;
        ensure!(
            stride.len() == rank && padding.len() == rank && dilation.len() == rank,
            "conv options must match spatial rank"
        );
        ensure!(
            groups > 0 && x[0] % groups == 0 && k[0] % groups == 0 && k[1] == x[0] / groups,
            "conv groups/channel mismatch"
        );
        ensure!(bias == [k[0]], "conv bias must have shape [Cout]");
        let mut g = Self {
            rank,
            input: [1; 3],
            kernel: [1; 3],
            output: [1; 3],
            stride: [1; 3],
            padding: [0; 3],
            dilation: [1; 3],
            channels: x[0],
            outputs: k[0],
            groups,
        };
        for a in 0..rank {
            ensure!(
                (1..=16384).contains(&stride[a])
                    && (1..=16384).contains(&dilation[a])
                    && padding[a] <= 16384,
                "conv stride/dilation must be 1..16384 and padding 0..16384"
            );
            let effective = dilation[a]
                .checked_mul(k[a + 2] - 1)
                .and_then(|n| n.checked_add(1))
                .context("conv extent overflow")?;
            let padded = x[a + 1]
                .checked_add(padding[a] * 2)
                .context("conv padding overflow")?;
            ensure!(padded >= effective, "conv kernel exceeds padded input");
            g.input[a] = x[a + 1];
            g.kernel[a] = k[a + 2];
            g.output[a] = (padded - effective) / stride[a] + 1;
            g.stride[a] = stride[a];
            g.padding[a] = padding[a];
            g.dilation[a] = dilation[a];
        }
        crate::model::validate_shape(&g.shape())?;
        g.work()?;
        Ok(g)
    }
    pub fn shape(self) -> Vec<usize> {
        let mut s = vec![self.outputs];
        s.extend_from_slice(&self.output[..self.rank]);
        s
    }
    pub fn input_volume(self) -> usize {
        self.input.iter().product()
    }
    pub fn kernel_volume(self) -> usize {
        self.kernel.iter().product()
    }
    pub fn output_volume(self) -> usize {
        self.output.iter().product()
    }
    pub fn work(self) -> Result<usize> {
        self.outputs
            .checked_mul(self.output_volume())
            .and_then(|n| n.checked_mul(self.channels / self.groups))
            .and_then(|n| n.checked_mul(self.kernel_volume()))
            .and_then(|n| n.checked_add(1))
            .context("conv work overflow")
    }
    /// Visit valid input/kernel pairs for one flattened output. No im2col buffer.
    pub fn pairs(self, out: usize, mut visit: impl FnMut(usize, usize)) {
        let oc = out / self.output_volume();
        let mut pos = out % self.output_volume();
        let mut coords = [0; 3];
        for a in (0..self.rank).rev() {
            coords[a] = pos % self.output[a];
            pos /= self.output[a];
        }
        for c in 0..self.channels / self.groups {
            for q in 0..self.kernel_volume() {
                let mut rem = q;
                let mut kc = [0; 3];
                for a in (0..self.rank).rev() {
                    kc[a] = rem % self.kernel[a];
                    rem /= self.kernel[a];
                }
                let mut index = 0;
                let mut valid = true;
                for a in 0..self.rank {
                    let p = (coords[a] * self.stride[a] + kc[a] * self.dilation[a]) as isize
                        - self.padding[a] as isize;
                    if p < 0 || p >= self.input[a] as isize {
                        valid = false;
                        break;
                    }
                    index = index * self.input[a] + p as usize;
                }
                if valid {
                    let ic = oc / (self.outputs / self.groups) * (self.channels / self.groups) + c;
                    visit(
                        ic * self.input_volume() + index,
                        (oc * (self.channels / self.groups) + c) * self.kernel_volume() + q,
                    );
                }
            }
        }
    }
}


/// Checked channel-first 2D pooling geometry. Pooling operates on [C,H,W].
#[derive(Clone, Copy, Debug)]
pub struct Pool2dGeometry {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub output_h: usize,
    pub output_w: usize,
}
impl Pool2dGeometry {
    pub fn new(input: &[usize], kernel: &[usize], stride: &[usize]) -> Result<Self> {
        ensure!(input.len() == 3, "pool2d input must have shape [C,H,W]");
        ensure!(kernel.len() == 2 && stride.len() == 2, "pool2d kernel and stride must have two dimensions");
        crate::model::validate_shape(input)?;
        ensure!(kernel.iter().all(|&n| n > 0 && n <= 16384)
            && stride.iter().all(|&n| n > 0 && n <= 16384),
            "pool2d kernel and stride must be in 1..=16384");
        ensure!(input[1] >= kernel[0] && input[2] >= kernel[1],
            "pool2d window exceeds input");
        let output_h = (input[1] - kernel[0]) / stride[0] + 1;
        let output_w = (input[2] - kernel[1]) / stride[1] + 1;
        let geometry = Self {
            channels: input[0], input_h: input[1], input_w: input[2],
            kernel_h: kernel[0], kernel_w: kernel[1],
            stride_h: stride[0], stride_w: stride[1], output_h, output_w,
        };
        crate::model::validate_shape(&geometry.shape())?;
        geometry.work()?;
        Ok(geometry)
    }
    pub fn shape(self) -> Vec<usize> { vec![self.channels, self.output_h, self.output_w] }
    pub fn input_len(self) -> usize { self.channels * self.input_h * self.input_w }
    pub fn output_len(self) -> usize { self.channels * self.output_h * self.output_w }
    pub fn window_len(self) -> usize { self.kernel_h * self.kernel_w }
    pub fn work(self) -> Result<usize> {
        self.output_len().checked_mul(self.window_len())
            .and_then(|n| n.checked_add(1)).context("pool2d work overflow")
    }
    pub fn visit(self, output: usize, mut visit: impl FnMut(usize)) {
        let spatial = self.output_h * self.output_w;
        let channel = output / spatial;
        let pos = output % spatial;
        let oy = pos / self.output_w;
        let ox = pos % self.output_w;
        for ky in 0..self.kernel_h {
            for kx in 0..self.kernel_w {
                let iy = oy * self.stride_h + ky;
                let ix = ox * self.stride_w + kx;
                visit((channel * self.input_h + iy) * self.input_w + ix);
            }
        }
    }
}
