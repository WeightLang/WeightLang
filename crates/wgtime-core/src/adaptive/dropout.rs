//! Stateless counter RNG: forward and VJP use the same checked immutable config.
use anyhow::{Result, ensure};
#[derive(Clone, Copy)]
pub(super) struct Config {
    seed: u32,
    stream: u32,
    probability: f32,
    training: bool,
}
impl Config {
    pub(super) fn new(rng: &[f32], probability: f32) -> Result<Self> {
        ensure!(
            rng.len() == 4
                && rng[..3]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0. && *v <= 65535. && v.fract() == 0.)
                && (rng[3] == 0. || rng[3] == 1.),
            "dropout rng requires three uint16 values and training 0/1"
        );
        ensure!(
            probability.is_finite() && (0.0..1.0).contains(&probability),
            "invalid dropout probability"
        );
        Ok(Self {
            seed: rng[0] as u32 | (rng[1] as u32) << 16,
            stream: rng[2] as u32,
            probability,
            training: rng[3] == 1.,
        })
    }
    pub(super) fn factor(self, index: usize) -> f32 {
        if !self.training || self.probability == 0. {
            return 1.;
        }
        let mut x = self.seed
            ^ self.stream.wrapping_mul(0x9e3779b9)
            ^ (index as u32).wrapping_mul(0x85ebca6b);
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846ca68b);
        x ^= x >> 16;
        if ((x >> 8) as f32) * (1. / 16777216.) >= self.probability {
            1. / (1. - self.probability)
        } else {
            0.
        }
    }
}
