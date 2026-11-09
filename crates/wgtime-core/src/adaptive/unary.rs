use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub(super) enum Unary {
    Tanh,
    Sigmoid,
    Relu,
    Softmax,
    Neg,
    Detach,
    Exp,
    Log,
    Sqrt,
    LogSoftmax,
    ValidateProbabilities,
}

impl Unary {
    pub fn forward(self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if matches!(self, Self::ValidateProbabilities) {
            let values = input.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
            crate::categorical::validate_probabilities(&values)?;
        }
        let normalizes = matches!(self, Self::Softmax | Self::LogSoftmax);
        let max = if normalizes {
            input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        } else {
            0.
        };
        let total = if normalizes {
            input.iter().map(|value| (f64::from(*value) - f64::from(max)).exp()).sum::<f64>()
        } else {
            1.
        };
        for (&value, output) in input.iter().zip(output) {
            *output = match self {
                Self::Tanh => value.tanh(),
                Self::Sigmoid => 1. / (1. + (-value).exp()),
                Self::Relu => value.max(0.),
                Self::Softmax => ((f64::from(value) - f64::from(max)).exp() / total) as f32,
                Self::Neg => -value,
                Self::Detach | Self::ValidateProbabilities => value,
                Self::Exp => value.exp(),
                Self::Log => value.ln(),
                Self::Sqrt => value.sqrt(),
                Self::LogSoftmax => (f64::from(value) - f64::from(max) - total.ln()) as f32,
            };
        }
        Ok(())
    }

    pub fn backward(self, input: &[f32], output: &[f32], gradient: &[f32], result: &mut [f32]) {
        match self {
            Self::Detach => {}
            Self::LogSoftmax => {
                let sum: f32 = gradient.iter().sum();
                for ((result, &g), &y) in result.iter_mut().zip(gradient).zip(output) {
                    *result += g - y.exp() * sum;
                }
            }
            Self::Softmax => {
                let dot: f32 = gradient.iter().zip(output).map(|(g, y)| g * y).sum();
                for ((result, &g), &y) in result.iter_mut().zip(gradient).zip(output) {
                    *result += y * (g - dot);
                }
            }
            _ => {
                for (index, &g) in gradient.iter().enumerate() {
                    if g == 0. {
                        continue;
                    }
                    let y = output[index];
                    let derivative = match self {
                        Self::Tanh => 1. - y * y,
                        Self::Sigmoid => y * (1. - y),
                        Self::Relu => if input[index] > 0. { 1. } else { 0. },
                        Self::Neg => -1.,
                        Self::ValidateProbabilities => 1.,
                        Self::Exp => y,
                        Self::Log => 1. / input[index],
                        Self::Sqrt => 0.5 / y,
                        Self::Detach | Self::Softmax | Self::LogSoftmax => unreachable!(),
                    };
                    result[index] += g * derivative;
                }
            }
        }
    }
}
