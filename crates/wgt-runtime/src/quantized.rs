//! Packed, immutable weight-only matrices. Decode in registers, never a dense arena.
use anyhow::{ensure, Context, Result};

#[derive(Debug)]
pub(crate) struct Packed {
    pub format: u32,
    pub group: usize,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
}

impl Packed {
    pub fn layout(
        storage: &str,
        group: usize,
        shape: &[usize],
    ) -> Result<(u32, usize, usize, usize)> {
        ensure!(
            shape.len() == 2 && group > 0 && group <= 4096,
            "packed weights require a matrix and group 1..4096"
        );
        let format = match storage {
            "int4" => 1,
            "int8" => 2,
            "f16" => 3,
            "bf16" => 4,
            _ => anyhow::bail!("unsupported packed format"),
        };
        let stride = if format <= 2 {
            shape[1]
                .div_ceil(group)
                .checked_mul(group)
                .context("packed stride overflow")?
        } else {
            shape[1]
        };
        let count = shape[0]
            .checked_mul(stride)
            .context("packed shape overflow")?;
        let bytes = match format {
            1 => count.div_ceil(2),
            2 => count,
            _ => count.checked_mul(2).context("packed size overflow")?,
        };
        let scales = if format <= 2 { count / group } else { 0 };
        Ok((format, stride, bytes, scales))
    }
    pub fn new(
        storage: &str,
        group: usize,
        shape: &[usize],
        data: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        let (format, stride, bytes, count) = Self::layout(storage, group, shape)?;
        ensure!(
            data.len() == bytes && scales.len() == count,
            "packed data/scale length mismatch"
        );
        ensure!(
            scales.iter().all(|s| s.is_finite() && *s > 0.),
            "packed scales must be finite and positive"
        );
        let result = Self {
            format,
            group,
            rows: shape[0],
            cols: shape[1],
            stride,
            data,
            scales,
        };
        // Validate encoded floating values, and scaled integer overflow, before any run.
        ensure!(
            (0..result.rows * result.cols).all(|i| result.value(i).is_finite()),
            "nonfinite decoded packed weight"
        );
        Ok(result)
    }
    pub fn bytes(&self) -> usize {
        self.data.len() + 4 * self.scales.len()
    }
    pub fn value(&self, index: usize) -> f32 {
        let index = index / self.cols * self.stride + index % self.cols;
        match self.format {
            1 => {
                (((self.data[index / 2] >> ((index % 2) * 4)) & 15) as f32 - 8.)
                    * self.scales[index / self.group]
            }
            2 => (self.data[index] as f32 - 128.) * self.scales[index / self.group],
            3 => half(u16::from_le_bytes([
                self.data[index * 2],
                self.data[index * 2 + 1],
            ])),
            _ => f32::from_bits(
                (u16::from_le_bytes([self.data[index * 2], self.data[index * 2 + 1]]) as u32) << 16,
            ),
        }
    }
    pub fn dot(&self, row: usize, x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.format <= 2
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            // SAFETY: ISA checked; the kernel bounds every load by group/input lengths.
            return unsafe {
                if self.format == 1 && self.group == 32 {
                    simd::int4_group32(self, row, x)
                } else {
                    simd::packed_dot(self, row, x)
                }
            };
        }
        #[cfg(target_arch = "x86_64")]
        if self.format >= 3
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            // SAFETY: feature detection and checked matrix/input bounds.
            return unsafe { simd::half_dot(self, row, x) };
        }
        let mut sum = 0.;
        for (col, value) in x.iter().enumerate() {
            sum += value * self.value(row * self.cols + col);
        }
        sum
    }
}

fn half(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exp = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    if exp == 0 {
        let value = mantissa as f32 * (1. / 16777216.);
        if sign != 0 {
            -value
        } else {
            value
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f800000 | ((mantissa as u32) << 13))
    } else {
        f32::from_bits(sign | (((exp as u32) + 112) << 23) | ((mantissa as u32) << 13))
    }
}

/// Same SIMD baseline for ordinary resident F32 weights; no artificial copying penalty.
pub(crate) fn dot(weight: &[f32], input: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: feature detection and equal, checked slice lengths at each call site.
        return unsafe { simd::float_dot(weight, input) };
    }
    weight.iter().zip(input).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::Packed;
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sum(v: __m256) -> f32 {
        let v = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let v = _mm_hadd_ps(v, v);
        _mm_cvtss_f32(_mm_hadd_ps(v, v))
    }
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn float_dot(w: &[f32], x: &[f32]) -> f32 {
        let mut a = _mm256_setzero_ps();
        let mut b = a;
        let mut i = 0;
        while i + 16 <= x.len() {
            a = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(i)),
                _mm256_loadu_ps(x.as_ptr().add(i)),
                a,
            );
            b = _mm256_fmadd_ps(
                _mm256_loadu_ps(w.as_ptr().add(i + 8)),
                _mm256_loadu_ps(x.as_ptr().add(i + 8)),
                b,
            );
            i += 16;
        }
        let mut value = sum(_mm256_add_ps(a, b));
        while i < x.len() {
            value += w[i] * x[i];
            i += 1;
        }
        value
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn half_dot(p: &Packed, row: usize, x: &[f32]) -> f32 {
        let mut a = _mm256_setzero_ps();
        let mut b = a;
        let mut i = 0;
        while i + 16 <= x.len() {
            let w0 = _mm_loadu_si128(p.data.as_ptr().add((row * p.cols + i) * 2).cast());
            let w1 = _mm_loadu_si128(p.data.as_ptr().add((row * p.cols + i + 8) * 2).cast());
            let f0 = if p.format == 3 {
                _mm256_cvtph_ps(w0)
            } else {
                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w0), 16))
            };
            let f1 = if p.format == 3 {
                _mm256_cvtph_ps(w1)
            } else {
                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w1), 16))
            };
            a = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x.as_ptr().add(i)), a);
            b = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x.as_ptr().add(i + 8)), b);
            i += 16;
        }
        let mut result = sum(_mm256_add_ps(a, b));
        while i < x.len() {
            result += x[i] * p.value(row * p.cols + i);
            i += 1;
        }
        result
    }
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn int4_group32(p: &Packed, row: usize, x: &[f32]) -> f32 {
        let mut total0 = _mm256_setzero_ps();
        let mut total1 = total0;
        let mask = _mm_set1_epi8(15);
        let offset = _mm256_set1_epi32(8);
        let full = x.len() / 32;
        let data = p.data.as_ptr().add(row * p.stride / 2);
        let scales = p.scales.as_ptr().add(row * (p.stride / 32));
        for g in 0..full {
            let packed = _mm_loadu_si128(data.add(g * 16).cast());
            let lo = _mm_and_si128(packed, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let low = _mm_unpacklo_epi8(lo, hi);
            let high = _mm_unpackhi_epi8(lo, hi);
            let q0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(low), offset));
            let q1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(low, 8)),
                offset,
            ));
            let q2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(high), offset));
            let q3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128(high, 8)),
                offset,
            ));
            let xx = x.as_ptr().add(g * 32);
            let a = _mm256_fmadd_ps(
                q2,
                _mm256_loadu_ps(xx.add(16)),
                _mm256_mul_ps(q0, _mm256_loadu_ps(xx)),
            );
            let b = _mm256_fmadd_ps(
                q3,
                _mm256_loadu_ps(xx.add(24)),
                _mm256_mul_ps(q1, _mm256_loadu_ps(xx.add(8))),
            );
            let scale = _mm256_set1_ps(*scales.add(g));
            total0 = _mm256_fmadd_ps(a, scale, total0);
            total1 = _mm256_fmadd_ps(b, scale, total1);
        }
        let mut value = sum(_mm256_add_ps(total0, total1));
        for (col, &input) in x.iter().enumerate().skip(full * 32) {
            value += input * p.value(row * p.cols + col);
        }
        value
    }
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn packed_dot(p: &Packed, row: usize, x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        let mut tail = 0.;
        let mask = _mm_set1_epi8(15);
        let offset = _mm256_set1_epi32(if p.format == 1 { 8 } else { 128 });
        for start in (0..x.len()).step_by(p.group) {
            let end = (start + p.group).min(x.len());
            let base = row * p.stride + start;
            let scale = p.scales[base / p.group];
            let mut a = _mm256_setzero_ps();
            let mut b = a;
            let mut col = start;
            if p.format == 1 && base.is_multiple_of(2) {
                while col + 16 <= end {
                    let packed =
                        _mm_loadl_epi64(p.data.as_ptr().add((row * p.stride + col) / 2).cast());
                    let lo = _mm_and_si128(packed, mask);
                    let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
                    let codes = _mm_unpacklo_epi8(lo, hi);
                    let q0 =
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(codes), offset));
                    let q1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                        _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8)),
                        offset,
                    ));
                    a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(x.as_ptr().add(col)), a);
                    b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(x.as_ptr().add(col + 8)), b);
                    col += 16;
                }
            } else if p.format == 2 {
                while col + 16 <= end {
                    let codes = _mm_loadu_si128(p.data.as_ptr().add(row * p.stride + col).cast());
                    let q0 =
                        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(codes), offset));
                    let q1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                        _mm256_cvtepu8_epi32(_mm_srli_si128(codes, 8)),
                        offset,
                    ));
                    a = _mm256_fmadd_ps(q0, _mm256_loadu_ps(x.as_ptr().add(col)), a);
                    b = _mm256_fmadd_ps(q1, _mm256_loadu_ps(x.as_ptr().add(col + 8)), b);
                    col += 16;
                }
            }
            total = _mm256_fmadd_ps(_mm256_add_ps(a, b), _mm256_set1_ps(scale), total);
            while col < end {
                tail += x[col] * p.value(row * p.cols + col);
                col += 1;
            }
        }
        sum(total) + tail
    }
}
