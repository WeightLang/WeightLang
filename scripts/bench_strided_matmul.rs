//! Standalone benchmark of the exact production matrix kernel.
//!
//! Run from the repository root (no Cargo build or third-party dependency):
//!   rustc --edition=2024 -O scripts/bench_strided_matmul.rs -o /tmp/weightlang/bench-strided-matmul
//!   /tmp/weightlang/bench-strided-matmul
//! The reference retains the old per-element logical-index decoding. Results
//! must match bit for bit before either implementation is timed. Paired sample
//! order alternates to reduce warmup and frequency bias; report median latency.

#[path = "../crates/wgtime-core/src/adaptive/matmul.rs"]
mod matmul;

use matmul::Layout;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
struct View {
    offset: usize,
    shape: [usize; 3],
    strides: [usize; 3],
    rank: usize,
}

struct Operand {
    data: Vec<f32>,
    view: Option<View>,
    layout: Layout,
}

impl Operand {
    fn new(shape: &[usize], axes: &[usize], viewed: bool, offset: usize) -> Self {
        assert_eq!(shape.len(), axes.len());
        let rank = shape.len();
        let mut raw_shape = [1; 3];
        for (index, &axis) in axes.iter().enumerate() {
            raw_shape[axis] = shape[index];
        }
        let mut raw_strides = [1; 3];
        let mut count = 1;
        for axis in (0..rank).rev() {
            raw_strides[axis] = count;
            count *= raw_shape[axis];
        }
        let mut logical_shape = [1; 3];
        logical_shape[..rank].copy_from_slice(shape);
        let mut strides = [0; 3];
        for (index, &axis) in axes.iter().enumerate() {
            strides[index] = raw_strides[axis];
        }
        let view = View {
            offset,
            shape: logical_shape,
            strides,
            rank,
        };
        assert!(viewed || (offset == 0 && axes.iter().enumerate().all(|(i, &a)| i == a)));
        Self {
            data: (0..offset + count)
                .map(|i| (i % 97) as f32 * 0.0078125 - 0.375)
                .collect(),
            view: viewed.then_some(view),
            layout: Layout {
                offset,
                batch_stride: if rank == 3 { strides[0] } else { 0 },
                row_stride: strides[rank - 2],
                column_stride: strides[rank - 1],
            },
        }
    }

    fn read_value(&self, mut logical_index: usize) -> f32 {
        if let Some(view) = self.view {
            let mut physical = view.offset;
            for axis in (0..view.rank).rev() {
                physical += (logical_index % view.shape[axis]) * view.strides[axis];
                logical_index /= view.shape[axis];
            }
            self.data[physical]
        } else {
            self.data[logical_index]
        }
    }
}

struct Case {
    name: &'static str,
    dimensions: [usize; 4],
    left: Operand,
    right: Operand,
}

#[inline(never)]
fn legacy(case: &Case, output: &mut [f32]) {
    let [batches, rows, inner, columns] = case.dimensions;
    for batch in 0..batches {
        let a_base = batch * rows * inner;
        let b_base = batch * inner * columns;
        let out_base = batch * rows * columns;
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0f64;
                for q in 0..inner {
                    sum += f64::from(case.left.read_value(a_base + row * inner + q))
                        * f64::from(case.right.read_value(b_base + q * columns + column));
                }
                output[out_base + row * columns + column] = sum as f32;
            }
        }
    }
}

#[inline(never)]
fn optimized(case: &Case, output: &mut [f32]) {
    matmul::multiply(
        &case.left.data,
        case.left.layout,
        &case.right.data,
        case.right.layout,
        case.dimensions,
        output,
    );
}

fn measure(kernel: fn(&Case, &mut [f32]), case: &Case, output: &mut [f32], iterations: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        kernel(black_box(case), black_box(output));
        black_box(&*output);
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(iterations)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let iterations: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("positive iteration count"))
        .unwrap_or(8);
    assert!(iterations > 0);
    let cases = [
        Case {
            name: "rank2-owned",
            dimensions: [1, 64, 96, 48],
            left: Operand::new(&[64, 96], &[0, 1], false, 0),
            right: Operand::new(&[96, 48], &[0, 1], false, 0),
        },
        Case {
            name: "rank2-reshape-view",
            dimensions: [1, 64, 96, 48],
            left: Operand::new(&[64, 96], &[0, 1], true, 0),
            right: Operand::new(&[96, 48], &[0, 1], true, 0),
        },
        Case {
            name: "rank2-transposed",
            dimensions: [1, 64, 96, 48],
            left: Operand::new(&[64, 96], &[1, 0], true, 0),
            right: Operand::new(&[96, 48], &[1, 0], true, 0),
        },
        Case {
            name: "rank2-offset-tail",
            dimensions: [1, 31, 65, 33],
            left: Operand::new(&[31, 65], &[0, 1], true, 13),
            right: Operand::new(&[65, 33], &[1, 0], true, 7),
        },
        Case {
            name: "rank3-permuted-batch",
            dimensions: [4, 32, 64, 40],
            left: Operand::new(&[4, 32, 64], &[2, 0, 1], true, 0),
            right: Operand::new(&[4, 64, 40], &[1, 2, 0], true, 0),
        },
    ];
    println!(
        "case,batches,rows,inner,columns,iterations,legacy_median_us,kernel_median_us,speedup"
    );
    for case in cases {
        let [batches, rows, inner, columns] = case.dimensions;
        let mut reference = vec![0.0; batches * rows * columns];
        let mut output = vec![0.0; reference.len()];
        legacy(&case, &mut reference);
        optimized(&case, &mut output);
        for (index, (&a, &b)) in reference.iter().zip(&output).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{} output {index}: {a} vs {b}",
                case.name
            );
        }
        // Allocation and validation are outside the timing window.
        let mut old_times = [0.0; 9];
        let mut new_times = [0.0; 9];
        for sample in 0..9 {
            if sample % 2 == 0 {
                old_times[sample] = measure(legacy, &case, &mut reference, iterations);
                new_times[sample] = measure(optimized, &case, &mut output, iterations);
            } else {
                new_times[sample] = measure(optimized, &case, &mut output, iterations);
                old_times[sample] = measure(legacy, &case, &mut reference, iterations);
            }
        }
        let old = median(&mut old_times);
        let new = median(&mut new_times);
        println!(
            "{},{batches},{rows},{inner},{columns},{iterations},{old:.3},{new:.3},{:.3}",
            case.name,
            old / new
        );
    }
}
