//! Allocation-free matrix products over already validated rank-2/rank-3 views.
//! Resolve tensor geometry once per operation, rather than dividing logical
//! indices into coordinates for every multiply. Four independent accumulators
//! reuse each left value without changing the q-order of any dot product.

#[derive(Clone, Copy, Debug)]
pub(super) struct Layout {
    pub offset: usize,
    pub batch_stride: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}

/// Multiply validated [batch, rows, inner] and [batch, inner, columns] views.
/// Rank-2 callers use one batch and a zero batch stride. The separate output
/// buffer preserves operands even when a destination aliases an input register.
pub(super) fn multiply(
    left: &[f32],
    left_layout: Layout,
    right: &[f32],
    right_layout: Layout,
    [batches, rows, inner, columns]: [usize; 4],
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), batches * rows * columns);
    for batch in 0..batches {
        let left_batch = left_layout.offset + batch * left_layout.batch_stride;
        let right_batch = right_layout.offset + batch * right_layout.batch_stride;
        for row in 0..rows {
            let left_row = left_batch + row * left_layout.row_stride;
            let output_row = (batch * rows + row) * columns;
            let mut column = 0;
            while columns - column >= 4 {
                let mut a_index = left_row;
                let mut b_index = right_batch + column * right_layout.column_stride;
                let step = right_layout.column_stride;
                let mut sums = [0.0f64; 4];
                for _ in 0..inner {
                    let a = f64::from(left[a_index]);
                    sums[0] += a * f64::from(right[b_index]);
                    sums[1] += a * f64::from(right[b_index + step]);
                    sums[2] += a * f64::from(right[b_index + 2 * step]);
                    sums[3] += a * f64::from(right[b_index + 3 * step]);
                    a_index += left_layout.column_stride;
                    b_index += right_layout.row_stride;
                }
                for lane in 0..4 {
                    output[output_row + column + lane] = sums[lane] as f32;
                }
                column += 4;
            }
            for column in column..columns {
                let mut a_index = left_row;
                let mut b_index = right_batch + column * right_layout.column_stride;
                let mut sum = 0.0f64;
                for _ in 0..inner {
                    sum += f64::from(left[a_index]) * f64::from(right[b_index]);
                    a_index += left_layout.column_stride;
                    b_index += right_layout.row_stride;
                }
                output[output_row + column] = sum as f32;
            }
        }
    }
}
