//! Allocation-free storage accounting before model values are materialized.
use anyhow::{Context, Result};

pub(super) fn storage_bytes<'a>(
    shapes: impl IntoIterator<Item = &'a [usize]>,
    copies: usize,
) -> Result<usize> {
    let bytes = shapes.into_iter().try_fold(0usize, |total, shape| {
        let elements = shape.iter().try_fold(1usize, |n, axis| n.checked_mul(*axis))
            .context("storage shape overflow")?;
        let bytes = elements.checked_mul(std::mem::size_of::<f32>())
            .context("storage byte overflow")?;
        total.checked_add(bytes).context("storage byte overflow")
    })?;
    bytes.checked_mul(copies).context("storage byte overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_parameters_and_states_before_transactional_copies() {
        let shapes: &[&[usize]] = &[&[2, 3], &[4]];
        assert_eq!(storage_bytes(shapes.iter().copied(), 1).unwrap(), 40);
        assert_eq!(storage_bytes(shapes.iter().copied(), 4).unwrap(), 160);
        assert_eq!(storage_bytes(std::iter::empty(), 4).unwrap(), 0);
        // Shape validity is checked by the language; arithmetic preserves scalar semantics.
        assert_eq!(storage_bytes([&[][..]], 1).unwrap(), 4);
    }

    #[test]
    fn rejects_element_byte_sum_and_copy_overflow() {
        let cases: &[(&[&[usize]], usize)] = &[
            (&[&[usize::MAX, 2]], 1),
            (&[&[usize::MAX / 4 + 1]], 1),
            (&[&[usize::MAX / 4], &[1]], 1),
            (&[&[usize::MAX / 8]], 4),
        ];
        for &(shapes, copies) in cases {
            assert!(storage_bytes(shapes.iter().copied(), copies).is_err());
        }
    }
}
