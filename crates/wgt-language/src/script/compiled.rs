use anyhow::{ensure, Result};
/// Shared half-open bounds for ordinary lists and tensor lowering. Never clamp.
pub fn slice_bounds(len: usize, start: Option<f64>, end: Option<f64>) -> Result<(usize, usize)> {
    let endpoint = |n: f64| -> Result<usize> {
        ensure!(n.is_finite() && n >= 0. && n.fract() == 0., "slice endpoints must be nonnegative integers; negative indices are unsupported");
        ensure!(n <= len as f64, "slice endpoint out of bounds (length {len})");
        Ok(n as usize)
    };
    let start = start.map(endpoint).transpose()?.unwrap_or(0);
    let end = end.map(endpoint).transpose()?.unwrap_or(len);
    ensure!(start <= end, "slice start must not exceed end");
    Ok((start, end))
}
