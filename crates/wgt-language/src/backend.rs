//! Source-level backend boundaries, independent of execution and operator support.
//!
//! Passing this preflight does not validate a model or promise that its operations
//! can run on a backend. Callers must still use that backend's frontend and checks.
use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Rust,
    Torch,
}

/// Check explicit source target compatibility before backend-specific parsing.
/// Plain WGT may be checked by either backend. The `target torch` compatibility
/// dialect requires the Python/PyTorch host and cannot be exported for Rust.
pub fn preflight_source(source: &str, backend: Backend) -> Result<()> {
    if backend == Backend::Rust {
        // Match the compatibility frontend's standalone target declaration.
        // Comments and strings mentioning the declaration are not declarations.
        if let Some(line) = torch_target_line(source) {
            bail!(
                "line {}: target torch requires Python/PyTorch and cannot be used with the Rust backend; migrate the model to shared WGT model/tensor syntax for Rust execution",
                line
            );
        }
    }
    Ok(())
}

pub(crate) fn torch_target_line(source: &str) -> Option<usize> {
    let mut quote: Option<(u8, bool)> = None;
    for (line_number, line) in source.lines().enumerate() {
        if quote.is_none() && line.trim() == "target torch" {
            return Some(line_number + 1);
        }
        let bytes = line.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            if let Some((delimiter, triple)) = quote {
                if byte == b'\\' {
                    index += 2;
                    continue;
                }
                if byte == delimiter {
                    if !triple {
                        quote = None;
                    } else if bytes.get(index..index + 3) == Some(&[delimiter; 3]) {
                        quote = None;
                        index += 2;
                    }
                }
            } else if byte == b'#' {
                break;
            } else if matches!(byte, b'\'' | b'"') {
                let triple = bytes.get(index..index + 3) == Some(&[byte; 3]);
                quote = Some((byte, triple));
                if triple {
                    index += 2;
                }
            }
            index += 1;
        }
    }
    None
}
