//! Bounded diagnostics shared by JSON and binary replies.
use anyhow::Result;
use std::fmt::{self, Write as _};
use std::io::Write;
use std::os::unix::net::UnixStream;

/// Kept well below MAX_FRAME even after JSON's worst-case escaping.
pub const MAX_ERROR_BYTES: usize = 8 * 1024;
const TRUNCATED: &str = "... [truncated]";

struct Diagnostic {
    text: String,
    truncated: bool,
}
impl fmt::Write for Diagnostic {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let remaining = MAX_ERROR_BYTES - self.text.len();
        if text.len() <= remaining {
            self.text.push_str(text);
            return Ok(());
        }
        let mut end = remaining;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.text.push_str(&text[..end]);
        self.truncated = true;
        // Stop formatting the rest of the chain, rather than allocating it and
        // truncating afterwards. This bounds the diagnostic buffer itself.
        Err(fmt::Error)
    }
}

pub(crate) fn bounded_error(error: &anyhow::Error) -> String {
    let mut diagnostic = Diagnostic {
        text: String::with_capacity(MAX_ERROR_BYTES),
        truncated: false,
    };
    let _ = write!(&mut diagnostic, "{error:#}");
    if diagnostic.truncated {
        let mut end = diagnostic.text.len().min(MAX_ERROR_BYTES - TRUNCATED.len());
        while !diagnostic.text.is_char_boundary(end) {
            end -= 1;
        }
        diagnostic.text.truncate(end);
        diagnostic.text.push_str(TRUNCATED);
    }
    diagnostic.text
}

/// Send one complete bounded error frame. A transport error requires closing
/// the connection; callers must not attempt to append a second response.
pub fn send_binary_error(stream: &mut UnixStream, error: &anyhow::Error) -> Result<usize> {
    let message = bounded_error(error);
    let size = u32::try_from(message.len())?;
    stream.write_all(&1u32.to_le_bytes())?;
    stream.write_all(&size.to_le_bytes())?;
    stream.write_all(message.as_bytes())?;
    Ok(message.len() + 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn diagnostics_are_utf8_bounded_and_preserve_short_errors() {
        for text in ["x".repeat(MAX_ERROR_BYTES * 2), "错".repeat(MAX_ERROR_BYTES)] {
            let message = bounded_error(&anyhow::anyhow!(text));
            assert!(message.len() <= MAX_ERROR_BYTES);
            assert!(message.ends_with(TRUNCATED));
        }
        let exact = "x".repeat(MAX_ERROR_BYTES);
        assert_eq!(bounded_error(&anyhow::anyhow!(exact.clone())), exact);
        assert_eq!(bounded_error(&anyhow::anyhow!("invalid handle")), "invalid handle");
    }

    #[test]
    fn escaped_json_error_stays_below_frame_limit() {
        let reply = crate::Reply::result(Err(anyhow::anyhow!("\0".repeat(MAX_ERROR_BYTES * 2))));
        let bytes = serde_json::to_vec(&reply).unwrap();
        assert!(bytes.len() < crate::MAX_FRAME);
        assert!(reply.error.unwrap().len() <= MAX_ERROR_BYTES);
    }

    #[test]
    fn binary_error_leaves_following_frame_aligned() -> Result<()> {
        let (mut sender, mut receiver) = UnixStream::pair()?;
        let sent = send_binary_error(&mut sender, &anyhow::anyhow!("错".repeat(MAX_ERROR_BYTES)))?;
        crate::send(&mut sender, &crate::Reply::result(Ok(serde_json::json!({"next": true}))))?;
        let mut header = [0; 8];
        receiver.read_exact(&mut header)?;
        assert_eq!(u32::from_le_bytes(header[..4].try_into().unwrap()), 1);
        let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        assert!(size <= MAX_ERROR_BYTES);
        assert_eq!(sent, size + 8);
        let mut bytes = vec![0; size];
        receiver.read_exact(&mut bytes)?;
        assert!(String::from_utf8(bytes).unwrap().ends_with(TRUNCATED));
        let reply: crate::Reply = crate::receive(&mut receiver)?;
        assert_eq!(reply.value.unwrap()["next"], true);
        Ok(())
    }
}
