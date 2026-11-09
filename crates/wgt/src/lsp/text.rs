//! Byte offsets internally, UTF-16 code units at the LSP boundary.
use serde_json::{Value, json};
use std::ops::Range;

pub const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;

pub struct Text {
    pub source: String,
    starts: Vec<usize>,
}

impl Text {
    pub fn new(source: String) -> Self {
        let mut starts = vec![0];
        let bytes = source.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\r' => {
                    i += 1;
                    if bytes.get(i) == Some(&b'\n') {
                        i += 1;
                    }
                    starts.push(i);
                }
                b'\n' => {
                    i += 1;
                    starts.push(i);
                }
                _ => i += 1,
            }
        }
        Self { source, starts }
    }

    pub fn lines(&self) -> usize {
        self.starts.len()
    }

    pub fn line_range(&self, line: usize) -> Range<usize> {
        let Some(&start) = self.starts.get(line) else {
            return self.source.len()..self.source.len();
        };
        let mut end = self
            .starts
            .get(line + 1)
            .copied()
            .unwrap_or(self.source.len());
        while end > start && matches!(self.source.as_bytes()[end - 1], b'\r' | b'\n') {
            end -= 1;
        }
        start..end
    }

    pub fn position(&self, offset: usize) -> Value {
        let mut offset = offset.min(self.source.len());
        while !self.source.is_char_boundary(offset) {
            offset -= 1;
        }
        let line = self.starts.partition_point(|&start| start <= offset) - 1;
        let span = self.line_range(line);
        let character = self.source[span.start..offset.min(span.end)]
            .encode_utf16()
            .count();
        json!({ "line": line, "character": character })
    }

    pub fn offset(&self, position: &Value) -> Option<usize> {
        let line = usize::try_from(position.get("line")?.as_u64()?).ok()?;
        let character = usize::try_from(position.get("character")?.as_u64()?).ok()?;
        self.starts.get(line)?;
        let span = self.line_range(line);
        let mut units = 0;
        for (offset, ch) in self.source[span.clone()].char_indices() {
            if units == character {
                return Some(span.start + offset);
            }
            units += ch.len_utf16();
            if units > character {
                return None; // A position cannot split a surrogate pair.
            }
        }
        Some(span.end) // LSP clamps positions past the end of a line.
    }

    pub fn range(&self, start: usize, end: usize) -> Value {
        json!({ "start": self.position(start), "end": self.position(end) })
    }

    /// Compiler columns count Unicode scalar values, not UTF-16 units.
    pub fn compiler_offset(&self, line: usize, column: Option<usize>) -> usize {
        let span = self.line_range(line.saturating_sub(1));
        if let Some(column) = column {
            let column = column.saturating_sub(1);
            span.start
                + self.source[span.clone()]
                    .char_indices()
                    .nth(column)
                    .map_or(span.len(), |(i, _)| i)
        } else {
            span.start + self.source[span.clone()].len() - self.source[span].trim_start().len()
        }
    }

    /// Apply an entire didChange batch atomically, in notification order.
    pub fn changed(&self, changes: &[Value]) -> Result<Self, &'static str> {
        let mut next = Self::new(self.source.clone());
        for change in changes {
            let replacement = change
                .get("text")
                .and_then(Value::as_str)
                .ok_or("missing change text")?;
            if let Some(range) = change.get("range") {
                let start = next.offset(&range["start"]).ok_or("invalid change start")?;
                let end = next.offset(&range["end"]).ok_or("invalid change end")?;
                if start > end {
                    return Err("reversed change range");
                }
                if next.source.len() - (end - start) + replacement.len() > MAX_DOCUMENT_BYTES {
                    return Err("document exceeds the 1 MiB analysis limit");
                }
                next.source.replace_range(start..end, replacement);
            } else {
                if replacement.len() > MAX_DOCUMENT_BYTES {
                    return Err("document exceeds the 1 MiB analysis limit");
                }
                next.source = replacement.into();
            }
            next = Self::new(next.source);
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_and_all_line_endings() {
        let text = Text::new("a😀中\r\nb\rc\n".into());
        assert_eq!(text.lines(), 4);
        assert_eq!(text.position(5), json!({"line":0,"character":3}));
        assert_eq!(text.offset(&json!({"line":0,"character":3})), Some(5));
        assert_eq!(text.offset(&json!({"line":0,"character":2})), None);
        assert_eq!(text.offset(&json!({"line":0,"character":99})), Some(8));
        assert_eq!(text.position(11), json!({"line":1,"character":1}));
        assert_eq!(text.position(14), json!({"line":3,"character":0}));
    }

    #[test]
    fn sequential_edits_and_invalid_batches() {
        let text = Text::new("😀abc".into());
        let edit = |start, end, value| json!({"range":{"start":{"line":0,"character":start},"end":{"line":0,"character":end}},"text":value});
        let next = text.changed(&[edit(2, 3, "XY"), edit(4, 5, "Z")]).unwrap();
        assert_eq!(next.source, "😀XYZc");
        assert!(text.changed(&[edit(2, 3, "XY"), edit(1, 2, "Z")]).is_err());
        assert_eq!(text.source, "😀abc");
    }
}
