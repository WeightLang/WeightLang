//! Incremental SSE decoder with bounded buffering and terminal response gates.
use super::wire;
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub struct Decoder {
    line: Vec<u8>,
    data: Vec<String>,
    event: String,
    skip_lf: bool,
    max_event: usize,
    event_bytes: usize,
    sequence: Option<u64>,
    items: BTreeMap<u64, (String, bool)>,
    completed_items: BTreeMap<u64, Value>,
    parts: BTreeMap<(u64, u64, bool), (String, bool)>,
    arguments: BTreeMap<u64, (String, bool)>,
    pub final_response: Option<Value>,
    pub events: Vec<Value>,
    max_events: usize,
    done: bool,
}
impl Decoder {
    pub fn new(max_event: usize, max_events: usize) -> Self {
        Self {
            line: Vec::new(),
            data: Vec::new(),
            event: String::new(),
            skip_lf: false,
            max_event,
            event_bytes: 0,
            sequence: None,
            items: BTreeMap::new(),
            completed_items: BTreeMap::new(),
            parts: BTreeMap::new(),
            arguments: BTreeMap::new(),
            final_response: None,
            events: Vec::new(),
            max_events,
            done: false,
        }
    }
    pub fn feed(&mut self, bytes: &[u8]) -> Result<()> {
        for &byte in bytes {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if byte == b'\r' || byte == b'\n' {
                self.skip_lf = byte == b'\r';
                let line = String::from_utf8(std::mem::take(&mut self.line))
                    .context("invalid SSE UTF-8")?;
                self.line(&line)?;
            } else {
                ensure!(!self.done, "data after SSE DONE");
                ensure!(self.line.len() < self.max_event, "SSE line exceeds limit");
                self.line.push(byte);
            }
        }
        Ok(())
    }
    fn line(&mut self, line: &str) -> Result<()> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        self.event_bytes = self
            .event_bytes
            .checked_add(line.len())
            .context("SSE event overflow")?;
        ensure!(
            self.event_bytes <= self.max_event,
            "SSE event exceeds limit"
        );
        match field {
            "event" => self.event = value.into(),
            "data" => self.data.push(value.into()),
            _ => {}
        }
        Ok(())
    }
    fn dispatch(&mut self) -> Result<()> {
        self.event_bytes = 0;
        let event = std::mem::take(&mut self.event);
        if self.data.is_empty() {
            return Ok(());
        }
        let data = std::mem::take(&mut self.data).join("\n");
        if data == "[DONE]" {
            ensure!(self.final_response.is_some(), "SSE DONE before terminal");
            self.done = true;
            return Ok(());
        }
        ensure!(
            self.final_response.is_none(),
            "event after terminal response"
        );
        let value: Value = serde_json::from_str(&data).context("invalid SSE JSON")?;
        let kind = wire::text(&value, "type")?;
        ensure!(event.is_empty() || event == kind, "SSE event/type mismatch");
        let seq = value["sequence_number"]
            .as_u64()
            .context("missing SSE sequence_number")?;
        ensure!(
            self.sequence.is_none_or(|n| seq > n),
            "SSE sequence_number did not increase"
        );
        self.sequence = Some(seq);
        ensure!(
            self.events.len() < self.max_events,
            "SSE event count exceeded"
        );
        match kind {
            "response.output_item.added" => {
                let index = value["output_index"]
                    .as_u64()
                    .context("missing output index")?;
                let id = wire::text(&value["item"], "id")?.to_owned();
                ensure!(
                    self.items.insert(index, (id, false)).is_none(),
                    "duplicate SSE output item"
                );
            }
            "response.output_item.done" => {
                let index = value["output_index"]
                    .as_u64()
                    .context("missing output index")?;
                let item = self
                    .items
                    .get_mut(&index)
                    .context("item done before added")?;
                ensure!(
                    !item.1 && item.0 == wire::text(&value["item"], "id")?,
                    "invalid item completion"
                );
                ensure!(
                    self.parts
                        .iter()
                        .filter(|((i, _, _), _)| *i == index)
                        .all(|(_, (_, done))| *done),
                    "item done before content completion"
                );
                if let Some((arguments, done)) = self.arguments.get(&index) {
                    ensure!(
                        *done && value["item"]["arguments"].as_str() == Some(arguments),
                        "item arguments differ from stream"
                    );
                }
                item.1 = true;
                self.completed_items.insert(index, value["item"].clone());
            }
            "response.completed" | "response.failed" | "response.incomplete" => {
                let response = value.get("response").context("missing terminal response")?;
                wire::response(response)?;
                ensure!(
                    kind.strip_prefix("response.") == response["status"].as_str(),
                    "terminal/status mismatch"
                );
                if kind == "response.completed" {
                    ensure!(
                        self.items.values().all(|(_, done)| *done),
                        "terminal before item completion"
                    );
                    let output = response["output"].as_array().unwrap();
                    ensure!(
                        output.len() == self.completed_items.len(),
                        "terminal output differs from streamed items"
                    );
                    for (index, item) in &self.completed_items {
                        ensure!(
                            output.get(*index as usize) == Some(item),
                            "terminal output item mismatch"
                        );
                    }
                }
                self.final_response = Some(response.clone());
            }
            _ => {
                if let Some(index) = value.get("output_index").and_then(Value::as_u64) {
                    let item = self.items.get(&index).context("event before item added")?;
                    ensure!(!item.1, "event modifies completed item");
                    if let Some(id) = value.get("item_id").and_then(Value::as_str) {
                        ensure!(id == item.0, "SSE item id mismatch");
                    }
                    if kind == "response.function_call_arguments.delta" {
                        let args = self
                            .arguments
                            .entry(index)
                            .or_insert_with(|| (String::new(), false));
                        ensure!(!args.1, "arguments delta after done");
                        args.0.push_str(wire::text(&value, "delta")?);
                    } else if kind == "response.function_call_arguments.done" {
                        let args = self
                            .arguments
                            .entry(index)
                            .or_insert_with(|| (String::new(), false));
                        ensure!(
                            !args.1 && args.0 == wire::text(&value, "arguments")?,
                            "arguments done mismatch"
                        );
                        args.1 = true;
                    } else if kind == "response.content_part.added"
                        || kind == "response.reasoning_summary_part.added"
                    {
                        let summary = kind.contains("summary");
                        let field = if summary {
                            "summary_index"
                        } else {
                            "content_index"
                        };
                        let content = value[field].as_u64().context("missing content index")?;
                        ensure!(
                            self.parts
                                .insert((index, content, summary), (String::new(), false))
                                .is_none(),
                            "duplicate content part"
                        );
                    } else if kind == "response.content_part.done"
                        || kind == "response.reasoning_summary_part.done"
                    {
                        let summary = kind.contains("summary");
                        let field = if summary {
                            "summary_index"
                        } else {
                            "content_index"
                        };
                        let content = value[field].as_u64().context("missing content index")?;
                        let part = self
                            .parts
                            .get_mut(&(index, content, summary))
                            .context("content done before added")?;
                        ensure!(!part.1, "duplicate content completion");
                        if let Some(text) = value["part"]
                            .get("text")
                            .or_else(|| value["part"].get("refusal"))
                            .and_then(Value::as_str)
                        {
                            ensure!(text == part.0, "content part differs from deltas");
                        }
                        part.1 = true;
                    } else if kind.ends_with(".delta") || kind.ends_with(".done") {
                        let summary = kind.contains("reasoning_summary_text");
                        let field = if summary {
                            "summary_index"
                        } else {
                            "content_index"
                        };
                        if let Some(content) = value[field].as_u64() {
                            let part = self
                                .parts
                                .get_mut(&(index, content, summary))
                                .context("text event before content added")?;
                            ensure!(!part.1, "text modifies completed content");
                            if kind.ends_with(".delta") {
                                part.0.push_str(wire::text(&value, "delta")?);
                            } else if let Some(text) = value
                                .get("text")
                                .or_else(|| value.get("refusal"))
                                .and_then(Value::as_str)
                            {
                                ensure!(text == part.0, "text done differs from deltas");
                            }
                        }
                    }
                }
            }
        }
        self.events.push(value);
        Ok(())
    }
    pub fn finish(self) -> Result<(Value, Vec<Value>)> {
        ensure!(
            self.line.is_empty() && self.data.is_empty() && self.event.is_empty(),
            "truncated SSE event"
        );
        Ok((
            self.final_response
                .context("SSE ended before terminal response")?,
            self.events,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_terminal_without_done() {
        let text="event: response.completed\r\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"r\",\"object\":\"response\",\"status\":\"completed\",\"output\":[]}}\r\n\r\n";
        let mut d = Decoder::new(1024, 10);
        for byte in text.as_bytes() {
            d.feed(&[*byte]).unwrap();
        }
        assert_eq!(d.finish().unwrap().0["status"], "completed");
    }
    fn event(decoder: &mut Decoder, value: Value) -> Result<()> {
        decoder.feed(
            format!(
                "event: {}\ndata: {}\n\n",
                value["type"].as_str().unwrap(),
                value
            )
            .as_bytes(),
        )
    }
    #[test]
    fn content_lifecycle_and_terminal_consistency() {
        use serde_json::json;
        let mut d = Decoder::new(4096, 20);
        let item = json!({"id":"m","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"你好"}]});
        event(&mut d,json!({"type":"response.output_item.added","sequence_number":0,"output_index":0,"item":{"id":"m"}})).unwrap();
        event(&mut d,json!({"type":"response.content_part.added","sequence_number":1,"output_index":0,"item_id":"m","content_index":0,"part":{"type":"output_text","text":""}})).unwrap();
        let delta = json!({"type":"response.output_text.delta","sequence_number":2,"output_index":0,"item_id":"m","content_index":0,"delta":"你好"});
        let encoded = format!("data: {delta}\n\n");
        for byte in encoded.bytes() {
            d.feed(&[byte]).unwrap();
        }
        event(&mut d,json!({"type":"response.output_text.done","sequence_number":3,"output_index":0,"item_id":"m","content_index":0,"text":"你好"})).unwrap();
        event(&mut d,json!({"type":"response.content_part.done","sequence_number":4,"output_index":0,"item_id":"m","content_index":0,"part":{"type":"output_text","text":"你好"}})).unwrap();
        event(&mut d,json!({"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":item})).unwrap();
        event(&mut d,json!({"type":"response.completed","sequence_number":6,"response":{"id":"r","object":"response","status":"completed","output":[item]}})).unwrap();
        d.finish().unwrap();
    }
    #[test]
    fn rejects_inconsistent_arguments_and_missing_item_events() {
        use serde_json::json;
        let mut d = Decoder::new(4096, 20);
        event(&mut d,json!({"type":"response.output_item.added","sequence_number":0,"output_index":0,"item":{"id":"f"}})).unwrap();
        event(&mut d,json!({"type":"response.function_call_arguments.delta","sequence_number":1,"output_index":0,"item_id":"f","delta":"{}"})).unwrap();
        assert!(event(&mut d,json!({"type":"response.function_call_arguments.done","sequence_number":2,"output_index":0,"item_id":"f","arguments":"{\"x\":1}"})).is_err());
        let mut d = Decoder::new(4096, 20);
        assert!(event(&mut d,json!({"type":"response.completed","sequence_number":0,"response":{"id":"r","object":"response","status":"completed","output":[{"id":"f","type":"function_call","call_id":"c","name":"x","arguments":"{}","status":"completed"}]}})).is_err());
    }
    #[test]
    fn no_terminal_and_early_done_fail() {
        assert!(Decoder::new(100, 2).finish().is_err());
        assert!(Decoder::new(100, 2).feed(b"data: [DONE]\n\n").is_err());
        assert!(Decoder::new(3, 2).feed(b"abcd").is_err());
    }
}
