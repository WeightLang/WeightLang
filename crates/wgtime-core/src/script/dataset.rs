//! Bounded native data streams shared by interpreted and compiled scripts.
use super::Value;
use anyhow::{ensure, Context, Result};
use reqwest::blocking::Client;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
    time::Duration,
};

const ROW_LIMIT: usize = 1_048_576;
const BATCH_LIMIT: usize = 16 * ROW_LIMIT;

pub(super) enum Reader {
    Jsonl(BufReader<File>),
    HuggingFace(Hub),
}

pub(super) struct Hub {
    client: Client,
    endpoint: String,
    dataset: String,
    config: String,
    split: String,
    token: Option<String>,
    offset: usize,
    total: Option<usize>,
}

impl Reader {
    pub fn jsonl(path: &Path) -> Result<Self> {
        Ok(Self::Jsonl(BufReader::new(File::open(path)?)))
    }

    pub fn huggingface(dataset: &str, config: &str, split: &str) -> Result<Self> {
        ensure!(
            !dataset.trim().is_empty() && !config.trim().is_empty() && !split.trim().is_empty(),
            "dataset, config and split must be nonempty"
        );
        Ok(Self::HuggingFace(Hub {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            endpoint: "https://datasets-server.huggingface.co/rows".into(),
            dataset: dataset.into(),
            config: config.into(),
            split: split.into(),
            token: std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty()),
            offset: 0,
            total: None,
        }))
    }

    pub fn batch(&mut self, count: usize) -> Result<Vec<Value>> {
        let mut result = Vec::new();
        let mut bytes = 0;
        match self {
            Self::Jsonl(reader) => {
                for _ in 0..count {
                    let mut line = String::new();
                    let read = reader
                        .by_ref()
                        .take((ROW_LIMIT + 1) as u64)
                        .read_line(&mut line)?;
                    if read == 0 {
                        break;
                    }
                    ensure!(read <= ROW_LIMIT, "JSONL row exceeds 1 MiB");
                    bytes += read;
                    ensure!(bytes <= BATCH_LIMIT, "JSONL batch exceeds 16 MiB");
                    result.push(Value::from_json(serde_json::from_str(&line)?)?);
                }
            }
            Self::HuggingFace(hub) => {
                // Commit the cursor only after the entire requested batch validates.
                let mut offset = hub.offset;
                while result.len() < count && hub.total.is_none_or(|total| offset < total) {
                    let length = (count - result.len()).min(100);
                    let mut request = hub.client.get(&hub.endpoint).query(&[
                        ("dataset", hub.dataset.clone()),
                        ("config", hub.config.clone()),
                        ("split", hub.split.clone()),
                        ("offset", offset.to_string()),
                        ("length", length.to_string()),
                    ]);
                    if let Some(token) = &hub.token {
                        request = request.bearer_auth(token);
                    }
                    let response = request.send().context("Hugging Face request failed")?;
                    ensure!(
                        response.status().is_success(),
                        "Hugging Face rows HTTP {}",
                        response.status()
                    );
                    let mut body = Vec::new();
                    response
                        .take((BATCH_LIMIT + 1) as u64)
                        .read_to_end(&mut body)?;
                    ensure!(
                        body.len() <= BATCH_LIMIT,
                        "Hugging Face response exceeds 16 MiB"
                    );
                    let page: serde_json::Value = serde_json::from_slice(&body)?;
                    let total = page["num_rows_total"]
                        .as_u64()
                        .context("missing num_rows_total")?;
                    let total = usize::try_from(total)?;
                    if let Some(previous) = hub.total {
                        ensure!(previous == total, "dataset changed during reading");
                    }
                    let rows = page["rows"].as_array().context("missing rows")?;
                    ensure!(
                        rows.len() <= length && offset + rows.len() <= total,
                        "invalid dataset page length"
                    );
                    ensure!(
                        !rows.is_empty() || offset == total,
                        "empty page before dataset end"
                    );
                    for (index, row) in rows.iter().enumerate() {
                        ensure!(
                            row["row_idx"].as_u64() == Some((offset + index) as u64),
                            "unexpected dataset row index"
                        );
                        ensure!(
                            row["truncated_cells"].as_array().is_some_and(Vec::is_empty),
                            "dataset row contains truncated or unverified cells"
                        );
                        let value = row.get("row").context("missing row payload")?;
                        ensure!(value.is_object(), "dataset row must be an object");
                        let size = serde_json::to_vec(value)?.len();
                        ensure!(size <= ROW_LIMIT, "dataset row exceeds 1 MiB");
                        bytes += size;
                        ensure!(bytes <= BATCH_LIMIT, "dataset batch exceeds 16 MiB");
                        result.push(Value::from_json(value.clone())?);
                    }
                    offset += rows.len();
                    hub.total = Some(total);
                }
                hub.offset = offset;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::TcpListener, thread};

    fn mock(pages: Vec<(usize, usize, serde_json::Value)>) -> (Reader, thread::JoinHandle<()>) {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/rows", server.local_addr().unwrap());
        let worker = thread::spawn(move || {
            for (offset, length, body) in pages {
                let (mut stream, _) = server.accept().unwrap();
                let mut input = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                input.read_line(&mut line).unwrap();
                assert!(line.contains(&format!("offset={offset}")));
                assert!(line.contains(&format!("length={length}")));
                assert!(line.contains("dataset=test%2Fdata"));
                loop {
                    line.clear();
                    input.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                let body = body.to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .unwrap();
            }
        });
        let mut reader = Reader::huggingface("test/data", "default", "train").unwrap();
        if let Reader::HuggingFace(hub) = &mut reader {
            hub.endpoint = endpoint;
            hub.token = None;
            hub.client = Client::builder().no_proxy().build().unwrap();
        }
        (reader, worker)
    }

    fn page(start: usize, end: usize, total: usize) -> serde_json::Value {
        serde_json::json!({"num_rows_total":total, "rows":(start..end).map(|i|
            serde_json::json!({"row_idx":i,"row":{"x":i},"truncated_cells":[]})
        ).collect::<Vec<_>>()})
    }

    #[test]
    fn pagination_tail_and_repeated_eof() {
        let (mut reader, worker) = mock(vec![
            (0, 100, page(0, 100, 103)),
            (100, 2, page(100, 102, 103)),
            (102, 8, page(102, 103, 103)),
        ]);
        let rows = reader.batch(102).unwrap();
        assert_eq!(rows.len(), 102);
        assert_eq!(rows[101].json().unwrap()["x"].as_f64(), Some(101.0));
        assert_eq!(reader.batch(8).unwrap().len(), 1);
        assert!(reader.batch(8).unwrap().is_empty());
        assert!(reader.batch(1).unwrap().is_empty());
        worker.join().unwrap();
    }

    #[test]
    fn rejects_truncation_and_invalid_pages_without_advancing() {
        let mut truncated = page(0, 1, 1);
        truncated["rows"][0]["truncated_cells"] = serde_json::json!(["x"]);
        for bad in [
            truncated,
            page(1, 2, 2),
            page(0, 0, 2),
            serde_json::json!({}),
        ] {
            let (mut reader, worker) = mock(vec![(0, 1, bad), (0, 1, page(0, 1, 1))]);
            assert!(reader.batch(1).is_err());
            assert_eq!(reader.batch(1).unwrap().len(), 1);
            worker.join().unwrap();
        }
    }
    #[test]
    fn http_failure_is_explicit() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut reader = Reader::huggingface("test/data", "default", "train").unwrap();
        if let Reader::HuggingFace(hub) = &mut reader {
            hub.endpoint = format!("http://{}/rows", server.local_addr().unwrap());
            hub.client = Client::builder().no_proxy().build().unwrap();
            hub.token = None;
        }
        let worker = thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut input = BufReader::new(stream.try_clone().unwrap());
            loop {
                let mut line = String::new();
                input.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
        });
        assert!(reader.batch(1).unwrap_err().to_string().contains("429"));
        worker.join().unwrap();
    }
}
