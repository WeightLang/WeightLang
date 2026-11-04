//! Authorized POST transport. Credentials are host-only and never serialized.
use super::{budget::Budget, sse::Decoder, wire};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::io::Read;
pub struct Endpoint {
    url: reqwest::Url,
    authorization: Option<reqwest::header::HeaderValue>,
}
impl Endpoint {
    pub fn new(url: &str, authorization: Option<&str>) -> Result<Self> {
        let url = reqwest::Url::parse(url).context("invalid response endpoint")?;
        ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.fragment().is_none()
                && url.query().is_none(),
            "endpoint must not contain credentials/query/fragment"
        );
        ensure!(
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))),
            "response endpoint requires HTTPS or loopback HTTP"
        );
        let authorization = authorization
            .map(|s| {
                let mut v = reqwest::header::HeaderValue::from_str(s)
                    .map_err(|_| anyhow::anyhow!("invalid authorization header"))?;
                v.set_sensitive(true);
                Ok::<_, anyhow::Error>(v)
            })
            .transpose()?;
        Ok(Self { url, authorization })
    }
    pub fn post(
        &self,
        request: &Value,
        budget: &Budget,
        max_body: usize,
        max_events: usize,
    ) -> Result<(Value, Vec<Value>)> {
        self.post_checked(request, budget, max_body, max_events, &|| Ok(()))
    }
    pub fn post_checked(
        &self,
        request: &Value,
        budget: &Budget,
        max_body: usize,
        max_events: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(Value, Vec<Value>)> {
        check()?;
        wire::request(request)?;
        let body = serde_json::to_vec(request)?;
        ensure!(body.len() <= max_body, "request body exceeds limit");
        let _flight = budget.flight()?;
        budget.bytes(body.len() as u64)?;
        let timeout = budget.remaining()?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(timeout)
            .connect_timeout(timeout)
            .build()
            .map_err(|_| anyhow::anyhow!("response client initialization failed"))?;
        let mut builder = client
            .post(self.url.clone())
            .header("Content-Type", "application/json")
            .header(
                "Accept",
                if request["stream"] == true {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .body(body);
        if let Some(auth) = &self.authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, auth.clone());
        }
        let mut response = builder
            .send()
            .map_err(|_| anyhow::anyhow!("response network failure; POST was not retried"))?;
        ensure!(
            response.status().is_success(),
            "response HTTP status {}",
            response.status().as_u16()
        );
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let streaming = request["stream"] == true;
        ensure!(
            content_type
                == if streaming {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            "unexpected response content type"
        );
        let mut bytes = Vec::new();
        let mut decoder = Decoder::new(max_body.min(1024 * 1024), max_events);
        let mut used = 0usize;
        let mut chunk = [0u8; 8192];
        loop {
            check()?;
            budget.remaining()?;
            let count = response
                .read(&mut chunk)
                .map_err(|_| anyhow::anyhow!("response body read failed; POST was not retried"))?;
            if count == 0 {
                break;
            }
            budget.bytes(count as u64)?;
            used = used.checked_add(count).context("body length overflow")?;
            ensure!(used <= max_body, "response body exceeds limit");
            if streaming {
                decoder.feed(&chunk[..count])?;
            } else {
                bytes.extend_from_slice(&chunk[..count]);
            }
        }
        if streaming {
            decoder.finish()
        } else {
            let value: Value = serde_json::from_slice(&bytes).context("invalid response JSON")?;
            wire::response(&value)?;
            ensure!(
                wire::terminal(wire::text(&value, "status")?),
                "nonterminal synchronous response"
            );
            Ok((value, Vec::new()))
        }
    }
}
