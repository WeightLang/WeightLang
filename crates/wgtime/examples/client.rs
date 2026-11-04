//! Standalone proof of the lightweight link boundary; not the WGT AOT migration.
use anyhow::Result;
use std::path::Path;
use wgtime::{Client, Request, DEFAULT_ENDPOINT};
fn main() -> Result<()> {
    let endpoint = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_ENDPOINT.into());
    let mut client = Client::connect(Path::new(&endpoint), unsafe { libc::geteuid() })?;
    let created = client.call(Request::Create { widths: vec![2, 4, 1], activation: "tanh".into(), seed: 42, optimizer: "adam".into(), memory: 1024 * 1024 })?;
    let handle = created["handle"].as_u64().unwrap();
    let output = client.call(Request::Predict { handle, input: vec![0.1, 0.2] })?;
    println!("{output}");
    client.call(Request::Drop { handle })?;
    Ok(())
}
