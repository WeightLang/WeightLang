//! Independent client load probe; no runtime/model execution dependency.
use anyhow::{Context,Result};
use std::{path::Path,time::Instant};
use wgtime::{Client,Request};
fn main()->Result<()>{
    let args:Vec<_>=std::env::args().collect();let endpoint=args.get(1).context("socket required")?;let clients:usize=args.get(2).context("client count required")?.parse()?;
    let barrier=std::sync::Arc::new(std::sync::Barrier::new(clients));let mut workers=Vec::new();
    let wall=Instant::now();
    for _ in 0..clients{let endpoint=endpoint.clone();let barrier=barrier.clone();workers.push(std::thread::spawn(move||->Result<serde_json::Value>{
        let mut client=Client::connect(Path::new(&endpoint),unsafe{libc::geteuid()})?;
        let handle=client.call(Request::Create{widths:vec![256,256,256],activation:"tanh".into(),seed:42,optimizer:"adam".into(),memory:16*1024*1024})?["handle"].as_u64().unwrap();
        let mut input=wgtime::buffer::WritableTensor::new(256)?;input.write(&[0.1;256])?;let sealed=input.commit()?;let tensor=client.register_tensor(&sealed,vec![256])?;
        let mut output=Vec::new();for _ in 0..100{client.predict_into(handle,tensor,0,256,&mut output)?;}
        barrier.wait();let start=Instant::now();let mut samples=Vec::new();for _ in 0..10000{let step=Instant::now();client.predict_into(handle,tensor,0,256,&mut output)?;samples.push(step.elapsed().as_nanos());}
        let elapsed=start.elapsed().as_secs_f64();samples.sort_unstable();Ok(serde_json::json!({"p50_ns":samples[5000],"p95_ns":samples[9500],"p99_ns":samples[9900],"requests":10000,"elapsed_s":elapsed,"requests_s":10000./elapsed,"control_bytes":client.transmitted_bytes}))
    }));}
    let mut results=Vec::new();for worker in workers{results.push(worker.join().unwrap()?);}
    let memory=std::fs::read_to_string("/proc/self/smaps_rollup")?.lines().filter(|s|s.starts_with("Rss:")||s.starts_with("Pss:")).collect::<Vec<_>>().join("; ");
    println!("{}",serde_json::json!({"clients":clients,"widths":[256,256,256],"dtype":"f32","backend":"cpu","workers":results,"wall_with_setup_s":wall.elapsed().as_secs_f64(),"client_process_memory":memory}));Ok(())
}
