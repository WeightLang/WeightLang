//! OS-process clients with a common parent timing window and active memory snapshots.
use anyhow::{ensure, Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
struct Run {
    root: PathBuf,
    children: Vec<Child>,
}
impl Run {
    fn wait(&mut self, path: &Path) -> Result<()> {
        let start = Instant::now();
        loop {
            for child in &mut self.children {
                if let Some(status) = child.try_wait()? {
                    anyhow::bail!("worker {} exited before release: {status}", child.id());
                }
            }
            if path.exists() {
                return Ok(());
            }
            ensure!(
                start.elapsed() < Duration::from_secs(30),
                "barrier timed out: {}",
                path.display()
            );
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}
impl Drop for Run {
    fn drop(&mut self) {
        for child in &mut self.children {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}
use wgtime::{Client, Request};
fn memory(pid: u32) -> Result<String> {
    Ok(fs::read_to_string(format!("/proc/{pid}/smaps_rollup"))?
        .lines()
        .filter(|line| line.starts_with("Rss:") || line.starts_with("Pss:"))
        .collect::<Vec<_>>()
        .join("; "))
}
fn wait(path: &Path) -> Result<()> {
    let start = Instant::now();
    while !path.exists() {
        ensure!(
            start.elapsed() < Duration::from_secs(30),
            "barrier timed out"
        );
        std::thread::sleep(Duration::from_micros(200));
    }
    Ok(())
}
fn worker(endpoint: &str, root: &Path, index: usize) -> Result<()> {
    let mut client = Client::connect(Path::new(endpoint), unsafe { libc::geteuid() })?;
    let handle = client.call(Request::Create {
        widths: vec![256, 256, 256],
        activation: "tanh".into(),
        seed: 42,
        optimizer: "adam".into(),
        memory: 16 * 1024 * 1024,
    })?["handle"]
        .as_u64()
        .unwrap();
    let mut input = wgtime::buffer::WritableTensor::new(256)?;
    input.write(&[0.1; 256])?;
    let sealed = input.commit()?;
    let tensor = client.register_tensor(&sealed, vec![256])?;
    let mut output = Vec::new();
    for _ in 0..100 {
        client.predict_into(handle, tensor, 0, 256, &mut output)?;
    }
    fs::write(root.join(format!("ready-{index}")), b"")?;
    wait(&root.join("start"))?;
    let begin = Instant::now();
    let mut samples = Vec::with_capacity(10000);
    for _ in 0..10000 {
        let step = Instant::now();
        client.predict_into(handle, tensor, 0, 256, &mut output)?;
        samples.push(step.elapsed().as_nanos());
    }
    let elapsed = begin.elapsed().as_secs_f64();
    samples.sort_unstable();
    fs::write(
        root.join(format!("result-{index}.json")),
        serde_json::to_vec(
            &serde_json::json!({"pid":std::process::id(),"p50_ns":samples[5000],"p95_ns":samples[9500],"p99_ns":samples[9900],"requests":10000,"elapsed_s":elapsed,"control_bytes":client.transmitted_bytes}),
        )?,
    )?;
    fs::write(root.join(format!("done-{index}")), b"")?;
    wait(&root.join("release"))?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|n| n == "--worker") {
        return worker(&args[2], Path::new(&args[3]), args[4].parse()?);
    }
    let endpoint = args.get(1).context("socket required")?;
    let service_pid: u32 = args.get(2).context("service pid required")?.parse()?;
    let count: usize = args.get(3).context("count required")?.parse()?;
    ensure!((1..=64).contains(&count), "process count must be 1..=64");
    let root = Path::new("/tmp/weightlang").join(format!("process-load-{}", std::process::id()));
    fs::create_dir(&root)?;
    let mut run = Run {
        root: root.clone(),
        children: Vec::new(),
    };
    for index in 0..count {
        run.children.push(
            Command::new(std::env::current_exe()?)
                .args([
                    "--worker",
                    endpoint,
                    root.to_str().unwrap(),
                    &index.to_string(),
                ])
                .stdout(Stdio::null())
                .spawn()?,
        );
    }
    for index in 0..count {
        run.wait(&root.join(format!("ready-{index}")))?;
    }
    let before = run
        .children
        .iter()
        .map(|child| memory(child.id()))
        .collect::<Result<Vec<_>>>()?;
    let service_before = memory(service_pid)?;
    let begin = Instant::now();
    fs::write(root.join("start"), b"")?;
    for index in 0..count {
        run.wait(&root.join(format!("done-{index}")))?;
    }
    let elapsed = begin.elapsed().as_secs_f64();
    let after = run
        .children
        .iter()
        .map(|child| memory(child.id()))
        .collect::<Result<Vec<_>>>()?;
    let service_after = memory(service_pid)?;
    let mut results = Vec::new();
    for index in 0..count {
        results.push(serde_json::from_slice::<serde_json::Value>(&fs::read(
            root.join(format!("result-{index}.json")),
        )?)?);
    }
    fs::write(root.join("release"), b"")?;
    let release_start = Instant::now();
    for child in &mut run.children {
        loop {
            if let Some(status) = child.try_wait()? {
                ensure!(status.success(), "worker failed: {status}");
                break;
            }
            ensure!(
                release_start.elapsed() < Duration::from_secs(30),
                "worker exit timed out"
            );
            std::thread::sleep(Duration::from_micros(200));
        }
    }
    println!(
        "{}",
        serde_json::json!({"processes":count,"cpu_configuration":"parent, clients and daemon inherit CPU8 only","requests":count*10000,"common_window_s":elapsed,"aggregate_requests_s":count as f64*10000./elapsed,"workers":results,"client_active_memory_before":before,"client_active_memory_after":after,"service_active_memory_before":service_before,"service_active_memory_after":service_after,"memory_is_peak":false,"weights_shared":false})
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
