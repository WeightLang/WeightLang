use anyhow::{Context, Result};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use weightlang_runtime::training::{mlp, Objective, Optimizer, Trainer};
use wgtime::{Client, Request};

struct Server {
    child: Child,
    root: PathBuf,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Server {
    fn new() -> Result<Self> {
        let root = PathBuf::from("/tmp/weightlang").join(format!(
            "wgtime-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir_all(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        let child = Command::new(env!("CARGO_BIN_EXE_wgtime"))
            .arg("--test-directory")
            .arg(&root)
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut server = Self { child, root };
        let start = Instant::now();
        loop {
            if server.client().is_ok() {
                break;
            }
            anyhow::ensure!(
                server.child.try_wait()?.is_none(),
                "daemon exited before ready"
            );
            anyhow::ensure!(
                start.elapsed() < Duration::from_secs(10),
                "daemon readiness timeout"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(server)
    }
    fn client(&self) -> Result<Client> {
        Client::connect(&self.root.join("service.sock"), unsafe { libc::geteuid() })
    }
}
fn create(client: &mut Client, widths: &[usize]) -> Result<u64> {
    client.call(Request::Create {
        widths: widths.to_vec(),
        activation: "tanh".into(),
        seed: 42,
        optimizer: "adam".into(),
        memory: 16 * 1024 * 1024,
    })?["handle"]
        .as_u64()
        .context("missing handle")
}
fn predict(client: &mut Client, handle: u64, input: &[f32]) -> Result<Vec<f32>> {
    Ok(serde_json::from_value(
        client.call(Request::Predict {
            handle,
            input: input.to_vec(),
        })?["output"]
            .clone(),
    )?)
}
#[test]
fn real_process_isolation_atomic_training_and_baseline() -> Result<()> {
    let cold = Instant::now();
    let server = Server::new()?;
    let cold_us = cold.elapsed().as_micros();
    assert!(server.child.id() > 0);
    let mut clients = Vec::new();
    for _ in 0..8 {
        let root = server.root.clone();
        clients.push(std::thread::spawn(move || {
            let status = Command::new(env!("CARGO_BIN_EXE_wgtime"))
                .arg("--test-directory")
                .arg(root)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap();
            assert!(
                !status.success(),
                "competing daemon acquired singleton lock"
            );
        }));
    }
    for client in clients {
        client.join().unwrap();
    }
    assert!(Client::connect(
        &server.root.join("service.sock"),
        unsafe { libc::geteuid() }.wrapping_add(1)
    )
    .is_err());
    let mut first = server.client()?;
    let mut second = server.client()?;
    let handle = create(&mut first, &[4, 8, 2])?;
    assert!(predict(&mut second, handle, &[0.1; 4]).is_err());
    let second_handle = create(&mut second, &[4, 8, 2])?;
    let mut direct = Trainer::new(
        mlp(&[4, 8, 2], "tanh", 42)?,
        std::path::Path::new("."),
        16 * 1024 * 1024,
        Optimizer::Adam,
    )?;
    let input = vec![0.1; 4];
    let target = vec![0.2, -0.3];
    let original = predict(&mut first, handle, &input)?;
    assert_eq!(original, direct.predict(&input)?);
    for _ in 0..12 {
        let expected = direct.step(&[input.clone()], Objective::Mse(&[target.clone()]), 0.01)?;
        let reply = first.call(Request::Train {
            handle,
            inputs: vec![input.clone()],
            targets: vec![target.clone()],
            rate: 0.01,
        })?;
        assert_eq!(reply["loss"].as_f64().unwrap() as f32, expected);
        assert_eq!(
            predict(&mut first, handle, &input)?,
            direct.predict(&input)?
        );
    }
    assert_eq!(
        predict(&mut second, second_handle, &input)?,
        original,
        "mutable sessions aliased"
    );
    let checkpoint = first.call(Request::Checkpoint { handle })?["checkpoint"].as_u64().unwrap();
    let restored = first.call(Request::Restore { checkpoint, memory: 16 * 1024 * 1024 })?["handle"].as_u64().unwrap();
    assert_eq!(predict(&mut first, handle, &input)?, predict(&mut first, restored, &input)?);
    let a = first.call(Request::Train { handle, inputs: vec![input.clone()], targets: vec![target.clone()], rate: 0.01 })?;
    let b = first.call(Request::Train { handle: restored, inputs: vec![input.clone()], targets: vec![target.clone()], rate: 0.01 })?;
    assert_eq!(a, b, "checkpoint lost optimizer state");
    assert_eq!(predict(&mut first, handle, &input)?, predict(&mut first, restored, &input)?);
    let before = predict(&mut first, handle, &input)?;
    assert!(first
        .call(Request::Train {
            handle,
            inputs: vec![input.clone()],
            targets: vec![vec![f32::MAX; 2]],
            rate: f32::MAX
        })
        .is_err());
    assert_eq!(
        predict(&mut first, handle, &input)?,
        before,
        "failed update changed parameters"
    );
    let dropped = second_handle;
    second.call(Request::Drop { handle: dropped })?;
    assert!(predict(&mut second, dropped, &input).is_err());
    drop(first);
    let mut third = server.client()?;
    assert!(predict(&mut third, handle, &input).is_err());

    let model = "wgt 0.5\nmodel Regression\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1,1] = [0]\nfn forward(x) { return linear(x,w); }\nfn loss(prediction,target) { return square(prediction-target).mean; }\n";
    let program = weightlang_runtime::adaptive::parse(model)?;
    let hosted = third.call(Request::CreateProgram { program: serde_json::to_value(program)?, memory: 1024 * 1024 })?["handle"].as_u64().unwrap();
    let source = format!("{model}\nfn strategy(net,inputs,targets) {{ let g=differentiate(net,inputs,targets); let s=gradient_snapshot(g); commit_gradients(g,[[s[\"parameters\"][0][0]-0.1*s[\"gradients\"][0][0]]],null); }}\nfn main() {{ return null; }}");
    let strategy = third.call(Request::RegisterStrategy { source, entry: "strategy".into() })?["strategy"].as_u64().unwrap();
    let metrics = third.call(Request::ExecuteStrategy { handle: hosted, strategy, inputs: vec![vec![1.],vec![2.]], targets: vec![vec![2.],vec![4.]], operations: 1_000_000, memory: 16 * 1024 * 1024 })?;
    assert_eq!(metrics["commits"], 1);
    assert_eq!(metrics["loss"], 10.0);
    assert_eq!(predict(&mut third, hosted, &[3.])?, vec![3.]);
    assert!(second.call(Request::ExecuteStrategy { handle: hosted, strategy, inputs: vec![vec![1.]], targets: vec![vec![2.]], operations: 100000, memory: 1024 * 1024 }).is_err());
    assert_eq!(metrics.as_object().unwrap().len(), 3, "resident parameters or gradients escaped");

    // Same seed/dtype/backend/one connection thread; small and compute-bound shapes.
    for widths in [&[4, 8, 2][..], &[256, 256, 256][..]] {
        let mut client = server.client()?;
        let load = Instant::now();
        let handle = create(&mut client, widths)?;
        let load_us = load.elapsed().as_micros();
        let mut direct = Trainer::new(
            mlp(widths, "tanh", 42)?,
            std::path::Path::new("."),
            16 * 1024 * 1024,
            Optimizer::Adam,
        )?;
        let input = vec![0.1; widths[0]];
        for _ in 0..100 {
            direct.predict(&input)?;
            predict(&mut client, handle, &input)?;
        }
        let mut local_ns = Vec::new();
        let mut ipc_ns = Vec::new();
        for _ in 0..1000 {
            let start = Instant::now();
            direct.predict(&input)?;
            local_ns.push(start.elapsed().as_nanos());
            let start = Instant::now();
            predict(&mut client, handle, &input)?;
            ipc_ns.push(start.elapsed().as_nanos());
        }
        let mut writable = wgtime::buffer::WritableTensor::new(input.len())?;
        writable.write(&input)?;
        let sealed = writable.commit()?;
        let tensor = client.register_tensor(&sealed, vec![input.len()])?;
        assert!(second.call(Request::PredictTensor { handle: second_handle, tensor, start: 0, count: input.len() }).is_err());
        assert!(client.call(Request::PredictTensor { handle, tensor, start: usize::MAX, count: 1 }).is_err());
        for _ in 0..100 { client.call(Request::PredictTensor { handle, tensor, start: 0, count: input.len() })?; }
        let mut shared_ns = Vec::new();
        let initial_bytes = client.transmitted_bytes;
        for _ in 0..1000 {
            let start = Instant::now();
            client.call(Request::PredictTensor { handle, tensor, start: 0, count: input.len() })?;
            shared_ns.push(start.elapsed().as_nanos());
        }
        let shared_control_bytes = client.transmitted_bytes - initial_bytes;
        let mut binary_ns = Vec::new();
        let mut binary_output = Vec::new();
        for _ in 0..100 { client.predict_into(handle, tensor, 0, input.len(), &mut binary_output)?; }
        for _ in 0..1000 {
            let start = Instant::now();
            client.predict_into(handle, tensor, 0, input.len(), &mut binary_output)?;
            binary_ns.push(start.elapsed().as_nanos());
        }
        assert_eq!(binary_output, direct.predict(&input)?);
        binary_ns.sort_unstable();
        println!("wgtime-binary {}",serde_json::json!({"widths":widths,"p50":binary_ns[500],"p95":binary_ns[950],"p99":binary_ns[990]}));
        shared_ns.sort_unstable();
        client.call(Request::ReleaseTensor { tensor })?;
        assert!(client.call(Request::PredictTensor { handle, tensor, start: 0, count: input.len() }).is_err());
        println!("wgtime-shared {}", serde_json::json!({"widths":widths,"p50":shared_ns[500],"p95":shared_ns[950],"p99":shared_ns[990],"requests":1000,"control_bytes":shared_control_bytes,"input_payload_ipc_bytes_after_registration":0}));
        local_ns.sort_unstable();
        ipc_ns.sort_unstable();
        let training_handle = create(&mut client, widths)?;
        let mut training_direct = Trainer::new(mlp(widths,"tanh",42)?,std::path::Path::new("."),16*1024*1024,Optimizer::Adam)?;
        let training_inputs = vec![input.clone(); 8];
        let training_targets = vec![vec![0.2; *widths.last().unwrap()]; 8];
        let mut direct_steps = Vec::new();
        let mut ipc_steps = Vec::new();
        for iteration in 0..40 {
            let begin = Instant::now();
            let expected = training_direct.step(&training_inputs,Objective::Mse(&training_targets),0.001)?;
            let direct_ns = begin.elapsed().as_nanos();
            let begin = Instant::now();
            let reply = client.call(Request::Train { handle: training_handle, inputs: training_inputs.clone(), targets: training_targets.clone(), rate:0.001 })?;
            let ipc_ns = begin.elapsed().as_nanos();
            assert_eq!(reply["loss"].as_f64().unwrap() as f32,expected);
            if iteration >= 10 { direct_steps.push(direct_ns); ipc_steps.push(ipc_ns); }
        }
        direct_steps.sort_unstable(); ipc_steps.sort_unstable();
        println!("wgtime-training {}",serde_json::json!({"widths":widths,"batch":8,"warmup":10,"samples":30,"direct_p50_ns":direct_steps[15],"ipc_p50_ns":ipc_steps[15],"direct_samples_s":8e9/direct_steps[15] as f64,"ipc_samples_s":8e9/ipc_steps[15] as f64}));
        println!(
            "wgtime-baseline {}",
            serde_json::json!({"widths": widths, "cold_us": cold_us, "load_us": load_us, "warmup":100, "samples":1000, "direct_ns":{"p50":local_ns[500],"p95":local_ns[950],"p99":local_ns[990]}, "ipc_ns":{"p50":ipc_ns[500],"p95":ipc_ns[950],"p99":ipc_ns[990]}, "client_control_bytes":client.transmitted_bytes})
        );
    }
    Ok(())
}
