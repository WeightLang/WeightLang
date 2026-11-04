use anyhow::Result;
use std::{
    fs,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, Instant},
};
use weightlang_runtime::training::{mlp, Optimizer, Trainer};
use wgtime::{Client, Request};
struct Server(Child, PathBuf);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = fs::remove_dir_all(&self.1);
    }
}
#[test]
fn diagnostic_segments() -> Result<()> {
    let root = PathBuf::from(format!("/tmp/weightlang/probe-{}", std::process::id()));
    fs::create_dir(&root)?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_wgtime"))
            .arg("--test-directory")
            .arg(&root)
            .spawn()?,
        root,
    );
    let endpoint = server.1.join("service.sock");
    let begin = Instant::now();
    let mut client = loop {
        if let Ok(c) = Client::connect(&endpoint, unsafe { libc::geteuid() }) {
            break c;
        }
        anyhow::ensure!(
            server.0.try_wait()?.is_none() && begin.elapsed() < Duration::from_secs(10),
            "server not ready"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    for widths in [vec![256, 256, 256], vec![512, 512, 512, 512, 512]] {
        let handle = client.call(Request::Create {
            widths: widths.clone(),
            activation: "tanh".into(),
            seed: 42,
            optimizer: "adam".into(),
            memory: 32 * 1024 * 1024,
        })?["handle"]
            .as_u64()
            .unwrap();
        let input = vec![0.1; widths[0]];
        assert!(input.iter().all(|v: &f32| v.is_finite()));
        let mut buffer = wgtime::buffer::WritableTensor::new(input.len())?;
        buffer.write(&input)?;
        let sealed = buffer.commit()?;
        let tensor = client.register_tensor(&sealed, vec![input.len()])?;
        let mut direct = Trainer::new(
            mlp(&widths, "tanh", 42)?,
            std::path::Path::new("."),
            32 * 1024 * 1024,
            Optimizer::Adam,
        )?;
        let mut output = Vec::new();
        for _ in 0..100 {
            direct.session.run(&input).map_err(anyhow::Error::msg)?;
            client.predict_into(handle, tensor, 0, input.len(), &mut output)?;
        }
        client.call(Request::Stats)?;
        let mut blocks = Vec::new();
        for block in 0..100 {
            let mut core = Vec::new();
            let mut ipc = Vec::new();
            for lane in 0..2 {
                if (lane == 0) == (block % 2 == 0) {
                    for _ in 0..20 {
                        let t = Instant::now();
                        let values = direct.session.run(&input).map_err(anyhow::Error::msg)?;
                        anyhow::ensure!(values.iter().all(|v| v.is_finite()), "nonfinite");
                        core.push(t.elapsed().as_nanos());
                    }
                } else {
                    for _ in 0..20 {
                        let t = Instant::now();
                        client.predict_into(handle, tensor, 0, input.len(), &mut output)?;
                        ipc.push(t.elapsed().as_nanos());
                    }
                }
            }
            assert_eq!(
                output,
                direct.session.run(&input).map_err(anyhow::Error::msg)?
            );
            blocks.push(serde_json::json!({"block":block,"order":if block%2==0{"AB"}else{"BA"},"core_ns":core,"ipc_ns":ipc}));
        }
        let compute = client.call(Request::Stats)?;
        #[allow(unused_mut)]
        let mut echo = Vec::<u128>::new();
        #[cfg(not(feature = "latency-probe"))]
        assert!(client.predict_into(handle, tensor, 0, input.len() | (1usize << 63), &mut output).is_err());
        #[cfg(feature = "latency-probe")]
        for _ in 0..1000 {
            let t = Instant::now();
            client.predict_into(handle, tensor, 0, input.len() | (1usize << 63), &mut output)?;
            echo.push(t.elapsed().as_nanos());
            assert!(output.is_empty());
        }
        let echo_stats = client.call(Request::Stats)?;
        assert!(client
            .predict_into(
                handle,
                tensor,
                usize::MAX,
                input.len() | (1usize << 63),
                &mut output
            )
            .is_err());
        assert!(client
            .predict_into(
                handle + 999,
                tensor,
                0,
                input.len() | (1usize << 63),
                &mut output
            )
            .is_err());
        println!(
            "wgtime-segments {}",
            serde_json::json!({"widths":widths,"blocks":blocks,"service_compute":compute,"echo_rtt_ns":echo,"service_echo":echo_stats,"input_finite":"prevalidated once both paths; sealed registration validates service input","core_timing":"Session::run plus output finite scan; no allocation/clone; excludes IPC encode and receive","budget":"same Session config; inference has no per-call compute budget in either path"})
        );
    }
    Ok(())
}
