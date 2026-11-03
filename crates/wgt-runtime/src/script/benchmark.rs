//! Synchronous host-to-host measurements with common warmup and percentile rules.
use super::*;
use std::time::Instant;

impl Runtime {
    pub(super) fn benchmark(&mut self, args: &[Value]) -> Result<Value> {
        ensure!(
            (5..=6).contains(&args.len()),
            "benchmark expects 5 or 6 arguments"
        );
        let index = args[0].model()?;
        let warmup = args[2].integer()?;
        let repeats = args[3].integer()?;
        let label = args[4].text()?;
        let mode = args.get(5).map(Value::text).transpose()?.unwrap_or("cpu");
        ensure!(
            warmup <= 10000 && (1..=100000).contains(&repeats),
            "invalid benchmark iterations"
        );
        ensure!(index < self.models.len(), "invalid model handle");
        let (device, scope, times) = if mode == "train:cpu" {
            ensure!(
                matches!(
                    self.models[index].lock()?.backend(),
                    crate::training::TrainingBackend::Cpu
                ),
                "train:cpu requires a CPU trainer; GPU training benchmarks are not yet supported"
            );
            let workload = args[1].list()?;
            ensure!(
                workload.len() == 4,
                "training workload must be [inputs, targets, rate, loss]"
            );
            let inputs = workload[0].matrix()?;
            let targets = workload[1].matrix()?;
            let rate = workload[2].float()?;
            ensure!(
                workload[3].text()? == "mse",
                "training benchmark currently supports mse only"
            );
            let times = measure(self, warmup, repeats, |rt| {
                std::hint::black_box(rt.models[index].lock()?.step(
                    &inputs,
                    Objective::Mse(&targets),
                    rate,
                )?);
                Ok(())
            })?;
            ("cpu".to_owned(), "training step: forward, batch-mean MSE, backward and optimizer; warmup and measured steps update the supplied model", times)
        } else {
            let input = args[1].vector()?;
            self.models[index].lock()?.predict(&input)?;
            if mode == "cpu" {
                let times = measure(self, warmup, repeats, |rt| {
                    std::hint::black_box(
                        rt.models[index]
                            .lock()?
                            .session
                            .run(&input)
                            .map_err(anyhow::Error::msg)?,
                    );
                    Ok(())
                })?;
                (
                    "cpu".to_owned(),
                    "single-vector forward; excludes script dispatch, data loading and training",
                    times,
                )
            } else {
                ensure!(
                    !mode.starts_with("train:"),
                    "GPU training benchmarks are not yet supported"
                );
                ensure!(
                    !self.gpu_benchmark_restricted && std::env::var_os("WGT_HOST_POLICY").is_none(),
                    "direct GPU benchmark is unavailable under host or compute budgets"
                );
                #[cfg(feature = "gpu")]
                {
                    let adapter = crate::gpu::adapters()
                        .into_iter()
                        .find(|a| mode == "gpu" || a.info.id == mode)
                        .with_context(|| format!("GPU unavailable: {mode}"))?;
                    let mut gpu = crate::gpu::Gpu::prepare(
                        &adapter,
                        &mut self.models[index].lock()?.session,
                        256 * 1024 * 1024,
                    )?;
                    // run waits for its submission and maps the result. This untimed run drains
                    // preparation work even with warmup=0; every timed run ends synchronized.
                    gpu.run(&mut self.models[index].lock()?.session, &input)?;
                    let times = measure(self, warmup, repeats, |rt| {
                        std::hint::black_box(
                            gpu.run(&mut rt.models[index].lock()?.session, &input)?,
                        );
                        Ok(())
                    })?;
                    (adapter.info.id, "synchronized host-to-host single-vector GPU forward: uploads, dispatch, wait and readback; excludes device preparation and script dispatch", times)
                }
                #[cfg(not(feature = "gpu"))]
                bail!("GPU support not compiled; enable gpu feature");
            }
        };
        let report = serde_json::json!({
            "device": device, "mode": mode, "unit": "us", "warmup": warmup, "repeats": repeats,
            "mean": times.iter().sum::<f64>() / repeats as f64,
            "p50": times[(repeats-1)/2], "p95": times[(repeats-1)*95/100], "scope": scope
        });
        self.metrics.insert(label.into(), report.clone());
        Value::from_json(report)
    }
}

fn measure(
    runtime: &mut Runtime,
    warmup: usize,
    repeats: usize,
    mut run: impl FnMut(&mut Runtime) -> Result<()>,
) -> Result<Vec<f64>> {
    for _ in 0..warmup {
        runtime.tick()?;
        run(runtime)?;
    }
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        runtime.tick()?;
        let start = Instant::now();
        run(runtime)?;
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    Ok(times)
}
