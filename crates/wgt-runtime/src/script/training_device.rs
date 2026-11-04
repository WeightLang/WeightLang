//! Explicit device selection shared by interpreted and compiled WGT calls.
use super::*;
use crate::training::TrainingBackend;

impl Runtime {
    pub(super) fn training_device_builtin(&mut self, name: &str, args: &[Value]) -> Result<Value> {
        if matches!(name, "wgsl_kernel" | "wgsl_run") {
            ensure!(args.len() == 4, "{name} expects four arguments");
            self.resources.allow_direct_gpu_training()?;
            #[cfg(not(feature = "gpu"))]
            bail!("GPU backend not compiled; build with --features gpu");
            #[cfg(feature = "gpu")]
            {
                if name == "wgsl_kernel" {
                    ensure!(
                        self.kernels.len() < 64,
                        "at most 64 WGSL kernels per runtime"
                    );
                    let kernel = crate::wgsl::Kernel::new(
                        args[0].text()?,
                        args[1].text()?,
                        args[2].text()?,
                        args[3].integer()?,
                    )?;
                    let index = self.kernels.len();
                    self.kernels.push(kernel);
                    return Ok(Value::Kernel(index));
                }
                let Value::Kernel(index) = args[0] else {
                    bail!("expected WGSL kernel handle")
                };
                let dimensions = args[3].list()?;
                ensure!(
                    dimensions.len() == 3,
                    "dispatch requires three workgroup counts"
                );
                let groups = [
                    u32::try_from(dimensions[0].integer()?)
                        .context("workgroup count exceeds u32")?,
                    u32::try_from(dimensions[1].integer()?)
                        .context("workgroup count exceeds u32")?,
                    u32::try_from(dimensions[2].integer()?)
                        .context("workgroup count exceeds u32")?,
                ];
                return Ok(vector(self.kernels[index].run(
                    &args[1].vector()?,
                    args[2].integer()?,
                    groups,
                )?));
            }
        }
        if name == "devices" {
            ensure!(args.is_empty(), "devices expects no arguments");
            return Value::from_json(serde_json::to_value(crate::execution::devices())?);
        }
        let count = if name == "training_device" { 4 } else { 1 };
        ensure!(args.len() == count, "{name} expects {count} arguments");
        let index = args[0].model().context("training device APIs require a sequential model; fn forward GPU training is unsupported")?;
        let mut trainer = self.models[index].lock()?;
        if name == "training_device" {
            let id = args[1].text()?;
            let bytes = args[2].integer()?;
            let batch = args[3].integer()?;
            ensure!(
                bytes > 0 && batch > 0,
                "memory_bytes and max_batch must be positive"
            );
            let backend = if id == "cpu" {
                TrainingBackend::Cpu
            } else {
                self.resources.allow_direct_gpu_training()?;
                ensure!(!id.starts_with("npu:"), "NPU training is unsupported");
                #[cfg(feature = "gpu")]
                {
                    TrainingBackend::Gpu(crate::training::GpuTraining::new(id, batch, bytes)?)
                }
                #[cfg(not(feature = "gpu"))]
                bail!("GPU backend not compiled; build with --features gpu");
            };
            // Construction can fail; preserve the old backend and optimizer until it succeeds.
            trainer.set_backend(backend)?;
        }
        Value::from_json(serde_json::json!({
            "device": trainer.backend().device_id(),
            "steps": trainer.steps(),
            "inference_device": "cpu",
            "state": "host_committed",
        }))
    }
}
