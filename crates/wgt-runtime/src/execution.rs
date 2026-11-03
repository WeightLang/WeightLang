//! Device discovery and measured selection for a fixed model and host-vector workload.
use crate::{Model, Report, Session};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub kind: String,
    pub memory_model: String,
}
fn cpu_info() -> DeviceInfo {
    DeviceInfo {
        id: "cpu".into(),
        name: "Rust CPU".into(),
        backend: "cpu".into(),
        kind: "Cpu".into(),
        memory_model: "host".into(),
    }
}
/// Enumerate hardware candidates. A listed adapter still needs model-specific validation.
pub fn devices() -> Vec<DeviceInfo> {
    let list = vec![cpu_info()];
    #[cfg(feature = "gpu")]
    let list = {
        let mut list = list;
        list.extend(crate::gpu::adapters().into_iter().map(|a| a.info));
        list
    };
    list
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPolicy {
    Flexible,
    Strict,
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub device: DeviceInfo,
    pub median_ns: Option<u128>,
    pub rejected: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionReport {
    #[serde(flatten)]
    pub memory: Report,
    pub host_managed_bytes: usize,
    pub gpu_buffer_bytes: usize,
    pub zero_heap_allocation: bool,
    pub selection: String,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug)]
pub struct ExecutionSession {
    cpu: Session,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::Gpu>,
    report: ExecutionReport,
}

impl ExecutionSession {
    /// `device`: auto (measured CPU/GPU), cpu, gpu (best GPU), or backend:index.
    /// Preparation benchmarks synthetic inputs; no user data or training is accessed.
    pub fn prepare(
        model: Model,
        root: &Path,
        budget: usize,
        device: &str,
        allocation: AllocationPolicy,
    ) -> Result<Self> {
        let explicit_gpu = device == "gpu"
            || device.split_once(':').is_some_and(|(b, i)| {
                matches!(b, "vulkan" | "metal" | "dx12")
                    && !i.is_empty()
                    && i.bytes().all(|b| b.is_ascii_digit())
            });
        ensure!(
            matches!(device, "cpu" | "auto") || explicit_gpu,
            "invalid native device: {device}"
        );
        ensure!(
            allocation != AllocationPolicy::Strict || !explicit_gpu,
            "strict zero-allocation mode currently supports CPU only"
        );
        let cpu = Session::prepare(model, root, budget)?;
        let report = ExecutionReport {
            memory: cpu.report.clone(),
            host_managed_bytes: cpu.report.managed_bytes,
            gpu_buffer_bytes: 0,
            zero_heap_allocation: true,
            selection: if allocation == AllocationPolicy::Strict {
                "strict allocation policy"
            } else {
                "explicit CPU"
            }
            .into(),
            candidates: vec![Candidate {
                device: cpu_info(),
                median_ns: None,
                rejected: None,
            }],
        };
        let mut session = Self {
            cpu,
            #[cfg(feature = "gpu")]
            gpu: None,
            report,
        };
        if device == "cpu" || allocation == AllocationPolicy::Strict {
            return Ok(session);
        }
        session.select(device, budget)?;
        Ok(session)
    }

    fn select(&mut self, requested: &str, budget: usize) -> Result<()> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = budget;
            ensure!(
                requested == "auto",
                "GPU support not compiled; enable the gpu feature"
            );
        }
        let probes: Vec<Vec<f32>> = (0..2)
            .map(|phase| {
                (0..self.cpu.report.input)
                    .map(|i| {
                        (((i % 29 + phase * 7) % 29) as f32 - 14.)
                            / (14. * (self.cpu.report.input as f32).sqrt())
                    })
                    .collect()
            })
            .collect();
        let expected: Vec<Vec<f32>> = probes
            .iter()
            .map(|x| {
                self.cpu
                    .run(x)
                    .map(|y| y.to_vec())
                    .map_err(anyhow::Error::msg)
            })
            .collect::<Result<_>>()?;
        ensure!(
            expected.iter().flatten().all(|x| x.is_finite()),
            "automatic calibration produced nonfinite CPU outputs"
        );
        let cpu_ns = measure(|| {
            std::hint::black_box(self.cpu.run(&probes[0]).map_err(anyhow::Error::msg)?);
            Ok(())
        })?;
        self.report.candidates[0].median_ns = Some(cpu_ns);
        if requested != "auto" {
            self.report.candidates[0].rejected = Some("explicit GPU requested".into());
        }
        #[cfg(feature = "gpu")]
        {
            let adapters = crate::gpu::adapters();
            if requested != "auto" && requested != "gpu" {
                ensure!(
                    adapters.iter().any(|a| a.info.id == requested),
                    "requested device unavailable: {requested}"
                );
            }
            for adapter in &adapters {
                if requested != "auto" && requested != "gpu" && requested != adapter.info.id {
                    continue;
                }
                // Only one candidate GPU is alive at a time. Do not exceed the
                // budget by holding the best-so-far device while probing another.
                let result = (|| -> Result<u128> {
                    let mut gpu = crate::gpu::Gpu::prepare(adapter, &mut self.cpu, budget)?;
                    for (input, reference) in probes.iter().zip(&expected) {
                        let actual = gpu.run(&mut self.cpu, input)?;
                        ensure!(
                            compatible(actual, reference),
                            "GPU result differs from CPU reference (atol=1e-4, rtol=1e-3)"
                        );
                    }
                    measure(|| {
                        std::hint::black_box(gpu.run(&mut self.cpu, &probes[0])?);
                        Ok(())
                    })
                })();
                self.report.candidates.push(Candidate {
                    device: adapter.info.clone(),
                    median_ns: result.as_ref().ok().copied(),
                    rejected: result.err().map(|e| format!("{e:#}")),
                });
            }
            loop {
                let winner = fastest(&self.report.candidates).ok_or_else(|| {
                    anyhow::anyhow!(
                        "no compatible device within budget: {:?}",
                        self.report.candidates
                    )
                })?;
                let id = self.report.candidates[winner].device.id.clone();
                if id == "cpu" {
                    break;
                }
                let adapter = adapters
                    .iter()
                    .find(|a| a.info.id == id)
                    .ok_or_else(|| anyhow::anyhow!("selected GPU disappeared"))?;
                match crate::gpu::Gpu::prepare(adapter, &mut self.cpu, budget) {
                    Ok(gpu) => {
                        self.report.gpu_buffer_bytes = gpu.bytes;
                        self.report.memory.managed_bytes += gpu.bytes;
                        self.report.memory.device = id;
                        self.report.zero_heap_allocation = false;
                        self.gpu = Some(gpu);
                        break;
                    }
                    Err(error) => {
                        self.report.candidates[winner].rejected =
                            Some(format!("final preparation failed: {error:#}"));
                    }
                }
            }
        }
        self.report.selection = "lowest measured median of 5 synchronized host-to-host runs after 2 warmups; synthetic fixed-shape inputs; not a global optimum guarantee".into();
        Ok(())
    }

    pub fn report(&self) -> &ExecutionReport {
        &self.report
    }

    pub fn parameter_info(&self) -> Vec<crate::ParameterInfo<'_>> {
        self.cpu.parameter_info()
    }

    pub fn revision(&self) -> u64 {
        self.cpu.revision()
    }

    pub fn read_parameter(&mut self, name: &str, start: usize, out: &mut [f32]) -> Result<()> {
        self.cpu.read_parameter(name, start, out)
    }

    pub fn write_parameters(&mut self, writes: &[crate::ParameterWrite<'_>]) -> Result<()> {
        self.cpu.write_parameters(writes)
    }

    pub fn run(&mut self, input: &[f32]) -> Result<&[f32]> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            return gpu.run(&mut self.cpu, input);
        }
        self.cpu.run(input).map_err(anyhow::Error::msg)
    }
}

#[cfg(any(feature = "gpu", test))]
fn compatible(actual: &[f32], reference: &[f32]) -> bool {
    actual.len() == reference.len()
        && actual
            .iter()
            .zip(reference)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4 + 1e-3 * b.abs())
}
#[cfg(any(feature = "gpu", test))]
fn fastest(candidates: &[Candidate]) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.rejected.is_none() && c.median_ns.is_some())
        .min_by_key(|(_, c)| c.median_ns)
        .map(|(i, _)| i)
}
fn measure(mut run: impl FnMut() -> Result<()>) -> Result<u128> {
    for _ in 0..2 {
        run()?;
    }
    let mut elapsed = [0; 5];
    for sample in &mut elapsed {
        let start = Instant::now();
        run()?;
        *sample = start.elapsed().as_nanos();
    }
    elapsed.sort_unstable();
    Ok(elapsed[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_requires_valid_measurement_not_device_rank() {
        let mut candidates = vec![
            Candidate {
                device: cpu_info(),
                median_ns: Some(10),
                rejected: None,
            },
            Candidate {
                device: cpu_info(),
                median_ns: Some(20),
                rejected: None,
            },
        ];
        assert_eq!(fastest(&candidates), Some(0));
        candidates[1].median_ns = Some(5);
        assert_eq!(fastest(&candidates), Some(1));
        candidates[1].rejected = Some("budget".into());
        assert_eq!(fastest(&candidates), Some(0));
        assert!(!compatible(&[f32::NAN], &[0.]));
        assert!(!compatible(&[1.], &[0.]));
        assert!(compatible(&[1.00001], &[1.]));
    }
}
