//! Conservative synchronous GPU training. The CPU Session is the committed owner;
//! each transaction uploads its current revision and publishes only checked readback.
use super::{Objective, TrainingError};
use crate::{accumulate, execution::DeviceInfo, Op, Session, Storage};
use anyhow::Result;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

type Updates = Vec<(usize, Vec<f32>)>;

/// Experimental Rust-only MSE/SGD backend. No WGT binding or CPU fallback.
/// Uses bounded per-call device arenas, including a complete batch tape. The
/// committed CPU weights survive device loss; this is not a throughput API.
#[derive(Debug)]
pub struct GpuTraining {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    info: DeviceInfo,
    failed: Arc<AtomicBool>,
    max_batch: usize,
    budget: usize,
}

// Flat uniforms avoid Rust/WGSL struct-layout and padding assumptions.
#[derive(Debug)]
struct Job([u32; 12]);
impl Job {
    fn new(op: u32, count: usize, a: usize, destination: usize) -> Self {
        Self([
            op,
            count as u32,
            0,
            0,
            0,
            a as u32,
            0,
            0,
            destination as u32,
            0,
            0,
            0,
        ])
    }
    fn shape(mut self, input: usize, output: usize, batch: usize) -> Self {
        self.0[2..5].copy_from_slice(&[input as u32, output as u32, batch as u32]);
        self
    }
    fn sources(mut self, b: usize, c: usize, extra: usize) -> Self {
        self.0[6] = b as u32;
        self.0[7] = c as u32;
        self.0[9] = extra as u32;
        self
    }
}

struct Arena {
    data: Vec<f32>,
    jobs: Vec<Job>,
    limit: usize,
    groups: usize,
}
impl Arena {
    fn new(gpu: &GpuTraining, session: &Session) -> Result<Self> {
        // Upper bound on managed numeric payloads: Session plus its two existing
        // optimizer arrays, six arena equivalents (host upload/readback/results,
        // device storage/staging and materialization), and uniform buffers.
        // Caller data, metadata, allocator capacity and driver allocations excluded.
        let params = session
            .parameters
            .iter()
            .try_fold(0usize, |n, p| n.checked_add(p.len))
            .ok_or(TrainingError::OutOfBudget)?;
        let base = params
            .checked_mul(8)
            .and_then(|n| n.checked_add(session.report.managed_bytes))
            .and_then(|n| n.checked_add(8))
            .ok_or(TrainingError::OutOfBudget)?;
        let available = gpu
            .budget
            .checked_sub(base)
            .ok_or(TrainingError::OutOfBudget)?;
        let limits = gpu.device.limits();
        let limit = (available / 24)
            .min((limits.max_storage_buffer_binding_size / 4) as usize)
            .min((limits.max_buffer_size.saturating_sub(4) / 4) as usize)
            .min(u32::MAX as usize);
        Ok(Self {
            data: Vec::new(),
            jobs: Vec::new(),
            limit,
            groups: limits.max_compute_workgroups_per_dimension as usize,
        })
    }
    fn zeros(&mut self, n: usize) -> Result<usize> {
        let start = self.data.len();
        let end = start.checked_add(n).ok_or(TrainingError::OutOfBudget)?;
        if end
            .checked_add(self.jobs.len() * 4)
            .is_none_or(|n| n > self.limit)
        {
            return Err(TrainingError::OutOfBudget.into());
        }
        self.data
            .try_reserve_exact(n)
            .map_err(|_| TrainingError::OutOfBudget)?;
        self.data.resize(end, 0.);
        Ok(start)
    }
    fn values(&mut self, values: &[f32]) -> Result<usize> {
        let start = self.zeros(values.len())?;
        self.data[start..].copy_from_slice(values);
        Ok(start)
    }
    fn job(&mut self, job: Job) -> Result<()> {
        if (job.0[1] as usize).div_ceil(64) > self.groups
            || self
                .data
                .len()
                .checked_add((self.jobs.len() + 1) * 4)
                .is_none_or(|n| n > self.limit)
        {
            return Err(TrainingError::OutOfBudget.into());
        }
        self.jobs.push(job);
        Ok(())
    }
}

fn device_error(e: impl std::fmt::Display) -> anyhow::Error {
    TrainingError::Device(e.to_string()).into()
}
fn activation(op: &Op) -> u32 {
    match op {
        Op::Relu => 1,
        Op::Tanh => 2,
        Op::Sigmoid => 3,
        Op::Identity => 4,
        Op::Linear { .. } => 0,
    }
}
fn count(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or_else(|| TrainingError::OutOfBudget.into())
}

impl GpuTraining {
    /// Prepare an exact hardware ID from execution::devices(), e.g. vulkan:0.
    /// max_batch and budget_bytes are mandatory and nonzero. Budget bounds managed
    /// numeric payloads per call, not process/driver memory. Does not run training.
    pub fn new(device: &str, max_batch: usize, budget_bytes: usize) -> Result<Self> {
        if max_batch == 0 || max_batch > u32::MAX as usize || budget_bytes == 0 {
            return Err(TrainingError::InvalidInput(
                "max_batch and budget must be positive and bounded",
            )
            .into());
        }
        let adapter = crate::gpu::adapters()
            .into_iter()
            .find(|a| a.info.id == device)
            .ok_or_else(|| TrainingError::Unavailable(device.into()))?;
        let limits = adapter.handle.limits();
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_storage_buffers_per_shader_stage < 2
        {
            return Err(TrainingError::Unsupported(
                "64-lane workgroups and two storage bindings required",
            )
            .into());
        }
        let (device, queue) =
            pollster::block_on(adapter.handle.request_device(&wgpu::DeviceDescriptor {
                label: Some("WeightLang experimental training"),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            }))
            .map_err(device_error)?;
        let failed = Arc::new(AtomicBool::new(false));
        let lost = Arc::clone(&failed);
        device.set_device_lost_callback(move |_, _| {
            lost.store(true, Ordering::SeqCst);
        });
        let uncaptured = Arc::clone(&failed);
        device.on_uncaptured_error(Arc::new(move |_| {
            uncaptured.store(true, Ordering::SeqCst);
        }));
        let pipeline = scoped(&device, &failed, || {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("batch MSE and SGD"),
                source: wgpu::ShaderSource::Wgsl(include_str!("train.wgsl").into()),
            });
            Ok(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("training kernels"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            )
        })?;
        Ok(Self {
            device,
            queue,
            pipeline,
            info: adapter.info,
            failed,
            max_batch,
            budget: budget_bytes,
        })
    }

    pub fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(super) fn gradients(
        &mut self,
        session: &mut Session,
        inputs: &[Vec<f32>],
        objective: Objective<'_>,
    ) -> Result<(f32, Vec<Vec<f32>>)> {
        let Objective::Mse(targets) = objective else {
            return Err(TrainingError::Unsupported("GPU supports MSE only").into());
        };
        let batch = inputs.len();
        if batch == 0
            || batch > self.max_batch
            || targets.len() != batch
            || inputs.iter().any(|x| x.len() != session.report.input)
            || targets.iter().any(|x| x.len() != session.report.output)
        {
            return Err(TrainingError::InvalidInput("batch or row shape").into());
        }
        if !inputs
            .iter()
            .chain(targets)
            .flatten()
            .all(|v| v.is_finite())
        {
            return Err(TrainingError::NonFinite("input or target").into());
        }
        let mut arena = Arena::new(self, session)?;
        let mut weights = Vec::new();
        let mut gradients = Vec::new();
        for i in 0..session.parameters.len() {
            let n = session.parameters[i].len;
            let offset = arena.zeros(n)?;
            accumulate(
                &mut session.parameters,
                i,
                0,
                &mut arena.data[offset..offset + n],
            )
            .map_err(device_error)?;
            weights.push(offset);
            gradients.push(arena.zeros(n)?);
        }
        let x = arena.zeros(count(batch, session.report.input)?)?;
        for (dst, row) in arena.data[x..]
            .chunks_exact_mut(session.report.input)
            .zip(inputs)
        {
            dst.copy_from_slice(row);
        }
        let target = arena.zeros(count(batch, session.report.output)?)?;
        for (dst, row) in arena.data[target..]
            .chunks_exact_mut(session.report.output)
            .zip(targets)
        {
            dst.copy_from_slice(row);
        }
        let mut tape = vec![(x, session.report.input)];
        for op in &session.ops {
            let (source, width) = tape[tape.len() - 1];
            let output = match op {
                Op::Linear { output, .. } => *output,
                _ => width,
            };
            let n = count(batch, output)?;
            let destination = arena.zeros(n)?;
            let job = match *op {
                Op::Linear {
                    weight,
                    bias,
                    input,
                    output,
                } => Job::new(0, n, source, destination)
                    .shape(input, output, batch)
                    .sources(
                        weights[weight],
                        bias.map_or(0, |b| weights[b]),
                        usize::from(bias.is_some()),
                    ),
                _ => Job::new(activation(op), n, source, destination),
            };
            arena.job(job)?;
            tape.push((destination, output));
        }
        let (prediction, output) = tape[tape.len() - 1];
        let n = count(batch, output)?;
        let losses = arena.zeros(n)?;
        arena.job(
            Job::new(5, n, prediction, losses)
                .shape(0, output, batch)
                .sources(target, 0, 0),
        )?;
        let loss = arena.zeros(1)?;
        arena.job(
            Job::new(7, 1, losses, loss)
                .shape(0, output, batch)
                .sources(0, 0, n),
        )?;
        let mut upstream = arena.zeros(n)?;
        arena.job(Job::new(6, n, prediction, upstream).sources(target, 0, 0))?;
        for (index, op) in session.ops.iter().enumerate().rev() {
            let (x, input) = tape[index];
            let (y, output) = tape[index + 1];
            let previous = arena.zeros(count(batch, input)?)?;
            match *op {
                Op::Linear { weight, bias, .. } => {
                    arena.job(
                        Job::new(8, count(batch, input)?, upstream, previous)
                            .shape(input, output, batch)
                            .sources(weights[weight], 0, 0),
                    )?;
                    arena.job(
                        Job::new(9, count(input, output)?, upstream, gradients[weight])
                            .shape(input, output, batch)
                            .sources(x, 0, 0),
                    )?;
                    if let Some(bias) = bias {
                        arena.job(
                            Job::new(10, output, upstream, gradients[bias])
                                .shape(input, output, batch),
                        )?;
                    }
                }
                _ => arena.job(
                    Job::new(
                        activation(op) + 10,
                        count(batch, input)?,
                        upstream,
                        previous,
                    )
                    .sources(x, y, 0),
                )?,
            }
            upstream = previous;
        }
        // Mirror CPU reverse declaration folding, including frozen dependency chains.
        for (i, p) in session.parameters.iter().enumerate().rev() {
            if p.frozen {
                arena.job(Job::new(16, p.len, 0, gradients[i]))?;
                continue;
            }
            match p.storage {
                Storage::View { source, start } => {
                    arena.job(Job::new(15, p.len, gradients[i], gradients[source] + start))?
                }
                Storage::Add { base, delta } => {
                    arena.job(Job::new(15, p.len, gradients[i], gradients[base]))?;
                    arena.job(Job::new(15, p.len, gradients[i], gradients[delta]))?;
                }
                _ => {}
            }
        }
        let data = self.execute(&arena)?;
        let gradients = session
            .parameters
            .iter()
            .zip(gradients)
            .map(|(p, start)| data[start..start + p.len].to_vec())
            .collect();
        Ok((data[loss], gradients))
    }

    pub(super) fn stage_sgd(
        &mut self,
        session: &Session,
        gradients: &[Vec<f32>],
        rate: f32,
    ) -> Result<Updates> {
        let mut arena = Arena::new(self, session)?;
        let mut outputs = Vec::new();
        for (i, p) in session.parameters.iter().enumerate() {
            if p.frozen {
                continue;
            }
            if let Storage::Resident(values) = &p.storage {
                let w = arena.values(values)?;
                let g = arena.values(&gradients[i])?;
                let candidate = arena.zeros(p.len)?;
                let mut job = Job::new(17, p.len, w, candidate).sources(g, 0, 0);
                job.0[10] = rate.to_bits();
                arena.job(job)?;
                outputs.push((i, candidate, p.len));
            }
        }
        if outputs.is_empty() {
            // Still check device health before committing a frozen-only step.
            scoped(&self.device, &self.failed, || Ok(()))?;
            return Ok(Vec::new());
        }
        let data = self.execute(&arena)?;
        Ok(outputs
            .into_iter()
            .map(|(i, start, n)| (i, data[start..start + n].to_vec()))
            .collect())
    }

    fn execute(&self, arena: &Arena) -> Result<Vec<f32>> {
        if !arena.data.iter().all(|v| v.is_finite()) {
            return Err(TrainingError::NonFinite("parameter").into());
        }
        scoped(&self.device, &self.failed, || {
            let storage = initialized_buffer(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("transaction tape and parameters"),
                    contents: bytemuck::cast_slice(&arena.data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                },
            )?;
            let status = initialized_buffer(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("arithmetic status"),
                    contents: &[0; 4],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                },
            )?;
            let byte_len = arena.data.len() as u64 * 4;
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("transaction readback"),
                size: byte_len + 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let layout = self.pipeline.get_bind_group_layout(0);
            for job in &arena.jobs {
                let uniform = initialized_buffer(
                    &self.device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("training operation"),
                        contents: bytemuck::cast_slice(&job.0),
                        usage: wgpu::BufferUsages::UNIFORM,
                    },
                )?;
                let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("training arena"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: storage.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: status.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups(job.0[1].div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, byte_len);
            encoder.copy_buffer_to_buffer(&status, 0, &readback, byte_len, 4);
            let submission = self.queue.submit([encoder.finish()]);
            let (send, receive) = mpsc::sync_channel(1);
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
            let result = (|| -> Result<Vec<f32>> {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission),
                        timeout: Some(Duration::from_secs(30)),
                    })
                    .map_err(device_error)?;
                receive
                    .recv_timeout(Duration::from_secs(30))
                    .map_err(device_error)?
                    .map_err(device_error)?;
                let view = readback.get_mapped_range(..).map_err(device_error)?;
                let words: &[u32] = bytemuck::cast_slice(&view);
                if words[arena.data.len()] != 0 {
                    return Err(TrainingError::NonFinite("GPU arithmetic").into());
                }
                let values: Vec<f32> = words[..arena.data.len()]
                    .iter()
                    .map(|v| f32::from_bits(*v))
                    .collect();
                if !values.iter().all(|v| v.is_finite()) {
                    return Err(TrainingError::NonFinite("GPU result").into());
                }
                Ok(values)
            })();
            readback.unmap();
            result
        })
    }
}

// Unlike DeviceExt::create_buffer_init, propagate mapping failure on device loss
// or allocation failure instead of panicking inside the convenience helper.
fn initialized_buffer(
    device: &wgpu::Device,
    desc: &wgpu::util::BufferInitDescriptor<'_>,
) -> Result<wgpu::Buffer> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: desc.label,
        size: desc.contents.len() as u64,
        usage: desc.usage,
        mapped_at_creation: true,
    });
    {
        let mut view = buffer.get_mapped_range_mut(..).map_err(device_error)?;
        view.copy_from_slice(desc.contents);
    }
    buffer.unmap();
    Ok(buffer)
}

// Drain every scope even when the body fails. Device loss poisons this backend,
// while the Trainer's last committed CPU state remains available for export.
fn scoped<T>(
    device: &wgpu::Device,
    failed: &AtomicBool,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if failed.load(Ordering::SeqCst) {
        return Err(TrainingError::Device("device lost or uncaptured error".into()).into());
    }
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = run();
    let errors = [
        pollster::block_on(validation.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(oom.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(device_error(error));
    }
    if failed.load(Ordering::SeqCst) {
        return Err(TrainingError::Device("device lost or uncaptured error".into()).into());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{mlp, Optimizer, Trainer, TrainingBackend};
    use std::path::Path;

    #[test]
    #[ignore = "requires hardware GPU; WGT_REQUIRE_GPU=1 requires availability"]
    fn device_loss_preserves_last_committed_model() {
        let adapters = crate::gpu::adapters();
        let Some(adapter) = adapters
            .iter()
            .find(|a| a.info.name.contains("4060"))
            .or_else(|| adapters.first())
        else {
            assert!(
                std::env::var_os("WGT_REQUIRE_GPU").is_none(),
                "required GPU absent"
            );
            eprintln!("SKIP: no hardware GPU; device-loss rollback was not tested");
            return;
        };
        eprintln!(
            "DEVICE LOSS TEST: {} / {}",
            adapter.info.id, adapter.info.name
        );
        let mut trainer = Trainer::new(
            mlp(&[1, 2, 1], "tanh", 7).unwrap(),
            Path::new("."),
            1_000_000,
            Optimizer::Sgd,
        )
        .unwrap();
        trainer
            .set_backend(TrainingBackend::Gpu(
                GpuTraining::new(&adapter.info.id, 4, 1_000_000).unwrap(),
            ))
            .unwrap();
        let x = [vec![0.2], vec![0.5]];
        let y = [vec![0.3], vec![-0.1]];
        trainer.step(&x, Objective::Mse(&y), 0.01).unwrap();
        let before = serde_json::to_vec(&trainer.snapshot().unwrap()).unwrap();
        let revision = trainer.session.revision;
        let (_, gradients) = trainer.gradients(&x, Objective::Mse(&y)).unwrap();
        let TrainingBackend::Gpu(gpu) = &mut trainer.backend else {
            panic!("GPU selected")
        };
        // Actual wgpu device destruction between backprop and candidate update,
        // not a synthetic error return. The last CPU commit must remain readable.
        gpu.device.destroy();
        let error = gpu
            .stage_sgd(&trainer.session, &gradients, 0.01)
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<TrainingError>(),
                Some(TrainingError::Device(_))
            ),
            "{error:#}"
        );
        assert!(trainer.step(&x, Objective::Mse(&y), 0.01).is_err());
        assert_eq!(
            serde_json::to_vec(&trainer.snapshot().unwrap()).unwrap(),
            before
        );
        assert_eq!(trainer.steps(), 1);
        assert_eq!(trainer.session.revision, revision);
        // Recovery is explicit and continues from the last good commit.
        trainer.set_backend(TrainingBackend::Cpu).unwrap();
        trainer.step(&x, Objective::Mse(&y), 0.01).unwrap();
        assert_eq!(trainer.steps(), 2);
    }
}
