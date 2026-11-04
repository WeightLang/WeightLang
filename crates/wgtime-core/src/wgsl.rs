//! Explicit native WGSL compute. Kernels are inference-only host calls, not AD nodes.
use anyhow::{ensure, Context, Result};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::time::Duration;

pub struct Kernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    failed: Arc<AtomicBool>,
    budget: usize,
}

impl Kernel {
    /// Compile source without dispatch. ABI: group 0 bindings 0/1 are respectively
    /// read-only input and read-write output storage arrays of f32. No uniforms.
    pub fn new(source: &str, entry: &str, device_id: &str, memory_bytes: usize) -> Result<Self> {
        ensure!(
            !source.is_empty() && source.len() <= 1_048_576,
            "WGSL source must be 1..1048576 bytes"
        );
        ensure!(
            !entry.is_empty() && memory_bytes > 0,
            "entry and positive memory budget required"
        );
        let adapter = crate::gpu::adapters()
            .into_iter()
            .find(|a| a.info.id == device_id)
            .context("WGSL GPU device unavailable; no CPU or NPU fallback")?;
        let (device, queue) =
            pollster::block_on(adapter.handle.request_device(&wgpu::DeviceDescriptor {
                label: Some("WeightLang WGSL kernel"),
                required_limits: adapter.handle.limits(),
                ..Default::default()
            }))?;
        let failed = Arc::new(AtomicBool::new(false));
        let lost = failed.clone();
        device.set_device_lost_callback(move |_, _| {
            lost.store(true, Ordering::SeqCst);
        });
        let uncaptured = failed.clone();
        device.on_uncaptured_error(Arc::new(move |_| {
            uncaptured.store(true, Ordering::SeqCst);
        }));
        let (layout, pipeline) = checked(&device, &failed, || {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("f32 input/output"),
                entries: &[binding(0, true), binding(1, false)],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("user WGSL"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok((layout, pipeline))
        })?;
        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            failed,
            budget: memory_bytes,
        })
    }

    /// Run one explicit dispatch; output is zero-initialized. Budget covers managed
    /// host numeric copies plus device/staging buffers, not driver/compiler memory.
    pub fn run(&mut self, input: &[f32], output_len: usize, groups: [u32; 3]) -> Result<Vec<f32>> {
        ensure!(
            !input.is_empty() && output_len > 0,
            "WGSL input/output must be nonempty"
        );
        ensure!(input.iter().all(|x| x.is_finite()), "nonfinite WGSL input");
        let input_bytes = input.len().checked_mul(4).context("WGSL size overflow")?;
        let output_bytes = output_len.checked_mul(4).context("WGSL size overflow")?;
        let managed = input_bytes
            .checked_mul(3)
            .and_then(|n| output_bytes.checked_mul(3).and_then(|m| n.checked_add(m)))
            .context("WGSL size overflow")?;
        ensure!(managed <= self.budget, "WGSL memory budget exceeded");
        let limits = self.device.limits();
        ensure!(
            groups
                .iter()
                .all(|&n| n > 0 && n <= limits.max_compute_workgroups_per_dimension),
            "invalid WGSL workgroup counts"
        );
        ensure!(
            [input_bytes, output_bytes]
                .iter()
                .all(|&n| n as u64 <= limits.max_buffer_size
                    && n as u64 <= limits.max_storage_buffer_binding_size),
            "WGSL buffer exceeds device limits"
        );
        checked(&self.device, &self.failed, || {
            let buffer = |size, usage| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let input_buffer = buffer(
                input_bytes as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let output = buffer(
                output_bytes as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let readback = buffer(
                output_bytes as u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );
            self.queue
                .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
            let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes as u64);
            let submission = self.queue.submit([encoder.finish()]);
            let (send, receive) = mpsc::sync_channel(1);
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
            let result = (|| {
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(30)),
                })?;
                receive.recv_timeout(Duration::from_secs(30))??;
                let view = readback.get_mapped_range(..)?;
                let values: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite WGSL output"
                );
                Ok(values)
            })();
            readback.unmap();
            result
        })
    }
}

fn binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn checked<T>(
    device: &wgpu::Device,
    failed: &AtomicBool,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    ensure!(!failed.load(Ordering::SeqCst), "WGSL device lost or failed");
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = run();
    if let Some(error) = [
        pollster::block_on(validation.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(oom.pop()),
    ]
    .into_iter()
    .flatten()
    .next()
    {
        anyhow::bail!("WGSL validation/device error: {error}");
    }
    ensure!(!failed.load(Ordering::SeqCst), "WGSL device lost or failed");
    result
}
