//! Cross-platform compute with explicit buffer accounting. Driver/command allocations
//! are outside that budget; this backend deliberately does not claim zero allocation.
use crate::{accumulate, bytes, Op, Session, Storage};
use anyhow::{ensure, Context, Result};
use std::{collections::HashMap, sync::mpsc, time::Duration};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub(crate) struct Adapter {
    pub info: crate::execution::DeviceInfo,
    pub handle: wgpu::Adapter,
}

pub(crate) fn adapters() -> Vec<Adapter> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(desc);
    let mut handles = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    handles.sort_by_key(|a| {
        let i = a.get_info();
        (
            format!("{:?}", i.backend),
            i.vendor,
            i.device,
            i.name,
            i.driver,
        )
    });
    let mut indices = HashMap::new();
    handles
        .into_iter()
        .filter_map(|handle| {
            let i = handle.get_info();
            // Software Vulkan adapters are not hardware acceleration candidates.
            if i.device_type == wgpu::DeviceType::Cpu {
                return None;
            }
            let backend = format!("{:?}", i.backend).to_lowercase();
            let index = indices.entry(backend.clone()).or_insert(0);
            let id = format!("{backend}:{index}");
            *index += 1;
            Some(Adapter {
                info: crate::execution::DeviceInfo {
                    id,
                    name: i.name,
                    backend,
                    kind: format!("{:?}", i.device_type),
                    memory_model: if i.device_type == wgpu::DeviceType::IntegratedGpu {
                        "integrated; shared allocation not assumed".into()
                    } else {
                        "device-local or driver-managed".into()
                    },
                },
                handle,
            })
        })
        .collect()
}

#[derive(Debug)]
struct Step {
    binding: wgpu::BindGroup,
    groups: u32,
    stream: Option<(usize, Option<usize>)>,
}

#[derive(Debug)]
pub(crate) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    activation: [wgpu::Buffer; 2],
    scratch: wgpu::Buffer,
    readback: wgpu::Buffer,
    steps: Vec<Step>,
    weights: HashMap<(usize, Option<usize>), wgpu::Buffer>,
    revision: u64,
    pub bytes: usize,
}

impl Gpu {
    pub fn prepare(adapter: &Adapter, cpu: &mut Session, budget: usize) -> Result<Self> {
        let limits = adapter.handle.limits();
        ensure!(
            limits.max_compute_invocations_per_workgroup >= 64
                && limits.max_compute_workgroup_size_x >= 64,
            "device lacks 64-lane workgroups"
        );
        ensure!(
            limits.max_storage_buffers_per_shader_stage >= 3
                && limits.max_uniform_buffers_per_shader_stage >= 1
                && limits.max_compute_workgroup_storage_size >= 256,
            "device lacks required compute bindings or workgroup memory"
        );
        let mut streamed = Vec::new();
        for p in &cpu.parameters {
            streamed.push(match p.storage {
                Storage::File { .. } => true,
                Storage::Resident(_) | Storage::Packed(_) => false,
                Storage::View { source, .. } => streamed[source],
                Storage::Add { base, delta } => streamed[base] || streamed[delta],
            });
        }
        let mut cached = HashMap::new();
        let mut scratch_len = 1;
        let mut weight_bytes = 0usize;
        let check = |n: usize| -> Result<()> {
            ensure!(
                n as u64 <= limits.max_buffer_size
                    && n as u64 <= limits.max_storage_buffer_binding_size,
                "model buffer exceeds device limits"
            );
            Ok(())
        };
        check(bytes(cpu.left.len())?)?;
        for op in &cpu.ops {
            if let Op::Linear {
                weight,
                bias,
                output,
                ..
            } = *op
            {
                ensure!(
                    output <= limits.max_compute_workgroups_per_dimension as usize,
                    "linear output exceeds GPU dispatch limit"
                );
                let n = match &cpu.parameters[weight].storage {
                    Storage::Packed(p) => {
                        ensure!(
                            !bias.is_some_and(|b| streamed[b]),
                            "GPU packed weights require resident bias"
                        );
                        p.data.len().div_ceil(4)
                            + p.scales.len()
                            + if bias.is_some() { output } else { 0 }
                    }
                    _ => cpu.parameters[weight].len + if bias.is_some() { output } else { 0 },
                };
                ensure!(
                    n <= u32::MAX as usize,
                    "parameter exceeds GPU u32 indexing range"
                );
                check(bytes(n)?)?;
                if streamed[weight] || bias.is_some_and(|b| streamed[b]) {
                    scratch_len = scratch_len.max(n);
                } else if let std::collections::hash_map::Entry::Vacant(entry) =
                    cached.entry((weight, bias))
                {
                    entry.insert(n);
                    weight_bytes = weight_bytes
                        .checked_add(bytes(n)?)
                        .context("GPU size overflow")?;
                }
            }
        }
        ensure!(
            cpu.left.len().div_ceil(64) <= limits.max_compute_workgroups_per_dimension as usize,
            "activation exceeds GPU dispatch limit"
        );
        let gpu_bytes = weight_bytes
            .checked_add(bytes(scratch_len)?)
            .and_then(|n| n.checked_add(cpu.report.activation_bytes))
            .and_then(|n| n.checked_add(cpu.report.output.checked_mul(4)?))
            .and_then(|n| n.checked_add(cpu.ops.len().checked_mul(32)?))
            .context("GPU memory accounting overflow")?;
        ensure!(
            cpu.report
                .managed_bytes
                .checked_add(gpu_bytes)
                .is_some_and(|n| n <= budget),
            "combined host/GPU budget exceeded: host {}, GPU {}, budget {}",
            cpu.report.managed_bytes,
            gpu_bytes,
            budget
        );
        let (device, queue) =
            pollster::block_on(adapter.handle.request_device(&wgpu::DeviceDescriptor {
                label: Some("WeightLang native compute"),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            }))
            .context("requesting GPU device")?;
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = Self::allocate(
            &device,
            &queue,
            cpu,
            &cached,
            &streamed,
            scratch_len,
            gpu_bytes,
        );
        let validation_error = pollster::block_on(validation.pop());
        let oom_error = pollster::block_on(oom.pop());
        ensure!(
            validation_error.is_none(),
            "GPU preparation validation: {:?}",
            validation_error
        );
        ensure!(
            oom_error.is_none(),
            "GPU allocation failed: {:?}",
            oom_error
        );
        result
    }

    fn allocate(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cpu: &mut Session,
        cached: &HashMap<(usize, Option<usize>), usize>,
        streamed: &[bool],
        scratch_len: usize,
        gpu_bytes: usize,
    ) -> Result<Self> {
        let buffer = |label, n, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let activation = [
            buffer("activation A", bytes(cpu.left.len())?, storage),
            buffer("activation B", bytes(cpu.left.len())?, storage),
        ];
        let scratch = buffer("streamed weights", bytes(scratch_len)?, storage);
        let readback = buffer(
            "readback",
            bytes(cpu.report.output)?,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native f32 kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("native vector compute"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let mut weights = HashMap::new();
        for (&(weight, bias), &n) in cached {
            let packed = if matches!(cpu.parameters[weight].storage, Storage::Packed(_)) {
                Some(packed_contents(cpu, weight, bias)?)
            } else {
                materialize(cpu, weight, bias)?;
                None
            };
            let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resident effective weights"),
                contents: packed
                    .as_deref()
                    .unwrap_or_else(|| bytemuck::cast_slice(&cpu.workspace[..n])),
                usage: storage,
            });
            weights.insert((weight, bias), b);
        }
        let mut steps = Vec::new();
        let mut width = cpu.report.input;
        for (index, op) in cpu.ops.iter().enumerate() {
            let (input, output, code, bias_flag, weight, stream) = match *op {
                Op::Linear {
                    weight,
                    bias,
                    input,
                    output,
                } => {
                    let stream = (streamed[weight] || bias.is_some_and(|b| streamed[b]))
                        .then_some((weight, bias));
                    (
                        input,
                        output,
                        0,
                        u32::from(bias.is_some()),
                        weights.get(&(weight, bias)).unwrap_or(&scratch),
                        stream,
                    )
                }
                _ => (
                    width,
                    width,
                    match op {
                        Op::Relu => 1,
                        Op::Tanh => 2,
                        Op::Sigmoid => 3,
                        _ => 4,
                    },
                    0,
                    &scratch,
                    None,
                ),
            };
            let (format, group, stride, words) = match *op {
                Op::Linear { weight, .. } => match &cpu.parameters[weight].storage {
                    Storage::Packed(p) => (
                        p.format,
                        p.group as u32,
                        p.stride as u32,
                        p.data.len().div_ceil(4) as u32,
                    ),
                    _ => (0, 1, input as u32, (input * output) as u32),
                },
                _ => (0, 1, 0, 0),
            };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("operator contract"),
                contents: bytemuck::cast_slice(&[
                    u32::try_from(input)?,
                    u32::try_from(output)?,
                    code,
                    bias_flag,
                    format,
                    group,
                    stride,
                    words,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("operator buffers"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: activation[index % 2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: activation[(index + 1) % 2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            steps.push(Step {
                binding,
                groups: if code == 0 {
                    output as u32
                } else {
                    output.div_ceil(64) as u32
                },
                stream,
            });
            width = output;
        }
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            activation,
            scratch,
            readback,
            steps,
            weights,
            revision: cpu.revision,
            bytes: gpu_bytes,
        })
    }

    pub fn run<'a>(&mut self, cpu: &'a mut Session, input: &[f32]) -> Result<&'a [f32]> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = self.run_inner(cpu, input);
        let validation_error = pollster::block_on(validation.pop());
        let oom_error = pollster::block_on(oom.pop());
        ensure!(
            validation_error.is_none(),
            "GPU execution validation: {:?}",
            validation_error
        );
        ensure!(
            oom_error.is_none(),
            "GPU execution out of memory: {:?}",
            oom_error
        );
        result
    }

    fn run_inner<'a>(&mut self, cpu: &'a mut Session, input: &[f32]) -> Result<&'a [f32]> {
        ensure!(input.len() == cpu.report.input, "input length mismatch");
        if self.revision != cpu.revision {
            for (&(weight, bias), buffer) in &self.weights {
                if matches!(cpu.parameters[weight].storage, Storage::Packed(_)) {
                    self.queue
                        .write_buffer(buffer, 0, &packed_contents(cpu, weight, bias)?);
                } else {
                    let n = materialize(cpu, weight, bias)?;
                    self.queue
                        .write_buffer(buffer, 0, bytemuck::cast_slice(&cpu.workspace[..n]));
                }
            }
            self.revision = cpu.revision;
        }
        self.queue
            .write_buffer(&self.activation[0], 0, bytemuck::cast_slice(input));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for step in &self.steps {
            if let Some((weight, bias)) = step.stream {
                // Queue writes execute before the next submission. Submit all earlier
                // consumers before replacing shared scratch, preserving asynchronous order.
                self.queue.submit([encoder.finish()]);
                encoder = self.device.create_command_encoder(&Default::default());
                let n = materialize(cpu, weight, bias)?;
                self.queue.write_buffer(
                    &self.scratch,
                    0,
                    bytemuck::cast_slice(&cpu.workspace[..n]),
                );
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &step.binding, &[]);
            pass.dispatch_workgroups(step.groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.activation[self.steps.len() % 2],
            0,
            &self.readback,
            0,
            bytes(cpu.report.output)? as u64,
        );
        let submission = self.queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::sync_channel(1);
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })?;
        receive.recv_timeout(Duration::from_secs(30))??;
        {
            let view = self.readback.get_mapped_range(..)?;
            cpu.left[..cpu.report.output].copy_from_slice(bytemuck::cast_slice(&view));
        }
        self.readback.unmap();
        Ok(&cpu.left[..cpu.report.output])
    }
}

fn materialize(cpu: &mut Session, weight: usize, bias: Option<usize>) -> Result<usize> {
    let n = cpu.parameters[weight].len;
    let count = n + bias.map_or(0, |b| cpu.parameters[b].len);
    cpu.workspace[..count].fill(0.);
    accumulate(&mut cpu.parameters, weight, 0, &mut cpu.workspace[..n])
        .map_err(anyhow::Error::msg)?;
    if let Some(bias) = bias {
        accumulate(&mut cpu.parameters, bias, 0, &mut cpu.workspace[n..count])
            .map_err(anyhow::Error::msg)?;
    }
    Ok(count)
}

fn packed_contents(cpu: &mut Session, weight: usize, bias: Option<usize>) -> Result<Vec<u8>> {
    let Storage::Packed(packed) = &cpu.parameters[weight].storage else {
        unreachable!()
    };
    let mut data = packed.data.clone();
    data.resize(data.len().div_ceil(4) * 4, 0);
    data.extend_from_slice(bytemuck::cast_slice(&packed.scales));
    if let Some(bias) = bias {
        let count = cpu.parameters[bias].len;
        cpu.workspace[..count].fill(0.);
        accumulate(&mut cpu.parameters, bias, 0, &mut cpu.workspace[..count])
            .map_err(anyhow::Error::msg)?;
        data.extend_from_slice(bytemuck::cast_slice(&cpu.workspace[..count]));
    }
    Ok(data)
}
