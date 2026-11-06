//! Reverse-mode differentiation of the executed program, using immutable value
//! versions. A write rebinds a parameter to a value node; it never severs an edge.
use super::{AdaptiveSession, Op};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Edge {
    Conv { x:usize,k:usize,b:usize,g:wgt_language::adaptive::ConvGeometry },
    MaxPool2d { x: usize, g: wgt_language::adaptive::Pool2dGeometry },
    AvgPool2d { x: usize, g: wgt_language::adaptive::Pool2dGeometry },
    Matmul { a: usize, b: usize, m: usize, k: usize, n: usize },
    BatchMatmul { a: usize, b: usize, batch: usize, m: usize, k: usize, n: usize },
    LayerNorm { x: usize, gamma: usize, beta: usize, groups: super::tensor::AxisGroups, epsilon: f32 },
    Transpose { x: usize, m: usize, n: usize },
    Permute { x: usize, input_shape: [usize; 8], axes: [usize; 8], rank: usize },
    SoftmaxAxis { x: usize, groups: super::tensor::AxisGroups },
    Scale { x: usize, factor: f32 },

    Minimum {
        a: usize,
        b: usize,
    },
    Clip {
        x: usize,
        low: f32,
        high: f32,
    },
    Sum {
        x: usize,
    },
    Leaf,
    MemoryRead {
        memory: usize,
        address: usize,
    },
    MemoryWrite {
        memory: usize,
        address: usize,
        value: usize,
    },
    Mean {
        x: usize,
    },
    Blend {
        p: usize,
        a: usize,
        b: Option<usize>,
    },
    Linear {
        x: usize,
        w: usize,
        b: Option<usize>,
    },
    Binary {
        a: usize,
        b: usize,
        mul: bool,
    },
    Unary {
        x: usize,
        kind: u8,
    },
    Slice {
        x: usize,
        start: usize,
    },
}

struct Node {
    value: Vec<f32>,
    edge: Edge,
}

pub(super) struct Tape {
    nodes: Vec<Node>,
    gradients: Vec<Vec<f32>>,
    registers: Vec<Option<usize>>,
    parameters: Vec<usize>,
    memories: Vec<Option<usize>>,
    bytes: usize,
    budget: usize,
}

impl Tape {
    fn new(session: &AdaptiveSession, input: &[f32], budget: usize) -> Result<Self> {
        let mut tape = Self {
            nodes: Vec::new(),
            gradients: Vec::new(),
            registers: vec![None; session.registers.len()],
            parameters: Vec::new(),
            memories: vec![None; session.registers.len()],
            bytes: 0,
            budget,
        };
        let node = tape.push(input, Edge::Leaf)?;
        tape.registers[session.input] = Some(node);
        for parameter in &session.program.parameters {
            let node = tape.push(&parameter.data, Edge::Leaf)?;
            tape.parameters.push(node);
        }
        Ok(tape)
    }

    fn push(&mut self, value: &[f32], edge: Edge) -> Result<usize> {
        // Both primals and adjoints are charged before allocation, so backward
        // cannot unexpectedly double the permitted tensor storage.
        let bytes = value
            .len()
            .checked_mul(8)
            .and_then(|n| self.bytes.checked_add(n))
            .context("tape size overflow")?;
        ensure!(
            bytes <= self.budget,
            "adaptive tape memory budget exceeded: need {bytes}, have {}",
            self.budget
        );
        let mut copy = crate::zero_buffer(value.len())?;
        copy.copy_from_slice(value);
        let gradient = crate::zero_buffer(value.len())?;
        let id = self.nodes.len();
        self.nodes.push(Node { value: copy, edge });
        self.gradients.push(gradient);
        self.bytes = bytes;
        Ok(id)
    }

    fn reg(&self, index: usize) -> Result<usize> {
        self.registers[index].context("uninitialized tape register")
    }

    pub(super) fn write(&mut self, parameter: usize, register: usize) -> Result<()> {
        self.parameters[parameter] = self.reg(register)?;
        Ok(())
    }

    pub(super) fn memory_allocate(&mut self, m: usize, value: &[f32]) -> Result<()> {
        self.memories[m] = Some(self.push(value, Edge::Leaf)?);
        Ok(())
    }
    pub(super) fn memory_write(
        &mut self,
        m: usize,
        a: usize,
        v: usize,
        value: &[f32],
    ) -> Result<()> {
        let edge = Edge::MemoryWrite {
            memory: self.memories[m].context("uninitialized tape memory")?,
            address: self.reg(a)?,
            value: self.reg(v)?,
        };
        self.memories[m] = Some(self.push(value, edge)?);
        Ok(())
    }

    pub(super) fn record(&mut self, op: Op, value: &[f32]) -> Result<()> {
        let (out, edge) = match op {
            Op::Conv(r,x,k,b,g) => (r, Edge::Conv {x:self.reg(x)?,k:self.reg(k)?,b:self.reg(b)?,g}),
            Op::MaxPool2d(r,x,g) => (r, Edge::MaxPool2d { x:self.reg(x)?, g }),
            Op::AvgPool2d(r,x,g) => (r, Edge::AvgPool2d { x:self.reg(x)?, g }),
            Op::Reshape(r,x) => (r, Edge::Slice { x:self.reg(x)?, start:0 }),
            Op::Transpose(r,x,m,n) => (r, Edge::Transpose { x:self.reg(x)?,m,n }),
            Op::Permute(r,x,input_shape,axes,rank) => (r, Edge::Permute { x:self.reg(x)?,input_shape,axes,rank }),
            Op::Matmul(r,a,b,m,k,n) => (r, Edge::Matmul { a:self.reg(a)?,b:self.reg(b)?,m,k,n }),
            Op::BatchMatmul(r,a,b,batch,m,k,n) => (r, Edge::BatchMatmul { a:self.reg(a)?,b:self.reg(b)?,batch,m,k,n }),
            Op::LayerNorm(r,x,gamma,beta,groups,epsilon) => (r, Edge::LayerNorm {
                x:self.reg(x)?, gamma:self.reg(gamma)?, beta:self.reg(beta)?, groups, epsilon
            }),
            Op::SoftmaxAxis(r,x,groups) => (r, Edge::SoftmaxAxis { x:self.reg(x)?,groups }),
            Op::Scale(r,x,factor) => (r, Edge::Scale { x:self.reg(x)?,factor }),
            Op::Constant(r, _) | Op::Unary(r, _, 5) => (r, Edge::Leaf),
            Op::Minimum(r, a, b) => (
                r,
                Edge::Minimum {
                    a: self.reg(a)?,
                    b: self.reg(b)?,
                },
            ),
            Op::Clip(r, x, low, high) => (
                r,
                Edge::Clip {
                    x: self.reg(x)?,
                    low,
                    high,
                },
            ),
            Op::Sum(r, x) => (r, Edge::Sum { x: self.reg(x)? }),
            Op::ResourceState(r) => (r, Edge::Leaf),
            Op::MemoryRead(r, m, a) => (
                r,
                Edge::MemoryRead {
                    memory: self.memories[m].context("uninitialized tape memory")?,
                    address: self.reg(a)?,
                },
            ),
            Op::Mean(r, x) => (r, Edge::Mean { x: self.reg(x)? }),
            // Preserve the hard forward primal while passing identity STE to
            // the continuous parameter node (also identity for ordinary reads).
            Op::Read(r, p) => (r, Edge::Slice { x: self.parameters[p], start: 0 }),
            Op::Linear(r, x, w, b) => (
                r,
                Edge::Linear {
                    x: self.reg(x)?,
                    w: self.parameters[w],
                    b: b.map(|b| self.parameters[b]),
                },
            ),
            Op::Binary(r, a, b, mul) => (
                r,
                Edge::Binary {
                    a: self.reg(a)?,
                    b: self.reg(b)?,
                    mul,
                },
            ),
            Op::Unary(r, x, kind) => (
                r,
                Edge::Unary {
                    x: self.reg(x)?,
                    kind,
                },
            ),
            Op::Blend(r, p, a, b) => (
                r,
                Edge::Blend {
                    p: self.reg(p)?,
                    a: self.reg(a)?,
                    b: b.map(|b| self.reg(b)).transpose()?,
                },
            ),
            Op::Slice(r, x, start) => (
                r,
                Edge::Slice {
                    x: self.reg(x)?,
                    start,
                },
            ),
            _ => anyhow::bail!("instruction does not produce a tape value"),
        };
        self.registers[out] = Some(self.push(value, edge)?);
        Ok(())
    }
}

/// An immutable forward snapshot with reusable, first-order vector-Jacobian
/// products. It does not borrow or mutate the originating session.
pub struct ForwardPass {
    tape: Tape,
    output: usize,
    initial: BTreeMap<String, usize>,
    final_state: BTreeMap<String, usize>,
    steps: u64,
    differentiated: bool,
}

impl AdaptiveSession {
    /// Record a forward without committing writes. The additional tape budget
    /// includes all saved f32 values AND their adjoints, but excludes metadata.
    pub fn forward(
        &mut self,
        input: &[f32],
        max_steps: u64,
        tape_budget_bytes: usize,
    ) -> Result<ForwardPass> {
        self.forward_with_parameters(input, &BTreeMap::new(), max_steps, tape_budget_bytes)
    }

    /// Explicit initial-state/parameter bindings apply only to this evaluation,
    /// including read-only-in-program (`frozen`) controller parameters.
    pub fn forward_with_parameters(
        &mut self,
        input: &[f32],
        bindings: &BTreeMap<String, Vec<f32>>,
        max_steps: u64,
        tape_budget_bytes: usize,
    ) -> Result<ForwardPass> {
        for (name, value) in bindings {
            let p = self
                .program
                .parameters
                .iter()
                .find(|p| &p.name == name)
                .context("unknown parameter binding")?;
            ensure!(
                value.len() == p.data.len() && value.iter().all(|v| v.is_finite()),
                "invalid parameter binding: {name}"
            );
        }
        self.begin(input, max_steps)?;
        let result = (|| {
            for p in &mut self.program.parameters {
                if let Some(value) = bindings.get(&p.name) {
                    p.data.copy_from_slice(value);
                }
            }
            let mut tape = Tape::new(self, input, tape_budget_bytes)?;
            let initial = self
                .program
                .parameters
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.clone(), tape.parameters[i]))
                .collect();
            let (r, _) = self.execute(max_steps, Some(&mut tape))?;
            let output = tape.reg(r)?;
            let final_state = self
                .program
                .parameters
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.clone(), tape.parameters[i]))
                .collect();
            Ok(ForwardPass {
                tape,
                output,
                initial,
                final_state,
                steps: self.steps,
                differentiated: false,
            })
        })();
        self.restore();
        result
    }
}

impl ForwardPass {
    pub fn output(&self) -> &[f32] {
        &self.tape.nodes[self.output].value
    }
    pub fn state(&self, name: &str) -> Result<&[f32]> {
        let id = *self
            .final_state
            .get(name)
            .context("unknown state parameter")?;
        Ok(&self.tape.nodes[id].value)
    }
    pub fn parameter_names(&self) -> impl Iterator<Item = &str> {
        self.initial.keys().map(String::as_str)
    }
    pub fn tape_bytes(&self) -> usize {
        self.tape.bytes
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn input_gradient(&self) -> Result<&[f32]> {
        ensure!(self.differentiated, "backward has not succeeded");
        Ok(&self.tape.gradients[0])
    }
    pub fn parameter_gradient(&self, name: &str) -> Result<&[f32]> {
        ensure!(self.differentiated, "backward has not succeeded");
        let id = *self.initial.get(name).context("unknown parameter")?;
        Ok(&self.tape.gradients[id])
    }

    /// Differentiate output and optional final-state cotangents with respect to
    /// the input and every INITIAL parameter. Missing state seeds mean zero.
    /// Frozen forbids forward writes, not mathematical differentiation. Discrete
    /// branches use the executed path; comparison/iteration count has no gradient.
    pub fn backward(
        &mut self,
        output_gradient: &[f32],
        state_gradients: &BTreeMap<String, Vec<f32>>,
    ) -> Result<()> {
        self.differentiated = false;
        ensure!(
            output_gradient.len() == self.output().len()
                && output_gradient.iter().all(|g| g.is_finite()),
            "invalid output gradient"
        );
        for (name, value) in state_gradients {
            let id = *self
                .final_state
                .get(name)
                .context("unknown state gradient")?;
            ensure!(
                value.len() == self.tape.nodes[id].value.len()
                    && value.iter().all(|v| v.is_finite()),
                "invalid state gradient: {name}"
            );
        }
        for gradient in &mut self.tape.gradients {
            gradient.fill(0.);
        }
        self.tape.gradients[self.output].copy_from_slice(output_gradient);
        for (name, seed) in state_gradients {
            let id = self.final_state[name];
            for (dst, src) in self.tape.gradients[id].iter_mut().zip(seed) {
                *dst += src;
            }
        }
        for index in (0..self.tape.nodes.len()).rev() {
            let (before, rest) = self.tape.nodes.split_at(index);
            let node = &rest[0];
            let (gradients, rest_gradients) = self.tape.gradients.split_at_mut(index);
            let gradient = &rest_gradients[0];
            // Dead paths must not evaluate 0 * overflow in a derivative formula.
            if gradient.iter().all(|g| *g == 0.) {
                continue;
            }
            match node.edge {
                Edge::Leaf => {}
                Edge::MemoryRead { memory, address } => {
                    let width = node.value.len();
                    for i in 0..before[address].value.len() {
                        for (j, g) in gradient.iter().enumerate() {
                            let index = i * width + j;
                            let a = before[address].value[i];
                            let m = before[memory].value[index];
                            gradients[address][i] += g * m;
                            gradients[memory][index] += g * a;
                        }
                    }
                }
                Edge::MemoryWrite {
                    memory,
                    address,
                    value,
                } => {
                    let width = before[value].value.len();
                    for (index, g) in gradient.iter().enumerate() {
                        let i = index / width;
                        let j = index % width;
                        let a = before[address].value[i];
                        let m = before[memory].value[index];
                        let v = before[value].value[j];
                        gradients[memory][index] += g * (1. - a);
                        gradients[address][i] += g * (v - m);
                        gradients[value][j] += g * a;
                    }
                }
                Edge::Mean { x } => {
                    let g = gradient[0] / before[x].value.len() as f32;
                    for v in &mut gradients[x] {
                        *v += g;
                    }
                }
                Edge::Linear { x, w, b } => {
                    let width = before[x].value.len();
                    for (row, &g) in gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        for col in 0..width {
                            let xv = before[x].value[col];
                            let wv = before[w].value[row * width + col];
                            gradients[x][col] += g * wv;
                            gradients[w][row * width + col] += g * xv;
                        }
                        if let Some(b) = b {
                            gradients[b][row] += g;
                        }
                    }
                }
                Edge::Binary { a, b, mul } => {
                    for (i, &g) in gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        let (da, db) = if mul {
                            (before[b].value[i], before[a].value[i])
                        } else {
                            (1., 1.)
                        };
                        gradients[a][i] += g * da;
                        gradients[b][i] += g * db;
                    }
                }
                Edge::Conv { x,k,b,g } => {
                    for (i,&dy) in gradient.iter().enumerate() {
                        gradients[b][i/g.output_volume()]+=dy;
                        g.pairs(i,|xi,ki| {
                            let xv=before[x].value[xi];let kv=before[k].value[ki];
                            gradients[x][xi]+=dy*kv;
                            gradients[k][ki]+=dy*xv;
                        });
                    }
                },
                Edge::MaxPool2d { x, g } => {
                    for (i, &dy) in gradient.iter().enumerate() {
                        let mut maximum = f32::NEG_INFINITY;
                        let mut selected = None;
                        g.visit(i, |index| {
                            let value = before[x].value[index];
                            if value > maximum {
                                maximum = value;
                                selected = Some(index);
                            }
                        });
                        if let Some(index) = selected {
                            gradients[x][index] += dy;
                        }
                    }
                },
                Edge::AvgPool2d { x, g } => {
                    let share = 1.0 / g.window_len() as f32;
                    for (i, &dy) in gradient.iter().enumerate() {
                        g.visit(i, |index| gradients[x][index] += dy * share);
                    }
                },
                Edge::Matmul { a,b,m,k,n } => {
                    for i in 0..m { for j in 0..n { for q in 0..k {
                        let g = gradient[i*n+j];
                        gradients[a][i*k+q] += g * before[b].value[q*n+j];
                        gradients[b][q*n+j] += g * before[a].value[i*k+q];
                    } } }
                },
                Edge::BatchMatmul { a,b,batch,m,k,n } => {
                    for batch_index in 0..batch {
                        let a_base = batch_index * m * k;
                        let b_base = batch_index * k * n;
                        let out_base = batch_index * m * n;
                        for i in 0..m { for j in 0..n { for q in 0..k {
                            let g = gradient[out_base+i*n+j];
                            gradients[a][a_base+i*k+q] += g * before[b].value[b_base+q*n+j];
                            gradients[b][b_base+q*n+j] += g * before[a].value[a_base+i*k+q];
                        } } }
                    }
                },
                Edge::Transpose { x,m,n } => {
                    for i in 0..m { for j in 0..n { gradients[x][i*n+j] += gradient[j*m+i]; } }
                },
                Edge::Permute { x,input_shape,axes,rank } => {
                    let mut output_shape = [1usize; 8];
                    for axis in 0..rank { output_shape[axis] = input_shape[axes[axis]]; }
                    for output_index in 0..gradient.len() {
                        let mut remainder = output_index;
                        let mut output_coords = [0usize; 8];
                        let mut input_coords = [0usize; 8];
                        for axis in (0..rank).rev() {
                            output_coords[axis] = remainder % output_shape[axis];
                            remainder /= output_shape[axis];
                        }
                        for axis in 0..rank { input_coords[axes[axis]] = output_coords[axis]; }
                        let mut input_index = 0usize;
                        for axis in 0..rank { input_index = input_index * input_shape[axis] + input_coords[axis]; }
                        gradients[x][input_index] += gradient[output_index];
                    }
                },
                Edge::LayerNorm { x,gamma,beta,groups,epsilon } => {
                    groups.layer_norm_vjp(&before[x].value, &before[gamma].value, gradient, epsilon,
                        |index, axis, dx, dgamma, dbeta| {
                            gradients[x][index] += dx;
                            gradients[gamma][axis] += dgamma;
                            gradients[beta][axis] += dbeta;
                        })?;
                },
                Edge::SoftmaxAxis { x,groups } => groups.softmax_vjp(&node.value,&gradient,&mut gradients[x]),
                Edge::Scale { x,factor } => {
                    for (i,g) in gradient.iter().enumerate() { gradients[x][i] += g * factor; }
                },
                Edge::Minimum { a, b } => {
                    for (i, &g) in gradient.iter().enumerate() {
                        let (av, bv) = (before[a].value[i], before[b].value[i]);
                        let da = if av < bv {
                            1.
                        } else if av > bv {
                            0.
                        } else {
                            0.5
                        };
                        gradients[a][i] += g * da;
                        gradients[b][i] += g * (1. - da);
                    }
                }
                Edge::Clip { x, low, high } => {
                    for (i, &g) in gradient.iter().enumerate() {
                        if before[x].value[i] > low && before[x].value[i] < high {
                            gradients[x][i] += g;
                        }
                    }
                }
                Edge::Sum { x } => {
                    for g in &mut gradients[x] {
                        *g += gradient[0];
                    }
                }
                Edge::Unary { x, kind } => {
                    if kind == 8 {
                        let sum: f32 = gradient.iter().sum();
                        for (i, &g) in gradient.iter().enumerate() {
                            gradients[x][i] += g - node.value[i].exp() * sum;
                        }
                        continue;
                    }
                    if kind == 3 {
                        let dot: f32 = gradient
                            .iter()
                            .zip(&node.value)
                            .map(|(g, y)| g * y)
                            .sum();
                        for (i, &g) in gradient.iter().enumerate() {
                            gradients[x][i] += node.value[i] * (g - dot);
                        }
                        continue;
                    }
                    for (i, &g) in gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        let y = node.value[i];
                        let derivative = match kind {
                            0 => 1. - y * y,
                            1 => y * (1. - y),
                            4 => -1.,
                            9 => 1.,
                            6 => y,
                            7 => 1. / before[x].value[i],
                            _ => {
                                if before[x].value[i] > 0. {
                                    1.
                                } else {
                                    0.
                                }
                            }
                        };
                        gradients[x][i] += g * derivative;
                    }
                }
                Edge::Blend { p, a, b } => {
                    for (i, &g) in gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        let j = if before[p].value.len() == 1 { 0 } else { i };
                        let probability = before[p].value[j];
                        let other = b.map_or(0., |b| before[b].value[i]);
                        gradients[p][j] += g * (before[a].value[i] - other);
                        gradients[a][i] += g * probability;
                        if let Some(b) = b {
                            gradients[b][i] += g * (1. - probability);
                        }
                    }
                }
                Edge::Slice { x, start } => {
                    for (i, &g) in gradient.iter().enumerate() {
                        gradients[x][start + i] += g;
                    }
                }
            }
        }
        ensure!(
            self.tape.gradients.iter().flatten().all(|g| g.is_finite()),
            "nonfinite gradient"
        );
        self.differentiated = true;
        Ok(())
    }
}
