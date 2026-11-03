//! Reverse-mode differentiation of the executed program, using immutable value
//! versions. A write rebinds a parameter to a value node; it never severs an edge.
use super::{AdaptiveSession, Op};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Edge {
    Minimum{a:usize,b:usize},
    Clip{x:usize,low:f32,high:f32},
    Sum{x:usize},
    Leaf,
    Mean { x: usize },
    Blend { p: usize, a: usize, b: Option<usize> },
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
    gradient: Vec<f32>,
    edge: Edge,
}

pub(super) struct Tape {
    nodes: Vec<Node>,
    registers: Vec<Option<usize>>,
    parameters: Vec<usize>,
    bytes: usize,
    budget: usize,
}

impl Tape {
    fn new(session: &AdaptiveSession, input: &[f32], budget: usize) -> Result<Self> {
        let mut tape = Self {
            nodes: Vec::new(),
            registers: vec![None; session.registers.len()],
            parameters: Vec::new(),
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
        self.nodes.push(Node {
            value: copy,
            gradient,
            edge,
        });
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

    pub(super) fn record(&mut self, op: Op, value: &[f32]) -> Result<()> {
        let (out, edge) = match op {
            Op::Constant(r,_) | Op::Unary(r,_,5) => (r,Edge::Leaf),
            Op::Minimum(r,a,b)=>(r,Edge::Minimum{a:self.reg(a)?,b:self.reg(b)?}),
            Op::Clip(r,x,low,high)=>(r,Edge::Clip{x:self.reg(x)?,low,high}),
            Op::Sum(r,x)=>(r,Edge::Sum{x:self.reg(x)?}),
            Op::Mean(r,x) => (r,Edge::Mean{x:self.reg(x)?}),
            Op::Read(r, p) => {
                self.registers[r] = Some(self.parameters[p]);
                return Ok(());
            }
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
            Op::Blend(r, p, a, b) => (r, Edge::Blend { p: self.reg(p)?, a: self.reg(a)?, b: b.map(|b| self.reg(b)).transpose()? }),
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
        Ok(&self.tape.nodes[0].gradient)
    }
    pub fn parameter_gradient(&self, name: &str) -> Result<&[f32]> {
        ensure!(self.differentiated, "backward has not succeeded");
        let id = *self.initial.get(name).context("unknown parameter")?;
        Ok(&self.tape.nodes[id].gradient)
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
        for node in &mut self.tape.nodes {
            node.gradient.fill(0.);
        }
        self.tape.nodes[self.output]
            .gradient
            .copy_from_slice(output_gradient);
        for (name, seed) in state_gradients {
            let id = self.final_state[name];
            for (dst, src) in self.tape.nodes[id].gradient.iter_mut().zip(seed) {
                *dst += src;
            }
        }
        for index in (0..self.tape.nodes.len()).rev() {
            let (before, rest) = self.tape.nodes.split_at_mut(index);
            let node = &rest[0];
            // Dead paths must not evaluate 0 * overflow in a derivative formula.
            if node.gradient.iter().all(|g| *g == 0.) {
                continue;
            }
            match node.edge {
                Edge::Leaf => {}
                Edge::Mean { x } => {
                    let g=node.gradient[0]/before[x].value.len() as f32;
                    for v in &mut before[x].gradient { *v+=g; }
                }
                Edge::Linear { x, w, b } => {
                    let width = before[x].value.len();
                    for (row, &g) in node.gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        for col in 0..width {
                            let xv = before[x].value[col];
                            let wv = before[w].value[row * width + col];
                            before[x].gradient[col] += g * wv;
                            before[w].gradient[row * width + col] += g * xv;
                        }
                        if let Some(b) = b {
                            before[b].gradient[row] += g;
                        }
                    }
                }
                Edge::Binary { a, b, mul } => {
                    for (i, &g) in node.gradient.iter().enumerate() {
                        if g == 0. {
                            continue;
                        }
                        let (da, db) = if mul {
                            (before[b].value[i], before[a].value[i])
                        } else {
                            (1., 1.)
                        };
                        before[a].gradient[i] += g * da;
                        before[b].gradient[i] += g * db;
                    }
                }
                Edge::Minimum{a,b} => {
                    for (i,&g) in node.gradient.iter().enumerate() {
                        let (av,bv)=(before[a].value[i],before[b].value[i]);
                        let da=if av<bv {1.} else if av>bv {0.} else {0.5};
                        before[a].gradient[i]+=g*da;before[b].gradient[i]+=g*(1.-da);
                    }
                }
                Edge::Clip{x,low,high} => {
                    for (i,&g) in node.gradient.iter().enumerate() {if before[x].value[i]>low && before[x].value[i]<high {before[x].gradient[i]+=g;}}
                }
                Edge::Sum{x} => {for g in &mut before[x].gradient {*g+=node.gradient[0];}}
                Edge::Unary { x, kind } => {
                    if kind==8 {
                        let sum:f32=node.gradient.iter().sum();
                        for (i,&g) in node.gradient.iter().enumerate() {before[x].gradient[i]+=g-node.value[i].exp()*sum;}
                        continue;
                    }
                    if kind == 3 {
                        let dot: f32 = node.gradient.iter().zip(&node.value).map(|(g,y)| g*y).sum();
                        for (i, &g) in node.gradient.iter().enumerate() {
                            before[x].gradient[i] += node.value[i] * (g-dot);
                        }
                        continue;
                    }
                    for (i, &g) in node.gradient.iter().enumerate() {
                        if g == 0. { continue; }
                        let y = node.value[i];
                        let derivative = match kind {
                            0 => 1. - y * y,
                            1 => y * (1. - y),
                            4 => -1.,
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
                        before[x].gradient[i] += g * derivative;
                    }
                }
                Edge::Blend { p, a, b } => {
                    for (i, &g) in node.gradient.iter().enumerate() {
                        if g == 0. { continue; }
                        let j = if before[p].value.len()==1 { 0 } else { i };
                        let probability = before[p].value[j];
                        let other = b.map_or(0., |b| before[b].value[i]);
                        before[p].gradient[j] += g * (before[a].value[i] - other);
                        before[a].gradient[i] += g * probability;
                        if let Some(b) = b { before[b].gradient[i] += g * (1.-probability); }
                    }
                }
                Edge::Slice { x, start } => {
                    for (i, &g) in node.gradient.iter().enumerate() {
                        before[x].gradient[start + i] += g;
                    }
                }
            }
        }
        ensure!(
            self.tape
                .nodes
                .iter()
                .all(|n| n.gradient.iter().all(|g| g.is_finite())),
            "nonfinite gradient"
        );
        self.differentiated = true;
        Ok(())
    }
}
