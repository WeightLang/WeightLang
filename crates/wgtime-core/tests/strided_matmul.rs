use std::collections::BTreeMap;
use wgtime_core::adaptive::{AdaptiveSession, Instruction, Program, Register, parse};

const BUDGET: usize = 1 << 20;
const WORK: u64 = 1_000_000;

fn same_values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {index}: {actual} vs {expected}"
        );
    }
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
            "gradient {index}: {actual} vs {expected}"
        );
    }
}

fn check(program: Program, input: &[f32], expected: &[f32], seed: &[f32], gradient: &[f32]) {
    let mut session = AdaptiveSession::prepare(program, BUDGET).unwrap();
    same_values(session.run(input, WORK).unwrap(), expected);
    let mut pass = session.forward(input, WORK, BUDGET).unwrap();
    same_values(pass.output(), expected);
    pass.backward(seed, &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), gradient);
    // Repeated execution resets views; stale output layouts must not leak into the next call.
    same_values(session.run(input, WORK).unwrap(), expected);
}

// The oracle maps logical coordinates back to a physical row-major tensor.
// It does not use the runtime's view layout or matrix kernel.
fn physical(coordinates: [usize; 3], dimensions: [usize; 3], axes: [usize; 3]) -> usize {
    let mut source = [0; 3];
    for axis in 0..3 {
        source[axes[axis]] = coordinates[axis];
    }
    (source[0] * dimensions[1] + source[1]) * dimensions[2] + source[2]
}

fn raw_dimensions(logical: [usize; 3], axes: [usize; 3]) -> [usize; 3] {
    let mut dimensions = [0; 3];
    for axis in 0..3 {
        dimensions[axes[axis]] = logical[axis];
    }
    dimensions
}

#[test]
fn rectangular_matmul_preserves_f64_results_and_gradients_for_each_transpose_pair() {
    let (m, k, n) = (3, 5, 7); // Covers a four-column block and a three-column tail.
    let input: Vec<f32> = (0..m * k + k * n)
        .map(|i| (i % 17) as f32 * 0.03125 - 0.25)
        .collect();
    let seed: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect();
    for transpose_a in [false, true] {
        for transpose_b in [false, true] {
            let (ar, ac) = if transpose_a { (k, m) } else { (m, k) };
            let (br, bc) = if transpose_b { (n, k) } else { (k, n) };
            let suffix_a = if transpose_a { ".T" } else { "" };
            let suffix_b = if transpose_b { ".T" } else { "" };
            let source = format!(
                "wgt 0.5\nmodel MatrixStrides\ninput x: f32[{}]\noutput y: f32[{}]\n\
                 fn forward(x) {{\n\
                   let a = (x[0:{}] as [{ar},{ac}]){suffix_a};\n\
                   let b = (x[{}:{}] as [{br},{bc}]){suffix_b};\n\
                   return (a @ b) as [{}];\n}}\n",
                input.len(),
                m * n,
                m * k,
                m * k,
                input.len(),
                m * n,
            );
            let mut expected = vec![0.0; m * n];
            let mut gradient = vec![0.0; input.len()];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for q in 0..k {
                        let ai = if transpose_a { q * m + i } else { i * k + q };
                        let bi = m * k + if transpose_b { j * k + q } else { q * n + j };
                        sum += f64::from(input[ai]) * f64::from(input[bi]);
                        gradient[ai] += seed[i * n + j] * input[bi];
                        gradient[bi] += seed[i * n + j] * input[ai];
                    }
                    expected[i * n + j] = sum as f32;
                }
            }
            check(parse(&source).unwrap(), &input, &expected, &seed, &gradient);
        }
    }
}

#[test]
fn batch_matmul_handles_all_axis_permutations_including_a_moved_batch_axis() {
    let (batch, m, k, n) = (2, 3, 4, 5);
    let a_len = batch * m * k;
    let input: Vec<f32> = (0..a_len + batch * k * n)
        .map(|i| (i % 19) as f32 * 0.03125 - 0.28125)
        .collect();
    let seed: Vec<f32> = (0..batch * m * n)
        .map(|i| (i % 5) as f32 * 0.0625 - 0.125)
        .collect();
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for a_axes in permutations {
        for b_axes in permutations {
            let a_shape = raw_dimensions([batch, m, k], a_axes);
            let b_shape = raw_dimensions([batch, k, n], b_axes);
            let source = format!(
                "wgt 0.5\nmodel BatchStrides\ninput x: f32[{}]\noutput y: f32[{}]\n\
                 fn forward(x) {{\n\
                   let a = x[0:{a_len}] as {a_shape:?};\n\
                   let b = x[{a_len}:{}] as {b_shape:?};\n\
                   return batch_matmul(permute(a,{a_axes:?}),permute(b,{b_axes:?})) as [{}];\n}}\n",
                input.len(),
                seed.len(),
                input.len(),
                seed.len(),
            );
            let mut expected = vec![0.0; seed.len()];
            let mut gradient = vec![0.0; input.len()];
            for z in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let output = (z * m + i) * n + j;
                        let mut sum = 0.0f64;
                        for q in 0..k {
                            let ai = physical([z, i, q], a_shape, a_axes);
                            let bi = a_len + physical([z, q, j], b_shape, b_axes);
                            sum += f64::from(input[ai]) * f64::from(input[bi]);
                            gradient[ai] += seed[output] * input[bi];
                            gradient[bi] += seed[output] * input[ai];
                        }
                        expected[output] = sum as f32;
                    }
                }
            }
            check(parse(&source).unwrap(), &input, &expected, &seed, &gradient);
        }
    }
}

fn register(name: &str, shape: &[usize]) -> Register {
    Register {
        name: name.into(),
        len: shape.iter().product(),
        shape: shape.to_vec(),
    }
}

#[test]
fn aliased_matmul_outputs_preserve_live_views_and_gradient_snapshots() {
    let weights: Vec<f32> = (0..25).map(|i| (i % 6) as f32 * 0.125 - 0.25).collect();
    let source = format!(
        "wgt 0.5\nmodel AliasProduct\ninput x: f32[25]\noutput y: f32[25]\n\
         fn forward(x) {{\n\
           let a = (-x) as [5,5];\n\
           let v = a.T;\n\
           let b = {weights:?} as [5,5];\n\
           return (((v @ b) @ b) + v) as [25];\n}}\n"
    );
    let mut program = parse(&source).unwrap();
    program.registers = vec![
        register("x", &[25]),
        register("a", &[5, 5]),
        register("b", &[5, 5]),
        register("v", &[5, 5]),
        register("y", &[25]),
    ];
    program.input = "x".into();
    program.instructions = vec![
        Instruction::Reshape {
            out: "a".into(),
            input: "x".into(),
        },
        // Materialize a canonical storage owner, then retain its transposed view.
        Instruction::Neg {
            out: "a".into(),
            input: "a".into(),
        },
        Instruction::Transpose {
            out: "v".into(),
            input: "a".into(),
        },
        Instruction::Constant {
            out: "b".into(),
            data: weights,
        },
        // Overwrite the backing register of the left operand.
        Instruction::Matmul {
            out: "a".into(),
            left: "v".into(),
            right: "b".into(),
        },
        // Overwrite the right operand. The old v must survive both writes.
        Instruction::Matmul {
            out: "b".into(),
            left: "a".into(),
            right: "b".into(),
        },
        Instruction::Add {
            out: "a".into(),
            left: "b".into(),
            right: "v".into(),
        },
        Instruction::Reshape {
            out: "y".into(),
            input: "a".into(),
        },
        Instruction::Return { value: "y".into() },
    ];
    let input: Vec<f32> = (0..25).map(|i| i as f32 * 0.03125 - 0.375).collect();
    let seed: Vec<f32> = (0..25).map(|i| (i % 3) as f32 * 0.125 - 0.125).collect();
    let mut reference = AdaptiveSession::prepare(parse(&source).unwrap(), BUDGET).unwrap();
    let mut pass = reference.forward(&input, WORK, BUDGET).unwrap();
    pass.backward(&seed, &BTreeMap::new()).unwrap();
    check(
        program,
        &input,
        pass.output(),
        &seed,
        pass.input_gradient().unwrap(),
    );
}

#[test]
fn one_register_can_be_both_matrix_operands_and_the_destination() {
    let source = "wgt 0.5\nmodel Square\ninput x: f32[25]\noutput y: f32[25]\n\
                  fn forward(x) { let a = x as [5,5]; return (a @ a) as [25]; }\n";
    let mut program = parse(source).unwrap();
    program.registers = vec![
        register("x", &[25]),
        register("a", &[5, 5]),
        register("y", &[25]),
    ];
    program.input = "x".into();
    program.instructions = vec![
        Instruction::Reshape {
            out: "a".into(),
            input: "x".into(),
        },
        Instruction::Matmul {
            out: "a".into(),
            left: "a".into(),
            right: "a".into(),
        },
        Instruction::Reshape {
            out: "y".into(),
            input: "a".into(),
        },
        Instruction::Return { value: "y".into() },
    ];
    let input: Vec<f32> = (0..25).map(|i| (i % 7) as f32 * 0.0625 - 0.1875).collect();
    let seed: Vec<f32> = (0..25).map(|i| (i % 3) as f32 * 0.125 - 0.125).collect();
    let mut reference = AdaptiveSession::prepare(parse(source).unwrap(), BUDGET).unwrap();
    let mut pass = reference.forward(&input, WORK, BUDGET).unwrap();
    pass.backward(&seed, &BTreeMap::new()).unwrap();
    check(
        program,
        &input,
        pass.output(),
        &seed,
        pass.input_gradient().unwrap(),
    );
}
