//! Interleaved release-mode matrix-vector benchmark: identical decoded weights.
use std::{hint::black_box, path::Path, time::Instant};
use wgtime_core::{Layer, Model, Parameter, Session, Value};
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .unwrap_or("3072".into())
        .parse()
        .unwrap();
    let group = 32;
    let stride = n.div_ceil(group) * group;
    let mut dense = Vec::with_capacity(n * n);
    let mut packed = vec![0u8; (n * stride).div_ceil(2)];
    for row in 0..n {
        for col in 0..stride {
            let q = ((row * 31 + col * 7) % 15) as i32 - 7;
            let i = row * stride + col;
            packed[i / 2] |= ((q + 8) as u8) << ((i % 2) * 4);
            if col < n {
                dense.push(q as f32 * 0.03125);
            }
        }
    }
    let create = |value| Model {
        output_type: Default::default(),
        version: 1,
        input: n,
        parameters: vec![Parameter {
            name: "w".into(),
            shape: vec![n, n],
            frozen: true,
            value,
        }],
        layers: vec![Layer::Linear {
            weight: "w".into(),
            bias: None,
        }],
    };
    let mut f32 = Session::prepare(
        create(Value::Resident { data: dense }),
        Path::new("."),
        1024 * 1024 * 1024,
    )
    .unwrap();
    let mut int4 = Session::prepare(
        create(Value::Packed {
            storage: "int4".into(),
            group,
            data: packed,
            scales: vec![0.03125; n * stride / group],
        }),
        Path::new("."),
        1024 * 1024 * 1024,
    )
    .unwrap();
    let x: Vec<_> = (0..n).map(|i| ((i % 23) as f32 - 11.) / 32.).collect();
    let expected = f32.run(&x).unwrap().to_vec();
    let actual = int4.run(&x).unwrap();
    let max_abs = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs < 1e-4, "incorrect result: {max_abs}");
    for _ in 0..20 {
        black_box(f32.run(&x).unwrap());
        black_box(int4.run(&x).unwrap());
    }
    let measure = |s: &mut Session| {
        let start = Instant::now();
        for _ in 0..50 {
            black_box(s.run(black_box(&x)).unwrap());
        }
        start.elapsed().as_nanos() / 50
    };
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..9 {
        if i % 2 == 0 {
            a.push(measure(&mut f32));
            b.push(measure(&mut int4));
        } else {
            b.push(measure(&mut int4));
            a.push(measure(&mut f32));
        }
    }
    println!(
        "{}",
        serde_json::json!({"shape":[n,n],"group":group,"threads":1,"iterations_per_sample":50,"samples":9,
        "f32_ns":a,"int4_ns":b,"max_abs_error":max_abs,"f32_memory":f32.report(),"int4_memory":int4.report(),
        "timing":"Rust Session::run only; excludes loading and JSON IPC"})
    );
}
