use std::path::Path;
use weightlang_runtime::execution::{AllocationPolicy, ExecutionSession};
use weightlang_runtime::{Model, Session};

fn make(storage: &str, group: usize, rows: usize, cols: usize) -> (Model, Vec<f32>) {
    let stride = if storage.starts_with("int") {
        cols.div_ceil(group) * group
    } else {
        cols
    };
    let mut data = Vec::new();
    let mut values = Vec::new();
    for i in 0..rows * stride {
        let q = (i % 15) as i32 - 7;
        let x = q as f32 * 0.125;
        if i % stride < cols {
            values.push(x);
        }
        match storage {
            "int4" => {
                if i % 2 == 0 {
                    data.push((q + 8) as u8);
                } else {
                    *data.last_mut().unwrap() |= ((q + 8) as u8) << 4;
                }
            }
            "int8" => data.push((q + 128) as u8),
            "bf16" => data.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes()),
            "f16" => {
                // Test values are exact binary multiples, including signed and zero.
                let bits = if x == 0. {
                    0
                } else {
                    (((x.to_bits() >> 16) & 0x8000)
                        | ((((x.to_bits() >> 23) & 255) - 112) << 10)
                        | ((x.to_bits() >> 13) & 1023)) as u16
                };
                data.extend_from_slice(&bits.to_le_bytes());
            }
            _ => unreachable!(),
        }
    }
    let scales = if storage.starts_with("int") {
        vec![0.125; rows * stride / group]
    } else {
        vec![]
    };
    let m = serde_json::json!({"version":1,"input":cols,"parameters":[
        {"name":"w","shape":[rows,cols],"kind":"packed","storage":storage,"group":group,"data":data,"scales":scales},
        {"name":"b","shape":[rows],"kind":"resident","data":vec![0.25;rows]}
    ],"layers":[{"op":"linear","weight":"w","bias":"b"}]});
    (serde_json::from_value(m).unwrap(), values)
}

#[test]
fn packed_cpu_tail_groups_formats_and_read_only() {
    for storage in ["int4", "int8", "f16", "bf16"] {
        for (rows, cols, group) in [(3, 65, 32), (5, 33, 3), (2, 7, 1), (7, 128, 64)] {
            let (m, weights) = make(storage, group, rows, cols);
            let mut session = Session::prepare(m, Path::new("."), 1 << 20).unwrap();
            let x: Vec<_> = (0..cols).map(|i| (i % 13) as f32 / 13.).collect();
            let expected: Vec<f32> = weights
                .chunks(cols)
                .map(|w| 0.25 + w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>())
                .collect();
            let actual = session.run(&x).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 2e-5, "{storage}: {a} != {b}");
            }
            let mut read = vec![0.; weights.len()];
            session.read_parameter("w", 0, &mut read).unwrap();
            assert_eq!(read, weights);
            assert!(
                !session
                    .parameter_info()
                    .iter()
                    .find(|p| p.name == "w")
                    .unwrap()
                    .writable
            );
            assert_eq!(session.report().workspace_bytes, rows * 4);
        }
    }
}

#[test]
fn packed_validation_and_budget() {
    let (m, _) = make("int4", 32, 2, 65);
    let mut value = serde_json::to_value(m).unwrap();
    value["parameters"][0]["scales"][0] = serde_json::json!(0);
    assert!(Session::prepare(
        serde_json::from_value(value).unwrap(),
        Path::new("."),
        1 << 20
    )
    .is_err());
    let (m, _) = make("int4", 32, 2, 65);
    assert!(Session::prepare(m, Path::new("."), 1).is_err());
}

#[test]
fn packed_gpu_matches_cpu_when_available() {
    if !weightlang_runtime::execution::devices()
        .iter()
        .any(|v| v.id != "cpu")
    {
        return;
    }
    for storage in ["int4", "int8", "f16", "bf16"] {
        for saturated in 0..if storage.starts_with("int") { 2 } else { 1 } {
            let (mut m, _) = make(storage, 3, 5, 65);
            // 0xffffffff is valid integer storage but a NaN if incorrectly loaded as F32.
            if saturated == 1 {
                if let weightlang_runtime::Value::Packed { data, .. } = &mut m.parameters[0].value {
                    data.fill(255);
                }
            }
            let reference = serde_json::from_value(serde_json::to_value(&m).unwrap()).unwrap();
            let mut cpu = Session::prepare(reference, Path::new("."), 1 << 20).unwrap();
            let mut gpu = ExecutionSession::prepare(
                m,
                Path::new("."),
                1 << 20,
                "gpu",
                AllocationPolicy::Flexible,
            )
            .unwrap();
            let x = vec![0.125; 65];
            let expected = cpu.run(&x).unwrap().to_vec();
            let actual = gpu.run(&x).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-5, "{storage}: {a} != {b}");
            }
            assert!(gpu.report().gpu_buffer_bytes < 65 * 5 * 4 + 4096);
        }
    }
}
