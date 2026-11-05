use std::collections::BTreeMap;
use wgtime_core::adaptive::{AdaptiveSession, parse};
const CASES: [(
    &str,
    &[usize],
    &[usize],
    &[usize],
    &[usize],
    &[usize],
    usize,
); 3] = [
    (
        include_str!("fixtures/cnn_1d/model.wgt"),
        &[2, 5],
        &[4, 1, 3],
        &[2],
        &[1],
        &[1],
        2,
    ),
    (
        include_str!("fixtures/cnn_2d/model.wgt"),
        &[2, 3, 5],
        &[2, 2, 2, 2],
        &[1, 2],
        &[1, 0],
        &[2, 1],
        1,
    ),
    (
        include_str!("fixtures/cnn_3d/model.wgt"),
        &[2, 2, 3, 4],
        &[2, 1, 2, 2, 2],
        &[1, 2, 1],
        &[0, 1, 1],
        &[1, 1, 2],
        2,
    ),
];
fn session(source: &str, native: bool) -> AdaptiveSession {
    let mut s = AdaptiveSession::prepare(parse(source).unwrap(), 1 << 20).unwrap();
    if native {
        s.enable_compiled().unwrap();
    }
    s
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < tol, "index {i}: {a} vs {b}");
    }
}
// Independent oracle enumerates complete spatial coordinates and explicitly zero pads.
// It uses no production geometry/indexing helpers, and accumulates in f64.
fn coordinates(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    for &n in dims {
        all = all
            .into_iter()
            .flat_map(|p| {
                (0..n).map(move |i| {
                    let mut p = p.clone();
                    p.push(i);
                    p
                })
            })
            .collect();
    }
    all
}
fn flat(coords: &[usize], dims: &[usize]) -> usize {
    coords.iter().zip(dims).fold(0, |n, (c, d)| n * d + c)
}
fn oracle(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    xs: &[usize],
    ks: &[usize],
    stride: &[usize],
    pad: &[usize],
    dil: &[usize],
    groups: usize,
) -> Vec<f32> {
    let out: Vec<_> = (0..stride.len())
        .map(|a| (xs[a + 1] + 2 * pad[a] - dil[a] * (ks[a + 2] - 1) - 1) / stride[a] + 1)
        .collect();
    let mut y = vec![];
    for oc in 0..ks[0] {
        for p in coordinates(&out) {
            let mut sum = f64::from(b[oc]);
            for c in 0..ks[1] {
                for q in coordinates(&ks[2..]) {
                    let pos: Vec<_> = p
                        .iter()
                        .zip(&q)
                        .enumerate()
                        .map(|(a, (&p, &q))| {
                            (p * stride[a] + q * dil[a]) as isize - pad[a] as isize
                        })
                        .collect();
                    if pos
                        .iter()
                        .enumerate()
                        .all(|(a, &p)| p >= 0 && p < xs[a + 1] as isize)
                    {
                        let ic = oc / (ks[0] / groups) * ks[1] + c;
                        let mut xc = vec![ic];
                        xc.extend(pos.iter().map(|&p| p as usize));
                        let mut wc = vec![oc, c];
                        wc.extend(q);
                        sum += f64::from(x[flat(&xc, xs)]) * f64::from(w[flat(&wc, ks)]);
                    }
                }
            }
            y.push(sum as f32);
        }
    }
    y
}
#[test]
fn convolution_native_reference_and_all_gradients() {
    for (source, xs, ks, stride, pad, dil, groups) in CASES {
        let input: Vec<_> = (0..xs.iter().product())
            .map(|i| ((i * 7 % 19) as f32 - 9.) / 10.)
            .collect();
        let mut s = session(source, true);
        let mut reference = session(source, false);
        let raw = oracle(
            &input,
            s.parameter("kernel").unwrap(),
            s.parameter("bias").unwrap(),
            xs,
            ks,
            stride,
            pad,
            dil,
            groups,
        );
        let expected: Vec<_> = raw.iter().map(|x| x.tanh()).collect();
        close(&s.evaluate(&input, 100000).unwrap(), &expected, 1e-7);
        close(
            &reference.evaluate(&input, 100000).unwrap(),
            &expected,
            1e-7,
        );
        let seed: Vec<_> = (0..expected.len())
            .map(|i| (i % 7) as f32 / 7. - 0.4)
            .collect();
        let mut pass = s.forward(&input, 100000, 1 << 20).unwrap();
        pass.backward(&seed, &BTreeMap::new()).unwrap();
        let mut rp = reference.forward(&input, 100000, 1 << 20).unwrap();
        rp.backward(&seed, &BTreeMap::new()).unwrap();
        close(
            pass.input_gradient().unwrap(),
            rp.input_gradient().unwrap(),
            1e-7,
        );
        let dot = |y: Vec<f32>| {
            y.iter()
                .zip(&seed)
                .map(|(y, g)| f64::from(*y) * f64::from(*g))
                .sum::<f64>()
        };
        for name in ["kernel", "bias"] {
            close(
                pass.parameter_gradient(name).unwrap(),
                rp.parameter_gradient(name).unwrap(),
                1e-7,
            );
            let original = s.parameter(name).unwrap().to_vec();
            for i in 0..original.len() {
                let mut p = original.clone();
                let mut m = original.clone();
                p[i] += 0.002;
                m[i] -= 0.002;
                s.replace_parameter(name, &p).unwrap();
                let yp = dot(s.evaluate(&input, 100000).unwrap());
                s.replace_parameter(name, &m).unwrap();
                let ym = dot(s.evaluate(&input, 100000).unwrap());
                let n = (yp - ym) / 0.004;
                let a = pass.parameter_gradient(name).unwrap()[i];
                assert!((n - f64::from(a)).abs() < 6e-5, "{name}[{i}]: {n} vs {a}");
            }
            s.replace_parameter(name, &original).unwrap();
        }
        for i in 0..input.len() {
            let mut p = input.clone();
            let mut m = input.clone();
            p[i] += 0.002;
            m[i] -= 0.002;
            let n = (dot(s.evaluate(&p, 100000).unwrap()) - dot(s.evaluate(&m, 100000).unwrap()))
                / 0.004;
            assert!(
                (n - f64::from(pass.input_gradient().unwrap()[i])).abs() < 6e-5,
                "input {i}"
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(s.evaluate(&input, 100000).unwrap());
        }
        eprintln!(
            "CNN rank {}: 1000 native forwards {:?}; managed {} B; tape {} B; work {}; native counts {:?}",
            stride.len(),
            start.elapsed(),
            s.managed_bytes(),
            pass.tape_bytes(),
            pass.steps(),
            s.compiled_forward_counts()
        );
        assert!(s.compiled_forward_counts().0 > 1000);
    }
}
#[test]
fn convolution_training_budget_finite_and_ir_failures() {
    for (source, xs, _, _, _, _, _) in CASES {
        let input = vec![0.4; xs.iter().product()];
        let mut s = session(source, true);
        s.set_optimizer(wgtime_core::training::Optimizer::Sgd)
            .unwrap();
        let y = s.evaluate(&input, 100000).unwrap();
        let target = vec![0.15; y.len()];
        let loss = |y: Vec<f32>| y.iter().map(|v| (v - 0.15).powi(2)).sum::<f32>() / y.len() as f32;
        let before = loss(y);
        let original = s.parameter("kernel").unwrap().to_vec();
        assert!(
            s.train_custom(&[input.clone()], &[target.clone()], 0.1, &mut 4, 1 << 20)
                .is_err()
        );
        assert_eq!(s.parameter("kernel").unwrap(), original);
        for _ in 0..100 {
            s.train_custom(
                &[input.clone()],
                &[target.clone()],
                0.3,
                &mut 100000,
                1 << 20,
            )
            .unwrap();
        }
        let after = loss(s.evaluate(&input, 100000).unwrap());
        assert!(after < before * 0.1, "{before} -> {after}");
        eprintln!("CNN training {before} -> {after}");
        assert!(s.evaluate(&input, 4).is_err());
        assert!(s.forward(&input, 100000, 1).is_err());
        let huge = vec![f32::MAX; s.parameter("kernel").unwrap().len()];
        s.replace_parameter("kernel", &huge).unwrap();
        assert!(s.evaluate(&vec![f32::MAX; input.len()], 100000).is_err());
        assert!(AdaptiveSession::prepare(parse(source).unwrap(), 1).is_err());
        let p = parse(source).unwrap();
        for version in [1, 2] {
            let mut p = p.clone();
            p.version = version;
            assert!(AdaptiveSession::prepare(p, 1 << 20).is_err());
        }
        let mut p = p.clone();
        for i in &mut p.instructions {
            if let wgt_language::adaptive::Instruction::Conv { groups, .. } = i {
                *groups = 0;
            }
        }
        assert!(AdaptiveSession::prepare(p, 1 << 20).is_err());
    }
}
#[test]
fn convolution_strict_shapes_and_boundary_windows() {
    let src = CASES[0].0;
    for (a, b) in [
        ("[2],[1],[1],2", "[0],[1],[1],2"),
        ("[2],[1],[1],2", "[2],[-1],[1],2"),
        ("[2],[1],[1],2", "[2],[1],[0],2"),
        ("[2],[1],[1],2", "[2],[1],[1],3"),
        ("[2],[1],[1],2", "[2,1],[1],[1],2"),
        ("reshape(kernel,[4,1,3])", "reshape(kernel,[2,2,3])"),
    ] {
        assert!(parse(&src.replace(a, b)).is_err(), "accepted {b}");
    }
    // A one-cell input, dilated three-cell kernel and padding: only the center tap is valid.
    let source = "wgt 0.5\nmodel Boundary\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return reshape(conv(reshape(x,[1,1]),reshape([2,3,4],[1,1,3]),[0.5],[1],[2],[2],1),[1]);}\n";
    for native in [false, true] {
        let mut s = session(source, native);
        assert_eq!(s.evaluate(&[2.], 100).unwrap(), [6.5]);
    }
}
