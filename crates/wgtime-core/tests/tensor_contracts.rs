use std::collections::BTreeMap;
use wgtime_core::adaptive::{AdaptiveSession, parse};

fn session(source: &str) -> AdaptiveSession {
    AdaptiveSession::prepare(parse(source).unwrap(), 1 << 20).unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tolerance, "index {index}: {a} vs {e}");
    }
}
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(a, b)| a * b).sum() }

#[test]
fn layer_norm_aliases_accumulate_all_vjp_contributions() {
    let input = [0.2, -0.7, 1.1];
    let seed = [0.3, -0.4, 0.8];
    let original = [0.8, -0.5, 1.2];
    // Cover all partitions of the x/gamma/beta IDs, including every alias pair.
    for (gamma, beta) in [("x", "x"), ("x", "affine"), ("affine", "x"), ("affine", "affine")] {
        let source = format!("wgt 0.5\nmodel AliasNorm\ninput x: f32[3]\noutput y: f32[3]\nparameter p: f32[3] = [0.8,-0.5,1.2]\nfn forward(x) {{ let affine = p; return layer_norm(x,{gamma},{beta},0,0.01); }}\n");
        let mut model = session(&source);
        let mut pass = model.forward(&input, 1000, 1 << 20).unwrap();
        let tape_bytes = pass.tape_bytes();
        pass.backward(&seed, &BTreeMap::new()).unwrap();
        let step = 0.001;
        for i in 0..3 {
            let mut plus = input; let mut minus = input;
            plus[i] += step; minus[i] -= step;
            let p = dot(model.forward(&plus, 1000, 1 << 20).unwrap().output(), &seed);
            let m = dot(model.forward(&minus, 1000, 1 << 20).unwrap().output(), &seed);
            close(&[pass.input_gradient().unwrap()[i]], &[(p-m)/(2.*step)], 0.003);
            let mut plus = original.to_vec(); let mut minus = original.to_vec();
            plus[i] += step; minus[i] -= step;
            let p = dot(model.forward_with_parameters(&input, &BTreeMap::from([("p".into(), plus)]), 1000, 1 << 20).unwrap().output(), &seed);
            let m = dot(model.forward_with_parameters(&input, &BTreeMap::from([("p".into(), minus)]), 1000, 1 << 20).unwrap().output(), &seed);
            close(&[pass.parameter_gradient("p").unwrap()[i]], &[(p-m)/(2.*step)], 0.003);
        }
        assert_eq!(pass.tape_bytes(), tape_bytes);
        let first = pass.input_gradient().unwrap().to_vec();
        pass.backward(&seed.map(|x| 2. * x), &BTreeMap::new()).unwrap();
        close(pass.input_gradient().unwrap(), &first.iter().map(|x| 2. * x).collect::<Vec<_>>(), 0.00001);
    }
}

#[test]
fn noncontiguous_reshape_materializes_logical_order_and_its_vjp() {
    let mut model = session("wgt 0.5\nmodel TransposeFlat\ninput x: f32[6]\noutput y: f32[6]\nfn forward(x) { let matrix = x as [2,3]; return matrix.T as [6]; }\n");
    let mut pass = model.forward(&[1., 2., 3., 4., 5., 6.], 1000, 1 << 20).unwrap();
    close(pass.output(), &[1., 4., 2., 5., 3., 6.], 0.00001);
    pass.backward(&[1., 2., 3., 4., 5., 6.], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &[1., 3., 5., 2., 4., 6.], 0.00001);
}

#[test]
fn permuted_batch_product_matches_independent_loop_and_finite_difference() {
    let source = "wgt 0.5\nmodel BatchView\ninput x: f32[20]\noutput y: f32[12]\nfn forward(x) { let a = x[0:12] as [2,2,3]; let b = x[12:20] as [2,2,2]; return batch_matmul(permute(a,[0,2,1]),b) as [12]; }\n";
    let mut model = session(source);
    let input: Vec<_> = (0..20).map(|i| i as f32 * 0.07 - 0.4).collect();
    let mut expected = vec![0.; 12];
    for batch in 0..2 { for row in 0..3 { for col in 0..2 {
        for k in 0..2 {
            expected[(batch*3+row)*2+col] += input[(batch*2+k)*3+row] * input[12+(batch*2+k)*2+col];
        }
    } } }
    let mut pass = model.forward(&input, 10_000, 1 << 20).unwrap();
    close(pass.output(), &expected, 0.00001);
    let seed: Vec<_> = (0..12).map(|i| i as f32 * 0.03 - 0.1).collect();
    pass.backward(&seed, &BTreeMap::new()).unwrap();
    for i in 0..20 {
        let mut plus = input.clone(); let mut minus = input.clone();
        plus[i] += 0.001; minus[i] -= 0.001;
        let p = dot(model.forward(&plus, 10_000, 1 << 20).unwrap().output(), &seed);
        let m = dot(model.forward(&minus, 10_000, 1 << 20).unwrap().output(), &seed);
        close(&[pass.input_gradient().unwrap()[i]], &[(p-m)/0.002], 0.001);
    }
}

#[test]
fn malformed_register_geometry_is_rejected_before_runtime_allocation() {
    let source = "wgt 0.5\nmodel Identity\ninput x: f32[4]\noutput y: f32[4]\nfn forward(x) { return x; }\n";
    let mut plan = parse(source).unwrap();
    plan.registers[0].shape = vec![2, 2];
    assert!(AdaptiveSession::prepare(plan, 1 << 20).is_err());
    let mut plan = parse(source).unwrap();
    plan.version = 3;
    plan.registers[0].shape = vec![0, 4];
    assert!(AdaptiveSession::prepare(plan, 1 << 20).is_err());
    let mut plan = parse(source).unwrap();
    plan.version = 3;
    plan.registers[0].shape = vec![2, 3];
    assert!(AdaptiveSession::prepare(plan, 1 << 20).is_err());
}

#[test]
fn annotated_computed_value_keeps_numeric_and_gradient_semantics() {
    let mut model = session("wgt 0.5\nmodel Annotated\ninput x: f32[6]\noutput y: f32[6]\nfn forward(x) { let matrix: f32[rows=2,cols=3] = tanh(x); let flat: f32[6] = matrix.T; return flat; }\n");
    let input = [-0.3f32, -0.2, -0.1, 0.1, 0.2, 0.3];
    let order = [0, 3, 1, 4, 2, 5];
    let expected: Vec<_> = order.map(|i| input[i].tanh()).to_vec();
    let mut pass = model.forward(&input, 1000, 1 << 20).unwrap();
    close(pass.output(), &expected, 0.00001);
    pass.backward(&[1.; 6], &BTreeMap::new()).unwrap();
    close(pass.input_gradient().unwrap(), &input.map(|x| 1. - x.tanh().powi(2)), 0.00001);
}

#[test]
fn maximum_rank_permutation_preserves_values_and_gradient_order() {
    let source = "wgt 0.5\nmodel RankEight\ninput x: f32[6]\noutput y: f32[6]\nfn forward(x) { let tensor = x as [1,1,1,1,1,1,2,3]; return permute(tensor,[7,6,5,4,3,2,1,0]) as [6]; }\n";
    for compiled in [false, true] {
        let mut model = session(source);
        if compiled { model.enable_compiled().unwrap(); }
        let mut pass = model.forward(&[1., 2., 3., 4., 5., 6.], 1000, 1 << 20).unwrap();
        assert_eq!(pass.output(), &[1., 4., 2., 5., 3., 6.]);
        pass.backward(&[1., 2., 3., 4., 5., 6.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[1., 3., 5., 2., 4., 6.]);
    }
}
