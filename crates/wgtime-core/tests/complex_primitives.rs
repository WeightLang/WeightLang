//! Stable masks and selections needed by native LightNest networks.
use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::AdaptiveSession;
fn session(body: &str, width: usize) -> AdaptiveSession {
    let source =
        format!("wgt 0.5\nmodel Complex\ninput x:f32[{width}]\noutput y:f32[{width}]\n{body}");
    let ModelPlan::Forward(plan) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(plan.version, 3);
    let mut legacy = plan.clone();
    legacy.version = 2;
    assert!(legacy.validate_tensor_contracts().is_err());
    let mut session = AdaptiveSession::prepare(plan, 1 << 20).unwrap();
    session.enable_compiled().unwrap();
    assert!(!session.is_compiled());
    session
}
#[test]
fn exact_gelu_extremes_and_numerical_derivative() {
    let mut s = session("fn forward(x){return gelu(x);}", 7);
    let x = [-8., -3., -1., 0., 1., 3., 8.];
    let mut pass = s.forward(&x, 10000, 1 << 20).unwrap();
    assert!(pass.output()[0].abs() < 1e-12);
    assert_eq!(pass.output()[3], 0.);
    assert_eq!(pass.output()[6], 8.);
    pass.backward(&[1.; 7], &BTreeMap::new()).unwrap();
    for i in 0..7 {
        let mut plus = x;
        let mut minus = x;
        plus[i] += 0.001;
        minus[i] -= 0.001;
        let a = s.forward(&plus, 10000, 1 << 20).unwrap().output()[i];
        let b = s.forward(&minus, 10000, 1 << 20).unwrap().output()[i];
        assert!((pass.input_gradient().unwrap()[i] - (a - b) / 0.002).abs() < 0.0005);
    }
}
#[test]
fn topk_last_axis_ties_and_selection_stop_gradient() {
    let mut s = session(
        "parameter count:f32[1]=[2]\nfn forward(x){let v=x as [2,4];return (v*topk_mask(v,count)) as [8];}",
        8,
    );
    let x = [4., 4., 2., 1., -5., -4., -4., -8.];
    let mut pass = s.forward(&x, 10000, 1 << 20).unwrap();
    assert_eq!(pass.output(), &[4., 4., 0., 0., 0., -4., -4., 0.]);
    pass.backward(&[1.; 8], &BTreeMap::new()).unwrap();
    assert_eq!(
        pass.input_gradient().unwrap(),
        &[1., 1., 0., 0., 0., 1., 1., 0.]
    );
    assert_eq!(pass.parameter_gradient("count").unwrap(), &[0.]);
    assert!(
        s.forward(&x, 31, 1 << 20).is_err(),
        "selection search work is charged"
    );
    for count in [-1., 1.5, 5.] {
        s.replace_parameter("count", &[count]).unwrap();
        assert!(s.forward(&x, 10000, 1 << 20).is_err());
    }
    s.replace_parameter("count", &[0.]).unwrap();
    assert_eq!(s.forward(&x, 10000, 1 << 20).unwrap().output(), &[0.; 8]);
}
#[test]
fn masked_softmax_ignores_extreme_invalid_logits_and_blocks_mask_gradients() {
    let mut s = session(
        "parameter mask:f32[2,2]=[0,1,1,0]\nfn forward(x){return masked_softmax(x as [2,2],mask as [2,2],1) as [4];}",
        4,
    );
    let mut pass = s.forward(&[1000., -1000., 2., 1.], 10000, 1 << 20).unwrap();
    assert_eq!(pass.output(), &[0., 1., 1., 0.]);
    pass.backward(&[1., -2., 3., -4.], &BTreeMap::new())
        .unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[0.; 4]);
    assert_eq!(pass.parameter_gradient("mask").unwrap(), &[0.; 4]);
    for mask in [[0., 0., 1., 0.], [0., 0.5, 1., 0.]] {
        s.replace_parameter("mask", &mask).unwrap();
        assert!(s.forward(&[1.; 4], 10000, 1 << 20).is_err());
    }
}

#[test]
fn explicit_shaped_large_embedding_read_has_no_oversized_flat_temporary() {
    let source = "wgt 0.5\nmodel Large\nparameter table:f32[257,128]=zeros()\nfn forward(x:f32[1])->(f32[128],){return (embedding(table as [257,128],x) as [128],);}";
    let ModelPlan::Forward(plan) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    assert!(
        plan.registers
            .iter()
            .all(|r| r.shape.iter().all(|&d| d <= 16384))
    );
    let mut s = AdaptiveSession::prepare(plan, 2 << 20).unwrap();
    s.replace_parameter("table", &vec![1.; 257 * 128]).unwrap();
    let mut pass = s.forward(&[256.], 1000000, 2 << 20).unwrap();
    assert_eq!(pass.output(), &[1.; 128]);
    pass.backward(&[1.; 128], &BTreeMap::new()).unwrap();
    let gradient = pass.parameter_gradient("table").unwrap();
    assert!(gradient[..256 * 128].iter().all(|&v| v == 0.));
    assert_eq!(&gradient[256 * 128..], &[1.; 128]);
    assert_eq!(pass.input_gradient().unwrap(), &[0.]);
}

#[test]
fn shaped_large_read_keeps_existing_compiled_matmul_path() {
    let source = "wgt 0.5\nmodel LargeMatmul\nparameter table:f32[257,128]=ones()\nfn forward(x:f32[257])->(f32[128],){return ((((table as [257,128]).T) @ (x as [257,1])) as [128],);}";
    let ModelPlan::Forward(plan) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    let mut s = AdaptiveSession::prepare(plan, 4 << 20).unwrap();
    s.enable_compiled().unwrap();
    assert!(s.is_compiled());
    let mut pass = s.forward(&[1.; 257], 1000000, 4 << 20).unwrap();
    assert_eq!(pass.output(), &[257.; 128]);
    pass.backward(&[1.; 128], &BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[128.; 257]);
    assert!(
        pass.parameter_gradient("table")
            .unwrap()
            .iter()
            .all(|&v| v == 1.)
    );
}
