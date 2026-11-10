use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::{adaptive::AdaptiveSession, training::Optimizer};
const SOURCE: &str =
    include_str!("../../../examples/native_training/signature_dimensions/model.wgt");
const BUDGET: usize = 8 * 1024 * 1024;
fn session(width: usize, compiled: bool) -> AdaptiveSession {
    let definition =
        ModelDefinition::parse_with_dimensions(SOURCE, &BTreeMap::from([("N".into(), width)]))
            .unwrap();
    let ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
        panic!()
    };
    let mut session = AdaptiveSession::prepare(program, BUDGET).unwrap();
    session.set_optimizer(Optimizer::Sgd).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    session
}
#[test]
fn signature_forward_trains_and_restores_in_native_and_compiled_execution() {
    for compiled in [false, true] {
        for width in [2, 3] {
            let input = vec![1.; width];
            let mut model = session(width, compiled);
            assert_eq!(model.run(&input, 100000).unwrap(), input);
            model
                .train_custom(
                    &[input.clone()],
                    &[vec![0.; width]],
                    0.1,
                    &mut 100000,
                    BUDGET,
                )
                .unwrap();
            let expected = 1. - 0.1 * (width as f32 + 1.) / width as f32;
            for value in model.run(&input, 100000).unwrap() {
                assert!((value - expected).abs() < 1e-6);
            }
            let weights = model.weights_bytes(BUDGET).unwrap();
            let mut restored = session(width, compiled);
            restored.bind_weights(&weights, "sgd").unwrap();
            assert_eq!(
                model.run(&input, 100000).unwrap(),
                restored.run(&input, 100000).unwrap()
            );
        }
    }
}
