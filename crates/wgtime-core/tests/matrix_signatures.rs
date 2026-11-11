use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::{adaptive::AdaptiveSession, training::Optimizer};

const SOURCE: &str = include_str!("../../../examples/native_training/matrix_signatures/model.wgt");
const BUDGET: usize = 8 * 1024 * 1024;

#[test]
fn matrix_signatures_execute_and_train_in_native_and_compiled_forward() {
    for compiled in [false, true] {
        let definition = ModelDefinition::parse(SOURCE).unwrap();
        let ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
            panic!()
        };
        let mut session = AdaptiveSession::prepare(program, BUDGET).unwrap();
        session.set_optimizer(Optimizer::Sgd).unwrap();
        if compiled {
            session.enable_compiled().unwrap();
        }
        let input = [1., 2., 3., 4., 5., 6.];
        assert_eq!(session.run(&input, 100000).unwrap(), [1.5, 1.5, 3.75, 3.75]);
        let loss = session
            .train_custom(
                &[input.to_vec()],
                &[vec![0.; 4]],
                0.001,
                &mut 100000,
                BUDGET,
            )
            .unwrap();
        assert!((loss - 4.078125).abs() < 1e-6);
        let result = session.run(&input, 100000).unwrap();
        // dL/dW[k,n] = sum_m(x[m,k]*prediction[m,n])/4.
        assert!((result[0] - 1.46475).abs() < 1e-6);
        assert!((result[2] - 3.6658125).abs() < 1e-6);
    }
}
