use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::{
    adaptive::{
        AdaptiveSession,
        recurrent::{OutputSlice, StateLink},
    },
    training::Optimizer,
};
const SOURCE: &str = "wgt 0.5\nmodel Recurrent\ninput x:f32[2]\noutput y:f32[1]\nparameter gain:f32[1]=[0.5]\nfn forward(x){return x[0:1]+x[1:2]*gain;}\n";
fn model() -> AdaptiveSession {
    let ModelPlan::Forward(program) = ModelDefinition::parse(SOURCE)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    let mut model = AdaptiveSession::prepare(program, 1024 * 1024).unwrap();
    model.set_optimizer(Optimizer::Sgd).unwrap();
    model
}
fn link() -> StateLink {
    StateLink {
        input_offset: 1,
        output_offset: 0,
        width: 1,
    }
}
fn output() -> OutputSlice {
    OutputSlice {
        offset: 0,
        width: 1,
    }
}
#[test]
fn final_only_loss_propagates_through_all_hidden_links() {
    let mut model = model();
    let inputs = vec![vec![1., 0.4], vec![2., 99.], vec![3., -99.]];
    let targets = vec![None, None, Some(vec![0.])];
    let report = model
        .recurrent_gradients(
            &inputs,
            &targets,
            link(),
            output(),
            &mut 100000,
            1024 * 1024,
        )
        .unwrap();
    assert!((report.loss - 18.49).abs() < 1e-5);
    for (a, b) in report.outputs.iter().flatten().zip([1.2, 2.6, 4.3]) {
        assert!((a - b).abs() < 1e-6);
    }
    assert!((report.parameter_gradients["gain"][0] - 28.38).abs() < 1e-5);
    for (a, b) in report
        .input_gradients
        .iter()
        .flatten()
        .zip([2.15, 1.075, 4.3, 0., 8.6, 0.])
    {
        assert!((a - b).abs() < 2e-6);
    }
    assert_eq!(model.parameter("gain").unwrap(), [0.5]);
    let loss = model
        .train_recurrent(
            &inputs,
            &targets,
            link(),
            output(),
            0.001,
            None,
            &mut 100000,
            1024 * 1024,
        )
        .unwrap();
    assert!((loss - report.loss).abs() < 1e-6);
    assert!((model.parameter("gain").unwrap()[0] - 0.47162).abs() < 1e-6);
}
#[test]
fn recurrent_budget_and_target_errors_do_not_commit_optimizer_updates() {
    let mut model = model();
    let before = serde_json::to_value(model.program()).unwrap();
    let inputs = vec![vec![1., 0.4], vec![2., 0.], vec![3., 0.]];
    let targets = vec![None, None, Some(vec![0.])];
    let revision = model.revision();
    assert!(
        model
            .train_recurrent(
                &inputs,
                &targets,
                link(),
                output(),
                0.01,
                None,
                &mut 100000,
                32
            )
            .is_err()
    );
    assert!(
        model
            .train_recurrent(
                &inputs,
                &[None, None, None],
                link(),
                output(),
                0.01,
                None,
                &mut 100000,
                1024 * 1024
            )
            .is_err()
    );
    assert_eq!(model.revision(), revision);
    assert_eq!(serde_json::to_value(model.program()).unwrap(), before);
    assert_eq!(model.run(&[1., 0.4], 100000).unwrap(), [1.2]);
}

#[test]
fn caller_tape_limit_cannot_exceed_session_memory_reservation() {
    let ModelPlan::Forward(program) = ModelDefinition::parse(SOURCE)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    let mut session = AdaptiveSession::prepare(program, 96).unwrap();
    session.set_optimizer(Optimizer::Adam).unwrap();
    let before = session.program().parameters[0].data.clone();
    let checkpoint = session.checkpoint_bytes_bounded(4096).unwrap();
    let error = session
        .train_recurrent(
            &[vec![1., 0.4], vec![2., 0.]],
            &[None, Some(vec![0.])],
            link(),
            output(),
            0.01,
            None,
            &mut 100000,
            64 * 1024 * 1024,
        )
        .unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    assert_eq!(session.program().parameters[0].data, before);
    assert_eq!(session.checkpoint_bytes_bounded(4096).unwrap(), checkpoint);
    assert_eq!(session.revision(), 0);
}
