use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::operations::OperationSession;
const SOURCE: &str = r#"wgt 0.5
model Cell
state activity: f32[2] = [0,0];
parameter gain: f32[2] = [1,1]
frozen fixed: f32[2] = [2,2]
fn observe(input: f32[2]) -> f32[2] { return activity + input * gain; }
fn advance(input: f32[2]) -> f32[2] { activity <- activity + input; return activity; }
fn bad(input: f32[2]) -> f32[2] { activity <- input; gain <- input; return log([0,0]); }
fn reset() { activity <- [0,0]; }
fn combine(left: f32[2], right: f32[2]) -> f32[2] { return left + right; }
fn update(input: f32[2]) { gain <- input; }
fn twice(input: f32[2]) -> f32[2] { activity <- input; activity <- activity + input; gain <- activity; return gain; }
fn stalled(input: f32[2]) -> f32[2] { activity <- input; while positive([1]) { activity <- activity + input; } return activity; }
"#;
fn session() -> OperationSession {
    let ModelPlan::Operations(plan) = ModelDefinition::parse(SOURCE)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    OperationSession::prepare(plan, 1024 * 1024).unwrap()
}
#[test]
fn state_transactions_isolation_and_reset() {
    let mut a = session();
    let mut b = session();
    assert_eq!(
        a.call("advance", &[vec![1., 2.]], 1000).unwrap(),
        Some(vec![1., 2.])
    );
    assert_eq!(
        a.call("observe", &[vec![1., 1.]], 1000).unwrap(),
        Some(vec![2., 3.])
    );
    assert_eq!(
        b.call("observe", &[vec![1., 1.]], 1000).unwrap(),
        Some(vec![1., 1.])
    );
    let before = a.state_values();
    let parameters = a.parameter_values();
    let revision = a.revision();
    assert!(a.call("bad", &[vec![4., 5.]], 1000).is_err());
    assert_eq!(before, a.state_values());
    assert_eq!(parameters, a.parameter_values());
    assert_eq!(revision, a.revision());
    assert_eq!(a.call("reset", &[], 1000).unwrap(), None);
    assert_eq!(
        a.call("observe", &[vec![1., 1.]], 1000).unwrap(),
        Some(vec![1., 1.])
    );
    assert_eq!(
        a.call("combine", &[vec![1., 2.], vec![3., 4.]], 1000)
            .unwrap(),
        Some(vec![4., 6.])
    );
    let calls = a.native_calls();
    let revision = a.revision();
    assert!(a.call("stalled", &[vec![1., 2.]], 100).is_err());
    assert_eq!(a.state_values()[0].2, vec![0., 0.]);
    assert_eq!(a.native_calls(), calls);
    assert_eq!(a.revision(), revision);
    assert_eq!(
        a.call("twice", &[vec![2., 3.]], 1000).unwrap(),
        Some(vec![4., 6.])
    );
    assert_eq!(a.state_values()[0].2, vec![4., 6.]);
    assert_eq!(
        a.parameter_values()
            .iter()
            .find(|p| p.0 == "gain")
            .unwrap()
            .2,
        vec![4., 6.]
    );
}
#[test]
fn search_candidates_are_atomic_and_weights_exclude_state() {
    let mut s = session();
    let mut best = f32::INFINITY;
    for gain in [0., 1., 2., 3.] {
        s.update_parameters(&[("gain".into(), vec![2], vec![gain, gain])])
            .unwrap();
        let y = s.call("observe", &[vec![1., 1.]], 1000).unwrap().unwrap();
        let error = (y[0] - 3.).powi(2);
        assert!(error < best);
        best = error;
    }
    assert_eq!(best, 0.);
    let before = s.parameter_values();
    assert!(
        s.update_parameters(&[
            ("gain".into(), vec![2], vec![9., 9.]),
            ("fixed".into(), vec![2], vec![2., 2.])
        ])
        .is_err()
    );
    assert_eq!(before, s.parameter_values());
    assert!(
        s.update_parameters(&[("gain".into(), vec![1, 2], vec![9., 9.])])
            .is_err()
    );
    assert!(
        s.update_parameters(&[("gain".into(), vec![2], vec![f32::NAN, 9.])])
            .is_err()
    );
    let bytes = s.weights_bytes(1024 * 1024).unwrap();
    let tensors = wgtime_core::weight_artifact::inspect(&bytes, 1024 * 1024).unwrap();
    assert!(tensors.iter().all(|p| p.name != "activity"));
    let mut restored = session();
    restored.call("advance", &[vec![2., 2.]], 1000).unwrap();
    restored.bind_weights(&bytes, "none").unwrap();
    assert_eq!(restored.state_values()[0].2, vec![2., 2.]);
    assert_eq!(
        restored.call("observe", &[vec![1., 1.]], 1000).unwrap(),
        Some(vec![5., 5.])
    );
}
#[test]
fn zero_storage_void_and_pure_operations() {
    for source in [
        "wgt 0.5\nmodel Empty\nfn reset() {}\n",
        "wgt 0.5\nmodel Pure\nfn observe(x:f32[1])->f32[1] {return x;}\n",
    ] {
        let definition = ModelDefinition::parse(source).unwrap();
        let ModelPlan::Operations(plan) = definition.instantiate_plan().unwrap() else {
            panic!()
        };
        let mut s = OperationSession::prepare(plan, 1024 * 1024).unwrap();
        if source.contains("reset") {
            assert_eq!(s.call("reset", &[], 100).unwrap(), None);
        } else {
            assert_eq!(s.call("observe", &[vec![2.]], 100).unwrap(), Some(vec![2.]));
        }
    }
    assert!(
        ModelDefinition::parse("wgt 0.5\nmodel Bad\nfn add(x:f32[1], y)->f32[1]{return x;}\n")
            .is_err()
    );
    assert!(
        ModelDefinition::parse("wgt 0.5\nmodel Bad\nfn run(x:f32[1]) {let a=x; a <- x;}\n")
            .is_err()
    );
}
#[test]
fn schemas_are_pure_and_invalid_updates_rejected() {
    let d = ModelDefinition::parse(SOURCE).unwrap();
    assert!(d.interface().is_none());
    assert_eq!(d.states().len(), 1);
    let bad = d.operations().iter().find(|s| s.name == "bad").unwrap();
    assert_eq!(bad.effects.writes_parameters, ["gain"]);
    assert_eq!(bad.effects.writes_state, ["activity"]);
    assert!(ModelDefinition::parse(&SOURCE.replace("gain <- input", "fixed <- input")).is_err());
    assert!(ModelDefinition::parse(&SOURCE.replace("fn update", "fn forward")).is_err());
}

#[test]
fn operation_bindings_preserve_identity_across_declaration_orders() {
    let source = r#"wgt 0.5
model Slots
state total: f32[1] = [5];
parameter gain: f32[1] = [2]
state stamp: f32[1] = [11];
parameter offset: f32[1] = [3]
fn read(x: f32[1]) -> f32[1] { return total + stamp + gain * x + offset; }
fn write(x: f32[1]) -> f32[1] { total <- x; stamp <- total + [7]; gain <- stamp; return gain + offset; }
fn fail(x: f32[1]) -> f32[1] { total <- x; gain <- x; return log([0]); }
"#;
    let ModelPlan::Operations(mut plan) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    plan.parameters.reverse();
    plan.states.reverse();
    for (index, (_, program)) in plan.entries.iter_mut().enumerate() {
        let len = program.parameters.len();
        program.parameters.rotate_left((index + 1) % len);
    }
    let mut s = OperationSession::prepare(plan, 1 << 20).unwrap();
    assert_eq!(s.call("read", &[vec![4.]], 1000).unwrap(), Some(vec![27.]));
    assert_eq!(
        s.call("write", &[vec![13.]], 1000).unwrap(),
        Some(vec![23.])
    );
    assert_eq!(
        s.parameter_values(),
        vec![
            ("offset".into(), vec![1], vec![3.]),
            ("gain".into(), vec![1], vec![20.]),
        ]
    );
    assert_eq!(
        s.state_values(),
        vec![
            ("stamp".into(), vec![1], vec![20.]),
            ("total".into(), vec![1], vec![13.]),
        ]
    );
    assert_eq!(s.revision(), 1);
    s.update_parameters(&[
        ("gain".into(), vec![1], vec![4.]),
        ("offset".into(), vec![1], vec![17.]),
    ])
    .unwrap();
    assert_eq!(s.call("read", &[vec![2.]], 1000).unwrap(), Some(vec![58.]));
    let before = (
        s.parameter_values(),
        s.state_values(),
        s.revision(),
        s.native_calls(),
    );
    assert!(s.call("fail", &[vec![9.]], 1000).is_err());
    assert!(
        s.update_parameters(&[("total".into(), vec![1], vec![9.])])
            .is_err()
    );
    assert_eq!(
        before,
        (
            s.parameter_values(),
            s.state_values(),
            s.revision(),
            s.native_calls()
        )
    );
}

#[test]
fn operation_storage_rejects_ambiguous_parameter_state_names() {
    let ModelPlan::Operations(mut plan) = ModelDefinition::parse(SOURCE)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!()
    };
    plan.states[0].name.clone_from(&plan.parameters[0].name);
    let error = OperationSession::prepare(plan, 1 << 20).err().unwrap();
    assert!(
        error
            .to_string()
            .contains("duplicate operation storage identity")
    );
}
#[test]
fn repeated_writes_roll_back_before_retry() {
    let source = SOURCE.replace(
        "activity <- input; gain <- input; return log([0,0]);",
        "activity <- input; activity <- activity + input; gain <- activity; return log([0,0]);",
    );
    let ModelPlan::Operations(plan) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("expected operations plan")
    };
    let mut session = OperationSession::prepare(plan, 1024 * 1024).unwrap();
    session.call("twice", &[vec![2., 3.]], 1000).unwrap();
    for input in [vec![7., 8.], vec![-4., 9.], vec![0., 0.]] {
        let parameters = session.parameter_values();
        let state = session.state_values();
        let revision = session.revision();
        let calls = session.native_calls();
        assert!(session.call("bad", &[input], 1000).is_err());
        assert_eq!(session.parameter_values(), parameters);
        assert_eq!(session.state_values(), state);
        assert_eq!(session.revision(), revision);
        assert_eq!(session.native_calls(), calls);
        assert_eq!(
            session.call("advance", &[vec![1., 1.]], 1000).unwrap(),
            Some(state[0].2.iter().map(|value| value + 1.).collect())
        );
        assert_eq!(session.revision(), revision + 1);
        assert_eq!(session.native_calls(), calls + 1);
    }
}

#[test]
fn candidate_validation_rejects_whole_batches_without_mutation() {
    let source = r#"wgt 0.5
model Domains
# Parameters include continuous, frozen, probability and discrete domains.
parameter value: f32[1] = [1]
frozen fixed: f32[1] = [2]
parameter chance: prob = 0.5
learn enabled: bool = true;
learn depth: int in 0..=3 = 1;
# State is separate from all external parameter candidates.
state total: f32[1] = [0]
fn observe() -> f32[1] { return total + value + enabled + depth; }
"#;
    let ModelPlan::Operations(plan) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("expected operations plan")
    };
    let mut session = OperationSession::prepare(plan, 1024 * 1024).unwrap();
    let parameters = session.parameter_values();
    let state = session.state_values();
    for invalid in [
        ("unknown", vec![1], vec![1.]),
        ("total", vec![1], vec![1.]),
        ("value", vec![1], vec![5.]),
        ("fixed", vec![1], vec![2.]),
        ("chance", vec![1], vec![0.]),
        ("enabled", vec![1], vec![0.25]),
        ("depth", vec![1], vec![1.5]),
        ("depth", vec![1], vec![4.]),
        ("depth", vec![1], vec![-1.]),
        ("enabled", vec![1, 1], vec![0.]),
        ("enabled", vec![1], vec![]),
        ("enabled", vec![1], vec![f32::NAN]),
        ("enabled", vec![1], vec![f32::INFINITY]),
    ] {
        assert!(
            session
                .update_parameters(&[
                    ("value".into(), vec![1], vec![9.]),
                    (invalid.0.into(), invalid.1, invalid.2),
                ])
                .is_err(),
            "accepted invalid candidate for {}",
            invalid.0
        );
        assert_eq!(session.parameter_values(), parameters);
        assert_eq!(session.state_values(), state);
        assert_eq!(session.revision(), 0);
        assert_eq!(session.native_calls(), 0);
    }
    assert!(session.update_parameters(&[]).is_err());
    session
        .update_parameters(&[
            ("value".into(), vec![1], vec![3.]),
            ("enabled".into(), vec![1], vec![0.]),
            ("depth".into(), vec![1], vec![2.]),
        ])
        .unwrap();
    assert_eq!(session.revision(), 1);
    assert_eq!(session.native_calls(), 0);
    assert_eq!(session.state_values(), state);
    assert_eq!(session.call("observe", &[], 1000).unwrap(), Some(vec![5.]));
    assert_eq!(session.revision(), 1);
}
