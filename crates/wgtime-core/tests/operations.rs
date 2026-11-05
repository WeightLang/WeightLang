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
