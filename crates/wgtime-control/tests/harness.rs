use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};
use wgtime_control::{
    harness::Harness,
    openresponses::{
        budget::{Budget, Grant, Limits},
        transport::Endpoint,
    },
    Value,
};
fn configured() -> (Harness, Budget) {
    let budget = Budget::new(Limits {
        requests: 1,
        bytes: 1000,
        tools: 1,
        rounds: 1,
        concurrency: 1,
        deadline_ms: 1000,
    })
    .unwrap();
    let grant = Grant {
        endpoints: BTreeSet::from(["local".into()]),
        tools: BTreeSet::from(["add".into()]),
        budget: budget.clone(),
    };
    let profiles = BTreeMap::from([(
        "local".into(),
        Arc::new(Endpoint::new("http://127.0.0.1:1/responses", None).unwrap()),
    )]);
    (Harness::from_grant(profiles, grant).unwrap(), budget)
}
#[test]
fn child_shares_ledger_and_cannot_forge_or_cross_use_handles() {
    let (mut parent, budget) = configured();
    let mut child = parent.child();
    let mut remaining = 100;
    let session = parent
        .call(
            "responses_session",
            &[Value::Data(json!({"model":"fake","input":"hi"}))],
            Path::new("."),
            &mut remaining,
        )
        .unwrap()
        .unwrap();
    assert!(session.data().is_err());
    assert!(child
        .call(
            "responses_cancel",
            &[session.clone()],
            Path::new("."),
            &mut remaining
        )
        .is_err());
    assert!(child
        .call(
            "responses_cancel",
            &[Value::Data(json!(0))],
            Path::new("."),
            &mut remaining
        )
        .is_err());
    assert!(child
        .call(
            "responses_restrict",
            &[Value::Data(json!(["local"])), Value::Data(json!(["other"]))],
            Path::new("."),
            &mut remaining
        )
        .is_err());
    child
        .call(
            "responses_restrict",
            &[Value::Data(json!(["local"])), Value::Data(json!([]))],
            Path::new("."),
            &mut remaining,
        )
        .unwrap();
    budget.bytes(999).unwrap();
    let usage = child
        .call("responses_usage", &[], Path::new("."), &mut remaining)
        .unwrap()
        .unwrap();
    assert_eq!(usage.data().unwrap()["bytes"], 999);
    assert!(budget.bytes(2).is_err());
    parent
        .call(
            "responses_cancel",
            &[session],
            Path::new("."),
            &mut remaining,
        )
        .unwrap();
}
