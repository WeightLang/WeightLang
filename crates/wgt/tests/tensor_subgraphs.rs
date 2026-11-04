use serde_json::{json, Value};
use std::collections::BTreeMap;
use weightlang_wgt::contracts::{infer_named, Contracts};

fn check(m: &Value) -> anyhow::Result<()> {
    infer_named(
        &m["layers"],
        &m["interface"],
        &mut BTreeMap::new(),
        &mut Contracts::new(),
    )?;
    weightlang_wgt::symbolic::validate(&m["layers"], &m["interface"])
}

#[test]
fn python_emitted_rotary_and_delta_contracts() {
    for fixture in [
        include_str!("fixtures/partial_rotary.json"),
        include_str!("fixtures/gated_delta_rule.json"),
        include_str!("fixtures/gated_attention.json"),
    ] {
        let m: Value = serde_json::from_str(fixture).unwrap();
        check(&m).unwrap();
        let mut bad = m.clone();
        bad["interface"]["outputs"][0]["dtype"] = json!("i64");
        assert!(check(&bad).is_err());
    }
}

#[test]
fn mismatched_heads_and_state_are_rejected() {
    let mut m: Value = serde_json::from_str(include_str!("fixtures/partial_rotary.json")).unwrap();
    m["layers"][0]["layers"][0]["args"]["parts"] = json!(3);
    assert!(check(&m).is_err());
    let mut m: Value =
        serde_json::from_str(include_str!("fixtures/gated_delta_rule.json")).unwrap();
    m["layers"][0]["returns"]["state0"] = json!("result");
    assert!(check(&m).is_err());
}
