use serde_json::{Value,json};
use weightlang_wgt::contracts::{infer_named, Contracts};
use std::collections::BTreeMap;

fn check(m: &Value) -> anyhow::Result<()> {
    infer_named(&m["layers"],&m["interface"],&mut BTreeMap::new(),&mut Contracts::new())?;
    weightlang_wgt::symbolic::validate(&m["layers"],&m["interface"])
}

#[test]
fn structured_cells_preserve_every_state_and_check_nested_results() {
    let m: Value = serde_json::from_str(include_str!("fixtures/multi_state.json")).unwrap();
    check(&m).unwrap();
    let mut bad = m.clone();
    bad["layers"][0]["returns"].as_object_mut().unwrap().remove("state1");
    assert!(check(&bad).is_err());
    let mut bad = m.clone();
    bad["layers"][0]["returns"]["state1"] = json!("missing.output");
    assert!(check(&bad).is_err());
    let mut bad = m;
    bad["interface"]["inputs"][1]["shape"] = json!([4]);
    assert!(check(&bad).is_err());
}
