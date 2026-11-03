use serde_json::json;
use std::collections::BTreeMap;
use weightlang_wgt::contracts::{infer_named, Contracts};

#[test]
fn named_bool_mask_and_outputs_are_checked_independently() {
    let layers = json!([
        {"name":"project","op":"linear","args":{"in":4,"out":4}},
        {"name":"select","op":"where","inputs":["mask","project","fallback"]}
    ]);
    let inputs = json!([
        {"name":"x","dtype":"f32","shape":[4]},
        {"name":"mask","dtype":"bool","shape":[4]},
        {"name":"fallback","dtype":"f32","shape":[4]}
    ]);
    let mut interface = json!({
        "input":inputs[0], "inputs": inputs,
        "output":{"name":"y","dtype":"f32","shape":[4]},
        "outputs":[{"name":"y","dtype":"f32","shape":[4],"from":"select"},
                   {"name":"raw","dtype":"f32","shape":[4],"from":"project"}]
    });
    let mut tensors = BTreeMap::new();
    infer_named(&layers, &interface, &mut tensors, &mut Contracts::new()).unwrap();
    assert_eq!(tensors["project.weight"], vec![4,4]);
    interface["outputs"][1]["shape"] = json!([5]);
    assert!(infer_named(&layers, &interface, &mut BTreeMap::new(), &mut Contracts::new()).is_err());
}

#[test]
fn scan_accepts_explicit_state_and_rejects_wrong_state_width() {
    let layers = json!([
        {"name":"memory","op":"scan","args":{"hidden":4},"inputs":["input","old"],"layers":[
            {"name":"joined","op":"concat","inputs":["input","state"]},
            {"name":"next","op":"linear","args":{"in":6,"out":4}}
        ]},
        {"name":"final","op":"last"}
    ]);
    let mut interface = json!({
        "input":{"name":"samples","dtype":"f32","shape":[3,2],"axes":["time","features"]},
        "output":{"name":"new","dtype":"f32","shape":[4]},
        "inputs":[{"name":"samples","dtype":"f32","shape":[3,2],"axes":["time","features"]},
                  {"name":"old","dtype":"f32","shape":[4]}],
        "outputs":[{"name":"new","dtype":"f32","shape":[4],"from":"final"}]
    });
    let mut contracts = Contracts::new();
    infer_named(&layers, &interface, &mut BTreeMap::new(), &mut contracts).unwrap();
    assert_eq!(contracts["memory"].1.axes, vec![Some("time".into()), None]);
    interface["inputs"][1]["shape"] = json!([5]);
    assert!(infer_named(&layers, &interface, &mut BTreeMap::new(), &mut Contracts::new()).is_err());
}
