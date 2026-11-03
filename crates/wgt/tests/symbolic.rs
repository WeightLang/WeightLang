use serde_json::json;
use weightlang_wgt::symbolic::validate;

fn fixture() -> (serde_json::Value, serde_json::Value) {
    let layers = json!([{"name":"head","op":"linear","args":{"in":4,"out":4}}]);
    let input = json!({"name":"x","dtype":"f32","shape":[2,4],"shape_expr":["T",4]});
    let output = json!({"name":"y","dtype":"f32","shape":[2,4],"shape_expr":["T",4]});
    let mut routed = output.clone();
    routed["from"] = json!("head");
    (
        layers,
        json!({"dimensions":{"T":{"min":1,"max":8}},
        "constraints":[["==",["%","T",2],0]],
        "input":input,"output":output,"inputs":[input],"outputs":[routed]}),
    )
}
#[test]
fn bounded_dimension_checks_all_admissible_shapes() {
    let (layers, interface) = fixture();
    validate(&layers, &interface).unwrap();
    let mut bad = interface.clone();
    bad["outputs"][0]["shape_expr"] = json!([2, 4]);
    bad["output"]["shape_expr"] = json!([2, 4]);
    assert!(validate(&layers, &bad).is_err());
    let mut bad = interface.clone();
    bad["constraints"] = json!([[">", "T", 8]]);
    assert!(validate(&layers, &bad).is_err());
    let mut bad = interface.clone();
    bad["dimensions"]["T"]["max"] = json!(4097);
    assert!(validate(&layers, &bad).is_err());
    let mut bad = interface.clone();
    bad["constraints"] = json!([["==", ["%", "T", 0], 0]]);
    assert!(validate(&layers, &bad).is_err());
}
#[test]
fn bounded_dimension_rejects_weight_width_changes_and_static_transforms() {
    let (mut layers, mut interface) = fixture();
    for key in ["input", "output"] {
        interface[key]["shape"] = json!([4, 2]);
        interface[key]["shape_expr"] = json!([4, "T"]);
    }
    interface["inputs"][0] = interface["input"].clone();
    interface["outputs"][0] = interface["output"].clone();
    interface["outputs"][0]["from"] = json!("head");
    layers[0]["args"] = json!({"in":2,"out":2});
    assert!(validate(&layers, &interface).is_err());
    let (mut layers, interface) = fixture();
    layers[0]["op"] = json!("rearrange");
    assert!(validate(&layers, &interface).is_err());
}
