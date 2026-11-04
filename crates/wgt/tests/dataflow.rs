use serde_json::json;
use std::collections::BTreeMap;
use weightlang_wgt::contracts::{infer, Contracts, TensorType};

#[test]
fn branches_and_shared_recurrence_infer_weights_once() {
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[4]})).unwrap();
    let layers = json!([{"name":"loop","op":"recur","args":{"steps":8},"layers":[
        {"name":"gate","op":"linear","args":{"in":4,"out":4},"inputs":["input"]},
        {"name":"sigmoid","op":"sigmoid"},
        {"name":"value","op":"linear","args":{"in":4,"out":4},"inputs":["input"]},
        {"name":"result","op":"mul","inputs":["sigmoid","value"]}
    ]}]);
    let mut weights = BTreeMap::new();
    let mut contracts = Contracts::new();
    assert_eq!(
        infer(&layers, "", input.clone(), &mut weights, &mut contracts).unwrap(),
        input
    );
    assert_eq!(weights.len(), 4);
    assert_eq!(weights["loop.gate.weight"], vec![4, 4]);
    let mut invalid = layers;
    invalid[0]["layers"][2]["inputs"] = json!(["future"]);
    assert!(infer(
        &invalid,
        "",
        input,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
}

#[test]
fn scan_state_concat_and_last_have_matching_contracts() {
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[5,2]})).unwrap();
    let layers = json!([
        {"name":"memory","op":"scan","args":{"hidden":4},"layers":[
            {"name":"joined","op":"concat","inputs":["input","state"]},
            {"name":"candidate","op":"linear","args":{"in":6,"out":4}},
            {"name":"next","op":"tanh"}
        ]},
        {"name":"final","op":"last"}
    ]);
    let mut weights = BTreeMap::new();
    let mut contracts = Contracts::new();
    assert_eq!(
        infer(&layers, "", input.clone(), &mut weights, &mut contracts)
            .unwrap()
            .shape,
        vec![4]
    );
    assert_eq!(contracts["memory.joined"].1.shape, vec![6]);
    assert_eq!(contracts["memory"].1.shape, vec![5, 4]);
    assert_eq!(weights["memory.candidate.weight"], vec![4, 6]);
    let mut invalid = layers;
    invalid[0]["args"]["hidden"] = json!(3);
    assert!(infer(
        &invalid,
        "",
        input,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
}

#[test]
fn decoder_primitives_preserve_shape_and_exact_parameter_layout() {
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[7,16]})).unwrap();
    let layers = json!([
        {"name":"norm","op":"rmsnorm","args":{"dim":16,"eps":0.0002}},
        {"name":"attention","op":"rope_attention","args":{"dim":16,"heads":4,"kv_heads":2,"head_dim":6,"qk_norm":1,"qkv_bias":1,"window":3}},
        {"name":"proj","op":"linear","args":{"in":16,"out":16,"bias":0}},
        {"name":"ln","op":"layernorm","args":{"dim":16,"bias":0,"eps":0.0002}},
        {"name":"activation","op":"gelu","args":{"approximate":1}}
    ]);
    let mut weights = BTreeMap::new();
    let output = infer(
        &layers,
        "",
        input.clone(),
        &mut weights,
        &mut Contracts::new(),
    )
    .unwrap();
    assert_eq!(output, input);
    assert_eq!(weights["attention.q_proj.weight"], vec![24, 16]);
    assert_eq!(weights["attention.k_proj.weight"], vec![12, 16]);
    assert_eq!(weights["attention.q_norm.weight"], vec![6]);
    assert_eq!(weights["attention.o_proj.weight"], vec![16, 24]);
    assert!(!weights.contains_key("attention.o_proj.bias"));
    assert!(!weights.contains_key("proj.bias"));
    assert!(!weights.contains_key("ln.bias"));
    let mut invalid = layers;
    invalid[1]["args"]["head_dim"] = json!(3);
    assert!(infer(
        &invalid,
        "",
        input,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
}

#[test]
fn full_projection_qk_norm_layout() {
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[7,16]})).unwrap();
    let mut layers = json!([{"name":"a","op":"rope_attention","args":{"dim":16,"heads":4,"kv_heads":2,"head_dim":6,"qk_norm":2}}]);
    let mut weights = BTreeMap::new();
    infer(
        &layers,
        "",
        input.clone(),
        &mut weights,
        &mut Contracts::new(),
    )
    .unwrap();
    assert_eq!(weights["a.q_norm.weight"], vec![24]);
    assert_eq!(weights["a.k_norm.weight"], vec![12]);
    layers[0]["args"]["qk_norm"] = json!(3);
    assert!(infer(
        &layers,
        "",
        input,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
}

#[test]
fn probabilistic_recur_checks_direct_scalar_probability() {
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[2]})).unwrap();
    let layers = json!([{"name":"loop","op":"recur","args":{"steps":5,"until":"stop"},"layers":[
        {"name":"next","op":"linear","args":{"in":2,"out":2}},
        {"name":"gate","op":"linear","args":{"in":2,"out":1}},
        {"name":"stop","op":"sigmoid"},
        {"name":"result","op":"identity","inputs":["next"]}
    ]}]);
    let mut weights = BTreeMap::new();
    let actual = infer(
        &layers,
        "",
        input.clone(),
        &mut weights,
        &mut Contracts::new(),
    )
    .unwrap();
    assert_eq!(actual, input);
    assert_eq!(weights.len(), 4);
    for gate in [json!("missing"), json!("next"), json!(3)] {
        let mut invalid = layers.clone();
        invalid[0]["args"]["until"] = gate;
        assert!(infer(
            &invalid,
            "",
            input.clone(),
            &mut BTreeMap::new(),
            &mut Contracts::new()
        )
        .is_err());
    }
}
