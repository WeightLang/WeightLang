use serde_json::json;
use std::collections::BTreeMap;
use weightlang_wgt::contracts::{infer, Contracts, TensorType};

#[test]
fn moe_and_tied_projection_contracts() {
    let input = TensorType::parse(&json!({"dtype":"i64","shape":[4]})).unwrap();
    let layers = json!([
        {"name":"embed","op":"embedding","args":{"vocab":13,"dim":8}},
        {"name":"experts","op":"sparse_moe","args":{"dim":8,"hidden":12,"experts":3,"top_k":2}},
        {"name":"head","op":"linear","args":{"in":8,"out":13,"bias":0,"tie_embedding":1}}
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
    assert_eq!(output.shape, vec![4, 13]);
    assert_eq!(weights["experts.experts.2.down_proj.weight"], vec![8, 12]);
    for (index, key, value) in [(1, "top_k", 4), (2, "tie_embedding", 2), (2, "bias", 1)] {
        let mut bad = layers.clone();
        bad[index]["args"][key] = json!(value);
        assert!(infer(
            &bad,
            "",
            input.clone(),
            &mut BTreeMap::new(),
            &mut Contracts::new()
        )
        .is_err());
    }
}
