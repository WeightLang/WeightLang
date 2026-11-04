use serde_json::json;
use std::collections::BTreeMap;
use weightlang_wgt::parameter_identity::adapters;

#[test]
fn adapter_methods_validate_canonical_shapes_and_reject_bad_contracts() {
    let layers = json!([{"name":"projection", "op":"linear", "args":{"in":8,"out":5}}]);
    for method in ["lora", "dora", "pissa", "qlora"] {
        let interface = json!({"parameter_ids":{}, "parameter_shares":{}, "adapters":{"task":{
            "method":method, "target":"projection", "rank":3, "alpha":6, "freeze":1
        }}});
        let initial = BTreeMap::from([
            ("projection.weight".to_owned(), vec![5, 8]),
            ("projection.bias".to_owned(), vec![5]),
        ]);
        let mut shapes = initial.clone();
        adapters(&interface, &layers, &mut shapes).unwrap();
        assert_eq!(shapes["projection.lora_a"], vec![3, 8]);
        assert_eq!(shapes["projection.lora_b"], vec![5, 3]);
        assert_eq!(
            shapes.contains_key("projection.magnitude"),
            method == "dora"
        );
        for (field, value) in [
            ("rank", json!(9)),
            ("method", json!("missing")),
            ("method", json!(null)),
            ("alpha", json!(0)),
        ] {
            let mut bad = interface.clone();
            bad["adapters"]["task"][field] = value;
            assert!(adapters(&bad, &layers, &mut initial.clone()).is_err());
        }
        if method != "lora" {
            let mut bad = interface.clone();
            bad["adapters"]["task"]["freeze"] = json!(0);
            assert!(adapters(&bad, &layers, &mut initial.clone()).is_err());
        }
    }
}
