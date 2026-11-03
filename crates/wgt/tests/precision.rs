use serde_json::json;
use weightlang_wgt::precision::validate;

#[test]
fn precision_policy_rejects_unknown_formats_targets_groups_and_sharing() {
    let layers = json!([{"name":"p","op":"linear","args":{"in":8,"out":8}},
                       {"name":"act","op":"relu","args":{}}]);
    let good = json!({"precision":{"p":{"storage":"auto","group":32}}});
    validate(&good,&layers,&[]).unwrap();
    for value in [json!(0),json!(4097),json!(2.5),json!(true)] {
        let mut bad = good.clone(); bad["precision"]["p"]["group"] = value;
        assert!(validate(&bad,&layers,&[]).is_err());
    }
    let mut bad = good.clone(); bad["precision"]["p"]["storage"] = json!("fp4");
    assert!(validate(&bad,&layers,&[]).is_err());
    let bad = json!({"precision":{"act":{"storage":"int4","group":32}}});
    assert!(validate(&bad,&layers,&[]).is_err());
    assert!(validate(&good,&layers,&[("p.weight".into(),"q.weight".into())]).is_err());
}
