use serde_json::json;
use std::collections::BTreeMap;
use weightlang_wgt::weight_types::validate;

#[test]
fn finite_types_mixtures_and_alias_conflicts() {
    let shapes = BTreeMap::from([("a.weight".into(),vec![3,4]),("b.weight".into(),vec![3,4])]);
    let aliases = vec![("b.weight".into(),"a.weight".into())];
    let good = json!({"weight_types":{"tri":[-1,0,1],"custom":[-2,0.25,3]},"weight_assignments":[
        {"target":"a.weight","type":"tri","axis":0,"start":0,"stop":1},
        {"target":"b.weight","type":"custom","axis":0,"start":1,"stop":3}
    ]});
    validate(&good,&shapes,&aliases).unwrap();
    for (key,value) in [("start",json!(0)),("axis",json!(1)),("stop",json!(4)),("type",json!("absent")),("target",json!("absent"))] {
        let mut bad = good.clone(); bad["weight_assignments"][1][key] = value;
        assert!(validate(&bad,&shapes,&aliases).is_err());
    }
    for values in [json!([]),json!([0,0]),json!([1,1.000000001]),json!([1e100])] {
        let mut bad = good.clone(); bad["weight_types"]["tri"] = values;
        assert!(validate(&bad,&shapes,&aliases).is_err());
    }
}

#[test]
fn implicit_embedding_ties_share_one_domain() {
    use weightlang_wgt::weight_types::parameter_aliases;
    let shapes = BTreeMap::from([("embed.weight".into(),vec![3,4]),("head.weight".into(),vec![3,4])]);
    let layers = json!([
        {"name":"embed","op":"embedding","args":{"vocab":3,"dim":4}},
        {"name":"head","op":"linear","args":{"in":4,"out":3,"bias":0,"tie_embedding":1}}
    ]);
    let aliases = parameter_aliases(&layers,&shapes,&[]).unwrap();
    assert_eq!(aliases,vec![("head.weight".into(),"embed.weight".into())]);
    let bad = json!({"weight_types":{"tri":[-1,0,1]},"weight_assignments":[
        {"target":"head.weight","type":"tri"}, {"target":"embed.weight","type":"tri"}
    ]});
    assert!(validate(&bad,&shapes,&aliases).is_err());
}
