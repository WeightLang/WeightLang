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
