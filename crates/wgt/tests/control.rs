use serde_json::{json, Value};
use std::collections::BTreeMap;
use weightlang_wgt::contracts::{infer, infer_named, Contracts, TensorType};

fn check(model: &Value) -> anyhow::Result<()> {
    if model["interface"]["inputs"].is_array() {
        infer_named(&model["layers"], &model["interface"], &mut BTreeMap::new(), &mut Contracts::new())
    } else {
        let actual = infer(&model["layers"], "", TensorType::parse(&model["interface"]["input"] )?, &mut BTreeMap::new(), &mut Contracts::new())?;
        anyhow::ensure!(actual.compatible(&TensorType::parse(&model["interface"]["output"])?), "output mismatch");
        Ok(())
    }
}

#[test]
fn python_control_and_multiscale_contracts_match() {
    for fixture in [include_str!("fixtures/control_route.json"), include_str!("fixtures/control_ponder.json"), include_str!("fixtures/control_multiscale.json")] {
        check(&serde_json::from_str(fixture).unwrap()).unwrap();
    }
}

#[test]
fn route_rejects_bad_selectors_and_branch_shapes() {
    let model: Value = serde_json::from_str(include_str!("fixtures/control_route.json")).unwrap();
    let mut bad = model.clone();
    bad["interface"]["inputs"][0]["dtype"] = json!("f32");
    assert!(check(&bad).is_err());
    let mut bad = model;
    bad["layers"][0]["layers"][0]["layers"][1]["args"]["out"] = json!(5);
    assert!(check(&bad).is_err());
}

#[test]
fn ponder_requires_probability_and_preserves_all_result_contracts() {
    let model: Value = serde_json::from_str(include_str!("fixtures/control_ponder.json")).unwrap();
    for name in ["missing", "next"] {
        let mut bad = model.clone();
        bad["layers"][0]["args"]["until"] = json!(name);
        assert!(check(&bad).is_err());
    }
    let mut bad = model.clone();
    bad["layers"][0]["results"]["halt_weights"]["shape"] = json!([7,1]);
    assert!(check(&bad).is_err());
    let mut bad = model;
    bad["layers"][0]["args"]["compact"] = json!(1);
    assert!(check(&bad).is_err());
}

#[test]
fn spatial_rejects_invalid_interpolation_and_pooling() {
    use weightlang_wgt::{contracts::TensorType, primitives};
    let input = TensorType::parse(&json!({"dtype":"f32","shape":[2,4,5]})).unwrap();
    for (op,args) in [
        ("resize2d",json!({"height":2,"width":3,"align_corners":1})),
        ("pad2d",json!({"left":5,"right":0,"top":0,"bottom":0,"mode":1})),
        ("avgpool2d",json!({"kernel":3,"stride":0})),
    ] { assert!(primitives::infer(op,&args,&[input.clone()]).is_err()); }
}
