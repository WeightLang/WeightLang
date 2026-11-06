use std::{fs, path::Path};
use wgtime_core::script;

#[test]
fn categorical_types_survive_model_and_checkpoint_roundtrips() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let text = r#"wgt 0.5
model Policy
input x: f32[3]
output scores: logits[3]
fn forward(x){return x;}
fn loss(p,t){return sum(p.probabilities*t);}
fn convert(value){let local=value;return local.probabilities;}
fn main(){
 let net=Policy("adam");
 save model net to "policy.json"; save checkpoint net to "checkpoint.json";
 let restored=load model from "policy.json" using "adam"; let resumed=load checkpoint from "checkpoint.json";
 let p=convert(restored([1000,1000,-1000]));
 assert p==convert(resumed([1000,1000,-1000])),"checkpoint output contract";
 seed(91); let a=sample from restored([0,0,0]); let tail=random();
 seed(91); let b=sample from net([0,0,0]);
 assert a==b && random()==tail,"typed sample RNG";
 return p.probabilities;
}"#;
    let reference = script::run(text, dir.path(), 100000).unwrap();
    assert_eq!(reference.result, serde_json::json!([0.5, 0.5, 0.]));
    let low_root = dir.path().join("low-budget");
    fs::create_dir(&low_root).unwrap();
    let error = script::run(text, &low_root, reference.operations - 1).unwrap_err();
    assert!(
        format!("{error:#}").contains("operation budget exhausted"),
        "{error:#}"
    );
    for path in ["policy.json", "checkpoint.json"] {
        let encoded: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.path().join(path)).unwrap()).unwrap();
        assert!(encoded.to_string().contains("\"output_type\":\"logits\""));
    }
}
