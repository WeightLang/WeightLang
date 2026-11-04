use std::{fs, path::Path, process::Command};
use weightlang_runtime::script;
use weightlang_wgt::script_compiler;

#[test]
fn categorical_types_survive_aot_model_and_checkpoint_roundtrips() {
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/aot-tests");
    fs::create_dir_all(&root).unwrap();
    let dir=tempfile::tempdir_in(root).unwrap();
    let text=r#"wgt 0.5
model Policy
input x: f32[3]
output scores: logits[3]
fn forward(x){return x;}
fn loss(p,t){return sum(p.probabilities*t);}
fn convert(value){let local=value;return local.probabilities;}
fn main(){
 let net=Policy("adam");
 save_model(net,"policy.json"); save_checkpoint(net,"checkpoint.json");
 let restored=load_model("policy.json","adam"); let resumed=load_checkpoint("checkpoint.json");
 let p=convert(restored([1000,1000,-1000]));
 assert(p==convert(resumed([1000,1000,-1000])),"checkpoint output contract");
 seed(91); let a=sample from restored([0,0,0]); let tail=random();
 seed(91); let b=sample from net([0,0,0]);
 assert(a==b && random()==tail,"typed sample RNG");
 return p.probabilities;
}"#;
    let source=dir.path().join("model.wgt"); let executable=dir.path().join("compiled");
    fs::write(&source,text).unwrap();
    script_compiler::compile(&source,&executable).unwrap();
    let reference_root=dir.path().join("reference"); let compiled_root=dir.path().join("compiled-data");
    fs::create_dir(&reference_root).unwrap();fs::create_dir(&compiled_root).unwrap();
    let reference=script::run(text,&reference_root,100000).unwrap();
    fs::remove_file(source).unwrap();
    let output=Command::new(&executable).env("PATH","/nonexistent").arg("--root").arg(&compiled_root).args(["--max-operations","100000"]).output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    let actual:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();
    let required=reference.operations;
    assert_eq!(serde_json::to_value(reference).unwrap(),actual);
    let low_root=dir.path().join("low-budget");fs::create_dir(&low_root).unwrap();
    assert!(script::run(text,&low_root,required-1).is_err());
    let low_aot=dir.path().join("low-budget-aot");fs::create_dir(&low_aot).unwrap();
    let rejected=Command::new(&executable).env("PATH","/nonexistent").arg("--root").arg(&low_aot).arg("--max-operations").arg((required-1).to_string()).output().unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("operation budget exhausted"));
    for path in ["policy.json","checkpoint.json"] {
        let encoded:serde_json::Value=serde_json::from_slice(&fs::read(compiled_root.join(path)).unwrap()).unwrap();
        assert!(encoded.to_string().contains("\"output_type\":\"logits\""));
    }
}
