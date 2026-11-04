use std::path::Path;
use weightlang_runtime::{adaptive, script};

fn source(kind: &str, main: &str) -> String {
    format!("wgt 0.5\nmodel Policy\ninput x: f32[3]\noutput scores: {kind}[3]\nfn forward(x) {{ return x; }}\n{main}")
}

#[test]
fn typed_probability_conversion_charges_linear_work() {
    for kind in ["logits","probabilities"] {
        let input=if kind=="logits" {"[0,0,0]"} else {"[0.25,0.25,0.5]"};
        let baseline=source(kind,&format!("fn main(){{let net=Policy(\"adam\");let value=net({input});return value;}}"));
        let converted=source(kind,&format!("fn main(){{let net=Policy(\"adam\");let value=net({input});return value.probabilities;}}"));
        let baseline=script::run(&baseline,Path::new("."),10000).unwrap();
        let report=script::run(&converted,Path::new("."),10000).unwrap();
        // A builtin expression and call cost two ticks; validating three entries costs three more.
        assert_eq!(report.operations-baseline.operations,5);
        let error=script::run(&converted,Path::new("."),report.operations-1).unwrap_err();
        assert!(format!("{error:#}").contains("operation budget exhausted"));
    }
}

#[test]
fn explicit_type_survives_locals_parameters_and_returns() {
    let text = source("logits", r#"
fn identity(value) { let local=value; return local; }
fn main() {
 let net=Policy("adam"); let scores=identity(net([1000,1000,-1000]));
 let __unused=0; let softmax=4;
 let p=scores.probabilities; let q=identity(p).probabilities;
 assert(p==q,"probabilities conversion must be idempotent");
 assert(p[0]==0.5 && p[1]==0.5 && p[2]==0,"stable conversion");
 return q;
}"#);
    assert_eq!(script::run(&text, Path::new("."), 10000).unwrap().result, serde_json::json!([0.5,0.5,0.]));
}

#[test]
fn probabilities_are_validated_without_normalization() {
    let text = source("probabilities", "fn main(){ let net=Policy(\"adam\"); let p=net([0.1,0.2,0.700001]); return p.probabilities; }");
    let result = script::run(&text, Path::new("."), 10000).unwrap().result;
    assert_eq!(result[2].as_f64().unwrap(), 0.700001_f32 as f64);
    for values in ["[0,0,0]", "[-0.1,0.2,0.9]", "[1.1,0,0]", "[0.5,0.5]"] {
        let text=source("probabilities", &format!("fn main(){{ let net=Policy(\"adam\"); return net({values}); }}"));
        assert!(script::run(&text,Path::new("."),10000).is_err(), "{values}");
    }
    let plain=source("f32", "fn main(){ let net=Policy(\"adam\"); return net([0,0,0]).probabilities; }");
    assert!(format!("{:#}",script::run(&plain,Path::new("."),10000).unwrap_err()).contains("requires logits"));
}

#[test]
fn model_serialization_retains_output_contract_and_legacy_defaults() {
    let text=source("logits", "fn main(){return 0;}");
    let plan=adaptive::parse(&text).unwrap();
    let encoded=serde_json::to_value(&plan).unwrap();
    assert_eq!(encoded["output_type"],"logits");
    assert_eq!(encoded["version"],2);
    let mut wrong_version=encoded.clone(); wrong_version["version"]=1.into();
    assert!(adaptive::AdaptiveSession::prepare(serde_json::from_value(wrong_version).unwrap(),1_000_000).is_err());
    let restored=serde_json::from_value::<adaptive::Program>(encoded.clone()).unwrap();
    assert_eq!(restored.output_type,weightlang_runtime::categorical::VectorType::Logits);
    adaptive::AdaptiveSession::prepare(restored,1_000_000).unwrap();
    let mut invalid_op=encoded.clone(); invalid_op["version"]=1.into();
    invalid_op.as_object_mut().unwrap().remove("output_type");
    invalid_op["instructions"]=serde_json::json!([
        {"op":"validate_probabilities","out":"checked","input":"x"},
        {"op":"return","value":"checked"}
    ]);
    invalid_op["registers"]=serde_json::json!([{"name":"x","len":3},{"name":"checked","len":3}]);
    assert!(adaptive::AdaptiveSession::prepare(serde_json::from_value(invalid_op).unwrap(),1_000_000).is_err());
    let mut future=encoded.clone(); future["version"]=3.into();
    assert!(adaptive::AdaptiveSession::prepare(serde_json::from_value(future).unwrap(),1_000_000).is_err());
    let mut legacy=encoded;
    legacy.as_object_mut().unwrap().remove("output_type");
    legacy["version"]=1.into();
    assert_eq!(serde_json::from_value::<adaptive::Program>(legacy.clone()).unwrap().output_type,weightlang_runtime::categorical::VectorType::F32);
    legacy["arbitrary_unknown_field"]=true.into();
    assert!(serde_json::from_value::<adaptive::Program>(legacy).is_err());
}

#[test]
fn declared_probability_sampling_uses_mass_and_preserves_rng() {
    let text=source("probabilities", r#"fn draw(net,x){return sample from net(x);}
fn main(){
 let net=Policy("adam"); seed(11); let count=0;
 for i in range(0,2000){let action=draw(net,[0,0.250001,0.75]); assert(action>0,"zero mass"); if action==1 {count=count+1;}}
 assert(count>400 && count<600,"categorical distribution");
 seed(19); let a=draw(net,[0,0.25,0.75]); let tail=random();
 seed(19); let b=draw(net,[0,0.25,0.75]); assert(a==b && tail==random(),"reproducible RNG");
 return true;
}"#);
    assert_eq!(script::run(&text,Path::new("."),200000).unwrap().result,true);
}

#[test]
fn native_model_versions_and_unknown_fields_remain_strict() {
    use weightlang_runtime::{Session, categorical::VectorType};
    let source="wgt 0.5\nmodel Native\ninput x: f32[3]\nlayer passthrough: identity\noutput scores: logits[3]\ninit seed=7\n";
    let (_,graph)=weightlang_runtime::source::graph(source).unwrap();
    let model=graph.instantiate().unwrap();
    let encoded=serde_json::to_value(model).unwrap();
    assert_eq!(encoded["version"],2);
    assert_eq!(encoded["output_type"],"logits");
    let restored=serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(Session::prepare(restored,Path::new("."),1_000_000).unwrap().output_type(),VectorType::Logits);
    for version in [1,3] {
        let mut invalid=encoded.clone();invalid["version"]=version.into();
        assert!(Session::prepare(serde_json::from_value(invalid).unwrap(),Path::new("."),1_000_000).is_err());
    }
    let mut legacy=encoded.clone(); legacy["version"]=1.into();legacy.as_object_mut().unwrap().remove("output_type");
    let legacy_model:weightlang_runtime::Model=serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(legacy_model.output_type,VectorType::F32);
    assert!(serde_json::to_value(legacy_model).unwrap().get("output_type").is_none());
    legacy["unknown_field"]=true.into();
    assert!(serde_json::from_value::<weightlang_runtime::Model>(legacy).is_err());
}

#[test]
fn loss_conversion_has_softmax_gradient_and_stable_log_softmax_stays_direct() {
    let text=source("logits", "fn loss(prediction,target){ let p=prediction.probabilities; return sum(p*target); }\nfn main(){return 0;}");
    let program=adaptive::parse(&text).unwrap();
    let loss=*program.loss.unwrap();
    let mut session=adaptive::AdaptiveSession::prepare(loss,1_000_000).unwrap();
    let input=[-3.,0.25,4.,0.3,-0.8,1.2];
    let mut pass=session.forward(&input,1000,1_000_000).unwrap();
    pass.backward(&[1.],&Default::default()).unwrap();
    let gradient=pass.input_gradient().unwrap().to_vec();
    for i in 0..3 {
        let mut plus=input; plus[i]+=0.001;
        let mut minus=input; minus[i]-=0.001;
        let numerical=(session.evaluate(&plus,1000).unwrap()[0]-session.evaluate(&minus,1000).unwrap()[0])/0.002;
        assert!((gradient[i]-numerical).abs()<0.0001,"axis {i}: {} vs {numerical}",gradient[i]);
    }
    let direct=source("logits", "fn loss(p,t){return sum(log_softmax(p)*t);}\nfn main(){return 0;}");
    let program=adaptive::parse(&direct).unwrap();
    assert!(program.loss.unwrap().instructions.iter().any(|i|matches!(i,adaptive::Instruction::LogSoftmax{..})));
}
