use std::collections::BTreeMap;
use wgtime_core::adaptive::{self, AdaptiveSession};

const CNN: &str = include_str!("fixtures/cnn_probabilities/model.wgt");
fn session(source: &str, native: bool) -> AdaptiveSession {
    let mut s = AdaptiveSession::prepare(adaptive::parse(source).unwrap(), 1 << 20).unwrap();
    if native { s.enable_compiled().unwrap(); }
    s
}
fn vector(body: &str, kind: &str) -> String {
    format!("wgt 0.5\nmodel P\ninput x:f32[2]\noutput y:{kind}[2]\nfn forward(x){{return {body};}}")
}

#[test]
fn cnn_native_softmax_equivalence_gradients_and_training() {
    let explicit = CNN.replace("linear(flat,weight,offset).probabilities.probabilities", "softmax(linear(flat,weight,offset))");
    let mut reference = session(&explicit, false);
    let mut native = session(CNN, true);
    let input = [-0.8,0.2,0.7,-0.3,0.5,0.9,0.1,-0.4,0.6];
    let expected = reference.evaluate(&input, 100000).unwrap();
    let actual = native.evaluate(&input, 100000).unwrap();
    for (a,b) in actual.iter().zip(expected) { assert!((a-b).abs()<1e-7); }
    assert!(native.compiled_forward_counts().0 > 0);
    let seed = [0.3,-0.8];
    let mut pass = native.forward(&input,100000,1<<20).unwrap();
    pass.backward(&seed,&BTreeMap::new()).unwrap();
    let dot = |v: Vec<f32>| v.iter().zip(seed).map(|(v,g)| f64::from(*v)*f64::from(g)).sum::<f64>();
    for name in ["kernel","bias","weight","offset"] {
        let original = native.parameter(name).unwrap().to_vec();
        for i in 0..original.len() {
            let mut p=original.clone(); p[i]+=0.002;
            native.replace_parameter(name,&p).unwrap();
            let plus=dot(native.evaluate(&input,100000).unwrap());
            p[i]=original[i]-0.002;
            native.replace_parameter(name,&p).unwrap();
            let minus=dot(native.evaluate(&input,100000).unwrap());
            let analytic=pass.parameter_gradient(name).unwrap()[i];
            assert!(analytic.is_finite());
            assert!(((plus-minus)/0.004-f64::from(analytic)).abs()<5e-5,"{name}[{i}]");
        }
        native.replace_parameter(name,&original).unwrap();
    }
    for i in 0..input.len() {
        let mut p=input; p[i]+=0.002;
        let mut m=input; m[i]-=0.002;
        let numerical=(dot(native.evaluate(&p,100000).unwrap())-dot(native.evaluate(&m,100000).unwrap()))/0.004;
        assert!((numerical-f64::from(pass.input_gradient().unwrap()[i])).abs()<5e-5);
    }
    native.set_optimizer(wgtime_core::training::Optimizer::Sgd).unwrap();
    let objective=|v:Vec<f32>| ((v[0]-1.).powi(2)+v[1].powi(2))/2.;
    let before=objective(actual);
    for _ in 0..80 { native.train_custom(&[input.to_vec()],&[vec![1.,0.]],0.3,&mut 100000,1<<20).unwrap(); }
    let after=objective(native.evaluate(&input,100000).unwrap());
    assert!(after<before*0.1,"{before} -> {after}");
}

#[test]
fn vector_contract_type_preservation_and_invalid_shapes() {
    for native in [false,true] {
        let mut s=session(&vector("x.probabilities.probabilities","probabilities"),native);
        for input in [[0.2,0.8],[f32::MAX,-f32::MAX],[1000.,1000.]] {
            let expected=wgtime_core::categorical::softmax(&input.map(f64::from)).unwrap();
            let actual=s.evaluate(&input,1000).unwrap();
            for (a,b) in actual.iter().zip(expected) { assert!((f64::from(*a)-b).abs()<1e-7); }
        }
        for input in [[f32::NAN,0.],[0.,f32::INFINITY],[f32::NEG_INFINITY,0.]] { assert!(s.evaluate(&input,1000).is_err()); }
        assert!(s.evaluate(&[0.],1000).is_err());
        let loss="\nfn loss(p,t){return sum(p.probabilities.probabilities*t);}";
        let program=adaptive::parse(&(vector("x","probabilities")+loss)).unwrap();
        let mut s=AdaptiveSession::prepare(*program.loss.unwrap(),1<<20).unwrap();
        if native { s.enable_compiled().unwrap(); }
        let mut p=s.forward(&[0.2,0.800001,0.3,-0.8],1000,1<<20).unwrap();
        p.backward(&[1.],&BTreeMap::new()).unwrap();
        assert_eq!(&p.input_gradient().unwrap()[..2],&[0.3,-0.8]);
        let mut declared=session(&vector("x","probabilities"),native);
        assert_eq!(declared.evaluate(&[0.2,0.800001],1000).unwrap(),[0.2,0.800001]);
        for input in [[0.2,0.2],[-0.1,1.1]] { assert!(declared.evaluate(&input,1000).is_err()); }
    }
    for body in ["x as [1,2].probabilities", "softmax(x as [1,2],1).probabilities"] {
        let error=adaptive::parse(&vector(body,"probabilities")).unwrap_err();
        assert!(format!("{error:#}").contains("rank-1"),"{error:#}");
    }
}

#[test]
fn pure_helper_and_import_preserve_probability_type() {
    let helper="fn convert(x){return x.probabilities;}";
    let local=vector("convert(x).probabilities","probabilities")+helper;
    let expected=session(&vector("softmax(x)","probabilities"),true).evaluate(&[0.2,0.8],1000).unwrap();
    assert_eq!(session(&local,true).evaluate(&[0.2,0.8],1000).unwrap(),expected);
    let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/probability_helper/model.wgt");
    let definition=wgt_language::model::ModelDefinition::from_file(root).unwrap();
    assert_eq!(session(definition.source(),true).evaluate(&[0.2,0.8],1000).unwrap(),expected);
}

#[test]
fn ordinary_lists_use_same_explicit_conversion() {
    let source="wgt 0.5\nfn identity(x){return x;}\nfn main(){let p=[0.2,0.8].probabilities; assert p==identity(p).probabilities.probabilities; return p;}";
    let report=wgtime_core::script::run(source,std::path::Path::new("."),10000).unwrap();
    assert_eq!(report.result,serde_json::json!(wgtime_core::categorical::softmax(&[0.2,0.8]).unwrap()));
    for value in ["[]","[[0.2,0.8]]","[true,false]","0.2"] {
        let source=format!("wgt 0.5\nfn main(){{return ({value}).probabilities;}}");
        assert!(wgtime_core::script::run(&source,std::path::Path::new("."),10000).is_err());
    }
}
