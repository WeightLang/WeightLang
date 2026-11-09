use std::collections::BTreeMap;
use wgtime_core::adaptive::{AdaptiveSession, Program, parse};
const SOURCE: &str = include_str!("fixtures/attention/model.wgt");
const INPUT: [f32; 6] = [0.7,-0.3,0.2,-0.4,0.8,0.5];
fn session(source: &str, native: bool) -> AdaptiveSession {
    let mut s = AdaptiveSession::prepare(parse(source).unwrap(), 1024*1024).unwrap();
    if native { s.enable_compiled().unwrap(); }
    s
}
// Independent scalar f64 reference: explicit token/channel loops, no runtime tensor helpers.
fn reference(x: &[f32], qk: &[f32], value: &[f32], output: &[f32]) -> Vec<f32> {
    let mut q = [[0f64;2];2]; let mut v = [[0f64;3];2];
    for t in 0..2 { for f in 0..3 {
        for d in 0..2 { q[t][d] += f64::from(x[t*3+f])*f64::from(qk[f*2+d]); }
        for d in 0..3 { v[t][d] += f64::from(x[t*3+f])*f64::from(value[f*3+d]); }
    } }
    let mut result = vec![0f64;6];
    for t in 0..2 {
        let mut scores = [0f64;2];
        for k in 0..2 { for d in 0..2 { scores[k] += q[t][d]*q[k][d]/2f64.sqrt(); } }
        let max = scores[0].max(scores[1]); let e = scores.map(|s|(s-max).exp()); let total = e[0]+e[1];
        for f in 0..3 { for d in 0..3 { for k in 0..2 {
            result[t*3+f] += e[k]/total*v[k][d]*f64::from(output[d*3+f]);
        } } }
    }
    result.into_iter().map(|v|v as f32).collect()
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(),b.len());
    for (i,(a,b)) in a.iter().zip(b).enumerate() { assert!((a-b).abs()<tolerance,"{i}: {a} vs {b}"); }
}
#[test]
fn attention_native_reference_and_shared_parameter_finite_differences() {
    let mut native = session(SOURCE,true);
    let expected = reference(&INPUT,native.parameter("qk").unwrap(),native.parameter("value").unwrap(),native.parameter("output").unwrap());
    close(&native.evaluate(&INPUT,10000).unwrap(),&expected,2e-7);
    let mut interpreted = session(SOURCE,false);
    close(&native.evaluate(&INPUT,10000).unwrap(),&interpreted.evaluate(&INPUT,10000).unwrap(),2e-7);
    let seed = [0.4,-0.8,0.2,0.6,-0.3,0.9];
    let reused = SOURCE.replace("let k = matmul(tokens,projection);", "let k = q;");
    let mut reused = session(&reused,true);
    let mut reused_pass = reused.forward(&INPUT,10000,1024*1024).unwrap();
    reused_pass.backward(&seed,&BTreeMap::new()).unwrap();

    let mut pass = native.forward(&INPUT,10000,1024*1024).unwrap();
    pass.backward(&seed,&BTreeMap::new()).unwrap();
    close(pass.output(),reused_pass.output(),1e-7);
    for name in ["qk","value","output"] { close(pass.parameter_gradient(name).unwrap(),reused_pass.parameter_gradient(name).unwrap(),1e-7); }
    assert!(reused_pass.steps() < pass.steps());
    let dot = |y: Vec<f32>| y.iter().zip(seed).map(|(y,g)|y*g).sum::<f32>();
    for name in ["qk","value","output"] {
        let initial = native.parameter(name).unwrap().to_vec();
        for i in 0..initial.len() {
            let mut plus = initial.clone(); plus[i] += 0.002;
            let mut minus = initial.clone(); minus[i] -= 0.002;
            native.replace_parameter(name,&plus).unwrap(); let p=dot(native.evaluate(&INPUT,10000).unwrap());
            native.replace_parameter(name,&minus).unwrap(); let m=dot(native.evaluate(&INPUT,10000).unwrap());
            let numerical=(p-m)/0.004; let analytical=pass.parameter_gradient(name).unwrap()[i];
            assert!((numerical-analytical).abs()<2e-5,"shared projection {name}[{i}]: {numerical} vs {analytical}");
        }
        native.replace_parameter(name,&initial).unwrap();
    }
    for i in 0..INPUT.len() {
        let mut plus=INPUT;plus[i]+=0.002;let mut minus=INPUT;minus[i]-=0.002;
        let numerical=(dot(native.evaluate(&plus,10000).unwrap())-dot(native.evaluate(&minus,10000).unwrap()))/0.004;
        assert!((numerical-pass.input_gradient().unwrap()[i]).abs()<2e-5);
    }
    assert!(native.compiled_forward_counts().0>40);
    assert_eq!(native.compiled_forward_counts().1,1);
    let start=std::time::Instant::now();
    for _ in 0..1000 { std::hint::black_box(native.evaluate(&INPUT,10000).unwrap()); }
    eprintln!("attention: 1000 forwards {:?}; managed tensor bytes {}; parameters {} bytes; tape {} bytes; charged work {}",start.elapsed(),native.managed_bytes(),native.program().parameters.iter().map(|p|p.data.len()*4).sum::<usize>(),pass.tape_bytes(),pass.steps());
}
#[test]
fn attention_training_descends_and_budget_failure_is_atomic() {
    let mut s = session(SOURCE,true); s.set_optimizer(wgtime_core::training::Optimizer::Sgd).unwrap();
    let targets=vec![vec![0.2,-0.1,0.3,-0.2,0.4,0.1]];
    let loss=|y:Vec<f32>|y.iter().zip(&targets[0]).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/6.;
    let before=loss(s.evaluate(&INPUT,10000).unwrap());
    let weights=s.parameter("qk").unwrap().to_vec();
    assert!(s.train_custom(&[INPUT.to_vec()],&targets,0.2,&mut 5,1024*1024).is_err());
    assert_eq!(weights,s.parameter("qk").unwrap());
    for _ in 0..100 { s.train_custom(&[INPUT.to_vec()],&targets,0.5,&mut 10000,1024*1024).unwrap(); }
    let after=loss(s.evaluate(&INPUT,10000).unwrap());
    eprintln!("attention training MSE {before} -> {after}");
    assert!(after<before*0.7,"{before} -> {after}");
}
#[test]
fn nonlast_axis_softmax_has_independent_groups_and_gradient() {
    let source="wgt 0.5\nmodel Axis\ninput x:f32[6]\noutput y:f32[6]\nfn forward(x){return reshape(softmax(reshape(x,[2,3]),0),[6]);}\n";
    let input=[1.,2.,3.,3.,1.,-1.];let seed=[0.3,-0.4,0.8,0.2,0.9,-0.7];
    for native in [false,true] {
        let mut s=session(source,native);let y=s.evaluate(&input,100).unwrap();
        for j in 0..3 { assert!((y[j]+y[3+j]-1.).abs()<1e-7);assert!((y[j]-1./(1.+(input[3+j]-input[j]).exp())).abs()<1e-7); }
        let mut pass=s.forward(&input,100,65536).unwrap();pass.backward(&seed,&BTreeMap::new()).unwrap();
        for i in 0..6 {
            let mut p=input;p[i]+=0.002;let mut m=input;m[i]-=0.002;
            let yp=s.evaluate(&p,100).unwrap();let ym=s.evaluate(&m,100).unwrap();
            let numerical=yp.iter().zip(ym).zip(seed).map(|((p,m),g)|(p-m)*g/0.004).sum::<f32>();
            assert!((numerical-pass.input_gradient().unwrap()[i]).abs()<3e-5);
        }
    }
}
#[test]
fn tensor_shape_and_serialized_ir_contracts_are_strict() {
    for expression in ["reshape(x,[2,2])","matmul(reshape(x,[2,3]),reshape(x,[2,3]))","reshape(x,[2,3])+reshape(x,[3,2])","softmax(reshape(x,[2,3]))","softmax(reshape(x,[2,3]),-1)","softmax(reshape(x,[2,3]),2)","transpose(x)","scale(x,reshape([2],[1,1]))"] {
        let source=format!("wgt 0.5\nmodel Invalid\ninput x:f32[6]\noutput y:f32[6]\nfn forward(x){{return {expression};}}\n");
        assert!(parse(&source).is_err(),"accepted {expression}");
    }
    let program=parse(SOURCE).unwrap(); assert_eq!(program.version,3);
    let encoded=serde_json::to_value(&program).unwrap();
    let roundtrip:Program=serde_json::from_value(encoded.clone()).unwrap();
    AdaptiveSession::prepare(roundtrip,1024*1024).unwrap();
    for version in [1,2,4] { let mut p=program.clone();p.version=version;assert!(AdaptiveSession::prepare(p,1024*1024).is_err()); }
    let mut bad=encoded.clone();bad["registers"][0].as_object_mut().unwrap().remove("shape");
    assert!(serde_json::from_value::<Program>(bad).is_err());
    let mut bad=program.clone();bad.registers[0].shape=vec![2,2];assert!(AdaptiveSession::prepare(bad,1024*1024).is_err());
    let mut bad=program.clone();let r=bad.registers.iter_mut().find(|r|r.name=="q").unwrap();r.shape=vec![1,4];assert!(AdaptiveSession::prepare(bad,1024*1024).is_err());
    assert!(AdaptiveSession::prepare(program,1).is_err());
    let mut s=session(SOURCE,true);assert!(s.evaluate(&[f32::MAX;6],10000).is_err());
}
