use wgtime_core::{adaptive::{self, AdaptiveSession}, training::Optimizer};
const BUDGET: usize = 8 * 1024 * 1024;
fn source(declaration: &str, forward: &str) -> String {
    format!("wgt 0.5\nmodel Learn\ninput x:f32[1]\noutput y:f32[1]\n{declaration}\nfn forward(x){{{forward}}}\nfn loss(prediction,target){{return [0.5]*square(prediction-target);}}\n")
}
fn session(text: &str, compiled: bool) -> AdaptiveSession {
    let mut s=AdaptiveSession::prepare(adaptive::parse(text).unwrap(), BUDGET).unwrap();
    if compiled { s.enable_compiled().unwrap(); }
    s.set_optimizer(Optimizer::Sgd).unwrap(); s
}
fn step(s: &mut AdaptiveSession, target: f32, rate: f32) {
    s.train_custom(&[vec![1.]], &[vec![target]], rate, &mut 100000, BUDGET).unwrap();
}
#[test]
fn float_and_projection_use_parameter_gradients() {
    for compiled in [false,true] {
        let mut s=session(&source("learn scale: float = 5.0;", "return x*scale;"),compiled);
        step(&mut s,10.,0.1); assert_eq!(s.program().parameters[0].data,[5.5]);
        let mut s=session(&source("learn scale: float in 0.0..=1.0 = 0.5;", "return x*scale;"),compiled);
        step(&mut s,10.,1.); assert_eq!(s.program().parameters[0].data,[1.]);
        let mut s=session(&source("learn scale: float > 0.0 = 1.0;", "return x*scale;"),compiled);
        step(&mut s,-10.,1.); assert_eq!(s.program().parameters[0].data,[f32::from_bits(1)]);
    }
}
#[test]
fn discrete_hard_primals_and_ste_cross_thresholds() {
    for compiled in [false,true] {
        let mut s=session(&source("learn scale: int in 1..=12 = 5;", "return square(scale)*x;"),compiled);
        // Hard primal 5^2=25, gradient (25-26)*2*5=-10.
        step(&mut s,26.,0.06); assert!((s.program().parameters[0].data[0]-5.6).abs()<1e-6);
        assert_eq!(s.run(&[1.],100000).unwrap(),[36.]);
        step(&mut s,36.,0.1); assert!((s.program().parameters[0].data[0]-5.6).abs()<1e-6);
        let mut s=session(&source("learn scale: bool = false;", "return mix scale {then {x} else {[0]}};"),compiled);
        for _ in 0..5 {step(&mut s,1.,0.1);}
        assert_eq!(s.run(&[1.],100000).unwrap(),[1.]);
        assert!((s.program().parameters[0].data[0]-0.5).abs()<1e-6);
        step(&mut s,1.,0.1); assert!((s.program().parameters[0].data[0]-0.5).abs()<1e-6);
    }
}
#[test]
fn shared_cell_bounded_depth_learns_more_iterations() {
    let text=source("learn depth: int in 1..=3 = 1;\nfrozen w:f32[1]=[2]", "let a=route(depth_gate(depth,1),x*w,x); let b=route(depth_gate(depth,2),a*w,a); return route(depth_gate(depth,3),b*w,b);");
    for compiled in [false,true] {
        let mut s=session(&text,compiled);
        step(&mut s,4.,0.1);
        assert!((s.program().parameters[0].data[0]-2.0).abs()<1e-5);
        assert_eq!(s.run(&[1.],100000).unwrap(),[4.]);
        step(&mut s,4.,0.1);
        assert!((s.program().parameters[0].data[0]-2.0).abs()<1e-5);
        if compiled {assert_eq!(s.compiled_forward_counts().1,2);}
    }
}
#[test]
fn shadow_restore_round_ties_and_atomic_domain_rejection() {
    for compiled in [false,true] {
        let text=source("learn scale: int in -12..=12 = -5;", "return x*scale;");
        let mut s=session(&text,compiled);
        s.replace_parameter("scale",&[-5.5]).unwrap();
        assert_eq!(s.run(&[1.],100000).unwrap(),[-6.]);
        let bytes=s.weights_bytes(BUDGET).unwrap();
        let mut restored=session(&text,compiled);
        restored.bind_weights(&bytes,"sgd").unwrap();
        assert_eq!(restored.parameter("scale").unwrap(),[-5.5]);
        assert_eq!(restored.run(&[1.],100000).unwrap(),[-6.]);
        let value=serde_json::from_slice(&s.checkpoint_bytes_bounded(BUDGET).unwrap()).unwrap();
        let mut checkpoint=AdaptiveSession::restore_checkpoint(value,BUDGET).unwrap();
        assert_eq!(checkpoint.parameter("scale").unwrap(),[-5.5]);
        step(&mut s,-4.,0.1); step(&mut checkpoint,-4.,0.1);
        assert_eq!(checkpoint.parameter("scale").unwrap(),s.parameter("scale").unwrap());
        let bad=wgtime_core::weight_artifact::encode(&[("scale".into(),vec![1],vec![13.])],BUDGET).unwrap();
        let before=restored.parameter("scale").unwrap().to_vec();
        assert!(restored.bind_weights(&bad,"sgd").is_err());
        assert_eq!(restored.parameter("scale").unwrap(),before);
        assert!(restored.replace_parameter("scale",&[13.]).is_err());
        let mut positive=session(&source("learn scale: float > -0.0 = 1;", "return x*scale;"),compiled);
        step(&mut positive,-10.,1.);
        assert_eq!(positive.parameter("scale").unwrap(),[f32::from_bits(1)]);
    }
}
#[test]
fn syntax_tour_compiles_and_executes() {
    let definition = wgt_language::model::ModelDefinition::parse(include_str!("../../../language/syntax-tour/model.wgt")).unwrap();
    let mut model = session(definition.source(), true);
    assert_eq!(model.run(&[2., 4.], 100000).unwrap(), [9.]);
    assert_eq!(model.run(&[0., 0.], 100000).unwrap(), [0.]);
}

#[test]
fn invalid_domains_and_values_are_rejected() {
    for declaration in ["learn a: float = NaN;", "learn a: float = 1e100;", "learn a: float in 1..=0 = 0.5;", "learn a: float in 0..=1 = 2;", "learn a: float > 0 = 0;", "learn a: int in 1..=3 = 1.5;", "learn a: int in 1..=3.5 = 2;", "learn a: int in 1..=16777217 = 2;", "learn a: int in 1..=16777216 = 16777217;", "learn a: bool = 1;", "learn a: choice {relu,tanh} = relu;"] {
        assert!(adaptive::parse(&source(declaration,"return x;")).is_err(),"{declaration}");
    }
    let mut p=adaptive::parse(&source("learn a: float in 0..=1 = 0.5;","return x*a;")).unwrap();
    p.parameters[0].data[0]=2.; assert!(AdaptiveSession::prepare(p,BUDGET).is_err());
}
