//! Runtime f32 indexing shares one frontend and never differentiates addresses.
use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::AdaptiveSession;

fn prepare(source: &str, compiled: bool) -> AdaptiveSession {
    let ModelPlan::Forward(plan) = ModelDefinition::parse(source).unwrap().instantiate_plan().unwrap() else { panic!("forward expected") };
    assert_eq!(plan.version, 3);
    let mut old = plan.clone();
    old.version = 2;
    assert!(old.validate_tensor_contracts().is_err());
    let mut session = AdaptiveSession::prepare(plan, 1 << 20).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
        assert!(!session.is_compiled(), "indexing explicitly falls back to interpreter");
    }
    session
}

const GATHER: &str = "wgt 0.5\nmodel Index\ninput x:f32[6]\noutput y:f32[6]\nparameter ids:f32[3]=[2,0,2]\nfn forward(x) { return gather(x as [2,3],ids,1) as [6]; }\n";

#[test]
fn gather_nonleading_axis_repeats_scatter_add_without_index_gradient() {
    for compiled in [false, true] {
        let mut session = prepare(GATHER, compiled);
        let mut pass = session.forward(&[1.,2.,3.,4.,5.,6.], 1000, 65536).unwrap();
        assert_eq!(pass.output(), &[3.,1.,3.,6.,4.,6.]);
        pass.backward(&[1.,2.,3.,4.,5.,6.], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[2.,0.,4.,5.,0.,10.]);
        assert_eq!(pass.parameter_gradient("ids").unwrap(), &[0.,0.,0.]);
        // The tape owns the old address value version even after a later binding.
        session.replace_parameter("ids", &[1.,1.,1.]).unwrap();
        pass.backward(&[1.;6], &BTreeMap::new()).unwrap();
        assert_eq!(pass.input_gradient().unwrap(), &[1.,0.,2.,1.,0.,2.]);
        assert!(session.forward(&[1.;6], 1000, 1).is_err(), "tape admission remains budgeted");
    }
}

#[test]
fn gather_validates_runtime_addresses_before_indexing() {
    let mut session = prepare(GATHER, false);
    for bad in [-1., 0.5, 3., 16_777_216.] {
        session.replace_parameter("ids", &[0.,bad,1.]).unwrap();
        assert!(session.evaluate(&[1.;6], 1000).is_err());
    }
    session.replace_parameter("ids", &[0.,1.,2.]).unwrap();
    assert_eq!(session.evaluate(&[1.,2.,3.,4.,5.,6.],1000).unwrap(), vec![1.,2.,3.,4.,5.,6.]);
}

#[test]
fn runtime_slice_end_boundary_and_start_stop_gradient() {
    let text = "wgt 0.5\nmodel Slice\ninput x:f32[5]\noutput y:f32[2]\nparameter start:f32[1]=[3]\nfn forward(x) { return dynamic_slice(x,start,2); }";
    let mut session = prepare(text, true);
    let mut pass = session.forward(&[1.,2.,3.,4.,5.],1000,65536).unwrap();
    assert_eq!(pass.output(), &[4.,5.]);
    pass.backward(&[2.,3.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[0.,0.,0.,2.,3.]);
    assert_eq!(pass.parameter_gradient("start").unwrap(), &[0.]);
    for bad in [-1.,0.5,4.] {
        session.replace_parameter("start", &[bad]).unwrap();
        assert!(session.evaluate(&[1.;5],1000).is_err());
    }
}

#[test]
fn concat_axis_one_and_alias_vjp() {
    let text = "wgt 0.5\nmodel Cat\ninput x:f32[4]\noutput y:f32[8]\nfn forward(x) { let a=x as [2,2]; return concat(a,a,1) as [8]; }";
    let mut session = prepare(text,true);
    let mut pass = session.forward(&[1.,2.,3.,4.],1000,65536).unwrap();
    assert_eq!(pass.output(), &[1.,2.,1.,2.,3.,4.,3.,4.]);
    pass.backward(&[1.,2.,3.,4.,5.,6.,7.,8.], &BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(), &[4.,6.,12.,14.]);
}

#[test]
fn gather_strided_input_and_rank_eight() {
    let text=GATHER.replace("gather(x as [2,3],ids,1)", "gather((x as [3,2]).T,ids,1)");
    let mut session=prepare(&text,false);
    assert_eq!(session.evaluate(&[1.,2.,3.,4.,5.,6.],1000).unwrap(),vec![5.,1.,5.,6.,2.,6.]);
    let text="wgt 0.5\nmodel RankEight\ninput x:f32[2]\noutput y:f32[3]\nfn forward(x) { return gather(x as [1,1,1,1,1,1,1,2],[1,0,1],7) as [3]; }";
    assert_eq!(prepare(text,false).evaluate(&[2.,3.],1000).unwrap(),vec![3.,2.,3.]);
}

#[test]
fn malformed_source_contracts_are_rejected() {
    for expression in ["gather(x,[0],2)", "gather(x,[0] as [1,1])", "gather(x,[0],0.5)", "dynamic_slice(x,[0,1],2)", "dynamic_slice(x,[0],0)", "dynamic_slice(x,[0],7)", "concat(x,x,1)", "concat(x as [2,3],x as [3,2],0)"] {
        let source=format!("wgt 0.5\nmodel Bad\ninput x:f32[6]\noutput y:f32[6]\nfn forward(x) {{ return {expression}; }}");
        assert!(ModelDefinition::parse(&source).and_then(|d| d.instantiate_plan()).is_err(),"{expression}");
    }
}
