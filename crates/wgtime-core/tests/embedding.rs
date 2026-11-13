//! Embedding padding changes only the table VJP, never forward row values.
use std::collections::BTreeMap;
use wgt_language::{adaptive::{Instruction, Program}, model::{ModelDefinition, ModelPlan}};
use wgtime_core::adaptive::AdaptiveSession;

fn plan(source: &str) -> Program {
    let ModelPlan::Forward(program) = ModelDefinition::parse(source).unwrap().instantiate_plan().unwrap() else { panic!("forward expected") };
    program
}
fn source(padding: &str) -> String {
    format!("wgt 0.5\nmodel Embedding\ninput ids:f32[4]\noutput y:f32[8]\nparameter table:f32[3,2]=[11,12,21,22,31,32]\nfn forward(ids) {{ return embedding(table as [3,2],ids{padding}) as [8]; }}")
}

#[test]
fn nonzero_padding_rows_are_returned_but_have_no_gradient() {
    for compiled in [false, true] {
        for padding in 0..3 {
            let mut session = AdaptiveSession::prepare(plan(&source(&format!(",{padding}"))),1<<20).unwrap();
            if compiled { session.enable_compiled().unwrap(); assert!(!session.is_compiled()); }
            let other = (padding + 1) % 3;
            let ids = [padding as f32,other as f32,other as f32,padding as f32];
            let mut pass = session.forward(&ids,1000,65536).unwrap();
            let table = [11.,12.,21.,22.,31.,32.];
            let expected: Vec<f32> = ids.iter().flat_map(|&id| table[2*id as usize..2*id as usize+2].iter().copied()).collect();
            assert_eq!(pass.output(),expected);
            pass.backward(&[1.,2.,3.,4.,5.,6.,7.,8.],&BTreeMap::new()).unwrap();
            let mut gradient = vec![0.;6];
            gradient[2*other]=8.; gradient[2*other+1]=10.;
            assert_eq!(pass.parameter_gradient("table").unwrap(),gradient);
            assert_eq!(pass.input_gradient().unwrap(),&[0.;4]);
            assert!(session.forward(&ids,1000,1).is_err());
        }
    }
}

#[test]
fn omitted_padding_means_no_padding_and_matches_generic_gather() {
    for suffix in [""] {
        let program=plan(&source(suffix));
        let instruction=program.instructions.iter().find(|i|matches!(i,Instruction::Embedding{..})).unwrap();
        assert!(matches!(instruction,Instruction::Embedding{padding_idx:None,..}));
        let mut session=AdaptiveSession::prepare(program,1<<20).unwrap();
        let mut pass=session.forward(&[0.,2.,2.,0.],1000,65536).unwrap();
        pass.backward(&[1.;8],&BTreeMap::new()).unwrap();
        assert_eq!(pass.parameter_gradient("table").unwrap(),&[2.,2.,0.,0.,2.,2.]);
    }
    let generic=source("").replace("embedding(","gather(");
    let mut pass=AdaptiveSession::prepare(plan(&generic),1<<20).unwrap().forward(&[0.,2.,2.,0.],1000,65536).unwrap();
    pass.backward(&[1.;8],&BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("table").unwrap(),&[2.,2.,0.,0.,2.,2.]);
}

#[test]
fn embedding_supports_runtime_table_values_and_strided_tables() {
    let text="wgt 0.5\nmodel TableInput\ninput x:f32[6]\noutput y:f32[6]\nparameter ids:f32[3]=[2,0,2]\nfn forward(x) { return embedding((x as [2,3]).T,ids,0) as [6]; }";
    let mut session=AdaptiveSession::prepare(plan(text),1<<20).unwrap();
    let mut pass=session.forward(&[1.,2.,3.,4.,5.,6.],1000,65536).unwrap();
    assert_eq!(pass.output(),&[3.,6.,1.,4.,3.,6.]);
    pass.backward(&[1.;6],&BTreeMap::new()).unwrap();
    assert_eq!(pass.input_gradient().unwrap(),&[0.,0.,2.,0.,0.,2.]);
    assert_eq!(pass.parameter_gradient("ids").unwrap(),&[0.;3]);
}

#[test]
fn rank_two_singleton_table_and_width_are_supported() {
    let text="wgt 0.5\nmodel Singleton\ninput ids:f32[2]\noutput y:f32[2]\nparameter table:f32[1,1]=[7]\nfn forward(ids) { return embedding(table as [1,1],ids,0) as [2]; }";
    let mut session=AdaptiveSession::prepare(plan(text),1<<20).unwrap();
    let mut pass=session.forward(&[0.,0.],1000,65536).unwrap();
    assert_eq!(pass.output(),&[7.,7.]);
    pass.backward(&[1.,1.],&BTreeMap::new()).unwrap();
    assert_eq!(pass.parameter_gradient("table").unwrap(),&[0.]);
}

#[test]
fn source_and_untrusted_ir_enforce_padding_rank_and_version_contracts() {
    for suffix in [",-4",",3",",0.5",",[0]",",ids",",1+1"] {
        assert!(ModelDefinition::parse(&source(suffix)).and_then(|d|d.instantiate_plan()).is_err(),"{suffix}");
    }
    for invalid in [source("").replace("table as [3,2]","table"),source("").replace(",ids)",",ids as [2,2])")] {
        assert!(ModelDefinition::parse(&invalid).and_then(|d|d.instantiate_plan()).is_err());
    }
    let program=plan(&source(",0"));
    assert_eq!(program.version,3);
    let mut old=program.clone(); old.version=2;
    assert!(old.validate_tensor_contracts().is_err());
    let mut encoded=serde_json::to_value(&program).unwrap();
    let embedding=encoded["instructions"].as_array_mut().unwrap().iter_mut().find(|i|i["op"]=="embedding").unwrap();
    embedding.as_object_mut().unwrap().remove("padding_idx");
    let restored:Program=serde_json::from_value(encoded).unwrap();
    restored.validate_tensor_contracts().unwrap();
    assert!(restored.instructions.iter().any(|i|matches!(i,Instruction::Embedding{padding_idx:None,..})));
    let mut invalid=program;
    for instruction in &mut invalid.instructions {
        if let Instruction::Embedding { padding_idx,.. }=instruction { *padding_idx=Some(3); }
    }
    assert!(invalid.validate_tensor_contracts().is_err());
}

#[test]
fn runtime_indices_keep_the_gather_bounds_contract() {
    let mut session=AdaptiveSession::prepare(plan(&source(",0")),1<<20).unwrap();
    for value in [-1.,0.5,3.,f32::INFINITY,f32::NAN] {
        assert!(session.evaluate(&[0.,value,2.,0.],1000).is_err());
    }
    assert!(session.evaluate(&[0.,1.,2.,0.],1000).is_ok());
}

#[test]
fn negative_padding_normalizes_to_the_equivalent_positive_row() {
    for (negative, positive) in [(-3,0),(-2,1),(-1,2)] {
        let program=plan(&source(&format!(",{negative}")));
        assert!(program.instructions.iter().any(|i| matches!(i,
            Instruction::Embedding { padding_idx: Some(row), .. } if *row == positive)));
        let mut a=AdaptiveSession::prepare(program,1<<20).unwrap();
        let mut b=AdaptiveSession::prepare(plan(&source(&format!(",{positive}"))),1<<20).unwrap();
        let mut a=a.forward(&[0.,2.,2.,0.],1000,65536).unwrap();
        let mut b=b.forward(&[0.,2.,2.,0.],1000,65536).unwrap();
        assert_eq!(a.output(),b.output());
        a.backward(&[1.;8],&BTreeMap::new()).unwrap();
        b.backward(&[1.;8],&BTreeMap::new()).unwrap();
        assert_eq!(a.parameter_gradient("table").unwrap(),b.parameter_gradient("table").unwrap());
        assert_eq!(a.input_gradient().unwrap(),&[0.;4]);
    }
}
