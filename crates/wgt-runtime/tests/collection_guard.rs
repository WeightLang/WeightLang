use weightlang_runtime::{adaptive::{self, AdaptiveSession}, training::Optimizer, weightfs::Weights};
const MODEL:&str="wgt 0.5\nmodel Policy\ninput x: f32[1]\noutput y: f32[1]\nparameter w: f32[1] = [1]\nfn forward(x){return w*x;}\nfn loss(p,t){return sum(square(p-t));}\n";
fn session()->AdaptiveSession {AdaptiveSession::prepare(adaptive::parse(MODEL).unwrap(),1<<20).unwrap()}

#[test]
fn collection_is_identity_scoped_and_drop_releases_all_mutations() {
 let mut model=session();model.set_optimizer(Optimizer::Sgd).unwrap();
 let token=model.begin_collection().unwrap();
 assert!(model.begin_collection().is_err());
 assert_eq!(model.run(&[2.],1000).unwrap(),[2.]);
 assert_eq!(model.evaluate(&[2.],1000).unwrap(),[2.]);
 let revision=model.revision();
 assert!(model.replace_parameter("w",&[2.]).is_err());
 // This is the actual weightfs backend trait dispatch, not a runtime builtin.
 assert!(Weights::replace(&mut model,"w",&[2.]).is_err());
 assert!(model.set_optimizer(Optimizer::Adam).is_err());
 assert!(model.train_custom(&[vec![1.]],[vec![0.]].as_slice(),0.1,&mut 10000,1<<20).is_err());
 assert_eq!(model.revision(),revision);
 drop(token);Weights::replace(&mut model,"w",&[2.]).unwrap();
 assert_eq!(model.run(&[2.],1000).unwrap(),[4.]);
}

#[test]
fn custom_gradients_remain_read_only_but_commit_is_protected() {
 let mut model=session();
 let snapshot=model.differentiate_custom(&[vec![1.]],[vec![0.]].as_slice(),&mut 10000,1<<20).unwrap();
 let token=model.begin_collection().unwrap();
 let trial=model.differentiate_at_custom(&snapshot,&[vec![2.]],&mut 10000,1<<20).unwrap();
 assert_eq!(model.parameter("w").unwrap(),[1.]);
 assert!(model.commit_custom(&trial,vec![vec![3.]],serde_json::Value::Null).is_err());
 assert_eq!(model.parameter("w").unwrap(),[1.]);
 drop(token);model.commit_custom(&trial,vec![vec![3.]],serde_json::Value::Null).unwrap();
}

#[test]
fn explicit_forward_writes_are_rejected_and_evaluate_discards_them() {
 let source=MODEL.replace("return w*x;","write(w,w+x);return w;");
 let mut model=AdaptiveSession::prepare(adaptive::parse(&source).unwrap(),1<<20).unwrap();
 let token=model.begin_collection().unwrap();
 assert!(model.run(&[1.],1000).is_err());
 assert_eq!(model.evaluate(&[1.],1000).unwrap(),[2.]);
 assert_eq!(model.parameter("w").unwrap(),[1.]);
 drop(token);model.run(&[1.],1000).unwrap();
 assert_eq!(model.parameter("w").unwrap(),[2.]);
}
