use std::{fs,process::Command,time::{Duration,Instant}};
use wgt::{Client,InstanceOptions,ModelDefinition,Optimizer,Loss,Targets,TrainOptions};
const SOURCE:&str=include_str!("fixtures/attention/model.wgt");
const INPUT:[f32;6]=[0.7,-0.3,0.2,-0.4,0.8,0.5];
#[test]
fn attention_macro_and_file_share_geometry() {
    let embedded=wgt::include_model!("tests/fixtures/attention/model.wgt");
    let parsed=ModelDefinition::parse(SOURCE).unwrap();
    assert_eq!(embedded.parameters()[0].shape,[3,2]);
    assert_eq!(parsed.interface().input,6);
    assert_eq!(parsed.interface().output,6);
    let root=std::env::var_os("CLAUDE_JOB_DIR").map(std::path::PathBuf::from).map(|p|p.join("tmp")).unwrap_or_else(||"/tmp/weightlang".into());fs::create_dir_all(&root).unwrap();
    let temp=tempfile::tempdir_in(&root).unwrap();
    fs::write(temp.path().join("ops.wgt"),"wgt 0.5\noperators Geometry\nfn matrix(x){return reshape(x,[2,3]);}\n").unwrap();
    let imported=SOURCE.replace("model TinyAttention","import geometry from \"./ops.wgt\"\nmodel TinyAttention").replace("reshape(x,[2,3])","geometry.matrix(x)");
    fs::write(temp.path().join("model.wgt"),imported).unwrap();
    let imported=ModelDefinition::from_file(temp.path().join("model.wgt")).unwrap();
    assert_eq!(imported.interface().output,6);

}
#[test]
#[ignore="requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn attention_service_compiles_trains_and_roundtrips_weights() {
    use std::os::unix::fs::PermissionsExt;
    struct Daemon(std::process::Child);
    impl Drop for Daemon {fn drop(&mut self){let _=self.0.kill();let _=self.0.wait();}}
    let root=std::env::var_os("CLAUDE_JOB_DIR").map(std::path::PathBuf::from).map(|p|p.join("tmp")).unwrap_or_else(||"/tmp/weightlang".into());fs::create_dir_all(&root).unwrap();
    let scratch=tempfile::Builder::new().prefix("attention-").tempdir_in(&root).unwrap();
    fs::set_permissions(scratch.path(),fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon=Daemon(Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(scratch.path()).spawn().unwrap());
    let endpoint=scratch.path().join("service.sock");let start=Instant::now();
    while !endpoint.exists(){assert!(start.elapsed()<Duration::from_secs(10));std::thread::sleep(Duration::from_millis(5));}
    let client=Client::connect_to(&endpoint).unwrap();
    let definition=wgt::include_model!("tests/fixtures/attention/model.wgt");
    let path=scratch.path().join("model.wgt");fs::write(&path,SOURCE).unwrap();
    let runtime=ModelDefinition::from_file(&path).unwrap();fs::remove_file(path).unwrap();
    let options=InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()};
    let model=client.initialize(&definition,options).unwrap();assert_eq!(model.backend(),"native-cpu");
    let initial=model.forward(&INPUT).unwrap();
    let target=vec![0.2,-0.1,0.3,-0.2,0.4,0.1];
    let mse=|y:&[f32]|y.iter().zip(&target).map(|(a,b)|(a-b).powi(2)).sum::<f32>()/6.;
    let before=mse(&initial);
    for _ in 0..100 {model.train(&[INPUT.to_vec()],Targets::Values(&[target.clone()]),TrainOptions{rate:0.5,loss:Loss::Custom,clip:None}).unwrap();}
    let trained=model.forward(&INPUT).unwrap();let after=mse(&trained);assert!(after<before*0.7);
    let weights=scratch.path().join("weights.safetensors");model.save_weights(&weights).unwrap();
    let restored=client.initialize(&runtime,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    restored.load_weights(&weights,Some(Optimizer::Sgd)).unwrap();assert_eq!(trained,restored.forward(&INPUT).unwrap());
    for net in [&model,&restored] {net.train(&[INPUT.to_vec()],Targets::Values(&[target.clone()]),TrainOptions{rate:0.1,loss:Loss::Custom,clip:None}).unwrap();}
    assert_eq!(model.forward(&INPUT).unwrap(),restored.forward(&INPUT).unwrap());
    let stats=client.stats().unwrap();
    for execution in stats["execution"].as_array().unwrap() {assert_eq!(execution["backend"],"native-cpu");assert!(execution["native_tape_forwards"].as_u64().unwrap()>0);}
    eprintln!("real service attention MSE {before} -> {after}; backend native-cpu; safetensors roundtrip and continued SGD identical");
    restored.close().unwrap();model.close().unwrap();assert_eq!(client.stats().unwrap()["sessions"],0);
    assert_eq!(client.stats().unwrap()["reserved"],0);
}
