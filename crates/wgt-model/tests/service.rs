use std::{fs,process::Command,time::{Duration,Instant}};
use wgt::{Client,InstanceOptions,ModelDefinition,Optimizer,Loss,Targets,TrainOptions};
const SOURCE:&str=include_str!("fixtures/model.wgt");

#[test]
fn definitions_are_pure_and_macros_use_actual_facade() {
    let inline=wgt::model!(r#"wgt 0.5
model Double
input x:f32[1]
output y:f32[1]
fn forward(x){return x+x;}
"#);
    let file=wgt::include_model!("tests/fixtures/model.wgt");
    assert_eq!(inline.interface().input,1);
    assert_eq!(file.interface().output,1);
    assert!(ModelDefinition::parse(SOURCE).is_ok());
}

#[test]
#[ignore="requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn three_definition_entries_execute_and_resources_release() {
    use std::os::unix::fs::PermissionsExt;
    struct Daemon(std::process::Child);
    impl Drop for Daemon {fn drop(&mut self){let _=self.0.kill();let _=self.0.wait();}}
    let scratch_root=std::env::var_os("CLAUDE_JOB_DIR").map(std::path::PathBuf::from).map(|p|p.join("tmp")).unwrap_or_else(||"/tmp/weightlang".into());
    fs::create_dir_all(&scratch_root).unwrap();
    let socket=tempfile::Builder::new().prefix("wm-").tempdir_in(&scratch_root).unwrap();
    fs::set_permissions(socket.path(),fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon=Daemon(Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required")).arg("--test-directory").arg(socket.path()).spawn().unwrap());
    let endpoint=socket.path().join("service.sock");let start=Instant::now();while !endpoint.exists(){assert!(start.elapsed()<Duration::from_secs(10));std::thread::sleep(Duration::from_millis(5));}
    let client=Client::connect_to(&endpoint).unwrap();
    let inline=wgt::model!(r#"wgt 0.5
model Double
input x:f32[1]
output y:f32[1]
fn forward(x){return x+x;}
"#);
    let compiled_file=wgt::include_model!("tests/fixtures/model.wgt");
    let scratch=tempfile::tempdir_in(&scratch_root).unwrap();let path=scratch.path().join("model.wgt");fs::write(&path,SOURCE).unwrap();
    let runtime=ModelDefinition::from_file(&path).unwrap();fs::remove_file(&path).unwrap();
    for definition in [&inline,&compiled_file,&runtime] {
        let model=client.initialize(definition,InstanceOptions::default()).unwrap();
        assert_eq!(model.forward(&[3.]).unwrap(),vec![6.]);
        let before=model.forward(&[3.]).unwrap();
        assert!(model.train(&[vec![3.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:None}).is_err());
        assert_eq!(before,model.forward(&[3.]).unwrap());
        model.close().unwrap();assert_eq!(client.stats().unwrap()["sessions"],0);assert_eq!(client.stats().unwrap()["reserved"],0);
    }
    let sequential=ModelDefinition::parse("wgt 0.5\nmodel Dense\ninput x:f32[1]\noutput y:f32[1]\nlayer dense: linear(1)\ninit seed=7\n").unwrap();
    let inference=client.initialize(&sequential,InstanceOptions::default()).unwrap();
    let before=inference.forward(&[1.]).unwrap();
    assert!(inference.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:None}).is_err());
    assert_eq!(before,inference.forward(&[1.]).unwrap());inference.close().unwrap();
    let trainable=ModelDefinition::parse("wgt 0.5\nmodel Scale\ninput x:f32[1]\noutput y:f32[1]\nparameter w:f32[1]=[2]\nfn forward(x){return x*w;}\n").unwrap();
    let stats=client.stats().unwrap();
    assert!(client.initialize(&trainable,InstanceOptions{memory_bytes:1,optimizer:None}).is_err());
    let after=client.stats().unwrap();assert_eq!(stats["sessions"],after["sessions"]);assert_eq!(stats["reserved"],after["reserved"]);
    let model=client.initialize(&trainable,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    let weights=scratch.path().join("weights.safetensors");model.save_weights(&weights).unwrap();assert!(model.save_weights(&weights).is_err());
    model.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:Some(1.)}).unwrap();
    assert!(model.forward(&[1.]).unwrap()[0]<2.);
    model.load_weights(&weights,None).unwrap();assert_eq!(model.forward(&[1.]).unwrap(),vec![2.]);
    let before=model.forward(&[1.]).unwrap();fs::write(scratch.path().join("bad.safetensors"),b"bad").unwrap();
    assert!(model.load_weights(&scratch.path().join("bad.safetensors"),Some(Optimizer::Adam)).is_err());assert_eq!(before,model.forward(&[1.]).unwrap());
    for rate in [f32::NAN,f32::INFINITY,0.] {assert!(model.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate,loss:Loss::Mse,clip:None}).is_err());}
    drop(model);assert_eq!(client.stats().unwrap()["sessions"],0);

    // Compile a real renamed facade consumer, then remove its included definition.
    let project=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache=project.join(".cache/weightlang/rust-model-consumers");fs::create_dir_all(&cache).unwrap();
    let consumer=tempfile::tempdir_in(&cache).unwrap();fs::create_dir(consumer.path().join("src")).unwrap();
    let facade=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap();
    fs::write(consumer.path().join("Cargo.toml"),format!("[package]\nname=\"wgt-model-runtime-consumer\"\nversion=\"0.0.0\"\nedition=\"2021\"\n[workspace]\n[dependencies]\nrenamed={{package=\"wgt-model\",path={:?}}}\n",facade.to_str().unwrap())).unwrap();
    fs::write(consumer.path().join("model.wgt"),SOURCE).unwrap();
    fs::write(consumer.path().join("src/main.rs"),r##"fn main(){
        let definition=renamed::include_model!(crate = renamed; "model.wgt");
        let inline=renamed::model!(crate = renamed; r#"wgt 0.5
model Double
input x:f32[1]
output y:f32[1]
fn forward(x){return x+x;}
"#);
        let client=renamed::Client::connect().unwrap();
        for definition in [&definition,&inline] {
            let model=client.initialize(definition,Default::default()).unwrap();
            assert_eq!(model.forward(&[3.]).unwrap(),vec![6.]);
            model.close().unwrap();
        }
        println!("[6]");
    }"##).unwrap();
    let target=project.join(".cache/weightlang/wgtime-target");
    let build=Command::new(env!("CARGO")).args(["build","--offline","--quiet","--manifest-path"]).arg(consumer.path().join("Cargo.toml")).arg("--target-dir").arg(&target).env("CARGO_BUILD_JOBS","2").output().unwrap();
    assert!(build.status.success(),"{}",String::from_utf8_lossy(&build.stderr));
    fs::remove_file(consumer.path().join("model.wgt")).unwrap();
    fs::remove_file(consumer.path().join("src/main.rs")).unwrap();
    let output=Command::new(target.join("debug/wgt-model-runtime-consumer")).env("WGTIME_ENDPOINT",&endpoint).env("PATH","/nonexistent").current_dir(consumer.path()).output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(),"[6]");
}
