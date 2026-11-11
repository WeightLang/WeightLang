mod support;
use std::{fs,process::Command};
use wgt::{InstanceOptions,ModelDefinition,Optimizer,Loss,Targets,TrainOptions};
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
    assert_eq!(inline.interface().unwrap().input,1);
    assert_eq!(file.interface().unwrap().output,1);
    assert!(ModelDefinition::parse(SOURCE).is_ok());
    let dimensioned = wgt::include_model!("../../examples/native_training/signature_dimensions/model.wgt");
    assert_eq!(dimensioned.interface().unwrap().input, 2);
}

#[test]
#[ignore="requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn three_definition_entries_execute_and_resources_release() {
    let service=support::Service::start().unwrap();
    let endpoint=service.endpoint();
    let client=service.client().unwrap();
    let inline=wgt::model!(r#"wgt 0.5
model Double
input x:f32[1]
output y:f32[1]
fn forward(x){return x+x;}
"#);
    let compiled_file=wgt::include_model!("tests/fixtures/model.wgt");
    let scratch=support::scratch().unwrap();let path=scratch.path().join("model.wgt");fs::write(&path,SOURCE).unwrap();
    let runtime=ModelDefinition::from_file(&path).unwrap();fs::remove_file(&path).unwrap();
    // Resolve a local operator module on the host, then remove both files before
    // service initialization: the service must use the embedded dependency closure.
    let module=scratch.path().join("operators.wgt");
    fs::write(&module,"wgt 0.5\noperators Local\nfn twice(x){return x+x;}\n").unwrap();
    fs::write(&path,"wgt 0.5\nimport local from \"./operators.wgt\"\nmodel Double\ninput x:f32[1]\noutput y:f32[1]\nfn forward(x){return local.twice(x);}\n").unwrap();
    let local=ModelDefinition::from_file(&path).unwrap();
    fs::remove_file(&path).unwrap();fs::remove_file(&module).unwrap();
    for definition in [&inline,&compiled_file,&runtime,&local] {
        let model=client.initialize(definition,InstanceOptions::default()).unwrap();
        assert_eq!(model.backend(),"native-cpu");
        assert_eq!(model.forward(&[3.]).unwrap(),vec![6.]);
        let before=model.forward(&[3.]).unwrap();
        assert!(model.train(&[vec![3.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:None}).is_err());
        assert_eq!(before,model.forward(&[3.]).unwrap());
        model.close().unwrap();assert_eq!(client.stats().unwrap()["sessions"],0);assert_eq!(client.stats().unwrap()["reserved"],0);
    }
    let standard=wgt::model!(r#"wgt 0.5
import activation from "std/activation"
model Activated
input x:f32[1]
output y:f32[1]
fn forward(x){return activation.squared_relu(x);}
"#);
    let activated=client.initialize(&standard,InstanceOptions::default()).unwrap();
    assert_eq!(activated.forward(&[-3.]).unwrap(),vec![0.]);
    assert_eq!(activated.forward(&[3.]).unwrap(),vec![9.]);
    activated.close().unwrap();
    let sequential=ModelDefinition::parse("wgt 0.5\nmodel Dense\ninput x:f32[1]\noutput y:f32[1]\nlayer dense: linear(1)\ninit weights=xavier_uniform(7) bias=zeros()\n").unwrap();
    let inference=client.initialize(&sequential,InstanceOptions::default()).unwrap();
    let before=inference.forward(&[1.]).unwrap();
    assert!(inference.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:None}).is_err());
    assert_eq!(before,inference.forward(&[1.]).unwrap());inference.close().unwrap();
    // A one-input dense layer has a weight and bias: one SGD step at rate .1
    // against zero changes their summed prediction by -0.4 times that prediction.
    let dense=client.initialize(&sequential,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    let initial=dense.forward(&[1.]).unwrap()[0];
    let loss=dense.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:None}).unwrap();
    assert!((loss-initial*initial).abs()<1e-6);
    assert!((dense.forward(&[1.]).unwrap()[0]-initial*0.6).abs()<1e-6);
    dense.close().unwrap();
    let trainable=ModelDefinition::parse("wgt 0.5\nmodel Scale\ninput x:f32[1]\noutput y:f32[1]\nparameter w:f32[1]=[2]\nfn forward(x){return x*w;}\n").unwrap();
    let stats=client.stats().unwrap();
    assert!(client.initialize(&trainable,InstanceOptions{memory_bytes:1,optimizer:None}).is_err());
    let after=client.stats().unwrap();assert_eq!(stats["sessions"],after["sessions"]);assert_eq!(stats["reserved"],after["reserved"]);
    let model=client.initialize(&trainable,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    let weights=scratch.path().join("weights.safetensors");model.save_weights(&weights).unwrap();assert!(model.save_weights(&weights).is_err());
    let loss=model.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Mse,clip:Some(1.)}).unwrap();
    assert!((loss-4.).abs()<1e-6);
    assert!((model.forward(&[1.]).unwrap()[0]-1.9).abs()<1e-6);
    let stats=client.stats().unwrap();
    let execution=&stats["execution"][0];
    assert_eq!(execution["backend"],"native-cpu");
    assert_eq!(execution["native_tape_forwards"],1);
    assert!(execution["native_forwards"].as_u64().unwrap()>=2);
    model.load_weights(&weights,None).unwrap();assert_eq!(model.forward(&[1.]).unwrap(),vec![2.]);
    let before=model.forward(&[1.]).unwrap();fs::write(scratch.path().join("bad.safetensors"),b"bad").unwrap();
    assert!(model.load_weights(&scratch.path().join("bad.safetensors"),Some(Optimizer::Adam)).is_err());assert_eq!(before,model.forward(&[1.]).unwrap());
    for rate in [f32::NAN,f32::INFINITY,0.] {assert!(model.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate,loss:Loss::Mse,clip:None}).is_err());}
    drop(model);assert_eq!(client.stats().unwrap()["sessions"],0);

    let learned_import=wgt::model!(r#"wgt 0.5
import losses from "std/loss"
model LearnedImport
input x:f32[1]
output y:f32[1]
learn scale: float = 5;
fn forward(x){return x*scale;}
fn loss(prediction,target){return losses.half_squared_error(prediction,target);}
"#);
    let imported=client.initialize(&learned_import,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    imported.train(&[vec![1.]],Targets::Values(&[vec![10.]]),TrainOptions{rate:0.1,loss:Loss::Custom,clip:None}).unwrap();
    assert_eq!(imported.forward(&[1.]).unwrap(),[5.5]); imported.close().unwrap();

    // Architecture parameters use the same actual native service and training tape.
    for (declaration,initial,target,rate,expected) in [
        ("learn scale: float = 5.0;",5.,10.,0.1,5.5),
        ("learn scale: float in 0..=1 = 0.5;",0.5,10.,1.,1.),
        ("learn scale: float > 0 = 1;",1.,-10.,1.,f32::from_bits(1)),
        ("learn scale: int in 1..=12 = 5;",5.,6.,0.6,6.),
        ("learn scale: bool = false;",0.,1.,0.5,1.),
    ] {
        let source=format!("wgt 0.5\nmodel Architecture\ninput x:f32[1]\noutput y:f32[1]\n{declaration}\nfn forward(x){{return x*scale;}}\nfn loss(prediction,target){{return [0.5]*square(prediction-target);}}\n");
        let definition=ModelDefinition::parse(&source).unwrap();
        assert!(definition.parameters()[0].learn.is_some());
        let net=client.initialize(&definition,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
        assert_eq!(net.forward(&[1.]).unwrap(),[initial]);
        net.train(&[vec![1.]],Targets::Values(&[vec![target]]),TrainOptions{rate,loss:Loss::Custom,clip:None}).unwrap();
        assert_eq!(net.forward(&[1.]).unwrap(),[expected]);
        let saved=scratch.path().join(format!("learn-{initial}-{expected}.safetensors"));
        net.save_weights(&saved).unwrap();
        // Restore into an independently initialized instance using the same definition.
        let restored=client.initialize(&definition,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
        restored.load_weights(&saved,Some(Optimizer::Sgd)).unwrap();
        assert_eq!(restored.forward(&[1.]).unwrap(),[expected]);
        for model in [&net,&restored] {
            model.train(&[vec![1.]],Targets::Values(&[vec![0.]]),TrainOptions{rate:0.1,loss:Loss::Custom,clip:None}).unwrap();
        }
        assert_eq!(restored.forward(&[1.]).unwrap(),net.forward(&[1.]).unwrap());
        restored.close().unwrap();net.close().unwrap();
    }

    let switch=ModelDefinition::parse("wgt 0.5\nmodel Switch\ninput x:f32[1]\noutput y:f32[1]\nlearn skip: bool = false;\nfn forward(x){return gate(skip,x);}\nfn loss(prediction,target){return [0.5]*square(prediction-target);}\n").unwrap();
    let original=client.initialize(&switch,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    original.train(&[vec![1.]],Targets::Values(&[vec![1.]]),TrainOptions{rate:0.4,loss:Loss::Custom,clip:None}).unwrap();
    assert_eq!(original.forward(&[1.]).unwrap(),[0.]);
    let path=scratch.path().join("switch-shadow.safetensors"); original.save_weights(&path).unwrap();
    let restored=client.initialize(&switch,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    restored.load_weights(&path,Some(Optimizer::Sgd)).unwrap();
    restored.train(&[vec![1.]],Targets::Values(&[vec![1.]]),TrainOptions{rate:0.2,loss:Loss::Custom,clip:None}).unwrap();
    assert_eq!(restored.forward(&[1.]).unwrap(),[1.]);
    restored.close().unwrap(); original.close().unwrap();
    let depth=ModelDefinition::parse("wgt 0.5\nmodel Depth\ninput x:f32[1]\noutput y:f32[1]\nlearn depth: int in 1..=3 = 1;\nfrozen w:f32[1]=[2]\nfn forward(x){let a=route(depth_gate(depth,1),x*w,x);let b=route(depth_gate(depth,2),a*w,a);return route(depth_gate(depth,3),b*w,b);}\nfn loss(prediction,target){return [0.5]*square(prediction-target);}\n").unwrap();
    let net=client.initialize(&depth,InstanceOptions{optimizer:Some(Optimizer::Sgd),..Default::default()}).unwrap();
    assert_eq!(net.forward(&[1.]).unwrap(),[2.]);
    net.train(&[vec![1.]],Targets::Values(&[vec![4.]]),TrainOptions{rate:0.1,loss:Loss::Custom,clip:None}).unwrap();
    assert_eq!(net.forward(&[1.]).unwrap(),[4.]); net.close().unwrap();

    // Compile a real renamed facade consumer, then remove its included definition.
    let project=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache=project.join(".cache/weightlang/rust-model-consumers");fs::create_dir_all(&cache).unwrap();
    let consumer=tempfile::tempdir_in(&cache).unwrap();fs::create_dir(consumer.path().join("src")).unwrap();
    let facade=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap();
    fs::write(consumer.path().join("Cargo.toml"),format!("[package]\nname=\"wgt-model-runtime-consumer\"\nversion=\"0.0.0\"\nedition=\"2024\"\n[workspace]\n[dependencies]\nrenamed={{package=\"wgt-model\",path={:?}}}\n",facade.to_str().unwrap())).unwrap();
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

#[test]
#[ignore="requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn signature_dimensions_execute_and_train_through_the_real_service() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let source = include_str!("../../../examples/native_training/signature_dimensions/model.wgt");
    let definition = ModelDefinition::parse_with_dimensions(source, &std::collections::BTreeMap::from([("N".into(), 3)])).unwrap();
    let model = client.initialize(&definition, InstanceOptions { optimizer: Some(Optimizer::Sgd), ..Default::default() }).unwrap();
    assert_eq!(model.forward(&[1.,1.,1.]).unwrap(), [1.,1.,1.]);
    model.train(&[vec![1.,1.,1.]], Targets::Values(&[vec![0.,0.,0.]]), TrainOptions { rate:0.1, loss:Loss::Custom, clip:None }).unwrap();
    for value in model.forward(&[1.,1.,1.]).unwrap() { assert!((value - (1. - 0.1 * 4. / 3.)).abs() < 1e-6); }
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn activation_capture_and_patch_use_real_service() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(include_str!("../../../examples/native_training/interpretability/model.wgt")).unwrap();
    let model = client.initialize(&definition, InstanceOptions::default()).unwrap();
    let baseline = model.inspect(&[2., 3.], Default::default(), 10, 1024, 10000).unwrap();
    assert_eq!(baseline["output"], serde_json::json!([6., 12.]));
    assert_eq!(baseline["events"][0]["before"], serde_json::json!([4., 9.]));
    let attribution = model.attribute(&[2.,3.], &[1.,1.], 10000, 262144).unwrap();
    assert_eq!(attribution["input_gradient"],serde_json::json!([5.,7.]));
    assert_eq!(attribution["events"][0]["gradient"],serde_json::json!([1.,1.]));
    let graph = model.circuit(&[2.,3.], &[1.,1.], 8, 10000, 1024, 0.).unwrap();
    assert_eq!(graph["edges"][0]["effect"],serde_json::json!(13.));
    let instruction = baseline["events"][0]["instruction"].as_u64().unwrap() as usize;
    let targeted = model.intervene(&[2.,3.],vec![wgt::ActivationIntervention {
        instruction,occurrence:0,value:vec![1.,1.],
    }],10,1024,10000).unwrap();
    assert_eq!(targeted["output"],serde_json::json!([3.,4.]));
    let dictionary=serde_json::json!({"width":2,"features":2,
        "encoder":[1.,0.,0.,1.],"encoder_bias":[0.,0.],"decoder":[0.5,0.,0.,0.5]});
    let feature_effects=model.feature_effects(&[2.,3.],&[1.,1.],dictionary,
        instruction,0,10000,1024).unwrap();
    assert_eq!(feature_effects["effects"],serde_json::json!([2.,4.5]));

    let mut replacements = std::collections::BTreeMap::new();
    replacements.insert("hidden".into(), vec![0., 0.]);
    let patched = model.inspect(&[2., 3.], replacements, 10, 1024, 10000).unwrap();
    assert_eq!(patched["output"], serde_json::json!([2., 3.]));
    assert!(model.inspect(&[2., 3.], Default::default(), 0, 0, 10000).is_err());
    assert_eq!(model.forward(&[2., 3.]).unwrap(), vec![6., 12.]);
}
