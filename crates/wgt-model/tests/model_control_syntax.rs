use std::{
    fs,
    process::{Child, Command},
    time::{Duration, Instant},
};
use wgt::{Client, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};
struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
const MODEL: &str = "wgt 0.5\nmodel Root\ninput x:f32[1]\noutput y:f32[1]\nlearn depth:int[2] in 0..=3=[1,2];\nlearn enabled:bool[2]=true;\nlearn gain:float[2]=1;\nmodel A {\ninput a:f32[1]\noutput b:f32[1]\nparameter weight:f32[1]=[2]\nfn forward(a){return a*weight;}\n}\nblocks:A[3];\nfn forward(x){let h=repeat depth[0] indexed i carrying h=x {blocks[i](h)};return if enabled[0] {blend {0.5 => gain[0]*h; 0.5 => x;}} else {x};}\nfn loss(prediction,target){return 0.5*((prediction-target)**2).mean;}\n";
#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn native_service_controls_instances_and_shadow_roundtrip() {
    fs::create_dir_all("/tmp/weightlang").unwrap();
    let directory = tempfile::Builder::new()
        .prefix("model-controls-")
        .tempdir_in("/tmp/weightlang")
        .unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon = Daemon(
        Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon required"))
            .arg("--test-directory")
            .arg(directory.path())
            .spawn()
            .unwrap(),
    );
    let socket = directory.path().join("service.sock");
    let start = Instant::now();
    while !socket.exists() {
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(5));
    }
    let client = Client::connect_to(&socket).unwrap();
    let definition = ModelDefinition::parse(MODEL).unwrap();
    let options = || InstanceOptions {
        optimizer: Some(Optimizer::Sgd),
        ..Default::default()
    };
    let model = client.initialize(&definition, options()).unwrap();
    assert_eq!(model.backend(), "native-cpu");
    assert_eq!(model.forward(&[1.]).unwrap(), [1.5]);
    model
        .train(
            &[vec![1.]],
            Targets::Values(&[vec![2.]]),
            TrainOptions {
                rate: 0.1,
                loss: Loss::Custom,
                clip: None,
            },
        )
        .unwrap();
    assert!(model.forward(&[1.]).unwrap()[0] > 1.5);
    let path = directory.path().join("weights.safetensors");
    model.save_weights(&path).unwrap();
    let restored = client.initialize(&definition, options()).unwrap();
    restored.load_weights(&path, Some(Optimizer::Sgd)).unwrap();
    assert_eq!(
        restored.forward(&[1.]).unwrap(),
        model.forward(&[1.]).unwrap()
    );
    for net in [&model, &restored] {
        net.train(
            &[vec![1.]],
            Targets::Values(&[vec![2.]]),
            TrainOptions {
                rate: 0.01,
                loss: Loss::Custom,
                clip: None,
            },
        )
        .unwrap();
    }
    assert_eq!(
        restored.forward(&[1.]).unwrap(),
        model.forward(&[1.]).unwrap()
    );
    restored.close().unwrap();
    model.close().unwrap();
    let stats = client.stats().unwrap();
    assert_eq!(stats["sessions"], 0);
    assert_eq!(stats["reserved"], 0);
}
