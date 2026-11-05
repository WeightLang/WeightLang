use std::{
    fs,
    process::Command,
    time::{Duration, Instant},
};
use wgt::{
    Client, DifferentiationPolicy, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets,
    TrainOptions,
};
const SOURCE: &str = r#"wgt 0.5
import activation from "std/activation"
model Cell
state activity: f32[2] = [0,0];
parameter gain: f32[2] = [1,1]
fn observe(input: f32[2]) -> f32[2] { return activity + input * gain; }
fn advance(input: f32[2]) -> f32[2] { activity <- activity + activation.squared_relu(input); return activity; }
fn bad(input: f32[2]) -> f32[2] { activity <- input; gain <- input; return log([0,0]); }
fn reset() { activity <- [0,0]; }
"#;
#[test]
fn typed_definition_import_roundtrip() {
    let inline = wgt::model!("wgt 0.5\nmodel Pure\nfn observe(x:f32[1])->f32[1]{return x;}\n");
    assert_eq!(inline.operations().len(), 1);
    let d = ModelDefinition::parse(SOURCE).unwrap();
    assert!(d.interface().is_none());
    assert_eq!(d.operations().len(), 4);
    assert_eq!(
        ModelDefinition::parse(d.source())
            .unwrap()
            .operations()
            .len(),
        4
    );
    let _ = d.instantiate_plan().unwrap();
}
#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn native_operations_ipc_search_persistence_and_policy() {
    use std::os::unix::fs::PermissionsExt;
    struct Daemon(std::process::Child);
    impl Drop for Daemon {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    fs::create_dir_all("/tmp/weightlang").unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("operations-")
        .tempdir_in("/tmp/weightlang")
        .unwrap();
    fs::set_permissions(scratch.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let _daemon = Daemon(
        Command::new(std::env::var_os("WGTIME_TEST_DAEMON").expect("prebuilt daemon"))
            .arg("--test-directory")
            .arg(scratch.path())
            .spawn()
            .unwrap(),
    );
    let endpoint = scratch.path().join("service.sock");
    let start = Instant::now();
    while !endpoint.exists() {
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(5));
    }
    let client = Client::connect_to(&endpoint).unwrap();
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let a = client
        .initialize(&definition, InstanceOptions::default())
        .unwrap();
    let b = client
        .initialize(&definition, InstanceOptions::default())
        .unwrap();
    assert_eq!(a.backend(), "native-operations");
    assert!(a.interface().is_none());
    assert_eq!(a.operations().len(), 4);
    assert_eq!(
        a.call("advance", &[vec![2., 3.]]).unwrap(),
        Some(vec![4., 9.])
    );
    assert_eq!(b.state().unwrap()[0].2, vec![0., 0.]);
    let before = a.state().unwrap();
    let parameters = a.parameters().unwrap();
    assert!(a.call("bad", &[vec![7., 7.]]).is_err());
    assert_eq!(a.state().unwrap(), before);
    assert_eq!(a.parameters().unwrap(), parameters);
    assert_eq!(a.call("reset", &[]).unwrap(), None);
    let mut best = f32::INFINITY;
    for gain in [0., 1., 2., 3.] {
        a.update_parameters(&[("gain".into(), vec![2], vec![gain, gain])])
            .unwrap();
        let y = a.call("observe", &[vec![1., 1.]]).unwrap().unwrap();
        let error = (y[0] - 3.).powi(2);
        assert!(error < best);
        best = error;
    }
    assert_eq!(best, 0.);
    let path = scratch.path().join("weights.safetensors");
    a.save_weights(&path).unwrap();
    b.call("advance", &[vec![2., 2.]]).unwrap();
    b.load_weights(&path, None).unwrap();
    assert_eq!(b.state().unwrap()[0].2, vec![4., 4.]);
    assert_eq!(
        b.call("observe", &[vec![1., 1.]]).unwrap(),
        Some(vec![7., 7.])
    );
    let options = TrainOptions {
        rate: 0.01,
        loss: Loss::Mse,
        clip: None,
    };
    assert!(
        a.train_with_policy(
            &[vec![1., 1.]],
            Targets::Values(&[vec![0., 0.]]),
            options,
            DifferentiationPolicy::StraightThrough
        )
        .is_err()
    );
    let discrete=ModelDefinition::parse("wgt 0.5\nmodel Discrete\ninput x: f32[1]\noutput y: f32[1]\nlearn gate: bool = true;\nfn forward(x) { return x * gate; }\n").unwrap();
    let d = client
        .initialize(
            &discrete,
            InstanceOptions {
                optimizer: Some(Optimizer::Sgd),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        d.train_with_policy(
            &[vec![1.]],
            Targets::Values(&[vec![0.]]),
            options,
            DifferentiationPolicy::ContinuousOnly
        )
        .is_err()
    );
    assert_eq!(d.parameters().unwrap()[0].2, vec![1.]);
    assert!(
        d.train_with_policy(
            &[vec![1.]],
            Targets::Values(&[vec![0.]]),
            options,
            DifferentiationPolicy::StraightThrough
        )
        .unwrap()
        .is_finite()
    );
    let values = d.parameters().unwrap();
    assert!(
        d.update_parameters(&[("gate".into(), vec![1], vec![0.4])])
            .is_err()
    );
    assert_eq!(d.parameters().unwrap(), values);
    assert!(
        d.train(&[vec![1.]], Targets::Values(&[vec![0.]]), options)
            .is_ok()
    );
    d.update_parameters(&[("gate".into(), vec![1], vec![0.])])
        .unwrap();
    assert!(
        d.train(&[vec![1.]], Targets::Values(&[vec![0.]]), options)
            .is_err()
    );
    a.close().unwrap();
    b.close().unwrap();
    d.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}
