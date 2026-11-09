mod support;

use wgt::{InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

// x = [observation, query]. A fresh two-slot bank routes the observation into
// persistent total; the query reads that candidate within the same transaction.
const SOURCE: &str = r#"wgt 0.5
model Accumulator
input x: f32[2]
output y: f32[1]
parameter total: f32[1] = [0]
parameter gain: f32[1] = [0.5]
fn forward(x) {
    let bank = memory_alloc(2, 1);
    let address = [0.25, 0.75];
    let value = slice(x, 0, 1) * gain;
    memory_write(bank, address, value);
    let recalled = memory_read(bank, address);
    memory_free(bank);
    let next = total + recalled;
    write(total, next);
    return next * slice(x, 1, 1);
}
fn loss(prediction, target) { return [0.5] * square(prediction - target); }
"#;

fn parameter(model: &wgt::Model, name: &str) -> Vec<f32> {
    model.parameters().unwrap().into_iter().find(|p| p.0 == name).unwrap().2
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn native_memory_writes_commit_rollback_and_reload() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let model = client.initialize(&definition, InstanceOptions::default()).unwrap();
    let independent = client.initialize(&definition, InstanceOptions::default()).unwrap();
    assert_eq!(model.backend(), "native-cpu");
    // Weighted write followed by read multiplies value by .25² + .75² = .625.
    assert_eq!(model.forward(&[2., 3.]).unwrap(), [1.875]);
    assert_eq!(model.forward(&[4., 3.]).unwrap(), [5.625]);
    assert_eq!(parameter(&model, "total"), [1.875]);
    assert_eq!(parameter(&independent, "total"), [0.]);
    let before = model.parameters().unwrap();
    assert!(model.forward(&[4., f32::MAX]).is_err());
    assert_eq!(model.parameters().unwrap(), before);
    // A failed invocation must leave neither candidate writes nor a live bank.
    assert_eq!(model.forward(&[0., 3.]).unwrap(), [5.625]);
    let weights = service.path().join("weights.safetensors");
    model.save_weights(&weights).unwrap();
    independent.load_weights(&weights, None).unwrap();
    assert_eq!(independent.forward(&[0., 3.]).unwrap(), [5.625]);
    let execution = client.stats().unwrap();
    assert!(execution["execution"].as_array().unwrap().iter().all(|s| s["backend"] == "native-cpu"));
    model.close().unwrap();
    independent.close().unwrap();
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn native_memory_custom_loss_updates_snapshot_parameters() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = ModelDefinition::parse(SOURCE).unwrap();
    let model = client.initialize(&definition, InstanceOptions {
        optimizer: Some(Optimizer::Sgd), ..Default::default()
    }).unwrap();
    let before = model.parameters().unwrap();
    let options = TrainOptions { rate: 0.1, loss: Loss::Custom, clip: None };
    assert!(model.train(&[vec![2., 1.], vec![4., f32::MAX]],
        Targets::Values(&[vec![0.], vec![0.]]), options).is_err());
    assert_eq!(model.parameters().unwrap(), before);
    let loss = model.train(&[vec![2., 1.]], Targets::Values(&[vec![0.]]), options).unwrap();
    assert_eq!(loss, 0.1953125);
    // y = total + 1.25*gain = .625; gradients are [.625, .78125].
    // SGD applies to the initial parameters, never the transient forward write.
    assert_eq!(parameter(&model, "total"), [-0.0625]);
    assert_eq!(parameter(&model, "gain"), [0.421875]);
    let stats = client.stats().unwrap();
    assert_eq!(stats["execution"][0]["backend"], "native-cpu");
    assert!(stats["execution"][0]["native_tape_forwards"].as_u64().unwrap() >= 1);
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn native_resource_state_tracks_call_local_memory() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let definition = wgt::model!(r#"wgt 0.5
model Resources
input x: f32[1]
output y: f32[3]
fn forward(x) {
    # Two slots of three f32 values occupy 24 live bytes until the free.
    let bank = memory_alloc(2, 3);
    let live = resource_state();
    memory_free(bank);
    return live;
}
"#);
    let model = client.initialize(&definition, InstanceOptions {
        memory_bytes: 65536, optimizer: None
    }).unwrap();
    let observed = model.forward(&[0.]).unwrap();
    assert_eq!(observed[0], 999992.);
    assert!(observed[1] > 0. && observed[1] < 65536.);
    assert_eq!(observed[2], 24.);
    assert_eq!(model.forward(&[0.]).unwrap(), observed);
    model.close().unwrap();
    assert_eq!(client.stats().unwrap()["reserved"], 0);
}
