mod support;
use std::{collections::BTreeMap, fs::File};
use wgt::{
    Client, InstanceOptions, ModelDefinition,
    weights::{Tensor, WeightSet},
};

fn weights(name: &str, shape: Vec<usize>, values: Vec<f32>) -> WeightSet {
    WeightSet::new(BTreeMap::from([(
        name.into(),
        Tensor::new(shape, values).unwrap(),
    )]))
    .unwrap()
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn complete_restore_skips_invalid_initializers_and_failed_loads_allocate_nothing() {
    let service = support::Service::start().unwrap();
    let client = service.client().unwrap();
    let scratch = support::scratch().unwrap();
    let definition = ModelDefinition::parse("wgt 0.5\nmodel Restored\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=sqrt(full(-1))\nfn forward(x){return x*gain;}\n").unwrap();
    assert!(
        client
            .initialize(&definition, InstanceOptions::default())
            .is_err()
    );
    assert_eq!(client.stats().unwrap()["sessions"], 0);
    let snapshot = weights("gain", vec![1], vec![7.]);
    let path = scratch.path().join("snapshot");
    snapshot.save(&path).unwrap();
    let restored = client
        .load_snapshot(
            &definition,
            &path,
            Some(snapshot.revision()),
            InstanceOptions::default(),
        )
        .unwrap();
    assert_eq!(restored.forward(&[3.]).unwrap(), [21.]);
    assert_eq!(
        restored
            .snapshot()
            .unwrap()
            .tensor("gain")
            .unwrap()
            .values(),
        [7.]
    );
    restored.close().unwrap();

    let unbound = ModelDefinition::parse(
        "wgt 0.5\nmodel Dense\ninput x:f32[1]\noutput y:f32[1]\nlayer dense:linear(1,bias=0)\n",
    )
    .unwrap();
    assert!(
        client
            .initialize(&unbound, InstanceOptions::default())
            .is_err()
    );
    let dense_path = scratch.path().join("dense");
    weights("dense.weight", vec![1, 1], vec![5.])
        .save(&dense_path)
        .unwrap();
    let dense = client
        .load_weights(
            &unbound,
            &dense_path.join("weights.safetensors"),
            InstanceOptions::default(),
        )
        .unwrap();
    assert_eq!(dense.forward(&[2.]).unwrap(), [10.]);
    dense.close().unwrap();

    let before = client.stats().unwrap();
    assert!(
        client
            .load_snapshot(
                &definition,
                &path,
                Some(&"0".repeat(64)),
                InstanceOptions::default()
            )
            .is_err()
    );
    assert!(
        client
            .load_weights(
                &definition,
                &dense_path.join("weights.safetensors"),
                InstanceOptions::default()
            )
            .is_err()
    );
    let after = client.stats().unwrap();
    assert_eq!(before["sessions"], after["sessions"]);
    assert_eq!(before["reserved"], after["reserved"]);

    // Pinning is also enforced on the service side after reading the actual FD.
    let mut raw = wgtime::Client::connect(&service.endpoint(), unsafe { libc::geteuid() }).unwrap();
    let request = wgtime::Request::InitializeWeightsFd {
        source: definition.source().into(),
        memory: 1024 * 1024,
        optimizer: None,
        revision: Some("0".repeat(64)),
    };
    assert!(
        raw.call_with_file(
            request,
            &File::open(path.join("weights.safetensors")).unwrap()
        )
        .is_err()
    );
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}

#[test]
#[ignore = "requires prebuilt WGTIME_TEST_DAEMON; isolated real service"]
fn frozen_restore_keeps_state_separate_and_preserves_continuous_learning_storage() {
    let service = support::Service::start().unwrap();
    let client: Client = service.client().unwrap();
    let scratch = support::scratch().unwrap();
    let definition = ModelDefinition::parse("wgt 0.5\nmodel Counter\nfrozen gain:f32[1]\nstate count:f32[1]=full(5)\nfn forward(x:f32[1])->f32[1]{return gain*x+count;}\nfn advance(x:f32[1]){count<-count+x;}\n").unwrap();
    let path = scratch.path().join("frozen");
    weights("gain", vec![1], vec![9.]).save(&path).unwrap();
    let model = client
        .load_snapshot(&definition, &path, None, InstanceOptions::default())
        .unwrap();
    model.call("advance", &[vec![3.]]).unwrap();
    assert_eq!(model.call("forward", &[vec![1.]]).unwrap(), Some(vec![17.]));
    let captured = model.snapshot().unwrap();
    assert_eq!(captured.names().collect::<Vec<_>>(), ["gain"]);
    assert!(
        model
            .update_parameters(&[("gain".into(), vec![1], vec![4.])])
            .is_err()
    );
    let next_path = scratch.path().join("captured");
    captured.save(&next_path).unwrap();
    let fresh = client
        .load_snapshot(&definition, &next_path, None, InstanceOptions::default())
        .unwrap();
    assert_eq!(fresh.call("forward", &[vec![1.]]).unwrap(), Some(vec![14.]));
    fresh.close().unwrap();
    model.close().unwrap();

    let learned = ModelDefinition::parse("wgt 0.5\nmodel Shadow\ninput x:f32[1]\noutput y:f32[1]\nlearn enabled:bool;\nlearn count:int in 0..=4;\nfn forward(x){return x;}\n").unwrap();
    let shadow = WeightSet::new(BTreeMap::from([
        ("enabled".into(), Tensor::new(vec![1], vec![0.3]).unwrap()),
        ("count".into(), Tensor::new(vec![1], vec![2.7]).unwrap()),
    ]))
    .unwrap();
    let shadow_path = scratch.path().join("shadow");
    shadow.save(&shadow_path).unwrap();
    let instance = client
        .load_snapshot(&learned, &shadow_path, None, InstanceOptions::default())
        .unwrap();
    assert_eq!(
        instance
            .snapshot()
            .unwrap()
            .tensor("count")
            .unwrap()
            .values(),
        [2.7]
    );
    assert_eq!(
        instance
            .snapshot()
            .unwrap()
            .tensor("enabled")
            .unwrap()
            .values(),
        [0.3]
    );
    instance.close().unwrap();
    assert_eq!(client.stats().unwrap()["sessions"], 0);
}
