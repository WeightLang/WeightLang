use std::path::Path;
use wgtime_core::{training::{mlp, Objective, Optimizer, Trainer}, weight_artifact};

#[test]
fn strict_tensor_contract_and_atomic_optimizer_reset() {
    let mut net = Trainer::new(mlp(&[1,1], "identity", 7).unwrap(), Path::new("."), 1024*1024, Optimizer::Adam).unwrap();
    let bytes = net.weights_bytes(1024*1024).unwrap();
    let schema = weight_artifact::inspect(&bytes, 1024*1024).unwrap();
    assert_eq!(schema.len(), 2);
    assert!(schema.iter().all(|s| s.dtype == "F32"));
    net.step(&[vec![1.]], Objective::Mse(&[vec![2.]]), 0.03).unwrap();
    let trained = net.checkpoint_bytes_bounded(1024*1024).unwrap();
    for invalid in [bytes[..bytes.len()-1].to_vec(), weight_artifact::encode(&[("wrong".into(),vec![1],vec![0.])],1024).unwrap()] {
        assert!(net.bind_weights(&invalid, "sgd").is_err());
        assert_eq!(trained, net.checkpoint_bytes_bounded(1024*1024).unwrap());
    }
    assert!(net.bind_weights(&bytes, "unsupported").is_err());
    assert_eq!(trained, net.checkpoint_bytes_bounded(1024*1024).unwrap());
    net.bind_weights(&bytes, "sgd").unwrap();
    assert_eq!(bytes, net.weights_bytes(1024*1024).unwrap());
    let state:serde_json::Value = serde_json::from_slice(&net.checkpoint_bytes_bounded(1024*1024).unwrap()).unwrap();
    assert_eq!(state["steps"],0);
    assert_eq!(state["optimizer"],"sgd");
    assert!(state["first"].as_array().unwrap().iter().flat_map(|v|v.as_array().unwrap()).all(|v|v.as_f64()==Some(0.)));
}

#[test]
fn dtype_shape_keys_and_finite_reject() {
    use safetensors::{tensor::TensorView,Dtype};
    let encode = |dtype, shape, data: &[u8]| safetensors::serialize(vec![("w",TensorView::new(dtype,shape,data).unwrap())],None).unwrap();
    let schema=vec![("w".into(),vec![1])];
    assert!(weight_artifact::decode(&schema,&encode(Dtype::I32,vec![1],&1i32.to_le_bytes()),1024).is_err());
    assert!(weight_artifact::decode(&schema,&encode(Dtype::F32,vec![1,1],&1f32.to_le_bytes()),1024).is_err());
    assert!(weight_artifact::decode(&schema,&encode(Dtype::F32,vec![1],&f32::NAN.to_le_bytes()),1024).is_err());
    let good=encode(Dtype::F32,vec![1],&3f32.to_le_bytes());
    assert_eq!(weight_artifact::decode(&schema,&good,1024).unwrap(),vec![vec![3.]]);
    assert!(weight_artifact::inspect(&good,good.len()-1).is_err());
}

#[test]
fn source_only_check_and_explicit_binding_preserve_handle_alias() {
    let root=tempfile::tempdir().unwrap();
    let source=r#"wgt 0.5
model Probe
input x:f32[1]
output y:f32[1]
parameter w:f32[1]=[2]
fn forward(x){return x*w;}
fn main(){let net=Probe("adam");let alias=net;save weights net to "weights.safetensors";train net on [[1]] against [[0]] {rate:0.1;};reload weights net from "weights.safetensors" using "sgd";assert alias([1])==[2];return net([1]);}
"#;
    wgtime_core::script::parse(source).unwrap();
    assert!(!root.path().join("weights.safetensors").exists());
    let report=wgtime_core::script::run(source,root.path(),100000).unwrap();
    assert_eq!(report.result,serde_json::json!([2.]));
    let bytes=std::fs::read(root.path().join("weights.safetensors")).unwrap();
    assert_eq!(weight_artifact::inspect(&bytes,1024).unwrap()[0].name,"w");
    assert!(wgtime_core::script::parse("wgt 0.5\nfn main(){let net=0;reload weights net from \"x\";}").is_err());
    assert!(wgtime_core::script::parse("wgt 0.5\nfn main(){return load weights from \"x\";}").is_err());
}

#[test]
fn adaptive_weight_binding_preserves_frozen_values_and_rejects_atomically() {
    use wgtime_core::adaptive::{self, AdaptiveSession};
    const MEMORY: usize = 1024 * 1024;
    let source = "wgt 0.5\nmodel Frozen\ninput x:f32[1]\noutput y:f32[1]\nparameter gain:f32[1]=[2]\nfrozen fixed:f32[1]=[3]\nfn forward(x){return x*gain+fixed;}\n";
    for compiled in [false, true] {
        let mut model = AdaptiveSession::prepare(adaptive::parse(source).unwrap(), MEMORY).unwrap();
        if compiled { model.enable_compiled().unwrap(); }
        model.set_optimizer(Optimizer::Adam).unwrap();
        model.train_step(&[vec![1.]], Objective::Mse(&[vec![0.]]), 0.03, &mut 100_000, MEMORY).unwrap();
        let before = model.checkpoint_bytes_bounded(MEMORY).unwrap();
        let revision = model.revision();
        let artifact = |fixed| weight_artifact::encode(&[
            ("gain".into(), vec![1], vec![7.]),
            ("fixed".into(), vec![1], vec![fixed]),
        ], MEMORY).unwrap();
        let error = model.bind_weights(&artifact(4.), "sgd").unwrap_err();
        assert!(error.to_string().contains("frozen"));
        assert_eq!(model.checkpoint_bytes_bounded(MEMORY).unwrap(), before);
        assert_eq!(model.revision(), revision);
        model.bind_weights(&artifact(3.), "sgd").unwrap();
        assert_eq!(model.parameter("gain").unwrap(), [7.]);
        assert_eq!(model.parameter("fixed").unwrap(), [3.]);
        assert_eq!(model.run(&[2.], 100_000).unwrap(), [17.]);
        assert_eq!(model.is_compiled(), compiled);
        let state: serde_json::Value = serde_json::from_slice(&model.checkpoint_bytes_bounded(MEMORY).unwrap()).unwrap();
        assert_eq!(state["optimizer"]["optimizer"], "sgd");
        assert_eq!(state["optimizer"]["steps"], 0);
    }
}

#[test]
fn trainer_weight_binding_preserves_frozen_values_and_optimizer_on_rejection() {
    use wgtime_core::{Parameter, Value};
    const MEMORY: usize = 1024 * 1024;
    for through_alias in [false, true] {
        let mut definition = mlp(&[1,1], "identity", 7).unwrap();
        definition.parameters[1].frozen = !through_alias;
        definition.parameters[1].value = Value::Resident { data: vec![3.] };
        if through_alias {
            definition.parameters.push(Parameter {
                name: "frozen_bias".into(), shape: vec![1], frozen: true,
                value: Value::View { source: "layer0.bias".into(), start: 0 },
            });
        }
        let mut model = Trainer::new(definition, Path::new("."), MEMORY, Optimizer::Adam).unwrap();
        model.step(&[vec![1.]], Objective::Mse(&[vec![0.]]), 0.03).unwrap();
        let before = model.checkpoint_bytes_bounded(MEMORY).unwrap();
        let artifact = |fixed| weight_artifact::encode(&[
            ("layer0.weight".into(), vec![1,1], vec![7.]),
            ("layer0.bias".into(), vec![1], vec![fixed]),
        ], MEMORY).unwrap();
        let error = model.bind_weights(&artifact(4.), "sgd").unwrap_err();
        assert!(error.to_string().contains("frozen"));
        assert_eq!(model.checkpoint_bytes_bounded(MEMORY).unwrap(), before);
        model.bind_weights(&artifact(3.), "sgd").unwrap();
        assert_eq!(model.weights_bytes(MEMORY).unwrap(), artifact(3.));
        let state: serde_json::Value = serde_json::from_slice(&model.checkpoint_bytes_bounded(MEMORY).unwrap()).unwrap();
        assert_eq!(state["optimizer"], "sgd");
        assert_eq!(state["steps"], 0);
    }
}
