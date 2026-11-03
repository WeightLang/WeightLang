use serde_json::{json, Value};
use std::{collections::BTreeMap, fs};
use weightlang_wgt::{
    artifact::{hash, Artifact},
    contracts::{infer, Contracts, TensorType},
    graph::Graph,
};

fn infer_layers(input: Value, layers: Value) -> anyhow::Result<TensorType> {
    infer(
        &layers,
        "",
        TensorType::parse(&input)?,
        &mut BTreeMap::new(),
        &mut Contracts::new(),
    )
}

fn transform(input: Value, op: &str, args: Value) -> anyhow::Result<TensorType> {
    infer_layers(input, json!([{"name":"transform", "op":op, "args":args}]))
}

#[test]
fn grouped_permutations_and_reductions_infer_names_without_weights() {
    let input = json!({"dtype":"f32", "shape":[6,8], "axes":[null,"features"]});
    let layers = json!([
        {"name":"split", "op":"rearrange", "args":{"pattern":"(h w) features -> h features w", "sizes":{"h":2}}},
        {"name":"pool", "op":"reduce", "args":{"pattern":"h features w -> features h", "sizes":{}, "reduction":"mean"}},
        {"name":"join", "op":"rearrange", "args":{"pattern":"features h -> (h features)", "sizes":{}}}
    ]);
    let mut weights = BTreeMap::new();
    let mut contracts = Contracts::new();
    let out = infer(
        &layers,
        "",
        TensorType::parse(&input).unwrap(),
        &mut weights,
        &mut contracts,
    )
    .unwrap();
    assert_eq!(out.shape, [16]);
    assert_eq!(out.axes, [None]);
    assert!(weights.is_empty());
    assert_eq!(contracts["split"].1.label(), "f32[h=2,features=8,w=3]");
    assert_eq!(contracts["pool"].1.label(), "f32[features=8,h=2]");
    for reduction in ["mean", "sum"] {
        let out = transform(
            json!({"dtype":"f32","shape":[2,3,4]}),
            "reduce",
            json!({"pattern":"a b c -> (c a)","sizes":{},"reduction":reduction}),
        )
        .unwrap();
        assert_eq!(out.shape, [8]);
    }
    let out = transform(
        json!({"dtype":"i64","shape":[6],"axes":["tokens"]}),
        "rearrange",
        json!({"pattern":"(a b) -> b a","sizes":{"a":2,"b":3}}),
    )
    .unwrap();
    assert_eq!(out.label(), "i64[b=3,a=2]");
    let out = transform(
        json!({"dtype":"f32","shape":[2,3]}),
        "reduce",
        json!({"pattern":"a b -> b a","sizes":{},"reduction":"sum"}),
    )
    .unwrap();
    assert_eq!(out.shape, [3, 2]);
}

#[test]
fn malformed_patterns_sizes_reductions_and_overflow_are_rejected() {
    let input = json!({"dtype":"f32","shape":[6]});
    for args in [
        json!({"pattern":"a -> a -> a","sizes":{}}),
        json!({"pattern":"a a -> a","sizes":{}}),
        json!({"pattern":"a -> a a","sizes":{}}),
        json!({"pattern":"a -> b","sizes":{}}),
        json!({"pattern":"(a b) -> a b","sizes":{}}),
        json!({"pattern":"(a b) -> a b","sizes":{"a":4}}),
        json!({"pattern":"(a b) -> a b","sizes":{"a":2,"b":4}}),
        json!({"pattern":"a -> a","sizes":{"a":0}}),
        json!({"pattern":"a -> a","sizes":{"a":true}}),
        json!({"pattern":"a -> a","sizes":{"a":2.5}}),
        json!({"pattern":"a -> a","sizes":{"a":-1}}),
        json!({"pattern":"a -> a","sizes":{"unused":2}}),
        json!({"pattern":"a -> a","sizes":[]}),
        json!({"pattern":123,"sizes":{}}),
        json!({"pattern":"((a)) -> a","sizes":{}}),
        json!({"pattern":"() -> a","sizes":{}}),
        json!({"pattern":"(a -> a","sizes":{}}),
        json!({"pattern":"(a)b -> a b","sizes":{}}),
        json!({"pattern":"a -> ...","sizes":{}}),
        json!({"pattern":"a -> 1 a","sizes":{}}),
    ] {
        assert!(
            transform(input.clone(), "rearrange", args.clone()).is_err(),
            "{args}"
        );
    }
    for reduction in [json!("max"), Value::Null, json!(3)] {
        assert!(transform(
            json!({"dtype":"f32","shape":[2,3]}),
            "reduce",
            json!({"pattern":"a b -> a","sizes":{},"reduction":reduction})
        )
        .is_err());
    }
    for op in ["rearrange"] {
        assert!(transform(
            input.clone(),
            op,
            json!({"pattern":"a ->","sizes":{},"reduction":"sum"})
        )
        .is_err());
    }
    assert!(transform(
        json!({"dtype":"i64","shape":[2,3]}),
        "reduce",
        json!({"pattern":"a b -> a","sizes":{},"reduction":"sum"})
    )
    .is_err());
    assert!(transform(
        json!({"dtype":"f32","shape":[u64::MAX,2]}),
        "rearrange",
        json!({"pattern":"a b -> (a b)","sizes":{}})
    )
    .is_err());
    assert!(transform(
        input,
        "rearrange",
        json!({"pattern":"(a b c) -> a b c","sizes":{"a":u64::MAX,"b":2}})
    )
    .is_err());
}

#[test]
fn named_contracts_validate_and_records_must_be_exact() {
    for axes in [
        json!([]),
        json!(["x", "x"]),
        json!(["bad-name", null]),
        json!([true, null]),
        Value::Null,
    ] {
        assert!(TensorType::parse(&json!({"dtype":"f32","shape":[2,3],"axes":axes})).is_err());
    }
    let input = json!({"dtype":"f32","shape":[2,3],"axes":["time","features"]});
    let output = json!({"dtype":"f32","shape":[2,4],"axes":["time","features"]});
    let mut layer =
        json!({"name":"fc","op":"linear","args":{"in":3,"out":4},"input":input,"output":output});
    let out = infer_layers(input.clone(), json!([layer.clone()])).unwrap();
    assert_eq!(out.label(), "f32[time=2,features=4]");
    layer["output"]["axes"] = json!(["time", null]);
    assert!(infer_layers(input.clone(), json!([layer.clone()])).is_err());
    layer["output"] = output;
    layer["input"]["extra"] = json!(true);
    assert!(infer_layers(input.clone(), json!([layer])).is_err());
    assert!(infer_layers(
        input.clone(),
        json!([{"name":"fc","op":"linear","args":{"out":4}}])
    )
    .is_err());
    assert!(transform(
        input,
        "rearrange",
        json!({"pattern":"other features -> features other","sizes":{}})
    )
    .is_err());
    let unnamed = TensorType::parse(&json!({"dtype":"f32","shape":[2,4]})).unwrap();
    assert!(unnamed.compatible(&out));
    let conflicting =
        TensorType::parse(&json!({"dtype":"f32","shape":[2,4],"axes":["other",null]})).unwrap();
    assert!(!conflicting.compatible(&out));
}

#[test]
fn existing_operators_propagate_axis_identity() {
    let input = json!({"dtype":"f32","shape":[3,4],"axes":["time","feature"]});
    for op in ["meanpool"] {
        assert_eq!(
            infer_layers(input.clone(), json!([{"name":"pool","op":op}]))
                .unwrap()
                .label(),
            "f32[feature=4]"
        );
    }
    let out = infer_layers(
        input.clone(),
        json!([
            {"name":"fc","op":"linear","args":{"in":4,"out":6}},
            {"name":"norm","op":"layernorm","args":{"dim":6}},
            {"name":"activation","op":"relu"}
        ]),
    )
    .unwrap();
    assert_eq!(out.label(), "f32[time=3,feature=6]");
    assert_eq!(
        infer_layers(input.clone(), json!([{"name":"flat","op":"flatten"}]))
            .unwrap()
            .label(),
        "f32[12]"
    );
    let out = infer_layers(
        json!({"dtype":"i64","shape":[3],"axes":["time"]}),
        json!([
            {"name":"tokens","op":"embedding","args":{"vocab":10,"dim":4}}
        ]),
    )
    .unwrap();
    assert_eq!(out.label(), "f32[time=3,4]");
    let out = infer_layers(
        json!({"dtype":"f32","shape":[2,4,4],"axes":["channels","height","width"]}),
        json!([
            {"name":"conv","op":"conv2d","args":{"in":2,"out":3,"kernel":1,"stride":1,"padding":0}},
            {"name":"pool","op":"globalavgpool2d"}
        ]),
    )
    .unwrap();
    assert_eq!(out.label(), "f32[channels=3]");
}

#[test]
fn unnamed_residual_boundaries_are_refined_before_record_validation() {
    let input = json!({"dtype":"f32","shape":[4,4]});
    let named = json!({"dtype":"f32","shape":[4,4],"axes":["time","feature"]});
    let mut layers = json!([{"name":"body","op":"residual",
        "input":input, "output":named,
        "layers":[{"name":"label","op":"rearrange",
            "args":{"pattern":"time feature -> time feature","sizes":{}},
            "input":named, "output":named}]}]);
    assert_eq!(
        infer_layers(input.clone(), layers.clone()).unwrap().label(),
        "f32[time=4,feature=4]"
    );
    layers[0]["layers"][0]["args"]["pattern"] = json!("time feature -> feature time");
    // No saved contracts can mask a semantic swap at an unnamed boundary.
    layers[0].as_object_mut().unwrap().remove("input");
    layers[0].as_object_mut().unwrap().remove("output");
    layers[0]["layers"][0]
        .as_object_mut()
        .unwrap()
        .remove("input");
    layers[0]["layers"][0]
        .as_object_mut()
        .unwrap()
        .remove("output");
    assert!(infer_layers(input.clone(), layers).is_err());
}

#[test]
fn sequential_containers_keep_named_contracts_and_validate_nested_records() {
    let input = json!({"dtype":"f32","shape":[2,3],"axes":["time","feature"]});
    let output = json!({"dtype":"f32","shape":[3,2],"axes":["feature","time"]});
    let mut layers = json!([{"name":"body","op":"seq","input":input,"output":output,
        "layers":[{"name":"swap","op":"rearrange",
            "args":{"pattern":"time feature -> feature time","sizes":{}},
            "input":input,"output":output}]}]);
    assert_eq!(
        infer_layers(input.clone(), layers.clone()).unwrap().label(),
        "f32[feature=3,time=2]"
    );
    layers[0]["layers"][0]["output"]["axes"] = json!(["time", "feature"]);
    assert!(infer_layers(input, layers).is_err());
}

fn artifact(root: &std::path::Path) -> Value {
    let source = b"wgt 0.5\nmodel demo\n";
    let header = b"{}      ";
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend(header);
    fs::write(root.join("source.wgt"), source).unwrap();
    fs::write(root.join("weights.safetensors"), &weights).unwrap();
    json!({"format":"weightlang.artifact","format_version":3,"model":"demo",
        "source":{"file":"source.wgt","sha256":hash(source)},
        "weights":{"file":"weights.safetensors","sha256":hash(&weights),"tensors":{}},
        "interface":{"input":{"dtype":"f32","shape":[2,3],"axes":["time","features"]},
            "output":{"dtype":"f32","shape":[3],"axes":["features"]}},
        "layers":[{"name":"pool","op":"reduce","args":{"pattern":"time features -> features","sizes":{},"reduction":"mean"}}]})
}

#[test]
fn artifact_validation_and_graph_rendering_use_inferred_named_shapes() {
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let temp = tempfile::tempdir_in(scratch).unwrap();
    let root = temp.path();
    let mut manifest = artifact(root);
    let write = |manifest: &Value| {
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap()
    };
    write(&manifest);
    let graph = Graph::read(root).unwrap();
    assert_eq!(graph.parameters, 0);
    for format in ["json", "toon", "mermaid", "dot", "svg", "html", "md"] {
        assert!(
            graph.render(format).unwrap().contains("features=3"),
            "{format}"
        );
    }
    manifest["layers"][0]["output"] = json!({"dtype":"f32","shape":[3]});
    write(&manifest);
    assert!(Artifact::read(root).is_err());
    manifest["layers"][0]["output"] = json!({"dtype":"f32","shape":[3],"axes":["features"]});
    manifest["interface"]["output"]["axes"] = json!([null]);
    write(&manifest);
    assert!(Artifact::read(root).is_ok());
    manifest["interface"]["output"]["axes"] = json!(["wrong"]);
    write(&manifest);
    assert!(Artifact::read(root).is_err());
    manifest["interface"]["output"]["axes"] = json!(["features"]);
    manifest["layers"][0]["args"]["sizes"] = json!({"time":9});
    write(&manifest);
    assert!(Artifact::read(root).is_err());
}

#[cfg(unix)]
#[test]
fn check_shapes_reports_removed_python_cli() {
    use std::{os::unix::fs::PermissionsExt, process::Command};
    let scratch = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&scratch).unwrap();
    let temp = tempfile::tempdir_in(scratch).unwrap();
    fs::write(
        temp.path().join("wgt.mod"),
        "module local/shapes\nwgt 0.5\n",
    )
    .unwrap();
    fs::write(temp.path().join("model.wgt"), "wgt 0.5\nmodel demo\n").unwrap();
    let compiler = temp.path().join("compiler");
    fs::write(&compiler, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n").unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .env("WEIGHTC", &compiler)
        .env("WEIGHTLANG_CACHE", temp.path().join("cache"))
        .args(["check", "model.wgt", "--shapes", "--offline"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Python weightc CLI has been removed"));
    assert!(!temp.path().join("args.txt").exists());
}
