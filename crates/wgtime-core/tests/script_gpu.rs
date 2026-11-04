use std::path::Path;
use wgtime_core::script;

#[test]
fn cpu_selection_preserves_steps_and_weights() {
    let result = script::run(
        r#"wgt 0.5
fn main() {
    let m = mlp([1,1], "identity", 42, "adam");
    train m on [[1]] against [[2]] {rate:0.01;loss:mse;};
    let before = m([1]);
    let state = training_device(m, "cpu", 1048576, 8);
    assert state["device"] == "cpu" && state["steps"] == 1, "state lost";
    assert m([1]) == before, "weights changed";
    return training_status(m);
}"#,
        Path::new("."),
        10000,
    )
    .unwrap();
    assert_eq!(result.result["inference_device"], "cpu");
}

#[test]
fn invalid_selection_is_explicit() {
    for (call, message) in [
        ("training_device(m, \"cpu\", 0, 1)", "must be positive"),
        (
            "training_device(m, \"npu:0\", 1024, 1)",
            "NPU training is unsupported",
        ),
        ("training_status()", "expects 1 arguments"),
        ("wgsl_run(0, [], 1)", "expects four arguments"),
    ] {
        let source = format!(
            "wgt 0.5\nfn main() {{let m=mlp([1,1],\"identity\",1,\"sgd\"); return {call};}}"
        );
        let error = script::run(&source, Path::new("."), 10000).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{error:#}");
    }
    let source = "wgt 0.5\nfn main(){ let m=mlp([1,1],\"identity\",1,\"sgd\"); compute_budget(100,1024,2,1000); return training_device(m,\"vulkan:0\",1024,1); }";
    assert!(format!(
        "{:#}",
        script::run(source, Path::new("."), 10000).unwrap_err()
    )
    .contains("unavailable under host or compute budgets"));
}

#[test]
fn new_examples_parse_without_gpu_execution() {
    for source in [
        r#"wgt 0.5
fn main() { let m=mlp([1,1], "identity", 1, "adam"); train m on [[1]] against [[2]] {rate:0.01;loss:mse;}; let state=training_device(m,"cpu",1048576,8); return training_status(m); }"#,
        r#"wgt 0.5
fn main() { let k=wgsl_kernel("@group(0) @binding(0) var<storage,read> x:array<f32>; @group(0) @binding(1) var<storage,read_write> y:array<f32>; @compute @workgroup_size(1) fn main(){y[0]=x[0];}","main","cpu",1024); return wgsl_run(k,[1],1); }"#,
    ] {
        script::parse(source).unwrap();
        script::codegen::rust_source(source, ".").unwrap();
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn cpu_build_rejects_gpu_and_wgsl_without_fallback() {
    for call in [
        "training_device(mlp([1,1],\"identity\",1,\"sgd\"),\"vulkan:0\",1024,1)",
        "wgsl_kernel(\"x\",\"main\",\"vulkan:0\",1024)",
    ] {
        let source = format!("wgt 0.5\nfn main() {{return {call};}}");
        assert!(format!(
            "{:#}",
            script::run(&source, Path::new("."), 10000).unwrap_err()
        )
        .contains("GPU backend not compiled"));
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires real GPU; serial hardware acceptance"]
fn real_gpu_scripts_and_wgsl_validation() {
    let hardware = wgtime_core::execution::devices();
    let device = hardware
        .iter()
        .find(|d| d.id != "cpu")
        .expect("hardware GPU required");
    eprintln!("script GPU: {} {}", device.id, device.name);
    for source in [
        r#"wgt 0.5
fn main() { let m=mlp([1,1],"identity",1,"adam"); training_device(m,"{}",1048576,8); return training_status(m); }"#,
        r#"wgt 0.5
fn main() { let k=wgsl_kernel("@group(0) @binding(0) var<storage,read> x:array<f32>; @group(0) @binding(1) var<storage,read_write> y:array<f32>; @compute @workgroup_size(1) fn main(){y[0]=x[0];}","main","{}",1024); return wgsl_run(k,[1],1); }"#,
    ] {
        let source = source.replace("{}", &device.id);
        let report = script::run(&source, Path::new("."), 100000).unwrap();
        eprintln!("{}", serde_json::to_string(&report).unwrap());
    }
    let overflow = format!(
        r#"wgt 0.5
fn main() {{
    let k = wgsl_kernel("@group(0) @binding(0) var<storage,read> x:array<f32>; @group(0) @binding(1) var<storage,read_write> y:array<f32>; @compute @workgroup_size(1) fn main(){{y[0]=x[0];}}", "main", "{}", 1024);
    return wgsl_run(k, [1], 1, [4294967297, 1, 1]);
}}"#,
        device.id
    );
    assert!(format!(
        "{:#}",
        script::run(&overflow, Path::new("."), 10000).unwrap_err()
    )
    .contains("expected integer in 0..1000000000"));
    assert!(
        wgtime_core::wgsl::Kernel::new("invalid shader", "main", &device.id, 1024).is_err()
    );
    let source = "@group(0) @binding(0) var<storage,read> x:array<f32>; @group(0) @binding(1) var<storage,read_write> y:array<f32>; @compute @workgroup_size(1) fn main(){y[0]=x[0];}";
    let mut kernel =
        wgtime_core::wgsl::Kernel::new(source, "main", &device.id, 1024).unwrap();
    assert!(kernel.run(&[1.], 1, [0, 1, 1]).is_err());
    assert!(kernel.run(&[1.], 1000, [1, 1, 1]).is_err());
    assert!(kernel.run(&[f32::NAN], 1, [1, 1, 1]).is_err());
    assert_eq!(kernel.run(&[7.], 2, [1, 1, 1]).unwrap(), [7., 0.]);
    assert_eq!(kernel.run(&[3.], 1, [1, 1, 1]).unwrap(), [3.]);
}
