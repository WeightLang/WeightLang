use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
use weightlang_wgt::torch_export;
use wgt_language::{
    adaptive::OPERATOR_CONTRACTS,
    model::{ModelDefinition, ModelPlan},
};

// Legacy source-snapshot fixture: bit-for-bit seeded initialization compatibility.
const SEQUENTIAL: &str = "wgt 0.5\nmodel Classifier\ninput x: f32[2]\noutput y: f32[1]\nlayer hidden: linear(3)\nlayer activation: relu\nlayer output: linear(1,bias=0)\ninit seed=7\n";
const FORWARD: &str = "wgt 0.5\nmodel Scale\ninput x: f32[2]\noutput y: f32[2]\nparameter gain: f32[2] = [2,3]\nfn forward(x) { return x * gain; }\n";

fn compile(source: &str) -> Value {
    serde_json::to_value(torch_export::compile_source(source).unwrap()).unwrap()
}

fn initial(parameter: &Value) -> Vec<f32> {
    let parameter: wgt_language::adaptive::Parameter = serde_json::from_value(parameter.clone()).unwrap();
    assert!(parameter.data.is_empty(), "compilation must not generate parameter data");
    parameter.initializer.unwrap().materialize(&parameter.shape).unwrap()
}

fn cli(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(directory)
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", directory.join("cache"));
    command
}

#[test]
fn sequential_export_preserves_shapes_names_and_initialization_recipe() {
    let architecture = compile(SEQUENTIAL);
    assert_eq!(architecture["format"], torch_export::FORMAT);
    assert_eq!(architecture["kind"], "sequential");
    assert_eq!(architecture["name"], "Classifier");
    assert_eq!(architecture["interface"]["input"], 2);
    assert_eq!(architecture["interface"]["output"], 1);
    assert_eq!(architecture["graph"]["seed"], 7);
    assert_eq!(architecture["graph"]["layers"][1]["op"], "relu");
    assert_eq!(architecture["graph"]["layers"][2]["bias"], false);
    assert_eq!(architecture["parameters"][0]["name"], "hidden.weight");
    assert_eq!(architecture["parameters"][0]["shape"], json!([3, 2]));
    assert_eq!(architecture["parameters"].as_array().unwrap().len(), 3);
    assert_eq!(architecture["source"], SEQUENTIAL);
    assert!(architecture.get("program").is_none());
}

#[test]
fn sequential_initializers_match_native_weights_bit_for_bit() {
    for seed in [0, 7, u64::MAX] {
        let source = SEQUENTIAL.replace("init seed=7", &format!("init seed={seed}"));
        let architecture = compile(&source);
        let (_, graph) = wgtime_core::source::graph(&source).unwrap();
        let native = graph.instantiate().unwrap();
        let exported = architecture["graph_parameters"].as_array().unwrap();
        assert_eq!(exported.len(), native.parameters.len());
        for (exported, parameter) in exported.iter().zip(native.parameters) {
            assert_eq!(exported["name"], parameter.name);
            assert_eq!(exported["shape"], json!(parameter.shape));
            let wgtime_core::Value::Resident { data } = parameter.value else {
                panic!("resident initializer")
            };
            let actual: Vec<f32> = initial(exported);
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                data.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    // Fixed native reference guards the initializer extraction itself, independent
    // of the two callers sharing its implementation.
    let first: Vec<f32> =
        initial(&compile(SEQUENTIAL)["graph_parameters"][0]);
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            3195480561, 3213328972, 1063306776, 1043991872, 3184878585, 3205269803
        ]
    );
}

#[test]
fn source_module_metadata_preserves_instance_identity_and_parameter_sharing() {
    let source = r#"wgt 0.5
model Root
input x:f32[1]
output y:f32[1]
model Block {
input a:f32[1]
output b:f32[1]
parameter weight:f32[1]=[2]
fn forward(a) { return a*weight; }
}
blocks:Block[2];
fn forward(x) { return blocks[0](blocks[0](x))+blocks[1](x); }
"#;
    let architecture = compile(source);
    assert_eq!(
        architecture["modules"],
        json!([
            {"path":"blocks[0]","model":"Block","input":1,"output":1,"parameters":{"weight":"blocks_0_weight"}},
            {"path":"blocks[1]","model":"Block","input":1,"output":1,"parameters":{"weight":"blocks_1_weight"}},
        ])
    );
    let mut program: wgt_language::adaptive::Program =
        serde_json::from_value(architecture["program"].clone()).unwrap();
    assert_eq!(program.parameters.len(), 2);
    let mut native_program = wgt_language::adaptive::parse(source).unwrap();
    for parameter in &mut native_program.parameters { parameter.data.clear(); }
    assert_eq!(serde_json::to_value(native_program).unwrap(), architecture["program"]);
    for parameter in &mut program.parameters {
        parameter.data = parameter.initializer.as_ref().unwrap().materialize(&parameter.shape).unwrap();
    }
    let mut native = wgtime_core::adaptive::AdaptiveSession::prepare(program, 65536).unwrap();
    assert_eq!(native.run(&[2.], 1000).unwrap(), [12.]);
}

#[test]
fn forward_export_has_native_ir_and_declared_values_without_execution() {
    let architecture = compile(FORWARD);
    assert_eq!(architecture["kind"], "forward");
    assert_eq!(
        initial(&architecture["program"]["parameters"][0]),
        vec![2., 3.]
    );
    assert_eq!(architecture["program"]["registers"][0]["shape"], json!([2]));
    assert_eq!(architecture["program"]["input"], "x");
    let ops: Vec<_> = architecture["program"]["instructions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|instruction| instruction["op"].as_str().unwrap())
        .collect();
    assert_eq!(ops, ["read", "mul", "return"]);
    assert!(architecture.get("graph").is_none());

    // Export never evaluates the forward graph, including its domain checks.
    let log = compile(&FORWARD.replace("return x * gain;", "return log(x * gain);"));
    assert!(
        log["program"]["instructions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|instruction| instruction["op"] == "log")
    );
}

#[test]
fn source_compilation_keeps_loss_frozen_and_float_parameters() {
    let source = FORWARD
        .replace("parameter gain", "frozen gain")
        .replace("fn forward", "learn scale: float = 1;\nfn forward")
        .replace("x * gain", "scale(x * gain,2)")
        + "fn loss(prediction,target) { return square(prediction-target).mean; }\n";
    let architecture = compile(&source);
    assert_eq!(architecture["parameters"][0]["frozen"], true);
    assert_eq!(architecture["parameters"][1]["learn"]["kind"], "float");
    assert_eq!(architecture["interface"]["target"], 2);
    assert!(architecture["program"]["loss"]["instructions"].is_array());
}

#[test]
fn probability_initializers_keep_one_explicit_sigmoid_and_frozen_values() {
    let source = "wgt 0.5\nmodel Probability\ninput x: f32[1]\noutput y: f32[1]\nparameter p: prob = 0.25\nfn forward(x) { return x * p; }\n";
    let architecture = compile(source);
    let value = initial(&architecture["program"]["parameters"][0])[0] as f64;
    assert!((value - (1.0_f64 / 3.0).ln()).abs() < 1e-6);
    let sigmoids = |program: &Value| {
        program["instructions"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|instruction| instruction["op"] == "sigmoid")
            .count()
    };
    assert_eq!(sigmoids(&architecture["program"]), 1);
    let frozen = compile(&source.replace("parameter p", "frozen p"));
    assert_eq!(initial(&frozen["program"]["parameters"][0]), [0.25]);
    assert_eq!(sigmoids(&frozen["program"]), 0);
}

#[test]
fn file_compilation_resolves_pure_imports_into_self_contained_source() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("twice.wgt"),
        "wgt 0.5\noperators Twice\nfn apply(x) { return x+x; }\n",
    )
    .unwrap();
    let source = FORWARD
        .replacen(
            "model Scale",
            "import helper from \"./twice.wgt\"\nmodel Scale",
            1,
        )
        .replace("x * gain", "helper.apply(x) * gain");
    let file = directory.path().join("model.wgt");
    fs::write(&file, source).unwrap();
    let architecture = torch_export::compile(&file).unwrap();
    assert!(!architecture.source.contains("import "));
    assert!(torch_export::compile_source(&architecture.source).is_ok());
    let standard = FORWARD
        .replacen(
            "model Scale",
            "import act from \"std/activation\"\nmodel Scale",
            1,
        )
        .replace("x * gain", "act.swish(x * gain)");
    assert!(torch_export::compile_source(&standard).is_ok());
}

#[test]
fn host_code_and_invalid_tensor_contracts_remain_rejected() {
    let host = format!("{FORWARD}\nfn main() {{save text \"no\" to \"must-not-exist\";}}\n");
    assert!(torch_export::compile_source(&host).is_err());
    assert!(torch_export::compile_source(&FORWARD.replace("x * gain", "x * [1]")).is_err());
    assert!(torch_export::compile_source("wgt 0.5\nmodel Bad\nstate value:f32[1]=[0];\nfn forward(x:f32[1])->f32[1] { value <- x; return value; }\n").is_err());
}

#[test]
fn every_native_instruction_survives_source_export_unchanged() {
    // These unused bindings are intentional: export must retain the complete IR,
    // including checks, working memory, writes and control flow with side effects.
    let source = r#"wgt 0.5
model Complete
input x:f32[4]
output y:f32[4]
parameter w:f32[4,4]=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
parameter b:f32[4]=[0,0,0,0]
parameter state:f32[4]=[0,0,0,0]
fn forward(x) {
    let joined=concat(x,x);
    let gathered=gather(w as [4,4],[0,2]);
    let selected=dynamic_slice(x,[1],2);
    let embedded=embedding(w as [4,4],[0,2],0);
    let image=x as [1,2,2];
    let kernel=[1] as [1,1,1,1];
    let convolved=conv(image,kernel,[0],[1,1],[0,0],[1,1],1);
    let maximum=max_pool2d(convolved,[2,2],[1,1]);
    let average=avg_pool2d(convolved,[2,2],[1,1]);
    let matrix=x as [2,2];
    let transposed=matrix.T;
    let permuted=permute(matrix,[1,0]);
    let product=matrix @ transposed;
    let batched=x as [1,2,2];
    let batch_product=batch_matmul(batched,batched);
    let normalized=layer_norm(matrix,[1,1],[0,0],1,0.00001);
    let axis_probabilities=softmax(matrix,1);
    let scaled=scale(x,0.5);
    let disconnected=detach(x);
    let exponential=exp(x);
    let logarithm=log(exponential);
    let log_probabilities=log_softmax(x);
    let smallest=minimum(x,x);
    let clipped=clip(x,-1,1);
    let total=sum(x);
    let compared=ge(x,x);
    let divided=div(x,exponential);
    let rooted=sqrt(exponential);
    let runtime_scaled=scale(x,mean(x));
    let bank=memory_alloc(4,4);
    let address=softmax(x);
    memory_write(bank,address,x);
    let retrieved=memory_read(bank,address);
    let resources=resource_state();
    write(state,retrieved);
    memory_free(bank);
    let stored=state;
    let dense=linear(x,w,b);
    let added=x+stored;
    let multiplied=x*stored;
    let hyperbolic=tanh(x);
    let sigmoid_value=sigmoid(x);
    let rectified=relu(x);
    let probabilities=softmax(x);
    let checked=probabilities.probabilities;
    let negated=-x;
    let average_value=mean(x);
    let gated=gate([0.5],x);
    let routed=route([0.5],x,stored);
    let first=slice(x,0,1);
    if positive(mean(x)) { return x; }
    return stored;
}
"#;
    let architecture = compile(source);
    let (ModelPlan::Forward(program), _) = ModelDefinition::parse(source)
        .unwrap()
        .compile_plan_with_modules()
        .unwrap()
    else {
        panic!("forward plan")
    };
    assert_eq!(
        architecture["program"],
        serde_json::to_value(&program).unwrap()
    );
    let actual: BTreeSet<_> = program
        .instructions
        .iter()
        .map(|instruction| instruction.contract().name)
        .collect();
    let expected: BTreeSet<_> = OPERATOR_CONTRACTS
        .iter()
        .map(|contract| contract.name)
        .collect();
    assert_eq!(
        actual, expected,
        "add a valid source example whenever the native instruction set grows"
    );
}

#[test]
fn constrained_learn_metadata_and_shadows_survive_export() {
    let source = "wgt 0.5\nmodel Rules\ninput x:f32[1]\noutput y:f32[1]\nlearn unbounded:float=2;\nlearn bounded:float in -1..=3=1;\nlearn positive:float > 0=0.5;\nlearn count:int in 0..=3=2;\nlearn enabled:bool=true;\nfn forward(x) { return x; }\n";
    let architecture = compile(source);
    let parameters = &architecture["program"]["parameters"];
    for (index, kind) in ["float", "bounded", "positive", "int", "bool"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(parameters[index]["learn"]["kind"], kind);
        assert_eq!(
            architecture["parameters"][index]["learn"],
            parameters[index]["learn"]
        );
    }
    assert_eq!(
        parameters[1]["learn"],
        json!({"kind":"bounded","low":-1.,"high":3.})
    );
    assert_eq!(parameters[2]["learn"], json!({"kind":"positive","low":0.}));
    assert_eq!(initial(&parameters[3]), [2.]);
    assert_eq!(initial(&parameters[4]), [1.]);
}

#[test]
fn typed_operations_export_shared_storage_and_complete_entry_schemas() {
    let source = r#"wgt 0.5
model Stateful
parameter gain:f32[1]=[2]
frozen baseline:f32[1]=[3]
learn count:int in 0..=4=2;
state value:f32[1]=[0];
fn observe() -> f32[1] { return value; }
fn accumulate(x:f32[1], y:f32[1]) -> f32[1] {
    value <- value+x+y;
    gain <- gain+x;
    return value*gain;
}
fn reset() { value <- [0]; }
"#;
    let architecture = compile(source);
    assert_eq!(architecture["kind"], "operations");
    assert!(architecture["interface"].is_null());
    assert_eq!(architecture["parameters"].as_array().unwrap().len(), 3);
    assert_eq!(
        initial(&architecture["operations"]["parameters"][0]),
        vec![2.]
    );
    assert_eq!(architecture["operations"]["parameters"][1]["frozen"], true);
    assert_eq!(
        architecture["operations"]["parameters"][2]["learn"]["kind"],
        "int"
    );
    assert_eq!(architecture["operations"]["states"][0]["name"], "value");
    assert_eq!(initial(&architecture["operations"]["states"][0]), [0.]);
    let entries = architecture["operations"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0]["schema"]["name"], "accumulate");
    assert_eq!(
        entries[0]["schema"]["inputs"],
        json!([{"name":"x","width":1},{"name":"y","width":1}])
    );
    assert_eq!(
        entries[0]["schema"]["effects"]["writes_state"],
        json!(["value"])
    );
    assert_eq!(
        entries[0]["schema"]["effects"]["writes_parameters"],
        json!(["gain"])
    );
    assert!(entries[2]["schema"]["output"].is_null());
    let ModelPlan::Operations(plan) = ModelDefinition::parse(source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("operations plan")
    };
    for (entry, (schema, program)) in entries.iter().zip(plan.entries) {
        assert_eq!(entry["schema"], serde_json::to_value(schema).unwrap());
        assert_eq!(entry["program"], serde_json::to_value(program).unwrap());
        for parameter in entry["program"]["parameters"].as_array().unwrap() {
            assert_eq!(parameter["data"], json!([]));
        }
    }
}

#[test]
fn cli_exports_json_without_python_or_project_environment() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("model.wgt"), SEQUENTIAL).unwrap();
    let result = cli(directory.path())
        .args(["compile", "model.wgt", "--target", "torch"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let architecture: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(architecture["format"], torch_export::FORMAT);
    assert_eq!(architecture["kind"], "sequential");
    assert!(!directory.path().join("cache").exists());
    assert!(!directory.path().join("weights").exists());
    assert!(!directory.path().join(".venv").exists());
}

#[test]
fn cli_accepts_stdin_and_preserves_existing_output() {
    let directory = tempfile::tempdir().unwrap();
    let mut child = cli(directory.path())
        .args(["compile", "-", "--target", "torch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(FORWARD.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let architecture: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        initial(&architecture["program"]["parameters"][0]),
        vec![2., 3.]
    );

    fs::write(directory.path().join("model.wgt"), FORWARD).unwrap();
    let result = cli(directory.path())
        .args(["compile", "-o", "architecture.json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let initial = fs::read(directory.path().join("architecture.json")).unwrap();
    fs::write(directory.path().join("model.wgt"), SEQUENTIAL).unwrap();
    let repeated = cli(directory.path())
        .args(["compile", "-o", "architecture.json"])
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("never overwritten"));
    assert_eq!(
        fs::read(directory.path().join("architecture.json")).unwrap(),
        initial
    );
}

#[test]
fn checking_and_packaging_validation_never_run_initializers() {
    let directory = tempfile::tempdir().unwrap();
    for initial in ["sqrt(full(-1))", "external"] {
        let source = FORWARD.replace("[2,3]", initial);
        let path = directory.path().join("model.wgt");
        fs::write(&path, &source).unwrap();
        let output = cli(directory.path()).args(["check", "model.wgt"]).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        weightlang_wgt::source_check::check_file(&path).unwrap();
        assert!(ModelDefinition::parse(&source).unwrap().instantiate_plan().is_err());
        // Legacy application validation shares the static boundary, while its
        // execution parser retains the explicit constructor behavior.
        fs::write(&path, format!("{source}\nfn main() {{return 0;}}\n")).unwrap();
        weightlang_wgt::source_check::check_file(&path).unwrap();
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn cli_dimension_binding_is_checked_and_both_export_targets_are_concrete() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("model.wgt");
    fs::write(&source, include_str!("../../../examples/native_training/signature_dimensions/model.wgt")).unwrap();
    let checked = cli(scratch.path()).args(["check", source.to_str().unwrap(), "--dim", "N=3"]).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
    for target in ["torch", "rust"] {
        let output = cli(scratch.path()).args(["compile", source.to_str().unwrap(), "--target", target, "--dim", "N=3"]).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        if target == "torch" {
            let json: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(json["interface"]["input"], 3);
            assert_eq!(json["interface"]["output"], 3);
        } else { assert!(String::from_utf8_lossy(&output.stdout).contains("f32[3]")); }
    }
    for args in [vec!["--dim", "TYPO=2"], vec!["--dim", "N=0"], vec!["--dim", "N=2", "--dim", "N=3"]] {
        let output = cli(scratch.path()).args(["check", source.to_str().unwrap()]).args(args).output().unwrap();
        assert!(!output.status.success());
    }
}

#[test]
fn explicit_torch_module_target_compiles_and_checks_without_python() {
    let directory = tempfile::tempdir().unwrap();
    let source = "wgt 0.5\ntarget torch\nhost example.host\nextern nn\nmodel Example extends Module {\ninit(self) {\nraise RuntimeError('must not execute');\n}\nfn forward(self, x) {\nreturn x;\n}\n}\n";
    fs::write(directory.path().join("model.wgt"), source).unwrap();
    let exported = compile(source);
    assert_eq!(exported["kind"], "torch_module");
    assert_eq!(exported["torch_module"]["models"], json!(["Example"]));
    assert!(exported["parameters"].as_array().unwrap().is_empty());
    assert!(cli(directory.path()).args(["check", "model.wgt"]).output().unwrap().status.success());
    assert!(cli(directory.path()).args(["compile", "model.wgt", "--target", "torch"]).output().unwrap().status.success());
    assert!(!cli(directory.path()).args(["compile", "model.wgt", "--target", "rust"]).output().unwrap().status.success());
}
