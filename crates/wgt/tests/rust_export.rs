use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use weightlang_wgt::rust_export::{self, DEFAULT_CRATE_PATH};
use wgt_language::model::{ModelDefinition, ModelPlan};

const GRAPH: &str = "wgt 0.5\nmodel Classifier\ninput x:f32[2]\noutput y:f32[1]\nlayer hidden:linear(3)\nlayer activation:relu\nlayer output:linear(1,bias=0)\ninit seed=7\n";
const FORWARD: &str = "wgt 0.5\nmodel Adaptive\ninput x:f32[1]\noutput y:f32[1]\nlearn depth:int in 0..=3=1;\nfrozen scale:f32[1]=[2]\nfn forward(x) { return repeat depth carrying h=x { h*scale }; }\nfn loss(prediction,target) { return 0.5*square(prediction-target).mean; }\n";
const OPERATIONS: &str = "wgt 0.5\nmodel Stateful\nparameter gain:f32[1]=[2]\nstate value:f32[1]=[0];\nfn observe() -> f32[1] { return value; }\nfn accumulate(x:f32[1],y:f32[1]) -> f32[1] { value <- value+x+y; gain <- gain+x; return value*gain; }\nfn reset() { value <- [0]; }\n";
const SCALE: &str = "wgt 0.5\nmodel Scale\ninput x:f32[2]\noutput y:f32[2]\nparameter gain:f32[2]=[2,3]\nfn forward(x) { return x*gain; }\n";

fn temporary_directory() -> tempfile::TempDir {
    let root = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("rust-export-")
        .tempdir_in(root)
        .unwrap()
}

fn cli(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wgt"));
    command
        .current_dir(directory)
        .env_clear()
        .env("PATH", "")
        .env("WEIGHTLANG_CACHE", directory.join("cache"))
        .env("WGTIME_ENDPOINT", directory.join("absent.socket"));
    command
}

fn native_plan(definition: &ModelDefinition) -> Value {
    match definition.instantiate_plan().unwrap() {
        ModelPlan::Graph(graph) => {
            json!({"kind":"graph", "parameters":graph.initial_parameters().unwrap(), "graph":graph})
        }
        ModelPlan::Forward(program) => json!({"kind":"forward", "program":program}),
        ModelPlan::Operations(plan) => json!({
            "kind":"operations", "parameters":plan.parameters,
            "states":plan.states, "entries":plan.entries,
        }),
    }
}

#[test]
fn file_and_stdin_export_need_no_python_or_service_and_never_overwrite() {
    let directory = temporary_directory();
    fs::write(directory.path().join("model.wgt"), FORWARD).unwrap();
    let result = cli(directory.path())
        .args(["compile", "--target", "rust", "-o", "model.rs"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let original = fs::read(directory.path().join("model.rs")).unwrap();
    let expected = rust_export::compile_source(FORWARD, DEFAULT_CRATE_PATH).unwrap();
    assert_eq!(original, expected.as_bytes());

    let mut stdin = cli(directory.path())
        .args(["compile", "-", "--target", "rust"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    stdin
        .stdin
        .take()
        .unwrap()
        .write_all(FORWARD.as_bytes())
        .unwrap();
    let result = stdin.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, original);

    let result = cli(directory.path())
        .args(["compile", "--target", "rust", "-o", "model.rs"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("never overwritten"));
    assert_eq!(
        fs::read(directory.path().join("model.rs")).unwrap(),
        original
    );
    assert!(!directory.path().join("cache").exists());
    assert!(!directory.path().join("absent.socket").exists());
}

#[test]
fn rust_target_rejects_host_code_bad_definitions_and_invalid_facade_paths() {
    for source in [
        format!("{SCALE}\nfn main() {{ return Scale(); }}\n"),
        SCALE.replace("x*gain", "x*[1]"),
        "# too large\n".repeat(100_000),
    ] {
        assert!(rust_export::compile_source(&source, DEFAULT_CRATE_PATH).is_err());
    }
    for path in [
        "",
        "::",
        "a::",
        "a::::b",
        "_",
        "std;panic!()",
        "wgt/*x*/",
        "self",
        "gen",
        "r#self",
    ] {
        assert!(
            rust_export::compile_source(SCALE, path).is_err(),
            "{path:?}"
        );
    }
    for path in ["::wgt", "renamed_wgt", "::renamed_wgt::facade", "r#type"] {
        assert!(rust_export::compile_source(SCALE, path).is_ok(), "{path:?}");
    }

    let directory = temporary_directory();
    fs::write(directory.path().join("model.wgt"), "not a model").unwrap();
    let result = cli(directory.path())
        .args(["compile", "--target", "rust", "-o", "model.rs"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!directory.path().join("model.rs").exists());
}

#[test]
fn torch_remains_default_and_rust_crate_path_is_explicit() {
    let directory = temporary_directory();
    fs::write(directory.path().join("model.wgt"), SCALE).unwrap();
    let result = cli(directory.path()).arg("compile").output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["format"], "weightlang.torch.v1");
    let result = cli(directory.path())
        .args(["compile", "--crate-path", "renamed_wgt"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--target rust"));

    let result = cli(directory.path())
        .args(["compile", "--target", "rust", "--crate-path", "renamed_wgt"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("renamed_wgt::ModelDefinition")
    );
}

#[test]
fn rust_export_keeps_frontend_import_and_validation_boundaries() {
    let imported = SCALE
        .replacen(
            "model Scale",
            "import act from \"std/activation\"\nmodel Scale",
            1,
        )
        .replace("x*gain", "act.swish(x*gain)");
    let resolved = ModelDefinition::parse(&imported).unwrap();
    assert!(!resolved.source().contains("import "));
    let output = rust_export::compile_source(&imported, DEFAULT_CRATE_PATH).unwrap();
    assert!(output.contains(&format!(
        "pub const SOURCE: &str = {:?};",
        resolved.source()
    )));

    let relative = imported.replace("std/activation", "./activation.wgt");
    assert!(rust_export::compile_source(&relative, DEFAULT_CRATE_PATH).is_err());
    // An undefined log input domain is not evaluated during architecture export.
    assert!(
        rust_export::compile_source(&SCALE.replace("x*gain", "log(x*gain)"), DEFAULT_CRATE_PATH)
            .is_ok()
    );
}

#[test]
fn generated_modules_compile_and_preserve_all_plans_without_wgt_source_files() {
    let directory = temporary_directory();
    let source_directory = directory.path().join("models");
    fs::create_dir_all(&source_directory).unwrap();
    let escaping = format!(
        "{FORWARD}# Unicode 模型; quotes \"; backslash \\; tab \t; NUL \0; raw delimiter \"{}\n",
        "#".repeat(300)
    );
    let mut modules = Vec::new();
    for (name, source, facade) in [
        ("graph", GRAPH, DEFAULT_CRATE_PATH),
        ("forward", escaping.as_str(), "renamed_wgt"),
        ("operations", OPERATIONS, "renamed_wgt"),
    ] {
        let folder = source_directory.join(name);
        fs::create_dir(&folder).unwrap();
        let path = folder.join("model.wgt");
        fs::write(&path, source).unwrap();
        let definition = ModelDefinition::from_file(&path).unwrap();
        let module = rust_export::compile(&path, facade).unwrap();
        fs::write(directory.path().join(format!("{name}.rs")), module).unwrap();
        modules.push((
            name,
            definition.source().to_owned(),
            native_plan(&definition),
        ));
    }

    let imports = source_directory.join("imports");
    fs::create_dir_all(imports.join("helpers")).unwrap();
    fs::write(
        imports.join("helpers/add.wgt"),
        "wgt 0.5\noperators Add\nfn twice(x) { return x+x; }\n",
    )
    .unwrap();
    fs::write(imports.join("helpers/twice.wgt"), "wgt 0.5\nimport add from \"./add.wgt\"\noperators Twice\nfn apply(x) { return add.twice(x); }\n").unwrap();
    let imported = SCALE
        .replacen(
            "model Scale",
            "import helper from \"./helpers/twice.wgt\"\nmodel Scale",
            1,
        )
        .replace("x*gain", "helper.apply(x)*gain");
    let path = imports.join("model.wgt");
    fs::write(&path, imported).unwrap();
    let definition = ModelDefinition::from_file(&path).unwrap();
    assert!(!definition.source().contains("import "));
    fs::write(
        directory.path().join("imports.rs"),
        rust_export::compile(&path, "renamed_wgt").unwrap(),
    )
    .unwrap();
    modules.push((
        "imports",
        definition.source().to_owned(),
        native_plan(&definition),
    ));

    // A consumer compiles and runs after every original WGT dependency is gone.
    // Both the default facade path and a Cargo-renamed dependency are exercised.
    fs::remove_dir_all(&source_directory).unwrap();
    let mut driver = String::from(
        r#"
extern crate renamed_wgt as wgt;
use wgt_language::model::{ModelDefinition, ModelPlan};
use serde_json::{Value, json};
fn native_plan(definition: &ModelDefinition) -> Value {
    match definition.instantiate_plan().unwrap() {
        ModelPlan::Graph(graph) => json!({"kind":"graph", "parameters":graph.initial_parameters().unwrap(), "graph":graph}),
        ModelPlan::Forward(program) => json!({"kind":"forward", "program":program}),
        ModelPlan::Operations(plan) => json!({"kind":"operations", "parameters":plan.parameters, "states":plan.states, "entries":plan.entries}),
    }
}
"#,
    );
    for (name, _, _) in &modules {
        driver.push_str(&format!("mod {name} {{ include!(\"{name}.rs\"); }}\n"));
    }
    driver.push_str("fn main() -> wgt::Result<()> {\n");
    for (name, source, expected) in &modules {
        driver.push_str(&format!(
            "let definition = {name}::definition()?;\nassert_eq!(definition.source(), {source:?});\nassert_eq!(definition.source(), {name}::SOURCE);\nassert_eq!(native_plan(&definition).to_string(), {expected:?});\n",
            expected = expected.to_string(),
        ));
    }
    driver.push_str("println!(\"verified 4 embedded model plans\");\nOk(())\n}\n");
    fs::write(directory.path().join("main.rs"), driver).unwrap();

    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let manifest = format!(
        "[package]\nname = \"wgt-rust-export-check\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[[bin]]\nname = \"check\"\npath = \"main.rs\"\n\n[dependencies]\nrenamed_wgt = {{ package = \"wgt-model\", path = {:?} }}\nwgt-language = {{ path = {:?} }}\nserde_json = \"1\"\n\n[workspace]\n",
        repository.join("crates/wgt-model"),
        repository.join("crates/wgt-language"),
    );
    fs::write(directory.path().join("Cargo.toml"), manifest).unwrap();
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repository.join("target"));
    let result = Command::new(env!("CARGO"))
        .current_dir(&repository)
        .args(["run", "--quiet", "--offline", "--manifest-path"])
        .arg(directory.path().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target)
        .env("WGTIME_ENDPOINT", directory.path().join("absent.socket"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap().trim(),
        "verified 4 embedded model plans"
    );
    assert!(!directory.path().join("absent.socket").exists());
}
