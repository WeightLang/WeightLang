use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use wgt_language::ide::{self, Analysis};

fn forward(body: &str) -> String {
    format!(
        "wgt 0.5\nmodel Matrices\ninput x: f32[12]\noutput y: f32[4]\nparameter weight: f32[2,3] = [1,2,3,4,5,6]\nfrozen bias: f32[2] = [0,0]\nfn forward(x) {{\n{body}\n}}\n"
    )
}

fn valid() -> String {
    forward(
        "let left: f32[rows=2, inner=3] = x[0:6];\nlet right: f32[inner=3, cols=2] = x[6:12];\nlet product = left @ right;\nreturn product as [4];",
    )
}

fn assert_valid(analysis: &Analysis) {
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
}

fn line(source: &str, text: &str) -> usize {
    source.lines().position(|line| line.contains(text)).unwrap() + 1
}

#[test]
fn compiler_types_preserve_matrix_axes_and_count_parameter_elements() {
    let source = valid();
    let analysis = ide::analyze(&source, None);
    assert_valid(&analysis);
    let product = analysis
        .bindings
        .iter()
        .find(|binding| binding.name == "product")
        .unwrap();
    assert_eq!(product.function, "forward");
    assert_eq!(product.line, line(&source, "let product"));
    assert_eq!(product.shape, [2, 2]);
    assert_eq!(product.axes, [Some("rows".into()), Some("cols".into())]);
    let model = analysis.model.unwrap();
    assert_eq!(model.parameter_count, 8);
    assert_eq!(model.trainable_parameter_count, 6);
    assert!(model.states.is_empty());
}

#[test]
fn matrix_reshape_and_arity_failures_use_the_original_statement() {
    for source in [
        valid().replace("f32[inner=3, cols=2]", "f32[inner=2, cols=3]"),
        valid().replace("left @ right", "matmul(left)"),
        valid().replace("left @ right", "left as [5]"),
    ] {
        let analysis = ide::analyze(&source, None);
        assert_eq!(analysis.diagnostics.len(), 1, "{:?}", analysis.diagnostics);
        assert_eq!(analysis.diagnostics[0].line, line(&source, "let product"));
        assert!(analysis.model.is_none());
        assert!(
            analysis.bindings.is_empty(),
            "invalid programs must not show confirmed shapes"
        );
    }
}

#[test]
fn named_axis_mismatch_is_reported_by_the_tensor_contract() {
    let source = valid().replace("f32[inner=3, cols=2]", "f32[other=3, cols=2]");
    let analysis = ide::analyze(&source, None);
    assert_eq!(analysis.diagnostics[0].code, "invalid-axis");
    assert_eq!(analysis.diagnostics[0].line, line(&source, "let product"));
}

#[test]
fn typed_operations_keep_independent_local_shapes_and_state_counts() {
    let source = "wgt 0.5\nmodel Operations\nstate memory: f32[4] = [0,0,0,0]\nparameter gain: f32[1] = [1]\nfn forward(x: f32[4]) -> f32[4] {\nlet matrix = x as [2,2];\nreturn matrix as [4];\n}\nfn inspect(x: f32[6]) -> f32[6] {\nlet matrix = x as [2,3];\nreturn matrix as [6];\n}\n";
    let analysis = ide::analyze(source, None);
    assert_valid(&analysis);
    let shapes: BTreeMap<_, _> = analysis
        .bindings
        .iter()
        .filter(|binding| binding.name == "matrix")
        .map(|binding| (binding.function.as_str(), binding.shape.clone()))
        .collect();
    assert_eq!(shapes["forward"], [2, 2]);
    assert_eq!(shapes["inspect"], [2, 3]);
    let model = analysis.model.unwrap();
    assert_eq!(model.parameter_count, 1);
    assert_eq!(model.trainable_parameter_count, 1);
    assert_eq!(model.states[0].elements, 4);
}

#[test]
fn standard_imports_preserve_user_shapes_and_error_line_mapping() {
    let source = valid()
        .replace(
            "model Matrices",
            "import activation from \"std/activation\"\nmodel Matrices",
        )
        .replace(
            "let product = left @ right;",
            "let product = softmax(activation.swish(left @ right), \"cols\");",
        );
    let analysis = ide::analyze(&source, None);
    assert_valid(&analysis);
    let product = analysis
        .bindings
        .iter()
        .find(|binding| binding.name == "product")
        .unwrap();
    assert_eq!(product.shape, [2, 2]);
    assert_eq!(product.line, line(&source, "let product"));
    let invalid = source.replace(
        "activation.swish(left @ right)",
        "activation.swish(left @ right, x)",
    );
    let analysis = ide::analyze(&invalid, None);
    assert_eq!(analysis.diagnostics[0].line, line(&invalid, "let product"));
    assert!(
        analysis.diagnostics[0].message.contains("arity")
            || analysis.diagnostics[0].message.contains("argument")
    );
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join("weightlang").join(format!(
            "ide-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn unsaved_root_and_import_overlays_are_checked_without_writes() {
    let directory = Directory::new();
    let path = directory.0.join("model.wgt");
    let imported = directory.0.join("operations.wgt");
    let source = "wgt 0.5\nimport ops from \"./operations.wgt\"\nmodel Buffer\ninput x: f32[4]\noutput y: f32[4]\nfn forward(x) {\nlet matrix = ops.matrix(x);\nreturn matrix as [4];\n}\n";
    let overlays = BTreeMap::from([(
        imported.clone(),
        "wgt 0.5\noperators Geometry\nfn matrix(x) { return x as [2,2]; }\n".into(),
    )]);
    let analysis = ide::analyze_with_overlays(source, Some(&path), &overlays);
    assert_valid(&analysis);
    assert!(analysis.dependencies.contains(&path));
    assert!(analysis.dependencies.contains(&imported));
    assert_eq!(
        analysis
            .bindings
            .iter()
            .find(|binding| binding.name == "matrix")
            .unwrap()
            .shape,
        [2, 2]
    );
    assert!(!path.exists() && !imported.exists());
    std::fs::write(
        &imported,
        "wgt 0.5\noperators Saved\nfn matrix(x) { return x; }\n",
    )
    .unwrap();
    let analysis = ide::analyze_with_overlays(source, Some(&path), &overlays);
    assert_valid(&analysis);
    assert_eq!(
        analysis
            .bindings
            .iter()
            .find(|binding| binding.name == "matrix")
            .unwrap()
            .shape,
        [2, 2]
    );
    let analysis = ide::analyze(source, Some(&path));
    assert_valid(&analysis);
    assert_eq!(
        analysis
            .bindings
            .iter()
            .find(|binding| binding.name == "matrix")
            .unwrap()
            .shape,
        [4]
    );
}

#[test]
fn import_failures_anchor_at_the_import_and_do_not_claim_shapes() {
    let source = valid().replace(
        "model Matrices",
        "import missing from \"std/does_not_exist\"\nmodel Matrices",
    );
    let analysis = ide::analyze(&source, None);
    assert_eq!(analysis.diagnostics[0].line, 2);
    assert!(analysis.bindings.is_empty());
}

#[test]
fn library_helpers_are_checked_without_inventing_argument_shapes() {
    let valid = "wgt 0.5\noperators Matrix\nfn multiply(left,right) { return left @ right; }\n";
    let analysis = ide::analyze(valid, None);
    assert_valid(&analysis);
    assert!(analysis.bindings.is_empty());
    let invalid = "wgt 0.5\noperators Matrix\nfn multiply(left,right) {\nreturn matmul(left);\n}\n";
    let analysis = ide::analyze(invalid, None);
    assert_eq!(analysis.diagnostics[0].line, 4);
}

#[test]
fn output_and_transaction_shapes_are_native_errors() {
    let bad_output = valid().replace("output y: f32[4]", "output y: f32[3]");
    let analysis = ide::analyze(&bad_output, None);
    assert!(
        analysis.diagnostics[0]
            .message
            .contains("return/output shape mismatch")
    );
    let bad_state = "wgt 0.5\nmodel State\nstate memory: f32[2] = [0,0]\nfn step(x: f32[3]) {\nmemory <- x;\n}\n";
    let analysis = ide::analyze(bad_state, None);
    assert!(analysis.diagnostics[0].message.contains("shape mismatch"));
    assert_eq!(analysis.diagnostics[0].line, line(bad_state, "memory <-"));
}

#[test]
fn submodel_functions_do_not_steal_root_argument_locations() {
    let source = "wgt 0.5\n# A separate submodel has a same-named forward.\nmodel Inner {\ninput value: f32[2]\noutput result: f32[2]\nfn forward(value) { return value; }\n}\nmodel Outer\ninput x: f32[2]\noutput y: f32[2]\nfn forward(x) { return x; }\n";
    let analysis = ide::analyze(source, None);
    assert_valid(&analysis);
    assert_eq!(
        analysis
            .bindings
            .iter()
            .find(|binding| binding.name == "x")
            .unwrap()
            .line,
        line(source, "fn forward(x)")
    );
    let invalid = source.replace("return value;", "return value as [3];");
    let analysis = ide::analyze(&invalid, None);
    assert_eq!(analysis.diagnostics[0].line, line(source, "model Outer"));
}

#[test]
fn source_and_import_readers_reject_directories() {
    let directory = Directory::new();
    let imported = directory.0.join("operations.wgt");
    std::fs::create_dir(&imported).unwrap();
    let source = "wgt 0.5\nimport ops from \"./operations.wgt\"\nmodel Buffer\ninput x: f32[1]\noutput y: f32[1]\nfn forward(x) { return x; }\n";
    let analysis = ide::analyze(source, Some(&directory.0.join("model.wgt")));
    assert_eq!(analysis.diagnostics.len(), 1);
    assert!(analysis.diagnostics[0].message.contains("regular file"));
    assert_eq!(analysis.diagnostics[0].line, 2);
    let error = wgt_language::model::ModelDefinition::from_file(&directory.0).unwrap_err();
    assert!(error.to_string().contains("regular file"));
}

#[cfg(unix)]
#[test]
fn source_reader_rejects_a_device_without_reading_it() {
    let error = wgt_language::model::ModelDefinition::from_file("/dev/null").unwrap_err();
    assert!(error.to_string().contains("regular file"));
}
