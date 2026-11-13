use std::collections::BTreeMap;
use wgt_language::{dimensions::Dimensions, model_template::ModelTemplate};

// M describes sample rows; K remains a fixed feature width. Binding M changes
// activation geometry without initializing the declared external parameter.
const MATRIX: &str = "wgt 0.5\ndim M = 2\ndim K = 3\ndim N = 2\nmodel Projection\nparameter weight: f32[K,N]\nfn forward(x: f32[M,K]) -> f32[M,N] {\nreturn x @ (weight as [K,N]);\n}\n";

fn shapes(shape: &[usize]) -> BTreeMap<String, Vec<usize>> {
    BTreeMap::from([("x".into(), shape.to_vec())])
}
fn dims(name: &str, value: usize) -> Dimensions {
    BTreeMap::from([(name.into(), value)])
}

#[test]
fn shape_binding_overrides_defaults_and_returns_uninitialized_fixed_definitions() {
    let template = ModelTemplate::parse(MATRIX).unwrap();
    for rows in [1, 7] {
        let definition = template
            .bind_input_shapes(&shapes(&[rows, 3]), &Dimensions::new())
            .unwrap();
        let interface = definition.interface().unwrap();
        assert_eq!((interface.input, interface.output), (rows * 3, rows * 2));
        assert_eq!(definition.parameters()[0].shape, [3, 2]);
        // Checking/lowering accepts external weights, materialization does not.
        assert!(definition.compile_plan_with_modules().is_ok());
        assert!(definition.instantiate_plan().is_err());
    }
}

#[test]
fn explicit_bindings_must_agree_with_inference_and_non_input_symbols_need_values() {
    let template = ModelTemplate::parse(&MATRIX.replace("dim N = 2", "dim N")).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[7, 3]), &Dimensions::new())
            .unwrap_err()
            .to_string()
            .contains("unbound dimension: N")
    );
    let bound = template
        .bind_input_shapes(&shapes(&[7, 3]), &dims("N", 4))
        .unwrap();
    assert_eq!(bound.interface().unwrap().output, 28);
    let template = ModelTemplate::parse(MATRIX).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[7, 3]), &dims("M", 2))
            .unwrap_err()
            .to_string()
            .contains("conflict")
    );
    assert!(
        template
            .bind_input_shapes(&shapes(&[7, 3]), &dims("M", 7))
            .is_ok()
    );
    for bindings in [dims("TYPO", 2), dims("N", 0), dims("N", 16385)] {
        assert!(
            template
                .bind_input_shapes(&shapes(&[7, 3]), &bindings)
                .is_err()
        );
    }
}

#[test]
fn fixed_repeated_and_rank_contracts_are_axis_sensitive() {
    let fixed = ModelTemplate::parse(&MATRIX.replace("f32[M,K]", "f32[M,3]")).unwrap();
    assert!(
        fixed
            .bind_input_shapes(&shapes(&[2, 4]), &Dimensions::new())
            .unwrap_err()
            .to_string()
            .contains("axis 1")
    );
    let repeated = ModelTemplate::parse(&MATRIX.replace("f32[M,K]", "f32[M,M]")).unwrap();
    assert!(
        repeated
            .bind_input_shapes(&shapes(&[2, 3]), &Dimensions::new())
            .unwrap_err()
            .to_string()
            .contains("conflict")
    );
    for shape in [
        vec![],
        vec![6],
        vec![1, 2, 3],
        vec![0, 3],
        vec![16385, 3],
        vec![8192, 3],
        vec![usize::MAX, 3],
    ] {
        assert!(
            fixed
                .bind_input_shapes(&shapes(&shape), &Dimensions::new())
                .is_err()
        );
    }
    assert!(
        fixed
            .bind_input_shapes(&BTreeMap::new(), &Dimensions::new())
            .is_err()
    );
    assert!(
        fixed
            .bind_input_shapes(
                &BTreeMap::from([("wrong".into(), vec![2, 3])]),
                &Dimensions::new()
            )
            .is_err()
    );
    assert!(
        fixed
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![2, 3]), ("extra".into(), vec![2, 3])]),
                &Dimensions::new()
            )
            .is_err()
    );
}

#[test]
fn binding_uses_full_frontend_and_does_not_accept_invalid_model_bodies() {
    let source = MATRIX.replace("return x @ (weight as [K,N]);", "return x as [1,6];");
    let template = ModelTemplate::parse(&source).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[2, 3]), &Dimensions::new())
            .is_err()
    );
    // String and comment contents do not close bodies or declare dimensions.
    let source = MATRIX.replace("return x @", "# dim M = 99 }\nreturn x @");
    assert!(
        ModelTemplate::parse(&source)
            .unwrap()
            .bind_input_shapes(&shapes(&[7, 3]), &Dimensions::new())
            .is_ok()
    );
}

#[test]
fn unsupported_entry_forms_and_invalid_declarations_are_rejected() {
    for source in [
        MATRIX.replace("fn forward(x: f32[M,K])", "fn forward(x)"),
        MATRIX.replace("fn forward", "async fn forward"),
        MATRIX.replace("fn forward", "fn other"),
        MATRIX.replace("f32[M,K]", "f32[UNDECLARED,K]"),
        MATRIX.replace("dim M = 2", "dim M = 0"),
        MATRIX.replace("dim M = 2", "dim M = 2\ndim M = 3"),
        MATRIX.replace("model Projection", "target torch\nmodel Projection"),
        MATRIX.replace(
            "model Projection",
            "model Projection\nstate hidden: f32[2] = zeros()",
        ),
        "wgt 0.5\nmodel Empty\n".into(),
    ] {
        assert!(ModelTemplate::parse(&source).is_err(), "{source}");
    }
    assert!(ModelTemplate::parse(&" ".repeat(1024 * 1024 + 1)).is_err());
}

#[test]
fn vector_forward_loss_uses_existing_signature_frontend() {
    let source = include_str!("../../../examples/native_training/signature_dimensions/model.wgt");
    let template = ModelTemplate::parse(source).unwrap();
    let definition = template
        .bind_input_shapes(&shapes(&[5]), &Dimensions::new())
        .unwrap();
    let interface = definition.interface().unwrap();
    assert_eq!(
        (interface.input, interface.output, interface.target),
        (5, 5, Some(5))
    );
}

#[test]
fn file_template_preserves_root_snapshot_and_independent_import_scope() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path = std::env::temp_dir().join("weightlang").join(format!(
        "model-template-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    let scratch = Scratch(path);
    let source = MATRIX
        .replace(
            "model Projection",
            "import ops from \"./ops.wgt\"\nmodel Projection",
        )
        .replace(
            "return x @ (weight as [K,N]);",
            "return ops.boost(x @ (weight as [K,N]));",
        );
    std::fs::write(
        scratch.0.join("ops.wgt"),
        "wgt 0.5\ndim M = 11\noperators Local\nfn boost(x) {return M * x;}\n",
    )
    .unwrap();
    let file = scratch.0.join("model.wgt");
    std::fs::write(&file, &source).unwrap();
    let template = ModelTemplate::from_file(&file).unwrap();
    std::fs::write(&file, "invalid root replacement").unwrap();
    let definition = template
        .bind_input_shapes(&shapes(&[7, 3]), &Dimensions::new())
        .unwrap();
    assert_eq!(definition.interface().unwrap().input, 21);
    assert!(definition.source().contains("11"));
    assert!(!definition.source().contains("dim M"));
    assert!(ModelTemplate::from_file(&scratch.0).is_err());
    std::fs::write(&file, " ".repeat(1024 * 1024 + 1)).unwrap();
    assert!(ModelTemplate::from_file(&file).is_err());
}

#[test]
fn fixed_templates_and_repeated_symbols_bind_without_changing_execution_syntax() {
    let fixed = MATRIX.replace("f32[M,K]", "f32[2,3]");
    let definition = ModelTemplate::parse(&fixed)
        .unwrap()
        .bind_input_shapes(&shapes(&[2, 3]), &Dimensions::new())
        .unwrap();
    assert_eq!(definition.interface().unwrap().input, 6);
    let repeated = MATRIX.replace("f32[M,K]", "f32[M,M]");
    assert!(
        ModelTemplate::parse(&repeated)
            .unwrap()
            .bind_input_shapes(&shapes(&[3, 3]), &Dimensions::new())
            .is_ok()
    );
    // Symbolic extents are available only to contract extraction. Ordinary
    // execution parsing still requires concrete integer extents.
    assert!(
        wgt_language::script::parse::parse_functions(
            "wgt 0.5\nfn forward(x: f32[M]) -> f32[M] {return x;}\n"
        )
        .is_err()
    );
}

#[test]
fn unrelated_typed_operations_are_rejected_after_full_lowering() {
    let source = "wgt 0.5\ndim N = 2\nmodel TwoEntries\nfn forward(x: f32[N]) -> f32[N] {return x;}\nfn other(y: f32[N]) -> f32[N] {return y;}\n";
    let template = ModelTemplate::parse(source).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .unwrap_err()
            .to_string()
            .contains("additional typed operations")
    );
}

#[test]
fn syntax_tour_shape_template_is_complete_and_binds_like_the_runnable_example() {
    let tour = include_str!("../../../language/syntax-tour/model.wgt");
    let (_, section) = tour.split_once("# T15 BEGIN").unwrap();
    let (_, body) = section.split_once('\n').unwrap();
    let (body, _) = body.split_once("# T15 END").unwrap();
    let source = body
        .lines()
        .map(|line| line.strip_prefix("# ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let definition = ModelTemplate::parse(&source)
        .unwrap()
        .bind_input_shapes(&shapes(&[7, 3]), &Dimensions::new())
        .unwrap();
    assert_eq!(definition.interface().unwrap().input, 21);
    assert_eq!(definition.interface().unwrap().output, 14);
    assert!(definition.compile_plan_with_modules().is_ok());
}

struct IdentityScratch(std::path::PathBuf);

impl IdentityScratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join("weightlang").join(format!(
            "template-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for IdentityScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const IMPORT_TEMPLATE: &str = "wgt 0.5\ndim M = 2\nimport ops from \"./ops.wgt\"\nmodel Snapshot\nfn forward(x: f32[M]) -> f32[M] {return ops.boost(x);}\n";
const IMPORT_ORIGINAL: &str = "wgt 0.5\ndim M = 11\noperators Local\nfn boost(x) {return M * x;}\n";

#[test]
fn fingerprint_requires_success_and_is_shape_independent_and_cloned() {
    let template = ModelTemplate::parse(MATRIX).unwrap();
    assert!(template.source_fingerprint().is_err());
    assert!(
        template
            .bind_input_shapes(&shapes(&[0, 3]), &Dimensions::new())
            .is_err()
    );
    assert!(template.source_fingerprint().is_err());
    template
        .bind_input_shapes(&shapes(&[2, 3]), &Dimensions::new())
        .unwrap();
    let fingerprint = template.source_fingerprint().unwrap();
    assert_eq!(fingerprint.len(), 64);
    assert!(fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()));
    template
        .bind_input_shapes(&shapes(&[7, 3]), &dims("N", 4))
        .unwrap();
    assert_eq!(template.source_fingerprint().unwrap(), fingerprint);
    assert_eq!(template.clone().source_fingerprint().unwrap(), fingerprint);
    let other = ModelTemplate::parse(&(MATRIX.to_owned() + "# different root\n")).unwrap();
    other
        .bind_input_shapes(&shapes(&[2, 3]), &Dimensions::new())
        .unwrap();
    assert_ne!(other.source_fingerprint().unwrap(), fingerprint);
}

#[test]
fn frozen_imports_survive_mutation_deletion_and_keep_independent_dimensions() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    std::fs::write(&root, "invalid physical root").unwrap();
    let original = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert!(original.source().contains("11"));
    let fingerprint = template.source_fingerprint().unwrap();
    let cloned = template.clone();
    std::fs::write(&import, "invalid replacement").unwrap();
    let same = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_eq!(same.source(), original.source());
    std::fs::remove_file(&root).unwrap();
    std::fs::remove_file(&import).unwrap();
    for snapshot in [&template, &cloned] {
        let bound = snapshot
            .bind_input_shapes(&shapes(&[7]), &Dimensions::new())
            .unwrap();
        let forward = bound
            .operations()
            .iter()
            .find(|op| op.name == "forward")
            .unwrap();
        assert_eq!(forward.inputs.len(), 1);
        assert_eq!(forward.inputs[0].width, 7);
        assert_eq!(forward.output, Some(7));
        assert!(bound.source().contains("11"));
        assert_eq!(snapshot.source_fingerprint().unwrap(), fingerprint);
    }
}

#[test]
fn unsuccessful_full_validation_does_not_freeze_imports() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    // Valid import resolution, but the return geometry fails model validation.
    std::fs::write(
        &import,
        "wgt 0.5\noperators Local\nfn boost(x) {return x[0:1];}\n",
    )
    .unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .is_err()
    );
    assert!(template.source_fingerprint().is_err());
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    let bound = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert!(bound.source().contains("11"));
    assert!(template.source_fingerprint().is_ok());
    let fresh = ModelTemplate::from_file(&root).unwrap();
    fresh
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_eq!(
        fresh.source_fingerprint().unwrap(),
        template.source_fingerprint().unwrap()
    );
    std::fs::write(&import, IMPORT_ORIGINAL.replace("11", "13")).unwrap();
    let changed = ModelTemplate::from_file(&root).unwrap();
    changed
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_ne!(
        changed.source_fingerprint().unwrap(),
        template.source_fingerprint().unwrap()
    );
}

#[test]
fn transitive_file_and_standard_imports_are_part_of_frozen_identity() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let outer = scratch.0.join("ops.wgt");
    let leaf = scratch.0.join("leaf.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    std::fs::write(&outer, "wgt 0.5\nimport leaf from \"./leaf.wgt\"\noperators Outer\nfn boost(x) {return leaf.apply(x);}\n").unwrap();
    std::fs::write(&leaf, "wgt 0.5\nimport activation from \"std/activation\"\noperators Leaf\nfn apply(x) {return activation.swish(x);}\n").unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    let original = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    let fingerprint = template.source_fingerprint().unwrap();
    std::fs::write(&leaf, "wgt 0.5\noperators Leaf\nfn apply(x) {return x;}\n").unwrap();
    let changed = ModelTemplate::from_file(&root).unwrap();
    changed
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_ne!(changed.source_fingerprint().unwrap(), fingerprint);
    std::fs::remove_file(&outer).unwrap();
    std::fs::remove_file(&leaf).unwrap();
    let frozen = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_eq!(frozen.source(), original.source());
    assert_eq!(template.source_fingerprint().unwrap(), fingerprint);

    let inline = "wgt 0.5\ndim M = 2\nimport activation from \"std/activation\"\nmodel Inline\nfn forward(x: f32[M]) -> f32[M] {return activation.swish(x);}\n";
    let template = ModelTemplate::parse(inline).unwrap();
    template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    let identity = template.source_fingerprint().unwrap();
    template
        .bind_input_shapes(&shapes(&[7]), &Dimensions::new())
        .unwrap();
    assert_eq!(template.source_fingerprint().unwrap(), identity);
}

#[test]
fn deleted_root_and_failed_import_resolution_do_not_prevent_later_capture() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    std::fs::remove_file(&root).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .is_err()
    );
    assert!(template.source_fingerprint().is_err());
    // Imports are captured at first success, unlike the root captured at creation.
    std::fs::write(&import, IMPORT_ORIGINAL.replace("11", "17")).unwrap();
    let bound = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert!(bound.source().contains("17"));
    let identity = template.source_fingerprint().unwrap();
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    assert_eq!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .unwrap()
            .source(),
        bound.source()
    );
    assert_eq!(template.source_fingerprint().unwrap(), identity);
}

#[cfg(unix)]
#[test]
fn frozen_overlays_do_not_bypass_import_symlink_boundary() {
    let scratch = IdentityScratch::new();
    let outside = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    let fingerprint = template.source_fingerprint().unwrap();
    let external = outside.0.join("external.wgt");
    std::fs::write(&external, IMPORT_ORIGINAL).unwrap();
    std::fs::remove_file(&import).unwrap();
    std::os::unix::fs::symlink(&external, &import).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .unwrap_err()
            .to_string()
            .contains("symlink escapes")
    );
    assert_eq!(template.source_fingerprint().unwrap(), fingerprint);
    std::fs::remove_file(&import).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .is_ok()
    );
}

#[test]
fn fingerprint_capture_supports_dimensions_without_defaults_and_std_imports() {
    let source = "wgt 0.5\ndim M\nimport activation from \"std/activation\"\nmodel Unbound\nfn forward(x: f32[M]) -> f32[M] {return activation.swish(x);}\n";
    let template = ModelTemplate::parse(source).unwrap();
    template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    let fingerprint = template.source_fingerprint().unwrap();
    template
        .bind_input_shapes(&shapes(&[7]), &Dimensions::new())
        .unwrap();
    assert_eq!(template.source_fingerprint().unwrap(), fingerprint);
    let independent = ModelTemplate::parse(source).unwrap();
    independent
        .bind_input_shapes(&shapes(&[7]), &Dimensions::new())
        .unwrap();
    assert_eq!(independent.source_fingerprint().unwrap(), fingerprint);

    let explicit = ModelTemplate::parse(&MATRIX.replace("dim N = 2", "dim N")).unwrap();
    explicit
        .bind_input_shapes(&shapes(&[3, 3]), &dims("N", 4))
        .unwrap();
    assert!(explicit.source_fingerprint().is_ok());
}

#[test]
fn import_identity_includes_source_text_even_when_lowered_computation_is_equal() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    let before = ModelTemplate::from_file(&root).unwrap();
    let original = before
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    std::fs::write(
        &import,
        format!("{IMPORT_ORIGINAL}# changed dependency identity\n"),
    )
    .unwrap();
    let after = ModelTemplate::from_file(&root).unwrap();
    let equivalent = after
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_eq!(original.source(), equivalent.source());
    assert_ne!(
        before.source_fingerprint().unwrap(),
        after.source_fingerprint().unwrap()
    );
}

#[test]
fn oversized_import_failure_is_retryable_and_frozen_imports_avoid_disk_reads() {
    let scratch = IdentityScratch::new();
    let root = scratch.0.join("model.wgt");
    let import = scratch.0.join("ops.wgt");
    std::fs::write(&root, IMPORT_TEMPLATE).unwrap();
    std::fs::write(&import, " ".repeat(1024 * 1024 + 1)).unwrap();
    let template = ModelTemplate::from_file(&root).unwrap();
    assert!(
        template
            .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
            .is_err()
    );
    assert!(template.source_fingerprint().is_err());
    std::fs::write(&import, IMPORT_ORIGINAL).unwrap();
    let original = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    let fingerprint = template.source_fingerprint().unwrap();
    std::fs::write(&import, " ".repeat(1024 * 1024 + 1)).unwrap();
    let frozen = template
        .bind_input_shapes(&shapes(&[3]), &Dimensions::new())
        .unwrap();
    assert_eq!(original.source(), frozen.source());
    assert_eq!(template.source_fingerprint().unwrap(), fingerprint);
}

#[test]
fn multiple_ports_infer_shared_dimensions_and_reject_cross_input_conflicts() {
    let source = "wgt 0.5\ndim B=2\nmodel Linked\nparameter gain:f32[1]=[0.5]\nfn forward(x:f32[B,1],hidden:f32[B,1])->(f32[B,1],f32[B,1]){let next=x+scale(hidden,gain);return(next,next);}\n";
    let template = ModelTemplate::parse(source).unwrap();
    let first = template
        .bind_input_shapes(
            &BTreeMap::from([("x".into(), vec![7, 1]), ("hidden".into(), vec![7, 1])]),
            &Dimensions::new(),
        )
        .unwrap();
    assert_eq!(first.forward_interface().unwrap().input, 14);
    assert_eq!(first.forward_interface().unwrap().output, 14);
    assert_eq!(first.parameters()[0].shape, [1]);
    assert!(
        template
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![7, 1]), ("hidden".into(), vec![2, 1])]),
                &Dimensions::new()
            )
            .is_err()
    );
    assert!(
        template
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![7, 1])]),
                &Dimensions::new()
            )
            .is_err()
    );
}

#[test]
fn multiple_matrix_inputs_without_tuple_opt_in_remain_rejected_on_binding() {
    let source = MATRIX.replace(
        "fn forward(x: f32[M,K])",
        "fn forward(x: f32[M,K], y: f32[M,K])",
    );
    let template = ModelTemplate::parse(&source).unwrap();
    assert!(
        template
            .bind_input_shapes(
                &BTreeMap::from([("x".into(), vec![2, 3]), ("y".into(), vec![2, 3])]),
                &Dimensions::new()
            )
            .is_err()
    );
}
