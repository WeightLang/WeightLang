use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use wgt_language::format::{format_source, format_with_context};

#[test]
fn expands_functions_without_changing_comments_literals_or_precedence() {
    let source = "# header\nwgt 0.5\nfn main(){let s=\"# } \\\" \\u0041\";# keep this\nif 1>0{return -2**2;}else{return 1e-3;}}\n";
    let expected = "# header\nwgt 0.5\nfn main() {\n    let s = \"# } \\\" \\u0041\"; # keep this\n    if 1 > 0 {\n        return -2 ** 2;\n    } else {\n        return 1e-3;\n    }\n}\n";
    let formatted = format_source(source).unwrap();
    assert_eq!(formatted, expected);
    assert_eq!(format_source(&formatted).unwrap(), formatted);
    let program = wgt_language::script::parse::parse_functions(&formatted).unwrap();
    let wgt_language::script::Stmt::If(_, yes, _) = &program["main"].body[1].kind else {
        panic!("if statement")
    };
    let wgt_language::script::Stmt::Return(wgt_language::script::Expr::Unary(_, value)) =
        &yes[0].kind
    else {
        panic!("unary outside exponent")
    };
    assert!(matches!(value.as_ref(), wgt_language::script::Expr::Binary(op, _, _) if op == "**"));
}

#[test]
fn preserves_line_bound_declarations_and_submodel_boundaries() {
    let source = "wgt 0.5\nmodel Cell {\nparameter w:f32[2]=[1e-3,-2.0]\nfn forward(x:f32[2])->f32[2]{return x+w;}\n}\nmodel Network\ncell:Cell;\nfn forward(x:f32[2])->f32[2]{return cell.forward(x);}\n";
    let output = format_source(source).unwrap();
    assert!(
        output.contains("    parameter w: f32[2] = [1e-3, -2.0]\n"),
        "{output}"
    );
    assert!(
        output.contains("\n}\nmodel Network\ncell: Cell;\n"),
        "{output}"
    );
    assert_eq!(format_source(&output).unwrap(), output);
    let before = wgt_language::model::ModelDefinition::parse(source).unwrap();
    let after = wgt_language::model::ModelDefinition::parse(&output).unwrap();
    assert_eq!(
        serde_json::to_value(before.parameters()).unwrap(),
        serde_json::to_value(after.parameters()).unwrap()
    );
}

#[test]
fn preserves_init_spelling_named_axes_ranges_and_multiline_arguments() {
    let graph = "wgt 0.5\nmodel G\ninput x:f32[2]\noutput y:f32[2]\ninit weights=xavier_uniform(7) bias=zeros()\nlayer out:linear(out=2,bias=1)\n";
    let formatted = format_source(graph).unwrap();
    assert!(formatted.contains("init weights=xavier_uniform(7) bias=zeros()\n"));
    let source = "wgt 0.5\nmodel M\ninput x:f32[4]\noutput y:f32[4]\nlearn depth:int in 0..=3=1;\nfn forward(x){let m:f32[rows=2,cols=2]=x as [2,2];return (m.T as [4])+[\n0,0,\n0,0\n];}\n";
    let formatted = format_source(source).unwrap();
    assert!(
        formatted.contains("learn depth: int in 0..=3 = 1;\n"),
        "{formatted}"
    );
    assert!(formatted.contains("f32[rows=2, cols=2]"), "{formatted}");
    assert!(
        formatted.contains("\n        0, 0,\n        0, 0\n    ];"),
        "{formatted}"
    );
    assert_eq!(format_source(&formatted).unwrap(), formatted);
}

#[test]
fn rejects_invalid_declarations_and_unfinished_functions() {
    for source in [
        "wgt 0.5\nfn main(){return 1}\n",
        "wgt 0.5\nmodel M\nparameter w:f32[2]=[1]\nfn forward(x:f32[2])->f32[2]{return x+w;}\n",
        "wgt 0.5\nfn main(){return \"open;}\n",
        "wgt 0.5\ngarbage\nfn main(){return 1;}\n",
    ] {
        assert!(format_source(source).is_err(), "{source}");
    }
}

#[test]
fn tuple_bindings_and_attached_comments_have_stable_spacing() {
    let source = "wgt 0.5\nfn main(){let (a,b)=(1,2);# tail\n# note\nreturn a+b;}\n";
    let expected =
        "wgt 0.5\nfn main() {\n    let (a, b) = (1, 2); # tail\n    # note\n    return a + b;\n}\n";
    assert_eq!(format_source(source).unwrap(), expected);
    assert_eq!(format_source(expected).unwrap(), expected);
}

#[test]
fn normalizes_line_endings_and_uses_unsaved_imports() {
    let source = "wgt 0.5\r\nfn main()\r\n{\r\nreturn 1;\r\n}\r\n";
    assert_eq!(
        format_source(source).unwrap(),
        "wgt 0.5\nfn main() {\n    return 1;\n}\n"
    );
    let directory = std::env::temp_dir().join(format!("wgt-format-import-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let path = directory.join("model.wgt");
    let imported = directory.join("ops.wgt");
    let source = "wgt 0.5\nimport ops from \"./ops.wgt\"\nmodel M\nfn forward(x:f32[1])->f32[1]{return ops.apply(x);}\n";
    let overlays = BTreeMap::from([(
        imported,
        "wgt 0.5\noperators Ops\nfn apply(x){return x;}\n".into(),
    )]);
    assert!(format_with_context(source, Some(&path), &overlays).is_ok());
    assert!(format_with_context(source, Some(&path), &BTreeMap::new()).is_err());
    fs::remove_dir(directory).unwrap();
}

fn visit(path: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            visit(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "wgt") {
            files.push(path);
        }
    }
}

#[test]
fn current_models_and_standard_modules_remain_valid_and_idempotent() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    for directory in [
        "examples",
        "packages",
        "python/examples",
        "std",
        "language/syntax-tour",
        "crates/wgt-model/tests/fixtures",
        "crates/wgt-model-macros/tests/fixtures",
        "crates/wgtime-core/tests/fixtures",
    ] {
        visit(&root.join(directory), &mut files);
    }
    let mut failures = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path).unwrap();
        let output = match format_with_context(&source, Some(&path), &BTreeMap::new()) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!("{}: {error:#}", path.display()));
                continue;
            }
        };
        assert_eq!(
            format_with_context(&output, Some(&path), &BTreeMap::new()).unwrap(),
            output,
            "{}",
            path.display()
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn existing_legacy_editor_sources_format_without_executing_or_migrating() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut failures = Vec::new();
    for relative in [
        "editors/vscode/media/example.wgt",
        "editors/tree-sitter-wgt/test/fixtures/forward.wgt",
        "editors/tree-sitter-wgt/test/fixtures/model_operations.wgt",
        "editors/tree-sitter-wgt/test/fixtures/model_control.wgt",
        "editors/tree-sitter-wgt/test/fixtures/tensor_learning/model.wgt",
        "crates/wgt/templates/native.wgt",
    ] {
        let path = root.join(relative);
        let source = fs::read_to_string(&path).unwrap();
        let output = match format_with_context(&source, Some(&path), &BTreeMap::new()) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!("{relative}: {error:#}"));
                continue;
            }
        };
        assert_eq!(
            format_with_context(&output, Some(&path), &BTreeMap::new()).unwrap(),
            output,
            "{relative}"
        );
        assert_eq!(source.contains("fn main"), output.contains("fn main"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // This mixed native-contract/learn snippet is deliberately an editor grammar
    // fixture, not a compilation unit accepted by the current Rust frontend.
    let path = root.join("editors/tree-sitter-wgt/test/fixtures/editor_current.wgt");
    let source = fs::read_to_string(&path).unwrap();
    assert!(format_with_context(&source, Some(&path), &BTreeMap::new()).is_err());
}

#[test]
fn legacy_models_use_static_validation_without_running_main_or_initializers() {
    // Literal spelling and legacy seed policy are preserved, not upgraded.
    let source = "wgt 0.5\nmodel M\ninput x:f32[1]\noutput y:f32[1]\ninit seed=7\nlayer y:linear(out=1)\nfn main(){save text \"never\" to \"must-not-be-written\";return 1;}\n";
    let output = format_source(source).unwrap();
    assert!(output.contains("init seed=7\n"));
    assert!(output.contains("save text \"never\" to \"must-not-be-written\";"));
    assert!(format_source(&source.replace("main()", "main(x)")).is_err());
    assert!(format_source(&source.replace("model M", "model main")).is_err());
}
