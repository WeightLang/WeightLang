use wgt_language::backend::{Backend, preflight_source};

#[test]
fn shared_model_source_is_eligible_for_both_backends() {
    let source = "wgt 0.5\nmodel Identity\nfn forward(x: f32[2]) -> f32[2] { return x; }\n";
    for backend in [Backend::Rust, Backend::Torch] {
        preflight_source(source, backend).unwrap();
    }
}

#[test]
fn torch_compatibility_is_eligible_only_for_torch() {
    let source = "# compatibility dialect\n\nwgt 0.5\n  target torch  \nhost demo\n";
    preflight_source(source, Backend::Torch).unwrap();
    let message = preflight_source(source, Backend::Rust)
        .unwrap_err()
        .to_string();
    for expected in [
        "line 4:",
        "target torch",
        "Python/PyTorch",
        "Rust backend",
        "shared WGT",
    ] {
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn comments_and_string_mentions_do_not_select_torch() {
    let source = "wgt 0.5\r\n# target torch\r\nmodel Identity\r\n# migration from target torch\r\nfn forward(x) { return observe(\"target torch\", x); }\r\n";
    preflight_source(source, Backend::Rust).unwrap();
}

#[test]
fn crlf_diagnostic_preserves_original_line_numbers() {
    let message = preflight_source("wgt 0.5\r\n\r\n\ttarget torch\r\n", Backend::Rust)
        .unwrap_err()
        .to_string();
    assert!(message.starts_with("line 3:"), "{message}");
}

#[test]
fn preflight_does_not_claim_syntax_or_operation_validation() {
    // Neither backend gains execution support from passing the target boundary.
    for backend in [Backend::Rust, Backend::Torch] {
        preflight_source("not a model; unknown_operation()", backend).unwrap();
    }
}

#[test]
fn multiline_strings_do_not_select_a_backend() {
    for delimiter in ["\"", "'", "\"\"\"", "'''"] {
        let source = format!(
            "wgt 0.5\n# ignored quote: '\nlet text = {delimiter}hello\ntarget torch\nworld{delimiter}\n"
        );
        preflight_source(&source, Backend::Rust).unwrap();
        let with_declaration = format!("{source}target torch\n");
        let error = preflight_source(&with_declaration, Backend::Rust).unwrap_err();
        assert!(error.to_string().starts_with("line 6:"), "{error}");
    }
}

#[test]
fn escaped_quotes_and_hashes_inside_strings_preserve_scanner_state() {
    let source =
        "wgt 0.5\nlet text = \"escaped \\\" # still text\ntarget torch\nend\"\n# target torch\n";
    preflight_source(source, Backend::Rust).unwrap();
    let error = preflight_source(&format!("{source}target torch\n"), Backend::Rust).unwrap_err();
    assert!(error.to_string().starts_with("line 6:"), "{error}");
}

#[test]
fn native_definition_entrypoints_report_the_original_target_line() {
    use wgt_language::model::ModelDefinition;
    let source = "# compatibility model\nwgt 0.5\ntarget torch\nhost demo\n";
    for error in [
        ModelDefinition::parse(source).unwrap_err(),
        ModelDefinition::parse_with_dimensions(source, &Default::default()).unwrap_err(),
    ] {
        let message = error.to_string();
        for expected in ["line 3:", "Python/PyTorch", "shared WGT"] {
            assert!(message.contains(expected), "{message}");
        }
    }
}

#[test]
fn compatibility_detection_and_native_boundary_agree_on_quoted_declarations() {
    for source in [
        "wgt 0.5\n# target torch\nmodel Identity\n",
        "wgt 0.5\nfn f() {return \"first\ntarget torch\nlast\";}\n",
    ] {
        preflight_source(source, Backend::Rust).unwrap();
        assert!(!wgt_language::torch_model::is_torch_model(source));
    }
    let source = "wgt 0.5\ntarget torch\nhost demo\n";
    assert!(wgt_language::torch_model::is_torch_model(source));
    assert!(preflight_source(source, Backend::Rust).is_err());
}
