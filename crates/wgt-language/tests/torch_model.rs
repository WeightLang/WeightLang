use wgt_language::{model::ModelDefinition, torch_model};

const SOURCE: &str = "wgt 0.5\ntarget torch\nhost example.host\nextern nn, torch\nmodel Recurrent extends Module {\ninit(self, width=4) {\nsuper().__init__();\nself.cell = nn.GRUCell(width, width);\n}\nfn forward(self, x, hidden) {\nif x.shape[1] != hidden.shape[1] {\nraise ValueError('width mismatch');\n}\nreturn self.cell(x, hidden);\n}\n}\nfn twice(x) {\nwhile False {\nx = x + x;\n}\nreturn x + x;\n}\n";

#[test]
fn dynamic_models_lower_without_constructing_parameters_or_executing_code() {
    let model = torch_model::parse(SOURCE).unwrap();
    assert_eq!(model.models, ["Recurrent"]);
    assert_eq!(model.functions, ["twice"]);
    assert!(model.code.contains("class Recurrent(Module):"));
    assert!(model.code.contains("def __init__(self, width=4):"));
    assert!(model.code.contains("return self.cell(x, hidden)"));
    assert!(
        ModelDefinition::parse(SOURCE).is_err(),
        "torch target must not execute as a native model"
    );
    let with_failure = SOURCE.replace("super().__init__();", "raise RuntimeError('not executed');");
    assert!(torch_model::parse(&with_failure).is_ok());
}

#[test]
fn invalid_declarations_signatures_and_historical_versions_are_rejected() {
    for source in [
        SOURCE.replace("wgt 0.5", "wgt 0.3"),
        SOURCE.replace("extern nn, torch", "extern nn, nn"),
        SOURCE.replace("self.cell(x, hidden)", "self.cell(x,, hidden)"),
        SOURCE.replace("fn twice(x)", "fn main(x)"),
        SOURCE.replace("host example.host", "host invalid/path"),
        SOURCE.trim_end_matches("}\n").to_owned(),
    ] {
        assert!(torch_model::parse(&source).is_err(), "{source}");
    }
}

#[test]
fn formatting_and_editor_validation_use_the_same_frontend() {
    let formatted = wgt_language::format::format_source(SOURCE).unwrap();
    assert_eq!(
        formatted,
        wgt_language::format::format_source(&formatted).unwrap()
    );
    assert!(
        wgt_language::ide::analyze(SOURCE, None)
            .diagnostics
            .is_empty()
    );
    assert!(wgt_language::ide::analyze(SOURCE, None).model.is_none());
}

#[test]
fn inline_comments_and_dictionary_continuations_preserve_expression_text() {
    let source = SOURCE.replace("return self.cell(x, hidden);", "result = {\n'output': self.cell(x, hidden),\n'label': '#literal',\n}; # comment\nreturn result['output'];");
    let parsed = torch_model::parse(&source).unwrap();
    assert!(parsed.code.contains("'label': '#literal'"));
    assert!(!parsed.code.contains("# comment"));
    let formatted = torch_model::format(&source).unwrap();
    assert_eq!(formatted, torch_model::format(&formatted).unwrap());
    assert!(
        torch_model::parse(&SOURCE.replace(
            "fn forward(self, x, hidden)",
            "fn __init__(self, x, hidden)"
        ))
        .is_err()
    );
}
