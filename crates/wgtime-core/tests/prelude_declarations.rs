use wgtime_core::script::{parse, prelude::declarations};

const SOURCE: &str = r#"prelude native 0.5;
# @contexts ["ordinary_function"]
# @purpose "Test declaration metadata consistency."
# @implementation "No runtime body is provided."
native fn responses_scope(limits, profiles, tools);
"#;
const CATALOG: &str = r#"{"entries":[{"contexts":["ordinary_function"],"purpose":"Test declaration metadata consistency.","implementation":"No runtime body is provided.","name":"responses_scope","kind":"function","signature":"responses_scope(limits,profiles,tools)","defaults":{},"arity":{"min":3,"max":3}}]}"#;

#[test]
fn bundled_contract_is_complete_and_inert() {
    declarations::validate(SOURCE, CATALOG).unwrap();
    let entries = declarations::parse(SOURCE).unwrap();
    assert!(entries
        .iter()
        .all(|e| !e["name"].as_str().unwrap().starts_with("__")));
    assert_eq!(entries.len(), 1);
    let declaration = &entries[0];
    assert_eq!(declaration["name"], "responses_scope");
    assert_eq!(declaration["arity"], serde_json::json!({"min":3,"max":3}));
    assert!(parse(SOURCE).is_err());
    assert!(parse("wgt 0.5\nnative fn hidden();\nfn main() {}\n").is_err());
    assert!(parse("wgt 0.5\nfn main() { return mean([1,2,3]); }\n").is_ok());
}

#[test]
fn declarations_reject_bodies_and_contract_drift() {
    for changed in [
        SOURCE.replacen(
            "native fn responses_scope(limits, profiles, tools);",
            "native fn responses_scope(limits, profiles, tools) {}",
            1,
        ),
        SOURCE.replacen(
            "native fn responses_scope",
            "native fn __responses_scope",
            1,
        ),
        SOURCE.replacen("# @contexts [\"ordinary_function\"]", "# @contexts []", 1),
        SOURCE.replacen("native fn responses_scope", "fn responses_scope", 1),
        format!("{SOURCE}\n# @purpose \"dangling\"\n"),
    ] {
        assert!(declarations::parse(&changed).is_err());
    }
    for changed in [
        SOURCE.replacen("limits, profiles, tools", "limits, profiles", 1),
        SOURCE.replacen(
            "Test declaration metadata consistency.",
            "Changed declaration metadata consistency.",
            1,
        ),
    ] {
        assert!(declarations::validate(&changed, CATALOG).is_err());
    }
}
