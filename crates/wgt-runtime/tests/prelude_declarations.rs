use weightlang_runtime::script::{parse, prelude::declarations};
const SOURCE: &str = include_str!("../../../std/prelude.wgt");
const CATALOG: &str = include_str!("../../../language/prelude.json");

#[test]
fn bundled_contract_is_complete_and_inert() {
    declarations::validate(SOURCE, CATALOG).unwrap();
    let entries = declarations::parse(SOURCE).unwrap();
    assert!(entries
        .iter()
        .all(|e| !e["name"].as_str().unwrap().starts_with("__")));
    assert!(entries.iter().all(|e| e["name"] != "fit"));
    let benchmark = entries.iter().find(|e| e["name"] == "benchmark").unwrap();
    assert_eq!(benchmark["arity"], serde_json::json!({"min":5,"max":6}));
    assert_eq!(benchmark["defaults"]["mode"], "cpu");
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
        format!("{SOURCE}\n# @purpose \"dangling\"\n"),
    ] {
        assert!(declarations::parse(&changed).is_err());
    }
    for changed in [
        SOURCE.replacen("profiles, tools);", "profiles);", 1),
        SOURCE.replacen("mode = \"cpu\"", "mode = \"gpu\"", 1),
        SOURCE.replacen("Configure one execution-only", "Changed execution-only", 1),
    ] {
        assert!(declarations::validate(&changed, CATALOG).is_err());
    }
}
