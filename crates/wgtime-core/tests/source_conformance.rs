use serde_json::Value;
use wgtime_core::script::parse;

#[test]
fn source_context_boundaries_match_shared_corpus() {
    let spec: Value =
        serde_json::from_str(include_str!("../../../language/source_capabilities.json")).unwrap();
    let corpus: Value =
        serde_json::from_str(include_str!("../../../language/conformance.json")).unwrap();
    assert_eq!(spec["source_edition"], "0.5");
    assert_eq!(
        spec["domains"]["artifact_graph"]["source_builtin_catalog"],
        false
    );
    let mut checked = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let context = case["context"].as_str().unwrap();
        assert!(
            spec["contexts"].get(context).is_some(),
            "unknown context: {context}"
        );
        let Some(expected) = case["rust_accepts"].as_bool() else {
            assert_eq!(context, "editor");
            continue;
        };
        let result = parse(case["source"].as_str().unwrap());
        assert_eq!(result.is_ok(), expected, "{}: {result:?}", case["id"]);
        checked += 1;
    }
    assert!(checked >= 12, "shared parser corpus unexpectedly shrank");
}
