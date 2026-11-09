fn main() {
    println!("cargo:rerun-if-changed=assets/prelude.json");
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("assets/prelude.json"))
            .expect("valid source prelude catalog JSON");
    let catalog_entries = catalog["entries"]
        .as_array()
        .expect("source prelude catalog entries array");
    assert_eq!(catalog["schema_version"], 1, "source prelude schema version");
    assert_eq!(catalog["source_edition"], "0.5", "source prelude edition");
    let mut declarations = std::collections::BTreeSet::new();
    for entry in catalog_entries {
        let name = entry["name"]
            .as_str()
            .expect("source prelude entry name string");
        let kind = entry["kind"]
            .as_str()
            .expect("source prelude entry kind string");
        assert!(
            declarations.insert((name, kind)),
            "duplicate source prelude declaration: {name} ({kind})"
        );
        let contexts = entry["contexts"].as_array().expect("prelude contexts array");
        assert!(!contexts.is_empty() && contexts.iter().all(|c| c.as_str().is_some()), "invalid prelude contexts: {name}");
        for field in ["signature", "purpose", "implementation"] {
            assert!(entry[field].as_str().is_some_and(|s| !s.is_empty()), "missing prelude {field}: {name}");
        }
        assert!(entry["defaults"].is_object(), "prelude defaults object: {name}");
        if matches!(kind, "function" | "internal") {
            assert!(contexts.iter().any(|c| c == "ordinary_function"), "ordinary prelude context: {name}");
            let min = entry["arity"]["min"].as_u64().expect("prelude minimum arity");
            let max = entry["arity"]["max"].as_u64().expect("prelude maximum arity");
            assert!(min <= max, "invalid prelude arity: {name}");
        }
        if kind == "intrinsic" {
            assert!(contexts.iter().all(|c| matches!(c.as_str(), Some("tensor_forward" | "tensor_loss"))), "tensor prelude context: {name}");
        }
        if let Some(arity) = entry.get("operator_arity") {
            let min = arity["min"].as_u64().expect("tensor operator minimum arity");
            let max = arity["max"].as_u64().expect("tensor operator maximum arity");
            assert!(kind == "intrinsic" && min > 0 && min <= max, "invalid tensor operator arity: {name}");
        }
        if let Some(reserved) = entry.get("reserved") {
            assert!(kind == "intrinsic" && reserved.is_boolean(), "invalid tensor name reservation: {name}");
        }
    }
}
