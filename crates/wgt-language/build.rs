fn main() {
    for path in [
        "../../language/prelude.json",
        "../../language/prelude_metadata.json",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("../../language/prelude.json"))
            .expect("valid source prelude catalog JSON");
    let catalog_entries = catalog["entries"]
        .as_array()
        .expect("source prelude catalog entries array");
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
        assert!(entry["contexts"].as_array().is_some(), "missing prelude entry contexts");
        assert!(entry["signature"].as_str().is_some(), "missing prelude entry signature");
    }

    let metadata: serde_json::Value =
        serde_json::from_str(include_str!("../../language/prelude_metadata.json"))
            .expect("valid source prelude metadata JSON");
    let entries: Vec<_> = catalog_entries
        .iter()
        .filter(|e| !matches!(e["kind"].as_str(), Some("function" | "intrinsic")))
        .collect();
    assert_eq!(
        serde_json::to_value(entries).unwrap(),
        metadata["entries"],
        "stale prelude metadata"
    );
}
