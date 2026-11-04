#[path = "src/script/prelude_declarations.rs"]
mod prelude_declarations;

fn main() {
    for path in [
        "../../std/prelude.wgt",
        "../../language/prelude.json",
        "../../language/prelude_metadata.json",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    prelude_declarations::validate(
        include_str!("../../std/prelude.wgt"),
        include_str!("../../language/prelude.json"),
    )
    .expect("native prelude declaration contract");
    let metadata: serde_json::Value =
        serde_json::from_str(include_str!("../../language/prelude_metadata.json")).unwrap();
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("../../language/prelude.json")).unwrap();
    let entries: Vec<_> = catalog["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| !matches!(e["kind"].as_str(), Some("function" | "intrinsic")))
        .collect();
    assert_eq!(
        serde_json::to_value(entries).unwrap(),
        metadata["entries"],
        "stale prelude metadata"
    );
}
