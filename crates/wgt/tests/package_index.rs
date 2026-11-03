use serde_json::json;
use weightlang_wgt::{
    manifest::{Lock, Manifest},
    package_index::{self, Record, Selection},
    registry,
};

fn record(v: &str) -> Record {
    Record {
        format_version: 1,
        name: "example/gates".into(),
        module: "github.com/Example/models/packages/gates".into(),
        version: v.into(),
        source_commit: "a".repeat(40),
        index_sha256: "b".repeat(64),
        license: "MIT".into(),
        edition: "0.5".into(),
        yanked: false,
    }
}
#[test]
fn names_ownership_versions_and_yanks() {
    for name in [
        "../evil",
        "a/../../x",
        "Example/pkg",
        "x/a.b",
        "one",
        "a/b/c",
        "a/-b",
    ] {
        assert!(package_index::package_name(name).is_err());
    }
    for repo in ["https://github.com/a/b", "a/b/c", "../b"] {
        assert!(package_index::repository(repo).is_err());
    }
    let mut r = record("v1.0.0");
    r.name = "other/gates".into();
    assert!(r.validate().is_err());
    let mut yanked = record("v2.0.0");
    yanked.yanked = true;
    let records = vec![
        (record("v0.9.0"), "a".into()),
        (record("v0.10.0"), "b".into()),
        (yanked, "c".into()),
    ];
    assert_eq!(
        package_index::select("example/gates", None, records.clone())
            .unwrap()
            .0
            .version,
        "v0.10.0"
    );
    assert_eq!(
        package_index::select("example/gates", Some("v0.9.0"), records.clone())
            .unwrap()
            .0
            .version,
        "v0.9.0"
    );
    assert!(package_index::select("example/gates", Some("v2.0.0"), records).is_err());
}
#[test]
fn pins_reject_replacement_and_preserve_initial_provenance() {
    let mut lock = Lock::new();
    let mut selected = Selection {
        record: record("v1.0.0"),
        registry: "WeightLang/registry".into(),
        commit: "c".repeat(40),
        record_sha256: "d".repeat(64),
    };
    selected.lock(&mut lock).unwrap();
    let original = lock.clone();
    selected.commit = "e".repeat(40);
    selected.lock(&mut lock).unwrap();
    assert_eq!(original, lock);
    selected.record.index_sha256 = "f".repeat(64);
    assert!(selected.lock(&mut lock).is_err());
    selected.record = record("v1.0.0");
    selected.registry = "attacker/index".into();
    assert!(selected.lock(&mut lock).is_err());
    let manifest = Manifest::parse(
        "module local/test\nwgt 0.5\nrequire github.com/Example/models/packages/gates v1.0.0\n",
    )
    .unwrap();
    assert_eq!(
        package_index::alias("example/gates", &manifest, &original).unwrap(),
        selected.record.module
    );
    assert!(package_index::alias("example/missing", &manifest, &original).is_err());
}
#[test]
fn registry_lock_survives_exact_refetch() {
    // Existing artifact fetch tests cover byte verification. An empty dependency set
    // must remove stale registry metadata just as it removes stale artifact entries.
    struct NoNetwork;
    impl registry::Remote for NoNetwork {
        fn resolve(&self, _: &str, _: &str) -> anyhow::Result<String> {
            panic!("network")
        }
        fn download(&self, _: &str, _: u64) -> anyhow::Result<Vec<u8>> {
            panic!("network")
        }
    }
    let mut lock = Lock::new();
    Selection {
        record: record("v1.0.0"),
        registry: "WeightLang/registry".into(),
        commit: "c".repeat(40),
        record_sha256: "d".repeat(64),
    }
    .lock(&mut lock)
    .unwrap();
    let manifest = Manifest::parse("module local/test\nwgt 0.5\n").unwrap();
    let cache = tempfile::tempdir().unwrap();
    assert!(
        registry::fetch(&manifest, &lock, cache.path(), false, &NoNetwork)
            .unwrap()
            .is_empty()
    );
    assert!(registry::fetch(&manifest, &lock, cache.path(), true, &NoNetwork).is_err());
    let mut value = serde_json::to_value(record("v1.0.0")).unwrap();
    value["untrusted_field"] = json!(true);
    assert!(serde_json::from_value::<Record>(value).is_err());
}

#[test]
fn old_registry_records_are_retained_but_not_selected_for_05() {
    let mut old = record("v9.0.0");
    old.edition = "0.3".into();
    let records = vec![(old, "old".into()), (record("v0.5.0"), "new".into())];
    assert_eq!(package_index::select("example/gates", None, records.clone()).unwrap().0.version, "v0.5.0");
    assert!(package_index::select("example/gates", Some("v9.0.0"), records).is_err());
}
