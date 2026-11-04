use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use weightlang_runtime::script::run;
static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let root = PathBuf::from("/tmp/weightlang").join(format!(
        "file-syntax-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    root
}
#[test]
fn text_json_and_model_round_trip() {
    let root = root();
    let report = run(
        r#"wgt 0.5
fn main() {
 save text "你好\n世界" to "文本.txt";
 let text = load text from "文本.txt";
 assert text == "你好\n世界", "unicode";
 save json [true,null,[1,2],text] to "data.json";
 let data = load json from "data.json";
 assert data == [true,null,[1,2],text];
 save text "new" to "文本.txt" overwrite;
 assert (load text from "文本.txt") == "new";
 let net = mlp([1,1],"identity",4,"adam");
 save model net to "model.json";
 save checkpoint net to "checkpoint.json";
 let restored = load model from "model.json" using "adam";
 let resumed = load checkpoint from "checkpoint.json";
 assert net([1]) == restored([1]);
 assert net([1]) == resumed([1]);
 return data;
}"#,
        &root,
        1_000_000,
    )
    .unwrap();
    assert_eq!(report.result[0], true);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn assertion_laziness_strict_bool_and_legacy_eagerness() {
    let root = root();
    run(
        "wgt 0.5\nfn main(){assert true, missing(); assert(true); return 1;}",
        &root,
        100,
    )
    .unwrap();
    for source in [
        "wgt 0.5\nfn main(){assert 1;}",
        "wgt 0.5\nfn main(){assert(true,missing());}",
        "wgt 0.5\nfn main(){assert false, \"diagnostic\";}",
    ] {
        let error = format!("{:#}", run(source, &root, 100).unwrap_err());
        assert!(error.contains("line 2"), "{error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn host_policy_blocks_native_file_syntax() {
    let root = root();
    let policy = root.join("policy.json");
    let script = root.join("model.wgt");
    std::fs::write(
        &policy,
        r#"{"work":100,"workspace_bytes":4096,"calls":2,"deadline_ms":10000}"#,
    )
    .unwrap();
    for body in [
        "save text \"secret\" to \"out.txt\";",
        "let x = load text from \"policy.json\";",
        "let x = load model from \"policy.json\" using \"adam\";",
    ] {
        std::fs::write(&script, format!("wgt 0.5\nfn main(){{{body}}}")).unwrap();
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_wgt-native"))
            .args(["script", script.to_str().unwrap()])
            .env("WGT_HOST_POLICY", &policy)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains("host policy requires capability-based I/O"));
        assert!(!root.join("out.txt").exists());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn encoded_json_budget_preserves_existing_content_and_utf8_cost() {
    let root = root();
    std::fs::write(root.join("old.json"), b"old").unwrap();
    let payload = "界".repeat(100);
    let source = format!("wgt 0.5\nfn main(){{save json \"{payload}\" to \"old.json\" overwrite;}}");
    assert!(run(&source, &root, 64).is_err());
    assert_eq!(std::fs::read(root.join("old.json")).unwrap(), b"old");
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    let ascii=run("wgt 0.5\nfn main(){save text \"a\" to \"a.txt\";}", &root, 100).unwrap();
    let unicode=run("wgt 0.5\nfn main(){save text \"界\" to \"b.txt\";}", &root, 100).unwrap();
    assert_eq!(unicode.operations, ascii.operations+2);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn record_batches_are_not_implicitly_json_serialized() {
    let root = root();
    let error=run("wgt 0.5\nfn main(){let rows=collect {yield {x:1};};save json rows to \"rows.json\";}", &root, 100_000).unwrap_err();
    assert!(format!("{error:#}").contains("resource handles cannot be saved as JSON"));
    assert!(!root.join("rows.json").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_reads_and_writes_preserve_outputs() {
    let root = root();
    std::fs::write(root.join("old.txt"), b"original").unwrap();
    std::fs::write(root.join("broken.json"), b"{bad}").unwrap();
    std::fs::write(root.join("invalid.txt"), [255]).unwrap();
    for body in [
        "save text \"bad\" to \"old.txt\";",
        "save json mlp([1,1],\"identity\",4,\"adam\") to \"old.txt\" overwrite;",
        "let x = load json from \"broken.json\";",
        "let x = load text from \"invalid.txt\";",
        "let x = load checkpoint from \"broken.json\";",
        "save text \"bad\" to \"absent/file\";",
    ] {
        assert!(
            run(&format!("wgt 0.5\nfn main(){{{body}}}"), &root, 100_000).is_err(),
            "{body}"
        );
        assert_eq!(std::fs::read(root.join("old.txt")).unwrap(), b"original");
    }
    assert!(run(
        "wgt 0.5\nfn main(){let x = load text from \"old.txt\";}",
        &root,
        4
    )
    .is_err());
    std::fs::remove_dir_all(root).unwrap();
}
