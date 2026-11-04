use std::path::Path;
use wgtime_core::script::run;

#[test]
fn native_assertion_contract() {
    for body in [
        "assert true, missing();",
        "assert(true);",
        "assert (1 == 1);",
        "assert (1) == 1;",
        "let assert=0; assert true;",
        "return assert(true,\"ok\");",
    ] {
        run(&format!("wgt 0.5\nfn main(){{{body}}}"), Path::new("."), 100).unwrap();
    }
    for (body, expected) in [
        ("assert 1;", "condition must be bool"),
        ("assert false;", "assertion failed"),
        ("assert false, \"failure\";", "failure"),
        ("assert true, missing(); assert false, 1;", "expected string"),
        ("assert(true,missing());", "unknown function"),
    ] {
        let error = format!("{:#}", run(&format!("wgt 0.5\nfn main(){{{body}}}"), Path::new("."), 100).unwrap_err());
        assert!(error.contains("line 2") && error.contains(expected), "{error}");
    }
}

#[test]
fn condition_is_evaluated_once_and_message_only_on_failure() {
    let source = "wgt 0.5\nfn main(){seed(42); assert random() >= 0, random(); return random();}";
    let expected = "wgt 0.5\nfn main(){seed(42); let first=random(); return random();}";
    assert_eq!(run(source, Path::new("."), 100).unwrap().result, run(expected, Path::new("."), 100).unwrap().result);
}
