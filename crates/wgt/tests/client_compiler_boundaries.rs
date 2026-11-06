use weightlang_wgt::{client_compiler, script_compiler};
use wgtime_core::script;

#[test]
fn accepted_core_syntax_reports_unimplemented_client_codegen_without_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/compiler-boundaries");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let model =
        "wgt 0.5\nmodel Policy\ninput x:f32[2]\noutput y:f32[2]\nfn forward(x){return x;}\n";
    for (name, source) in [
        (
            "sample",
            format!("{model}fn main(){{let net=Policy(\"none\");return sample from net([0,0]);}}"),
        ),
        (
            "sample_record",
            format!(
                "{model}fn main(){{let net=Policy(\"none\");let {{action,log_prob}}=sample from net([0,0]);return [action,log_prob];}}"
            ),
        ),
        (
            "slice",
            "wgt 0.5\nfn main(){return [1,2,3][1:];}".to_owned(),
        ),
    ] {
        script::parse(&source).unwrap();
        let output = directory.path().join(name);
        let error = script_compiler::compile_text(&source, directory.path(), &output).unwrap_err();
        assert!(
            format!("{error:#}").contains("not yet migrated to client-only compiler"),
            "{name}: {error:#}"
        );
        assert!(
            !output.exists(),
            "unsupported {name} published an executable"
        );
    }
}

#[test]
fn client_lowering_follows_the_selected_entry_and_recursive_helpers() {
    let source = r#"wgt 0.5
fn main(){return [1,2,3][1:];}
fn condition(){return true;}
fn count(n){if n==0{return 0;}return 1+count(n-1);}
fn chosen(){if condition(){return count(4);}return 0;}
"#;
    let root = std::path::Path::new(".");
    let error = client_compiler::source(source, root).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not yet migrated to client-only compiler")
    );
    client_compiler::source_entry(source, root, "chosen").unwrap();
    let error = client_compiler::source_entry(source, root, "missing").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("client compilation requires entry missing")
    );
}

#[test]
fn imported_sources_reject_only_reachable_resident_strategies_before_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.cache/weightlang/compiler-boundaries");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    std::fs::write(
        directory.path().join("ops.wgt"),
        "wgt 0.5\noperators Shift\nfn shifted(x){return x+[1];}\n",
    )
    .unwrap();
    let definitions = r#"wgt 0.5
import ops from "./ops.wgt"
model Identity
input x:f32[1]
output y:f32[1]
fn forward(x){return ops.shifted(x);}
fn strategy(net,inputs,targets){return inputs[0][0];}
fn using_strategy(){
    let net=Identity("none");
    return train net on [[2]] against [[0]] using strategy;
}
"#;
    let ordinary = format!("{definitions}fn main(){{let net=Identity(\"none\");return net([2]);}}");
    client_compiler::source(&ordinary, directory.path()).unwrap();

    let reachable = format!("{definitions}fn main(){{return using_strategy();}}");
    for (entry, source) in [("using_strategy", &ordinary), ("main", &reachable)] {
        let output = directory.path().join(entry);
        let error = script_compiler::compile_text_entry(source, directory.path(), &output, entry)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("resident strategies do not support imports"),
            "{entry}: {error:#}"
        );
        assert!(
            !output.exists(),
            "unsupported strategy published an executable"
        );
    }
}

#[test]
fn strategy_import_admission_uses_declarations_not_comments_or_strings() {
    let source = r#"wgt 0.5
# import ghost from "./missing.wgt"
model Identity
input x:f32[1]
output y:f32[1]
fn forward(x){return x;}
fn strategy(net,inputs,targets){
    let note="import ghost from \"./missing.wgt\"";
    return inputs[0][0];
}
fn main(){
    let net=Identity("none");
    return train net on [[2]] against [[0]] using strategy;
}
"#;
    client_compiler::source(source, std::path::Path::new(".")).unwrap();
}
