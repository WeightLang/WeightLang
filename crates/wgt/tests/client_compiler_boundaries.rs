use weightlang_wgt::script_compiler;
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
