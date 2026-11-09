use std::{fs, path::Path, process::Command};
use wgtime_core::{Model, Session, categorical::VectorType};

#[test]
fn typed_native_build_package_and_export_preserve_contract() {
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.cache/weightlang/package-tests");
    fs::create_dir_all(&root).unwrap(); let dir=tempfile::tempdir_in(root).unwrap();
    let source=dir.path().join("model.wgt");
    fs::write(&source,"wgt 0.5\nmodel Categories\ninput x: f32[3]\nlayer same: identity\noutput scores: logits[3]\ninit weights=xavier_uniform(7) bias=zeros()\nfn main(){return Categories(\"adam\");}\n").unwrap();
    let artifact=dir.path().join("artifact"); let packaged=dir.path().join("package"); let exported=dir.path().join("inference.json");
    let output=Command::new(env!("CARGO_BIN_EXE_weightc")).arg("legacy-build").arg(&source).arg("-o").arg(&artifact).output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    for args in [vec!["package","--entry",artifact.to_str().unwrap(),"-o",packaged.to_str().unwrap()], vec!["export",packaged.to_str().unwrap(),"-o",exported.to_str().unwrap()]] {
        let output=Command::new(env!("CARGO_BIN_EXE_wgt")).args(args).output().unwrap();
        assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    }
    for model_path in [artifact.join("model.json"),packaged.join("model.json"),exported] {
        let encoded:serde_json::Value=serde_json::from_slice(&fs::read(model_path).unwrap()).unwrap();
        assert_eq!(encoded["version"],2); assert_eq!(encoded["output_type"],"logits");
        let model:Model=serde_json::from_value(encoded).unwrap();
        let mut session=Session::prepare(model,Path::new("."),1_000_000).unwrap();
        assert_eq!(session.output_type(),VectorType::Logits);
        assert_eq!(session.run(&[1000.,999.,-1000.]).unwrap(),[1000.,999.,-1000.]);
    }
}
