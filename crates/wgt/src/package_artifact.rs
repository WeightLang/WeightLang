//! Verified release payloads. Source language versions and artifact formats are independent.
use crate::{artifact::{hash_file, Artifact, FILES}, manifest};
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};

pub const NATIVE_FORMAT: &str = "weightlang.native-build.v1";
pub const NATIVE_FILES: [&str;4] = ["manifest.json","model.json","checkpoint.json","source.wgt"];
pub struct Info {
    pub name: String,
    pub source_hash: String,
    pub untrained: Vec<String>,
    pub native: bool,
}
impl Info {
    pub fn files(&self) -> &'static [&'static str] { if self.native { &NATIVE_FILES } else { &FILES } }
    pub fn read(path:&Path) -> Result<Self> {
        let m:serde_json::Value=serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
        if m["format"] != NATIVE_FORMAT {
            let artifact=Artifact::read(path)?;
            return Ok(Self{name:artifact.manifest["model"].as_str().context("model name")?.into(),source_hash:artifact.manifest["source"]["sha256"].as_str().context("source hash")?.into(),untrained:artifact.untrained(),native:false});
        }
        ensure!(m["edition"]=="0.5","native package requires edition 0.5");
        let name=m["name"].as_str().context("native model name")?.to_string();
        ensure!(manifest::identifier(&name),"invalid native model name");
        for (key,file) in [("model","model.json"),("source","source.wgt"),("checkpoint","checkpoint.json")] {
            ensure!(m[key]==file,"invalid native payload path");
            ensure!(hash_file(&path.join(file))?==m[format!("{key}_sha256")],"native {key} checksum mismatch");
        }
        let source=fs::read_to_string(path.join("source.wgt"))?;
        weightlang_runtime::script::parse(&source)?;
        let model=serde_json::from_slice(&fs::read(path.join("model.json"))?)?;
        let session=weightlang_runtime::Session::prepare(model,path,256*1024*1024)?;
        ensure!(m["interface"]["input"]["shape"]==serde_json::json!([session.report().input]) && m["interface"]["output"]["shape"]==serde_json::json!([session.report().output]),"native interface mismatch");
        let checkpoint:serde_json::Value=serde_json::from_slice(&fs::read(path.join("checkpoint.json"))?)?;
        let inference:serde_json::Value=serde_json::from_slice(&fs::read(path.join("model.json"))?)?;
        ensure!(checkpoint["model"]==inference,"checkpoint/model weights differ");
        let trainer=weightlang_runtime::training::Trainer::restore(&path.join("checkpoint.json"),256*1024*1024)?;
        let steps=m["training_steps"].as_u64().context("training_steps")?;
        ensure!(trainer.steps()==steps,"checkpoint step mismatch");
        Ok(Self{name,source_hash:m["source_sha256"].as_str().context("source hash")?.into(),untrained:if steps==0 {vec!["all parameters".into()]}else{Vec::new()},native:true})
    }
}
pub fn index_files(artifact:&serde_json::Value) -> Result<&'static [&'static str]> {
    match artifact["format"].as_str() {
        Some(NATIVE_FORMAT)=>Ok(&NATIVE_FILES),
        None | Some("weightlang.artifact")=>Ok(&FILES),
        _=>anyhow::bail!("unsupported package artifact format")
    }
}
