//! Verified release payloads. Source language versions and artifact formats are independent.
use crate::{
    artifact::{hash_file, Artifact, FILES},
    manifest,
};
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};

pub const NATIVE_FORMAT: &str = "weightlang.native-build.v1";
pub const OPERATOR_FORMAT: &str = "weightlang.operators.v1";
pub const OPERATOR_FILES: [&str; 2] = ["manifest.json", "source.wgt"];
pub const NATIVE_FILES: [&str; 4] = [
    "manifest.json",
    "model.json",
    "checkpoint.json",
    "source.wgt",
];
pub struct Info {
    pub name: String,
    pub source_hash: String,
    pub untrained: Vec<String>,
    pub native: bool,
    pub operator: bool,
}
impl Info {
    pub fn files(&self) -> &'static [&'static str] {
        if self.operator {
            &OPERATOR_FILES
        } else if self.native {
            &NATIVE_FILES
        } else {
            &FILES
        }
    }
    pub fn read(path: &Path) -> Result<Self> {
        let m: serde_json::Value = serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
        if m["format"] == OPERATOR_FORMAT {
            ensure!(
                m["edition"] == "0.5" && m["source"] == "source.wgt",
                "invalid operator manifest"
            );
            ensure!(
                fs::metadata(path.join("source.wgt"))?.len() <= 1024 * 1024,
                "operator library exceeds 1 MiB"
            );
            let source_hash = hash_file(&path.join("source.wgt"))?;
            ensure!(
                m["source_sha256"] == source_hash,
                "operator source checksum mismatch"
            );
            let name = wgt_language::script::operators::library(&fs::read_to_string(
                path.join("source.wgt"),
            )?)?;
            ensure!(m["name"] == name, "operator library identity mismatch");
            return Ok(Self {
                name,
                source_hash,
                untrained: Vec::new(),
                native: false,
                operator: true,
            });
        }
        if m["format"] != NATIVE_FORMAT {
            let artifact = Artifact::read(path)?;
            return Ok(Self {
                name: artifact.manifest["model"]
                    .as_str()
                    .context("model name")?
                    .into(),
                source_hash: artifact.manifest["source"]["sha256"]
                    .as_str()
                    .context("source hash")?
                    .into(),
                untrained: artifact.untrained(),
                native: false,
                operator: false,
            });
        }
        ensure!(m["edition"] == "0.5", "native package requires edition 0.5");
        let name = m["name"].as_str().context("native model name")?.to_string();
        ensure!(manifest::identifier(&name), "invalid native model name");
        for (key, file) in [
            ("model", "model.json"),
            ("source", "source.wgt"),
            ("checkpoint", "checkpoint.json"),
        ] {
            ensure!(m[key] == file, "invalid native payload path");
            ensure!(
                hash_file(&path.join(file))? == m[format!("{key}_sha256")],
                "native {key} checksum mismatch"
            );
        }
        let source = fs::read_to_string(path.join("source.wgt"))?;
        wgt_language::script::parse(&source)?;
        let model_report = wgtime_control::persistence::inspect_artifact(&path.join("model.json"), false)?;
        ensure!(
            m["interface"]["input"]["shape"] == serde_json::json!([model_report["input"].as_u64().context("missing input")?])
                && m["interface"]["output"]["shape"]
                    == serde_json::json!([model_report["output"].as_u64().context("missing output")?]),
            "native interface mismatch"
        );
        let checkpoint: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("checkpoint.json"))?)?;
        let inference: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("model.json"))?)?;
        ensure!(
            checkpoint["model"] == inference,
            "checkpoint/model weights differ"
        );
        let checkpoint_report = wgtime_control::persistence::inspect_artifact(&path.join("checkpoint.json"), true)?;
        let steps = m["training_steps"].as_u64().context("training_steps")?;
        ensure!(checkpoint_report["steps"].as_u64().context("missing checkpoint steps")? == steps, "checkpoint step mismatch");
        Ok(Self {
            name,
            source_hash: m["source_sha256"].as_str().context("source hash")?.into(),
            untrained: if steps == 0 {
                vec!["all parameters".into()]
            } else {
                Vec::new()
            },
            native: true,
            operator: false,
        })
    }
}
pub fn index_files(artifact: &serde_json::Value) -> Result<&'static [&'static str]> {
    match artifact["format"].as_str() {
        Some(NATIVE_FORMAT) => Ok(&NATIVE_FILES),
        Some(OPERATOR_FORMAT) => Ok(&OPERATOR_FILES),
        None | Some("weightlang.artifact") => Ok(&FILES),
        _ => anyhow::bail!("unsupported package artifact format"),
    }
}
