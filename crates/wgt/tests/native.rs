use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command};
use weightlang_wgt::{
    artifact::{hash, Artifact, FILES},
    graph::Graph,
    manifest::{self, Lock, Manifest},
    registry::{self, Remote},
};
const MODULE: &str = "github.com/example/models/gates";
const VERSION: &str = "v0.2.0";
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn graph_discovers_model_css_and_explicit_style_wins() {
    let t = tempfile::tempdir().unwrap();
    artifact(&t.path().join("artifact"), false);
    fs::write(t.path().join("model.wgt"), "model source").unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_wgt"))
            .current_dir(t.path())
            .args(args)
            .output()
            .unwrap()
    };
    let html = |args: &[&str]| {
        let result = run(args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    };
    let args = ["graph", "artifact", "--stdout"];
    assert!(html(&args).contains("<weightlang-viewer>"));
    fs::write(t.path().join("model.css"), "/* project-companion */").unwrap();
    assert!(html(&args).contains("/* project-companion */"));
    fs::write(
        t.path().join("artifact/model.css"),
        "/* bundled-companion */",
    )
    .unwrap();
    assert!(html(&args).contains("/* bundled-companion */"));
    assert!(!html(&args).contains("/* project-companion */"));
    assert!(
        html(&["graph", "artifact/manifest.json", "--stdout"]).contains("/* bundled-companion */")
    );
    fs::write(t.path().join("custom.css"), "/* explicit-style */").unwrap();
    let explicit = html(&["graph", "artifact", "--style", "custom.css", "--stdout"]);
    assert!(explicit.contains("/* explicit-style */"));
    assert!(!explicit.contains("/* bundled-companion */"));
    assert!(
        !run(&["graph", "artifact", "--style", "missing.css", "--stdout"])
            .status
            .success()
    );
    fs::write(t.path().join("artifact/model.css"), [0xff]).unwrap();
    assert!(!run(&args).status.success());
    assert!(run(&["graph", "artifact", "--json"]).status.success());
    fs::remove_file(t.path().join("artifact/model.css")).unwrap();
    fs::create_dir(t.path().join("models")).unwrap();
    fs::write(t.path().join("models/model.wgt"), "source").unwrap();
    fs::write(t.path().join("models/model.css"), "/* nested-companion */").unwrap();
    fs::write(t.path().join("wgt.toml"), "[project]\nmodule = \"local/demo\"\nedition = \"0.5\"\n[model]\nsource = \"models/model.wgt\"\n").unwrap();
    assert!(html(&args).contains("/* nested-companion */"));
}

fn artifact(root: &Path, residual: bool) {
    fs::create_dir_all(root).unwrap();
    let layers = if residual {
        json!([{"name":"body","op":"residual","block":"std/blocks.demo","layers":[{"name":"fc","op":"linear","args":{"in":2,"out":2}}]}])
    } else {
        json!([{"name":"fc","op":"linear","args":{"in":2,"out":2}}])
    };
    let prefix = if residual { "body.fc" } else { "fc" };
    let weight = format!("{prefix}.weight");
    let bias = format!("{prefix}.bias");
    let header = json!({&bias:{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
        &weight:{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}});
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0u8; 24]);
    let source=b"wgt 0.5\nmodel demo\ninput x: f32[2]\noutput y: f32[2]\nlayer fc: linear(in=2, out=2)\ninit weights=xavier_uniform(42) bias=zeros()\n";
    let m = json!({"format":"weightlang.artifact","format_version":2,"model":"demo",
        "source":{"file":"source.wgt","sha256":hash(source)},
        "weights":{"file":"weights.safetensors","sha256":hash(&bytes),"tensors":{
            &weight:{"shape":[2,2],"dtype":"float32","origin":"random-init"},
            &bias:{"shape":[2],"dtype":"float32","origin":"random-init"}}},
        "interface":{"input":{"name":"x","dtype":"f32","shape":[2]},"output":{"name":"y","dtype":"f32","shape":[2]}},
        "layers":layers});
    fs::write(root.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    fs::write(root.join("weights.safetensors"), bytes).unwrap();
    fs::write(root.join("source.wgt"), source).unwrap();
}
struct Fake {
    commit: String,
    files: BTreeMap<String, Vec<u8>>,
}
impl Fake {
    fn new(root: &Path) -> Self {
        artifact(root, false);
        let mut f = Self {
            commit: COMMIT.into(),
            files: BTreeMap::new(),
        };
        let mut records = serde_json::Map::new();
        let base = format!("https://github.com/example/models/releases/download/gates/{VERSION}");
        for file in FILES {
            let bytes = fs::read(root.join(file)).unwrap();
            let asset = format!("demo.{file}");
            records.insert(
                file.into(),
                json!({"asset":asset,"sha256":hash(&bytes),"size":bytes.len()}),
            );
            f.files.insert(format!("{base}/{asset}"), bytes);
        }
        let index = json!({"format":"weightlang.package-index","format_version":1,"module":MODULE,"version":VERSION,
            "tag":format!("gates/{VERSION}"),"commit":COMMIT,"artifacts":{"demo":{"source_path":"main.wgt","files":records}}});
        f.files.insert(
            format!("{base}/wgt.index.json"),
            serde_json::to_vec(&index).unwrap(),
        );
        f.files.insert(
            format!("https://raw.githubusercontent.com/example/models/{COMMIT}/gates/main.wgt"),
            fs::read(root.join("source.wgt")).unwrap(),
        );
        f
    }
}
impl Remote for Fake {
    fn resolve(&self, _: &str, _: &str) -> Result<String> {
        Ok(self.commit.clone())
    }
    fn download(&self, url: &str, max: u64) -> Result<Vec<u8>> {
        let bytes = self
            .files
            .get(url)
            .ok_or_else(|| anyhow::anyhow!("missing {url}"))?;
        if bytes.len() as u64 > max {
            bail!("oversize");
        }
        Ok(bytes.clone())
    }
}
fn manifest() -> Manifest {
    Manifest::parse(&format!(
        "module local/demo\nwgt 0.5\nrequire {MODULE} {VERSION}\n"
    ))
    .unwrap()
}

#[test]
fn malicious_paths_and_duplicate_declarations_are_rejected() {
    for m in [
        "github.com/a/b/../escape",
        "github.com/./b",
        "github.com/a/b//c",
        "github.com/a/b/",
        "github.com/a/b?token",
    ] {
        assert!(manifest::module_parts(m).is_err(), "{m}");
    }
    for v in ["latest", "v01.2.3", "v1.2", "v1.2.3/../x", "1.2.3"] {
        assert!(manifest::version(v).is_err());
    }
    assert!(Manifest::parse("module a\nmodule b").is_err());
    assert!(Manifest::parse("module a\nwgt 0.5\nwgt 0.5").is_err());
    assert!(Manifest::parse(&format!(
        "module a\nrequire {MODULE} {VERSION}\nrequire {MODULE} {VERSION}"
    ))
    .is_err());
}
#[test]
fn manifest_edits_preserve_comments() {
    let m = Manifest::parse(&format!(
        "# hello\nmodule local/a\nwgt 0.5\nrequire {MODULE} v0.1.0 # pinned\n"
    ))
    .unwrap();
    let text = m.edited(MODULE, Some(VERSION));
    assert!(text.starts_with("# hello\n"));
    assert!(text.ends_with("v0.2.0 # pinned\n"));
    assert!(!m.edited(MODULE, None).contains("require"));
}
#[test]
fn native_fetch_lock_offline_and_tamper() {
    let t = tempfile::tempdir().unwrap();
    let cache = t.path().join("cache");
    let fake = Fake::new(&t.path().join("artifact"));
    let lock = registry::fetch(&manifest(), &Lock::new(), &cache, false, &fake).unwrap();
    assert_eq!(lock.len(), 5);
    assert_eq!(
        registry::verify(&manifest(), &lock, &cache).unwrap().len(),
        1
    );
    assert_eq!(
        registry::fetch(&manifest(), &lock, &cache, true, &fake).unwrap(),
        lock
    );
    let p = registry::module_dir(&cache, MODULE, VERSION).join("demo/source.wgt");
    fs::write(p, "tampered").unwrap();
    assert!(registry::verify(&manifest(), &lock, &cache).is_err());
}
#[test]
fn failed_fetch_preserves_verified_cache_and_rejects_moved_tag() {
    let t = tempfile::tempdir().unwrap();
    let cache = t.path().join("cache");
    let mut fake = Fake::new(&t.path().join("artifact"));
    let lock = registry::fetch(&manifest(), &Lock::new(), &cache, false, &fake).unwrap();
    fake.commit = "b".repeat(40);
    assert!(registry::fetch(&manifest(), &lock, &cache, false, &fake).is_err());
    fake.commit = COMMIT.into();
    let asset = fake
        .files
        .keys()
        .find(|s| s.ends_with("demo.weights.safetensors"))
        .unwrap()
        .clone();
    fake.files.get_mut(&asset).unwrap()[10] ^= 1;
    assert!(registry::fetch(&manifest(), &lock, &cache, false, &fake).is_err());
    assert!(registry::verify(&manifest(), &lock, &cache).is_ok());
}
#[test]
fn lock_prevents_first_use_and_tagged_source_is_checked() {
    let t = tempfile::tempdir().unwrap();
    let mut fake = Fake::new(&t.path().join("artifact"));
    let cache = t.path().join("cache");
    assert!(registry::fetch(&manifest(), &Lock::new(), &cache, true, &fake).is_err());
    assert!(!registry::module_dir(&cache, MODULE, VERSION).exists());
    let source = fake
        .files
        .keys()
        .find(|s| s.contains("raw.githubusercontent"))
        .unwrap()
        .clone();
    fake.files.insert(source, b"wrong source".to_vec());
    assert!(registry::fetch(&manifest(), &Lock::new(), &cache, false, &fake).is_err());
}
#[test]
fn graph_residual_edges_shapes_counts_and_formats() {
    let t = tempfile::tempdir().unwrap();
    artifact(t.path(), true);
    let g = Graph::read(t.path()).unwrap();
    assert_eq!(g.parameters, 6);
    assert_eq!(g.nodes.iter().map(|n| n.parameters).sum::<u64>(), 6);
    assert_eq!(g.edges.iter().filter(|e| e.kind == "skip").count(), 1);
    assert!(g
        .nodes
        .iter()
        .any(|n| n.path == "body.fc" && n.input == 2 && n.output == 2 && n.parent == "body"));
    for format in ["md", "json", "toon", "html", "svg", "mermaid", "dot"] {
        assert!(!g.render(format).unwrap().is_empty());
    }
    let value: Value = serde_json::from_str(&g.render("json").unwrap()).unwrap();
    assert_eq!(value["format"], "weightlang.graph");
    assert!(g.render("toon").unwrap().contains("nodes[5]{"));
    assert!(g.render("mermaid").unwrap().contains("-. skip .->"));
    let html = g.render("html").unwrap();
    assert!(!html.contains("<details>"));
    assert!(!html.contains("Agent-readable"));
    assert!(html.contains("<weightlang-viewer>"));
    assert!(html.contains("<script type=\"module\">"));
}
#[test]
fn graph_rejects_corruption_and_bad_safetensors() {
    let t = tempfile::tempdir().unwrap();
    artifact(t.path(), false);
    let mut m: Value =
        serde_json::from_slice(&fs::read(t.path().join("manifest.json")).unwrap()).unwrap();
    let file = t.path().join("weights.safetensors");
    let mut bytes = fs::read(&file).unwrap();
    bytes.extend([0u8; 4]);
    fs::write(&file, &bytes).unwrap();
    assert!(Graph::read(t.path()).is_err());
    m["weights"]["sha256"] = json!(hash(&bytes));
    fs::write(
        t.path().join("manifest.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    assert!(Artifact::read(t.path()).is_err()); // hashed, but malformed tensor layout
}
#[test]
fn cli_works_without_python_or_weightc_and_preserves_files() {
    let t = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_wgt");
    let run = |args: &[&str]| {
        Command::new(bin)
            .args(args)
            .env_clear()
            .env("PATH", "")
            .env("WEIGHTLANG_CACHE", t.path().join("cache"))
            .current_dir(t.path())
            .output()
            .unwrap()
    };
    assert!(run(&["new", "demo"]).status.success());
    assert!(run(&["-C", "demo", "metadata"]).status.success());
    assert!(run(&["-C", "demo", "fetch", "--offline"]).status.success());
    let manifest_before = fs::read(t.path().join("demo/wgt.toml")).unwrap();
    assert!(!run(&[
        "-C",
        "demo",
        "--offline",
        "add",
        MODULE,
        "--version",
        VERSION
    ])
    .status
    .success());
    assert_eq!(
        fs::read(t.path().join("demo/wgt.toml")).unwrap(),
        manifest_before
    );
    assert!(fs::read_to_string(t.path().join("demo/wgt.sum"))
        .unwrap()
        .is_empty());
    let before = fs::read(t.path().join("demo/model.wgt")).unwrap();
    assert!(!run(&["-C", "demo", "init"]).status.success());
    assert_eq!(fs::read(t.path().join("demo/model.wgt")).unwrap(), before);
    let check = run(&["-C", "demo", "check"]);
    assert!(!check.status.success());
    assert!(String::from_utf8_lossy(&check.stderr).contains("weightc"));
    artifact(&t.path().join("artifact"), true);
    assert!(run(&[
        "graph",
        "artifact",
        "--format",
        "json",
        "-o",
        "diagram.json"
    ])
    .status
    .success());
    let before = fs::read(t.path().join("diagram.json")).unwrap();
    assert!(!run(&["graph", "artifact", "-o", "diagram.json"])
        .status
        .success());
    assert_eq!(fs::read(t.path().join("diagram.json")).unwrap(), before);
}
#[cfg(unix)]
#[test]
fn extension_forwards_arguments_workdir_and_exit_code() {
    use std::os::unix::fs::PermissionsExt;
    let t = tempfile::tempdir().unwrap();
    let plugin = t.path().join("wgt-hello");
    fs::write(
        &plugin,
        "#!/bin/sh\nprintf '%s|%s|%s' \"$1\" \"$WGT_OFFLINE\" \"$PWD\"\nexit 7\n",
    )
    .unwrap();
    fs::set_permissions(&plugin, fs::Permissions::from_mode(0o755)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .args(["--offline", "hello", "a b"])
        .env("PATH", t.path())
        .current_dir(t.path())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(7));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("a b|1|{}", t.path().display())
    );
}

#[test]
fn native_package_pins_committed_sources_and_refuses_partial_or_untrained() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    let pkg = repo.join("gates");
    fs::create_dir_all(&pkg).unwrap();
    let arts = t.path().join("artifacts");
    let art = arts.join("demo");
    artifact(&art, false);
    fs::copy(art.join("source.wgt"), pkg.join("main.wgt")).unwrap();
    fs::write(pkg.join("wgt.mod"), format!("module {MODULE}\nwgt 0.5\n")).unwrap();
    registry::git(&repo, &["init", "-q"]).unwrap();
    registry::git(&repo, &["add", "."]).unwrap();
    registry::git(
        &repo,
        &[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    )
    .unwrap();
    let output = t.path().join("dist");
    assert!(registry::package(&pkg, &arts, VERSION, &output, false).is_err());
    assert!(!output.exists());
    let index = registry::package(&pkg, &arts, VERSION, &output, true).unwrap();
    assert_eq!(index["artifacts"]["demo"]["source_path"], "main.wgt");
    assert_eq!(
        index["artifacts"]["demo"]["untrained_tensors"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(output.join("wgt.index.json").exists());
    assert!(registry::package(&pkg, &arts, VERSION, &output, true).is_err());
    fs::write(pkg.join("missing.wgt"), "wgt 0.5\nmodel missing\n").unwrap();
    registry::git(&repo, &["add", "."]).unwrap();
    registry::git(
        &repo,
        &[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "missing model",
        ],
    )
    .unwrap();
    assert!(registry::package(&pkg, &arts, VERSION, &t.path().join("dist2"), true).is_err());
    fs::write(pkg.join("main.wgt"), "dirty").unwrap();
    assert!(registry::package(&pkg, &arts, VERSION, &t.path().join("dist3"), true).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn graph_temp_reuse_stdout_and_default_browser() {
    use std::os::unix::fs::PermissionsExt;
    let t = tempfile::tempdir().unwrap();
    artifact(&t.path().join("artifact"), true);
    let bin = env!("CARGO_BIN_EXE_wgt");
    let opener = t.path().join("xdg-open");
    fs::write(&opener, "#!/bin/sh\nprintf '%s' \"$1\" > \"$OPEN_LOG\"\n").unwrap();
    fs::set_permissions(&opener, fs::Permissions::from_mode(0o755)).unwrap();
    let log = t.path().join("opened-path");
    let run = |args: &[&str]| {
        Command::new(bin)
            .args(args)
            .current_dir(t.path())
            .env("TMPDIR", t.path())
            .env("PATH", t.path())
            .env("OPEN_LOG", &log)
            .output()
            .unwrap()
    };
    assert!(run(&["graph", "artifact", "--open"]).status.success());
    let output = t.path().join("weightlang/model.html");
    assert_eq!(fs::read_to_string(&log).unwrap(), output.to_str().unwrap());
    fs::write(&output, "old diagram").unwrap();
    assert!(run(&["graph", "artifact", "--open"]).status.success());
    assert!(fs::read_to_string(&output)
        .unwrap()
        .starts_with("<!doctype html>"));
    assert_eq!(fs::read_dir(output.parent().unwrap()).unwrap().count(), 1);
    let json = run(&["graph", "artifact", "--format", "json", "--stdout"]);
    assert!(json.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&json.stdout).unwrap()["format"],
        "weightlang.graph"
    );
    assert!(!t.path().join("weightlang/model.json").exists());
    assert!(!run(&["graph", "artifact", "--open", "--stdout"])
        .status
        .success());
    assert!(!run(&["graph", "artifact", "--open", "--format", "json"])
        .status
        .success());
    fs::write(&opener, "#!/bin/sh\nexit 3\n").unwrap();
    let result = run(&["graph", "artifact", "--open"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("diagram saved at"));
    assert!(output.is_file());
}

#[test]
fn multidimensional_contracts_infer_attention_and_cnn_weights() {
    use weightlang_wgt::contracts::{infer, Contracts, TensorType};
    let mut shapes = BTreeMap::new();
    let mut contracts = Contracts::new();
    let tokens = TensorType::parse(&json!({"dtype":"i64","shape":[4]})).unwrap();
    let layers = json!([
        {"name":"embed","op":"embedding","args":{"vocab":5,"dim":8}},
        {"name":"pos","op":"position_embedding","args":{"dim":8,"max_len":4}},
        {"name":"body","op":"residual","layers":[
            {"name":"attn","op":"self_attention","args":{"dim":8,"heads":2,"causal":1}}
        ]},
        {"name":"pool","op":"meanpool"},
        {"name":"fc","op":"linear","args":{"in":8,"out":2}}
    ]);
    let output = infer(&layers, "", tokens.clone(), &mut shapes, &mut contracts).unwrap();
    assert_eq!(output.label(), "f32[2]");
    assert_eq!(shapes["body.attn.qkv.weight"], vec![24, 8]);
    assert_eq!(shapes["pos.weight"], vec![4, 8]);
    let mut broken = layers.clone();
    broken[2]["layers"][0]["args"]["heads"] = json!(3);
    assert!(infer(
        &broken,
        "",
        tokens.clone(),
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
    broken = layers;
    broken[0]["output"] = json!({"dtype":"f32","shape":[9,8]});
    assert!(infer(
        &broken,
        "",
        tokens,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
    let image = TensorType::parse(&json!({"dtype":"f32","shape":[1,8,8]})).unwrap();
    let cnn = json!([
        {"name":"conv","op":"conv2d","args":{"in":1,"out":4,"kernel":3,"stride":1,"padding":1}},
        {"name":"pool","op":"maxpool2d","args":{"kernel":2,"stride":2}},
        {"name":"flat","op":"flatten"}
    ]);
    let out = infer(&cnn, "", image.clone(), &mut shapes, &mut contracts).unwrap();
    assert_eq!(out.shape, vec![64]);
    assert_eq!(shapes["conv.weight"], vec![4, 1, 3, 3]);
    assert_eq!(contracts["pool"].1.shape, vec![4, 4, 4]);
    let mut invalid = cnn;
    invalid[0]["args"]["padding"] = json!(u64::MAX);
    assert!(infer(
        &invalid,
        "",
        image,
        &mut BTreeMap::new(),
        &mut Contracts::new()
    )
    .is_err());
}

#[test]
fn edited_weights_remain_untrained_and_visible_in_graphs() {
    let temp = tempfile::tempdir().unwrap();
    artifact(temp.path(), false);
    let path = temp.path().join("manifest.json");
    let mut m: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    m["format_version"] = json!(3);
    m["weights"]["tensors"]["fc.weight"]["origin"] = json!("edited");
    m["weights"]["tensors"]["fc.bias"]["origin"] = json!("trained");
    fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let a = Artifact::read(temp.path()).unwrap();
    assert_eq!(a.untrained(), vec!["fc.weight"]);
    let graph = Graph::read(temp.path()).unwrap().render("json").unwrap();
    assert!(graph.contains("edited"));
    m["weights"]["tensors"]["fc.weight"]["origin"] = json!("imported");
    m["weights"]["tensors"]["fc.weight"]["upstream_origin"] = json!("edited");
    fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    assert_eq!(
        Artifact::read(temp.path()).unwrap().untrained(),
        vec!["fc.weight"]
    );
}

#[test]
fn missing_graph_artifact_reports_resolved_path_and_directory_hint() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .args(["graph", "build/missing", "--open"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(
        temp.path()
            .join("build/missing/manifest.json")
            .to_str()
            .unwrap()
    ));
    assert!(error.contains("-C"));
    assert!(error.contains("build the model first"));
    artifact(&temp.path().join("nested/build/demo"), false);
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .current_dir(temp.path())
        .args([
            "-C",
            "nested",
            "graph",
            "build/demo",
            "--format",
            "json",
            "--stdout",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
}

#[test]
fn weight_domain_v7_artifact_validation() {
    let root = tempfile::tempdir().unwrap();
    artifact(root.path(), false);
    let path = root.path().join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["format_version"] = json!(7);
    manifest["interface"]["inputs"] = json!([{"name":"x","dtype":"f32","shape":[2]}]);
    manifest["interface"]["outputs"] = json!([{"name":"y","dtype":"f32","shape":[2],"from":"fc"}]);
    manifest["interface"]["parameter_ids"] = json!({});
    manifest["interface"]["parameter_shares"] = json!({});
    manifest["interface"]["weight_types"] = json!({"tri":[-1,0,1]});
    manifest["interface"]["weight_assignments"] =
        json!([{"target":"fc.weight","type":"tri","axis":0,"start":0,"stop":1}]);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    Artifact::read(root.path()).unwrap();
    manifest["interface"]["weight_assignments"][0]["stop"] = json!(3);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(Artifact::read(root.path()).is_err());
    manifest["format_version"] = json!(6);
    manifest["interface"]["weight_assignments"][0]["stop"] = json!(1);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(Artifact::read(root.path()).is_err());
}

#[test]
fn inline_script_dependencies_resolve_verified_cache_in_both_execution_modes() {
    use sha2::{Digest, Sha256};
    let temp = std::env::temp_dir().join("weightlang");
    fs::create_dir_all(&temp).unwrap();
    let dir = tempfile::tempdir_in(temp).unwrap();
    let cache = dir.path().join("cache");
    let fake = Fake::new(&dir.path().join("fixture"));
    let lock = registry::fetch(&manifest(), &Lock::new(), &cache, false, &fake).unwrap();
    let source = dir.path().join("train.wgt");
    let body = format!(
        r#"# /// script
# [dependencies]
# "{MODULE}" = "{VERSION}"
# ///
wgt 0.5
fn main() {{
    let reader = jsonl(dependency("{MODULE}") + "/demo/manifest.json");
    let rows = batch(reader, 1);
    close(reader);
    return rows[0]["model"];
}}
"#
    );
    fs::write(&source, body).unwrap();
    let key = format!(
        "{:x}",
        Sha256::digest(source.canonicalize().unwrap().to_string_lossy().as_bytes())
    );
    let state = cache.join("scripts").join(key);
    fs::create_dir_all(&state).unwrap();
    fs::write(state.join("wgt.sum"), manifest::lock_text(&lock)).unwrap();
    for compiled in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_wgt"));
        cmd.env("WEIGHTLANG_CACHE", &cache)
            .arg("--offline")
            .arg("--locked")
            .arg(&source);
        if compiled {
            cmd.arg("--compiled");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["result"], "demo");
    }
    assert!(!dir.path().join("wgt.toml").exists());
    assert!(!dir.path().join("wgt.sum").exists());
    // Project scripts use the project's existing declaration and lock instead.
    fs::write(
        &source,
        format!("wgt 0.5\nfn main() {{ return dependency(\"{MODULE}\"); }}"),
    )
    .unwrap();
    fs::write(dir.path().join("wgt.toml"), format!("[project]\nmodule=\"local/test\"\nedition=\"0.5\"\n[dependencies]\n\"{MODULE}\"=\"{VERSION}\"\n")).unwrap();
    fs::write(dir.path().join("wgt.sum"), manifest::lock_text(&lock)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .env("WEIGHTLANG_CACHE", &cache)
        .args(["--offline", "--locked"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::write(
        registry::module_dir(&cache, MODULE, VERSION).join("demo/source.wgt"),
        "tampered",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .env("WEIGHTLANG_CACHE", &cache)
        .arg("--offline")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("cache hash mismatch"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // An explicit empty block must not inherit the now-corrupted project dependency.
    fs::write(
        &source,
        "# /// script\n# [dependencies]\n# ///\nwgt 0.5\nfn main() { return 42; }",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_wgt"))
        .env("WEIGHTLANG_CACHE", &cache)
        .args(["--offline", "--locked"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["result"], 42.0);
}
