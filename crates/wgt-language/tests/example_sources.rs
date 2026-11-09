use std::{fs, path::{Path, PathBuf}};
use wgt_language::model::ModelDefinition;

fn visit(directory: &Path, models: &mut Vec<PathBuf>) {
    if !directory.is_dir() { return; }
    for entry in fs::read_dir(directory).expect("read model examples") {
        let entry = entry.expect("read model example entry");
        let path = entry.path();
        if path.is_dir() {
            visit(&path, models);
        } else if path.file_name().is_some_and(|name| name == "model.wgt") {
            models.push(path);
        }
    }
}

#[test]
fn every_current_model_example_passes_the_shared_frontend() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut models = Vec::new();
    for directory in [
        "examples", "packages", "python/examples", "language/syntax-tour",
        "crates/wgt-model/tests/fixtures", "crates/wgt-model-macros/tests/fixtures",
        "crates/wgtime-core/tests/fixtures",
    ] {
        visit(&root.join(directory), &mut models);
    }
    models.sort();
    assert!(models.len() >= 25, "missing model examples from inventory");
    for path in models {
        // This compatibility fixture intentionally contains the retired host `fn main`.
        if path.ends_with("crates/wgtime-core/tests/fixtures/ppo_legacy/model.wgt") {
            continue;
        }
        let model = ModelDefinition::from_file(&path)
            .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
        model.instantiate_plan()
            .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
    }
}
