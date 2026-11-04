//! Embed the bundled `std/**/*.wgt` sources so `wgt graph` can ship the modules a
//! model uses (with their comments) to the architecture viewer.
use std::{env, fs, path::Path};

fn collect(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("read std")
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "wgt") {
            let module = path.strip_prefix(root).unwrap().with_extension("");
            let module = format!("std/{}", module.to_string_lossy().replace('\\', "/"));
            out.push((
                module,
                path.canonicalize().unwrap().to_string_lossy().into_owned(),
            ));
        }
    }
}

fn main() {
    let manifest = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must supply CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).join("../../std");
    println!("cargo:rerun-if-env-changed=CARGO_MANIFEST_DIR");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut modules = Vec::new();
    collect(&root, &root, &mut modules);
    let table: String = modules
        .iter()
        .map(|(module, path)| format!("    ({module:?}, include_str!({path:?})),\n"))
        .collect();
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("std_sources.rs");
    fs::write(
        out,
        format!("pub const STD_SOURCES: &[(&str, &str)] = &[\n{table}];\n"),
    )
    .unwrap();
}
