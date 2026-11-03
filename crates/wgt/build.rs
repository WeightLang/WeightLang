use std::{env, fs, path::Path};

fn collect(root: &Path, path: &Path, entries: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(path).expect("read bundled compiler source") {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().unwrap() != "__pycache__" {
                collect(root, &path, entries);
            }
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("py" | "wgt")
        ) {
            entries.push((
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
                path.to_str().unwrap().to_owned(),
            ));
        }
    }
}
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let source = root.join("src");
    println!(
        "cargo:rerun-if-changed={}",
        source.join("weightlang").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("pyproject.toml").display()
    );
    let mut entries = Vec::new();
    collect(&source, &source.join("weightlang"), &mut entries);
    entries.sort();
    let mut generated = String::from("pub const COMPILER_FILES: &[(&str, &[u8])] = &[\n");
    for (name, path) in entries {
        generated.push_str(&format!("({name:?}, include_bytes!({path:?})),\n"));
    }
    generated.push_str("];\n");
    generated.push_str(&format!(
        "pub const COMPILER_PROJECT: &str = include_str!({:?});\n",
        root.join("pyproject.toml").to_str().unwrap()
    ));
    fs::write(
        Path::new(&env::var_os("OUT_DIR").unwrap()).join("compiler.rs"),
        generated,
    )
    .unwrap();
}
