use std::{fs, io, path::Path};

const DEFAULTS: &str = include_str!("project.gitignore");

pub fn initialize(project: &Path) -> io::Result<()> {
    let path = project.join(".gitignore");
    let existing = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return fs::write(path, DEFAULTS);
        }
        Err(error) => return Err(error),
    };
    let missing: Vec<_> = DEFAULTS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter(|rule| !existing.lines().any(|line| line == *rule))
        .collect();
    if !missing.is_empty() {
        // Keep user rules last so explicit exceptions retain their precedence.
        fs::write(
            path,
            format!(
                "# WeightLang defaults\n{}\n\n{existing}",
                missing.join("\n")
            ),
        )?;
    }
    Ok(())
}
