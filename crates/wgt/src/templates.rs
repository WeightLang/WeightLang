use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};
use weightlang_wgt::{manifest, toolchain};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Template {
    #[default]
    Native,
    Mlp,
    TinyLm,
}

#[derive(Args, Default)]
pub struct Options {
    /// Start from a native model, a regression model, or a small language model
    #[arg(long, value_enum, conflicts_with = "interactive")]
    pub template: Option<Template>,
    /// Choose a template and Python version with a short guided menu
    #[arg(long, short = 'i')]
    pub interactive: bool,
    /// Python version for uv to select, for example 3.12 (Python templates only)
    #[arg(long)]
    pub python: Option<String>,
    /// Install the template's Python and dependencies with uv after creation
    #[arg(long)]
    pub sync: bool,
}

impl Template {
    pub fn is_python(self) -> bool {
        !matches!(self, Self::Native)
    }
}

pub fn list() {
    println!(
        "Template   What you can do\n\
              native     Define a native WGT model; check without Python\n\
              mlp        Train a small regression model with PyTorch\n\
              tiny-lm    Train and generate text with a small GPT-2 architecture\n\n\
              Start: wgt init my-model --template tiny-lm\n\
              Guided: wgt init my-model --interactive"
    );
}

fn answer(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
    io::stderr().flush()?;
    let mut text = String::new();
    ensure!(
        io::stdin().read_line(&mut text)? > 0,
        "input ended; use --template for non-interactive initialization"
    );
    Ok(text.trim().to_owned())
}

fn select(options: &Options) -> Result<(Template, Option<String>)> {
    let template = if options.interactive {
        eprintln!(
            "What would you like to build?\n  1. Small language model — train on text, then generate text\n  2. Regression model — learn a relationship between numbers\n  3. Native WGT model — describe a model for the Rust runtime"
        );
        match answer("Choose [1]: ")?.as_str() {
            "" | "1" => Template::TinyLm,
            "2" => Template::Mlp,
            "3" => Template::Native,
            _ => bail!("choose 1, 2 or 3; or use --template tiny-lm, mlp or native"),
        }
    } else {
        options.template.unwrap_or_default()
    };
    ensure!(
        template.is_python() || options.python.is_none(),
        "--python requires the mlp or tiny-lm template"
    );
    let python = if template.is_python() {
        let version = match &options.python {
            Some(version) => version.clone(),
            None if options.interactive => {
                let text = answer("Python version [3.12]: ")?;
                if text.is_empty() { "3.12".into() } else { text }
            }
            None => "3.12".into(),
        };
        let parts: Vec<_> = version.split('.').collect();
        ensure!(
            (parts.len() == 2 || parts.len() == 3)
                && parts[0] == "3"
                && ["11", "12", "13"].contains(&parts[1])
                && parts
                    .iter()
                    .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())),
            "Python starters support 3.11, 3.12 or 3.13, optionally with a patch version"
        );
        Some(version)
    } else {
        None
    };
    Ok((template, python))
}

/// Create source/configuration only. Downloads and training are separate actions.
pub fn initialize(path: &Path, module: Option<String>, options: &Options) -> Result<Template> {
    let (template, python) = select(options)?;
    let resolved = if path.exists() {
        fs::canonicalize(path)?
    } else {
        path.to_path_buf()
    };
    let directory_name = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid project directory")?;
    let mut name: String = directory_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    name = name.trim_matches('-').to_owned();
    if name.is_empty() {
        name = "my-model".into();
    }
    let module = module.unwrap_or_else(|| format!("local/{name}"));
    ensure!(
        !module.is_empty() && !module.chars().any(char::is_whitespace) && !module.contains('#'),
        "invalid module name"
    );

    let mut files = vec![
        ("wgt.sum", String::new()),
        (toolchain::PIN, format!("{}\n", toolchain::CURRENT)),
    ];
    match template {
        Template::Native => {
            files.push(("wgt.toml", format!(
                "[project]\nmodule = {}\nedition = \"0.5\"\n\n[model]\nsource = \"model.wgt\"\noutput = \"weights\"\n\n[revisions]\nmax_versions = 5\n\n[dependencies]\n",
                serde_json::to_string(&module)?
            )));
            files.push(("model.wgt", include_str!("../templates/native.wgt").into()));
        }
        Template::Mlp | Template::TinyLm => {
            let source = if matches!(template, Template::TinyLm) {
                include_str!("../templates/tiny-lm.toml")
            } else {
                include_str!("../templates/mlp.toml")
            };
            let config = source
                .replace("PROJECT_NAME", &name)
                .replace("\"PROJECT_MODULE\"", &serde_json::to_string(&module)?);
            manifest::ProjectConfig::parse(&config)?;
            files.push(("pyproject.toml", config));
            files.push((".python-version", format!("{}\n", python.unwrap())));
            files.push((
                "workflow.py",
                include_str!("../templates/workflow.py").into(),
            ));
            files.push(if matches!(template, Template::TinyLm) {
                ("data.txt", include_str!("../templates/tiny-lm.txt").into())
            } else {
                ("data.csv", include_str!("../templates/mlp.csv").into())
            });
        }
    }
    // Preflight every output before creating anything, including an existing
    // manifest from another ecosystem. A partial starter is hard to recover from.
    for file in [
        "wgt.mod",
        "wgt.toml",
        "pyproject.toml",
        "wgt.lock",
        "main.wgt",
        "model.wgt",
        "tests/main.json",
    ]
    .into_iter()
    .chain(files.iter().map(|(name, _)| *name))
    {
        ensure!(
            fs::symlink_metadata(resolved.join(file))
                .is_err_and(|e| e.kind() == io::ErrorKind::NotFound),
            "refusing to overwrite {file}"
        );
    }
    match fs::symlink_metadata(resolved.join(".gitignore")) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file(),
                ".gitignore must be a regular file, not a directory or symlink"
            );
            fs::read_to_string(resolved.join(".gitignore")).context("read existing .gitignore")?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect existing .gitignore"),
    }
    fs::create_dir_all(&resolved)?;
    for (name, content) in files {
        fs::write(resolved.join(name), content).with_context(|| format!("write {name}"))?;
    }
    super::project_ignore::initialize(&resolved)?;
    println!("Created {}", resolved.display());
    if template.is_python() {
        println!(
            "Edit pyproject.toml to choose model size, training and inference settings.\n\
                  From this directory:\n  wgt train\n  wgt infer\n\
                  uv prepares Python and dependencies automatically on the first run.\n\
                  Add Python libraries with: uv add PACKAGE"
        );
    } else {
        println!("From this directory: wgt check");
    }
    Ok(template)
}
