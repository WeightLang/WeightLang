//! Test discovery reuses the script parser; each invocation owns all runtime state.
use super::*;

pub struct Suite {
    program: Program,
    names: Vec<String>,
}

impl Suite {
    /// Discover ordinary test_* functions. Invalid signatures fail individually at run time.
    pub fn parse(source: &str) -> Result<Self> {
        let program = super::parse(source)?;
        Self::from_program(program)
    }
    pub fn parse_with_imports(
        source: &str,
        root: &Path,
        dependencies: &BTreeMap<String, String>,
    ) -> Result<Self> {
        Self::from_program(super::parse_with_imports(source, root, dependencies)?)
    }
    fn from_program(program: Program) -> Result<Self> {
        let mut names: Vec<_> = program
            .iter()
            .filter(|(name, function)| {
                name.starts_with("test_") && !matches!(function.body.first(), Some(Statement { kind: Stmt::Return(Expr::Call(builtin, _)), .. }) if builtin == "__graph" || builtin == "__adaptive")
            })
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        Ok(Self { program, names })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn run(
        &self,
        name: &str,
        root: &Path,
        max_operations: u64,
        dependencies: BTreeMap<String, String>,
    ) -> Result<Report> {
        ensure!(self.names.iter().any(|n| n == name), "unknown test: {name}");
        let (runtime, value) = execute_program(
            self.program.clone(),
            root,
            max_operations,
            dependencies,
            name,
        )?;
        Ok(Report {
            result: value.json()?,
            metrics: runtime.metrics,
            operations: max_operations - runtime.remaining,
        })
    }
}
