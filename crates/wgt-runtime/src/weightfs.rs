//! weightfs v1: live model access serialized with execution, without disk snapshots.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

#[cfg(target_os = "linux")]
mod fuse;

#[derive(Clone, Serialize)]
pub struct Entry {
    pub name: String,
    pub shape: Vec<usize>,
    pub writable: bool,
}

pub trait Weights: Send + 'static {
    fn entries(&self) -> Vec<Entry>;
    fn revision(&self) -> u64;
    fn read(&mut self, name: &str) -> Result<Vec<f32>>;
    fn replace(&mut self, name: &str, values: &[f32]) -> Result<()>;
}

macro_rules! access {
    ($ty:ty) => {
        impl Weights for $ty {
            fn entries(&self) -> Vec<Entry> {
                self.parameter_info()
                    .into_iter()
                    .map(|p| Entry {
                        name: p.name.into(),
                        shape: p.shape.into(),
                        writable: p.writable,
                    })
                    .collect()
            }
            fn revision(&self) -> u64 {
                self.revision()
            }
            fn read(&mut self, name: &str) -> Result<Vec<f32>> {
                let entry = self
                    .entries()
                    .into_iter()
                    .find(|e| e.name == name)
                    .context("unknown parameter")?;
                let mut values = vec![0.; entry.shape.iter().product()];
                self.read_parameter(name, 0, &mut values)?;
                Ok(values)
            }
            fn replace(&mut self, name: &str, values: &[f32]) -> Result<()> {
                self.write_parameters(&[crate::ParameterWrite {
                    name,
                    start: 0,
                    values,
                }])
            }
        }
    };
}
access!(crate::Session);
access!(crate::execution::ExecutionSession);
impl Weights for crate::training::Trainer {
    fn entries(&self) -> Vec<Entry> {
        self.session.entries()
    }
    fn revision(&self) -> u64 {
        self.session.revision()
    }
    fn read(&mut self, name: &str) -> Result<Vec<f32>> {
        self.session.read(name)
    }
    fn replace(&mut self, name: &str, values: &[f32]) -> Result<()> {
        // Manual editing preserves optimizer history, like explicit parameter writes.
        self.session.replace(name, values)
    }
}
impl Weights for crate::adaptive::AdaptiveSession {
    fn entries(&self) -> Vec<Entry> {
        self.program()
            .parameters
            .iter()
            .map(|p| Entry {
                name: p.name.clone(),
                shape: p.shape.clone(),
                writable: !p.frozen,
            })
            .collect()
    }
    fn revision(&self) -> u64 {
        self.revision()
    }
    fn read(&mut self, name: &str) -> Result<Vec<f32>> {
        Ok(self.parameter(name)?.to_vec())
    }
    fn replace(&mut self, name: &str, values: &[f32]) -> Result<()> {
        self.replace_parameter(name, values)
    }
}

/// CLI/script lifecycle wrapper. Bare Rust sessions remain free of OS side effects.
pub struct Mounted<T: Weights> {
    value: Arc<Mutex<T>>,
    #[cfg(target_os = "linux")]
    mount: Option<fuse::Mount>,
}
impl<T: Weights> Mounted<T> {
    pub fn auto(value: T) -> Result<Self> {
        let mode = std::env::var("WGT_WEIGHTFS").unwrap_or_else(|_| "auto".into());
        ensure!(
            matches!(mode.as_str(), "auto" | "on" | "off"),
            "WGT_WEIGHTFS must be auto, on or off"
        );
        let value = Arc::new(Mutex::new(value));
        let mut result = Self {
            value,
            #[cfg(target_os = "linux")]
            mount: None,
        };
        // A host I/O policy must not be bypassed through an automatically exposed filesystem.
        if std::env::var_os("WGT_HOST_POLICY").is_some() {
            ensure!(
                mode != "on",
                "weightfs is unavailable under WGT_HOST_POLICY"
            );
            return Ok(result);
        }
        if mode == "off" {
            return Ok(result);
        }
        #[cfg(target_os = "linux")]
        match fuse::Mount::new(result.value.clone()) {
            Ok(mount) => {
                eprintln!("weightfs: {}", mount.path.display());
                result.mount = Some(mount);
            }
            Err(e) if mode == "auto" => {
                eprintln!("weightfs unavailable: {e:#}; use WGT_WEIGHTFS=on to require mounting")
            }
            Err(e) => return Err(e),
        }
        #[cfg(not(target_os = "linux"))]
        {
            ensure!(mode != "on", "weightfs currently requires Linux FUSE");
            eprintln!("weightfs unavailable: currently requires Linux FUSE");
        }
        Ok(result)
    }
    pub fn lock(&self) -> Result<MutexGuard<'_, T>> {
        self.value
            .lock()
            .map_err(|_| anyhow::anyhow!("weightfs model lock poisoned"))
    }
    pub fn path(&self) -> Option<&Path> {
        #[cfg(target_os = "linux")]
        {
            self.mount.as_ref().map(|m| m.path.as_path())
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
    pub fn into_inner(mut self) -> Result<T> {
        #[cfg(target_os = "linux")]
        drop(self.mount.take());
        Arc::try_unwrap(self.value)
            .map_err(|_| anyhow::anyhow!("weightfs still holds model"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("weightfs model lock poisoned"))
    }
}
