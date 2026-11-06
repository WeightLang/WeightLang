//! Client-owned staging for descriptor writes; publication never reads model bytes.
use anyhow::{Result, ensure};
use std::{
    fs::{self, File, OpenOptions},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
pub enum Mode {
    CreateNew,
    Replace,
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Write beside the destination, then atomically publish the completed file.
/// Existing replacement targets must be writable regular files; their permissions are preserved.
pub fn write(
    destination: &Path,
    mode: Mode,
    writer: impl FnOnce(&File) -> Result<()>,
) -> Result<()> {
    let permissions = if matches!(mode, Mode::Replace) {
        match fs::symlink_metadata(destination) {
            Ok(metadata) => {
                ensure!(
                    metadata.is_file() && !metadata.permissions().readonly(),
                    "overwrite requires writable regular file"
                );
                Some(metadata.permissions())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        }
    } else {
        None
    };
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    if let Some(permissions) = &permissions {
        options.mode(permissions.mode());
    }
    let (temporary, file) = loop {
        let candidate = parent.join(format!(
            ".wgtime-stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        if candidate.file_name() == destination.file_name() {
            continue;
        }
        match options.open(&candidate) {
            Ok(file) => break (Temporary(candidate), file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    if let Some(permissions) = permissions {
        file.set_permissions(permissions)?;
    }
    writer(&file)?;
    file.sync_all()?;
    drop(file);
    match mode {
        Mode::CreateNew => fs::hard_link(&temporary.0, destination)?,
        Mode::Replace => {
            if let Ok(metadata) = fs::symlink_metadata(destination) {
                ensure!(metadata.is_file(), "overwrite requires regular file");
            }
            fs::rename(&temporary.0, destination)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::sync::Mutex;

    static COUNTER: Mutex<()> = Mutex::new(());

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Result<Self> {
            let path = std::env::temp_dir().join("weightlang").join(format!(
                "publication-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.parent().unwrap())?;
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn complete_file_is_published_once_without_clobber_or_staging_leaks() -> Result<()> {
        let _counter_guard = COUNTER.lock().unwrap();
        let directory = Directory::new()?;
        let path = directory.0.join("weights");
        write(&path, Mode::CreateNew, |mut file| {
            Ok(file.write_all(b"original")?)
        })?;
        assert!(
            write(&path, Mode::CreateNew, |mut file| Ok(
                file.write_all(b"replacement")?
            ))
            .is_err()
        );
        assert_eq!(fs::read(&path)?, b"original");
        assert_eq!(fs::read_dir(&directory.0)?.count(), 1);
        Ok(())
    }

    #[test]
    fn failed_writers_leave_existing_or_absent_destination_unchanged() -> Result<()> {
        let _counter_guard = COUNTER.lock().unwrap();
        let directory = Directory::new()?;
        let path = directory.0.join("weights");
        let fail = |mut file: &File| -> Result<()> {
            file.write_all(b"partial")?;
            anyhow::bail!("writer failed")
        };
        assert!(write(&path, Mode::CreateNew, fail).is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&directory.0)?.count(), 0);
        fs::write(&path, b"original")?;
        assert!(write(&path, Mode::Replace, fail).is_err());
        assert_eq!(fs::read(&path)?, b"original");
        assert_eq!(fs::read_dir(&directory.0)?.count(), 1);
        Ok(())
    }

    #[test]
    fn replacement_preserves_permissions_and_rejects_protected_targets() -> Result<()> {
        let _counter_guard = COUNTER.lock().unwrap();
        let directory = Directory::new()?;
        let path = directory.0.join("weights");
        fs::write(&path, b"original")?;
        for mode in [0o600, 0o640] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))?;
            write(&path, Mode::Replace, |mut file| {
                assert_eq!(file.metadata()?.permissions().mode() & 0o777, mode);
                Ok(file.write_all(b"replacement")?)
            })?;
            assert_eq!(fs::read(&path)?, b"replacement");
            assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, mode);
        }
        fs::set_permissions(&path, fs::Permissions::from_mode(0o440))?;
        assert!(
            write(&path, Mode::Replace, |_| panic!(
                "readonly target reached writer"
            ))
            .is_err()
        );
        assert_eq!(fs::read(&path)?, b"replacement");
        let link = directory.0.join("link");
        symlink(&path, &link)?;
        assert!(
            write(&link, Mode::Replace, |_| panic!(
                "symlink target reached writer"
            ))
            .is_err()
        );
        assert!(fs::symlink_metadata(&link)?.file_type().is_symlink());
        assert!(
            write(&directory.0, Mode::Replace, |_| panic!(
                "directory target reached writer"
            ))
            .is_err()
        );
        assert_eq!(fs::read_dir(&directory.0)?.count(), 2);
        Ok(())
    }

    #[test]
    fn replacement_destination_may_match_the_next_staging_name() -> Result<()> {
        let _counter_guard = COUNTER.lock().unwrap();
        let directory = Directory::new()?;
        let path = directory.0.join(format!(
            ".wgtime-stage-{}-{}",
            std::process::id(),
            NEXT.load(Ordering::Relaxed)
        ));
        write(&path, Mode::Replace, |mut file| {
            Ok(file.write_all(b"replacement")?)
        })?;
        assert_eq!(fs::read(&path)?, b"replacement");
        assert_eq!(fs::read_dir(&directory.0)?.count(), 1);
        Ok(())
    }
}
