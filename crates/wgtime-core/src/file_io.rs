//! Bounded serialization, file reads and atomic core artifact publication.
//!
//! Callers authorize the operation and charge their resource budget before using
//! these helpers. Parents must already exist, matching native model persistence.
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Stop serialization before extending its output buffer beyond the IO budget.
pub(crate) fn encode_json(value: &impl serde::Serialize, limit: usize) -> anyhow::Result<Vec<u8>> {
    struct Bounded { bytes: Vec<u8>, limit: usize }
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "serialized file exceeds IO byte budget"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut writer = Bounded { bytes: Vec::new(), limit };
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes)
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Limit the actual read, including when a regular file grows after metadata.
pub(crate) fn read_bounded(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    // Refuse special files without blocking on a FIFO substituted before open.
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "load requires a regular file",
        ));
    }
    if metadata.len() > limit as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file exceeds IO byte budget",
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file exceeds IO byte budget",
        ));
    }
    Ok(bytes)
}

/// Fully stage and sync data before publishing. Overwrite never follows the
/// destination symlink. Replacement preserves the previous regular file's mode.
/// Like existing model saves, this does not create missing parent directories.
pub(crate) fn publish(path: &Path, bytes: &[u8], overwrite: bool) -> io::Result<()> {
    let previous = match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if !overwrite {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "save target already exists; use overwrite explicitly",
                ));
            }
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "overwrite requires a regular file, not a symlink or directory",
                ));
            }
            if metadata.permissions().readonly() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "save target is read-only",
                ));
            }
            Some(metadata)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let (temporary, mut file) = loop {
        let nonce = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".wgt-io-{}-{nonce}", std::process::id()));
        if candidate.file_name() == path.file_name() {
            continue;
        }
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (Temporary(candidate), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    file.write_all(bytes)?;
    if let Some(metadata) = previous {
        file.set_permissions(metadata.permissions())?;
    }
    file.sync_all()?;
    drop(file);
    if overwrite {
        // Recheck immediately before publication. Rename replaces the directory
        // entry itself, so even a raced-in symlink is never followed or written.
        match fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.is_file() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "overwrite requires a regular file",
                ))
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        fs::rename(&temporary.0, path)?;
    } else {
        // Atomic no-clobber publication, including a raced-in symlink.
        fs::hard_link(&temporary.0, path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static COUNTER: Mutex<()> = Mutex::new(());

    fn directory() -> PathBuf {
        let path = PathBuf::from("/tmp/weightlang").join(format!(
            "file-io-{}-{}",
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn publication_is_explicit_and_read_is_bounded() {
        let _counter_guard = COUNTER.lock().unwrap();
        let root = directory();
        let path = root.join("文本.txt");
        publish(&path, "初始".as_bytes(), false).unwrap();
        assert_eq!(
            publish(&path, b"new", false).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(read_bounded(&path, 6).unwrap(), "初始".as_bytes());
        assert!(read_bounded(&path, 5).is_err());
        publish(&path, b"new", true).unwrap();
        assert_eq!(read_bounded(&path, 3).unwrap(), b"new");
        assert!(publish(&root, b"bad", true).is_err());
        assert!(publish(&root.join("missing/file"), b"bad", false).is_err());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn links_are_rejected_and_permissions_survive_replacement() {
        let _counter_guard = COUNTER.lock().unwrap();
        use std::os::unix::fs::{symlink, PermissionsExt};
        let root = directory();
        let path = root.join("target");
        publish(&path, b"old", false).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        let link = root.join("link");
        symlink(&path, &link).unwrap();
        assert!(publish(&link, b"bad", false).is_err());
        assert!(publish(&link, b"bad", true).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
        publish(&path, b"new", true).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        fs::set_permissions(&path, fs::Permissions::from_mode(0o440)).unwrap();
        assert_eq!(
            publish(&path, b"bad", true).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(fs::read(&path).unwrap(), b"new");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn replacement_destination_may_match_the_next_staging_name() {
        let _counter_guard = COUNTER.lock().unwrap();
        let root = directory();
        let path = root.join(format!(
            ".wgt-io-{}-{}",
            std::process::id(),
            NEXT_TEMPORARY.load(Ordering::Relaxed)
        ));
        publish(&path, b"complete", true).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"complete");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn serialization_respects_exact_byte_budget() {
        let value = serde_json::json!({"text": "雪\n", "values": [1, 2, 3]});
        let expected = serde_json::to_vec(&value).unwrap();
        assert_eq!(encode_json(&value, expected.len()).unwrap(), expected);
        assert!(encode_json(&value, expected.len() - 1).is_err());
        assert!(encode_json(&value, 0).is_err());
    }
}
