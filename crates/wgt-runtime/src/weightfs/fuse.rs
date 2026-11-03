use super::{Entry, Weights};
use anyhow::{ensure, Context, Result};
use fuser::*;
use libc::{EACCES, EAGAIN, EBADF, EFBIG, EINVAL, EIO, ENOENT, ENOSPC, EROFS};
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs,
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const TTL: Duration = Duration::ZERO;
const MAX_OPEN: usize = 64;
const MAX_BUFFER: usize = 64 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(super) struct Mount {
    pub path: PathBuf,
    session: Option<(
        SessionUnmounter,
        std::thread::JoinHandle<std::io::Result<()>>,
    )>,
}
impl Mount {
    pub fn new<T: Weights>(model: Arc<Mutex<T>>) -> Result<Self> {
        let uid = unsafe { libc::geteuid() };
        let base = match std::env::var_os("XDG_RUNTIME_DIR") {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(format!("/run/user/{uid}")),
        };
        ensure!(base.is_absolute(), "XDG_RUNTIME_DIR must be absolute");
        let mut path = base;
        for component in [None, Some("weightlang"), Some("weightfs")] {
            if let Some(c) = component {
                path.push(c);
                match fs::DirBuilder::new().mode(0o700).create(&path) {
                    Ok(()) => (),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
                    Err(e) => return Err(e.into()),
                }
            }
            let meta = fs::symlink_metadata(&path)
                .context("weightfs requires a private runtime directory")?;
            ensure!(
                meta.is_dir()
                    && !meta.file_type().is_symlink()
                    && meta.uid() == uid
                    && meta.mode() & 0o077 == 0,
                "unsafe weightfs runtime directory: {}",
                path.display()
            );
        }
        // create_dir, never reuse or hide an existing directory, including PID reuse.
        loop {
            let candidate = path.join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::DirBuilder::new().mode(0o700).create(&candidate) {
                Ok(()) => {
                    path = candidate;
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let entries = model
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned model"))?
            .entries();
        let filesystem = WeightFs {
            model,
            entries,
            handles: HashMap::new(),
            next: 0,
            uid,
            gid: unsafe { libc::getegid() },
        };
        match fuser::Session::new(
            filesystem,
            &path,
            &[
                MountOption::FSName("weightfs".into()),
                MountOption::Subtype("weightfs".into()),
                MountOption::DefaultPermissions,
                MountOption::NoSuid,
                MountOption::NoDev,
                MountOption::NoExec,
            ],
        ) {
            Ok(mut session) => {
                let unmount = session.unmount_callable();
                match std::thread::Builder::new()
                    .name("weightfs".into())
                    .spawn(move || session.run())
                {
                    Ok(thread) => Ok(Self {
                        path,
                        session: Some((unmount, thread)),
                    }),
                    Err(error) => {
                        let _ = fs::remove_dir(&path);
                        Err(error.into())
                    }
                }
            }
            Err(e) => {
                let _ = fs::remove_dir(&path);
                Err(e.into())
            }
        }
    }
}
impl Drop for Mount {
    fn drop(&mut self) {
        if let Some((mut unmount, thread)) = self.session.take() {
            if let Err(error) = unmount.unmount() {
                eprintln!("weightfs unmount failed: {error}");
            }
            match thread.join() {
                Ok(Ok(())) => (),
                Ok(Err(error)) => eprintln!("weightfs worker failed: {error}"),
                Err(_) => eprintln!("weightfs worker panicked"),
            }
        }
        let _ = fs::remove_dir(&self.path);
    }
}

struct Handle {
    ino: u64,
    data: Vec<u8>,
    revision: u64,
    writable: bool,
    dirty: bool,
}
struct WeightFs<T> {
    model: Arc<Mutex<T>>,
    entries: Vec<Entry>,
    handles: HashMap<u64, Handle>,
    next: u64,
    uid: u32,
    gid: u32,
}
impl<T: Weights> WeightFs<T> {
    fn entry(&self, ino: u64) -> std::result::Result<&Entry, i32> {
        ino.checked_sub(3)
            .and_then(|i| self.entries.get(i as usize))
            .ok_or(ENOENT)
    }
    fn snapshot(&self, ino: u64) -> std::result::Result<(Vec<u8>, u64), i32> {
        let mut model = self.model.lock().map_err(|_| EIO)?;
        let revision = model.revision();
        let mut data = if ino == 2 {
            serde_json::to_vec_pretty(&serde_json::json!({"format":"weightlang.weightfs.v1", "pid":std::process::id(), "revision":revision,
                "parameters":self.entries.iter().enumerate().map(|(i,e)| serde_json::json!({"file":format!("{i:06}.json"),"name":e.name,"shape":e.shape,"dtype":"f32","writable":e.writable})).collect::<Vec<_>>() }))
                .map_err(|_| EIO)?
        } else {
            let e = self.entry(ino)?;
            if e.shape.iter().product::<usize>() > MAX_BUFFER / 32 {
                return Err(EFBIG);
            }
            serde_json::to_vec(&model.read(&e.name).map_err(|_| EIO)?).map_err(|_| EIO)?
        };
        data.push(b'\n');
        Ok((data, revision))
    }
    fn attr(&self, ino: u64, size: u64) -> std::result::Result<FileAttr, i32> {
        let (kind, perm) = match ino {
            1 => (FileType::Directory, 0o500),
            2 => (FileType::RegularFile, 0o400),
            _ => (
                FileType::RegularFile,
                if self.entry(ino)?.writable {
                    0o600
                } else {
                    0o400
                },
            ),
        };
        Ok(FileAttr {
            ino,
            size,
            blocks: size.div_ceil(512),
            atime: UNIX_EPOCH,
            mtime: UNIX_EPOCH,
            ctime: UNIX_EPOCH,
            crtime: UNIX_EPOCH,
            kind,
            perm,
            nlink: if ino == 1 { 2 } else { 1 },
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            blksize: 4096,
            flags: 0,
        })
    }
    fn stat(&self, ino: u64, fh: Option<u64>) -> std::result::Result<FileAttr, i32> {
        let size = if ino == 1 {
            0
        } else if let Some(h) = fh.and_then(|h| self.handles.get(&h)) {
            h.data.len()
        } else {
            self.snapshot(ino)?.0.len()
        };
        self.attr(ino, size as u64)
    }
    fn commit(&mut self, fh: u64) -> std::result::Result<(), i32> {
        let h = self.handles.get_mut(&fh).ok_or(EBADF)?;
        if !h.dirty {
            return Ok(());
        }
        let e = &self.entries[(h.ino - 3) as usize];
        let values: Vec<f32> = serde_json::from_slice(&h.data).map_err(|_| EINVAL)?;
        if values.len() != e.shape.iter().product::<usize>()
            || !values.iter().all(|v| v.is_finite())
        {
            return Err(EINVAL);
        }
        let mut model = self.model.lock().map_err(|_| EIO)?;
        if model.revision() != h.revision {
            return Err(EAGAIN);
        }
        model.replace(&e.name, &values).map_err(|_| EINVAL)?;
        h.revision = model.revision();
        h.dirty = false;
        Ok(())
    }
}
impl<T: Weights> Filesystem for WeightFs<T> {
    fn init(&mut self, _: &Request<'_>, config: &mut KernelConfig) -> std::result::Result<(), i32> {
        config
            .add_capabilities(consts::FUSE_ATOMIC_O_TRUNC)
            .map_err(|_| libc::ENOTSUP)
    }
    fn lookup(&mut self, _: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let ino = if parent != 1 {
            None
        } else if name == "manifest.json" {
            Some(2)
        } else {
            self.entries
                .iter()
                .enumerate()
                .find(|(i, _)| name == OsStr::new(&format!("{i:06}.json")))
                .map(|(i, _)| i as u64 + 3)
        };
        match ino.ok_or(ENOENT).and_then(|i| self.stat(i, None)) {
            Ok(a) => reply.entry(&TTL, &a, 0),
            Err(e) => reply.error(e),
        }
    }
    fn getattr(&mut self, _: &Request<'_>, ino: u64, fh: Option<u64>, reply: ReplyAttr) {
        match self.stat(ino, fh) {
            Ok(a) => reply.attr(&TTL, &a),
            Err(e) => reply.error(e),
        }
    }
    fn readdir(
        &mut self,
        _: &Request<'_>,
        ino: u64,
        _: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        if ino != 1 || offset < 0 {
            reply.error(EINVAL);
            return;
        }
        let entries = [
            (1, FileType::Directory, ".".into()),
            (1, FileType::Directory, "..".into()),
            (2, FileType::RegularFile, "manifest.json".into()),
        ]
        .into_iter()
        .chain(
            self.entries
                .iter()
                .enumerate()
                .map(|(i, _)| (i as u64 + 3, FileType::RegularFile, format!("{i:06}.json"))),
        );
        for (i, (ino, kind, name)) in entries.enumerate().skip(offset as usize) {
            if reply.add(ino, (i + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }
    fn open(&mut self, _: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        let writable = flags & libc::O_ACCMODE != libc::O_RDONLY;
        if writable && !self.entry(ino).is_ok_and(|e| e.writable) {
            reply.error(EACCES);
            return;
        }
        if flags & libc::O_APPEND != 0 {
            reply.error(EINVAL);
            return;
        }
        if self.handles.len() >= MAX_OPEN {
            reply.error(ENOSPC);
            return;
        }
        match self.snapshot(ino) {
            Ok((mut data, revision)) => {
                if writable
                    && (flags & libc::O_ACCMODE == libc::O_WRONLY || flags & libc::O_TRUNC != 0)
                {
                    data.clear();
                }
                if self
                    .handles
                    .values()
                    .map(|h| h.data.capacity())
                    .sum::<usize>()
                    + data.capacity()
                    > MAX_BUFFER
                {
                    reply.error(ENOSPC);
                    return;
                }
                self.next += 1;
                self.handles.insert(
                    self.next,
                    Handle {
                        ino,
                        data,
                        revision,
                        writable,
                        dirty: flags & libc::O_TRUNC != 0,
                    },
                );
                reply.opened(self.next, consts::FOPEN_DIRECT_IO); // FOPEN_DIRECT_IO: never serve stale kernel pages.
            }
            Err(e) => reply.error(e),
        }
    }
    fn read(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _: i32,
        _: Option<u64>,
        reply: ReplyData,
    ) {
        match self.handles.get(&fh) {
            Some(h) if offset >= 0 => {
                let start = (offset as usize).min(h.data.len());
                reply.data(&h.data[start..start.saturating_add(size as usize).min(h.data.len())]);
            }
            _ => reply.error(EBADF),
        }
    }
    fn write(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        _: u32,
        _: i32,
        _: Option<u64>,
        reply: ReplyWrite,
    ) {
        let used: usize = self.handles.values().map(|h| h.data.capacity()).sum();
        let Some(h) = self.handles.get_mut(&fh) else {
            reply.error(EBADF);
            return;
        };
        if !h.writable || offset < 0 {
            reply.error(EACCES);
            return;
        }
        let end = (offset as usize).saturating_add(data.len());
        let limit = self.entries[(h.ino - 3) as usize]
            .shape
            .iter()
            .product::<usize>()
            .saturating_mul(32)
            .saturating_add(2)
            .min(MAX_BUFFER);
        if end > limit || used.saturating_add(end.saturating_sub(h.data.capacity())) > MAX_BUFFER {
            reply.error(EFBIG);
            return;
        }
        if end > h.data.len() {
            h.data.reserve_exact(end - h.data.len());
            h.data.resize(end, 0);
        }
        h.data[offset as usize..end].copy_from_slice(data);
        h.dirty = true;
        reply.written(data.len() as u32);
    }
    fn setattr(
        &mut self,
        _: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        _: Option<TimeOrNow>,
        _: Option<TimeOrNow>,
        _: Option<SystemTime>,
        fh: Option<u64>,
        _: Option<SystemTime>,
        _: Option<SystemTime>,
        _: Option<SystemTime>,
        flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        if mode.is_some()
            || uid.is_some()
            || gid.is_some()
            || flags.is_some()
            || !self.entry(ino).is_ok_and(|e| e.writable)
        {
            reply.error(EROFS);
            return;
        }
        if let Some(size) = size {
            if let Some(h) = fh.and_then(|h| self.handles.get_mut(&h)) {
                if !h.writable || size > h.data.len() as u64 {
                    reply.error(EINVAL);
                    return;
                }
                h.data.truncate(size as usize);
                h.dirty = true;
            } else {
                reply.error(EINVAL);
                return;
            }
        }
        match self.attr(ino, size.unwrap_or(0)) {
            Ok(a) => reply.attr(&TTL, &a),
            Err(e) => reply.error(e),
        }
    }
    fn flush(&mut self, _: &Request<'_>, _: u64, fh: u64, _: u64, reply: ReplyEmpty) {
        match self.commit(fh) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(e),
        }
    }
    fn fsync(&mut self, _: &Request<'_>, _: u64, fh: u64, _: bool, reply: ReplyEmpty) {
        match self.commit(fh) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(e),
        }
    }
    fn release(
        &mut self,
        _: &Request<'_>,
        _: u64,
        fh: u64,
        _: i32,
        _: Option<u64>,
        _: bool,
        reply: ReplyEmpty,
    ) {
        self.handles.remove(&fh);
        reply.ok();
    }
}
