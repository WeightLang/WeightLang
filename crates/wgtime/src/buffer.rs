//! Linux immutable small shared tensors. Large datasets must use disk-backed storage.
use anyhow::{ensure, Context, Result};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::ptr::NonNull;

pub const MAX_SHARED_BYTES: usize = 64 * 1024;
const SEALS: i32 = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
/// No writable mappings are exposed. Committing consumes the only write API.
pub struct WritableTensor { file: File, count: usize }
pub struct SealedTensor { file: File, pub count: usize }
impl WritableTensor {
    pub fn new(count: usize) -> Result<Self> {
        let bytes = count.checked_mul(4).context("tensor length overflow")?;
        ensure!(bytes > 0 && bytes <= MAX_SHARED_BYTES, "small shared tensor limit exceeded");
        let fd = unsafe { libc::memfd_create(c"wgtime-input".as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
        ensure!(fd >= 0, "cannot create shared tensor: {}", std::io::Error::last_os_error());
        let file = unsafe { File::from_raw_fd(fd) };
        file.set_len(bytes as u64)?;
        Ok(Self { file, count })
    }
    pub fn write(&mut self, data: &[f32]) -> Result<()> {
        ensure!(data.len() == self.count && data.iter().all(|n| n.is_finite()), "invalid tensor data");
        self.file.seek(SeekFrom::Start(0))?;
        // Native f32 is valid only within this local same-architecture service.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.file.write_all(bytes)?;
        Ok(())
    }
    pub fn commit(self) -> Result<SealedTensor> {
        ensure!(unsafe { libc::fcntl(self.file.as_raw_fd(), libc::F_ADD_SEALS, SEALS) } == 0, "cannot seal shared tensor: {}", std::io::Error::last_os_error());
        Ok(SealedTensor { file: self.file, count: self.count })
    }
}
impl SealedTensor {
    pub fn send(&self, stream: &UnixStream) -> Result<()> { send_fd(stream, self.file.as_raw_fd()) }
}
fn send_fd(stream: &UnixStream, fd: RawFd) -> Result<()> {
    let mut byte = [0u8];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    let mut control = [0usize; 8];
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as usize;
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as usize;
        std::ptr::write_unaligned(libc::CMSG_DATA(header).cast::<RawFd>(), fd);
    }
    ensure!(unsafe { libc::sendmsg(stream.as_raw_fd(), &message, libc::MSG_NOSIGNAL) } == 1, "cannot send buffer descriptor");
    Ok(())
}
pub fn receive_file(stream: &UnixStream) -> Result<File> {
    let mut byte = [0u8];
    let mut iov = libc::iovec { iov_base: byte.as_mut_ptr().cast(), iov_len: 1 };
    let mut control = [0usize; 8];
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = std::mem::size_of_val(&control);
    let read = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut message, libc::MSG_CMSG_CLOEXEC) };
    ensure!(read == 1, "missing buffer descriptor");
    let mut files = Vec::new();
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&message);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let base = libc::CMSG_LEN(0) as usize;
                let count = (*header).cmsg_len.saturating_sub(base) / std::mem::size_of::<RawFd>();
                for i in 0..count {
                    let fd = std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<RawFd>().add(i));
                    files.push(File::from_raw_fd(fd));
                }
            }
            header = libc::CMSG_NXTHDR(&message, header);
        }
    }
    ensure!(message.msg_flags & libc::MSG_CTRUNC == 0 && files.len() == 1, "invalid buffer descriptors");
    Ok(files.pop().unwrap())
}
/// Mapping remains immutable even if the untrusted sender retains the original FD.
pub struct Mapping { pointer: NonNull<f32>, count: usize, _file: File }
impl Mapping {
    pub fn open(file: File, count: usize) -> Result<Self> {
        let bytes = count.checked_mul(4).context("tensor length overflow")?;
        ensure!(bytes > 0 && bytes <= MAX_SHARED_BYTES && file.metadata()?.len() == bytes as u64, "invalid tensor extent");
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        ensure!(seals >= 0 && seals & SEALS == SEALS, "tensor must be irreversibly sealed before submission");
        let raw = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        ensure!(raw != libc::MAP_FAILED, "cannot map tensor");
        let mapping = Self { pointer: NonNull::new(raw.cast()).context("null mapping")?, count, _file: file };
        ensure!(mapping.values().iter().all(|n| n.is_finite()), "nonfinite shared tensor");
        Ok(mapping)
    }
    pub fn values(&self) -> &[f32] { unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.count) } }
}
impl Drop for Mapping {
    fn drop(&mut self) { unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.count * 4); } }
}
