use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd};
use wgtime::buffer::{Mapping, WritableTensor};
#[test]
fn committed_tensor_is_immutable_and_unsealed_input_is_rejected() -> Result<()> {
    let raw = unsafe { libc::memfd_create(c"wgtime-test".as_ptr(), libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC) };
    assert!(raw >= 0);
    let file = unsafe { File::from_raw_fd(raw) };
    file.set_len(8)?;
    assert!(Mapping::open(file, 2).is_err());
    let mut writable = WritableTensor::new(2)?;
    writable.write(&[1., 2.])?;
    let sealed = writable.commit()?;
    let (sender, receiver) = std::os::unix::net::UnixStream::pair()?;
    sealed.send(&sender)?;
    let received = wgtime::buffer::receive_file(&receiver)?;
    let mut retained = received.try_clone()?;
    assert!(retained.write_all(&[0; 8]).is_err());
    assert_eq!(unsafe { libc::ftruncate(retained.as_raw_fd(), 4) }, -1);
    let mapping = Mapping::open(received, 2)?;
    assert_eq!(mapping.values(), &[1., 2.]);
    assert!(WritableTensor::new(usize::MAX).is_err());
    assert!(WritableTensor::new(65536).is_err());
    Ok(())
}
