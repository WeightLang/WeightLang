//! Descriptor-backed artifacts share one byte bound and staging contract.
use super::{Connection, MAX_SESSION_BYTES, Session, Shared};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs::File,
    io::{Read, Seek, Write},
};
use wgtime::Request;
use wgtime_core::training::{Optimizer, Trainer};

const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

impl Connection {
    pub(super) fn execute_persistence(
        &mut self,
        request: Request,
        mut file: File,
        shared: &Shared,
    ) -> Result<serde_json::Value> {
        ensure!(
            self.ready && file.metadata()?.is_file(),
            "invalid persistence descriptor"
        );
        match request {
            Request::InitializeWeightsFd { source, memory, optimizer, revision } => {
                let bytes = read_payload(&mut file, "weights")?;
                if let Some(expected) = revision {
                    use sha2::{Digest, Sha256};
                    ensure!(expected == format!("{:x}", Sha256::digest(&bytes)), "pinned weight revision mismatch");
                }
                self.initialize_definition(source, memory, optimizer, Some(&bytes), shared)
            }
            Request::ExportWeightsFd { handle } => {
                write_payload(&mut file, || match self.sessions.get_mut(&handle) {
                    Some(Session::Operations(session)) => session.weights_bytes(MAX_ARTIFACT_BYTES),
                    Some(Session::Adaptive(session)) => session.weights_bytes(MAX_ARTIFACT_BYTES),
                    Some(Session::Trainer(trainer)) => trainer.weights_bytes(MAX_ARTIFACT_BYTES),
                    _ => bail!("unknown session or foreign owner"),
                })
            }
            Request::BindWeightsFd { handle, optimizer } => {
                let bytes = read_payload(&mut file, "weights")?;
                match self.sessions.get_mut(&handle) {
                    Some(Session::Operations(session)) => {
                        session.bind_weights(&bytes, &optimizer)?
                    }
                    Some(Session::Adaptive(session)) => session.bind_weights(&bytes, &optimizer)?,
                    Some(Session::Trainer(trainer)) => trainer.bind_weights(&bytes, &optimizer)?,
                    _ => bail!("unknown session or foreign owner"),
                }
                Ok(serde_json::json!({}))
            }
            Request::PersistFd { handle, checkpoint } => {
                write_payload(&mut file, || match self.sessions.get_mut(&handle) {
                    Some(Session::Adaptive(session)) => {
                        if checkpoint {
                            session.checkpoint_bytes_bounded(MAX_ARTIFACT_BYTES)
                        } else {
                            Ok(serde_json::to_vec(session.program())?)
                        }
                    }
                    Some(Session::Trainer(trainer)) => {
                        if checkpoint {
                            trainer.checkpoint_bytes_bounded(MAX_ARTIFACT_BYTES)
                        } else {
                            Ok(serde_json::to_vec(&trainer.snapshot()?)?)
                        }
                    }
                    _ => bail!("unknown session or foreign owner"),
                })
            }
            Request::InspectArtifactFd { checkpoint, memory } => {
                ensure!(
                    memory > 0 && memory <= MAX_SESSION_BYTES,
                    "inspection memory budget exceeded"
                );
                let bytes = read_payload(&mut file, "artifact")?;
                let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                let trainer = if checkpoint {
                    Trainer::restore_self_contained_value(value, memory)?
                } else {
                    Trainer::new_self_contained(
                        serde_json::from_value(value)?,
                        memory,
                        Optimizer::Sgd,
                    )?
                };
                let report = trainer.session.report();
                Ok(
                    serde_json::json!({"input":report.input,"output":report.output,"steps":trainer.steps()}),
                )
            }
            Request::LoadFd {
                checkpoint,
                optimizer,
                memory,
            } => {
                let admission = self
                    .sessions
                    .admit(memory)
                    .context("load admission budget exceeded")?;
                let bytes = read_payload(&mut file, "checkpoint")?;
                let value: serde_json::Value = serde_json::from_slice(&bytes)?;
                let artifact_checkpoint = match value.get("format") {
                    None => false,
                    Some(tag)
                        if matches!(
                            tag.as_str(),
                            Some(
                                "weightlang.forward-training.v1"
                                    | "weightlang.forward-training.v2"
                                    | "weightlang.native-training.v1"
                            )
                        ) =>
                    {
                        true
                    }
                    Some(_) => bail!("unsupported checkpoint format"),
                };
                ensure!(
                    checkpoint == artifact_checkpoint,
                    "load format does not match model/checkpoint kind"
                );
                let handle = shared.next_handle()?;
                if value
                    .get("format")
                    .and_then(|n| n.as_str())
                    .is_some_and(|n| n.starts_with("weightlang.forward"))
                    || (!checkpoint && value.get("instructions").is_some())
                {
                    let mut session = if checkpoint {
                        wgtime_core::adaptive::AdaptiveSession::restore_checkpoint(value, memory)?
                    } else {
                        wgtime_core::adaptive::AdaptiveSession::prepare(
                            serde_json::from_value(value)?,
                            memory,
                        )?
                    };
                    if !checkpoint {
                        match optimizer.as_str() {
                            "none" => {}
                            "adam" => session.set_optimizer(Optimizer::Adam)?,
                            "sgd" => session.set_optimizer(Optimizer::Sgd)?,
                            _ => bail!("unsupported optimizer"),
                        }
                    }
                    admission.insert(handle, Session::Adaptive(session));
                } else {
                    let trainer = if checkpoint {
                        Trainer::restore_self_contained_value(value, memory)?
                    } else {
                        let optimizer = match optimizer.as_str() {
                            "adam" => Optimizer::Adam,
                            "sgd" => Optimizer::Sgd,
                            _ => bail!("unsupported optimizer"),
                        };
                        Trainer::new_self_contained(
                            serde_json::from_value(value)?,
                            memory,
                            optimizer,
                        )?
                    };
                    admission.insert(handle, Session::Trainer(trainer));
                }
                Ok(serde_json::json!({"handle":handle}))
            }
            _ => unreachable!(),
        }
    }
}

fn read_payload(file: &mut File, kind: &str) -> Result<Vec<u8>> {
    ensure!(
        file.metadata()?.len() <= MAX_ARTIFACT_BYTES as u64,
        "{kind} byte limit exceeded"
    );
    file.rewind()?;
    let mut bytes = Vec::new();
    file.take(MAX_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_ARTIFACT_BYTES,
        "{kind} byte limit exceeded"
    );
    Ok(bytes)
}

fn write_payload(
    file: &mut File,
    payload: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<serde_json::Value> {
    ensure!(
        file.metadata()?.len() == 0,
        "destination must be empty staging file"
    );
    let bytes = payload()?;
    ensure!(
        bytes.len() <= MAX_ARTIFACT_BYTES,
        "persistence byte limit exceeded"
    );
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(serde_json::json!({"bytes": bytes.len()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::FromRawFd;

    fn file(bytes: &[u8]) -> Result<File> {
        let fd =
            unsafe { libc::memfd_create(c"wgtime-persistence-test".as_ptr(), libc::MFD_CLOEXEC) };
        ensure!(
            fd >= 0,
            "test descriptor creation failed: {}",
            std::io::Error::last_os_error()
        );
        let mut file = unsafe { File::from_raw_fd(fd) };
        file.write_all(bytes)?;
        Ok(file)
    }

    #[test]
    fn payload_reads_rewind_and_reject_oversized_files() -> Result<()> {
        let mut file = file(b"complete artifact")?;
        assert_eq!(read_payload(&mut file, "artifact")?, b"complete artifact");
        file.set_len(64 * 1024 * 1024 + 1)?;
        for kind in ["artifact", "weights", "checkpoint"] {
            let error = read_payload(&mut file, kind).unwrap_err();
            assert_eq!(error.to_string(), format!("{kind} byte limit exceeded"));
        }
        Ok(())
    }

    #[test]
    fn payload_writes_require_empty_staging_and_preserve_failed_destinations() -> Result<()> {
        let mut file = file(&[])?;
        assert!(write_payload(&mut file, || bail!("serialization failed")).is_err());
        assert_eq!(file.metadata()?.len(), 0);
        let result = write_payload(&mut file, || Ok(b"complete artifact".to_vec()))?;
        assert_eq!(result["bytes"], 17);
        assert_eq!(read_payload(&mut file, "artifact")?, b"complete artifact");
        assert!(
            write_payload(&mut file, || panic!("nonempty staging reached serializer")).is_err()
        );
        assert_eq!(read_payload(&mut file, "artifact")?, b"complete artifact");
        Ok(())
    }
}
