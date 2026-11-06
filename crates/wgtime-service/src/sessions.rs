use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use wgtime_core::{adaptive::AdaptiveSession, operations::OperationSession, training::Trainer};

pub(super) const MAX_SESSION_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONNECTION_BYTES: usize = 512 * 1024 * 1024;
const MAX_SESSIONS: usize = 64;

pub(super) enum Session {
    Trainer(Trainer),
    Adaptive(AdaptiveSession),
    Operations(OperationSession),
}

struct Entry {
    session: Box<Session>,
    memory: usize,
}

#[derive(Default)]
pub(super) struct Sessions {
    entries: HashMap<u64, Entry>,
    reserved: usize,
}

/// Construction holds exclusive admission without charging failed preparations.
pub(super) struct Admission<'a> {
    sessions: &'a mut Sessions,
    memory: usize,
}

impl Sessions {
    pub(super) fn admit(&mut self, memory: usize) -> Result<Admission<'_>> {
        ensure!(self.entries.len() < MAX_SESSIONS, "session limit exceeded");
        ensure!(
            memory > 0
                && memory <= MAX_SESSION_BYTES
                && self
                    .reserved
                    .checked_add(memory)
                    .is_some_and(|n| n <= MAX_CONNECTION_BYTES),
            "session memory budget exceeded"
        );
        Ok(Admission {
            sessions: self,
            memory,
        })
    }

    pub(super) fn get(&self, handle: &u64) -> Option<&Session> {
        self.entries.get(handle).map(|entry| entry.session.as_ref())
    }

    pub(super) fn get_mut(&mut self, handle: &u64) -> Option<&mut Session> {
        self.entries
            .get_mut(handle)
            .map(|entry| entry.session.as_mut())
    }

    pub(super) fn remove(&mut self, handle: &u64) -> Result<()> {
        let entry = self
            .entries
            .remove(handle)
            .context("unknown session or foreign owner")?;
        self.reserved -= entry.memory;
        Ok(())
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn reserved(&self) -> usize {
        self.reserved
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&u64, &Session)> {
        self.entries
            .iter()
            .map(|(handle, entry)| (handle, entry.session.as_ref()))
    }
}

impl Admission<'_> {
    pub(super) fn insert(self, handle: u64, session: Session) {
        self.sessions.entries.insert(
            handle,
            Entry {
                session: Box::new(session),
                memory: self.memory,
            },
        );
        self.sessions.reserved += self.memory;
    }
}
