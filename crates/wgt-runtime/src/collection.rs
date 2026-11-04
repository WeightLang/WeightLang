//! Runtime-only protection of a live model while a batch is collected.
//! Tokens never enter model artifacts, optimizer snapshots or checkpoints.
use anyhow::{ensure, Result};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[derive(Debug, Default)]
pub struct CollectionState(Arc<AtomicBool>);

#[derive(Debug)]
pub struct CollectionToken(Arc<AtomicBool>);
impl CollectionState {
    pub fn begin(&self) -> Result<CollectionToken> {
        ensure!(self.0.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_ok(),
            "model is already being collected; nested collection of the same model is unsupported");
        Ok(CollectionToken(self.0.clone()))
    }
    pub fn ensure_mutable(&self) -> Result<()> {
        ensure!(!self.0.load(Ordering::Acquire), "model mutation is forbidden while collecting its batch");
        Ok(())
    }
}
impl Drop for CollectionToken {
    fn drop(&mut self) { self.0.store(false,Ordering::Release); }
}
