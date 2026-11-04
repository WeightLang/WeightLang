//! Process-local allocator accounting; counters include all service threads.
use std::alloc::{GlobalAlloc,Layout,System};
use std::sync::atomic::{AtomicU64,Ordering};
pub struct Counting;
static ALLOCATIONS:AtomicU64=AtomicU64::new(0);
static ALLOCATED_BYTES:AtomicU64=AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self,layout:Layout)->*mut u8{let p=System.alloc(layout);if !p.is_null(){ALLOCATIONS.fetch_add(1,Ordering::Relaxed);ALLOCATED_BYTES.fetch_add(layout.size() as u64,Ordering::Relaxed);}p}
    unsafe fn dealloc(&self,p:*mut u8,layout:Layout){System.dealloc(p,layout)}
    unsafe fn realloc(&self,p:*mut u8,layout:Layout,size:usize)->*mut u8{let p=System.realloc(p,layout,size);if !p.is_null(){ALLOCATIONS.fetch_add(1,Ordering::Relaxed);ALLOCATED_BYTES.fetch_add(size as u64,Ordering::Relaxed);}p}
}
pub fn snapshot()->serde_json::Value{serde_json::json!({"enabled":cfg!(feature="allocation-metrics"),"allocation_calls":ALLOCATIONS.load(Ordering::Relaxed),"allocated_bytes_total":ALLOCATED_BYTES.load(Ordering::Relaxed)})}
