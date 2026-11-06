//! Process-local allocator accounting; counters include all service threads.
#[cfg(feature = "allocation-metrics")]
use std::alloc::{GlobalAlloc,Layout,System};
use std::sync::atomic::{AtomicU64,Ordering};
#[cfg(feature = "allocation-metrics")]
pub struct Counting;
static ALLOCATIONS:AtomicU64=AtomicU64::new(0);
static ALLOCATED_BYTES:AtomicU64=AtomicU64::new(0);
#[cfg(feature = "allocation-metrics")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc requires forwarding the caller's valid layout to the system allocator.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        // SAFETY: GlobalAlloc requires the pointer and layout returned by the matching allocation.
        unsafe { System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: GlobalAlloc requires the pointer/layout pair from the prior allocation and requested new size.
        let p = unsafe { System.realloc(p, layout, size) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        p
    }
}
pub fn snapshot()->serde_json::Value{serde_json::json!({"enabled":cfg!(feature="allocation-metrics"),"allocation_calls":ALLOCATIONS.load(Ordering::Relaxed),"allocated_bytes_total":ALLOCATED_BYTES.load(Ordering::Relaxed)})}
