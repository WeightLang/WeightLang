use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::operations::OperationSession;

struct CountAlloc;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn measured_call(session: &mut OperationSession, name: &str, inputs: &[Vec<f32>]) -> usize {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = session.call(name, inputs, 10_000);
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert_eq!(result.unwrap(), Some(vec![2., 3.]));
    allocations
}

#[test]
fn repeated_writes_reuse_candidate_storage() {
    let source = format!(
        "wgt 0.5\nmodel Reuse\n\
         # The same two-element state is overwritten; only the final candidate commits.\n\
         state value: f32[2] = [0,0]\n\
         fn once(input: f32[2]) -> f32[2] {{ value <- input; return value; }}\n\
         fn repeated(input: f32[2]) -> f32[2] {{ {} return value; }}\n",
        "value <- input;".repeat(32)
    );
    let ModelPlan::Operations(plan) = ModelDefinition::parse(&source)
        .unwrap()
        .instantiate_plan()
        .unwrap()
    else {
        panic!("expected operations plan")
    };
    let mut session = OperationSession::prepare(plan, 1024 * 1024).unwrap();
    let inputs = [vec![2., 3.]];
    session.call("once", &inputs, 10_000).unwrap();
    session.call("repeated", &inputs, 10_000).unwrap();
    let once = measured_call(&mut session, "once", &inputs);
    let repeated = measured_call(&mut session, "repeated", &inputs);
    eprintln!("operation allocations: once={once}, repeated={repeated}");
    assert!(
        repeated <= once,
        "repeated writes allocated {repeated} times, single write {once}"
    );
    assert_eq!(session.state_values()[0].2, vec![2., 3.]);
}
