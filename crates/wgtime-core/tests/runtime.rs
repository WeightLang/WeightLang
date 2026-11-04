use std::{
    alloc::{GlobalAlloc, Layout, System},
    path::Path,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use wgtime_core::{
    execution::{AllocationPolicy, ExecutionSession},
    Model, Session,
};

struct CountAlloc;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
// SAFETY: every allocation operation forwards its original pointer/layout to System.
// Instrumentation uses atomics only, never recursive allocation.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn model() -> Model {
    serde_json::from_str(
        r#"{"version":1,"input":2,"parameters":[
        {"name":"base","shape":[2,2],"kind":"resident","data":[1,2,3,4]},
        {"name":"delta","shape":[2,2],"kind":"resident","data":[0.5,0,0,-0.5]},
        {"name":"effective","shape":[2,2],"kind":"add","base":"base","delta":"delta"},
        {"name":"bias","shape":[2],"kind":"view","source":"base","start":1}
    ],"layers":[{"op":"linear","weight":"effective","bias":"bias"},{"op":"relu"}]}"#,
    )
    .unwrap()
}

// One test owns the allocator counter, so no parallel tests can pollute it.
#[test]
fn contracts_streaming_and_allocation_free_execution() {
    let mut session = Session::prepare(model(), Path::new("."), 72).unwrap();
    assert_eq!(session.report().managed_bytes, 72);
    assert!(Session::prepare(model(), Path::new("."), 71).is_err());
    assert!(session.run(&[1.]).is_err());
    let mut wrong = model();
    wrong.parameters[1].shape = vec![4];
    assert!(Session::prepare(wrong, Path::new("."), 1000).is_err());
    let mut wrong = model();
    wrong.parameters[3].value = wgtime_core::Value::View {
        source: "base".into(),
        start: 3,
    };
    assert!(Session::prepare(wrong, Path::new("."), 1000).is_err());
    let mut wrong = model();
    wrong.parameters[2].value = wgtime_core::Value::Add {
        base: "effective".into(),
        delta: "delta".into(),
    };
    assert!(Session::prepare(wrong, Path::new("."), 1000).is_err());

    let directory = std::env::temp_dir()
        .join("weightlang")
        .join(format!("native-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("weights.f32");
    let raw: Vec<u8> = [1f32, 2., 3., 4.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    std::fs::write(&path, raw).unwrap();
    let mut streamed_model = model();
    streamed_model.parameters[0].value = wgtime_core::Value::File {
        path: path.clone(),
        offset: 0,
    };
    let mut streamed = Session::prepare(streamed_model, Path::new("."), 56).unwrap();
    assert_eq!(streamed.report().managed_bytes, 56);
    let mut strict = ExecutionSession::prepare(
        model(),
        Path::new("."),
        72,
        "auto",
        AllocationPolicy::Strict,
    )
    .unwrap();
    let packed_model: Model = serde_json::from_str(r#"{"version":1,"input":2,"parameters":[
        {"name":"w","shape":[1,2],"kind":"packed","storage":"int4","group":2,"data":[169],"scales":[1]}
    ],"layers":[{"op":"linear","weight":"w"}]}"#).unwrap();
    let mut packed = Session::prepare(packed_model, Path::new("."), 100).unwrap();
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..100 {
        let mut effective = [0.; 4];
        session
            .read_parameter("effective", 0, &mut effective)
            .unwrap();
        assert_eq!(effective, [1.5, 2., 3., 3.5]);
        session
            .write_parameters(&[wgtime_core::ParameterWrite {
                name: "delta",
                start: 0,
                values: &[0.5, 0., 0., -0.5],
            }])
            .unwrap();
        assert_eq!(session.run(&[2., 1.]).unwrap(), &[7., 12.5]);
        assert_eq!(streamed.run(&[2., 1.]).unwrap(), &[7., 12.5]);
        assert_eq!(strict.run(&[2., 1.]).unwrap(), &[7., 12.5]);
        assert_eq!(packed.run(&[2., 1.]).unwrap(), &[4.]);
    }
    COUNTING.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
    std::fs::write(&path, []).unwrap();
    assert_eq!(
        streamed.run(&[2., 1.]).unwrap_err(),
        "parameter read failed"
    );
    drop(streamed);
    std::fs::remove_dir_all(directory).unwrap();
}
