//! Native structured-task substrate. Not yet exposed as WGT syntax.
//!
//! Futures are cold until spawned and driven by `TaskScope::finish`. The executor
//! polls local futures on its caller thread; it never requires Runtime or Value
//! to be Send. Blocking work uses a separate fixed-size pool with bounded queue.
//! Dropping a scope cancels and drops its futures; dropping a pool joins workers.
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

static NEXT_SCOPE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskError {
    pub source: String,
    pub message: String,
}
impl TaskError {
    pub fn new(source: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            message: message.into(),
        }
    }
}
impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.source, self.message)
    }
}
impl std::error::Error for TaskError {}

#[derive(Clone)]
pub struct TaskContext {
    cancelled: Arc<AtomicBool>,
    remaining: Arc<AtomicU64>,
    in_flight: Arc<(Mutex<usize>, Condvar)>,
    deadline: Instant,
    source: String,
    seed: u64,
}
impl TaskContext {
    /// Charge before performing work. Failed/cancelled work is never refunded.
    pub fn charge(&self, units: u64) -> Result<(), TaskError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(TaskError::new(&self.source, "task cancelled"));
        }
        if Instant::now() >= self.deadline {
            return Err(TaskError::new(&self.source, "task deadline exceeded"));
        }
        self.remaining
            .try_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                left.checked_sub(units)
            })
            .map(|_| ())
            .map_err(|_| TaskError::new(&self.source, "task work budget exhausted"))
    }
    pub(crate) fn refund_reserved(&self, units: u64) { self.remaining.fetch_add(units, Ordering::AcqRel); }
    fn wait_for_io(&self) {
        let (count, done) = &*self.in_flight;
        let mut count = count.lock().unwrap();
        while *count != 0 {
            count = done.wait(count).unwrap();
        }
    }
    /// Seed derives from scope seed and spawn ordinal, never poll order.
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskId {
    scope: u64,
    index: usize,
}

type LocalTask<T> = Pin<Box<dyn Future<Output = Result<T, TaskError>>>>;
struct Entry<T> {
    future: Option<LocalTask<T>>,
    result: Option<Result<T, TaskError>>,
    source: String,
}

/// Limits bound retained tasks (including completed results), polls, and wall time.
/// Admission fails with backpressure rather than allocating an unbounded queue.
/// A language adapter may await/drain and retry admission explicitly.
pub struct TaskScope<T> {
    id: u64,
    tasks: Vec<Entry<T>>,
    limit: usize,
    context: TaskContext,
    finished: bool,
    failures: Vec<TaskError>,
}
struct Notify(thread::Thread);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
impl<T> TaskScope<T> {
    pub fn new(limit: usize, work: u64, timeout: Duration, seed: u64) -> Result<Self, TaskError> {
        if limit == 0 || limit > 10_000 {
            return Err(TaskError::new("tasks", "task limit must be in 1..10000"));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| TaskError::new("tasks", "deadline overflow"))?;
        Ok(Self {
            id: NEXT_SCOPE.fetch_add(1, Ordering::Relaxed),
            tasks: Vec::new(),
            limit,
            context: TaskContext {
                cancelled: Arc::new(AtomicBool::new(false)),
                remaining: Arc::new(AtomicU64::new(work)),
                in_flight: Arc::new((Mutex::new(0), Condvar::new())),
                deadline,
                source: "tasks".into(),
                seed,
            },
            finished: false,
            failures: Vec::new(),
        })
    }
    pub fn spawn<F, Fut>(&mut self, source: impl Into<String>, make: F) -> Result<TaskId, TaskError>
    where
        F: FnOnce(TaskContext) -> Fut,
        Fut: Future<Output = Result<T, TaskError>> + 'static,
    {
        let source = source.into();
        if self.finished || self.context.is_cancelled() {
            return Err(TaskError::new(source, "task scope is closed"));
        }
        if self.tasks.len() == self.limit {
            return Err(TaskError::new(
                source,
                "task capacity exhausted; await or leave scope",
            ));
        }
        self.context.charge(1)?;
        let index = self.tasks.len();
        let mut context = self.context.clone();
        context.source = source.clone();
        context.seed = split_seed(context.seed, index as u64);
        self.tasks.push(Entry {
            future: Some(Box::pin(make(context))),
            result: None,
            source,
        });
        Ok(TaskId {
            scope: self.id,
            index,
        })
    }
    pub fn charge(&self, units: u64) -> Result<(), TaskError> {
        self.context.charge(units)
    }
    #[cfg(test)]
    pub(crate) fn test_context(&self) -> TaskContext { self.context.clone() }
    /// Join all children. Any failure cancels siblings and retains every observed
    /// error. Pending futures are dropped before this returns, running their guards.
    pub fn finish(&mut self) -> Result<(), Vec<TaskError>> { self.drive(None) }
    pub fn wait(&mut self, id: TaskId) -> Result<(), Vec<TaskError>> {
        if id.scope != self.id || id.index >= self.tasks.len() {
            return Err(vec![TaskError::new("tasks", "task belongs to another scope")]);
        }
        self.drive(Some(id.index))
    }
    fn drive(&mut self, target: Option<usize>) -> Result<(), Vec<TaskError>> {
        if self.finished {
            return self.errors();
        }
        let waker = Waker::from(Arc::new(Notify(thread::current())));
        let mut cx = Context::from_waker(&waker);
        loop {
            let mut pending = false;
            let mut failed = false;
            if target.is_some_and(|index| self.tasks[index].future.is_none()) { return self.errors(); }
            for entry in &mut self.tasks {
                let Some(future) = entry.future.as_mut() else {
                    continue;
                };
                if let Err(mut error) = self.context.charge(0) {
                    error.source = entry.source.clone();
                    entry.result = Some(Err(error));
                    entry.future = None;
                    failed = true;
                    continue;
                }
                // A panic must not abandon siblings or escape as an unobserved
                // worker failure. Local task destructors run on this same thread.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    future.as_mut().poll(&mut cx)
                }));
                match outcome {
                    Ok(Poll::Pending) => pending = true,
                    Ok(Poll::Ready(result)) => {
                        failed |= result.is_err();
                        entry.result = Some(
                            result
                                .map_err(|error| TaskError::new(&entry.source, error.to_string())),
                        );
                        entry.future = None;
                    }
                    Err(_) => {
                        entry.result = Some(Err(TaskError::new(&entry.source, "task panicked")));
                        entry.future = None;
                        failed = true;
                    }
                }
            }
            if failed {
                self.cancel();
                break;
            }
            if target.is_some_and(|index| self.tasks[index].future.is_none()) { return self.errors(); }
            if !pending {
                self.finished = true;
                break;
            }
            // Wake tokens are retained by Thread::unpark across the poll/park race.
            // Timeout also makes deadlines observable for broken/non-waking futures.
            thread::park_timeout(Duration::from_millis(1));
        }
        self.context.wait_for_io();
        self.failures = self
            .tasks
            .iter()
            .filter_map(|entry| match &entry.result {
                Some(Err(error)) => Some(error.clone()),
                _ => None,
            })
            .collect();
        self.errors()
    }
    fn errors(&self) -> Result<(), Vec<TaskError>> {
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(self.failures.clone())
        }
    }
    pub fn cancel(&mut self) {
        if self.finished {
            return;
        }
        self.context.cancelled.store(true, Ordering::Release);
        for entry in &mut self.tasks {
            if entry.future.take().is_some() {
                entry.result = Some(Err(TaskError::new(&entry.source, "task cancelled")));
            }
        }
        self.context.wait_for_io();
        self.failures = self
            .tasks
            .iter()
            .filter_map(|entry| match &entry.result {
                Some(Err(error)) => Some(error.clone()),
                _ => None,
            })
            .collect();
        self.finished = true;
    }
    pub fn take(&mut self, id: TaskId) -> Result<T, TaskError> {
        if id.scope != self.id {
            return Err(TaskError::new("tasks", "task belongs to another scope"));
        }
        self.tasks
            .get_mut(id.index)
            .and_then(|entry| entry.result.take())
            .unwrap_or_else(|| {
                Err(TaskError::new(
                    "tasks",
                    "task is pending or result already consumed",
                ))
            })
    }
    pub fn remaining(&self) -> u64 {
        self.context.remaining.load(Ordering::Acquire)
    }
}
impl<T> Drop for TaskScope<T> {
    fn drop(&mut self) {
        self.cancel();
    }
}
fn split_seed(seed: u64, ordinal: u64) -> u64 {
    let mut z = seed.wrapping_add(ordinal.wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

struct InFlight(Arc<(Mutex<usize>, Condvar)>);
impl Drop for InFlight {
    fn drop(&mut self) {
        let (count, done) = &*self.0;
        *count.lock().unwrap() -= 1;
        done.notify_all();
    }
}
type Job = Box<dyn FnOnce() + Send>;
/// Fixed worker count and bounded waiting queue; no thread-per-operation fallback.
/// Running OS calls cannot be interrupted. Drop closes admission and joins all jobs.
pub struct BlockingPool {
    sender: Option<mpsc::SyncSender<Job>>,
    workers: Vec<JoinHandle<()>>,
}
struct Completion<T> {
    result: Option<Result<T, TaskError>>,
    waker: Option<Waker>,
}
pub struct BlockingTask<T>(Arc<Mutex<Completion<T>>>);
impl<T> Future for BlockingTask<T> {
    type Output = Result<T, TaskError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.0.lock().unwrap();
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
impl BlockingPool {
    pub fn new(workers: usize, queue: usize) -> Result<Self, TaskError> {
        if !(1..=64).contains(&workers) || !(1..=10_000).contains(&queue) {
            return Err(TaskError::new(
                "blocking IO",
                "workers must be 1..64 and queue 1..10000",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Job>(queue);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut pool = Self {
            sender: Some(sender),
            workers: Vec::new(),
        };
        for _ in 0..workers {
            let receiver = receiver.clone();
            match thread::Builder::new()
                .name("wgt-blocking-io".into())
                .spawn(move || loop {
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                }) {
                Ok(worker) => pool.workers.push(worker),
                Err(error) => return Err(TaskError::new("blocking IO", error.to_string())),
            }
        }
        Ok(pool)
    }
    pub fn submit<T, F>(&self, context: TaskContext, job: F) -> Result<BlockingTask<T>, TaskError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, TaskError> + Send + 'static,
    {
        context.charge(1)?;
        let state = Arc::new(Mutex::new(Completion {
            result: None,
            waker: None,
        }));
        let completed = state.clone();
        let source = context.source.clone();
        *context.in_flight.0.lock().unwrap() += 1;
        let in_flight = InFlight(context.in_flight.clone());
        let work = Box::new(move || {
            let _in_flight = in_flight;
            let result = context.charge(1).and_then(|_| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or_else(|_| {
                    Err(TaskError::new(
                        &context.source,
                        "blocking operation panicked",
                    ))
                })
            });
            let mut state = completed.lock().unwrap();
            state.result = Some(result);
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
        });
        self.sender
            .as_ref()
            .unwrap()
            .try_send(work)
            .map_err(|_| TaskError::new(source, "blocking IO queue exhausted or closed"))?;
        Ok(BlockingTask(state))
    }
}
impl Drop for BlockingPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    fn scope(limit: usize) -> TaskScope<u64> {
        TaskScope::new(limit, 100_000, Duration::from_secs(5), 42).unwrap()
    }
    #[test]
    fn cold_until_driven_and_results_scoped() {
        let polled = Arc::new(AtomicBool::new(false));
        let mut s = scope(1);
        let p = polled.clone();
        let id = s
            .spawn("example:3", |_| async move {
                p.store(true, Ordering::SeqCst);
                Ok(7)
            })
            .unwrap();
        assert!(!polled.load(Ordering::SeqCst));
        assert!(s.spawn("overflow", |_| async { Ok(0) }).is_err());
        s.finish().unwrap();
        assert_eq!(s.take(id).unwrap(), 7);
        assert!(scope(1).take(id).is_err());
        assert!(s.take(id).is_err());
    }
    #[test]
    fn blocking_work_really_overlaps() {
        let pool = Arc::new(BlockingPool::new(2, 2).unwrap());
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut s = scope(2);
        for _ in 0..2 {
            let (pool, active, peak, barrier) =
                (pool.clone(), active.clone(), peak.clone(), barrier.clone());
            s.spawn("overlap", move |cx| async move {
                pool.submit(cx, move || {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    barrier.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(1)
                })?
                .await
            })
            .unwrap();
        }
        s.finish().unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        drop(s);
        drop(pool);
    }
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn failure_cancels_siblings_and_drops_captures() {
        let drops = Arc::new(AtomicUsize::new(0));
        let guard = Guard(drops.clone());
        let mut s = scope(3);
        s.spawn("pending:4", |_| async move {
            let _guard = guard;
            std::future::pending::<()>().await;
            Ok(0)
        })
        .unwrap();
        s.spawn("bad:9", |_| async { Err(TaskError::new("io", "broken")) })
            .unwrap();
        let errors = s.finish().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.source == "bad:9" && e.message.contains("broken")));
        assert!(errors
            .iter()
            .any(|e| e.source == "pending:4" && e.message.contains("cancelled")));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn drop_cancels_cold_tasks_and_preserves_local_non_send_values() {
        let drops = Arc::new(AtomicUsize::new(0));
        let guard = Guard(drops.clone());
        let local = std::rc::Rc::new(9);
        let mut s = scope(1);
        s.spawn("local", |_| async move {
            let _guard = guard;
            Ok(*local)
        })
        .unwrap();
        drop(s);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn deterministic_seed_and_budget_are_not_refunded() {
        let mut seeds = Vec::new();
        for _ in 0..2 {
            let mut s = scope(2);
            let a = s
                .spawn("a", |cx| async move {
                    cx.charge(7)?;
                    Ok(cx.seed())
                })
                .unwrap();
            let b = s.spawn("b", |cx| async move { Ok(cx.seed()) }).unwrap();
            s.finish().unwrap();
            seeds.push((s.take(a).unwrap(), s.take(b).unwrap()));
            assert_eq!(s.remaining(), 100_000 - 2 - 7);
        }
        assert_eq!(seeds[0], seeds[1]);
        assert_ne!(seeds[0].0, seeds[0].1);
        let mut s = TaskScope::new(1, 2, Duration::from_secs(1), 0).unwrap();
        s.spawn("budget", |cx| async move {
            cx.charge(10)?;
            Ok(0_u64)
        })
        .unwrap();
        assert!(s.finish().unwrap_err()[0].message.contains("budget"));
        assert_eq!(s.remaining(), 1);
    }
    #[test]
    fn awaiting_short_target_returns_before_long_sibling_and_exit_joins() {
        let pool=Arc::new(BlockingPool::new(1,2).unwrap());
        let (release_tx,release_rx)=mpsc::channel();
        let (started_tx,started_rx)=mpsc::channel();
        let cleaned=Arc::new(AtomicUsize::new(0));
        let guard=Guard(cleaned.clone());
        let mut s=scope(3);
        let long=s.spawn("long",move |cx| async move {
            pool.submit(cx,move || {
                let _guard=guard;
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(9)
            })?.await
        }).unwrap();
        let short=s.spawn("short",|_|async{Ok(1)}).unwrap();
        s.wait(short).unwrap();
        assert_eq!(s.take(short).unwrap(),1);
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(cleaned.load(Ordering::SeqCst),0,"await incorrectly joined long sibling");
        // Parent progress supplies the signal the long child needs; waiting for
        // every sibling in await would deadlock/fail before reaching this send.
        release_tx.send(()).unwrap();
        s.finish().unwrap();
        assert_eq!(s.take(long).unwrap(),9);
        assert_eq!(cleaned.load(Ordering::SeqCst),1);
    }
    #[test]
    fn deadline_and_panics_propagate() {
        let mut s = TaskScope::<u64>::new(1, 100_000, Duration::from_millis(10), 0).unwrap();
        s.spawn("waiting", |_| std::future::pending()).unwrap();
        assert!(s.finish().unwrap_err()[0].message.contains("deadline"));
        let mut s = scope(1);
        s.spawn("panic:8", |_| async { panic!("expected test panic") })
            .unwrap();
        assert!(s.finish().unwrap_err()[0].message.contains("panicked"));
    }
    #[test]
    fn consumed_failure_stays_visible_at_scope_exit() {
        let mut s = scope(1);
        let id = s
            .spawn("bad", |_| async { Err(TaskError::new("bad", "failure")) })
            .unwrap();
        let errors = s.finish().unwrap_err();
        assert!(s.take(id).is_err());
        assert_eq!(s.finish().unwrap_err(), errors);
    }
    #[test]
    fn cancellation_waits_for_in_flight_cleanup() {
        let pool = Arc::new(BlockingPool::new(1, 1).unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let cleaned = Arc::new(AtomicUsize::new(0));
        let guard = Guard(cleaned.clone());
        let mut s = scope(2);
        s.spawn("io", move |cx| async move {
            pool.submit(cx, move || {
                let _guard = guard;
                started_tx.send(()).unwrap();
                thread::sleep(Duration::from_millis(20));
                Ok(1)
            })?
            .await
        })
        .unwrap();
        s.spawn("failure", |_| async move {
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            Err(TaskError::new("failure", "cancel siblings"))
        })
        .unwrap();
        assert!(s.finish().is_err());
        assert_eq!(cleaned.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn nested_scope_drop_cancels_captured_children() {
        let cleaned = Arc::new(AtomicUsize::new(0));
        let guard = Guard(cleaned.clone());
        let mut child = scope(1);
        child
            .spawn("child", |_| async move {
                let _guard = guard;
                std::future::pending::<()>().await;
                Ok(0)
            })
            .unwrap();
        let mut parent = scope(1);
        parent
            .spawn("parent", |_| async move {
                let _child = child;
                std::future::pending::<()>().await;
                Ok(0)
            })
            .unwrap();
        parent.cancel();
        assert_eq!(cleaned.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn queue_backpressure_and_pool_join_cleanup() {
        let pool = BlockingPool::new(1, 1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let drops = Arc::new(AtomicUsize::new(0));
        let guard = Guard(drops.clone());
        let s = scope(1);
        let cx = s.context.clone();
        let running = pool
            .submit(cx.clone(), move || {
                let _guard = guard;
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(())
            })
            .unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let queued = pool.submit(cx.clone(), || Ok(())).unwrap();
        assert!(pool.submit(cx, || Ok(())).is_err());
        drop(running);
        drop(queued);
        release_tx.send(()).unwrap();
        drop(pool);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
