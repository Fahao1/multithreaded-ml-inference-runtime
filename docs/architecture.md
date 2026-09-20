# Architecture and ownership

## Components and data flow

`Tensor` owns a `std::vector<float>` plus dimensions. Linear algebra operates on
rank-two row-major matrices; a rank-one vector can broadcast as a bias. Dimensions
and allocation-size arithmetic are checked at construction. `at({row,column})`
always checks bounds; `operator[]` asserts in Debug. Copies have value semantics,
and moves transfer storage. A default or moved-from tensor is a storage holder;
assign a valid tensor before using it for inference.

`Model` owns validated layers and their weights. The loader reads the portable
[versioned binary format](model_format.md) once. `forward` accepts its input by
value: callers can move an owned tensor to avoid a copy. Each linear layer creates
one output matrix and then moves it into the current activation. Bias, ReLU,
sigmoid, and softmax operate in place. The old activation is released after the
next linear output is calculated. At most the current activation and next output
are live inside the forward pass. There is no persistent intermediate tensor for
every layer and no weight copy per request.

The matrix kernel uses i-k-j loops so the inner loop visits contiguous weight
and output rows. It deliberately uses no BLAS, intrinsics, or architecture-specific
packing. The compiler can vectorize contiguous loops. Sigmoid uses separate
positive/negative formulas; softmax subtracts each row's maximum. Inputs and each
layer's output must be finite. Overflow reports an error rather than returning
invalid JSON or propagating NaNs into predictions.

`ThreadPool` owns a fixed set of workers and a FIFO queue of packaged tasks.
`InferenceEngine` owns a pool, optional scheduler, pending-request queue, and a
`shared_ptr<const Model>`. The application may share that model with several
engines, but must not mutate it through a separate mutable alias. No model state
changes during inference.

For dynamically batched calls, an input vector and promise move into the request
queue. The scheduler removes a FIFO group, packs its input rows into one matrix,
and submits one job to the worker pool. A worker runs `forward`, splits the output
into independently owned rows, and resolves each request's own promise. Input
packing and output splitting are the intentional copies required by this API.
Temporary request storage is released after completion.

Individual calls with batch size 1 bypass the scheduler and enter the worker
queue directly. Explicit batches also bypass coalescing, regardless of their row
count; their rows stay in the original order. The configured maximum applies to
dynamic batches only. Synchronous calls use the same path and block on the future.
A single synchronous caller with batching enabled generally waits for the timeout,
since it cannot submit the next request until the current one completes.

## Thread ownership and synchronization

- The application owns the engine and keeps it alive through all API calls.
- The engine owns one scheduler thread when maximum batch size exceeds 1.
- The pool owns exactly the configured number of worker threads.
- The CLI's concurrent mode additionally creates a fixed number of load-generator
  client threads for each phase. These are application threads, not inference workers.
- The engine mutex protects acceptance state and pending requests. Its condition
  variable waits for arrivals, sufficient batch size, the oldest deadline, or shutdown.
- The pool mutex protects its queue and shutdown state. Idle workers sleep on a
  condition variable. They remove one task under lock and execute it after unlocking.
- Three relaxed atomic counters collect completed batch count, sample count, and
  summed compute time. A live snapshot is approximate across fields; metrics taken
  after all request futures finish are coherent for the completed phase.

Weights are read-only. Every forward pass has private activation buffers. Queues
are accessed only under their owning mutex. Promises belong to exactly one
accepted request; futures transfer its output or error to the caller. Workers
never acquire the engine mutex while running the model. Explicit submission
briefly takes the engine mutex followed by the pool mutex; no path takes these
locks in reverse order. No join occurs while either queue mutex is held.

Cross-worker completion order is unspecified. A future always corresponds to the
input that created it, even if other requests finish sooner. The CLI stores each
output in its request-index slot, then emits results in input order. It does not
infer completion time by consuming futures in that order.

## Batch scheduling

The scheduler waits until the queue is nonempty, then uses the oldest request's
enqueue timestamp plus `max_batch_wait` as its deadline. It wakes when the queue
reaches `max_batch_size`, that deadline expires, or shutdown begins. New arrivals
never reset the oldest deadline. It takes at most the configured maximum and
repeats. A zero timeout immediately dispatches whatever is available.

The timeout limits collection time under normal scheduling; it is **not** an
end-to-end latency guarantee. Scheduler preemption, work already queued in the
pool, inference, result delivery, and caller rescheduling all add latency. Larger
batches can amortize scheduling and improve weight locality, but the scalar kernel
still performs the work for every row. A batch does not promise a speedup.

## Shutdown lifecycle

1. Under the engine mutex, mark the engine as stopping and reject further submissions.
2. Notify the scheduler. It drains pending requests into full or partial jobs without
   waiting for deadlines. Already queued work remains owned by the engine/pool.
3. Join the scheduler, ensuring it cannot submit another pool task.
4. Mark the pool stopping, wake workers, drain accepted tasks, and join every worker.
5. Destroy queues, synchronization objects, and model references only after joining.

`std::call_once` makes concurrent explicit shutdown calls safe and idempotent.
Destructors call shutdown. Submission racing shutdown either succeeds before the
acceptance flag changes and completes, or throws. Destruction itself must not race
another method call: normal C++ object-lifetime rules still apply.

Call shutdown from an external thread, never from one of the object's workers.
Similarly, a generic pool task must not synchronously wait for another task in the
same exhausted pool. These self-join/nested-wait patterns are outside the API
contract. The inference API does not expose user callbacks inside workers.

## Errors and limits

Invalid dimensions and non-finite input throw before acceptance, including in the
asynchronous API. Model loading errors include the relevant header, shape, or
unsupported feature. Compute errors travel through futures. A dynamic batch
failure is delivered to every caller in that batch, including otherwise valid
rows if another row overflows. Workers remain available after task errors.
Scheduler allocation/submission errors stop further acceptance and fail pending
promises. If pool construction creates only some of its requested threads, it
joins those threads before propagating the failure.

Queues are unbounded and there is no cancellation, admission control, priority,
fairness guarantee across producers, NUMA placement, or deadline-aware worker
scheduling. The benchmark bounds outstanding requests with its fixed client count.
A service would need admission control before exposing this runtime to unbounded
traffic. Sanitizers and stress tests improve confidence but are not a formal proof
of race/deadlock freedom.
