# Benchmark methodology

## Workload and reproducibility

The NumPy generator trains a 32→64→32→4 MLP on 512 synthetic examples from a
seeded linear teacher. ReLU follows the hidden layers and softmax produces four
scores. Training is deliberately small and educational; classification accuracy
does not establish usefulness on a real dataset. The generated model metadata
records training settings and accuracy. NumPy version and BLAS implementation can
affect low-order bits; fixed seeds reproduce the workload, not universal bitwise
equality across software/hardware.

The benchmark loads the binary once per process, reads the same 256 input rows,
and cycles through those rows for the requested count. There is no prediction
cache. Each configuration/repetition runs in a fresh CLI process. A fixed seed
shuffles the order of all repetitions to reduce systematic order/thermal bias.
Worker and client threads are persistent within each phase, never created per
request. Peak memory includes both kinds of thread.

Default sweep: 1, 2, 4 workers; maximum dynamic batches 1, 4, 8, 16, 32;
32 concurrent clients; 2 ms batch wait; 500 warm-up requests; 5,000 measured
requests; three repetitions. `--include-hardware` also tests the detected logical
CPU count; it is opt-in to keep the default run modest. Hardware concurrency is
not necessarily the appropriate number of workers on a mixed-core or shared host.

There is also a sequential-caller baseline with one inference worker and batching
disabled. This baseline uses the same future/pool engine path. It is not a direct
`Model::forward` microbenchmark. The concurrent batch-1, one-worker configuration
is the matching-load baseline for isolating worker/batching changes. Comparing
sequential and concurrent modes changes offered concurrency as well as configuration.

## Timing boundaries

The CLI uses `std::chrono::steady_clock`, a monotonic clock. It first completes all
warm-up requests. It then allocates measurement/output storage and creates the
fixed client threads behind a start gate. The measured wall interval starts just
before opening the gate and ends after all clients join. It includes gate wake-up,
dispatch, synchronization, and final join overhead. It excludes model loading,
CSV parsing, thread creation, warm-up, output serialization, and engine shutdown.

Each client has one outstanding request. Per-request latency starts before copying
the sample into `predict` and ends after that request's own future returns to its
caller. This **caller-observed end-to-end latency** includes input validation and
copying, both queue waits, dynamic batch collection, packing, inference, output
splitting, synchronization, and caller rescheduling. Storing predictions is outside
the individual latency interval; benchmark mode suppresses prediction storage.

This is a **closed-loop** load generator: a client submits the next request only
after the previous one completes. It does not simulate an independent arrival
process or measure latency under arbitrary overload; coordinated omission can
hide that behavior. Run counts are finite, and final partial batches can increase
tail latency. Configured maximum batch size is not actual average batch size:
both `executed_batches` and `mean_batch_size` are recorded.

`model_compute_ms` sums worker time spent in `Model::forward`, including its
validation, allocations, kernels, and in-place activations. It excludes batching
input packing and output splitting. It is elapsed worker time, not process CPU
time, and overlapping workers can make the sum exceed measured wall time.
The per-request value is that sum divided by completed samples, not end-to-end
latency or a separately timed request inside a fused batch.

## Recorded metrics

Every CSV row is one measured repetition and records total requests, warm-up count,
thread/client counts, maximum batch size, timeout, wall seconds, requests/second,
mean/median/p95/p99 caller latency in milliseconds, process CPU seconds, peak RSS,
executed batches, average batch size, and aggregate/per-request model compute time.

Percentiles use linear interpolation between sorted samples at `(N−1) × p`.
Throughput is completed measured requests divided by measured wall seconds.
CPU time is the difference in process user+system `getrusage` times immediately
around the measured phase; it includes clients, scheduler, and workers and can
exceed wall time. On Linux, `ru_maxrss` is KiB; on macOS it is bytes. Both are
converted to MiB. Peak RSS is a **whole-process lifetime high-water mark** including
model load, input storage, and warm-up; it cannot be reset at the measurement gate.
Unsupported resource counters become JSON `null`/empty CSV cells, never fabricated
zeroes. The measurement code targets Linux/macOS resource conventions.

`environment.json` records OS, architecture, CPU count, compiler, CMake build type
and flags, artifact hashes, command arguments, run order, and timestamps. It records
the executable and model identities rather than assuming they match a later rebuild.
Project/home paths are normalized for sharing and temporary metrics-file paths
are placeholders. Review metadata before sharing runs created outside the project.
Raw CSV rows are flushed only after successful execution and metric validation;
stale JSON is removed before each invocation. Metadata says `complete` only after
the sweep and all plots succeed. Ordinary exceptions and keyboard interrupts set
`failed`; a forcibly killed process can leave partial data and `running` status.
Neither partial state represents a complete experiment.

## Plots and interpretation

The four plots show throughput, mean/p95 caller latency, CPU seconds, and peak RSS
against maximum batch size. Values are medians across repetitions; throughput
bands show the observed min/max, not confidence intervals. Latency plots show the
median of run-level mean/p95 values, not percentiles pooled across all runs. The
horizontal sequential-caller line is context, not an identical-load control.

These short local runs characterize this implementation, model, compiler, workload,
and machine only. Small models can be dominated by synchronization and client
wake-ups. Additional workers can reduce throughput; larger batches can increase
latency or fail to fill. CPU frequency, background load, scheduling, power state,
allocator, build flags, and instrumentation affect results. Never use Debug or
sanitizer timings as Release performance evidence.

For stronger conclusions, increase request counts and repetitions, isolate the
machine, repeat on Linux and larger models, and use an open-loop arrival generator
with admission control. No comparison with TensorFlow, PyTorch, or ONNX Runtime is
claimed. Saved local results are observations, not portable performance promises.
