# Verification

Linux verification completed on **2026-09-20** in a private GitHub repository before
public publication. The [successful implementation run](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885)
verified commit `64b7a6f636c526f7e2cf42669e07f04399bcfadf`. All six jobs passed.
The README badge follows subsequent runs on `main`; documentation changes also run
the complete workflow.

## Linux environment and results

GitHub-hosted `ubuntu-24.04`, **Ubuntu 24.04.5 LTS, x86_64**, runner image
`20260907.300.1`. Build/test compiler: **GCC 13.3.0**
(`Ubuntu 13.3.0-6ubuntu2~24.04.1`). CMake/CTest **3.31.6**, Python **3.12.14**,
NumPy **2.5.3**, matplotlib **3.11.2**. Quality tools and analysis compiler:
**Clang, clang-format, and clang-tidy 17.0.6** from Ubuntu's LLVM 17 packages.

Each job starts from a clean checkout. The build jobs use GCC 13.3.0; the quality
job uses LLVM 17.0.6 and the actual exported compilation database.

- [Release — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132246):
  five CTest entries, three additional stress repetitions, and benchmark smoke.
- [Debug — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132266):
  five CTest entries and three additional stress repetitions.
- [ASan and leaks — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132285):
  known-leak detection probe, five CTest entries with leak detection enabled,
  and three additional stress repetitions.
- [UBSan — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132356):
  five CTest entries and three additional stress repetitions.
- [TSan — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132333):
  five CTest entries and three additional stress repetitions; no race report.
- [LLVM 17 quality — passed](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537535885/job/106149132093):
  all 12 owned C++ files passed formatting, and all seven translation units,
  including tests, passed clang-tidy. This job analyzes sources; it does not run CTest.

The five CTest entries are `runtime_tests`, `concurrency_stress`, `metrics_tests`,
`numpy_validation`, and `tool_tests`. This gives **25 successful CTest entries**
across five configurations, plus **15 additional stress repetitions**. Each stress
invocation checks 16,384 request/result associations, accepted shutdown-race work,
and 74 engine lifecycles. The complete run therefore checked **327,680 normal
stress results and 1,480 engine lifecycles**, plus the varying shutdown-race counts.
No accepted future was left unresolved, and no deadlock or sanitizer failure was observed.

NumPy validation tested ten configurations per build, including synchronous,
concurrent, and dynamically batched execution, sigmoid extremes, and softmax.
Maximum absolute error across these Linux jobs was **4.76e-7**, within unchanged
`atol=2e-6, rtol=2e-5`; predicted classes also matched.

The Release smoke benchmark completed five configurations with 128 measured and
32 warm-up requests per run: **640 measured requests**, excluding 160 warm-ups.
The script validated CLI metrics JSON, saved raw CSV/environment metadata, and
produced all four plots. This is a short pipeline check, not evidence of stable
performance gains. Published longer benchmark data remains explicitly labeled
as historical macOS measurements in [the measurement report](../benchmarks/example-run/README.md).

## Sanitizer evidence

ASan, UBSan, and TSan use separate Debug builds with the matching `-fsanitize`
compile and link option. TSan is never combined with another sanitizer.

The Linux ASan job uses `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`LSAN_OPTIONS=exitcode=23`. Its temporary canary reported **123 bytes leaked in one
allocation** and returned the required exit code 23. The workflow explicitly checks
both the expected diagnostic and exit status. This proves leak detection was active;
the following real runtime/integration tests then passed with that detection enabled.
The canary is outside the source tree and is not a normal build or CTest target.

UBSan uses `halt_on_error=1:print_stacktrace=1`; TSan uses `halt_on_error=1`.
No suppressions, reduced coverage, `continue-on-error`, or relaxed numerical
tolerances were introduced to obtain successful results.

## Failures corrected during private verification

The failed runs and normal corrective commits remain visible in repository history.

1. [Initial run](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537133042):
   all five build/test jobs passed, but the quality job failed before analysis.
   Bare executable-name overrides were interpreted as relative CMake FILEPATH
   values. Removing those redundant overrides let the existing `find_program`
   discovery resolve the installed LLVM 17 binaries. This was a workflow
   configuration error, not an inference defect.
2. [Second run](https://github.com/Fahao1/multithreaded-ml-inference-runtime/actions/runs/35537259586):
   formatting and all five build/test jobs passed; clang-tidy reported 32 project
   diagnostics. Fixes made numeric conversions explicit in metrics/test code,
   performed size/count multiplication in the intended wide type, reserved known
   test-vector capacities, and made the metrics test report unexpected exceptions
   while explicitly checking expected empty-input rejection. Its numeric test
   helper now also rejects non-finite results. These were type-clarity,
   test-performance, and test-error-reporting findings; no race was reported.
3. The resulting implementation passed all six jobs in the successful run linked
   above. Local Release, Debug, separate sanitizers, NumPy tests, repeated stress,
   formatting, and a short benchmark were rerun after the C++ changes.

All enabled project clang-tidy checks remain active and warnings are errors.
LLVM's default filtering excludes diagnostics from non-user/system code; it is
not a project-warning suppression. No project-specific suppression was added.

## Local macOS verification

Local checks ran on **macOS 15.7.9 / Darwin 24.6.0 ARM64**, with Apple Clang and
Apple clang-format **17.0.0** (`clang-1700.0.13.5`), CMake/CTest **4.4.3**,
Python **3.13.7**, NumPy **2.5.3**, and matplotlib **3.11.2**.

Clean Release/Debug and separate ASan/UBSan/TSan builds passed all five CTest entries
and three extra stress repetitions each after the analysis fixes. The example
model's earlier explicit NumPy comparison had maximum error **1.79e-7**; the seeded
medium **256→512→256→16** model had maximum error **5.98e-8**, with unchanged
tolerances. A pre/post-format comparison produced byte-identical predictions for
256 rows of each model. Medium weights were generated with zero training steps;
no trained-accuracy claim is made for that workload.

A local test-harness defect was also corrected: direct interpreter shebangs failed
when the virtual-environment path contained a space. Mock CLI scripts now select
the current interpreter through PATH. The benchmark-failure regression cases pass
from a workspace containing a space.

Apple Clang rejected `-fsanitize=leak` for `arm64-apple-darwin24.6.0`. An ASan probe
with `detect_leaks=1` also aborted with `detect_leaks is not supported on this
platform`. Local ASan therefore used `detect_leaks=0`; its pass is **not** leak
verification. Leak verification comes from Linux CI. clang-tidy is unavailable
locally; its verified result likewise comes from Linux.

## Reproducing checks

[README build and test instructions](../README.md#build-and-test) cover local
setup, example generation, NumPy comparison, and stress tests. Its
[developer-tool instructions](../README.md#sanitizers-and-developer-tools)
cover formatting, analysis, and macOS sanitizer settings.

On Linux, with Python requirements installed, use a new directory for every build:

```bash
cmake -S . -B build-linux-asan -DCMAKE_BUILD_TYPE=Debug \
  -DMLRT_SANITIZER=address -DMLRT_REQUIRE_PYTHON_TESTS=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build-linux-asan --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 LSAN_OPTIONS=exitcode=23 \
  ctest --test-dir build-linux-asan --output-on-failure --verbose
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 LSAN_OPTIONS=exitcode=23 \
  ctest --test-dir build-linux-asan -R concurrency_stress \
    --repeat until-fail:3 --output-on-failure --verbose
```

Use separate builds with `MLRT_SANITIZER=undefined` and `thread` for UBSan and TSan,
with the environment options described above. The exact CI commands and leak
canary are in [the workflow](../.github/workflows/ci.yml). Its default permissions
are read-only, checkout credentials are not persisted, and all failures propagate.

Local workflow checks parsed YAML and checked shell syntax. `actionlint` and Linux
container tools were unavailable locally; actual GitHub Actions execution now
provides Linux verification. Other operating systems, CPU architectures, and
compiler versions have not been tested. Passing these checks is not production
hardening or a formal concurrency proof. Scalar kernels, unbounded queues, and
closed-loop benchmark limitations remain documented.
