# Verification

Local macOS ARM64 checks passed on 2026-09-20 after correcting one test-harness
portability defect. Linux execution, Linux leak detection, and clang-tidy analysis
remain pending the private GitHub verification workflow.

## Local environment

macOS 15.7.9 / Darwin 24.6.0, ARM64, 12 reported logical CPUs. Apple Clang
17.0.0 (`clang-1700.0.13.5`), CMake/CTest 4.4.3, Python 3.13.7, NumPy 2.5.3,
matplotlib 3.11.2. This follow-up used the existing project virtual environment.
An isolated requirements-only environment was also verified before publication.

Five build trees were created under `/tmp/mlrt-quality-cIxd2X`; existing project
builds were preserved. Logs, command/exit-code records (`checks.json` and
`smoke-checks.json`), before-format snapshots, and the leak probes are in that
local temporary directory. Temporary files are not part of the proposed commit.

## Formatting and static analysis

- **Local formatting passed.** `xcrun --find clang-format` found the executable
  in the Apple command-line tools although it was absent from PATH. Version:
  `Apple clang-format version 17.0.0 (clang-1700.0.13.5)`.
- The original `--dry-run --Werror` produced 621 formatting diagnostics across
  all 12 owned C++ files. The repository's `.clang-format` was applied to only those
  files. The final dry-run and CMake `format-check` target passed.
- A deliberately misformatted function in a temporary source copy made
  `format-check` fail with exit code 2. No intentional violation remains in the
  project. The formatted files exactly equal the formatter's output from their
  saved originals; there were no manual C++ logic changes.
- **Local clang-tidy is unavailable.** PATH, versioned executable names, xcrun,
  Apple toolchains, and common Homebrew LLVM locations were checked. Its version,
  warning count, and analysis outcome are unknown; no static-analysis pass or
  warning-free result is claimed. No analyzer warning was suppressed or declared
  harmless without analysis.
- CMake now supplies optional `format-check`, `format`, and `tidy-check` targets.
  They explicitly select owned `src`, `apps`, `tests`, and public-header files.
  Tests are included in analysis. Seven translation units appear in the actual
  compilation database; dependencies and generated files are not selected.
- `.clang-tidy` retains the enabled bugprone, performance, and selected modernize
  checks, includes CLI headers as well as public headers, and treats enabled
  warnings as errors. Linux CI installs LLVM major version 17 and runs both checks.

Local formatting commands (run from the repository root):

```bash
source .venv/bin/activate
xcrun --find clang-format
"$(xcrun --find clang-format)" --version
"$(xcrun --find clang-format)" --style=file --dry-run --Werror \
  include/ml_runtime/*.hpp src/*.cpp apps/*.cpp apps/*.hpp tests/*.cpp
cmake -S . -B /tmp/mlrt-quality-cIxd2X/release -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" -DMLRT_REQUIRE_PYTHON_TESTS=ON \
  -DMLRT_CLANG_FORMAT="$(xcrun --find clang-format)"
cmake --build /tmp/mlrt-quality-cIxd2X/release --target format
cmake --build /tmp/mlrt-quality-cIxd2X/release --target format-check
```

## Builds, tests, sanitizers, and the corrected defect

Clean Release and Debug builds each passed **all five CTest entries**:
`runtime_tests`, `concurrency_stress`, `metrics_tests`, `numpy_validation`, and
`tool_tests`. Separate AddressSanitizer, UndefinedBehaviorSanitizer, and
ThreadSanitizer builds also passed all five entries. Each configuration additionally
passed three stress repetitions. Compiler warnings `-Wall -Wextra -Wpedantic`
were active; no compiler warnings were emitted.

The first Release test run failed in `tool_tests`: its generated mock executable
put the virtual-environment interpreter path directly into a shebang. The space in
the workspace path prevented launching it. Earlier tests from a temporary path
without spaces did not expose this. The mock now uses `/usr/bin/env python3` with
PATH explicitly selecting the current interpreter's directory. The existing
failed/stale/missing/non-finite benchmark regression cases all pass from this
workspace, including its space. This was a test-harness defect, not a runtime
inference defect. No numerical tolerances were changed.

Twenty successful post-fix stress invocations exercised **327,680 uniquely checked
normal request results**, additional accepted shutdown-race requests, and **1,480
engine lifecycles**. Checks cover 1/4 workers, batch sizes 1/16, eight concurrent
producers, drained shutdown races, repeated empty shutdown/destruction, and exact
request-result association. Unit tests additionally verify size and timeout
batch triggers, partial flushes, future exceptions, and submission after shutdown.
No sanitizer diagnostic, unresolved accepted future, or deadlock was observed.

Compile databases were inspected: all seven translation units in each sanitizer
build include the requested sanitizer flag; all three executable link commands
also include it. The sanitizers were never combined in this follow-up.

These are the commands used for each configuration (`release`, `debug`, `asan`,
`ubsan`, `tsan`); the first Release build was fresh, and its tests were rerun after
the Python-only launcher fix:

```bash
source .venv/bin/activate
quality_dir=/tmp/mlrt-quality-cIxd2X
for configuration in release debug asan ubsan tsan; do
  build_type=Debug
  sanitizer=
  case "$configuration" in
    release) build_type=Release ;;
    asan) sanitizer=address ;;
    ubsan) sanitizer=undefined ;;
    tsan) sanitizer=thread ;;
  esac
  cmake -S . -B "$quality_dir/$configuration" \
    -DCMAKE_BUILD_TYPE="$build_type" -DMLRT_SANITIZER="$sanitizer" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPython3_EXECUTABLE="$PWD/.venv/bin/python" -DMLRT_REQUIRE_PYTHON_TESTS=ON \
    -DMLRT_CLANG_FORMAT="$(xcrun --find clang-format)"
  cmake --build "$quality_dir/$configuration" --parallel 4
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 TSAN_OPTIONS=halt_on_error=1 \
    ctest --test-dir "$quality_dir/$configuration" --output-on-failure
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 TSAN_OPTIONS=halt_on_error=1 \
    ctest --test-dir "$quality_dir/$configuration" -R concurrency_stress \
      --repeat until-fail:3 --output-on-failure
done
```

The loop consolidates the separately executed commands; each run enabled the
relevant sanitizer environment option. Use a new temporary directory to reproduce
fresh builds. The zero-leak-detection setting above is specific to this macOS host.

## LeakSanitizer: unsupported locally, pending on Linux

A standalone temporary known-leak program was compiled with:

```bash
clang++ -std=c++17 -g -O0 -fsanitize=leak \
  /tmp/mlrt-quality-cIxd2X/leak_probe.cpp -o /tmp/mlrt-quality-cIxd2X/leak_probe
```

Apple Clang returned exit code 1: `unsupported option '-fsanitize=leak' for target
'arm64-apple-darwin24.6.0'`. The same source compiled successfully with
`-fsanitize=address`; running with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`
aborted with `AddressSanitizer: detect_leaks is not supported on this platform.`
These are direct support-probe results, **not successful leak checks**. Local ASan
runs explicitly used `detect_leaks=0`.

Linux CI sets `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`LSAN_OPTIONS=exitcode=23`. Before project tests, a separate temporary ASan canary
must report `LeakSanitizer: detected memory leaks` and exit 23. A missing report,
wrong status, or unexpectedly successful canary fails CI. The subsequent normal
tests must all exit successfully. Intentionally leaking code is not in a project
build target or the normal public CTest suite. Linux execution remains pending;
the canary has not been claimed to detect leaks on this host.

## NumPy agreement and benchmark smoke

The example model passed ten configurations with maximum absolute error
**1.79e-7**. The seeded medium network, **256→512→256→16**, has 267,024 parameters
and passed with maximum error **5.98e-8**. Both use the unchanged
`atol=2e-6, rtol=2e-5`, with matching predicted classes. Medium weights were
seeded with zero training steps; no accuracy/training claim is made.

A separately compiled pre-format snapshot and the formatted Release build produced
**byte-identical prediction CSVs for 256 tiny-model rows and 256 medium-model rows**.

Two short sweeps each tested a synchronous baseline and four concurrent settings:
1/2 workers × batch sizes 1/4, eight clients, 256 measured requests and 32 warm-up
requests per run, one repetition. All ten runs completed: **2,560 measured requests**;
320 warm-up requests were excluded. Results, environment metadata, and eight plots
are under ignored `benchmarks/local/quality-followup/{tiny,medium}`. Throughput
identities, measured counts, JSON schema/numbers, and complete-run status were
checked. A separate 17-request / 7-warm-up CLI run also passed metrics JSON parsing.
Matplotlib emitted nonfatal font-cache warnings on first use; all plots were produced.

Machine-specific observed throughput ranges were **187,764–359,067 requests/s**
for the tiny model and **42,214–76,601 requests/s** for the medium model. These
millisecond-scale single-repetition runs are pipeline smoke tests, not evidence of
stable speedups or a replacement for the saved longer benchmark experiment.

```bash
source .venv/bin/activate
quality_dir=/tmp/mlrt-quality-cIxd2X
python python/validate.py --cli "$quality_dir/release/ml_inference_cli" \
  --model models/example_mlp.bin --input data/sample_inputs.csv
python python/generate_model.py --widths 256 512 256 16 --steps 0 --samples 64 \
  --model "$quality_dir/medium.bin" --data "$quality_dir/medium.csv"
python python/validate.py --cli "$quality_dir/release/ml_inference_cli" \
  --model "$quality_dir/medium.bin" --input "$quality_dir/medium.csv"
python python/benchmark.py --cli "$quality_dir/release/ml_inference_cli" \
  --requests 256 --warmup 32 --repeats 1 --threads 1 2 --batch-sizes 1 4 \
  --clients 8 --output benchmarks/local/quality-followup/tiny
python python/benchmark.py --cli "$quality_dir/release/ml_inference_cli" \
  --model "$quality_dir/medium.bin" --input "$quality_dir/medium.csv" \
  --requests 256 --warmup 32 --repeats 1 --threads 1 2 --batch-sizes 1 4 \
  --clients 8 --output benchmarks/local/quality-followup/medium
"$quality_dir/release/ml_inference_cli" --model models/example_mlp.bin \
  --input data/sample_inputs.csv --requests 17 --warmup 7 --threads 2 \
  --batch-size 4 --clients 8 --metrics "$quality_dir/metrics.json" \
  --output "$quality_dir/predictions.csv" --quiet
python -m json.tool "$quality_dir/metrics.json"
```

The actual benchmark launches additionally set `MPLCONFIGDIR` inside the temporary
quality directory to keep plotting caches out of the repository.

## Linux workflow: configured, not executed

The [workflow](../.github/workflows/ci.yml) targets Ubuntu 24.04 with read-only
`contents` permission and checkout credential persistence disabled. It uses the
current stable major tags [checkout v7](https://github.com/actions/checkout/releases/tag/v7.0.1)
and [setup-python v7](https://github.com/actions/setup-python/releases/tag/v7.0.0).

Five independent jobs build Release, Debug, ASan with leak detection, UBSan, and
TSan. Every configuration installs `requirements.txt`, requires Python/NumPy test
registration, runs all five CTest entries, and repeats concurrency stress three
more times. The Release job smoke-tests benchmark CSV/JSON and plotting. Any test
or sanitizer failure fails the job; `continue-on-error` is not used.

A sixth job installs Ubuntu's LLVM 17 tools and configures a real Clang compilation
database, then runs formatting and tidy analysis without an unnecessary preliminary
compilation. [Ubuntu's clang-tidy-17 package](https://packages.ubuntu.com/noble/clang-tidy-17)
and [clang-format-17 package](https://packages.ubuntu.com/noble/arm64/clang-format-17)
belong to the same 17.0.6 toolchain series. The workflow pins the LLVM major version;
Ubuntu package revisions may receive updates.

The analysis job's commands are documented in the README. Unlike the normal test
jobs, it disables only Python test registration; the C++ test translation units
remain in the compilation database and are analyzed.

Local validation parsed the workflow with Ruby's YAML parser, asserted trigger,
runner, matrix, permission, and failure-handling settings, and syntax-checked each
embedded script with `bash -n` after substituting matrix placeholders. `actionlint`
is unavailable. YAML parsing and shell syntax checks do not validate GitHub's full
workflow semantics or replace execution on a runner.

Docker, Podman, and Colima are unavailable. No Linux build, sanitizer, formatter,
static analyzer, or leak-detection success is claimed. The first push must be
followed by a successful Linux workflow; any resulting diagnostics need resolution
before describing those checks as verified.
