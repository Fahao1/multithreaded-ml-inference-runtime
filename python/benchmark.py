"""Run bounded, closed-loop CLI workloads; retain every measured repetition."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_metadata(path, metadata):
    # Keep hardware/build evidence, but do not publish a user's home/project path.
    text = json.dumps(metadata, indent=2)
    text = text.replace(str(Path.cwd()), ".").replace(str(Path.home()), "~")
    path.write_text(text + "\n")


def read_metrics(path, expected):
    values = json.loads(path.read_text())
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"CLI metrics disagree with requested {key}")
    numeric = ("total_seconds", "throughput_rps", "mean_latency_ms", "median_latency_ms",
               "p95_latency_ms", "p99_latency_ms", "executed_batches", "mean_batch_size",
               "model_compute_ms", "model_compute_ms_per_request")
    for key in numeric:
        if not isinstance(values.get(key), (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
            raise ValueError(f"invalid CLI metric: {key}")
    if values["total_seconds"] <= 0 or values["executed_batches"] <= 0:
        raise ValueError("CLI reported no measured work")
    if not math.isclose(values["throughput_rps"], expected["total_requests"] / values["total_seconds"], rel_tol=1e-8):
        raise ValueError("inconsistent throughput")
    if not math.isclose(values["mean_batch_size"] * values["executed_batches"], expected["total_requests"], rel_tol=1e-8):
        raise ValueError("warm-up or missing requests in measured batch counters")
    if not 0 <= values["median_latency_ms"] <= values["p95_latency_ms"] <= values["p99_latency_ms"]:
        raise ValueError("invalid latency percentile ordering")
    for key in ("cpu_seconds", "peak_rss_mib"):
        if key not in values or (values[key] is not None and (not math.isfinite(values[key]) or values[key] < 0)):
            raise ValueError(f"invalid process metric: {key}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path("build/ml_inference_cli"))
    parser.add_argument("--model", type=Path, default=Path("models/example_mlp.bin"))
    parser.add_argument("--input", type=Path, default=Path("data/sample_inputs.csv"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/local"))
    parser.add_argument("--threads", type=positive, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch-sizes", type=positive, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--include-hardware", action="store_true", help="also test os.cpu_count() workers")
    parser.add_argument("--clients", type=positive, default=32)
    parser.add_argument("--requests", type=positive, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--batch-timeout-ms", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.warmup < 0 or not 0 <= args.batch_timeout_ms <= 60000:
        parser.error("warmup must be nonnegative and timeout in [0, 60000]")
    workers = sorted(set(args.threads + ([os.cpu_count() or 1] if args.include_hardware else [])))
    if max(workers) > 256 or args.clients > 256 or max(args.batch_sizes) > 65536:
        parser.error("CLI limits: workers/clients <=256, batches <=65536")
    args.output.mkdir(parents=True, exist_ok=True)
    cli, model, inputs = args.cli.resolve(), args.model.resolve(), args.input.resolve()
    configs = [("sync", 1, 1, 1)] + [("concurrent", t, b, args.clients) for t in workers for b in args.batch_sizes]
    work = [(repeat, config) for repeat in range(1, args.repeats + 1) for config in configs]
    random.Random(args.seed).shuffle(work)
    cache_path = cli.parent / "CMakeCache.txt"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.startswith("CMAKE_") and "=" in line and ":" in line:
                key, value = line.split("=", 1)
                cache[key.split(":")[0]] = value
    compiler = cache.get("CMAKE_CXX_COMPILER", "c++")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
        "python": sys.version, "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "build_type": cache.get("CMAKE_BUILD_TYPE", "unknown"),
        "cxx_flags": cache.get("CMAKE_CXX_FLAGS", ""),
        "build_type_flags": cache.get("CMAKE_CXX_FLAGS_" + cache.get("CMAKE_BUILD_TYPE", "").upper(), ""),
        "cli_sha256": sha256(cli), "model_sha256": sha256(model), "input_sha256": sha256(inputs),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "commands": [], "status": "running",
    }
    if sys.platform == "darwin":
        for name in ("machdep.cpu.brand_string", "hw.memsize"):
            process = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
            if process.returncode == 0:
                metadata[name] = process.stdout.strip()
    environment = args.output / "environment.json"
    save_metadata(environment, metadata)
    results = args.output / "results.csv"
    try:
        with tempfile.TemporaryDirectory(prefix="mlrt-benchmark-") as directory, results.open("w", newline="") as file:
            metrics = Path(directory) / "metrics.json"
            writer = None
            for index, (repeat, (mode, threads, batch, clients)) in enumerate(work, 1):
                command = [str(cli), "--model", str(model), "--input", str(inputs), "--mode", mode,
                           "--threads", str(threads), "--batch-size", str(batch), "--clients", str(clients),
                           "--batch-timeout-ms", str(args.batch_timeout_ms), "--requests", str(args.requests),
                           "--warmup", str(args.warmup), "--metrics", str(metrics), "--quiet"]
                metadata["commands"].append([arg if arg != str(metrics) else "<temporary>/metrics.json" for arg in command])
                metrics.unlink(missing_ok=True)  # never reuse a previous run's JSON
                process = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                values = read_metrics(metrics, {"mode": mode, "threads": threads, "batch_size": batch,
                                      "clients": clients, "total_requests": args.requests,
                                      "warmup_requests": args.warmup, "batch_timeout_ms": args.batch_timeout_ms})
                row = {"run_order": index, "repeat": repeat, **values}
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(f"[{index}/{len(work)}] {mode} threads={threads} batch={batch}: "
                      f"{row['throughput_rps']:.0f} req/s; p95={row['p95_latency_ms']:.3f} ms", flush=True)
        from plot_benchmarks import plot
        plot(results, args.output)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", error=str(error))
        save_metadata(environment, metadata)
        raise
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    save_metadata(environment, metadata)
    print(f"Saved raw repetitions, environment, and four plots to {args.output}")


if __name__ == "__main__":
    main()
