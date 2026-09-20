"""CLI trust boundaries, metrics accounting, and failed-benchmark regression tests."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from benchmark import read_metrics
from generate_model import generate
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()
    cli = str(args.cli.resolve())
    with tempfile.TemporaryDirectory(prefix="mlrt-tools-") as directory:
        temp = Path(directory)
        model, data = temp / "model.bin", temp / "input.csv"
        generate(model, data, samples=17, steps=5)
        duplicate, duplicate_data = temp / "duplicate.bin", temp / "duplicate.csv"
        generate(duplicate, duplicate_data, samples=17, steps=5)
        assert model.read_bytes() == duplicate.read_bytes() and data.read_bytes() == duplicate_data.read_bytes()
        bad_model = temp / "bad.bin"; bad_model.write_bytes(b"not a model")
        base = [cli, "--model", str(model), "--warmup", "0", "--quiet"]
        result = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0 and "Usage:" in result.stdout
        cases = [[cli], [cli, "--model"], [cli, "--model", str(temp / "missing.bin")]]
        cases += [base + extra for extra in (
            ["--input", str(temp / "missing.csv")], ["--input"], ["--model", str(bad_model)],
            ["--threads", "0"], ["--threads", "-1"], ["--threads", "257"], ["--threads", "abc"],
            ["--threads", "999999999999999999999999999999"],
            ["--batch-size", "0"], ["--batch-size", "65537"], ["--batch-timeout-ms", "-1"],
            ["--batch-timeout-ms", "60001"], ["--batch-timeout-ms", "nan"],
            ["--batch-timeout-ms", "inf"], ["--batch-timeout-ms", "abc"], ["--unknown", "1"],
            ["--output", str(model)], ["--metrics", str(model)],
            ["--input", str(data), "--output", str(data)],
            ["--output", str(temp / "same"), "--metrics", str(temp / "same")])]
        malformed = ["", "\n", "1,2\n", "x," + ",".join(["0"]*31), ",".join(["nan"]*32),
                     ",".join(["0"]*32) + ",\n", ",".join(["0"]*31 + ["1junk"])]
        for i, text in enumerate(malformed):
            path = temp / f"bad-{i}.csv"; path.write_text(text)
            cases.append(base + ["--input", str(path)])
        link = temp / "model-link.bin"; os.link(model, link)
        cases.append(base + ["--output", str(link)])
        for command in cases:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert result.returncode == 1 and "error:" in result.stderr, command
            assert result.stderr.strip() not in ("error: stod", "error: stoull"), "unhelpful numeric parser error"
        assert model.read_bytes() == duplicate.read_bytes(), "rejected output damaged the model"

        prediction, metrics = temp / "predictions.csv", temp / "metrics.json"
        command = base + ["--input", str(data), "--threads", "1", "--batch-size", "1",
                          "--clients", "4", "--requests", "11", "--warmup", "7",
                          "--output", str(prediction), "--metrics", str(metrics)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        assert not result.stdout, "quiet mode polluted stdout"
        values = read_metrics(metrics, {"mode": "concurrent", "threads": 1, "batch_size": 1,
                              "clients": 4, "total_requests": 11, "warmup_requests": 7, "batch_timeout_ms": 2})
        assert values["executed_batches"] == 11 and values["mean_batch_size"] == 1
        reference = np.loadtxt(temp / "reference_predictions.csv", delimiter=",", ndmin=2)
        np.testing.assert_allclose(np.loadtxt(prediction, delimiter=",", ndmin=2), reference[:11], atol=2e-6, rtol=2e-5)
        print(f"PASS {len(cases)} CLI rejection cases; help; clean CSV/JSON; warm-up exclusion; deterministic export")

        # Exit failures, missing JSON, and a missing second result may never
        # become successful CSV rows. The fake executable is strictly local.
        for scenario in ("exit", "missing", "stale", "invalid"):
            fake = temp / f"fake-{scenario}"
            # A shebang cannot quote an interpreter path containing spaces.
            # Resolve python3 through PATH, set to this interpreter's directory below.
            fake.write_text("#!/usr/bin/env python3\n" + '''import json, pathlib, sys
args = dict(zip(sys.argv[1:-1:2], sys.argv[2:-1:2]))
scenario = pathlib.Path(__file__).name.split('-')[-1]
count_file = pathlib.Path(__file__).with_suffix('.count')
count = int(count_file.read_text()) if count_file.exists() else 0
count_file.write_text(str(count + 1))
if scenario == 'exit': sys.exit(3)
if scenario == 'missing' or (scenario == 'stale' and count): sys.exit(0)
values = {'mode':args['--mode'], 'threads':int(args['--threads']), 'batch_size':int(args['--batch-size']),
          'clients':int(args['--clients']), 'warmup_requests':int(args['--warmup']),
          'total_requests':8, 'batch_timeout_ms':2, 'total_seconds':1, 'throughput_rps':8,
          'mean_latency_ms':1, 'median_latency_ms':1, 'p95_latency_ms':1, 'p99_latency_ms':1,
          'cpu_seconds':None, 'peak_rss_mib':None, 'executed_batches':8, 'mean_batch_size':1,
          'model_compute_ms':1, 'model_compute_ms_per_request':0.125}
if scenario == 'invalid': values['throughput_rps'] = float('nan')
pathlib.Path(args['--metrics']).write_text(json.dumps(values))
''')
            fake.chmod(0o700)
            output = temp / f"failure-{scenario}"
            result = subprocess.run([sys.executable, str(ROOT / "python/benchmark.py"), "--cli", str(fake),
                "--model", str(model), "--input", str(data), "--output", str(output), "--requests", "8",
                "--warmup", "3", "--threads", "1", "--batch-sizes", "1", "--repeats", "1"],
                capture_output=True, text=True, timeout=20,
                env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
            assert result.returncode != 0 and json.loads((output / "environment.json").read_text())["status"] == "failed"
            with (output / "results.csv").open() as file:
                assert len(list(csv.DictReader(file))) == (1 if scenario == "stale" else 0), (scenario, result.stderr)
        print("PASS benchmark failure/absent/stale/non-finite metric rejection")


if __name__ == "__main__":
    main()
