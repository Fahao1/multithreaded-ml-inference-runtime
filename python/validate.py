"""Compare real CLI predictions with an independent NumPy forward pass."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np

from generate_model import generate
from model_io import LINEAR, SIGMOID, SOFTMAX, export_model, forward, load_model

ATOL, RTOL = 2e-6, 2e-5


def compare(cli, model, data, temp):
    _, layers = load_model(model)
    inputs = np.loadtxt(data, delimiter=",", dtype=np.float32, ndmin=2)
    expected = forward(inputs, layers)
    maximum_error = 0.0
    for mode, threads, batch in [("sync", 1, 1), ("concurrent", 4, 1),
                                  ("concurrent", 2, 4), ("concurrent", 4, 16), ("concurrent", 2, 32)]:
        output = temp / "predictions.csv"
        command = [str(cli), "--model", str(model), "--input", str(data), "--output", str(output),
                   "--mode", mode, "--threads", str(threads), "--batch-size", str(batch),
                   "--batch-timeout-ms", "1", "--warmup", "0", "--quiet"]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        actual = np.loadtxt(output, delimiter=",", ndmin=2)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    return maximum_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path("build/ml_inference_cli"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if bool(args.model) != bool(args.input):
        parser.error("--model and --input must be provided together")
    cli = args.cli.resolve()
    with tempfile.TemporaryDirectory(prefix="mlrt-validation-") as directory:
        temp = Path(directory)
        model, data = args.model, args.input
        if model is None:
            model, data = temp / "model.bin", temp / "inputs.csv"
            generate(model, data, samples=53, steps=40)
        error = compare(cli, model, data, temp)
        # A second model exercises sigmoid, including numerically extreme inputs.
        sigmoid_model, sigmoid_data = temp / "sigmoid.bin", temp / "sigmoid.csv"
        export_model(sigmoid_model, 3, [(LINEAR, np.eye(3, dtype=np.float32), np.zeros(3)),
                                        (SIGMOID, None, None), (SOFTMAX, None, None)])
        np.savetxt(sigmoid_data, [[-1000, 0, 1000], [2, -3, 5], [0, 0, 0]], delimiter=",")
        error = max(error, compare(cli, sigmoid_model, sigmoid_data, temp))
        base = [str(cli), "--model", str(model), "--quiet", "--warmup", "0"]
        bad_csv = temp / "bad.csv"
        bad_csv.write_text("1,2,garbage\n")
        for extra in [["--threads", "0"], ["--batch-size", "0"], ["--mode", "unknown"],
                      ["--requests", "0"], ["--batch-timeout-ms", "nan"], ["--unknown", "1"],
                      ["--input", str(bad_csv)]]:
            result = subprocess.run(base + extra, capture_output=True, text=True, timeout=10)
            if result.returncode != 1 or "error:" not in result.stderr:
                raise AssertionError(f"invalid CLI input not rejected: {extra}")
        print(f"PASS NumPy/CLI: 10 configurations; max absolute error={error:.3g}; atol={ATOL}, rtol={RTOL}")
        print("PASS malformed CLI/CSV rejection")


if __name__ == "__main__":
    main()
