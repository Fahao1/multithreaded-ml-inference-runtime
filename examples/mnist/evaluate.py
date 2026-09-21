"""Validate the exported MNIST model through real C++ single/batched inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from model_io import LINEAR, RELU, SOFTMAX, forward, load_model
from data import load_split, validate_inputs

HERE = Path(__file__).resolve().parent
ATOL, RTOL = 2e-6, 2e-5


def model_layers(path):
    width, layers = load_model(path)
    if width != 784 or [x[0] for x in layers] != [LINEAR, RELU, LINEAR, SOFTMAX]:
        raise ValueError("expected 784 -> 128 ReLU -> 10 softmax MNIST model")
    if layers[0][1].shape != (784, 128) or layers[2][1].shape != (128, 10):
        raise ValueError("incorrect MNIST layer dimensions")
    return layers


def cpp_predict(cli, model, inputs, batch=16, threads=2):
    with tempfile.TemporaryDirectory(prefix="mlrt-mnist-") as directory:
        temp = Path(directory)
        source, output = temp / "inputs.csv", temp / "predictions.csv"
        np.savetxt(source, inputs, delimiter=",", fmt="%.9g")
        command = [str(Path(cli).resolve()), "--model", str(model), "--input", str(source),
                   "--output", str(output), "--threads", str(threads), "--batch-size", str(batch),
                   "--clients", "32", "--batch-timeout-ms", "2", "--warmup", "0", "--quiet"]
        process = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if process.returncode:
            raise RuntimeError(f"C++ inference failed: {process.stderr.strip()}")
        actual = np.loadtxt(output, delimiter=",", ndmin=2)
    if actual.shape != (len(inputs), 10) or not np.isfinite(actual).all():
        raise ValueError("invalid C++ probability output")
    return actual


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    return float(np.max(np.abs(actual - expected)))


def evaluate(cli, model, inputs, labels):
    validate_inputs(inputs, labels)
    expected = forward(inputs, model_layers(model))
    maximum = 0.0
    for batch, threads in [(1, 1), (16, 2)]:
        actual = cpp_predict(cli, model, inputs, batch, threads)
        maximum = max(maximum, compare(actual, expected))
    return {
        "test_accuracy": float(np.mean(actual.argmax(axis=1) == labels)),
        "evaluated_samples": len(inputs), "class_agreement": 1.0,
        "max_probability_difference": maximum,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "configurations": [{"threads": 1, "batch_size": 1}, {"threads": 2, "batch_size": 16}],
    }, actual


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/ml_inference_cli")
    parser.add_argument("--model", type=Path, default=HERE / "mnist_mlp.bin")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    args = parser.parse_args()
    try:
        model_layers(args.model)  # Reject missing models before any download.
        inputs, labels = load_split(args.cache)
        result, _ = evaluate(args.cli, args.model, inputs, labels)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
