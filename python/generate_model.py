"""Train a seeded synthetic classifier with NumPy SGD and export MLRTBIN v1."""
import argparse
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np

from model_io import LINEAR, RELU, SOFTMAX, export_model, forward


def generate(model_path, data_path, seed=42, samples=256, steps=400, widths=None):
    widths = [32, 64, 32, 4] if widths is None else list(widths)
    if len(widths) < 2 or len(widths) > 129 or any(w < 1 or w > 65536 for w in widths):
        raise ValueError("widths must contain 2..129 integers in [1, 65536]")
    if 20 + 16 * (len(widths) - 1) + 4 * sum(a*b+b for a,b in zip(widths, widths[1:])) > 256*1024*1024:
        raise ValueError("model exceeds 256 MiB format limit")
    if seed < 0 or samples < 1 or steps < 0:
        raise ValueError("seed/steps must be nonnegative and samples positive")
    rng = np.random.default_rng(seed)
    teacher = rng.normal(size=(widths[0], widths[-1])).astype(np.float32)
    train = rng.normal(size=(512, widths[0])).astype(np.float32)
    labels = (train @ teacher).argmax(axis=1)
    weights = [(rng.normal(size=(a, b)) * np.sqrt(2 / a)).astype(np.float32)
               for a, b in zip(widths, widths[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in widths[1:]]
    for _ in range(steps):
        a = [train]
        for w, b in zip(weights[:-1], biases[:-1]):
            a.append(np.maximum(a[-1] @ w + b, 0))
        logits = a[-1] @ weights[-1] + biases[-1]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = probabilities
        gradient[np.arange(len(train)), labels] -= 1
        gradient /= len(train)
        for i in reversed(range(len(weights))):
            dw, db = a[i].T @ gradient, gradient.sum(axis=0)
            if i:
                gradient = (gradient @ weights[i].T) * (a[i] > 0)
            weights[i] -= np.float32(0.15) * dw
            biases[i] -= np.float32(0.15) * db
    layers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        layers.extend([(LINEAR, w, b), (SOFTMAX if i == len(weights) - 1 else RELU, None, None)])
    inputs = rng.normal(size=(samples, widths[0])).astype(np.float32)
    predictions = forward(inputs, layers)
    export_model(model_path, widths[0], layers)
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_path, inputs, delimiter=",", fmt="%.9g")
    reference_path = data_path.with_name("reference_predictions.csv")
    np.savetxt(reference_path, predictions, delimiter=",", fmt="%.9g")
    metadata = {
        "seed": seed, "sgd_steps": steps, "training_samples": len(train), "widths": widths,
        "sample_count": samples, "task": "synthetic linear-teacher classification",
        "training_accuracy": float(np.mean(forward(train, layers).argmax(axis=1) == labels)),
        "sample_accuracy": float(np.mean(predictions.argmax(axis=1) == (inputs @ teacher).argmax(axis=1))),
        "numpy_version": np.__version__,
    }
    Path(model_path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/example_mlp.bin"))
    parser.add_argument("--data", type=Path, default=Path("data/sample_inputs.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 32, 4],
                        help="input, hidden, and output widths; use --steps 0 for a seeded compute workload")
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 0:
        parser.error("samples must be positive and steps nonnegative")
    try:
        print(json.dumps(generate(args.model, args.data, args.seed, args.samples, args.steps, args.widths), indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
