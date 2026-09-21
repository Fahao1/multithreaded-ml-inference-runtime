"""Offline MNIST quick demo, or full download/train/export/evaluate reproduction."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from data import validate_inputs
from evaluate import HERE, ROOT, compare, cpp_predict, forward, model_layers


def prediction_grid(inputs, labels, probabilities, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4
    rows = (len(inputs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(8, 2.5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(inputs):
            predicted = int(probabilities[index].argmax())
            axis.imshow(inputs[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Actual {int(labels[index])} | C++ {predicted}\n"
                           f"Confidence {probabilities[index, predicted]:.1%}", fontsize=11,
                           color="#166534" if predicted == labels[index] else "#b91c1c")
    figure.suptitle("MNIST · 784 → 128 ReLU → 10 softmax", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def quick(cli, model, inputs_path, labels_path, output):
    inputs = np.loadtxt(inputs_path, delimiter=",", dtype=np.float32, ndmin=2)
    labels = np.loadtxt(labels_path, delimiter=",", ndmin=1)
    validate_inputs(inputs, labels)
    expected = forward(inputs, model_layers(model))
    actual = cpp_predict(cli, model, inputs)
    error = compare(actual, expected)
    for label, probabilities in zip(labels, actual):
        predicted = int(probabilities.argmax())
        print(f"Actual {int(label)}  predicted {predicted}  confidence {probabilities[predicted]:.2%}")
    prediction_grid(inputs, labels, actual, output)
    print(json.dumps({"samples": len(inputs), "class_agreement": 1.0,
                      "max_probability_difference": error}))
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/ml_inference_cli")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--model", type=Path, default=HERE / "mnist_mlp.bin")
    parser.add_argument("--input", type=Path, default=HERE / "sample_inputs.csv")
    parser.add_argument("--labels", type=Path, default=HERE / "sample_labels.csv")
    parser.add_argument("--output", type=Path, default=HERE / "local/predictions.png")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    parser.add_argument("--artifact-dir", type=Path, default=HERE / "local/reproduced")
    args = parser.parse_args()
    try:
        if args.mode == "full":
            from train_export import train_export
            train_export(args.cli, args.cache, args.artifact_dir)
        else:
            quick(args.cli, args.model, args.input, args.labels, args.output)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
