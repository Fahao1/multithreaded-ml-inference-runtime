"""Train on all 60,000 MNIST training examples; export and verify on 10,000 test digits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from evaluate import HERE, ROOT, compare, evaluate, forward, model_layers, sha256
from data import FILES, SEED, SOURCE, load_split, sample_indices
from model_io import LINEAR, RELU, SOFTMAX, export_model
import numpy as np


def train_export(cli, cache, output):
    import sklearn
    import scipy
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits
    from run_demo import prediction_grid

    if not Path(cli).is_file():
        raise ValueError("C++ CLI not found; build the runtime before training")
    train_x, train_y = load_split(cache, train=True)
    test_x, test_y = load_split(cache)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(hidden_layer_sizes=(128,), activation="relu", solver="adam", batch_size=256,
                  learning_rate_init=0.001, alpha=0.0001, random_state=SEED, shuffle=True)
    classifier = MLPClassifier(**config)
    start = time.monotonic()
    # Fixed epochs, no test-set selection or early stopping; one BLAS thread bounds load.
    with threadpool_limits(limits=1):
        for epoch in range(20):
            classifier.partial_fit(train_x, train_y, classes=np.arange(10))
            print(f"epoch {epoch + 1}/20 loss={classifier.loss_:.6f}", flush=True)
        python_probabilities = classifier.predict_proba(test_x)
    training_seconds = time.monotonic() - start
    if not np.array_equal(classifier.classes_, np.arange(10)):
        raise ValueError("unexpected classifier class ordering")
    model = output / "mnist_mlp.bin"
    export_model(model, 784, [(LINEAR, classifier.coefs_[0], classifier.intercepts_[0]),
                            (RELU, None, None),
                            (LINEAR, classifier.coefs_[1], classifier.intercepts_[1]),
                            (SOFTMAX, None, None)])
    exported_reference = forward(test_x, model_layers(model))
    export_error = compare(exported_reference, python_probabilities)
    result, actual = evaluate(cli, model, test_x, test_y)
    sklearn_cpp_error = compare(actual, python_probabilities)
    if result["test_accuracy"] < 0.90:
        raise ValueError("test accuracy below 90%; investigate training/export before publishing")
    indices = sample_indices()
    np.savetxt(output / "sample_inputs.csv", test_x[indices], delimiter=",", fmt="%.9g")
    np.savetxt(output / "sample_labels.csv", test_y[indices], delimiter=",", fmt="%d")
    np.savetxt(output / "sample_predictions.csv", actual[indices], delimiter=",", fmt="%.9g")
    prediction_grid(test_x[indices], test_y[indices], actual[indices], output / "predictions.png")
    metadata = {
        "dataset": {"name": "MNIST", "source": SOURCE, "source_documentation": "https://github.com/cvdfoundation/mnist",
                    "compressed_md5": FILES, "train_samples": 60000, "test_samples": 10000},
        "architecture": [784, 128, 10], "activations": ["relu", "softmax"],
        "preprocessing": "28x28 row-major grayscale pixels / 255, float32; no centering or augmentation",
        "seed": SEED, "sample_test_indices": indices.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
        "training": {**config, "epochs": 20, "blas_threads": 1, "seconds": training_seconds,
                     "method": "partial_fit over full training split each epoch"},
        "model_sha256": sha256(model), "format": "MLRTBIN v1",
        "validation": {**result, "sklearn_export_max_difference": export_error,
                       "sklearn_cpp_max_difference": sklearn_cpp_error},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["validation"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/ml_inference_cli")
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    parser.add_argument("--output", type=Path, default=HERE / "local/reproduced")
    args = parser.parse_args()
    try:
        train_export(args.cli, args.cache, args.output)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
