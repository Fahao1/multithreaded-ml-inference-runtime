"""Offline MNIST integration tests: included model/fixtures only, never downloads."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data import load_split, normalize, sample_indices, validate_inputs
from evaluate import ATOL, RTOL, HERE, evaluate, model_layers, sha256


class MnistTests(unittest.TestCase):
    def test_normalization_and_labels(self):
        raw = np.zeros((2, 784), dtype=np.uint8)
        raw[0, 0], raw[1, 0] = 255, 128
        normalized = normalize(raw)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized[0, 0], 1)
        self.assertEqual(normalized[1, 0], np.float32(128) / np.float32(255))
        validate_inputs(normalized, np.array([0, 9]))
        for inputs, labels in [(np.zeros((1, 783)), np.array([0])),
                               (np.full((1, 784), 1.1), np.array([0])),
                               (np.full((1, 784), np.nan), np.array([0])),
                               (normalized, np.array([-1, 10])),
                               (normalized, np.array([0, 1.5])),
                               (normalized, np.array([0]))]:
            with self.assertRaises(ValueError):
                validate_inputs(inputs, labels)
        for pixels in [np.zeros((1, 783)), np.full((1, 784), 256), np.full((1, 784), -1)]:
            with self.assertRaises(ValueError):
                normalize(pixels)

    def test_metadata_and_dimensions(self):
        metadata = json.loads((HERE / "model_metadata.json").read_text())
        self.assertEqual(metadata["architecture"], [784, 128, 10])
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["model_sha256"], sha256(HERE / "mnist_mlp.bin"))
        self.assertEqual(metadata["dataset"]["train_samples"], 60000)
        self.assertEqual(metadata["validation"]["evaluated_samples"], 10000)
        self.assertGreaterEqual(metadata["validation"]["test_accuracy"], 0.90)
        self.assertEqual(metadata["validation"]["class_agreement"], 1.0)
        self.assertEqual(metadata["validation"]["tolerance"], {"atol": ATOL, "rtol": RTOL})
        for key in ("max_probability_difference", "sklearn_export_max_difference", "sklearn_cpp_max_difference"):
            self.assertTrue(np.isfinite(metadata["validation"][key]))
            self.assertGreaterEqual(metadata["validation"][key], 0)
        self.assertTrue(metadata["versions"]["scikit_learn"])
        indices = sample_indices()
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(metadata["sample_test_indices"], indices.tolist())
        np.testing.assert_array_equal(indices, [858, 891, 941, 2013, 4325, 4383, 5262, 6537,
                                                6968, 7174, 7355, 7609, 7728, 7860, 8577, 9752])
        layers = model_layers(HERE / "mnist_mlp.bin")
        self.assertEqual(layers[0][2].shape, (128,))
        self.assertEqual(layers[2][2].shape, (10,))

    def test_cpp_agreement_and_loading(self):
        inputs = np.loadtxt(HERE / "sample_inputs.csv", delimiter=",", dtype=np.float32, ndmin=2)
        labels = np.loadtxt(HERE / "sample_labels.csv", delimiter=",", ndmin=1)
        self.assertEqual(inputs.shape, (16, 784))
        result, actual = evaluate(CLI, HERE / "mnist_mlp.bin", inputs, labels)
        reference = np.loadtxt(HERE / "sample_predictions.csv", delimiter=",", ndmin=2)
        np.testing.assert_allclose(actual, reference, atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(actual.argmax(1), reference.argmax(1))
        self.assertEqual(result["class_agreement"], 1.0)
        print(f"MNIST offline agreement: {result}", flush=True)

    def test_quick_demo_offline(self):
        # Run in process with downloads forbidden; still invokes the real CLI.
        from run_demo import quick
        with tempfile.TemporaryDirectory() as directory, patch("data.urlopen", side_effect=AssertionError("network forbidden")):
            image = Path(directory) / "predictions.png"
            quick(CLI, HERE / "mnist_mlp.bin", HERE / "sample_inputs.csv", HERE / "sample_labels.csv", image)
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_download_failure_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("data.urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(RuntimeError, "use --mode quick"):
                    load_split(directory)
            (Path(directory) / "t10k-images-idx3-ubyte.gz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_split(directory)

    def test_bad_files(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            malformed = temp / "malformed.csv"
            malformed.write_text("1,2,not-a-pixel\n")
            wrong_width = temp / "width.csv"
            wrong_width.write_text("0,1\n")
            for extra in [["--model", str(temp / "missing.bin")],
                          ["--input", str(temp / "missing.csv")],
                          ["--input", str(malformed)], ["--input", str(wrong_width)]]:
                result = subprocess.run([sys.executable, str(HERE / "run_demo.py"), "--cli", str(CLI),
                                         "--output", str(temp / "predictions.png"), *extra],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    CLI = args.cli.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
