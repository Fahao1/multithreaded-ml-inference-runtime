"""Original MNIST IDX splits from the CVDF mirror; no training dependency."""
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

SOURCE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
SEED = 42


def sample_indices():
    return np.sort(np.random.default_rng(SEED).choice(10000, 16, replace=False))


def normalize(pixels):
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 784 or not len(pixels):
        raise ValueError("MNIST requires nonempty rows of 784 pixels")
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("raw pixels must be finite and in [0, 255]")
    return pixels.astype(np.float32) / np.float32(255)


def validate_inputs(inputs, labels):
    if inputs.ndim != 2 or inputs.shape[1] != 784 or not len(inputs):
        raise ValueError("MNIST requires nonempty rows of 784 features")
    if not np.isfinite(inputs).all() or np.any((inputs < 0) | (inputs > 1)):
        raise ValueError("normalized pixels must be finite and in [0, 1]")
    if labels.shape != (len(inputs),) or not np.isfinite(labels).all():
        raise ValueError("one finite label is required per input")
    if np.any((labels < 0) | (labels > 9) | (labels != np.floor(labels))):
        raise ValueError("labels must be integer digits 0..9")


def load_split(cache, train=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    prefix, count = ("train", 60000) if train else ("t10k", 10000)
    arrays = []
    for kind, suffix, magic, shape in [
        ("images", "idx3", 2051, (count, 28, 28)),
        ("labels", "idx1", 2049, (count,)),
    ]:
        name = f"{prefix}-{kind}-{suffix}-ubyte.gz"
        path = cache / name
        if not path.exists():
            try:
                with urlopen(SOURCE + name, timeout=60) as response:
                    compressed = response.read(16 * 1024 * 1024)
            except (OSError, URLError) as error:
                raise RuntimeError(f"MNIST download failed ({error}); check network/CA certificates or use --mode quick") from error
        else:
            compressed = path.read_bytes()
        # Published MNIST MD5s identify the original files; HTTPS protects transport.
        if hashlib.md5(compressed).hexdigest() != FILES[name]:
            raise ValueError(f"MNIST checksum mismatch: {name}; remove the corrupt cache file and retry")
        raw = gzip.decompress(compressed)
        header_size = 4 * (1 + len(shape))
        if struct.unpack(">" + "I" * (1 + len(shape)), raw[:header_size]) != (magic, *shape):
            raise ValueError(f"invalid MNIST IDX dimensions: {name}")
        if len(raw) != header_size + int(np.prod(shape)):
            raise ValueError(f"invalid MNIST IDX payload size: {name}")
        if not path.exists():
            path.write_bytes(compressed)
        arrays.append(np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape))
    inputs, labels = normalize(arrays[0].reshape(count, 784)), arrays[1]
    validate_inputs(inputs, labels)
    if set(np.unique(labels)) != set(range(10)):
        raise ValueError("MNIST split must contain all ten digits")
    return inputs, labels
