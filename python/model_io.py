"""Independent NumPy reference and portable MLRTBIN v1 writer/reader."""
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MLRTBIN\0"
LINEAR, RELU, SIGMOID, SOFTMAX = 1, 2, 3, 4


def export_model(path, input_size, layers):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(MAGIC + struct.pack("<III", 1, input_size, len(layers)))
        for kind, weights, bias in layers:
            stream.write(struct.pack("<I", kind))
            if kind == LINEAR:
                stream.write(struct.pack("<II", *weights.shape))
                stream.write(np.asarray(weights, dtype="<f4").tobytes(order="C"))
                stream.write(np.asarray(bias, dtype="<f4").tobytes(order="C"))


def load_model(path):
    """Reference tool, with the same format limits as the C++ loader."""
    path = Path(path)
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("model exceeds 256 MiB limit")
    data = path.read_bytes()
    cursor = 0

    def read(n):
        nonlocal cursor
        if n > len(data) - cursor:
            raise ValueError("truncated model")
        value = data[cursor:cursor + n]
        cursor += n
        return value

    def u32():
        return struct.unpack("<I", read(4))[0]

    if read(8) != MAGIC or u32() != 1:
        raise ValueError("unsupported model header/version")
    width, count = u32(), u32()
    if not 1 <= width <= 65536 or not 1 <= count <= 256:
        raise ValueError("invalid model dimensions/layer count")
    input_size = width
    layers = []
    for _ in range(count):
        kind = u32()
        weights = bias = None
        if kind == LINEAR:
            rows, cols = u32(), u32()
            if rows != width or not 1 <= cols <= 65536:
                raise ValueError("incompatible linear dimensions")
            weights = np.frombuffer(read(rows * cols * 4), dtype="<f4").reshape(rows, cols)
            bias = np.frombuffer(read(cols * 4), dtype="<f4")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError("non-finite model parameters")
            width = cols
        elif kind not in (RELU, SIGMOID, SOFTMAX):
            raise ValueError("unknown layer type")
        layers.append((kind, weights, bias))
    if cursor != len(data):
        raise ValueError("trailing model bytes")
    return input_size, layers


def forward(inputs, layers):
    x = np.asarray(inputs, dtype=np.float32).copy()
    for kind, weights, bias in layers:
        if kind == LINEAR:
            x = x @ weights + bias
        elif kind == RELU:
            x = np.maximum(x, 0)
        elif kind == SIGMOID:
            z = np.exp(-np.abs(x))
            x = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
        elif kind == SOFTMAX:
            x = np.exp(x - x.max(axis=1, keepdims=True))
            x /= x.sum(axis=1, keepdims=True)
        else:
            raise ValueError("unsupported reference operation")
    return x
