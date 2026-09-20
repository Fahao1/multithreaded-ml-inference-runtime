# MLRTBIN version 1

All integers are unsigned 32-bit **little-endian**. All values are IEEE-754
float32, little-endian, contiguous and row-major. No native C++ structures are
written to disk; there is no padding, compression, or external metadata dependency.

The header is exactly 20 bytes, in this order:

1. Eight bytes: `MLRTBIN\0` (including the terminating zero).
2. Format version: `1`.
3. Input feature count.
4. Layer count.

Each layer starts with its type identifier:

- `1`: linear. Followed by input width, output width, `input × output` weights,
  and `output` bias values. Computes `X[batch,input] @ W[input,output] + b[output]`.
- `2`: ReLU. No additional bytes.
- `3`: sigmoid. No additional bytes.
- `4`: row-wise softmax. No additional bytes.

Example sequence: linear 32→64, ReLU, linear 64→32, ReLU, linear 32→4,
softmax. An activation preserves the current width. Activation-only sequences
are supported too. Softmax need not be last, although it normally is in a classifier.

The C++ and Python loaders reject unknown versions/types, truncated data,
non-finite parameters, incompatible adjacent widths, and trailing bytes. Widths
must be 1..65,536, layer counts 1..256, and files at most 256 MiB. Lengths are
validated against remaining file bytes before tensor allocation. These are
educational resource limits, not a comprehensive hostile-input security boundary.

There is no checksum; changing a finite weight produces another valid model.
Benchmark metadata records SHA-256 hashes to identify exactly what was tested.
Model saving writes directly to the requested path and is not crash-atomic.

`python/model_io.py` exports compatible weights, including weights trained
elsewhere after conversion to this orientation and operation subset.
`Model::save` supports round trips in C++. Train outside the runtime: it has no
backpropagation, autograd, optimizers, or training state.
