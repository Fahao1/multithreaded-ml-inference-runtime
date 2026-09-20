#include "ml_runtime/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ml_runtime {
namespace {
std::size_t elements(const std::vector<std::size_t> &shape) {
    if (shape.empty())
        throw std::invalid_argument("tensor shape must not be empty");
    std::size_t n = 1;
    for (auto d : shape) {
        if (d == 0 || n > std::numeric_limits<std::size_t>::max() / d)
            throw std::invalid_argument("tensor dimension is zero or element count overflows");
        n *= d;
    }
    if (n > std::vector<float>().max_size())
        throw std::length_error("tensor is too large");
    return n;
}
void matrix(const Tensor &x) {
    if (x.shape().size() != 2 || x.size() == 0)
        throw std::invalid_argument("operation requires a nonempty rank-2 tensor");
}
} // namespace

Tensor::Tensor(std::vector<std::size_t> shape) : shape_(std::move(shape)), values_(elements(shape_), 0.0f) {}
Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), values_(std::move(values)) {
    if (elements(shape_) != values_.size())
        throw std::invalid_argument("tensor data/shape mismatch");
}
std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != shape_.size())
        throw std::out_of_range("tensor index rank mismatch");
    std::size_t result = 0, axis = 0;
    for (auto i : indices) {
        if (i >= shape_[axis])
            throw std::out_of_range("tensor index out of bounds");
        result = result * shape_[axis++] + i;
    }
    if (result >= size())
        throw std::out_of_range("empty tensor");
    return result;
}
float &Tensor::at(std::initializer_list<std::size_t> indices) {
    return values_[offset(indices)];
}
const float &Tensor::at(std::initializer_list<std::size_t> indices) const {
    return values_[offset(indices)];
}

Tensor matmul(const Tensor &a, const Tensor &b) {
    matrix(a);
    matrix(b);
    const auto rows = a.shape()[0], inner = a.shape()[1], cols = b.shape()[1];
    if (inner != b.shape()[0])
        throw std::invalid_argument("matmul inner dimensions differ");
    Tensor result({rows, cols});
    // i-k-j visits contiguous weight/output rows; no transposes or weight copies.
    // ponytail: scalar kernel; use BLAS only when profiling justifies a dependency.
    for (std::size_t i = 0; i < rows; ++i) {
        auto *out = result.data() + i * cols;
        for (std::size_t k = 0; k < inner; ++k) {
            const auto value = a[i * inner + k];
            const auto *weight = b.data() + k * cols;
            for (std::size_t j = 0; j < cols; ++j)
                out[j] += value * weight[j];
        }
    }
    return result;
}
void add_inplace(Tensor &a, const Tensor &b) {
    if (a.size() == 0 || b.size() == 0)
        throw std::invalid_argument("cannot add empty tensors");
    if (a.shape() == b.shape()) {
        for (std::size_t i = 0; i < a.size(); ++i)
            a[i] += b[i];
    } else if (a.shape().size() == 2 && b.shape().size() == 1 && a.shape()[1] == b.size()) {
        for (std::size_t i = 0; i < a.shape()[0]; ++i)
            for (std::size_t j = 0; j < b.size(); ++j)
                a[i * b.size() + j] += b[j];
    } else
        throw std::invalid_argument("addition requires equal shapes or a matching vector bias");
}
void relu_inplace(Tensor &x) {
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::max(0.0f, x[i]);
}
void sigmoid_inplace(Tensor &x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float z = std::exp(-std::abs(x[i]));
        x[i] = x[i] >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
    }
}
void softmax_inplace(Tensor &x) {
    matrix(x);
    const auto cols = x.shape()[1];
    for (std::size_t i = 0; i < x.shape()[0]; ++i) {
        auto *row = x.data() + i * cols;
        const auto maximum = *std::max_element(row, row + cols);
        double sum = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - maximum);
            sum += row[j];
        }
        for (std::size_t j = 0; j < cols; ++j)
            row[j] = static_cast<float>(row[j] / sum);
    }
}
std::vector<std::size_t> argmax(const Tensor &x) {
    matrix(x);
    std::vector<std::size_t> result(x.shape()[0]);
    const auto cols = x.shape()[1];
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto *row = x.data() + i * cols;
        result[i] = static_cast<std::size_t>(std::max_element(row, row + cols) - row);
    }
    return result;
}
} // namespace ml_runtime
