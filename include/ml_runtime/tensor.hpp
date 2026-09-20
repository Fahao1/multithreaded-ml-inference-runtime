#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ml_runtime {

// Row-major, contiguous float storage. Copies are explicit value copies; moves
// transfer the allocation. A default/moved-from tensor is only a storage holder.
class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    const std::vector<std::size_t> &shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return values_.size(); }
    float *data() noexcept { return values_.data(); }
    const float *data() const noexcept { return values_.data(); }
    float &operator[](std::size_t i) {
        assert(i < size());
        return values_[i];
    }
    const float &operator[](std::size_t i) const {
        assert(i < size());
        return values_[i];
    }
    float &at(std::initializer_list<std::size_t> indices);
    const float &at(std::initializer_list<std::size_t> indices) const;

  private:
    std::size_t offset(std::initializer_list<std::size_t> indices) const;
    std::vector<std::size_t> shape_;
    std::vector<float> values_;
};

Tensor matmul(const Tensor &a, const Tensor &b);
void add_inplace(Tensor &a, const Tensor &b); // equal shape or [columns] bias
void relu_inplace(Tensor &x);
void sigmoid_inplace(Tensor &x);
void softmax_inplace(Tensor &x);                  // each row of a matrix
std::vector<std::size_t> argmax(const Tensor &x); // first maximum per row

} // namespace ml_runtime
