#pragma once

#include "ml_runtime/tensor.hpp"
#include <cstdint>
#include <string>

namespace ml_runtime {
enum class LayerType : std::uint32_t { Linear = 1, Relu = 2, Sigmoid = 3, Softmax = 4 };
struct Layer {
    LayerType type;
    Tensor weights; // [input, output] for Linear, empty for activations
    Tensor bias;    // [output] for Linear
};

// Validated at construction. No mutators: one model can serve every worker.
class Model {
  public:
    Model(std::size_t input_size, std::vector<Layer> layers);
    static Model load(const std::string &path);
    void save(const std::string &path) const;
    Tensor forward(Tensor input) const;
    void validate_input(const Tensor &input) const;
    std::size_t input_size() const noexcept { return input_size_; }
    std::size_t output_size() const noexcept { return output_size_; }

  private:
    std::size_t input_size_, output_size_;
    std::vector<Layer> layers_;
};
} // namespace ml_runtime
