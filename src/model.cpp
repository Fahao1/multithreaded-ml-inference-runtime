#include "ml_runtime/model.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ml_runtime {
namespace {
constexpr std::size_t max_bytes = 256 * 1024 * 1024;
constexpr std::size_t max_width = 65536;
constexpr std::size_t max_layers = 256;
constexpr std::array<char, 8> magic = {'M', 'L', 'R', 'T', 'B', 'I', 'N', '\0'};
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "model format requires IEEE-754 float32");

void finite(const Tensor &x, const char *message) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i]))
            throw std::invalid_argument(message);
}
void width(std::size_t n) {
    if (n == 0 || n > max_width)
        throw std::invalid_argument("model width must be in [1, 65536]");
}
// Decode bytes explicitly: the on-disk format never depends on host endianness
// or compiler struct padding. Reading validates lengths before allocating tensors.
class Reader {
  public:
    explicit Reader(const std::string &path) : file(path, std::ios::binary | std::ios::ate) {
        if (!file)
            throw std::runtime_error("cannot open model: " + path);
        const auto n = file.tellg();
        if (n < 0 || static_cast<std::uint64_t>(n) > max_bytes)
            throw std::runtime_error("model exceeds 256 MiB limit");
        remaining = static_cast<std::size_t>(n);
        file.seekg(0);
    }
    void read(char *p, std::size_t n) {
        if (n > remaining || !file.read(p, static_cast<std::streamsize>(n)))
            throw std::runtime_error("truncated model file");
        remaining -= n;
    }
    std::uint32_t u32() {
        std::array<unsigned char, 4> b{};
        read(reinterpret_cast<char *>(b.data()), b.size());
        return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) | (std::uint32_t(b[2]) << 16) |
               (std::uint32_t(b[3]) << 24);
    }
    Tensor tensor(std::vector<std::size_t> shape, std::size_t n) {
        if (n > remaining / 4)
            throw std::runtime_error("truncated model tensor");
        Tensor t(std::move(shape));
        for (std::size_t i = 0; i < n; ++i) {
            const auto bits = u32();
            std::memcpy(t.data() + i, &bits, 4);
        }
        return t;
    }
    std::size_t remaining = 0;

  private:
    std::ifstream file;
};
void write_u32(std::ostream &out, std::uint32_t x) {
    const char b[] = {static_cast<char>(x & 255), static_cast<char>((x >> 8) & 255),
                      static_cast<char>((x >> 16) & 255), static_cast<char>((x >> 24) & 255)};
    out.write(b, 4);
}
void write_tensor(std::ostream &out, const Tensor &t) {
    for (std::size_t i = 0; i < t.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, t.data() + i, 4);
        write_u32(out, bits);
    }
}
} // namespace

Model::Model(std::size_t input_size, std::vector<Layer> layers)
    : input_size_(input_size), output_size_(input_size), layers_(std::move(layers)) {
    width(input_size_);
    if (layers_.empty() || layers_.size() > max_layers)
        throw std::invalid_argument("model must contain 1..256 layers");
    std::size_t bytes = 20;
    for (const auto &layer : layers_) {
        bytes += 4;
        switch (layer.type) {
        case LayerType::Linear:
            if (layer.weights.shape().size() != 2 || layer.weights.shape()[0] != output_size_ ||
                layer.bias.shape().size() != 1 || layer.bias.size() != layer.weights.shape()[1])
                throw std::invalid_argument(
                    "linear layer weights/bias are incompatible with preceding layer");
            output_size_ = layer.weights.shape()[1];
            width(output_size_);
            if (layer.weights.size() > max_bytes / 4)
                throw std::invalid_argument("model exceeds 256 MiB limit");
            bytes += 8 + 4 * (layer.weights.size() + layer.bias.size());
            finite(layer.weights, "model weights must be finite");
            finite(layer.bias, "model bias must be finite");
            break;
        case LayerType::Relu:
        case LayerType::Sigmoid:
        case LayerType::Softmax:
            if (layer.weights.size() || layer.bias.size())
                throw std::invalid_argument("activation layers must not contain parameters");
            break;
        default:
            throw std::invalid_argument("unsupported model layer type");
        }
        if (bytes > max_bytes)
            throw std::invalid_argument("model exceeds 256 MiB limit");
    }
}

Model Model::load(const std::string &path) {
    Reader r(path);
    std::array<char, 8> header{};
    r.read(header.data(), header.size());
    if (header != magic)
        throw std::runtime_error("invalid model magic; expected MLRTBIN");
    if (r.u32() != 1)
        throw std::runtime_error("unsupported model format version (expected 1)");
    const auto input = r.u32(), count = r.u32();
    width(input);
    if (count == 0 || count > max_layers)
        throw std::runtime_error("invalid model layer count");
    std::vector<Layer> layers;
    std::size_t current = input;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto type = static_cast<LayerType>(r.u32());
        if (type == LayerType::Linear) {
            const auto in = r.u32(), out = r.u32();
            width(in);
            width(out);
            if (in != current)
                throw std::runtime_error("linear layer input dimension mismatch");
            auto weights = r.tensor({in, out}, static_cast<std::size_t>(in) * out);
            auto bias = r.tensor({out}, out);
            layers.push_back({type, std::move(weights), std::move(bias)});
            current = out;
        } else if (type == LayerType::Relu || type == LayerType::Sigmoid || type == LayerType::Softmax) {
            layers.push_back({type, {}, {}});
        } else
            throw std::runtime_error("unsupported model layer type: " +
                                     std::to_string(static_cast<std::uint32_t>(type)));
    }
    if (r.remaining != 0)
        throw std::runtime_error("unexpected trailing bytes in model");
    return Model(input, std::move(layers));
}
void Model::save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot create model: " + path);
    out.write(magic.data(), magic.size());
    write_u32(out, 1);
    write_u32(out, static_cast<std::uint32_t>(input_size_));
    write_u32(out, static_cast<std::uint32_t>(layers_.size()));
    for (const auto &layer : layers_) {
        write_u32(out, static_cast<std::uint32_t>(layer.type));
        if (layer.type == LayerType::Linear) {
            write_u32(out, static_cast<std::uint32_t>(layer.weights.shape()[0]));
            write_u32(out, static_cast<std::uint32_t>(layer.weights.shape()[1]));
            write_tensor(out, layer.weights);
            write_tensor(out, layer.bias);
        }
    }
    out.close();
    if (!out)
        throw std::runtime_error("failed writing model: " + path);
}
void Model::validate_input(const Tensor &input) const {
    if (input.shape().size() != 2 || input.shape()[0] == 0 || input.shape()[1] != input_size_)
        throw std::invalid_argument("input must have shape [batch, " + std::to_string(input_size_) + "]");
    finite(input, "input values must be finite");
}
Tensor Model::forward(Tensor input) const {
    validate_input(input);
    for (const auto &layer : layers_) {
        switch (layer.type) {
        case LayerType::Linear:
            input = matmul(input, layer.weights);
            add_inplace(input, layer.bias);
            break;
        case LayerType::Relu:
            relu_inplace(input);
            break;
        case LayerType::Sigmoid:
            sigmoid_inplace(input);
            break;
        case LayerType::Softmax:
            softmax_inplace(input);
            break;
        }
        finite(input, "non-finite intermediate result; input or weights overflow float32");
    }
    return input;
}
} // namespace ml_runtime
