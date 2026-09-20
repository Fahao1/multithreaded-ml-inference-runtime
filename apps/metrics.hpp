#pragma once

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ml_runtime {
struct LatencySummary {
    double mean, median, p95, p99;
};
inline LatencySummary summarize_latencies(std::vector<double> &values) {
    if (values.empty())
        throw std::invalid_argument("latency samples must not be empty");
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double p) {
        const double index = (values.size() - 1) * p;
        const auto low = static_cast<std::size_t>(index);
        const auto high = std::min(low + 1, values.size() - 1);
        return values[low] + (values[high] - values[low]) * (index - low);
    };
    return {mean, percentile(0.5), percentile(0.95), percentile(0.99)};
}
} // namespace ml_runtime
