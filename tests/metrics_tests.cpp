#include "../apps/metrics.hpp"
#include <cmath>
#include <iostream>

void near(double a, double b) {
    if (std::abs(a - b) > 1e-12)
        throw std::runtime_error("incorrect latency statistic");
}
int main() {
    std::vector<double> values{100, 1, 3, 2};
    auto summary = ml_runtime::summarize_latencies(values);
    near(summary.mean, 26.5);
    near(summary.median, 2.5);
    near(summary.p95, 85.45);
    near(summary.p99, 97.09);
    values = {7};
    summary = ml_runtime::summarize_latencies(values);
    near(summary.mean, 7);
    near(summary.median, 7);
    near(summary.p95, 7);
    near(summary.p99, 7);
    values = {8, 2};
    summary = ml_runtime::summarize_latencies(values);
    near(summary.mean, 5);
    near(summary.median, 5);
    near(summary.p95, 7.7);
    near(summary.p99, 7.94);
    values.clear();
    try {
        ml_runtime::summarize_latencies(values);
        return 1;
    } catch (const std::invalid_argument &) {
    }
    std::cout << "PASS deterministic latency statistics\n";
}
