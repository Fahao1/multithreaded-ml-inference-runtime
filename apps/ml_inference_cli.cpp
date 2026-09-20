#include "metrics.hpp"
#include "ml_runtime/engine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace ml_runtime;
namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    std::string model, input, output, metrics, mode = "concurrent";
    std::size_t threads = 1, batch = 1, clients = 32, requests = 0, warmup = 100, seed = 42;
    double timeout_ms = 2;
    bool quiet = false;
};
void help() {
    std::cout << R"(Multithreaded ML Inference Runtime (C++17)
Usage: ml_inference_cli --model FILE [options]
  --input FILE             Headerless numeric CSV; columns must match the model
                           Without a file, generate 256 seeded uniform samples
  --mode sync|concurrent   Sequential caller or fixed concurrent clients (default concurrent)
  --threads N              Inference worker threads (default 1)
  --batch-size N           Maximum dynamic batch; 1 disables batching (default 1)
  --batch-timeout-ms MS    Oldest-request deadline, 0..60000 (default 2)
  --clients N              Concurrent load-generator clients (default 32)
  --requests N             Measured requests; cycles input rows (default input row count)
  --warmup N               Separate warm-up requests (default 100)
  --seed N                 Generated input seed (default 42)
  --output FILE            Prediction CSV in original request order
  --metrics FILE           Measured metrics as JSON
  --quiet                  Suppress prediction printing
  --help                   Show this help

Examples:
  ml_inference_cli --model models/example_mlp.bin --input data/sample_inputs.csv
  ml_inference_cli --model models/example_mlp.bin --threads 4 --batch-size 16
    --batch-timeout-ms 2 --requests 2000 --metrics run.json --quiet
)";
}
std::size_t integer(const std::string &value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned integer: " + value);
    std::size_t end = 0;
    unsigned long long n;
    try {
        n = std::stoull(value, &end);
    } catch (const std::out_of_range &) {
        throw std::invalid_argument("integer out of range: " + value);
    }
    if (end != value.size() || n > 10000000)
        throw std::invalid_argument("integer out of range: " + value);
    return static_cast<std::size_t>(n);
}
Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--quiet") {
            o.quiet = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model")
            o.model = value;
        else if (key == "--input")
            o.input = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--metrics")
            o.metrics = value;
        else if (key == "--mode")
            o.mode = value;
        else if (key == "--threads")
            o.threads = integer(value);
        else if (key == "--batch-size")
            o.batch = integer(value);
        else if (key == "--clients")
            o.clients = integer(value);
        else if (key == "--requests") {
            o.requests = integer(value);
            if (!o.requests)
                throw std::invalid_argument("requests must be positive");
        } else if (key == "--warmup")
            o.warmup = integer(value);
        else if (key == "--seed")
            o.seed = integer(value);
        else if (key == "--batch-timeout-ms") {
            std::size_t end = 0;
            try {
                o.timeout_ms = std::stod(value, &end);
            } catch (const std::exception &) {
                throw std::invalid_argument("invalid --batch-timeout-ms: expected a number in [0, 60000]");
            }
            if (end != value.size() || !std::isfinite(o.timeout_ms) || o.timeout_ms < 0 ||
                o.timeout_ms > 60000)
                throw std::invalid_argument("timeout must be in [0, 60000] milliseconds");
        } else
            throw std::invalid_argument("unknown option: " + key);
    }
    if (o.model.empty())
        throw std::invalid_argument("--model is required");
    if (o.mode != "sync" && o.mode != "concurrent")
        throw std::invalid_argument("mode must be sync or concurrent");
    if (o.threads == 0 || o.threads > 256 || o.clients == 0 || o.clients > 256 || o.batch == 0 ||
        o.batch > 65536)
        throw std::invalid_argument("threads/clients must be 1..256; batch size must be 1..65536");
    if (o.mode == "sync")
        o.clients = 1;
    return o;
}
std::vector<std::vector<float>> samples(const Options &o, std::size_t width) {
    std::vector<std::vector<float>> rows;
    if (o.input.empty()) {
        std::mt19937 random(static_cast<unsigned>(o.seed));
        std::uniform_real_distribution<float> uniform(-1, 1);
        rows.assign(256, std::vector<float>(width));
        for (auto &row : rows)
            for (float &x : row)
                x = uniform(random);
        return rows;
    }
    std::ifstream file(o.input);
    if (!file)
        throw std::runtime_error("cannot open input CSV: " + o.input);
    std::string line;
    std::size_t number = 0;
    while (std::getline(file, line)) {
        ++number;
        if (line.empty())
            throw std::runtime_error("empty CSV row at line " + std::to_string(number));
        std::stringstream stream(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(stream, cell, ',')) {
            std::istringstream value(cell);
            float x;
            if (!(value >> x) || !std::isfinite(x))
                throw std::runtime_error("invalid CSV number at line " + std::to_string(number));
            value >> std::ws;
            if (!value.eof())
                throw std::runtime_error("trailing text in CSV number");
            row.push_back(x);
        }
        if (line.back() == ',' || row.size() != width)
            throw std::runtime_error("CSV dimension mismatch at line " + std::to_string(number));
        rows.push_back(std::move(row));
    }
    if (!file.eof())
        throw std::runtime_error("failed reading input CSV");
    if (rows.empty())
        throw std::runtime_error("input CSV contains no samples");
    return rows;
}
struct Resources {
    double cpu_seconds = -1, peak_rss_mib = -1;
};
Resources resources() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        const double cpu = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 + usage.ru_stime.tv_sec +
                           usage.ru_stime.tv_usec / 1e6;
#if defined(__APPLE__)
        return {cpu, usage.ru_maxrss / (1024.0 * 1024.0)};
#else
        return {cpu, usage.ru_maxrss / 1024.0};
#endif
    }
#endif
    return {};
}
struct Run {
    double seconds = 0, cpu_seconds = -1, rss = -1;
    std::vector<double> latency_ms;
    std::vector<Tensor> outputs;
    EngineStats stats{};
};
Run run(InferenceEngine &engine, const Options &o, const std::vector<std::vector<float>> &rows,
        std::size_t count, bool keep_outputs) {
    Run result;
    result.latency_ms.resize(count);
    if (keep_outputs)
        result.outputs.resize(count);
    if (!count)
        return result;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex mutex;
    std::condition_variable gate;
    bool start = false;
    const auto client = [&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            gate.wait(lock, [&] { return start; });
        }
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const auto i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count)
                    break;
                const auto begin = Clock::now();
                auto prediction = engine.predict(rows[i % rows.size()]);
                result.latency_ms[i] =
                    std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
                if (keep_outputs)
                    result.outputs[i] = std::move(prediction);
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex);
            if (!error)
                error = std::current_exception();
        }
    };
    std::vector<std::thread> clients;
    if (o.mode == "concurrent") {
        try {
            for (std::size_t i = 0; i < std::min(count, o.clients); ++i)
                clients.emplace_back(client);
        } catch (...) {
            failed.store(true);
            {
                std::lock_guard<std::mutex> lock(mutex);
                start = true;
            }
            gate.notify_all();
            for (auto &t : clients)
                t.join();
            throw;
        }
    }
    const auto stats_before = engine.stats();
    const auto before = resources();
    const auto begin = Clock::now();
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
    }
    gate.notify_all();
    if (o.mode == "sync")
        client();
    for (auto &t : clients)
        t.join();
    result.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    const auto after = resources();
    if (before.cpu_seconds >= 0 && after.cpu_seconds >= 0)
        result.cpu_seconds = after.cpu_seconds - before.cpu_seconds;
    result.rss = after.peak_rss_mib;
    const auto stats_after = engine.stats();
    result.stats = {stats_after.batches - stats_before.batches, stats_after.samples - stats_before.samples,
                    stats_after.compute_ns - stats_before.compute_ns};
    if (error)
        std::rethrow_exception(error);
    return result;
}
void metrics(std::ostream &out, const Options &o, Run &r) {
    const auto latency = summarize_latencies(r.latency_ms);
    out << std::setprecision(10) << "{\n"
        << "  \"mode\": \"" << o.mode << "\",\n"
        << "  \"threads\": " << o.threads << ",\n  \"batch_size\": " << o.batch
        << ",\n  \"batch_timeout_ms\": " << o.timeout_ms << ",\n  \"clients\": " << o.clients
        << ",\n  \"warmup_requests\": " << o.warmup << ",\n  \"total_requests\": " << o.requests
        << ",\n  \"total_seconds\": " << r.seconds << ",\n  \"throughput_rps\": " << o.requests / r.seconds
        << ",\n  \"mean_latency_ms\": " << latency.mean << ",\n  \"median_latency_ms\": " << latency.median
        << ",\n  \"p95_latency_ms\": " << latency.p95 << ",\n  \"p99_latency_ms\": " << latency.p99
        << ",\n  \"cpu_seconds\": ";
    if (r.cpu_seconds >= 0)
        out << r.cpu_seconds;
    else
        out << "null";
    out << ",\n  \"peak_rss_mib\": ";
    if (r.rss >= 0)
        out << r.rss;
    else
        out << "null";
    out << ",\n  \"executed_batches\": " << r.stats.batches
        << ",\n  \"mean_batch_size\": " << static_cast<double>(r.stats.samples) / r.stats.batches
        << ",\n  \"model_compute_ms\": " << r.stats.compute_ns / 1e6
        << ",\n  \"model_compute_ms_per_request\": " << r.stats.compute_ns / 1e6 / o.requests << "\n}\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--help") {
                help();
                return 0;
            }
        auto options = parse(argc, argv);
        const auto same_file = [](const std::string &a, const std::string &b) {
            if (a.empty() || b.empty())
                return false;
            return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
                   (std::filesystem::exists(a) && std::filesystem::exists(b) &&
                    std::filesystem::equivalent(a, b));
        };
        for (const auto &destination : {options.output, options.metrics})
            for (const auto &source : {options.model, options.input})
                if (same_file(destination, source))
                    throw std::invalid_argument("output must not overwrite a model or input file");
        if (same_file(options.output, options.metrics))
            throw std::invalid_argument("prediction and metrics files must be different");
        auto model = std::make_shared<const Model>(Model::load(options.model));
        const auto inputs = samples(options, model->input_size());
        if (!options.requests)
            options.requests = inputs.size();
        InferenceEngine engine(
            model, {options.threads, options.batch,
                    std::chrono::microseconds(static_cast<std::int64_t>(options.timeout_ms * 1000))});
        run(engine, options, inputs, options.warmup, false);
        auto result =
            run(engine, options, inputs, options.requests, !options.quiet || !options.output.empty());
        engine.shutdown();
        std::ofstream output;
        if (!options.output.empty()) {
            output.open(options.output);
            if (!output)
                throw std::runtime_error("cannot create predictions file");
        }
        for (std::size_t i = 0; i < result.outputs.size(); ++i) {
            const auto &row = result.outputs[i];
            if (output.is_open()) {
                for (std::size_t j = 0; j < row.size(); ++j)
                    output << (j ? "," : "") << std::setprecision(9) << row[j];
                output << '\n';
            }
            if (!options.quiet) {
                std::cout << "request " << i << " class=" << argmax(row)[0] << " scores=";
                for (std::size_t j = 0; j < row.size(); ++j)
                    std::cout << (j ? "," : "") << std::setprecision(6) << row[j];
                std::cout << '\n';
            }
        }
        if (output.is_open()) {
            output.close();
            if (!output)
                throw std::runtime_error("failed writing predictions");
        }
        if (!options.metrics.empty()) {
            std::ofstream file(options.metrics);
            if (!file)
                throw std::runtime_error("cannot create metrics file");
            metrics(file, options, result);
            file.close();
            if (!file)
                throw std::runtime_error("failed writing metrics");
        }
        std::cerr << "Processed " << options.requests << " requests in " << result.seconds << " s ("
                  << options.requests / result.seconds << " requests/s)\n";
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
