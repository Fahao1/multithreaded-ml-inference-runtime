#include "ml_runtime/engine.hpp"

#include <algorithm>
#include <cmath>

namespace ml_runtime {
InferenceEngine::InferenceEngine(std::shared_ptr<const Model> model, EngineConfig config)
    : model_(std::move(model)), config_(config), pool_(config.threads) {
    if (!model_)
        throw std::invalid_argument("model must not be null");
    if (config_.max_batch_size == 0 || config_.max_batch_size > 65536 || config_.max_batch_wait.count() < 0 ||
        config_.max_batch_wait > std::chrono::seconds(60))
        throw std::invalid_argument("batch size must be 1..65536 and wait must be 0..60 seconds");
    if (config_.max_batch_size > 1)
        scheduler_ = std::thread([this] { schedule(); });
}
InferenceEngine::~InferenceEngine() {
    shutdown();
}
Tensor InferenceEngine::predict(std::vector<float> input) {
    return predict_async(std::move(input)).get();
}
Tensor InferenceEngine::predict_batch(Tensor input) {
    return predict_batch_async(std::move(input)).get();
}

std::future<Tensor> InferenceEngine::predict_async(std::vector<float> input) {
    if (input.size() != model_->input_size())
        throw std::invalid_argument("request input dimension mismatch");
    for (float x : input)
        if (!std::isfinite(x))
            throw std::invalid_argument("input values must be finite");
    if (config_.max_batch_size == 1)
        return predict_batch_async(Tensor({1, model_->input_size()}, std::move(input)));
    Request request{std::move(input), {}, Clock::now()};
    auto future = request.result.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_)
            throw std::runtime_error("inference engine is shut down");
        request.queued = Clock::now();
        requests_.push_back(std::move(request));
    }
    ready_.notify_one();
    return future;
}
std::future<Tensor> InferenceEngine::predict_batch_async(Tensor input) {
    model_->validate_input(input);
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
        throw std::runtime_error("inference engine is shut down");
    return pool_.submit([this, input = std::move(input)]() mutable { return compute(std::move(input)); });
}
Tensor InferenceEngine::compute(Tensor input) {
    const auto count = input.shape()[0];
    const auto start = Clock::now();
    auto output = model_->forward(std::move(input));
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    compute_ns_.fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
    samples_.fetch_add(count, std::memory_order_relaxed);
    batches_.fetch_add(1, std::memory_order_relaxed);
    return output;
}
void InferenceEngine::execute(const std::shared_ptr<std::vector<Request>> &batch) {
    try {
        Tensor input({batch->size(), model_->input_size()});
        for (std::size_t i = 0; i < batch->size(); ++i)
            std::copy((*batch)[i].input.begin(), (*batch)[i].input.end(),
                      input.data() + i * model_->input_size());
        const auto output = compute(std::move(input));
        // Split once before fulfilling promises, so allocation errors fail the
        // entire batch consistently; every caller owns its independent output.
        std::vector<Tensor> results;
        results.reserve(batch->size());
        for (std::size_t i = 0; i < batch->size(); ++i) {
            const auto *row = output.data() + i * model_->output_size();
            results.emplace_back(std::vector<std::size_t>{1, model_->output_size()},
                                 std::vector<float>(row, row + model_->output_size()));
        }
        for (std::size_t i = 0; i < batch->size(); ++i)
            (*batch)[i].result.set_value(std::move(results[i]));
    } catch (...) {
        const auto error = std::current_exception();
        for (auto &request : *batch)
            request.result.set_exception(error);
    }
}
void InferenceEngine::schedule() {
    std::shared_ptr<std::vector<Request>> batch;
    try {
        for (;;) {
            batch = std::make_shared<std::vector<Request>>();
            batch->reserve(config_.max_batch_size);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !requests_.empty(); });
                if (requests_.empty())
                    return;
                // The deadline belongs to the oldest request; arrivals never
                // extend it. During shutdown, every remaining partial batch runs.
                const auto deadline = requests_.front().queued + config_.max_batch_wait;
                ready_.wait_until(lock, deadline,
                                  [this] { return stopping_ || requests_.size() >= config_.max_batch_size; });
                const auto count = std::min(requests_.size(), config_.max_batch_size);
                for (std::size_t i = 0; i < count; ++i) {
                    batch->push_back(std::move(requests_.front()));
                    requests_.pop_front();
                }
            }
            pool_.submit([this, batch] { execute(batch); });
            batch.reset();
        }
    } catch (...) {
        // A scheduler allocation/submission failure must reach every accepted caller.
        const auto error = std::current_exception();
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        if (batch)
            for (auto &request : *batch)
                request.result.set_exception(error);
        for (auto &request : requests_)
            request.result.set_exception(error);
        requests_.clear();
    }
}
void InferenceEngine::shutdown() {
    std::call_once(shutdown_once_, [this] {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        if (scheduler_.joinable())
            scheduler_.join();
        pool_.shutdown();
    });
}
EngineStats InferenceEngine::stats() const noexcept {
    return {batches_.load(std::memory_order_relaxed), samples_.load(std::memory_order_relaxed),
            compute_ns_.load(std::memory_order_relaxed)};
}
} // namespace ml_runtime
