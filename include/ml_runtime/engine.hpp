#pragma once

#include "ml_runtime/model.hpp"
#include "ml_runtime/thread_pool.hpp"
#include <atomic>
#include <chrono>
#include <deque>

namespace ml_runtime {
struct EngineConfig {
    std::size_t threads = 1;
    std::size_t max_batch_size = 1; // 1 bypasses the batching scheduler entirely
    std::chrono::microseconds max_batch_wait{2000};
};
struct EngineStats {
    std::uint64_t batches;
    std::uint64_t samples;
    std::uint64_t compute_ns; // sum of worker Model::forward durations, can exceed wall time
};

class InferenceEngine {
  public:
    explicit InferenceEngine(std::shared_ptr<const Model> model, EngineConfig config = {});
    ~InferenceEngine();
    InferenceEngine(const InferenceEngine &) = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;

    Tensor predict(std::vector<float> input);
    std::future<Tensor> predict_async(std::vector<float> input);
    Tensor predict_batch(Tensor input);
    std::future<Tensor> predict_batch_async(Tensor input); // explicit batches bypass coalescing
    void shutdown();                                       // external callers only; concurrent calls are safe
    EngineStats stats() const noexcept;

  private:
    using Clock = std::chrono::steady_clock;
    struct Request {
        std::vector<float> input;
        std::promise<Tensor> result;
        Clock::time_point queued;
    };
    void schedule();
    Tensor compute(Tensor input);
    void execute(const std::shared_ptr<std::vector<Request>> &batch);

    std::shared_ptr<const Model> model_;
    EngineConfig config_;
    ThreadPool pool_;
    std::mutex mutex_;
    std::condition_variable ready_;
    // ponytail: unbounded queues; add admission control for a long-running service.
    std::deque<Request> requests_;
    bool stopping_ = false;
    std::thread scheduler_;
    std::once_flag shutdown_once_;
    std::atomic<std::uint64_t> batches_{0}, samples_{0}, compute_ns_{0};
};
} // namespace ml_runtime
