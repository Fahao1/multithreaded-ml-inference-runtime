#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ml_runtime {
class ThreadPool {
  public:
    explicit ThreadPool(std::size_t threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    template <class F> auto submit(F &&function) -> std::future<std::invoke_result_t<F>> {
        using Result = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(function));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_)
                throw std::runtime_error("thread pool is shut down");
            tasks_.emplace([task] { (*task)(); });
        }
        ready_.notify_one();
        return future;
    }
    // Drains all accepted tasks. Call from an external thread, never a pool task.
    void shutdown();

  private:
    void worker();
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
    std::once_flag shutdown_once_;
};
} // namespace ml_runtime
