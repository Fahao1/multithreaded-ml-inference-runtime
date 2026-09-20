#include "ml_runtime/thread_pool.hpp"

namespace ml_runtime {
ThreadPool::ThreadPool(std::size_t threads) {
    if (threads == 0)
        throw std::invalid_argument("worker count must be positive");
    try {
        for (std::size_t i = 0; i < threads; ++i)
            workers_.emplace_back([this] { worker(); });
    } catch (...) {
        shutdown(); // join any threads created before a resource-allocation failure
        throw;
    }
}
ThreadPool::~ThreadPool() {
    shutdown();
}
void ThreadPool::worker() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (tasks_.empty())
                return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task(); // packaged_task captures user exceptions in its future
    }
}
void ThreadPool::shutdown() {
    std::call_once(shutdown_once_, [this] {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto &thread : workers_)
            if (thread.joinable())
                thread.join();
    });
}
} // namespace ml_runtime
