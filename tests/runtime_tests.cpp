#include "ml_runtime/engine.hpp"
#include "ml_runtime/model.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ml_runtime;
namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message); // active in Release too
}
void near(float a, float b, float tolerance = 1e-5f) {
    check(std::isfinite(a) && std::abs(a - b) <= tolerance, "numeric mismatch");
}
template <class F> void throws(F &&f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "expected an exception");
}
Model example() {
    return Model(2,
                 {{LayerType::Linear, Tensor({2, 3}, {1, -1, 2, 0.5f, 2, -1}), Tensor({3}, {0, 1, -1})},
                  {LayerType::Relu, {}, {}},
                  {LayerType::Linear, Tensor({3, 2}, {1, 0, 0, 1, 0.5f, -0.5f}), Tensor({2}, {0.5f, -0.5f})},
                  {LayerType::Softmax, {}, {}}});
}
void tensor_tests() {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    check(t.size() == 6 && t.shape() == std::vector<std::size_t>({2, 3}), "shape mismatch");
    near(t.at({1, 2}), 6);
    t.at({0, 1}) = 7;
    near(t[1], 7);
    const auto &ct = t;
    near(ct.at({0, 1}), 7);
    Tensor copied(t);
    copied[0] = 99;
    near(t[0], 1);
    Tensor copy_assigned;
    copy_assigned = t;
    copy_assigned[2] = 99;
    near(t[2], 3);
    check(copied.data() != t.data() && copy_assigned.data() != t.data(), "copy aliases source storage");
    throws([&] { t.at({2, 0}); });
    throws([&] { t.at({1}); });
    throws([] { Tensor({0, 3}); });
    throws([] { Tensor(std::vector<std::size_t>{}); });
    throws([] { Tensor({std::numeric_limits<std::size_t>::max(), 2}); });
    throws([] { Tensor({2}, {1}); });
    const auto *allocation = t.data();
    Tensor moved(std::move(t));
    check(moved.data() == allocation, "move copied tensor storage");
    Tensor assigned;
    assigned = std::move(moved);
    check(assigned.data() == allocation, "move assignment copied");
    auto product = matmul(Tensor({2, 2}, {1, 2, 3, 4}), Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
    const std::vector<float> expected{9, 12, 15, 19, 26, 33};
    for (std::size_t i = 0; i < expected.size(); ++i)
        near(product[i], expected[i]);
    add_inplace(product, Tensor({3}, {1, 2, 3}));
    near(product.at({1, 2}), 36);
    add_inplace(product, Tensor({2, 3}, {1, 1, 1, 1, 1, 1}));
    near(product[0], 11);
    Tensor v({2}, {1, 2});
    add_inplace(v, v);
    near(v[1], 4);
    throws([&] { add_inplace(product, Tensor({2})); });
    throws([] { matmul(Tensor({2, 3}), Tensor({2, 1})); });
    throws([] { matmul(Tensor({2}), Tensor({2})); });
    Tensor empty;
    throws([&] { empty.at({}); });
    throws([&] { matmul(empty, Tensor({1, 1})); });
    throws([&] { add_inplace(empty, empty); });
    throws([&] { softmax_inplace(empty); });
    throws([&] { argmax(empty); });
}
void activation_tests() {
    Tensor x({1, 3}, {-1000, 0, 1000});
    sigmoid_inplace(x);
    near(x[0], 0);
    near(x[1], 0.5f);
    near(x[2], 1);
    Tensor r({3}, {-2, 0, 3});
    relu_inplace(r);
    near(r[0], 0);
    near(r[2], 3);
    Tensor s({2, 3}, {1000, 1001, 1002, -1000, -1000, -1000});
    softmax_inplace(s);
    near(s[0], 0.09003057f);
    near(s[2], 0.66524096f);
    near(s[3], 1.0f / 3);
    Tensor shifted({1, 3}, {-1002, -1001, -1000});
    softmax_inplace(shifted);
    for (std::size_t i = 0; i < 3; ++i)
        near(shifted[i], s[i]);
    check(argmax(s) == std::vector<std::size_t>({2, 0}), "argmax/tie handling failed");
    throws([] {
        auto t = Tensor({3});
        softmax_inplace(t);
    });
}
void model_tests() {
    const auto model = example();
    auto output = model.forward(Tensor({1, 2}, {1, 2}));
    near(output[0], 1.0f / (1.0f + std::exp(1.0f)));
    auto batch = model.forward(Tensor({2, 2}, {1, 2, -2, 3}));
    auto second = model.forward(Tensor({1, 2}, {-2, 3}));
    near(batch[0], output[0]);
    near(batch[2], second[0]);
    throws([&] { model.forward(Tensor({1, 3})); });
    throws([&] { model.forward(Tensor({1, 2}, {0, std::numeric_limits<float>::infinity()})); });
    throws([] { Model(2, {}); });
    throws([] { Model(3, {{LayerType::Linear, Tensor({2, 3}), Tensor({3})}}); });
    throws([] { Model(2, {{static_cast<LayerType>(99), {}, {}}}); });

    // CTest runs each build in its own directory; these files are local to it.
    const std::string path = "test_model.bin";
    struct Cleanup {
        std::string path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{path};
    model.save(path);
    auto restored = Model::load(path).forward(Tensor({1, 2}, {1, 2}));
    near(restored[0], output[0], 0);
    std::ifstream input(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    input.close();
    const auto reject = [&](std::vector<char> bad) {
        std::ofstream f(path, std::ios::binary);
        f.write(bad.data(), static_cast<std::streamsize>(bad.size()));
        f.close();
        throws([&] { Model::load(path); });
    };
    for (std::size_t n = 0; n < bytes.size(); ++n)
        reject(std::vector<char>(bytes.begin(), bytes.begin() + n));
    auto bad = bytes;
    bad[0] = 'X';
    reject(bad);
    bad = bytes;
    bad[8] = 2;
    reject(bad);
    bad = bytes;
    bad[16] = 0;
    reject(bad);
    bad = bytes;
    bad[17] = 1;
    reject(bad); // more than 256 layers
    bad = bytes;
    bad[12] = 0;
    reject(bad); // zero input width
    bad = bytes;
    bad[20] = 99;
    reject(bad);
    bad = bytes;
    bad[24] = 3;
    reject(bad);
    bad = bytes;
    bad[28] = 0;
    reject(bad);
    bad = bytes;
    bad[34] = static_cast<char>(0x80);
    bad[35] = static_cast<char>(0x7f);
    reject(bad);
    bad = bytes;
    bad.push_back(0);
    reject(bad);
    throws([] { Model::load("nonexistent-model-file.bin"); });
}
void pool_tests() {
    throws([] { ThreadPool pool(0); });
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 200; ++i)
        futures.push_back(pool.submit([i] { return i * i; }));
    for (int i = 0; i < 200; ++i)
        check(futures[i].get() == i * i, "lost pool task");
    auto error = pool.submit([]() -> int { throw std::runtime_error("task failed"); });
    throws([&] { error.get(); });
    auto movable = pool.submit([p = std::make_unique<int>(7)] { return *p; });
    check(movable.get() == 7, "move-only task failed");
    std::atomic<int> done{0};
    for (int i = 0; i < 200; ++i)
        pool.submit([&] { ++done; });
    auto stop = std::async(std::launch::async, [&] { pool.shutdown(); });
    pool.shutdown();
    stop.get();
    check(done == 200, "shutdown abandoned work");
    throws([&] { pool.submit([] {}); });
    std::future<int> result;
    {
        ThreadPool short_lived(1);
        result = short_lived.submit([] { return 11; });
    }
    check(result.get() == 11, "destructor failed to drain");
}
void engine_tests() {
    const auto model = std::make_shared<const Model>(example());
    throws([&] { InferenceEngine bad(model, {1, 0, std::chrono::microseconds(0)}); });
    throws([] { InferenceEngine bad(nullptr); });
    for (auto batch_size : {1u, 8u}) {
        InferenceEngine engine(model, {4, batch_size, std::chrono::microseconds(500)});
        auto one = engine.predict({1, 2});
        near(one[0], model->forward(Tensor({1, 2}, {1, 2}))[0]);
        auto batch = engine.predict_batch(Tensor({2, 2}, {1, 2, -2, 3}));
        near(batch[2], model->forward(Tensor({1, 2}, {-2, 3}))[0]);
        auto explicit_async = engine.predict_batch_async(Tensor({1, 2}, {1, 2}));
        near(explicit_async.get()[0], one[0]);
        throws([&] { engine.predict_async({1}); });
        throws([&] { engine.predict_batch(Tensor({2, 3})); });
        throws([&] { engine.predict_async({0, std::numeric_limits<float>::quiet_NaN()}); });
        std::vector<std::future<void>> clients;
        for (int client = 0; client < 8; ++client)
            clients.push_back(std::async(std::launch::async, [&, client] {
                std::vector<std::future<Tensor>> results;
                for (int i = 0; i < 64; ++i)
                    results.push_back(engine.predict_async({i * 0.1f, client * -0.2f}));
                for (int i = 0; i < 64; ++i) {
                    const auto expected = model->forward(Tensor({1, 2}, {i * 0.1f, client * -0.2f}));
                    const auto actual = results[i].get();
                    for (std::size_t j = 0; j < actual.size(); ++j)
                        near(actual[j], expected[j]);
                }
            }));
        for (auto &client : clients)
            client.get();
        auto stop = std::async(std::launch::async, [&] { engine.shutdown(); });
        engine.shutdown();
        stop.get();
        check(engine.stats().samples == 516, "wrong executed sample count");
        throws([&] { engine.predict_async({1, 2}); });
        throws([&] { engine.predict_batch_async(Tensor({1, 2})); });
    }
    const auto overflowing = std::make_shared<const Model>(
        1, std::vector<Layer>{
               {LayerType::Linear, Tensor({1, 1}, {std::numeric_limits<float>::max()}), Tensor({1})}});
    for (auto size : {1u, 4u}) {
        InferenceEngine engine(overflowing, {2, size, std::chrono::microseconds(100)});
        auto a = engine.predict_async({2}), b = engine.predict_async({3});
        throws([&] { a.get(); });
        throws([&] { b.get(); });
        near(engine.predict({0})[0], 0); // worker survives failed inference
    }
}
void batching_tests() {
    using namespace std::chrono_literals;
    const auto model = std::make_shared<const Model>(example());
    { // Size must trigger before the deliberately long timeout.
        InferenceEngine engine(model, {2, 4, 5s});
        auto first = engine.predict_async({1, 2});
        check(first.wait_for(10ms) == std::future_status::timeout, "partial batch executed too early");
        std::vector<std::future<Tensor>> rest;
        for (int i = 0; i < 3; ++i)
            rest.push_back(engine.predict_async({static_cast<float>(i), 1}));
        check(first.wait_for(2s) == std::future_status::ready, "full batch did not trigger");
        near(first.get()[0], model->forward(Tensor({1, 2}, {1, 2}))[0]);
        for (std::size_t i = 0; i < rest.size(); ++i)
            near(rest[i].get()[0], model->forward(Tensor({1, 2}, {static_cast<float>(i), 1}))[0]);
        check(engine.stats().batches == 1 && engine.stats().samples == 4, "size batching failed");
    }
    {
        InferenceEngine engine(model, {1, 8, 40ms});
        const auto begin = std::chrono::steady_clock::now();
        auto partial = engine.predict_async({1, 2});
        check(partial.wait_for(2s) == std::future_status::ready, "partial batch never timed out");
        check(std::chrono::steady_clock::now() - begin >= 30ms, "batch timeout ignored");
        partial.get();
        check(engine.stats().batches == 1, "timeout batch count wrong");
    }
    std::vector<std::future<Tensor>> pending;
    {
        InferenceEngine engine(model, {2, 32, 60s});
        for (int i = 0; i < 11; ++i)
            pending.push_back(engine.predict_async({float(i), 1}));
    } // destructor must flush without waiting 60 seconds
    for (int i = 0; i < 11; ++i)
        near(pending[i].get()[0], model->forward(Tensor({1, 2}, {float(i), 1}))[0]);
    { // Submission racing shutdown is either accepted and completed or rejected.
        InferenceEngine engine(model, {2, 8, 1ms});
        auto producer = std::async(std::launch::async, [&] {
            std::vector<std::future<Tensor>> accepted;
            for (int i = 0; i < 500; ++i) {
                try {
                    accepted.push_back(engine.predict_async({1, 2}));
                } catch (const std::runtime_error &) {
                    break;
                }
            }
            for (auto &f : accepted)
                near(f.get()[0], model->forward(Tensor({1, 2}, {1, 2}))[0]);
        });
        engine.shutdown();
        producer.get();
    }
}
void stress_tests() {
    using namespace std::chrono_literals;
    const auto model = std::make_shared<const Model>(
        2, std::vector<Layer>{{LayerType::Linear, Tensor({2, 2}, {1, 0, 0, 1}), Tensor({2})}});
    constexpr int clients = 8, requests = 512;
    std::uint64_t checked = 0, raced = 0;
    for (const auto workers : {1u, 4u})
        for (const auto batch : {1u, 16u}) {
            InferenceEngine engine(model, {workers, batch, 1ms});
            std::promise<void> go;
            auto start = go.get_future().share();
            std::vector<std::future<void>> producers;
            for (int c = 0; c < clients; ++c)
                producers.push_back(std::async(std::launch::async, [&, c] {
                    start.wait();
                    std::vector<std::future<Tensor>> results;
                    for (int i = 0; i < requests; ++i) {
                        const auto id = static_cast<float>(c * requests + i);
                        results.push_back(engine.predict_async({id, -id}));
                    }
                    for (int i = requests - 1; i >= 0; --i) {
                        check(results[i].wait_for(10s) == std::future_status::ready,
                              "unresolved accepted future");
                        const auto value = results[i].get();
                        near(value[0], static_cast<float>(c * requests + i), 0);
                        near(value[1], -static_cast<float>(c * requests + i), 0);
                    }
                }));
            go.set_value();
            for (auto &producer : producers)
                producer.get();
            engine.shutdown();
            check(engine.stats().samples == clients * requests, "lost or duplicated inference");
            checked += engine.stats().samples;
        }
    for (int repetition = 0; repetition < 20; ++repetition) {
        InferenceEngine engine(model, {repetition % 2 ? 1u : 4u, repetition % 2 ? 1u : 16u, 10ms});
        std::vector<std::promise<void>> ready(4);
        std::promise<void> go;
        const auto start = go.get_future().share();
        std::vector<std::future<std::size_t>> producers;
        for (int c = 0; c < 4; ++c)
            producers.push_back(std::async(std::launch::async, [&, c] {
                std::vector<std::future<Tensor>> accepted;
                accepted.push_back(engine.predict_async({static_cast<float>(c), 0}));
                ready[c].set_value();
                start.wait();
                for (int i = 1; i < 256; ++i) {
                    try {
                        accepted.push_back(
                            engine.predict_async({static_cast<float>(c), static_cast<float>(i)}));
                    } catch (const std::runtime_error &) {
                        break;
                    }
                }
                for (std::size_t i = 0; i < accepted.size(); ++i) {
                    check(accepted[i].wait_for(10s) == std::future_status::ready,
                          "shutdown left unresolved future");
                    const auto value = accepted[i].get();
                    near(value[0], static_cast<float>(c), 0);
                    near(value[1], static_cast<float>(i), 0);
                }
                return accepted.size();
            }));
        for (auto &signal : ready)
            signal.get_future().wait();
        go.set_value();
        auto shutdown = std::async(std::launch::async, [&] { engine.shutdown(); });
        engine.shutdown();
        shutdown.get();
        std::size_t accepted = 0;
        for (auto &producer : producers)
            accepted += producer.get();
        check(accepted >= 4 && engine.stats().samples == accepted,
              "shutdown lost or duplicated accepted work");
        raced += accepted;
        throws([&] { engine.predict_async({0, 0}); });
    }
    for (int i = 0; i < 50; ++i) {
        InferenceEngine empty(model, {1, i % 2 ? 1u : 8u, 1ms});
        empty.shutdown();
        empty.shutdown();
    }
    std::cout << "PASS stress: " << checked << " ordered results, " << raced
              << " accepted shutdown-race results, 74 engine lifecycles\n";
}
} // namespace
int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--stress") {
        try {
            stress_tests();
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL stress: " << e.what() << '\n';
            return 1;
        }
    }
    const std::vector<std::pair<const char *, std::function<void()>>> tests{
        {"tensor", tensor_tests},
        {"activations", activation_tests},
        {"model and malformed files", model_tests},
        {"thread pool", pool_tests},
        {"inference and concurrent ordering", engine_tests},
        {"batch size, timeout, and shutdown", batching_tests}};
    for (const auto &test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << test.first << ": " << e.what() << '\n';
            return 1;
        }
    }
}
