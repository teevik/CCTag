// THROWAWAY: the only detector translation unit compiled for SYCL.
#include <sycl/sycl.hpp>
#include "backends/prototype_sycl/backend.hpp"
#include "host/context.hpp"
#include "kernels/gradient.hpp"
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace cctag::portable::prototype_sycl {
namespace {
sycl::device selected_device() {
    const char* setting = std::getenv("PROTOTYPE_SYCL_PLATFORM");
    if (!setting) throw std::runtime_error("set PROTOTYPE_SYCL_PLATFORM=cpu|cuda|hip");
    const std::string wanted(setting);
    std::vector<sycl::device> matches;
    for (const auto& device : sycl::device::get_devices()) {
        const auto name = device.get_info<sycl::info::device::name>();
        if ((wanted == "cpu" && device.is_cpu())
            || (wanted == "cuda" && name.find("RTX 4090") != std::string::npos)
            || (wanted == "hip" && (name.find("gfx1032") != std::string::npos || name == "AMD Radeon RX 6800S")))
            matches.push_back(device);
    }
    if (matches.size() != 1) throw std::runtime_error("required SYCL platform missing or ambiguous");
    if (!matches[0].has(sycl::aspect::usm_device_allocations))
        throw std::runtime_error("required device USM unsupported");
    return matches[0];
}
bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string(value) == "1";
}
// Fixed workers; stage futures include all OpenMP work and propagate exceptions.
struct Workers {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::packaged_task<void()>> jobs;
    std::vector<std::thread> threads;
    std::vector<std::shared_future<void>> outstanding;
    bool stopping = false;
    Workers() {
        const char* value = std::getenv("PROTOTYPE_SYCL_WORKERS");
        const int count = value ? std::stoi(value) : 0;
        if (count < 0 || count > 2) throw std::invalid_argument("workers must be 0, 1 or 2");
        for (int i = 0; i < count; ++i) threads.emplace_back([this] {
            for (;;) {
                std::packaged_task<void()> job;
                {
                    std::unique_lock lock(mutex);
                    ready.wait(lock, [&] { return stopping || !jobs.empty(); });
                    if (jobs.empty() && stopping) return;
                    job = std::move(jobs.front()); jobs.pop_front();
                }
                job();
            }
        });
    }
    ~Workers() {
        { std::lock_guard lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto& thread : threads) thread.join();
    }
    std::shared_future<void> submit(std::shared_future<void> previous, std::function<void()> action) {
        std::packaged_task<void()> job([previous, action = std::move(action)] {
            if (previous.valid()) previous.get();
            action();
        });
        auto result = job.get_future().share();
        outstanding.push_back(result);
        { std::lock_guard lock(mutex); jobs.push_back(std::move(job)); }
        ready.notify_one();
        return result;
    }
    void wait() {
        std::exception_ptr first;
        for (auto& future : outstanding) {
            try { future.get(); } catch (...) { if (!first) first = std::current_exception(); }
        }
        outstanding.clear();
        if (first) std::rethrow_exception(first);
    }
};
}

struct ExecutionState::Impl {
    std::mutex mutex;
    std::exception_ptr error;
    std::atomic<bool> invalid = false;
    Workers workers;
    bool pinned = enabled("PROTOTYPE_SYCL_PINNED");
    bool serialized = enabled("PROTOTYPE_SYCL_SERIALIZED");
    bool profiling = enabled("PROTOTYPE_SYCL_PROFILE");
    sycl::device device = selected_device();
    sycl::context context{device};
    sycl::queue queue{context, device, [this](sycl::exception_list errors) {
        std::lock_guard lock(mutex);
        if (!error && errors.size()) error = *errors.begin();
    }, profiling ? sycl::property_list{sycl::property::queue::enable_profiling{}} : sycl::property_list{}};
    Impl() {
        if (pinned && !device.has(sycl::aspect::usm_host_allocations))
            throw std::runtime_error("requested host USM unsupported");
        std::cerr << "PROTOTYPE SYCL platform=" << device.get_info<sycl::info::device::name>()
                  << " runtime=" << device.get_info<sycl::info::device::driver_version>()
                  << " staging=" << (pinned ? "host-usm" : "ordinary")
                  << " schedule=" << (serialized ? "serialized" : "dependencies") << '\n';
        std::cerr << "PROTOTYPE CPU workers=" << workers.threads.size() << '\n';
    }
    void check() {
        std::lock_guard lock(mutex);
        if (error) { invalid = true; std::rethrow_exception(error); }
        if (invalid) throw std::runtime_error("failed Context must be recreated");
    }
};
ExecutionState::ExecutionState() : impl(std::make_unique<Impl>()) {}
ExecutionState::~ExecutionState() {
    try { impl->workers.wait(); } catch (...) {}
    try { impl->queue.wait(); } catch (...) {}
}
void ExecutionState::wait() {
    try { impl->workers.wait(); impl->queue.wait_and_throw(); impl->check(); }
    catch (...) { impl->invalid = true; throw; }
}
void ExecutionState::invalidate() noexcept { impl->invalid = true; }

struct Buffers::Impl {
    ExecutionState* execution = nullptr;
    std::size_t capacity = 0;
    std::uint8_t* source = nullptr;
    std::int16_t* dx = nullptr;
    std::int16_t* dy = nullptr;
    std::uint8_t* host_source = nullptr;
    std::int16_t* host_dx = nullptr;
    std::int16_t* host_dy = nullptr;
    sycl::event gradient;
    std::vector<sycl::event> uploads;
    std::shared_future<void> host_completion;
    bool gradient_current = false;
    bool gradient_submitted = false;
    ~Impl() {
        if (!execution) return;
        if (host_completion.valid()) { try { host_completion.get(); } catch (...) {} }
        auto& queue = execution->impl->queue;
        try { queue.wait(); } catch (...) {}
        for (void* p : {static_cast<void*>(source), static_cast<void*>(dx), static_cast<void*>(dy),
                       static_cast<void*>(host_source), static_cast<void*>(host_dx), static_cast<void*>(host_dy)})
            if (p) sycl::free(p, queue);
    }
    void reserve(std::size_t size) {
        if (size <= capacity) return;
        execution->wait();
        auto& queue = execution->impl->queue;
        // Allocate before releasing old storage; any failure invalidates this Context.
        auto* next_source = sycl::malloc_device<std::uint8_t>(size, queue);
        auto* next_dx = sycl::malloc_device<std::int16_t>(size, queue);
        auto* next_dy = sycl::malloc_device<std::int16_t>(size, queue);
        std::uint8_t* next_host_source = nullptr;
        std::int16_t *next_host_dx = nullptr, *next_host_dy = nullptr;
        if (execution->impl->pinned) {
            next_host_source = sycl::malloc_host<std::uint8_t>(size, queue);
            next_host_dx = sycl::malloc_host<std::int16_t>(size, queue);
            next_host_dy = sycl::malloc_host<std::int16_t>(size, queue);
        }
        if (!next_source || !next_dx || !next_dy
            || (execution->impl->pinned && (!next_host_source || !next_host_dx || !next_host_dy))) {
            for (void* p : {static_cast<void*>(next_source), static_cast<void*>(next_dx), static_cast<void*>(next_dy),
                           static_cast<void*>(next_host_source), static_cast<void*>(next_host_dx), static_cast<void*>(next_host_dy)})
                if (p) sycl::free(p, queue);
            execution->impl->invalid = true;
            throw std::bad_alloc();
        }
        for (void* p : {static_cast<void*>(source), static_cast<void*>(dx), static_cast<void*>(dy),
                       static_cast<void*>(host_source), static_cast<void*>(host_dx), static_cast<void*>(host_dy)})
            if (p) sycl::free(p, queue);
        source = next_source; dx = next_dx; dy = next_dy;
        host_source = next_host_source; host_dx = next_host_dx; host_dy = next_host_dy;
        capacity = size;
    }
};
Buffers::Buffers() : impl(std::make_unique<Impl>()) {}
Buffers::~Buffers() {
    // Drain while host staging is still alive (members die in reverse order).
    if (impl && impl->host_completion.valid()) {
        try { impl->host_completion.get(); } catch (...) {}
    }
}
Buffers::Buffers(Buffers&&) noexcept = default;
Buffers& Buffers::operator=(Buffers&&) noexcept = default;
void Buffers::bind(ExecutionState& execution) { impl->execution = &execution; }
void Buffers::ensure(std::uint32_t w, std::uint32_t h, std::uint32_t iw, std::uint32_t ih) {
    if (!w || !h || w > INT_MAX || h > INT_MAX
        || std::size_t(w) > std::numeric_limits<std::size_t>::max() / h / sizeof(std::int16_t))
        throw std::invalid_argument("invalid gradient dimensions");
    impl->reserve(std::size_t(w) * h);
    host.ensure(w, h, iw, ih);
    impl->gradient_current = impl->gradient_submitted = false;
}

void Backend::load(Buffers& level, kernels::Plane<const std::uint8_t> input) {
    cpu::Backend::load(level.host, input);
}
void Backend::pyramid(Buffers& level, const Buffers& finer) {
    cpu::Backend::pyramid(level.host, finer.host);
}
void Backend::gradient(Buffers& level) {
    auto& state = *level.impl;
    auto& execution = *state.execution->impl;
    execution.check();
    auto& queue = execution.queue;
    const auto src = level.host.src_plane();
    const auto w = src.width, h = src.height;
    auto* source = state.source;
    auto* dx = state.dx;
    auto* dy = state.dy;
    std::vector<sycl::event> uploads;
    if (execution.pinned) {
        for (std::uint32_t y = 0; y < h; ++y)
            std::memcpy(state.host_source + std::size_t(y) * w, src.row(y), w);
        uploads.push_back(queue.memcpy(source, state.host_source, std::size_t(w) * h));
    } else if (src.stride == w) {
        uploads.push_back(queue.memcpy(source, src.data, std::size_t(w) * h));
    } else {
        for (std::uint32_t y = 0; y < h; ++y)
            uploads.push_back(queue.memcpy(source + std::size_t(y) * w, src.row(y), w));
    }
    const auto dx_taps = kernels::kDxTaps, dy_taps = kernels::kDyTaps;
    state.gradient_current = false;
    state.gradient = queue.submit([&](sycl::handler& handler) {
        handler.depends_on(uploads);
        handler.parallel_for(sycl::range<2>(h, w), [=](sycl::id<2> id) {
            const auto y = static_cast<std::uint32_t>(id[0]);
            const auto x = static_cast<std::uint32_t>(id[1]);
            dx[std::size_t(y) * w + x] = kernels::gradient_at(source, w, w, h, x, y, dx_taps);
            dy[std::size_t(y) * w + x] = kernels::gradient_at(source, w, w, h, x, y, dy_taps);
        });
    });
    state.gradient_submitted = true;
    state.uploads = uploads;
    if (enabled("PROTOTYPE_SYCL_RESIDENCY_CHECK")) {
        // Two real consumers use the same device allocations before any host transfer.
        for (int pass = 0; pass < 2; ++pass) {
            const auto dependency = state.gradient;
            state.gradient = queue.submit([&](sycl::handler& handler) {
                handler.depends_on(dependency);
                handler.parallel_for(sycl::range<1>(std::size_t(w) * h), [=](sycl::id<1> id) {
                    dx[id[0]] = -dx[id[0]];
                    dy[id[0]] = -dy[id[0]];
                });
            });
        }
    }
    if (execution.serialized) state.gradient.wait_and_throw();
}
GradientHost Backend::host_gradient(Buffers& level) {
    auto& state = *level.impl;
    if (!state.gradient_submitted) throw std::logic_error("gradient not produced");
    if (!state.gradient_current) {
        auto& execution = *state.execution->impl;
        auto& queue = execution.queue;
        const auto w = level.host.width, h = level.host.height;
        auto host_dx = level.host.dx_plane(), host_dy = level.host.dy_plane();
        std::vector<sycl::event> copies;
        auto copy = [&](std::int16_t* out, const std::int16_t* in, std::size_t size) {
            copies.push_back(queue.submit([&](sycl::handler& handler) {
                handler.depends_on(state.gradient);
                handler.memcpy(out, in, size * sizeof(std::int16_t));
            }));
        };
        if (execution.pinned) {
            copy(state.host_dx, state.dx, std::size_t(w) * h);
            copy(state.host_dy, state.dy, std::size_t(w) * h);
        } else if (host_dx.stride == w && host_dy.stride == w) {
            copy(host_dx.data, state.dx, std::size_t(w) * h);
            copy(host_dy.data, state.dy, std::size_t(w) * h);
        } else {
            for (std::uint32_t y = 0; y < h; ++y) {
                copy(host_dx.row(y), state.dx + std::size_t(y) * w, w);
                copy(host_dy.row(y), state.dy + std::size_t(y) * w, w);
            }
        }
        for (auto& event : copies) event.wait_and_throw();
        execution.check();
        if (execution.profiling) {
            auto elapsed = [](const sycl::event& event) {
                return event.get_profiling_info<sycl::info::event_profiling::command_end>()
                    - event.get_profiling_info<sycl::info::event_profiling::command_start>();
            };
            std::uint64_t upload_ns = 0, download_ns = 0;
            for (const auto& event : state.uploads) upload_ns += elapsed(event);
            for (const auto& event : copies) download_ns += elapsed(event);
            std::cerr << "PROTOTYPE_TRANSFER width=" << w << " height=" << h
                      << " h2d_bytes=" << std::size_t(w) * h
                      << " d2h_bytes=" << std::size_t(w) * h * 4
                      << " h2d_count=" << state.uploads.size() << " d2h_count=" << copies.size()
                      << " h2d_event_ns=" << upload_ns << " d2h_event_ns=" << download_ns
                      << " method=SYCL_event_profiling boundary=gradient_to_edges"
                      << " observer=" << (enabled("PROTOTYPE_SYCL_ISOLATION") ? "isolation" : "detection") << '\n';
        }
        if (execution.pinned) {
            for (std::uint32_t y = 0; y < h; ++y) {
                std::memcpy(host_dx.row(y), state.host_dx + std::size_t(y) * w, w * sizeof(std::int16_t));
                std::memcpy(host_dy.row(y), state.host_dy + std::size_t(y) * w, w * sizeof(std::int16_t));
            }
        }
        state.gradient_current = true;
    }
    return cpu::Backend::host_gradient(level.host);
}
void Backend::edges(Buffers& level, const Parameters& params) {
    host_gradient(level);
    auto& workers = level.impl->execution->impl->workers;
    if (workers.threads.empty()) cpu::Backend::edges(level.host, params);
    else level.impl->host_completion = workers.submit({}, [host = &level.host, params] {
        if (enabled("PROTOTYPE_SYCL_INJECT_WORKER_FAILURE"))
            throw std::runtime_error("injected prototype worker failure");
        cpu::Backend::edges(*host, params);
    });
}
namespace {
void submit_cpu(Buffers& level, std::function<void()> action) {
    auto& workers = level.impl->execution->impl->workers;
    if (workers.threads.empty()) action();
    else level.impl->host_completion = workers.submit(level.impl->host_completion, std::move(action));
}
void wait_host(Buffers& level) {
    if (level.impl->host_completion.valid()) {
        try { level.impl->host_completion.get(); }
        catch (...) { level.impl->execution->impl->invalid = true; throw; }
    }
}
}
void Backend::edge_points(Buffers& level) {
    submit_cpu(level, [host = &level.host] { cpu::Backend::edge_points(*host); });
}
void Backend::vote(Buffers& level, const Parameters& params) {
    submit_cpu(level, [host = &level.host, params] { cpu::Backend::vote(*host, params); });
}
void Backend::linking(Buffers& level, const Parameters& params) {
    submit_cpu(level, [host = &level.host, params] { cpu::Backend::linking(*host, params); });
}
void Backend::candidates(Context<Backend>& context, const Parameters& params) {
    std::vector<PrototypeCandidateInput> inputs;
    for (auto& level : context.levels) {
        wait_host(level);
        inputs.push_back(cpu::prototype_candidate_input(level.host));
    }
    prototype_candidates(context, inputs, context.width, context.height, params);
}
void Backend::markers(Context<Backend>& context, const Parameters& params) {
    prototype_markers(context, host_pyramid(context.levels[0]).src, params);
}
PyramidHost Backend::host_pyramid(Buffers& level) { return cpu::Backend::host_pyramid(level.host); }
EdgesHost Backend::host_edges(Buffers& level) { wait_host(level); return cpu::Backend::host_edges(level.host); }
EdgePointsHost Backend::host_edge_points(Buffers& level) { wait_host(level); return cpu::Backend::host_edge_points(level.host); }
VoteHost Backend::host_vote(Buffers& level) { wait_host(level); return cpu::Backend::host_vote(level.host); }
LinkingHost Backend::host_linking(Buffers& level) { wait_host(level); return cpu::Backend::host_linking(level.host); }
void Backend::wait(Context<Backend>& context) { context.execution->wait(); }
void Backend::invalidate(Context<Backend>& context) noexcept { context.execution->invalidate(); }
}
