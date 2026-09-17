// THROWAWAY: the only detector translation unit compiled for SYCL.
#include <sycl/sycl.hpp>
#include "backends/prototype_sycl/backend.hpp"
#include "host/context.hpp"
#include "kernels/gradient.hpp"
#include "kernels/canny.hpp"
#include "kernels/thinning.hpp"
#include <chrono>
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
    std::string section = std::getenv("PROTOTYPE_SYCL_SECTION") ? std::getenv("PROTOTYPE_SYCL_SECTION") : "gradient";
    sycl::device device = selected_device();
    sycl::context context{device};
    sycl::queue queue{context, device, [this](sycl::exception_list errors) {
        std::lock_guard lock(mutex);
        if (!error && errors.size()) error = *errors.begin();
    }, profiling ? sycl::property_list{sycl::property::queue::enable_profiling{}} : sycl::property_list{}};
    Impl() {
        if (section != "gradient" && section != "full" && section != "compact")
            throw std::invalid_argument("section must be gradient, full or compact");
        if (section != "gradient" && (pinned || !workers.threads.empty()))
            throw std::invalid_argument("extension requires ordinary staging and direct CPU calls");
        std::cerr << "PROTOTYPE section=" << section << '\n';
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
    int *magnitude = nullptr, *classes = nullptr, *front_a = nullptr, *front_b = nullptr;
    int *count = nullptr, *offsets = nullptr, *edge_map = nullptr, *xy = nullptr;
    float* point_gradients = nullptr;
    std::uint8_t *edges = nullptr, *thin = nullptr;
    std::size_t extra_capacity = 0, point_capacity = 0;
    bool edges_current = false, points_current = false;
    sycl::event edge_event, point_event;
    unsigned rounds = 0;
    void reserve_extra(std::size_t size) {
        if (size <= extra_capacity) return;
        execution->wait();
        auto& q = execution->impl->queue;
        for (auto** ptr : {&magnitude,&classes,&front_a,&front_b,&count,&offsets,&edge_map}) {
            if (*ptr) sycl::free(*ptr,q);
            *ptr = sycl::malloc_device<int>(size+1,q);
            if (!*ptr) throw std::bad_alloc();
        }
        for (auto** ptr : {&edges,&thin}) {
            if (*ptr) sycl::free(*ptr,q);
            *ptr = sycl::malloc_device<std::uint8_t>(size,q);
            if (!*ptr) throw std::bad_alloc();
        }
        extra_capacity = size;
    }
    void reserve_points(std::size_t n) {
        if (n <= point_capacity) return;
        execution->wait();
        auto& q = execution->impl->queue;
        if (xy) sycl::free(xy,q);
        if (point_gradients) sycl::free(point_gradients,q);
        xy = sycl::malloc_device<int>(2*n,q);
        point_gradients = sycl::malloc_device<float>(2*n,q);
        if (!xy || !point_gradients) throw std::bad_alloc();
        point_capacity = n;
    }
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
        for (void* p : {static_cast<void*>(magnitude),static_cast<void*>(classes),
            static_cast<void*>(front_a),static_cast<void*>(front_b),static_cast<void*>(count),
            static_cast<void*>(offsets),static_cast<void*>(edge_map),static_cast<void*>(xy),
            static_cast<void*>(point_gradients),static_cast<void*>(edges),static_cast<void*>(thin)})
            if (p) sycl::free(p,queue);
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
        || std::size_t(w) > std::numeric_limits<std::size_t>::max() / h / sizeof(std::int16_t)
        || std::size_t(w)*h > INT_MAX)
        throw std::invalid_argument("invalid gradient dimensions");
    impl->reserve(std::size_t(w) * h);
    if (impl->execution->impl->section != "gradient") impl->reserve_extra(std::size_t(w)*h);
    host.ensure(w, h, iw, ih);
    host.prototype_compact_vote = impl->execution->impl->section == "compact";
    impl->edges_current = impl->points_current = false;
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
                      << " observer=" << (enabled("PROTOTYPE_SYCL_ISOLATION") ? "isolation" : execution.section == "compact" ? "snapshot" : "detection") << '\n';
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
void baseline_edges(Buffers& level, const Parameters& params) {
    Backend::host_gradient(level);
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
void baseline_edge_points(Buffers& level) {
    submit_cpu(level, [host = &level.host] { cpu::Backend::edge_points(*host); });
}
void Backend::vote(Buffers& level, const Parameters& params) {
    if (level.impl->execution->impl->section != "gradient") host_edge_points(level);
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


VoteHost Backend::host_vote(Buffers& level) { wait_host(level); return cpu::Backend::host_vote(level.host); }
LinkingHost Backend::host_linking(Buffers& level) { wait_host(level); return cpu::Backend::host_linking(level.host); }
void Backend::wait(Context<Backend>& context) { context.execution->wait(); }
void Backend::invalidate(Context<Backend>& context) noexcept { context.execution->invalidate(); }

// Disposable consecutive device section. No shared scheduling abstraction.
namespace {
using Atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed,
    sycl::memory_scope::device, sycl::access::address_space::global_space>;
using Clock = std::chrono::steady_clock;
auto nanoseconds(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
// Each weak pixel is claimed once, so each frontier fits P and termination takes at most P rounds.
sycl::event flood(Buffers& level, sycl::event dependency) {
    auto& s=*level.impl; auto& q=s.execution->impl->queue;
    const int w=level.host.width,h=level.host.height,p=w*h;
    auto *classes=s.classes,*a=s.front_a,*b=s.front_b,*count=s.count;
    auto zero=q.submit([&](sycl::handler& c){c.depends_on(dependency);c.memset(count,0,4);});
    auto event=q.submit([&](sycl::handler& c){c.depends_on(zero);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){
            int i=id[0]; if(classes[i]==2) a[Atomic(*count).fetch_add(1)]=i;
        });});
    int active=0;
    q.submit([&](sycl::handler& c){c.depends_on(event);c.memcpy(&active,count,4);}).wait_and_throw();
    s.rounds=0;
    while(active) {
        if (++s.rounds>static_cast<unsigned>(p)) throw std::runtime_error("hysteresis failed to terminate");
        zero=q.memset(count,0,4);
        event=q.submit([&](sycl::handler& c){c.depends_on(zero);
            c.parallel_for(sycl::range<1>(active),[=](sycl::id<1> id){
                int i=a[id[0]],x=i%w,y=i/w;
                for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox) {
                    int nx=x+ox,ny=y+oy;
                    if(nx<0||nx>=w||ny<0||ny>=h||(ox==0&&oy==0)) continue;
                    int j=ny*w+nx,weak=1;
                    if(Atomic(classes[j]).compare_exchange_strong(weak,2))
                        b[Atomic(*count).fetch_add(1)]=j;
                }
            });});
        q.submit([&](sycl::handler& c){c.depends_on(event);c.memcpy(&active,count,4);}).wait_and_throw();
        std::swap(a,b);
    }
    s.execution->impl->check();
    return event;
}
}
void Backend::upload_gradient(Buffers& level) {
    auto& s=*level.impl; auto& q=s.execution->impl->queue;
    s.execution->wait();
    const auto w=level.host.width,h=level.host.height;
    std::vector<sycl::event> events;
    for(unsigned y=0;y<h;++y) {
        events.push_back(q.memcpy(s.dx+y*w,level.host.dx[y],w*2));
        events.push_back(q.memcpy(s.dy+y*w,level.host.dy[y],w*2));
    }
    s.gradient=q.submit([&](sycl::handler& c){c.depends_on(events);c.single_task([]{});});
    s.gradient_submitted=true;s.gradient_current=true;
}
void Backend::upload_edges(Buffers& level) {
    upload_gradient(level);
    auto& s=*level.impl;auto& q=s.execution->impl->queue;
    std::vector<sycl::event> events{s.gradient};
    for(unsigned y=0;y<level.host.height;++y)
        events.push_back(q.memcpy(s.edges+y*level.host.width,level.host.edges[y],level.host.width));
    s.edge_event=q.submit([&](sycl::handler& c){c.depends_on(events);c.single_task([]{});});
    s.edges_current=true;s.points_current=false;
}
std::vector<int> Backend::hysteresis_case(Buffers& level,const std::vector<int>& input) {
    auto& s=*level.impl;auto& q=s.execution->impl->queue;
    if(input.size()!=std::size_t(level.host.width)*level.host.height) throw std::invalid_argument("case size");
    auto copy=q.memcpy(s.classes,input.data(),input.size()*4);
    auto event=flood(level,copy);
    std::vector<int> output(input.size());
    q.submit([&](sycl::handler& c){c.depends_on(event);c.memcpy(output.data(),s.classes,input.size()*4);}).wait_and_throw();
    return output;
}
void Backend::edges(Buffers& level,const Parameters& params) {
    auto& s=*level.impl;auto& e=*s.execution->impl;auto& q=e.queue;
    if(e.section=="gradient") {baseline_edges(level,params);return;}
    const int w=level.host.width,h=level.host.height,p=w*h;
    auto *dx=s.dx,*dy=s.dy;auto *mag=s.magnitude,*classes=s.classes;
    auto *edges=s.edges,*thin=s.thin;
    auto [lo,hi]=std::minmax(params._cannyThrLow,params._cannyThrHigh);
    int low=cvFloor(lo*256.f),high=cvFloor(hi*256.f);
    auto magnitude=q.submit([&](sycl::handler& c){c.depends_on(s.gradient);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){int i=id[0];mag[i]=kernels::magnitude_at(dx[i],dy[i]);});});
    auto nms=q.submit([&](sycl::handler& c){c.depends_on(magnitude);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){
            int i=id[0],x=i%w,y=i/w;std::array<int,9> near{};
            for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox)
                if(x+ox>=0&&x+ox<w&&y+oy>=0&&y+oy<h) near[(oy+1)*3+ox+1]=mag[(y+oy)*w+x+ox];
            classes[i]=kernels::nms_class_at(dx[i],dy[i],near,low,high);
        });});
    auto flooded=flood(level,nms);
    auto raw=q.submit([&](sycl::handler& c){c.depends_on(flooded);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){edges[id[0]]=classes[id[0]]==2?255:0;});});
    std::array<std::uint8_t,512> lut1,lut2;
    std::copy_n(kernels::kThinning1,512,lut1.begin());std::copy_n(kernels::kThinning2,512,lut2.begin());
    auto pass1=q.submit([&](sycl::handler& c){c.depends_on(raw);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){int i=id[0],x=i%w,y=i/w;
            thin[i]=(x>0&&x<w-1&&y>0&&y<h-1)?kernels::thinning_at({edges,unsigned(w),unsigned(h),std::size_t(w)},x,y,lut1.data()):0;
        });});
    s.edge_event=q.submit([&](sycl::handler& c){c.depends_on(pass1);
        c.parallel_for(sycl::range<1>(p),[=](sycl::id<1> id){int i=id[0],x=i%w,y=i/w;
            if(x>0&&x<w-1&&y>0&&y<h-1) edges[i]=kernels::thinning_at({thin,unsigned(w),unsigned(h),std::size_t(w)},x,y,lut2.data());
        });});
    s.edges_current=s.points_current=false;
}
void Backend::edge_points(Buffers& level) {
    auto& s=*level.impl;auto& e=*s.execution->impl;auto& q=e.queue;
    if(e.section=="gradient") {baseline_edge_points(level);return;}
    int w=level.host.width,h=level.host.height;
    auto* edges=s.edges;auto* offsets=s.offsets;auto* count=s.count;
    auto rows=q.submit([&](sycl::handler& c){c.depends_on(s.edge_event);
        c.parallel_for(sycl::range<1>(h),[=](sycl::id<1> id){int y=id[0],n=0;
            for(int x=0;x<w;++x)n+=edges[y*w+x]==255;offsets[y]=n;
        });});
    auto scan=q.submit([&](sycl::handler& c){c.depends_on(rows);c.single_task([=]{
        int n=0;for(int y=0;y<h;++y){int k=offsets[y];offsets[y]=n;n+=k;}*count=n;
    });});
    int n=0;q.submit([&](sycl::handler& c){c.depends_on(scan);c.memcpy(&n,count,4);}).wait_and_throw();
    if(n<0||unsigned(n)>cpu::kMaxEdgePoints)throw std::length_error("edge point limit");
    s.reserve_points(n);level.host.n=n;
    auto* xy=s.xy;auto* gradients=s.point_gradients;auto* map=s.edge_map;auto* dx=s.dx;auto* dy=s.dy;
    s.point_event=q.submit([&](sycl::handler& c){c.depends_on(scan);
        c.parallel_for(sycl::range<1>(h),[=](sycl::id<1> id){int y=id[0],index=offsets[y];
            for(int x=0;x<w;++x){int i=y*w+x;
                if(edges[i]==255){xy[2*index]=x;xy[2*index+1]=y;gradients[2*index]=dx[i];gradients[2*index+1]=dy[i];map[i]=index++;}
                else map[i]=-1;
            }
        });});
    s.points_current=false;
}
EdgesHost Backend::host_edges(Buffers& level) {
    auto& s=*level.impl;auto& q=s.execution->impl->queue;
    if(s.execution->impl->section=="gradient"){wait_host(level);return cpu::Backend::host_edges(level.host);}
    if(!s.edges_current){
        if(level.host.edges.step1()==level.host.width)
            q.submit([&](sycl::handler& c){c.depends_on(s.edge_event);c.memcpy(level.host.edges[0],s.edges,std::size_t(level.host.width)*level.host.height);});
        else for(unsigned y=0;y<level.host.height;++y)
            q.submit([&](sycl::handler& c){c.depends_on(s.edge_event);c.memcpy(level.host.edges[y],s.edges+y*level.host.width,level.host.width);});
        q.wait_and_throw();s.execution->impl->check();s.edges_current=true;
        if(s.execution->impl->profiling)std::cerr<<"PROTOTYPE_SNAPSHOT_TRANSFER boundary=edges d2h_bytes="
            <<std::size_t(level.host.width)*level.host.height<<'\n';
    }
    return cpu::Backend::host_edges(level.host);
}
EdgePointsHost Backend::host_edge_points(Buffers& level) {
    auto& s=*level.impl;auto& e=*s.execution->impl;auto& q=e.queue;auto& host=level.host;
    if(e.section=="gradient"){wait_host(level);return cpu::Backend::host_edge_points(host);}
    if(!s.points_current){
        auto start=Clock::now();std::vector<sycl::event> copies;
        host.xy.resize(2*host.n);host.gradients.resize(2*host.n);
        auto copy=[&](void* out,const void* in,std::size_t bytes){if(bytes) copies.push_back(q.submit([&](sycl::handler& c){c.depends_on(s.point_event);c.memcpy(out,in,bytes);}));};
        copy(host.xy.data(),s.xy,8*host.n);copy(host.gradients.data(),s.point_gradients,8*host.n);
        if(e.section=="full") {
            if(host.edge_map.step1()==host.width)copy(host.edge_map[0],s.edge_map,std::size_t(host.width)*host.height*4);
            else for(unsigned y=0;y<host.height;++y)copy(host.edge_map[y],s.edge_map+y*host.width,host.width*4);
        }
        for(auto& event:copies)event.wait_and_throw();e.check();
        auto transfer_ns=nanoseconds(start);start=Clock::now();
        if(e.section=="compact"){
            host.edge_map.setTo(-1);
            for(unsigned i=0;i<host.n;++i)host.edge_map(host.xy[2*i+1],host.xy[2*i])=i;
        }
        auto rebuild_ns=nanoseconds(start);
        if(e.section=="full")host_gradient(level);
        s.points_current=true;
        if(e.profiling)std::cerr<<"PROTOTYPE_SECTION width="<<host.width<<" height="<<host.height<<" points="<<host.n
            <<" boundary="<<e.section<<" rounds="<<s.rounds<<" control_d2h_bytes="<<4*(s.rounds+2)
            <<" continuation_d2h_bytes="<<(16*std::size_t(host.n)+(e.section=="full"?8*std::size_t(host.width)*host.height:0))
            <<" point_map_copy_wall_ns="<<transfer_ns<<" host_rebuild_ns="<<rebuild_ns
            <<" extra_device_capacity_bytes="<<(30*s.extra_capacity+28+16*s.point_capacity)<<'\n';
    }
    return cpu::Backend::host_edge_points(host);
}

}
