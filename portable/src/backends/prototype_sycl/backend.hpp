// THROWAWAY: direct-call control for the SYCL ownership/delegation experiment.
#pragma once
#include "backends/cpu/backend.hpp"
#include <memory>

namespace cctag::portable::prototype_sycl {
struct ExecutionState {
    struct Impl;
    std::unique_ptr<Impl> impl;
    ExecutionState();
    ~ExecutionState();
    void wait();
    void invalidate() noexcept;
};

struct Buffers {
    struct Impl;
    std::unique_ptr<Impl> impl;
    cpu::Buffers host;
    Buffers();
    ~Buffers();
    Buffers(Buffers&&) noexcept;
    Buffers& operator=(Buffers&&) noexcept;
    void bind(ExecutionState&);
    void ensure(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);
};

struct Backend {
    using ExecutionState = prototype_sycl::ExecutionState;
    using Buffers = prototype_sycl::Buffers;
    static void load(Buffers&, kernels::Plane<const std::uint8_t>);
    static void pyramid(Buffers&, const Buffers&);
    static void gradient(Buffers&);
    // THROWAWAY reference-fed stage inputs and independent connectivity experiments.
    static void upload_gradient(Buffers&);
    static void upload_edges(Buffers&);
    static std::vector<int> hysteresis_case(Buffers&, const std::vector<int>&);
    static void edges(Buffers&, const Parameters&);
    static void edge_points(Buffers&);
    static void check_device_points(Buffers&);
    static void vote(Buffers&, const Parameters&);
    static void linking(Buffers&, const Parameters&);
    static void candidates(Context<Backend>&, const Parameters&);
    static void markers(Context<Backend>&, const Parameters&);
    static PyramidHost host_pyramid(Buffers&);
    static GradientHost host_gradient(Buffers&);
    static EdgesHost host_edges(Buffers&);
    static EdgePointsHost host_edge_points(Buffers&);
    static VoteHost host_vote(Buffers&);
    static LinkingHost host_linking(Buffers&);
    static void wait(Context<Backend>&);
    static void invalidate(Context<Backend>&) noexcept;
};
static_assert(ExecutionBackend<Backend>);
}
