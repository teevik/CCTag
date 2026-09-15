// THROWAWAY: borrowed whole-image inputs for the SYCL delegation experiment.
#pragma once
#include "host/views.hpp"
#include <cctag/Params.hpp>
#include <span>

namespace cctag::portable {
struct PrototypeHostState;

struct PrototypeCandidateInput {
    std::uint32_t width, height, n;
    EdgePointsHost points;
    VoteHost vote;
    kernels::Plane<const std::int32_t> edge_map;
    std::span<const std::int32_t> link_seeds, loop_one_order;
    std::span<const std::int32_t> children_offsets, children_values, child_counts;
};

void prototype_candidates(PrototypeHostState&, std::span<const PrototypeCandidateInput>,
                          std::uint32_t width, std::uint32_t height, const Parameters&);
void prototype_markers(PrototypeHostState&, kernels::Plane<const std::uint8_t>, const Parameters&);
}
