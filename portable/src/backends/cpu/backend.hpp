/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP

#include "host/backend.hpp"
#include "host/views.hpp"
#include "kernels/linking.hpp"
#include "kernels/plane.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace cctag::portable::cpu {

/// Maximum number of edge points in one pyramid level
inline constexpr std::uint32_t kMaxEdgePoints = 1u << 24;

/// Holds one pyramid level's buffers
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    /// Descent uses the input image's bounds at every pyramid level
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;

    /// Source grayscale image
    cv::Mat1b src;
    /// Horizontal gradients
    cv::Mat1s dx;
    /// Vertical gradients
    cv::Mat1s dy;
    /// Thinned edges, with raw Canny output on the border
    cv::Mat1b edges;

    /// Canonical index at each pixel, or -1 when no edge point is present
    cv::Mat1i edge_map;
    /// Edge-point collection in canonical order, with interleaved (x, y) and (dx, dy) pairs
    std::uint32_t n = 0;
    std::vector<std::int32_t> xy;
    std::vector<float> gradients;
    /// Per-row counts, replaced by exclusive offsets before scattering edge points
    std::vector<std::uint32_t> row_offsets;

    /// Vote graph in snapshot layout, with interleaved (before, after) links
    std::vector<std::int32_t> links;
    std::vector<std::int32_t> voters_offsets;
    std::vector<std::int32_t> voters_values;
    std::vector<std::int32_t> is_max;
    std::vector<float> flow_length;
    /// Seed set in canonical order and seeds in ownership-resolution order
    std::vector<std::int32_t> seeds;
    std::vector<std::int32_t> seed_order;
    /// Each point's chosen seed and total flow distance
    std::vector<std::int32_t> voted_for;
    std::vector<float> vote_distance;
    /// Per-point sub-segment distances and per-row cursors for the voter CSR
    std::vector<float> vote_segments;
    std::vector<std::int32_t> voter_cursors;

    /// Each processed seed's walk and the slots accepted by ownership resolution
    std::vector<kernels::SegmentSlot> arena;
    std::vector<std::int32_t> accepted_slots;
    std::vector<std::uint8_t> processed_in;
    /// Compacted segments in ascending seed order, with points in walk order
    std::vector<std::int32_t> link_seeds;
    std::vector<std::int32_t> segment_offsets;
    std::vector<std::int32_t> segment_values;
    std::vector<std::int32_t> child_counts;
    std::vector<float> avg_vote;
    /// Children in segment-walk order and the candidate order consumed by loop two
    std::vector<std::int32_t> children_offsets;
    std::vector<std::int32_t> children_values;
    std::vector<std::int32_t> loop_one_order;

    /// Magnitudes and NMS classes, each with a zero border outside the image
    cv::Mat1i magnitude;
    cv::Mat1b nms_class;
    /// First thinning pass, with a zero border inside the image
    cv::Mat1b thinning;
    /// Pixels still to visit in the hysteresis flood
    std::vector<std::uint8_t*> hysteresis_stack;

    Buffers() = default;
    Buffers(Buffers&&) noexcept = default;
    Buffers& operator=(Buffers&&) noexcept = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

    void ensure(
        std::uint32_t level_width,
        std::uint32_t level_height,
        std::uint32_t image_width = 0,
        std::uint32_t image_height = 0
    );

    kernels::Plane<std::uint8_t> src_plane() {
        return {src[0], width, height, src.step1()};
    }
    kernels::Plane<std::int16_t> dx_plane() {
        return {dx[0], width, height, dx.step1()};
    }
    kernels::Plane<std::int16_t> dy_plane() {
        return {dy[0], width, height, dy.step1()};
    }
    kernels::Plane<std::uint8_t> edges_plane() {
        return {edges[0], width, height, edges.step1()};
    }
};

/// The CPU execution backend, implementing ExecutionBackend
struct Backend {
    using Buffers = cpu::Buffers;

    /// Copies the input grayscale image into `level0.src`
    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    /// Downsamples `finer.src` into `coarser.src` to build the next pyramid level
    static void pyramid(Buffers& coarser, const Buffers& finer);
    /// Computes horizontal (`dx`) and vertical (`dy`) gradients from `src`
    static void gradient(Buffers& level);
    /// Finds and thins edges from `dx` and `dy`
    static void edges(Buffers& level, const Parameters& params);
    /// Compacts edges into the edge-point collection and rewrites the edge map
    static void edge_points(Buffers& level);
    /// Links edge points and gathers their votes into the vote graph
    static void vote(Buffers& level, const Parameters& params);
    /// Walks seeds, resolves ownership and gathers segments and their children
    static void linking(Buffers& level, const Parameters& params);
    /// Fits candidate markers across levels and refits their outer ellipses at level zero
    static void candidates(Context<Backend>& context, const Parameters& params);
    static void markers(Context<Backend>& context, const Parameters& params);

    static PyramidHost host_pyramid(Buffers& level);
    static GradientHost host_gradient(Buffers& level);
    static EdgesHost host_edges(Buffers& level);
    static EdgePointsHost host_edge_points(Buffers& level);
    static VoteHost host_vote(Buffers& level);
    static LinkingHost host_linking(Buffers& level);

    static void wait(Context<Backend>&) {}
};

static_assert(ExecutionBackend<Backend>);

} // namespace cctag::portable::cpu

#endif
