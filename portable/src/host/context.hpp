/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_CONTEXT_HPP
#define CCTAG_PORTABLE_HOST_CONTEXT_HPP

#include "host/candidates.hpp"
#include "host/markers.hpp"
#include "host/views.hpp"

#include <cctag/Params.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cctag::portable {

// THROWAWAY: exactly one host-state owner, shared by both stage implementations.
struct PrototypeHostState {
    std::vector<CandidateLevel> candidate_levels;
    std::vector<CandidateMarker> candidate_markers;
    /// Contiguous probe rows, filled from the raw candidate markers
    std::vector<float> candidate_ellipses;
    std::vector<std::int32_t> candidate_pyramid_levels;
    std::vector<float> candidate_quality;
    MarkerBank bank;
    std::vector<IdentificationScratch> identification;
    std::vector<Marker> identified_markers;
    std::vector<Marker> preliminary_markers;
    std::vector<Marker> markers;
    std::vector<float> marker_xy;
    std::vector<std::int32_t> marker_ids;
    std::vector<std::int32_t> marker_statuses;

};

/// Holds one detection pipe's stage buffers and reuses them between frames
template <class Backend>
struct Context : PrototypeHostState {
    std::unique_ptr<typename Backend::ExecutionState> execution =
        std::make_unique<typename Backend::ExecutionState>();
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<typename Backend::Buffers> levels;

    void ensure(std::uint32_t input_width, std::uint32_t input_height, const Parameters& params) {
        const std::size_t count = params._numberOfProcessedMultiresLayers;
        // Every configured level must remain nonempty after integer halving.
        // Validate before changing buffers so a rejected request leaves the context reusable.
        const auto max_levels = std::bit_width(std::min(input_width, input_height));
        if (count == 0 || count > static_cast<std::size_t>(max_levels)) {
            throw std::invalid_argument(
                "cctagDetection: processed pyramid level count must be positive and keep every "
                "level's width and height at least one pixel"
            );
        }
        if (input_width == width && input_height == height && levels.size() == count) {
            return;
        }

        // Resize the stage buffers when the input dimensions or level count change
        Backend::wait(*this);
        levels.resize(count);
        std::uint32_t level_width = input_width;
        std::uint32_t level_height = input_height;
        for (auto& level : levels) {
            level.bind(*execution);
            level.ensure(level_width, level_height, input_width, input_height);
            level_width /= 2;
            level_height /= 2;
        }
        width = input_width;
        height = input_height;
    }
};

/// Builds probe rows without sorting or deduplicating the candidate markers
template <class Backend>
CandidatesHost host_candidates(Context<Backend>& context) {
    context.candidate_ellipses.clear();
    context.candidate_pyramid_levels.clear();
    context.candidate_quality.clear();
    for (const auto& marker : context.candidate_markers) {
        const auto& ellipse = marker.rescaled_outer_ellipse;
        context.candidate_ellipses.insert(
            context.candidate_ellipses.end(),
            {ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.angle}
        );
        context.candidate_pyramid_levels.push_back(marker.level);
        context.candidate_quality.push_back(marker.quality);
    }
    return {
        static_cast<std::uint32_t>(context.candidate_markers.size()),
        context.candidate_ellipses,
        context.candidate_pyramid_levels,
        context.candidate_quality
    };
}

/// Builds the final detection-candidate rows for the probe
template <class Backend>
MarkersHost host_markers(Context<Backend>& context) {
    context.marker_xy.clear();
    context.marker_ids.clear();
    context.marker_statuses.clear();
    for (const auto& marker : context.markers) {
        context.marker_xy.insert(context.marker_xy.end(), {marker.center.x(), marker.center.y()});
        context.marker_ids.push_back(marker.id);
        context.marker_statuses.push_back(marker.status);
    }
    return {
        static_cast<std::uint32_t>(context.markers.size()),
        context.marker_xy,
        context.marker_ids,
        context.marker_statuses
    };
}

} // namespace cctag::portable

#ifdef CCTAG_TEST
#include "backends/cpu/backend.hpp"

#include <boost/ut.hpp>

namespace cctag::portable::tests::context {

using namespace boost::ut;

inline suite<"context"> context_suite = [] {
    "context sizes levels by integer halving"_test = [] {
        Context<cpu::Backend> context;
        const cctag::Parameters params(3);

        // Use odd dimensions to check rounding when halving
        constexpr std::uint32_t input_width = 37;
        constexpr std::uint32_t input_height = 23;
        context.ensure(input_width, input_height, params);

        // Check that all configured levels were created
        expect(eq(context.levels.size(), params._numberOfProcessedMultiresLayers)) << fatal;

        // Level 1 halves each input dimension, rounding down
        expect(eq(context.levels[1].width, input_width / 2));
        expect(eq(context.levels[1].height, input_height / 2));

        // Level 3 divides each input dimension by 8, rounding down
        expect(eq(context.levels[3].width, input_width / 8));
        expect(eq(context.levels[3].height, input_height / 8));
    };

    "context reuses level zero storage when dimensions are unchanged"_test = [] {
        Context<cpu::Backend> context;
        const cctag::Parameters params(3);
        context.ensure(37, 23, params);
        const void* before = context.levels[0].src.data;
        context.ensure(37, 23, params);
        // Check that level 0 keeps the same allocation
        expect(eq(before, static_cast<const void*>(context.levels[0].src.data)));
    };
};

} // namespace cctag::portable::tests::context
#endif // CCTAG_TEST

#endif
