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
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cctag::portable {

/// Holds one detection pipe's stage buffers and reuses them between frames
template <class Backend>
struct Context {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<typename Backend::Buffers> levels;
    std::vector<CandidateLevel> candidate_levels;
    CandidateMarkers candidate_markers;
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

    void ensure(std::uint32_t input_width, std::uint32_t input_height, const Parameters& params) {
        const std::size_t count = params._numberOfProcessedMultiresLayers;
        // Levels halve with rounding up, so small images repeat 1x1 levels. The last level
        // index is at most 30, keeping its scale 2^30 a positive 32-bit int.
        // Validate before changing buffers so a rejected request leaves the context reusable.
        constexpr std::size_t max_levels = 31;
        if (input_width == 0 || input_height == 0 || count == 0 || count > max_levels) {
            throw std::invalid_argument(
                "cctagDetection: invalid dimensions or processed pyramid level count"
            );
        }
        if (input_width == width && input_height == height && levels.size() == count) {
            return;
        }

        // Resize the stage buffers when the input dimensions or level count change
        levels.resize(count);
        std::uint32_t level_width = input_width;
        std::uint32_t level_height = input_height;
        for (auto& level : levels) {
            level.ensure(level_width, level_height);
            level_width = level_width / 2 + level_width % 2;
            level_height = level_height / 2 + level_height % 2;
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
    const auto candidates = context.candidate_markers.view();
    for (const auto& marker : candidates) {
        const auto& ellipse = marker.rescaled_outer_ellipse;
        context.candidate_ellipses.insert(
            context.candidate_ellipses.end(),
            {ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.angle}
        );
        context.candidate_pyramid_levels.push_back(marker.level);
        context.candidate_quality.push_back(marker.quality);
    }
    return {
        static_cast<std::uint32_t>(candidates.size()),
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

#include <array>

namespace cctag::portable::tests::context {

using namespace boost::ut;

inline suite<"context"> context_suite = [] {
    "pyramid levels halve each dimension rounding up and repeat one pixel levels"_test = [] {
        Context<cpu::Backend> context;
        cctag::Parameters params(3);
        params._numberOfProcessedMultiresLayers = 4;
        using Sizes = std::vector<std::array<std::uint32_t, 2>>;
        const auto level_sizes = [&] {
            Sizes sizes;
            for (const auto& level : context.levels) {
                sizes.push_back({level.width, level.height});
            }
            return sizes;
        };

        context.ensure(37, 23, params);
        expect(std::ranges::equal(level_sizes(), Sizes{{37, 23}, {19, 12}, {10, 6}, {5, 3}}));
        context.ensure(5, 1, params);
        expect(std::ranges::equal(level_sizes(), Sizes{{5, 1}, {3, 1}, {2, 1}, {1, 1}}));
        context.ensure(1, 2, params);
        expect(std::ranges::equal(level_sizes(), Sizes{{1, 2}, {1, 1}, {1, 1}, {1, 1}}));
    };

    "context accepts one to thirty one levels and a nonempty image"_test = [] {
        Context<cpu::Backend> context;
        cctag::Parameters params(3);
        for (const std::size_t count : {1u, 31u}) {
            params._numberOfProcessedMultiresLayers = count;
            context.ensure(1, 1, params);
            expect(eq(context.levels.size(), count));
        }
        // The last level's scale 2^(count - 1) must fit in a positive 32-bit int
        for (const std::size_t count : {0u, 32u}) {
            params._numberOfProcessedMultiresLayers = count;
            expect(throws<std::invalid_argument>([&] { context.ensure(1, 1, params); }));
        }
        params._numberOfProcessedMultiresLayers = 4;
        expect(throws<std::invalid_argument>([&] { context.ensure(0, 5, params); }));
        expect(throws<std::invalid_argument>([&] { context.ensure(5, 0, params); }));
    };

    "context reuses level zero storage when dimensions are unchanged"_test = [] {
        Context<cpu::Backend> context;
        const cctag::Parameters params(3);
        context.ensure(37, 23, params);
        const void* before = context.levels[0].src.data;
        context.ensure(37, 23, params);
        expect(eq(before, static_cast<const void*>(context.levels[0].src.data)));
    };
};

} // namespace cctag::portable::tests::context
#endif // CCTAG_TEST

#endif
