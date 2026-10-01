/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "kernels/frontend.hpp"

#include <algorithm>
#include <cstdint>

namespace cctag::portable::cpu {

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    // Copy row by row since the input may be a view of a larger image
    for (std::uint32_t y = 0; y < input.height; ++y) {
        std::copy_n(input.row(y), input.width, level0.src[static_cast<int>(y)]);
    }
}

void Backend::pyramid(Buffers& level, const Buffers& level0) {
    const kernels::Plane<const std::uint8_t>
        source{level0.src[0], level0.width, level0.height, level0.src.step1()};
#pragma omp parallel for schedule(static)
    for (int y = 0; y < int(level.height); ++y) {
        for (int x = 0; x < int(level.width); ++x) {
            level.src(y, x) = kernels::pyramid_at(source, x, y, level.width, level.height);
        }
    }
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include "host/context.hpp"

#include <boost/ut.hpp>

#include <array>

namespace cctag::portable::tests::pyramid_stage {

using namespace boost::ut;

inline suite<"pyramid_stage"> pyramid_stage_suite = [] {
    "pyramid rounds odd sizes up and samples the source origin"_test = [] {
        cctag::Parameters params(3);
        params._numberOfProcessedMultiresLayers = 2;
        Context<cpu::Backend> context;
        context.ensure(5, 3, params);
        expect(eq(context.levels[1].width, 3u));
        expect(eq(context.levels[1].height, 2u));
        // Source f(x, y) = 12x + 24y is sampled at (5x/3 - 1/2, 3y/2 - 1/2). The weights
        // 1/6 and 5/6 round to 43/256 and 213/256, so the first row is
        // trunc(12 * (1 + 43/256)) = 14 and trunc(12 * (2 + 213/256)) = 33.
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 5; ++x) {
                context.levels[0].src(y, x) = 12 * x + 24 * y;
            }
        }
        cpu::Backend::pyramid(context.levels[1], context.levels[0]);
        const std::array<std::uint8_t, 6> expected{0, 14, 33, 24, 38, 57};
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 3; ++x) {
                expect(eq(context.levels[1].src(y, x), expected[y * 3 + x]));
            }
        }
    };
};

} // namespace cctag::portable::tests::pyramid_stage
#endif // CCTAG_TEST
