/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_CONTEXT_HPP
#define CCTAG_PORTABLE_HOST_CONTEXT_HPP

#include <cctag/Params.hpp>

#include <cstdint>
#include <vector>

namespace cctag::portable {

/// Owns buffers for a CCTag detection pipe. Keeps the same buffers between frames to avoid
/// reallocations
template <class Backend>
struct Context {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<typename Backend::Buffers> levels;

    void ensure(std::uint32_t input_width, std::uint32_t input_height, const Parameters& params) {
        const std::size_t count = params._numberOfProcessedMultiresLayers;
        if (input_width == width && input_height == height && levels.size() == count) {
            return;
        }

        // Size has changed, resize levels and buffers
        levels.resize(count);
        std::uint32_t level_width = input_width;
        std::uint32_t level_height = input_height;
        for (auto& level : levels) {
            level.ensure(level_width, level_height);
            level_width /= 2;
            level_height /= 2;
        }
        width = input_width;
        height = input_height;
    }
};

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

        // Create odd sized image
        constexpr std::uint32_t input_width = 37;
        constexpr std::uint32_t input_height = 23;
        context.ensure(input_width, input_height, params);

        // Ensure correct number of levels is created
        expect(eq(context.levels.size(), params._numberOfProcessedMultiresLayers)) << fatal;

        // Level 0 has the input dimensions. Level 1 halves each dimension, rounding down:
        expect(eq(context.levels[1].width, input_width / 2));
        expect(eq(context.levels[1].height, input_height / 2));

        // Three halvings divide each dimension by 8, again rounding down:
        expect(eq(context.levels[3].width, input_width / 8));
        expect(eq(context.levels[3].height, input_height / 8));
    };

    "context reuses level zero storage when dimensions are unchanged"_test = [] {
        Context<cpu::Backend> context;
        const cctag::Parameters params(3);
        context.ensure(37, 23, params);
        const void* before = context.levels[0].src.data;
        context.ensure(37, 23, params);
        // Ensure no reallocation
        expect(eq(before, static_cast<const void*>(context.levels[0].src.data)));
    };
};

} // namespace cctag::portable::tests::context
#endif // CCTAG_TEST

#endif
