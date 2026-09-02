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

/// The owner of all portable-pipeline state for one detection pipe: the stage buffers of every
/// pyramid level. Created once per `pipeId` and reused by every frame; `ensure` adapts it to new
/// dimensions or parameters and is a no-op in steady state, so frame two allocates nothing.
template <class Backend>
struct Context {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<typename Backend::Buffers> levels;

    void ensure(std::uint32_t input_width, std::uint32_t input_height, const Parameters& params) {
        // The legacy pipeline sizes its pyramid by the processed layer count (Detection.cpp:813).
        const std::size_t count = params._numberOfProcessedMultiresLayers;
        if (input_width == width && input_height == height && levels.size() == count) {
            return;
        }
        levels.resize(count);
        std::uint32_t level_width = input_width;
        std::uint32_t level_height = input_height;
        std::uint32_t finer_width = 0;
        std::uint32_t finer_height = 0;
        for (auto& level : levels) {
            level.ensure(level_width, level_height, finer_width, finer_height);
            finer_width = level_width;
            finer_height = level_height;
            level_width /= 2;
            level_height /= 2;
        }
        width = input_width;
        height = input_height;
    }
};

} // namespace cctag::portable

#endif
