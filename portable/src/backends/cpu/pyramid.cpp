/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

namespace cctag::portable::cpu {

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    // Copy row by row since the input may be view of a larger image
    for (std::uint32_t y = 0; y < input.height; ++y) {
        std::copy_n(input.row(y), input.width, level0.src[static_cast<int>(y)]);
    }
}

void Backend::pyramid(Buffers& coarser, const Buffers& finer) {
    cv::resize(finer.src, coarser.src, coarser.src.size());
}

PyramidHost Backend::host_pyramid(Buffers& level) {
    return PyramidHost{level.src_plane().as_const()};
}

} // namespace cctag::portable::cpu
