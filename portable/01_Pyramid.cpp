/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Stages.hpp"

namespace cctag {
namespace portable {

// Downsamples by averaging every 2x2 block
void downsample(sycl::queue& queue, const Level& source_level, const Level& destination_level) {
    const Plane<const std::uint8_t> src = source_level.source_plane();
    const Plane<std::uint8_t> dest = destination_level.source_plane();

    queue.parallel_for(sycl::range<2>(dest.height, dest.width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];

        // Sum 2x2 block
        const int corner_y = 2 * y;
        const int corner_x = 2 * x;
        const std::uint32_t sum = src.at(corner_y, corner_x) + src.at(corner_y, corner_x + 1)
            + src.at(corner_y + 1, corner_x) + src.at(corner_y + 1, corner_x + 1);

        // Divide, round to nearest and store
        dest.at(y, x) = static_cast<std::uint8_t>((sum + 2) / 4);
    });
}

} // namespace portable
} // namespace cctag
