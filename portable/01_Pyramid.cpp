/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Stages.hpp"

#include <cstddef>

namespace cctag {
namespace portable {

void downsample(sycl::queue& queue, const Level& src, const Level& dst) {
    const std::uint8_t* in = src.source;
    std::uint8_t* out = dst.source;
    const std::size_t inWidth = src.width;
    const std::size_t outWidth = dst.width;

    queue.parallel_for(sycl::range<2>(dst.height, dst.width), [=](sycl::id<2> id) {
        const std::size_t y = id[0];
        const std::size_t x = id[1];
        const std::uint8_t* top = in + 2 * y * inWidth + 2 * x;
        const std::uint8_t* bottom = top + inWidth;
        const std::uint32_t sum = top[0] + top[1] + bottom[0] + bottom[1];
        out[y * outWidth + x] = static_cast<std::uint8_t>((sum + 2) / 4);
    });
}

} // namespace portable
} // namespace cctag
