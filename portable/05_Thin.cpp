/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Stages.hpp"

#include <array>

namespace cctag {
namespace portable {

namespace {

/// A LUT of filter/thinning.cpp at the indices with the centre set, one bit per neighbourhood
using Lut = std::array<std::uint32_t, 8>;

constexpr Lut thin1{
    0x3333337f,
    0x33ff31ff,
    0xffffffff,
    0xffffffff,
    0x3323ffff,
    0xffffffff,
    0xff22ffff,
    0xffffffff
};
constexpr Lut thin2{
    0xfffff7ff,
    0xfffff5ff,
    0xeceef4ff,
    0xeceef4ff,
    0xffefffff,
    0xffffffff,
    0xeceef4ff,
    0xeceef4fe
};

/// One imageIter of filter/thinning.cpp over a whole level, where in marks edges with edge: each
/// interior pixel of out is marked edge where the LUT keeps one, else none, and the border is none
void thinPass(
    sycl::queue& queue,
    const Level& level,
    const Mark* in,
    Mark edge,
    const Lut lut,
    Mark* out
) {
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            out[y * width + x] = Mark::none;
            return;
        }
        const auto at = [=](int u, int v) -> std::uint32_t {
            return in[(y + v) * width + x + u] == edge;
        };
        const std::uint32_t index = at(-1, -1) | at(-1, 0) << 1 | at(-1, 1) << 2 | at(0, -1) << 3
            | at(0, 1) << 4 | at(1, -1) << 5 | at(1, 0) << 6 | at(1, 1) << 7;
        const bool keep = at(0, 0) && ((lut[index / 32] >> index % 32) & 1);
        out[y * width + x] = keep ? Mark::edge : Mark::none;
    });
}

} // namespace

void thin(sycl::queue& queue, const Level& level, Mark* scratch) {
    thinPass(queue, level, level.map, Mark::strong, thin1, scratch);
    thinPass(queue, level, scratch, Mark::edge, thin2, level.map);
}

} // namespace portable
} // namespace cctag
