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

namespace {

// tan(22.5°) and tan(67.5°), the bounds of the diagonal sector of non-maximum suppression
constexpr float tan22 = 0.41421356f;
constexpr float tan67 = 2.41421356f;

/// The gradient magnitude at index i, rounded
int magnitude(const Plane<const std::int16_t> dx, const Plane<const std::int16_t> dy, int i) {
    const int gx = dx[i];
    const int gy = dy[i];
    return sycl::rint(sycl::sqrt(float(gx * gx + gy * gy)));
}

} // namespace

void suppress(sycl::queue& queue, const Level& level, float low, float high) {
    const Plane<const std::int16_t> dx = level.dx_plane();
    const Plane<const std::int16_t> dy = level.dy_plane();
    const Plane<Mark> map = level.map_plane();
    const int width = map.width;
    const int height = map.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        // Indexed in scan order rather than with at, as the neighbours along the gradient are one
        // flat step away from i
        const int i = y * width + x;
        const bool border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
        const int m = magnitude(dx, dy, i);
        if (border || m <= low) {
            map[i] = Mark::none;
            return;
        }

        // The index step to the next pixel along the gradient: right when the gradient is within
        // 22.5° of horizontal, down when within 22.5° of vertical, else down-right or down-left
        const int gx = dx[i];
        const int gy = dy[i];
        const float ax = sycl::abs(gx);
        const float ay = sycl::abs(gy);
        int step;
        bool diagonal = false;
        if (ay < tan22 * ax) {
            step = 1;
        } else if (ay > tan67 * ax) {
            step = width;
        } else {
            const bool same_sign = (gx < 0) == (gy < 0);
            step = same_sign ? width + 1 : width - 1;
            diagonal = true;
        }

        // A tie with the neighbour before suppresses the pixel. A tie with the one after keeps it,
        // so one pixel of a plateau survives, except along a diagonal, where cvRecodedCanny
        // suppresses both; keeping one there adds 5-10 % edge pixels.
        const int before = magnitude(dx, dy, i - step);
        const int after = magnitude(dx, dy, i + step);
        const bool maximum = m > before && (diagonal ? m > after : m >= after);
        map[i] = !maximum ? Mark::none : m > high ? Mark::strong : Mark::weak;
    });
}

} // namespace portable
} // namespace cctag
