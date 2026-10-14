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

/// Lists the edge pixels of the map in scan order and the index of every pixel's edge point, or -1.
/// Edge pixels past the capacity are counted but get no edge point.
void compact(
    sycl::queue& queue,
    const Level& level,
    const Plane<std::int32_t> index,
    std::uint32_t* rows
) {
    const Plane<const Mark> map = level.map_plane();
    const Plane<const std::int16_t> dx = level.dx_plane();
    const Plane<const std::int16_t> dy = level.dy_plane();
    const int width = map.width;
    const int height = map.height;
    const std::uint32_t capacity = level.capacity;
    Edgel* edgels = level.edgels;
    std::uint32_t* count = level.edgel_count;

    queue.parallel_for(sycl::range<1>(height), [=](sycl::id<1> id) {
        const int y = id[0];
        std::uint32_t found = 0;
        for (int x = 0; x < width; ++x) {
            found += map.at(y, x) == Mark::edge;
        }
        rows[y] = found;
    });

    queue.single_task([=]() {
        std::uint32_t total = 0;
        for (int y = 0; y < height; ++y) {
            const std::uint32_t found = rows[y];
            rows[y] = total;
            total += found;
        }
        *count = total;
    });

    queue.parallel_for(sycl::range<1>(height), [=](sycl::id<1> id) {
        const int y = id[0];
        std::uint32_t i = rows[y];
        for (int x = 0; x < width; ++x) {
            if (map.at(y, x) != Mark::edge || i >= capacity) {
                index.at(y, x) = -1;
                continue;
            }
            edgels[i] = {std::int16_t(x), std::int16_t(y), dx.at(y, x), dy.at(y, x), -1, -1};
            index.at(y, x) = i;
            ++i;
        }
    });
}

int sign(float v) {
    return (v > 0) - (v < 0);
}

/// The index of the first edge point gradientDirectionDescent of Bresenham.cpp reaches from the
/// edge point e against its gradient for direction -1 or along it for 1, or -1
std::int32_t descend(
    const Plane<const std::int32_t> index,
    const Edgel e,
    int direction,
    std::uint32_t dist_search
) {
    // Walks in coordinates (u, v) where u is the axis the gradient changes most along, like
    // updateXY. The threshold test of the CPU code recomputes the same direction, so it is left
    // out.
    const float dx = direction * e.dx;
    const float dy = direction * e.dy;
    const bool steep = sycl::fabs(dy) > sycl::fabs(dx);
    const float du = steep ? dy : dx;
    const float dv = steep ? dx : dy;
    const int su = sign(du);
    const int sv = sign(dv);
    const float slope = sycl::fabs(dv / du);
    int u = steep ? e.y : e.x;
    int v = steep ? e.x : e.y;
    float error = 0;
    const auto step = [&]() {
        error += slope;
        u += su;
        if (error >= 0.5f) {
            v += sv;
            error -= 1;
        }
    };
    // The edge point at (u, v), -1 if none, or -2 outside the level
    const auto at = [&](int u, int v) -> std::int32_t {
        const int x = steep ? v : u;
        const int y = steep ? u : v;
        if (x < 0 || y < 0 || x >= int(index.width) || y >= int(index.height)) {
            return -2;
        }
        return index.at(y, x);
    };

    step();
    step();
    std::uint32_t n = 2;
    std::int32_t found = at(u, v);
    if (found != -1) {
        return sycl::max(found, -1);
    }
    while (n <= dist_search) {
        step();
        ++n;
        found = at(u, v);
        if (found != -1) {
            return sycl::max(found, -1);
        }
        // The pixel beside, one step back along u, catches lines the walk cuts diagonally
        found = at(u - su, v);
        if (found != -1) {
            return sycl::max(found, -1);
        }
    }
    return -1;
}

} // namespace

void link(
    sycl::queue& queue,
    const Level& level,
    std::uint32_t dist_search,
    std::int32_t* index,
    std::uint32_t* rows
) {
    const Plane<std::int32_t> index_plane{index, level.width, level.height};
    compact(queue, level, index_plane, rows);

    // The count is only known on the device, so every slot gets a work-item
    const Plane<const std::int32_t> edge_index = index_plane;
    Edgel* edgels = level.edgels;
    const std::uint32_t* count = level.edgel_count;
    queue.parallel_for(sycl::range<1>(level.capacity), [=](sycl::id<1> id) {
        const std::uint32_t i = id[0];
        if (i >= *count) {
            return;
        }
        Edgel& e = edgels[i];
        e.before = descend(edge_index, e, -1, dist_search);
        e.after = descend(edge_index, e, 1, dist_search);
    });
}

} // namespace portable
} // namespace cctag
