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

/// Branch points a hysteresis walker remembers to come back to
constexpr int walkStack = 64;

/// The 8-neighbours of an interior pixel with a mark the pixel itself does not have: the index of
/// the first in scan order, or -1, and how many there are
struct Neighbours {
    int first;
    int count;
};

Neighbours neighbours(const Plane<const Mark> map, int i, Mark mark) {
    const int width = map.width;
    Neighbours found{-1, 0};
    for (int v = -1; v <= 1; ++v) {
        for (int u = -1; u <= 1; ++u) {
            const int j = i + v * width + u;
            if (map[j] == mark) {
                if (found.first < 0) {
                    found.first = j;
                }
                ++found.count;
            }
        }
    }
    return found;
}

/// Every weak pixel with a strong 8-neighbour becomes strong and walks its chain depth first,
/// promoting the first weak neighbour and stacking the pixel if it has more. Sets pending if a full
/// stack makes a walker leave a weak neighbour behind.
void walk(sycl::queue& queue, const Level& level, std::uint32_t* pending) {
    const Plane<Mark> map = level.mapPlane();

    queue.parallel_for(sycl::range<1>(map.size()), [=](sycl::id<1> id) {
        int i = id[0];
        if (map[i] != Mark::weak || neighbours(map, i, Mark::strong).count == 0) {
            return;
        }
        map[i] = Mark::strong;
        int stack[walkStack];
        int top = 0;
        for (;;) {
            const Neighbours weak = neighbours(map, i, Mark::weak);
            if (weak.first < 0) {
                if (top == 0) {
                    return;
                }
                i = stack[--top];
                continue;
            }
            if (weak.count > 1) {
                if (top < walkStack) {
                    stack[top++] = i;
                } else {
                    *pending = 1;
                }
            }
            map[weak.first] = Mark::strong;
            i = weak.first;
        }
    });
}

} // namespace

/// Walks until no walker leaves work behind, usually after one launch. Walkers race on the map,
/// which is safe as pixels only turn strong.
void hysteresis(sycl::queue& queue, const Level& level) {
    std::uint32_t* pending = sycl::malloc_device<std::uint32_t>(1, queue);
    std::uint32_t any = 1;
    while (any) {
        queue.memset(pending, 0, sizeof(any));
        walk(queue, level, pending);
        queue.memcpy(&any, pending, sizeof(any)).wait();
    }
    sycl::free(pending, queue);
}

} // namespace portable
} // namespace cctag
