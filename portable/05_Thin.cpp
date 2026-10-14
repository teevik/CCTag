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

/// The neighbours x_1 ... x_8 of a pixel, one bit each, counterclockwise from x_1 east: E, NE, N,
/// NW, W, SW, S, SE, bit k - 1 holding x_k
using Neighbourhood = std::uint32_t;

/// Which subiteration of Guo-Hall thinning a pass runs: first peels south-east boundary pixels and
/// north-west corners, second the opposite
enum class Subiteration {
    first,
    second
};

/// Whether a subiteration of Guo-Hall thinning, as in Lam, Lee and Suen, "Thinning methodologies:
/// a comprehensive survey" (1992) and MATLAB's bwmorph thin, keeps a set pixel. It deletes one with
/// one crossing (G1), 2 or 3 neighbour pairs either way round (G2) and on the side the subiteration
/// peels (G3).
constexpr bool keeps(Neighbourhood n, Subiteration subiteration) {
    const auto x = [n](int k) -> bool { return (n >> (k - 1) % 8) & 1; }; // x_9 is x_1
    int crossings = 0;
    int n1 = 0;
    int n2 = 0;
    for (int i = 1; i <= 4; ++i) {
        crossings += !x(2 * i - 1) && (x(2 * i) || x(2 * i + 1));
        n1 += x(2 * i - 1) || x(2 * i);
        n2 += x(2 * i) || x(2 * i + 1);
    }
    const int pairs = n1 < n2 ? n1 : n2;
    const bool side = subiteration == Subiteration::first ? (x(2) || x(3) || !x(8)) && x(1)
                                                          : (x(6) || x(7) || !x(4)) && x(5);
    return !(crossings == 1 && 2 <= pairs && pairs <= 3 && !side);
}

/// keeps of one subiteration for every neighbourhood, bit n % 32 of word n / 32 for neighbourhood n
using Lut = std::array<std::uint32_t, 8>;

constexpr Lut tabulate(Subiteration subiteration) {
    Lut table{};
    for (Neighbourhood n = 0; n < 256; ++n) {
        table[n / 32] |= std::uint32_t(keeps(n, subiteration)) << n % 32;
    }
    return table;
}

/// One subiteration over a whole level, as imageIter of filter/thinning.cpp, where in marks edges
/// with edge: each interior pixel of out is marked edge where lut keeps one, else none, and the
/// border is none
void thin_pass(
    sycl::queue& queue,
    const Plane<const Mark> in,
    Mark edge,
    const Lut lut,
    const Plane<Mark> out
) {
    const int width = out.width;
    const int height = out.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            out.at(y, x) = Mark::none;
            return;
        }
        const auto at = [=](int u, int v) -> std::uint32_t { return in.at(y + v, x + u) == edge; };
        const Neighbourhood n = at(1, 0) | at(1, -1) << 1 | at(0, -1) << 2 | at(-1, -1) << 3
            | at(-1, 0) << 4 | at(-1, 1) << 5 | at(0, 1) << 6 | at(1, 1) << 7;
        const bool keep = at(0, 0) && ((lut[n / 32] >> n % 32) & 1);
        out.at(y, x) = keep ? Mark::edge : Mark::none;
    });
}

} // namespace

void thin(sycl::queue& queue, const Level& level, Mark* scratch) {
    constexpr Lut first = tabulate(Subiteration::first);
    constexpr Lut second = tabulate(Subiteration::second);
    const Plane<Mark> map = level.map_plane();
    const Plane<Mark> thinned{scratch, level.width, level.height};
    thin_pass(queue, map, Mark::strong, first, thinned);
    thin_pass(queue, thinned, Mark::edge, second, map);
}

} // namespace portable
} // namespace cctag
