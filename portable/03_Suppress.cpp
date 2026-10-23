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
constexpr float tan22_5 = 0.41421356f;
constexpr float tan67_5 = 2.41421356f;
} // namespace

// Runs canny edge supression based on `level.dx` and `level.dy`, storing the result in `level.map`
// Conditions:
// - `none` if:
//   - `magnitude <= low`
//   - Pixel is on the border
//   - Isn't a local maximum
// - `weak` if local maximum and `low < magnitude <= high`
// - `strong` if local maximum and `magnitude > high`
void suppress(sycl::queue& queue, const Level& level, float low, float high) {
    const auto width = level.width;
    const auto height = level.height;
    const auto size = std::size_t(width) * std::size_t(height);

    auto dx_src = std::vector<std::int16_t>(size);
    auto dy_src = std::vector<std::int16_t>(size);
    queue.memcpy(dx_src.data(), level.dx, size * sizeof(std::int16_t));
    queue.memcpy(dy_src.data(), level.dy, size * sizeof(std::int16_t));
    queue.wait();

    const Plane<std::int16_t> dx{dx_src.data(), width, height};
    const Plane<std::int16_t> dy{dy_src.data(), width, height};

    std::vector<Mark> map_src(size, Mark::none);
    Plane<Mark> map{map_src.data(), width, height};

    const auto magnitude_at = [&dx, &dy](int i) {
        const int gx = dx[i];
        const int gy = dy[i];
        return std::lrint(std::sqrt(float(gx * gx + gy * gy)));
    };

    // TODO: Array2D Iterator?
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int i = y * width + x;

            const int gx = dx.at(y, x);
            const int gy = dy.at(y, x);

            // Use simd?
            const bool is_border = (y == 0) || (y == height - 1) || (x == 0) || (x == width - 1);
            if (is_border) {
                map.at(y, x) = Mark::none;
                continue;
            }

            const int magnitude = magnitude_at(i);
            if (magnitude <= low) {
                map.at(y, x) = Mark::none;
                continue;
            }

            const float ax = std::abs(float(gx));
            const float ay = std::abs(float(gy));

            int step;
            bool diagonal;

            if (ay < tan22_5 * ax) { // Within 22.5 degrees of horizontal
                step = 1;
                diagonal = false;
            } else if (ay > tan67_5 * ax) { // Within 22.5 degrees of vertical
                step = width;
                diagonal = false;
            } else if (std::signbit(gx) == std::signbit(gy)) {
                step = width + 1;
                diagonal = true;
            } else {
                step = width - 1;
                diagonal = true;
            }

            auto magnitude_before = magnitude_at(i - step);
            auto magnitude_after = magnitude_at(i + step);

            auto beats_before = magnitude > magnitude_before; // Always strict

            bool beats_after;
            if (diagonal) { // Strict for diagonal
                beats_after = magnitude > magnitude_after;
            } else { // A tie with the next pixel keeps this one
                beats_after = magnitude >= magnitude_after;
            }

            if (!(beats_before && beats_after)) {
                map[i] = Mark::none;
                continue;
            }

            // Placeholder until step 4
            if (magnitude > high) {
                map[i] = Mark::strong;
            } else {
                map[i] = Mark::weak;
            }

            // if (magnitude > low) {
            //     map.at(y, x) = Mark::strong;
            // }

            // steps 1-4 go here
            // map.at(y, x) = ;
        }
    }

    // wait, because map is freed when the function returns
    queue.memcpy(level.map, map_src.data(), size * sizeof(Mark)).wait();
}

} // namespace portable
} // namespace cctag
